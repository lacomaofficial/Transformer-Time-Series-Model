# Univariate time series Transformer-based model 



import math
import os
import logging
import collections
import dataclasses
from pathlib import Path
from typing import Callable, Optional, Sequence, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file, save_file


# ============================================================================
# SECTION 1: CONFIGURATION DATA CLASSES
# Defines the architecture hyperparameters and model structure
# ============================================================================

@dataclasses.dataclass(frozen=False)
class ResidualBlockConfig:
    """Configuration for residual blocks used in tokenizer and output projections."""
    input_dims: int
    hidden_dims: int
    output_dims: int
    use_bias: bool
    activation: str  # "relu", "swish", or "none"


@dataclasses.dataclass(frozen=False)
class TransformerConfig:
    """Configuration for individual transformer layers."""
    model_dims: int          # Hidden dimension of the transformer
    hidden_dims: int         # Feedforward hidden dimension
    num_heads: int           # Number of attention heads
    attention_norm: str      # Normalization type for attention ("rms")
    feedforward_norm: str    # Normalization type for FFN ("rms")
    qk_norm: str            # Query-Key normalization ("rms")
    use_bias: bool          # Whether to use bias in linear layers
    use_rotary_position_embeddings: bool  # Use RoPE for positional encoding
    ff_activation: str      # Activation function in FFN
    fuse_qkv: bool          # Fuse QKV projection into single linear layer


@dataclasses.dataclass(frozen=False)
class StackedTransformersConfig:
    """Configuration for the stack of transformer layers."""
    num_layers: int
    transformer: TransformerConfig


@dataclasses.dataclass(frozen=True)
class TimesFM_2p5_200M_Definition:
    """
    Framework-agnostic configuration of TimesFM 2.5 with 200M parameters.
    
    Key architectural choices:
    - Input patches of 32 timesteps
    - Output patches of 128 timesteps  
    - 20 transformer layers with 16 attention heads
    - Model dimension of 1280
    - Forecasts 9 quantiles (0.1 to 0.9) plus mean
    """
    context_limit: int = 16384                    # Maximum context length
    input_patch_len: int = 32                     # P: Input patch size
    output_patch_len: int = 128                   # O: Output patch size
    output_quantile_len: int = 1024               # OS: Quantile output length
    quantiles: list = dataclasses.field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    decode_index: int = 5                         # Index of median (0.5) quantile
    
    # Tokenizer: Converts raw time series patches to embeddings
    tokenizer: ResidualBlockConfig = ResidualBlockConfig(
        input_dims=64,       # 32 timesteps + 32 mask values
        hidden_dims=1280,
        output_dims=1280,
        use_bias=True,
        activation="swish",
    )
    
    # Stack of 20 transformer layers
    stacked_transformers: StackedTransformersConfig = StackedTransformersConfig(
        num_layers=20,
        transformer=TransformerConfig(
            model_dims=1280,
            hidden_dims=1280,
            num_heads=16,
            attention_norm="rms",
            feedforward_norm="rms",
            qk_norm="rms",
            use_bias=False,
            use_rotary_position_embeddings=True,
            ff_activation="swish",
            fuse_qkv=True,
        ),
    )
    
    # Output projections for point forecasts and quantiles
    output_projection_point: ResidualBlockConfig = ResidualBlockConfig(
        input_dims=1280,
        hidden_dims=1280,
        output_dims=1280,
        use_bias=False,
        activation="swish",
    )
    
    output_projection_quantiles: ResidualBlockConfig = ResidualBlockConfig(
        input_dims=1280,
        hidden_dims=1280,
        output_dims=10240,   # 128 timesteps * 10 quantiles * 8 (special encoding)
        use_bias=False,
        activation="swish",
    )


@dataclasses.dataclass(frozen=False)
class DecodeCache:
    """
    Cache for efficient autoregressive decoding.
    Stores key-value pairs from previous timesteps to avoid recomputation.
    """
    next_index: torch.Tensor      # Next position to write in cache
    num_masked: torch.Tensor      # Number of masked (padding) positions
    key: torch.Tensor             # Cached attention keys
    value: torch.Tensor           # Cached attention values


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# Helper functions for normalization, masking, and data preprocessing
# ============================================================================

_TOLERANCE = 1e-6


def revin(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, reverse: bool = False):
    """
    Reversible Instance Normalization (RevIN).
    
    Normalizes input by subtracting mean and dividing by standard deviation.
    Can be reversed to restore original scale after prediction.
    
    Args:
        x: Input tensor
        mu: Mean for normalization
        sigma: Standard deviation for normalization
        reverse: If True, denormalize; if False, normalize
    
    Returns:
        Normalized or denormalized tensor
    """
    # Expand dimensions to match input shape
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]
    
    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)


def update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple:
    """
    Updates running statistics (count, mean, std) incrementally.
    
    Used during autoregressive decoding to maintain normalization statistics
    as new predictions are generated.
    
    Args:
        n: Current count of valid observations
        mu: Current mean
        sigma: Current standard deviation
        x: New observations
        mask: Boolean mask indicating invalid/missing values
    
    Returns:
        Updated (n, mu, sigma) tuple
    """
    is_legit = torch.logical_not(mask)
    inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)
    
    # Calculate incremental mean
    inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
    inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
    inc_mu = inc_mu_numerator / inc_n_safe
    inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)
    
    # Calculate incremental variance
    inc_var_numerator = torch.sum(
        ((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1
    )
    inc_var = inc_var_numerator / inc_n_safe
    inc_var = torch.where(inc_n == 0, 0.0, inc_var)
    inc_sigma = torch.sqrt(inc_var)
    
    # Update combined statistics
    new_n = n + inc_n
    new_n_safe = torch.where(new_n == 0, 1.0, new_n)
    
    new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
    new_mu = torch.where(new_n == 0, 0.0, new_mu)
    
    # Welford's online algorithm for combining variances
    term1 = n * sigma.pow(2)
    term2 = inc_n * inc_sigma.pow(2)
    term3 = n * (mu - new_mu).pow(2)
    term4 = inc_n * (inc_mu - new_mu).pow(2)
    
    new_var = (term1 + term2 + term3 + term4) / new_n_safe
    new_var = torch.where(new_n == 0, 0.0, new_var)
    new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))
    
    return (new_n, new_mu, new_sigma), (new_n, new_mu, new_sigma)


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: torch.Tensor,
    query_index_offset: torch.Tensor | None = None,
    kv_length: int = 0,
) -> torch.Tensor:
    """
    Creates causal attention mask for autoregressive modeling.
    
    Ensures each position can only attend to itself and previous positions,
    while also respecting masked (padding) positions.
    
    Args:
        query_length: Length of query sequence
        num_all_masked_kv: Number of masked key-value positions per batch
        query_index_offset: Offset for query indices (used in decoding)
        kv_length: Length of key-value sequence (defaults to query_length)
    
    Returns:
        Boolean attention mask (True = attend, False = don't attend)
    """
    if kv_length == 0:
        kv_length = query_length
    
    q_index = torch.arange(query_length, device=num_all_masked_kv.device)[
        None, None, :, None
    ]
    if query_index_offset is not None:
        q_index = q_index + query_index_offset[:, None, None, None]
    kv_index = torch.arange(kv_length, device=num_all_masked_kv.device)[
        None, None, None, :
    ]
    
    # Causal mask: query can only attend to current and past positions
    # Also exclude masked positions
    return torch.logical_and(
        q_index >= kv_index,
        kv_index >= num_all_masked_kv[:, None, None, None],
    )


def strip_leading_nans(arr):
    """Removes contiguous NaN values from the beginning of a NumPy array."""
    isnan = np.isnan(arr)
    first_valid_index = np.argmax(~isnan)
    return arr[first_valid_index:]


def linear_interpolation(arr):
    """
    Performs linear interpolation to fill NaN values in a 1D numpy array.
    
    Args:
        arr: The 1D numpy array containing NaN values
    
    Returns:
        Array with NaNs filled via linear interpolation
    """
    nans = np.isnan(arr)
    if not np.any(nans):
        return arr
    
    def x(z):
        return z.nonzero()[0]
    
    nans_indices = x(nans)
    non_nans_indices = x(~nans)
    non_nans_values = arr[~nans]
    
    try:
        arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
    except ValueError:
        if non_nans_values:
            mu = np.nanmean(arr)
        else:
            mu = 0.0
        arr = np.where(np.isfinite(arr), arr, mu)
    return arr


# ============================================================================
# SECTION 3: NEURAL NETWORK COMPONENTS
# Building blocks of the transformer architecture
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes inputs by their root mean square, then scales with learned parameters.
    More stable than LayerNorm for some applications.
    """
    def __init__(self, num_features: int, epsilon: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        self.epsilon = epsilon
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs


class ResidualBlock(nn.Module):
    """
    Residual block with two linear layers and skip connection.
    
    Architecture: output = activation(hidden_layer(input)) + residual_layer(input)
    
    Used for tokenization and output projection.
    """
    def __init__(self, config: ResidualBlockConfig):
        super().__init__()
        self.config = config
        
        self.hidden_layer = nn.Linear(
            in_features=config.input_dims,
            out_features=config.hidden_dims,
            bias=config.use_bias,
        )
        self.output_layer = nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.output_dims,
            bias=config.use_bias,
        )
        self.residual_layer = nn.Linear(
            in_features=config.input_dims,
            out_features=config.output_dims,
            bias=config.use_bias,
        )
        
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "swish":
            self.activation = nn.SiLU()
        elif config.activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation: {config.activation} not supported.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(
            self.activation(self.hidden_layer(x))
        ) + self.residual_layer(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes positional information by rotating query and key vectors.
    Provides better extrapolation to longer sequences than absolute positional encodings.
    """
    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
    
    def forward(self, inputs: torch.Tensor, position: torch.Tensor | None = None):
        """
        Applies rotary positional embeddings to inputs.
        
        Args:
            inputs: Input tensor of shape (batch, seq_len, ..., embedding_dims)
            position: Position indices (defaults to 0, 1, 2, ...)
        
        Returns:
            Inputs with rotary positional embeddings applied
        """
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                f"Embedding dims ({self.embedding_dims}) must match "
                f"input hidden dimension ({inputs.shape[-1]})."
            )
        
        half_embedding_dim = self.embedding_dims // 2
        fraction = (
            2 * torch.arange(0, half_embedding_dim, device=inputs.device)
            / self.embedding_dims
        )
        timescale = (
            self.min_timescale 
            * (self.max_timescale / self.min_timescale) ** fraction
        ).to(inputs.device)
        
        if position is None:
            seq_length = inputs.shape[1]
            position = torch.arange(
                seq_length, dtype=torch.float32, device=inputs.device
            )[None, :]
        
        # Adjust dimensions based on input rank
        if len(inputs.shape) == 4:
            position = position[..., None, None]
            timescale = timescale[None, None, None, :]
        elif len(inputs.shape) == 3:
            position = position[..., None]
            timescale = timescale[None, None, :]
        else:
            raise ValueError("Inputs must be of rank 3 or 4.")
        
        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        
        # Apply rotation to pairs of dimensions
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        
        return torch.cat([first_part, second_part], dim=-1)


class PerDimScale(nn.Module):
    """
    Per-dimension scaling for attention queries.
    
    Learns a scaling factor for each dimension to improve attention stability.
    """
    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_factor = (
            1.442695041 / math.sqrt(self.num_dims) 
            * F.softplus(self.per_dim_scale)
        )
        return x * scale_factor


def _torch_dot_product_attention(query, key, value, mask=None):
    """
    Efficient dot-product attention using PyTorch's fused implementation.
    
    Args:
        query: Query tensor of shape (B, L, H, D)
        key: Key tensor of shape (B, L, H, D)
        value: Value tensor of shape (B, L, H, D)
        mask: Attention mask
    
    Returns:
        Attention output of shape (B, L, H, D)
    """
    # Permute from (B, L, H, D) to (B, H, L, D) for PyTorch's expected format
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    
    # Use fused attention kernel with scale=1.0 (no sqrt(d_k) scaling)
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=1.0
    )
    
    # Permute back to (B, L, H, D)
    output = output.permute(0, 2, 1, 3)
    return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE and RMSNorm.
    
    Key features:
    - Rotary positional embeddings for better sequence modeling
    - RMS normalization on queries and keys
    - Optional fused QKV projection for efficiency
    - Support for KV caching during autoregressive decoding
    """
    def __init__(
        self,
        num_heads: int,
        in_features: int,
        use_per_dim_scale: bool = True,
        use_rotary_position_embeddings: bool = True,
        use_bias: bool = False,
        attention_fn: Callable = _torch_dot_product_attention,
        qk_norm: str = "rms",
        fuse_qkv: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.qk_norm = qk_norm
        self.fuse_qkv = fuse_qkv
        
        if self.in_features % self.num_heads != 0:
            raise ValueError(
                f"Model dimension ({self.in_features}) must be divisible by "
                f"number of heads ({self.num_heads})."
            )
        
        # QKV projections (fused or separate)
        if self.fuse_qkv:
            self.qkv_proj = nn.Linear(
                self.in_features, 3 * self.in_features, bias=use_bias
            )
        else:
            self.query = nn.Linear(self.in_features, self.in_features, bias=use_bias)
            self.key = nn.Linear(self.in_features, self.in_features, bias=use_bias)
            self.value = nn.Linear(self.in_features, self.in_features, bias=use_bias)
        
        self.out = nn.Linear(self.in_features, self.in_features, bias=use_bias)
        
        # Query-Key normalization
        if self.qk_norm == "rms":
            self.query_ln = RMSNorm(self.head_dim)
            self.key_ln = RMSNorm(self.head_dim)
        else:
            self.query_ln = nn.Identity()
            self.key_ln = nn.Identity()
        
        # Rotary positional embeddings
        self.use_rotary_position_embeddings = use_rotary_position_embeddings
        if self.use_rotary_position_embeddings:
            self.rotary_position_embedding = RotaryPositionalEmbedding(
                embedding_dims=self.head_dim
            )
        
        # Per-dimension scaling
        self.use_per_dim_scale = use_per_dim_scale
        if use_per_dim_scale:
            self.per_dim_scale = PerDimScale(num_dims=self.head_dim)
    
    def forward(
        self,
        inputs_q: torch.Tensor,
        decode_cache: DecodeCache | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecodeCache | None]:
        """
        Forward pass through multi-head attention.
        
        Args:
            inputs_q: Input tensor of shape (batch, seq_len, hidden_dim)
            decode_cache: Cache for autoregressive decoding
            patch_mask: Boolean mask indicating padded positions
        
        Returns:
            Tuple of (output tensor, updated decode cache)
        """
        b, n_patches, _ = inputs_q.shape
        
        if patch_mask is None:
            patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)
        
        # Project to Q, K, V
        if self.fuse_qkv:
            qkv = self.qkv_proj(inputs_q)
            query, key, value = torch.chunk(qkv, 3, dim=-1)
            query = query.view(b, n_patches, self.num_heads, self.head_dim)
            key = key.view(b, n_patches, self.num_heads, self.head_dim)
            value = value.view(b, n_patches, self.num_heads, self.head_dim)
        else:
            query = self.query(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
            key = self.key(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
            value = self.value(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
        
        # Handle caching for autoregressive decoding
        if decode_cache is None:
            num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1)
            next_index = torch.zeros_like(num_masked, dtype=torch.int32)
        else:
            num_masked = (
                torch.sum(patch_mask.to(torch.int32), dim=-1) 
                + decode_cache.num_masked
            )
            next_index = decode_cache.next_index.clone()
        
        # Apply rotary positional embeddings
        if self.use_rotary_position_embeddings:
            position = (
                torch.arange(n_patches, device=inputs_q.device)[None, :]
                + next_index[:, None]
                - num_masked[:, None]
            )
            query = self.rotary_position_embedding(query, position)
            key = self.rotary_position_embedding(key, position)
        
        # Normalize queries and keys
        query = self.query_ln(query)
        key = self.key_ln(key)
        
        # Apply per-dimension scaling
        if self.use_per_dim_scale:
            query = self.per_dim_scale(query)
        
        # Update cache if in decoding mode
        if decode_cache is not None:
            _, decode_cache_size, _, _ = decode_cache.value.shape
            start = decode_cache.next_index[0]
            end = start + n_patches
            
            # Vectorized cache update
            decode_cache.key[:, start:end] = key
            decode_cache.value[:, start:end] = value
            
            key = decode_cache.key
            value = decode_cache.value
            decode_cache.next_index += n_patches
            decode_cache.num_masked = num_masked
            
            attn_mask = make_attn_mask(
                query_length=n_patches,
                num_all_masked_kv=num_masked,
                query_index_offset=next_index,
                kv_length=decode_cache_size,
            )
        else:
            attn_mask = make_attn_mask(
                query_length=n_patches, 
                num_all_masked_kv=num_masked
            )
        
        # Compute attention
        x = self.attention_fn(query, key, value, mask=attn_mask)
        x = x.reshape(b, n_patches, self.in_features)
        
        # Final output projection
        out = self.out(x)
        return out, decode_cache


class Transformer(nn.Module):
    """
    Single Transformer layer with pre-normalization architecture.
    
    Structure:
    1. Pre-attention normalization → Multi-head attention → Post-attention normalization + residual
    2. Pre-FFN normalization → Feedforward network → Post-FFN normalization + residual
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Attention block with RMS normalization
        if config.attention_norm == "rms":
            self.pre_attn_ln = RMSNorm(num_features=config.model_dims)
            self.post_attn_ln = RMSNorm(num_features=config.model_dims)
        else:
            raise ValueError(f"Layer norm: {config.attention_norm} not supported.")
        
        self.attn = MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.model_dims,
            use_per_dim_scale=True,
            use_rotary_position_embeddings=config.use_rotary_position_embeddings,
            qk_norm=config.qk_norm,
            fuse_qkv=config.fuse_qkv,
        )
        
        # Feedforward block with RMS normalization
        if config.feedforward_norm == "rms":
            self.pre_ff_ln = RMSNorm(num_features=config.model_dims)
            self.post_ff_ln = RMSNorm(num_features=config.model_dims)
        else:
            raise ValueError(f"Layer norm: {config.feedforward_norm} not supported.")
        
        self.ff0 = nn.Linear(
            in_features=config.model_dims,
            out_features=config.hidden_dims,
            bias=config.use_bias,
        )
        self.ff1 = nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.model_dims,
            bias=config.use_bias,
        )
        
        if config.ff_activation == "relu":
            self.activation = nn.ReLU()
        elif config.ff_activation == "swish":
            self.activation = nn.SiLU()
        elif config.ff_activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation: {config.ff_activation} not supported.")
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
        decode_cache: DecodeCache | None = None,
    ) -> tuple[torch.Tensor, DecodeCache | None]:
        """
        Forward pass through transformer layer.
        
        Args:
            input_embeddings: Input embeddings
            patch_mask: Mask for padded positions
            decode_cache: Cache for autoregressive decoding
        
        Returns:
            Tuple of (output embeddings, updated decode cache)
        """
        # Attention sub-layer with residual connection
        attn_output, decode_cache = self.attn(
            inputs_q=self.pre_attn_ln(input_embeddings),
            decode_cache=decode_cache,
            patch_mask=patch_mask,
        )
        attn_output = self.post_attn_ln(attn_output) + input_embeddings
        
        # Feedforward sub-layer with residual connection
        output_embeddings = (
            self.post_ff_ln(
                self.ff1(
                    self.activation(
                        self.ff0(self.pre_ff_ln(attn_output))
                    )
                )
            )
            + attn_output
        )
        
        return output_embeddings, decode_cache


# ============================================================================
# SECTION 4: TIMESFM MODEL CORE
# Main model architecture integrating all components
# ============================================================================

class TimesFM_2p5_200M_torch_module(nn.Module):
    """
    TimesFM 2.5 Core Model with 200M parameters.
    
    Architecture overview:
    1. Tokenizer: Converts time series patches to embeddings
    2. Transformer Stack: 20 layers of self-attention for pattern recognition
    3. Output Projections: Generate point forecasts and quantile predictions
    
    Key innovations:
    - Patch-based processing (32 timesteps per patch)
    - Reversible Instance Normalization for distribution shift handling
    - Autoregressive decoding with KV caching
    - Probabilistic forecasts via quantile regression
    """
    
    config = TimesFM_2p5_200M_Definition()
    
    def __init__(self):
        super().__init__()
        
        # Extract configuration constants for convenience
        self.p = self.config.input_patch_len           # 32: input patch length
        self.o = self.config.output_patch_len          # 128: output patch length
        self.os = self.config.output_quantile_len      # 1024: quantile output length
        self.m = self.o // self.p                      # 4: expansion ratio
        self.x = self.config.stacked_transformers.num_layers  # 20: num layers
        self.h = self.config.stacked_transformers.transformer.num_heads  # 16 heads
        self.md = self.config.stacked_transformers.transformer.model_dims  # 1280 dims
        self.hd = self.md // self.h                    # 80: head dimension
        self.q = len(self.config.quantiles) + 1        # 10: num quantiles (9 + mean)
        self.aridx = self.config.decode_index          # 5: index of median quantile
        
        # Model components
        self.tokenizer = ResidualBlock(self.config.tokenizer)
        
        self.stacked_xf = nn.ModuleList([
            Transformer(self.config.stacked_transformers.transformer)
            for _ in range(self.x)
        ])
        
        self.output_projection_point = ResidualBlock(
            self.config.output_projection_point
        )
        self.output_projection_quantiles = ResidualBlock(
            self.config.output_projection_quantiles
        )
        
        # Device configuration
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_count = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.device_count = 1
    
    def load_checkpoint(self, path: str, **kwargs):
        """
        Loads model weights from a checkpoint file.
        
        Args:
            path: Path to safetensors checkpoint file
            kwargs: Additional options (e.g., torch_compile)
        """
        tensors = load_file(path)
        self.load_state_dict(tensors, strict=True)
        self.to(self.device)
        
        torch_compile = kwargs.get("torch_compile", True)
        if torch_compile:
            logging.info("Compiling model...")
            self = torch.compile(self)
        
        self.eval()
    
    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        decode_caches: list[DecodeCache] | None = None,
    ):
        """
        Forward pass through the model.
        
        Args:
            inputs: Normalized input patches of shape (batch, num_patches, patch_len)
            masks: Boolean masks indicating valid timesteps
            decode_caches: List of caches for each transformer layer
        
        Returns:
            Tuple of (input_embeddings, output_embeddings, point_forecast, quantile_spread)
            and updated decode caches
        """
        # Concatenate inputs and masks for tokenization
        tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
        input_embeddings = self.tokenizer(tokenizer_inputs)
        
        if decode_caches is None:
            decode_caches = [None] * self.x
        
        # Pass through transformer stack
        output_embeddings = input_embeddings
        new_decode_caches = []
        for i, layer in enumerate(self.stacked_xf):
            output_embeddings, new_cache = layer(
                output_embeddings, masks[..., -1], decode_caches[i]
            )
            new_decode_caches.append(new_cache)
        
        # Generate outputs
        output_ts = self.output_projection_point(output_embeddings)
        output_quantile_spread = self.output_projection_quantiles(output_embeddings)
        
        return (
            input_embeddings,
            output_embeddings,
            output_ts,
            output_quantile_spread,
        ), new_decode_caches
    
    def decode(self, horizon: int, inputs, masks):
        """
        Autoregressive decoding for long-horizon forecasting.
        
        Process:
        1. Prefill: Encode all available context
        2. Compute normalization statistics for context
        3. Autoregressively generate future patches
        
        Args:
            horizon: Number of timesteps to forecast
            inputs: Input time series
            masks: Masks indicating missing values
        
        Returns:
            Tuple of (prefill_outputs, quantile_spreads, autoregressive_outputs)
        """
        with torch.no_grad():
            batch_size, context = inputs.shape[0], inputs.shape[1]
            num_decode_steps = (horizon - 1) // self.o
            num_input_patches = context // self.p
            decode_cache_size = num_input_patches + num_decode_steps * self.m
            
            # Reshape into patches
            patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
            patched_masks = torch.reshape(masks, (batch_size, -1, self.p))
            
            # Compute running statistics for normalization
            n = torch.zeros(batch_size, device=inputs.device)
            mu = torch.zeros(batch_size, device=inputs.device)
            sigma = torch.zeros(batch_size, device=inputs.device)
            patch_mu = []
            patch_sigma = []
            
            for i in range(num_input_patches):
                (n, mu, sigma), _ = update_running_stats(
                    n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
                )
                patch_mu.append(mu)
                patch_sigma.append(sigma)
            
            last_n, last_mu, last_sigma = n, mu, sigma
            context_mu = torch.stack(patch_mu, dim=1)
            context_sigma = torch.stack(patch_sigma, dim=1)
            
            # Initialize decode caches for each transformer layer
            decode_caches = [
                DecodeCache(
                    next_index=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
                    num_masked=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
                    key=torch.zeros(batch_size, decode_cache_size, self.h, self.hd, device=inputs.device),
                    value=torch.zeros(batch_size, decode_cache_size, self.h, self.hd, device=inputs.device),
                )
                for _ in range(self.x)
            ]
            
            # Normalize inputs and run through model
            normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
            normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
            
            (_, _, normed_outputs, normed_quantile_spread), decode_caches = self(
                normed_inputs, patched_masks, decode_caches
            )
            
            # Denormalize outputs
            renormed_outputs = torch.reshape(
                revin(normed_outputs, context_mu, context_sigma, reverse=True),
                (batch_size, -1, self.o, self.q),
            )
            renormed_quantile_spread = torch.reshape(
                revin(normed_quantile_spread, context_mu, context_sigma, reverse=True),
                (batch_size, -1, self.os, self.q),
            )[:, -1, ...]
            
            # Autoregressive decoding loop
            ar_outputs = []
            last_renormed_output = renormed_outputs[:, -1, :, self.aridx]
            
            for _ in range(num_decode_steps):
                # Prepare next input patch from previous predictions
                new_patched_input = torch.reshape(
                    last_renormed_output, (batch_size, self.m, self.p)
                )
                new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)
                
                # Update running statistics
                n, mu, sigma = last_n, last_mu, last_sigma
                new_mus, new_sigmas = [], []
                
                for i in range(self.m):
                    (n, mu, sigma), _ = update_running_stats(
                        n, mu, sigma, new_patched_input[:, i], new_mask[:, i]
                    )
                    new_mus.append(mu)
                    new_sigmas.append(sigma)
                
                last_n, last_mu, last_sigma = n, mu, sigma
                new_mu = torch.stack(new_mus, dim=1)
                new_sigma = torch.stack(new_sigmas, dim=1)
                
                # Normalize and predict next patch
                new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
                (_, _, new_normed_output, _), decode_caches = self(
                    new_normed_input, new_mask, decode_caches
                )
                
                new_renormed_output = torch.reshape(
                    revin(new_normed_output, new_mu, new_sigma, reverse=True),
                    (batch_size, self.m, self.o, self.q),
                )
                ar_outputs.append(new_renormed_output[:, -1, ...])
                last_renormed_output = new_renormed_output[:, -1, :, self.aridx]
            
            ar_renormed_outputs = torch.stack(ar_outputs, dim=1) if num_decode_steps > 0 else None
        
        return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs
    
    def forecast_naive(self, horizon: int, inputs: Sequence[np.ndarray]) -> list[np.ndarray]:
        """
        Simple forecasting method for debugging.
        
        Args:
            horizon: Forecast horizon
            inputs: List of input time series
        
        Returns:
            List of forecast arrays
        """
        outputs = []
        for each_input in inputs:
            input_t = torch.tensor(each_input, dtype=torch.float32)
            mask = torch.zeros_like(input_t, dtype=torch.bool)
            
            # Pad to multiple of patch size
            len_front_mask = self.p - (len(each_input) % self.p)
            if len_front_mask < self.p:
                input_t = torch.cat([
                    torch.zeros(len_front_mask, dtype=torch.float32), 
                    input_t
                ], dim=0)
                mask = torch.cat([
                    torch.ones(len_front_mask, dtype=torch.bool), 
                    mask
                ], dim=0)
            
            input_t = input_t[None, ...]
            mask = mask[None, ...]
            
            t_pf, _, t_ar = self.decode(horizon, input_t, mask)
            
            to_concat = [t_pf[:, -1, ...]]
            if t_ar is not None:
                to_concat.append(t_ar.reshape(1, -1, self.q))
            
            torch_forecast = torch.cat(to_concat, dim=1)[:, :horizon, :]
            torch_forecast = torch_forecast.squeeze(0)
            outputs.append(torch_forecast.detach().cpu().numpy())
        
        return outputs


# ============================================================================
# SECTION 5: HIGH-LEVEL API AND UTILITIES
# User-facing interface and helper functions
# ============================================================================

class TimesFM_2p5_200M_torch(
    PyTorchModelHubMixin,
    library_name="timesfm",
    repo_url="https://github.com/google-research/timesfm",
    paper_url="https://arxiv.org/abs/2310.10688",
    docs_url="https://github.com/google-research/timesfm",
    license="apache-2.0",
    pipeline_tag="time-series-forecasting",
    tags=["pytorch", "timeseries", "forecasting", "timesfm-2.5"],
):
    """
    High-level wrapper for TimesFM 2.5 with Hugging Face integration.
    
    Provides convenient methods for:
    - Loading pretrained models
    - Compiling for fast inference
    - Batch forecasting with covariates
    """
    
    DEFAULT_REPO_ID = "google/timesfm-2.5-200m-pytorch"
    WEIGHTS_FILENAME = "model.safetensors"
    
    forecast_config = None
    compiled_decode = None
    global_batch_size = 0
    
    def __init__(self, torch_compile: bool = True, config: Optional[dict] = None, **kwargs):
        self.model = TimesFM_2p5_200M_torch_module()
        self.torch_compile = torch_compile
        if config is not None:
            self._hub_mixin_config = config
    
    def load_checkpoint(self, path: str, **kwargs):
        """Loads model from checkpoint file or directory."""
        if os.path.isdir(path):
            model_file_path = os.path.join(path, self.WEIGHTS_FILENAME)
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(
                    f"{self.WEIGHTS_FILENAME} not found in directory {path}"
                )
        else:
            model_file_path = path
        
        self.model.load_checkpoint(model_file_path, **kwargs)
    
    @classmethod
    def _from_pretrained(cls, *, model_id: str = DEFAULT_REPO_ID, **kwargs):
        """Loads model from Hugging Face Hub or local path."""
        model_file_path = ""
        if os.path.isdir(model_id):
            logging.info("Loading from local directory: %s", model_id)
            model_file_path = os.path.join(model_id, cls.WEIGHTS_FILENAME)
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(
                    f"{cls.WEIGHTS_FILENAME} not found in directory {model_id}"
                )
        else:
            logging.info("Downloading from Hugging Face: %s", model_id)
            model_file_path = hf_hub_download(
                repo_id=model_id,
                filename=cls.WEIGHTS_FILENAME,
                **{k: v for k, v in kwargs.items() 
                   if k in ['revision', 'cache_dir', 'force_download', 
                           'token', 'local_files_only']}
            )
        
        instance = cls(**kwargs)
        logging.info("Loading checkpoint from: %s", model_file_path)
        instance.load_checkpoint(model_file_path, torch_compile=instance.torch_compile)
        return instance
    
    def _save_pretrained(self, save_directory: Union[str, Path]):
        """Saves model to directory."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        weights_path = os.path.join(save_directory, self.WEIGHTS_FILENAME)
        save_file(self.model.state_dict(), weights_path)
    
    def compile(self, forecast_config, **kwargs) -> None:
        """
        Compiles model for optimized inference.
        
        Args:
            forecast_config: Configuration specifying max context, horizon, etc.
        """
        self.global_batch_size = forecast_config.per_core_batch_size * self.model.device_count
        fc = forecast_config
        
        # Validate and adjust configuration
        if fc.max_context % self.model.p != 0:
            new_context = math.ceil(fc.max_context / self.model.p) * self.model.p
            logging.info(
                "Adjusting max context from %d to %d (must be multiple of %d)",
                fc.max_context, new_context, self.model.p
            )
            fc = dataclasses.replace(fc, max_context=new_context)
        
        if fc.max_horizon % self.model.o != 0:
            new_horizon = math.ceil(fc.max_horizon / self.model.o) * self.model.o
            logging.info(
                "Adjusting max horizon from %d to %d (must be multiple of %d)",
                fc.max_horizon, new_horizon, self.model.o
            )
            fc = dataclasses.replace(fc, max_horizon=new_horizon)
        
        if fc.max_context + fc.max_horizon > self.model.config.context_limit:
            raise ValueError(
                f"Context + horizon exceeds limit: {fc.max_context} + {fc.max_horizon} > "
                f"{self.model.config.context_limit}"
            )
        
        self.forecast_config = fc
        
        def _compiled_decode(horizon, inputs, masks):
            """Optimized decoding function with various post-processing options."""
            if horizon > fc.max_horizon:
                raise ValueError(
                    f"Horizon {horizon} exceeds max horizon {fc.max_horizon}"
                )
            
            inputs = torch.from_numpy(np.array(inputs)).to(self.model.device).to(torch.float32)
            masks = torch.from_numpy(np.array(masks)).to(self.model.device).to(torch.bool)
            batch_size = inputs.shape[0]
            
            # Check if inputs are positive
            is_positive = torch.all(inputs >= 0, dim=-1, keepdim=True) if fc.infer_is_positive else None
            
            # Normalize inputs if requested
            if fc.normalize_inputs:
                mu = torch.mean(inputs, dim=-1, keepdim=True)
                sigma = torch.std(inputs, dim=-1, keepdim=True)
                inputs = revin(inputs, mu, sigma, reverse=False)
            else:
                mu, sigma = None, None
            
            # Run model
            pf_outputs, quantile_spreads, ar_outputs = self.model.decode(
                forecast_config.max_horizon, inputs, masks
            )
            
            # Combine prefill and autoregressive outputs
            to_cat = [pf_outputs[:, -1, ...]]
            if ar_outputs is not None:
                to_cat.append(ar_outputs.reshape(batch_size, -1, self.model.q))
            full_forecast = torch.cat(to_cat, dim=1)
            
            # Flip invariance for improved robustness
            def flip_quantile_fn(x):
                return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)
            
            if fc.force_flip_invariance:
                flipped_pf, flipped_qs, flipped_ar = self.model.decode(
                    forecast_config.max_horizon, -inputs, masks
                )
                flipped_qs = flip_quantile_fn(flipped_qs)
                flipped_pf = flip_quantile_fn(flipped_pf)
                
                to_cat = [flipped_pf[:, -1, ...]]
                if flipped_ar is not None:
                    to_cat.append(flipped_ar.reshape(batch_size, -1, self.model.q))
                flipped_full = torch.cat(to_cat, dim=1)
                
                quantile_spreads = (quantile_spreads - flipped_qs) / 2
                pf_outputs = (pf_outputs - flipped_pf) / 2
                full_forecast = (full_forecast - flipped_full) / 2
            
            # Continuous quantile head adjustment
            if fc.use_continuous_quantile_head:
                for qi in [1, 2, 3, 4, 6, 7, 8, 9]:
                    full_forecast[:, :, qi] = (
                        quantile_spreads[:, :fc.max_horizon, qi]
                        - quantile_spreads[:, :fc.max_horizon, 5]
                        + full_forecast[:, :fc.max_horizon, 5]
                    )
            
            full_forecast = full_forecast[:, :horizon, :]
            
            # Return backcast if requested
            if fc.return_backcast:
                full_backcast = pf_outputs[:, :-1, :self.model.p, :].reshape(
                    batch_size, -1, self.model.q
                )
                full_forecast = torch.cat([full_backcast, full_forecast], dim=1)
            
            # Fix quantile crossing (ensure monotonicity)
            if fc.fix_quantile_crossing:
                for i in [4, 3, 2, 1]:
                    full_forecast[:, :, i] = torch.where(
                        full_forecast[:, :, i] < full_forecast[:, :, i + 1],
                        full_forecast[:, :, i],
                        full_forecast[:, :, i + 1],
                    )
                for i in [6, 7, 8, 9]:
                    full_forecast[:, :, i] = torch.where(
                        full_forecast[:, :, i] > full_forecast[:, :, i - 1],
                        full_forecast[:, :, i],
                        full_forecast[:, :, i - 1],
                    )
            
            # Denormalize if needed
            if fc.normalize_inputs:
                full_forecast = revin(full_forecast, mu, sigma, reverse=True)
            
            # Ensure positivity if required
            if is_positive is not None:
                full_forecast = torch.where(
                    is_positive[..., None],
                    torch.maximum(full_forecast, torch.zeros_like(full_forecast)),
                    full_forecast,
                )
            
            full_forecast = full_forecast.detach().cpu().numpy()
            return full_forecast[..., 5], full_forecast
        
        self.compiled_decode = _compiled_decode
    
    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Main forecasting interface.
        
        Args:
            horizon: Number of timesteps to forecast
            inputs: List of input time series
        
        Returns:
            Tuple of (point_forecasts, quantile_forecasts)
        """
        if self.compiled_decode is None:
            raise RuntimeError("Model is not compiled. Please call compile() first.")
        
        assert self.global_batch_size > 0
        assert self.forecast_config is not None
        
        context = self.forecast_config.max_context
        num_inputs = len(inputs)
        
        # Pad to batch size
        if (w := num_inputs % self.global_batch_size) != 0:
            inputs += [np.array([0.0] * 3)] * (self.global_batch_size - w)
        
        output_points = []
        output_quantiles = []
        values = []
        masks = []
        idx = 0
        
        for each_input in inputs:
            # Preprocess: remove leading NaNs and interpolate
            value = linear_interpolation(strip_leading_nans(np.array(each_input)))
            
            if len(value) >= context:
                value = value[-context:]
                mask = np.zeros_like(value, dtype=bool)
            else:
                mask = np.array([True] * (context - len(value)) + [False] * len(value))
                value = np.pad(value, (context - len(value), 0), "constant", constant_values=0.0)
            
            values.append(value)
            masks.append(mask)
            idx += 1
            
            if idx == self.global_batch_size:
                idx = 0
                point_forecast, quantile_forecast = self.compiled_decode(horizon, values, masks)
                output_points.append(point_forecast)
                output_quantiles.append(quantile_forecast)
                values = []
                masks = []
        
        output_points = np.concatenate(output_points, axis=0)
        output_quantiles = np.concatenate(output_quantiles, axis=0)
        
        return output_points[:num_inputs], output_quantiles[:num_inputs]
    
    def forecast_with_covariates(
        self,
        inputs: list[Sequence[float]],
        dynamic_numerical_covariates: dict[str, Sequence[Sequence[float]]] | None = None,
        dynamic_categorical_covariates: dict[str, Sequence[Sequence[Any]]] | None = None,
        static_numerical_covariates: dict[str, Sequence[float]] | None = None,
        static_categorical_covariates: dict[str, Sequence[Any]] | None = None,
        xreg_mode: str = "xreg + timesfm",
        normalize_xreg_target_per_input: bool = True,
        ridge: float = 0.0,
        max_rows_per_col: int = 0,
        force_on_cpu: bool = False,
    ):
        """
        Forecast with exogenous covariates using linear regression residuals.
        
        Two modes:
        - "xreg + timesfm": Fit XReg first, forecast residuals with TimesFM
        - "timesfm + xreg": Forecast with TimesFM first, fit XReg on residuals
        
        Args:
            inputs: List of target time series
            dynamic_numerical_covariates: Time-varying numerical features
            dynamic_categorical_covariates: Time-varying categorical features
            static_numerical_covariates: Static numerical features
            static_categorical_covariates: Static categorical features
            xreg_mode: Regression strategy
            normalize_xreg_target_per_input: Normalize targets per series
            ridge: Ridge regression penalty
            max_rows_per_col: Max rows for linear model
            force_on_cpu: Force CPU for linear model
        
        Returns:
            Tuple of (adjusted_point_forecasts, adjusted_quantile_forecasts)
        """
        if self.forecast_config is None:
            raise ValueError("Model is not compiled. Please call compile() first.")
        elif not self.forecast_config.return_backcast:
            raise ValueError(
                "For XReg, `return_backcast` must be set to True. Please recompile."
            )
        
        # Import here to avoid circular dependency
        from ..utils import xreg_lib
        
        if not (dynamic_numerical_covariates or dynamic_categorical_covariates 
                or static_numerical_covariates or static_categorical_covariates):
            raise ValueError("At least one type of covariate must be provided.")
        
        # Track lengths
        input_lens, train_lens, test_lens = [], [], []
        
        for i, input_ts in enumerate(inputs):
            input_len = len(input_ts)
            input_lens.append(input_len)
            
            if xreg_mode == "timesfm + xreg":
                train_lens.append(max(0, input_len - self.model.p))
            elif xreg_mode == "xreg + timesfm":
                train_lens.append(input_len)
            else:
                raise ValueError(f"Unsupported mode: {xreg_mode}")
            
            if dynamic_numerical_covariates:
                test_lens.append(
                    len(list(dynamic_numerical_covariates.values())[0][i]) - input_len
                )
            elif dynamic_categorical_covariates:
                test_lens.append(
                    len(list(dynamic_categorical_covariates.values())[0][i]) - input_len
                )
            else:
                test_lens.append(self.forecast_config.max_horizon)
            
            if test_lens[-1] > self.forecast_config.max_horizon:
                raise ValueError(
                    f"Inferred horizon {test_lens[-1]} exceeds max_horizon "
                    f"{self.forecast_config.max_horizon}"
                )
        
        # Prepare covariates
        train_dyn_num = collections.defaultdict(list)
        test_dyn_num = collections.defaultdict(list)
        train_dyn_cat = collections.defaultdict(list)
        test_dyn_cat = collections.defaultdict(list)
        
        for covariates, train_cov, test_cov in [
            (dynamic_numerical_covariates, train_dyn_num, test_dyn_num),
            (dynamic_categorical_covariates, train_dyn_cat, test_dyn_cat),
        ]:
            if not covariates:
                continue
            for name, values in covariates.items():
                for in_len, tr_len, val in zip(input_lens, train_lens, values):
                    train_cov[name].append(val[(in_len - tr_len):in_len])
                    test_cov[name].append(val[in_len:])
        
        # Execute chosen mode
        if xreg_mode == "timesfm + xreg":
            # Step 1: TimesFM forecast
            point_outputs, quantile_outputs = self.forecast(
                horizon=self.forecast_config.max_horizon, inputs=inputs
            )
            
            # Step 2: Compute residuals
            targets = [
                np.array(inp)[-tr_len:] - pt_out[:-self.forecast_config.max_horizon][-tr_len:]
                for inp, pt_out, tr_len in zip(inputs, point_outputs, train_lens)
            ]
            
            # Step 3: Normalize if requested
            per_instance_stats = None
            if normalize_xreg_target_per_input:
                targets, per_instance_stats = xreg_lib.normalize(targets)
            
            # Step 4: Fit linear model on residuals
            xregs = xreg_lib.BatchedInContextXRegLinear(
                targets=targets,
                train_lens=train_lens,
                test_lens=test_lens,
                train_dynamic_numerical_covariates=train_dyn_num,
                test_dynamic_numerical_covariates=test_dyn_num,
                train_dynamic_categorical_covariates=train_dyn_cat,
                test_dynamic_categorical_covariates=test_dyn_cat,
                static_numerical_covariates=static_numerical_covariates,
                static_categorical_covariates=static_categorical_covariates,
            ).fit(
                ridge=ridge,
                one_hot_encoder_drop=None if ridge > 0 else "first",
                max_rows_per_col=max_rows_per_col,
                force_on_cpu=force_on_cpu,
                debug_info=False,
                assert_covariates=True,
                assert_covariate_shapes=True,
            )
            
            if normalize_xreg_target_per_input:
                xregs = xreg_lib.renormalize(xregs, per_instance_stats)
            
            xregs = np.array(xregs)
            
            # Step 5: Add XReg predictions to TimesFM forecasts
            new_points = [
                pt_out[-self.forecast_config.max_horizon:][:tl] + xreg
                for pt_out, tl, xreg in zip(point_outputs, test_lens, xregs)
            ]
            new_quantiles = [
                qt_out[-self.forecast_config.max_horizon:][:tl] + xreg[..., None]
                for qt_out, tl, xreg in zip(quantile_outputs, test_lens, xregs)
            ]
        
        else:  # xreg + timesfm
            # Step 1: Fit linear model on targets
            targets = [
                np.array(inp)[-tr_len:]
                for inp, tr_len in zip(inputs, train_lens)
            ]
            
            per_instance_stats = None
            if normalize_xreg_target_per_input:
                targets, per_instance_stats = xreg_lib.normalize(targets)
            
            xregs, xregs_on_context, _, _, _ = xreg_lib.BatchedInContextXRegLinear(
                targets=targets,
                train_lens=train_lens,
                test_lens=test_lens,
                train_dynamic_numerical_covariates=train_dyn_num,
                test_dynamic_numerical_covariates=test_dyn_num,
                train_dynamic_categorical_covariates=train_dyn_cat,
                test_dynamic_categorical_covariates=test_dyn_cat,
                static_numerical_covariates=static_numerical_covariates,
                static_categorical_covariates=static_categorical_covariates,
            ).fit(
                ridge=ridge,
                one_hot_encoder_drop=None if ridge > 0 else "first",
                max_rows_per_col=max_rows_per_col,
                force_on_cpu=force_on_cpu,
                debug_info=True,
                assert_covariates=True,
                assert_covariate_shapes=True,
            )
            
            # Step 2: Forecast residuals with TimesFM
            point_outputs, quantile_outputs = self.forecast(
                horizon=self.forecast_config.max_horizon,
                inputs=[
                    target - xreg_ctx
                    for target, xreg_ctx in zip(targets, xregs_on_context)
                ],
            )
            
            # Step 3: Add back XReg component
            new_points = [
                pt_out[-self.forecast_config.max_horizon:][:tl] + xreg
                for pt_out, tl, xreg in zip(point_outputs, test_lens, xregs)
            ]
            new_quantiles = [
                qt_out[-self.forecast_config.max_horizon:][:tl] + xreg[..., None]
                for qt_out, tl, xreg in zip(quantile_outputs, test_lens, xregs)
            ]
            
            if normalize_xreg_target_per_input:
                new_points = xreg_lib.renormalize(new_points, per_instance_stats)
                new_quantiles = xreg_lib.renormalize(new_quantiles, per_instance_stats)
        
        return new_points, new_quantiles