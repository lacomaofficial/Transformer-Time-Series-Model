# Multivariate time series Transformer-based model



import math
from typing import Any, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# -----------------------------------------------------------------------------
# PART 1: dd_unit_scaling Package Implementation
# -----------------------------------------------------------------------------

# Note: In a real environment, 'unit_scaling' and 'dion' are external dependencies.
# We assume they are installed. If not, imports will fail.
try:
    import unit_scaling as uu_base
    from unit_scaling import functional as _U_upstream
    from unit_scaling.constraints import apply_constraint
    from unit_scaling.optim import _get_fan_in as _get_fan_in_base, lr_scale_for_depth, scaled_parameters
except ImportError:
    raise ImportError("Please install 'unit_scaling' and 'dion' packages.")

# -------------------------------------------------------------------------
# Module: scale.py
# Compile-friendly reimplementation of unit_scaling's scale_fwd / scale_bwd.
# -------------------------------------------------------------------------

class _ScaledGrad(torch.autograd.Function):
    """Apply different scales in forward and backward passes (compile-friendly)."""

    @staticmethod
    def forward(X: torch.Tensor, fwd_scale: float, bwd_scale: float) -> torch.Tensor:
        return fwd_scale * X  # type: ignore[return-value]

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple, output: torch.Tensor) -> None:
        _, _, bwd_scale = inputs
        ctx.bwd_scale = bwd_scale

    @staticmethod
    def backward(ctx: Any, grad_Y: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        return ctx.bwd_scale * grad_Y, None, None


def scale_fwd(input: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale a tensor in the forward pass only (gradient is unchanged)."""
    return _ScaledGrad.apply(input, scale, 1.0)  # type: ignore[return-value]


def scale_bwd(input: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale a tensor's gradient in the backward pass only (forward is identity)."""
    return _ScaledGrad.apply(input, 1.0, scale)  # type: ignore[return-value]


# -------------------------------------------------------------------------
# Module: functional.py
# World-size-aware unit-scaled functional operations.
# -------------------------------------------------------------------------

# Global gradient accumulation steps — set once at script startup.
GRAD_ACCUMULATION_STEPS = 1

# Cached world size — defaults to 1 (single-GPU).
_CACHED_WORLD_SIZE: int = 1


def set_grad_accumulation_steps(steps: int) -> None:
    """Set the gradient accumulation steps for unit scaling."""
    global GRAD_ACCUMULATION_STEPS
    if steps < 1:
        raise ValueError(f"accumulate_grad_batches must be >= 1, got {steps}")
    GRAD_ACCUMULATION_STEPS = steps


def init_world_size_cache(world_size: int = 1) -> None:
    """Set the cached distributed world size."""
    global _CACHED_WORLD_SIZE
    _CACHED_WORLD_SIZE = world_size


def _get_effective_batch_multiplier() -> int:
    """Get the multiplier for GLOBAL batch."""
    return GRAD_ACCUMULATION_STEPS * _CACHED_WORLD_SIZE


def residual_split(
    input: torch.Tensor, tau: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split into (residual, skip) with τ-weighted backward scaling."""
    denom = (1 + tau**2) ** 0.5
    residual = scale_bwd(input, tau / denom)
    skip = scale_bwd(input, 1 / denom)
    return residual, skip


def residual_add(
    residual: torch.Tensor, skip: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """Combine residual + skip with τ-weighted forward scaling."""
    denom = (1 + tau**2) ** 0.5
    residual = scale_fwd(residual, tau / denom)
    skip = scale_fwd(skip, 1 / denom)
    return residual + skip


def _unscaled_silu(x: torch.Tensor, mult: float = 1.0) -> torch.Tensor:
    if mult == 1.0:
        return F.silu(x)
    return x * F.sigmoid(x * mult)


def silu_glu(
    input: torch.Tensor, gate: torch.Tensor, mult: float = 1.0
) -> torch.Tensor:
    """Unit-scaled gated linear unit: ``input * silu(gate)``."""
    alpha = 1.0 / (1.0 + 1.0 / (mult**2.0))
    scale = math.exp(alpha * math.log(2.0**0.5) + (1.0 - alpha) * math.log(2.0))
    input = scale_bwd(input, scale)
    gate = scale_bwd(gate, scale)
    output = input * _unscaled_silu(gate, mult=mult)
    return scale_fwd(output, scale)


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    constraint: Optional[str] = "to_output_scale",
    scale_power: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """World-size-aware unit-scaled linear transformation."""
    fan_out, fan_in = weight.shape
    effective_multiplier = _get_effective_batch_multiplier()
    global_numel = input.numel() * effective_multiplier
    batch_size = global_numel // fan_in

    output_scale = 1.0 / fan_in ** scale_power[0]
    grad_input_scale = 1.0 / fan_out ** scale_power[1]
    grad_weight_scale = grad_bias_scale = 1.0 / batch_size ** scale_power[2]

    if constraint is not None:
        output_scale, grad_input_scale = apply_constraint(
            constraint, output_scale, grad_input_scale
        )

    input = scale_bwd(input, grad_input_scale)
    weight = scale_bwd(weight, grad_weight_scale)
    bias = scale_bwd(bias, grad_bias_scale) if bias is not None else None
    output = F.linear(input, weight, bias)
    return scale_fwd(output, output_scale)


def rms_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Accumulation-aware unit-scaled RMS normalization."""
    if weight is not None:
        effective_multiplier = _get_effective_batch_multiplier()
        global_numel = input.numel() * effective_multiplier
        scale = math.sqrt(math.prod(normalized_shape) / global_numel)
        weight = scale_bwd(weight, scale)
    return _U_upstream._unscaled_rms_norm(input, normalized_shape, weight, eps=eps)


def softplus(
    x: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    constraint: Optional[str] = None,
) -> torch.Tensor:
    """Unit-scaled softplus."""
    y_scale = 1.0 / 0.52103
    grad_input_scale = 1.0 / 0.20833444

    if constraint is not None:
        y_scale, grad_input_scale = apply_constraint(
            constraint, y_scale, grad_input_scale
        )

    x = scale_bwd(x, grad_input_scale)
    output = F.softplus(x, beta=beta, threshold=threshold)
    return scale_fwd(output, y_scale)


def per_dim_scale(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Accumulation-aware unit-scaled per-dimension scaling."""
    output_scale = 0.52103
    grad_input_scale = 0.52103
    effective_multiplier = _get_effective_batch_multiplier()
    global_numel = input.numel() * effective_multiplier
    grad_scale = math.sqrt(input.shape[-1] / global_numel)
    weight = scale_bwd(weight, grad_scale)
    input = scale_bwd(input, grad_input_scale)
    return scale_fwd(input * weight, output_scale)


# -------------------------------------------------------------------------
# Module: _modules.py
# World-size-aware unit-scaled nn.Module wrappers.
# -------------------------------------------------------------------------

class Linear(torch.nn.Linear):
    """World-size-aware unit-scaled Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = "to_output_scale",
        weight_mup_type: str = "weight",
        scale_power: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.constraint = constraint
        self.scale_power = scale_power
        # Note: uu.Parameter is from unit_scaling package
        self.weight = uu_base.Parameter(self.weight.data, mup_type=weight_mup_type)
        if self.bias is not None:
            self.bias = uu_base.Parameter(self.bias.data, mup_type="bias")

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear(input, self.weight, self.bias, self.constraint, self.scale_power)


class LinearReadout(Linear):
    """World-size-aware unit-scaled LinearReadout layer (final output projection)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = None,
        weight_mup_type: str = "output",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
            constraint=constraint,
            weight_mup_type=weight_mup_type,
            scale_power=(1.0, 0.5, 0.5),
        )


class RMSNorm(torch.nn.RMSNorm):
    """World-size-aware unit-scaled RMSNorm."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = False,
        include_weight: Optional[bool] = None,
    ) -> None:
        weight_mode = (
            include_weight if include_weight is not None else elementwise_affine
        )

        super().__init__(normalized_shape, eps=eps, elementwise_affine=bool(weight_mode))
        self.dim = (
            normalized_shape
            if isinstance(normalized_shape, int)
            else normalized_shape[0]
        )

        if weight_mode:
            self.weight = uu_base.Parameter(torch.ones(self.dim), mup_type="norm")
        else:
            self.weight = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return rms_norm(
            input,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            eps=self.eps,
        )


class PerDimScale(torch.nn.Module):
    """Learned per-dimension scaling with unit-scaled gradients."""

    def __init__(self, dim: int):
        super().__init__()
        self.per_dim_scale = uu_base.Parameter(torch.zeros(dim), mup_type="norm")

    _param_grad_compensation: float = math.log(2.0) / (0.5 / 0.20833444)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        p = scale_bwd(self.per_dim_scale, self._param_grad_compensation)
        r = (softplus(p) / math.log(2.0)).to(input.dtype)
        return per_dim_scale(input, r)


# -------------------------------------------------------------------------
# Module: optim.py
# MuP-aware optimizer with FSDP2 support.
# -------------------------------------------------------------------------

_UMUP_METADATA_BY_NAME: Dict[str, Dict[str, Any]] = {}


def cache_fan_values(named_parameters) -> None:
    """Cache MuP metadata by parameter name before FSDP wrapping."""
    _UMUP_METADATA_BY_NAME.clear()
    for name, param in named_parameters:
        if not hasattr(param, "mup_type"):
            continue
        # Note: _get_fan_in is defined below in this consolidated script
        fan_in = _get_fan_in(name, param)
        fan_out = param.shape[0] if param.ndim >= 2 else 1
        _UMUP_METADATA_BY_NAME[name] = {
            "mup_type": param.mup_type,
            "mup_scaling_depth": getattr(param, "mup_scaling_depth", None),
            "fan_in": fan_in,
            "fan_out": fan_out,
        }


def get_cached_metadata(param_name: str) -> Dict[str, Any]:
    """Get cached MuP metadata for a parameter by name."""
    return _UMUP_METADATA_BY_NAME.get(param_name, {})


def _get_fan_in(param_name: str, param: torch.Tensor) -> int:
    """Get fan-in, checking cache first then falling back to unit_scaling."""
    metadata = get_cached_metadata(param_name)
    if "fan_in" in metadata:
        return metadata["fan_in"]
    return _get_fan_in_base(param) if param.ndim >= 2 else 1


def _lr_scale_func_adam(param: torch.Tensor) -> float:
    """Calculate the LR scaling factor for AdamW with FSDP2 support."""
    if not hasattr(param, "mup_type"):
        return 1.0

    mup_type = param.mup_type
    scale = lr_scale_for_depth(param)

    if mup_type in ("bias", "norm", "output"):
        return scale
    elif mup_type == "weight":
        fan_in = getattr(param, "_original_fan_in", None)
        if fan_in is None:
            fan_in = _get_fan_in_base(param) if param.ndim >= 2 else 1
        return scale * fan_in**-0.5
    else:
        return scale


def _lr_scale_func_muon(param: torch.Tensor) -> float:
    """LR scaling for Muon-family optimizers."""
    if not hasattr(param, "mup_type"):
        return 1.0
    return lr_scale_for_depth(param)


class Dion2:
    """Dion2 optimizer with u-MuP LR scaling and FSDP2 support."""

    def __new__(
        cls,
        params,
        *,
        lr: float = 0.02,
        fraction: float = 0.5,
        ef_decay: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        epsilon: float = 1e-7,
        distributed_mesh=None,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        use_triton: bool = True,
    ):
        try:
            from dion import Dion2 as _Dion2
        except ImportError:
            raise ImportError("Please install 'dion' package.")

        params = scaled_parameters(
            params,
            _lr_scale_func_muon,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )

        return _Dion2(
            params,
            distributed_mesh=distributed_mesh,
            lr=lr,
            fraction=fraction,
            ef_decay=ef_decay,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            adjust_lr=adjust_lr,
            use_triton=use_triton,
        )


class NorMuon:
    """NorMuon optimizer with u-MuP LR scaling and FSDP2 support."""

    def __new__(
        cls,
        params,
        *,
        lr: float = 0.02,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        epsilon: float = 1e-7,
        distributed_mesh=None,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        nesterov: bool = True,
        cautious_wd: bool = True,
        adjust_lr: Optional[str] = "spectral_norm",
        use_polar_express: bool = True,
        use_triton: bool = True,
    ):
        try:
            from dion import NorMuon as _NorMuon
        except ImportError:
            raise ImportError("Please install 'dion' package.")

        params = scaled_parameters(
            params,
            _lr_scale_func_muon,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )

        return _NorMuon(
            params,
            distributed_mesh=distributed_mesh,
            lr=lr,
            mu=mu,
            muon_beta2=muon_beta2,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov,
            cautious_wd=cautious_wd,
            adjust_lr=adjust_lr,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
        )


class AdamW(torch.optim.AdamW):
    """World-size-aware AdamW optimizer with u-MuP support."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        *args,
        weight_decay: float = 0.0,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        **kwargs,
    ) -> None:
        params = scaled_parameters(
            params,
            _lr_scale_func_adam,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )
        super().__init__(params, *args, **kwargs)


# -------------------------------------------------------------------------
# Namespace: dd_unit_scaling
# Simulating the package structure for the model import
# -------------------------------------------------------------------------

class dd_unit_scaling:
    Linear = Linear
    LinearReadout = LinearReadout
    RMSNorm = RMSNorm
    PerDimScale = PerDimScale
    
    class functional:
        silu_glu = silu_glu
        residual_split = residual_split
        residual_add = residual_add
        linear = linear
        rms_norm = rms_norm
        softplus = softplus
        per_dim_scale = per_dim_scale
        set_grad_accumulation_steps = set_grad_accumulation_steps
        init_world_size_cache = init_world_size_cache

# Alias for convenience within this script
uu = dd_unit_scaling
U = dd_unit_scaling.functional


# -----------------------------------------------------------------------------
# PART 2: Multivariate Regressor Model (MVR_Model)
# -----------------------------------------------------------------------------

class ExtrapolatableRotaryProjection(nn.Module):
    """RoPE with xPos scaling for length extrapolation."""
    def __init__(self, dim: int, base: float = 10000.0, xpos_scale_base: int = 256, xpos_scale_exponent: float = 1.0):
        super().__init__()
        self.dim = dim
        self.xpos_scale_base = xpos_scale_base
        self.xpos_scale_exponent = xpos_scale_exponent
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        xpos_base_scale = (torch.arange(0, dim, 2).float() + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("xpos_base_scale", xpos_base_scale, persistent=False)

    def forward(self, x: torch.Tensor, seq_ids: Optional[torch.Tensor] = None):
        B, H, L, D = x.shape
        if seq_ids is None:
            t = torch.arange(L, device=x.device, dtype=torch.float32)
        else:
            t = seq_ids.float()
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.float())
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
            
        x_rope = (x * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(x) * sin.unsqueeze(0).unsqueeze(0))
        
        if seq_ids is None:
            seq_ids = torch.arange(L, device=x.device, dtype=torch.float32)
        
        max_pos = seq_ids.max()
        center = torch.div(max_pos + 1, 2, rounding_mode="floor")
        power = (seq_ids.float() - center) / self.xpos_scale_base
        
        scale_half = self.xpos_base_scale.unsqueeze(0) ** power.unsqueeze(-1)
        scale_full = scale_half.repeat(1, 2)
        scale_full = scale_full ** self.xpos_scale_exponent
        scale_full = scale_full.unsqueeze(0).unsqueeze(0)
        
        return x_rope * scale_full.to(x.dtype)

class InputResidualMLP(nn.Module):
    """Residual MLP with MuP scaling using τ-rule residual connections."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_p: float = 0.0, bias: bool = True):
        super().__init__()
        self.tau = 1.0
        self.linear1 = uu.Linear(in_dim, hidden_dim, bias=bias, constraint=None)
        self.linear2 = uu.Linear(hidden_dim, out_dim, bias=bias)
        self.skip_proj = uu.Linear(in_dim, out_dim, bias=bias, constraint=None)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main, x_skip = U.residual_split(x, self.tau)
        h = F.silu(self.linear1(x_main))
        h = self.dropout(self.linear2(h))
        skip = self.skip_proj(x_skip)
        return U.residual_add(h, skip, self.tau)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = uu.Linear(d_model, d_ff, bias=False)
        self.w2 = uu.Linear(d_ff, d_model, bias=False)
        self.w3 = uu.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GQAWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None, is_causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal

        self.q_proj = uu.Linear(d_model, d_model, bias=False)
        self.k_proj = uu.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = uu.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = uu.Linear(d_model, d_model, bias=False)
        self.rope = ExtrapolatableRotaryProjection(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q), self.rope(k)
        
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=self.is_causal)
        return self.out_proj(attn.transpose(1, 2).contiguous().view(B, L, D))

class MVR_Model(nn.Module):
    def __init__(self, feature_dim=11, embed_dim=256, num_heads=8, num_layers=8,
                 patch_size=3, context_length=96, horizon=7,
                 layer_group_size=2, num_variate_layers_per_group=1):
        super().__init__()
        self.patch_size = patch_size
        self.horizon = horizon
        
        self.input_norm = uu.RMSNorm(feature_dim, eps=1e-6, include_weight=True)
        self.input_proj = InputResidualMLP(
            in_dim=feature_dim, 
            hidden_dim=4 * embed_dim, 
            out_dim=embed_dim, 
            dropout_p=0.0, 
            bias=True
        )
        
        self.patch_proj = uu.Linear(patch_size * embed_dim, embed_dim, bias=False)
        self.patch_norm = uu.RMSNorm(embed_dim, eps=1e-6, include_weight=True)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_variate = (i % layer_group_size) < num_variate_layers_per_group
            self.layers.append(nn.ModuleDict({
                'norm1': uu.RMSNorm(embed_dim, eps=1e-6, include_weight=True),
                'attn': GQAWithRoPE(embed_dim, num_heads, is_causal=not is_variate),
                'norm2': uu.RMSNorm(embed_dim, eps=1e-6, include_weight=True),
                'ffn': SwiGLU(embed_dim, embed_dim * 4),
            }))

        self.head = nn.Sequential(
            uu.Linear(embed_dim, embed_dim, bias=False),
            uu.RMSNorm(embed_dim, eps=1e-6, include_weight=True),
            nn.SiLU(),
            uu.LinearReadout(embed_dim, 9 * horizon, bias=True)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (uu.Linear, uu.LinearReadout)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        assert T % self.patch_size == 0
        x = x.reshape(B, T // self.patch_size, self.patch_size * x.shape[-1])
        x = self.patch_norm(self.patch_proj(x))

        for layer in self.layers:
            r = x; x = layer['norm1'](x); x = layer['attn'](x) + r
            r = x; x = layer['norm2'](x); x = layer['ffn'](x) + r

        return self.head(x.mean(dim=1)).reshape(B, 9, self.horizon)