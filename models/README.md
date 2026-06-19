# Time Series Foundation Models: Univariate & Multivariate Transformers

<br>




## Differences

| Feature | UVR | MVR |
| :--- | :--- | :--- |
| **Input Type** | Univariate (Single variable per series) | Multivariate (Multiple correlated variables) |
| **Primary Goal** | Zero-shot forecasting, robustness to distribution shift | Modeling complex inter-variable dependencies |
| **Normalization** | Reversible Instance Norm (RevIN) | RMSNorm with MuP scaling |
| **Attention** | Standard Multi-Head Attention | Grouped Query Attention (GQA) |
| **Positional Encoding** | Standard RoPE | Extrapolatable RoPE with xPos scaling |
| **Output** | Point forecast + 9 Quantiles | Multi-variable point forecasts |
| **Training Stability** | Standard initialization | MuP (Unit Scaling) for width-invariant training |

---

<br>

## Architecture

### UVR
1. **Tokenization**: Input series are padded and split into patches of 32 timesteps.
2. **Normalization**: RevIN normalizes each patch based on running statistics.
3. **Embedding**: A residual MLP projects patches into a 1280-dimensional space.
4. **Transformer Stack**: 20 layers of self-attention with causal masking process the sequence.
5. **Decoding**: 
   - **Prefill**: Encodes the entire context history.
   - **Autoregressive Loop**: Generates future patches one by one, updating normalization stats and KV caches dynamically.
6. **Output Head**: Projects embeddings back to time domain for point and quantile forecasts.

### MVR 
1. **Input Projection**: An `InputResidualMLP` projects the multi-feature vector into an embedding space, mixing variables immediately.
2. **Patching**: The temporal sequence is divided into patches to reduce sequence length.
3. **Transformer Stack**: 
   - Layers alternate between **Temporal Attention** (causal, looking back in time) and **Variate Attention** (non-causal, looking across features).
   - Uses **SwiGLU** feedforward networks for improved non-linearity.
4. **Global Pooling**: Mean pooling aggregates information across the temporal dimension.
5. **Output Head**: A `LinearReadout` layer maps the pooled embedding to the final multivariate forecast matrix `(Batch, 9, Horizon)`.

<br>

## Details

### ⌚ Univariate Regressor Model



#### Key Features:
- **Patch-Based Processing**: Converts raw time series into patches of 32 timesteps, enabling efficient long-sequence modeling.
- **Reversible Instance Normalization (RevIN)**: Handles distribution shifts by normalizing inputs and denormalizing outputs, improving robustness across different data scales.
- **Autoregressive Decoding**: Uses KV-caching and causal masking to generate long-horizon forecasts step-by-step.
- **Probabilistic Forecasting**: Outputs 9 quantiles (0.1–0.9) plus the mean, providing uncertainty estimates alongside point predictions.
- **Zero-Shot Capability**: Pre-trained on large-scale datasets, allowing it to forecast new series without task-specific fine-tuning.

#### Architecture Stats:
- **Parameters**: ~200 Million
- **Layers**: 20 Transformer blocks
- **Heads**: 16 Attention heads
- **Hidden Dim**: 1280

<br>

### ⏱️ Multivariate Regressor Model


#### Key Features:
- **Cross-Variable Mixing**: The input projection layer immediately mixes information from all input features, allowing the model to learn inter-variable dependencies.
- **MuP Scaling**: Implements Maximal Update Parametrization (unit scaling) to ensure stable training gradients regardless of model width or batch size.
- **Grouped Query Attention (GQA)**: Uses efficient attention mechanisms with fewer key/value heads than query heads, reducing memory usage while maintaining performance.
- **Extrapolatable RoPE**: Enhanced Rotary Positional Embeddings with xPos scaling for better generalization to sequence lengths not seen during training.
- **Alternating Attention Layers**: Switches between temporal attention (causal) and variate attention (non-causal) to capture both time-dependent patterns and cross-feature correlations.

#### Architecture Stats:
- **Input**: 11 features (configurable)
- **Output**: 9 target variables over a 7-step horizon (configurable)
- **Embedding Dim**: 256
- **Layers**: 8 Transformer blocks
- **Heads**: 8 Attention heads

<br>

