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

### ⌚ Univariate Regressor Model
1. **Tokenization**: Input series are padded and split into patches of 32 timesteps.
2. **Normalization**: RevIN normalizes each patch based on running statistics.
3. **Embedding**: A residual MLP projects patches into a 1280-dimensional space.
4. **Transformer Stack**: 20 layers of self-attention with causal masking process the sequence.
5. **Decoding**: 
   - **Prefill**: Encodes the entire context history.
   - **Autoregressive Loop**: Generates future patches one by one, updating normalization stats and KV caches dynamically.
6. **Output Head**: Projects embeddings back to time domain for point and quantile forecasts.

### ⏱️ Multivariate Regressor Model
1. **Input Projection**: An `InputResidualMLP` projects the multi-feature vector into an embedding space, mixing variables immediately.
2. **Patching**: The temporal sequence is divided into patches to reduce sequence length.
3. **Transformer Stack**: 
   - Layers alternate between **Temporal Attention** (causal, looking back in time) and **Variate Attention** (non-causal, looking across features).
   - Uses **SwiGLU** feedforward networks for improved non-linearity.
4. **Global Pooling**: Mean pooling aggregates information across the temporal dimension.
5. **Output Head**: A `LinearReadout` layer maps the pooled embedding to the final multivariate forecast matrix `(Batch, 9, Horizon)`.



