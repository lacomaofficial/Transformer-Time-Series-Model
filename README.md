# Time Series Transformers Models

A collection of organized transformer architectures and experimental notebooks for time series analysis, ranging from foundation models to LLM integrations.

## 📂 Structure

### `models/` — Core Architectures
Clean, educational implementations of two distinct forecasting approaches:
*   **Univariate Model (UVR):** Novel architecture with patch-based processing, RevIN normalization, and probabilistic quantile outputs.
*   **Multivariate Model (MVR):** MuP-scaled transformer with Grouped Query Attention (GQA) and alternating temporal/variate layers for cross-variable dependency modeling.

### `notebooks/` — Experimental Research
Exploratory notebooks testing various deep learning and novel LLM applications:
*   **Baselines:** `deep-classifier-tsm.ipynb` & `trend-classifier-tsm.ipynb`
*   **Multivariate:** `multivariate-gt.ipynb` & `multivariate-yf.ipynb` (Yahoo Finance)
*   **LLM Integration:** `qwen-classifier-tsm.ipynb` (Using Qwen for time series classification)


---
*For educational and research purposes only.*
