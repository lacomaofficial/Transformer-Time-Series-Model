# Time Series Transformer for Forecasting

This repository contains code to train a Transformer model specifically designed for time series forecasting using IBM's `tsfm` library and Hugging Face’s `transformers`. The model is built with NVIDIA stock data as a sample, aiming to predict future stock prices based on past historical data. 

## Overview

This model uses **PatchTST**, a Transformer architecture optimized for time series forecasting. With a direct forecasting approach, the model leverages self-attention to learn temporal patterns and make predictions based on them.

### Key Features
- Patch extraction from historical data (context) for more efficient learning
- Dynamic hyperparameter tuning and customization options
- Integrated early stopping and performance tracking with `Weights & Biases`
- Scalable data preprocessing for train, validation, and test datasets


## Usage

### 1. Data Preparation
The code loads stock data, splits it into training, validation, and test sets, and preprocesses it for time series modeling.

   ```python
   # Load and preprocess data
   data = pd.read_csv("path/to/NVDA_ticker.csv", parse_dates=['Date'])
   ```

### 2. Train the Model
Run the `trainer.train()` function to start training with pre-defined hyperparameters:

```python
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback]
)
trainer.train()
```

### 3. Evaluate and Save the Model
After training, the model can be evaluated and saved:

```python
results = trainer.evaluate(test_dataset)
trainer.save_model(save_dir)
```

### 4. Visualize Metrics
The script includes a function to visualize evaluation metrics as a bar plot:

```python
plot_evaluation_metrics(results)
```

## Configuration

The main hyperparameters of the Transformer model are set in `PatchTSTConfig`, including:
- `context_length` – History window length for each input sample
- `patch_length` – Length of data segments used for each prediction
- `d_model`, `num_attention_heads`, and `num_hidden_layers` for model complexity

## Dependencies

- Python 3.8+
- Hugging Face `transformers`
- IBM `tsfm` library
- `Weights & Biases` for experiment tracking
- Other packages listed in `requirements.txt`

## Results

After training, you can monitor the loss through each epoch in `Weights & Biases`, while the final MSE loss on the test set provides an overview of the model's performance.

