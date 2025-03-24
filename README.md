# Time Series Financial Data Transformers

## Overview

This repository contains two projects centered on using Transformer architectures for time series forecasting, specifically focused on stock market data. The projects utilize advanced deep learning techniques to predict future stock prices based on historical data, enhancing predictive accuracy through innovative model designs and optimization strategies.

## Project 1: Univariate Time Series Transformer

### Description

The first project implements a Transformer model designed for univariate time series forecasting. Using NVIDIA stock market data, the model predicts future closing prices by analyzing past closing price patterns.

### Key Features

- **Data Preparation**: Loads and preprocesses stock market data, extracting the 'Close' price for modeling.
- **Deep Learning Architecture**: 
  - Utilizes an Enhanced Transformer model with components such as Learnable Positional Encoding and ProbSparse Attention to efficiently handle time series data.
  - Implements a sequence-based approach where historical prices are organized into sequences for model input.
- **Hyperparameter Optimization**: 
  - Leverages Optuna for dynamic tuning of hyperparameters like model dimensions, attention heads, dropout rates, and learning rates to maximize model performance.
- **Training and Evaluation**: 
  - Trains the model with early stopping based on validation loss, ensuring robust performance on unseen data.
  - Evaluates the model's accuracy using root mean squared error (RMSE).

## Project 2: Multivariate Time Series Transformer

### Description

The second project expands on the first by utilizing a Transformer model specifically designed for multivariate time series forecasting using IBM's `tsfm` library and Hugging Face's `transformers`. This model leverages a different architecture known as PatchTST, optimized for capturing complex temporal patterns from multiple input features.

### Key Features

- **PatchTST Architecture**: 
  - This architecture optimizes learning by extracting relevant patches from historical data (context) for improved efficiency and performance.
- **Dynamic Hyperparameter Tuning**: 
  - Users can customize key hyperparameters, such as context length, patch length, model complexity (including `d_model`, number of attention heads, and hidden layers), enhancing the model's adaptability to various datasets.
- **Integrated Training Framework**: 
  - Utilizes the Trainer class for seamless training and evaluation, including built-in early stopping and performance tracking through Weights & Biases.
- **Scalable Data Preprocessing**: 
  - Supports preparation of train, validation, and test datasets to ensure the model generalizes well across different segments of the data.

### Usage Steps

1. **Data Preparation**: 
   - Load stock data and preprocess it for model training, ensuring it is split into appropriate training, validation, and test datasets.
   
   ```python
   data = pd.read_csv("path/to/NVDA_ticker.csv", parse_dates=['Date'])
   ```

2. **Train the Model**: 
   - Execute the training process using predefined hyperparameters.

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

3. **Evaluate and Save the Model**: 
   - Evaluate the model's performance and save the trained model for future use.

   ```python
   results = trainer.evaluate(test_dataset)
   trainer.save_model(save_dir)
   ```

4. **Visualize Metrics**: 
   - Includes functionality to visualize evaluation metrics, helping to analyze model performance across epochs.

### Dependencies

- Python 3.8+
- Hugging Face `transformers`
- IBM `tsfm` library
- Weights & Biases for experiment tracking
- Additional packages as specified in the requirements.txt

## Results

Both projects allow for monitoring and visualization of model performance metrics throughout the training process. The final mean squared error (MSE) loss on the test set provides a comprehensive view of how well the models can predict future stock prices based on historical data.

