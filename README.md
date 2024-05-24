##Transformer-based Time Series Model (Stock Price Prediction )

### Overview
This Python code implements a transformer-based time series model to predict stock prices using historical market data. The model is trained on features derived from technical indicators such as Simple Moving Averages (SMA), Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Bollinger Bands, Stochastic Oscillator, and combined trading signals.

### Libraries Used
- PyTorch: For building and training neural networks.
- NumPy: For numerical computing.
- Pandas: For data manipulation and analysis.
- yfinance: For fetching historical market data.
- Matplotlib: For data visualization.
- scikit-learn: For preprocessing and evaluation.

### Workflow
1. **Data Preprocessing**: Fetch historical market data using `yfinance` and preprocess the data by normalizing features using `MinMaxScaler`.
2. **Sequence Creation**: Generate sequences of data samples with fixed sequence length using the `create_sequences` function.
3. **Data Splitting**: Split the data into training, validation, and test sets using `train_test_split`.
4. **Data Loading**: Convert the data into PyTorch tensors and create DataLoader objects for efficient batch processing.
5. **Model Definition**: Define a Transformer-based time series model using the `TimeSeriesTransformer` class.
6. **Model Training**: Train the model using the defined hyperparameters and optimization algorithm.
7. **Hyperparameter Optimization**: Use Bayesian optimization with Gaussian processes to search for the best hyperparameters.
8. **Final Model Evaluation**: Evaluate the final trained model on the test set and compute the test loss and R2 score.

### Hyperparameter Optimization
- The hyperparameters optimized include learning rate, number of layers, and dropout rate.
- Bayesian optimization with Gaussian processes is used to find the best hyperparameters that minimize the negative R2 score.

![image](https://github.com/lacomaofficial/Transformers/assets/132283879/73efde58-6c52-42e8-b544-1c3221c39b04)


### Example
```python
# Fetch and preprocess data
data = pd.read_csv('nvda_tf.csv',index_col='Date')
# Define your independent variables (features)
features = ['SMA_Signal', 'MACD_Signal', 'RSI_Signal', 'BB_Signal',
       'Stochastic_Signal', 'Combined_Signal', 'Volume']
# Normalize the features
data[features] = scaler.fit_transform(data[features])
# Create sequences
X, y = create_sequences(data.to_numpy(), sequence_length)
# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# Define model, criterion, and optimizer
model = TimeSeriesTransformer(input_dim, d_model, nhead, num_layers, dropout)
# Run Bayesian optimization
res = gp_minimize(objective, space, n_calls=20, random_state=42)
# Define model with best hyperparameters
best_model = TimeSeriesTransformer(input_dim, d_model, nhead, best_num_layers, best_dropout)
# Training loop for final model
for epoch in range(num_epochs):
    best_model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = best_model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
# Testing the final model
best_model.eval()
test_loss = 0.0
test_predictions = []
test_targets = []
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        test_outputs = best_model(X_test_batch)
        test_loss += criterion(test_outputs.squeeze(), y_test_batch).item() * X_test_batch.size(0)
        test_predictions.extend(test_outputs.squeeze().tolist())
        test_targets.extend(y_test_batch.tolist())
test_loss /= len(test_loader.dataset)
test_r2 = r2_score(test_targets, test_predictions)
print(f'Final Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}')
```
