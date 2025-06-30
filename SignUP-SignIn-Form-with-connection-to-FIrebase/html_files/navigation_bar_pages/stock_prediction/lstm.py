import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
df = pd.read_csv('HistoricalData\KOTAKBAANK.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Feature engineering
df['Range'] = df['High'] - df['Low']
df['MA5'] = df['Close'].rolling(window=5).mean().fillna(method='bfill')
df['Volume'] = np.log1p(df['Volume'])

# Feature selection
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Range', 'MA5']
target = 'Close'
data = df[features].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences with multi-step targets
def create_sequences(data, window_size=30, prediction_steps=7):
    X, y = [], []
    for i in range(window_size, len(data) - prediction_steps + 1):
        X.append(data[i - window_size:i])
        y.append(data[i:i + prediction_steps, features.index(target)])
    return np.array(X), np.array(y)

window_size = 30
prediction_steps = 7
X, y = create_sequences(scaled_data, window_size, prediction_steps)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, len(features))),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(prediction_steps)
])

# Compile model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# Train model
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)

# Inverse transformation
def inverse_transform(predictions, scaler, feature_index):
    n_samples, n_steps = predictions.shape
    flat_predictions = predictions.flatten()
    dummy = np.zeros((len(flat_predictions), scaled_data.shape[1]))
    dummy[:, feature_index] = flat_predictions
    dummy = scaler.inverse_transform(dummy)
    return dummy[:, feature_index].reshape(n_samples, n_steps)

# Generate predictions
test_predictions = model.predict(X_test)
actual_prices = inverse_transform(y_test, scaler, features.index(target))
predicted_prices = inverse_transform(test_predictions, scaler, features.index(target))

# Save training predictions
train_pred = inverse_transform(model.predict(X_train), scaler, features.index(target))
train_dates = df['Date'][window_size:split+window_size]
training_actual = df['Close'][window_size:split+window_size].values

training_df = pd.DataFrame({
    'Date': train_dates,
    'Actual_Close': training_actual,
    'Training_Prediction': train_pred[:, 0]  # First prediction step
})
training_df.to_csv('training_predictions.csv', index=False)

# Save test predictions
test_predictions_list = []
for i in range(len(X_test)):
    start_idx = split + window_size + i
    end_idx = start_idx + prediction_steps
    dates = df['Date'].iloc[start_idx:end_idx]
    actuals = df['Close'].iloc[start_idx:end_idx]
    preds = predicted_prices[i]
    
    for j in range(prediction_steps):
        test_predictions_list.append({
            'Date': dates.iloc[j],
            'Actual_Close': actuals.iloc[j],
            'Test_Prediction': preds[j]
        })

test_df = pd.DataFrame(test_predictions_list)
test_df.to_csv('test_predictions.csv', index=False)

# Generate and save forecast
def get_market_dates(last_date, days=7):
    dates = []
    current = last_date
    while len(dates) < days:
        current += pd.DateOffset(days=1)
        if current.weekday() < 5:  # Monday-Friday
            dates.append(current)
    return dates[:days]

last_sequence = scaled_data[-window_size:]
next_pred = inverse_transform(model.predict(np.expand_dims(last_sequence, 0)), scaler, features.index(target))[0]
future_dates = get_market_dates(df['Date'].iloc[-1])

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast_Close': next_pred
})
forecast_df.to_csv('forecast.csv', index=False)

# Calculate and save metrics
mae = np.mean(np.abs(actual_prices.flatten()[:len(test_df)] - predicted_prices.flatten()[:len(test_df)]))
rmse = np.sqrt(np.mean((actual_prices.flatten()[:len(test_df)] - predicted_prices.flatten()[:len(test_df)])**2))

metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE'],
    'Value': [mae, rmse]
})
metrics_df.to_csv('metrics.csv', index=False)

print("CSV files generated successfully!")