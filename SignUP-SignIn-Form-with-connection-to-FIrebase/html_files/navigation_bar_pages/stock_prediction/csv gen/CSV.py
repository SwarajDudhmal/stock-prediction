import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
df = pd.read_csv('HistoricalData\ITC.csv', parse_dates=['Date'])
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

# Create sequences
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, features.index(target)])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(scaled_data, window_size)

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
    Dense(1)
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

# Inverse transformation function
def inverse_transform(predictions, scaler, feature_index):
    dummy = np.zeros(shape=(len(predictions), scaled_data.shape[1]))
    dummy[:, feature_index] = predictions.reshape(-1)
    return scaler.inverse_transform(dummy)[:, feature_index]

# Generate all predictions
train_predictions = inverse_transform(model.predict(X_train), scaler, features.index(target))
test_predictions = inverse_transform(model.predict(X_test), scaler, features.index(target))

# Create main results dataframe
results_df = pd.DataFrame({
    'Date': df['Date'][window_size:].values,
    'Actual': inverse_transform(np.concatenate([y_train, y_test]), scaler, features.index(target)),
    'Predicted': np.concatenate([train_predictions, test_predictions]),
    'Type': ['Train'] * len(y_train) + ['Test'] * len(y_test)
})

# Mark last week predictions
last_week_mask = results_df['Date'] >= results_df['Date'].iloc[-7]
results_df.loc[last_week_mask, 'Type'] = 'LastWeek'

# Add next day prediction
next_day_date = df['Date'].iloc[-1] + pd.DateOffset(days=1)
last_sequence = scaled_data[-window_size:].reshape(1, window_size, len(features))
next_day_pred = inverse_transform(model.predict(last_sequence), scaler, features.index(target))[0]
results_df = pd.concat([
    results_df,
    pd.DataFrame([{
        'Date': next_day_date,
        'Actual': np.nan,
        'Predicted': next_day_pred,
        'Type': 'NextDay'
    }])
], ignore_index=True)

# Calculate metrics
test_mask = results_df['Type'].isin(['Test', 'LastWeek'])
actual_prices = results_df.loc[test_mask, 'Actual']
predicted_prices = results_df.loc[test_mask, 'Predicted']

metrics = {
    'MAE': np.mean(np.abs(actual_prices - predicted_prices)),
    'RMSE': np.sqrt(np.mean((actual_prices - predicted_prices) ** 2)),
    'MeanPrice': results_df.loc[results_df['Type'].isin(['Train', 'Test']), 'Actual'].mean(),
    'ErrorPercentage': (np.mean(np.abs(actual_prices - predicted_prices)) /
                        results_df.loc[results_df['Type'].isin(['Train', 'Test']), 'Actual'].mean() * 100)
}

# Create metrics dataframe
metrics_df = pd.DataFrame({
    'Date': ['Metric: MAE', 'Metric: RMSE', 'Metric: Average Price', 'Metric: Error Percentage'],
    'Actual': [metrics['MAE'], metrics['RMSE'], metrics['MeanPrice'], metrics['ErrorPercentage']],
    'Predicted': ['INR', 'INR', 'INR', '%'],
    'Type': ['Metric'] * 4
})

# Combine all data
final_df = pd.concat([results_df, metrics_df], ignore_index=True)

# Save to single CSV
final_df.to_csv('all_predictions.csv', index=False)

print("Single CSV file generated successfully!")
print(f"Predicted Next Day Price: ₹{next_day_pred:.2f}")
