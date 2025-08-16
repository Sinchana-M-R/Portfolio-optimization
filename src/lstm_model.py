import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf
import multiprocessing

def prepare_lstm_data(series, look_back=12):
    """Prepare data for LSTM"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(look_back):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def forecast_with_lstm(series, look_back=12):
    """Forecast returns and volatility using LSTM with reduced retracing"""
    @tf.function(reduce_retracing=True)
    def predict(model, input_data):
        return model(input_data, training=False)
    try:
        if len(series) < look_back + 1:
            print(f"Skipping series with insufficient data points: {len(series)} < {look_back + 1}")
            val = series.iloc[-1] if len(series) > 0 else 0.0
            return val, val, val
        X, y, scaler = prepare_lstm_data(series, look_back)
        if len(X) < 10:
            print(f"Skipping series with insufficient training data: {len(X)} samples")
            val = series.iloc[-1] if len(series) > 0 else 0.0
            return val, val, val
        model = build_lstm_model(look_back)
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        last_sequence = series[-look_back:].values.reshape(1, look_back, 1)
        last_sequence = tf.convert_to_tensor(last_sequence, dtype=tf.float32)
        scaled_forecast = predict(model, last_sequence)[0, 0].numpy()
        forecast = scaler.inverse_transform([[scaled_forecast]])[0, 0]
        X_tensor = tf.convert_to_tensor(X.reshape(-1, look_back, 1), dtype=tf.float32)
        predictions = predict(model, X_tensor).numpy().flatten()
        residuals = y - predictions
        std_error = np.std(residuals) if len(residuals) > 0 else 0.0
        z_score = 1.96 # For 95% CI
        lower = forecast - z_score * std_error
        upper = forecast + z_score * std_error
        return forecast, lower, upper
    except Exception as e:
        print(f"LSTM forecast error for series: {e}")
        val = series.iloc[-1] if len(series) > 0 else 0.0
        return val, val, val

def parallel_lstm_forecast(returns, est_period_idx):
    """Parallel LSTM forecasting for all assets"""
    max_workers = min(multiprocessing.cpu_count(), 4)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for col in returns.columns:
            series = returns[col].loc[est_period_idx].dropna()
            if len(series) >= 12 + 1:
                futures.append(executor.submit(forecast_with_lstm, series))
            else:
                print(f"Skipping {col} due to insufficient data: {len(series)} points")
                results.append((0.0, 0.0, 0.0))
        results.extend([f.result() for f in futures])
    return pd.DataFrame(
        results,
        index=returns.columns,
        columns=['forecast', 'lower_bound', 'upper_bound']
    )
