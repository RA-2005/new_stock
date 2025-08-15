import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras

def train_predict_model(df):
    # Ensure numeric types

    df['Close'] = df['Close'].replace(',', '', regex=True).astype(float)
    
    if 'Sentiment' not in df.columns:
        df['Sentiment'] = 0.0  # default if no sentiment available

    # Select both Close price and Sentiment
    data = df[['Close', 'Sentiment']].values

    # Scale both features together
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    seq_len = 60
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i])         # last 60 steps of both features
        y.append(scaled_data[i, 0])               # predict only Close price

    X, y = np.array(X), np.array(y)

    # LSTM model for multivariate time series
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # Forecast future prices
    last_sequence = scaled_data[-seq_len:]
    next_input = last_sequence.reshape((1, seq_len, X.shape[2]))

    future = []
    for _ in range(7):
        pred = model.predict(next_input, verbose=0)[0][0]  # Only predict Close price
        future.append(pred)
        # Append predicted Close with same sentiment as last known day (or 0)
        last_sentiment = next_input[0, -1, 1] if X.shape[2] > 1 else 0
        next_row = np.array([[pred, last_sentiment]])
        next_input = np.append(next_input[:, 1:, :], [next_row], axis=1)

    # Inverse transform: Need array with 2 columns to match scaler
    future_full = np.zeros((len(future), 2))
    future_full[:, 0] = future  # Close price predictions
    predicted_prices = scaler.inverse_transform(future_full)[:, 0]

    return predicted_prices
