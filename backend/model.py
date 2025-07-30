import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_predict_model(csv_path):
    df = pd.read_csv(csv_path)
    data = df['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    seq_len = 60
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i])
        y.append(scaled_data[i])
        
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    last_sequence = scaled_data[-seq_len:]
    next_input = last_sequence.reshape((1, seq_len, 1))
    
    future = []
    for _ in range(7):  # Predict next 7 days
        pred = model.predict(next_input)[0][0]
        future.append(pred)
        next_input = np.append(next_input[:, 1:, :], [[[pred]]], axis=1)
    
    predicted_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1))
    return predicted_prices
