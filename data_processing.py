import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def download_data():
    sp500 = yf.Ticker("^GSPC")
    df = sp500.history(start="2004-06-30", end="2024-06-30")
    return df

def calculate_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['RSI'] = calculate_rsi(df['Close'])
    df = df.dropna()
    return df

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed>=0].sum()/window
    down = -seed[seed<0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1.+rs)

    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[target].iloc[i + seq_length])
    return np.array(X), np.array(y)
