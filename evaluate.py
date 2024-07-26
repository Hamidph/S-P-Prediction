import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from src.data_processing import download_data, calculate_features, create_sequences

def plot_predictions(y_true, y_pred, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Values')
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted Values', color='red')
    plt.xlabel('Time')
    plt.ylabel('S&P 500 Closing Price')
    plt.legend()
    plt.title('S&P 500 Closing Price Prediction')
    plt.savefig(filename)
    plt.show()

def main():
    df = download_data()
    df = calculate_features(df)
    
    seq_length = 60
    features = ['Returns', 'MA_50', 'MA_200', 'Volatility', 'RSI', 'Volume']
    target = 'Close'
    
    X, y = create_sequences(df[features + [target]], target, seq_length)
