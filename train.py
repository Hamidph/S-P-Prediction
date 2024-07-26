import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data_processing import download_data, calculate_features, create_sequences
from src.model import build_model

def main():
    df = download_data()
    df = calculate_features(df)
    
    seq_length = 60
    features = ['Returns', 'MA_50', 'MA_200', 'Volatility', 'RSI', 'Volume']
    target = 'Close'
    
    X, y = create_sequences(df[features + [target]], target, seq_length)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    model = build_model(seq_length, X_train_scaled.shape[2])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss')
    
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=[early_stopping, model_checkpoint],
        shuffle=False
    )
    
    model.save('models/final_model.h5')

if __name__ == "__main__":
    main()
