# src/05_lstm_model_training_v2.py (Modified to Reduce Overfitting)

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import argparse
import os


def build_gru_model(input_shape):
    """Builds the OPTIMIZED GRU model for USDCHF based on tuning results."""

    # Create the Adam optimizer with the best learning rate for USDCHF
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Use the best parameters found by KerasTuner for USDCHF
        layers.GRU(96, return_sequences=True),
        layers.Dropout(0.3),
        layers.GRU(96, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(20, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    # Compile the model with the new optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def main(ticker):
    print(f"\n--- Step 5: LSTM Model Training for {ticker} ---")

    PROCESSED_DATA_PATH = f'data/{ticker}_processed_data.npz'
    MODEL_PATH = f'models/lstm_{ticker}.keras'

    try:
        with np.load(PROCESSED_DATA_PATH, allow_pickle=True) as data:
            X_train = data['X_train']
            y_train = data['y_train']
            X_val = data['X_val']
            y_val = data['y_val']
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
        return

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    print("\nBuilding GRU model...")
    # Pass the input shape directly to the build function
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    os.makedirs('models', exist_ok=True)
    model_checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')

    print(f"\nTraining GRU model for {ticker}...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    print(f"\nModel training complete. Best model saved to '{MODEL_PATH}'")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'GRU Model Loss for {ticker}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'models/{ticker}_loss_curve.png')
    print(f"Loss curve plot saved to 'models/{ticker}_loss_curve.png'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GRU model for a given ticker.')
    parser.add_argument('--ticker', type=str, required=True, help='The ticker symbol to train the model for.')
    args = parser.parse_args()
    main(args.ticker)