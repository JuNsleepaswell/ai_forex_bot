# src/04_lstm_data_preparation.py (Memory-Optimized Version)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import gc  # --- MEMORY OPTIMIZATION: Import the Garbage Collector ---


def create_sequences(input_data, target_data, look_back, predict_ahead):
    """Helper function to create sequences for LSTM/GRU."""
    X, y = [], []
    for i in range(look_back, len(input_data) - predict_ahead + 1):
        X.append(input_data[i - look_back:i, :])
        y.append(target_data[i + predict_ahead - 1])
    return np.array(X), np.array(y)


def main(ticker):
    print(f"\n--- Step 4: Preparing LSTM/GRU Data for {ticker} (Advanced Features) ---")

    FEATURES_DATA_PATH = f'data/{ticker}_data_features.csv'
    PROCESSED_DATA_PATH = f'data/{ticker}_processed_data.npz'

    # --- 1. Load Data with Memory Optimization ---

    # This list should match the `numeric_features` from your Step 3 script
    features_to_use = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'EMA_96', 'SMA_480', 'RSI_96',
        'BB_lower', 'BB_middle', 'BB_upper', 'BB_bandwidth', 'BB_percent',
        'MACD', 'MACD_hist', 'MACD_signal', 'ATR'
    ]

    # --- MEMORY OPTIMIZATION 1: Define Data Types Before Loading ---
    # We create a dictionary telling pandas to load all numeric features as 32-bit floats
    # instead of the default 64-bit. This simple change cuts the initial DataFrame's
    # memory usage by nearly 50%.
    col_dtypes = {col: 'float32' for col in features_to_use}

    try:
        # The 'dtype' parameter is added here.
        data = pd.read_csv(
            FEATURES_DATA_PATH,
            index_col='Date',
            parse_dates=True,
            dtype=col_dtypes
        )
        print(f"Feature data for {ticker} loaded successfully with optimized memory.")
    except FileNotFoundError:
        print(f"Error: Feature file not found at {FEATURES_DATA_PATH}")
        return

    # Verify all columns are present
    missing_cols = [col for col in features_to_use if col not in data.columns]
    if missing_cols:
        print(f"FATAL ERROR: The following feature columns are missing from the data: {missing_cols}")
        return

    # --- MEMORY OPTIMIZATION: Explicitly select and then delete original large dataframe ---
    # This ensures we don't keep multiple copies of the data in RAM.
    data_selected = data[features_to_use].copy()
    del data
    gc.collect()
    print("Cleaned up initial data frame from memory.")

    # --- 2. Split Data by Percentage (60/20/20) BEFORE Scaling ---
    print("Splitting data into training, validation, and testing sets...")
    train_size = int(0.6 * len(data_selected))
    val_size = int(0.2 * len(data_selected))

    train_df = data_selected.iloc[:train_size]
    val_df = data_selected.iloc[train_size: train_size + val_size]
    test_df = data_selected.iloc[train_size + val_size:]

    # --- MEMORY OPTIMIZATION: Delete the now-redundant combined dataframe ---
    del data_selected
    gc.collect()

    if train_df.empty or val_df.empty or test_df.empty:
        print("FATAL ERROR: One or more data splits are empty. Not enough data available.")
        return

    print(f"Data split into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test bars.")

    # --- 3. Scale the Data based ONLY on the Training Set ---
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df)  # Fit ONLY on training data

    scaled_train_data = scaler.transform(train_df)
    scaled_val_data = scaler.transform(val_df)
    scaled_test_data = scaler.transform(test_df)

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(train_df[['Close']])

    scaled_train_target = scaler_target.transform(train_df[['Close']])
    scaled_val_target = scaler_target.transform(val_df[['Close']])
    scaled_test_target = scaler_target.transform(test_df[['Close']])

    # --- MEMORY OPTIMIZATION: Delete the split dataframes, which are now redundant ---
    del train_df, val_df, test_df
    gc.collect()
    print("Cleaned up intermediate dataframes from memory.")

    # --- 4. Create Sequences for all three sets ---
    print("Creating time-series sequences...")
    look_back = 240
    predict_ahead = 32

    X_train, y_train = create_sequences(scaled_train_data, scaled_train_target, look_back, predict_ahead)
    X_val, y_val = create_sequences(scaled_val_data, scaled_val_target, look_back, predict_ahead)
    X_test, y_test = create_sequences(scaled_test_data, scaled_test_target, look_back, predict_ahead)

    # --- MEMORY OPTIMIZATION: Delete the large scaled arrays before saving ---
    # The sequence arrays are the final product; we don't need the intermediate arrays anymore.
    del scaled_train_data, scaled_val_data, scaled_test_data
    del scaled_train_target, scaled_val_target, scaled_test_target
    gc.collect()
    print("Cleaned up scaled numpy arrays from memory.")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("FATAL ERROR: Zero sequences were created for one or more sets. Check data length.")
        return

    print(f"Training sequences created: {X_train.shape}")
    print(f"Validation sequences created: {X_val.shape}")
    print(f"Testing sequences created: {X_test.shape}")

    # --- 5. Save ALL THREE sets to the .npz file ---
    os.makedirs('data', exist_ok=True)
    # Using savez_compressed for slightly better disk space usage
    np.savez_compressed(
        PROCESSED_DATA_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        scaler_target=scaler_target
    )
    print(f"Processed GRU/LSTM data saved to '{PROCESSED_DATA_PATH}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare advanced M15 data for LSTM/GRU model training.')
    parser.add_argument('--ticker', type=str, required=True, help='The symbol name as used in your MT5 terminal')
    args = parser.parse_args()
    main(args.ticker)