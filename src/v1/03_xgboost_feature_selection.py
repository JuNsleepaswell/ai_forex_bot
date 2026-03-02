# src/03_xgboost_feature_selection.py (Final Version with Advanced Features & Train/Val/Test Split)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
import os
import argparse


def main(ticker):
    print(f"\n--- Step 3: XGBoost Model Training for {ticker} (Advanced Features) ---")

    FEATURES_DATA_PATH = f'data/{ticker}_data_features.csv'
    MODEL_PATH = f'models/xgboost_{ticker}_model.joblib'
    PLOT_PATH = f'plots/feature_importance_{ticker}.png'

    try:
        data = pd.read_csv(FEATURES_DATA_PATH, index_col=0, parse_dates=True)
        data.index.name = 'Date'
        print(f"Feature data for {ticker} loaded successfully. Initial rows: {len(data)}")
    except FileNotFoundError:
        print(f"Error: Feature file not found at {FEATURES_DATA_PATH}")
        return

    # --- Robust Data Cleaning with ALL new features ---
    numeric_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'EMA_96', 'SMA_480', 'RSI_96',
        'BB_lower', 'BB_middle', 'BB_upper', 'BB_bandwidth', 'BB_percent',
        'MACD', 'MACD_hist', 'MACD_signal', 'ATR'
    ]
    for col in numeric_features:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)
    print(f"Data cleaned. Rows remaining: {len(data)}")

    if len(data) == 0:
        print("\nFATAL ERROR: All data was removed during the cleaning process.")
        return

    # Prediction Horizon for M15 (8 hours)
    N = 32
    data['Future_Close'] = data['Close'].shift(-N)
    data['Target'] = (data['Future_Close'] > data['Close']).astype(int)
    data.dropna(inplace=True)

    if len(data) < 500:
        print("\nFATAL ERROR: Not enough data remains to create a training set.")
        return

    # Use ALL new features for training
    features = numeric_features
    X = data[features]
    y = data['Target']

    # --- CORRECTED: 3-Way Train-Validation-Test Split (60/20/20) ---
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_val = X.iloc[train_size: train_size + val_size]
    y_val = y.iloc[train_size: train_size + val_size]

    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]

    if any(df.empty for df in [X_train, X_val, X_test]):
        print("\nFATAL ERROR: One of the data splits is empty. Not enough data available.")
        return

    print(f"Data split into {len(X_train)} train, {len(X_val)} validation, and {len(X_test)} test bars.")

    # --- Calculate scale_pos_weight for class imbalance ---
    num_zeros = (y_train == 0).sum()
    num_ones = (y_train == 1).sum()
    scale_pos_weight = num_zeros / num_ones if num_ones > 0 else 1
    print(f"Class balance check: Zeros={num_zeros}, Ones={num_ones}, scale_pos_weight={scale_pos_weight:.2f}")

    # --- Train the XGBoost Model with Validation ---
    print(f"Training XGBoost model for {ticker} with early stopping...")
    model = xgb.XGBClassifier(
        n_estimators=1000,  # Use a high number for early stopping
        max_depth=4,  # Slightly increased complexity
        learning_rate=0.05,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20  # Stop if validation loss doesn't improve for 20 rounds
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # --- Evaluate on the FINAL Test Set ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    # --- Analyze, Plot, and Save ---
    print("Analyzing and saving feature importance plot...")
    os.makedirs('plots', exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))  # Made the plot bigger for more features
    xgb.plot_importance(model, ax=ax, importance_type='weight')
    ax.set_title(f'XGBoost Feature Importance for {ticker} (Advanced Features)')
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"Feature importance plot saved to {PLOT_PATH}")

    print(f"Saving XGBoost model for {ticker}...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an XGBoost model with advanced features and train/val/test split.')
    parser.add_argument('--ticker', type=str, required=True, help='The symbol name as used in your MT5 terminal')
    args = parser.parse_args()
    main(args.ticker)