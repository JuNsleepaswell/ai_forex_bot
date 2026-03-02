# src/old_04_multiframe_risk_training.py (STABLE VERSION)

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import os
import argparse

def main(ticker):
    print(f"\n--- Training Risk Model for {ticker} (Stable) ---")

    DATA_PATH = f'data/{ticker}_SUPER_dataset.csv'
    MODEL_PATH = f'models/risk_{ticker}.joblib'

    if not os.path.exists(DATA_PATH): return

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # 1. Target
    N = 4
    df['Future_ATR'] = df['M15_ATR'].shift(-N)
    df.dropna(inplace=True)

    # 2. Features
    features = [
        'D1_EMA_200', 'D1_RSI',
        'H4_EMA_200', 'H4_ATR', 'H4_RSI',
        'H1_EMA_50', 'H1_RSI', 'H1_MACD',
        'M15_RSI', 'M15_ATR', 'Close',
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
    ]
    available_features = [f for f in features if f in df.columns]

    # 3. Split
    split_idx = int(len(df) * 0.8)
    X_train = df[available_features].iloc[:split_idx]
    y_train = df['Future_ATR'].iloc[:split_idx]
    X_test = df[available_features].iloc[split_idx:]
    y_test = df['Future_ATR'].iloc[split_idx:]

    # 4. Train
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        n_jobs=1
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  MAE: {mae:.5f}")
    
    # 6. Save
    joblib.dump(model, MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    args = parser.parse_args()
    main(args.ticker)