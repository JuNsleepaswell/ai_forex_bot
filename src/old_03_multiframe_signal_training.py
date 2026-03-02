# src/old_03_multiframe_signal_training.py (STABLE VERSION)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os
import argparse

def main(ticker):
    print(f"\n--- Training Signal Model for {ticker} (Stable) ---")

    DATA_PATH = f'data/{ticker}_SUPER_dataset.csv'
    MODEL_PATH = f'models/signal_{ticker}.joblib'

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # 1. Target
    N = 4 
    df['Future_Close'] = df['Close'].shift(-N)
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    df.dropna(inplace=True)

    # 2. Features (D1 + Time + Technicals)
    features = [
        'D1_EMA_200', 'D1_RSI',
        'H4_EMA_200', 'H4_ATR', 'H4_RSI',
        'H1_EMA_50', 'H1_RSI', 'H1_MACD', 'H1_MACD_Signal',
        'M15_RSI', 'M15_ATR', 'M15_BB_Upper', 'M15_BB_Lower', 'Close',
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
    ]
    
    available_features = [f for f in features if f in df.columns]

    # 3. Split
    split_idx = int(len(df) * 0.8)
    X_train = df[available_features].iloc[:split_idx]
    y_train = df['Target'].iloc[:split_idx]
    X_test = df[available_features].iloc[split_idx:]
    y_test = df['Target'].iloc[split_idx:]

    # 4. Train (Proven Stable Hyperparameters)
    # n_jobs=1 prevents threading locks during save
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        n_jobs=1 
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    print(f"  Accuracy: {acc:.2%} | Precision: {prec:.2%}")

    # 6. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    args = parser.parse_args()
    main(args.ticker)