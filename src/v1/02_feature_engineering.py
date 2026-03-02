# src/02_feature_engineering.py (Upgraded with pandas_ta)

import pandas as pd
import pandas_ta as ta  # Import the new library
import argparse


def main(ticker):
    print(f"\n--- Step 2: Advanced Feature Engineering for {ticker} ---")

    RAW_DATA_PATH = f'data/{ticker}_data.csv'
    FEATURES_DATA_PATH = f'data/{ticker}_data_features.csv'

    try:
        data = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
        data.index.name = 'Date'
        print(f"Raw data for {ticker} loaded successfully. Initial rows: {len(data)}")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        print(f"Please run '01_mt5_data_ingestion.py --ticker {ticker}' first.")
        return

    # Check for sufficient data BEFORE calculating indicators
    required_bars = 480  # Our largest window size is for SMA_480
    if len(data) < required_bars:
        print(f"\nFATAL ERROR: Loaded only {len(data)} bars, but at least {required_bars} are needed.")
        return

    print("Calculating advanced technical indicators using pandas_ta...")

    # --- NEW: Create a custom strategy of indicators ---
    custom_strategy = ta.Strategy(
        name="M15_Advanced",
        description="A collection of indicators for an M15 strategy",
        ta=[
            # Your original indicators
            {"kind": "ema", "length": 96, "col_names": "EMA_96"},
            {"kind": "sma", "length": 480, "col_names": "SMA_480"},
            {"kind": "rsi", "length": 96, "col_names": "RSI_96"},

            # NEW INDICATORS
            # Volatility: Bollinger Bands
            {"kind": "bbands", "length": 96, "std": 2,
             "col_names": ("BB_lower", "BB_middle", "BB_upper", "BB_bandwidth", "BB_percent")},

            # Momentum: Moving Average Convergence Divergence (MACD)
            # Parameters are scaled for M15 (12h, 26h, 9h equivalent)
            {"kind": "macd", "fast": 48, "slow": 104, "signal": 36, "col_names": ("MACD", "MACD_hist", "MACD_signal")},

            # Volatility: Average True Range (ATR)
            {"kind": "atr", "length": 96, "col_names": "ATR"}
        ]
    )

    # Append all the indicators to your dataframe in one go
    data.ta.strategy(custom_strategy)
    # --- END OF NEW SECTION ---

    data.dropna(inplace=True)

    data.to_csv(FEATURES_DATA_PATH)

    print(f"Data with advanced features saved successfully to {FEATURES_DATA_PATH}")
    print(f"Data shape after adding features and cleaning: {data.shape}")
    print("New data columns:")
    print(data.columns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create advanced M15 features for financial data.')
    parser.add_argument('--ticker', type=str, required=True, help='The symbol name as used in your MT5 terminal')
    args = parser.parse_args()
    main(args.ticker)