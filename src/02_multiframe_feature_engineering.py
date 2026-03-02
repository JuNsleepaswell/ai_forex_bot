import pandas as pd
import pandas_ta as ta
import numpy as np
import argparse
import os
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


def apply_frac_diff(series, d=0.5, window=10):
    """
    Standard differentiation (d=1) removes too much memory.
    Fractional differentiation (d=0.5) keeps trend memory while becoming stationary.
    """
    weights = np.array([1.0, -d, d * (d - 1) / 2, -d * (d - 1) * (d - 2) / 6])  # Simple 4-tap weights
    res = series.rolling(window=len(weights)).apply(lambda x: np.dot(x[::-1], weights), raw=True)
    return res.fillna(0)


def process_d1(ticker):
    df = pd.read_csv(f'data/{ticker}_D1.csv', index_col='time', parse_dates=True)
    df['D1_ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # Normalized Momentum
    df['D1_Norm_Ret'] = df['Close'].diff(1) / (df['D1_ATR'] + 1e-9)
    return df[['D1_ATR', 'D1_Norm_Ret']]


def process_h4(ticker):
    df = pd.read_csv(f'data/{ticker}_H4.csv', index_col='time', parse_dates=True)
    df['H4_ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['H4_ER'] = ta.er(df['Close'], length=14)
    return df[['H4_ATR', 'H4_ER']]


def process_h1(ticker):
    df = pd.read_csv(f'data/{ticker}_H1.csv', index_col='time', parse_dates=True)
    df['H1_ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 1. Stationary Velocity (Targeting the -1.0 to 1.0 range)
    df['H1_Norm_Ret_1'] = df['Close'].diff(1) / (df['H1_ATR'] + 1e-9)
    df['H1_Norm_Ret_4'] = df['Close'].diff(4) / (df['H1_ATR'] + 1e-9)

    # --- ENHANCEMENT: CYCLE MEMORY LAGS ---
    # Helps LSTM see what happened exactly 12h and 24h ago
    df['H1_Norm_Ret_12'] = df['H1_Norm_Ret_1'].shift(12)
    df['H1_Norm_Ret_24'] = df['H1_Norm_Ret_1'].shift(24)

    # --- ENHANCEMENT: VOLATILITY REGIME ---
    # Is the market currently exploding or quiet compared to the last week?
    df['Vol_Regime'] = df['H1_ATR'] / (df['H1_ATR'].rolling(168).mean() + 1e-9)

    # --- ENHANCEMENT: FRACTIONAL DIFFERENTIATION ---
    # Preserves some 'price level' memory without being non-stationary
    df['FracDiff_Close'] = apply_frac_diff(df['Close'], d=0.4)

    # 3. Autocorrelation & Z-Score
    df['H1_Ret_1'] = df['Close'].pct_change(1)
    df['H1_Autocorr'] = df['H1_Ret_1'].rolling(10).apply(lambda x: x.autocorr() if x.std() > 0 else 0, raw=False)


    # 1. The "Stretch" Feature (Leading Signal)
    # Define the 100-hour window specifically for Stretch/Speed
    rolling_mean_100 = df['Close'].rolling(100).mean()
    rolling_std_100 = df['Close'].rolling(100).std()

    df['Price_Stretch'] = (df['Close'] - rolling_mean_100) / (rolling_std_100 + 1e-9)

    # 2. The "Velocity" Feature (Removes Lag)
    df['MA_Speed'] = rolling_mean_100.diff(3) / (df['H1_ATR'] + 1e-9)

    # 3. RSI Velocity
    rsi = ta.rsi(df['Close'], length=14)
    df['RSI_Velocity'] = rsi.diff(1)

    # Is volatility 'normal' (near 1.0)?
    # Values < 0.5 mean 'Dead Zone', Values > 2.5 mean 'Panic Zone'
    df['ATR_Relative'] = df['H1_ATR'] / (df['H1_ATR'].rolling(168).mean() + 1e-9)

    # Keep the 50-period Z-Score for a different perspective
    rolling_mean_50 = df['Close'].rolling(50).mean()
    rolling_std_50 = df['Close'].rolling(50).std()
    df['H1_ZScore_50'] = (df['Close'] - rolling_mean_50) / (rolling_std_50 + 1e-9)

    df['H1_ER'] = ta.er(df['Close'], length=10)

    # Time Features
    df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return df


def get_macro_data(start_date, end_date):
    print("  Fetching Global Macro Liquidity Data...")
    try:
        # Using adjusted tickers for better reliability
        dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)

        macro = pd.DataFrame(index=dxy.index)
        macro['DXY_Ret'] = dxy['Close'].pct_change(1)
        macro['SPY_Ret'] = spy['Close'].pct_change(1)

        if macro.index.tz is not None:
            macro.index = macro.index.tz_localize(None)

        return macro.dropna()
    except:
        return pd.DataFrame()


def main(ticker):
    print(f"\n--- Step 2: Advanced Feature Engineering for {ticker} ---")
    try:
        d1 = process_d1(ticker)
        h4 = process_h4(ticker)
        h1 = process_h1(ticker)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Look-Ahead Bias Prevention
    h4.index = h4.index + pd.Timedelta(hours=4)
    d1.index = d1.index + pd.Timedelta(days=1)

    # Merge
    merged = pd.merge_asof(h1.sort_index(), h4.sort_index(), left_index=True, right_index=True, direction='backward')
    merged = pd.merge_asof(merged, d1.sort_index(), left_index=True, right_index=True, direction='backward')

    # Macro
    start_str = merged.index[0].strftime('%Y-%m-%d')
    end_str = merged.index[-1].strftime('%Y-%m-%d')
    macro_df = get_macro_data(start_str, end_str)

    if not macro_df.empty:
        merged = pd.merge_asof(merged, macro_df, left_index=True, right_index=True, direction='backward')

    # ATR Ratio (Volatility Compression)
    merged['ATR_Ratio'] = merged['H1_ATR'] / (merged['H4_ATR'] + 1e-9)

    # Final Clean up
    merged.fillna(method='ffill', inplace=True)
    merged.dropna(inplace=True)

    output_path = f'data/{ticker}_SUPER_dataset.csv'
    merged.to_csv(output_path)
    print(f"  Super Dataset saved | Final Shape: {merged.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    args = parser.parse_args()
    main(args.ticker)