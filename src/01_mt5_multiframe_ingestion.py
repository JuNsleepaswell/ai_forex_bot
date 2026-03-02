# src/01_mt5_multiframe_ingestion.py (Desperate Fallback Version)

import MetaTrader5 as mt5
import pandas as pd
import os
import argparse
from datetime import datetime

# --- CONFIGURATION ---
BROKER_SUFFIX = "" 

def download_timeframe(ticker, timeframe_const, timeframe_name):
    broker_symbol = f"{ticker}{BROKER_SUFFIX}"
    print(f"  Requesting {timeframe_name} data for {broker_symbol}...")

    # --- PLAN A: Date Range Loop ---
    start_years = [2010, 2015, 2018, 2020, 2021, 2022]
    end_date = datetime.now()
    
    rates = None
    
    for year in start_years:
        start_date = datetime(year, 1, 1)
        rates = mt5.copy_rates_range(broker_symbol, timeframe_const, start_date, end_date)
        if rates is not None and len(rates) > 0:
            print(f"    [Plan A] Success! Retrieved data starting approx {year}.")
            break 
    
    # --- PLAN B: Decreasing Bar Counts (The "Desperate" Strategy) ---
    if rates is None or len(rates) == 0:
        print(f"    [Plan A] Failed. Attempting Plan B (Step-down counts)...")
        
        # Try requesting these amounts in order. If 200k fails, try 100k, etc.
        counts_to_try = [200000, 100000, 50000, 10000, 1000]
        
        for count in counts_to_try:
            print(f"      Trying to fetch last {count} bars...")
            rates = mt5.copy_rates_from_pos(broker_symbol, timeframe_const, 0, count)
            if rates is not None and len(rates) > 0:
                print(f"    [Plan B] Success! Retrieved last {len(rates)} bars.")
                break
        
    if rates is None or len(rates) == 0:
        print(f"    ERROR: Could not retrieve ANY data for {timeframe_name}.")
        return None

    # --- Process Data ---
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    
    cols_to_drop = ['spread', 'real_volume']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    return df

def main(ticker):
    print(f"\n--- Step 1: Multi-Timeframe Ingestion for {ticker} ---")

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    timeframes = {
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1
    }

    os.makedirs('data', exist_ok=True)

    for tf_name, tf_const in timeframes.items():
        df = download_timeframe(ticker, tf_const, tf_name)
        if df is not None:
            filename = f'data/{ticker}_{tf_name}.csv'
            df.to_csv(filename)
            print(f"    Saved {len(df)} rows to {filename}")

    mt5.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download M15, H1, H4, and D1 data adaptively.')
    parser.add_argument('--ticker', type=str, required=True, help='Standard Symbol name (e.g., EURUSD without suffix)')
    args = parser.parse_args()
    main(args.ticker)