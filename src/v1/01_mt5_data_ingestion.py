# src/01_mt5_data_ingestion.py (Using Date Range)

import MetaTrader5 as mt5
import pandas as pd
import os
import argparse
from datetime import datetime


def main(ticker):
    print(f"\n--- Step 1: MT5 Data Ingestion for {ticker} ---")

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    print("MetaTrader5 connection successful")

    # --- MODIFIED: Explicitly define the desired date range ---
    start_date = datetime(2011, 1, 1)
    # Let's get data right up to the present
    end_date = datetime.now()
    timeframe = mt5.TIMEFRAME_M15

    print(f"Requesting data for {ticker} from {start_date.strftime('%Y-%m-%d')} to present on M15 timeframe...")

    # Use copy_rates_range to request data between two dates
    rates = mt5.copy_rates_range(ticker, timeframe, start_date, end_date)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"No data received for {ticker}, error code =", mt5.last_error())
        print("Please ensure the symbol is correct and history is downloaded in the MT5 terminal.")
        return

    print(f"{len(rates)} bars of data received successfully.")

    # Convert to DataFrame to check the actual start date
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')

    actual_start_date = data['time'].iloc[0]
    print(f"Broker returned data starting from: {actual_start_date.strftime('%Y-%m-%d')}")

    # Check if we have enough data for our largest indicator window (SMA_480)
    required_bars = 480
    if len(rates) < required_bars:
        print(f"\nFATAL ERROR: Downloaded only {len(rates)} bars, but at least {required_bars} are required.")
        return

    # --- Data processing ---
    data.set_index('time', inplace=True)
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)

    cols_to_drop = ['spread', 'real_volume']
    data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True)

    # --- Save the Data ---
    DATA_PATH = f'data/{ticker}_data.csv'
    os.makedirs('data', exist_ok=True)
    data.to_csv(DATA_PATH)
    print(f"Data for {ticker} saved successfully to {DATA_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download all available M15 data from MetaTrader 5 for a specific date range.')
    parser.add_argument('--ticker', type=str, required=True, help='The symbol name as used in your MT5 terminal')
    args = parser.parse_args()
    main(args.ticker)