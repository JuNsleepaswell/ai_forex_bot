# src/01_data_ingestion.py
import yfinance as yf
import pandas as pd
import os
import argparse  # Import argparse


def main(ticker):  # Put the logic into a main function
    print("--- Step 1: Data Ingestion ---")
    START_DATE = '2010-01-01'
    END_DATE = '2024-01-01'
    DATA_PATH = f'data/{ticker}_data.csv'  # Dynamic filename

    os.makedirs('data', exist_ok=True)
    print(f"Downloading data for {ticker} from {START_DATE} to {END_DATE}...")
    data = yf.download(ticker, start=START_DATE, end=END_DATE)

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, level='Ticker', axis=1)

    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    data[cols_to_numeric] = data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    data.to_csv(DATA_PATH)
    print(f"Clean data saved successfully to {DATA_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download financial data.')
    parser.add_argument('--ticker', type=str, required=True, help='The ticker symbol to download (e.g., EURUSD=X)')
    args = parser.parse_args()
    main(args.ticker)