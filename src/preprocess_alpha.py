import pandas as pd
import numpy as np
import pywt
from tsfracdiff import FractionalDifferentiator  # Corrected import
import os


def wavelet_denoise(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")[:len(data)]


def process_ticker(ticker):
    print(f"🧹 Cleaning {ticker}...")
    file_path = f'data/{ticker}_SUPER_dataset.csv'
    if not os.path.exists(file_path):
        print(f"  [Error] {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    # 1. WAVELET DENOISING (Remove high-frequency spikes)
    df['Close_Clean'] = wavelet_denoise(df['Close'].values)

    # 2. FRACTIONAL DIFFERENTIATION (The Stationarity Secret)
    # This automatically finds the best 'd' value to preserve memory
    # while making the data stationary for the AI.
    frac_diff = FractionalDifferentiator()
    # We apply it to our denoised close price
    df['FracDiff_Close'] = frac_diff.FitTransform(df[['Close_Clean']])

    # 3. Z-SCORE NORMALIZATION (Standardizing the signal)
    window = 100
    df['FracDiff_Z'] = (df['FracDiff_Close'] - df['FracDiff_Close'].rolling(window).mean()) / \
                       (df['FracDiff_Close'].rolling(window).std() + 1e-8)

    # Fill any NaNs created by rolling windows or differentiation
    df.fillna(method='bfill', inplace=True)

    save_path = f'data/{ticker}_CLEAN_dataset.csv'
    df.to_csv(save_path, index=False)
    print(f"✨ Successfully saved to {save_path}")


if __name__ == "__main__":
    tickers = ["EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "USDJPY", "AUDUSD", "NZDUSD"]
    for t in tickers:
        process_ticker(t)