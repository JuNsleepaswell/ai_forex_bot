import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import itertools
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Add every pair you want to test here. The script will test every possible combination.
TICKERS = ['AUDNZD', 'EURGBP', 'CADCHF', 'AUDCAD', 'NZDCAD']
TIMEFRAME = 'H1'

print(f"--- Statistical Arbitrage: Cointegration Research Engine ---")


def calculate_half_life(spread):
    """
    Calculates the Half-Life of mean reversion using the Ornstein-Uhlenbeck process.
    Tells us how many bars it typically takes for the spread to revert to the mean.
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    # Align the series
    spread_lag, spread_diff = spread_lag.align(spread_diff, join='inner')

    # Run OLS: dSpread_t = lambda * Spread_{t-1} + error
    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, X).fit()

    # The slope (lambda) tells us the speed of mean reversion
    lam = model.params.iloc[1]

    # If lambda is positive, it's diverging, not mean-reverting
    if lam >= 0:
        return np.inf

    half_life = -np.log(2) / lam
    return half_life


def main():
    # 1. Load and align all Close prices into a single DataFrame
    print("Loading MT5 Data...")
    price_data = {}

    for ticker in TICKERS:
        path = f'data/{ticker}_{TIMEFRAME}.csv'
        if not os.path.exists(path):
            print(f"  [WARNING] Could not find {path}. Did you run step 1?")
            continue

        df = pd.read_csv(path, index_col='time', parse_dates=True)
        price_data[ticker] = df['Close']

    if len(price_data) < 2:
        print("Need at least two valid tickers to find a pair.")
        return

    prices = pd.DataFrame(price_data).dropna()

    # --- THE REGIME FILTER ---
    # Only test the last 6000 bars (~1 year of H1 data) to find CURRENT cointegration
    LOOKBACK_BARS = 6000
    prices = prices.tail(LOOKBACK_BARS)

    print(f"Recent Data Shape: {prices.shape} (Testing the most recent market regime)")

    # 2. Iterate through every possible pair combination
    pairs = list(itertools.combinations(prices.columns, 2))
    results = []

    print(f"\nAnalyzing {len(pairs)} combinations for mathematical cointegration...\n")

    for asset_Y, asset_X in pairs:
        Y = prices[asset_Y]
        X = prices[asset_X]

        # Calculate Hedge Ratio (Beta) using Ordinary Least Squares (OLS)
        X_const = sm.add_constant(X)
        model = sm.OLS(Y, X_const).fit()
        beta = model.params.iloc[1]

        # Calculate the Spread: Y - (Beta * X)
        spread = Y - (beta * X)

        # Run Cointegration Test (Engle-Granger)
        # p-value < 0.05 means we are 95% confident they are cointegrated
        score, p_value, _ = coint(Y, X)

        # Calculate Half-Life
        half_life = calculate_half_life(spread)

        results.append({
            'Pair_Y': asset_Y,
            'Pair_X': asset_X,
            'P_Value': p_value,
            'Hedge_Ratio_Beta': beta,
            'Half_Life_Hours': half_life
        })

    # 3. Sort and Display Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='P_Value')

    print(f"{'Asset Y':<10} | {'Asset X':<10} | {'P-Value':<12} | {'Beta (Hedge)':<12} | {'Half-Life'}")
    print("-" * 65)
    for _, row in results_df.iterrows():
        p_val_str = f"{row['P_Value']:.5f}"
        if row['P_Value'] < 0.05: p_val_str += " ***"  # Highlight the winners

        hl_str = f"{row['Half_Life_Hours']:.1f} hrs" if row['Half_Life_Hours'] != np.inf else "No Mean Rev"
        print(
            f"{row['Pair_Y']:<10} | {row['Pair_X']:<10} | {p_val_str:<12} | {row['Hedge_Ratio_Beta']:<12.4f} | {hl_str}")

    # 4. Visualize the absolute best pair
    best_pair = results_df.iloc[0]
    if best_pair['P_Value'] > 0.05:
        print("\n[WARNING] None of these pairs are statistically cointegrated (P-Value < 0.05).")
        return

    print(f"\n>>> BEST PAIR FOUND: {best_pair['Pair_Y']} and {best_pair['Pair_X']} <<<")

    Y = prices[best_pair['Pair_Y']]
    X = prices[best_pair['Pair_X']]
    beta = best_pair['Hedge_Ratio_Beta']

    spread = Y - (beta * X)

    # Calculate Z-Score of the spread
    spread_mean = spread.rolling(window=200).mean()
    spread_std = spread.rolling(window=200).std()
    z_score = (spread - spread_mean) / spread_std

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

    axs[0].plot(Y.index, Y, label=best_pair['Pair_Y'], color='blue')
    axs[0].set_ylabel(f"{best_pair['Pair_Y']} Price")
    axs[0].legend(loc='upper left')

    ax0_twin = axs[0].twinx()
    ax0_twin.plot(X.index, X, label=best_pair['Pair_X'], color='orange')
    ax0_twin.set_ylabel(f"{best_pair['Pair_X']} Price")
    ax0_twin.legend(loc='upper right')
    axs[0].set_title(f"Normalized Prices: {best_pair['Pair_Y']} vs {best_pair['Pair_X']}")

    axs[1].plot(z_score.index, z_score, label='Spread Z-Score', color='purple')
    axs[1].axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Short Y, Buy X')
    axs[1].axhline(-2.0, color='green', linestyle='--', alpha=0.5, label='Buy Y, Short X')
    axs[1].axhline(0, color='black', alpha=0.5)
    axs[1].set_title(f"Spread Z-Score (Mean Reverts every ~{best_pair['Half_Life_Hours']:.1f} hours)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()