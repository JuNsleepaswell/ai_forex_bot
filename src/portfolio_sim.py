import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
import os

# Import your exact environment so the simulation matches training perfectly
from drl_training_pro import ForexTradingEnvPro, OBSERVATION_FEATURES

TICKERS = ["EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "USDJPY", "AUDUSD", "NZDUSD"]
VERSION = "20260308_2132"
PORTFOLIO_STARTING_BALANCE = 100000.0


def run_portfolio():
    print(f"--- 🌐 Launching Portfolio Simulation (Fleet v{VERSION}) ---")

    all_pnls = {}

    for ticker in TICKERS:
        model_path = f"models/pro_lstm_{ticker}_{VERSION}_final.zip"
        data_path = f"data/{ticker}_CLEAN_dataset.csv"

        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print(f"⚠️ Missing files for {ticker}. Skipping...")
            continue

        print(f"  [Simulating] {ticker}...")

        # Load Data and grab the exact same 10% Test Split used in your backtester
        df = pd.read_csv(data_path)
        df.dropna(subset=OBSERVATION_FEATURES, inplace=True)
        val_split = int(len(df) * 0.9)
        test_df = df.iloc[val_split:].copy().reset_index(drop=True)

        # Initialize Environment & Model
        env = ForexTradingEnvPro(test_df)
        model = RecurrentPPO.load(model_path)

        obs, _ = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        step_pnls = []

        # Run the unseen hours
        done = False
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts,
                                                deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.item())
            episode_starts = done = terminated or truncated

            # Collect the raw fractional step return (e.g., +0.0015 for a 0.15% gain)
            step_pnls.append(info.get('pnl', 0.0))

        all_pnls[ticker] = step_pnls

    # --- PORTFOLIO AGGREGATION ---
    print("\n--- 📊 Compiling Portfolio Metrics ---")

    # Pad shorter arrays with 0s at the start so everything ends on the exact same hour
    max_len = max(len(p) for p in all_pnls.values())
    padded_pnls = {t: ([0.0] * (max_len - len(p)) + p) for t, p in all_pnls.items()}
    df_pnl = pd.DataFrame(padded_pnls)

    # Each bot gets an equal slice of the $100k pie
    capital_per_bot = PORTFOLIO_STARTING_BALANCE / len(all_pnls)
    portfolio_equity = np.zeros(max_len)

    plt.figure(figsize=(14, 8))

    for ticker in df_pnl.columns:
        # Calculate compounded equity for this specific bot
        bot_equity = [capital_per_bot]
        for pnl in df_pnl[ticker]:
            bot_equity.append(bot_equity[-1] * (1 + pnl))

        bot_equity_array = np.array(bot_equity[1:])
        portfolio_equity += bot_equity_array

        # Plot individual bot performance faintly in the background
        plt.plot(bot_equity_array, label=ticker, alpha=0.3, linewidth=1.5)

    # Plot the Master Portfolio Curve heavily
    plt.plot(portfolio_equity, label='Total Portfolio', color='black', linewidth=3)

    # Calculate Master Metrics
    total_return = ((portfolio_equity[-1] - PORTFOLIO_STARTING_BALANCE) / PORTFOLIO_STARTING_BALANCE) * 100

    # Calculate Max Drawdown for the whole portfolio
    peak = np.maximum.accumulate(portfolio_equity)
    drawdown = (portfolio_equity - peak) / peak
    max_dd = np.min(drawdown) * 100

    print(f"Final Portfolio Balance: ${portfolio_equity[-1]:,.2f}")
    print(f"Total Fleet Return:    {total_return:,.2f}%")
    print(f"Fleet Max Drawdown:    {max_dd:,.2f}%")

    plt.title(
        f'AI Fleet Portfolio Performance (v{VERSION})\nStart: $100k | End: ${portfolio_equity[-1]:,.0f} | Max DD: {max_dd:.2f}%',
        fontsize=14)
    plt.ylabel('Account Balance ($)', fontsize=12)
    plt.xlabel('Testing Hours', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_portfolio()