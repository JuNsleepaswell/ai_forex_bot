import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# IMPORT THE NEW CLASS DIRECTLY TO SYNC LOGIC
from drl_training_pro import ForexTradingEnvPro, OBSERVATION_FEATURES


def calculate_max_drawdown(equity_curve):
    equity_series = pd.Series(equity_curve)
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min() * 100


def run_backtest(ticker, version): # 1. Added version parameter
    print(f"--- Pro Backtester v2: Evaluating {ticker} (Version: {version}) ---")

    data_path = f'data/{ticker}_CLEAN_dataset.csv'

    # 2. UPDATED PATH LOGIC: Use the version string provided by the user
    best_path = f"./models/best_model_{ticker}_{version}/best_model.zip"
    final_path = f"models/pro_lstm_{ticker}_{version}_final.zip"

    if os.path.exists(best_path):
        model_path = best_path
        print(f"  [Status] Loading BEST model from validation phase.")
    elif os.path.exists(final_path):
        model_path = final_path
        print(f"  [Status] Loading FINAL model from end of training.")
    else:
        # This will now tell you EXACTLY where it looked and failed
        print(f"  [Error] No model found at:")
        print(f"   - {best_path}")
        print(f"   - {final_path}")
        return

    df = pd.read_csv(data_path)
    val_split = int(len(df) * 0.9)  # Test on the final 10% (pure unseen data)
    test_df = df.iloc[val_split:].copy().reset_index(drop=True)

    print(f"  [Data] Testing on {len(test_df)} unseen hours...")

    # Initialize Env (The class now handles the +2 features internally)
    env = DummyVecEnv([lambda: ForexTradingEnvPro(test_df)])
    model = RecurrentPPO.load(model_path, env=env)

    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    equity_curve = [100000.0]
    current_equity = 100000.0
    positions = []
    pnl_history = []

    for i in range(len(test_df) - 1):
        # Predict with LSTM state tracking
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        obs, reward, done, info = env.step(action)
        episode_starts = done

        step_pnl_pct = info[0]['pnl']
        dollar_pnl = current_equity * step_pnl_pct

        current_equity += dollar_pnl
        equity_curve.append(current_equity)
        positions.append(info[0]['pos'])
        pnl_history.append(step_pnl_pct)

        if done:
            break

    # Metrics
    total_return = ((equity_curve[-1] - 100000.0) / 100000.0) * 100
    avg_return = np.mean(pnl_history)
    std_return = np.std(pnl_history) + 1e-9
    sharpe = (avg_return / std_return) * np.sqrt(24 * 252)
    max_dd = calculate_max_drawdown(equity_curve)
    total_flips = np.sum(np.abs(np.diff(positions)))

    print(f"\n--- Performance Summary: {ticker} ---")
    print(f"Final Balance: ${equity_curve[-1]:,.2f}")
    print(f"Total Return:  {total_return:.2f}%")
    print(f"Annual Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown:  {max_dd:.2f}%")
    print(f"Total Flips:   {total_flips}")

    # Plot
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve, label='Equity Curve', color='blue')
    plt.title(f"Pro Agent Backtest v2: {ticker}")
    plt.ylabel("Account Value ($)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.fill_between(range(len(positions)), positions, color='gray', alpha=0.3, label='Position (1=L, -1=S)')
    plt.ylabel("Market Exposure")
    plt.xlabel("Hours")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--version", type=str, required=True, help="Format: YYYYMMDD_HHMM_Targeted")
    args = parser.parse_args()

    # 3. Pass both arguments to the function
    run_backtest(args.ticker, args.version)