import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure your env class is being imported successfully
from drl_training_pro import ForexTradingEnvPro, OBSERVATION_FEATURES


def calculate_max_drawdown(equity_curve):
    equity_series = pd.Series(equity_curve)
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min() * 100


def run_backtest(ticker):
    print(f"--- Pro Backtester: Evaluating {ticker} ---")

    # 1. Load the exact matching dataset and model for the requested ticker
    data_path = f'data/{ticker}_SUPER_dataset.csv'
    model_path = f"./models/best_model_{ticker}/best_model.zip"

    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Did the training run successfully for {ticker}?")
        return

    df = pd.read_csv(data_path)

    # 2. Extract out the test data
    val_split = int(len(df) * 0.9)
    test_df = df.iloc[val_split:].copy().reset_index(drop=True)

    print(f"Testing on {len(test_df)} unseen hours...")

    # 3. Initialize the environment
    env = DummyVecEnv([lambda: ForexTradingEnvPro(test_df)])
    model = RecurrentPPO.load(model_path, env=env)

    obs = env.reset()

    # LSTM State tracking
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    equity_curve = [100000.0]
    current_equity = 100000.0
    positions = []
    pnl_history = []

    # 4. Step through the test data
    for i in range(len(test_df) - 1):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        obs, reward, done, info = env.step(action)
        episode_starts = done

        # Extract real PnL from info
        step_pnl_pct = info[0]['pnl']
        dollar_pnl = current_equity * step_pnl_pct

        current_equity += dollar_pnl
        equity_curve.append(current_equity)
        positions.append(info[0]['pos'])
        pnl_history.append(step_pnl_pct)

        if done:
            break

    # 5. Calculate Metrics
    total_return = ((equity_curve[-1] - 100000.0) / 100000.0) * 100
    avg_return = np.mean(pnl_history)
    std_return = np.std(pnl_history)
    sharpe = (avg_return / (std_return + 1e-9)) * np.sqrt(24 * 252)
    max_dd = calculate_max_drawdown(equity_curve)

    # Using np.diff to calculate the number of actual flips (changing positions)
    total_flips = np.sum(np.abs(np.diff(positions)))

    print(f"\n--- Performance Summary ---")
    print(f"Final Balance: ${equity_curve[-1]:,.2f}")
    print(f"Total Return:  {total_return:.2f}%")
    print(f"Annual Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown:  {max_dd:.2f}%")
    print(f"Total Flips:   {total_flips}")

    # 6. Plot the results
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve, label='Equity Curve', color='blue')
    plt.title(f"Pro Agent Backtest: {ticker}")
    plt.ylabel("Account Value ($)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.fill_between(range(len(positions)), positions, color='gray', alpha=0.3, label='Position (1=L, -1=S)')
    plt.ylabel("Market Exposure")
    plt.xlabel("Hours")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="The currency pair to test")
    args = parser.parse_args()

    run_backtest(args.ticker)