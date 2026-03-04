import pandas as pd
import numpy as np
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your environment
from src.drl_training_pro import ForexTradingEnvPro, OBSERVATION_FEATURES

TICKERS = ["EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "USDJPY", "AUDUSD", "NZDUSD"]


def calculate_max_drawdown(equity_curve):
    equity_series = pd.Series(equity_curve)
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min() * 100


def evaluate_fleet():
    print("==================================================")
    print("📊 INITIATING PRO FLEET BACKTEST (UNSEEN DATA) 📊")
    print("==================================================\n")

    results = []

    for ticker in TICKERS:
        data_path = f'data/{ticker}_SUPER_dataset.csv'
        model_path = f"./models/best_model_{ticker}/best_model.zip"

        if not os.path.exists(data_path) or not os.path.exists(model_path):
            print(f"⚠️ Skipping {ticker}: Model or data missing.")
            continue

        # 1. Load Data & Isolate Test Set (Last 10%)
        df = pd.read_csv(data_path)
        val_split = int(len(df) * 0.9)
        test_df = df.iloc[val_split:].copy().reset_index(drop=True)

        # 2. Setup Env & Load Model
        env = DummyVecEnv([lambda: ForexTradingEnvPro(test_df)])
        model = RecurrentPPO.load(model_path, env=env)

        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        equity = 100000.0
        equity_curve = [equity]
        pnl_history = []

        # 3. Step through the test data
        for i in range(len(test_df) - 1):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True  # Important: True means the bot uses its best logic, no random exploration
            )

            obs, reward, done, info = env.step(action)
            episode_starts = done

            step_pnl_pct = info[0]['pnl']
            dollar_pnl = equity * step_pnl_pct
            equity += dollar_pnl

            equity_curve.append(equity)
            pnl_history.append(step_pnl_pct)

            if done:
                break

        # 4. Calculate Metrics
        total_return = ((equity_curve[-1] - 100000.0) / 100000.0) * 100
        avg_return = np.mean(pnl_history)
        std_return = np.std(pnl_history) + 1e-9
        sharpe = (avg_return / std_return) * np.sqrt(24 * 252)  # Annualized for hourly data
        max_dd = calculate_max_drawdown(equity_curve)

        results.append({
            "Pair": ticker,
            "Return (%)": round(total_return, 2),
            "Max DD (%)": round(max_dd, 2),
            "Sharpe": round(sharpe, 2),
            "Final Balance": f"${equity_curve[-1]:,.2f}"
        })

    # 5. Print Leaderboard
    if results:
        results_df = pd.DataFrame(results)
        # Sort by best return
        results_df = results_df.sort_values(by="Return (%)", ascending=False).reset_index(drop=True)
        print(results_df.to_markdown(index=False))
    else:
        print("No results to display.")


if __name__ == "__main__":
    evaluate_fleet()