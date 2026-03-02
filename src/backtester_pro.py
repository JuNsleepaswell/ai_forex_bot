import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# --- CONFIGURATION ---
TICKER = "AUDCAD"
OBSERVATION_FEATURES = [
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Norm_Ret_12', 'H1_Norm_Ret_24',
    'Vol_Regime', 'FracDiff_Close', 'H1_Autocorr', 'H1_ZScore_50',
    'H1_ER', 'ATR_Ratio', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'Price_Stretch', 'MA_Speed', 'RSI_Velocity', 'ATR_Relative'  # <--- ADDED THESE 4
]

# Re-defining the Env Class inside or importing it is necessary for the model to load
from drl_training_pro import ForexTradingEnvPro


def calculate_max_drawdown(equity_curve):
    equity_series = pd.Series(equity_curve)
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min() * 100


def main():
    print(f"--- Pro Backtester: Evaluating {TICKER} ---")

    model_path = "./models/best_model/best_model.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first!")
        return

    # 1. Load the Brain
    model = RecurrentPPO.load(model_path)

    # 2. Load and Prepare Test Data (Last 20% of the dataset)
    df = pd.read_csv(f'data/{TICKER}_SUPER_dataset.csv')
    df.dropna(subset=OBSERVATION_FEATURES, inplace=True)

    test_size = int(len(df) * 0.2)
    test_df = df.iloc[-test_size:].copy().reset_index(drop=True)
    print(f"Testing on {len(test_df)} unseen hours...")

    # 3. Environment Simulation
    env = DummyVecEnv([lambda: ForexTradingEnvPro(test_df)])
    obs = env.reset()

    # LSTM State tracking
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    equity_curve = [100000.0]
    current_equity = 100000.0
    positions = []
    pnl_history = []

    # 4. The Loop
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

    # 5. Professional Metrics
    total_return = ((equity_curve[-1] - 100000.0) / 100000.0) * 100
    avg_return = np.mean(pnl_history)
    std_return = np.std(pnl_history)
    sharpe = (avg_return / (std_return + 1e-9)) * np.sqrt(24 * 252)  # Annualized Sharpe
    max_dd = calculate_max_drawdown(equity_curve)

    print(f"\n--- Performance Summary ---")
    print(f"Final Balance: ${equity_curve[-1]:,.2f}")
    print(f"Total Return:  {total_return:.2f}%")
    print(f"Annual Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown:  {max_dd:.2f}%")
    print(f"Total Flips:   {np.sum(np.abs(np.diff(positions)))}")

    # 6. Plotting
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve, label='Equity Curve', color='blue')
    plt.title(f"Pro Agent Backtest: {TICKER}")
    plt.ylabel("Account Value ($)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.fill_between(range(len(positions)), positions, color='gray', alpha=0.3, label='Position (1=L, -1=S)')
    plt.ylabel("Market Exposure")
    plt.xlabel("Hours")
    plt.show()


if __name__ == "__main__":
    main()