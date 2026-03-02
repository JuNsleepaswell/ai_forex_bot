import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import importlib
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Import the environment we built in the previous script
# We can just redefine the config variables here to ensure they match
TICKER = "AUDCAD"
OBSERVATION_FEATURES = [
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Autocorr', 'H1_ZScore_50', 'H1_ER',
    'ATR_Ratio', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]

# We must import the exact class from your training script so the model recognizes it
drl_env = importlib.import_module("04_drl_training_environment")
ForexTradingEnv = drl_env.ForexTradingEnv


def main():
    print(f"--- DRL Backtester: Evaluating {TICKER} PPO Agent ---")

    # 1. Load the Model
    model_path = f"models/ppo_agent_{TICKER}.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Let the training finish first!")
        return

    print("Loading AI Brain...")
    model = PPO.load(model_path)

    # 2. Load the Data
    data_path = f'data/{TICKER}_SUPER_dataset.csv'
    df = pd.read_csv(data_path)
    df.dropna(subset=OBSERVATION_FEATURES, inplace=True)

    # 3. Extract the Out-of-Sample Test Data (The future it has never seen)
    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    print(f"Testing on {len(test_df)} unseen hours of data...")

    # 4. Initialize the Environment in "Test Mode"
    env = DummyVecEnv([lambda: ForexTradingEnv(test_df)])
    obs = env.reset()

    # Tracking Variables
    equity_curve = [100000.0]  # Start with $100k
    current_equity = 100000.0
    positions = []

    print("Simulating trades... Please wait.")

    # 5. The Evaluation Loop
    for i in range(len(test_df) - 24):  # -24 to account for WINDOW_SIZE
        # Let the AI look at the chart and predict the best action
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, done, info = env.step(action)

        # Calculate PnL (De-scale the reward back to real dollars based on $100k account)
        # info is a list of dicts because of DummyVecEnv
        actual_reward = info[0]['step_reward'] / 10000

        # Simulate risk (Assume the agent is trading 1 standard lot for simplicity)
        # $10 per pip rough estimate, mapped to price change
        pnl = actual_reward * 100000

        current_equity += pnl
        equity_curve.append(current_equity)
        positions.append(info[0]['position'])

        if done:
            break

    # 6. Analyze Results
    total_return = ((equity_curve[-1] - 100000.0) / 100000.0) * 100

    print(f"\n--- Final Results ---")
    print(f"Initial Capital: $100,000.00")
    print(f"Final Capital:   ${equity_curve[-1]:,.2f}")
    print(f"Total Return:    {total_return:.2f}%")

    # 7. Plotting the AI's Journey
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve, color='purple', label="DRL Agent Equity")
    plt.axhline(100000, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Deep Reinforcement Learning (PPO) Out-of-Sample Performance - {TICKER}")
    plt.xlabel("Simulated Hours")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()