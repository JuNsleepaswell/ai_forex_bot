import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Import your pro environment
from drl_training_pro import ForexTradingEnvPro, OBSERVATION_FEATURES


def run_simulation(model, df, spread):
    """Runs a full backtest with a custom forced spread."""

    # We monkey-patch the spread for this specific run
    class StressEnv(ForexTradingEnvPro):
        def step(self, action):
            # Temporarily override global SPREAD_COST for this instance
            actual_spread = spread
            mapped_action = {0: 0, 1: 1, 2: -1}[action]
            change_magnitude = abs(mapped_action - self.current_position)

            # Apply the stressed cost
            costs = (actual_spread * change_magnitude) + (0.00012 * (change_magnitude > 1))

            current_close = self.df.loc[self.current_step, 'Close']
            prev_close = self.df.loc[self.current_step - 1, 'Close']
            log_return = np.log(current_close / prev_close)

            step_pnl = (log_return * self.current_position) - costs
            self.current_position = mapped_action
            self.current_step += 1

            obs = self._get_observation()
            done = self.current_step >= self.max_steps
            return obs, step_pnl, done, False, {}

    env = DummyVecEnv([lambda: StressEnv(df)])
    obs = env.reset()

    equity = 100000.0
    history = [equity]
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for _ in range(len(df) - 1):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, pnl_pct, done, _, _ = env.step(action)

        equity *= (1 + pnl_pct)
        history.append(equity)
        if done: break

    return history


def main():
    ticker = "AUDCAD"
    model_path = f"models/pro_lstm_{ticker}_final.zip"

    if not os.path.exists(model_path):
        print("Waiting for training to finish...")
        return

    model = RecurrentPPO.load(model_path)
    df = pd.read_csv(f'data/{ticker}_SUPER_dataset.csv').iloc[-5000:].reset_index(drop=True)

    # Stress Levels (in Pips/Price units)
    # 0.00010 = 1.0 Pip
    stress_levels = {
        "Tight (1.0 pip)": 0.00010,
        "Standard (1.8 pips)": 0.00018,
        "Wide (3.5 pips)": 0.00035,
        "Extreme (6.0 pips)": 0.00060
    }

    plt.figure(figsize=(12, 6))
    for label, spread_val in stress_levels.items():
        print(f"Running Stress Test: {label}...")
        curve = run_simulation(model, df, spread_val)
        plt.plot(curve, label=label)

    plt.title(f"Slippage Stress Test - {ticker} LSTM Agent")
    plt.ylabel("Equity ($)")
    plt.xlabel("Hours")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()