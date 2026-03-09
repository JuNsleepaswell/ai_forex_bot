import pandas as pd
import numpy as np
import gymnasium as gym
import time # Ensure this is imported
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
import argparse # <--- Add this at the very top of your file with the other imports
from collections import deque
import os

# --- CONFIGURATION ---
SPREAD_COST = 0.00018  # Typical spread + slippage buffer
CHURN_PENALTY = 0.00040
WINDOW_SIZE = 72

# Updated Observation Features to use the "Clean Alpha"
OBSERVATION_FEATURES = [
    'FracDiff_Z',      # <--- Our new "Master Signal"
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Norm_Ret_12', # Keep these for momentum
    'Vol_Regime', 'H1_Autocorr', 'H1_ZScore_50',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'RSI_Velocity', 'ATR_Relative'
]


def make_env(df):
    # This helper function is required for SubprocVecEnv
    def _init():
        return ForexTradingEnvPro(df)

    return _init

class ForexTradingEnvPro(gym.Env):
    def __init__(self, df):
        super(ForexTradingEnvPro, self).__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # --- REMOVED THE ARTIFICIAL LEASH ---
        # No more MIN_HOLD_TIME or hold_timer. The bot is free.

        self.action_space = spaces.Discrete(3)

        # --- NEW: EXPANDED OBSERVATION SPACE ---
        # We add +2 to the feature length to account for Position and Unrealized PnL
        self.obs_shape = len(OBSERVATION_FEATURES) + 2

        self.observation_space = spaces.Box(
            low=-5, high=5,
            shape=(WINDOW_SIZE, self.obs_shape),
            dtype=np.float32
        )

        self.returns_history = deque(maxlen=48)
        self.current_step = WINDOW_SIZE
        self.current_position = 0

        # NEW: Track the exact price we entered the trade to calculate live PnL
        self.entry_price = 0.0

    def _get_observation(self):
        # 1. Get the base market features
        obs_df = self.df.loc[self.current_step - WINDOW_SIZE + 1: self.current_step, OBSERVATION_FEATURES].values

        # 2. Calculate Unrealized PnL for the CURRENT step
        current_close = self.df.loc[self.current_step, 'Close']
        if self.current_position != 0 and self.entry_price > 0:
            # Multiplied by 100 so the neural network can clearly "see" the percentage change
            unrealized_pnl = ((current_close - self.entry_price) / self.entry_price) * self.current_position * 100.0
        else:
            unrealized_pnl = 0.0

        # 3. Create the Agent State matrix
        # We broadcast the current position and PnL across the window so it matches the 2D shape
        agent_state = np.array([[self.current_position, unrealized_pnl]] * WINDOW_SIZE)

        # 4. Stitch the market data and agent state together
        obs = np.hstack((obs_df, agent_state))

        return np.clip(np.nan_to_num(obs), -5, 5).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.returns_history.clear()

        train_max = int(self.max_steps * 0.8) - 1
        self.current_step = np.random.randint(WINDOW_SIZE, train_max)

        # Reset internal states
        self.current_position = 0
        self.entry_price = 0.0

        return self._get_observation(), {}

    def step(self, action):
        # Action Map: 0=Flat, 1=Long, 2=Short
        target_position = {0: 0, 1: 1, 2: -1}[action]
        current_close = self.df.loc[self.current_step, 'Close']

        # 1. Update Entry Price Logic
        if target_position != self.current_position:
            if target_position != 0:
                self.entry_price = current_close  # We opened a new trade, record the price
            else:
                self.entry_price = 0.0  # We went flat

        # 2. Transaction Costs (The true enemy)
        change_magnitude = abs(target_position - self.current_position)
        costs = (SPREAD_COST * change_magnitude) + (0.00050 * (change_magnitude > 1))

        # 3. Step PnL (Calculated based on PREVIOUS position)
        prev_close = self.df.loc[self.current_step - 1, 'Close']
        log_return = np.log(current_close / prev_close)

        step_pnl = (log_return * self.current_position) - costs
        self.returns_history.append(step_pnl)

        # 4. Pure Sharpe Reward (Removed arbitrary drawdowns/comfort zones)
        # 4. Pure Sharpe Reward
        if len(self.returns_history) < 20:
            reward = step_pnl * 10.0
        else:
            mean_ret = np.mean(self.returns_history)
            std_ret = np.std(self.returns_history) + 1e-8
            reward = (mean_ret / std_ret) * 10.0

        # --- ADD THE INACTIVITY PENALTY HERE ---
        if self.current_position == 0:
            # A tiny negative reward every hour the bot does nothing
            # This forces it to at least try to find profitable setups
            reward -= 0.001

        self.current_position = target_position
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        return self._get_observation(), reward, terminated, False, {"pnl": step_pnl, "pos": self.current_position}


def main(ticker, version_id, ent_coef, lr): # <--- Ensure these 4 are here
    # FIX 2: Use the passed version_id instead of generating a new one
    run_name = f"{ticker}_{version_id}"

    print(f"--- Deep Reinforcement Learning: PRO Agent ({run_name}) ---")

    data_path = f'data/{ticker}_CLEAN_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Skipping {ticker}.")
        return

    df = pd.read_csv(data_path)
    df.dropna(subset=OBSERVATION_FEATURES, inplace=True)

    # Split Data
    train_split = int(len(df) * 0.8)
    val_split = int(len(df) * 0.9)
    train_df = df.iloc[:train_split].copy()
    val_df = df.iloc[train_split:val_split].copy()

    # Vectorized Environments
    n_envs = 2
    env = SubprocVecEnv([make_env(train_df) for _ in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(val_df) for _ in range(n_envs)])

    # Model Configuration
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        lstm_hidden_size=256,
        n_lstm_layers=2,
        ortho_init=True
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,
        ent_coef=ent_coef,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda",
        tensorboard_log=f"./tensorboard_logs/{run_name}/"
    )

    # --- CALLBACKS ---
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=40, min_evals=10, verbose=1)

    best_model_path = f'./models/best_model_{run_name}/'
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        eval_freq=10000,
        callback_after_eval=stop_train_callback,
        verbose=1
    )

    callback_list = CallbackList([eval_callback])

    # --- START TRAINING ---
    print(f"Beginning Training for {ticker} (2 Million Timesteps)...")
    model.learn(total_timesteps=2000000, callback=callback_list)

    final_save_path = f"models/pro_lstm_{run_name}_final"
    model.save(final_save_path)
    print(f"✅ Model saved as {final_save_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    # ADD THESE TWO NEW ARGUMENTS:
    parser.add_argument("--ent_coef", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main(args.ticker, args.version, args.ent_coef, args.lr)