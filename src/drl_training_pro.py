import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from collections import deque
import os

# --- CONFIGURATION ---
TICKER = "AUDCAD"
SPREAD_COST = 0.00018  # Typical spread + slippage buffer
CHURN_PENALTY = 0.00040
WINDOW_SIZE = 72

# Updated Observation Features based on Step 2 Enhancement
OBSERVATION_FEATURES = [
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Norm_Ret_12', 'H1_Norm_Ret_24',
    'Vol_Regime', 'FracDiff_Close', 'H1_Autocorr', 'H1_ZScore_50',
    'H1_ER', 'ATR_Ratio', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'Price_Stretch', 'MA_Speed', 'RSI_Velocity','ATR_Relative'  # <--- ADD THESE
]


class ForexTradingEnvPro(gym.Env):
    def __init__(self, df):
        super(ForexTradingEnvPro, self).__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # --- NEW CONSTRAINTS ---
        self.MIN_HOLD_TIME = 6  # Agent must hold a position for 6 bars
        self.hold_timer = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-5, high=5,
            shape=(WINDOW_SIZE, len(OBSERVATION_FEATURES)),
            dtype=np.float32
        )

        self.returns_history = deque(maxlen=48)
        self.current_step = WINDOW_SIZE
        self.current_position = 0

    def _get_observation(self):
        # We grab the window of data from the dataframe
        obs = self.df.loc[self.current_step - WINDOW_SIZE + 1: self.current_step, OBSERVATION_FEATURES].values

        # We use np.nan_to_num to prevent training crashes if a stray NaN exists
        # We also clip between -5 and 5 to keep gradients stable for the LSTM
        return np.clip(np.nan_to_num(obs), -5, 5).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.returns_history.clear()
        self.hold_timer = 0

        # Ensure we don't start too close to the end of the training set
        train_max = int(self.max_steps * 0.8) - 1
        self.current_step = np.random.randint(WINDOW_SIZE, train_max)
        self.current_position = 0

        return self._get_observation(), {}

    def step(self, action):
        # Action Map: 0=Flat, 1=Long, 2=Short
        target_position = {0: 0, 1: 1, 2: -1}[action]

        # 1. ENFORCE POSITION LOCK
        # If the timer hasn't expired, the agent CANNOT change its mind
        if self.hold_timer > 0:
            mapped_action = self.current_position
            self.hold_timer -= 1
        else:
            mapped_action = target_position
            # If we just opened a new non-flat position, start the timer
            if mapped_action != 0 and mapped_action != self.current_position:
                self.hold_timer = self.MIN_HOLD_TIME

        # 2. Harsh Cost Logic
        change_magnitude = abs(mapped_action - self.current_position)
        # We increase CHURN_PENALTY here to 0.00050
        costs = (SPREAD_COST * change_magnitude) + (0.00050 * (change_magnitude > 1))

        # 3. Performance (Log Returns)
        current_close = self.df.loc[self.current_step, 'Close']
        prev_close = self.df.loc[self.current_step - 1, 'Close']
        log_return = np.log(current_close / prev_close)

        step_pnl = (log_return * self.current_position) - costs
        self.returns_history.append(step_pnl)

        # 4. Volatility Normalized Reward
        vol = 1.0
        if len(self.returns_history) >= 20:
            vol = np.std(self.returns_history) + 1e-8

        current_atr_rel = self.df.loc[self.current_step, 'ATR_Relative']

        if mapped_action == 0:
            reward = 0.0  # Safe harbor. No free money, but no losses.
        else:
            # THE SNIPER RULE: If volatility is low, do not trade.
            if current_atr_rel < 0.8:
                # Massive penalty for trading in the chop
                reward = step_pnl - 0.002
            else:
                # Normal reward scaling for valid volatile regimes
                reward = step_pnl / (vol + 1e-8)

        self.current_position = mapped_action
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        return self._get_observation(), reward, terminated, False, {"pnl": step_pnl, "pos": self.current_position}

def main():
    print(f"--- Deep Reinforcement Learning: PRO Agent ({TICKER}) ---")

    data_path = f'data/{TICKER}_SUPER_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run Step 2 first!")
        return

    df = pd.read_csv(data_path)
    df.dropna(subset=OBSERVATION_FEATURES, inplace=True)

    train_df = df.iloc[:int(len(df) * 0.8)].copy()
    env = DummyVecEnv([lambda: ForexTradingEnvPro(train_df)])

    # Split Policy/Value Network architecture
    # Pi (Policy) decides the action, Qf (Value) estimates the reward
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        lstm_hidden_size=256,
        n_lstm_layers=2,
        ortho_init=True
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=1e-4,
        n_steps=8192,  # <--- INCREASED: Lets the model see longer sequences before updating
        batch_size=512,  # <--- INCREASED: Looks at more examples at once
        ent_coef=0.10,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda",
        patience=10
    )

    # Checkpoint every 500k steps
    checkpoint = CheckpointCallback(save_freq=500000, save_path='./models/', name_prefix='pro_agent')

    print("Beginning Training (2 Million Timesteps)...")
    model.learn(total_timesteps=2000000, callback=checkpoint)

    model.save(f"models/pro_lstm_{TICKER}_final")
    print(f"Model saved as models/pro_lstm_{TICKER}_final.zip")


if __name__ == "__main__":
    main()