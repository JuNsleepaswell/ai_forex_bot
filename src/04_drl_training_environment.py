import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# --- CONFIGURATION ---
TICKER = "AUDCAD"  # Let's use your best mean-reverting cross pair
SPREAD_COST = 0.00015  # Approximating 1.5 pips cost per trade
WINDOW_SIZE = 24  # The AI will look at the last 24 hours of data to make a decision

# The exact features the Neural Network is allowed to "see"
# Notice we DO NOT give it raw prices. Neural networks need scaled/normalized data.
OBSERVATION_FEATURES = [
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Autocorr', 'H1_ZScore_50', 'H1_ER',
    'ATR_Ratio', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]


class ForexTradingEnv(gym.Env):
    """A Custom Trading Environment for OpenAI Gymnasium"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df):
        super(ForexTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # Actions: 0 = Hold/Flat, 1 = Buy (Long), 2 = Sell (Short)
        self.action_space = spaces.Discrete(3)

        # Observation: A 2D array representing the last N bars of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WINDOW_SIZE, len(OBSERVATION_FEATURES)),
            dtype=np.float32
        )

        # Environment State Variables
        self.current_step = WINDOW_SIZE
        self.current_position = 0  # 0=Flat, 1=Long, -1=Short

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a random point in the dataset for better training variance
        self.current_step = np.random.randint(WINDOW_SIZE, int(self.max_steps * 0.8))
        self.current_position = 0
        return self._get_observation(), {}

    def _get_observation(self):
        # The AI "sees" the last 24 bars of the selected quantitative features
        obs = self.df.loc[self.current_step - WINDOW_SIZE + 1: self.current_step, OBSERVATION_FEATURES].values
        # Replace NaNs with 0 to prevent Neural Network explosion
        return np.nan_to_num(obs).astype(np.float32)

    def step(self, action):
        # Map Discrete Action (0, 1, 2) to Position (0, 1, -1)
        mapped_action = 0
        if action == 1:
            mapped_action = 1
        elif action == 2:
            mapped_action = -1

        # 1. Calculate transaction costs if we changed position
        transaction_cost = 0.0
        if mapped_action != self.current_position:
            # We pay the spread every time we flip our position
            transaction_cost = SPREAD_COST * abs(mapped_action - self.current_position)

        # Update current position
        self.current_position = mapped_action

        # Move forward one bar in time
        self.current_step += 1

        # 2. Calculate the Reward (The most important part of DRL)
        # We look at the actual price change of the current bar
        current_close = self.df.loc[self.current_step, 'Close']
        previous_close = self.df.loc[self.current_step - 1, 'Close']
        price_change = current_close - previous_close

        # Reward Formula: (Price Change * Position) - Costs
        step_reward = (price_change * self.current_position) - transaction_cost

        # We multiply the reward by a large number (e.g. 10000) so the neural network
        # isn't trying to learn from microscopic decimal numbers like 0.0001
        scaled_reward = step_reward * 10000

        # 3. Check if the game is over (we reached the end of the historical data)
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Info dictionary for debugging
        info = {'step_reward': scaled_reward, 'position': self.current_position}

        return self._get_observation(), scaled_reward, terminated, truncated, info


def main():
    print("--- Deep Reinforcement Learning: PPO Agent Training ---")

    # 1. Load the Quant Features Dataset
    data_path = f'data/{TICKER}_SUPER_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run Step 02 first.")
        return

    df = pd.read_csv(data_path)
    df.dropna(subset=OBSERVATION_FEATURES, inplace=True)
    print(f"Loaded {len(df)} bars of environment data for {TICKER}.")

    # 2. Split Data (Train on the past, Test on the future)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()

    # 3. Initialize the Environment
    env = DummyVecEnv([lambda: ForexTradingEnv(train_df)])

    # 4. Create the Neural Network Agent (PPO)
    print("Initializing PPO Neural Network...")
    model = PPO(
        "MlpPolicy",  # Multi-Layer Perceptron (Standard Neural Net)
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,  # Discount factor (cares about long-term future rewards)
        ent_coef=0.01,  # Entropy coefficient (forces AI to explore new strategies)
        verbose=1
    )

    # 5. Train the Agent (Let it play the game)
    print("Beginning Training Loop (This will take time)...")
    # 200,000 timesteps means the AI will experience 200,000 hours of simulated trading
    model.learn(total_timesteps=200000)

    # 6. Save the Brain
    os.makedirs('models', exist_ok=True)
    model.save(f"models/ppo_agent_{TICKER}")
    print(f"Training Complete. AI Brain saved as models/ppo_agent_{TICKER}.zip")


if __name__ == "__main__":
    main()