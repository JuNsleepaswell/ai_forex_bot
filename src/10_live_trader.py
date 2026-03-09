import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pywt
from tsfracdiff import FractionalDifferentiator
import ta
from sb3_contrib import RecurrentPPO
import time
import sys
from datetime import datetime

# --- CONFIGURATION ---
TICKERS = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDCAD', 'USDJPY', 'AUDUSD', 'NZDUSD']
VERSION = "20260308_2132"
BROKER_SUFFIX = ""
MAGIC_NUMBER = 999000
LOT_SIZE = 0.10  # Fixed lot size for Paper Trading testing
WINDOW_SIZE = 72

OBSERVATION_FEATURES = [
    'FracDiff_Z', 'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Norm_Ret_12',
    'Vol_Regime', 'H1_Autocorr', 'H1_ZScore_50',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'RSI_Velocity', 'ATR_Relative'
]

print(f"--- 🚀 AI Fleet Paper Trader (v{VERSION}) ---")

# 1. Initialize MT5
if not mt5.initialize():
    print("MT5 initialize() failed. Error code =", mt5.last_error())
    sys.exit()
print(f"Connected to Account: {mt5.account_info().login}")

# 2. Load RL Models
print("Loading RecurrentPPO Models...")
models = {}
for ticker in TICKERS:
    try:
        models[ticker] = RecurrentPPO.load(f"models/pro_lstm_{ticker}_{VERSION}_final.zip")
        print(f"  Loaded {ticker}")
    except Exception as e:
        print(f"  [ERROR] Could not load {ticker}: {e}")


# --- DATA PREPROCESSING PIPELINE ---
def wavelet_denoise(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")[:len(data)]


def get_live_observation(ticker):
    """Fetches MT5 data, cleans it, and builds the 72-hour matrix for the LSTM"""
    symbol = f"{ticker}{BROKER_SUFFIX}"
    # Fetch 200 bars so we have enough data for rolling windows and FracDiff
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
    if rates is None or len(rates) < 200: return None, None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 1. Clean the Alpha
    df['Close_Clean'] = wavelet_denoise(df['close'].values)
    frac_diff = FractionalDifferentiator()
    df['FracDiff_Close'] = frac_diff.FitTransform(df[['Close_Clean']])
    df['FracDiff_Z'] = (df['FracDiff_Close'] - df['FracDiff_Close'].rolling(100).mean()) / (
                df['FracDiff_Close'].rolling(100).std() + 1e-8)

    # 2. Build Momentum & Volatility Features
    df['H1_ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['H1_Norm_Ret_1'] = df['close'].diff(1) / df['H1_ATR']
    df['H1_Norm_Ret_4'] = df['close'].diff(4) / df['H1_ATR']
    df['H1_Norm_Ret_12'] = df['close'].diff(12) / df['H1_ATR']
    df['H1_Autocorr'] = df['close'].pct_change().rolling(10).apply(lambda x: x.autocorr(), raw=False)
    df['H1_ZScore_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()

    df['Vol_Regime'] = df['H1_ATR'] / df['H1_ATR'].rolling(50).mean()
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['RSI_Velocity'] = df['RSI'].diff(3)
    df['ATR_Relative'] = df['H1_ATR'] / df['close']

    # Time features based on MT5 server time
    df['Hour_Sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df['time'].dt.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['time'].dt.dayofweek / 7)

    df.bfill(inplace=True)

    # 3. Get Current Position & PnL
    positions = mt5.positions_get(symbol=symbol)
    current_position = 0
    unrealized_pnl = 0.0

    if positions:
        pos = positions[0]
        current_position = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
        # Calculate PnL percentage exactly like the training env
        unrealized_pnl = ((df['close'].iloc[-1] - pos.price_open) / pos.price_open) * current_position * 100.0

    # 4. Construct the 72-Hour Matrix
    obs_df = df[OBSERVATION_FEATURES].iloc[-WINDOW_SIZE:].values
    agent_state = np.array([[current_position, unrealized_pnl]] * WINDOW_SIZE)
    obs_matrix = np.hstack((obs_df, agent_state))

    # Reshape for SB3: (batch_size=1, window_size=72, features=15)
    obs = np.clip(np.nan_to_num(obs_matrix), -5, 5).astype(np.float32)
    obs = np.expand_dims(obs, axis=0)

    return obs, current_position, symbol


# --- MT5 EXECUTION ---
def close_all_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions: return

    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "AI Bot Flip/Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)


def execute_trade(symbol, action_target):
    # Map RL target: 0=Flat, 1=Long, 2=Short
    target_pos = {0: 0, 1: 1, 2: -1}[action_target]

    close_all_positions(symbol)

    if target_pos != 0:
        tick = mt5.symbol_info_tick(symbol)
        order_type = mt5.ORDER_TYPE_BUY if target_pos == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if target_pos == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "AI Bot Entry",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ Executed: {'LONG' if target_pos == 1 else 'SHORT'} on {symbol}")
        else:
            print(f"  ❌ Failed: {res.comment}")


# --- MAIN LOOP ---
def run_fleet():
    print(f"\n--- Scanning Market: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    for ticker in TICKERS:
        if ticker not in models: continue

        obs, current_pos, symbol = get_live_observation(ticker)
        if obs is None: continue

        # Get Action from the LSTM Brain
        action, _ = models[ticker].predict(obs, deterministic=True)
        action = action.item()

        target_pos = {0: 0, 1: 1, 2: -1}[action]
        action_name = {0: "FLAT", 1: "LONG", 2: "SHORT"}[action]

        # If the bot wants to change its position, execute!
        if target_pos != current_pos:
            print(f"⚡ {ticker} Action Required: Current=({current_pos}) -> Target=({action_name})")
            execute_trade(symbol, action)
        else:
            print(f"  {ticker}: Holding current state ({action_name})")


if __name__ == "__main__":
    # RL was trained on H1 data, so checking every 60 minutes on the hour is optimal
    # For testing, you can run it manually to see if it executes.
    while True:
        run_fleet()
        # Sleep for 1 hour (3600 seconds)
        print("Sleeping until next hourly candle...")
        time.sleep(3600)