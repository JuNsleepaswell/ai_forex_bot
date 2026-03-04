import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
import sys
from datetime import datetime
from sb3_contrib import RecurrentPPO

# --- CONFIGURATION ---
TICKER = 'XAUUSD'
BROKER_SUFFIX = ""
MAGIC_NUMBER = 999100
TIMEFRAME = mt5.TIMEFRAME_H1  # The bot trained on H1 data!
WINDOW_SIZE = 72
RISK_PERCENT = 0.01  # 1% Risk for the catastrophic Stop Loss

# EXACTLY match the training features
OBSERVATION_FEATURES = [
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Norm_Ret_12', 'H1_Norm_Ret_24',
    'Vol_Regime', 'FracDiff_Close', 'H1_Autocorr', 'H1_ZScore_50',
    'H1_ER', 'ATR_Ratio', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'Price_Stretch', 'MA_Speed', 'RSI_Velocity', 'ATR_Relative'
]

print(f"--- AI Forex Bot: DRL PRO Sniper ({TICKER}) ---")

# 1. Initialize MT5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    sys.exit()
print(f"Connected to Account: {mt5.account_info().login}")

trade_symbol = f"{TICKER}{BROKER_SUFFIX}"
if not mt5.symbol_select(trade_symbol, True):
    print(f"Failed to select {trade_symbol}")
    mt5.shutdown()
    sys.exit()

# 2. Load the Brain
model_path = f"./models/best_model_{TICKER}/best_model.zip"
if not os.path.exists(model_path):
    print(f"Error: Could not find {model_path}")
    sys.exit()

print("Loading LSTM Network...")
model = RecurrentPPO.load(model_path)
lstm_states = None  # The bot's short-term memory


# --- HELPER FUNCTIONS ---

def get_current_mt5_position():
    """Returns 0 (Flat), 1 (Long), or 2 (Short) based on live MT5 positions."""
    positions = mt5.positions_get(symbol=trade_symbol)
    if not positions:
        return 0, None

    pos = positions[0]
    if pos.type == mt5.ORDER_TYPE_BUY:
        return 1, pos
    elif pos.type == mt5.ORDER_TYPE_SELL:
        return 2, pos
    return 0, None


def close_position(position):
    """Sends an opposite market order to close the current open position."""
    tick = mt5.symbol_info_tick(trade_symbol)

    if position.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": trade_symbol,
        "volume": position.volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  [CLOSED] Position {position.ticket} closed successfully.")
    else:
        print(f"  [ERROR] Failed to close position: {result.comment}")


def open_position(target_action, atr_value):
    """Opens a new position with a catastrophic Hard Stop Loss based on ATR."""
    tick = mt5.symbol_info_tick(trade_symbol)
    symbol_info = mt5.symbol_info(trade_symbol)

    # The AI manages exits, but we use a 3x ATR hard stop just in case MT5 crashes
    hard_sl_distance = atr_value * 3.0

    if target_action == 1:  # LONG
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - hard_sl_distance
    elif target_action == 2:  # SHORT
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + hard_sl_distance
    else:
        return

    # Basic lot sizing (You can replace this with your dynamic lot calc)
    volume = symbol_info.volume_min

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": trade_symbol,
        "volume": volume,
        "type": order_type,
        "price": float(round(price, symbol_info.digits)),
        "sl": float(round(sl, symbol_info.digits)),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI DRL Open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        direction = "LONG" if target_action == 1 else "SHORT"
        print(f"  [OPENED] {direction} at {price} (Vol: {volume})")
    else:
        print(f"  [ERROR] Failed to open position: {result.comment}")


def apply_frac_diff(series, d=0.4):  # Matched to your d=0.4 call
    weights = np.array([1.0, -d, d * (d - 1) / 2, -d * (d - 1) * (d - 2) / 6])
    res = series.rolling(window=len(weights)).apply(lambda x: np.dot(x[::-1], weights), raw=True)
    return res.fillna(0)


def calculate_features(df):
    """
    Live implementation of the V7.0 PRO Feature Engineering.
    Converts live MT5 H1 data into the exact format the LSTM trained on.
    """
    # 1. H1 Basic Math
    df['H1_ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Stationary Velocity
    df['H1_Norm_Ret_1'] = df['Close'].diff(1) / (df['H1_ATR'] + 1e-9)
    df['H1_Norm_Ret_4'] = df['Close'].diff(4) / (df['H1_ATR'] + 1e-9)
    df['H1_Norm_Ret_12'] = df['H1_Norm_Ret_1'].shift(12)
    df['H1_Norm_Ret_24'] = df['H1_Norm_Ret_1'].shift(24)

    # Volatility Regimes
    df['Vol_Regime'] = df['H1_ATR'] / (df['H1_ATR'].rolling(168).mean() + 1e-9)
    df['ATR_Relative'] = df['H1_ATR'] / (df['H1_ATR'].rolling(168).mean() + 1e-9)

    # FracDiff
    df['FracDiff_Close'] = apply_frac_diff(df['Close'], d=0.4)

    # Autocorrelation
    df['H1_Ret_1'] = df['Close'].pct_change(1)
    df['H1_Autocorr'] = df['H1_Ret_1'].rolling(10).apply(lambda x: x.autocorr() if x.std() > 0 else 0, raw=False)

    # Stretch & Speed
    rolling_mean_100 = df['Close'].rolling(100).mean()
    rolling_std_100 = df['Close'].rolling(100).std()
    df['Price_Stretch'] = (df['Close'] - rolling_mean_100) / (rolling_std_100 + 1e-9)
    df['MA_Speed'] = rolling_mean_100.diff(3) / (df['H1_ATR'] + 1e-9)

    # RSI & Z-Score
    rsi = ta.rsi(df['Close'], length=14)
    df['RSI_Velocity'] = rsi.diff(1)

    rolling_mean_50 = df['Close'].rolling(50).mean()
    rolling_std_50 = df['Close'].rolling(50).std()
    df['H1_ZScore_50'] = (df['Close'] - rolling_mean_50) / (rolling_std_50 + 1e-9)

    df['H1_ER'] = ta.er(df['Close'], length=10)

    # 2. Time Features
    df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # 3. Dynamic H4 Resampling for ATR_Ratio
    # We must synthesize H4 candles from our live H1 stream to match your original script
    df_h4 = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    df_h4['H4_ATR'] = ta.atr(df_h4['High'], df_h4['Low'], df_h4['Close'], length=14)

    # Map the H4 ATR back to the live H1 DataFrame
    df = pd.merge_asof(df, df_h4[['H4_ATR']], left_index=True, right_index=True, direction='backward')
    df['ATR_Ratio'] = df['H1_ATR'] / (df['H4_ATR'] + 1e-9)

    return df
# --- MAIN LIVE LOOP ---

print("Bot is live. Waiting for the next H1 candle close...")

# We only want to predict ONCE per hour, right when the new H1 candle opens
last_processed_hour = -1

while True:
    time.sleep(1)  # Check every second
    current_time = datetime.now()

    if current_time.hour != last_processed_hour and current_time.minute == 0 and current_time.second < 10:
        print(f"\n--- AI Evaluation: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        # 1. Fetch enough history to calculate indicators + the 72 window
        rates = mt5.copy_rates_from_pos(trade_symbol, TIMEFRAME, 0, 300)
        if rates is None: continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'},
                  inplace=True)

        # 2. Build Features
        df = calculate_features(df)
        df.dropna(inplace=True)

        if len(df) < WINDOW_SIZE:
            print("  Not enough data to form a 72-hour window. Skipping.")
            continue

        # 3. Extract the exact 72x18 grid
        latest_window = df[OBSERVATION_FEATURES].iloc[-WINDOW_SIZE:].values
        # Match the normalization/clipping from training: np.clip(np.nan_to_num(obs), -5, 5)
        obs = np.clip(np.nan_to_num(latest_window), -5, 5).astype(np.float32)

        # 4. Neural Network Prediction
        # We pass deterministic=True for strict mathematical trading, and update lstm_states
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        target_action = int(action)

        print(f"  Network Output State: {target_action}")

        # 5. Execution Logic
        current_state, open_pos = get_current_mt5_position()

        if target_action == current_state:
            print("  Action matches current state. Holding.")
        else:
            print(f"  State Shift: {current_state} -> {target_action}. Executing...")

            # Step A: Close existing if we have one
            if current_state != 0:
                close_position(open_pos)
                time.sleep(1)  # Give MT5 a second to process

            # Step B: Open new if target is not flat
            if target_action != 0:
                # We grab a proxy ATR for the hard stop loss calculation
                current_atr = df['High'].iloc[-1] - df['Low'].iloc[-1]
                open_position(target_action, current_atr)

        last_processed_hour = current_time.hour