import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import joblib
import time
import sys
import math
from datetime import datetime

# --- CONFIGURATION ---
TICKERS = ['AUDCAD', 'AUDNZD', 'NZDCAD', 'EURGBP', 'EURAUD']
BROKER_SUFFIX = ""  # Change if your broker uses "AUDCAD.a" etc.
MAGIC_NUMBER = 888000
DEVIATION = 20

# --- RISK & TARGET SETTINGS ---
RISK_PERCENT = 0.01  # 1% Risk per trade
SL_MULTIPLIER = 1.5  # 1.5 ATR Stop Loss
TP_MULTIPLIER = 1.0  # 1.0 ATR Take Profit
MAX_SPREAD_ATR = 0.15  # Skip trade if live spread is > 15% of H1 ATR

# --- AI SETTINGS ---
MIN_PROBABILITY = 0.60  # Must be 60% confident
Z_SCORE_EXTREME = 2.0  # Must be 2 standard deviations stretched

# --- FEATURES ---
SIGNAL_FEATURES = [
    'D1_Norm_Ret', 'H4_ER',
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Autocorr', 'H1_ZScore_50', 'H1_ER',
    'ATR_Ratio', 'DXY_Ret', 'SPY_Ret', 'US10Y_Ret',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]
RISK_FEATURES = SIGNAL_FEATURES

print(f"--- AI Forex Bot: Asian Mean Reversion (Live) ---")

# 1. Initialize MT5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    sys.exit()

print(f"Connected to Account: {mt5.account_info().login}")

# 2. Load Models
print("Loading AI Models...")
models = {}
for ticker in TICKERS:
    try:
        models[ticker] = {
            'signal': joblib.load(f'models/signal_{ticker}.joblib'),
            'risk': joblib.load(f'models/risk_{ticker}.joblib')
        }
        print(f"  Loaded {ticker}")
    except Exception as e:
        print(f"  [ERROR] Could not load {ticker}. Did you run step 11 for this pair? Error: {e}")


# --- Helper Functions ---

def get_broker_symbol(ticker):
    return f"{ticker}{BROKER_SUFFIX}"


def get_live_macro():
    """Fetches the latest daily macro data from Yahoo Finance"""
    try:
        # Cross pairs don't rely heavily on this, but the model expects the columns
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
        spy = yf.Ticker("SPY").history(period="5d")
        tnx = yf.Ticker("^TNX").history(period="5d")
        return dxy['Close'].pct_change().iloc[-1], spy['Close'].pct_change().iloc[-1], tnx['Close'].pct_change().iloc[
            -1]
    except:
        return 0.0, 0.0, 0.0


def calculate_lot_size(trade_symbol, sl_dist_price):
    account = mt5.account_info()
    symbol_info = mt5.symbol_info(trade_symbol)

    risk_amount = account.balance * RISK_PERCENT
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value

    sl_ticks = sl_dist_price / tick_size
    loss_per_lot = sl_ticks * tick_value
    lots = risk_amount / loss_per_lot

    step = symbol_info.volume_step
    lots = math.floor(lots / step) * step
    lots = max(lots, symbol_info.volume_min)
    lots = min(lots, symbol_info.volume_max)
    return float(f"{lots:.2f}")


def get_live_features(trade_symbol):
    """Calculates the exact quantitative features for the live bar"""
    rates_h1 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_H1, 0, 100)
    rates_h4 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_H4, 0, 100)
    rates_d1 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_D1, 0, 100)

    if rates_h1 is None or len(rates_h1) < 60: return None

    df_h1 = pd.DataFrame(rates_h1)
    df_h4 = pd.DataFrame(rates_h4)
    df_d1 = pd.DataFrame(rates_d1)

    try:
        # D1 Features
        d1_atr = ta.atr(df_d1['high'], df_d1['low'], df_d1['close'], length=14).iloc[-1]
        d1_norm_ret = (df_d1['close'].iloc[-1] - df_d1['close'].iloc[-2]) / d1_atr

        # H4 Features
        h4_atr = ta.atr(df_h4['high'], df_h4['low'], df_h4['close'], length=14).iloc[-1]
        h4_er = ta.er(df_h4['close'], length=14).iloc[-1]

        # H1 Features
        h1_atr = ta.atr(df_h1['high'], df_h1['low'], df_h1['close'], length=14).iloc[-1]
        h1_ret_1 = df_h1['close'].pct_change(1)
        h1_norm_ret_1 = (df_h1['close'].iloc[-1] - df_h1['close'].iloc[-2]) / h1_atr
        h1_norm_ret_4 = (df_h1['close'].iloc[-1] - df_h1['close'].iloc[-5]) / h1_atr

        h1_autocorr = h1_ret_1.rolling(10).apply(lambda x: x.autocorr(), raw=False).iloc[-1]

        rolling_mean = df_h1['close'].rolling(50).mean().iloc[-1]
        rolling_std = df_h1['close'].rolling(50).std().iloc[-1]
        h1_zscore = (df_h1['close'].iloc[-1] - rolling_mean) / rolling_std

        h1_er = ta.er(df_h1['close'], length=10).iloc[-1]

        dxy_ret, spy_ret, us10y_ret = get_live_macro()

        server_time = datetime.fromtimestamp(rates_h1[-1]['time'])

        features = {
            'D1_Norm_Ret': d1_norm_ret,
            'H4_ER': h4_er,
            'H1_Norm_Ret_1': h1_norm_ret_1,
            'H1_Norm_Ret_4': h1_norm_ret_4,
            'H1_Autocorr': h1_autocorr,
            'H1_ZScore_50': h1_zscore,
            'H1_ER': h1_er,
            'ATR_Ratio': h1_atr / h4_atr if h4_atr != 0 else 1.0,
            'DXY_Ret': dxy_ret,
            'SPY_Ret': spy_ret,
            'US10Y_Ret': us10y_ret,
            'Hour_Sin': np.sin(2 * np.pi * server_time.hour / 24),
            'Hour_Cos': np.cos(2 * np.pi * server_time.hour / 24),
            'Day_Sin': np.sin(2 * np.pi * server_time.weekday() / 7),
            'Day_Cos': np.cos(2 * np.pi * server_time.weekday() / 7)
        }

        return pd.DataFrame([features]), h1_atr
    except Exception as e:
        print(f"  Feature Error: {e}")
        return None, None


def close_position(pos, comment="Hard Time Exit"):
    """Force closes an open position"""
    tick = mt5.symbol_info_tick(pos.symbol)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(request)
    print(f"  [CLOSED] {pos.symbol} - {comment}")


def strategy_tick():
    # 1. Check Server Time
    tick = mt5.symbol_info_tick(get_broker_symbol("AUDCAD"))
    if tick is None: return
    server_time = datetime.fromtimestamp(tick.time)
    hour = server_time.hour
    minute = server_time.minute

    print(f"\n--- Scan: {server_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 2. HARD TIME EXIT (Kill trades when Europe wakes up at 07:00 to 18:00)
    is_active_session = 7 <= hour <= 18
    positions = mt5.positions_get()

    if positions:
        for pos in positions:
            if pos.magic == MAGIC_NUMBER and is_active_session:
                close_position(pos, "07:00 Session Exit")

    # If we are in the active session, do not look for new trades.
    if is_active_session:
        print("  European/US Session Active. Monitoring only.")
        return

    # 3. ROLLOVER SPREAD PAUSE (Block entries from 23:55 to 01:15)
    if (hour == 23 and minute >= 55) or (hour == 0) or (hour == 1 and minute <= 15):
        print("  Rollover Pause Active. Waiting for spreads to settle.")
        return

    # 4. Scan Tickers for Mean Reversion
    for ticker in TICKERS:
        trade_symbol = get_broker_symbol(ticker)

        # Don't overlap trades on the same pair
        if mt5.positions_get(symbol=trade_symbol):
            continue

        data, h1_atr = get_live_features(trade_symbol)
        if data is None or ticker not in models: continue

        z_score = data['H1_ZScore_50'].iloc[0]

        # --- The Z-Score Filter ---
        if abs(z_score) < Z_SCORE_EXTREME:
            continue  # Market is not stretched enough

        # Get AI Prediction (0 = Flat, 1 = Long, 2 = Short)
        probs = models[ticker]['signal'].predict_proba(data[SIGNAL_FEATURES])[0]
        prob_long = probs[1]
        prob_short = probs[2]
        vol_pred = models[ticker]['risk'].predict(data[RISK_FEATURES])[0]

        signal = 0
        if z_score < -Z_SCORE_EXTREME and prob_long > MIN_PROBABILITY:
            signal = 1
        elif z_score > Z_SCORE_EXTREME and prob_short > MIN_PROBABILITY:
            signal = -1

        if signal == 0:
            print(
                f"  {ticker} Stretched (Z: {z_score:.2f}) but AI rejected (Prob L: {prob_long:.2f}, S: {prob_short:.2f})")
            continue

        # --- Spread Safety Check ---
        symbol_info = mt5.symbol_info(trade_symbol)
        live_spread = symbol_info.ask - symbol_info.bid
        max_allowed_spread = h1_atr * MAX_SPREAD_ATR

        if live_spread > max_allowed_spread:
            print(f"  [SKIPPED] {ticker} Spread too high! Live: {live_spread:.5f} | Max: {max_allowed_spread:.5f}")
            continue

        # --- EXECUTE TRADE ---
        print(f"  >> EXECUTING {ticker} | Z-Score: {z_score:.2f} | AI Confidence: {max(prob_long, prob_short):.2f}")

        sl_dist = vol_pred * SL_MULTIPLIER
        tp_dist = vol_pred * TP_MULTIPLIER

        digits = symbol_info.digits
        price = symbol_info.ask if signal == 1 else symbol_info.bid
        sl = price - sl_dist if signal == 1 else price + sl_dist
        tp = price + tp_dist if signal == 1 else price - tp_dist

        price = float(round(price, digits))
        sl = float(round(sl, digits))
        tp = float(round(tp, digits))

        volume = calculate_lot_size(trade_symbol, sl_dist)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": trade_symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "Asian Z-Sniper",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  SUCCESS! Opened {volume} {trade_symbol} @ {price}")
        else:
            err = res.comment if res else "No Response"
            print(f"  ORDER FAILED: {err}")


if __name__ == "__main__":
    while True:
        # Run exactly once every 15 minutes to align with candle closes
        strategy_tick()
        time.sleep(60 * 15)