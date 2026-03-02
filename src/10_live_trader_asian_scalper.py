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

# --- V9.0 ASIAN QUANT SCALPER CONFIGURATION ---
TICKERS = ['AUDCAD', 'AUDNZD', 'NZDCAD', 'EURGBP', 'EURAUD']
BROKER_SUFFIX = ""  # Change if your broker uses suffix like ".a"
MAGIC_NUMBER = 900001
DEVIATION = 20

# --- RISK & TARGET SETTINGS ---
RISK_PERCENT = 0.01  # 1% Risk per trade
SL_MULTIPLIER = 1.5  # 1.5 ATR Stop Loss
TP_MULTIPLIER = 1.0  # 1.0 ATR Take Profit (High Probability Scalp)
MAX_SPREAD_ATR = 0.15  # Max allowed spread is 15% of H1 ATR

# --- AI & QUANT SETTINGS ---
MIN_PROBABILITY = 0.60  # AI must be 60% confident
Z_SCORE_EXTREME = 2.0  # Must be 2 standard deviations stretched

# --- FEATURES ARRAY (Must match step 02 exactly) ---
SIGNAL_FEATURES = [
    'D1_Norm_Ret', 'H4_ER',
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Autocorr', 'H1_ZScore_50', 'H1_ER',
    'ATR_Ratio', 'DXY_Ret', 'SPY_Ret', 'US10Y_Ret',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]
RISK_FEATURES = SIGNAL_FEATURES

print(f"--- AI Forex Bot: V9.0 ASIAN QUANT SCALPER (Live) ---")

if not mt5.initialize():
    print("MT5 Init failed, error code =", mt5.last_error())
    sys.exit()

print(f"Connected to Account: {mt5.account_info().login}")

# Load AI Models
print("Loading AI Models...")
models = {}
for ticker in TICKERS:
    try:
        models[ticker] = {
            'signal': joblib.load(f'models/signal_{ticker}.joblib'),
            'risk': joblib.load(f'models/risk_{ticker}.joblib')
        }
        print(f"  [OK] Loaded models for {ticker}")
    except Exception as e:
        print(f"  [WARNING] Could not load {ticker}. Error: {e}")


def get_broker_symbol(ticker):
    return f"{ticker}{BROKER_SUFFIX}"


def get_live_macro():
    """Fetches the latest daily macro data from Yahoo Finance"""
    try:
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

    if tick_size == 0 or tick_value == 0: return symbol_info.volume_min

    sl_ticks = sl_dist_price / tick_size
    loss_per_lot = sl_ticks * tick_value

    if loss_per_lot == 0: return symbol_info.volume_min

    lots = risk_amount / loss_per_lot

    step = symbol_info.volume_step
    lots = math.floor(lots / step) * step
    lots = max(lots, symbol_info.volume_min)
    lots = min(lots, symbol_info.volume_max)
    return float(f"{lots:.2f}")


def get_live_features(trade_symbol):
    rates_h1 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_H1, 0, 100)
    rates_h4 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_H4, 0, 100)
    rates_d1 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_D1, 0, 100)

    if rates_h1 is None or len(rates_h1) < 60: return None, None

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
        server_time = datetime.utcfromtimestamp(rates_h1[-1]['time'])

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

        # Replace NaN with 0.0 to prevent XGBoost errors
        df_features = pd.DataFrame([features]).fillna(0.0)
        return df_features, h1_atr
    except Exception as e:
        print(f"  Feature Generation Error: {e}")
        return None, None


def close_position(pos, comment="Hard Time Exit"):
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None: return

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
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  [CLOSED] {pos.symbol} - {comment}")
    else:
        print(f"  [FAILED TO CLOSE] {pos.symbol} - Error: {res.comment if res else 'Unknown'}")


def strategy_tick():
    tick = mt5.symbol_info_tick(get_broker_symbol(TICKERS[0]))
    # Block empty ticks from the 1970 epoch
    if tick is None or tick.time == 0:
        print("  Waiting for valid MT5 ticks (Broker warming up)...")
        return

    server_time = datetime.utcfromtimestamp(tick.time)
    hour = server_time.hour
    minute = server_time.minute

    print(f"\n--- Bot Ping: {server_time.strftime('%Y-%m-%d %H:%M:%S')} (Broker Time) ---")

    # 1. HARD TIME EXIT: Active Session (07:00 to 18:00)
    is_active_session = 7 <= hour <= 18
    positions = mt5.positions_get()

    if positions:
        for pos in positions:
            if pos.magic == MAGIC_NUMBER and is_active_session:
                close_position(pos, "07:00 Session Exit")

    if is_active_session:
        print("  Status: [MONITORING] European/US session active. No new trades allowed.")
        return

    # 2. ROLLOVER SPREAD PAUSE (23:55 to 01:15)
    is_rollover = (hour == 23 and minute >= 55) or (hour == 0) or (hour == 1 and minute <= 15)
    if is_rollover:
        print("  Status: [PAUSED] Midnight Rollover active. Waiting for spreads to settle.")
        return

    print("  Status: [HUNTING] Asian Session Active. Scanning for Z-Score Extremes...")

    # 3. SCAN TICKERS
    for ticker in TICKERS:
        if ticker not in models: continue
        trade_symbol = get_broker_symbol(ticker)

        # Anti-Overlap: Only 1 trade per pair
        if mt5.positions_get(symbol=trade_symbol):
            continue

        data, h1_atr = get_live_features(trade_symbol)
        if data is None: continue

        z_score = data['H1_ZScore_50'].iloc[0]

        if abs(z_score) < Z_SCORE_EXTREME:
            continue

        probs = models[ticker]['signal'].predict_proba(data[SIGNAL_FEATURES])[0]
        prob_long, prob_short = probs[1], probs[2]
        vol_pred = models[ticker]['risk'].predict(data[RISK_FEATURES])[0]

        signal = 0
        if z_score < -Z_SCORE_EXTREME and prob_long > MIN_PROBABILITY:
            signal = 1
        elif z_score > Z_SCORE_EXTREME and prob_short > MIN_PROBABILITY:
            signal = -1

        if signal == 0:
            print(
                f"    ~ {ticker} Stretched (Z: {z_score:.2f}) but AI Rejected (Confidence Low L:{prob_long:.2f} S:{prob_short:.2f})")
            continue

        # Spread Safety Check
        symbol_info = mt5.symbol_info(trade_symbol)
        if symbol_info is None: continue

        live_spread_price = symbol_info.ask - symbol_info.bid
        max_allowed_spread_price = h1_atr * MAX_SPREAD_ATR

        if live_spread_price > max_allowed_spread_price:
            print(
                f"    [BLOCKED] {ticker} Spread too high! Live: {live_spread_price:.5f} | Max Allowed: {max_allowed_spread_price:.5f}")
            continue

        # --- EXECUTE TRADE ---
        direction = "BUY" if signal == 1 else "SELL"
        print(
            f"    >>> EXECUTING {direction} {ticker} | Z-Score: {z_score:.2f} | Conf: {max(prob_long, prob_short):.2f} <<<")

        sl_dist = float(vol_pred * SL_MULTIPLIER)
        tp_dist = float(vol_pred * TP_MULTIPLIER)

        price = float(symbol_info.ask if signal == 1 else symbol_info.bid)
        sl = float(price - sl_dist if signal == 1 else price + sl_dist)
        tp = float(price + tp_dist if signal == 1 else price - tp_dist)

        digits = symbol_info.digits
        price = round(price, digits)
        sl = round(sl, digits)
        tp = round(tp, digits)

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
            "comment": "V9_Asian_Scalp",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"    [SUCCESS] Opened {volume} {trade_symbol} @ {price}")
        else:
            print(f"    [FAILED] MT5 Error: {res.comment if res else 'Unknown'}")


if __name__ == "__main__":

    print("\nForcing MT5 to subscribe to live market data...")
    for ticker in TICKERS:
        trade_symbol = get_broker_symbol(ticker)
        success = mt5.symbol_select(trade_symbol, True)
        if not success:
            print(f"  [ERROR] MT5 cannot find '{trade_symbol}'.")
        else:
            print(f"  [OK] '{trade_symbol}' data stream activated.")

    print("  Waiting 3 seconds for broker data feeds to populate...")
    time.sleep(3)  # <--- ADD THIS LINE

    while True:
        try:
            strategy_tick()
        except Exception as e:
            print(f"Critical Loop Error: {e}")
        # Run every 5 minutes to ensure we don't miss the strict time exits
        time.sleep(60 * 5)