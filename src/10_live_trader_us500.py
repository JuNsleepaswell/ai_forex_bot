# src/10_live_trader_old.py (US500 AI PRODUCTION ENGINE)

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import joblib
import time
import sys
import math
from datetime import datetime, timedelta

# --- CONFIGURATION ---
TICKER = "US500"  # Update to match your exact MT5 Market Watch symbol
MAGIC_NUMBER = 777500
RISK_PERCENT = 0.01  # 1% Risk per trade
HOLD_HOURS = 72  # Close trades after 72 hours
SL_MULTIPLIER = 4.0  # 4x ATR Stop Loss
MIN_REQUIRED_MOVE = 0.25  # AI must predict 0.25 ATR move

# --- AI FEATURES ---
SIGNAL_FEATURES = [
    'D1_RSI', 'H4_ATR', 'H4_RSI', 'H1_RSI', 'H1_MACD', 'H1_MACD_Signal', 'H1_ATR',
    'Dist_H1_EMA', 'Dist_H4_EMA', 'Dist_D1_EMA',
    'Ret_1', 'Ret_4', 'Ret_24',
    'ATR_Ratio', 'DXY_Ret', 'SPY_Ret', 'US10Y_Ret',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]

print(f"--- AI Equity Engine: V8.0 LIVE ({TICKER}) ---")

if not mt5.initialize():
    print("MT5 Init failed.")
    sys.exit()

print(f"Connected to Account: {mt5.account_info().login}")

# Load AI
try:
    sig_model = joblib.load(f'models/signal_{TICKER}.joblib')
    risk_model = joblib.load(f'models/risk_{TICKER}.joblib')
    print("AI Models Loaded Successfully.")
except Exception as e:
    print(f"Failed to load AI Models: {e}")
    sys.exit()


def get_live_macro():
    """Fetches the latest daily macro data from Yahoo Finance"""
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
        spy = yf.Ticker("SPY").history(period="5d")
        tnx = yf.Ticker("^TNX").history(period="5d")

        dxy_ret = dxy['Close'].pct_change().iloc[-1]
        spy_ret = spy['Close'].pct_change().iloc[-1]
        us10y_ret = tnx['Close'].pct_change().iloc[-1]

        return dxy_ret, spy_ret, us10y_ret
    except Exception as e:
        print(f"Macro fetch failed: {e}")
        return 0.0, 0.0, 0.0


def get_live_features():
    """Generates the exact feature vector for the AI"""
    # Get H1 (Base)
    rates_h1 = mt5.copy_rates_from_pos(TICKER, mt5.TIMEFRAME_H1, 1, 100)
    df_h1 = pd.DataFrame(rates_h1)
    df_h1['close'] = df_h1['close'].astype(float)

    # Get H4
    rates_h4 = mt5.copy_rates_from_pos(TICKER, mt5.TIMEFRAME_H4, 1, 100)
    df_h4 = pd.DataFrame(rates_h4)

    # Get D1
    rates_d1 = mt5.copy_rates_from_pos(TICKER, mt5.TIMEFRAME_D1, 1, 100)
    df_d1 = pd.DataFrame(rates_d1)

    try:
        # Calculate Indicators on the most recent CLOSED candle
        h1_close = df_h1['close'].iloc[-1]

        h1_ema = ta.ema(df_h1['close'], length=50).iloc[-1]
        h1_rsi = ta.rsi(df_h1['close'], length=14).iloc[-1]
        h1_atr = ta.atr(df_h1['high'], df_h1['low'], df_h1['close'], length=14).iloc[-1]
        macd = ta.macd(df_h1['close'])
        h1_macd = macd['MACD_12_26_9'].iloc[-1]
        h1_macd_sig = macd['MACDs_12_26_9'].iloc[-1]

        h4_ema = ta.ema(df_h4['close'], length=200).iloc[-1]
        h4_atr = ta.atr(df_h4['high'], df_h4['low'], df_h4['close'], length=14).iloc[-1]
        h4_rsi = ta.rsi(df_h4['close'], length=14).iloc[-1]

        d1_ema = ta.ema(df_d1['close'], length=200).iloc[-1]
        d1_rsi = ta.rsi(df_d1['close'], length=14).iloc[-1]

        # Fetch Macro
        dxy_ret, spy_ret, us10y_ret = get_live_macro()

        # Build Feature Dictionary
        features = {
            'D1_RSI': d1_rsi,
            'H4_ATR': h4_atr,
            'H4_RSI': h4_rsi,
            'H1_RSI': h1_rsi,
            'H1_MACD': h1_macd,
            'H1_MACD_Signal': h1_macd_sig,
            'H1_ATR': h1_atr,
            'Dist_H1_EMA': (h1_close - h1_ema) / h1_close,
            'Dist_H4_EMA': (h1_close - h4_ema) / h1_close,
            'Dist_D1_EMA': (h1_close - d1_ema) / h1_close,
            'Ret_1': df_h1['close'].pct_change(1).iloc[-1],
            'Ret_4': df_h1['close'].pct_change(4).iloc[-1],
            'Ret_24': df_h1['close'].pct_change(24).iloc[-1],
            'ATR_Ratio': h1_atr / h4_atr if h4_atr != 0 else 1.0,
            'DXY_Ret': dxy_ret,
            'SPY_Ret': spy_ret,
            'US10Y_Ret': us10y_ret
        }

        # Time Features
        now = datetime.now()
        features['Hour_Sin'] = np.sin(2 * np.pi * now.hour / 24)
        features['Hour_Cos'] = np.cos(2 * np.pi * now.hour / 24)
        features['Day_Sin'] = np.sin(2 * np.pi * now.weekday() / 7)
        features['Day_Cos'] = np.cos(2 * np.pi * now.weekday() / 7)

        return pd.DataFrame([features])

    except Exception as e:
        print(f"Error building features: {e}")
        return None


def calculate_lot_size(sl_distance_price):
    account = mt5.account_info()
    symbol_info = mt5.symbol_info(TICKER)

    risk_amount = account.balance * RISK_PERCENT
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value

    sl_ticks = sl_distance_price / tick_size
    loss_per_lot = sl_ticks * tick_value

    lots = risk_amount / loss_per_lot
    lots = max(lots, symbol_info.volume_min)
    lots = min(lots, symbol_info.volume_max)

    # Cap Retail Lots
    lots = min(lots, 50.0)

    # Round to volume step
    step = symbol_info.volume_step
    lots = math.floor(lots / step) * step
    return float(f"{lots:.2f}")


def manage_open_trades():
    """Closes trades that have been open for exactly 72 hours"""
    positions = mt5.positions_get(symbol=TICKER)
    if not positions: return 0

    current_time = datetime.now()
    open_trades_count = 0

    for pos in positions:
        if pos.magic != MAGIC_NUMBER: continue
        open_trades_count += 1

        open_time = datetime.fromtimestamp(pos.time)
        hours_open = (current_time - open_time).total_seconds() / 3600.0

        if hours_open >= HOLD_HOURS:
            print(f"  [TIME EXIT] Closing Trade #{pos.ticket} (Open for {hours_open:.1f} hrs)")
            tick = mt5.symbol_info_tick(TICKER)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": TICKER,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": "Time Exit 72H",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(request)
            open_trades_count -= 1

    return open_trades_count


def strategy_tick():
    print(f"\n--- Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. Manage Time-Based Exits
    open_trades = manage_open_trades()

    # 2. Check Time Filter (Only buy 6 AM to 12 PM server time)
    current_hour = datetime.now().hour
    if current_hour < 6 or current_hour > 12:
        print("  Outside execution window. Monitoring exits only.")
        return

    # 3. Anti-Overlap: Max 1 Trade per day.
    # Check if we already opened a trade today
    today_start = datetime.now().replace(hour=0, minute=0, second=0)
    deals = mt5.history_deals_get(today_start, datetime.now())
    traded_today = any(
        d.symbol == TICKER and d.magic == MAGIC_NUMBER and d.entry == mt5.DEAL_ENTRY_IN for d in (deals or []))

    if traded_today or open_trades > 0:
        print("  Anti-Overlap Active. Already traded today or position open.")
        return

    # 4. Get Features & AI Prediction
    data = get_live_features()
    if data is None: return

    pred_return = sig_model.predict(data[SIGNAL_FEATURES])[0]
    pred_vol = risk_model.predict(data[RISK_FEATURES])[0]

    h1_atr = data['H1_ATR'].iloc[0]
    min_move = h1_atr * MIN_REQUIRED_MOVE

    print(f"  AI Pred Return: {pred_return:.5f} | Required: {min_move:.5f} | H1 ATR: {h1_atr:.5f}")

    prob_win = sig_model.predict_proba(data[SIGNAL_FEATURES])[0][1]

    if prob_win > 0.65:  # e.g., AI is 65% confident it's a win
        print(f"  *** AI BUY SIGNAL GENERATED (Confidence: {prob_win:.2f}) ***")

        tick = mt5.symbol_info_tick(TICKER)
        price = tick.ask

        sl_dist_price = pred_vol * SL_MULTIPLIER
        sl = price - sl_dist_price

        # Rounding
        digits = mt5.symbol_info(TICKER).digits
        price = float(round(price, digits))
        sl = float(round(sl, digits))

        vol = calculate_lot_size(sl_dist_price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": TICKER,
            "volume": vol,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": 0.0,  # NO TAKE PROFIT. Time Exit Only.
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "AI Macro Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  SUCCESS! Bought {vol} {TICKER} @ {price}")
        else:
            print(f"  ORDER FAILED: {res.comment}")


if __name__ == "__main__":
    while True:
        # Run every hour on the hour (H1 Timeframe execution)
        # Sleep for 60 seconds to avoid spamming the broker
        strategy_tick()
        time.sleep(60 * 60)