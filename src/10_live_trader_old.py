# src/10_live_trader_old.py (V7.0 PRO - MATCHES WFO MODELS)

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import time
import sys
import math
from datetime import datetime, timedelta

# --- CONFIGURATION ---
TICKERS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD']
BROKER_SUFFIX = "" # CHECK YOUR MARKET WATCH! If symbols are "EURUSD", change to ""

# --- STRATEGY SETTINGS ---
TIMEFRAME = mt5.TIMEFRAME_M15
MAGIC_NUMBER = 999000
DEVIATION = 20

# --- RISK MANAGEMENT ---
RISK_PERCENT = 0.0003       # 1% Risk per trade (Aggressive but backed by data)
MAX_OPEN_TRADES = 5       # Allow up to 3 simultaneous trades (Portfolio diversification)
DAILY_LOSS_LIMIT = 0.04   # Stop if down 4% in a day
MAX_LOTS_PER_TRADE = 10.0 
MIN_LOTS_PER_TRADE = 0.01

# --- AI THRESHOLDS ---
CONFIDENCE_THRESHOLD = 0.65 # Matched to your successful backtest
SL_MULTIPLIER = 1.5         # Matched to backtest
TP_MULTIPLIER = 3.0         # Matched to backtest (2:1 Ratio)
ADX_THRESHOLD = 30          # Chop Filter

# --- ENTRY SETTINGS ---
ORDER_EXPIRATION_HRS = 2

# --- COOLDOWN ---
COOLDOWN_MINUTES = 60

# --- TIME FILTER (GMT+0 / Broker Time) ---
START_HOUR = 18  # London Open
END_HOUR = 8   # NY Close

print(f"--- AI Forex Bot: V7.0 PRO (Live) ---")

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
        print(f"  Loaded models for {ticker}")
    except Exception as e:
        print(f"  ERROR loading {ticker}: {e}")

# --- Feature Lists (MUST MATCH WFO TRAINING) ---
SIGNAL_FEATURES = [
    'D1_EMA_200', 'D1_RSI',
    'H4_EMA_200', 'H4_ATR', 'H4_RSI',
    'H1_EMA_50', 'H1_RSI', 'H1_MACD', 'H1_MACD_Signal',
    'M15_RSI', 'M15_ATR', 'M15_BB_Upper', 'M15_BB_Lower', 'Close',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]

RISK_FEATURES = [
    'D1_EMA_200', 'D1_RSI',
    'H4_EMA_200', 'H4_ATR', 'H4_RSI',
    'H1_EMA_50', 'H1_RSI', 'H1_MACD',
    'M15_RSI', 'M15_ATR', 'Close',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]

# --- Helper Functions ---

def get_broker_symbol(ticker):
    return f"{ticker}{BROKER_SUFFIX}"

def cancel_pending_orders(trade_symbol):
    """Cancels ALL pending orders for a specific symbol."""
    orders = mt5.orders_get(symbol=trade_symbol)
    if orders:
        for order in orders:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "magic": MAGIC_NUMBER,
            }
            res = mt5.order_send(request)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  [CLEANUP] Cancelled Order #{order.ticket}")

def calculate_lot_size(trade_symbol, entry_price, sl_price):
    account = mt5.account_info()
    if not account: return MIN_LOTS_PER_TRADE
    
    balance = account.balance
    risk_amount = balance * RISK_PERCENT
    
    symbol_info = mt5.symbol_info(trade_symbol)
    if not symbol_info: return MIN_LOTS_PER_TRADE
    
    sl_distance = abs(entry_price - sl_price)
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value
    
    if tick_size == 0 or tick_value == 0: return MIN_LOTS_PER_TRADE
    
    sl_ticks = sl_distance / tick_size
    loss_per_lot = sl_ticks * tick_value
    
    if loss_per_lot == 0: return MIN_LOTS_PER_TRADE

    lots = risk_amount / loss_per_lot

    step = symbol_info.volume_step
    if step > 0:
        lots = math.floor(lots / step) * step

    lots = max(lots, symbol_info.volume_min)
    lots = min(lots, symbol_info.volume_max)

    # --- ADD THIS INSTITUTIONAL CAP ---
    # Most retail brokers cap trades at 50 to 100 standard lots
    MAX_RETAIL_LOTS = 50.0
    lots = min(lots, MAX_RETAIL_LOTS)

    return float(f"{lots:.2f}")

def check_daily_drawdown():
    history = mt5.history_deals_get(datetime.now().replace(hour=0, minute=0, second=0), datetime.now())
    if history is None: return False
    daily_profit = sum([deal.profit for deal in history])
    account = mt5.account_info()
    if daily_profit < -(account.balance * DAILY_LOSS_LIMIT):
        print(f"!!! DAILY LOSS LIMIT HIT ({daily_profit:.2f}). STOPPING !!!")
        return True
    return False

def check_cooldown(trade_symbol):
    tick = mt5.symbol_info_tick(trade_symbol)
    if tick is None: return False
    server_now = datetime.fromtimestamp(tick.time)
    from_time = server_now - timedelta(minutes=COOLDOWN_MINUTES)
    deals = mt5.history_deals_get(trade_symbol, from_time, server_now)
    if deals and len(deals) > 0: return True
    return False

def check_correlation_conflict(ticker, signal_type):
    # Simplified: Don't buy USD and Sell USD at same time
    # USD Base: USDCAD, USDCHF, USDJPY
    # USD Quote: EURUSD, GBPUSD, AUDUSD, NZDUSD
    
    is_usd_base = ticker in ['USDCAD', 'USDCHF', 'USDJPY']
    desired_usd_direction = "SHORT" if (is_usd_base and signal_type == "SELL") or (not is_usd_base and signal_type == "BUY") else "LONG"
    
    for other in TICKERS:
        if other == ticker: continue
        sym = get_broker_symbol(other)
        pos = mt5.positions_get(symbol=sym)
        if pos:
            other_is_base = other in ['USDCAD', 'USDCHF', 'USDJPY']
            existing_type = "BUY" if pos[0].type == 0 else "SELL"
            existing_usd_dir = "SHORT" if (other_is_base and existing_type == "SELL") or (not other_is_base and existing_type == "BUY") else "LONG"
            
            if desired_usd_direction != existing_usd_dir:
                print(f"  BLOCKED: Correlation Conflict. Existing {other} trade implies USD {existing_usd_dir}.")
                return True
    return False

def get_live_features(trade_symbol):
    # 1. Download Data (M15, H1, H4, D1)
    # Note: We need enough data for D1 EMA 200
    rates_m15 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_M15, 1, 1000)
    if rates_m15 is None: return None
    df_m15 = pd.DataFrame(rates_m15)
    df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
    df_m15.set_index('time', inplace=True)
    df_m15.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)

    rates_h1 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_H1, 1, 200)
    df_h1 = pd.DataFrame(rates_h1)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
    df_h1.set_index('time', inplace=True)
    df_h1.rename(columns={'close': 'Close'}, inplace=True)

    rates_h4 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_H4, 1, 300)
    df_h4 = pd.DataFrame(rates_h4)
    df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s')
    df_h4.set_index('time', inplace=True)
    df_h4.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}, inplace=True)
    
    rates_d1 = mt5.copy_rates_from_pos(trade_symbol, mt5.TIMEFRAME_D1, 1, 300)
    df_d1 = pd.DataFrame(rates_d1)
    df_d1['time'] = pd.to_datetime(df_d1['time'], unit='s')
    df_d1.set_index('time', inplace=True)
    df_d1.rename(columns={'close': 'Close'}, inplace=True)

    try:
        # Indicators
        d1_ema = ta.ema(df_d1['Close'], length=200)
        d1_rsi = ta.rsi(df_d1['Close'], length=14)

        h4_ema = ta.ema(df_h4['Close'], length=200)
        h4_atr = ta.atr(df_h4['High'], df_h4['Low'], df_h4['Close'], length=14)
        h4_rsi = ta.rsi(df_h4['Close'], length=14)
        
        h1_ema = ta.ema(df_h1['Close'], length=50)
        h1_rsi = ta.rsi(df_h1['Close'], length=14)
        h1_macd = ta.macd(df_h1['Close'])
        
        m15_rsi = ta.rsi(df_m15['Close'], length=14)
        m15_atr = ta.atr(df_m15['High'], df_m15['Low'], df_m15['Close'], length=14)
        m15_bb = ta.bbands(df_m15['Close'], length=20, std=2)
        m15_adx = ta.adx(df_m15['High'], df_m15['Low'], df_m15['Close'], length=14)
        
        # Merge
        last_row = df_m15.iloc[[-1]].copy()
        last_row['D1_EMA_200'] = d1_ema.iloc[-1]
        last_row['D1_RSI'] = d1_rsi.iloc[-1]
        last_row['H4_EMA_200'] = h4_ema.iloc[-1]
        last_row['H4_ATR'] = h4_atr.iloc[-1]
        last_row['H4_RSI'] = h4_rsi.iloc[-1]
        last_row['H1_EMA_50'] = h1_ema.iloc[-1]
        last_row['H1_RSI'] = h1_rsi.iloc[-1]
        last_row['H1_MACD'] = h1_macd['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in h1_macd else np.nan
        last_row['H1_MACD_Signal'] = h1_macd['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in h1_macd else np.nan
        last_row['M15_RSI'] = m15_rsi.iloc[-1]
        last_row['M15_ATR'] = m15_atr.iloc[-1]
        last_row['M15_BB_Upper'] = m15_bb['BBU_20_2.0'].iloc[-1]
        last_row['M15_BB_Lower'] = m15_bb['BBL_20_2.0'].iloc[-1]
        last_row['ADX'] = m15_adx['ADX_14'].iloc[-1]
        
        # Time Features
        current_time = last_row.index[0] 
        last_row['Hour_Sin'] = np.sin(2 * np.pi * current_time.hour / 24)
        last_row['Hour_Cos'] = np.cos(2 * np.pi * current_time.hour / 24)
        last_row['Day_Sin'] = np.sin(2 * np.pi * current_time.dayofweek / 7)
        last_row['Day_Cos'] = np.cos(2 * np.pi * current_time.dayofweek / 7)
        
        last_row['Prev_High'] = df_m15['High'].iloc[-2]
        last_row['Prev_Low'] = df_m15['Low'].iloc[-2]
        
        if last_row.isnull().values.any(): return None
    except Exception: return None
    return last_row

def calculate_dynamic_entry(signal_type, df, atr_value):
    current_close = df['Close'].iloc[-1]
    current_open = df['Open'].iloc[-1]
    prev_low = df['Prev_Low'].iloc[-1] 
    prev_high = df['Prev_High'].iloc[-1]
    rsi = df['M15_RSI'].iloc[-1]
    
    entry_price = current_close
    mode = "Standard"

    if signal_type == "BUY":
        if rsi > 70: 
            entry_price = current_close - (atr_value * 0.1)
            mode = "Aggressive"
        elif rsi > 55: 
            entry_price = (current_open + current_close) / 2
            mode = "Balanced"
        else: 
            entry_price = prev_low + (atr_value * 0.1) 
            mode = "Sniper"

    elif signal_type == "SELL":
        if rsi < 30: 
            entry_price = current_close + (atr_value * 0.1)
            mode = "Aggressive"
        elif rsi < 45: 
            entry_price = (current_open + current_close) / 2
            mode = "Balanced"
        else: 
            entry_price = prev_high - (atr_value * 0.1)
            mode = "Sniper"
            
    return entry_price, mode

def execute_trade(ticker, signal, vol_pred, df):
    trade_symbol = get_broker_symbol(ticker)
    
    # 1. Clean up old pending orders
    cancel_pending_orders(trade_symbol)

    atr_value = df['M15_ATR'].iloc[-1]
    limit_price, mode = calculate_dynamic_entry(signal, df, atr_value)
    
    sl_pips = vol_pred * SL_MULTIPLIER
    tp_pips = vol_pred * TP_MULTIPLIER
    
    symbol_info = mt5.symbol_info(trade_symbol)
    if symbol_info is None: return False
    
    digits = symbol_info.digits
    point = symbol_info.point
    
    # Get Minimum Stop Distance
    min_dist_points = symbol_info.trade_stops_level
    if min_dist_points == 0: min_dist_points = 10 
    min_dist_price = min_dist_points * point

    # --- CRITICAL FIX: FORCE NATIVE FLOAT TYPES ---
    # Numpy types cause order_send to fail silently (return None)
    limit_price = float(limit_price)
    sl_pips = float(sl_pips)
    tp_pips = float(tp_pips)
    # ----------------------------------------------

    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(trade_symbol).ask
        sl = price - sl_pips
        tp = price + tp_pips
    else:  # SELL
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(trade_symbol).bid
        sl = price + sl_pips
        tp = price - tp_pips

        # Rounding
    price = float(round(price, digits))
    sl = float(round(sl, digits))
    tp = float(round(tp, digits))

    volume = calculate_lot_size(trade_symbol, price, sl)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,  # Changed from PENDING
        "symbol": trade_symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"AI {signal}",
        "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or cancel
    }
    
    print(f"  ... Placing [{mode}] Limit Order at {limit_price:.{digits}f} (Vol: {volume}) ...")
    
    # --- DIAGNOSTIC CHECK ---
    # Ask MT5 if this order is valid BEFORE sending it
    check_result = mt5.order_check(request)
    if check_result is None:
         print(f"  FAILED: Connection Issue (order_check returned None)")
         return False
    if check_result.retcode != 0:
         print(f"  PRE-CHECK FAILED: {check_result.comment} (Code: {check_result.retcode})")
         # We try to send anyway in case check is strict, but this log is vital
    # ------------------------

    # Send Order
    for _ in range(3):
        res = mt5.order_send(request)
        if res is not None: break
        time.sleep(0.5)
        
    if res is None:
        print(f"  FAILED: No Response from MT5 (Still None after retries).")
        return False

    if res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  PENDING ORDER PLACED: {signal} {ticker} @ {limit_price:.{digits}f}")
        return True
    
    # If we get here, it failed with a code
    print(f"  FAILED: {res.comment} (Retcode: {res.retcode})")
    return False

def strategy_tick():
    if check_daily_drawdown(): return 

    try:
        server_time = mt5.symbol_info_tick(get_broker_symbol("EURUSD")).time
        print(f"\n--- Scan: {datetime.fromtimestamp(server_time)} ---")
        current_hour = datetime.fromtimestamp(server_time).hour
    except: return

    is_trading_time = False
    if START_HOUR < END_HOUR: is_trading_time = START_HOUR <= current_hour < END_HOUR
    else: is_trading_time = current_hour >= START_HOUR or current_hour < END_HOUR

    if not is_trading_time:
        print("  Market Closed. Monitoring...")
        
    total_positions = mt5.positions_total()
    total_orders = mt5.orders_total()
    
    for ticker in TICKERS:
        trade_symbol = get_broker_symbol(ticker)
        
        # 1. Manage Positions
        positions = mt5.positions_get(symbol=trade_symbol)
        if positions:
            print(f"  {ticker}: Open. P&L: {positions[0].profit:.2f}")
            continue
            
        orders = mt5.orders_get(symbol=trade_symbol)
        if orders:
            print(f"  {ticker}: Pending Order Waiting.")
            continue

        if not is_trading_time: continue
        if check_cooldown(trade_symbol): continue
        if (total_positions + total_orders) >= MAX_OPEN_TRADES: continue

        # 2. Analysis
        data = get_live_features(trade_symbol)
        if data is None: continue
        
        adx_val = data['ADX'].iloc[0]
        if adx_val < ADX_THRESHOLD:
            print(f"  {ticker}: Chop (ADX {adx_val:.1f}). Skipping.")
            continue

        try:
            prob_up = models[ticker]['signal'].predict_proba(data[SIGNAL_FEATURES])[0][1]
            vol_pred = models[ticker]['risk'].predict(data[RISK_FEATURES])[0]
            
            print(f"  {ticker}: Prob={prob_up:.2f} | Vol={vol_pred:.5f}")

            if prob_up > CONFIDENCE_THRESHOLD:
                if not check_correlation_conflict(ticker, "BUY"):
                    print(f"  >> BUY SIGNAL: {ticker}")
                    execute_trade(ticker, "BUY", vol_pred, data)
            elif prob_up < (1.0 - CONFIDENCE_THRESHOLD):
                if not check_correlation_conflict(ticker, "SELL"):
                    print(f"  >> SELL SIGNAL: {ticker}")
                    execute_trade(ticker, "SELL", vol_pred, data)

        except Exception as e:
            print(f"Error {ticker}: {e}")

if __name__ == "__main__":
    while True:
        strategy_tick()
        time.sleep(60)