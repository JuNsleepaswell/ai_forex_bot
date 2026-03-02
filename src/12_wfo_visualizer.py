import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

TICKERS = ['AUDCAD', 'AUDNZD', 'NZDCAD', 'EURGBP', 'EURAUD']

# --- CONFIGURATION FOR FOREX ---
INITIAL_CAPITAL = 100000.0
RISK_PER_TRADE = 0.01
HOLD_BARS = 10  # Max 10 hours. Do not let trades bleed into the next session.
SL_MULTIPLIER = 1.5       # Keep the stop wide enough to survive noise
TP_MULTIPLIER = 1.0       # Cash out faster before the session ends
SPREAD_IMPACT_ATR = 0.10
MIN_PROBABILITY = 0.55  # AI must be >55% confident to execute

print("--- Analyzing WFO Results (Exact-Path Mean Reversion) ---")

all_trades = []

for ticker in TICKERS:
    path = f'data/{ticker}_WFO_predictions.csv'
    if not os.path.exists(path):
        print(f"Skipping {ticker}, WFO data not found.")
        continue

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    # --- 1. GENERATE SIGNALS ---
    df['Signal'] = 0

    # Require 60% confidence from the XGBoost model
    MIN_PROBABILITY = 0.60
    ai_wants_long = df['WFO_Prob_Long'] > MIN_PROBABILITY
    ai_wants_short = df['WFO_Prob_Short'] > MIN_PROBABILITY

    # TRUE Statistical Extremes (2 Standard Deviations)
    market_is_oversold = df['H1_ZScore_50'] < -2.0
    market_is_overbought = df['H1_ZScore_50'] > 2.0

    df.loc[ai_wants_long & market_is_oversold, 'Signal'] = 1
    df.loc[ai_wants_short & market_is_overbought, 'Signal'] = -1

    # --- TIME OF DAY FILTER: The Asian Session Edge ---
    # We ONLY want to trade Mean Reversion when London and NY are CLOSED.
    # Keep signals ONLY between 18:00 (Late NY / Rollover) and 07:00 (Pre-London)
    df['Hour'] = df.index.hour
    df.loc[(df['Hour'] >= 7) & (df['Hour'] <= 18), 'Signal'] = 0

    # Anti-Overlap: Max 1 trade per day
    df['Date'] = df.index.date
    trades = df[df['Signal'] != 0].copy()
    trades = trades.drop_duplicates(subset=['Date'], keep='first')

    if trades.empty: continue

    # --- 2. EXACT CHRONOLOGICAL OUTCOME SIMULATOR ---
    results = []
    close_arr = df['Close'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values

    for idx, row in trades.iterrows():
        pos = df.index.get_loc(idx)
        entry_price = row['Close']
        cost = row['H1_ATR'] * SPREAD_IMPACT_ATR
        sl_dist = row['H1_ATR'] * SL_MULTIPLIER
        tp_dist = row['H1_ATR'] * TP_MULTIPLIER
        sig = row['Signal']

        # Adjust entry for broker spread
        if sig == 1:
            entry_price += cost
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:
            entry_price -= cost
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist

        # Extract the look-ahead window
        end_pos = min(pos + 1 + HOLD_BARS, len(df))
        w_high = high_arr[pos + 1: end_pos]
        w_low = low_arr[pos + 1: end_pos]
        w_close = close_arr[pos + 1: end_pos]
        w_hours = df.index[pos + 1: end_pos].hour  # Get the hours of the future bars

        if len(w_high) == 0:
            results.append(0)
            continue

        # Find exact index where barriers are breached
        if sig == 1:
            tp_hits = np.where(w_high >= tp_price)[0]
            sl_hits = np.where(w_low <= sl_price)[0]
        else:
            tp_hits = np.where(w_low <= tp_price)[0]
            sl_hits = np.where(w_high >= sl_price)[0]

        first_tp = tp_hits[0] if len(tp_hits) > 0 else 999
        first_sl = sl_hits[0] if len(sl_hits) > 0 else 999

        # HARD TIME STOP: Exit immediately if we hit the London/Frankfurt open.
        # The freight trains run from hour 7 to 18. If we hit these hours, kill the trade.
        session_breaks = np.where((w_hours >= 7) & (w_hours <= 18))[0]
        first_break = session_breaks[0] if len(session_breaks) > 0 else 999

        # Determine strict chronological winner
        if first_tp < first_sl and first_tp < first_break:
            results.append(TP_MULTIPLIER / SL_MULTIPLIER)  # Hit TP cleanly (+1R)
        elif first_sl < first_tp and first_sl < first_break:
            results.append(-1.0)  # Hit SL cleanly (-1R)
        elif first_tp == first_sl and first_tp < first_break:
            results.append(-1.0)  # Hit both in same bar, assume loss
        else:
            # TIME EXIT: Session ended or HOLD_BARS ran out. Calculate floating PnL.
            exit_idx = min(first_break, len(w_close) - 1) if first_break != 999 else len(w_close) - 1
            exit_price = w_close[exit_idx]

            if sig == 1:
                r_float = (exit_price - entry_price) / sl_dist
            else:
                r_float = (entry_price - exit_price) / sl_dist

            # Cap the float at -1.0 just in case of severe gap
            r_float = max(r_float, -1.0)
            results.append(r_float)

    trades['Trade_Result_R'] = results
    trades['Ticker'] = ticker
    all_trades.append(trades[['Ticker', 'Trade_Result_R', 'Signal']])

if not all_trades:
    print("No valid Forex trades found.")
    exit()

# --- 3. COMPUTE PORTFOLIO EQUITY ---
master_trades = pd.concat(all_trades).sort_index()
current_equity = INITIAL_CAPITAL
equity_curve, dates = [], []

print("\n--- Simulating Live Compounding ---")
for date, daily_trades in master_trades.groupby(master_trades.index.date):
    if current_equity <= 0:
        equity_curve.append(0)
        dates.append(date)
        continue

    daily_pct_change = daily_trades['Trade_Result_R'].sum() * RISK_PER_TRADE
    current_equity = max(0, current_equity * (1 + daily_pct_change))

    equity_curve.append(current_equity)
    dates.append(date)

equity_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
final_equity = equity_series.iloc[-1]
total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital:   ${final_equity:,.2f}")
print(f"Total Return:    {total_return:.2f}%")

drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax() * 100
max_drawdown = drawdown.min()
print(f"Maximum Drawdown: {max_drawdown:.2f}%")

total_trades = len(master_trades)
win_rate = len(master_trades[master_trades['Trade_Result_R'] > 0]) / total_trades * 100
long_trades = len(master_trades[master_trades['Signal'] == 1])
short_trades = len(master_trades[master_trades['Signal'] == -1])

print(f"Total Trades:    {total_trades} (Longs: {long_trades}, Shorts: {short_trades})")
print(f"Win Rate:        {win_rate:.2f}%")

plt.figure(figsize=(14, 7))
equity_series.plot(color='blue' if final_equity > INITIAL_CAPITAL else 'red', linewidth=1.5)
plt.axhline(INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5)
plt.title("Forex Exact-Path Mean Reversion Engine (1:1 Risk/Reward)")
plt.ylabel("Account Balance ($)")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.show()