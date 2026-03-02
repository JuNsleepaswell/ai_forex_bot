# src/09_multiframe_backtester.py (V6.0 - PRO MODEL COMPATIBLE)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import argparse

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.60  # Only trade if model is >60% sure
SL_MULTIPLIER = 1.5          
TP_MULTIPLIER = 3.0          
RISK_PER_TRADE = 0.02        
INITIAL_CAPITAL = 10000.0

# --- SAFETY CONSTRAINTS ---
MAX_LEVERAGE = 30.0          
MIN_ATR_PCT = 0.0005         
MAX_POSITION_VALUE = 1000000.0 

def run_backtest(ticker):
    print(f"\n--- Backtesting {ticker} ---")
    
    data_path = f'data/{ticker}_SUPER_dataset.csv'
    signal_model_path = f'models/signal_{ticker}.joblib'
    risk_model_path = f'models/risk_{ticker}.joblib'

    if not os.path.exists(data_path): return None, 0

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # --- UPDATED FEATURE LISTS (Matches PRO Training) ---
    # Must match src/old_03_multiframe_signal_training.py features exactly
    signal_features = [
        'D1_EMA_200', 'D1_RSI',
        'H4_EMA_200', 'H4_ATR', 'H4_RSI',
        'H1_EMA_50', 'H1_RSI', 'H1_MACD', 'H1_MACD_Signal',
        'M15_RSI', 'M15_ATR', 'M15_BB_Upper', 'M15_BB_Lower', 'Close',
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
    ]

    # Must match src/old_04_multiframe_risk_training.py features exactly
    risk_features = [
        'D1_EMA_200', 'D1_RSI',
        'H4_EMA_200', 'H4_ATR', 'H4_RSI',
        'H1_EMA_50', 'H1_RSI', 'H1_MACD',
        'M15_RSI', 'M15_ATR', 'Close',
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
    ]
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    if len(test_df) == 0: return None, 0

    # Predictions
    print("  Generating AI predictions...")
    try:
        signal_model = joblib.load(signal_model_path)
        risk_model = joblib.load(risk_model_path)
        
        # Verify columns exist
        missing = [c for c in signal_features if c not in test_df.columns]
        if missing:
            print(f"  ERROR: Missing columns: {missing}")
            return None, 0

        probs = signal_model.predict_proba(test_df[signal_features])
        test_df['Prob_Up'] = probs[:, 1]
        test_df['Pred_Vol'] = risk_model.predict(test_df[risk_features])
    except Exception as e:
        print(f"  Model Error: {e}")
        return None, 0

    # Simulation
    print("  Simulating trades...")
    balance = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    in_position = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    position_size = 0 

    for i in range(len(test_df) - 1):
        date = test_df.index[i]
        current_close = test_df['Close'].iloc[i]
        current_high = test_df['High'].iloc[i] if 'High' in test_df else current_close
        current_low = test_df['Low'].iloc[i] if 'Low' in test_df else current_close
        
        prob_up = test_df['Prob_Up'].iloc[i]
        pred_vol = test_df['Pred_Vol'].iloc[i]

        # --- Exit Logic ---
        if in_position:
            pnl = 0
            exit_reason = ""
            
            if current_low <= stop_loss:
                pnl = (stop_loss - entry_price) * position_size
                exit_reason = "SL"
                in_position = False
            elif current_high >= take_profit:
                pnl = (take_profit - entry_price) * position_size
                exit_reason = "TP"
                in_position = False
            
            if not in_position:
                balance += pnl
                trades.append({
                    'Date': date, 'Type': 'Sell', 'PnL': pnl, 'Reason': exit_reason
                })

        # --- Entry Logic ---
        if not in_position:
            if prob_up > CONFIDENCE_THRESHOLD:
                entry_price = current_close
                
                min_vol = entry_price * MIN_ATR_PCT
                safe_vol = max(abs(pred_vol), min_vol)
                
                sl_dist = safe_vol * SL_MULTIPLIER
                tp_dist = safe_vol * TP_MULTIPLIER
                
                stop_loss = entry_price - sl_dist
                take_profit = entry_price + tp_dist
                
                risk_amt = balance * RISK_PER_TRADE
                if sl_dist > 0:
                    raw_units = risk_amt / sl_dist
                else:
                    raw_units = 0
                
                leverage_cap_units = (balance * MAX_LEVERAGE) / entry_price
                liquidity_cap_units = MAX_POSITION_VALUE / entry_price
                
                position_size = min(raw_units, leverage_cap_units, liquidity_cap_units)

                if position_size > 0:
                    in_position = True
                    trades.append({'Date': date, 'Type': 'Buy', 'Price': entry_price, 'PnL': 0, 'Reason': 'Entry'})

        equity_curve.append(balance)

    # Report
    total_trades = len([t for t in trades if t['Reason'] in ['SL', 'TP']])
    wins = len([t for t in trades if t['PnL'] > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    final_return = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"  Final Balance: ${balance:,.2f}")
    print(f"  Return: {final_return:.2f}%")
    print(f"  Trades: {total_trades} | Win Rate: {win_rate:.2f}%")
    
    return equity_curve, balance

def main():
    tickers = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD']
    
    total_initial = INITIAL_CAPITAL * len(tickers)
    total_final = 0
    all_curves = {}

    for ticker in tickers:
        curve, final_bal = run_backtest(ticker)
        if curve:
            all_curves[ticker] = curve
            total_final += final_bal
        else:
            total_final += INITIAL_CAPITAL
    
    print("\n" + "="*30)
    print("PORTFOLIO SUMMARY (Pro Models)")
    print("="*30)
    print(f"Total Initial Capital: ${total_initial:,.2f}")
    print(f"Total Final Capital:   ${total_final:,.2f}")
    net_profit = total_final - total_initial
    total_return = (net_profit / total_initial) * 100
    print(f"Net Profit:            ${net_profit:,.2f}")
    print(f"Total Return:          {total_return:,.2f}%")
    
    plt.figure(figsize=(12,6))
    for ticker, curve in all_curves.items():
        if curve:
            norm_curve = [(x - INITIAL_CAPITAL)/INITIAL_CAPITAL * 100 for x in curve]
            plt.plot(norm_curve, label=ticker)
    
    plt.title(f"Pro Strategy (GridSearch Tuned + Time Features)")
    plt.xlabel("Time (Bars)")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()