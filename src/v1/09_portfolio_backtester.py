import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

print("--- Step 9: Final Portfolio Backtesting (with Take Profit & Stop Loss) ---")

# --- 1. Define Portfolio and NEW TRADE MANAGEMENT PARAMETERS ---
TICKERS = [
    'AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD',
    'USDCAD', 'USDCHF'
]

COST_CONFIG = {
    'SPREAD_COST_PIPS': 1.0,
    'SWAP_COST_PIPS_PER_NIGHT': 0.1
}

# --- THE FIX IS HERE: Define the correct parameters ---
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 1.5   # This was the missing variable


def get_pip_value(ticker):
    return 0.01 if "JPY" in ticker else 0.0001


# --- 2. Load All Models and Data ---
print("Loading all models and data...")
models, all_data = {}, {}
for ticker in TICKERS:
    try:
        models[ticker] = {
            'keras_model': load_model(f'models/lstm_{ticker}.keras'),
            'xgb_model': joblib.load(f'models/xgboost_{ticker}_model.joblib'),
        }
        features_df = pd.read_csv(f'data/{ticker}_data_features.csv', index_col='Date', parse_dates=True)
        with np.load(f'data/{ticker}_processed_data.npz', allow_pickle=True) as data:
            X_test_keras = data['X_test']
            scaler_target = data['scaler_target'].item()
        test_size = len(X_test_keras)
        test_data = features_df.iloc[-test_size:].copy()
        all_data[ticker] = {'features_df': test_data, 'X_test_keras': X_test_keras, 'scaler_target': scaler_target}
        print(f"Successfully loaded models and data for {ticker}")
    except Exception as e:
        print(f"WARNING: Could not load models/data for {ticker}. It will be excluded. Error: {e}")

TICKERS = list(all_data.keys())
print(f"\nProceeding with backtest for the following assets: {TICKERS}")

# --- 3. Generate Signals and ATR ---
print("\nGenerating signals and calculating ATR...")
signals = {}
for ticker in TICKERS:
    keras_pred_scaled = models[ticker]['keras_model'].predict(all_data[ticker]['X_test_keras'], verbose=0)
    keras_pred = all_data[ticker]['scaler_target'].inverse_transform(keras_pred_scaled).flatten()
    xgb_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_96', 'SMA_480', 'RSI_96',
                    'BB_lower', 'BB_middle', 'BB_upper', 'BB_bandwidth', 'BB_percent',
                    'MACD', 'MACD_hist', 'MACD_signal', 'ATR']
    xgb_pred_proba = models[ticker]['xgb_model'].predict_proba(all_data[ticker]['features_df'][xgb_features])
    signal_df = all_data[ticker]['features_df'][['Close', 'High', 'Low', 'Open']].copy()
    keras_buy_cond = keras_pred > signal_df['Close']
    xgb_buy_cond = xgb_pred_proba[:, 1] > 0.44
    keras_sell_cond = keras_pred < signal_df['Close']
    xgb_sell_cond = xgb_pred_proba[:, 0] > 0.44
    signal_df['Signal'] = np.select([keras_buy_cond & xgb_buy_cond, keras_sell_cond & xgb_sell_cond], [1, -1],
                                    default=0)
    high_low = signal_df['High'] - signal_df['Low']
    high_close = np.abs(signal_df['High'] - signal_df['Close'].shift())
    low_close = np.abs(signal_df['Low'] - signal_df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    signal_df['ATR'] = true_range.rolling(96).mean()
    signals[ticker] = signal_df

# --- 4. Prepare DataFrame for Simulation ---
print("Preparing data for event-driven backtest...")
portfolio_df = pd.DataFrame()
for ticker in TICKERS:
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Signal', 'ATR']
    unique_cols = {col: f'{ticker}_{col}' for col in cols_to_keep}
    signals[ticker] = signals[ticker].rename(columns=unique_cols)
    if portfolio_df.empty:
        portfolio_df = signals[ticker]
    else:
        portfolio_df = portfolio_df.join(signals[ticker], how='outer')

for ticker in TICKERS:
    portfolio_df[f'{ticker}_Signal'] = portfolio_df[f'{ticker}_Signal'].shift(1)

portfolio_df.ffill(inplace=True)
portfolio_df.dropna(inplace=True)

# --- 5. Run the EVENT-DRIVEN Backtest with Take Profit & Stop Loss ---
print("\nRunning event-driven backtest with Take Profit / Stop Loss...")
results_list = []
all_trades = {ticker: [] for ticker in TICKERS}

for ticker in TICKERS:
    pip_value = get_pip_value(ticker)
    spread_cost_abs = COST_CONFIG['SPREAD_COST_PIPS'] * pip_value

    trade_pnls = []
    in_trade = False
    trade_entry_price = 0.0
    trade_direction = 0
    stop_loss_price = 0.0
    take_profit_price = 0.0

    for i in range(len(portfolio_df)):
        row = portfolio_df.iloc[i]
        signal = row[f'{ticker}_Signal']
        atr = row[f'{ticker}_ATR']
        bar_open, bar_high, bar_low, bar_close = row[f'{ticker}_Open'], row[f'{ticker}_High'], row[f'{ticker}_Low'], \
        row[f'{ticker}_Close']

        if in_trade:
            exit_price = None
            if trade_direction == 1:  # Long trade
                if bar_low <= stop_loss_price:
                    exit_price = stop_loss_price
                elif bar_high >= take_profit_price:
                    exit_price = take_profit_price
            elif trade_direction == -1:  # Short trade
                if bar_high >= stop_loss_price:
                    exit_price = stop_loss_price
                elif bar_low <= take_profit_price:
                    exit_price = take_profit_price

            if exit_price:
                pnl = (exit_price - trade_entry_price) * trade_direction - spread_cost_abs
                trade_pnls.append(pnl)
                in_trade = False

        if not in_trade and signal != 0:
            in_trade = True
            trade_direction = signal
            trade_entry_price = bar_open

            if trade_direction == 1:
                stop_loss_price = trade_entry_price - (STOP_LOSS_ATR_MULTIPLIER * atr)
                take_profit_price = trade_entry_price + (TAKE_PROFIT_ATR_MULTIPLIER * atr)
            elif trade_direction == -1:
                stop_loss_price = trade_entry_price + (STOP_LOSS_ATR_MULTIPLIER * atr)
                take_profit_price = trade_entry_price - (TAKE_PROFIT_ATR_MULTIPLIER * atr)

    if in_trade:
        final_pnl = (portfolio_df.iloc[-1][f'{ticker}_Close'] - trade_entry_price) * trade_direction - spread_cost_abs
        trade_pnls.append(final_pnl)

    all_trades[ticker] = trade_pnls

    trade_count = len(all_trades[ticker])
    if trade_count > 0:
        total_pnl_percent = sum(all_trades[ticker]) / portfolio_df[f'{ticker}_Close'].mean()
        win_count = sum(1 for pnl in all_trades[ticker] if pnl > 0)
        win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
        avg_pnl_percent = total_pnl_percent / trade_count if trade_count > 0 else 0
    else:
        total_pnl_percent, win_count, win_rate, avg_pnl_percent = 0, 0, 0, 0

    results_list.append({
        "Asset": ticker, "Total P&L (% Return)": f"{total_pnl_percent:.2%}",
        "Num Trades": trade_count, "Win Count": win_count, "Win Rate (%)": round(win_rate, 2),
        "Avg P&L / Trade (%)": f"{avg_pnl_percent:.4%}"
    })

results_df = pd.DataFrame(results_list).set_index("Asset")

# --- 6. Display All Results ---
print("\n--- Individual Asset Performance (with TP/SL & Costs) ---")
print(results_df)

total_net_return = sum(pd.to_numeric(results_df['Total P&L (% Return)'].str.replace('%', ''))) / 100
avg_net_return = total_net_return / len(TICKERS)
initial_capital = 100000.0
final_value = initial_capital * (1 + avg_net_return)

print("\n--- Estimated NET Portfolio Performance (with TP/SL & Costs) ---")
print(f"Initial Capital:               ${initial_capital:,.2f}")
print(f"Estimated Final Portfolio Value: ${final_value:,.2f}")
print(f"Estimated Net Profit / Loss:     ${final_value - initial_capital:,.2f}")
print(f"Estimated Total Net Return:    {avg_net_return:.2%}")