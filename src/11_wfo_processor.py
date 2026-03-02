import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import argparse
from datetime import timedelta

# --- CONFIGURATION ---
TRAIN_WINDOW_YEARS = 3
TEST_WINDOW_MONTHS = 3
MIN_DATA_REQUIRED = 5000
N_BARS = 24
TP_MULT = 1.5             # CHANGED: 1.5 ATR Target
SL_MULT = 1.5             # CHANGED: 1.5 ATR Stop

# --- FEATURES (Institutional Macro Version) ---
# --- FEATURES (Quant Microstructure Version) ---
SIGNAL_FEATURES =[
    'D1_Norm_Ret', 'H4_ER',
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Autocorr', 'H1_ZScore_50', 'H1_ER',
    'ATR_Ratio', 'DXY_Ret', 'SPY_Ret', 'US10Y_Ret',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
]
RISK_FEATURES = SIGNAL_FEATURES


def apply_triple_barrier(df, tp_mult=3.0, sl_mult=1.5, max_bars=24):
    """
    Generates 3-Class Targets:
    0 = Flat/Chop (Stop Loss Hit or Time Expired)
    1 = Long Win  (Long TP Hit before Long SL)
    2 = Short Win (Short TP Hit before Short SL)
    """
    print(f"  Generating Triple Barrier Targets (TP: {tp_mult}x, SL: {sl_mult}x, Horizon: {max_bars} bars)...")

    close_arr = df['Close'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    atr_arr = df['H1_ATR'].values

    labels = np.zeros(len(df))

    for i in range(len(df) - max_bars):
        entry = close_arr[i]
        atr = atr_arr[i]

        if pd.isna(atr) or atr <= 0:
            continue

        long_tp = entry + (atr * tp_mult)
        long_sl = entry - (atr * sl_mult)
        short_tp = entry - (atr * tp_mult)
        short_sl = entry + (atr * sl_mult)

        window_high = high_arr[i + 1: i + 1 + max_bars]
        window_low = low_arr[i + 1: i + 1 + max_bars]

        # Long Barrier Hits
        l_tp_hits = np.where(window_high >= long_tp)[0]
        l_sl_hits = np.where(window_low <= long_sl)[0]
        first_l_tp = l_tp_hits[0] if len(l_tp_hits) > 0 else max_bars + 1
        first_l_sl = l_sl_hits[0] if len(l_sl_hits) > 0 else max_bars + 1

        # Short Barrier Hits
        s_tp_hits = np.where(window_low <= short_tp)[0]
        s_sl_hits = np.where(window_high >= short_sl)[0]
        first_s_tp = s_tp_hits[0] if len(s_tp_hits) > 0 else max_bars + 1
        first_s_sl = s_sl_hits[0] if len(s_sl_hits) > 0 else max_bars + 1

        # Evaluate Outcomes
        long_win = first_l_tp < first_l_sl and first_l_tp < max_bars
        short_win = first_s_tp < first_s_sl and first_s_tp < max_bars

        if long_win and not short_win:
            labels[i] = 1
        elif short_win and not long_win:
            labels[i] = 2
        elif long_win and short_win:
            # Extreme volatility: Pick the one that hit its target FIRST
            labels[i] = 1 if first_l_tp < first_s_tp else 2
        else:
            labels[i] = 0  # Chop / Loss

    df['Signal_Target'] = labels
    return df.iloc[:-max_bars].copy()


def train_signal_model(X_train, y_train):
    # CHANGED TO MULTI-CLASS CLASSIFIER
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,  # 0 (Flat), 1 (Long), 2 (Short)
        eval_metric='mlogloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_risk_model(X_train, y_train):
    # Risk model remains a Regressor to predict future ATR (Pip distance)
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def main(ticker):
    print(f"\n--- Starting Walk-Forward Optimization for {ticker} (TRIPLE BARRIER CLASSIFIER) ---")

    data_path = f'data/{ticker}_SUPER_dataset.csv'
    if not os.path.exists(data_path): return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    # 1. Prepare Targets
    df = apply_triple_barrier(df, tp_mult=TP_MULT, sl_mult=SL_MULT, max_bars=N_BARS)
    df['Risk_Target'] = df['H1_ATR'].shift(-N_BARS)

    df.dropna(subset=['Signal_Target', 'Risk_Target'] + SIGNAL_FEATURES, inplace=True)
    if len(df) < MIN_DATA_REQUIRED:
        print("  Not enough data after dropna.")
        return

    start_date = df.index[0] + timedelta(days=TRAIN_WINDOW_YEARS * 365)
    end_date = df.index[-1]
    current_date = start_date
    wfo_results = []

    print("  Processing Windows...")
    while current_date < end_date:
        train_start = current_date - timedelta(days=TRAIN_WINDOW_YEARS * 365)
        test_end = current_date + timedelta(days=TEST_WINDOW_MONTHS * 30)

        train_data = df[(df.index >= train_start) & (df.index < current_date)]
        test_data = df[(df.index >= current_date) & (df.index < test_end)]

        if len(test_data) == 0: break
        if len(train_data) < 1000:
            current_date = test_end
            continue

        # Train Models
        sig_model = train_signal_model(train_data[SIGNAL_FEATURES], train_data['Signal_Target'])
        risk_model = train_risk_model(train_data[RISK_FEATURES], train_data['Risk_Target'])

        # Predict Probabilities instead of raw magnitude
        probs = sig_model.predict_proba(test_data[SIGNAL_FEATURES])
        vol_preds = risk_model.predict(test_data[RISK_FEATURES])

        result_chunk = test_data.copy()
        result_chunk['WFO_Prob_Flat'] = probs[:, 0]
        result_chunk['WFO_Prob_Long'] = probs[:, 1]
        result_chunk['WFO_Prob_Short'] = probs[:, 2]
        result_chunk['WFO_Pred_Vol'] = vol_preds
        wfo_results.append(result_chunk)

        current_date = test_end

    if not wfo_results: return
    full_wfo_df = pd.concat(wfo_results)

    output_path = f'data/{ticker}_WFO_predictions.csv'
    full_wfo_df.to_csv(output_path)
    print(f"  WFO Complete. Classification Probabilities saved to {output_path}")

    # Save final live model
    final_train_start = end_date - timedelta(days=TRAIN_WINDOW_YEARS * 365)
    final_train = df[df.index >= final_train_start]
    final_sig_model = train_signal_model(final_train[SIGNAL_FEATURES], final_train['Signal_Target'])
    final_risk_model = train_risk_model(final_train[RISK_FEATURES], final_train['Risk_Target'])

    os.makedirs('models', exist_ok=True)
    joblib.dump(final_sig_model, f'models/signal_{ticker}.joblib')
    joblib.dump(final_risk_model, f'models/risk_{ticker}.joblib')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    args = parser.parse_args()
    main(args.ticker)