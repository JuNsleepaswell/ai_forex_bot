# dashboard.py (Final Corrected Version 2)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import subprocess
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Forex Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Forex Trading Bot Dashboard")
st.markdown("An end-to-end application for training and backtesting a hybrid XGBoost + LSTM Forex trading model.")


# --- Helper Function to Run Scripts ---
def run_script(script_path):
    try:
        python_executable = sys.executable
        with st.expander(f"Logs for {script_path}", expanded=False):
            process = subprocess.Popen([python_executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
            stdout, stderr = process.communicate()
            if stdout:
                st.text(stdout)
            if stderr:
                st.error(stderr)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args, stderr=stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error running {script_path}:")
        st.error(e.stderr)
        return False


# --- Ensemble Backtesting Function ---
def run_ensemble_backtest():
    st.write("### 📈 Ensemble (XGBoost + LSTM) Backtest Results")

    try:
        lstm_model = load_model('models/lstm_forex_model.keras')
        xgb_model = joblib.load('models/xgboost_forex_model.joblib')

        # *** CORRECTED LOGIC HERE ***
        # All data loading must happen INSIDE the 'with' block
        with np.load('data/lstm_processed_data.npz', allow_pickle=True) as data:
            X_test_lstm = data['X_test']
            scaler_target = data['scaler_target'].item()

    except FileNotFoundError:
        st.error("Could not find trained models or data. Please run the full training pipeline first.")
        return

    # Prepare data and generate predictions
    test_size = len(X_test_lstm)
    data_full = pd.read_csv('data/EURUSD_data_features.csv', index_col=0, parse_dates=True)
    test_data = data_full.iloc[-test_size:].copy()

    predictions_scaled_lstm = lstm_model.predict(X_test_lstm)
    test_data['LSTM_Pred_Price'] = scaler_target.inverse_transform(predictions_scaled_lstm).flatten()

    xgb_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'EMA_20', 'RSI']
    X_test_xgb = test_data[xgb_features]
    test_data['XGB_Up_Probability'] = xgb_model.predict_proba(X_test_xgb)[:, 1]

    # Create Ensemble Signal
    lstm_condition = test_data['LSTM_Pred_Price'] > test_data['Close']
    xgb_condition = test_data['XGB_Up_Probability'] > 0.55
    test_data['Ensemble_Signal'] = np.where(lstm_condition & xgb_condition, 1, 0)

    # Simulation Logic
    initial_balance = 10000
    balance = initial_balance
    position = None
    balance_history = [initial_balance]

    for i in range(len(test_data)):
        if position is not None:
            if test_data['Ensemble_Signal'].iloc[i] == 0:
                balance *= (test_data['Close'].iloc[i] / position)
                position = None
        if position is None:
            if test_data['Ensemble_Signal'].iloc[i] == 1:
                position = test_data['Close'].iloc[i]
        balance_history.append(balance)

    performance_df = pd.DataFrame({'Balance': balance_history[1:]}, index=test_data.index)
    performance_df['Cumulative_Returns'] = performance_df['Balance'] / initial_balance

    # --- Display Metrics and Plot ---
    col1, col2, col3 = st.columns(3)
    final_balance = performance_df['Balance'].iloc[-1]
    total_return = (final_balance / initial_balance) - 1
    buy_and_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0]) - 1

    col1.metric("Final Balance", f"${final_balance:,.2f}")
    col2.metric("Ensemble Strategy Return", f"{total_return:.2%}")
    col3.metric("Buy & Hold Return", f"{buy_and_hold_return:.2%}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(performance_df['Cumulative_Returns'], label='Ensemble AI Strategy')
    buy_and_hold_cum_returns = (1 + test_data['Close'].pct_change()).cumprod()
    ax.plot(buy_and_hold_cum_returns, label='Buy and Hold', linestyle='--')
    ax.set_title('Ensemble (XGBoost + LSTM) Backtest Performance')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# --- Main Application Logic ---
def main():
    st.sidebar.header("Controls")

    if st.sidebar.button("🚀 Run Full Training Pipeline"):
        st.info("This will retrain all models. It may take several minutes.")
        with st.status("Running full pipeline...", expanded=True) as status:
            if run_script('src/v1/01_data_ingestion.py'):
                status.write("✅ Step 1/5: Data ingestion complete.")
            else:
                status.update(label="Pipeline failed!", state="error"); return

            if run_script('src/v1/02_feature_engineering.py'):
                status.write("✅ Step 2/5: Feature engineering complete.")
            else:
                status.update(label="Pipeline failed!", state="error"); return

            if run_script('src/v1/03_xgboost_feature_selection.py'):
                status.write("✅ Step 3/5: XGBoost model trained and saved.")
            else:
                status.update(label="Pipeline failed!", state="error"); return

            if run_script('src/v1/04_lstm_data_preparation.py'):
                status.write("✅ Step 4/5: LSTM data prepared.")
            else:
                status.update(label="Pipeline failed!", state="error"); return

            if run_script('src/v1/05_lstm_model_training.py'):
                status.write("✅ Step 5/5: LSTM model trained and saved.")
            else:
                status.update(label="Pipeline failed!", state="error"); return

            status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

        st.success("All models have been retrained successfully!")
        st.balloons()
        run_ensemble_backtest()

    st.sidebar.markdown("---")
    if st.sidebar.button("📊 Run Ensemble Backtest Only"):
        st.info("Running backtest with existing trained models...")
        run_ensemble_backtest()


if __name__ == '__main__':
    main()