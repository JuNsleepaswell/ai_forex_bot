#!/usr/bin/env python3
"""
AUDUSD XGBoost Live Trader — MT5 Demo Account

Features computed on CLOSED H1 bars only (start_pos=1 in copy_rates_from_pos
skips the currently-forming bar).  Every signal is logged to CSV so live
behaviour can be compared against backtest assumptions later.

Hard risk limits
  - Max 1 open AUDUSD position at any time
  - Fixed 0.01 lots per trade
  - Stop trading for the day if daily loss exceeds 2 % of session-opening balance

Usage
  # First run: train and save model, then start live loop
  python src/live_audusd_xgb.py

  # Skip retrain if model already saved
  python src/live_audusd_xgb.py --no_retrain

  # Session filter: only enter trades 13:00-16:00 UTC
  python src/live_audusd_xgb.py --session_filter
"""

import argparse
import csv
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL             = "AUDUSD"
MAGIC              = 20260614
LOTS               = 0.01
MAX_DAILY_LOSS_PCT = 2.0        # halt if (start_bal - current_bal) / start_bal >= this
SESSION_START_UTC  = 13         # only used with --session_filter
SESSION_END_UTC    = 16

LONG_THRESHOLD  = 0.65
SHORT_THRESHOLD = 0.65
FORWARD_BARS    = 4
MIN_HOLD_HOURS  = 12            # minimum bars to hold before allowing exit/reversal

TRAIN_HOURS     = 8_640         # ~12 months of H1 bars for model training
FEATURE_BUFFER  = 300           # bars to pull from MT5 for feature computation

MODEL_PATH = os.path.join("models", "audusd_xgb_live.pkl")
LOG_PATH   = os.path.join("live_logs", "audusd_xgb_signals.csv")

# Columns the XGBClassifier was trained on (must exactly match walk-forward)
FEATURE_COLS = [
    "FracDiff_Close",
    "H1_Norm_Ret_1",
    "H1_Norm_Ret_4",
    "H1_Norm_Ret_12",
    "Vol_Regime",
    "H1_Autocorr",
    "H1_ZScore_50",
    "Hour_Sin",
    "Hour_Cos",
    "Day_Sin",
    "Day_Cos",
    "RSI_Velocity",
    "ATR_Relative",
]

SUPER_CSV = os.path.join("data", "AUDUSD_SUPER_dataset.csv")


# ---------------------------------------------------------------------------
# Feature engineering — must exactly replicate 02_multiframe_feature_engineering.py
# ---------------------------------------------------------------------------

def _frac_diff(series: pd.Series, d: float = 0.4) -> pd.Series:
    """4-tap fractional differentiation (matches apply_frac_diff in training pipeline)."""
    weights = np.array([1.0, -d, d * (d - 1) / 2, -d * (d - 1) * (d - 2) / 6])
    return (
        series.rolling(window=4)
        .apply(lambda x: np.dot(x[::-1], weights), raw=True)
        .fillna(0)
    )


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate process_h1 from 02_multiframe_feature_engineering.py exactly.
    Input df must have columns: Open, High, Low, Close, Volume (standard OHLCV).
    Index must be a DatetimeIndex (UTC).
    Needs >= 200 bars for Vol_Regime (rolling 168) to stabilise.
    Returns a copy with all FEATURE_COLS present.
    """
    df = df.copy()

    # ATR-14
    df["H1_ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Norm returns (diff, not shift — matches process_h1 line 42-43)
    df["H1_Norm_Ret_1"] = df["Close"].diff(1) / (df["H1_ATR"] + 1e-9)
    df["H1_Norm_Ret_4"] = df["Close"].diff(4) / (df["H1_ATR"] + 1e-9)

    # Norm_Ret_12 is Norm_Ret_1 shifted back 12 bars (matches line 47)
    df["H1_Norm_Ret_12"] = df["H1_Norm_Ret_1"].shift(12)

    # Vol_Regime = ATR / ATR.rolling(168) (matches line 52)
    df["Vol_Regime"] = df["H1_ATR"] / (df["H1_ATR"].rolling(168).mean() + 1e-9)

    # Fractional differentiation (d=0.4, 4-tap)
    df["FracDiff_Close"] = _frac_diff(df["Close"], d=0.4)

    # Autocorrelation of pct_change over rolling 10 (matches lines 59-60)
    df["H1_Ret_1"] = df["Close"].pct_change(1)
    df["H1_Autocorr"] = df["H1_Ret_1"].rolling(10).apply(
        lambda x: x.autocorr() if x.std() > 0 else 0.0, raw=False
    )

    # Z-Score 50 (matches lines 82-84)
    roll50_mean = df["Close"].rolling(50).mean()
    roll50_std  = df["Close"].rolling(50).std()
    df["H1_ZScore_50"] = (df["Close"] - roll50_mean) / (roll50_std + 1e-9)

    # RSI velocity (matches lines 74-75)
    rsi = ta.rsi(df["Close"], length=14)
    df["RSI_Velocity"] = rsi.diff(1)

    # ATR_Relative = same formula as Vol_Regime (matches line 79)
    df["ATR_Relative"] = df["H1_ATR"] / (df["H1_ATR"].rolling(168).mean() + 1e-9)

    # Time features (matches lines 89-92)
    df["Hour_Sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["Day_Sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["Day_Cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _build_label(closes: np.ndarray, forward_bars: int) -> np.ndarray:
    """1 if price is higher after forward_bars, 0 if lower, -1 if flat (excluded)."""
    n = len(closes)
    labels = np.full(n, -1, dtype=int)
    for i in range(n - forward_bars):
        ret = closes[i + forward_bars] - closes[i]
        if ret > 0:
            labels[i] = 1
        elif ret < 0:
            labels[i] = 0
    return labels


def train_model(super_csv: str = SUPER_CSV) -> object:
    """Train XGBClassifier on the last TRAIN_HOURS rows of the SUPER dataset."""
    from xgboost import XGBClassifier

    print(f"[TRAIN] Loading {super_csv} ...")
    df = pd.read_csv(super_csv, index_col=0, parse_dates=True)

    # Identify the frac-diff column (SUPER datasets use FracDiff_Close)
    frac_col = "FracDiff_Close" if "FracDiff_Close" in df.columns else "FracDiff_Z"
    if frac_col == "FracDiff_Z":
        df = df.rename(columns={"FracDiff_Z": "FracDiff_Close"})

    # Use only the most recent TRAIN_HOURS bars
    df = df.iloc[-TRAIN_HOURS:].copy()

    # Check all feature columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"SUPER dataset missing columns: {missing}")

    closes = df["Close"].values
    labels = _build_label(closes, FORWARD_BARS)

    valid = labels != -1
    X = df[FEATURE_COLS].values[valid]
    y = labels[valid]

    print(f"[TRAIN] {X.shape[0]} samples  (long={y.sum()}  short={(y==0).sum()})")

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    print("[TRAIN] Done.")
    return clf


def save_model(clf, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[MODEL] Saved → {path}")


def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"[MODEL] Loaded ← {path}")
    return clf


# ---------------------------------------------------------------------------
# Signal logger
# ---------------------------------------------------------------------------

LOG_FIELDS = [
    "timestamp_utc", "signal", "p_long", "p_short",
    "spread_pips", "close_price",
    "FracDiff_Close", "H1_Norm_Ret_1", "H1_Norm_Ret_4", "H1_Norm_Ret_12",
    "Vol_Regime", "H1_Autocorr", "H1_ZScore_50",
    "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos",
    "RSI_Velocity", "ATR_Relative",
]


def _ensure_log(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def log_signal(
    ts: datetime,
    signal: str,
    p_long: float,
    p_short: float,
    spread_pips: float,
    close_price: float,
    feat_row: dict,
    path: str = LOG_PATH,
) -> None:
    _ensure_log(path)
    row = {
        "timestamp_utc": ts.strftime("%Y-%m-%d %H:%M"),
        "signal":        signal,
        "p_long":        round(p_long, 4),
        "p_short":       round(p_short, 4),
        "spread_pips":   round(spread_pips, 2),
        "close_price":   close_price,
    }
    row.update({k: round(feat_row.get(k, float("nan")), 6) for k in FEATURE_COLS})
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

def _mt5_import():
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 package not found. Install with: pip install MetaTrader5"
        )


def connect_mt5(mt5) -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    info = mt5.account_info()
    print(f"[MT5] Connected  account={info.login}  balance={info.balance:.2f}  "
          f"server={info.server}")


def get_h1_bars(mt5, n: int = FEATURE_BUFFER) -> pd.DataFrame:
    """
    Pull n CLOSED H1 bars.  start_pos=1 skips the currently-forming bar.
    Returns a DataFrame with OHLCV and a UTC DatetimeIndex, sorted oldest→newest.
    """
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 1, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_spread_pips(mt5) -> float:
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return float("nan")
    # AUDUSD pip = 0.0001
    return round((tick.ask - tick.bid) / 0.0001, 2)


def get_current_position(mt5):
    """Return the open AUDUSD position managed by this bot, or None."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return None
    for p in positions:
        if p.magic == MAGIC:
            return p
    return None


def bars_held_since_open(position) -> int:
    """
    Derive hold duration from the broker's position open time, not an in-memory
    counter.  Survives reconnects and missed bars: position.time is a Unix
    timestamp (UTC seconds) set by MT5 when the trade was filled.
    """
    opened_at = datetime.fromtimestamp(position.time, tz=timezone.utc)
    elapsed   = datetime.now(timezone.utc) - opened_at
    return int(elapsed.total_seconds() / 3600)


def close_position(mt5, position) -> bool:
    order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

    req = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    SYMBOL,
        "volume":    position.volume,
        "type":      order_type,
        "position":  position.ticket,
        "price":     price,
        "deviation": 10,
        "magic":     MAGIC,
        "comment":   "xgb_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    ok = result.retcode == mt5.TRADE_RETCODE_DONE
    print(f"[CLOSE] ticket={position.ticket}  retcode={result.retcode}  {'OK' if ok else 'FAIL'}")
    return ok


def open_position(mt5, direction: int) -> bool:
    """direction: 1 = long, -1 = short."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("[OPEN] No tick data — skipping")
        return False

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price      = tick.ask if direction == 1 else tick.bid

    req = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    SYMBOL,
        "volume":    LOTS,
        "type":      order_type,
        "price":     price,
        "deviation": 10,
        "magic":     MAGIC,
        "comment":   "xgb_entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    ok = result.retcode == mt5.TRADE_RETCODE_DONE
    print(f"[OPEN] dir={'BUY' if direction==1 else 'SELL'}  price={price}  "
          f"retcode={result.retcode}  {'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Daily risk tracking
# ---------------------------------------------------------------------------

class DailyRiskGuard:
    def __init__(self, mt5):
        self._mt5        = mt5
        self._start_bal  = mt5.account_info().balance
        self._trade_date = datetime.now(timezone.utc).date()
        self._halted     = False
        print(f"[RISK] Session opening balance = {self._start_bal:.2f}")

    def _refresh_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self._trade_date:
            self._start_bal  = self._mt5.account_info().balance
            self._trade_date = today
            self._halted     = False
            print(f"[RISK] New trading day — reset balance = {self._start_bal:.2f}")

    def is_halted(self) -> bool:
        self._refresh_if_new_day()
        if self._halted:
            return True
        bal  = self._mt5.account_info().balance
        loss = (self._start_bal - bal) / self._start_bal * 100.0
        if loss >= MAX_DAILY_LOSS_PCT:
            print(f"[RISK] Daily loss {loss:.2f}% >= {MAX_DAILY_LOSS_PCT}% — HALTED for today")
            self._halted = True
        return self._halted


# ---------------------------------------------------------------------------
# Main trader loop
# ---------------------------------------------------------------------------

def _wait_for_bar_close() -> None:
    """Sleep until 5 s after the top of the next hour (UTC)."""
    now     = datetime.now(timezone.utc)
    seconds = (60 - now.minute) * 60 - now.second + 5
    print(f"[WAIT] Next bar in {seconds // 60}m {seconds % 60}s  "
          f"(target {(now.hour + 1) % 24:02d}:00:05 UTC)")
    time.sleep(max(seconds, 5))


def _predict(clf, feat_row: dict) -> tuple[int, float, float]:
    """Return (signal, p_long, p_short).  signal: 1=long, -1=short, 0=flat."""
    X = np.array([[feat_row[c] for c in FEATURE_COLS]])
    probs   = clf.predict_proba(X)[0]   # [p_down, p_up]
    p_short, p_long = float(probs[0]), float(probs[1])

    if p_long  >= LONG_THRESHOLD:
        return  1, p_long, p_short
    if p_short >= SHORT_THRESHOLD:
        return -1, p_long, p_short
    return 0, p_long, p_short


def run(
    clf,
    session_filter: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Main loop.

    dry_run=True: compute features and log signals but never touch MT5 orders.
    """
    mt5 = _mt5_import()
    connect_mt5(mt5)

    risk_guard = DailyRiskGuard(mt5)
    _ensure_log(LOG_PATH)

    print(f"[LIVE] Starting  session_filter={session_filter}  dry_run={dry_run}")
    print(f"[LIVE] Logging → {LOG_PATH}")

    while True:
        _wait_for_bar_close()

        now_utc = datetime.now(timezone.utc)

        # ---- Risk check ----
        if risk_guard.is_halted():
            print("[LOOP] Daily loss limit hit — waiting for next trading day")
            continue

        # ---- Fetch closed bars ----
        try:
            bars = get_h1_bars(mt5, FEATURE_BUFFER)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            continue

        if len(bars) < 200:
            print(f"[WARN] Only {len(bars)} bars — need >=200 for Vol_Regime; skipping")
            continue

        # ---- Compute features on last closed bar ----
        try:
            feat_df = compute_features(bars)
        except Exception as e:
            print(f"[ERROR] Feature computation: {e}")
            continue

        feat_df = feat_df.dropna(subset=FEATURE_COLS)
        if feat_df.empty:
            print("[WARN] All feature rows NaN — skipping")
            continue

        last_row   = feat_df.iloc[-1]
        feat_dict  = {c: float(last_row[c]) for c in FEATURE_COLS}
        close_price = float(last_row["Close"])
        bar_ts      = last_row.name  # DatetimeIndex entry (UTC-aware)
        spread_pips = get_spread_pips(mt5)

        # ---- Predict ----
        signal, p_long, p_short = _predict(clf, feat_dict)

        signal_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal]
        print(
            f"[{bar_ts.strftime('%Y-%m-%d %H:%M')} UTC]  "
            f"signal={signal_label:<5}  p_long={p_long:.3f}  p_short={p_short:.3f}  "
            f"spread={spread_pips:.1f}pip  close={close_price:.5f}"
        )

        # ---- Session filter: only enter new positions in London/NY overlap ----
        bar_hour = bar_ts.hour  # already UTC-aware from MT5 copy
        in_session = SESSION_START_UTC <= bar_hour < SESSION_END_UTC

        # ---- Log every signal regardless of session / position state ----
        log_signal(
            ts=bar_ts.to_pydatetime(),
            signal=signal_label,
            p_long=p_long,
            p_short=p_short,
            spread_pips=spread_pips,
            close_price=close_price,
            feat_row=feat_dict,
        )

        if dry_run:
            continue

        # ---- Position management ----
        # Exit rule mirrors the walk-forward backtest exactly:
        #   hold-until-opposite-signal, with MIN_HOLD_HOURS enforcing a floor.
        #   (_apply_min_hold in xgb_walkforward.py locks a signal for min_hold bars;
        #    here we enforce the same floor by checking actual elapsed hours from the
        #    broker's position open timestamp — robust to reconnects and missed bars.)
        pos = get_current_position(mt5)

        # Re-derive hold duration from broker state, not an in-memory counter.
        bars_held = bars_held_since_open(pos) if pos is not None else 0

        can_exit  = pos is None or bars_held >= MIN_HOLD_HOURS
        can_enter = (not session_filter) or in_session

        if pos is not None:
            pos_dir = 1 if pos.type == 0 else -1  # MT5: type 0=BUY, 1=SELL
            # Exit on an opposite (non-flat) signal once min-hold is satisfied
            if can_exit and signal != 0 and signal != pos_dir:
                print(f"[EXEC] Close {'BUY' if pos.type==0 else 'SELL'}  "
                      f"held={bars_held}h  min_hold={MIN_HOLD_HOURS}h")
                close_position(mt5, pos)
                pos = None
                time.sleep(1)  # brief pause before re-entering on same bar

        if pos is None and signal != 0 and can_enter:
            print(f"[EXEC] Open {'BUY' if signal==1 else 'SELL'}")
            open_position(mt5, signal)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AUDUSD XGBoost Live Trader (MT5 Demo)")
    parser.add_argument("--no_retrain",     action="store_true",
                        help="Load existing model instead of retraining")
    parser.add_argument("--session_filter", action="store_true",
                        help=f"Only enter new positions {SESSION_START_UTC}:00-{SESSION_END_UTC}:00 UTC")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Compute features and log signals only — no orders sent")
    args = parser.parse_args()

    os.makedirs("models",    exist_ok=True)
    os.makedirs("live_logs", exist_ok=True)

    if args.no_retrain and os.path.exists(MODEL_PATH):
        clf = load_model(MODEL_PATH)
    else:
        clf = train_model(SUPER_CSV)
        save_model(clf, MODEL_PATH)

    run(clf, session_filter=args.session_filter, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
