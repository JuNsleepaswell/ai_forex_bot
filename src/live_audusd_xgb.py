#!/usr/bin/env python3
"""
AUDUSD XGBoost Live Trader — MT5 / Prop-Firm Account

Features computed on CLOSED H1 bars only (start_pos=1 in copy_rates_from_pos
skips the currently-forming bar).  Every signal is logged to CSV so live
behaviour can be compared against backtest assumptions later.

Risk layer (all configurable in the PROP-FIRM RISK LIMITS block below)
  - ATR-based hard stop-loss on every order (no naked positions)
  - Dynamic lot sizing: risk RISK_PCT_PER_TRADE% of equity per trade
  - Daily loss halt: close position and stop if equity drops DAILY_HALT_BUFFER_PCT
    below MAX_DAILY_LOSS_PCT; resets at UTC midnight
  - Total drawdown circuit breaker: permanent halt if equity drops
    TOTAL_DD_BUFFER_PCT below MAX_TOTAL_DRAWDOWN_PCT vs challenge start equity
    (persisted to live_logs/challenge_state.json; delete file to reset)

Usage
  python src/live_audusd_xgb.py                 # train + live loop
  python src/live_audusd_xgb.py --no_retrain    # skip retrain
  python src/live_audusd_xgb.py --session_filter
  python src/live_audusd_xgb.py --dry_run
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL            = "AUDUSD"
MAGIC             = 20260614
SESSION_START_UTC = 13          # only used with --session_filter
SESSION_END_UTC   = 16

# ── PROP-FIRM RISK LIMITS ─────────────────────────────────────────────────────
# Fill in FundingPips' actual numbers; placeholders below are conservative.
# The bot halts BEFORE the hard limit by the buffer so there is headroom for
# a position that is already open to be closed without breaching the rule.
MAX_DAILY_LOSS_PCT      = 4.0   # FundingPips hard daily loss limit (% of day-start equity)
MAX_TOTAL_DRAWDOWN_PCT  = 8.0   # FundingPips hard total drawdown limit (% of challenge start)
DAILY_HALT_BUFFER_PCT   = 0.5   # halt this many % BEFORE the hard daily limit
TOTAL_DD_BUFFER_PCT     = 0.5   # halt this many % BEFORE the hard total limit

# ── POSITION SIZING ───────────────────────────────────────────────────────────
RISK_PCT_PER_TRADE = 0.5        # % of current equity to risk per trade
ATR_SL_MULT        = 1.5        # stop-loss = ATR_SL_MULT × H1 ATR (in price units)
PIP                = 0.0001     # AUDUSD pip size
PIP_VALUE_PER_LOT  = 10.0       # USD per pip per standard lot (AUDUSD, USD account)
MIN_LOTS           = 0.01       # broker minimum
MAX_LOTS           = 1.0        # self-imposed cap per trade
LOT_STEP           = 0.01       # broker lot increment

CHALLENGE_STATE_PATH = os.path.join("live_logs", "challenge_state.json")

# ── DAILY RESET TIME ──────────────────────────────────────────────────────────
# Hour (UTC) at which the prop firm's "new trading day" begins.
# FundingPips (and most MT4/MT5 prop firms) align to the New York close:
#   5 pm ET = 22:00 UTC in winter (EST) / 21:00 UTC in summer (EDT)
# Set to 0 for a straight UTC-midnight reset.
# IMPORTANT: Verify your exact account type at dashboard.fundingpips.com
# Current setting: 22:00 UTC (EST / winter).  Change to 21 when clocks spring forward.
DAILY_RESET_HOUR_UTC = 22

# ── TRAILING vs STATIC DRAWDOWN ───────────────────────────────────────────────
# True  = FundingPips standard: drawdown floor follows your equity HIGH-WATER MARK
#         upward as you profit, so the allowed loss is always measured from peak.
# False = simpler static: floor is pinned at challenge_start_equity forever.
# FundingPips standard challenges use TRAILING.  Verify for your specific plan.
TRAILING_DRAWDOWN = True

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
    print(f"[MODEL] Loaded <- {path}")
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


def connect_mt5(mt5, login: int | None = None,
                server: str | None = None,
                password: str | None = None) -> None:
    """
    Connect to the running MT5 terminal.
    If login/server/password are supplied (e.g. --login 123 --server MetaQuotes-Demo),
    MT5 logs into that account instead of whatever is already open in the terminal.
    Use this to point the bot at a MetaQuotes demo without touching the FundingPips login.
    """
    kwargs = {}
    if login    is not None: kwargs["login"]    = login
    if server   is not None: kwargs["server"]   = server
    if password is not None: kwargs["password"] = password

    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    info = mt5.account_info()
    print(f"[MT5] Connected  account={info.login}  balance={info.balance:.2f}  "
          f"server={info.server}")


def print_broker_constraints(mt5) -> float:
    """
    Print SYMBOL constraints from the broker and return the minimum safe SL
    distance in price units (stops_level + 5 points buffer).

    The buffer prevents retcode 10016 (invalid stops) when the ATR-based SL
    lands inside the broker's forbidden zone around the current price.
    """
    si = mt5.symbol_info(SYMBOL)
    if si is None:
        print(f"[BROKER] WARNING: symbol_info({SYMBOL}) returned None — "
              f"SL clamping disabled")
        return 0.0

    point     = si.point                    # e.g. 0.00001 for 5-digit broker
    stops_lvl = si.trade_stops_level        # minimum distance in points
    # Add 5 points of breathing room above the hard minimum
    min_dist  = (stops_lvl + 5) * point

    print(f"[BROKER] {SYMBOL} constraints:")
    print(f"[BROKER]   digits          = {si.digits}")
    print(f"[BROKER]   point           = {point}")
    print(f"[BROKER]   stops_level     = {stops_lvl} points  "
          f"({stops_lvl * point / PIP:.1f} pips)")
    print(f"[BROKER]   SL clamped to  >= {min_dist / PIP:.1f} pips  "
          f"(stops_level + 5 point buffer)")
    print(f"[BROKER]   volume_min      = {si.volume_min}")
    print(f"[BROKER]   volume_max      = {si.volume_max}")
    print(f"[BROKER]   volume_step     = {si.volume_step}")
    print(f"[BROKER]   spread (now)    = {si.spread} points  "
          f"({si.spread * point / PIP:.1f} pips)")
    return min_dist


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


_filling_logged = False  # log chosen filling mode once per session


def _resolve_filling(mt5) -> int:
    """Return the first ORDER_FILLING constant the symbol actually supports."""
    global _filling_logged
    info = mt5.symbol_info(SYMBOL)
    mask = info.filling_mode if info is not None else 0
    # Bitmask: bit-0 = FOK (1), bit-1 = IOC (2), bit-2 = RETURN (4)
    if mask & 1:
        mode, name = mt5.ORDER_FILLING_FOK, "FOK"
    elif mask & 2:
        mode, name = mt5.ORDER_FILLING_IOC, "IOC"
    else:
        mode, name = mt5.ORDER_FILLING_RETURN, "RETURN"
    if not _filling_logged:
        print(f"[ORDER] filling_mode bitmask={mask}  selected={name} ({mode})")
        _filling_logged = True
    return mode


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
        "type_filling": _resolve_filling(mt5),
    }
    result = mt5.order_send(req)
    ok = result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        print(f"[CLOSE] ticket={position.ticket}  retcode={result.retcode}  OK")
    else:
        print(f"[CLOSE] FAIL ticket={position.ticket}  retcode={result.retcode}  comment='{result.comment}'")
    return ok


def open_position(mt5, direction: int, sl_price: float, lot_size: float) -> bool:
    """
    direction: 1 = long, -1 = short.
    sl_price:  hard stop-loss price sent with the order (ATR-based).
    lot_size:  dynamically sized to risk RISK_PCT_PER_TRADE% of equity.
    """
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("[OPEN] No tick data — skipping")
        return False

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price      = tick.ask if direction == 1 else tick.bid

    req = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    SYMBOL,
        "volume":    lot_size,
        "type":      order_type,
        "price":     price,
        "sl":        round(sl_price, 5),
        "deviation": 10,
        "magic":     MAGIC,
        "comment":   "xgb_entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _resolve_filling(mt5),
    }
    result = mt5.order_send(req)
    ok = result.retcode == mt5.TRADE_RETCODE_DONE
    sl_pips = abs(price - sl_price) / PIP
    if ok:
        print(f"[OPEN] {'BUY' if direction==1 else 'SELL'}  price={price:.5f}  "
              f"sl={sl_price:.5f} ({sl_pips:.1f}pip)  lots={lot_size}  retcode={result.retcode}  OK")
    else:
        print(f"[OPEN] FAIL {'BUY' if direction==1 else 'SELL'}  price={price:.5f}  "
              f"retcode={result.retcode}  comment='{result.comment}'")
    return ok


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

def _load_or_init_challenge_equity(mt5) -> float:
    """
    Load challenge-start equity from disk so it survives restarts.
    On first run the current equity is written as the baseline.
    Delete CHALLENGE_STATE_PATH to reset the circuit breaker.
    """
    os.makedirs(os.path.dirname(CHALLENGE_STATE_PATH), exist_ok=True)
    if os.path.exists(CHALLENGE_STATE_PATH):
        with open(CHALLENGE_STATE_PATH) as f:
            eq = float(json.load(f)["challenge_start_equity"])
        print(f"[RISK] Challenge start equity loaded  = {eq:.2f}  "
              f"(delete {CHALLENGE_STATE_PATH} to reset)")
    else:
        eq = mt5.account_info().equity
        with open(CHALLENGE_STATE_PATH, "w") as f:
            json.dump({"challenge_start_equity": eq}, f, indent=2)
        print(f"[RISK] Challenge start equity recorded = {eq:.2f}  -> {CHALLENGE_STATE_PATH}")
    return eq


def compute_lot_size(equity: float, sl_dist_price: float) -> float:
    """
    Risk RISK_PCT_PER_TRADE% of equity on this trade.

    sl_dist_price  stop-loss distance in price units (e.g. 0.0012 for 12 pips).
    Returns lot size rounded to LOT_STEP and clamped to [MIN_LOTS, MAX_LOTS].
    """
    sl_pips  = max(sl_dist_price / PIP, 1.0)   # floor at 1 pip to avoid division blow-up
    risk_usd = equity * (RISK_PCT_PER_TRADE / 100.0)
    lots     = risk_usd / (sl_pips * PIP_VALUE_PER_LOT)
    lots     = round(lots / LOT_STEP) * LOT_STEP
    lots     = max(MIN_LOTS, min(MAX_LOTS, lots))
    return round(lots, 2)


class RiskGuard:
    """
    Enforces two independent drawdown limits using EQUITY (balance + floating P&L).

    DAILY
      Equity is compared to the day-start equity, where "day" resets at
      DAILY_RESET_HOUR_UTC (default 22:00 UTC = 5pm ET / New York close).
      Halts when loss >= MAX_DAILY_LOSS_PCT - DAILY_HALT_BUFFER_PCT.
      Resets automatically at the next period boundary.

    TOTAL (permanent until manual restart)
      If TRAILING_DRAWDOWN = True (FundingPips standard):
        The reference tracks your peak equity (high-water mark).
        Drawdown = (peak_equity - current_equity) / challenge_start * 100
        The floor rises as you profit, matching FundingPips' trailing rule.
      If TRAILING_DRAWDOWN = False:
        Drawdown is static from challenge_start_equity.
      Halts when >= MAX_TOTAL_DRAWDOWN_PCT - TOTAL_DD_BUFFER_PCT.
    """

    @staticmethod
    def _trading_period(now: datetime) -> object:
        """
        Return an opaque period key for the current trading day.
        Shifting back by DAILY_RESET_HOUR_UTC hours means the boundary
        lands at DAILY_RESET_HOUR_UTC:00 UTC instead of midnight.
        Example: DAILY_RESET_HOUR_UTC=22 → day flips at 22:00 UTC.
        """
        return (now - timedelta(hours=DAILY_RESET_HOUR_UTC)).date()

    def __init__(self, mt5, challenge_start_eq: float):
        self._mt5             = mt5
        self._challenge_start = challenge_start_eq
        now                   = datetime.now(timezone.utc)
        eq                    = mt5.account_info().equity
        self._day_start_eq    = eq
        self._peak_equity     = eq          # high-water mark for trailing DD
        self._period          = self._trading_period(now)
        self._daily_halted    = False
        self._total_halted    = False

        daily_soft = MAX_DAILY_LOSS_PCT    - DAILY_HALT_BUFFER_PCT
        total_soft = MAX_TOTAL_DRAWDOWN_PCT - TOTAL_DD_BUFFER_PCT
        dd_type    = "TRAILING (HWM)" if TRAILING_DRAWDOWN else "STATIC"
        reset_desc = f"{DAILY_RESET_HOUR_UTC:02d}:00 UTC"

        print(f"[RISK] Day-start equity       = {self._day_start_eq:.2f}")
        print(f"[RISK] Challenge-start equity  = {self._challenge_start:.2f}")
        print(f"[RISK] Daily reset at          {reset_desc}  (DAILY_RESET_HOUR_UTC={DAILY_RESET_HOUR_UTC})")
        print(f"[RISK] Daily halt at          -{daily_soft:.1f}%  (hard limit -{MAX_DAILY_LOSS_PCT:.1f}%)")
        print(f"[RISK] Total DD type           {dd_type}")
        print(f"[RISK] Total DD halt at       -{total_soft:.1f}%  (hard limit -{MAX_TOTAL_DRAWDOWN_PCT:.1f}%)")

    def _reset_if_new_period(self) -> None:
        now    = datetime.now(timezone.utc)
        period = self._trading_period(now)
        if period != self._period:
            self._day_start_eq = self._mt5.account_info().equity
            self._period       = period
            self._daily_halted = False
            reset_label = f"{DAILY_RESET_HOUR_UTC:02d}:00 UTC"
            print(f"[RISK] New trading period ({reset_label}) "
                  f"— day_start_eq = {self._day_start_eq:.2f}")

    def check(self, pos, close_fn) -> bool:
        """
        Returns True if trading must be blocked this iteration.
        Closes pos (if open) the first time either limit is breached.
        close_fn: callable(mt5, position) -> bool
        """
        # Permanent total halt — always checked first.
        if self._total_halted:
            return True

        self._reset_if_new_period()

        eq = self._mt5.account_info().equity

        # Update high-water mark BEFORE computing trailing DD.
        self._peak_equity = max(self._peak_equity, eq)

        # ── Total drawdown ────────────────────────────────────────────────────
        total_soft = MAX_TOTAL_DRAWDOWN_PCT - TOTAL_DD_BUFFER_PCT
        if TRAILING_DRAWDOWN:
            # Drawdown from peak; % expressed relative to challenge start balance.
            total_dd  = (self._peak_equity - eq) / self._challenge_start * 100.0
            dd_ref    = f"HWM={self._peak_equity:.2f}"
        else:
            total_dd  = (self._challenge_start - eq) / self._challenge_start * 100.0
            dd_ref    = f"start={self._challenge_start:.2f}"

        if total_dd >= total_soft:
            print(f"[RISK] TOTAL DD {total_dd:.2f}% >= soft limit {total_soft:.1f}% "
                  f"({dd_ref}, hard = {MAX_TOTAL_DRAWDOWN_PCT:.1f}%) — PERMANENT HALT")
            print(f"[RISK] Delete {CHALLENGE_STATE_PATH} and restart to reset.")
            if pos is not None:
                print("[RISK] Closing open position before halting.")
                close_fn(self._mt5, pos)
            self._total_halted = True
            return True

        # ── Daily loss ────────────────────────────────────────────────────────
        if self._daily_halted:
            return True

        daily_soft = MAX_DAILY_LOSS_PCT - DAILY_HALT_BUFFER_PCT
        daily_loss = (self._day_start_eq - eq) / self._day_start_eq * 100.0

        if daily_loss >= daily_soft:
            reset_label = f"{DAILY_RESET_HOUR_UTC:02d}:00 UTC"
            print(f"[RISK] DAILY LOSS {daily_loss:.2f}% >= soft limit {daily_soft:.1f}% "
                  f"(hard = {MAX_DAILY_LOSS_PCT:.1f}%) — halted until {reset_label}")
            if pos is not None:
                print("[RISK] Closing open position before halting.")
                close_fn(self._mt5, pos)
            self._daily_halted = True
            return True

        return False


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
    login: int | None = None,
    server: str | None = None,
    password: str | None = None,
) -> None:
    """
    Main loop.

    dry_run=True: compute features and log signals but never touch MT5 orders.
    login/server/password: override the default MT5 account (useful for pointing
        at a MetaQuotes demo instead of the FundingPips prop account).
    """
    mt5 = _mt5_import()
    connect_mt5(mt5, login=login, server=server, password=password)

    # Print broker constraints; capture minimum SL distance for clamping.
    min_sl_dist = print_broker_constraints(mt5)

    challenge_eq = _load_or_init_challenge_equity(mt5)
    risk_guard   = RiskGuard(mt5, challenge_eq)
    _ensure_log(LOG_PATH)

    print(f"[LIVE] Starting  session_filter={session_filter}  dry_run={dry_run}")
    print(f"[LIVE] Logging -> {LOG_PATH}")

    while True:
        _wait_for_bar_close()

        # ---- Risk check (reads current position so the guard can close it) ----
        # Do this BEFORE fetching new bars so a halt closes the position promptly.
        pos = get_current_position(mt5) if not dry_run else None
        if not dry_run and risk_guard.check(pos, close_position):
            print("[LOOP] Risk halt active — waiting")
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

        last_row    = feat_df.iloc[-1]
        feat_dict   = {c: float(last_row[c]) for c in FEATURE_COLS}
        close_price = float(last_row["Close"])
        atr_value   = float(last_row["H1_ATR"])   # raw ATR in price units for SL sizing
        bar_ts      = last_row.name                # DatetimeIndex entry (UTC-aware)
        spread_pips = get_spread_pips(mt5)

        # ---- Predict ----
        signal, p_long, p_short = _predict(clf, feat_dict)

        signal_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal]
        print(
            f"[{bar_ts.strftime('%Y-%m-%d %H:%M')} UTC]  "
            f"signal={signal_label:<5}  p_long={p_long:.3f}  p_short={p_short:.3f}  "
            f"spread={spread_pips:.1f}pip  close={close_price:.5f}  atr={atr_value/PIP:.1f}pip"
        )

        # ---- Session filter: only enter new positions in London/NY overlap ----
        bar_hour   = bar_ts.hour
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
        pos = get_current_position(mt5)  # re-read: guard may have closed it above

        # Re-derive hold duration from broker state, not an in-memory counter.
        bars_held = bars_held_since_open(pos) if pos is not None else 0

        can_exit  = pos is None or bars_held >= MIN_HOLD_HOURS
        can_enter = (not session_filter) or in_session

        if pos is not None:
            pos_dir = 1 if pos.type == 0 else -1  # MT5: type 0=BUY, 1=SELL
            # Exit when signal differs from current direction (including FLAT=0),
            # matching _apply_min_hold: out[i]=0 on any raw_sig != cur once unlocked.
            if can_exit and signal != pos_dir:
                print(f"[EXEC] Close {'BUY' if pos.type==0 else 'SELL'}  "
                      f"held={bars_held}h  min_hold={MIN_HOLD_HOURS}h")
                close_position(mt5, pos)
                pos = None
                time.sleep(1)  # brief pause before re-entering on same bar

        if pos is None and signal != 0 and can_enter:
            atr_dist = ATR_SL_MULT * atr_value
            # Clamp SL distance to broker minimum (stops_level + buffer) so the
            # order is never rejected with retcode 10016 (invalid stops).
            sl_dist  = max(atr_dist, min_sl_dist)
            if sl_dist > atr_dist:
                print(f"[RISK] SL clamped: ATR gave {atr_dist/PIP:.1f}pip, "
                      f"broker minimum is {min_sl_dist/PIP:.1f}pip")
            sl_price = close_price - sl_dist if signal == 1 else close_price + sl_dist
            equity   = mt5.account_info().equity
            lot_size = compute_lot_size(equity, sl_dist)
            print(f"[RISK] Sizing: equity={equity:.2f}  sl_dist={sl_dist/PIP:.1f}pip  "
                  f"risk={RISK_PCT_PER_TRADE}%  -> lots={lot_size}")
            open_position(mt5, signal, sl_price=sl_price, lot_size=lot_size)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Reconfigure stdout to UTF-8 so non-ASCII in print() doesn't crash on
    # Windows terminals that default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="AUDUSD XGBoost Live Trader (MT5)")
    parser.add_argument("--no_retrain",     action="store_true",
                        help="Load existing model instead of retraining")
    parser.add_argument("--session_filter", action="store_true",
                        help=f"Only enter new positions {SESSION_START_UTC}:00-{SESSION_END_UTC}:00 UTC")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Compute features and log signals only — no orders sent")
    parser.add_argument("--login",    type=int,  default=None,
                        help="MT5 account login (overrides the open terminal's account)")
    parser.add_argument("--server",   type=str,  default=None,
                        help="MT5 server name, e.g. MetaQuotes-Demo")
    parser.add_argument("--password", type=str,  default=None,
                        help="MT5 account password")
    args = parser.parse_args()

    os.makedirs("models",    exist_ok=True)
    os.makedirs("live_logs", exist_ok=True)

    if args.no_retrain and os.path.exists(MODEL_PATH):
        clf = load_model(MODEL_PATH)
    else:
        clf = train_model(SUPER_CSV)
        save_model(clf, MODEL_PATH)

    run(
        clf,
        session_filter=args.session_filter,
        dry_run=args.dry_run,
        login=args.login,
        server=args.server,
        password=args.password,
    )


if __name__ == "__main__":
    main()
