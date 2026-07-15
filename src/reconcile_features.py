#!/usr/bin/env python3
"""
src/reconcile_features.py

Verifies that feature values logged by live_audusd_xgb.py match the values
produced by the training pipeline (src/02_multiframe_feature_engineering.py).

Two recomputation sources — the script auto-selects per row:
  1. SUPER dataset CSV  — for timestamps that exist in the training data.
     Direct comparison against the exact values the model trained on.
  2. MT5 (read-only)   — for timestamps beyond the SUPER dataset's range.
     Pulls LOOKBACK fresh H1 bars, recomputes from scratch using the same
     formulas as process_h1, and compares against the logged value.

No orders are sent.  MT5 is used in read-only mode.

Usage:
  python src/reconcile_features.py
  python src/reconcile_features.py --rows 50
  python src/reconcile_features.py --rows 20 --tol 1e-5
  python src/reconcile_features.py --no_mt5          # SUPER dataset only
  python src/reconcile_features.py --super_csv path/to/other.csv
"""

import argparse
import sys
from datetime import timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL      = "AUDUSD"
LOG_PATH    = "live_logs/audusd_xgb_signals.csv"
SUPER_CSV   = "data/AUDUSD_SUPER_dataset.csv"

# Bars pulled from MT5 before each target timestamp.
# Vol_Regime needs rolling(168); ATR needs 14 more -> 182 minimum.
# 250 gives a safe margin so all windows are fully warmed up.
LOOKBACK    = 250

# Relative tolerance for flagging a mismatch: |live - ref| / |ref| > REL_TOL
REL_TOL     = 1e-6

N_ROWS_DEFAULT = 20

FEATURE_COLS = [
    "FracDiff_Close",
    "H1_Norm_Ret_1", "H1_Norm_Ret_4", "H1_Norm_Ret_12",
    "Vol_Regime", "H1_Autocorr", "H1_ZScore_50",
    "Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos",
    "RSI_Velocity", "ATR_Relative",
]


# ---------------------------------------------------------------------------
# Feature pipeline — exact replica of process_h1 in
#   src/02_multiframe_feature_engineering.py
# Any change here must be mirrored in live_audusd_xgb.py::compute_features
# ---------------------------------------------------------------------------

def _frac_diff(series: pd.Series, d: float = 0.4) -> pd.Series:
    """4-tap fractional differentiation.  Matches apply_frac_diff(d=0.4)."""
    weights = np.array([1.0, -d, d * (d - 1) / 2, -d * (d - 1) * (d - 2) / 6])
    return (
        series.rolling(window=4)
        .apply(lambda x: np.dot(x[::-1], weights), raw=True)
        .fillna(0)
    )


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's ATR.  Uses pandas_ta if available (matches training pipeline exactly);
    falls back to a manual Wilder EMA so the script runs without pandas_ta installed
    (--no_mt5 path never calls this at all).
    """
    try:
        import pandas_ta as ta
        return ta.atr(high, low, close, length=length)
    except ImportError:
        prev = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / length, adjust=False, ignore_na=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's RSI.  Same fallback strategy as _atr().
    """
    try:
        import pandas_ta as ta
        return ta.rsi(close, length=length)
    except ImportError:
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, ignore_na=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, ignore_na=False).mean()
        rs       = avg_gain / (avg_loss + 1e-12)
        return 100.0 - 100.0 / (1.0 + rs)


def _recompute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 13 FEATURE_COLS from raw OHLCV.
    df must have Open/High/Low/Close/Volume and a UTC DatetimeIndex.
    Only called on the MT5 path; --no_mt5 skips this entirely.
    """
    df = df.copy()

    df["H1_ATR"]         = _atr(df["High"], df["Low"], df["Close"], length=14)
    df["H1_Norm_Ret_1"]  = df["Close"].diff(1) / (df["H1_ATR"] + 1e-9)
    df["H1_Norm_Ret_4"]  = df["Close"].diff(4) / (df["H1_ATR"] + 1e-9)
    df["H1_Norm_Ret_12"] = df["H1_Norm_Ret_1"].shift(12)   # shift, not diff — matches line 47
    df["Vol_Regime"]     = df["H1_ATR"] / (df["H1_ATR"].rolling(168).mean() + 1e-9)
    df["FracDiff_Close"] = _frac_diff(df["Close"], d=0.4)

    df["_pct1"]       = df["Close"].pct_change(1)
    df["H1_Autocorr"] = df["_pct1"].rolling(10).apply(
        lambda x: x.autocorr() if x.std() > 0 else 0.0, raw=False
    )

    _r50              = df["Close"].rolling(50)
    df["H1_ZScore_50"]= (df["Close"] - _r50.mean()) / (_r50.std() + 1e-9)

    df["RSI_Velocity"] = _rsi(df["Close"], length=14).diff(1)
    df["ATR_Relative"] = df["H1_ATR"] / (df["H1_ATR"].rolling(168).mean() + 1e-9)

    df["Hour_Sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["Day_Sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["Day_Cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return df


# ---------------------------------------------------------------------------
# Data-source helpers
# ---------------------------------------------------------------------------

def _load_super(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None

    # Normalise index to UTC-aware Timestamps
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Ensure the frac-diff column name matches FEATURE_COLS
    if "FracDiff_Z" in df.columns and "FracDiff_Close" not in df.columns:
        df = df.rename(columns={"FracDiff_Z": "FracDiff_Close"})

    return df


def _mt5_module():
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        return None


def _fetch_mt5_bars(mt5, target_ts: pd.Timestamp, lookback: int) -> tuple[pd.DataFrame | None, str]:
    """
    Pull bars from MT5 ending at (and including) target_ts, using
    copy_rates_from_pos — the SAME API call as live_audusd_xgb.py.

    This is deliberate.  copy_rates_range and copy_rates_from_pos use
    different timestamp conventions on many brokers (copy_rates_range
    takes UTC date parameters but copy_rates_from_pos returns bar times
    stamped in server time).  Using the same call as the live trader
    ensures the timestamps we see here match the ones in the live log.

    We fetch enough bars to cover: lookback warmup + bars from target_ts
    to now.  An estimate with a 1.5x calendar-to-trading multiplier plus
    a hard buffer covers weekend gaps without fetching excessively.
    """
    now_utc   = pd.Timestamp.now(tz="UTC")
    elapsed_h = max((now_utc - target_ts).total_seconds() / 3600, 0)
    # 1.5x factor converts calendar hours to trading hours (weekends ≈ 2/7)
    n_fetch   = int(elapsed_h * 1.5) + lookback + 300
    n_fetch   = min(n_fetch, 5000)   # hard cap

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 1, n_fetch)
    if rates is None or len(rates) == 0:
        return None, f"copy_rates_from_pos({n_fetch}) returned empty: {mt5.last_error()}"

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = (
        df.set_index("time")
        .sort_index()
        .rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume",
        })
    )
    return df[["Open", "High", "Low", "Close", "Volume"]], (
        f"copy_rates_from_pos({n_fetch}) -> {len(df)} bars  "
        f"({df.index[0].strftime('%m-%d %H:%M')} to {df.index[-1].strftime('%m-%d %H:%M')})"
    )


# ---------------------------------------------------------------------------
# Per-row comparison
# ---------------------------------------------------------------------------

_ABS_ZERO_FLOOR = 1e-9  # values smaller than this are "effectively zero"

def _diff_entry(live_val: float, ref_val: float, tol: float) -> dict:
    abs_diff = abs(live_val - ref_val)
    denom    = abs(ref_val)
    if denom < _ABS_ZERO_FLOOR:
        # Near-zero reference: sin(pi), cos(3pi/2) etc. are ~1e-16, not truly zero.
        # Use absolute tolerance to avoid a rel_diff blow-up on machine epsilon.
        rel_diff = 0.0
        mismatch = abs_diff > _ABS_ZERO_FLOOR
    else:
        rel_diff = abs_diff / denom
        mismatch = rel_diff > tol
    return {
        "live":      live_val,
        "ref":       ref_val,
        "abs_diff":  abs_diff,
        "rel_diff":  rel_diff,
        "mismatch":  mismatch,
    }


def _compare_timestamp(
    log_row:    pd.Series,
    target_ts:  pd.Timestamp,
    super_df:   pd.DataFrame | None,
    mt5,
    tol:        float,
    lookback:   int,
    verbose:    bool = True,
) -> tuple[list[dict], str]:
    """
    Returns (comparison_rows, source_label).
    source_label is 'SUPER', 'MT5', or 'SKIP:<reason>'.
    """
    ts_str  = target_ts.strftime("%Y-%m-%d %H:%M")
    ref_row = None
    source  = None

    def _dbg(msg):
        if verbose:
            print(f"    [{ts_str}] {msg}")

    # ---- Try SUPER dataset first ----
    if super_df is not None:
        if target_ts in super_df.index:
            missing = [c for c in FEATURE_COLS if c not in super_df.columns]
            if missing:
                _dbg(f"SUPER: found timestamp but missing columns {missing}")
            else:
                ref_row = super_df.loc[target_ts]
                source  = "SUPER"
                _dbg("SUPER: exact match")
        else:
            _dbg(f"SUPER: timestamp not in index (SUPER ends {super_df.index[-1].strftime('%Y-%m-%d')})")
    else:
        _dbg("SUPER: not loaded")

    # ---- Fall back to MT5 recomputation ----
    if ref_row is None:
        if mt5 is None:
            _dbg("MT5: skipped (not connected)")
        else:
            raw_bars, fetch_msg = _fetch_mt5_bars(mt5, target_ts, lookback)
            _dbg(f"MT5 fetch: {fetch_msg}")

            if raw_bars is None or len(raw_bars) < 50:
                n = len(raw_bars) if raw_bars is not None else 0
                _dbg(f"MT5: not enough bars ({n} returned, need >= 50)")
            else:
                feat_df = _recompute(raw_bars)
                feat_df = feat_df.dropna(subset=FEATURE_COLS)
                if feat_df.empty:
                    _dbg("MT5: all feature rows are NaN after dropna")
                else:
                    # Exact match first
                    if target_ts in feat_df.index:
                        ref_row = feat_df.loc[target_ts]
                        _dbg(f"MT5: exact index match at {target_ts}")
                    else:
                        # Broker may stamp bars in server time (UTC+2/+3).
                        # Find the bar whose timestamp is closest to target_ts
                        # within a ±4-hour window.
                        window_lo = target_ts - pd.Timedelta(hours=4)
                        window_hi = target_ts + pd.Timedelta(hours=4)
                        candidates = feat_df[
                            (feat_df.index >= window_lo) & (feat_df.index <= window_hi)
                        ]
                        if not candidates.empty:
                            # TimedeltaIndex has no .abs(); use argmin() for positional access
                            pos     = (candidates.index - target_ts).to_series().abs().values.argmin()
                            ref_row = candidates.iloc[pos]
                            offset  = candidates.index[pos] - target_ts
                            _dbg(f"MT5: nearest bar at {candidates.index[pos]}  "
                                 f"(offset {offset}, broker TZ shift?)")
                        else:
                            _dbg(f"MT5: no bar within +-4h of target.  "
                                 f"Index range: {feat_df.index[0]} to {feat_df.index[-1]}")
                    source = "MT5" if ref_row is not None else None

    if ref_row is None:
        reason = "no SUPER match and MT5 unavailable/empty" if mt5 is None else "all paths failed"
        return [], f"SKIP:{reason}"

    rows = []
    for feat in FEATURE_COLS:
        live_val = float(log_row[feat])
        ref_val  = float(ref_row[feat])
        entry    = _diff_entry(live_val, ref_val, tol)
        rows.append({
            "timestamp": ts_str,
            "feature":   feat,
            "source":    source,
            **entry,
        })
    return rows, source


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_COL_W = max(len(c) for c in FEATURE_COLS) + 2   # width for feature name column


def _print_summary(df: pd.DataFrame, tol: float) -> None:
    print("\n" + "=" * 82)
    print(f"{'FEATURE RECONCILIATION — SUMMARY':^82}")
    print("=" * 82)
    print(f"  Tolerance: |live - ref| / |ref| < {tol:.0e}   "
          f"(ref = SUPER dataset or MT5 recompute)")
    print()
    print(f"  {'Feature':<{_COL_W}} {'Status':>6}  {'Fail/Total':>10}  "
          f"{'Max rel err':>12}  {'Mean rel err':>13}  {'Source':>6}")
    print(f"  {'-' * 78}")

    all_pass = True
    for feat in FEATURE_COLS:
        sub       = df[df["feature"] == feat]
        n_total   = len(sub)
        n_fail    = sub["mismatch"].sum()
        max_rel   = sub["rel_diff"].max()   if n_total else float("nan")
        mean_rel  = sub["rel_diff"].mean()  if n_total else float("nan")
        sources   = "/".join(sorted(sub["source"].unique())) if n_total else "-"
        status    = "FAIL" if n_fail > 0 else "PASS"
        marker    = "  <-- DIVERGES" if n_fail > 0 else ""
        if n_fail > 0:
            all_pass = False
        print(
            f"  {feat:<{_COL_W}} {status:>6}  {n_fail:>4}/{n_total:<5}  "
            f"{max_rel:>12.2e}  {mean_rel:>13.2e}  {sources:>6}{marker}"
        )

    print()
    if all_pass:
        print(f"  PASS  All {len(FEATURE_COLS)} features match within tolerance.")
        print("  Live pipeline is faithful to the training pipeline.")
    else:
        bad = df[df["mismatch"]]["feature"].nunique()
        print(f"  FAIL  {bad} feature(s) diverge.  See detail table below.")
    print()


def _print_detail(df: pd.DataFrame, tol: float) -> None:
    mismatches = df[df["mismatch"]].copy()
    if mismatches.empty:
        return

    print("-" * 82)
    print(f"  MISMATCH DETAIL  (rel diff >= {tol:.0e})")
    print("-" * 82)
    print(
        f"  {'Timestamp':<18} {'Feature':<{_COL_W}} {'Src':>4}  "
        f"{'Live':>13} {'Ref':>13} {'Abs diff':>12} {'Rel diff':>10}"
    )
    print(f"  {'-' * 78}")

    prev_ts = None
    for _, r in mismatches.sort_values(["timestamp", "feature"]).iterrows():
        sep = "" if r["timestamp"] == prev_ts else "\n" if prev_ts else ""
        print(
            f"{sep}  {r['timestamp']:<18} {r['feature']:<{_COL_W}} "
            f"{r['source']:>4}  "
            f"{r['live']:>13.7f} {r['ref']:>13.7f} "
            f"{r['abs_diff']:>12.2e} {r['rel_diff']:>10.2e}"
        )
        prev_ts = r["timestamp"]
    print()


def _print_per_ts_summary(df: pd.DataFrame) -> None:
    """One line per timestamp: how many features matched."""
    print("-" * 82)
    print("  PER-TIMESTAMP PASS/FAIL")
    print("-" * 82)
    for ts, grp in df.groupby("timestamp"):
        n_total  = len(grp)
        n_pass   = (~grp["mismatch"]).sum()
        src      = grp["source"].iloc[0]
        bar      = "ok" if n_pass == n_total else f"FAIL ({n_total - n_pass} mismatches)"
        print(f"  {ts}  [{src:>5}]  {n_pass:>2}/{n_total} features ok  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(log_path: str, n_rows: int, tol: float, lookback: int, use_mt5: bool,
        super_csv: str = SUPER_CSV) -> None:
    # ---- Load live log ----
    try:
        log = pd.read_csv(log_path)
    except FileNotFoundError:
        sys.exit(
            f"Log not found: {log_path}\n"
            "Run live_audusd_xgb.py (even with --dry_run) to generate it first."
        )

    log["timestamp_utc"] = pd.to_datetime(log["timestamp_utc"], utc=True)
    log = log.tail(n_rows).reset_index(drop=True)
    print(f"Loaded {len(log)} rows from {log_path}")

    # ---- Load SUPER dataset (optional, fast path) ----
    super_df = _load_super(super_csv)
    if super_df is not None:
        print(f"SUPER dataset loaded: {len(super_df)} rows  "
              f"({super_df.index[0].date()} to {super_df.index[-1].date()})")
    else:
        print(f"SUPER dataset not found at {SUPER_CSV} — will use MT5 only")

    # ---- Connect MT5 (read-only) ----
    mt5 = None
    if use_mt5:
        mt5_mod = _mt5_module()
        if mt5_mod is None:
            print("MetaTrader5 not installed — MT5 recomputation unavailable")
        elif not mt5_mod.initialize():
            print(f"MT5 initialize() failed: {mt5_mod.last_error()} — skipping MT5 source")
        else:
            mt5 = mt5_mod
            acc = mt5.account_info()
            print(f"MT5 connected (read-only)  account={acc.login}  server={acc.server}")
    else:
        print("MT5 skipped (--no_mt5)")

    if super_df is None and mt5 is None:
        sys.exit("No reference data available.  Provide SUPER dataset or enable MT5.")

    # ---- Compare each logged row ----
    print(f"\nComparing {len(log)} timestamps ...\n")
    all_rows = []

    for _, log_row in log.iterrows():
        target_ts = log_row["timestamp_utc"]
        rows, source = _compare_timestamp(
            log_row, target_ts, super_df, mt5, tol, lookback
        )
        ts_str = target_ts.strftime("%Y-%m-%d %H:%M")
        if source.startswith("SKIP"):
            reason = source.split(":", 1)[1] if ":" in source else "no reference data"
            print(f"  [{ts_str}]  SKIP -- {reason}")
            continue
        all_rows.extend(rows)
        n_fail = sum(r["mismatch"] for r in rows)
        status = "ok" if n_fail == 0 else f"FAIL ({n_fail} features)"
        print(f"  [{ts_str}]  [{source:>5}]  {status}")

    if mt5 is not None:
        mt5.shutdown()

    if not all_rows:
        print("\nNo rows could be compared.")
        return

    result_df = pd.DataFrame(all_rows)
    _print_summary(result_df, tol)
    _print_per_ts_summary(result_df)
    _print_detail(result_df, tol)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reconcile live feature logs vs training pipeline (read-only)"
    )
    p.add_argument("--rows",       type=int,   default=N_ROWS_DEFAULT,
                   help=f"Last N rows of the live log to check (default {N_ROWS_DEFAULT})")
    p.add_argument("--csv",        type=str,   default=LOG_PATH,
                   help="Path to live signal log CSV")
    p.add_argument("--super_csv",  type=str,   default=SUPER_CSV,
                   help="Path to SUPER dataset CSV (reference for in-sample timestamps)")
    p.add_argument("--tol",        type=float, default=REL_TOL,
                   help=f"Relative tolerance for mismatch flag (default {REL_TOL:.0e})")
    p.add_argument("--lookback",   type=int,   default=LOOKBACK,
                   help=f"H1 bars pulled from MT5 before each timestamp (default {LOOKBACK})")
    p.add_argument("--no_mt5",     action="store_true",
                   help="Skip MT5 recomputation; only use SUPER dataset")
    args = p.parse_args()

    run(
        log_path  = args.csv,
        n_rows    = args.rows,
        tol       = args.tol,
        lookback  = args.lookback,
        use_mt5   = not args.no_mt5,
        super_csv = args.super_csv,
    )


if __name__ == "__main__":
    main()
