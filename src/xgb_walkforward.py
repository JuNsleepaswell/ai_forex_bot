#!/usr/bin/env python3
"""
XGBoost Walk-Forward Baseline
-------------------------------
Reports GROSS and NET metrics side-by-side, holding-period decay, and
cost-scenario analysis.  If the strategy loses money, it shows.

Usage:
  python src/xgb_walkforward.py --ticker AUDUSD
  python src/xgb_walkforward.py --ticker AUDUSD --forward_bars 4 --threshold 0.65 --min_hold 12
  python src/xgb_walkforward.py --ticker AUDUSD --forward_bars 4 --threshold 0.65 --min_hold 12 --limit_order
  python src/xgb_walkforward.py --ticker AUDUSD --forward_bars 4 --threshold 0.65 --min_hold 12 --session_filter
  python src/xgb_walkforward.py --tickers EURUSD,GBPUSD,USDJPY,AUDUSD --forward_bars 4 --threshold 0.65 --min_hold 12
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

CANDIDATE_FEATURES = [
    'FracDiff_Z',        # present in CLEAN datasets
    'FracDiff_Close',    # fallback in SUPER datasets
    'H1_Norm_Ret_1', 'H1_Norm_Ret_4', 'H1_Norm_Ret_12',
    'Vol_Regime', 'H1_Autocorr', 'H1_ZScore_50',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'RSI_Velocity', 'ATR_Relative',
]
TIME_FEATURES = frozenset(['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'])

TRAIN_HOURS     = 12 * 30 * 24   # ~8 640 H1 bars  (12 months)
TEST_HOURS      =  1 * 30 * 24   # ~  720 H1 bars  (1 month)
PURGE_BARS      = 24
FORWARD_BARS    = 4
LONG_THRESHOLD  = 0.58
SHORT_THRESHOLD = 0.58
MIN_HOLD        = 0
SPREAD_COST     = 0.00018         # 1.8 pips baseline
N_RANDOM_RUNS   = 30

HOLD_HORIZONS  = [1, 2, 4, 8, 12, 24]   # bars for decay analysis
COST_SCENARIOS = [1.8, 1.0, 0.5]        # pips

# Limit-order simulation parameters
LIMIT_OFFSET_PIPS = 0.5   # place limit 0.5 pips from close (passive fill price improvement)
LIMIT_COST_PIPS   = 0.5   # spread cost when limit-filled (maker, not taker)

# Session filter: London/NY overlap (assumes SUPER dataset timestamps are UTC)
SESSION_UTC_START = 13
SESSION_UTC_END   = 16

ALL_TICKERS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDJPY", "XAUUSD",
    "AUDCAD", "AUDNZD", "EURAUD", "NZDCAD", "EURGBP", "CADCHF",
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def annualized_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    std = r.std()
    return 0.0 if std == 0 else (r.mean() / std) * np.sqrt(24 * 252)


def max_drawdown_pct(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-10)
    return float(dd.min()) * 100


def _pip_size(avg_close: float) -> float:
    return 0.01 if avg_close > 10 else 0.0001


def _resolve_features(df: pd.DataFrame, drop_time: bool = False) -> list:
    cols = set(df.columns)
    result = []
    frac_added = False
    for f in CANDIDATE_FEATURES:
        if drop_time and f in TIME_FEATURES:
            continue
        if f in ('FracDiff_Z', 'FracDiff_Close'):
            if not frac_added and f in cols:
                result.append(f)
                frac_added = True
        elif f in cols:
            result.append(f)
    return result


def _apply_min_hold(raw_sigs: np.ndarray, min_hold: int) -> np.ndarray:
    if min_hold <= 0:
        return raw_sigs.copy()
    n = len(raw_sigs)
    out = np.zeros(n, dtype=float)
    cur = 0.0
    locked = 0
    for i in range(n):
        if locked > 0:
            out[i] = cur
            locked -= 1
        else:
            if raw_sigs[i] != cur:
                locked = min_hold - 1
            cur = raw_sigs[i]
            out[i] = cur
    return out


def _apply_session_filter(
    sigs: np.ndarray,
    timestamps: pd.DatetimeIndex,
    start_h: int = SESSION_UTC_START,
    end_h: int   = SESSION_UTC_END,
) -> np.ndarray:
    """
    Zero out any position entry (sigs[i] != prev and sigs[i] != 0) whose bar
    falls outside [start_h, end_h) UTC.  Existing positions continue unaffected
    so holds are not forcibly cut; only NEW entries are blocked.

    NOTE: assumes timestamps are UTC-aligned (standard for MT5 H1 exported data).
    """
    out  = sigs.copy()
    n    = len(out)
    i    = 0
    while i < n:
        s = out[i]
        if s == 0.0:
            i += 1
            continue
        # Start of a non-zero block
        block_end = i + 1
        while block_end < n and out[block_end] == s:
            block_end += 1

        hour = timestamps[i].hour
        if not (start_h <= hour < end_h):
            out[i:block_end] = 0.0   # block this entire entry + hold period

        i = block_end
    return out


def _apply_limit_fill_filter(
    sigs:   np.ndarray,
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    p_size: float,
    limit_offset_pips: float = LIMIT_OFFSET_PIPS,
) -> tuple:
    """
    Simulate limit-order entries: for every trade block, check whether bar[i+1]'s
    High/Low actually crossed the limit price set at bar[i]'s close.
    Unfilled entries have the entire hold-block zeroed out.

    Limit BUY  @ close[i] - offset  →  fills if Low[i+1]  <= limit_price
    Limit SELL @ close[i] + offset  →  fills if High[i+1] >= limit_price

    Returns (filtered_sigs, fill_rate, n_attempted).
    """
    offset  = limit_offset_pips * p_size
    out     = sigs.copy()
    n       = len(out)
    attempted = 0
    filled    = 0

    i = 0
    while i < n:
        s = out[i]
        if s == 0.0:
            i += 1
            continue

        # Locate end of this block
        block_end = i + 1
        while block_end < n and out[block_end] == s:
            block_end += 1

        check = i + 1   # bar whose H/L we inspect for fill
        attempted += 1

        if check < len(highs):
            if s == 1.0:
                hit = lows[check] <= closes[i] - offset
            else:   # s == -1.0
                hit = highs[check] >= closes[i] + offset

            if hit:
                filled += 1
            else:
                out[i:block_end] = 0.0   # limit not reached — skip entire block
        else:
            out[i:block_end] = 0.0       # no next bar available

        i = block_end

    fill_rate = filled / attempted if attempted > 0 else 0.0
    return out, fill_rate, attempted


def _backtest_window(closes: np.ndarray, sigs: np.ndarray,
                     spread_cost: float = SPREAD_COST):
    """
    Returns (gross_pnls, net_pnls).
    sigs[i]  = position during [close[i], close[i+1]].
    gross[i] = sigs[i] * log(close[i+1] / close[i])
    net[i]   = gross[i] - spread_cost * |sigs[i] - sigs[i-1]|
    """
    log_rets = np.log(closes[1:] / (closes[:-1] + 1e-12))
    n = min(len(log_rets), len(sigs))
    gross = np.empty(n, dtype=float)
    net   = np.empty(n, dtype=float)
    prev  = 0.0
    for i in range(n):
        s        = sigs[i]
        cost     = spread_cost * abs(s - prev)
        gross[i] = s * log_rets[i]
        net[i]   = gross[i] - cost
        prev     = s
    return gross, net


def _trade_stats(sigs: np.ndarray, gross_pnls: np.ndarray, closes: np.ndarray):
    """Returns (n_trades, avg_gross_pips, avg_hold_bars)."""
    avg_close = float(np.nanmean(closes))
    p_size = _pip_size(avg_close)
    trades_gross, trades_bars = [], []
    cur, t_start, t_accum = 0.0, None, 0.0
    for i in range(len(gross_pnls)):
        s = sigs[i]
        if s != cur:
            if cur != 0.0 and t_start is not None:
                trades_gross.append(t_accum)
                trades_bars.append(i - t_start)
            cur = s
            if s != 0.0:
                t_start, t_accum = i, gross_pnls[i]
            else:
                t_start, t_accum = None, 0.0
        elif s != 0.0:
            t_accum += gross_pnls[i]
    if cur != 0.0 and t_start is not None:
        trades_gross.append(t_accum)
        trades_bars.append(len(gross_pnls) - t_start)
    if not trades_gross:
        return 0, 0.0, 0.0
    arr_g = np.array(trades_gross)
    arr_b = np.array(trades_bars)
    return len(trades_gross), float((arr_g * avg_close / p_size).mean()), float(arr_b.mean())


def _random_equity_paths(closes: np.ndarray, sigs: np.ndarray,
                         n_runs: int = N_RANDOM_RUNS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_runs):
        _, net = _backtest_window(closes, rng.permutation(sigs))
        paths.append(net)
    return np.array(paths)


# ---------------------------------------------------------------------------
# Holding-period decay
# ---------------------------------------------------------------------------

def _holding_period_decay(
    all_sigs_list:   list,
    all_closes_list: list,
    horizons:        list = HOLD_HORIZONS,
) -> dict:
    """
    At every trade entry (sigs changes to a non-zero value), compute the
    cumulative gross pips at each horizon within the same test window.
    Works window-by-window to prevent cross-window contamination.
    Returns {h: np.ndarray([pip_obs, ...])} for each horizon.
    """
    data = defaultdict(list)
    for sigs, closes in zip(all_sigs_list, all_closes_list):
        avg_close = float(np.nanmean(closes))
        p_size    = _pip_size(avg_close)
        prev = 0.0
        for i in range(len(sigs)):
            s = sigs[i]
            if s != 0.0 and s != prev:          # new trade entry
                for h in horizons:
                    j = i + h
                    if j < len(closes):
                        raw_ret = np.log(closes[j] / (closes[i] + 1e-12))
                        data[h].append(s * raw_ret * avg_close / p_size)
            prev = s
    return {h: np.array(v) for h, v in data.items()}


def _print_decay_table(decay: dict, ticker: str) -> None:
    print(f"\n  Holding-Period Decay Analysis -- {ticker}")
    print(f"  (avg gross pips measured from each trade entry; breakeven = +1.8 pips)")
    print(f"  {'Horizon':>8}  {'Avg Pips':>9}  {'Std':>8}  {'t-stat':>7}  {'N':>6}  Sig")
    print(f"  {'-'*55}")
    for h in sorted(decay.keys()):
        arr = decay[h]
        if len(arr) < 5:
            continue
        mu  = arr.mean()
        std = arr.std()
        n   = len(arr)
        t   = mu / (std / np.sqrt(n) + 1e-10)
        sig = '*' if abs(t) > 1.96 else ('~' if abs(t) > 1.3 else '')
        print(f"  {h:>6}h  {mu:>+9.2f}  {std:>8.2f}  {t:>+7.2f}  {n:>6}  {sig}")


# ---------------------------------------------------------------------------
# Cost-scenario rescaling
# ---------------------------------------------------------------------------

def _cost_scenario_analysis(agg_gross: np.ndarray, agg_net: np.ndarray) -> list:
    """
    The cost vector is (agg_gross - agg_net).  For a new spread X pips, scale
    that vector by (X / 1.8) and recompute net metrics.
    """
    agg_costs = agg_gross - agg_net   # per-bar cost at 1.8 pips baseline
    results = []
    for pips in COST_SCENARIOS:
        net_s = agg_gross - agg_costs * (pips / 1.8)
        eq_s  = np.exp(np.cumsum(net_s)) * 100_000
        results.append({
            'cost_pips'  : pips,
            'net_ret_pct': float(np.expm1(net_s.sum())) * 100,
            'net_sharpe' : annualized_sharpe(net_s),
            'net_dd_pct' : max_drawdown_pct(eq_s),
        })
    return results


def _print_cost_table(scenarios: list) -> None:
    print(f"\n  Cost-Scenario Analysis  (same signals, different spreads):")
    print(f"  {'Spread':>8}  {'Net Return %':>13}  {'Net Sharpe':>11}  {'Max DD %':>9}")
    print(f"  {'-'*48}")
    for s in scenarios:
        tag = '  <- actual' if s['cost_pips'] == 1.8 else ''
        print(f"  {s['cost_pips']:>7.1f}p  {s['net_ret_pct']:>+13.2f}  "
              f"{s['net_sharpe']:>11.3f}  {s['net_dd_pct']:>9.2f}{tag}")


# ---------------------------------------------------------------------------
# Per-ticker walk-forward
# ---------------------------------------------------------------------------

def run_ticker(
    ticker: str,
    *,
    forward_bars:       int   = FORWARD_BARS,
    long_threshold:     float = LONG_THRESHOLD,
    short_threshold:    float = SHORT_THRESHOLD,
    min_hold:           int   = MIN_HOLD,
    drop_time_features: bool  = False,
    limit_order:        bool  = False,
    session_filter:     bool  = False,
) -> dict | None:

    path = f'data/{ticker}_SUPER_dataset.csv'
    if not os.path.exists(path):
        print(f"  [Skip] {path} not found.")
        return None

    df = pd.read_csv(path, index_col='time', parse_dates=True)
    features = _resolve_features(df, drop_time=drop_time_features)
    if len(features) < 4:
        print(f"  [Skip] Only {len(features)} features for {ticker}.")
        return None

    df['_fwd_ret'] = df['Close'].pct_change(forward_bars).shift(-forward_bars)
    df = df.dropna(subset=features + ['_fwd_ret', 'Close']).copy()
    df['_label'] = (df['_fwd_ret'] > 0).astype(int)

    # Verify High/Low exist for limit-order simulation
    has_hl = 'High' in df.columns and 'Low' in df.columns
    if limit_order and not has_hl:
        print(f"  [Warn] {ticker} SUPER dataset missing High/Low — limit_order disabled.")
        limit_order = False

    n = len(df)
    if n < TRAIN_HOURS + PURGE_BARS + TEST_HOURS:
        print(f"  [Skip] {ticker}: {n} bars, need {TRAIN_HOURS + PURGE_BARS + TEST_HOURS}.")
        return None

    flags = []
    if session_filter:
        flags.append(f"session={SESSION_UTC_START}-{SESSION_UTC_END}UTC")
    if limit_order:
        flags.append(f"limit={LIMIT_OFFSET_PIPS}pip")
    cfg_str = (f"fwd={forward_bars}  thr={long_threshold:.2f}  hold={min_hold}"
               + (f"  [{', '.join(flags)}]" if flags else ""))
    n_exp = (n - TRAIN_HOURS - PURGE_BARS) // TEST_HOURS
    print(f"\n[{ticker}] {n} bars | {len(features)} features | ~{n_exp} windows | {cfg_str}")

    # Per-window accumulators (primary mode)
    window_stats    = []
    all_gross_pnls  = []
    all_net_pnls    = []
    all_sigs_list   = []
    all_closes_list = []
    last_model      = None

    # Limit-order comparison accumulators (populated when limit_order=True)
    lim_all_gross   = []
    lim_all_net     = []
    lim_fill_rates  = []

    pos = 0
    rng_seed = 0
    while pos + TRAIN_HOURS + PURGE_BARS + TEST_HOURS <= n:
        t_end    = pos + TRAIN_HOURS
        te_start = t_end + PURGE_BARS
        te_end   = te_start + TEST_HOURS

        X_tr      = df.iloc[pos:t_end][features].values
        y_tr      = df.iloc[pos:t_end]['_label'].values
        X_te      = df.iloc[te_start:te_end][features].values
        closes_te = df.iloc[te_start:te_end]['Close'].values

        if len(np.unique(y_tr)) < 2:
            pos += TEST_HOURS
            continue

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', verbosity=0, random_state=42,
        )
        model.fit(X_tr, y_tr)
        last_model = model

        proba = model.predict_proba(X_te)
        raw_sigs = np.zeros(len(X_te), dtype=float)
        raw_sigs[proba[:, 1] > long_threshold]  =  1.0
        raw_sigs[proba[:, 0] > short_threshold] = -1.0
        sigs = _apply_min_hold(raw_sigs, min_hold)

        # Session filter: block entries outside 13-16 UTC
        if session_filter:
            ts_te = df.index[te_start:te_end]
            sigs  = _apply_session_filter(sigs, ts_te)

        # Primary backtest: market orders at SPREAD_COST
        gross, net = _backtest_window(closes_te, sigs, SPREAD_COST)
        n_tr, avg_pips, avg_hold = _trade_stats(sigs, gross, closes_te)

        w_gross_ret = float(np.expm1(gross.sum())) * 100
        w_net_ret   = float(np.expm1(net.sum()))   * 100

        window_stats.append({
            'window_start'  : df.index[te_start],
            'gross_ret_pct' : w_gross_ret,
            'net_ret_pct'   : w_net_ret,
            'edge_pct'      : w_gross_ret - w_net_ret,
            'net_sharpe'    : annualized_sharpe(net),
            'max_dd_pct'    : max_drawdown_pct(np.exp(np.cumsum(net)) * 100_000),
            'n_trades'      : n_tr,
            'avg_gross_pips': avg_pips,
            'avg_hold_bars' : avg_hold,
        })

        all_gross_pnls.append(gross)
        all_net_pnls.append(net)
        all_sigs_list.append(sigs[:len(gross)])
        all_closes_list.append(closes_te[:len(gross) + 1])

        # Limit-order simulation: fill only if bar[i+1] crosses the limit
        if limit_order:
            avg_close_te = float(np.nanmean(closes_te))
            p_size_te    = _pip_size(avg_close_te)
            highs_te     = df.iloc[te_start:te_end]['High'].values
            lows_te      = df.iloc[te_start:te_end]['Low'].values
            lim_sigs, fr, _ = _apply_limit_fill_filter(
                sigs, closes_te, highs_te, lows_te, p_size_te
            )
            lim_cost = LIMIT_COST_PIPS * p_size_te
            lg, ln   = _backtest_window(closes_te, lim_sigs, lim_cost)
            lim_all_gross.append(lg)
            lim_all_net.append(ln)
            lim_fill_rates.append(fr)

        pos      += TEST_HOURS
        rng_seed += 1

    if not window_stats:
        print(f"  [Skip] No completed windows for {ticker}.")
        return None

    # ----- Aggregate -----
    agg_gross = np.concatenate(all_gross_pnls)
    agg_net   = np.concatenate(all_net_pnls)

    gross_equity = np.exp(np.cumsum(agg_gross)) * 100_000
    net_equity   = np.exp(np.cumsum(agg_net))   * 100_000

    total_gross_ret  = float(np.expm1(agg_gross.sum())) * 100
    total_net_ret    = float(np.expm1(agg_net.sum()))   * 100
    total_cost_paid  = total_gross_ret - total_net_ret
    agg_gross_sharpe = annualized_sharpe(agg_gross)
    agg_net_sharpe   = annualized_sharpe(agg_net)
    agg_net_dd       = max_drawdown_pct(net_equity)
    agg_gross_dd     = max_drawdown_pct(gross_equity)

    total_n_trades = sum(w['n_trades'] for w in window_stats)
    agg_avg_pips   = float(np.mean([w['avg_gross_pips'] for w in window_stats if w['n_trades'] > 0] or [0.0]))
    agg_avg_hold   = float(np.mean([w['avg_hold_bars']  for w in window_stats if w['n_trades'] > 0] or [0.0]))
    win_rate       = float((agg_net > 0).mean()) * 100

    fi = dict(zip(features, last_model.feature_importances_)) if last_model else {}

    # ----- BAH benchmark -----
    bah_lr = np.concatenate([np.log(c[1:] / (c[:-1] + 1e-12)) for c in all_closes_list])
    bah_eq = np.exp(np.cumsum(bah_lr)) * 100_000
    bah_ret = float(np.expm1(bah_lr.sum())) * 100

    # ----- Random-signal benchmark -----
    rng = np.random.default_rng(42)
    rand_net_paths = [
        _random_equity_paths(c, s, n_runs=N_RANDOM_RUNS, seed=int(rng.integers(9999)))
        for s, c in zip(all_sigs_list, all_closes_list)
    ]
    rand_cumlog = np.zeros((N_RANDOM_RUNS, len(agg_net)), dtype=float)
    idx = 0
    for paths_w in rand_net_paths:
        T = paths_w.shape[1]
        rand_cumlog[:, idx:idx+T] = paths_w
        idx += T
    rand_equity = np.exp(np.cumsum(rand_cumlog, axis=1)) * 100_000
    rand_med = np.median(rand_equity, axis=0)
    rand_lo  = np.percentile(rand_equity, 10, axis=0)
    rand_hi  = np.percentile(rand_equity, 90, axis=0)

    # ----- Holding-period decay -----
    decay = _holding_period_decay(all_sigs_list, all_closes_list, HOLD_HORIZONS)

    # ----- Cost scenarios -----
    scenarios = _cost_scenario_analysis(agg_gross, agg_net)

    # ----- Limit-order aggregate -----
    lim_net_sharpe = lim_net_ret = lim_gross_ret = lim_net_dd = 0.0
    lim_n_trades   = 0
    lim_net_equity = None
    avg_fill_rate  = 0.0
    if limit_order and lim_all_gross:
        agg_lim_gross  = np.concatenate(lim_all_gross)
        agg_lim_net    = np.concatenate(lim_all_net)
        lim_gross_eq   = np.exp(np.cumsum(agg_lim_gross)) * 100_000
        lim_net_equity = np.exp(np.cumsum(agg_lim_net))   * 100_000
        lim_gross_ret  = float(np.expm1(agg_lim_gross.sum())) * 100
        lim_net_ret    = float(np.expm1(agg_lim_net.sum()))   * 100
        lim_net_sharpe = annualized_sharpe(agg_lim_net)
        lim_net_dd     = max_drawdown_pct(lim_net_equity)
        avg_fill_rate  = float(np.mean(lim_fill_rates)) * 100
        # Count trades from limit sigs
        for lg, ln, cl_list in zip(lim_all_gross, lim_all_net, all_closes_list):
            # Derive limit sigs indirectly via trade_stats on limit net
            pass   # use lim_n_trades from direct count below
        lim_n_trades = int(round(
            sum(
                len(np.where(np.diff(np.concatenate([[0], lg != 0 ])))[0])
                for lg in lim_all_gross
            ) // 2
        ))

    # ----- Print primary results -----
    print(f"\n{'='*65}")
    print(f"AGGREGATE RESULTS: {ticker}  ({len(window_stats)} windows)")
    print(f"{'='*65}")
    print(f"  {'Metric':<28} {'GROSS':>10}  {'NET':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Total Return %':<28} {total_gross_ret:>+10.2f}  {total_net_ret:>+10.2f}")
    print(f"  {'Ann. Sharpe':<28} {agg_gross_sharpe:>10.3f}  {agg_net_sharpe:>10.3f}")
    print(f"  {'Max Drawdown %':<28} {agg_gross_dd:>10.2f}  {agg_net_dd:>10.2f}")
    print(f"  {'-'*50}")
    print(f"  {'Cost paid (gross-net) %':<28} {total_cost_paid:>+10.2f}")
    print(f"  {'Buy-and-hold Return %':<28} {bah_ret:>+10.2f}")
    print(f"  {'-'*50}")
    print(f"  {'Completed Trades':<28} {total_n_trades:>10}")
    print(f"  {'Avg Gross Pips/Trade':<28} {agg_avg_pips:>10.2f}  (need > 1.8 to break even)")
    print(f"  {'Avg Hold (H1 bars)':<28} {agg_avg_hold:>10.1f}")
    print(f"  {'Win Rate (bar-level)':<28} {win_rate:>9.1f}%")
    if session_filter:
        active_bars = int((np.concatenate(all_sigs_list) != 0).sum())
        total_bars  = len(agg_gross)
        print(f"  {'Session exposure %':<28} {100*active_bars/total_bars:>9.1f}%  (only 13-16 UTC)")
    print(f"  {'-'*50}")
    print(f"  Feature Importance -- last window (top 5):")
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:5]:
        print(f"    {feat:30s}: {imp:.4f}")

    # ----- Print limit-order comparison -----
    if limit_order:
        print(f"\n  Limit-Order Comparison  "
              f"(offset={LIMIT_OFFSET_PIPS}pip  cost={LIMIT_COST_PIPS}pip):")
        print(f"  {'Mode':<24} {'Fill%':>6} {'Gross%':>8} {'Net%':>8} "
              f"{'Net Sharpe':>11} {'Net DD%':>8}")
        print(f"  {'-'*72}")
        print(f"  {'Market (1.8pip)':<24} {'100.0':>6} "
              f"{total_gross_ret:>+8.2f} {total_net_ret:>+8.2f} "
              f"{agg_net_sharpe:>11.3f} {agg_net_dd:>8.2f}")
        print(f"  {f'Limit ({LIMIT_COST_PIPS}pip)':<24} {avg_fill_rate:>5.1f}% "
              f"{lim_gross_ret:>+8.2f} {lim_net_ret:>+8.2f} "
              f"{lim_net_sharpe:>11.3f} {lim_net_dd:>8.2f}")
        if lim_net_sharpe > agg_net_sharpe:
            delta = lim_net_sharpe - agg_net_sharpe
            print(f"  --> Limit-order improves net Sharpe by {delta:+.3f}")
        else:
            delta = lim_net_sharpe - agg_net_sharpe
            print(f"  --> Limit-order changes net Sharpe by {delta:+.3f} "
                  f"(fill rate {avg_fill_rate:.1f}% filters out trades)")

    _print_decay_table(decay, ticker)
    _print_cost_table(scenarios)

    # ----- Save -----
    os.makedirs('results', exist_ok=True)
    windows_df = pd.DataFrame(window_stats)
    suffix = f"_fwd{forward_bars}_thr{int(long_threshold*100)}_hold{min_hold}"
    if drop_time_features:
        suffix += "_notime"
    if session_filter:
        suffix += "_session"
    if limit_order:
        suffix += "_limit"
    windows_df.to_csv(f'results/{ticker}_xgb_wfo_windows{suffix}.csv', index=False)

    # ---- Figure 1: equity curves + gross window bars ----
    fig, axes = plt.subplots(2, 1, figsize=(15, 11),
                             gridspec_kw={'height_ratios': [3, 1.5]})

    T = len(net_equity)
    x = np.arange(T)

    ax = axes[0]
    ax.fill_between(x, rand_lo, rand_hi, color='gray', alpha=0.12,
                    label='Random 10-90th pct')
    ax.plot(rand_med,     color='gray',       lw=0.9, ls='--',
            label=f'Random median  (n={N_RANDOM_RUNS})')
    ax.plot(bah_eq,       color='darkorange', lw=1.0, ls='-',
            label=f'Buy & hold  ({bah_ret:+.1f}%)')
    ax.plot(gross_equity, color='limegreen',  lw=1.0, ls='--',
            label=f'Strategy GROSS  ({total_gross_ret:+.1f}%  Sharpe {agg_gross_sharpe:.2f})')
    ax.plot(net_equity,   color='royalblue',  lw=1.3, ls='-',
            label=f'Strategy NET  ({total_net_ret:+.1f}%  Sharpe {agg_net_sharpe:.2f})')
    if lim_net_equity is not None:
        ax.plot(lim_net_equity, color='mediumorchid', lw=1.2, ls='-.',
                label=f'Limit-order NET  ({lim_net_ret:+.1f}%  Sharpe {lim_net_sharpe:.2f}'
                       f'  fill={avg_fill_rate:.0f}%)')
    ax.axhline(100_000, color='black', lw=0.5, ls=':', alpha=0.4)

    mode_notes = []
    if session_filter:
        mode_notes.append(f'session={SESSION_UTC_START}-{SESSION_UTC_END}UTC')
    if limit_order:
        mode_notes.append(f'limit={LIMIT_OFFSET_PIPS}pip  cost={LIMIT_COST_PIPS}pip')
    ax.set_title(
        f'XGBoost Walk-Forward: {ticker}  |  fwd={forward_bars}  thr={long_threshold:.2f}  '
        f'hold>={min_hold}' + (f'  [{", ".join(mode_notes)}]' if mode_notes else '') + '\n'
        f'Net MaxDD: {agg_net_dd:.1f}%   Trades: {total_n_trades}   '
        f'Avg gross pips/trade: {agg_avg_pips:.1f}   Avg hold: {agg_avg_hold:.0f}h'
    )
    ax.set_ylabel('Equity ($)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    gcolors = ['#2ecc71' if v > 0 else '#e74c3c' for v in windows_df['gross_ret_pct']]
    ax.bar(np.arange(len(windows_df)), windows_df['gross_ret_pct'],
           color=gcolors, width=0.85, label='Gross return %')
    ax.axhline(0, color='black', lw=0.5)
    running_mean = windows_df['gross_ret_pct'].expanding().mean()
    ax.plot(np.arange(len(windows_df)), running_mean,
            color='navy', lw=1.2, ls='--', label='Expanding mean')
    n_pos = (windows_df['gross_ret_pct'] > 0).sum()
    ax.set_title(
        f'Gross Return per Walk-Forward Window  '
        f'({n_pos}/{len(windows_df)} positive = {100*n_pos/len(windows_df):.0f}%)'
    )
    ax.set_xlabel('Window index (oldest -> newest)')
    ax.set_ylabel('Gross Return %')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'results/{ticker}_xgb_equity{suffix}.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ---- Figure 2: holding-period decay + feature importance ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    hx    = sorted(decay.keys())
    means = [decay[h].mean() if len(decay[h]) > 0 else 0 for h in hx]
    stds  = [decay[h].std()  if len(decay[h]) > 0 else 0 for h in hx]
    ns    = [len(decay[h]) for h in hx]
    sems  = [s / np.sqrt(max(n, 1)) for s, n in zip(stds, ns)]
    ax.fill_between(hx,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color='steelblue', label='+-1 std')
    ax.errorbar(hx, means, yerr=sems, fmt='o-', color='steelblue',
                lw=1.5, capsize=4, label='Avg gross pips +- SEM')
    ax.axhline(0,   color='black',  lw=0.6, ls='-')
    ax.axhline(1.8, color='red',    lw=1.0, ls='--', label='1.8 pip breakeven')
    ax.axhline(1.0, color='orange', lw=0.8, ls=':',  label='1.0 pip (limit-order cost)')
    ax.set_xticks(hx)
    ax.set_xticklabels([f'{h}h' for h in hx])
    ax.set_title(f'Holding-Period Decay -- {ticker}\n'
                 f'(from {sum(ns)//max(len(hx),1):,} entry points)')
    ax.set_xlabel('Bars held after entry')
    ax.set_ylabel('Avg gross pips')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    if fi:
        top_fi = sorted(fi.items(), key=lambda x: -x[1])[:10]
        ax.barh([f[0] for f in top_fi], [f[1] for f in top_fi], color='steelblue')
        ax.set_title('Feature Importance -- last window (top 10)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'results/{ticker}_xgb_analysis{suffix}.png', dpi=100, bbox_inches='tight')
    plt.close()

    return {
        'ticker'              : ticker,
        'forward_bars'        : forward_bars,
        'threshold'           : long_threshold,
        'min_hold'            : min_hold,
        'drop_time'           : drop_time_features,
        'session_filter'      : session_filter,
        'limit_order'         : limit_order,
        'gross_return_pct'    : total_gross_ret,
        'net_return_pct'      : total_net_ret,
        'cost_paid_pct'       : total_cost_paid,
        'gross_sharpe'        : agg_gross_sharpe,
        'net_sharpe'          : agg_net_sharpe,
        'net_max_dd_pct'      : agg_net_dd,
        'bah_return_pct'      : bah_ret,
        'total_trades'        : total_n_trades,
        'avg_gross_pips_trade': agg_avg_pips,
        'avg_hold_bars'       : agg_avg_hold,
        'win_rate_bar_pct'    : win_rate,
        'windows'             : len(window_stats),
        # Limit comparison (if run)
        'lim_net_return_pct'  : lim_net_ret    if limit_order else None,
        'lim_net_sharpe'      : lim_net_sharpe if limit_order else None,
        'lim_fill_rate_pct'   : avg_fill_rate  if limit_order else None,
        '_decay'              : decay,
        '_scenarios'          : scenarios,
    }


# ---------------------------------------------------------------------------
# Cross-ticker comparison plot
# ---------------------------------------------------------------------------

def _plot_comparison(results: list, suffix: str) -> None:
    tickers   = [r['ticker']               for r in results]
    g_sharpe  = [r['gross_sharpe']         for r in results]
    n_sharpe  = [r['net_sharpe']           for r in results]
    avg_pips  = [r['avg_gross_pips_trade'] for r in results]
    g_ret     = [r['gross_return_pct']     for r in results]
    bah_rets  = [r['bah_return_pct']       for r in results]

    x = np.arange(len(tickers))
    w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(x - w/2, g_sharpe, width=w, label='Gross Sharpe',
           color=['#2ecc71' if v > 0 else '#e74c3c' for v in g_sharpe])
    ax.bar(x + w/2, n_sharpe, width=w, label='Net Sharpe',
           color=['#27ae60' if v > 0 else '#c0392b' for v in n_sharpe], alpha=0.85)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tickers, rotation=30, ha='right')
    ax.set_title('Gross vs Net Ann. Sharpe\n(gross = does the edge generalise?)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

    ax = axes[1]
    colors = ['#2ecc71' if v > 1.8 else ('#f39c12' if v > 0 else '#e74c3c') for v in avg_pips]
    ax.bar(x, avg_pips, color=colors, width=0.6)
    ax.axhline(1.8, color='red',    lw=1.2, ls='--', label='1.8 pip breakeven (market order)')
    ax.axhline(1.0, color='orange', lw=0.8, ls=':',  label='1.0 pip (limit order)')
    ax.axhline(0,   color='black',  lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tickers, rotation=30, ha='right')
    ax.set_title('Avg Gross Pips / Trade\n(green = above market-order breakeven)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

    ax = axes[2]
    ax.bar(x - w/2, g_ret,    width=w, label='Strategy gross %',
           color=['#2ecc71' if v > 0 else '#e74c3c' for v in g_ret])
    ax.bar(x + w/2, bah_rets, width=w, label='Buy & hold %',
           color='darkorange', alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tickers, rotation=30, ha='right')
    ax.set_title('Strategy Gross Return vs Buy-and-Hold\n(over test periods only)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle(
        f'Cross-Ticker Comparison  |  fwd={results[0]["forward_bars"]}  '
        f'thr={results[0]["threshold"]:.2f}  hold={results[0]["min_hold"]}',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(f'results/multi_comparison{suffix}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n  [Saved] results/multi_comparison{suffix}.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='XGBoost Walk-Forward Baseline')
    parser.add_argument('--ticker',  type=str, default=None,
                        help='Single ticker symbol')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated list, e.g. EURUSD,GBPUSD,USDJPY')
    parser.add_argument('--forward_bars',        type=int,   default=FORWARD_BARS)
    parser.add_argument('--threshold',           type=float, default=LONG_THRESHOLD)
    parser.add_argument('--min_hold',            type=int,   default=MIN_HOLD)
    parser.add_argument('--drop_time_features',  action='store_true')
    parser.add_argument('--limit_order',         action='store_true',
                        help='Simulate limit-order fills: fill only if next bar H/L crosses limit price')
    parser.add_argument('--session_filter',      action='store_true',
                        help=f'Only allow new entries {SESSION_UTC_START}:00-{SESSION_UTC_END}:00 UTC '
                             f'(London/NY overlap).  Assumes SUPER dataset timestamps are UTC.')
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = ALL_TICKERS

    summary = []
    for t in tickers:
        result = run_ticker(
            t,
            forward_bars        = args.forward_bars,
            long_threshold      = args.threshold,
            short_threshold     = args.threshold,
            min_hold            = args.min_hold,
            drop_time_features  = args.drop_time_features,
            limit_order         = args.limit_order,
            session_filter      = args.session_filter,
        )
        if result:
            summary.append(result)

    if not summary:
        print("\nNo tickers produced results.")
        return

    os.makedirs('results', exist_ok=True)
    suffix = f"_fwd{args.forward_bars}_thr{int(args.threshold*100)}_hold{args.min_hold}"
    if args.drop_time_features:
        suffix += "_notime"
    if args.session_filter:
        suffix += "_session"
    if args.limit_order:
        suffix += "_limit"

    # Strip internal fields before saving CSV
    skip = {'_decay', '_scenarios'}
    csv_rows = [{k: v for k, v in r.items() if k not in skip} for r in summary]
    summary_df = pd.DataFrame(csv_rows).sort_values('net_sharpe', ascending=False)
    summary_df.to_csv(f'results/xgb_wfo_summary{suffix}.csv', index=False)

    print(f"\n{'='*95}")
    print("FLEET SUMMARY  (sorted by net Sharpe)")
    print(f"{'='*95}")
    cols = ['ticker', 'gross_return_pct', 'net_return_pct', 'cost_paid_pct',
            'gross_sharpe', 'net_sharpe', 'net_max_dd_pct',
            'bah_return_pct', 'avg_gross_pips_trade', 'avg_hold_bars', 'total_trades']
    print(summary_df[cols].to_string(index=False, float_format='{:.2f}'.format))
    print(f"\nAll results saved to results/")

    if len(summary) > 1:
        _plot_comparison(summary, suffix)


if __name__ == '__main__':
    main()