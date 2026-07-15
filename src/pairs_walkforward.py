#!/usr/bin/env python3
"""
pairs_walkforward.py  —  Statistical Arbitrage / Pairs Trading Walk-Forward

Phase 1  Cointegration scan on the first 60% of data (no peeking).
Phase 2  Walk-forward backtest: 12-month train / 1-month test / 24-bar purge.
         Cointegration re-tested each window; only trades windows where p < 0.05.

Usage:
    python src/pairs_walkforward.py             # full run
    python src/pairs_walkforward.py --scan_only # Phase 1 table only
"""

import argparse, itertools, os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
# USD majors (original universe + USDCHF which is in data/)
MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDJPY", "USDCHF", "XAUUSD"]
# Non-USD crosses available in data/
CROSSES = ["AUDNZD", "AUDCAD", "EURGBP", "EURAUD", "NZDCAD", "CADCHF"]
# Full scan universe: all combinations of MAJORS+CROSSES are tested as pairs;
# CROSSES are also tested individually for ADF stationarity.
SCAN_TICKERS = MAJORS + CROSSES

DATA_DIR    = "data"
RESULTS_DIR = "results"

TRAIN_MONTHS = 12
PURGE_BARS   = 24

Z_ENTRY  = 2.0
Z_STOP   = 4.0
COOLDOWN = 500

ROLL_HEDGE  = 1000
ROLL_ZSCORE = 500

COST_SCENARIOS = [1.8, 1.0, 0.5]   # pips per leg, per side

PIP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.10,
    "AUDNZD": 0.0001, "AUDCAD": 0.0001, "EURGBP": 0.0001,
    "EURAUD": 0.0001, "NZDCAD": 0.0001, "CADCHF": 0.0001,
}

# Typical retail-broker bid-ask spread in pips per instrument (conservative H1 average).
# Used in the scan cost column; the backtest uses COST_SCENARIOS which models
# total transaction cost per leg (spread + slippage).
TYPICAL_SPREAD = {
    "EURUSD": 1.0, "GBPUSD": 1.2, "AUDUSD": 1.2, "NZDUSD": 1.5,
    "USDCAD": 1.5, "USDJPY": 1.0, "USDCHF": 1.5,
    "XAUUSD": 4.0,   # 4 × $0.10 = $0.40 typical for gold
    "AUDNZD": 2.5, "AUDCAD": 2.5, "EURGBP": 1.5,
    "EURAUD": 2.5, "NZDCAD": 3.5, "CADCHF": 3.5,
}

COINT_P = 0.05
HL_MIN  = 12     # hours
HL_MAX  = 500    # hours


# ── DATA ─────────────────────────────────────────────────────────────────────
def load_prices() -> dict:
    """
    Return {ticker: pd.Series} — one Close series per instrument.
    Pairs are aligned lazily (inner join per pair) so a missing instrument
    in one cross does not shrink the full dataset for unrelated pairs.
    """
    series = {}
    for tk in SCAN_TICKERS:
        path = os.path.join(DATA_DIR, f"{tk}_H1.csv")
        if not os.path.exists(path):
            print(f"  [WARN] {path} missing — skipping {tk}")
            continue
        df = pd.read_csv(path, index_col="time", parse_dates=True)
        s  = df["Close"].sort_index().rename(tk)
        series[tk] = s
        print(f"  {tk:<8} {len(s):>7,} bars  "
              f"({s.index[0].date()} to {s.index[-1].date()})")
    return series


def _align(all_series: dict, tk_a: str, tk_b: str) -> tuple:
    """Return (A, B) as aligned pd.Series (inner join on timestamps)."""
    df = pd.concat([all_series[tk_a], all_series[tk_b]], axis=1).dropna()
    return df[tk_a], df[tk_b]


# ── HALF-LIFE (OU process) — reused from 03_pairs_cointegration_research.py ──
def half_life(spread: pd.Series) -> float:
    lag  = spread.shift(1).dropna()
    diff = spread.diff().dropna()
    lag, diff = lag.align(diff, join="inner")
    lam = sm.OLS(diff, sm.add_constant(lag)).fit().params.iloc[1]
    return np.inf if lam >= 0 else -np.log(2) / lam


# ── PHASE 1 ───────────────────────────────────────────────────────────────────
def _rt_cost_pips(tk_a: str, tk_b: str, beta: float) -> float:
    """
    Estimated round-trip cost in pips of leg A for one full trade
    (entry + exit on both legs).

    Formula: 2 × (spread_A + |beta| × pip_B/pip_A × spread_B)
    Converts the beta-weighted B-leg cost into A-leg pip units.
    """
    sp_a = TYPICAL_SPREAD.get(tk_a, 2.0)
    sp_b = TYPICAL_SPREAD.get(tk_b, 2.0)
    scale = PIP.get(tk_b, 0.0001) / PIP.get(tk_a, 0.0001)
    return 2.0 * (sp_a + abs(beta) * scale * sp_b)


def phase1_scan(all_series: dict) -> tuple[pd.DataFrame, list]:
    available = list(all_series.keys())
    pairs = list(itertools.combinations(available, 2))

    # ── PAIRS: Engle-Granger cointegration ──────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  PHASE 1A — Pairs Cointegration Scan  ({len(pairs)} combinations)")
    print(f"  Training: first 60% of each pair's aligned history (no peeking)")
    print(f"  Candidate: Engle-Granger p < {COINT_P}  AND  {HL_MIN}h <= half-life <= {HL_MAX}h")
    print(f"{'='*76}\n")
    print(f"  {'Pair':<18} {'N-train':>8} {'P-Value':>9} {'HL(h)':>9} "
          f"{'Beta':>10} {'RT-cost(pips-A)':>16}  Status")
    print(f"  {'-'*78}")

    pair_rows = []
    pair_candidates = []
    for tk_a, tk_b in pairs:
        if tk_a not in all_series or tk_b not in all_series:
            continue
        A_full, B_full = _align(all_series, tk_a, tk_b)
        cutoff = int(len(A_full) * 0.60)
        A, B = A_full.iloc[:cutoff], B_full.iloc[:cutoff]
        if len(A) < 500:
            continue

        X    = sm.add_constant(B)
        beta = sm.OLS(A, X).fit().params.iloc[1]
        sp   = A - beta * B
        _, p, _ = coint(A, B)
        hl  = half_life(sp)
        rtc = _rt_cost_pips(tk_a, tk_b, beta)
        is_cand = (p < COINT_P) and (HL_MIN <= hl <= HL_MAX)
        if is_cand:
            pair_candidates.append(dict(A=tk_a, B=tk_b, Beta=beta,
                                        P_Value=p, Half_Life=hl))

        hl_s   = f"{hl:>7.0f}" if not np.isinf(hl) else "    INF"
        status = ("CANDIDATE **" if is_cand
                  else ("p<.05 HL out of range" if p < COINT_P else ""))
        print(f"  {tk_a}/{tk_b:<13} {cutoff:>8,} {p:>9.5f} {hl_s:>9} "
              f"{beta:>10.4f} {rtc:>16.1f}  {status}")
        pair_rows.append(dict(Pair=f"{tk_a}/{tk_b}", A=tk_a, B=tk_b,
                              N_train=cutoff, P_Value=p, Half_Life=hl,
                              Beta=beta, RT_cost_pips=round(rtc, 1)))

    pairs_df = (pd.DataFrame(pair_rows).sort_values("P_Value")
                .reset_index(drop=True))

    print(f"\n  {len(pair_candidates)} pair candidate(s) passed.\n")

    # ── SINGLES: ADF stationarity on each cross ──────────────────────────────
    print(f"{'='*76}")
    print(f"  PHASE 1B — Single-Cross ADF Stationarity  ({len(CROSSES)} instruments)")
    print(f"  A stationary cross price needs only one leg — half the cost of a pair.")
    print(f"  Note: forex H1 prices almost never pass ADF; report is for completeness.")
    print(f"{'='*76}\n")
    print(f"  {'Ticker':<10} {'N-train':>8} {'ADF p-val':>10} {'HL(h)':>9} "
          f"{'Spread(pips)':>13}  Status")
    print(f"  {'-'*60}")

    single_candidates = []
    for tk in CROSSES:
        if tk not in all_series:
            continue
        s_full = all_series[tk].sort_index()
        cutoff = int(len(s_full) * 0.60)
        s = s_full.iloc[:cutoff].dropna()
        if len(s) < 500:
            continue

        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        hl = half_life(s) if adf_p < 0.10 else np.inf
        sp = TYPICAL_SPREAD.get(tk, 3.0)
        rt = 2.0 * sp   # single-leg round-trip cost in pips
        is_cand = (adf_p < COINT_P) and (HL_MIN <= hl <= HL_MAX)
        if is_cand:
            single_candidates.append(dict(A=tk, B=None, P_Value=adf_p,
                                          Half_Life=hl, type="single"))

        hl_s   = f"{hl:>7.0f}" if not np.isinf(hl) else "    INF"
        status = ("CANDIDATE **" if is_cand
                  else ("p<.05 HL out of range" if adf_p < COINT_P else ""))
        print(f"  {tk:<10} {cutoff:>8,} {adf_p:>10.5f} {hl_s:>9} "
              f"{sp:>10.1f} ({rt:.0f} RT)  {status}")

    print(f"\n  {len(single_candidates)} single-cross candidate(s) passed.\n")

    # ── COMBINED SUMMARY ──────────────────────────────────────────────────────
    all_cands = pair_candidates + single_candidates
    print(f"{'='*76}")
    print(f"  COMBINED CANDIDATES: {len(all_cands)} total")
    print(f"{'='*76}")
    if not all_cands:
        print("  None — nothing passes both p < 0.05 AND 12h <= half-life <= 500h.")
        print("  Possible causes: macro regime spans years (HL too long), or")
        print("  insufficient comovement (p too high).\n")
    else:
        for c in all_cands:
            kind = "PAIR" if c.get("B") else "SINGLE"
            lbl  = f"{c['A']}/{c['B']}" if c.get("B") else c["A"]
            print(f"  [{kind}] {lbl}  p={c['P_Value']:.5f}  HL={c['Half_Life']:.0f}h")

    return pairs_df, all_cands


# ── ROLLING FEATURES (vectorised, zero look-ahead) ───────────────────────────
def compute_features(A: pd.Series, B: pd.Series) -> pd.DataFrame:
    """
    Rolling OLS beta via rolling cov/var, each shifted 1 bar forward so that
    bar-i signals use only bars 0..(i-1).  Spread and z-score follow the same
    convention.
    """
    # shift(1): window [i-N, i-1] for the value at i  → no look-ahead
    roll_cov  = A.rolling(ROLL_HEDGE).cov(B).shift(1)
    roll_varB = B.rolling(ROLL_HEDGE).var().shift(1)
    hedge     = (roll_cov / roll_varB).ffill()

    spread     = A - hedge * B
    roll_mu    = spread.rolling(ROLL_ZSCORE).mean().shift(1)
    roll_sigma = spread.rolling(ROLL_ZSCORE).std().shift(1)
    z          = (spread - roll_mu) / (roll_sigma + 1e-12)

    return pd.DataFrame(
        {"hedge": hedge, "spread": spread, "z": z}, index=A.index
    )


# ── SINGLE-WINDOW SIMULATION ─────────────────────────────────────────────────
def simulate_window(
    feats: pd.DataFrame,
    pip_a: float,
    pip_b: float,
) -> dict:
    """
    Simulate the stat-arb strategy on pre-computed z/spread/hedge.

    Returns bar-level arrays:
      bar_gross      P&L in raw spread-price units, gross (no cost)
      bar_cf         cost factor per bar at 1.0 pip/leg (pip_a + |beta|*pip_b)
                     non-zero only at trade events (entry and exit)

    Net P&L at cost scenario cp  =  bar_gross - cp * bar_cf

    Positions: +1 = long spread (long A, short beta*B), -1 = short spread.
    """
    z_a  = feats["z"].values
    sp_a = feats["spread"].values
    hg_a = feats["hedge"].values
    n    = len(feats)

    bar_gross = np.zeros(n)
    bar_cf    = np.zeros(n)   # cost-factor per bar (at 1 pip)

    pos       = 0
    cooldown  = 0
    entry_i   = 0
    trd_gross = 0.0
    trd_cf    = 0.0
    trades    = []   # (gross, total_cf, hold_bars)

    def _cf(i: int) -> float:
        b = abs(hg_a[i]) if not np.isnan(hg_a[i]) else 1.0
        return pip_a + b * pip_b

    for i in range(1, n):
        zi  = z_a[i]
        spi = sp_a[i]
        ds  = spi - sp_a[i - 1]

        if np.isnan(zi) or np.isnan(spi) or np.isnan(sp_a[i - 1]):
            if cooldown > 0:
                cooldown -= 1
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        if pos != 0:
            mtm = pos * ds
            bar_gross[i] += mtm
            trd_gross    += mtm

            # Hard stop: |z| > Z_STOP
            if abs(zi) > Z_STOP:
                cf = _cf(i)
                bar_cf[i] += cf
                trd_cf    += cf
                trades.append((trd_gross, trd_cf, i - entry_i))
                pos = 0; trd_gross = 0.0; trd_cf = 0.0
                cooldown = COOLDOWN
                continue

            # Exit: z crossed back through 0
            if (pos == 1 and zi >= 0) or (pos == -1 and zi <= 0):
                cf = _cf(i)
                bar_cf[i] += cf
                trd_cf    += cf
                trades.append((trd_gross, trd_cf, i - entry_i))
                pos = 0; trd_gross = 0.0; trd_cf = 0.0
                continue

        # Entry (only when flat)
        if pos == 0:
            if zi < -Z_ENTRY:
                pos = 1; entry_i = i
                cf = _cf(i)
                bar_cf[i] += cf
                trd_gross = 0.0; trd_cf = cf
            elif zi > Z_ENTRY:
                pos = -1; entry_i = i
                cf = _cf(i)
                bar_cf[i] += cf
                trd_gross = 0.0; trd_cf = cf

    # Force-close any open position at end of window
    if pos != 0:
        cf = _cf(n - 1)
        bar_cf[-1] += cf
        trd_cf     += cf
        trades.append((trd_gross, trd_cf, n - 1 - entry_i))

    gross_list = [t[0] for t in trades]
    cf_list    = [t[1] for t in trades]   # total cost factor per trade
    hold_list  = [t[2] for t in trades]

    return {
        "bar_gross": bar_gross,
        "bar_cf":    bar_cf,
        "n_trades":  len(trades),
        "avg_hold":  float(np.mean(hold_list)) if hold_list else 0.0,
        "gross_list": gross_list,
        "cf_list":    cf_list,
    }


# ── WALK-FORWARD ─────────────────────────────────────────────────────────────
def walkforward_pair(
    all_series: dict,
    ticker_a: str,
    ticker_b: str,
) -> dict | None:
    A, B = _align(all_series, ticker_a, ticker_b)
    pip_a, pip_b = PIP[ticker_a], PIP[ticker_b]
    label = f"{ticker_a}/{ticker_b}"

    print(f"  {label} ...", end="", flush=True)

    feats = compute_features(A, B)

    # Monthly window boundaries
    months = pd.date_range(
        A.index[0].normalize(),
        A.index[-1] + pd.offsets.MonthEnd(),
        freq="MS",
    )
    if len(months) < TRAIN_MONTHS + 2:
        print(" not enough data")
        return None

    # Accumulators across test windows
    gross_all  = []
    cf_all     = []
    A_level    = []
    ts_index   = []
    n_trades   = 0
    hold_list  = []
    gross_list = []
    win_total  = 0
    win_coint  = 0

    for i in range(TRAIN_MONTHS, len(months) - 1):
        tr_start = months[i - TRAIN_MONTHS]
        tr_end   = months[i]
        te_start = tr_end + pd.Timedelta(hours=PURGE_BARS)
        te_end   = months[i + 1]

        A_tr = A[(A.index >= tr_start) & (A.index < tr_end)]
        B_tr = B[(B.index >= tr_start) & (B.index < tr_end)]
        if len(A_tr) < ROLL_HEDGE:
            continue

        win_total += 1

        _, p_win, _ = coint(A_tr, B_tr)
        held = p_win < COINT_P
        if held:
            win_coint += 1

        te_mask = (feats.index >= te_start) & (feats.index < te_end)
        f_te    = feats[te_mask]
        A_te    = A[te_mask]

        if len(f_te) < 5:
            continue

        # Append zero bars if cointegration failed (equity curve continuity)
        if not held:
            gross_all.append(np.zeros(len(f_te)))
            cf_all.append(np.zeros(len(f_te)))
            A_level.append(A_te.values)
            ts_index.append(f_te.index)
            continue

        res = simulate_window(f_te, pip_a, pip_b)
        gross_all.append(res["bar_gross"])
        cf_all.append(res["bar_cf"])
        A_level.append(A_te.values)
        ts_index.append(f_te.index)
        n_trades  += res["n_trades"]
        hold_list += [res["avg_hold"]] if res["avg_hold"] > 0 else []
        gross_list += res["gross_list"]

    if not gross_all:
        print(" no valid windows")
        return None

    gross_arr = np.concatenate(gross_all)
    cf_arr    = np.concatenate(cf_all)
    A_arr     = np.concatenate(A_level)
    full_idx  = np.concatenate([ix for ix in ts_index])

    ref_price = float(np.nanmean(A_arr[A_arr > 0])) if np.any(A_arr > 0) else 1.0

    def sharpe(r: np.ndarray) -> float:
        mu, sd = r.mean(), r.std()
        return float((mu / (sd + 1e-12)) * np.sqrt(252 * 24)) if sd > 1e-12 else 0.0

    def max_dd(cum_pct: np.ndarray) -> float:
        peak = np.maximum.accumulate(cum_pct)
        dd   = peak - cum_pct
        return float(dd.max()) if len(dd) else 0.0

    # Metrics for each cost scenario
    scenario_metrics = {}
    for cp in COST_SCENARIOS:
        net_arr    = gross_arr - cp * cf_arr
        cum_g      = np.cumsum(gross_arr) / ref_price * 100
        cum_n      = np.cumsum(net_arr)   / ref_price * 100
        gross_ret  = gross_arr / ref_price
        net_ret    = net_arr   / ref_price
        scenario_metrics[cp] = {
            "gross_ret_pct": float(cum_g[-1]),
            "net_ret_pct":   float(cum_n[-1]),
            "gross_sharpe":  sharpe(gross_ret),
            "net_sharpe":    sharpe(net_ret),
            "max_dd_pct":    max_dd(cum_n),
            "cum_gross":     cum_g,
            "cum_net":       cum_n,
        }

    avg_gross_pips = (float(np.mean(gross_list)) / pip_a) if gross_list else 0.0
    coint_pct      = 100 * win_coint / win_total if win_total else 0.0

    primary = scenario_metrics[1.8]
    print(f" done | trades={n_trades:3d} | gross={primary['gross_ret_pct']:+.1f}% | "
          f"net(1.8pip)={primary['net_ret_pct']:+.1f}% | coint={coint_pct:.0f}%")

    return {
        "pair":           label,
        "ticker_a":       ticker_a,
        "ticker_b":       ticker_b,
        "pip_a":          pip_a,
        "full_idx":       full_idx,
        "scenarios":      scenario_metrics,
        "n_trades":       n_trades,
        "avg_hold_h":     float(np.mean(hold_list)) if hold_list else 0.0,
        "avg_gross_pips": avg_gross_pips,
        "coint_pct":      coint_pct,
        "win_total":      win_total,
        "win_coint":      win_coint,
        "ref_price":      ref_price,
    }


# ── REPORTING ────────────────────────────────────────────────────────────────
def print_pair_results(r: dict) -> None:
    print(f"\n  {'─'*68}")
    print(f"  {r['pair']}   ({r['n_trades']} trades | avg hold {r['avg_hold_h']:.0f}h | "
          f"coint held {r['coint_pct']:.0f}% of {r['win_total']} windows)")
    print(f"  avg gross per trade: {r['avg_gross_pips']:.1f} pips (leg A)\n")

    print(f"  {'Cost/leg/side':>14} {'Gross Ret%':>11} {'Net Ret%':>10} "
          f"{'Net Sharpe':>11} {'Max DD%':>9}")
    print(f"  {'-'*58}")
    for cp in COST_SCENARIOS:
        m = r["scenarios"][cp]
        print(f"  {cp:.1f} pip        {m['gross_ret_pct']:>+11.2f} {m['net_ret_pct']:>+10.2f} "
              f"{m['net_sharpe']:>11.3f} {m['max_dd_pct']:>9.2f}")

    survives = [cp for cp in COST_SCENARIOS
                if r["scenarios"][cp]["net_ret_pct"] > 0
                and r["scenarios"][cp]["net_sharpe"] > 0]
    if not survives:
        print(f"\n  RESULT: does NOT survive costs at any scenario tested.")
    else:
        print(f"\n  RESULT: net-positive at cost scenarios: {survives} pip/leg/side")


def plot_equity(r: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    idx = np.arange(len(r["scenarios"][1.8]["cum_net"]))
    ax.plot(idx, r["scenarios"][1.8]["cum_gross"], color="steelblue",
            lw=1.2, label="Gross")
    ax.plot(idx, r["scenarios"][1.8]["cum_net"], color="darkorange",
            lw=1.2, label="Net (1.8 pip)")
    ax.plot(idx, r["scenarios"][0.5]["cum_net"], color="green",
            lw=0.8, ls="--", label="Net (0.5 pip)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_title(f"{r['pair']}  —  Walk-Forward Equity Curve (% of ref price)")
    ax.set_ylabel("Cumulative Return %")
    ax.set_xlabel("Test Bars (all windows concatenated)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(out_dir, f"equity_{r['pair'].replace('/', '_')}.png")
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  Saved: {fname}")


def save_leaderboard(results: list, out_dir: str) -> None:
    rows = []
    for r in results:
        base = {
            "pair":           r["pair"],
            "n_trades":       r["n_trades"],
            "avg_hold_h":     round(r["avg_hold_h"], 1),
            "avg_gross_pips": round(r["avg_gross_pips"], 2),
            "coint_pct":      round(r["coint_pct"], 1),
            "win_total":      r["win_total"],
        }
        for cp in COST_SCENARIOS:
            m = r["scenarios"][cp]
            base[f"gross_ret_pct"]               = round(m["gross_ret_pct"], 2)
            base[f"net_ret_pct_{cp}pip"]         = round(m["net_ret_pct"], 2)
            base[f"net_sharpe_{cp}pip"]          = round(m["net_sharpe"], 3)
            base[f"max_dd_pct_{cp}pip"]          = round(m["max_dd_pct"], 2)
        rows.append(base)

    lb = pd.DataFrame(rows).sort_values("net_ret_pct_1.8pip", ascending=False)
    path = os.path.join(out_dir, "pairs_leaderboard.csv")
    lb.to_csv(path, index=False)
    print(f"\n  Leaderboard saved: {path}")
    print(lb[["pair", "n_trades", "avg_hold_h", "coint_pct",
              "gross_ret_pct", "net_ret_pct_1.8pip",
              "net_sharpe_1.8pip", "max_dd_pct_1.8pip"]].to_string(index=False))


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_only", action="store_true",
                        help="Print Phase 1 candidate table and exit")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading prices...")
    all_series = load_prices()

    # Phase 1
    _, candidates = phase1_scan(all_series)

    if args.scan_only or not candidates:
        return

    # Phase 2: only run pair candidates (single-cross backtest is a different strategy)
    pair_cands = [c for c in candidates if c.get("B") is not None]
    if not pair_cands:
        print("  No pair candidates to backtest (single-cross candidates require"
              " a separate mean-reversion strategy, not implemented here).")
        return

    print(f"{'='*72}")
    print(f"  PHASE 2 -- Walk-Forward Backtest  ({len(pair_cands)} pair(s))")
    print(f"  Train {TRAIN_MONTHS}m / Test 1m / Purge {PURGE_BARS}h")
    print(f"  Entry |z|>{Z_ENTRY}  Exit z=0  Stop |z|>{Z_STOP}  Cooldown {COOLDOWN}bars")
    print(f"  Rolling hedge {ROLL_HEDGE}bars  Rolling z-score {ROLL_ZSCORE}bars")
    print(f"  Cost scenarios: {COST_SCENARIOS} pips/leg/side")
    print(f"{'='*72}\n")

    results = []
    for c in pair_cands:
        r = walkforward_pair(all_series, c["A"], c["B"])
        if r is not None:
            results.append(r)

    if not results:
        print("\n  No results to report.\n")
        return

    print(f"\n{'='*72}")
    print(f"  RESULTS — per pair")
    print(f"{'='*72}")
    for r in results:
        print_pair_results(r)
        plot_equity(r, RESULTS_DIR)

    print(f"\n{'='*72}")
    print(f"  LEADERBOARD")
    print(f"{'='*72}")
    save_leaderboard(results, RESULTS_DIR)


if __name__ == "__main__":
    main()