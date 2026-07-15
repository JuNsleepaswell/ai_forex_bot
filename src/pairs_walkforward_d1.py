#!/usr/bin/env python3
"""
pairs_walkforward_d1.py  —  Pairs Trading Walk-Forward  (D1 / Daily Resolution)

Phase 1  Engle-Granger cointegration scan on first 60% of D1 data.
         Half-life acceptance: 5–120 trading DAYS.
Phase 2  Walk-forward: 36-month train / 6-month test / 5-bar purge.
         Cointegration re-tested each window; only trades p < 0.05 windows.
         Flags total trade count < 30 as insufficient for confidence.

Usage:
    python src/pairs_walkforward_d1.py [--scan_only]
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
MAJORS  = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDJPY", "USDCHF", "XAUUSD"]
CROSSES = ["AUDNZD", "AUDCAD", "EURGBP", "EURAUD", "NZDCAD", "CADCHF"]
SCAN_TICKERS = MAJORS + CROSSES

DATA_DIR    = "data"
DATA_SUFFIX = "_D1.csv"
RESULTS_DIR = "results"

TRAIN_MONTHS = 36      # 3-year training window
TEST_MONTHS  = 6       # 6-month test window (slide forward by 6 months)
PURGE_BARS   = 5       # trading days gap between train-end and test-start

Z_ENTRY  = 2.0
Z_STOP   = 4.0
COOLDOWN = 60          # trading days blocked after a stop (≈ 3 months at D1)

ROLL_HEDGE  = 250      # trailing D1 bars for rolling OLS hedge ratio (~1 year)
ROLL_ZSCORE = 120      # trailing D1 bars for rolling z-score (~6 months)

COST_SCENARIOS = [1.8, 1.0, 3.0]   # pips per leg, per side
                                     # 3.0 covers illiquid crosses (NZDCAD, CADCHF)

PIP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.10,
    "AUDNZD": 0.0001, "AUDCAD": 0.0001, "EURGBP": 0.0001,
    "EURAUD": 0.0001, "NZDCAD": 0.0001, "CADCHF": 0.0001,
}

TYPICAL_SPREAD = {
    "EURUSD": 1.0, "GBPUSD": 1.2, "AUDUSD": 1.2, "NZDUSD": 1.5,
    "USDCAD": 1.5, "USDJPY": 1.0, "USDCHF": 1.5,
    "XAUUSD": 4.0,
    "AUDNZD": 2.5, "AUDCAD": 2.5, "EURGBP": 1.5,
    "EURAUD": 2.5, "NZDCAD": 3.5, "CADCHF": 3.5,
}

COINT_P    = 0.05
HL_MIN     = 5      # trading days
HL_MAX     = 120    # trading days
BARS_PER_YEAR     = 252
MIN_TRADES_FLAG   = 30   # flag if total trades below this


# ── DATA ─────────────────────────────────────────────────────────────────────
def load_prices() -> dict:
    series = {}
    for tk in SCAN_TICKERS:
        path = os.path.join(DATA_DIR, f"{tk}{DATA_SUFFIX}")
        if not os.path.exists(path):
            print(f"  [WARN] {path} missing — skipping {tk}")
            continue
        df = pd.read_csv(path, index_col="time", parse_dates=True)
        s  = df["Close"].sort_index().rename(tk)
        series[tk] = s
        print(f"  {tk:<8} {len(s):>5,} D1 bars  "
              f"({s.index[0].date()} to {s.index[-1].date()})")
    return series


def _align(all_series: dict, tk_a: str, tk_b: str) -> tuple:
    df = pd.concat([all_series[tk_a], all_series[tk_b]], axis=1).dropna()
    return df[tk_a], df[tk_b]


# ── HALF-LIFE (OU process) ────────────────────────────────────────────────────
def half_life(spread: pd.Series) -> float:
    lag  = spread.shift(1).dropna()
    diff = spread.diff().dropna()
    lag, diff = lag.align(diff, join="inner")
    lam = sm.OLS(diff, sm.add_constant(lag)).fit().params.iloc[1]
    return np.inf if lam >= 0 else -np.log(2) / lam


# ── PHASE 1 ───────────────────────────────────────────────────────────────────
def _rt_cost_pips(tk_a: str, tk_b: str, beta: float) -> float:
    sp_a   = TYPICAL_SPREAD.get(tk_a, 2.0)
    sp_b   = TYPICAL_SPREAD.get(tk_b, 2.0)
    scale  = PIP.get(tk_b, 0.0001) / PIP.get(tk_a, 0.0001)
    return 2.0 * (sp_a + abs(beta) * scale * sp_b)


def phase1_scan(all_series: dict) -> tuple[pd.DataFrame, list]:
    available = list(all_series.keys())
    pairs     = list(itertools.combinations(available, 2))

    print(f"\n{'='*78}")
    print(f"  PHASE 1A — D1 Pairs Cointegration Scan  ({len(pairs)} combinations)")
    print(f"  Training: first 60% of each pair's aligned D1 history (no peeking)")
    print(f"  Candidate: p < {COINT_P}  AND  {HL_MIN} <= half-life <= {HL_MAX} trading days")
    print(f"{'='*78}\n")
    print(f"  {'Pair':<18} {'N':>5} {'P-Value':>9} {'HL(days)':>10} "
          f"{'Beta':>10} {'RT-cost(pip-A)':>15}  Status")
    print(f"  {'-'*80}")

    pair_rows      = []
    pair_candidates = []

    for tk_a, tk_b in pairs:
        if tk_a not in all_series or tk_b not in all_series:
            continue
        A_full, B_full = _align(all_series, tk_a, tk_b)
        cutoff = int(len(A_full) * 0.60)
        A, B   = A_full.iloc[:cutoff], B_full.iloc[:cutoff]
        if len(A) < 200:
            continue

        X    = sm.add_constant(B)
        beta = sm.OLS(A, X).fit().params.iloc[1]
        sp   = A - beta * B
        _, p, _ = coint(A, B)
        hl   = half_life(sp)
        rtc  = _rt_cost_pips(tk_a, tk_b, beta)
        is_cand = (p < COINT_P) and (HL_MIN <= hl <= HL_MAX)

        if is_cand:
            pair_candidates.append(dict(A=tk_a, B=tk_b, Beta=beta,
                                        P_Value=p, Half_Life=hl))

        hl_s   = f"{hl:>8.1f}" if not np.isinf(hl) else "     INF"
        status = ("CANDIDATE **" if is_cand
                  else ("p<.05 HL out of range" if p < COINT_P else ""))
        print(f"  {tk_a}/{tk_b:<13} {cutoff:>5,} {p:>9.5f} {hl_s:>10} "
              f"{beta:>10.4f} {rtc:>15.1f}  {status}")
        pair_rows.append(dict(Pair=f"{tk_a}/{tk_b}", A=tk_a, B=tk_b,
                              N=cutoff, P_Value=p, Half_Life=hl,
                              Beta=beta, RT_cost_pips=round(rtc, 1)))

    pairs_df = (pd.DataFrame(pair_rows).sort_values("P_Value")
                .reset_index(drop=True))
    print(f"\n  {len(pair_candidates)} pair candidate(s) passed.\n")

    # ── SINGLES: ADF on each cross ────────────────────────────────────────────
    print(f"{'='*78}")
    print(f"  PHASE 1B — Single-Cross ADF Stationarity  ({len(CROSSES)} instruments)")
    print(f"{'='*78}\n")
    print(f"  {'Ticker':<10} {'N':>5} {'ADF p-val':>10} {'HL(days)':>10} "
          f"{'Spread(pip)':>12}  Status")
    print(f"  {'-'*58}")

    single_candidates = []
    for tk in CROSSES:
        if tk not in all_series:
            continue
        s      = all_series[tk].sort_index().dropna()
        cutoff = int(len(s) * 0.60)
        s_tr   = s.iloc[:cutoff]
        if len(s_tr) < 200:
            continue

        adf_p = adfuller(s_tr, autolag="AIC")[1]
        hl    = half_life(s_tr) if adf_p < 0.10 else np.inf
        sp    = TYPICAL_SPREAD.get(tk, 3.0)
        is_cand = (adf_p < COINT_P) and (HL_MIN <= hl <= HL_MAX)
        if is_cand:
            single_candidates.append(dict(A=tk, B=None, P_Value=adf_p, Half_Life=hl))

        hl_s   = f"{hl:>8.1f}" if not np.isinf(hl) else "     INF"
        status = ("CANDIDATE **" if is_cand
                  else ("p<.05 HL out of range" if adf_p < COINT_P else ""))
        print(f"  {tk:<10} {cutoff:>5,} {adf_p:>10.5f} {hl_s:>10} "
              f"{sp:>9.1f} (RT {2*sp:.0f})  {status}")

    print(f"\n  {len(single_candidates)} single-cross candidate(s) passed.\n")

    all_cands = pair_candidates + single_candidates
    print(f"{'='*78}")
    print(f"  COMBINED CANDIDATES: {len(all_cands)}")
    print(f"{'='*78}")
    if not all_cands:
        print("  None pass p < 0.05 AND half-life 5–120 trading days.\n")
    else:
        for c in all_cands:
            kind = "PAIR" if c.get("B") else "SINGLE"
            lbl  = f"{c['A']}/{c['B']}" if c.get("B") else c["A"]
            print(f"  [{kind}] {lbl:<18}  p={c['P_Value']:.5f}  "
                  f"HL={c['Half_Life']:.1f} days")
    print()
    return pairs_df, all_cands


# ── ROLLING FEATURES (vectorised, zero look-ahead) ───────────────────────────
def compute_features(A: pd.Series, B: pd.Series) -> pd.DataFrame:
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
def simulate_window(feats: pd.DataFrame, pip_a: float, pip_b: float) -> dict:
    """
    Gross/net P&L per bar at D1 resolution.
    bar_cf: cost factor at 1.0 pip/leg/side — scale by cost scenario for net.
    Returns hold times in DAYS (bars at D1).
    """
    z_a  = feats["z"].values
    sp_a = feats["spread"].values
    hg_a = feats["hedge"].values
    n    = len(feats)

    bar_gross = np.zeros(n)
    bar_cf    = np.zeros(n)

    pos       = 0
    cooldown  = 0
    entry_i   = 0
    trd_gross = 0.0
    trd_cf    = 0.0
    trades    = []

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

            if abs(zi) > Z_STOP:
                cf = _cf(i)
                bar_cf[i] += cf; trd_cf += cf
                trades.append((trd_gross, trd_cf, i - entry_i))
                pos = 0; trd_gross = 0.0; trd_cf = 0.0
                cooldown = COOLDOWN
                continue

            if (pos == 1 and zi >= 0) or (pos == -1 and zi <= 0):
                cf = _cf(i)
                bar_cf[i] += cf; trd_cf += cf
                trades.append((trd_gross, trd_cf, i - entry_i))
                pos = 0; trd_gross = 0.0; trd_cf = 0.0
                continue

        if pos == 0:
            if zi < -Z_ENTRY:
                pos = 1; entry_i = i
                cf = _cf(i); bar_cf[i] += cf
                trd_gross = 0.0; trd_cf = cf
            elif zi > Z_ENTRY:
                pos = -1; entry_i = i
                cf = _cf(i); bar_cf[i] += cf
                trd_gross = 0.0; trd_cf = cf

    if pos != 0:
        cf = _cf(n - 1)
        bar_cf[-1] += cf; trd_cf += cf
        trades.append((trd_gross, trd_cf, n - 1 - entry_i))

    gross_list = [t[0] for t in trades]
    cf_list    = [t[1] for t in trades]
    hold_list  = [t[2] for t in trades]   # in trading DAYS at D1

    return {
        "bar_gross":  bar_gross,
        "bar_cf":     bar_cf,
        "n_trades":   len(trades),
        "avg_hold_d": float(np.mean(hold_list)) if hold_list else 0.0,
        "gross_list": gross_list,
        "cf_list":    cf_list,
    }


# ── WALK-FORWARD ─────────────────────────────────────────────────────────────
def walkforward_pair(all_series: dict, ticker_a: str, ticker_b: str) -> dict | None:
    A, B = _align(all_series, ticker_a, ticker_b)
    pip_a, pip_b = PIP[ticker_a], PIP[ticker_b]
    label = f"{ticker_a}/{ticker_b}"

    print(f"  {label} ...", end="", flush=True)

    feats = compute_features(A, B)

    months = pd.date_range(
        A.index[0].normalize(),
        A.index[-1] + pd.offsets.MonthEnd(),
        freq="MS",
    )
    if len(months) < TRAIN_MONTHS + TEST_MONTHS + 1:
        print(" not enough data")
        return None

    gross_all  = []
    cf_all     = []
    A_level    = []
    n_trades   = 0
    hold_list  = []
    gross_list = []
    win_total  = 0
    win_coint  = 0

    # Slide test window forward by TEST_MONTHS each step (no test-period overlap)
    for i in range(TRAIN_MONTHS, len(months) - TEST_MONTHS, TEST_MONTHS):
        tr_start = months[i - TRAIN_MONTHS]
        tr_end   = months[i]
        # ~5 trading days purge  ≈  7 calendar days
        te_start = tr_end + pd.Timedelta(days=7)
        te_end   = months[i + TEST_MONTHS]

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

        if not held:
            gross_all.append(np.zeros(len(f_te)))
            cf_all.append(np.zeros(len(f_te)))
            A_level.append(A_te.values)
            continue

        res = simulate_window(f_te, pip_a, pip_b)
        gross_all.append(res["bar_gross"])
        cf_all.append(res["bar_cf"])
        A_level.append(A_te.values)
        n_trades  += res["n_trades"]
        hold_list += [res["avg_hold_d"]] if res["avg_hold_d"] > 0 else []
        gross_list += res["gross_list"]

    if not gross_all:
        print(" no valid windows")
        return None

    gross_arr = np.concatenate(gross_all)
    cf_arr    = np.concatenate(cf_all)
    A_arr     = np.concatenate(A_level)

    ref_price = float(np.nanmean(A_arr[A_arr > 0])) if np.any(A_arr > 0) else 1.0

    def sharpe(r: np.ndarray) -> float:
        mu, sd = r.mean(), r.std()
        return float((mu / (sd + 1e-12)) * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0

    def max_dd(cum_pct: np.ndarray) -> float:
        peak = np.maximum.accumulate(cum_pct)
        return float((peak - cum_pct).max()) if len(cum_pct) else 0.0

    scenario_metrics = {}
    for cp in COST_SCENARIOS:
        net_arr   = gross_arr - cp * cf_arr
        cum_g     = np.cumsum(gross_arr) / ref_price * 100
        cum_n     = np.cumsum(net_arr)   / ref_price * 100
        scenario_metrics[cp] = {
            "gross_ret_pct": float(cum_g[-1]),
            "net_ret_pct":   float(cum_n[-1]),
            "gross_sharpe":  sharpe(gross_arr / ref_price),
            "net_sharpe":    sharpe(net_arr   / ref_price),
            "max_dd_pct":    max_dd(cum_n),
            "cum_gross":     cum_g,
            "cum_net":       cum_n,
        }

    avg_gross_pips = float(np.mean(gross_list)) / pip_a if gross_list else 0.0
    avg_hold_d     = float(np.mean(hold_list))  if hold_list else 0.0
    coint_pct      = 100 * win_coint / win_total if win_total else 0.0

    primary = scenario_metrics[1.8]
    flag = " [LOW TRADES]" if n_trades < MIN_TRADES_FLAG else ""
    print(f" done | trades={n_trades:3d}{flag} | gross={primary['gross_ret_pct']:+.1f}% | "
          f"net(1.8pip)={primary['net_ret_pct']:+.1f}% | coint={coint_pct:.0f}%")

    return {
        "pair":           label,
        "ticker_a":       ticker_a,
        "ticker_b":       ticker_b,
        "pip_a":          pip_a,
        "scenarios":      scenario_metrics,
        "n_trades":       n_trades,
        "avg_hold_d":     avg_hold_d,
        "avg_gross_pips": avg_gross_pips,
        "coint_pct":      coint_pct,
        "win_total":      win_total,
        "win_coint":      win_coint,
        "ref_price":      ref_price,
        "gross_arr":      gross_arr,
        "cf_arr":         cf_arr,
    }


# ── REPORTING ────────────────────────────────────────────────────────────────
def print_pair_results(r: dict) -> None:
    flag = "  [!] FEWER THAN 30 TRADES — treat statistics with caution" \
           if r["n_trades"] < MIN_TRADES_FLAG else ""
    print(f"\n  {'─'*72}")
    print(f"  {r['pair']}   trades={r['n_trades']}  avg_hold={r['avg_hold_d']:.1f}d  "
          f"coint={r['coint_pct']:.0f}% of {r['win_total']} windows")
    if flag:
        print(f"  {flag}")
    print(f"  avg gross per trade: {r['avg_gross_pips']:.1f} pips (leg A)\n")

    print(f"  {'Cost/leg/side':>14} {'Gross Ret%':>11} {'Net Ret%':>10} "
          f"{'Net Sharpe':>11} {'Max DD%':>9}")
    print(f"  {'-'*60}")
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
        print(f"\n  RESULT: net-positive at: {survives} pip/leg/side")


def plot_equity(r: dict, out_dir: str) -> None:
    m18 = r["scenarios"][1.8]
    m10 = r["scenarios"][1.0]
    m30 = r["scenarios"][3.0]

    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(m18["cum_net"]))
    ax.plot(x, m18["cum_gross"], color="steelblue", lw=1.3, label="Gross")
    ax.plot(x, m18["cum_net"],   color="darkorange", lw=1.3, label="Net 1.8 pip")
    ax.plot(x, m10["cum_net"],   color="green",      lw=0.9, ls="--", label="Net 1.0 pip")
    ax.plot(x, m30["cum_net"],   color="red",        lw=0.9, ls=":",  label="Net 3.0 pip")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_title(f"{r['pair']}  D1  —  Walk-Forward Equity  "
                 f"(train 36m / test 6m)  |  {r['n_trades']} trades")
    ax.set_ylabel("Cumulative Return % of avg price")
    ax.set_xlabel("Test-window bars (all windows concatenated)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(out_dir, f"d1_equity_{r['pair'].replace('/', '_')}.png")
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  Chart: {fname}")


def save_leaderboard(results: list, out_dir: str) -> None:
    rows = []
    for r in results:
        base = {
            "pair":           r["pair"],
            "n_trades":       r["n_trades"],
            "low_sample":     r["n_trades"] < MIN_TRADES_FLAG,
            "avg_hold_days":  round(r["avg_hold_d"], 1),
            "avg_gross_pips": round(r["avg_gross_pips"], 2),
            "coint_pct":      round(r["coint_pct"], 1),
            "win_total":      r["win_total"],
            "gross_ret_pct":  round(r["scenarios"][1.8]["gross_ret_pct"], 2),
        }
        for cp in COST_SCENARIOS:
            m = r["scenarios"][cp]
            base[f"net_ret_{cp}pip"]    = round(m["net_ret_pct"], 2)
            base[f"net_sharpe_{cp}pip"] = round(m["net_sharpe"], 3)
            base[f"max_dd_{cp}pip"]     = round(m["max_dd_pct"], 2)
        rows.append(base)

    lb = (pd.DataFrame(rows)
          .sort_values("net_ret_1.8pip", ascending=False)
          .reset_index(drop=True))
    path = os.path.join(out_dir, "pairs_d1_leaderboard.csv")
    lb.to_csv(path, index=False)
    print(f"\n  Leaderboard: {path}\n")

    cols = ["pair", "n_trades", "low_sample", "avg_hold_days", "coint_pct",
            "gross_ret_pct", "net_ret_1.8pip", "net_sharpe_1.8pip", "max_dd_1.8pip"]
    print(lb[cols].to_string(index=False))


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading D1 prices...")
    all_series = load_prices()

    _, candidates = phase1_scan(all_series)

    if args.scan_only or not candidates:
        return

    pair_cands = [c for c in candidates if c.get("B") is not None]
    if not pair_cands:
        print("  No pair candidates — single-cross backtest not implemented.")
        return

    print(f"{'='*78}")
    print(f"  PHASE 2 — D1 Walk-Forward Backtest  ({len(pair_cands)} pair(s))")
    print(f"  Train {TRAIN_MONTHS}m / Test {TEST_MONTHS}m / Purge {PURGE_BARS} bars")
    print(f"  |z|>{Z_ENTRY} enter  z=0 exit  |z|>{Z_STOP} stop  cooldown {COOLDOWN}d")
    print(f"  Rolling hedge {ROLL_HEDGE}d  z-score {ROLL_ZSCORE}d")
    print(f"  Cost scenarios: {COST_SCENARIOS} pip/leg/side  (total trades <{MIN_TRADES_FLAG} = flagged)")
    print(f"{'='*78}\n")

    results = []
    for c in pair_cands:
        r = walkforward_pair(all_series, c["A"], c["B"])
        if r is not None:
            results.append(r)

    if not results:
        print("\n  No results.\n")
        return

    print(f"\n{'='*78}")
    print(f"  RESULTS — per pair")
    print(f"{'='*78}")
    for r in results:
        print_pair_results(r)
        plot_equity(r, RESULTS_DIR)

    print(f"\n{'='*78}")
    print(f"  LEADERBOARD")
    print(f"{'='*78}")
    save_leaderboard(results, RESULTS_DIR)


if __name__ == "__main__":
    main()
