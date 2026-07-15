#!/usr/bin/env python3
"""
pairs_walkforward_etf.py  —  ETF Statistical Arbitrage Walk-Forward

Phase 1  Engle-Granger cointegration scan on first 60% of daily data.
         Candidate gate: p < 0.05  AND  5 <= half-life <= 120 trading days.

Phase 2  Walk-forward: 36-month train / 6-month test / 5-bar purge.
         Cointegration re-checked each window; only live windows trade.
         Graduation gate: n_trades >= 30  AND  net return > 0 after cost.

Cost model: fraction of portfolio value (A_price + |beta| * B_price) paid
            at each trade event (entry or exit).  Not pips — actual ETF cost.

Data: yfinance daily Adj Close, cached to data/etf/*.csv

Usage:
    python src/pairs_walkforward_etf.py               # full run
    python src/pairs_walkforward_etf.py --scan_only   # Phase 1 only
    python src/pairs_walkforward_etf.py --force_download  # re-fetch yfinance
"""

import argparse, itertools, os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
# 12 liquid US ETFs — C(12,2) = 66 combinations scanned
TICKERS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000 (small-cap)
    "TLT",   # iShares 20+ yr Treasury
    "IEF",   # iShares 7-10 yr Treasury
    "GLD",   # SPDR Gold
    "SLV",   # iShares Silver
    "XLE",   # Energy Select Sector
    "XLF",   # Financial Select Sector
    "XLK",   # Technology Select Sector
    "XLU",   # Utilities Select Sector
    "XLB",   # Materials Select Sector
]

DATA_DIR    = os.path.join("data", "etf")
RESULTS_DIR = "results"

YF_START    = "2004-01-01"
YF_END      = "2026-01-01"

TRAIN_MONTHS = 36      # 3-year training window
TEST_MONTHS  = 6       # 6-month test window, no overlap
PURGE_BARS   = 5       # trading-day gap between train-end and test-start

Z_ENTRY  = 2.0
Z_STOP   = 4.0
COOLDOWN = 60          # trading days blocked after stop-loss

ROLL_HEDGE  = 250      # trailing bars for rolling OLS hedge ratio  (~1 year)
ROLL_ZSCORE = 120      # trailing bars for rolling z-score          (~6 months)

# Cost = fraction of (A_price + |beta|*B_price) per trade event (entry OR exit).
# 0.0005 = 5 bps, 0.0010 = 10 bps, 0.0020 = 20 bps.
# Round-trip total = 2× this (entry + exit).
COST_SCENARIOS = [0.0005, 0.0010, 0.0020]

COINT_P         = 0.05
HL_MIN          = 5       # trading days
HL_MAX          = 120     # trading days
BARS_PER_YEAR   = 252
MIN_TRADES_FLAG = 30


# ── DATA ─────────────────────────────────────────────────────────────────────
def fetch_prices(force: bool = False) -> dict:
    """
    Returns {ticker: pd.Series} with daily Adj Close prices.
    Downloads from yfinance on first run; re-uses cached CSV afterwards.
    """
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    series = {}
    print("Loading ETF prices (yfinance / cache) ...")
    for tk in TICKERS:
        cache = os.path.join(DATA_DIR, f"{tk}_D1.csv")
        if os.path.exists(cache) and not force:
            s = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze()
        else:
            raw = yf.download(tk, start=YF_START, end=YF_END,
                              auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  [WARN] yfinance returned nothing for {tk}")
                continue
            s = raw["Close"].squeeze()
            s.index = pd.to_datetime(s.index)
            s.name  = tk
            s.to_frame("Close").to_csv(cache)
            s = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze()
        s = s.sort_index().rename(tk)
        series[tk] = s
        print(f"  {tk:<6}  {len(s):>5,} bars  "
              f"({s.index[0].date()} to {s.index[-1].date()})")
    return series


def _align(series: dict, tk_a: str, tk_b: str) -> tuple:
    df = pd.concat([series[tk_a], series[tk_b]], axis=1).dropna()
    return df[tk_a], df[tk_b]


# ── HALF-LIFE (Ornstein–Uhlenbeck) ───────────────────────────────────────────
def half_life(spread: pd.Series) -> float:
    lag  = spread.shift(1).dropna()
    diff = spread.diff().dropna()
    lag, diff = lag.align(diff, join="inner")
    lam = sm.OLS(diff, sm.add_constant(lag)).fit().params.iloc[1]
    return np.inf if lam >= 0 else -np.log(2) / lam


# ── PHASE 1 ───────────────────────────────────────────────────────────────────
def phase1_scan(series: dict) -> tuple:
    available = list(series.keys())
    pairs     = list(itertools.combinations(available, 2))

    print(f"\n{'='*76}")
    print(f"  PHASE 1 — ETF Cointegration Scan  ({len(pairs)} combinations)")
    print(f"  Training window: first 60% of aligned history (no peeking)")
    print(f"  Gate: p < {COINT_P}  AND  {HL_MIN} <= half-life <= {HL_MAX} trading days")
    print(f"{'='*76}\n")
    hdr = f"  {'Pair':<14} {'N':>5} {'P-Value':>9} {'HL(days)':>10} {'Beta':>8}  Status"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    rows       = []
    candidates = []

    for tk_a, tk_b in pairs:
        A_full, B_full = _align(series, tk_a, tk_b)
        cutoff = int(len(A_full) * 0.60)
        A, B   = A_full.iloc[:cutoff], B_full.iloc[:cutoff]
        if len(A) < 200:
            continue

        beta    = sm.OLS(A, sm.add_constant(B)).fit().params.iloc[1]
        sp      = A - beta * B
        _, p, _ = coint(A, B)
        hl      = half_life(sp)
        passed  = (p < COINT_P) and (HL_MIN <= hl <= HL_MAX)

        if passed:
            candidates.append(dict(A=tk_a, B=tk_b, Beta=beta,
                                   P_Value=p, Half_Life=hl))

        hl_s   = f"{hl:>8.1f}" if not np.isinf(hl) else "     INF"
        status = ("CANDIDATE **" if passed
                  else ("p<.05 HL out of range" if p < COINT_P else ""))
        print(f"  {tk_a}/{tk_b:<9} {cutoff:>5,} {p:>9.5f} {hl_s:>10} "
              f"{beta:>8.4f}  {status}")
        rows.append(dict(Pair=f"{tk_a}/{tk_b}", A=tk_a, B=tk_b,
                         N=cutoff, P_Value=p, Half_Life=hl, Beta=beta))

    scan_df = pd.DataFrame(rows).sort_values("P_Value").reset_index(drop=True)

    print(f"\n  {len(candidates)} candidate(s) passed Phase 1.")
    if candidates:
        print(f"\n  {'Pair':<18}  {'p-value':>9}  {'HL (days)':>10}")
        for c in candidates:
            print(f"  {c['A']}/{c['B']:<14}  {c['P_Value']:>9.5f}  "
                  f"{c['Half_Life']:>10.1f}")
    else:
        print("\n  Honesty note: no ETF pair passes both the cointegration")
        print("  test AND the half-life gate on first-60% training data.")
        print("  Walking forward on pairs with HL > 120 days is not advisable —")
        print("  mean reversion is too slow relative to typical holding costs.\n")
    print()
    return scan_df, candidates


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
        {"hedge": hedge, "spread": spread, "z": z, "A_price": A, "B_price": B},
        index=A.index,
    )


# ── SINGLE-WINDOW SIMULATION ─────────────────────────────────────────────────
def simulate_window(feats: pd.DataFrame) -> dict:
    """
    bar_gross[i]: raw spread P&L at bar i (in A-price dollar units).
    bar_cf[i]:    cost factor at bar i — multiply by cost_pct to get dollar cost.
                  cf = A_price + |beta| * B_price  (full portfolio notional).
    """
    z_a   = feats["z"].values
    sp_a  = feats["spread"].values
    hg_a  = feats["hedge"].values
    A_p   = feats["A_price"].values
    B_p   = feats["B_price"].values
    n     = len(feats)

    bar_gross = np.zeros(n)
    bar_cf    = np.zeros(n)

    pos       = 0
    cooldown  = 0
    entry_i   = 0
    trd_gross = 0.0
    trd_cf    = 0.0
    trades    = []

    def _cf(i: int) -> float:
        b   = abs(hg_a[i]) if not np.isnan(hg_a[i]) else 1.0
        a_p = A_p[i] if not np.isnan(A_p[i]) else 1.0
        b_p = B_p[i] if not np.isnan(B_p[i]) else 1.0
        return a_p + b * b_p

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
                bar_cf[i] += cf
                trd_cf    += cf
                trades.append((trd_gross, trd_cf, i - entry_i))
                pos = 0; trd_gross = 0.0; trd_cf = 0.0
                cooldown = COOLDOWN
                continue

            if (pos == 1 and zi >= 0) or (pos == -1 and zi <= 0):
                cf = _cf(i)
                bar_cf[i] += cf
                trd_cf    += cf
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
        bar_cf[-1] += cf
        trd_cf     += cf
        trades.append((trd_gross, trd_cf, n - 1 - entry_i))

    gross_list = [t[0] for t in trades]
    cf_list    = [t[1] for t in trades]
    hold_list  = [t[2] for t in trades]

    return {
        "bar_gross":  bar_gross,
        "bar_cf":     bar_cf,
        "n_trades":   len(trades),
        "avg_hold_d": float(np.mean(hold_list)) if hold_list else 0.0,
        "gross_list": gross_list,
        "cf_list":    cf_list,
    }


# ── WALK-FORWARD ─────────────────────────────────────────────────────────────
def walkforward_pair(series: dict, ticker_a: str, ticker_b: str) -> dict | None:
    A, B  = _align(series, ticker_a, ticker_b)
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
    port_all   = []
    n_trades   = 0
    hold_list  = []
    gross_list = []
    win_total  = 0
    win_coint  = 0

    for i in range(TRAIN_MONTHS, len(months) - TEST_MONTHS, TEST_MONTHS):
        tr_start = months[i - TRAIN_MONTHS]
        tr_end   = months[i]
        te_start = tr_end + pd.Timedelta(days=7)   # ~5 trading-day purge
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
        B_te    = B[te_mask]

        if len(f_te) < 5:
            continue

        h_abs  = np.where(np.isnan(f_te["hedge"].values), 1.0,
                          np.abs(f_te["hedge"].values))
        port_v = A_te.values + h_abs * B_te.values

        if not held:
            gross_all.append(np.zeros(len(f_te)))
            cf_all.append(np.zeros(len(f_te)))
            port_all.append(port_v)
            continue

        res = simulate_window(f_te)
        gross_all.append(res["bar_gross"])
        cf_all.append(res["bar_cf"])
        port_all.append(port_v)
        n_trades  += res["n_trades"]
        hold_list += [res["avg_hold_d"]] if res["avg_hold_d"] > 0 else []
        gross_list += res["gross_list"]

    if not gross_all:
        print(" no valid windows")
        return None

    gross_arr = np.concatenate(gross_all)
    cf_arr    = np.concatenate(cf_all)
    port_arr  = np.concatenate(port_all)

    ref_port = float(np.nanmean(port_arr[port_arr > 0])) if np.any(port_arr > 0) else 1.0

    def sharpe(r: np.ndarray) -> float:
        mu, sd = r.mean(), r.std()
        return float((mu / (sd + 1e-12)) * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0

    def max_dd(cum: np.ndarray) -> float:
        peak = np.maximum.accumulate(cum)
        return float((peak - cum).max()) if len(cum) else 0.0

    scenario_metrics = {}
    for cp in COST_SCENARIOS:
        net_arr = gross_arr - cp * cf_arr
        cum_g   = np.cumsum(gross_arr) / ref_port * 100
        cum_n   = np.cumsum(net_arr)   / ref_port * 100
        scenario_metrics[cp] = {
            "gross_ret_pct": float(cum_g[-1]),
            "net_ret_pct":   float(cum_n[-1]),
            "gross_sharpe":  sharpe(gross_arr / ref_port),
            "net_sharpe":    sharpe(net_arr   / ref_port),
            "max_dd_pct":    max_dd(cum_n),
            "cum_gross":     cum_g,
            "cum_net":       cum_n,
        }

    avg_gross_bps = (float(np.mean(gross_list)) / ref_port * 10_000) if gross_list else 0.0
    avg_hold_d    = float(np.mean(hold_list))  if hold_list else 0.0
    coint_pct     = 100 * win_coint / win_total if win_total else 0.0

    primary = scenario_metrics[COST_SCENARIOS[0]]
    flag    = " [LOW TRADES]" if n_trades < MIN_TRADES_FLAG else ""
    print(f" done | trades={n_trades:3d}{flag} | gross={primary['gross_ret_pct']:+.1f}% | "
          f"net(5bp)={primary['net_ret_pct']:+.1f}% | coint={coint_pct:.0f}%")

    return {
        "pair":           label,
        "ticker_a":       ticker_a,
        "ticker_b":       ticker_b,
        "scenarios":      scenario_metrics,
        "n_trades":       n_trades,
        "avg_hold_d":     avg_hold_d,
        "avg_gross_bps":  avg_gross_bps,
        "coint_pct":      coint_pct,
        "win_total":      win_total,
        "win_coint":      win_coint,
        "ref_port":       ref_port,
        "gross_arr":      gross_arr,
        "cf_arr":         cf_arr,
    }


# ── REPORTING ────────────────────────────────────────────────────────────────
def print_pair_results(r: dict) -> None:
    low_flag = "  [!] FEWER THAN 30 TRADES — statistics unreliable" \
               if r["n_trades"] < MIN_TRADES_FLAG else ""
    print(f"\n  {'─'*72}")
    print(f"  {r['pair']}   trades={r['n_trades']}  avg_hold={r['avg_hold_d']:.1f}d  "
          f"coint={r['coint_pct']:.0f}% of {r['win_total']} windows")
    if low_flag:
        print(f"  {low_flag}")
    print(f"  avg gross per trade: {r['avg_gross_bps']:.1f} bps\n")

    print(f"  {'Cost/event':>12} {'Gross Ret%':>11} {'Net Ret%':>10} "
          f"{'Net Sharpe':>11} {'Max DD%':>9}")
    print(f"  {'-'*58}")
    for cp in COST_SCENARIOS:
        m     = r["scenarios"][cp]
        label = f"{cp * 10_000:.0f} bp"
        print(f"  {label:<12} {m['gross_ret_pct']:>+11.2f} {m['net_ret_pct']:>+10.2f} "
              f"{m['net_sharpe']:>11.3f} {m['max_dd_pct']:>9.2f}")

    survives = [cp for cp in COST_SCENARIOS
                if r["scenarios"][cp]["net_ret_pct"] > 0
                and r["scenarios"][cp]["net_sharpe"] > 0]
    if not survives:
        print(f"\n  RESULT: does NOT survive costs at any scenario tested.")
    else:
        labels = [f"{c * 10_000:.0f}bp" for c in survives]
        print(f"\n  RESULT: net-positive at cost scenarios: {labels}")


def plot_equity(r: dict, out_dir: str) -> None:
    m05 = r["scenarios"][0.0005]
    m10 = r["scenarios"][0.0010]
    m20 = r["scenarios"][0.0020]

    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(m05["cum_net"]))
    ax.plot(x, m05["cum_gross"], color="steelblue",   lw=1.3, label="Gross")
    ax.plot(x, m05["cum_net"],   color="darkorange",  lw=1.3, label="Net  5 bp")
    ax.plot(x, m10["cum_net"],   color="green",       lw=0.9, ls="--", label="Net 10 bp")
    ax.plot(x, m20["cum_net"],   color="red",         lw=0.9, ls=":",  label="Net 20 bp")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Test-bar index (trading days)")
    ax.set_ylabel("Cumulative return (%)")
    low = " [LOW TRADES]" if r["n_trades"] < MIN_TRADES_FLAG else ""
    ax.set_title(f"{r['pair']}  ETF pairs{low}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    name = r["pair"].replace("/", "_")
    path = os.path.join(out_dir, f"{name}_etf_equity.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    saved: {path}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_only",      action="store_true",
                    help="Only run Phase 1 cointegration scan")
    ap.add_argument("--force_download", action="store_true",
                    help="Re-download data from yfinance even if cache exists")
    args = ap.parse_args()

    series = fetch_prices(force=args.force_download)
    if len(series) < 2:
        print("Need at least 2 tickers.  Check yfinance or data/etf/ cache.")
        return

    _, candidates = phase1_scan(series)

    if args.scan_only or not candidates:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    print(f"\n{'='*76}")
    print(f"  PHASE 2 — Walk-Forward: {TRAIN_MONTHS}m train / {TEST_MONTHS}m test / "
          f"{PURGE_BARS}-bar purge")
    print(f"  Cointegration re-tested per window; skips non-cointegrated windows")
    print(f"  Cost scenarios: {[f'{c*10_000:.0f}bp' for c in COST_SCENARIOS]} "
          f"per event (entry or exit)")
    print(f"{'='*76}\n")

    for c in candidates:
        r = walkforward_pair(series, c["A"], c["B"])
        if r:
            results.append(r)

    if not results:
        print("\nNo pairs completed the walk-forward successfully.")
        return

    print(f"\n{'='*76}")
    print(f"  DETAILED RESULTS")
    print(f"{'='*76}")
    surviving = []
    for r in results:
        print_pair_results(r)
        plot_equity(r, RESULTS_DIR)
        if any(r["scenarios"][cp]["net_ret_pct"] > 0
               and r["scenarios"][cp]["net_sharpe"] > 0
               for cp in COST_SCENARIOS):
            surviving.append(r["pair"])

    print(f"\n{'='*76}")
    print(f"  GRADUATION GATE  (n_trades >= {MIN_TRADES_FLAG}  AND  net return > 0)")
    print(f"{'='*76}")
    for r in results:
        ok_trades = r["n_trades"] >= MIN_TRADES_FLAG
        ok_net    = r["pair"] in surviving
        grad      = "GRADUATED" if (ok_trades and ok_net) else "REJECTED"
        reasons   = []
        if not ok_trades:
            reasons.append(f"only {r['n_trades']} trades")
        if not ok_net:
            reasons.append("negative net return")
        detail = ", ".join(reasons) if reasons else "all checks pass"
        print(f"  {r['pair']:<18}  {grad:<10}  ({detail})")

    print()


if __name__ == "__main__":
    main()
