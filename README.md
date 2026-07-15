# AI Forex / Statistical-Arbitrage Research Pipeline

A personal research project exploring whether machine-learning and statistical-arbitrage strategies can produce a net-positive edge in FX and ETF markets after realistic transaction costs. The short answer, documented here in full, is no — not with the frameworks tested.

---

## 1  Project Summary

The pipeline progresses through three distinct research phases:

1. **Deep Reinforcement Learning (DRL)** — PPO/A2C agents trained on multi-timeframe FX features with custom reward shaping. Several iterations, each attempting to fix reward-signal and observation-space defects identified in the prior run.

2. **XGBoost Directional Classification** — walk-forward backtester predicting next-N-bar direction on 14 FX pairs at H1 resolution. More interpretable than DRL; easier to reconcile against live execution.

3. **Statistical Arbitrage** — Engle-Granger cointegration scan across FX H1, FX D1, and ETF daily universes. Pre-committed graduation gate: requires both sufficient OOS trade count and net-positive return after costs to advance.

Each phase was built with a live MT5 paper-trading bridge so findings could be validated outside the backtest.

---

## 2  Architecture

```
MT5 / yfinance
     │
     ▼
01_mt5_multiframe_ingestion.py   ← pulls H1 + D1 OHLCV from MT5
02_multiframe_feature_engineering.py  ← FracDiff, RSI, ATR, time
     │                                    cyclicals, vol-regime, z-scores
     ▼
┌────────────────────────┐   ┌──────────────────────────────────┐
│  DRL path              │   │  XGBoost path                    │
│  drl_training_pro.py   │   │  xgb_walkforward.py              │
│  backtester_pro.py     │   │  12m train / 1m test / 24-bar    │
│  PPO / A2C (SB3)       │   │  purge; costs shown gross+net    │
└────────┬───────────────┘   └────────────┬─────────────────────┘
         │                                │
         └──────────┬─────────────────────┘
                    ▼
         live_audusd_xgb.py          ← MT5 demo paper-trader
         10_live_trader.py           ← DRL variant
         reconcile_features.py       ← offline live/backtest feature diff

┌─────────────────────────────────────────────────────────────┐
│  Statistical Arbitrage                                       │
│  pairs_walkforward.py      ← FX H1, 91 pairs               │
│  pairs_walkforward_d1.py   ← FX D1, 91 pairs               │
│  pairs_walkforward_etf.py  ← ETF daily, 66 pairs, yfinance │
│  All use: Engle-Granger + rolling OLS hedge + z-score entry │
│  Walk-forward: 36m train / 6m test / 5-bar purge           │
└─────────────────────────────────────────────────────────────┘
```

**Key files:**

| File | Purpose |
|---|---|
| `src/xgb_walkforward.py` | XGBoost WFO harness; reports gross and net side-by-side across cost scenarios |
| `src/live_audusd_xgb.py` | MT5 AUDUSD paper trader; reconnect loop, filling-mode auto-detect, min-hold guard |
| `src/reconcile_features.py` | Compares live bar features against offline-recomputed values; flags drift |
| `src/pairs_walkforward_etf.py` | ETF scan; downloads via yfinance, caches to `data/etf/`, % cost model |
| `src/backtester_pro.py` | DRL backtester with fixed episode-start and purge-aware evaluation |

---

## 3  Engineering Issues Found and Fixed

These are listed because they each produced misleading results before the fix. None are unusual; they are the standard failure modes of trading-system research.

### 3.1  Random-start backtest invalidating DRL results

Early DRL episodes started at a random bar in the dataset. This inflated apparent performance by letting the model see different start conditions across episodes, mixing in-sample and OOS data. Fixed by requiring episodes to start strictly after the training cutoff.

### 3.2  Macro-data lookahead

A feature derived from macro releases (economic calendar data) was computed using the release *value* at bar time, before the release was actually published. The feature column existed in the dataset because the ingestion script back-filled. Removed from the feature set.

### 3.3  PPO qf/vf misconfiguration

The SB3 PPO actor/critic shared the wrong policy-kwargs key (`qf_kwargs` instead of `vf_coef`), silently falling back to defaults. Reward variance was also unscaled, making the value function diverge. Fixed by explicit `vf_coef` and normalising rewards to unit variance before the critic update.

### 3.4  FLAT-signal exit mismatch (live vs backtest)

The backtest `_apply_min_hold()` closes a position on *any* signal change once the minimum hold expires, including a change to FLAT (signal = 0). The live trader had:

```python
if can_exit and signal != 0 and signal != pos_dir:   # wrong
```

The `signal != 0` guard prevented the live trader from closing on a FLAT signal. This caused the live system to hold through extended losing periods that the backtest would have exited. Fixed to:

```python
if can_exit and signal != pos_dir:   # matches backtest
```

Commit: `d69ac39`.

### 3.5  Stale-bar guard missing

The live polling loop could, on a slow tick, fetch the same closed bar twice and fire a trade signal twice on the same candle. Fixed by tracking `last_processed_bar_time` and skipping the signal loop if the fetched bar timestamp matches.

### 3.6  Broker timezone offset

MT5 server timestamps were in broker local time (UTC+2/3 DST). The session filter was written assuming UTC, so London/NY overlap filtering fired 2–3 hours early. Fixed by detecting server offset from the broker's `symbol_info_tick().time` vs `datetime.utcnow()` at startup.

### 3.7  MT5 filling-mode rejection (retcode 10030)

The order request hardcoded `ORDER_FILLING_IOC`. Some demo brokers require `ORDER_FILLING_FOK` or `ORDER_FILLING_RETURN`. Fixed with a `_resolve_filling()` helper that reads `symbol_info(SYMBOL).filling_mode` bitmask and selects the best supported mode at startup, logging the choice once.

---

## 4  Research Findings

### 4.1  XGBoost directional (H1, AUDUSD representative)

Walk-forward: 12-month train / 1-month test / 24-bar purge, 30 randomised initialisation runs to guard against lucky seeds.

| Metric | Value |
|---|---|
| Gross Sharpe (annualised, H1) | ~1.0 |
| Gross edge per bar | ~0.43 pips |
| Net return at 1.8 pip RT cost | ~0 |
| Net return at 1.0 pip RT cost | marginally positive |
| Net Sharpe at any realistic spread | < 0.3 |

The model learns a real signal — the gross equity curve is consistently upward-sloping across OOS windows and across 13 FX pairs. The problem is that H1 mean-reversion signals are small: the average winning trade is only a few pips, and the bid-ask spread plus slippage consumes the entire edge at realistic retail costs.

A limit-order variant (passive fills at 0.5 pip inside the spread) improves the picture but requires reliable fill rates that cannot be guaranteed in an H1 live system.

**Conclusion:** The signal exists. It is not large enough to survive realistic execution costs at H1 on retail spreads.

### 4.2  FX statistical arbitrage (H1 and D1)

Universe: 14 instruments → 91 pair combinations. Engle-Granger cointegration on the first 60% of aligned history; candidate gate: p < 0.05 AND 12h ≤ half-life ≤ 500h (H1) or 5d ≤ HL ≤ 120d (D1).

**H1 result:** 0 of 91 pairs passed. Every pair that achieves p < 0.05 has a half-life of 1,000–4,000 hours — cointegration exists, but only at a regime scale of months to years. At that speed the spread cannot be traded profitably with per-bar holding costs.

**D1 result:** 8 of 91 pairs passed Phase 1 (including EURUSD/AUDUSD, USDCHF/AUDCAD, AUDCAD/EURAUD). Walk-forward on those 8 pairs produced 0–9 OOS trades each over the 13-year backtest period. The graduation gate requires ≥ 30 trades for the statistics to be interpretable. No pair reached that threshold.

**Root cause:** The cointegration found on the training window does not persist in OOS windows. Across 37 six-month walk-forward windows, cointegration held (p < 0.05) in fewer than 15% of periods for any pair tested.

**Conclusion:** FX cointegration is episodic, not structural. The Engle-Granger framework cannot reliably identify pairs that will be cointegrated in the next test period.

### 4.3  ETF statistical arbitrage (daily, yfinance)

Universe: 12 liquid US ETFs (SPY, QQQ, IWM, TLT, IEF, GLD, SLV, XLE, XLF, XLK, XLU, XLB) → 66 combinations. Same walk-forward structure as FX D1. Cost model: fraction of portfolio notional (A + |β| × B) per trade event, not pips.

**Phase 1:** 3 pairs passed — QQQ/XLU (p = 0.045, HL = 115d), XLK/XLU (p = 0.027, HL = 99d), XLU/XLB (p = 0.019, HL = 84d). All three involve XLU (Utilities), which is rate-sensitive and moves differently from growth/tech and cyclicals.

**Phase 2 walk-forward:**

| Pair | OOS trades | Coint windows | Gross ret | Net ret (5 bp) |
|---|---|---|---|---|
| QQQ/XLU | 1 | 3% of 37 | −11.1% | −11.1% |
| XLK/XLU | 3 | 8% of 37 | −17.2% | −17.3% |
| XLU/XLB | 4 | 5% of 37 | −11.7% | −12.3% |

All three rejected at the graduation gate (< 30 trades, negative net return).

The cointegration found in the 2004–2018 training window does not hold in 92–97% of subsequent 6-month periods. When the system trades, it is almost always entering a spread that does not mean-revert on the expected timescale.

**Conclusion:** ETF pairs cointegration in the standard Engle-Granger framework is as unstable as FX pairs cointegration. The same failure mode — in-sample signal that does not persist OOS — applies across asset classes.

---

## 5  Methodology Lessons

**Pre-committed graduation gates.** Setting acceptance criteria (≥ 30 trades AND net return > 0) before running the backtest prevents post-hoc rationalisation of marginal results. If the gate is defined after seeing the numbers, it will be moved to let promising-looking strategies through.

**Gross and net reporting side-by-side.** Every table in this codebase shows gross returns alongside net returns at multiple cost scenarios. A strategy with a genuine gross edge but negative net return is not a usable strategy — but it is a research finding (the signal exists; execution is the bottleneck).

**Walk-forward with purge gaps.** A 24-bar (H1) or 5-bar (D1) purge between training and test windows prevents labels from leaking across the boundary. Without the purge, the model implicitly trains on the outcome of the first OOS bars.

**Live/backtest feature reconciliation.** `reconcile_features.py` recomputes every feature offline using the same bar data the live system used, then diffs the values. This catches silent divergences (e.g., floating-point differences in pandas_ta vs manual Wilder EMA, timestamp misalignment) before they compound into strategy drift.

**Cointegration is not the same as tradeable mean-reversion.** Statistical significance at a given training window (p < 0.05) does not imply the relationship persists. A half-life of 80–120 days means a single failed trade can remain open for months. Regime tests should be rerun frequently, and non-cointegrated windows should go flat.

---

## 6  Usage

### Prerequisites

```
conda activate trading_bot_env
pip install xgboost statsmodels yfinance MetaTrader5
```

### XGBoost walk-forward

```bash
# Single ticker
python src/xgb_walkforward.py --ticker AUDUSD

# Custom parameters
python src/xgb_walkforward.py --ticker AUDUSD \
    --forward_bars 4 --threshold 0.65 --min_hold 12

# All majors
python src/xgb_walkforward.py \
    --tickers EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDJPY,XAUUSD \
    --forward_bars 4 --threshold 0.65 --min_hold 12
```

### Pairs scan (FX)

```bash
python src/pairs_walkforward.py --scan_only   # H1, Phase 1 only
python src/pairs_walkforward_d1.py            # D1, full run
```

### Pairs scan (ETF)

```bash
python src/pairs_walkforward_etf.py --scan_only      # Phase 1 only
python src/pairs_walkforward_etf.py                  # full run
python src/pairs_walkforward_etf.py --force_download # refresh yfinance cache
```

ETF price data is cached to `data/etf/` on first download. Subsequent runs use the cache.

### Live AUDUSD paper trader

```bash
python src/live_audusd_xgb.py
```

Requires MT5 terminal open and connected to a demo account. The trader reconnects automatically and skips signal logic on disconnects.

### Feature reconciliation

```bash
python src/reconcile_features.py
```

Reads the last N bars from MT5 and from the offline CSV, recomputes all features both ways, and prints a diff table. Any row where relative error exceeds 1 × 10⁻⁵ is flagged FAIL.

---

## 7  Repository Layout

```
data/              MT5 H1 and D1 CSVs (EURUSD_H1.csv etc.)
data/etf/          yfinance daily cache (SPY_D1.csv etc.)
results/           Walk-forward output: equity PNGs, window CSVs
src/               All source files (see table in §2)
```

---

*No live capital was risked. All results are from MT5 demo accounts or historical backtests.*
