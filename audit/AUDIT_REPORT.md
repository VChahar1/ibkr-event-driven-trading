# Backtest audit — `ibkr_bot`

Audit date 2026-10-03, code at commit `32b386d` (clean tree). Every number below comes from
`audit/run_audit.py` (→ `audit/results.json`) or `audit/pit_experiment.py` (→ `audit/pit_results.json`),
both run on this date. The original code was not modified: variants (extra lag, inverted/random
signals, cost changes, data fixes) are wrappers or scratch copies, and the copies were first checked
to reproduce the original engines exactly (`scratch_copies_identical`: both `true`).

Backtests audited:
- **Stock** (traded): `backtest_stocks.py` — S&P 100 12-1 momentum, top 10, SPY 200-day filter, 2007-06-01 → 2026-10-02.
  Headline: **CAGR 25.6%, Sharpe 1.05, max DD −30.9%**.
- **ETF** (research): `backtest.py --proxy` — dual momentum on SPY/EEM/TLT/GLD/BIL, 2008-05-30 → 2026-10-02.
  Headline: CAGR 7.4%, Sharpe 0.55, max DD −18.2%.
- **Pre-registered H1/H2** (`prereg_test.py`): rules exercised on **synthetic data only**. Their
  real-data result must be produced exactly once by `prereg_test.py`, so it was not run here.

## 1. Architecture

| File | Role in the backtests |
|---|---|
| `universe.py`, `universe/sp100.csv` | Stock universe: S&P 100 snapshot **as of 2026-10-03** |
| `data.py` | Yahoo adjusted closes (`load_stock_prices` keeps each stock's own history; `load_prices` for ETFs) |
| `stock_strategy.py` | Pure stock signal: `decide()` = SPY filter (replayed), 12-1 momentum, top-10 with top-20 buffer |
| `backtest_stocks.py` | Stock engine `run_monthly()`: USD NAV, signal on month-end close, trade next day's close, IBKR commissions + slippage, SPY and equal-weight benchmarks |
| `strategy.py`, `backtest.py` | ETF signal (`signal_frames`, `weights_from_signals`) and engine `run_backtest()`: daily, trade on the signal bar's close, flat bps costs |
| `prereg_test.py` | Pre-registered H1/H2 targets, simulation, bootstrap, deflated Sharpe |
| `analytics.py` | All metrics (`perf_stats`), drawdown, regimes, log parsing |
| `report.py` | HTML charts from the same result objects as the metrics |
| `audit/run_audit.py` | This audit (checklist §1–7); `audit/pit_experiment.py` measures survivorship bias |

## 2. Summary

| # | Check | Result | Evidence |
|---|---|---|---|
| 1.1 | Sorted, unique, one time zone | **PASS** | 5,220 × 102 stock panel and 4,868 × 5 ETF panel: sorted, unique, tz-naive dates |
| 1.2 | Missing values, no future fill | **PARTIAL** | Fill is `ffill(limit=5)` (past only), but `data.py:46` masks with `bfill()` (future info). Measured effect 0 (no interior or trailing gaps exist) |
| 1.3 | Split/dividend adjusted | **PASS** | Yahoo `auto_adjust=True` total-return closes; ratio signals are invariant to later re-adjustment |
| 1.4 | >20% daily moves checked | **PARTIAL** | 189 stock moves flagged; IBKR's own bars confirm 51 of the 52 it covers within 2 pts. **1 data error**: DELL 2018-12-24 (Yahoo −21.6% vs IBKR −1.7%, Yahoo history spliced before the 2018-12-28 listing) |
| 1.5 | Survivorship | **FAIL** (stocks) / PARTIAL (ETF) | Today's S&P 100 applied to 2007–2026. Same rules on the S&P 500: biased CAGR 32.7% / Sharpe 1.09 vs point-in-time 10.5% / 0.46 |
| 1.6 | Bar alignment | **PASS** | Single joined index; 0 misaligned rows; 0 interior NaN cells |
| 2.1 | Signal at t traded at t+1 | **PASS** (stocks) / **PARTIAL** (ETF) | Stocks: month-end signal, next-day-close fill. ETF: fills on the signal bar's own close (`backtest.py:38-46`) |
| 2.2 | Rolling stats past-only | **PASS** | Only `shift()` and trailing `rolling()`; no scalers, z-scores or models |
| 2.3 | Fundamentals/events by release date | **N/A** | Price data only |
| 2.4 | Truncation test | **PASS** | 8 cut dates (stocks), 5 (ETF), 3 (H1/H2 synthetic): max weight difference 0.0; first differing index: none |
| 2.5 | Future-perturbation test | **PASS** | 4 (stocks), 5 (ETF), 3 (H1/H2 synthetic) random-walk perturbations after t: max difference 0.0 |
| 3.1 | Fill price realistic | **PASS** (stocks) / **PARTIAL** (ETF) | Stocks: next close (live fills about 10:30 New York). ETF: signal close |
| 3.2 | Costs on every leg | **PASS** | 1,025 buys and 939 sells all charged; ETF cost = bps × Σ\|Δw\| |
| 3.3 | Commission, spread, slippage realistic | **PASS** (stocks) / PARTIAL (ETF) | IBKR fixed ($0.005/sh, $1 min, 1% max) + 5 bps; median order $2.3k. ETF: flat 10 bps |
| 3.4 | No unintended leverage | **PARTIAL** | Cash dips to **−0.85%** (gross exposure up to 100.85%): costs and skipped small trades aren't netted before buys |
| 3.5 | Borrow costs | **N/A** | Long-only |
| 3.6 | Derivatives handling | **N/A** | None |
| 4.1 | Lagged position × return, compounded | **PASS** | Always-long equals buy-and-hold to 4.1e-14 (stocks) and 0.0 (ETF) |
| 4.2 | Annualization | **PASS** | 252 (daily equities) |
| 4.3 | Independent metric recompute | **PASS** | CAGR, Sharpe, Sortino, max DD, Calmar recomputed from the equity curve: difference 0.000000 (stocks and ETF) |
| 4.4 | Benchmark | **PASS** | Stocks: SPY and equal-weight universe. ETF: buy & hold, equal weight, 60/40 |
| 5.1 | Always-long = buy-and-hold | **PASS** | 4.1e-14 / 0.0 |
| 5.2 | Zero position = flat | **PASS** | NAV range 0.0; max \|return\| 0.0 |
| 5.3 | Random signals | **PASS** (long-only reading) | Random 10-stock picks: net Sharpe 0.67 ± 0.08 (always in), 0.81 ± 0.09 (with the filter); costs cut ~0.09. Momentum 1.05 beats all 50 filtered random runs (best 1.02) |
| 5.4 | Inverted signal | **PASS** (long-only reading) | Bottom-10 momentum gross Sharpe 0.82 ≈ equal-weight 0.81, vs top-10 1.07. Inverted filter: 0.10 |
| 5.5 | Doubled costs | **PASS** | Stocks: Sharpe 1.054 → 1.039. ETF: 0.545 → 0.457 |
| 5.6 | One extra bar of lag | **PASS** | Stocks: Sharpe 1.054 → 1.057 (no collapse). ETF: 0.545 → 0.497 (lag 1), 0.540 (lag 2): non-monotonic, path noise |
| 6.1 | In-sample / out-of-sample | **PARTIAL** | No split. Stock parameters fixed from the literature before any run (`STOCK_STRATEGY.md`, commit cc9b5d5); ETF parameter history undocumented. Sub-periods: Sharpe 1.12 (2007–16), 1.03 (2017–26) |
| 6.2 | Walk-forward | **FAIL** | Not implemented (no parameters are fitted, so it would equal the full backtest) |
| 6.3 | Parameter sensitivity | **PASS** (stocks) / UNVERIFIED (ETF) | Stocks: broad plateau, Sharpe 0.92–1.18; chosen point 1.05 is mid-plateau. ETF grid deliberately not run (see §4) |
| 6.4 | ≥ 30 trades | **PASS** | 233 rebalances and 1,964 orders (stocks); 192 rebalances (ETF) |
| 6.5 | Dominated by few trades or periods | **FAIL** | Top 5 stocks = **33%** of total contribution. Without them: CAGR 25.6% → 18.6%. 2026 YTD alone +110.7%; CAGR excluding 2026: 21.8% |
| 7.1 | Plots use the metrics' series | **PASS** | Plotted final NAV $2,040,750 vs backtest $2,040,752 (6-sig-fig rounding); plotted max DD −30.92% = metric; CAGR tile 25.6% = metric |
| 7.2 | Equity and drawdown from net returns | **PASS** | Plotted NAV = net run ($2.04M), not gross ($2.19M) |

**Not verified at all:** S&P 100 point-in-time membership (no free source; the S&P 500 was used
instead); prices of the 22% of point-in-time members missing from Yahoo; live fills vs backtest
(only 5 paper fills so far, all ETFs).

## 3. Critical issues (inflate results)

### C1. Survivorship / look-ahead in the stock universe — `backtest_stocks.py:184-189`
```python
uni = load_universe()  # S&P 100 as of 2026-10-03
cfg = stock_config(conf, uni)  # ... used for every date since 2007
res = run_all(prices, cash_ret, cfg)
```
The universe is today's index. Stocks enter the index because they rose; losers leave or are
delisted. Momentum buys recent winners, so it harvests exactly the stocks whose later inclusion is
known only in hindsight. The time-series tests (§2.4–2.5) cannot catch this: the look-ahead is in
*which* columns exist, not in how they are used.

**Test output** (`audit/pit_experiment.py`): the same rules and costs on the S&P 500, with
point-in-time membership from github.com/fja05680/sp500 (full member list at each change date, 1996
→ 2026-08-18), June 2007 → October 2026:

| S&P 500 universe | CAGR | Sharpe | Max DD | Most-picked |
|---|---|---|---|---|
| Momentum, today's members (biased) | **32.7%** | **1.09** | −38.4% | NVDA, CVNA, NFLX, MU, AMD, SMCI, GNRC, TPL |
| Momentum, point-in-time | **10.5%** | **0.46** | −37.7% | NVDA, MU, NFLX, AMD, WDC, NRG, STX, AVGO |
| Equal weight, biased | 14.3% | 0.68 | −52.7% | |
| Equal weight, point-in-time | 9.8% | 0.48 | −58.8% | |
| SPY | 10.7% | 0.54 | −55.2% | |

The biased run "owned" CVNA, SMCI, GNRC and TPL during the run-ups that later got them into the
index. On point-in-time data, momentum's edge over equal-weight falls from +0.41 to −0.02 Sharpe, and
both trail SPY. Even this is optimistic: on average only **77.6%** of point-in-time members still
have Yahoo prices (55.1% in 2006); the missing ones are mostly delisted losers.

**Proposed fix** (needs a membership-history file; S&P 100 history is not freely available, so the
honest option is to research on the S&P 500 point-in-time and trade the S&P 100):
```diff
--- a/backtest_stocks.py
+++ b/backtest_stocks.py
-def strategy_fn(cfg: StockConfig) -> DecideFn:
+def strategy_fn(cfg: StockConfig,
+                members_at: Callable[[pd.Timestamp], set[str]] | None = None) -> DecideFn:
     def fn(hist: pd.DataFrame, held: set[str], today: dt.date) -> tuple[pd.Series, dict]:
+        if members_at is not None:                       # point-in-time universe
+            members = members_at(hist.index[-1])
+            hist = hist[[c for c in hist.columns if c in members or c == cfg.market]]
+            held = held & members
         d = decide(hist, cfg, held, today)
```
…plus a loader for `sp500_hist.csv` with prices for every ticker ever in the index (as in
`audit/pit_experiment.py`), and a data source that includes delisted stocks (e.g. CRSP/Norgate) to
close the remaining gap.

### C2. ETF backtest trades on the signal bar's own close — `backtest.py:38-46`
```python
if i >= cfg.warmup:
    tgt = weights_from_signals(mom.iloc[i], ratio.iloc[i], cfg, held)  # uses close[i]
    ...
    w = tgt.copy()  # earns from i+1
```
The signal uses close *i* and the trade also fills at close *i*, which needs a market-on-close order
placed after the close is known. The live bot trades one or two days later. The docstring calls this
"slightly optimistic"; measured:

| ETF dual momentum | CAGR | Sharpe | Max DD |
|---|---|---|---|
| As coded (same-bar fill) | 7.4% | 0.545 | −18.2% |
| Signal from t−1, fill at t (fix) | 6.8% | 0.497 | −22.5% |
| Two bars of lag | 7.4% | 0.540 | −21.7% |

The non-monotonic lag-2 result says most of this is path noise rather than systematic leakage, but
the drawdown is consistently worse once the fill is realistic.
```diff
--- a/backtest.py
+++ b/backtest.py
     mom, ratio = signal_frames(prices, cfg)
+    mom, ratio = mom.shift(1), ratio.shift(1)      # decide on close t-1, fill at close t
 ...
-        if i >= cfg.warmup:
+        if i >= cfg.warmup + 1:
```
**The same same-bar convention is written into `HYPOTHESES.md` for H1/H2.** It is not locked yet
(no `HYPOTHESES.lock`), so it can still be amended to "signal t, trade t+1" before locking without
breaking the pre-registration. That is your decision.

## 4. Lower-severity issues and suggestions

1. **Yahoo data error: DELL.** Yahoo's series starts 2016-08-17, but Dell re-listed on 2018-12-28; the earlier history is spliced from another security, with a fake −21.6% move on 2018-12-24. The strategy held DELL from May 2019 on a signal that reached into the spliced period. Removing the pre-listing data: Sharpe 1.0542 → 1.0546 (negligible). *Fix:* store real listing dates in `universe/sp100.csv` and blank prices before them; keep cross-checking flagged moves against IBKR (`run_audit.py --ibkr`).
2. **Future-information mask in the loader** (`data.py:46`, `df.where(df.bfill().notna())`). It uses later data to decide whether a forward-filled value survives. No effect today (0 interior gaps, 0 trailing gaps), but it would leak with delisted stocks. *Fix:* delete the line and end each series at an explicit delisting date.
3. **Slight leverage** (`backtest_stocks.py:113`). Cash dips to −0.85% because costs and skipped sub-$100 sells aren't netted before buys. The live bot cannot do this (`plan_orders` scales buys to cash). *Fix:* scale the buys by the excess before setting holdings:
   ```diff
   -            h[doable] = target[doable]
   +            h_new = np.where(doable, target, h)
   +            excess = h_new.sum() + day_cost - nav            # never exceed the book's value
   +            buys = doable & (delta > 0)
   +            if excess > 0 and buys.any():
   +                h_new[buys] -= excess * delta[buys] / delta[buys].sum()
   +            h = h_new
                cash = nav - h.sum() - day_cost
   ```
4. **Drawdown ignores the starting capital as a peak** (`analytics.py:34-36`). A loss in the first days would be understated. No effect on the current numbers (independent recompute: 0.0 difference). *Fix:* `return eq / eq.cummax().clip(lower=1.0) - 1`.
5. **Concentration.** One third of total return comes from five stocks, and current picks are 6/10 semiconductors. Consider a sector cap; report results with and without the top contributors.
6. **ETF parameter provenance.** `trend_buffer`, `mom_margin` and `band` have no documented origin, which could mean data snooping. The ETF parameter grid was deliberately **not** run: `HYPOTHESES.md` declares 3 trials for the deflated-Sharpe correction, and viewing a dual-momentum grid before locking would add uncounted trials. If you want it, raise the trial count in `HYPOTHESES.md` first.
7. **No out-of-sample or walk-forward.** With fixed literature parameters there is nothing to walk forward; the real out-of-sample tests are point-in-time data (C1) and forward paper trading.
8. **ETF cost model** is a flat 10 bps with no commission minimums or UCITS spreads. Doubling it costs 0.09 Sharpe, so it matters here, unlike in the stock backtest.
9. **Reproducibility.** Yahoo re-adjusts history over time. The cached CSV pins today's run; record a hash of the price file in every backtest output.
10. **Live vs backtest gaps** (already documented): fills about 10:30 New York vs next close; whole shares; unaffordable picks replaced.

## 5. Before / after estimate

| Backtest | Fix | CAGR | Sharpe | Max DD | Basis |
|---|---|---|---|---|---|
| Stock, S&P 100 (headline) | — | 25.6% | 1.05 | −30.9% | measured |
| | C1 survivorship | **≈ 8–12%** | **≈ 0.4–0.6** | ≈ −30 to −40% | **extrapolated** from the measured S&P 500 experiment (32.7% / 1.09 → 10.5% / 0.46), and still optimistic because 22% of point-in-time members lack prices |
| | DELL data error | 25.6% | 1.055 | −30.9% | measured (no effect) |
| | Leverage fix (§4.3) | ≈ unchanged | ≈ unchanged | ≈ unchanged | at most −0.85% exposure; not re-run |
| ETF dual momentum | — | 7.4% | 0.545 | −18.2% | measured |
| | C2 realistic fill | 6.8% | 0.497 | −22.5% | measured |

## 6. Verdict

**The stock backtest's headline (25.6% CAGR, Sharpe 1.05) cannot be trusted as evidence of an
edge.** The *code* holds up well:
- no time-series look-ahead (truncation and perturbation differences exactly 0);
- correct accounting (always-long equals buy-and-hold to 1e-14);
- metrics confirmed by an independent recompute;
- charts consistent with the metrics;
- costs realistic and on every leg.

The *universe* is the problem. On point-in-time data, the same rules on the S&P 500 return about 10.5%
a year with Sharpe 0.46, slightly *below* SPY (0.54) and equal-weight (0.48). In the biased data, a
third of the return comes from five hindsight winners. The honest current estimate is **no
demonstrated edge over the index**. The strategy remains fine as the bot's plumbing workload and a
forward paper test.

**The ETF backtest is a sound implementation with a modestly optimistic fill.** Its edge is weak:
- Sharpe 0.50 with a realistic fill, vs 0.60 for buy-and-hold SPY over the same period;
- random daily picks among the same ETFs reach 0.49 before costs.

**Next steps, in priority order:**
1. Research on point-in-time S&P 500 membership with delisted prices (C1).
2. Fix the ETF fill timing (C2), and decide on the H1/H2 timing rule *before* locking `HYPOTHESES.md`.
3. Apply the small fixes in §4.
