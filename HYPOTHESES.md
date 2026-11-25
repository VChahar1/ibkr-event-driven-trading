# Pre-registered hypotheses — written 2026-09-29, before any test was run

Rules below are frozen. `python prereg_test.py --lock` records a SHA-256 hash of this
file; the test refuses to run if the file changes afterwards, and refuses a second run
unless a reason is given (which is then printed in the results as a re-run).

## Context and what has already been seen
- Already observed (NOT part of this test): dual momentum strategy, buy & hold SPY,
  static equal-weight and static 60/40 statistics on 2007–2026 proxy data.
- Not yet observed: any result for the two strategies below.
- Trials counted for multiple-testing purposes: 3 (dual momentum + H1 + H2).

## Common setup (identical for H1, H2 and the benchmark)
| Item | Rule |
|---|---|
| Data | Yahoo Finance adjusted daily closes: SPY (equity, "EQ"), TLT (bonds, "BD"), BIL (T-bills, "CASH"); download start 2006-01-01, all dates up to the run date |
| Common sample | Dates where all three exist (BIL starts 2007-05-30) |
| Warm-up | First 252 trading days used only for signals; evaluation starts on day 253 for every series |
| Timing | Signal from the close of day t; new weights earn returns from day t+1 |
| Rebalancing | Trade to target when any weight differs from target by more than 0.05 (absolute) |
| Costs | 10 bps × turnover (sum of absolute weight changes) |
| Leverage | None: weights ≥ 0 and sum to 1; unused weight goes to CASH |
| Benchmark | Static 60/40 SPY/TLT, rebalanced on the first trading day of each month, 10 bps |

## H1 — Volatility-targeted 60/40
Rule, evaluated every day:
1. Σ_t = sample covariance of the last 63 daily returns of EQ and BD.
2. σ_t = sqrt(252 · wᵀ Σ_t w) with w = (0.6, 0.4).
3. scale_t = min(1, 0.10 / σ_t).
4. Target: EQ 0.6·scale_t, BD 0.4·scale_t, CASH 1 − scale_t.

Hypothesis: scaling exposure down when recent volatility is high improves risk-adjusted
return vs static 60/40 (Moreira & Muir 2017; Harvey et al. 2018).

## H2 — 60/40 with a trend crash filter
Rule, evaluated every day, for each sleeve independently (EQ at 0.6, BD at 0.4):
1. ratio = close / 200-day simple moving average.
2. State starts "in" if ratio > 1 on the first evaluation day, otherwise "out".
3. "in" → switch to "out" when ratio < 0.98; "out" → switch to "in" when ratio > 1.02.
4. Target: sleeve weight when "in", moved to CASH when "out".

Hypothesis: holding each sleeve only while it is above its long-term trend reduces
crash losses enough to improve risk-adjusted return vs static 60/40 (Faber 2007).

## Primary metric and decision rule (per hypothesis)
- Primary metric: annualised Sharpe ratio of daily returns in excess of CASH (BIL).
- ΔSharpe = Sharpe(strategy) − Sharpe(static 60/40), on the same evaluation dates.
- Uncertainty: paired circular block bootstrap of daily returns, block length 21 days,
  5,000 resamples, random seed 20260929.
- Two hypotheses → Bonferroni: each tested at one-sided α = 0.025.
- **SUPPORTED** only if BOTH hold:
  (a) the 2.5th percentile of bootstrap ΔSharpe is > 0, and
  (b) the strategy's maximum drawdown is smaller in magnitude than static 60/40's.
- Otherwise **NOT SUPPORTED**. Results are reported either way; no parameter changes after running.

## Secondary results (descriptive only — they cannot change the decision)
- CAGR, volatility, maximum drawdown, Calmar, turnover per year.
- Same statistics at 20 bps costs.
- Sub-periods: evaluation start – 2016-12-31 and 2017-01-01 – end.
- Deflated Sharpe ratio of each strategy with N = 3 trials.

## Out-of-sample follow-up
Whatever the result, no strategy replaces the paper bot's placeholder until it has also
been paper-traded forward for at least 3 months from the date it is adopted.
