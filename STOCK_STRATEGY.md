# Stock strategy: S&P 100 momentum with a market filter

Written 2026-10-03, **before any backtest of this strategy was run**. The commit that adds this
file is the timestamp. Rules and parameters are taken from the literature, not fitted. The
backtest is run once; any later change is logged under *Deviations* with the reason.

## Purpose and status
- Replaces the ETF dual-momentum placeholder as the strategy the paper bot trades.
- It is a **new trial** (stock family, trial #1), separate from the pre-registered H1/H2 tests
  in `HYPOTHESES.md`, which are unaffected.
- The backtest below is **descriptive**: the universe is survivorship-biased (see *Known biases*),
  so it cannot establish an edge. Forward paper trading is the real test.

## Rules
| Item | Rule |
|---|---|
| Universe | `universe/sp100.csv`: the S&P 100 as of 2026-10-03, one share class per company (100 stocks) |
| Data | Yahoo Finance adjusted daily closes; SPY for the market filter; BIL as the T-bill / cash return |
| Eligibility | A stock needs a close 252 and 21 trading days before the signal date (≥ 253 days of history) |
| Signal | 12-1 momentum: close[t−21] / close[t−252] − 1 (skips the most recent month) |
| Selection | Rank eligible stocks by momentum. Keep a current holding while it ranks ≤ 20; fill the remaining slots, up to 10, from the top of the ranking |
| Weights | 10% of the portfolio per stock (equal weight); empty slots stay in cash |
| Market filter | ratio = SPY close / SPY 200-day SMA. Starts "on" if ratio > 1 at the first evaluation. On → off when ratio < 0.98; off → on when ratio > 1.02. When off: 100% cash, no stocks |
| Rebalance | Monthly. Signal from the last close of each month. At every rebalance, the selected stocks are reset to equal weight |
| Minimum trade | Trades below $100 are skipped |

## Backtest conventions
| Item | Convention |
|---|---|
| Sample | Download from 2006-01-01. Evaluation starts at the first month-end where SPY has a 200-day SMA, BIL has data, and at least 50 stocks are eligible; it runs to the latest close |
| Execution | At the **next trading day's close** after the signal (live trades about 10:30 New York on that day) |
| Capital | NAV starts at $25,000 and compounds; cash earns the BIL return |
| Commission | IBKR fixed: $0.005 per share, minimum $1.00, maximum 1% of trade value, per order; shares ≈ value / adjusted close |
| Slippage | 5 bps of traded value |
| Benchmarks | SPY buy & hold; equal-weight portfolio of all eligible universe stocks, rebalanced monthly with the same costs |
| Metrics | CAGR, volatility, Sharpe and Sortino (excess of BIL), max drawdown, Calmar, turnover, cost drag, stress windows (2008, 2020, 2022) |

## Known biases (declared before seeing results)
- **Survivorship / look-ahead in the universe.** Today's S&P 100 contains stocks that were added
  *because* they rose (recent examples include PLTR, GEV, DELL, SNDK) and excludes those that
  fell out or were delisted. Momentum buys recent winners, so this bias flatters it more than
  most strategies. Absolute returns are an **upper bound**.
- **Momentum vs equal-weight** uses the same biased universe, so the *difference* between them is
  more informative than either level. It is still not clean: momentum may concentrate in exactly
  the names whose inclusion was decided ex post.
- **Researcher knowledge.** The author knows in general terms how large US tech stocks performed
  over this period. Fixing the rules before running, and not tuning afterwards, limits but cannot
  remove this contamination.
- **Short histories.** Recent IPOs and spin-offs enter only after 253 days, so the early sample
  holds fewer names.
- Adjusted prices make per-share commissions approximate (split-adjusted prices overstate share
  counts early on; the 1% cap bounds this).

## Live vs backtest differences (by design)
- Live executes about 10:30 New York on the first trading day after the signal; the backtest uses that day's close.
- Live trades whole shares; a stock priced above its slot (~$2.4k at $25k / 10) is skipped and the next-ranked name used.
- The first live rebalance happens whenever the bot first runs in a month, not necessarily on the first trading day.
- Live cash earns IBKR's paper interest (if any), not BIL.

## Deviations
- 2026-11-05, `feature/kafka-redis` only (the event-driven path in `eventbot/`; the daily bot is unchanged).
  No rule or parameter changes: picks, weights, sizing and the 10:30/15:00 New York schedule are the same code.
  Two execution details differ. Sizing prices come from the latest streamed tick rather than a one-shot
  `reqTickers` snapshot, using the same `marketPrice()`-else-close rule and the same delayed data. A new
  placement-side cap (`max_position_notional_usd`, $5,000 per symbol) can refuse an order; at a ~$2.4k slot
  no planned buy should reach it. Decided before any results exist from this path.
