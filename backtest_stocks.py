#!/usr/bin/env python3
"backtest_stocks.py - monthly backtest of stock_strategy with IBKR-style costs."

import argparse
import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import perf_stats
from stock_strategy import StockConfig, decide, eligible_count, market_states, momentum, month_ends

ROOT = Path(__file__).parent
CACHE = ROOT / "data_cache" / "stocks_yahoo.csv"
START = "2006-01-01"
CAPITAL = 25_000.0
MIN_TRADE = 100.0

DecideFn = Callable[[pd.DataFrame, set[str], dt.date], tuple[pd.Series, dict]]


@dataclass(frozen=True)
class CostModel:
    """IBKR fixed pricing (US stocks) plus slippage, per order."""

    per_share: float = 0.005
    min_order: float = 1.0
    max_pct: float = 0.01
    slippage_bps: float = 5.0

    def order_cost(self, value: float, price: float) -> float:
        if value <= 0 or not price > 0:
            return 0.0
        commission = min(max(self.min_order, self.per_share * value / price), self.max_pct * value)
        return commission + value * self.slippage_bps / 1e4


DEFAULT_COSTS = CostModel()


@dataclass
class StockBacktest:
    returns: pd.Series
    nav: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def evaluation_start(prices: pd.DataFrame, cash_ret: pd.Series, cfg: StockConfig) -> pd.Timestamp:
    """Execution day after the first month-end where the SPY filter, the cash return and at
    least cfg.min_eligible stock signals all exist (STOCK_STRATEGY.md)."""
    ends = month_ends(prices.index)
    states = market_states(prices[cfg.market], ends, cfg)
    n_ok = eligible_count(prices, cfg)
    cash_ok = cash_ret.notna()
    for s in ends:
        if s in states.index and cash_ok.get(s, False) and n_ok[s] >= cfg.min_eligible:
            nxt = prices.index[prices.index > s]
            if len(nxt):
                return nxt[0]
    raise ValueError("no month-end satisfies the start conditions")


def run_monthly(
    prices: pd.DataFrame,
    cash_ret: pd.Series,
    decide_fn: DecideFn,
    symbols: list[str],
    start: pd.Timestamp,
    costs: CostModel = DEFAULT_COSTS,
    capital: float = CAPITAL,
    min_trade: float = MIN_TRADE,
) -> StockBacktest:
    """Daily mark-to-market; at the close of the trading day after each month-end, trade to
    decide_fn's weights (signal on the month-end close). Cash earns cash_ret."""
    px = prices[symbols]
    days = prices.index[prices.index >= start]
    signal_for = {}
    for s in month_ends(prices.index):
        nxt = prices.index[prices.index > s]
        if len(nxt) and nxt[0] >= start:
            signal_for[nxt[0]] = s
    h = np.zeros(len(symbols))
    cash = capital
    prev: np.ndarray | None = None
    navs, wrows, cost_days, trades, decisions = [], [], [], [], []
    for d in days:
        p = px.loc[d].to_numpy(dtype=float)
        if prev is not None:
            ok = np.isfinite(p) & np.isfinite(prev) & (prev > 0)
            h = h * np.where(ok, p / np.where(ok, prev, 1.0), 1.0)
            r = cash_ret.get(d, 0.0)
            cash *= 1 + (0.0 if pd.isna(r) else float(r))
        day_cost = 0.0
        if d in signal_for:
            held = {symbols[i] for i in np.flatnonzero(h > 0)}
            w, info = decide_fn(prices.loc[: signal_for[d]], held, d.date())
            nav = h.sum() + cash
            target = w.reindex(symbols).fillna(0.0).to_numpy(dtype=float) * nav
            delta = target - h
            doable = np.isfinite(p) & (np.abs(delta) >= min_trade)
            for i in np.flatnonzero(doable):
                c = costs.order_cost(abs(delta[i]), p[i])
                day_cost += c
                trades.append(
                    {
                        "date": d,
                        "symbol": symbols[i],
                        "side": "BUY" if delta[i] > 0 else "SELL",
                        "value": abs(delta[i]),
                        "cost": c,
                    }
                )
            h[doable] = target[doable]
            cash = nav - h.sum() - day_cost
            decisions.append({"signal_date": signal_for[d], "exec_date": d, **info})
        nav = h.sum() + cash
        navs.append(nav)
        wrows.append(np.append(h / nav, cash / nav))
        cost_days.append(day_cost)
        prev = np.where(np.isfinite(p), p, prev if prev is not None else p)
    nav_s = pd.Series(navs, index=days)
    rets = nav_s.pct_change()
    rets.iloc[0] = nav_s.iloc[0] / capital - 1
    return StockBacktest(
        returns=rets,
        nav=nav_s,
        weights=pd.DataFrame(wrows, index=days, columns=[*symbols, "CASH"]),
        trades=pd.DataFrame(trades, columns=["date", "symbol", "side", "value", "cost"]),
        decisions=pd.DataFrame(decisions),
        costs=pd.Series(cost_days, index=days),
    )


def strategy_fn(cfg: StockConfig) -> DecideFn:
    def fn(hist: pd.DataFrame, held: set[str], today: dt.date) -> tuple[pd.Series, dict]:
        d = decide(hist, cfg, held, today)
        return d.weights, {"market_on": d.market_on, "picks": d.picks, "n_eligible": len(d.ranks)}

    return fn


def equal_weight_fn(cfg: StockConfig) -> DecideFn:
    """Benchmark: every eligible universe stock (same eligibility as the strategy), 1/n each."""

    def fn(hist: pd.DataFrame, held: set[str], today: dt.date) -> tuple[pd.Series, dict]:
        universe = [s for s in cfg.universe if s in hist.columns]
        mom = momentum(hist[universe], cfg).iloc[-1]
        ok = mom.notna() & hist[universe].iloc[-1].notna()
        w = pd.Series(0.0, index=universe)
        if ok.any():
            w[ok] = 1.0 / int(ok.sum())
        return w, {"n_eligible": int(ok.sum())}

    return fn


def run_all(
    prices: pd.DataFrame, cash_ret: pd.Series, cfg: StockConfig, costs: CostModel = DEFAULT_COSTS
) -> dict:
    start = evaluation_start(prices, cash_ret, cfg)
    symbols = [s for s in cfg.universe if s in prices.columns]
    strat = run_monthly(prices, cash_ret, strategy_fn(cfg), symbols, start, costs)
    ew = run_monthly(prices, cash_ret, equal_weight_fn(cfg), symbols, start, costs)
    spy = prices[cfg.market].pct_change().loc[strat.returns.index].fillna(0.0)
    spy.iloc[0] = 0.0
    return {
        "start": start,
        "strategy": strat,
        "equal_weight": ew,
        "spy": spy,
        "cash": cash_ret.loc[strat.returns.index].fillna(0.0),
    }


def summary(res: dict, cfg: StockConfig) -> pd.DataFrame:
    rf = res["cash"]
    rows = {
        "S&P 100 momentum": res["strategy"].returns,
        "Equal-weight S&P 100": res["equal_weight"].returns,
        "SPY buy & hold": res["spy"],
    }
    table = pd.DataFrame({k: perf_stats(v, rf) for k, v in rows.items()}).T
    years = len(rf) / 252
    for name, bt in (("S&P 100 momentum", res["strategy"]), ("Equal-weight S&P 100", res["equal_weight"])):
        avg_nav = float(bt.nav.mean())
        table.loc[name, "Turnover / yr"] = float(bt.trades["value"].sum()) / avg_nav / years
        table.loc[name, "Cost drag / yr"] = float(bt.costs.sum()) / avg_nav / years
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download prices from Yahoo")
    args = ap.parse_args()

    from config import load_config
    from data import load_stock_prices
    from logsetup import code_version
    from universe import load_universe

    conf, _ = load_config()
    uni = load_universe()
    cfg = stock_config(conf, uni)
    ymap = dict(zip(uni["symbol"], uni["yahoo"], strict=True)) | {cfg.market: cfg.market, "BIL": "BIL"}
    prices = load_stock_prices(ymap, START, CACHE, args.refresh)
    cash_ret = prices.pop("BIL").pct_change()
    res = run_all(prices, cash_ret, cfg)
    s = res["strategy"]
    print(
        f"Code {code_version(ROOT)} | universe {len(uni)} stocks as of {uni['as_of'].iloc[0]} "
        f"(SURVIVORSHIP-BIASED: today's members)"
    )
    print(
        f"Data {prices.index[0].date()} -> {prices.index[-1].date()} | "
        f"evaluation from {res['start'].date()} "
        f"({len(s.returns)} days, {len(s.decisions)} rebalances)"
    )
    print(
        f"Params: top_n={cfg.top_n}, hold_rank={cfg.hold_rank}, mom {cfg.mom_lookback}-{cfg.mom_skip}, "
        f"{cfg.market} SMA{cfg.sma_window} {cfg.exit_ratio}/{cfg.enter_ratio}"
    )
    print(f"Costs: {DEFAULT_COSTS} | capital ${CAPITAL:,.0f} | min trade ${MIN_TRADE:.0f}")
    pd.set_option("display.width", 200)
    table = summary(res, cfg)
    pct_cols = ["CAGR", "Volatility", "Max drawdown", "Turnover / yr", "Cost drag / yr"]
    out = table[
        [
            "CAGR",
            "Volatility",
            "Sharpe",
            "Sharpe SE",
            "Max drawdown",
            "Calmar",
            "Turnover / yr",
            "Cost drag / yr",
        ]
    ].copy()
    for c in pct_cols:
        out[c] = out[c].map(lambda v: "–" if not math.isfinite(v) else f"{v * 100:.1f}%")
    print(out.round(2).to_string())
    on = s.decisions["market_on"].mean()
    print(
        f"Months in the market: {on:.0%} | final NAV ${s.nav.iloc[-1]:,.0f} | "
        f"total costs ${s.costs.sum():,.0f}"
    )


def stock_config(conf: dict, uni: pd.DataFrame) -> StockConfig:
    """StockConfig from config.toml [stocks] (defaults = STOCK_STRATEGY.md) and the universe file."""
    params = {k: v for k, v in conf.get("stocks", {}).items() if k in StockConfig.__dataclass_fields__}
    return StockConfig(universe=tuple(uni["symbol"]), **params)


if __name__ == "__main__":
    main()
