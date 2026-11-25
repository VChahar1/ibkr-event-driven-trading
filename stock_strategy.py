"stock_strategy.py - S&P 100 12-1 momentum with a SPY trend filter (pure pandas)."

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StockConfig:
    universe: tuple[str, ...]
    market: str = "SPY"
    top_n: int = 10
    hold_rank: int = 20
    mom_lookback: int = 252
    mom_skip: int = 21
    sma_window: int = 200
    exit_ratio: float = 0.98
    enter_ratio: float = 1.02
    min_eligible: int = 50


@dataclass(frozen=True)
class Decision:
    signal_date: pd.Timestamp
    weights: pd.Series
    market_on: bool
    picks: list[str]
    ranks: pd.Series


def momentum(prices: pd.DataFrame, cfg: StockConfig) -> pd.DataFrame:
    """12-1 momentum, same formula as strategy.signal_frames (row t uses closes up to t)."""
    return prices.shift(cfg.mom_skip) / prices.shift(cfg.mom_lookback) - 1.0


def trend_ratio(close: pd.Series, cfg: StockConfig) -> pd.Series:
    return close / close.rolling(cfg.sma_window).mean()


def month_ends(index: pd.DatetimeIndex, today: dt.date | None = None) -> pd.DatetimeIndex:
    """Last trading day of each month in `index`. The final row counts only if `today` is
    already in a later month (otherwise its month may not be over)."""
    if len(index) == 0:
        return index
    nxt = index[1:].to_period("M") != index[:-1].to_period("M")
    ends = list(index[:-1][nxt])
    last = index[-1]
    if today is not None and (today.year, today.month) > (last.year, last.month):
        ends.append(last)
    return pd.DatetimeIndex(ends)


def market_states(close: pd.Series, dates: pd.DatetimeIndex, cfg: StockConfig) -> pd.Series:
    """Filter state at each evaluation date, with hysteresis, replayed from the first date
    where the SMA exists. Starts "on" if the ratio is above 1 at that first evaluation."""
    ratio = trend_ratio(close, cfg).reindex(dates)
    state: bool | None = None
    out = {}
    for d, r in ratio.items():
        if pd.isna(r):
            continue
        if state is None:
            state = bool(r > 1)
        elif state and r < cfg.exit_ratio:
            state = False
        elif not state and r > cfg.enter_ratio:
            state = True
        out[d] = state
    return pd.Series(out, dtype=bool)


def select(mom_row: pd.Series, held: Iterable[str], cfg: StockConfig) -> tuple[list[str], pd.Series]:
    """Buffer rule: keep holdings ranked <= hold_rank, then fill from the top to top_n.
    Ties break alphabetically so the choice is deterministic."""
    eligible = mom_row.dropna()
    order = sorted(eligible.index, key=lambda s: (-float(eligible[s]), s))
    ranks = pd.Series(range(1, len(order) + 1), index=order, dtype=int)
    picks = sorted(
        (s for s in set(held) if s in ranks.index and ranks[s] <= cfg.hold_rank), key=lambda s: ranks[s]
    )[: cfg.top_n]
    for s in order:
        if len(picks) >= cfg.top_n:
            break
        if s not in picks:
            picks.append(s)
    return sorted(picks, key=lambda s: ranks[s]), ranks


def decide(
    prices: pd.DataFrame, cfg: StockConfig, held: Iterable[str] = (), today: dt.date | None = None
) -> Decision:
    "The decision for the latest completed month-end in `prices` (columns: universe + market)."
    ends = month_ends(prices.index, today)
    if len(ends) == 0:
        raise ValueError("no completed month-end in the price history")
    t = ends[-1]
    hist = prices.loc[:t]
    states = market_states(hist[cfg.market], ends[ends <= t], cfg)
    if t not in states.index:
        raise ValueError(
            f"market filter undefined at {t.date()} (need {cfg.sma_window} days of {cfg.market})"
        )
    on = bool(states[t])
    universe = [s for s in cfg.universe if s in hist.columns]
    mom = momentum(hist[universe], cfg).iloc[-1]
    mom = mom[hist[universe].iloc[-1].notna()]
    picks, ranks = select(mom, held, cfg) if on else ([], select(mom, (), cfg)[1])
    w = pd.Series(0.0, index=universe)
    if picks:
        w[picks] = 1.0 / cfg.top_n
    return Decision(signal_date=t, weights=w, market_on=on, picks=picks, ranks=ranks)


def rebalance_due(today: dt.date, last_rebalance: dt.date | None) -> bool:
    """Monthly: due on the first run of a calendar month that hasn't rebalanced yet."""
    return last_rebalance is None or (today.year, today.month) != (last_rebalance.year, last_rebalance.month)


def eligible_count(prices: pd.DataFrame, cfg: StockConfig) -> pd.Series:
    """Number of stocks with a valid momentum signal on each date (backtest start rule)."""
    universe = [s for s in cfg.universe if s in prices.columns]
    mom = momentum(prices[universe], cfg)
    return (mom.notna() & prices[universe].notna()).sum(axis=1)
