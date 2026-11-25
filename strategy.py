"strategy.py - dual-momentum + trend filter on UCITS ETFs (pure pandas)."

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

HELD_MIN_WEIGHT = 0.02


@dataclass(frozen=True)
class StrategyConfig:
    risky: tuple[str, ...] = ("CSPX", "EIMI", "IDTL", "IGLN")
    safe: str = "IB01"
    top_n: int = 2
    mom_lookback: int = 252
    mom_skip: int = 21
    sma_window: int = 200
    band: float = 0.05
    trend_buffer: float = 0.02
    mom_margin: float = 0.02

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.risky) + (self.safe,)

    @property
    def warmup(self) -> int:
        return max(self.mom_lookback, self.sma_window) + 1


def signal_frames(prices: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Momentum and price/SMA ratio for the whole history (no look-ahead:
    row t uses closes up to and including t)."""
    mom = prices.shift(cfg.mom_skip) / prices.shift(cfg.mom_lookback) - 1.0
    ratio = prices / prices.rolling(cfg.sma_window).mean()
    return mom, ratio


def weights_from_signals(
    mom_row: pd.Series, ratio_row: pd.Series, cfg: StrategyConfig, held: frozenset[str] = frozenset()
) -> pd.Series:
    """held: risky ETFs currently owned (drives the hysteresis)."""
    hurdle = mom_row[cfg.safe]

    def in_trend(a: str) -> bool:
        edge = 1 - cfg.trend_buffer if a in held else 1 + cfg.trend_buffer
        return bool(ratio_row[a] > edge)

    eligible = [a for a in cfg.risky if in_trend(a) and np.isfinite(mom_row[a]) and mom_row[a] > hurdle]
    score = {a: mom_row[a] + (cfg.mom_margin if a in held else 0.0) for a in eligible}
    picks = sorted(eligible, key=lambda a: -score[a])[: cfg.top_n]
    w = pd.Series(0.0, index=list(cfg.assets))
    for a in picks:
        w[a] = 1.0 / cfg.top_n
    w[cfg.safe] += 1.0 - w.sum()
    return w


def target_weights(prices: pd.DataFrame, cfg: StrategyConfig, held: Iterable[str] = frozenset()) -> pd.Series:
    """Target weights from the latest row of a price history."""
    if len(prices) < cfg.warmup:
        raise ValueError(f"need >= {cfg.warmup} daily closes, got {len(prices)}")
    mom, ratio = signal_frames(prices[list(cfg.assets)], cfg)
    last_mom = mom.iloc[-1]
    if last_mom.isna().any():
        raise ValueError(f"NaN momentum for {list(last_mom[last_mom.isna()].index)}")
    return weights_from_signals(last_mom, ratio.iloc[-1], cfg, frozenset(held))


def held_set(weights: pd.Series, risky: Iterable[str]) -> frozenset[str]:
    """Risky ETFs that count as held for the hysteresis rules."""
    return frozenset(a for a in risky if weights.get(a, 0.0) > HELD_MIN_WEIGHT)


def needs_rebalance(current_w: pd.Series, target_w: pd.Series, band: float) -> bool:
    cur = current_w.reindex(target_w.index, fill_value=0.0)
    held = set(cur[cur > HELD_MIN_WEIGHT].index)
    tgt = set(target_w[target_w > 0].index)
    if held != tgt:
        return True
    return bool((cur - target_w).abs().max() > band)
