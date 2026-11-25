"""Stock strategy rules (STOCK_STRATEGY.md): ranking, buffer, market filter, timing, no look-ahead."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stock_strategy import (
    StockConfig,
    decide,
    eligible_count,
    market_states,
    momentum,
    month_ends,
    rebalance_due,
    select,
    trend_ratio,
)
from strategy import StrategyConfig, signal_frames
from tests.conftest import make_stock_prices

UNIVERSE = tuple(f"S{i:02d}" for i in range(1, 31))


def scfg(**kw: object) -> StockConfig:
    return StockConfig(universe=UNIVERSE, **kw)  # type: ignore[arg-type]


def test_formulas_match_the_etf_strategy() -> None:
    """Same 12-1 momentum and price/SMA ratio as strategy.signal_frames."""
    p = make_stock_prices()
    mom_etf, ratio_etf = signal_frames(p, StrategyConfig())
    pd.testing.assert_frame_equal(momentum(p, scfg()), mom_etf)
    pd.testing.assert_series_equal(trend_ratio(p["SPY"], scfg()), ratio_etf["SPY"])


def test_select_takes_top_n_by_momentum() -> None:
    mom = pd.Series({f"S{i:02d}": float(i) for i in range(1, 31)})
    picks, ranks = select(mom, (), scfg())
    assert picks == [f"S{i:02d}" for i in range(30, 20, -1)]
    assert ranks["S30"] == 1 and ranks["S01"] == 30


def test_buffer_keeps_holdings_ranked_within_hold_rank() -> None:
    mom = pd.Series({f"S{i:02d}": float(i) for i in range(1, 31)})
    held = {"S15", "S05"}
    picks, _ = select(mom, held, scfg())
    assert "S15" in picks and "S05" not in picks and len(picks) == 10
    assert picks[-1] == "S15"
    assert "S21" not in picks


def test_select_breaks_ties_alphabetically_and_ignores_nan() -> None:
    mom = pd.Series({"S02": 0.1, "S01": 0.1, "S03": np.nan})
    picks, ranks = select(mom, (), scfg(top_n=1))
    assert picks == ["S01"] and "S03" not in ranks.index


def test_fewer_eligible_than_slots_leaves_cash() -> None:
    picks, _ = select(pd.Series({"S01": 0.2, "S02": 0.1}), (), scfg())
    assert picks == ["S01", "S02"]


def closes_for_ratios(ratios: list[float], start: float = 100.0) -> pd.Series:
    """With a 2-day SMA, ratio_t = 2c_t / (c_t + c_{t-1}), so c_t = r * c_{t-1} / (2 - r)."""
    c = [start]
    for r in ratios:
        c.append(r * c[-1] / (2 - r))
    return pd.Series(c, index=pd.bdate_range("2024-01-01", periods=len(c)))


def test_market_filter_hysteresis() -> None:
    cfg = scfg(sma_window=2)

    ratios = [1.01, 0.99, 0.97, 1.01, 1.03, 0.985]
    close = closes_for_ratios(ratios)
    np.testing.assert_allclose(trend_ratio(close, cfg).dropna().to_numpy(), ratios)
    states = market_states(close, close.index, cfg)
    assert list(states) == [True, True, False, False, True, True]
    assert list(
        market_states(closes_for_ratios([0.995, 1.01]), closes_for_ratios([0.995, 1.01]).index, cfg)
    ) == [False, False]


def test_filter_replay_converges_from_different_start_dates() -> None:
    p = make_stock_prices(n_days=900)
    ends = month_ends(p.index)
    full = market_states(p["SPY"], ends, scfg())
    late = market_states(p["SPY"].iloc[300:], ends[ends > p.index[300]], scfg())
    common = late.index[late.index >= late.index[6]]
    assert (full.loc[common] == late.loc[common]).all()


def test_month_ends_need_the_month_to_be_over() -> None:
    idx = pd.bdate_range("2026-09-24", "2026-10-02")
    assert list(month_ends(idx)) == [pd.Timestamp("2026-09-30")]
    idx2 = pd.bdate_range("2026-09-24", "2026-09-30")
    assert list(month_ends(idx2)) == []
    assert list(month_ends(idx2, dt.date(2026, 10, 1))) == [pd.Timestamp("2026-09-30")]


def test_decide_off_market_means_all_cash() -> None:
    p = make_stock_prices(market_drift=-2e-3)
    d = decide(p, scfg())
    assert not d.market_on and d.picks == [] and d.weights.sum() == 0.0
    assert len(d.ranks) > 0


def test_decide_on_market_equal_weights_top_n() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    d = decide(p, scfg())
    assert d.market_on and len(d.picks) == 10
    assert d.weights.sum() == pytest.approx(1.0) and set(d.weights[d.weights > 0]) == {0.1}
    assert d.signal_date == month_ends(p.index)[-1]


def test_decide_has_no_look_ahead() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    d = decide(p, scfg())
    shocked = p.copy()
    shocked.loc[shocked.index > d.signal_date] *= 5.0
    d2 = decide(shocked, scfg())
    assert d2.signal_date == d.signal_date and d2.picks == d.picks


def test_stock_without_a_close_at_signal_date_is_not_eligible() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    t = month_ends(p.index)[-1]
    best = decide(p, scfg()).picks[0]
    p.loc[t, best] = np.nan
    assert best not in decide(p, scfg()).picks


def test_rebalance_due_monthly() -> None:
    assert rebalance_due(dt.date(2026, 10, 5), None)
    assert not rebalance_due(dt.date(2026, 10, 20), dt.date(2026, 10, 5))
    assert rebalance_due(dt.date(2026, 11, 2), dt.date(2026, 10, 5))


def test_eligible_count_grows_after_warmup() -> None:
    p = make_stock_prices()
    n = eligible_count(p, scfg())
    assert n.iloc[0] == 0 and n.iloc[-1] == 30
