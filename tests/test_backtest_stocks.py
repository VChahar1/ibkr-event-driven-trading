"""Stock backtest engine: costs, accounting, timing, no look-ahead (synthetic data, offline)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backtest_stocks import (
    CostModel,
    DecideFn,
    equal_weight_fn,
    evaluation_start,
    run_all,
    run_monthly,
    strategy_fn,
)
from stock_strategy import StockConfig, decide, month_ends
from tests.conftest import make_stock_prices

UNIVERSE = tuple(f"S{i:02d}" for i in range(1, 31))
CFG = StockConfig(universe=UNIVERSE, min_eligible=25)
ZERO = CostModel(per_share=0.0, min_order=0.0, max_pct=0.0, slippage_bps=0.0)


def cash(prices: pd.DataFrame, daily: float = 0.0001) -> pd.Series:
    return pd.Series(daily, index=prices.index)


def test_cost_model_minimum_per_share_and_cap() -> None:
    c = CostModel(slippage_bps=0.0)
    assert c.order_cost(1000, 100) == pytest.approx(1.00)
    assert c.order_cost(100_000, 10) == pytest.approx(50.0)
    assert c.order_cost(50, 0.01) == pytest.approx(0.5)
    assert CostModel().order_cost(10_000, 100) == pytest.approx(1.0 + 5.0)
    assert c.order_cost(0, 100) == 0.0


def constant_fn(weights: dict) -> DecideFn:
    def fn(hist: pd.DataFrame, held: set, today: dt.date) -> tuple[pd.Series, dict]:
        return pd.Series(weights), {}

    return fn


def test_accounting_matches_a_hand_computed_portfolio() -> None:
    p = make_stock_prices(n_stocks=2, n_days=400)
    start = p.index[300]
    bt = run_monthly(
        p,
        cash(p, 0.0),
        constant_fn({"S01": 0.5, "S02": 0.5}),
        ["S01", "S02"],
        start,
        ZERO,
        capital=10_000,
        min_trade=0.0,
    )
    np.testing.assert_allclose(bt.weights.sum(axis=1).to_numpy(), 1.0)

    execs = list(bt.decisions["exec_date"])
    a, b = execs[0], execs[1]
    seg = p.loc[a:b, ["S01", "S02"]]
    hold = (bt.nav[a] / 2) * seg.iloc[-2] / seg.iloc[0]
    assert float(hold.sum()) == pytest.approx(float(bt.nav[seg.index[-2]]))


def test_trades_happen_the_day_after_each_month_end() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    bt = run_monthly(p, cash(p), strategy_fn(CFG), list(UNIVERSE), evaluation_start(p, cash(p), CFG))
    for _, row in bt.decisions.iterrows():
        nxt = p.index[p.index > row["signal_date"]][0]
        assert row["exec_date"] == nxt and row["signal_date"] in month_ends(p.index)
    assert set(bt.trades["date"]) <= set(bt.decisions["exec_date"])
    held = (bt.weights.drop(columns="CASH") > 0).sum(axis=1)
    assert held.max() <= CFG.top_n


def test_market_off_moves_everything_to_cash() -> None:
    p = make_stock_prices(market_drift=-2e-3)
    start = p.index[300]
    bt = run_monthly(p, cash(p), strategy_fn(CFG), list(UNIVERSE), start)
    off = bt.decisions[~bt.decisions["market_on"]]
    assert len(off) > 0
    for d in off["exec_date"]:
        assert bt.weights.loc[d, "CASH"] == pytest.approx(1.0)


def test_costs_reduce_nav_by_what_was_charged() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    start = evaluation_start(p, cash(p, 0.0), CFG)
    free = run_monthly(p, cash(p, 0.0), strategy_fn(CFG), list(UNIVERSE), start, ZERO)
    paid = run_monthly(p, cash(p, 0.0), strategy_fn(CFG), list(UNIVERSE), start, CostModel())
    assert paid.nav.iloc[-1] < free.nav.iloc[-1]
    assert paid.costs.sum() == pytest.approx(paid.trades["cost"].sum())
    assert (paid.trades["cost"] >= 1.0).all()


def test_small_trades_are_skipped() -> None:
    p = make_stock_prices(n_stocks=2, n_days=400)
    bt = run_monthly(
        p,
        cash(p),
        constant_fn({"S01": 0.5, "S02": 0.5}),
        ["S01", "S02"],
        p.index[300],
        ZERO,
        capital=10_000,
        min_trade=10_000,
    )
    assert bt.trades.empty and bt.weights["CASH"].iloc[-1] > 0.99


def test_no_look_ahead_in_the_nav_path() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    start = evaluation_start(p, cash(p), CFG)
    cut = p.index[-120]
    base = run_monthly(p, cash(p), strategy_fn(CFG), list(UNIVERSE), start)
    shocked = p.copy()
    shocked.loc[shocked.index > cut] *= 3.0
    alt = run_monthly(shocked, cash(shocked), strategy_fn(CFG), list(UNIVERSE), start)
    pd.testing.assert_series_equal(base.nav.loc[:cut], alt.nav.loc[:cut])


def test_live_mid_month_decision_equals_the_month_end_decision() -> None:
    """The bot may first run mid-month: extra rows after the month-end must not matter."""
    p = make_stock_prices(market_drift=1.5e-3)
    t = month_ends(p.index)[-2]
    later = p.index[(p.index > t)][5]
    at_end = decide(p.loc[:t], CFG, today=later.date())
    mid = decide(p.loc[:later], CFG, today=later.date())
    assert mid.signal_date == at_end.signal_date == t and mid.picks == at_end.picks


def test_equal_weight_benchmark_spreads_over_eligible_stocks() -> None:
    p = make_stock_prices()
    w, info = equal_weight_fn(CFG)(p, set(), p.index[-1].date())
    assert info["n_eligible"] == 30 and w.sum() == pytest.approx(1.0) and w.nunique() == 1


def test_evaluation_start_respects_min_eligible() -> None:
    p = make_stock_prices()
    late = p.copy()
    late.loc[: late.index[400], [f"S{i:02d}" for i in range(1, 11)]] = np.nan
    assert evaluation_start(late, cash(late), CFG) > evaluation_start(p, cash(p), CFG)


def test_run_all_produces_aligned_series() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    res = run_all(p, cash(p), CFG)
    idx = res["strategy"].returns.index
    assert idx.equals(res["equal_weight"].returns.index) and idx.equals(res["spy"].index)
    assert res["spy"].iloc[0] == 0.0
