"""Rule 2 in test form: eventbot.decision and stock_day() must produce identical orders from identical inputs.

stock_day() runs unmodified against the repo's own FakeIB in dry mode, and its logged plan is the reference.
"""

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bot import Abort
from config import load_config as load_daily_config
from eventbot.decision import (
    failed_price_checks,
    held_symbols,
    make_intents,
    managed_positions,
    plan_stock_orders,
    price_frame_hash,
    require_tradable,
    wanted_symbols,
)
from eventbot.schemas import Side
from stock_strategy import StockConfig, decide
from stockbot import ny_date, stock_day, symbol_maps
from tests.conftest import make_stock_prices
from tests.fakes import FAKE_ACCOUNT, FakeIB

SYMS = [f"S{i:02d}" for i in range(1, 31)]
CFG = StockConfig(universe=tuple(SYMS), min_eligible=25)
RISK = dict(
    capital_usd=25_000,
    max_order_usd=5_000,
    max_orders_per_run=24,
    min_trade_usd=100,
    cash_buffer=0.02,
    fill_timeout_s=0,
)
STOCKS: dict[str, Any] = {"max_price_gap": 0.15, "candidates": 10}


def universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": SYMS,
            "yahoo": SYMS,
            "ib_symbol": SYMS,
            "name": SYMS,
            "sector": "X",
            "primary_exchange": "NYSE",
            "con_id": 0,
            "as_of": "2026-10-03",
        }
    )


def daily_plan(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    prices: pd.DataFrame,
    ib_prices: dict[str, float],
    positions: dict[str, int],
    usd_cash: float,
    **risk_over: Any,
) -> tuple[list, float]:
    """What the daily bot plans, from its own log record."""
    conf, _ = load_daily_config(env_file=None)
    conf["ib"]["account"] = FAKE_ACCOUNT
    conf["risk"].update(RISK | risk_over)
    conf["stocks"] = STOCKS
    day = (prices.index[-1] + pd.offsets.BDay(1)).date()
    clock = dt.datetime.combine(day, dt.time(15, 0), tzinfo=dt.UTC)
    hours = f"{day:%Y%m%d}:0930-{day:%Y%m%d}:1600"
    ib = FakeIB(ib_prices, positions=positions, usd_cash=usd_cash, liquid_hours=hours)
    with caplog.at_level(logging.INFO, logger="dailybot"):
        caplog.clear()
        stock_day(ib, conf, CFG, universe(), prices, "dry", tmp_path, clock=lambda: clock)
    plans = [r for r in caplog.records if getattr(r, "event", "") == "plan"]
    assert len(plans) == 1
    return plans[0].data["orders"], plans[0].data["budget"]  # type: ignore[attr-defined]


def new_plan(
    prices: pd.DataFrame,
    ib_prices: dict[str, float],
    positions: dict[str, int],
    usd_cash: float,
    **risk_over: Any,
) -> tuple[list, float]:
    """The same inputs through eventbot.decision, in the order strategy_service calls it."""
    day = (prices.index[-1] + pd.offsets.BDay(1)).date()
    to_ib, _ = symbol_maps(universe())
    managed = managed_positions(positions, to_ib, set())
    held = held_symbols(managed)
    dec = decide(prices, CFG, held & set(to_ib), day)
    wanted = wanted_symbols(held, dec, CFG, STOCKS["candidates"])
    ibkr = {s: ib_prices[s] for s in wanted if s in ib_prices}
    bad = failed_price_checks(wanted, ibkr, prices.iloc[-1], STOCKS["max_price_gap"])
    require_tradable(bad, held, dec, STOCKS["max_price_gap"])
    plan = plan_stock_orders(dec, managed, ibkr, bad, usd_cash, RISK | risk_over, CFG)
    return plan.orders, plan.budget


def assert_same(a: tuple[list, float], b: tuple[list, float]) -> None:
    assert [tuple(o) for o in a[0]] == [tuple(o) for o in b[0]]
    assert a[1] == b[1]


@pytest.fixture
def market_on() -> pd.DataFrame:
    return make_stock_prices(market_drift=1.5e-3)


@pytest.fixture
def market_off() -> pd.DataFrame:
    return make_stock_prices(market_drift=-1.5e-3)


def last(prices: pd.DataFrame) -> dict[str, float]:
    return prices.iloc[-1].to_dict()


def test_first_rebalance_from_all_cash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    args: Any = (market_on, last(market_on), {}, 30_000.0)
    reference = daily_plan(tmp_path, caplog, *args)
    assert len(reference[0]) == 10
    assert_same(reference, new_plan(*args))


def test_rotation_with_existing_holdings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    picks = decide(market_on, CFG, (), (market_on.index[-1] + pd.offsets.BDay(1)).date())
    laggards = [s for s in SYMS if s not in picks.ranks.index[:20]][:3]
    positions = {s: 20 for s in [*picks.picks[:2], *laggards]}
    args: Any = (market_on, last(market_on), positions, 15_000.0)
    reference = daily_plan(tmp_path, caplog, *args)
    assert {o[1] for o in reference[0]} == {"BUY", "SELL"}
    assert [o[1] for o in reference[0]] == sorted((o[1] for o in reference[0]), reverse=True)
    assert_same(reference, new_plan(*args))


def test_market_filter_off_sells_everything(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_off: pd.DataFrame
) -> None:
    held = {"S01": 10, "S02": 15}
    args: Any = (market_off, last(market_off), held, 5_000.0)
    reference = daily_plan(tmp_path, caplog, *args)
    if reference[0]:
        assert {o[1] for o in reference[0]} == {"SELL"}
    assert_same(reference, new_plan(*args))


def test_an_unaffordable_pick_is_replaced_identically(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    day = (market_on.index[-1] + pd.offsets.BDay(1)).date()
    expensive = decide(market_on, CFG, (), day).picks[0]
    scaled = market_on.copy()
    scaled[expensive] = scaled[expensive] * 200
    args: Any = (scaled, last(scaled), {}, 30_000.0)
    reference = daily_plan(tmp_path, caplog, *args)
    assert expensive not in [o[0] for o in reference[0]]
    assert_same(reference, new_plan(*args))


def test_buys_scaled_down_by_available_cash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    args: Any = (market_on, last(market_on), {}, 3_000.0)
    assert_same(daily_plan(tmp_path, caplog, *args), new_plan(*args))


def test_a_small_budget_and_the_minimum_trade_rule(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    args: Any = (market_on, last(market_on), {}, 25_000.0)
    assert_same(
        daily_plan(tmp_path, caplog, *args, min_trade_usd=2_000), new_plan(*args, min_trade_usd=2_000)
    )


def test_a_cap_breach_aborts_in_both(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    args: Any = (market_on, last(market_on), {}, 30_000.0)
    with pytest.raises(Abort, match="max_order_usd"):
        daily_plan(tmp_path, caplog, *args, max_order_usd=100)
    with pytest.raises(Abort, match="max_order_usd"):
        new_plan(*args, max_order_usd=100)


def test_a_failed_price_check_on_a_pick_aborts_in_both(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    day = (market_on.index[-1] + pd.offsets.BDay(1)).date()
    pick = decide(market_on, CFG, (), day).picks[0]
    prices = last(market_on)
    prices[pick] *= 1.5
    args: Any = (market_on, prices, {}, 30_000.0)
    with pytest.raises(Abort, match="Price check failed"):
        daily_plan(tmp_path, caplog, *args)
    with pytest.raises(Abort, match="Price check failed"):
        new_plan(*args)


def test_a_failed_price_check_on_a_candidate_only_drops_the_candidate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, market_on: pd.DataFrame
) -> None:
    day = (market_on.index[-1] + pd.offsets.BDay(1)).date()
    dec = decide(market_on, CFG, (), day)
    candidate = [s for s in dec.ranks.index if s not in dec.picks][0]
    prices = last(market_on)
    prices[candidate] *= 1.5
    args: Any = (market_on, prices, {}, 30_000.0)
    assert_same(daily_plan(tmp_path, caplog, *args), new_plan(*args))


def test_a_missing_price_counts_as_failed() -> None:
    assert failed_price_checks(["A", "B"], {"A": 100.0}, {"A": 100.0, "B": 100.0}, 0.15) == {"B"}


def test_positions_outside_the_universe_are_not_managed() -> None:
    assert managed_positions({"S01": 3, "CSPX": 5, "S02": 0}, SYMS, set()) == {"S01": 3}
    assert managed_positions({"OLD": 3}, SYMS, {"OLD"}) == {"OLD": 3}


DAY = dt.date(2026, 11, 2)


def test_intents_carry_the_position_after_the_order_as_target() -> None:
    orders = [("S01", "SELL", 5, 500.0), ("S02", "BUY", 4, 400.0)]
    intents = make_intents(orders, {"S01": 8, "S02": 1}, "stocks", DAY, "stocks-20261102")
    assert [(i.symbol, i.side, i.target_qty, i.qty, i.est_price) for i in intents] == [
        ("S01", Side.SELL, 3, 5, 100.0),
        ("S02", Side.BUY, 5, 4, 100.0),
    ]


def test_a_rerun_after_a_partial_fill_keeps_the_same_intent_id() -> None:
    """A buy of 10 from 0 that filled 4 is a buy of 6 from 4: the same target, so the same ID."""
    first = make_intents([("S01", "BUY", 10, 1000.0)], {}, "stocks", DAY, "d")[0]
    rerun = make_intents([("S01", "BUY", 6, 600.0)], {"S01": 4}, "stocks", DAY, "d")[0]
    assert first.intent_id == rerun.intent_id


def test_two_intents_of_one_decision_have_different_ids() -> None:
    a, b = make_intents([("S01", "BUY", 1, 10.0), ("S02", "BUY", 1, 10.0)], {}, "stocks", DAY, "d")
    assert a.intent_id != b.intent_id


def test_price_frame_hash_changes_when_history_is_revised() -> None:
    prices = make_stock_prices(n_days=300)
    revised = prices.copy()
    revised.iloc[10, 0] *= 1.0001
    assert price_frame_hash(prices) == price_frame_hash(prices.copy())
    assert price_frame_hash(prices) != price_frame_hash(revised)


def test_ny_date_comes_from_the_daily_bot() -> None:
    assert ny_date(dt.datetime(2026, 11, 3, 1, 0, tzinfo=dt.UTC)) == dt.date(2026, 11, 2)
