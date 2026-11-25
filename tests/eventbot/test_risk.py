"""Each pre-trade check, in isolation: one failing condition at a time against an otherwise clean context."""

import datetime as dt
from dataclasses import replace
from typing import Any

import pytest

from eventbot.config import load_config
from eventbot.risk import BLOCKING, PreTradeContext, RejectCode, check, gross_notional
from eventbot.schemas import OrderIntent, Side

CFG = load_config()
NOW = dt.datetime(2026, 11, 2, 15, 30, tzinfo=dt.UTC)
DAY = dt.date(2026, 11, 2)


def intent(
    symbol: str = "AMD",
    side: Side = Side.BUY,
    qty: float = 4,
    target: float = 4,
    age_s: float = 0,
    day: dt.date = DAY,
) -> OrderIntent:
    built = OrderIntent.build(
        strategy="stocks",
        trading_date=day,
        symbol=symbol,
        side=side,
        target_qty=target,
        qty=qty,
        est_price=600.0,
        decision_id="d",
    )
    return built.model_copy(update={"ts_utc": NOW - dt.timedelta(seconds=age_s)})


def ctx(**over: Any) -> PreTradeContext:
    base: dict[str, Any] = dict(
        now=NOW,
        today=DAY,
        connected=True,
        kill_switch=None,
        stop_file=False,
        md_heartbeat_age_s=1.0,
        reconciler_heartbeat_age_s=1.0,
        market_open=True,
        clock_skew_s=1.0,
        price=600.0,
    )
    base.update(over)
    return PreTradeContext(**base)


def codes(i: OrderIntent, c: PreTradeContext, caps: Any = CFG.caps) -> list[RejectCode]:
    return [r.code for r in check(i, c, caps, CFG.adapter)]


def test_a_clean_order_passes_every_check() -> None:
    assert codes(intent(), ctx()) == []


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"kill_switch": "reconcile mismatch"}, RejectCode.KILL_SWITCH),
        ({"stop_file": True}, RejectCode.STOP_FILE),
        ({"connected": False}, RejectCode.BROKER_DISCONNECTED),
        ({"md_heartbeat_age_s": None}, RejectCode.STALE_MARKET_DATA),
        ({"reconciler_heartbeat_age_s": None}, RejectCode.RECONCILER_NOT_RUNNING),
        ({"market_open": False}, RejectCode.MARKET_CLOSED),
        ({"clock_skew_s": None}, RejectCode.CLOCK_SKEW),
        ({"clock_skew_s": 31.0}, RejectCode.CLOCK_SKEW),
        ({"clock_skew_s": -31.0}, RejectCode.CLOCK_SKEW),
    ],
)
def test_system_state_checks_block(over: dict, expected: RejectCode) -> None:
    assert codes(intent(), ctx(**over)) == [expected]
    assert expected in BLOCKING


def test_clock_skew_at_the_limit_passes() -> None:
    assert codes(intent(), ctx(clock_skew_s=30.0)) == []


def test_an_intent_from_another_day_is_stale() -> None:
    assert codes(intent(day=DAY - dt.timedelta(days=1)), ctx()) == [RejectCode.STALE_INTENT]


def test_an_old_intent_is_stale() -> None:
    assert codes(intent(age_s=301), ctx()) == [RejectCode.STALE_INTENT]
    assert codes(intent(age_s=299), ctx()) == []


def test_order_notional_cap_rejects_above_and_allows_at_the_cap() -> None:
    caps = CFG.caps.model_copy(update={"max_position_notional_usd": 1e9})
    assert codes(intent(qty=5, target=5), ctx(price=1000.0), caps) == []
    assert codes(intent(qty=5, target=5), ctx(price=1000.01), caps) == [RejectCode.ORDER_NOTIONAL]


def test_order_notional_cap_also_applies_to_sells() -> None:
    """Same as plan_orders today: any order above max_order_usd is refused, whichever side."""
    c = ctx(price=1001.0, positions={"AMD": 5.0}, marks={"AMD": 1001.0})
    assert codes(intent(side=Side.SELL, qty=5, target=0), c) == [RejectCode.ORDER_NOTIONAL]


def test_per_symbol_cap_rejects_a_position_that_would_exceed_it() -> None:
    caps = CFG.caps.model_copy(update={"max_order_notional_usd": 1e9})
    c = ctx(price=1000.0, positions={"AMD": 4.0}, marks={"AMD": 1000.0})
    assert codes(intent(qty=1, target=5), c, caps) == []
    assert codes(intent(qty=2, target=6), c, caps) == [RejectCode.POSITION_NOTIONAL]


def test_per_symbol_cap_never_blocks_reducing_a_position() -> None:
    """A holding that grew past the cap must still be sellable."""
    c = ctx(price=1000.0, positions={"AMD": 8.0}, marks={"AMD": 1000.0})
    assert codes(intent(side=Side.SELL, qty=1, target=7), c) == []


def test_gross_cap_rejects_when_the_order_would_exceed_it() -> None:
    held = {f"S{n}": 10.0 for n in range(10)}
    marks = {s: 240.0 for s in held}
    allowed = ctx(positions=held, marks=marks, price=500.0)
    assert codes(intent(qty=2, target=2), allowed) == []
    assert codes(intent(qty=3, target=3), allowed) == [RejectCode.GROSS_NOTIONAL]


def test_gross_cap_never_blocks_a_sell_when_already_above_it() -> None:
    held = {f"S{n}": 10.0 for n in range(10)} | {"AMD": 4.0}
    marks = {s: 300.0 for s in held}
    assert codes(intent(side=Side.SELL, qty=4, target=0), ctx(positions=held, marks=marks, price=300.0)) == []


def test_gross_notional_uses_average_cost_when_there_is_no_mark() -> None:
    assert gross_notional({"A": 2.0, "B": 1.0}, {"A": 10.0}, {"B": 5.0}) == 25.0
    assert gross_notional({"A": 2.0}, {}, {}) is None


def test_a_position_with_no_price_at_all_fails_closed_on_gross() -> None:
    c = ctx(positions={"OLD": 5.0}, marks={})
    assert codes(intent(), c) == [RejectCode.GROSS_NOTIONAL]


def test_orders_per_day_cap() -> None:
    assert codes(intent(), ctx(orders_today=23)) == []
    assert codes(intent(), ctx(orders_today=24)) == [RejectCode.ORDERS_PER_DAY]


def test_no_price_rejects() -> None:
    assert codes(intent(), ctx(price=None)) == [RejectCode.NO_PRICE]
    assert codes(intent(), ctx(price=0.0)) == [RejectCode.NO_PRICE]


def test_selling_more_than_held_would_short_and_is_rejected() -> None:
    i = OrderIntent.build(
        strategy="stocks",
        trading_date=DAY,
        symbol="AMD",
        side=Side.SELL,
        target_qty=0,
        qty=5,
        est_price=600.0,
        decision_id="d",
    )
    found = codes(i, ctx(positions={"AMD": 3.0}))
    assert RejectCode.WOULD_SHORT in found


def test_target_mismatch_catches_an_order_that_has_already_filled() -> None:
    """Position is already at the target, so buying again would double it. The last line of defence."""
    assert codes(intent(qty=4, target=4), ctx(positions={"AMD": 4.0})) == [RejectCode.TARGET_MISMATCH]


def test_an_open_order_on_the_symbol_rejects() -> None:
    assert codes(intent(), ctx(open_order_symbols=frozenset({"AMD"}))) == [RejectCode.OPEN_ORDER_ON_SYMBOL]
    assert codes(intent(), ctx(open_order_symbols=frozenset({"MU"}))) == []


def test_every_failing_check_is_reported_not_just_the_first() -> None:
    c = replace(ctx(), kill_switch="x", market_open=False, md_heartbeat_age_s=None, price=None)
    assert set(codes(intent(), c)) == {
        RejectCode.KILL_SWITCH,
        RejectCode.MARKET_CLOSED,
        RejectCode.STALE_MARKET_DATA,
        RejectCode.NO_PRICE,
    }


def test_rejections_know_whether_they_are_blocking() -> None:
    blocking = check(intent(), ctx(kill_switch="x"), CFG.caps, CFG.adapter)
    cap = check(intent(qty=9, target=9), ctx(price=1000.0), CFG.caps, CFG.adapter)
    assert all(r.blocking for r in blocking) and not any(r.blocking for r in cap)
