import datetime as dt

import pytest
from pydantic import ValidationError

from eventbot.schemas import (
    Alert,
    AlertType,
    Fill,
    OrderEvent,
    OrderIntent,
    PositionSnapshot,
    Side,
    Tick,
    alert,
    is_intent_ref,
    make_intent_id,
)

DAY = dt.date(2026, 11, 2)


def intent(**over: object) -> OrderIntent:
    base: dict = dict(
        strategy="stocks",
        trading_date=DAY,
        symbol="AMD",
        side=Side.BUY,
        target_qty=4,
        qty=4,
        est_price=600.0,
        decision_id="d1",
    )
    base.update(over)
    return OrderIntent.build(**base)


def test_intent_id_is_deterministic_and_covers_every_input() -> None:
    a = make_intent_id("stocks", DAY, "AMD", Side.BUY, 4)
    assert a == make_intent_id("stocks", DAY, "AMD", Side.BUY, 4.0)
    assert a == make_intent_id("stocks", DAY, "AMD", "BUY", 4)
    assert is_intent_ref(a)
    others = [
        make_intent_id("other", DAY, "AMD", Side.BUY, 4),
        make_intent_id("stocks", DAY + dt.timedelta(days=1), "AMD", Side.BUY, 4),
        make_intent_id("stocks", DAY, "MU", Side.BUY, 4),
        make_intent_id("stocks", DAY, "AMD", Side.SELL, 4),
        make_intent_id("stocks", DAY, "AMD", Side.BUY, 3),
    ]
    assert len({a, *others}) == 6


def test_a_rerun_of_the_same_decision_keeps_the_id_even_with_new_msg_ids() -> None:
    first, second = intent(), intent()
    assert first.intent_id == second.intent_id and first.msg_id != second.msg_id


def test_an_intent_whose_id_does_not_match_its_fields_is_invalid() -> None:
    good = intent().model_dump()
    good["target_qty"] = 5
    with pytest.raises(ValidationError):
        OrderIntent.model_validate(good)


def test_every_message_has_version_timestamp_and_unique_id() -> None:
    a, b = Tick(symbol="SPY", price=1.0, md_type=3), Tick(symbol="SPY", price=1.0, md_type=3)
    assert a.schema_version == 1 and a.ts_utc.tzinfo is not None and a.msg_id != b.msg_id


def test_json_round_trip_for_every_schema() -> None:
    now = dt.datetime(2026, 11, 2, 15, 0, tzinfo=dt.UTC)
    for message in (
        Tick(symbol="SPY", price=500.5, bid=500.4, ask=500.6, md_type=3),
        intent(),
        OrderEvent(order_ref="r", symbol="AMD", side=Side.BUY, status="Submitted", qty=4),
        Fill(exec_id="e.01", order_ref="r", symbol="AMD", side=Side.BUY, qty=4, price=600, exec_time_utc=now),
        PositionSnapshot(account="DU***567", kind="periodic", positions={"AMD": 4.0}),
        alert(AlertType.SMOKE, "info", "t", "m", when=now),
    ):
        assert type(message).model_validate_json(message.model_dump_json()) == message


@pytest.mark.parametrize(
    "bad",
    [
        {"symbol": "SPY", "price": -1.0, "md_type": 3},
        {"symbol": "SPY", "price": float("nan"), "md_type": 3},
        {"symbol": "SPY", "price": 1.0, "md_type": 9},
        {"symbol": "", "price": 1.0, "md_type": 3},
        {"symbol": "SPY", "price": 1.0, "md_type": 3, "surprise": 1},
        {"symbol": "SPY", "price": 1.0, "md_type": 3, "schema_version": 2},
        {"symbol": "SPY", "price": 1.0, "md_type": 3, "ts_utc": "2026-11-02T15:00:00"},
    ],
)
def test_invalid_payloads_are_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        Tick.model_validate(bad)


def test_snapshot_refuses_an_unmasked_account() -> None:
    with pytest.raises(ValidationError):
        PositionSnapshot(account="DU" + "1234567", kind="periodic", positions={})


def test_intent_cannot_target_a_short_position() -> None:
    with pytest.raises(ValidationError):
        intent(target_qty=-1)


def test_order_event_terminal_flag() -> None:
    make = lambda s: OrderEvent(order_ref="r", symbol="A", side=Side.BUY, status=s, qty=1)  # noqa: E731
    assert all(make(s).terminal for s in ("Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"))
    live = ("PendingSubmit", "Submitted", "PreSubmitted", "ValidationError")
    assert not any(make(s).terminal for s in live)


def test_alert_helper_makes_details_json_safe() -> None:
    a = alert(AlertType.SMOKE, "info", "t", "m", day=DAY, side=Side.BUY)
    assert a.details == {"day": "2026-11-02", "side": "BUY"}
    assert Alert.model_validate_json(a.model_dump_json()) == a
