"""The adapter's intent handling and event flow, through the in-memory bus, fakeredis and FakeBroker."""

import asyncio
import json
import logging
from pathlib import Path

import pytest

from eventbot.adapter.fake_broker import FAKE_ACCOUNT, FakeOrder
from eventbot.logs import JsonFormatter
from eventbot.schemas import AlertType, IntentStatus, Side
from tests.eventbot.conftest import Harness, make_intent, today, with_caps


async def test_an_intent_becomes_one_order_with_intent_id_as_order_ref(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    assert await harness.adapter.handle_intent(intent) == "placed"

    assert harness.broker.placed == [intent.intent_id]
    order = harness.broker.state.orders[intent.intent_id]
    assert (order.symbol, order.side, order.qty) == ("AMD", "BUY", 4)
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.PLACED
    assert harness.order_events()[-1].order_ref == intent.intent_id
    assert await harness.state.orders_today(today().isoformat()) == 1


async def test_the_same_intent_delivered_twice_results_in_exactly_one_order(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.deliver(intent)
    await harness.deliver(intent)
    consumer = await harness.intent_consumer()

    assert await harness.adapter.process_one_intent(consumer, 0.01) == "placed"
    assert await harness.adapter.process_one_intent(consumer, 0.01) == "duplicate"

    assert harness.broker.placed == [intent.intent_id]
    assert AlertType.DUPLICATE_INTENT in [a.type for a in harness.alerts()]


async def test_concurrent_delivery_of_one_intent_places_one_order(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    results = await asyncio.gather(*(harness.adapter.handle_intent(intent) for _ in range(5)))
    assert sorted(results) == ["duplicate"] * 4 + ["placed"]
    assert len(harness.broker.placed) == 1


async def test_a_rerun_of_the_same_decision_with_a_new_msg_id_is_still_a_duplicate(harness: Harness) -> None:
    await harness.ready()
    first, second = make_intent(), make_intent()
    assert first.msg_id != second.msg_id
    await harness.adapter.handle_intent(first)
    assert await harness.adapter.handle_intent(second) == "duplicate"
    assert len(harness.broker.placed) == 1


async def test_a_resized_decision_on_the_same_day_is_blocked(harness: Harness) -> None:
    """2026-10-09: AMD sized 3 at 10:30 and 4 at 15:00. Two IDs, and without this guard, 7 shares."""
    await harness.ready()
    three, four = make_intent(qty=3, target=3), make_intent(qty=4, target=4)
    assert three.intent_id != four.intent_id
    assert await harness.adapter.handle_intent(three) == "placed"
    harness.broker.fill(three.intent_id)
    await harness.adapter.drain_broker_events()
    await harness.adapter.publish_snapshot()

    top_up = make_intent(qty=1, target=4)
    assert await harness.adapter.handle_intent(top_up) == "blocked"
    assert harness.broker.placed == [three.intent_id]
    assert AlertType.CONFLICTING_INTENT in [a.type for a in harness.alerts()]


async def test_redis_wiped_then_redelivery_does_not_replace_an_open_order(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)

    await harness.state._r.flushall()
    await harness.fresh_beats()

    assert await harness.adapter.handle_intent(intent) == "rejected"
    assert harness.broker.placed == [intent.intent_id]


async def test_redis_wiped_then_redelivery_does_not_replace_a_filled_order(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.fill(intent.intent_id)
    await harness.adapter.drain_broker_events()

    await harness.state._r.flushall()
    await harness.fresh_beats()

    assert await harness.adapter.handle_intent(intent) == "rejected"
    assert harness.broker.placed == [intent.intent_id]


async def test_redis_wiped_then_redelivery_of_a_cancelled_order_finds_it_at_the_broker(
    harness: Harness,
) -> None:
    """Nothing open and the position unchanged: only the orderRef lookup can catch this one."""
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    await harness.broker.cancel(intent.intent_id)
    await harness.adapter.drain_broker_events()

    await harness.state._r.flushall()
    await harness.fresh_beats()

    assert await harness.adapter.handle_intent(intent) == "already_at_broker"
    assert harness.broker.placed == [intent.intent_id]
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.PLACED


async def assert_refused(h: Harness, status: str, code: str, **intent_kwargs: object) -> None:
    intent = make_intent(**intent_kwargs)  # type: ignore[arg-type]
    assert await h.adapter.handle_intent(intent) == status
    assert h.broker.placed == []
    assert await h.state.intent_status(intent.intent_id) is IntentStatus(status)
    assert code in h.alert_codes(), "every rejected check publishes an alert"
    last = h.order_events()[-1]
    assert last.status == "Rejected" and code in last.reason and last.terminal


async def test_the_kill_switch_blocks_orders(harness: Harness) -> None:
    await harness.ready()
    await harness.state.set_kill_switch("test")
    await assert_refused(harness, "blocked", "kill_switch")


async def test_the_stop_file_blocks_orders(harness: Harness) -> None:
    await harness.ready()
    (harness.root / "STOP").touch()
    await assert_refused(harness, "blocked", "stop_file")


async def test_a_stale_market_data_heartbeat_blocks_trading(harness: Harness) -> None:
    await harness.ready()
    await harness.state._r.delete("hb:md")
    await assert_refused(harness, "blocked", "stale_market_data")


async def test_a_heartbeat_really_expires_after_its_ttl(harness: Harness) -> None:
    await harness.ready()
    harness.state._heartbeat_ttl_s = 1
    await harness.state.beat("md")
    await asyncio.sleep(1.1)
    await assert_refused(harness, "blocked", "stale_market_data")


async def test_no_reconciler_heartbeat_blocks_trading(harness: Harness) -> None:
    await harness.ready()
    await harness.state._r.delete("hb:reconciler")
    await assert_refused(harness, "blocked", "reconciler_not_running")


async def test_a_closed_market_blocks_orders(harness: Harness) -> None:
    harness.broker.state.market_open = False
    await harness.ready()
    await assert_refused(harness, "blocked", "market_closed")


async def test_a_stale_local_clock_blocks_orders(harness: Harness) -> None:
    """Boots with a clock hours behind have happened on this machine."""
    harness.broker.state.clock_offset_s = 3 * 3600
    await harness.ready()
    await assert_refused(harness, "blocked", "clock_skew")


async def test_a_disconnected_broker_blocks_orders(harness: Harness) -> None:
    await harness.ready()
    harness.broker.drop_connection()
    await assert_refused(harness, "blocked", "broker_disconnected")


@pytest.mark.parametrize(
    "caps,qty,price,code",
    [
        ({"max_order_notional_usd": 100.0}, 1, 600.0, "order_notional"),
        ({"max_position_notional_usd": 100.0}, 1, 600.0, "position_notional"),
        ({"max_gross_notional_usd": 100.0}, 1, 600.0, "gross_notional"),
    ],
)
async def test_each_cap_rejects(harness: Harness, caps: dict, qty: int, price: float, code: str) -> None:
    harness.cfg = with_caps(harness.cfg, **caps)
    harness.new_adapter()
    await harness.ready(price=price)
    await assert_refused(harness, "rejected", code, qty=qty, target=qty)


async def test_the_orders_per_day_cap_rejects(harness: Harness) -> None:
    harness.cfg = with_caps(harness.cfg, max_orders_per_day=1)
    harness.new_adapter()
    await harness.ready()
    await harness.state.count_order(today().isoformat())
    await assert_refused(harness, "rejected", "orders_per_day")


async def test_no_current_price_rejects(harness: Harness) -> None:
    await harness.adapter.startup()
    await harness.fresh_beats()
    await assert_refused(harness, "rejected", "no_price")


async def test_a_rejected_intent_stays_rejected_when_redelivered(harness: Harness) -> None:
    await harness.ready()
    await harness.state.set_kill_switch("test")
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    await harness.state.clear_kill_switch()
    assert await harness.adapter.handle_intent(intent) == "duplicate"
    assert harness.broker.placed == []


async def test_partial_fills_are_published_individually_and_track_position(harness: Harness) -> None:
    await harness.ready(price=100.0)
    intent = make_intent(qty=10, target=10)
    await harness.adapter.handle_intent(intent)
    for qty in (3, 3, 4):
        harness.broker.fill(intent.intent_id, qty=qty, price=100.0)
    await harness.adapter.drain_broker_events()

    fills = harness.fills()
    assert [f.qty for f in fills] == [3, 3, 4]
    assert [f.cum_qty for f in fills] == [3, 6, 10]
    assert all(f.order_ref == intent.intent_id and not f.late for f in fills)
    assert harness.adapter.positions["AMD"] == 10
    assert harness.order_events()[-1].status == "Filled"
    assert intent.intent_id not in harness.adapter.open_orders


async def test_a_duplicate_execution_report_publishes_one_fill(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    execution = harness.broker.fill(intent.intent_id)
    harness.broker.report_execution(execution)
    await harness.adapter.drain_broker_events()
    assert len(harness.fills()) == 1
    assert harness.adapter.positions["AMD"] == 4


async def test_a_broker_rejection_marks_the_intent_rejected_and_alerts(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.reject(intent.intent_id, 201, "Order rejected - reason: margin")
    await harness.adapter.drain_broker_events()

    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.REJECTED
    assert AlertType.ORDER_REJECTED in [a.type for a in harness.alerts()]
    assert harness.order_events()[-1].status == "Cancelled"
    assert intent.intent_id not in harness.adapter.open_orders


async def test_an_inactive_status_counts_as_a_rejection(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.reject(intent.intent_id, status="Inactive", from_error=False)
    await harness.adapter.drain_broker_events()
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.REJECTED


async def test_a_user_cancel_is_terminal_but_not_a_rejection(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    await harness.broker.cancel(intent.intent_id)
    await harness.adapter.drain_broker_events()
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.PLACED
    assert harness.order_events()[-1].status == "Cancelled"


async def test_an_error_on_an_order_still_live_at_ib_is_not_reported_as_terminal(harness: Harness) -> None:
    """ib_async marks a Trade Cancelled locally on any error, even when the order stays live at IB."""
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.spurious_error_cancel(intent.intent_id)
    await harness.adapter.drain_broker_events()

    assert all(e.status != "Cancelled" for e in harness.order_events())
    assert intent.intent_id in harness.adapter.open_orders
    assert AlertType.ORDER_UNCERTAIN in [a.type for a in harness.alerts()]


async def test_ib_asyncs_synthetic_validation_error_is_not_terminal(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.validation_warning(intent.intent_id)
    await harness.adapter.drain_broker_events()
    assert intent.intent_id in harness.adapter.open_orders
    assert not harness.order_events()[-1].terminal


async def test_an_unfilled_order_is_cancelled_after_the_timeout(harness: Harness) -> None:
    harness.new_adapter()
    harness.adapter._fill_timeout_s = 0.0
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    await asyncio.sleep(0.01)
    await harness.adapter.cancel_timed_out_orders()
    await harness.adapter.cancel_timed_out_orders()
    await harness.adapter.drain_broker_events()
    assert harness.broker.cancels == [intent.intent_id]
    assert harness.order_events()[-1].status == "Cancelled"


async def test_an_execution_that_did_not_come_from_an_intent_alerts_and_is_not_a_fill(
    harness: Harness,
) -> None:
    await harness.adapter.startup()
    harness.broker.state.orders["stockbot-20261102"] = FakeOrder("stockbot-20261102", "AMD", "BUY", 1.0)
    harness.broker.fill("stockbot-20261102")
    await harness.adapter.drain_broker_events()
    assert harness.fills() == []
    assert AlertType.FOREIGN_EXECUTION in [a.type for a in harness.alerts()]


async def test_an_invalid_intent_produces_an_alert_and_the_next_one_is_processed(harness: Harness) -> None:
    await harness.ready()
    await harness.bus.publish_raw(harness.topics.orders_intents, "bad", b'{"not": "an intent"}')
    tampered = make_intent().model_dump(mode="json")
    tampered["qty"], tampered["target_qty"] = 400, 400
    await harness.bus.publish_raw(harness.topics.orders_intents, "bad2", json.dumps(tampered).encode())
    good = make_intent()
    await harness.deliver(good)

    consumer = await harness.intent_consumer()
    assert await harness.adapter.process_one_intent(consumer, 0.01) == "placed"
    assert await harness.adapter.process_one_intent(consumer, 0.01) is None
    assert harness.broker.placed == [good.intent_id]
    assert [a.type for a in harness.alerts()].count(AlertType.INVALID_MESSAGE) == 2


async def test_redis_down_places_nothing_and_keeps_the_intent_for_retry(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.deliver(intent)
    consumer = await harness.intent_consumer()
    harness.server.connected = False

    task = asyncio.create_task(harness.adapter.process_one_intent(consumer, 0.01))
    await asyncio.sleep(0.05)
    assert not task.done() and harness.broker.placed == []

    harness.server.connected = True
    await harness.fresh_beats()
    assert await asyncio.wait_for(task, 3.0) == "placed"
    assert harness.broker.placed == [intent.intent_id]


async def test_a_crash_before_the_commit_redelivers_and_does_not_double_place(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.deliver(intent)
    consumer = await harness.intent_consumer()
    record = await consumer.getone(0.01)
    assert record is not None
    assert await harness.adapter.handle_intent(intent) == "placed"
    await consumer.stop()

    restarted = await harness.intent_consumer()
    assert await harness.adapter.process_one_intent(restarted, 0.01) == "duplicate"
    assert harness.broker.placed == [intent.intent_id]


async def test_a_kafka_outage_does_not_lose_a_fill(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.bus.fail_publishes = 3
    harness.broker.fill(intent.intent_id)
    await harness.adapter.drain_broker_events()
    assert harness.fills() == []

    await harness.adapter.flush_outbox()
    await harness.adapter.flush_outbox()
    await harness.adapter.flush_outbox()
    assert len(harness.fills()) == 1


async def test_snapshots_carry_positions_cash_and_a_masked_account(harness: Harness) -> None:
    harness.broker.state.positions = {"XOM": 3.0}
    harness.broker.state.usd_cash = 1234.5
    await harness.adapter.startup()
    snapshot = harness.snapshots()[-1]
    assert snapshot.kind == "startup" and snapshot.positions == {"XOM": 3.0} and snapshot.usd_cash == 1234.5
    assert snapshot.account != FAKE_ACCOUNT and FAKE_ACCOUNT[-3:] in snapshot.account


async def test_the_real_account_id_never_appears_on_the_bus(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.fill(intent.intent_id)
    await harness.adapter.drain_broker_events()
    await harness.adapter.publish_snapshot()
    await harness.adapter.flush_ticks()
    for topic in harness.topics.all():
        for record in harness.bus.records(topic):
            assert FAKE_ACCOUNT.encode() not in record.value and FAKE_ACCOUNT != record.key


async def test_market_data_heartbeat_needs_a_connection_and_fresh_ticks(harness: Harness) -> None:
    await harness.adapter.startup()
    await harness.adapter.beat()
    assert await harness.state.heartbeat_age_s("adapter") is not None
    assert await harness.state.heartbeat_age_s("md") is None

    harness.broker.tick("SPY", 500.0)
    await harness.adapter.beat()
    assert await harness.state.heartbeat_age_s("md") is not None


async def test_market_data_heartbeat_stops_when_ticks_stop(harness: Harness) -> None:
    harness.cfg = harness.cfg.model_copy(
        update={"heartbeats": harness.cfg.heartbeats.model_copy(update={"md_max_tick_age_s": 0.01})}
    )
    harness.new_adapter()
    await harness.adapter.startup()
    harness.broker.tick("SPY", 500.0)
    await asyncio.sleep(0.05)
    await harness.adapter.beat()
    assert await harness.state.heartbeat_age_s("md") is None


async def test_market_data_heartbeat_stops_on_disconnect(harness: Harness) -> None:
    await harness.adapter.startup()
    harness.broker.tick("SPY", 500.0)
    harness.broker.drop_connection()
    await harness.adapter.beat()
    assert await harness.state.heartbeat_age_s("md") is None


async def test_ticks_are_published_latest_per_symbol_and_cached(harness: Harness) -> None:
    await harness.adapter.startup()
    for price in (500.0, 501.0, 502.0):
        harness.broker.tick("SPY", price)
    harness.broker.tick("AMD", 600.0)
    await harness.adapter.flush_ticks()
    assert sorted(r.key or "" for r in harness.bus.records(harness.topics.md_ticks)) == ["AMD", "SPY"]
    cached = await harness.state.get_last("SPY")
    assert cached is not None and cached[0] == 502.0
    assert harness.adapter.marks["SPY"] == 502.0


async def test_subscriptions_cover_base_positions_and_watchlist_within_the_line_limit(
    harness: Harness,
) -> None:
    harness.broker.state.positions = {"XOM": 1.0}
    await harness.adapter.startup()
    await harness.state.add_to_watchlist(["AMD", "MU"])
    await harness.adapter.sync_subscriptions()
    assert harness.broker.subscriptions == {"SPY", "XOM", "AMD", "MU"}


async def test_the_line_limit_keeps_base_and_positions_ahead_of_the_watchlist(harness: Harness) -> None:
    harness.cfg = harness.cfg.model_copy(
        update={"adapter": harness.cfg.adapter.model_copy(update={"max_market_data_lines": 2})}
    )
    harness.new_adapter()
    harness.broker.state.positions = {"XOM": 1.0}
    await harness.adapter.startup()
    await harness.state.add_to_watchlist(["AMD", "MU"])
    await harness.adapter.sync_subscriptions()
    assert harness.broker.subscriptions == {"SPY", "XOM"}


async def test_order_log_lines_carry_intent_id_and_order_ref(
    harness: Harness, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    await harness.ready()
    intent = make_intent()
    with caplog.at_level(logging.INFO, logger="eventbot.adapter"):
        await harness.adapter.handle_intent(intent)
        harness.broker.fill(intent.intent_id)
        await harness.adapter.drain_broker_events()
    formatter = JsonFormatter("adapter")
    entries = [json.loads(formatter.format(r)) for r in caplog.records]
    relevant = [e for e in entries if e["event"] in ("order_placed", "fill", "order_status")]
    assert {e["event"] for e in relevant} == {"order_placed", "fill", "order_status"}
    assert all(e["intent_id"] == intent.intent_id and e["order_ref"] == intent.intent_id for e in relevant)


async def test_a_sell_uses_the_broker_position(harness: Harness) -> None:
    harness.broker.state.positions = {"AMD": 4.0}
    await harness.ready()
    sell = make_intent(side=Side.SELL, qty=4, target=0)
    assert await harness.adapter.handle_intent(sell) == "placed"
    assert harness.broker.state.orders[sell.intent_id].side == "SELL"
