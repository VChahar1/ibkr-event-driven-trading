"""Restarts, crashes and reconnects. A restart here is a new adapter (empty memory) over the same broker
state, Redis and bus, which is exactly what survives a real process crash."""

import asyncio

import pytest

from eventbot.adapter.fake_broker import FakeBroker, FakeOrder
from eventbot.ledger import FillLedger
from eventbot.paper_guard import PaperGuardError
from eventbot.schemas import AlertType, IntentStatus
from tests.eventbot.conftest import Harness, make_intent


async def test_restart_while_placing_with_no_order_at_ib_blocks_and_never_places(harness: Harness) -> None:
    """Crash after SET NX, before placeOrder. Missing the order is the safe outcome."""
    intent = make_intent()
    assert await harness.state.claim_intent(intent.intent_id)

    harness.new_adapter()
    await harness.ready()

    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.BLOCKED
    assert AlertType.RECOVERY_BLOCKED in [a.type for a in harness.alerts()]
    assert await harness.adapter.handle_intent(intent) == "duplicate"
    assert harness.broker.state.orders == {}


async def test_restart_while_placing_with_the_order_at_ib_does_not_double_place(harness: Harness) -> None:
    """Crash after placeOrder, before marking placed: the orderRef lookup finds the order."""
    harness.new_adapter(FakeBroker(harness.broker.state, fail_after_send=True))
    await harness.ready()
    intent = make_intent()
    assert await harness.adapter.handle_intent(intent) == "uncertain"
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.PLACING
    assert list(harness.broker.state.orders) == [intent.intent_id]

    harness.new_adapter()
    await harness.ready()

    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.PLACED
    assert AlertType.RECOVERY_ADOPTED in [a.type for a in harness.alerts()]
    assert await harness.adapter.handle_intent(intent) == "duplicate"
    assert harness.broker.placed == []
    assert list(harness.broker.state.orders) == [intent.intent_id]
    assert intent.intent_id in harness.adapter.open_orders


async def test_full_restart_through_the_bus_with_an_uncommitted_intent(harness: Harness) -> None:
    """Delivered, order at IB, crash before mark and before commit, restart, redelivery."""
    harness.new_adapter(FakeBroker(harness.broker.state, fail_after_send=True))
    await harness.ready()
    intent = make_intent()
    await harness.deliver(intent)
    consumer = await harness.intent_consumer()
    await consumer.getone(0.01)
    await harness.adapter.handle_intent(intent)
    await consumer.stop()

    harness.new_adapter()
    await harness.ready()
    restarted = await harness.intent_consumer()
    assert await harness.adapter.process_one_intent(restarted, 0.01) == "duplicate"
    assert list(harness.broker.state.orders) == [intent.intent_id]


@pytest.mark.parametrize("where", ["open", "completed", "executions"])
async def test_the_orderref_lookup_covers_every_place_ib_can_report_it(harness: Harness, where: str) -> None:
    intent = make_intent()
    await harness.state.claim_intent(intent.intent_id)
    broker = harness.broker
    order = FakeOrder(intent.intent_id, "AMD", "BUY", 4)
    broker.state.orders[intent.intent_id] = order
    if where == "completed":
        order.status = "Cancelled"
    elif where == "executions":
        await broker.connect()
        broker.fill_silently(intent.intent_id)
        order.status = "Filled"
        del broker.state.orders[intent.intent_id]

    harness.new_adapter()
    await harness.adapter.startup()
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.PLACED


async def test_placing_failing_before_send_leaves_nothing_at_ib_and_resolves_to_blocked(
    harness: Harness,
) -> None:
    harness.new_adapter(FakeBroker(harness.broker.state, fail_before_send=True))
    await harness.ready()
    intent = make_intent()
    assert await harness.adapter.handle_intent(intent) == "uncertain"
    assert AlertType.ORDER_UNCERTAIN in [a.type for a in harness.alerts()]

    await harness.adapter._refresh_order_knowledge()
    await harness.adapter._resolve_placing()
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.BLOCKED
    assert harness.broker.state.orders == {}


async def test_an_intent_being_placed_right_now_is_not_touched_by_recovery(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.state.claim_intent(intent.intent_id)
    harness.adapter._in_flight.add(intent.intent_id)
    await harness.adapter._resolve_placing()
    assert await harness.state.intent_status(intent.intent_id) is IntentStatus.PLACING


async def test_fills_missed_while_the_adapter_was_down_are_published_on_restart(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    execution = harness.broker.fill_silently(intent.intent_id)

    harness.new_adapter()
    await harness.adapter.startup()

    fills = harness.fills()
    assert [f.exec_id for f in fills] == [execution.exec_id] and fills[0].late
    assert harness.adapter.positions["AMD"] == 4


async def test_a_restart_republishes_todays_fills_and_the_ledger_counts_them_once(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.fill(intent.intent_id)
    await harness.adapter.drain_broker_events()

    harness.new_adapter()
    await harness.adapter.startup()

    fills = harness.fills()
    assert len(fills) == 2 and fills[0].exec_id == fills[1].exec_id
    ledger = FillLedger()
    for fill in fills:
        ledger.apply(fill)
    assert ledger.position("AMD") == 4


async def test_periodic_snapshots_catch_a_fill_that_fired_no_event(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.fill_silently(intent.intent_id)
    await harness.adapter.publish_snapshot()
    assert len(harness.fills()) == 1 and harness.fills()[0].late
    assert harness.snapshots()[-1].positions == {"AMD": 4.0}


async def test_reconnect_recovers_missed_fills_and_verifies_the_account_again(harness: Harness) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.drop_connection()
    assert not harness.adapter.connected
    harness.broker.fill_silently(intent.intent_id)

    await harness.adapter.reconnect()

    assert harness.adapter.connected
    assert [f.late for f in harness.fills()] == [True]
    assert AlertType.BROKER_DISCONNECT in [a.type for a in harness.alerts()]
    assert harness.snapshots()[-1].kind == "startup"


async def test_reconnect_retries_with_backoff_until_the_gateway_is_back(harness: Harness) -> None:
    harness.cfg = harness.cfg.model_copy(
        update={"adapter": harness.cfg.adapter.model_copy(update={"reconnect_delays_s": [0.001]})}
    )
    harness.new_adapter()
    await harness.adapter.startup()
    harness.broker.connect_failures = 3
    harness.broker.drop_connection()
    await asyncio.wait_for(harness.adapter.reconnect(), 2.0)
    assert harness.adapter.connected and harness.broker.is_connected()


async def test_startup_refuses_a_live_account_and_disconnects(harness: Harness) -> None:
    harness.broker.state.accounts = ["U" + "1234567"]
    with pytest.raises(PaperGuardError):
        await harness.adapter.startup()
    assert not harness.broker.is_connected() and not harness.adapter.connected
    assert harness.snapshots() == []


async def test_reconnect_refuses_if_the_gateway_has_switched_to_a_live_account(harness: Harness) -> None:
    await harness.ready()
    harness.broker.drop_connection()
    harness.broker.state.accounts = ["U" + "1234567"]
    with pytest.raises(PaperGuardError):
        await harness.adapter.reconnect()
    assert not harness.broker.is_connected()


async def test_nothing_is_consumed_before_recovery_has_finished(harness: Harness) -> None:
    """run() starts the intent consumer only after startup(), so a live-account refusal consumes nothing."""
    harness.broker.state.accounts = ["U" + "1234567"]
    await harness.deliver(make_intent())
    with pytest.raises(PaperGuardError):
        await harness.adapter.run()
    assert harness.bus.committed("adapter", harness.topics.orders_intents, 0) is None
    assert harness.broker.state.orders == {}


async def test_run_loop_end_to_end_then_stop(harness: Harness) -> None:
    """The real concurrent loops, not the single-step helpers."""
    harness.broker.auto_fill = True
    harness.broker.state.prices["AMD"] = 600.0
    task = asyncio.create_task(harness.adapter.run())
    await asyncio.sleep(0.05)
    harness.broker.tick("AMD", 600.0)
    await harness.fresh_beats()

    intent = make_intent()
    await harness.deliver(intent)
    for _ in range(100):
        if harness.fills():
            break
        await asyncio.sleep(0.02)

    harness.adapter.request_stop()
    await asyncio.wait_for(task, 5.0)
    assert list(harness.broker.state.orders) == [intent.intent_id]
    assert [f.order_ref for f in harness.fills()] == [intent.intent_id]
    assert harness.bus.committed("adapter", harness.topics.orders_intents, 0) == 1
    assert not harness.broker.is_connected()


async def test_a_paper_guard_failure_on_reconnect_stops_the_run_loop(harness: Harness) -> None:
    task = asyncio.create_task(harness.adapter.run())
    await asyncio.sleep(0.05)
    harness.broker.state.accounts = ["U" + "1234567"]
    harness.broker.drop_connection()
    with pytest.raises(PaperGuardError):
        await asyncio.wait_for(task, 5.0)
