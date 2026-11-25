"""Reconciliation: the comparison, cold start, the halt, and the heartbeat the adapter depends on."""

import asyncio
import datetime as dt
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import fakeredis
import pytest
from fakeredis import FakeAsyncRedis

from eventbot.bus import InMemoryBus, TopicMap, publish
from eventbot.config import EventbotConfig, load_config
from eventbot.reconciler import Halted, Reconciler, build_baseline
from eventbot.schemas import Alert, AlertType, Fill, OrderIntent, PositionSnapshot, Side, utc_now
from eventbot.state import StateStore

REF = "a" * 32


def recent(seconds_ago: float = 0.0) -> dt.datetime:
    return utc_now() - dt.timedelta(seconds=seconds_ago)


@dataclass
class Env:
    cfg: EventbotConfig
    bus: InMemoryBus
    state: StateStore
    topics: TopicMap
    service: Reconciler

    def alerts(self) -> list[Alert]:
        return [Alert.model_validate_json(r.value) for r in self.bus.records(self.topics.alerts)]

    def new_service(self) -> Reconciler:
        self.service = Reconciler(self.cfg, self.bus, self.state, self.topics)
        return self.service

    async def fill(
        self, exec_id: str, qty: float = 4, side: Side = Side.BUY, symbol: str = "AMD", ago: float = 60.0
    ) -> None:
        message = Fill(
            exec_id=exec_id,
            order_ref=REF,
            symbol=symbol,
            side=side,
            qty=qty,
            price=100.0,
            exec_time_utc=recent(ago),
            ts_utc=recent(ago),
        )
        await publish(self.bus, self.topics.fills, exec_id, message)

    async def broker(
        self, positions: dict[str, float], exec_ids: list[str] | None = None, ago: float = 0.0, **over: Any
    ) -> None:
        message = PositionSnapshot(
            account="DU***000",
            kind=over.pop("kind", "periodic"),
            positions=positions,
            exec_ids=exec_ids or [],
            ts_utc=recent(ago),
            **over,
        )
        await publish(self.bus, self.topics.positions_snapshot, "DU***000", message)


@pytest.fixture
async def env() -> AsyncIterator[Env]:
    cfg = load_config()
    bus = InMemoryBus()
    client = FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    state = StateStore(client)
    topics = TopicMap.from_config(cfg.topics)
    service = Reconciler(cfg, bus, state, topics)
    await service.start()
    yield Env(cfg, bus, state, topics, service)
    await client.aclose()


async def test_fills_that_match_the_broker_reconcile(env: Env) -> None:
    await env.fill("e.1.01", 3)
    await env.fill("e.2.01", 1)
    await env.broker({"AMD": 4.0}, ["e.1.01", "e.2.01"])
    assert (await env.service.check()).status == "ok"
    assert env.service.verified and not env.service.halted
    assert await env.state.kill_switch() is None


async def test_a_flat_account_with_no_fills_reconciles(env: Env) -> None:
    await env.broker({})
    assert (await env.service.check()).status == "ok"


async def test_duplicate_fills_do_not_cause_a_false_mismatch(env: Env) -> None:
    await env.fill("e.1.01", 4)
    await env.fill("e.1.01", 4)
    await env.broker({"AMD": 4.0}, ["e.1.01"])
    assert (await env.service.check()).status == "ok"


async def test_a_fill_is_written_to_the_position_cache(env: Env) -> None:
    await env.fill("e.1.01", 4)
    await env.service.apply_available()
    assert await env.state.get_pos("AMD") == 4.0
    await env.fill("e.2.01", 4, Side.SELL)
    await env.service.apply_available()
    assert await env.state.get_pos("AMD") == 0.0


async def test_a_correction_alerts_and_updates_the_position(env: Env) -> None:
    await env.fill("e.1.01", 5)
    await env.fill("e.1.02", 3)
    await env.service.apply_available()
    assert env.service.ledger.position("AMD") == 3
    assert AlertType.EXEC_CORRECTION in [a.type for a in env.alerts()]


@pytest.mark.parametrize(
    "fills,broker,exec_ids",
    [
        ([("e.1.01", 4.0)], {"AMD": 5.0}, ["e.1.01"]),
        ([("e.1.01", 4.0)], {"AMD": 3.0}, ["e.1.01"]),
        ([("e.1.01", 4.0)], {}, ["e.1.01"]),
        ([], {"AMD": 4.0}, []),
        ([("e.1.01", 4.0)], {"AMD": 4.0, "MU": 1.0}, ["e.1.01"]),
    ],
)
async def test_any_divergence_sets_the_kill_switch_alerts_and_halts(
    env: Env, fills: list, broker: dict, exec_ids: list
) -> None:
    for exec_id, qty in fills:
        await env.fill(exec_id, qty)
    await env.broker(broker, exec_ids)
    with pytest.raises(Halted):
        await env.service.check()
    assert env.service.halted and not env.service.verified
    assert (await env.state.kill_switch() or "").startswith("reconcile mismatch")
    critical = [a for a in env.alerts() if a.type is AlertType.RECONCILE_MISMATCH]
    assert len(critical) == 1 and critical[0].severity == "critical"
    assert "diffs" in critical[0].details


async def test_a_mismatch_never_flattens_it_publishes_no_intents_at_all(env: Env) -> None:
    await env.broker({"AMD": 100.0, "MU": 50.0})
    with pytest.raises(Halted):
        await env.service.check()
    assert env.bus.records(env.topics.orders_intents) == []
    assert not hasattr(env.service, "_broker")


async def test_a_lost_fill_is_a_mismatch_even_when_positions_agree(env: Env) -> None:
    """Two fills lost that happen to cancel out would otherwise go unnoticed."""
    await env.fill("e.1.01", 4)
    await env.broker({"AMD": 4.0}, ["e.1.01", "lost.1.01"])
    with pytest.raises(Halted):
        await env.service.check()
    alert_ = [a for a in env.alerts() if a.type is AlertType.RECONCILE_MISMATCH][0]
    assert alert_.details["missing_exec_ids"] == ["lost.1.01"]


async def test_a_fill_published_after_a_snapshot_is_read_before_comparing(env: Env) -> None:
    """Fills and snapshots are different topics: arrival order between them means nothing."""
    await env.broker({"AMD": 4.0}, ["e.1.01"])
    await env.service.apply_available()
    await env.fill("e.1.01", 4)
    assert (await env.service.check()).status == "ok"


async def test_a_very_recent_fill_defers_the_comparison_instead_of_halting(env: Env) -> None:
    """The broker's position report can lag a fill by a moment. That is not a divergence."""
    await env.fill("e.1.01", 4, ago=1.0)
    await env.broker({}, ["e.1.01"])
    assert (await env.service.check()).status == "unsettled"
    assert not env.service.halted and await env.state.kill_switch() is None

    await env.broker({"AMD": 4.0}, ["e.1.01"])
    env.service._rcfg = env.cfg.reconciler.model_copy(update={"settle_s": 0.0})
    assert (await env.service.check()).status == "ok"


async def test_a_settled_mismatch_after_a_recent_fill_still_halts(env: Env) -> None:
    await env.fill("e.1.01", 4, ago=60.0)
    await env.broker({}, ["e.1.01"])
    with pytest.raises(Halted):
        await env.service.check()


async def test_a_republished_old_fill_does_not_reset_the_settling_clock(env: Env) -> None:
    """An adapter restart republishes the day's fills. That must not postpone reconciliation forever."""
    await env.fill("e.1.01", 4, ago=60.0)
    await env.service.apply_available()
    await env.fill("e.1.01", 4, ago=0.0)
    await env.broker({"AMD": 4.0}, ["e.1.01"])
    assert (await env.service.check()).status == "ok"


async def test_no_snapshot_is_stale_not_a_divergence(env: Env) -> None:
    assert (await env.service.check()).status == "stale"
    assert not env.service.halted and await env.state.kill_switch() is None


async def test_an_old_snapshot_stops_vouching_and_alerts_once(env: Env) -> None:
    await env.broker({})
    assert (await env.service.check()).status == "ok"
    assert env.service.verified

    await env.broker({}, ago=1000.0)
    assert (await env.service.check()).status == "stale"
    assert (await env.service.check()).status == "stale"
    assert not env.service.verified
    assert [a.type for a in env.alerts()].count(AlertType.RECONCILE_STALE) == 1


async def test_cold_start_rebuilds_from_fill_history_not_from_redis(env: Env) -> None:
    await env.fill("e.1.01", 4)
    await env.state.set_pos("AMD", 999.0)
    await env.state.set_pos("GHOST", 1.0)
    await env.broker({"AMD": 4.0}, ["e.1.01"])
    await env.service.cold_start()
    assert env.service.ledger.positions == {"AMD": 4.0}
    assert await env.state.get_pos("AMD") == 4.0
    assert await env.state.heartbeat_age_s("reconciler") is not None


async def test_cold_start_halts_on_a_mismatch_and_never_writes_a_heartbeat(env: Env) -> None:
    await env.fill("e.1.01", 4)
    await env.broker({"AMD": 7.0}, ["e.1.01"])
    with pytest.raises(Halted):
        await env.service.cold_start()
    assert await env.state.heartbeat_age_s("reconciler") is None
    assert await env.state.kill_switch() is not None


async def test_no_heartbeat_until_the_cold_start_check_has_passed(env: Env) -> None:
    task = asyncio.create_task(env.service.cold_start())
    await asyncio.sleep(0.05)
    assert await env.state.heartbeat_age_s("reconciler") is None
    await env.broker({})
    await asyncio.wait_for(task, 3.0)
    assert await env.state.heartbeat_age_s("reconciler") is not None


async def test_a_non_flat_account_with_no_baseline_halts(env: Env) -> None:
    await env.broker({"XOM": 10.0})
    with pytest.raises(Halted):
        await env.service.cold_start()


async def test_cold_start_after_a_restart_gets_the_same_positions(env: Env) -> None:
    await env.fill("e.1.01", 4)
    await env.fill("e.2.01", 1, Side.SELL)
    await env.broker({"AMD": 3.0}, ["e.1.01", "e.2.01"])
    await env.service.check()
    await env.service.stop()

    restarted = env.new_service()
    await restarted.start()
    await restarted.cold_start()
    assert restarted.ledger.positions == {"AMD": 3.0}


async def test_a_baseline_explains_positions_that_came_from_elsewhere(env: Env) -> None:
    await env.broker({"XOM": 10.0}, foreign_exec_ids=["old.1.01"])
    await env.service.apply_available()
    assert env.service.snapshot is not None
    baseline = build_baseline(env.service.snapshot, env.service.ledger.exec_ids)
    assert baseline.kind == "baseline" and baseline.baseline_exec_ids == ["old.1.01"]
    await publish(env.bus, env.topics.positions_snapshot, "DU***000", baseline)
    await env.broker({"XOM": 10.0})
    assert (await env.service.check()).status == "ok"
    assert env.service.ledger.positions == {"XOM": 10.0}


async def test_fills_after_a_baseline_add_to_it(env: Env) -> None:
    await env.broker({"XOM": 10.0})
    await env.service.apply_available()
    assert env.service.snapshot is not None
    await publish(
        env.bus,
        env.topics.positions_snapshot,
        "DU***000",
        build_baseline(env.service.snapshot, set()),
    )
    await env.fill("e.1.01", 2, Side.SELL, symbol="XOM")
    await env.broker({"XOM": 8.0}, ["e.1.01"])
    assert (await env.service.check()).status == "ok"


async def test_a_baseline_excludes_fills_it_already_includes_even_after_a_restart(env: Env) -> None:
    await env.fill("e.1.01", 4)
    await env.broker({"AMD": 9.0}, ["e.1.01"])
    await env.service.apply_available()
    assert env.service.snapshot is not None
    baseline = build_baseline(env.service.snapshot, env.service.ledger.exec_ids)
    await publish(env.bus, env.topics.positions_snapshot, "DU***000", baseline)
    await env.broker({"AMD": 9.0}, ["e.1.01"])

    restarted = env.new_service()
    await restarted.start()
    assert (await restarted.check()).status == "ok"
    assert restarted.ledger.positions == {"AMD": 9.0}


async def test_the_run_loop_keeps_the_heartbeat_alive_then_stops(env: Env) -> None:
    await env.broker({})
    await env.state._r.delete("hb:reconciler")
    task = asyncio.create_task(env.service.run())
    await asyncio.sleep(0.1)
    assert await env.state.heartbeat_age_s("reconciler") is not None
    env.service.request_stop()
    await asyncio.wait_for(task, 3.0)


async def test_the_run_loop_finds_a_later_divergence_and_stops_beating(env: Env) -> None:
    fast = env.cfg.reconciler.model_copy(update={"interval_minutes": 0.001})
    env.service._rcfg = fast
    await env.broker({})
    task = asyncio.create_task(env.service.run())
    await asyncio.sleep(0.1)
    await env.broker({"AMD": 1.0})
    with pytest.raises(Halted):
        await asyncio.wait_for(task, 5.0)
    assert await env.state.kill_switch() is not None


async def test_an_invalid_message_alerts_and_does_not_stop_reconciling(env: Env) -> None:
    await env.bus.publish_raw(env.topics.fills, "x", b"garbage")
    await env.broker({})
    assert (await env.service.check()).status == "ok"
    assert AlertType.INVALID_MESSAGE in [a.type for a in env.alerts()]


async def test_an_intent_is_refused_after_a_halt(env: Env) -> None:
    """The whole chain, with the intent going through the real pre-trade check."""
    from eventbot.risk import PreTradeContext, RejectCode, check

    await env.broker({"AMD": 1.0})
    with pytest.raises(Halted):
        await env.service.check()
    now = utc_now()
    intent = OrderIntent.build(
        strategy="s",
        trading_date=now.date(),
        symbol="AMD",
        side=Side.BUY,
        target_qty=1,
        qty=1,
        est_price=10.0,
        decision_id="d",
    )
    ctx = PreTradeContext(
        now=now,
        today=now.date(),
        connected=True,
        kill_switch=await env.state.kill_switch(),
        stop_file=False,
        md_heartbeat_age_s=1.0,
        reconciler_heartbeat_age_s=await env.state.heartbeat_age_s("reconciler"),
        market_open=True,
        clock_skew_s=0.0,
        price=10.0,
    )
    codes = {r.code for r in check(intent, ctx, env.cfg.caps, env.cfg.adapter)}
    assert {RejectCode.KILL_SWITCH, RejectCode.RECONCILER_NOT_RUNNING} <= codes
