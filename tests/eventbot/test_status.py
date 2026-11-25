"""make status: every section, every problem it should flag, and survival when its sources are down."""

import asyncio
import time
from typing import Any

import pytest

from eventbot.bus import publish
from eventbot.bus.memory import InMemoryInspector
from eventbot.schemas import Fill, PositionSnapshot, Side, utc_now
from eventbot.status import StatusReport, collect, render
from tests.eventbot.conftest import Harness, make_intent, today


async def beat_all(h: Harness) -> None:
    for process in ("adapter", "md", "strategy", "reconciler", "recorder"):
        await h.state.beat(process)


async def report_of(h: Harness, inspector: Any = "default", kafka_error: str | None = None) -> StatusReport:
    chosen = InMemoryInspector(h.bus) if inspector == "default" else inspector
    return await collect(h.cfg, h.state, chosen, h.topics, today().isoformat(), kafka_error)


async def healthy(h: Harness) -> None:
    """Everything running, flat, with fresh data."""
    await h.adapter.startup()
    h.broker.tick("SPY", 500.0)
    await h.adapter.flush_ticks()
    await beat_all(h)
    h.bus.set_committed("adapter", h.topics.md_ticks, 0, 0)


async def test_a_healthy_system_reports_ok(harness: Harness) -> None:
    await healthy(harness)
    report = await report_of(harness)
    lag_free = {g: {t: 0 for t in harness.topics.all()} for g in ("adapter", "recorder")}
    report.lag = lag_free
    assert report.kill_switch is None and all(a is not None for a in report.heartbeats.values())
    assert report.broker_positions == {} and report.position_diffs == {}
    assert report.newest_tick_age_s is not None and report.newest_tick_age_s < 5


async def test_the_output_answers_every_question_the_brief_asks(harness: Harness) -> None:
    await healthy(harness)
    text = render(await report_of(harness))
    for section in (
        "Kill switch",
        "Heartbeats",
        "Positions",
        "Exposure vs caps",
        "gross",
        "newest tick",
        "Consumer lag",
        "Collected in",
    ):
        assert section in text, section


async def test_it_is_fast(harness: Harness) -> None:
    await healthy(harness)
    started = time.perf_counter()
    report = await report_of(harness)
    assert time.perf_counter() - started < 5 and report.elapsed_s < 5


async def test_the_kill_switch_is_flagged_with_its_reason(harness: Harness) -> None:
    await healthy(harness)
    await harness.state.set_kill_switch("reconcile mismatch")
    report = await report_of(harness)
    assert not report.healthy and "reconcile mismatch" in render(report)


async def test_each_missing_heartbeat_is_named(harness: Harness) -> None:
    await healthy(harness)
    await harness.state._r.delete("hb:md", "hb:recorder")
    report = await report_of(harness)
    assert report.heartbeats["md"] is None and report.heartbeats["recorder"] is None
    assert "No heartbeat from md" in report.problems and "No heartbeat from recorder" in report.problems
    assert "EXPIRED" in render(report)


async def test_positions_that_differ_from_the_broker_are_flagged(harness: Harness) -> None:
    harness.broker.state.positions = {"AMD": 5.0}
    await healthy(harness)
    await harness.state.set_pos("AMD", 3.0)
    report = await report_of(harness)
    assert report.position_diffs == {"AMD": {"internal": 3.0, "broker": 5.0}}
    assert "DIFFERS" in render(report) and not report.healthy


async def test_gross_and_per_symbol_exposure_are_shown_against_their_caps(harness: Harness) -> None:
    harness.broker.state.positions = {"AMD": 100.0}
    await healthy(harness)
    await harness.state.set_pos("AMD", 100.0)
    harness.broker.tick("AMD", 300.0)
    await harness.adapter.flush_ticks()
    report = await report_of(harness)
    assert report.gross_notional == 30_000.0 and report.largest_position == ("AMD", 30_000.0)
    assert any("Gross notional" in p for p in report.problems)
    assert any("per-symbol" in p for p in report.problems)
    assert "30,000 of 25,000" in render(report)


async def test_a_stale_broker_snapshot_is_flagged(harness: Harness) -> None:
    await healthy(harness)
    old = utc_now().replace(year=2020)
    await publish(
        harness.bus,
        harness.topics.positions_snapshot,
        "DU***000",
        PositionSnapshot(account="DU***000", kind="periodic", positions={}, ts_utc=old),
    )
    assert any("snapshot" in p for p in (await report_of(harness)).problems)


async def test_stale_market_data_during_market_hours_is_flagged(harness: Harness) -> None:
    await harness.adapter.startup()
    await beat_all(harness)
    assert "The market is open but market data is stale" in (await report_of(harness)).problems


async def test_stale_market_data_is_fine_when_the_market_is_closed(harness: Harness) -> None:
    harness.broker.state.market_open = False
    await harness.adapter.startup()
    await beat_all(harness)
    assert "The market is open but market data is stale" not in (await report_of(harness)).problems


async def test_consumer_lag_is_reported_per_group_and_flagged_when_large(harness: Harness) -> None:
    await healthy(harness)
    fill = Fill(
        exec_id="e", order_ref="a" * 32, symbol="AMD", side=Side.BUY, qty=1, price=1, exec_time_utc=utc_now()
    )
    for _ in range(150):
        await publish(harness.bus, harness.topics.fills, "e", fill)
    report = await report_of(harness)
    assert report.lag["recorder"][harness.topics.fills] == 150
    assert any("recorder is" in p for p in report.problems)


async def test_an_intent_stuck_in_placing_is_flagged(harness: Harness) -> None:
    await healthy(harness)
    await harness.state.claim_intent(make_intent().intent_id)
    report = await report_of(harness)
    assert report.intent_counts == {"placing": 1}
    assert any("stuck in placing" in p for p in report.problems)


async def test_redis_down_is_reported_not_fatal(harness: Harness) -> None:
    await healthy(harness)
    harness.server.connected = False
    report = await report_of(harness)
    harness.server.connected = True
    assert report.redis_error and not report.healthy
    assert "Redis unavailable" in render(report)


async def test_kafka_down_is_reported_not_fatal(harness: Harness) -> None:
    await healthy(harness)
    report = await report_of(harness, inspector=None, kafka_error="KafkaConnectionError: no broker")
    assert report.kafka_error and not report.healthy
    assert "Kafka unavailable" in render(report)


async def test_kafka_failing_midway_is_reported_not_fatal(harness: Harness) -> None:
    class Broken:
        async def latest(self, topic: str) -> None:
            raise RuntimeError("connection lost")

        async def lag(self, group: str, topics: list[str]) -> dict[str, int]:
            return {}

    await healthy(harness)
    report = await report_of(harness, inspector=Broken())
    assert "connection lost" in (report.kafka_error or "")


async def test_both_down_still_prints_a_useful_answer(harness: Harness) -> None:
    harness.server.connected = False
    report = await report_of(harness, inspector=None, kafka_error="KafkaConnectionError: no broker")
    harness.server.connected = True
    text = render(report)
    assert "Redis unavailable" in text and "Kafka unavailable" in text


async def test_a_hung_source_times_out_rather_than_hanging_the_tool(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("eventbot.status.SECTION_TIMEOUT_S", 0.05)

    class Hung:
        async def latest(self, topic: str) -> None:
            await asyncio.sleep(60)

        async def lag(self, group: str, topics: list[str]) -> dict[str, int]:
            return {}

    await healthy(harness)
    report = await asyncio.wait_for(report_of(harness, inspector=Hung()), 3.0)
    assert report.kafka_error


async def test_the_account_is_masked_in_the_output(harness: Harness) -> None:
    from eventbot.adapter.fake_broker import FAKE_ACCOUNT

    await healthy(harness)
    assert FAKE_ACCOUNT not in render(await report_of(harness))


async def test_unknown_positions_are_not_reported_as_zero_exposure(harness: Harness) -> None:
    harness.server.connected = False
    report = await report_of(harness, inspector=None, kafka_error="no broker")
    harness.server.connected = True
    assert report.gross_notional is None and "gross        unknown" in render(report)
