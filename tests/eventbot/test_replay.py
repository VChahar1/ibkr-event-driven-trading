"""Replay: a session recorded from the real services must reproduce the same final positions."""

import ast
import json
from pathlib import Path

import pandas as pd

from eventbot import replay as replay_module
from eventbot.recorder import Recorder, load_session
from eventbot.replay import replay_decision, replay_session
from eventbot.schemas import Side
from tests.eventbot.conftest import Harness, make_intent, with_caps
from tests.eventbot.test_strategy_service import (  # noqa: F401  (env is a fixture)
    STOCK_CFG,
    TRADING,
    Env,
    env,
    ready,
)


async def record_session(h: Harness, directory: Path) -> Recorder:
    """A messy but realistic session: partial fills, a duplicate report, a re-delivered intent, a kill switch
    refusal, a cap rejection, a sell, a broker cancel, and an adapter restart that republishes fills."""
    h.cfg = with_caps(h.cfg, max_order_notional_usd=2_000.0)
    h.new_adapter()
    recorder = Recorder(h.cfg, h.bus, h.state, h.topics, directory)
    await recorder.start()
    await h.ready(price=100.0)
    h.broker.tick("MU", 50.0)
    h.broker.tick("XOM", 80.0)
    h.broker.tick("NVDA", 500.0)
    await h.adapter.flush_ticks()

    consumer = await h.intent_consumer()

    async def send(intent: object) -> None:
        await h.deliver(intent)  # type: ignore[arg-type]
        await h.adapter.process_one_intent(consumer, 0.01)

    buy_amd = make_intent("AMD", qty=10, target=10, est_price=100.0)
    await send(buy_amd)
    h.broker.fill(buy_amd.intent_id, qty=3, price=100.0)
    execution = h.broker.fill(buy_amd.intent_id, qty=3, price=100.5)
    h.broker.report_execution(execution)
    h.broker.fill(buy_amd.intent_id, qty=4, price=101.0)
    await h.adapter.drain_broker_events()

    await send(buy_amd)

    buy_mu = make_intent("MU", qty=5, target=5, est_price=50.0)
    await send(buy_mu)
    h.broker.fill(buy_mu.intent_id, price=50.0)
    await h.adapter.drain_broker_events()

    await h.state.set_kill_switch("test")
    refused = make_intent("XOM", qty=1, target=1, est_price=80.0)
    await send(refused)
    await h.state.clear_kill_switch()

    over_cap = make_intent("NVDA", qty=10, target=10, est_price=500.0)
    await send(over_cap)

    sell = make_intent("AMD", Side.SELL, qty=2, target=8, est_price=100.0, strategy="other")
    await h.adapter.publish_snapshot()
    await send(sell)
    h.broker.fill(sell.intent_id, price=100.0)
    await h.adapter.drain_broker_events()

    cancelled = make_intent("XOM", qty=2, target=2, est_price=80.0, strategy="other")
    await send(cancelled)
    await h.broker.cancel(cancelled.intent_id)
    await h.adapter.drain_broker_events()

    h.new_adapter()
    await h.ready(price=100.0)
    await h.adapter.publish_snapshot()

    await recorder.process_available(0.01)
    await recorder.flush()
    return recorder


async def test_replaying_a_recorded_session_reproduces_the_same_final_positions(
    harness: Harness, tmp_path: Path
) -> None:
    directory = tmp_path / "pq"
    await record_session(harness, directory)
    assert harness.broker.state.positions == {"AMD": 8.0, "MU": 5.0}

    messages = load_session(directory)
    report = await replay_session(messages, harness.cfg)

    assert report.replayed_positions == {"AMD": 8.0, "MU": 5.0}
    assert report.recorded_positions == report.replayed_positions
    assert report.last_broker_snapshot == report.replayed_positions
    assert report.position_diffs == {} and report.snapshot_diffs == {}
    assert report.matches, report


async def test_replay_makes_the_same_placement_decisions(harness: Harness, tmp_path: Path) -> None:
    """Not just the same positions: the same intents become orders, and the refused ones are refused again."""
    await record_session(harness, tmp_path / "pq")
    report = await replay_session(load_session(tmp_path / "pq"), harness.cfg)
    assert report.order_diffs == set()
    assert len(report.replayed_orders) == 4


async def test_replay_is_deterministic(harness: Harness, tmp_path: Path) -> None:
    await record_session(harness, tmp_path / "pq")
    messages = load_session(tmp_path / "pq")
    first, second = await replay_session(messages, harness.cfg), await replay_session(messages, harness.cfg)
    assert first.replayed_positions == second.replayed_positions
    assert first.replayed_orders == second.replayed_orders


async def test_a_lost_fill_in_the_recording_is_reported_not_hidden(harness: Harness, tmp_path: Path) -> None:
    await record_session(harness, tmp_path / "pq")
    messages = load_session(tmp_path / "pq")
    exec_ids = {m.key for m in messages if m.topic == harness.topics.fills}
    victim = sorted(k for k in exec_ids if k)[0]
    tampered = [m for m in messages if not (m.topic == harness.topics.fills and m.key == victim)]
    report = await replay_session(tampered, harness.cfg)
    assert not report.matches and report.snapshot_diffs


async def test_a_duplicate_intent_in_the_recording_places_one_order(harness: Harness, tmp_path: Path) -> None:
    await record_session(harness, tmp_path / "pq")
    messages = load_session(tmp_path / "pq")
    intents = [m for m in messages if m.topic == harness.topics.orders_intents]
    keys = [m.key for m in intents]
    assert len(keys) > len(set(keys))
    report = await replay_session(messages, harness.cfg)
    assert len(report.replayed_orders) == 4


async def test_replay_reports_throughput(harness: Harness, tmp_path: Path) -> None:
    await record_session(harness, tmp_path / "pq")
    report = await replay_session(load_session(tmp_path / "pq"), harness.cfg)
    assert report.messages > 10 and report.seconds > 0 and report.messages_per_second > 0


async def test_an_empty_recording_is_trivially_consistent(harness: Harness) -> None:
    report = await replay_session([], harness.cfg)
    assert report.matches and report.messages == 0


async def test_replay_starts_from_a_recorded_baseline(harness: Harness, tmp_path: Path) -> None:
    from eventbot.bus import publish
    from eventbot.reconciler import build_baseline

    harness.broker.state.positions = {"XOM": 10.0}
    await harness.ready(price=100.0)
    harness.broker.tick("XOM", 80.0)
    snapshot = harness.snapshots()[-1]
    await publish(
        harness.bus, harness.topics.positions_snapshot, snapshot.account, build_baseline(snapshot, set())
    )
    recorder = Recorder(harness.cfg, harness.bus, harness.state, harness.topics, tmp_path / "pq")
    await recorder.start()
    await harness.adapter.flush_ticks()
    intent = make_intent("AMD", qty=2, target=2, est_price=100.0)
    await harness.deliver(intent)
    assert await harness.adapter.handle_intent(intent) == "placed"
    harness.broker.fill(intent.intent_id, price=100.0)
    await harness.adapter.drain_broker_events()
    await harness.adapter.publish_snapshot()
    await recorder.process_available(0.01)
    await recorder.flush()

    report = await replay_session(load_session(tmp_path / "pq"), harness.cfg)
    assert report.replayed_positions == {"XOM": 10.0, "AMD": 2.0}
    assert report.matches, report


def test_replay_cannot_reach_kafka_or_ib() -> None:
    """Replay must never be able to touch live topics or IB, whatever its arguments."""
    tree = ast.parse(Path(replay_module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
    assert not {m for m in modules if "kafka" in m or "ib_client" in m or "ib_async" in m or "runtime" in m}


async def test_replay_leaves_the_callers_bus_and_redis_alone(harness: Harness, tmp_path: Path) -> None:
    await record_session(harness, tmp_path / "pq")
    before = {t: len(harness.bus.records(t)) for t in harness.topics.all()}
    placed = list(harness.broker.state.orders)
    await replay_session(load_session(tmp_path / "pq"), harness.cfg)
    assert {t: len(harness.bus.records(t)) for t in harness.topics.all()} == before
    assert list(harness.broker.state.orders) == placed


async def test_a_frozen_decision_reproduces_its_intent_ids(env: Env) -> None:  # noqa: F811
    await ready(env, positions={"S01": 40.0, "S02": 40.0}, cash=20_000.0)
    await env.run_with_fake_adapter()
    marker = env.marker()
    frame = env.root / env.cfg.strategy.state_dir / "decisions" / f"stocks-{env.day:%Y%m%d}.parquet"
    prices = pd.read_parquet(frame)

    check = replay_decision(marker, prices, STOCK_CFG, TRADING["risk"], 10, 0.15, "stocks", env.day)

    assert check.prices_hash_ok, "the frozen frame must hash to what the marker recorded"
    assert check.recorded == check.recomputed and len(check.recorded) > 0
    assert check.matches


async def test_a_revised_price_history_is_detected(env: Env) -> None:  # noqa: F811
    await ready(env)
    await env.run_with_fake_adapter()
    marker = env.marker()
    revised = env.prices.copy()
    revised.iloc[5, 0] *= 1.001
    check = replay_decision(marker, revised, STOCK_CFG, TRADING["risk"], 10, 0.15, "stocks", env.day)
    assert not check.prices_hash_ok and not check.matches


async def test_the_marker_is_valid_json_with_the_frozen_inputs(env: Env) -> None:  # noqa: F811
    await ready(env)
    await env.run_with_fake_adapter()
    path = env.root / env.cfg.strategy.state_dir / f"decided-stocks-{env.day:%Y%m%d}.json"
    stored = json.loads(path.read_text())
    assert set(stored["inputs"]) == {"prices_sha256", "positions", "usd_cash", "ibkr_prices"}
