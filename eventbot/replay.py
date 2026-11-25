"""Replay: feed a recorded session back through the real adapter and reconciler, and check that it
ends in the same positions."""

import argparse
import asyncio
import datetime as dt
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fakeredis
import pandas as pd
from fakeredis import FakeAsyncRedis

from eventbot.adapter.fake_broker import FakeBroker, FakeBrokerState
from eventbot.adapter.service import GROUP as ADAPTER_GROUP
from eventbot.adapter.service import GatewayAdapter
from eventbot.bus import InMemoryBus, TopicMap, publish
from eventbot.config import ROOT, EventbotConfig, Settings, load_config
from eventbot.decision import (
    failed_price_checks,
    held_symbols,
    make_intents,
    plan_stock_orders,
    price_frame_hash,
    wanted_symbols,
)
from eventbot.ledger import FillLedger, diff_positions
from eventbot.recorder import RecordedMessage, load_session
from eventbot.risk import BLOCKING
from eventbot.schemas import Fill, OrderEvent, OrderIntent, PositionSnapshot, Tick, utc_now
from eventbot.state import StateStore
from stock_strategy import StockConfig, decide

DONE_WITHOUT_FILL = frozenset({"Cancelled", "ApiCancelled", "Inactive"})
UNBOUNDED_CAPS = {
    "max_orders_per_day": 10**9,
}


@dataclass
class ReplayReport:
    messages: int = 0
    seconds: float = 0.0
    recorded_positions: dict[str, float] = field(default_factory=dict)
    replayed_positions: dict[str, float] = field(default_factory=dict)
    last_broker_snapshot: dict[str, float] | None = None
    recorded_orders: set[str] = field(default_factory=set)
    replayed_orders: set[str] = field(default_factory=set)
    position_diffs: dict[str, dict[str, float]] = field(default_factory=dict)
    snapshot_diffs: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def order_diffs(self) -> set[str]:
        return self.recorded_orders ^ self.replayed_orders

    @property
    def matches(self) -> bool:
        return not (self.position_diffs or self.snapshot_diffs or self.order_diffs)

    @property
    def messages_per_second(self) -> float:
        return self.messages / self.seconds if self.seconds else 0.0


class _ReplayClock:
    """Time inside the recording. Pre-trade checks compare intents against now, so now has to follow."""

    def __init__(self) -> None:
        self.now = utc_now()

    def __call__(self) -> dt.datetime:
        return self.now


class _RecordedTimeBroker(FakeBroker):
    """FakeBroker whose server clock is the recording's, so the clock-skew check sees no skew."""

    def __init__(self, state: FakeBrokerState, clock: _ReplayClock) -> None:
        super().__init__(state)
        self._replay_clock = clock

    async def server_time(self) -> dt.datetime:
        return self._replay_clock()


def _parse(topics: TopicMap, messages: list[RecordedMessage]) -> list[Any]:
    return [
        topics.model_for(m.topic).model_validate_json(m.payload) for m in messages if m.topic in topics.all()
    ]


def _opening_positions(parsed: list[Any]) -> tuple[dict[str, float], list[str]]:
    """The position the session started from: the last baseline if there is one, else the first snapshot,
    which is the broker's state before anything in this session traded."""
    baselines = [m for m in parsed if isinstance(m, PositionSnapshot) and m.kind == "baseline"]
    if baselines:
        return dict(baselines[-1].positions), list(baselines[-1].baseline_exec_ids)
    for message in parsed:
        if isinstance(message, PositionSnapshot):
            return dict(message.positions), list(message.exec_ids)
    return {}, []


def _recorded_outcomes(parsed: list[Any]) -> tuple[set[str], dict[str, str]]:
    """Which intents really became orders, and the reason each refused one gave."""
    placed: set[str] = set()
    refused: dict[str, str] = {}
    for message in parsed:
        if isinstance(message, OrderEvent):
            if message.status == "Rejected":
                refused[message.order_ref] = message.reason
            else:
                placed.add(message.order_ref)
    return placed, refused


def _blocked_by_system_state(reason: str) -> bool:
    """Refused because of system state at the time (kill switch, stale data), which topics do not record
    directly. The replay reproduces it, rather than reporting a difference that is not one."""
    return any(code.value in reason for code in BLOCKING)


async def replay_session(
    messages: list[RecordedMessage], cfg: EventbotConfig | None = None, root: Path | None = None
) -> ReplayReport:
    """Run a recording through a fresh adapter and report whether it ends in the same place."""
    base = cfg or load_config()

    config = base.model_copy(update={"caps": base.caps.model_copy(update=UNBOUNDED_CAPS)})
    topics = TopicMap.from_config(config.topics)
    parsed = _parse(topics, messages)
    opening, opening_execs = _opening_positions(parsed)
    recorded_placed, refused = _recorded_outcomes(parsed)

    recorded_ledger = FillLedger(opening, opening_execs)
    last_snapshot: dict[str, float] | None = None
    for message in parsed:
        if isinstance(message, Fill):
            recorded_ledger.apply(message)
        elif isinstance(message, PositionSnapshot) and message.kind != "baseline":
            last_snapshot = dict(message.positions)

    clock = _ReplayClock()
    first_ts = next((m.ts_utc for m in parsed), None)
    if first_ts is not None:
        clock.now = first_ts
    bus = InMemoryBus()
    client = FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    state = StateStore(client, config.heartbeats.ttl_s)
    broker_state = FakeBrokerState(positions=dict(opening))
    broker = _RecordedTimeBroker(broker_state, clock)
    with tempfile.TemporaryDirectory() as scratch:
        adapter = GatewayAdapter(
            config,
            Settings(),
            bus,
            state,
            broker,
            topics,
            fill_timeout_s=float("inf"),
            clock=clock,
            root=root or Path(scratch),
        )
        await adapter.startup()
        consumer = bus.consumer([topics.orders_intents], ADAPTER_GROUP)
        await consumer.start()

        started = time.perf_counter()
        for message in parsed:
            clock.now = message.ts_utc
            await _play(message, adapter, broker, bus, state, topics, consumer, refused)
        await adapter.drain_broker_events()
        elapsed = time.perf_counter() - started

        ledger = FillLedger(opening, opening_execs)
        for record in bus.records(topics.fills):
            ledger.apply(Fill.model_validate_json(record.value))
        replayed_positions = ledger.positions
        replayed_orders = set(broker.placed)
        await consumer.stop()
    await client.aclose()

    report = ReplayReport(
        messages=len(parsed),
        seconds=elapsed,
        recorded_positions=recorded_ledger.positions,
        replayed_positions=replayed_positions,
        last_broker_snapshot=last_snapshot,
        recorded_orders=recorded_placed,
        replayed_orders=replayed_orders,
    )
    report.position_diffs = diff_positions(report.recorded_positions, report.replayed_positions)
    if last_snapshot is not None:
        report.snapshot_diffs = diff_positions(last_snapshot, report.replayed_positions)
    return report


async def _play(
    message: Any,
    adapter: GatewayAdapter,
    broker: FakeBroker,
    bus: InMemoryBus,
    state: StateStore,
    topics: TopicMap,
    consumer: Any,
    refused: dict[str, str],
) -> None:
    if isinstance(message, Tick):
        broker.tick(message.symbol, message.price, message.md_type)
    elif isinstance(message, OrderIntent):
        await state.beat("md")
        await state.beat("reconciler")
        adapter.market_open = True
        adapter.clock_skew_s = 0.0
        await adapter.drain_broker_events()
        blocked = _blocked_by_system_state(refused.get(message.intent_id, ""))
        if blocked:
            await state.set_kill_switch("reproducing a recorded refusal")
        await publish(bus, topics.orders_intents, message.intent_id, message)
        await adapter.process_one_intent(consumer, 0.0)
        if blocked:
            await state.clear_kill_switch()
    elif isinstance(message, Fill):
        await adapter.drain_broker_events()
        if message.exec_id in {e.exec_id for e in broker.state.executions}:
            broker.report_execution(_execution_from(message))
        elif message.order_ref in broker.state.orders:
            broker.fill(message.order_ref, qty=message.qty, price=message.price, exec_id=message.exec_id)
        await adapter.drain_broker_events()
    elif isinstance(message, OrderEvent) and message.status in DONE_WITHOUT_FILL:
        if message.order_ref in broker.state.orders:
            broker.reject(
                message.order_ref,
                message.error_code or 0,
                message.reason,
                from_error=False,
                status=message.status,
            )
            await adapter.drain_broker_events()


def _execution_from(fill: Fill) -> Any:
    from eventbot.adapter.broker import BrokerExecution

    return BrokerExecution(
        exec_id=fill.exec_id,
        order_ref=fill.order_ref,
        symbol=fill.symbol,
        side=fill.side.value,
        qty=fill.qty,
        price=fill.price,
        cum_qty=fill.cum_qty,
        avg_price=fill.avg_price,
        time=fill.exec_time_utc,
        perm_id=fill.perm_id,
    )


@dataclass(frozen=True)
class DecisionCheck:
    decision_id: str
    recorded: list[str]
    recomputed: list[str]
    prices_hash_ok: bool

    @property
    def matches(self) -> bool:
        return self.prices_hash_ok and self.recorded == self.recomputed


def replay_decision(
    marker: dict[str, Any],
    prices: pd.DataFrame,
    stock_cfg: StockConfig,
    risk: dict[str, Any],
    candidates: int,
    gap_limit: float,
    strategy: str,
    today: dt.date,
) -> DecisionCheck:
    """Re-run the sizing from what the decision actually saw: same closes, positions, cash and IBKR prices.
    Same inputs must give the same intent IDs."""
    inputs = marker["inputs"]
    positions = {s: int(q) for s, q in inputs["positions"].items()}
    held = held_symbols(positions)
    dec = decide(prices, stock_cfg, held & set(stock_cfg.universe), today)
    wanted = wanted_symbols(held, dec, stock_cfg, candidates)
    ibkr = {s: float(p) for s, p in inputs["ibkr_prices"].items()}
    bad = failed_price_checks(wanted, ibkr, prices.iloc[-1], gap_limit)
    plan = plan_stock_orders(dec, positions, ibkr, bad, float(inputs["usd_cash"]), risk, stock_cfg)
    intents = make_intents(plan.orders, positions, strategy, today, marker["decision_id"])
    return DecisionCheck(
        decision_id=marker["decision_id"],
        recorded=[i["intent_id"] for i in marker["intents"]],
        recomputed=[i.intent_id for i in intents],
        prices_hash_ok=price_frame_hash(prices) == inputs["prices_sha256"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cfg = load_config()
    parser.add_argument("--dir", type=Path, default=ROOT / cfg.recorder.out_dir)
    parser.add_argument("--date", help="only this date (YYYY-MM-DD)")
    args = parser.parse_args()
    messages = load_session(args.dir, args.date)
    if not messages:
        print(f"Nothing recorded under {args.dir}")
        return 1
    report = asyncio.run(replay_session(messages, cfg))
    print(
        json.dumps(
            {
                "messages": report.messages,
                "seconds": round(report.seconds, 3),
                "messages_per_second": round(report.messages_per_second),
                "recorded_positions": report.recorded_positions,
                "replayed_positions": report.replayed_positions,
                "last_broker_snapshot": report.last_broker_snapshot,
                "orders_recorded": len(report.recorded_orders),
                "orders_replayed": len(report.replayed_orders),
                "position_diffs": report.position_diffs,
                "snapshot_diffs": report.snapshot_diffs,
                "order_diffs": sorted(report.order_diffs),
                "matches": report.matches,
            },
            indent=2,
        )
    )
    return 0 if report.matches else 1


if __name__ == "__main__":
    sys.exit(main())
