'make status: one screen answering "is this system healthy, and if not, why?", in well under a minute.'

import asyncio
import datetime as dt
import logging
import sys
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import redis.asyncio as aioredis

from eventbot.bus import Record, TopicMap
from eventbot.bus.base import decode
from eventbot.config import EventbotConfig, load_config, load_settings
from eventbot.ledger import diff_positions
from eventbot.risk import gross_notional
from eventbot.schemas import PositionSnapshot, utc_now
from eventbot.state import StateStore, StateUnavailable

SECTION_TIMEOUT_S = 8.0
PROCESSES = ("adapter", "md", "strategy", "reconciler", "recorder")
GROUPS = ("adapter", "recorder")
LAG_WARNING = 100
T = TypeVar("T")


class BusInspector(Protocol):
    async def latest(self, topic: str) -> Record | None: ...
    async def lag(self, group: str, topics: list[str]) -> dict[str, int]: ...


@dataclass
class StatusReport:
    generated_at: dt.datetime
    elapsed_s: float = 0.0
    redis_error: str | None = None
    kafka_error: str | None = None
    kill_switch: str | None = None
    heartbeats: dict[str, float | None] = field(default_factory=dict)
    internal_positions: dict[str, float] = field(default_factory=dict)
    broker_positions: dict[str, float] | None = None
    usd_cash: float | None = None
    snapshot_age_s: float | None = None
    open_orders: int = 0
    market_open: bool | None = None
    clock_skew_s: float | None = None
    position_diffs: dict[str, dict[str, float]] = field(default_factory=dict)
    gross_notional: float | None = None
    gross_cap: float = 0.0
    largest_position: tuple[str, float] | None = None
    position_cap: float = 0.0
    orders_today: int = 0
    orders_cap: int = 0
    newest_tick_age_s: float | None = None
    tick_symbols: int = 0
    tick_stale_after_s: float = 0.0
    intent_counts: dict[str, int] = field(default_factory=dict)
    lag: dict[str, dict[str, int]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.problems


async def _bounded(call: Awaitable[T]) -> T:
    return await asyncio.wait_for(call, SECTION_TIMEOUT_S)


async def collect(
    cfg: EventbotConfig,
    state: StateStore,
    inspector: BusInspector | None,
    topics: TopicMap,
    today: str,
    kafka_error: str | None = None,
) -> StatusReport:
    started = time.perf_counter()
    report = StatusReport(generated_at=utc_now(), kafka_error=kafka_error)
    report.gross_cap = cfg.caps.max_gross_notional_usd
    report.position_cap = cfg.caps.max_position_notional_usd
    report.orders_cap = cfg.caps.max_orders_per_day
    report.tick_stale_after_s = cfg.heartbeats.md_max_tick_age_s

    last_prices: dict[str, tuple[float, float]] = {}
    try:
        report.kill_switch = await _bounded(state.kill_switch())
        for process in PROCESSES:
            report.heartbeats[process] = await _bounded(state.heartbeat_age_s(process))
        report.internal_positions = {
            s: q for s, q in (await _bounded(state.all_pos())).items() if abs(q) > 1e-9
        }
        last_prices = await _bounded(state.all_last())
        report.orders_today = await _bounded(state.orders_today(today))
        report.intent_counts = await _bounded(state.status_counts())
    except (StateUnavailable, TimeoutError) as exc:
        report.redis_error = str(exc) or "timed out"

    if inspector is not None:
        try:
            record = await _bounded(inspector.latest(topics.positions_snapshot))
            if record is not None:
                _apply_snapshot(report, topics, record)
            for group in GROUPS:
                report.lag[group] = await _bounded(inspector.lag(group, topics.all()))
        except Exception as exc:
            report.kafka_error = str(exc) or type(exc).__name__

    if last_prices:
        now = time.time()
        report.tick_symbols = len(last_prices)
        report.newest_tick_age_s = now - max(ts for _, ts in last_prices.values())

    _derive(report, cfg, last_prices)
    report.elapsed_s = time.perf_counter() - started
    return report


def _apply_snapshot(report: StatusReport, topics: TopicMap, record: Record) -> None:
    snapshot = decode(topics, record)
    assert isinstance(snapshot, PositionSnapshot)
    report.broker_positions = dict(snapshot.positions)
    report.usd_cash = snapshot.usd_cash
    report.snapshot_age_s = (utc_now() - snapshot.ts_utc).total_seconds()
    report.open_orders = len(snapshot.open_order_refs)
    report.market_open = snapshot.market_open
    report.clock_skew_s = snapshot.clock_skew_s


def _derive(report: StatusReport, cfg: EventbotConfig, last_prices: dict[str, tuple[float, float]]) -> None:
    problems = report.problems
    if report.redis_error:
        problems.append(report.redis_error)
    if report.kafka_error:
        problems.append(f"Kafka unavailable: {report.kafka_error}")
    if report.kill_switch is not None:
        problems.append(f"KILL SWITCH IS ON: {report.kill_switch}")
    if not report.redis_error:
        for process, age in report.heartbeats.items():
            if age is None:
                problems.append(f"No heartbeat from {process}")

    held = report.broker_positions if report.broker_positions is not None else report.internal_positions
    marks = {s: p for s, (p, _) in last_prices.items()}

    positions_known = report.broker_positions is not None or not report.redis_error
    report.gross_notional = gross_notional(held, marks, {}) if positions_known else None
    if report.gross_notional is not None and report.gross_notional > report.gross_cap:
        problems.append(
            f"Gross notional {report.gross_notional:,.0f} is above the cap {report.gross_cap:,.0f}"
        )
    priced = [(s, abs(q) * marks[s]) for s, q in held.items() if s in marks]
    if priced:
        report.largest_position = max(priced, key=lambda item: item[1])
        if report.largest_position[1] > report.position_cap:
            problems.append(f"{report.largest_position[0]} is above the per-symbol cap")

    if report.broker_positions is not None and not report.redis_error:
        report.position_diffs = diff_positions(report.internal_positions, report.broker_positions)
        if report.position_diffs:
            problems.append(f"Positions differ from the broker on {sorted(report.position_diffs)}")
    if report.snapshot_age_s is not None and report.snapshot_age_s > cfg.reconciler.max_snapshot_age_s:
        problems.append(f"The broker snapshot is {report.snapshot_age_s:.0f} s old")
    if report.broker_positions is None and not report.kafka_error:
        problems.append("No broker snapshot has been published yet")
    if report.market_open and (
        report.newest_tick_age_s is None or report.newest_tick_age_s > report.tick_stale_after_s
    ):
        problems.append("The market is open but market data is stale")
    if report.clock_skew_s is not None and abs(report.clock_skew_s) > cfg.adapter.max_clock_skew_s:
        problems.append(f"Local clock is {report.clock_skew_s:.0f} s off IB server time")
    for group, lags in report.lag.items():
        total = sum(lags.values())
        if total > LAG_WARNING:
            problems.append(f"Consumer group {group} is {total} messages behind")
    if report.intent_counts.get("placing"):
        problems.append(f"{report.intent_counts['placing']} intents are stuck in placing")


def _age(age: float | None) -> str:
    return "EXPIRED" if age is None else f"{age:.1f}s ago"


def render(report: StatusReport) -> str:
    lines: list[str] = []
    verdict = "OK" if report.healthy else f"ATTENTION ({len(report.problems)})"
    lines.append(f"eventbot status  {report.generated_at:%Y-%m-%d %H:%M:%S} UTC  {verdict}")
    lines.append("")
    lines.append(f"Kill switch   {'ON: ' + report.kill_switch if report.kill_switch is not None else 'off'}")
    lines.append("")
    lines.append("Heartbeats")
    if report.redis_error:
        lines.append(f"  unknown: {report.redis_error}")
    for process, age in report.heartbeats.items():
        lines.append(f"  {process:<12} {_age(age)}")
    lines.append("")
    lines.append("Positions (internal from fills, broker from the adapter's snapshot)")
    if report.broker_positions is None:
        lines.append("  no broker snapshot available")
    else:
        lines.append(f"  snapshot     {report.snapshot_age_s:.0f}s old, {report.open_orders} open orders")
        lines.append(f"  usd cash     {report.usd_cash if report.usd_cash is not None else 'unknown'}")
        symbols = sorted(set(report.internal_positions) | set(report.broker_positions))
        if not symbols:
            lines.append("  flat")
        for symbol in symbols:
            a, b = report.internal_positions.get(symbol, 0.0), report.broker_positions.get(symbol, 0.0)
            lines.append(f"  {symbol:<12} internal {a:>8g}   broker {b:>8g}{'   DIFFERS' if a != b else ''}")
    lines.append("")
    lines.append("Exposure vs caps")
    gross = "unknown" if report.gross_notional is None else f"{report.gross_notional:,.0f}"
    lines.append(f"  gross        {gross} of {report.gross_cap:,.0f}")
    if report.largest_position:
        symbol, value = report.largest_position
        lines.append(f"  largest      {symbol} {value:,.0f} of {report.position_cap:,.0f}")
    lines.append(f"  orders today {report.orders_today} of {report.orders_cap}")
    lines.append(f"  intents      {report.intent_counts or 'none'}")
    lines.append("")
    lines.append("Market data")
    market = {True: "open", False: "closed", None: "unknown"}[report.market_open]
    lines.append(f"  market       {market}   clock skew {report.clock_skew_s}")
    if report.newest_tick_age_s is None:
        lines.append("  newest tick  none")
    else:
        lines.append(f"  newest tick  {report.newest_tick_age_s:.1f}s ago, {report.tick_symbols} symbols")
    lines.append("")
    lines.append("Consumer lag")
    if report.kafka_error:
        lines.append(f"  unknown: {report.kafka_error}")
    for group, lags in report.lag.items():
        nonzero = {t: n for t, n in lags.items() if n}
        lines.append(f"  {group:<12} {sum(lags.values())} {nonzero if nonzero else ''}".rstrip())
    lines.append("")
    if report.problems:
        lines.append("Needs attention")
        lines.extend(f"  - {problem}" for problem in report.problems)
        lines.append("")
    lines.append(f"Collected in {report.elapsed_s:.2f} s")
    return "\n".join(lines)


async def amain() -> int:
    from eventbot.bus.kafka import KafkaInspector
    from stockbot import ny_date

    logging.getLogger("aiokafka").setLevel(logging.CRITICAL)
    started = time.perf_counter()
    cfg, settings = load_config(), load_settings()
    topics = TopicMap.from_config(cfg.topics)
    client = aioredis.Redis.from_url(
        settings.redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
    )
    inspector: Any = KafkaInspector(settings.kafka_bootstrap)
    kafka_error: str | None = None
    try:
        await asyncio.wait_for(inspector.start(), SECTION_TIMEOUT_S)
    except Exception as exc:
        kafka_error, inspector = str(exc) or type(exc).__name__, None
    try:
        report = await collect(
            cfg, StateStore(client), inspector, topics, ny_date(utc_now()).isoformat(), kafka_error
        )
    finally:
        if inspector is not None:
            await inspector.stop()
        await client.aclose()
    report.elapsed_s = time.perf_counter() - started
    print(render(report))
    return 0 if report.healthy else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
