"""The recorder: consumes every topic and writes Parquet, readable by DuckDB, as the audit trail
and the input to replay."""

import asyncio
import datetime as dt
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from eventbot.bus import Bus, BusError, Consumer, Record, TopicMap, publish
from eventbot.bus.base import decode
from eventbot.config import EventbotConfig
from eventbot.logs import log_event
from eventbot.schemas import AlertType, Message, alert, utc_now
from eventbot.state import StateStore, StateUnavailable

LOG = logging.getLogger("eventbot.recorder")
SERVICE = "recorder"
GROUP = "recorder"
INVALID_TOPIC_SUFFIX = "._invalid"
POLL_S = 0.25


def to_columns(message: Message) -> dict[str, Any]:
    """Top-level fields as columns for SQL; nested values become JSON. _payload stays authoritative."""
    out: dict[str, Any] = {}
    for key, value in message.model_dump(mode="json").items():
        out[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
    return out


def write_durably(table: pa.Table, path: Path) -> None:
    """After this returns the file survives a power cut, and a reader can never see it half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temp)
    with open(temp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class Recorder:
    def __init__(
        self,
        cfg: EventbotConfig,
        bus: Bus,
        state: StateStore,
        topics: TopicMap,
        out_dir: Path,
        *,
        group: str = GROUP,
    ) -> None:
        self._rcfg = cfg.recorder
        self._beat_every = cfg.heartbeats.interval_s
        self._bus = bus
        self._state = state
        self._topics = topics
        self._out_dir = Path(out_dir)
        self._group = group
        self._consumer: Consumer | None = None
        self._rows: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        self._last_record: dict[tuple[str, int], Record] = {}
        self._buffered = 0
        self._last_flush = time.monotonic()
        self.records_written = 0
        self.files_written = 0
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def start(self) -> None:
        self._consumer = self._bus.consumer(self._topics.all(), self._group)
        await self._consumer.start()

    async def stop(self) -> None:
        await self.flush()
        if self._consumer is not None:
            await self._consumer.stop()

    def _buffer(self, record: Record, message: Message | None) -> None:
        received = utc_now()
        row: dict[str, Any] = {
            "_topic": record.topic,
            "_partition": record.partition,
            "_offset": record.offset,
            "_key": record.key,
            "_recv_ts_utc": received.isoformat(),
            "_payload": record.value.decode("utf-8", errors="replace"),
            "_valid": message is not None,
        }
        if message is not None:
            row.update(to_columns(message))
            date = message.ts_utc.date().isoformat()
            topic_dir = record.topic
        else:
            date = received.date().isoformat()
            topic_dir = record.topic + INVALID_TOPIC_SUFFIX
        self._rows[(topic_dir, record.partition, date)].append(row)
        self._last_record[(record.topic, record.partition)] = record
        self._buffered += 1

    async def process_available(self, timeout_s: float = POLL_S) -> int:
        """Buffer whatever has arrived. Returns how many records were read."""
        assert self._consumer is not None
        count = 0
        record = await self._consumer.getone(timeout_s)
        while record is not None:
            await self._ingest(record)
            count += 1
            if self._buffered >= self._rcfg.flush_records:
                await self.flush()
            record = await self._consumer.getone(0.0)
        return count

    async def _ingest(self, record: Record) -> None:
        try:
            self._buffer(record, decode(self._topics, record))
        except ValueError as exc:
            self._buffer(record, None)
            await self._alert_invalid(record, exc)

    async def _alert_invalid(self, record: Record, exc: Exception) -> None:
        log_event(LOG, "invalid_message", level=logging.WARNING, topic=record.topic, offset=record.offset)
        try:
            await publish(
                self._bus,
                self._topics.alerts,
                AlertType.INVALID_MESSAGE.value,
                alert(
                    AlertType.INVALID_MESSAGE,
                    "warning",
                    SERVICE,
                    f"Invalid message on {record.topic} (recorded under {INVALID_TOPIC_SUFFIX})",
                    topic=record.topic,
                    partition=record.partition,
                    offset=record.offset,
                    error=str(exc)[:500],
                ),
            )
        except BusError as bus_exc:
            log_event(LOG, "invalid_alert_failed", level=logging.ERROR, msg=str(bus_exc))

    async def flush(self) -> int:
        """Write everything buffered, then commit. In that order: the commit says "safely on disk"."""
        if not self._buffered:
            self._last_flush = time.monotonic()
            return 0
        assert self._consumer is not None
        written = 0
        for (topic_dir, partition, date), rows in self._rows.items():
            first, last = rows[0]["_offset"], rows[-1]["_offset"]
            path = (
                self._out_dir
                / f"topic={topic_dir}"
                / f"date={date}"
                / f"part-p{partition}-{first:012d}-{last:012d}.parquet"
            )
            write_durably(pa.Table.from_pylist(rows), path)
            written += len(rows)
            self.files_written += 1
        for record in self._last_record.values():
            try:
                await self._consumer.commit(record)
            except BusError as exc:
                log_event(LOG, "commit_failed", level=logging.WARNING, msg=str(exc), topic=record.topic)
        self._rows.clear()
        self._last_record.clear()
        self._buffered = 0
        self._last_flush = time.monotonic()
        self.records_written += written
        log_event(LOG, "flushed", records=written, total=self.records_written)
        return written

    async def run(self) -> None:
        await self.start()
        next_beat = 0.0
        try:
            while not self._stop.is_set():
                try:
                    await self.process_available()
                except BusError as exc:
                    log_event(LOG, "consume_failed", level=logging.WARNING, msg=str(exc))
                    await asyncio.sleep(1.0)
                now = time.monotonic()
                if now - self._last_flush >= self._rcfg.flush_seconds:
                    await self.flush()
                if now >= next_beat:
                    next_beat = now + self._beat_every
                    try:
                        await self._state.beat(SERVICE)
                    except StateUnavailable as exc:
                        log_event(LOG, "heartbeat_skipped", level=logging.WARNING, msg=str(exc))
        finally:
            await self.stop()


@dataclass(frozen=True)
class RecordedMessage:
    topic: str
    partition: int
    offset: int
    key: str | None
    payload: str
    ts_utc: dt.datetime | None
    valid: bool


def load_session(
    out_dir: Path, date: str | None = None, include_invalid: bool = False
) -> list[RecordedMessage]:
    """Every recorded message exactly once, in time order. Offsets identify records, so a range written twice
    after a crash collapses to one copy."""
    root = Path(out_dir)
    if not any(root.rglob("*.parquet")):
        return []
    pattern = str(root / "topic=*" / (f"date={date}" if date else "date=*") / "*.parquet")
    if not list(root.glob(f"topic=*/{f'date={date}' if date else 'date=*'}/*.parquet")):
        return []
    rows = duckdb.execute(
        """
        SELECT _topic, _partition, _offset, _key, _payload, _valid, _recv_ts_utc
        FROM read_parquet(?, union_by_name = true, hive_partitioning = false)
        QUALIFY row_number() OVER (PARTITION BY _topic, _partition, _offset ORDER BY _recv_ts_utc) = 1
        """,
        [pattern],
    ).fetchall()
    out: list[RecordedMessage] = []
    for topic, partition, offset, key, payload, valid, _received in rows:
        if not valid and not include_invalid:
            continue
        ts: dt.datetime | None = None
        if valid:
            ts = dt.datetime.fromisoformat(json.loads(payload)["ts_utc"])
        out.append(RecordedMessage(topic, partition, offset, key, payload, ts, bool(valid)))
    floor = dt.datetime.min.replace(tzinfo=dt.UTC)
    out.sort(key=lambda m: (m.ts_utc or floor, m.topic, m.partition, m.offset))
    return out


async def amain() -> int:
    """Run the Kafka-to-Parquet recorder with the configured state store."""
    from eventbot.config import ROOT
    from eventbot.runtime import build_runtime, install_stop_handlers

    rt = build_runtime(SERVICE)
    service = Recorder(rt.cfg, rt.bus, rt.state, rt.topics, ROOT / rt.cfg.recorder.out_dir)
    install_stop_handlers(service.request_stop)
    await rt.bus.start()
    try:
        await service.run()
        return 0
    finally:
        await rt.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
