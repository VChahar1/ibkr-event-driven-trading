"""The recorder: Parquet output readable by DuckDB, commits only after durable writes, no loss on a crash."""

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
import pytest

from eventbot.adapter.fake_broker import FAKE_ACCOUNT
from eventbot.bus import BusError, publish
from eventbot.recorder import GROUP, Recorder, load_session
from eventbot.schemas import Alert, AlertType, Fill, PositionSnapshot, Side, Tick, utc_now
from tests.eventbot.conftest import Harness, make_intent


def tick(n: int = 0, symbol: str = "SPY", **over: Any) -> Tick:
    return Tick(symbol=symbol, price=100.0 + n, md_type=3, **over)


def fill(exec_id: str = "e.1.01", qty: float = 2) -> Fill:
    return Fill(
        exec_id=exec_id,
        order_ref="a" * 32,
        symbol="AMD",
        side=Side.BUY,
        qty=qty,
        price=10.0,
        exec_time_utc=utc_now(),
    )


def parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


@pytest.fixture
async def recorder(harness: Harness, tmp_path: Path) -> Recorder:
    instance = Recorder(harness.cfg, harness.bus, harness.state, harness.topics, tmp_path / "pq")
    await instance.start()
    return instance


async def test_every_topic_is_recorded_under_its_own_directory(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    await harness.ready()
    intent = make_intent()
    await harness.adapter.handle_intent(intent)
    harness.broker.fill(intent.intent_id)
    await harness.adapter.drain_broker_events()
    await harness.adapter.flush_ticks()
    await harness.deliver(intent)
    await publish(
        harness.bus, harness.topics.alerts, "x", harness.alerts()[0] if harness.alerts() else _alert()
    )

    await recorder.process_available(0.01)
    await recorder.flush()

    dirs = {p.parent.parent.name for p in parquet_files(tmp_path / "pq")}
    assert dirs == {f"topic={t}" for t in harness.topics.all()}


def _alert() -> Alert:
    from eventbot.schemas import alert

    return alert(AlertType.SMOKE, "info", "t", "m")


async def test_duckdb_can_query_the_flattened_columns(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    for n in range(3):
        await publish(harness.bus, harness.topics.md_ticks, "SPY", tick(n))
    await recorder.process_available(0.01)
    await recorder.flush()
    rows = duckdb.execute(
        "SELECT symbol, max(price), count(*) FROM read_parquet(?, hive_partitioning = false) GROUP BY symbol",
        [str(tmp_path / "pq" / "topic=md.ticks" / "*" / "*.parquet")],
    ).fetchall()
    assert rows == [("SPY", 102.0, 3)]


async def test_columns_that_are_empty_in_one_file_still_union(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    await publish(harness.bus, harness.topics.md_ticks, "SPY", tick(0))
    await recorder.process_available(0.01)
    await recorder.flush()
    await publish(harness.bus, harness.topics.md_ticks, "SPY", tick(1, bid=100.5))
    await recorder.process_available(0.01)
    await recorder.flush()
    assert len(parquet_files(tmp_path / "pq")) == 2
    rows = duckdb.execute(
        "SELECT bid FROM read_parquet(?, union_by_name = true, hive_partitioning = false) ORDER BY _offset",
        [str(tmp_path / "pq" / "topic=md.ticks" / "*" / "*.parquet")],
    ).fetchall()
    assert rows == [(None,), (100.5,)]


async def test_nothing_is_committed_until_the_file_is_written(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    await publish(harness.bus, harness.topics.fills, "e", fill())
    await recorder.process_available(0.01)
    assert harness.bus.committed(GROUP, harness.topics.fills, 0) is None
    assert parquet_files(tmp_path / "pq") == []
    await recorder.flush()
    assert harness.bus.committed(GROUP, harness.topics.fills, 0) == 1
    assert len(parquet_files(tmp_path / "pq")) == 1


async def test_a_failed_write_commits_nothing(
    harness: Harness, tmp_path: Path, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    await publish(harness.bus, harness.topics.fills, "e", fill())
    await recorder.process_available(0.01)

    def disk_full(*args: Any) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("eventbot.recorder.write_durably", disk_full)
    with pytest.raises(OSError):
        await recorder.flush()
    assert harness.bus.committed(GROUP, harness.topics.fills, 0) is None


async def test_a_crash_before_the_flush_loses_nothing(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    for n in range(5):
        await publish(harness.bus, harness.topics.fills, f"e{n}", fill(f"e{n}.1.01"))
    await recorder.process_available(0.01)

    restarted = Recorder(harness.cfg, harness.bus, harness.state, harness.topics, tmp_path / "pq")
    await restarted.start()
    await restarted.process_available(0.01)
    await restarted.flush()
    assert len(load_session(tmp_path / "pq")) == 5


async def test_a_crash_between_write_and_commit_records_twice_and_readers_see_one(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    for n in range(3):
        await publish(harness.bus, harness.topics.md_ticks, "SPY", tick(n))
    await recorder.process_available(0.01)

    async def commit_fails(record: Any) -> None:
        raise BusError("broker went away")

    assert recorder._consumer is not None
    recorder._consumer.commit = commit_fails  # type: ignore[method-assign]
    await recorder.flush()
    for n in range(3, 5):
        await publish(harness.bus, harness.topics.md_ticks, "SPY", tick(n))

    restarted = Recorder(harness.cfg, harness.bus, harness.state, harness.topics, tmp_path / "pq")
    await restarted.start()
    await restarted.process_available(0.01)
    await restarted.flush()

    total = pq.read_table(str(parquet_files(tmp_path / "pq")[0])).num_rows
    assert total == 3 and len(parquet_files(tmp_path / "pq")) == 2
    assert [m.offset for m in load_session(tmp_path / "pq")] == [0, 1, 2, 3, 4]


async def test_files_appear_atomically_with_no_temporary_files_left(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    await publish(harness.bus, harness.topics.fills, "e", fill())
    await recorder.process_available(0.01)
    await recorder.flush()
    leftovers = [p for p in (tmp_path / "pq").rglob("*") if p.is_file() and p.suffix != ".parquet"]
    assert leftovers == []


async def test_a_flush_happens_at_the_record_limit(harness: Harness, tmp_path: Path) -> None:
    cfg = harness.cfg.model_copy(
        update={"recorder": harness.cfg.recorder.model_copy(update={"flush_records": 2})}
    )
    instance = Recorder(cfg, harness.bus, harness.state, harness.topics, tmp_path / "pq")
    await instance.start()
    for n in range(5):
        await publish(harness.bus, harness.topics.md_ticks, "SPY", tick(n))
    await instance.process_available(0.01)
    assert instance.records_written == 4
    await instance.flush()
    assert instance.records_written == 5


async def test_the_run_loop_flushes_on_stop_and_writes_a_heartbeat(harness: Harness, tmp_path: Path) -> None:
    instance = Recorder(harness.cfg, harness.bus, harness.state, harness.topics, tmp_path / "pq")
    task = asyncio.create_task(instance.run())
    await asyncio.sleep(0.02)
    await publish(harness.bus, harness.topics.md_ticks, "SPY", tick())
    await asyncio.sleep(0.05)
    instance.request_stop()
    await asyncio.wait_for(task, 3.0)
    assert len(load_session(tmp_path / "pq")) == 1
    assert await harness.state.heartbeat_age_s("recorder") is not None
    assert harness.bus.committed(GROUP, harness.topics.md_ticks, 0) == 1


async def test_an_invalid_message_is_recorded_alerted_and_does_not_stop_the_recorder(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    await harness.bus.publish_raw(harness.topics.fills, "bad", b"not json")
    await publish(harness.bus, harness.topics.fills, "e", fill())
    await recorder.process_available(0.01)
    await recorder.flush()

    assert sorted(m.topic for m in load_session(tmp_path / "pq")) == [
        harness.topics.alerts,
        harness.topics.fills,
    ]

    everything = load_session(tmp_path / "pq", include_invalid=True)
    assert len(everything) == 3 and sum(not m.valid for m in everything) == 1
    assert any(p.parent.parent.name == "topic=fills._invalid" for p in parquet_files(tmp_path / "pq"))
    assert AlertType.INVALID_MESSAGE in [a.type for a in harness.alerts()]
    assert harness.bus.committed(GROUP, harness.topics.fills, 0) == 2


async def test_load_session_returns_time_order_across_topics(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    base = utc_now()
    await publish(harness.bus, harness.topics.fills, "e", fill().model_copy(update={"ts_utc": base}))
    await publish(harness.bus, harness.topics.md_ticks, "SPY", tick(ts_utc=base - dt.timedelta(seconds=5)))
    await recorder.process_available(0.01)
    await recorder.flush()
    assert [m.topic for m in load_session(tmp_path / "pq")] == [harness.topics.md_ticks, harness.topics.fills]


async def test_load_session_filters_by_date(harness: Harness, tmp_path: Path, recorder: Recorder) -> None:
    await publish(
        harness.bus, harness.topics.md_ticks, "SPY", tick(ts_utc=dt.datetime(2026, 11, 2, 15, tzinfo=dt.UTC))
    )
    await publish(
        harness.bus, harness.topics.md_ticks, "SPY", tick(ts_utc=dt.datetime(2026, 11, 3, 15, tzinfo=dt.UTC))
    )
    await recorder.process_available(0.01)
    await recorder.flush()
    assert len(load_session(tmp_path / "pq")) == 2
    assert len(load_session(tmp_path / "pq", date="2026-11-02")) == 1
    assert load_session(tmp_path / "pq", date="2020-01-01") == []


def test_load_session_on_an_empty_directory(tmp_path: Path) -> None:
    assert load_session(tmp_path / "nothing") == []


async def test_the_real_account_id_never_reaches_parquet(
    harness: Harness, tmp_path: Path, recorder: Recorder
) -> None:
    await harness.ready()
    await harness.adapter.publish_snapshot()
    await recorder.process_available(0.01)
    await recorder.flush()
    snapshots = [m for m in load_session(tmp_path / "pq") if m.topic == harness.topics.positions_snapshot]
    assert snapshots
    rows = duckdb.execute(
        "SELECT * FROM read_parquet(?, union_by_name = true, hive_partitioning = false)",
        [str(tmp_path / "pq" / "topic=*" / "*" / "*.parquet")],
    ).fetchall()
    assert FAKE_ACCOUNT not in str(rows)
    assert all(PositionSnapshot.model_validate_json(m.payload).account != FAKE_ACCOUNT for m in snapshots)
