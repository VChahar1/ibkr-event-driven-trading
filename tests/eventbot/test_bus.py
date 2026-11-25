"""The bus contract, on the in-memory implementation. tests/integration/test_kafka_bus.py runs the same
scenarios against a real broker; test_kafka_bus_stub.py pins what KafkaBus passes to aiokafka."""

import pytest

from eventbot.bus import BusError, Consumer, InMemoryBus, Record, TopicMap, next_message, publish
from eventbot.config import load_config
from eventbot.schemas import Alert, AlertType, Tick

TOPICS = TopicMap.from_config(load_config().topics)


def tick(n: int = 0, symbol: str = "SPY") -> Tick:
    return Tick(symbol=symbol, price=100.0 + n, md_type=3)


async def one(consumer: Consumer) -> Record:
    record = await consumer.getone(0.01)
    assert record is not None
    return record


async def read_all(consumer: Consumer, limit: int = 100) -> list[Record]:
    out: list[Record] = []
    while len(out) < limit:
        record = await consumer.getone(0.01)
        if record is None:
            return out
        out.append(record)
    return out


async def test_publish_and_consume_in_key_order() -> None:
    bus = InMemoryBus()
    for n in range(5):
        await publish(bus, TOPICS.md_ticks, "SPY", tick(n))
    consumer = bus.consumer([TOPICS.md_ticks], "g")
    await consumer.start()
    records = await read_all(consumer)
    assert [Tick.model_validate_json(r.value).price for r in records] == [100, 101, 102, 103, 104]


async def test_a_key_always_goes_to_the_same_partition() -> None:
    bus = InMemoryBus(partitions={TOPICS.md_ticks: 3})
    for n in range(10):
        await publish(bus, TOPICS.md_ticks, "AMD", tick(n, "AMD"))
    assert sorted(len(bus.log(TOPICS.md_ticks, p)) for p in range(3)) == [0, 0, 10]


async def test_consumer_restart_resumes_from_the_committed_offset_with_no_loss_or_duplicates() -> None:
    bus = InMemoryBus()
    for n in range(10):
        await publish(bus, TOPICS.md_ticks, "SPY", tick(n))

    first = bus.consumer([TOPICS.md_ticks], "g")
    await first.start()
    seen = []
    for _ in range(4):
        record = await one(first)
        seen.append(record.offset)
        await first.commit(record)
    await first.stop()

    second = bus.consumer([TOPICS.md_ticks], "g")
    await second.start()
    seen += [r.offset for r in await read_all(second)]
    assert seen == list(range(10))


async def test_a_record_read_but_not_committed_is_redelivered_after_a_restart() -> None:
    bus = InMemoryBus()
    for n in range(3):
        await publish(bus, TOPICS.md_ticks, "SPY", tick(n))
    first = bus.consumer([TOPICS.md_ticks], "g")
    await first.start()
    await first.commit(await one(first))
    await first.getone(0.01)
    await first.stop()

    second = bus.consumer([TOPICS.md_ticks], "g")
    await second.start()
    assert [r.offset for r in await read_all(second)] == [1, 2]


async def test_groups_are_independent() -> None:
    bus = InMemoryBus()
    await publish(bus, TOPICS.md_ticks, "SPY", tick())
    a, b = bus.consumer([TOPICS.md_ticks], "a"), bus.consumer([TOPICS.md_ticks], "b")
    await a.start()
    await b.start()
    await a.commit(await one(a))
    assert (await one(b)).offset == 0


async def test_from_beginning_false_starts_at_the_end() -> None:
    bus = InMemoryBus()
    await publish(bus, TOPICS.md_ticks, "SPY", tick(0))
    consumer = bus.consumer([TOPICS.md_ticks], None, from_beginning=False)
    await consumer.start()
    assert await consumer.getone(0.01) is None
    await publish(bus, TOPICS.md_ticks, "SPY", tick(1))
    assert (await one(consumer)).offset == 1


async def test_commit_without_a_group_is_a_programming_error() -> None:
    bus = InMemoryBus()
    await publish(bus, TOPICS.md_ticks, "SPY", tick())
    consumer = bus.consumer([TOPICS.md_ticks], None)
    await consumer.start()
    record = await one(consumer)
    with pytest.raises(RuntimeError):
        await consumer.commit(record)


async def test_caught_up() -> None:
    bus = InMemoryBus(partitions={TOPICS.fills: 3})
    await publish(bus, TOPICS.fills, "a", tick())
    await publish(bus, TOPICS.fills, "b", tick())
    consumer = bus.consumer([TOPICS.fills], None)
    await consumer.start()
    assert not await consumer.caught_up()
    await read_all(consumer)
    assert await consumer.caught_up()


async def test_a_waiting_consumer_wakes_on_publish() -> None:
    import asyncio

    bus = InMemoryBus()
    consumer = bus.consumer([TOPICS.md_ticks], "g")
    await consumer.start()
    waiter = asyncio.create_task(consumer.getone(5.0))
    await asyncio.sleep(0)
    await publish(bus, TOPICS.md_ticks, "SPY", tick())
    record = await asyncio.wait_for(waiter, 1.0)
    assert record is not None and record.offset == 0


async def test_publish_failure_raises_bus_error() -> None:
    bus = InMemoryBus()
    bus.fail_publishes = 1
    with pytest.raises(BusError):
        await publish(bus, TOPICS.md_ticks, "SPY", tick())
    await publish(bus, TOPICS.md_ticks, "SPY", tick())
    assert len(bus.records(TOPICS.md_ticks)) == 1


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"",
        b"\xff\xfe\x00binary",
        b'{"symbol": "SPY"}',
        b'{"schema_version": 99, "symbol": "SPY", "price": 1.0, "md_type": 3}',
    ],
)
async def test_invalid_message_produces_an_alert_and_does_not_crash_or_block(raw: bytes) -> None:
    bus = InMemoryBus()
    await bus.publish_raw(TOPICS.md_ticks, "SPY", raw)
    await publish(bus, TOPICS.md_ticks, "SPY", tick(7))
    consumer = bus.consumer([TOPICS.md_ticks], "g")
    await consumer.start()

    good = await next_message(consumer, bus, TOPICS, service="t", timeout_s=0.01)
    assert good is not None and isinstance(good.message, Tick) and good.message.price == 107.0
    assert await next_message(consumer, bus, TOPICS, service="t", timeout_s=0.01) is None

    alerts = [Alert.model_validate_json(r.value) for r in bus.records(TOPICS.alerts)]
    assert len(alerts) == 1 and alerts[0].type is AlertType.INVALID_MESSAGE
    assert alerts[0].details["topic"] == TOPICS.md_ticks and alerts[0].details["offset"] == 0


async def test_an_invalid_message_is_committed_so_a_restart_does_not_reprocess_it() -> None:
    bus = InMemoryBus()
    await bus.publish_raw(TOPICS.md_ticks, "SPY", b"not json")
    consumer = bus.consumer([TOPICS.md_ticks], "g")
    await consumer.start()
    await next_message(consumer, bus, TOPICS, service="t", timeout_s=0.01)
    await consumer.stop()
    restarted = bus.consumer([TOPICS.md_ticks], "g")
    await restarted.start()
    assert await restarted.getone(0.01) is None
    assert len(bus.records(TOPICS.alerts)) == 1


def test_topic_map_prefix_keeps_replay_away_from_the_real_topics() -> None:
    replay = TopicMap.from_config(load_config().topics, prefix="replay.")
    assert all(t.startswith("replay.") for t in replay.all())
    assert not set(replay.all()) & set(TOPICS.all())
    assert replay.model_for("replay.md.ticks") is Tick


async def test_invalid_records_on_a_groupless_reader_are_alerted_and_skipped() -> None:
    bus = InMemoryBus()
    await bus.publish_raw(TOPICS.md_ticks, "SPY", b"bad")
    await publish(bus, TOPICS.md_ticks, "SPY", tick(1))
    consumer = bus.consumer([TOPICS.md_ticks], None)
    await consumer.start()
    assert await next_message(consumer, bus, TOPICS, service="t", timeout_s=0.01) is not None
    assert len(bus.records(TOPICS.alerts)) == 1


async def test_a_failure_to_publish_the_alert_does_not_crash_the_consumer() -> None:
    bus = InMemoryBus()
    await bus.publish_raw(TOPICS.md_ticks, "SPY", b"bad")
    await publish(bus, TOPICS.md_ticks, "SPY", tick(1))
    consumer = bus.consumer([TOPICS.md_ticks], "g")
    await consumer.start()
    bus.fail_publishes = 1
    good = await next_message(consumer, bus, TOPICS, service="t", timeout_s=0.01)
    assert good is not None and good.record.offset == 1
