"""The bus contract against a real broker: the same scenarios as tests/eventbot/test_bus.py."""

import asyncio
import uuid

import pytest

from eventbot.bus.kafka import KafkaBus

pytestmark = pytest.mark.integration


async def read_until(consumer, count: int, timeout_s: float = 20.0) -> list:
    out: list = []
    deadline = asyncio.get_running_loop().time() + timeout_s
    while len(out) < count and asyncio.get_running_loop().time() < deadline:
        record = await consumer.getone(0.5)
        if record is not None:
            out.append(record)
    return out


@pytest.fixture
async def bus(bootstrap: str, make_topic):
    instance = KafkaBus(bootstrap, "it")
    await instance.start()
    yield instance
    await instance.stop()


async def test_publish_consume_round_trip_with_key(bus: KafkaBus, make_topic) -> None:
    topic = await make_topic()
    await bus.publish_raw(topic, "k1", b"v1")
    consumer = bus.consumer([topic], None)
    await consumer.start()
    try:
        records = await read_until(consumer, 1)
    finally:
        await consumer.stop()
    assert [(r.key, r.value) for r in records] == [("k1", b"v1")]


async def test_consumer_restart_resumes_from_committed_offset(bus: KafkaBus, make_topic) -> None:
    topic = await make_topic()
    for n in range(10):
        await bus.publish_raw(topic, "k", str(n).encode())
    group = f"it-{uuid.uuid4().hex[:8]}"

    first = bus.consumer([topic], group)
    await first.start()
    seen = []
    for record in await read_until(first, 4):
        seen.append(record.value)
        await first.commit(record)
    await first.stop()

    second = bus.consumer([topic], group)
    await second.start()
    try:
        seen += [r.value for r in await read_until(second, 6)]
        assert await second.getone(1.0) is None
    finally:
        await second.stop()
    assert seen == [str(n).encode() for n in range(10)]


async def test_uncommitted_records_are_redelivered_after_a_restart(bus: KafkaBus, make_topic) -> None:
    topic = await make_topic()
    for n in range(3):
        await bus.publish_raw(topic, "k", str(n).encode())
    group = f"it-{uuid.uuid4().hex[:8]}"

    first = bus.consumer([topic], group)
    await first.start()
    records = await read_until(first, 3)
    await first.commit(records[0])
    await first.stop()

    second = bus.consumer([topic], group)
    await second.start()
    try:
        redelivered = await read_until(second, 2)
    finally:
        await second.stop()
    assert [r.value for r in redelivered] == [b"1", b"2"]


async def test_caught_up_on_a_multi_partition_topic(bus: KafkaBus, make_topic) -> None:
    topic = await make_topic(partitions=3)
    for n in range(6):
        await bus.publish_raw(topic, f"k{n}", b"v")
    consumer = bus.consumer([topic], None)
    await consumer.start()
    try:
        await read_until(consumer, 6)
        assert await consumer.caught_up()
    finally:
        await consumer.stop()
