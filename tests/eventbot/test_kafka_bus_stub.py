"""What KafkaBus passes to aiokafka, using stand-ins for aiokafka's classes. This pins the settings the
design depends on. It does NOT show that they work against a broker: tests/integration does that."""

from types import SimpleNamespace
from typing import Any

import pytest
from aiokafka import TopicPartition
from aiokafka.errors import KafkaConnectionError

from eventbot.bus import BusError
from eventbot.bus import kafka as kafka_module
from eventbot.bus.base import Record
from eventbot.bus.kafka import KafkaBus


class StubProducer:
    instances: list["StubProducer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.sent: list[tuple] = []
        self.fail = False
        StubProducer.instances.append(self)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def send_and_wait(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        if self.fail:
            raise KafkaConnectionError()
        self.sent.append((topic, key, value))


class StubConsumer:
    instances: list["StubConsumer"] = []

    def __init__(self, *topics: str, **kwargs: Any) -> None:
        self.subscribed, self.kwargs = topics, kwargs
        self.committed: list[dict] = []
        self.batch: dict = {}
        self.ends: dict = {}
        self.positions: dict = {}
        StubConsumer.instances.append(self)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def getmany(self, timeout_ms: int, max_records: int) -> dict:
        batch, self.batch = self.batch, {}
        return batch

    async def commit(self, offsets: dict) -> None:
        self.committed.append(offsets)

    def assignment(self) -> set:
        return set(self.ends)

    async def end_offsets(self, tps: list) -> dict:
        return self.ends

    async def position(self, tp: TopicPartition) -> int:
        return self.positions[tp]


@pytest.fixture(autouse=True)
def stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    StubProducer.instances.clear()
    StubConsumer.instances.clear()
    monkeypatch.setattr(kafka_module, "AIOKafkaProducer", StubProducer)
    monkeypatch.setattr(kafka_module, "AIOKafkaConsumer", StubConsumer)


def test_producer_is_idempotent_with_acks_all() -> None:
    KafkaBus("b:9092", "c")
    assert StubProducer.instances[0].kwargs["enable_idempotence"] is True
    assert StubProducer.instances[0].kwargs["acks"] == "all"


def test_consumer_has_auto_commit_off_and_reads_from_earliest() -> None:
    KafkaBus("b:9092", "c").consumer(["fills"], "g")
    kwargs = StubConsumer.instances[0].kwargs
    assert kwargs["enable_auto_commit"] is False
    assert kwargs["group_id"] == "g"
    assert kwargs["auto_offset_reset"] == "earliest"


def test_from_beginning_false_maps_to_latest() -> None:
    KafkaBus("b:9092", "c").consumer(["fills"], None, from_beginning=False)
    assert StubConsumer.instances[0].kwargs["auto_offset_reset"] == "latest"


async def test_commit_is_offset_plus_one_for_that_partition_only() -> None:
    consumer = KafkaBus("b:9092", "c").consumer(["fills"], "g")
    await consumer.commit(Record("fills", 2, 41, "k", b""))
    assert StubConsumer.instances[0].committed == [{TopicPartition("fills", 2): 42}]


async def test_getone_returns_a_record_with_a_decoded_key_and_buffers_the_rest() -> None:
    consumer = KafkaBus("b:9092", "c").consumer(["fills"], "g")
    tp = TopicPartition("fills", 0)
    StubConsumer.instances[0].batch = {
        tp: [
            SimpleNamespace(topic="fills", partition=0, offset=5, key=b"e1", value=b"v1"),
            SimpleNamespace(topic="fills", partition=0, offset=6, key=None, value=b"v2"),
        ]
    }
    first = await consumer.getone(0.01)
    second = await consumer.getone(0.01)
    assert first is not None and second is not None
    assert (first.offset, first.key, first.value) == (5, "e1", b"v1")
    assert (second.offset, second.key) == (6, None)
    assert await consumer.getone(0.01) is None


async def test_publish_encodes_the_key_and_wraps_failures_as_bus_error() -> None:
    bus = KafkaBus("b:9092", "c")
    await bus.publish_raw("fills", "e1", b"v")
    assert StubProducer.instances[0].sent == [("fills", b"e1", b"v")]
    StubProducer.instances[0].fail = True
    with pytest.raises(BusError):
        await bus.publish_raw("fills", "e1", b"v")


async def test_caught_up_compares_positions_with_end_offsets() -> None:
    consumer = KafkaBus("b:9092", "c").consumer(["fills"], "g")
    stub = StubConsumer.instances[0]
    tp = TopicPartition("fills", 0)
    stub.ends, stub.positions = {tp: 10}, {tp: 9}
    assert not await consumer.caught_up()
    stub.positions = {tp: 10}
    assert await consumer.caught_up()


async def test_caught_up_is_false_before_any_assignment() -> None:
    consumer = KafkaBus("b:9092", "c").consumer(["fills"], "g")
    assert not await consumer.caught_up()


async def test_inspector_lag_counts_uncommitted_messages_per_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    from eventbot.bus.kafka import KafkaInspector

    tp0, tp1 = TopicPartition("fills", 0), TopicPartition("fills", 1)

    class StubAdmin:
        def __init__(self, **kwargs: Any) -> None: ...

        async def list_consumer_group_offsets(self, group: str) -> dict:
            return {tp0: SimpleNamespace(offset=7)}

    class InspectConsumer(StubConsumer):
        async def topics(self) -> set:
            return {"fills"}

        def partitions_for_topic(self, topic: str) -> set | None:
            return {0, 1} if topic == "fills" else None

        async def end_offsets(self, tps: list) -> dict:
            return {tp0: 10, tp1: 5}

        async def beginning_offsets(self, tps: list) -> dict:
            return {tp0: 0, tp1: 2}

    monkeypatch.setattr(kafka_module, "AIOKafkaAdminClient", StubAdmin)
    monkeypatch.setattr(kafka_module, "AIOKafkaConsumer", InspectConsumer)
    inspector = KafkaInspector("b:9092")
    assert await inspector.lag("recorder", ["fills", "unknown"]) == {"fills": 3 + 3}
