"Kafka bus on aiokafka, which shares the asyncio loop that ib_async runs on."

from collections import deque

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.errors import KafkaError

from eventbot.bus.base import BusError, Consumer, Record

FETCH_BATCH = 500


class KafkaConsumer:
    def __init__(
        self, bootstrap: str, topics: list[str], group: str | None, from_beginning: bool, client_id: str
    ) -> None:
        self._consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap,
            client_id=client_id,
            group_id=group,
            enable_auto_commit=False,
            auto_offset_reset="earliest" if from_beginning else "latest",
        )
        self._buffer: deque[ConsumerRecord] = deque()
        self.group = group

    async def start(self) -> None:
        try:
            await self._consumer.start()
        except KafkaError as exc:
            raise BusError(f"Kafka consumer failed to start: {exc}") from exc

    async def stop(self) -> None:
        await self._consumer.stop()

    async def getone(self, timeout_s: float) -> Record | None:
        if not self._buffer:
            try:
                batches = await self._consumer.getmany(
                    timeout_ms=int(timeout_s * 1000), max_records=FETCH_BATCH
                )
            except KafkaError as exc:
                raise BusError(f"Kafka fetch failed: {exc}") from exc
            for messages in batches.values():
                self._buffer.extend(messages)
        if not self._buffer:
            return None
        m = self._buffer.popleft()
        return Record(m.topic, m.partition, m.offset, m.key.decode() if m.key is not None else None, m.value)

    async def commit(self, record: Record) -> None:
        if self.group is None:
            raise RuntimeError("cannot commit without a consumer group")
        try:
            await self._consumer.commit({TopicPartition(record.topic, record.partition): record.offset + 1})
        except KafkaError as exc:
            raise BusError(f"Offset commit failed: {exc}") from exc

    async def caught_up(self) -> bool:
        if self._buffer:
            return False
        assignment = self._consumer.assignment()
        if not assignment:
            return False
        ends = await self._consumer.end_offsets(list(assignment))
        for tp in assignment:
            if await self._consumer.position(tp) < ends[tp]:
                return False
        return True


class KafkaBus:
    def __init__(self, bootstrap: str, client_id: str) -> None:
        self._bootstrap = bootstrap
        self._client_id = client_id
        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap,
            client_id=client_id,
            enable_idempotence=True,
            acks="all",
        )

    async def start(self) -> None:
        try:
            await self._producer.start()
        except KafkaError as exc:
            raise BusError(f"Kafka producer failed to start: {exc}") from exc

    async def stop(self) -> None:
        await self._producer.stop()

    async def publish_raw(self, topic: str, key: str | None, value: bytes) -> None:
        try:
            encoded = key.encode() if key is not None else None
            await self._producer.send_and_wait(topic, value=value, key=encoded)
        except KafkaError as exc:
            raise BusError(f"Publish to {topic} failed: {exc}") from exc

    def consumer(self, topics: list[str], group: str | None, from_beginning: bool = True) -> Consumer:
        return KafkaConsumer(self._bootstrap, topics, group, from_beginning, self._client_id)


class KafkaInspector:
    """Read-only views for make status: the newest record on a topic, and consumer group lag."""

    def __init__(self, bootstrap: str) -> None:
        self._admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap, client_id="eventbot-status")
        self._consumer = AIOKafkaConsumer(
            bootstrap_servers=bootstrap,
            client_id="eventbot-status",
            group_id=None,
            enable_auto_commit=False,
        )

    async def start(self) -> None:
        """If the broker is down this raises, with nothing left open."""
        try:
            await self._admin.start()
            await self._consumer.start()
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        await self._consumer.stop()
        await self._admin.close()

    async def _partitions(self, topic: str) -> list[TopicPartition]:
        await self._consumer.topics()
        known = self._consumer.partitions_for_topic(topic) or set()
        return [TopicPartition(topic, p) for p in sorted(known)]

    async def latest(self, topic: str) -> Record | None:
        partitions = await self._partitions(topic)
        if not partitions:
            return None
        ends = await self._consumer.end_offsets(partitions)
        begins = await self._consumer.beginning_offsets(partitions)
        newest: ConsumerRecord | None = None
        for tp in partitions:
            if ends[tp] <= begins[tp]:
                continue
            self._consumer.assign([tp])
            self._consumer.seek(tp, ends[tp] - 1)
            batches = await self._consumer.getmany(tp, timeout_ms=3000, max_records=1)
            for message in batches.get(tp, []):
                if newest is None or message.timestamp > newest.timestamp:
                    newest = message
        if newest is None:
            return None
        key = newest.key.decode() if newest.key is not None else None
        return Record(newest.topic, newest.partition, newest.offset, key, newest.value)

    async def lag(self, group: str, topics: list[str]) -> dict[str, int]:
        """Messages not yet committed by this group, per topic. A group with no commit on a partition counts
        everything retained on it."""
        committed = await self._admin.list_consumer_group_offsets(group)
        out: dict[str, int] = {}
        for topic in topics:
            partitions = await self._partitions(topic)
            if not partitions:
                continue
            ends = await self._consumer.end_offsets(partitions)
            begins = await self._consumer.beginning_offsets(partitions)
            total = 0
            for tp in partitions:
                entry = committed.get(tp)
                position = entry.offset if entry is not None and entry.offset >= 0 else begins[tp]
                total += max(0, ends[tp] - position)
            out[topic] = total
        return out
