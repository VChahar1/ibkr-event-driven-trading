"""In-memory bus for tests, replay and measurements. It models the parts of Kafka that the services
rely on: per-key partition ordering, per-group committed offsets that outlive a consumer, and
redelivery after a restart without a commit. It does not model rebalancing or broker failure.
"""

import asyncio
import zlib

from eventbot.bus.base import Consumer, Record


class InMemoryConsumer:
    def __init__(
        self, bus: "InMemoryBus", topics: list[str], group: str | None, from_beginning: bool
    ) -> None:
        self._bus = bus
        self._topics = topics
        self.group = group
        self._from_beginning = from_beginning
        self._positions: dict[tuple[str, int], int] = {}
        self._started = False

    async def start(self) -> None:
        for topic in self._topics:
            for partition in range(self._bus.partition_count(topic)):
                committed = self._bus.committed(self.group, topic, partition)
                if committed is not None:
                    start = committed
                elif self._from_beginning:
                    start = 0
                else:
                    start = len(self._bus.log(topic, partition))
                self._positions[(topic, partition)] = start
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def _next_ready(self) -> Record | None:
        for (topic, partition), position in self._positions.items():
            log = self._bus.log(topic, partition)
            if position < len(log):
                self._positions[(topic, partition)] = position + 1
                return log[position]
        return None

    async def getone(self, timeout_s: float) -> Record | None:
        if not self._started:
            raise RuntimeError("consumer is not started")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            record = self._next_ready()
            if record is not None:
                return record
            remaining = deadline - loop.time()
            if remaining <= 0:
                await asyncio.sleep(0)
                return None
            waiter = self._bus.new_data_waiter()
            try:
                await asyncio.wait_for(waiter.wait(), remaining)
            except TimeoutError:
                return None

    async def commit(self, record: Record) -> None:
        if self.group is None:
            raise RuntimeError("cannot commit without a consumer group")
        self._bus.set_committed(self.group, record.topic, record.partition, record.offset + 1)

    async def caught_up(self) -> bool:
        return all(
            position >= len(self._bus.log(topic, partition))
            for (topic, partition), position in self._positions.items()
        )


class InMemoryBus:
    def __init__(self, partitions: dict[str, int] | None = None, default_partitions: int = 1) -> None:
        self._partition_counts = dict(partitions or {})
        self._default_partitions = default_partitions
        self._logs: dict[tuple[str, int], list[Record]] = {}
        self._committed: dict[tuple[str, str, int], int] = {}
        self._waiter = asyncio.Event()
        self.fail_publishes = 0

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def partition_count(self, topic: str) -> int:
        return self._partition_counts.get(topic, self._default_partitions)

    def log(self, topic: str, partition: int) -> list[Record]:
        return self._logs.setdefault((topic, partition), [])

    def committed(self, group: str | None, topic: str, partition: int) -> int | None:
        return None if group is None else self._committed.get((group, topic, partition))

    def set_committed(self, group: str, topic: str, partition: int, offset: int) -> None:
        self._committed[(group, topic, partition)] = offset

    def new_data_waiter(self) -> asyncio.Event:
        return self._waiter

    def records(self, topic: str) -> list[Record]:
        """Every record on a topic. Test and replay helper; not part of the Bus interface."""
        out: list[Record] = []
        for partition in range(self.partition_count(topic)):
            out.extend(self.log(topic, partition))
        return out

    async def publish_raw(self, topic: str, key: str | None, value: bytes) -> None:
        if self.fail_publishes > 0:
            self.fail_publishes -= 1
            from eventbot.bus.base import BusError

            raise BusError("injected publish failure")
        count = self.partition_count(topic)
        partition = zlib.crc32((key or "").encode()) % count
        log = self.log(topic, partition)
        log.append(Record(topic, partition, len(log), key, value))

        waiter, self._waiter = self._waiter, asyncio.Event()
        waiter.set()

    def consumer(self, topics: list[str], group: str | None, from_beginning: bool = True) -> Consumer:
        return InMemoryConsumer(self, topics, group, from_beginning)


class InMemoryInspector:
    """The same read-only views as KafkaInspector, for tests."""

    def __init__(self, bus: InMemoryBus) -> None:
        self._bus = bus

    async def latest(self, topic: str) -> Record | None:
        newest: Record | None = None
        for partition in range(self._bus.partition_count(topic)):
            log = self._bus.log(topic, partition)
            if log:
                newest = log[-1]
        return newest

    async def lag(self, group: str, topics: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for topic in topics:
            total = 0
            for partition in range(self._bus.partition_count(topic)):
                end = len(self._bus.log(topic, partition))
                total += end - (self._bus.committed(group, topic, partition) or 0)
            out[topic] = total
        return out
