"Bus interface plus typed publish and validated consume."

import logging
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from eventbot.config import TopicsConfig
from eventbot.logs import log_event
from eventbot.schemas import (
    Alert,
    AlertType,
    Fill,
    Message,
    OrderEvent,
    OrderIntent,
    PositionSnapshot,
    Tick,
    alert,
)

LOG = logging.getLogger("eventbot.bus")
RAW_PREVIEW_BYTES = 500


class BusError(RuntimeError):
    """The bus could not carry out a request. Callers treat this as transient: they retry and do
    not commit, so nothing is lost."""


@dataclass(frozen=True)
class Record:
    topic: str
    partition: int
    offset: int
    key: str | None
    value: bytes


class Consumer(Protocol):
    group: str | None

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def getone(self, timeout_s: float) -> Record | None:
        """The next record, or None if there is none within the timeout."""
        ...

    async def commit(self, record: Record) -> None:
        """Mark this record and everything before it in its partition as processed."""
        ...

    async def caught_up(self) -> bool:
        """True once every record that existed at the time of the call has been returned."""
        ...


class Bus(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def publish_raw(self, topic: str, key: str | None, value: bytes) -> None:
        """Returns once the broker has acknowledged the message (acks=all on Kafka)."""
        ...

    def consumer(self, topics: list[str], group: str | None, from_beginning: bool = True) -> Consumer:
        """group=None reads with no committed offsets. from_beginning applies when there is no
        committed offset for the group."""
        ...


@dataclass(frozen=True)
class TopicMap:
    """Topic names as configured, and which schema each one carries."""

    md_ticks: str
    orders_intents: str
    orders_events: str
    fills: str
    positions_snapshot: str
    alerts: str

    @classmethod
    def from_config(cls, topics: TopicsConfig, prefix: str = "") -> "TopicMap":
        return cls(
            md_ticks=prefix + topics.md_ticks.name,
            orders_intents=prefix + topics.orders_intents.name,
            orders_events=prefix + topics.orders_events.name,
            fills=prefix + topics.fills.name,
            positions_snapshot=prefix + topics.positions_snapshot.name,
            alerts=prefix + topics.alerts.name,
        )

    def all(self) -> list[str]:
        return [
            self.md_ticks,
            self.orders_intents,
            self.orders_events,
            self.fills,
            self.positions_snapshot,
            self.alerts,
        ]

    def model_for(self, topic: str) -> type[Message]:
        mapping: dict[str, type[Message]] = {
            self.md_ticks: Tick,
            self.orders_intents: OrderIntent,
            self.orders_events: OrderEvent,
            self.fills: Fill,
            self.positions_snapshot: PositionSnapshot,
            self.alerts: Alert,
        }
        return mapping[topic]


@dataclass(frozen=True)
class Delivery:
    record: Record
    message: Message


async def publish(bus: Bus, topic: str, key: str | None, message: Message) -> None:
    await bus.publish_raw(topic, key, message.model_dump_json().encode())


def decode(topics: TopicMap, record: Record) -> Message:
    """Raises ValidationError or KeyError for an invalid message or an unknown topic."""
    return topics.model_for(record.topic).model_validate_json(record.value)


async def next_message(
    consumer: Consumer, bus: Bus, topics: TopicMap, *, service: str, timeout_s: float
) -> Delivery | None:
    "The next valid message, or None if there is none available."
    while True:
        record = await consumer.getone(timeout_s)
        if record is None:
            return None
        try:
            return Delivery(record, decode(topics, record))
        except (ValidationError, KeyError, ValueError) as exc:
            preview = record.value[:RAW_PREVIEW_BYTES].decode("utf-8", errors="replace")
            try:
                await publish(
                    bus,
                    topics.alerts,
                    AlertType.INVALID_MESSAGE.value,
                    alert(
                        AlertType.INVALID_MESSAGE,
                        "warning",
                        service,
                        f"Invalid message on {record.topic}",
                        topic=record.topic,
                        partition=record.partition,
                        offset=record.offset,
                        key=record.key,
                        error=str(exc)[:RAW_PREVIEW_BYTES],
                        raw=preview,
                    ),
                )
                if consumer.group is not None:
                    await consumer.commit(record)
            except BusError as bus_exc:
                log_event(
                    LOG,
                    "invalid_message_alert_failed",
                    level=logging.ERROR,
                    msg=str(bus_exc),
                    topic=record.topic,
                    offset=record.offset,
                )
        timeout_s = 0.0
