"""Unit-test harness: in-memory bus, fakeredis and FakeBroker. No Docker and no IB."""

import datetime as dt
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from fakeredis import FakeAsyncRedis

from eventbot.adapter.fake_broker import FakeBroker, FakeBrokerState
from eventbot.adapter.service import GROUP, GatewayAdapter
from eventbot.bus import Consumer, InMemoryBus, TopicMap, publish
from eventbot.config import EventbotConfig, Settings, load_config
from eventbot.schemas import Alert, Fill, Message, OrderEvent, OrderIntent, PositionSnapshot, Side, utc_now
from eventbot.state import StateStore
from stockbot import ny_date


def today() -> dt.date:
    return ny_date(utc_now())


def make_intent(
    symbol: str = "AMD",
    side: Side = Side.BUY,
    qty: float = 4,
    target: float | None = None,
    est_price: float = 600.0,
    strategy: str = "stocks",
    day: dt.date | None = None,
) -> OrderIntent:
    return OrderIntent.build(
        strategy=strategy,
        trading_date=day or today(),
        symbol=symbol,
        side=side,
        target_qty=qty if target is None else target,
        qty=qty,
        est_price=est_price,
        decision_id="d1",
    )


def with_caps(cfg: EventbotConfig, **caps: Any) -> EventbotConfig:
    return cfg.model_copy(update={"caps": cfg.caps.model_copy(update=caps)})


@dataclass
class Harness:
    cfg: EventbotConfig
    settings: Settings
    bus: InMemoryBus
    server: fakeredis.FakeServer
    state: StateStore
    broker: FakeBroker
    topics: TopicMap
    adapter: GatewayAdapter
    root: Path

    def new_adapter(self, broker: FakeBroker | None = None, **kwargs: Any) -> GatewayAdapter:
        """A fresh adapter process: empty memory, same bus, Redis and broker state."""
        self.broker = broker or FakeBroker(self.broker.state)
        self.adapter = GatewayAdapter(
            self.cfg,
            self.settings,
            self.bus,
            self.state,
            self.broker,
            self.topics,
            fill_timeout_s=120.0,
            root=self.root,
            **kwargs,
        )
        return self.adapter

    async def ready(self, price: float = 600.0, symbol: str = "AMD") -> None:
        """Connected, recovered, with fresh heartbeats and a current price: every check would pass."""
        if not self.adapter.connected:
            await self.adapter.startup()
        await self.fresh_beats()
        self.broker.tick(symbol, price)

    async def fresh_beats(self) -> None:
        await self.state.beat("md")
        await self.state.beat("reconciler")

    def messages(self, topic: str, model: type[Message]) -> list[Any]:
        return [model.model_validate_json(r.value) for r in self.bus.records(topic)]

    def alerts(self) -> list[Alert]:
        return self.messages(self.topics.alerts, Alert)

    def alert_codes(self) -> list[Any]:
        return [a.details.get("code") for a in self.alerts()]

    def fills(self) -> list[Fill]:
        return self.messages(self.topics.fills, Fill)

    def order_events(self) -> list[OrderEvent]:
        return self.messages(self.topics.orders_events, OrderEvent)

    def snapshots(self) -> list[PositionSnapshot]:
        return self.messages(self.topics.positions_snapshot, PositionSnapshot)

    async def deliver(self, intent: OrderIntent) -> None:
        await publish(self.bus, self.topics.orders_intents, intent.intent_id, intent)

    async def intent_consumer(self) -> Consumer:
        consumer = self.bus.consumer([self.topics.orders_intents], GROUP)
        await consumer.start()
        return consumer


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    cfg = load_config()
    settings = Settings()
    bus = InMemoryBus()
    server = fakeredis.FakeServer()
    client = FakeAsyncRedis(server=server, decode_responses=True)
    state = StateStore(client, cfg.heartbeats.ttl_s)
    broker = FakeBroker(FakeBrokerState())
    topics = TopicMap.from_config(cfg.topics)
    adapter = GatewayAdapter(cfg, settings, bus, state, broker, topics, fill_timeout_s=120.0, root=tmp_path)
    yield Harness(cfg, settings, bus, server, state, broker, topics, adapter, tmp_path)
    await client.aclose()
