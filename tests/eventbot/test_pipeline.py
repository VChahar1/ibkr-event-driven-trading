"""The whole system at once: strategy, adapter and reconciler running concurrently on one in-memory bus, with
FakeBroker in IB's place. These use the real run() loops, not single-step helpers."""

import asyncio
import datetime as dt
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fakeredis
import pandas as pd
import pytest
from fakeredis import FakeAsyncRedis

from eventbot.adapter.fake_broker import FakeBroker, FakeBrokerState
from eventbot.adapter.service import GatewayAdapter
from eventbot.bus import InMemoryBus, TopicMap
from eventbot.config import EventbotConfig, Settings, load_config
from eventbot.reconciler import Halted, Reconciler
from eventbot.schemas import Alert, AlertType, Fill, OrderEvent, OrderIntent, utc_now
from eventbot.state import StateStore
from eventbot.strategy_service import StrategyService
from stock_strategy import StockConfig
from stockbot import ny_date
from tests.conftest import make_stock_prices

SYMS = [f"S{i:02d}" for i in range(1, 31)]
STOCK_CFG = StockConfig(universe=tuple(SYMS), min_eligible=25)
TRADING: dict[str, Any] = {
    "risk": dict(
        capital_usd=25_000,
        max_order_usd=5_000,
        max_orders_per_run=24,
        min_trade_usd=100,
        cash_buffer=0.02,
        fill_timeout_s=5,
        max_data_age_days=5,
    ),
    "stocks": {"max_price_gap": 0.15, "candidates": 10},
}


def universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": SYMS,
            "yahoo": SYMS,
            "ib_symbol": SYMS,
            "name": SYMS,
            "sector": "X",
            "primary_exchange": "NYSE",
            "con_id": 0,
            "as_of": "2026-10-03",
        }
    )


@dataclass
class System:
    cfg: EventbotConfig
    bus: InMemoryBus
    state: StateStore
    topics: TopicMap
    broker: FakeBroker
    adapter: GatewayAdapter
    strategy: StrategyService
    reconciler: Reconciler
    prices: pd.DataFrame
    tasks: dict[str, asyncio.Task[Any]]

    def fills(self) -> list[Fill]:
        return [Fill.model_validate_json(r.value) for r in self.bus.records(self.topics.fills)]

    def intents(self) -> list[OrderIntent]:
        return [
            OrderIntent.model_validate_json(r.value) for r in self.bus.records(self.topics.orders_intents)
        ]

    def events(self) -> list[OrderEvent]:
        return [OrderEvent.model_validate_json(r.value) for r in self.bus.records(self.topics.orders_events)]

    def alerts(self) -> list[Alert]:
        return [Alert.model_validate_json(r.value) for r in self.bus.records(self.topics.alerts)]

    def stream_prices(self, symbols: list[str] | None = None) -> None:
        latest = self.prices.iloc[-1]
        for symbol in symbols or list(self.broker.subscriptions):
            if symbol in latest.index:
                self.broker.tick(symbol, float(latest[symbol]))

    async def wait_for(self, condition: Callable[[], bool], timeout_s: float = 15.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while not condition():
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("condition not reached")
            self.stream_prices()
            await asyncio.sleep(0.02)

    async def settle_and_reconcile(self) -> str:
        """Publish a broker snapshot after the fills have settled, and compare."""
        await self.adapter.publish_snapshot()
        return (await self.reconciler.check()).status


@pytest.fixture
async def system(tmp_path: Path) -> AsyncIterator[System]:
    cfg = load_config()
    fast = cfg.model_copy(
        update={
            "reconciler": cfg.reconciler.model_copy(update={"settle_s": 0.0}),
            "adapter": cfg.adapter.model_copy(update={"snapshot_interval_s": 0.05}),
            "heartbeats": cfg.heartbeats.model_copy(update={"interval_s": 0.05}),
            "strategy": cfg.strategy.model_copy(update={"price_wait_s": 5.0}),
        }
    )

    today = ny_date(utc_now())
    index_end = pd.Timestamp(today) - pd.offsets.BDay(1)
    prices = make_stock_prices(market_drift=1.5e-3)
    prices.index = pd.bdate_range(end=index_end, periods=len(prices))

    bus = InMemoryBus()
    client = FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    state = StateStore(client, fast.heartbeats.ttl_s)
    topics = TopicMap.from_config(fast.topics)
    broker_state = FakeBrokerState(usd_cash=30_000.0)
    broker = FakeBroker(broker_state, auto_fill=True)
    adapter = GatewayAdapter(fast, Settings(), bus, state, broker, topics, fill_timeout_s=5.0, root=tmp_path)
    strategy = StrategyService(
        fast,
        bus,
        state,
        topics,
        TRADING,
        universe(),
        STOCK_CFG,
        load_prices=lambda *args: prices,
        clock=lambda: dt.datetime.combine(today, dt.time(15, 0), tzinfo=dt.UTC),
        root=tmp_path,
        mode="execute",
    )
    reconciler = Reconciler(fast, bus, state, topics)
    await strategy.start()
    system = System(fast, bus, state, topics, broker, adapter, strategy, reconciler, prices, {})
    system.tasks["adapter"] = asyncio.create_task(adapter.run())
    await asyncio.sleep(0.05)
    system.stream_prices(["SPY"])
    await adapter.beat()
    system.tasks["reconciler"] = asyncio.create_task(reconciler.run())
    await system.wait_for(lambda: reconciler.verified)
    yield system
    system.adapter.request_stop()
    reconciler.request_stop()
    await asyncio.gather(*system.tasks.values(), return_exceptions=True)
    await strategy.stop()
    await client.aclose()


async def rebalance(system: System) -> Any:
    task = asyncio.create_task(system.strategy.evaluate())
    await system.wait_for(task.done, 30.0)
    return task.result()


async def test_a_full_rebalance_through_every_service(system: System) -> None:
    await system.wait_for(lambda: system.adapter.market_open)
    result = await rebalance(system)

    assert result.outcome == "trade", [a.message for a in system.alerts()]
    assert len(system.intents()) == 10
    assert sorted(system.broker.placed) == sorted(i.intent_id for i in system.intents())
    assert len(system.fills()) == 10

    await system.wait_for(lambda: len(system.reconciler.ledger.exec_ids) == 10)
    expected = {i.symbol: i.target_qty for i in system.intents()}
    assert system.broker.state.positions == expected
    assert system.reconciler.ledger.positions == expected
    assert await system.settle_and_reconcile() == "ok"
    assert await system.state.kill_switch() is None
    assert all(await asyncio.gather(*(system.state.get_pos(s) for s in expected)))


async def test_partial_fills_reach_the_right_position_through_every_service(system: System) -> None:
    from eventbot.bus import publish
    from tests.eventbot.conftest import make_intent

    system.broker.auto_fill = False
    system.stream_prices(["AMD"])
    system.broker.tick("AMD", 100.0)
    intent = make_intent(qty=10, target=10, est_price=100.0)
    await publish(system.bus, system.topics.orders_intents, intent.intent_id, intent)
    await system.wait_for(lambda: intent.intent_id in system.broker.state.orders)
    for qty in (3, 3, 4):
        system.broker.fill(intent.intent_id, qty=qty, price=100.0)
    await system.wait_for(lambda: len(system.reconciler.ledger.exec_ids) == 3)
    assert system.reconciler.ledger.position("AMD") == 10
    assert system.adapter.positions["AMD"] == 10
    assert await system.settle_and_reconcile() == "ok"


async def test_a_divergence_halts_the_system_and_leaves_positions_alone(system: System) -> None:
    await system.wait_for(lambda: system.adapter.market_open)
    assert (await rebalance(system)).outcome == "trade"
    await system.wait_for(lambda: len(system.reconciler.ledger.exec_ids) == 10)
    before_orders = list(system.broker.state.orders)

    symbol = next(iter(system.broker.state.positions))
    system.broker.state.positions[symbol] += 5
    before_positions = dict(system.broker.state.positions)

    with pytest.raises(Halted):
        await system.settle_and_reconcile()

    assert await system.state.kill_switch() is not None
    assert AlertType.RECONCILE_MISMATCH in [a.type for a in system.alerts()]
    await asyncio.sleep(0.1)
    assert system.broker.state.positions == before_positions
    assert list(system.broker.state.orders) == before_orders


async def test_the_system_can_not_trade_after_a_halt(system: System) -> None:
    from eventbot.bus import publish
    from tests.eventbot.conftest import make_intent

    await system.wait_for(lambda: system.adapter.market_open)
    await system.state.set_kill_switch("test")
    system.broker.tick("AMD", 100.0)
    intent = make_intent(est_price=100.0)
    await publish(system.bus, system.topics.orders_intents, intent.intent_id, intent)
    await system.wait_for(lambda: bool(system.events()))
    assert system.events()[-1].status == "Rejected"
    assert system.broker.state.orders == {}


async def test_an_adapter_restart_mid_stream_does_not_double_place(system: System) -> None:
    await system.wait_for(lambda: system.adapter.market_open)
    assert (await rebalance(system)).outcome == "trade"
    await system.wait_for(lambda: len(system.reconciler.ledger.exec_ids) == 10)

    system.adapter.request_stop()
    await asyncio.gather(system.tasks["adapter"], return_exceptions=True)
    fresh = FakeBroker(system.broker.state, auto_fill=True)
    placed_before = list(fresh.state.orders)
    system.broker = fresh
    system.adapter = GatewayAdapter(
        system.cfg, Settings(), system.bus, system.state, fresh, system.topics, fill_timeout_s=5.0
    )
    system.tasks["adapter"] = asyncio.create_task(system.adapter.run())
    await system.wait_for(lambda: system.adapter.connected)

    result = await rebalance(system)
    assert result.outcome == "resume"
    assert list(fresh.state.orders) == placed_before and fresh.placed == []
    assert await system.settle_and_reconcile() == "ok"
