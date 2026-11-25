"""The strategy service alone. A small scripted stand-in plays the adapter's part on the bus."""

import asyncio
import datetime as dt
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fakeredis
import pandas as pd
import pytest
from fakeredis import FakeAsyncRedis

from eventbot.bus import InMemoryBus, TopicMap, publish
from eventbot.config import EventbotConfig, load_config
from eventbot.schemas import (
    Alert,
    AlertType,
    Fill,
    OrderEvent,
    OrderIntent,
    PositionSnapshot,
    Side,
    Tick,
)
from eventbot.state import StateStore
from eventbot.strategy_service import EvaluationResult, StrategyService, due_slot
from stock_strategy import StockConfig
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
        fill_timeout_s=0,
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
class Env:
    service: StrategyService
    bus: InMemoryBus
    state: StateStore
    topics: TopicMap
    prices: pd.DataFrame
    root: Path
    day: dt.date
    cfg: EventbotConfig
    calls: dict[str, int]

    def latest(self) -> dict[str, float]:
        return self.prices.iloc[-1].to_dict()

    def intents(self) -> list[OrderIntent]:
        return [
            OrderIntent.model_validate_json(r.value) for r in self.bus.records(self.topics.orders_intents)
        ]

    def alerts(self) -> list[Alert]:
        return [Alert.model_validate_json(r.value) for r in self.bus.records(self.topics.alerts)]

    async def snapshot(
        self, positions: dict[str, float] | None = None, cash: float = 30_000.0, **over: Any
    ) -> None:
        fields: dict[str, Any] = dict(
            account="DU***000",
            kind="periodic",
            positions=positions or {},
            usd_cash=cash,
            market_open=True,
            clock_skew_s=0.5,
        )
        fields.update(over)
        await publish(self.bus, self.topics.positions_snapshot, "DU***000", PositionSnapshot(**fields))

    async def ticks(self, overrides: dict[str, float] | None = None) -> None:
        for symbol, price in (self.latest() | (overrides or {})).items():
            if price == price and price > 0:
                await publish(
                    self.bus, self.topics.md_ticks, symbol, Tick(symbol=symbol, price=price, md_type=3)
                )

    async def order_event(self, intent: OrderIntent, status: str = "Filled") -> None:
        await publish(
            self.bus,
            self.topics.orders_events,
            intent.intent_id,
            OrderEvent(
                order_ref=intent.intent_id,
                symbol=intent.symbol,
                side=intent.side,
                status=status,
                qty=intent.qty,
                filled=intent.qty if status == "Filled" else 0.0,
            ),
        )

    def marker(self) -> dict[str, Any]:
        path = self.root / self.cfg.strategy.state_dir / f"decided-stocks-{self.day:%Y%m%d}.json"
        return json.loads(path.read_text())

    async def run_with_fake_adapter(
        self, respond: Callable[[OrderIntent], Awaitable[None]] | None = None
    ) -> EvaluationResult:
        """evaluate() while a stand-in adapter answers each intent."""
        answered: set[str] = set()

        async def answer() -> None:
            while True:
                for intent in self.intents():
                    if intent.intent_id not in answered:
                        answered.add(intent.intent_id)
                        if respond is None:
                            await self.order_event(intent)
                        else:
                            await respond(intent)
                await asyncio.sleep(0.005)

        task = asyncio.create_task(answer())
        try:
            return await asyncio.wait_for(self.service.evaluate(), 10.0)
        finally:
            task.cancel()


@pytest.fixture
async def env(tmp_path: Path) -> AsyncIterator[Env]:
    cfg = load_config()
    fast = cfg.strategy.model_copy(update={"price_wait_s": 0.05, "fill_wait_margin_s": 0.2})
    cfg = cfg.model_copy(update={"strategy": fast})
    bus = InMemoryBus()
    client = FakeAsyncRedis(server=fakeredis.FakeServer(), decode_responses=True)
    state = StateStore(client)
    topics = TopicMap.from_config(cfg.topics)
    prices = make_stock_prices(market_drift=1.5e-3)
    day = (prices.index[-1] + pd.offsets.BDay(1)).date()
    now = dt.datetime.combine(day, dt.time(15, 0), tzinfo=dt.UTC)
    calls = {"load": 0}

    def load_prices(*args: Any) -> pd.DataFrame:
        calls["load"] += 1
        return prices

    service = StrategyService(
        cfg,
        bus,
        state,
        topics,
        TRADING,
        universe(),
        STOCK_CFG,
        load_prices=load_prices,
        clock=lambda: now,
        root=tmp_path,
        mode="execute",
    )
    await service.start()
    yield Env(service, bus, state, topics, prices, tmp_path, day, cfg, calls)
    await service.stop()
    await client.aclose()


async def ready(env: Env, positions: dict[str, float] | None = None, cash: float = 30_000.0) -> None:
    await env.snapshot(positions, cash)
    await env.ticks()


async def test_dry_mode_logs_a_plan_and_publishes_nothing(env: Env) -> None:
    env.service.mode = "dry"
    await ready(env)
    result = await env.service.evaluate()
    assert result.outcome == "dry" and env.intents() == []
    assert not (env.root / env.cfg.strategy.state_dir).exists()


async def test_the_committed_default_mode_is_dry() -> None:
    assert load_config().strategy.mode == "dry"


async def test_a_rebalance_publishes_one_intent_per_order_keyed_by_intent_id(env: Env) -> None:
    await ready(env)
    result = await env.run_with_fake_adapter()
    intents = env.intents()
    assert result.outcome == "trade" and len(intents) == 10
    assert all(i.side is Side.BUY and i.strategy == "stocks" and i.trading_date == env.day for i in intents)
    records = env.bus.records(env.topics.orders_intents)
    assert all(r.key == OrderIntent.model_validate_json(r.value).intent_id for r in records)


async def test_the_decision_is_frozen_with_its_inputs(env: Env) -> None:
    await ready(env)
    await env.run_with_fake_adapter()
    marker = env.marker()
    assert marker["decision"] == "trade" and len(marker["intents"]) == 10
    assert marker["inputs"]["usd_cash"] == 30_000.0 and len(marker["inputs"]["prices_sha256"]) == 64
    frame = env.root / env.cfg.strategy.state_dir / "decisions" / f"stocks-{env.day:%Y%m%d}.parquet"
    assert pd.read_parquet(frame).shape == env.prices.shape


async def test_sells_go_first_and_buys_only_after_every_sell_filled(env: Env) -> None:
    await ready(env, positions={"S01": 40.0, "S02": 40.0}, cash=20_000.0)
    published_when_sell_answered: list[int] = []

    async def respond(intent: OrderIntent) -> None:
        if intent.side is Side.SELL:
            published_when_sell_answered.append(sum(i.side is Side.BUY for i in env.intents()))
        await env.order_event(intent)

    result = await env.run_with_fake_adapter(respond)
    sides = [i.side for i in env.intents()]
    assert result.outcome == "trade" and Side.SELL in sides and Side.BUY in sides
    assert sides == sorted(sides, key=lambda s: s is Side.BUY)
    assert all(count == 0 for count in published_when_sell_answered)


@pytest.mark.parametrize("status", ["Cancelled", "Rejected", "Inactive"])
async def test_an_unfilled_sell_stops_the_buys(env: Env, status: str) -> None:
    await ready(env, positions={"S01": 40.0, "S02": 40.0}, cash=20_000.0)

    async def respond(intent: OrderIntent) -> None:
        await env.order_event(intent, status)

    result = await env.run_with_fake_adapter(respond)
    assert result.outcome == "abort"
    assert all(i.side is Side.SELL for i in env.intents())
    assert AlertType.STRATEGY_ABORT in [a.type for a in env.alerts()]


async def test_no_answer_from_the_adapter_stops_the_buys_after_the_timeout(env: Env) -> None:
    await ready(env, positions={"S01": 40.0, "S02": 40.0}, cash=20_000.0)

    async def respond(intent: OrderIntent) -> None:
        return None

    result = await env.run_with_fake_adapter(respond)
    assert result.outcome == "abort" and all(i.side is Side.SELL for i in env.intents())


async def test_a_restart_on_the_same_day_republishes_the_same_ids_without_resizing(env: Env) -> None:
    """A re-sized decision is how two IDs could exist for one decision. Prices move here, which would
    change the sizing if it ran again, as it did for AMD on 2026-10-09."""
    await ready(env)

    async def unanswered(intent: OrderIntent) -> None:
        return None

    await env.run_with_fake_adapter(unanswered)
    first = {i.intent_id: i.target_qty for i in env.intents()}
    loads_before = env.calls["load"]

    await env.ticks({s: p * 1.02 for s, p in env.latest().items()})
    result = await env.run_with_fake_adapter()

    assert result.outcome == "resume" and env.calls["load"] == loads_before
    assert {i.intent_id: i.target_qty for i in env.intents()} == first


async def test_a_resume_does_not_resend_intents_already_handled(env: Env) -> None:
    await ready(env)
    await env.run_with_fake_adapter()
    count = len(env.intents())
    result = await env.service.evaluate()
    assert result.outcome == "resume" and len(env.intents()) == count


async def test_a_later_day_in_the_month_holds_and_publishes_nothing(env: Env, tmp_path: Path) -> None:
    await ready(env)
    await env.run_with_fake_adapter()
    count = len(env.intents())
    if env.day.month == (env.day + dt.timedelta(days=1)).month:
        later = dt.datetime.combine(env.day + dt.timedelta(days=1), dt.time(15, 0), tzinfo=dt.UTC)
        env.service._clock = lambda: later
        assert (await env.service.evaluate()).outcome in ("hold", "closed")
        assert len(env.intents()) == count


async def test_a_no_op_decision_is_recorded_and_holds_for_the_month(env: Env) -> None:
    prices = make_stock_prices(market_drift=-1.5e-3)
    env.service._load_prices = lambda *args: prices
    env.prices = prices
    await ready(env)
    result = await env.service.evaluate()
    assert env.intents() == []
    if result.outcome == "no-op":
        assert env.marker()["decision"] == "no-op"


async def test_the_kill_switch_stops_a_decision(env: Env) -> None:
    await ready(env)
    await env.state.set_kill_switch("mismatch")
    assert (await env.service.evaluate()).outcome == "abort"
    assert env.intents() == [] and env.calls["load"] == 0


async def test_the_kill_switch_set_after_the_sells_stops_the_buys(env: Env) -> None:
    await ready(env, positions={"S01": 40.0, "S02": 40.0}, cash=20_000.0)

    async def respond(intent: OrderIntent) -> None:
        await env.state.set_kill_switch("mismatch")
        await env.order_event(intent)

    result = await env.run_with_fake_adapter(respond)
    assert result.outcome == "abort" and all(i.side is Side.SELL for i in env.intents())


async def test_the_stop_file_stops_a_decision_before_anything_is_recorded(env: Env) -> None:
    await ready(env)
    (env.root / "STOP").touch()
    assert (await env.service.evaluate()).outcome == "abort"
    assert env.intents() == [] and not (env.root / env.cfg.strategy.state_dir).exists()


async def test_no_snapshot_means_no_decision(env: Env) -> None:
    await env.ticks()
    assert (await env.service.evaluate()).outcome == "abort"
    assert env.intents() == []


async def test_a_stale_snapshot_means_no_decision(env: Env) -> None:
    await ready(env)
    await env.service.pump()
    assert env.service.snapshot_received is not None
    env.service.snapshot_received -= 1000
    assert (await env.service.evaluate()).outcome == "abort"
    assert env.intents() == []


async def test_a_skewed_clock_means_no_decision(env: Env) -> None:
    await env.snapshot(clock_skew_s=3600.0)
    await env.ticks()
    assert (await env.service.evaluate()).outcome == "abort"
    assert env.intents() == []


async def test_a_closed_market_per_ibkr_means_no_decision_and_no_marker(env: Env) -> None:
    await env.snapshot(market_open=False)
    await env.ticks()
    assert (await env.service.evaluate()).outcome == "closed"
    assert env.intents() == [] and not (env.root / env.cfg.strategy.state_dir).exists()


async def test_outside_new_york_hours_is_closed_without_loading_anything(env: Env) -> None:
    night = dt.datetime.combine(env.day, dt.time(3, 0), tzinfo=dt.UTC)
    env.service._clock = lambda: night
    assert (await env.service.evaluate()).outcome == "closed"
    assert env.calls["load"] == 0


async def test_a_yahoo_outage_alerts_decides_nothing_and_a_later_slot_can_retry(env: Env) -> None:
    """2026-10-08: the 10:30 run crashed on 'Yahoo returned no data'; the 15:00 run was fine."""
    prices = env.prices

    def broken(*args: Any) -> pd.DataFrame:
        raise RuntimeError("Yahoo returned no data for ['META']")

    await ready(env)
    env.service._load_prices = broken
    assert (await env.service.evaluate()).outcome == "abort"
    assert env.intents() == [] and AlertType.STRATEGY_ABORT in [a.type for a in env.alerts()]

    env.service._load_prices = lambda *args: prices
    assert (await env.run_with_fake_adapter()).outcome == "trade"


async def test_a_missing_market_data_price_on_a_pick_aborts(env: Env) -> None:
    await env.snapshot()
    assert (await env.service.evaluate()).outcome == "abort"
    assert env.intents() == []
    assert any("Price check failed" in a.message for a in env.alerts())


async def test_prices_come_through_the_watchlist(env: Env) -> None:
    await ready(env)
    await env.run_with_fake_adapter()
    watch = await env.state.watchlist()
    assert {i.symbol for i in env.intents()} <= watch and len(watch) >= 10


async def test_a_stock_bought_by_fills_that_left_the_universe_is_sold(env: Env) -> None:
    """stockbot's orphan rule, with the ledger in place of journal.jsonl."""
    now = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    await publish(
        env.bus,
        env.topics.fills,
        "e.1.01",
        Fill(
            exec_id="e.1.01",
            order_ref="a" * 32,
            symbol="GONE",
            side=Side.BUY,
            qty=5,
            price=50.0,
            exec_time_utc=now,
        ),
    )
    await env.snapshot({"GONE": 5.0})
    await env.ticks({"GONE": 50.0})
    env.prices["GONE"] = 50.0
    await env.run_with_fake_adapter()
    sold = [i for i in env.intents() if i.symbol == "GONE"]
    assert [(i.side, i.qty, i.target_qty) for i in sold] == [(Side.SELL, 5, 0)]


async def test_a_position_outside_the_universe_with_no_fills_is_not_touched(env: Env) -> None:
    await ready(env, positions={"CSPX": 5.0})
    await env.run_with_fake_adapter()
    assert all(i.symbol != "CSPX" for i in env.intents())


async def test_an_invalid_message_on_a_history_topic_does_not_crash_the_service(env: Env) -> None:
    await env.bus.publish_raw(env.topics.orders_events, "x", b"garbage")
    await env.bus.publish_raw(env.topics.md_ticks, "x", b"garbage")
    await ready(env)
    assert (await env.run_with_fake_adapter()).outcome == "trade"
    assert [a.type for a in env.alerts()].count(AlertType.INVALID_MESSAGE) == 2


def ny(hour: int, minute: int, day: int = 2) -> dt.datetime:
    return dt.datetime(2026, 11, day, hour + 5, minute, tzinfo=dt.UTC)


@pytest.mark.parametrize(
    "now,attempted,expected",
    [
        (ny(9, 0), set(), None),
        (ny(10, 30), set(), "2026-11-02 10:30"),
        (ny(11, 0), {"2026-11-02 10:30"}, None),
        (ny(15, 0), {"2026-11-02 10:30"}, "2026-11-02 15:00"),
        (ny(15, 1), {"2026-11-02 10:30", "2026-11-02 15:00"}, None),
        (ny(12, 0), set(), "2026-11-02 10:30"),
        (ny(15, 30), set(), "2026-11-02 15:00"),
        (ny(10, 30, day=3), {"2026-11-02 10:30"}, "2026-11-03 10:30"),
    ],
)
def test_due_slot(now: dt.datetime, attempted: set[str], expected: str | None) -> None:
    assert due_slot(now, ["10:30", "15:00"], attempted) == expected


async def test_the_service_loop_runs_a_slot_once_and_stops_cleanly(env: Env) -> None:
    env.service.mode = "dry"
    await ready(env)
    task = asyncio.create_task(env.service.run())
    await asyncio.sleep(0.2)
    env.service.request_stop()
    await asyncio.wait_for(task, 5.0)
    assert env.intents() == []
    assert await env.state.heartbeat_age_s("strategy") is not None
