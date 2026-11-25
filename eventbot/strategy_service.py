"""The strategy service: consumes market data and broker snapshots, publishes order intents."""

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bot import Abort, decision_path, record_decision
from eventbot.bus import Bus, BusError, Consumer, TopicMap, next_message, publish
from eventbot.config import ROOT, EventbotConfig
from eventbot.decision import (
    failed_price_checks,
    held_symbols,
    make_intents,
    managed_positions,
    plan_stock_orders,
    price_frame_hash,
    require_tradable,
    wanted_symbols,
)
from eventbot.ledger import FillLedger
from eventbot.logs import log_event
from eventbot.schemas import (
    AlertType,
    Fill,
    OrderEvent,
    OrderIntent,
    PositionSnapshot,
    Side,
    Tick,
    alert,
    utc_now,
)
from eventbot.state import StateStore, StateUnavailable
from stock_strategy import StockConfig, decide
from stockbot import NEW_YORK, PREFIX, month_done, ny_date, ny_session_open, symbol_maps

LOG = logging.getLogger("eventbot.strategy")
SERVICE = "strategy"
PUMP_SLICE_S = 0.05
SCHEDULE_TICK_S = 5.0


@dataclass(frozen=True)
class EvaluationResult:
    outcome: str
    intents: tuple[OrderIntent, ...] = ()


def due_slot(now: dt.datetime, times_ny: list[str], attempted: set[str]) -> str | None:
    """The latest scheduled time today that has passed and has not been tried, like the daily timer's
    Persistent=true catch-up. Earlier missed slots are skipped: one attempt covers them."""
    local = now.astimezone(NEW_YORK)
    day = local.date().isoformat()
    past = [t for t in sorted(times_ny) if local.strftime("%H:%M") >= t]
    if not past:
        return None
    slot = f"{day} {past[-1]}"
    return None if slot in attempted else slot


class StrategyService:
    def __init__(
        self,
        cfg: EventbotConfig,
        bus: Bus,
        state: StateStore,
        topics: TopicMap,
        trading: dict[str, Any],
        universe: pd.DataFrame,
        stock_cfg: StockConfig,
        *,
        load_prices: Callable[[pd.DataFrame, StockConfig, dict[str, Any], dt.datetime], pd.DataFrame],
        clock: Callable[[], dt.datetime] = utc_now,
        root: Path = ROOT,
        mode: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._scfg = cfg.strategy
        self._bus = bus
        self._state = state
        self._topics = topics
        self._trading = trading
        self._risk: dict[str, Any] = trading["risk"]
        self._universe = universe
        self._stock_cfg = stock_cfg
        self._load_prices = load_prices
        self._clock = clock
        self._root = root
        self.mode = mode or self._scfg.mode
        self._state_dir = root / self._scfg.state_dir
        self._to_ib, _ = symbol_maps(universe)

        self.ledger = FillLedger()
        self.snapshot: PositionSnapshot | None = None
        self.snapshot_received: float | None = None
        self.order_states: dict[str, OrderEvent] = {}
        self.prices: dict[str, tuple[float, float]] = {}
        self._history: Consumer | None = None
        self._ticks: Consumer | None = None
        self._attempted: set[str] = set()
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """History topics from the beginning with no group: their state is rebuilt on each start, and
        applying them is idempotent. Ticks only from now on: an old price is of no use."""
        self._history = self._bus.consumer(
            [self._topics.orders_events, self._topics.fills, self._topics.positions_snapshot], None
        )
        self._ticks = self._bus.consumer([self._topics.md_ticks], None, from_beginning=False)
        await self._history.start()
        await self._ticks.start()

    async def stop(self) -> None:
        for consumer in (self._history, self._ticks):
            if consumer is not None:
                await consumer.stop()

    def request_stop(self) -> None:
        self._stop.set()

    def _apply(self, message: Any) -> None:
        if isinstance(message, Tick):
            self.prices[message.symbol] = (message.price, time.monotonic())
        elif isinstance(message, OrderEvent):
            self.order_states[message.order_ref] = message
        elif isinstance(message, Fill):
            self.ledger.apply(message)
        elif isinstance(message, PositionSnapshot):
            if message.kind == "baseline":
                self.ledger.rebase(message.positions, message.baseline_exec_ids)
            else:
                self.snapshot = message
                self.snapshot_received = time.monotonic()

    async def pump(self, timeout_s: float = 0.0) -> None:
        """Apply everything available now, waiting at most timeout_s for the first message."""
        assert self._history is not None and self._ticks is not None
        for consumer in (self._history, self._ticks):
            delivery = await next_message(
                consumer, self._bus, self._topics, service=SERVICE, timeout_s=timeout_s
            )
            while delivery is not None:
                self._apply(delivery.message)
                delivery = await next_message(
                    consumer, self._bus, self._topics, service=SERVICE, timeout_s=0.0
                )

    async def _alert_abort(self, message: str, **details: Any) -> None:
        log_event(LOG, "strategy_abort", level=logging.ERROR, msg=message, **details)
        await publish(
            self._bus,
            self._topics.alerts,
            AlertType.STRATEGY_ABORT.value,
            alert(AlertType.STRATEGY_ABORT, "critical", SERVICE, message, **details),
        )

    def _load_marker(self, day: dt.date) -> dict[str, Any] | None:
        path = decision_path(self._state_dir, day, PREFIX)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    async def evaluate(self) -> EvaluationResult:
        """One scheduled run. An Abort means nothing was decided, so the next slot can try again, as the
        daily bot's 15:00 run retries a failed 10:30 run."""
        try:
            return await self._evaluate()
        except Abort as exc:
            await self._alert_abort(str(exc))
            return EvaluationResult("abort")
        except StateUnavailable as exc:
            await self._alert_abort(f"Redis is unavailable, no decision made: {exc}")
            return EvaluationResult("abort")
        except BusError:
            raise
        except Exception as exc:
            LOG.exception("Evaluation failed")
            await self._alert_abort(f"Evaluation failed: {exc!r}")
            return EvaluationResult("abort")

    async def _evaluate(self) -> EvaluationResult:
        now = self._clock()
        today = ny_date(now)
        await self.pump()
        execute = self.mode == "execute"

        if execute:
            if not ny_session_open(now):
                return EvaluationResult("closed")
            reason = await self._state.kill_switch()
            if reason is not None:
                raise Abort(f"Kill switch is on: {reason}")
            marker = self._load_marker(today)
            if marker is not None:
                return await self._resume(marker)
        if month_done(self._state_dir, today):
            log_event(LOG, "hold", msg="This month's rebalance is already recorded")
            return EvaluationResult("hold")

        snapshot = self._require_fresh_snapshot()
        if abs(snapshot.clock_skew_s or 0.0) > self._cfg.adapter.max_clock_skew_s:
            raise Abort(f"Local clock is {snapshot.clock_skew_s} s off IB server time")

        orphans = {s for s in self.ledger.bought_net() if s not in self._to_ib}
        positions = managed_positions(snapshot.positions, self._to_ib, orphans)
        held = held_symbols(positions)
        usd_cash = snapshot.usd_cash
        if usd_cash is None:
            raise Abort("The latest broker snapshot has no USD cash")

        prices_df = await asyncio.to_thread(
            self._load_prices, self._universe, self._stock_cfg, self._trading, now
        )
        dec = await asyncio.to_thread(decide, prices_df, self._stock_cfg, held & set(self._to_ib), today)
        log_event(
            LOG,
            "signal",
            signal_date=dec.signal_date.date(),
            market_on=dec.market_on,
            picks=dec.picks,
            held=sorted(held),
        )

        stocks = self._trading.get("stocks", {})
        wanted = wanted_symbols(held, dec, self._stock_cfg, int(stocks.get("candidates", 10)))
        ibkr = await self._collect_prices(wanted)
        yahoo = prices_df.iloc[-1]
        gap_limit = float(stocks.get("max_price_gap", 0.15))
        bad = failed_price_checks(wanted, ibkr, yahoo, gap_limit)
        require_tradable(bad, held, dec, gap_limit)

        plan = plan_stock_orders(dec, positions, ibkr, bad, usd_cash, self._risk, self._stock_cfg)
        log_event(LOG, "plan", budget=plan.budget, orders=plan.orders, picks=plan.picks)
        if not execute:
            return EvaluationResult("dry")

        if not snapshot.market_open:
            log_event(LOG, "skip", reason="market closed (IBKR calendar)")
            return EvaluationResult("closed")
        if plan.orders and (self._root / self._cfg.adapter.stop_file).exists():
            raise Abort("Kill switch file STOP appeared - no orders placed")
        decision_id = f"{self._scfg.name}-{today:%Y%m%d}"
        intents = make_intents(plan.orders, positions, self._scfg.name, today, decision_id)
        self._record(today, decision_id, dec, plan, intents, prices_df, positions, usd_cash, ibkr)
        return await self._run_batch(intents, "trade" if intents else "no-op")

    def _require_fresh_snapshot(self) -> PositionSnapshot:
        if self.snapshot is None or self.snapshot_received is None:
            raise Abort("No broker snapshot received yet: is the adapter running?")
        age = time.monotonic() - self.snapshot_received
        if age > self._scfg.snapshot_max_age_s:
            raise Abort(f"The latest broker snapshot is {age:.0f} s old")
        return self.snapshot

    async def _collect_prices(self, wanted: set[str]) -> dict[str, float]:
        """Ask the adapter for market data and wait for it. A symbol with no tick stays missing, which the
        price check then treats exactly as ib_prices' NaN."""
        await self._state.add_to_watchlist(sorted(wanted))
        max_age = self._cfg.heartbeats.md_max_tick_age_s
        deadline = time.monotonic() + self._scfg.price_wait_s

        def fresh() -> dict[str, float]:
            cutoff = time.monotonic() - max_age
            return {s: self.prices[s][0] for s in wanted if s in self.prices and self.prices[s][1] >= cutoff}

        while time.monotonic() < deadline:
            await self.pump(PUMP_SLICE_S)
            collected = fresh()
            if set(collected) == wanted:
                return collected
        collected = fresh()
        log_event(
            LOG,
            "prices_incomplete",
            level=logging.WARNING,
            missing=sorted(wanted - set(collected)),
            waited_s=self._scfg.price_wait_s,
        )
        return collected

    def _record(
        self,
        today: dt.date,
        decision_id: str,
        dec: Any,
        plan: Any,
        intents: list[OrderIntent],
        prices_df: pd.DataFrame,
        positions: dict[str, int],
        usd_cash: float,
        ibkr: dict[str, float],
    ) -> None:
        """Freeze the decision, with the inputs that replay needs. The price frame goes first: a marker
        must never point at a file that does not exist."""
        decisions_dir = self._state_dir / "decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        prices_df.to_parquet(decisions_dir / f"{decision_id}.parquet")
        record_decision(
            self._state_dir,
            today,
            {
                "ts": self._clock().isoformat(),
                "decision_id": decision_id,
                "signal_date": dec.signal_date.date(),
                "market_on": dec.market_on,
                "picks": plan.picks,
                "decision": "trade" if plan.orders else "no-op",
                "orders": plan.orders,
                "intents": [i.model_dump(mode="json", exclude={"msg_id", "ts_utc"}) for i in intents],
                "inputs": {
                    "prices_sha256": price_frame_hash(prices_df),
                    "positions": positions,
                    "usd_cash": usd_cash,
                    "ibkr_prices": ibkr,
                },
            },
            prefix=PREFIX,
        )

    async def _resume(self, marker: dict[str, Any]) -> EvaluationResult:
        """Today's decision exists: never re-size. Same intents, same IDs; the adapter drops any already
        handled. Those already seen as handled are not even re-sent."""
        intents = [OrderIntent.model_validate(stored) for stored in marker.get("intents", [])]
        log_event(LOG, "resume", decision_id=marker.get("decision_id"), intents=len(intents))
        if not intents:
            return EvaluationResult("no-op")
        return await self._run_batch(intents, "resume")

    async def _run_batch(self, intents: list[OrderIntent], label: str) -> EvaluationResult:
        """Sells first, then buys only if every sell filled. The same rule as bot.execute_orders."""
        sells = [i for i in intents if i.side is Side.SELL]
        buys = [i for i in intents if i.side is Side.BUY]
        for name, batch in (("SELL", sells), ("BUY", buys)):
            if not batch:
                continue
            reason = await self._state.kill_switch()
            if reason is not None:
                await self._alert_abort(f"Kill switch set before the {name} orders: {reason}")
                return EvaluationResult("abort", tuple(intents))
            await self._publish_unhandled(batch)
            if not await self._await_filled(batch):
                unfilled = [
                    i.intent_id
                    for i in batch
                    if (s := self.order_states.get(i.intent_id)) is None or s.status != "Filled"
                ]
                await self._alert_abort(f"{name} orders not fully filled - stopping", unfilled=unfilled)
                return EvaluationResult("abort", tuple(intents))
        return EvaluationResult(label, tuple(intents))

    async def _publish_unhandled(self, batch: list[OrderIntent]) -> None:
        await self.pump()
        for stored in batch:
            if stored.intent_id in self.order_states:
                continue

            intent = OrderIntent.build(
                strategy=stored.strategy,
                trading_date=stored.trading_date,
                symbol=stored.symbol,
                side=stored.side,
                target_qty=stored.target_qty,
                qty=stored.qty,
                est_price=stored.est_price,
                decision_id=stored.decision_id,
            )
            await publish(self._bus, self._topics.orders_intents, intent.intent_id, intent)
            log_event(
                LOG,
                "intent_published",
                intent_id=intent.intent_id,
                order_ref=intent.intent_id,
                symbol=intent.symbol,
                side=intent.side.value,
                qty=intent.qty,
                target_qty=intent.target_qty,
            )

    async def _await_filled(self, batch: list[OrderIntent]) -> bool:
        """Wait for a terminal state on every order. The adapter cancels at fill_timeout_s, so the margin
        here only matters if the adapter itself has gone away."""
        timeout = float(self._risk["fill_timeout_s"]) + self._scfg.fill_wait_margin_s
        deadline = time.monotonic() + timeout
        refs = [i.intent_id for i in batch]
        while True:
            await self.pump(PUMP_SLICE_S)
            states = [self.order_states.get(r) for r in refs]
            if all(s is not None and s.terminal for s in states):
                return all(s is not None and s.status == "Filled" for s in states)
            if time.monotonic() >= deadline:
                return False

    async def run(self) -> None:
        await self.start()
        try:
            await asyncio.gather(self._schedule_loop(), self._heartbeat_loop())
        finally:
            await self.stop()

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._state.beat(SERVICE)
            except StateUnavailable as exc:
                log_event(LOG, "heartbeat_skipped", level=logging.WARNING, msg=str(exc))
            await self._sleep(self._cfg.heartbeats.interval_s)

    async def _schedule_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.pump(PUMP_SLICE_S)
                slot = due_slot(self._clock(), self._scfg.run_times_ny, self._attempted)
                if slot is not None:
                    self._attempted.add(slot)
                    result = await self.evaluate()
                    log_event(
                        LOG, "evaluated", slot=slot, outcome=result.outcome, intents=len(result.intents)
                    )
            except BusError as exc:
                log_event(LOG, "bus_unavailable", level=logging.WARNING, msg=str(exc))
            await self._sleep(SCHEDULE_TICK_S)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), seconds)
        except TimeoutError:
            pass


async def amain(once: bool) -> int:
    """Build and run the stock-strategy scheduling service."""
    from backtest_stocks import stock_config
    from eventbot.runtime import build_runtime, install_stop_handlers
    from stockbot import load_signal_prices
    from universe import load_universe

    rt = build_runtime(SERVICE)
    uni = load_universe(ROOT / rt.trading["stocks"]["universe_file"])
    service = StrategyService(
        rt.cfg,
        rt.bus,
        rt.state,
        rt.topics,
        rt.trading,
        uni,
        stock_config(rt.trading, uni),
        load_prices=load_signal_prices,
    )
    install_stop_handlers(service.request_stop)
    await rt.bus.start()
    try:
        if once:
            await service.start()
            await service.pump(2.0)
            result = await service.evaluate()
            print(f"outcome={result.outcome} intents={len(result.intents)} mode={service.mode}")
            await service.stop()
            return 0
        await service.run()
        return 0
    finally:
        await rt.close()


def main() -> int:
    """Start the configured asynchronous strategy service."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--once", action="store_true", help="evaluate once now, then exit")
    return asyncio.run(amain(parser.parse_args().once))


if __name__ == "__main__":
    sys.exit(main())
