"""The gateway adapter: consumes order intents, runs pre-trade checks, places orders, and
publishes market data, order events, fills and position snapshots."""

import asyncio
import datetime as dt
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eventbot.adapter.broker import (
    BrokerClient,
    BrokerExecution,
    BrokerHandlers,
    BrokerOrder,
    BrokerTick,
    BrokerUnavailable,
)
from eventbot.bus import Bus, BusError, Consumer, TopicMap, next_message, publish
from eventbot.config import ROOT, EventbotConfig, Settings
from eventbot.logs import log_event
from eventbot.paper_guard import assert_paper_account, assert_paper_port, mask_account
from eventbot.risk import PreTradeContext, check
from eventbot.schemas import (
    AlertType,
    Fill,
    IntentStatus,
    Message,
    OrderEvent,
    OrderIntent,
    PositionSnapshot,
    Side,
    Tick,
    alert,
    is_intent_ref,
    utc_now,
)
from eventbot.state import StateStore, StateUnavailable
from stockbot import ny_date

LOG = logging.getLogger("eventbot.adapter")
SERVICE = "adapter"
GROUP = "adapter"
TICK_FLUSH_S = 0.25
RETRY_DELAY_S = 1.0
CONNECTIVITY_LOST = 1100
CONNECTIVITY_RESTORED = frozenset({1101, 1102})

INFORMATIONAL_CODES = frozenset({10167, *range(2100, 2200)})


class GatewayAdapter:
    def __init__(
        self,
        cfg: EventbotConfig,
        settings: Settings,
        bus: Bus,
        state: StateStore,
        broker: BrokerClient,
        topics: TopicMap,
        *,
        fill_timeout_s: float,
        clock: Callable[[], dt.datetime] = utc_now,
        root: Path = ROOT,
        on_placed: Callable[[OrderIntent, int], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._acfg = cfg.adapter
        self._settings = settings
        self._bus = bus
        self._state = state
        self._broker = broker
        self._topics = topics
        self._fill_timeout_s = fill_timeout_s
        self._clock = clock
        self._root = root
        self._on_placed = on_placed

        self.account = ""
        self.connected = False
        self.market_open = False
        self.clock_skew_s: float | None = None
        self.positions: dict[str, float] = {}
        self.avg_costs: dict[str, float] = {}
        self.usd_cash: float | None = None
        self.marks: dict[str, float] = {}
        self.known_refs: set[str] = set()
        self.seen_exec_ids: set[str] = set()
        self.open_orders: dict[str, BrokerOrder] = {}

        self._order_started: dict[str, float] = {}
        self._cancel_sent: set[str] = set()
        self._in_flight: set[str] = set()
        self._last_order_state: dict[str, tuple[str, float]] = {}
        self._last_tick_monotonic: float | None = None
        self._pending_ticks: dict[str, BrokerTick] = {}
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._outbox: deque[tuple[str, str | None, Message]] = deque()
        self._pending_marks: dict[str, IntentStatus] = {}
        self._reconnect_needed = asyncio.Event()
        self._stop = asyncio.Event()
        self._broker.set_handlers(
            BrokerHandlers(
                on_execution=lambda e: self._events.put_nowait(e),
                on_order=lambda o: self._events.put_nowait(o),
                on_tick=self._on_tick,
                on_disconnect=self._on_disconnect,
                on_error=self._on_error,
            )
        )

    async def startup(self) -> None:
        """Guard, connect, guard again, recover. Nothing is consumed until this has finished."""
        assert_paper_port(self._settings.ib_port)
        await self._broker.connect()
        await self._verify_account()
        self.connected = True
        await self._recover()

    async def _verify_account(self) -> None:
        try:
            self.account = assert_paper_account(self._broker.managed_accounts(), self._settings.ib_account)
        except Exception:
            await self._broker.disconnect()
            raise
        log_event(LOG, "account_verified", account=mask_account(self.account))

    async def _recover(self) -> None:
        """Rebuild from the broker, never from Redis: publish any executions missed while away, resolve
        intents stuck in "placing", publish a snapshot."""
        await self._refresh_order_knowledge()
        await self._resolve_placing()
        await self.publish_snapshot("startup")
        log_event(LOG, "recovered", known_refs=len(self.known_refs), open_orders=len(self.open_orders))

    async def _refresh_order_knowledge(self) -> list[BrokerExecution]:
        """Ask IB what it knows. Order matters: executions last, so a just-filled order cannot slip
        between the lists."""
        opens = await self._broker.open_orders()
        completed = await self._broker.completed_orders()
        executions = await self._broker.executions()
        self.known_refs |= {o.order_ref for o in (*opens, *completed) if o.order_ref}
        self.known_refs |= {e.order_ref for e in executions if e.order_ref}
        for order in opens:
            if not is_intent_ref(order.order_ref):
                continue
            self.open_orders[order.order_ref] = order
            self._order_started.setdefault(order.order_ref, time.monotonic())
        open_refs = {o.order_ref for o in opens}
        for ref in [r for r in self.open_orders if r not in open_refs and r not in self._in_flight]:
            self.open_orders.pop(ref, None)
            self._order_started.pop(ref, None)
        for execution in executions:
            await self._handle_execution(execution, late=True)
        return executions

    async def _resolve_placing(self) -> None:
        for intent_id in await self._state.intents_with_status(IntentStatus.PLACING):
            if intent_id in self._in_flight:
                continue
            if intent_id in self.known_refs:
                await self._mark(intent_id, IntentStatus.PLACED)
                log_event(
                    LOG, "recovery_adopted", level=logging.WARNING, intent_id=intent_id, order_ref=intent_id
                )
                await self._emit_alert(
                    alert(
                        AlertType.RECOVERY_ADOPTED,
                        "warning",
                        SERVICE,
                        "Intent was left in placing; the order exists at IB, so it is marked placed",
                        intent_id=intent_id,
                        order_ref=intent_id,
                    )
                )
            else:
                await self._mark(intent_id, IntentStatus.BLOCKED)
                log_event(
                    LOG, "recovery_blocked", level=logging.ERROR, intent_id=intent_id, order_ref=intent_id
                )
                await self._emit_alert(
                    alert(
                        AlertType.RECOVERY_BLOCKED,
                        "critical",
                        SERVICE,
                        "Intent was left in placing and no order exists at IB. Not retrying: a missed order "
                        "is safer than a doubled one.",
                        intent_id=intent_id,
                        order_ref=intent_id,
                    )
                )

    async def handle_intent(self, intent: OrderIntent) -> str:
        "Returns the outcome: placed, duplicate, blocked, rejected, already_at_broker or uncertain."
        intent_id = intent.intent_id
        fields: dict[str, Any] = {
            "intent_id": intent_id,
            "order_ref": intent_id,
            "symbol": intent.symbol,
            "side": intent.side.value,
            "qty": intent.qty,
        }

        existing = await self._state.intent_status(intent_id)
        if existing is not None:
            return await self._on_duplicate(intent, existing, fields)
        ctx = await self._context(intent)
        if not await self._state.claim_intent(intent_id):
            status = await self._state.intent_status(intent_id)
            return await self._on_duplicate(intent, status, fields)

        self._in_flight.add(intent_id)
        try:
            return await self._handle_claimed(intent, ctx, fields)
        finally:
            self._in_flight.discard(intent_id)

    async def _handle_claimed(self, intent: OrderIntent, ctx: PreTradeContext, fields: dict[str, Any]) -> str:
        intent_id = intent.intent_id
        rejections = check(intent, ctx, self._cfg.caps, self._acfg)
        if rejections:
            status = IntentStatus.BLOCKED if any(r.blocking for r in rejections) else IntentStatus.REJECTED
            await self._mark(intent_id, status)
            for rejection in rejections:
                log_event(
                    LOG,
                    "pre_trade_reject",
                    level=logging.WARNING,
                    msg=rejection.message,
                    code=rejection.code.value,
                    **fields,
                )
                await self._emit_alert(
                    alert(
                        AlertType.PRE_TRADE_REJECT,
                        "warning",
                        SERVICE,
                        rejection.message,
                        intent_id=intent_id,
                        order_ref=intent_id,
                        code=rejection.code.value,
                    )
                )
            await self._emit_rejected(intent, "; ".join(r.code.value for r in rejections))
            return status.value

        if intent_id in self.known_refs:
            await self._mark(intent_id, IntentStatus.PLACED)
            log_event(LOG, "already_at_broker", level=logging.WARNING, **fields)
            await self._emit_alert(
                alert(
                    AlertType.RECOVERY_ADOPTED,
                    "warning",
                    SERVICE,
                    "IB already has an order with this orderRef: not placing another",
                    intent_id=intent_id,
                    order_ref=intent_id,
                )
            )
            return "already_at_broker"

        holder = await self._state.claim_intent_day(
            intent.strategy, intent.trading_date.isoformat(), intent.symbol, intent_id
        )
        if holder is not None:
            await self._mark(intent_id, IntentStatus.BLOCKED)
            message = f"{intent.symbol} already has intent {holder} for {intent.trading_date}"
            log_event(LOG, "conflicting_intent", level=logging.ERROR, msg=message, holder=holder, **fields)
            await self._emit_alert(
                alert(
                    AlertType.CONFLICTING_INTENT,
                    "critical",
                    SERVICE,
                    message,
                    intent_id=intent_id,
                    order_ref=intent_id,
                    holder=holder,
                )
            )
            await self._emit_rejected(intent, "conflicting_intent")
            return IntentStatus.BLOCKED.value

        try:
            order = await self._broker.place_market_order(
                intent.symbol, intent.side.value, intent.qty, intent_id, self.account
            )
        except Exception as exc:
            log_event(LOG, "order_uncertain", level=logging.ERROR, msg=repr(exc), **fields)
            await self._emit_alert(
                alert(
                    AlertType.ORDER_UNCERTAIN,
                    "critical",
                    SERVICE,
                    f"Placing the order failed and its state is unknown: {exc!r}",
                    intent_id=intent_id,
                    order_ref=intent_id,
                )
            )
            return "uncertain"

        placed_ns = time.time_ns()
        self.known_refs.add(intent_id)
        self.open_orders[intent_id] = order
        self._order_started[intent_id] = time.monotonic()
        await self._mark(intent_id, IntentStatus.PLACED)
        try:
            await self._state.count_order(ctx.today.isoformat())
        except StateUnavailable:
            log_event(LOG, "count_order_failed", level=logging.WARNING, **fields)
        log_event(LOG, "order_placed", status=order.status, perm_id=order.perm_id, **fields)
        if self._on_placed is not None:
            self._on_placed(intent, placed_ns)
        await self._emit_order_event(order)
        return IntentStatus.PLACED.value

    async def _on_duplicate(
        self, intent: OrderIntent, status: IntentStatus | None, fields: dict[str, Any]
    ) -> str:
        log_event(LOG, "duplicate_intent", status=status.value if status else None, **fields)
        await self._emit_alert(
            alert(
                AlertType.DUPLICATE_INTENT,
                "info",
                SERVICE,
                f"Intent delivered again, already {status.value if status else 'unknown'}: ignored",
                intent_id=intent.intent_id,
                order_ref=intent.intent_id,
            )
        )
        return "duplicate"

    async def _context(self, intent: OrderIntent) -> PreTradeContext:
        now = self._clock()
        today = ny_date(now)
        open_symbols = frozenset(o.symbol for o in self.open_orders.values())
        return PreTradeContext(
            now=now,
            today=today,
            connected=self.connected and self._broker.is_connected(),
            kill_switch=await self._state.kill_switch(),
            stop_file=(self._root / self._acfg.stop_file).exists(),
            md_heartbeat_age_s=await self._state.heartbeat_age_s("md"),
            reconciler_heartbeat_age_s=await self._state.heartbeat_age_s("reconciler"),
            market_open=self.market_open,
            clock_skew_s=self.clock_skew_s,
            price=self.marks.get(intent.symbol),
            positions=dict(self.positions),
            marks=dict(self.marks),
            avg_costs=dict(self.avg_costs),
            open_order_symbols=open_symbols,
            orders_today=await self._state.orders_today(today.isoformat()),
        )

    async def _mark(self, intent_id: str, status: IntentStatus) -> None:
        """Set an intent's status. If Redis is down, remember it and retry from the heartbeat loop."""
        try:
            await self._state.set_intent_status(intent_id, status)
            self._pending_marks.pop(intent_id, None)
        except StateUnavailable:
            self._pending_marks[intent_id] = status
            log_event(
                LOG,
                "mark_deferred",
                level=logging.WARNING,
                intent_id=intent_id,
                order_ref=intent_id,
                status=status.value,
            )

    async def process_one_intent(self, consumer: Consumer, timeout_s: float = 0.5) -> str | None:
        """One consume-process-commit cycle. The commit comes only after processing has finished, and a
        retryable failure keeps retrying the same record, so nothing is skipped."""
        delivery = await next_message(consumer, self._bus, self._topics, service=SERVICE, timeout_s=timeout_s)
        if delivery is None:
            return None
        assert isinstance(delivery.message, OrderIntent)
        while True:
            try:
                outcome = await self.handle_intent(delivery.message)
                break
            except StateUnavailable as exc:
                log_event(
                    LOG,
                    "intent_retry",
                    level=logging.WARNING,
                    msg=str(exc),
                    intent_id=delivery.message.intent_id,
                    order_ref=delivery.message.intent_id,
                )
                if self._stop.is_set():
                    return "not_processed"
                await asyncio.sleep(RETRY_DELAY_S)
        try:
            await consumer.commit(delivery.record)
        except BusError as exc:
            log_event(
                LOG,
                "commit_failed",
                level=logging.WARNING,
                msg=str(exc),
                intent_id=delivery.message.intent_id,
                order_ref=delivery.message.intent_id,
            )
        return outcome

    def _on_tick(self, tick: BrokerTick) -> None:
        self.marks[tick.symbol] = tick.price
        self._last_tick_monotonic = time.monotonic()
        self._pending_ticks[tick.symbol] = tick

    def _on_disconnect(self) -> None:
        self.connected = False
        self.market_open = False
        self._reconnect_needed.set()

    def _on_error(self, code: int, message: str, order_ref: str | None) -> None:
        self._events.put_nowait((code, message, order_ref))

    async def drain_broker_events(self) -> None:
        """Process everything queued so far."""
        while not self._events.empty():
            await self._process_event(self._events.get_nowait())

    async def _process_event(self, event: Any) -> None:
        if isinstance(event, BrokerExecution):
            await self._handle_execution(event)
        elif isinstance(event, BrokerOrder):
            await self._handle_order(event)
        else:
            await self._handle_error(*event)

    async def _handle_execution(self, execution: BrokerExecution, late: bool = False) -> None:
        if execution.exec_id in self.seen_exec_ids:
            return
        self.seen_exec_ids.add(execution.exec_id)
        fields: dict[str, Any] = {
            "intent_id": execution.order_ref,
            "order_ref": execution.order_ref,
            "exec_id": execution.exec_id,
            "symbol": execution.symbol,
        }
        if not is_intent_ref(execution.order_ref):
            log_event(LOG, "foreign_execution", level=logging.WARNING, **fields)
            await self._emit_alert(
                alert(
                    AlertType.FOREIGN_EXECUTION,
                    "warning",
                    SERVICE,
                    f"Execution on {execution.symbol} that did not come from an intent (orderRef "
                    f"{execution.order_ref!r}). It is not added to fills, so positions will not reconcile.",
                    order_ref=execution.order_ref,
                    exec_id=execution.exec_id,
                )
            )
            return
        if not late:
            signed = execution.qty if execution.side == Side.BUY.value else -execution.qty
            self.positions[execution.symbol] = self.positions.get(execution.symbol, 0.0) + signed
        log_event(
            LOG, "fill", late=late, side=execution.side, qty=execution.qty, price=execution.price, **fields
        )
        await self._emit(
            self._topics.fills,
            execution.exec_id,
            Fill(
                exec_id=execution.exec_id,
                order_ref=execution.order_ref,
                symbol=execution.symbol,
                side=Side(execution.side),
                qty=execution.qty,
                price=execution.price,
                cum_qty=execution.cum_qty,
                avg_price=execution.avg_price,
                exec_time_utc=execution.time,
                perm_id=execution.perm_id,
                late=late,
            ),
        )

    async def _handle_order(self, order: BrokerOrder) -> None:
        ref = order.order_ref
        if not is_intent_ref(ref):
            return
        state = (order.status, order.filled)
        if self._last_order_state.get(ref) == state:
            return
        self._last_order_state[ref] = state
        fields: dict[str, Any] = {
            "intent_id": ref,
            "order_ref": ref,
            "status": order.status,
            "filled": order.filled,
        }

        if order.status == "ValidationError":
            log_event(
                LOG, "order_warning", level=logging.WARNING, msg=order.reason, code=order.error_code, **fields
            )
            await self._emit_alert(
                alert(
                    AlertType.ORDER_UNCERTAIN,
                    "warning",
                    SERVICE,
                    f"Broker warning on order: {order.reason}",
                    intent_id=ref,
                    order_ref=ref,
                    code=order.error_code,
                )
            )
            await self._emit_order_event(order)
            return

        event = OrderEvent(
            order_ref=ref,
            symbol=order.symbol,
            side=Side(order.side),
            status=order.status,
            qty=order.qty,
            filled=order.filled,
            remaining=order.remaining,
            avg_fill_price=order.avg_fill_price,
            perm_id=order.perm_id,
            error_code=order.error_code,
            reason=order.reason,
        )
        if event.terminal and order.from_error and await self._still_open_at_broker(ref):
            log_event(LOG, "spurious_terminal_ignored", level=logging.WARNING, msg=order.reason, **fields)
            await self._emit_alert(
                alert(
                    AlertType.ORDER_UNCERTAIN,
                    "warning",
                    SERVICE,
                    f"Error on an order that is still live at IB: {order.reason}",
                    intent_id=ref,
                    order_ref=ref,
                    code=order.error_code,
                )
            )
            return
        if event.terminal:
            self.open_orders.pop(ref, None)
            self._order_started.pop(ref, None)
            self._cancel_sent.discard(ref)
        else:
            self.open_orders[ref] = order
        if order.status == "Inactive" or (order.status == "Cancelled" and order.from_error):
            await self._mark(ref, IntentStatus.REJECTED)
            log_event(
                LOG, "order_rejected", level=logging.ERROR, msg=order.reason, code=order.error_code, **fields
            )
            await self._emit_alert(
                alert(
                    AlertType.ORDER_REJECTED,
                    "critical",
                    SERVICE,
                    f"IB rejected the order: {order.reason}",
                    intent_id=ref,
                    order_ref=ref,
                    code=order.error_code,
                )
            )
        log_event(LOG, "order_status", **fields)
        await self._emit(self._topics.orders_events, ref, event)

    async def _still_open_at_broker(self, order_ref: str) -> bool:
        try:
            return any(o.order_ref == order_ref for o in await self._broker.open_orders())
        except BrokerUnavailable:
            return True

    async def _handle_error(self, code: int, message: str, order_ref: str | None) -> None:
        if code in INFORMATIONAL_CODES:
            return
        if code == CONNECTIVITY_LOST:
            log_event(LOG, "ib_connectivity_lost", level=logging.ERROR, msg=message, code=code)
            self.connected = False
            await self._emit_alert(
                alert(
                    AlertType.BROKER_DISCONNECT,
                    "critical",
                    SERVICE,
                    f"IB connectivity lost: {message}",
                    code=code,
                )
            )
            self._reconnect_needed.set()
            return
        if code in CONNECTIVITY_RESTORED:
            self._reconnect_needed.set()
            return
        log_event(LOG, "broker_error", level=logging.WARNING, msg=message, code=code, order_ref=order_ref)
        await self._emit_alert(
            alert(AlertType.BROKER_ERROR, "warning", SERVICE, message, order_ref=order_ref, code=code)
        )

    async def _emit(self, topic: str, key: str | None, message: Message) -> None:
        if self._outbox:
            self._outbox.append((topic, key, message))
            return
        try:
            await publish(self._bus, topic, key, message)
        except BusError as exc:
            log_event(LOG, "publish_queued", level=logging.WARNING, msg=str(exc), topic=topic, key=key)
            self._outbox.append((topic, key, message))

    async def flush_outbox(self) -> None:
        while self._outbox:
            topic, key, message = self._outbox[0]
            try:
                await publish(self._bus, topic, key, message)
            except BusError:
                return
            self._outbox.popleft()

    async def _emit_alert(self, message: Any) -> None:
        await self._emit(self._topics.alerts, message.type.value, message)

    async def _emit_rejected(self, intent: OrderIntent, reason: str) -> None:
        await self._emit(
            self._topics.orders_events,
            intent.intent_id,
            OrderEvent(
                order_ref=intent.intent_id,
                symbol=intent.symbol,
                side=intent.side,
                status="Rejected",
                qty=intent.qty,
                remaining=intent.qty,
                reason=reason,
            ),
        )

    async def _emit_order_event(self, order: BrokerOrder) -> None:
        await self._emit(
            self._topics.orders_events,
            order.order_ref,
            OrderEvent(
                order_ref=order.order_ref,
                symbol=order.symbol,
                side=Side(order.side),
                status=order.status,
                qty=order.qty,
                filled=order.filled,
                remaining=order.remaining,
                avg_fill_price=order.avg_fill_price,
                perm_id=order.perm_id,
                error_code=order.error_code,
                reason=order.reason,
            ),
        )

    async def flush_ticks(self) -> None:
        pending, self._pending_ticks = self._pending_ticks, {}
        now = time.time()
        for tick in pending.values():
            try:
                await self._state.set_last(tick.symbol, tick.price, now)
            except StateUnavailable:
                pass
            try:
                await publish(
                    self._bus,
                    self._topics.md_ticks,
                    tick.symbol,
                    Tick(
                        symbol=tick.symbol,
                        price=tick.price,
                        bid=tick.bid,
                        ask=tick.ask,
                        last=tick.last,
                        close=tick.close,
                        md_type=tick.md_type,
                    ),
                )
            except BusError:
                pass

    async def _refresh_broker_view(self) -> None:
        positions = await self._broker.positions(self.account)
        self.positions = {p.symbol: p.qty for p in positions}
        self.avg_costs = {p.symbol: p.avg_cost for p in positions if p.avg_cost}
        self.usd_cash = await self._broker.usd_cash(self.account)
        self.market_open = await self._broker.market_open()
        server = await self._broker.server_time()
        self.clock_skew_s = (self._clock() - server).total_seconds()

    async def publish_snapshot(self, kind: str = "periodic") -> None:
        """Executions are diffed here too: ib_async fires no event for a fill on an order it has no Trade
        for, so this is what catches them."""
        executions = await self._refresh_order_knowledge()
        await self._refresh_broker_view()
        await self.flush_outbox()
        await self._emit(
            self._topics.positions_snapshot,
            mask_account(self.account),
            PositionSnapshot(
                account=mask_account(self.account),
                kind=kind,  # type: ignore[arg-type]
                positions=dict(self.positions),
                usd_cash=self.usd_cash,
                exec_ids=[e.exec_id for e in executions if is_intent_ref(e.order_ref)],
                foreign_exec_ids=[e.exec_id for e in executions if not is_intent_ref(e.order_ref)],
                open_order_refs=sorted(self.open_orders),
                market_open=self.market_open,
                clock_skew_s=self.clock_skew_s,
            ),
        )

    async def beat(self) -> None:
        """Write heartbeats. hb:md is only refreshed while market data is really arriving."""
        await self._state.beat(SERVICE)
        fresh = (
            self._last_tick_monotonic is not None
            and time.monotonic() - self._last_tick_monotonic < self._cfg.heartbeats.md_max_tick_age_s
        )
        if self.connected and fresh:
            await self._state.beat("md")
        for intent_id, status in list(self._pending_marks.items()):
            await self._mark(intent_id, status)

    async def sync_subscriptions(self) -> None:
        wanted: list[str] = list(self._acfg.base_watchlist) + sorted(self.positions)
        wanted += sorted(await self._state.watchlist())
        chosen = list(dict.fromkeys(wanted))
        limit = self._acfg.max_market_data_lines
        if len(chosen) > limit:
            log_event(LOG, "watchlist_truncated", level=logging.WARNING, wanted=len(chosen), limit=limit)
        await self._broker.set_subscriptions(set(chosen[:limit]))

    async def cancel_timed_out_orders(self) -> None:
        """The same rule as bot.wait_filled: an order not done within fill_timeout_s is cancelled."""
        now = time.monotonic()
        for ref, started in list(self._order_started.items()):
            if now - started > self._fill_timeout_s and ref not in self._cancel_sent:
                self._cancel_sent.add(ref)
                log_event(LOG, "order_timeout_cancel", level=logging.WARNING, intent_id=ref, order_ref=ref)
                try:
                    await self._broker.cancel(ref)
                except BrokerUnavailable as exc:
                    self._cancel_sent.discard(ref)
                    log_event(
                        LOG, "cancel_failed", level=logging.ERROR, msg=str(exc), intent_id=ref, order_ref=ref
                    )

    async def reconnect(self) -> None:
        """Back off, reconnect, verify the account again, re-run recovery. A failed paper check raises."""
        self.connected = False
        await self._emit_alert(
            alert(
                AlertType.BROKER_DISCONNECT,
                "warning",
                SERVICE,
                "Disconnected from IB Gateway. No orders until reconnected.",
            )
        )
        attempt = 0
        delays = self._acfg.reconnect_delays_s
        while not self._stop.is_set():
            try:
                await self._broker.disconnect()
                await self._broker.connect()
                await self._verify_account()
                self.connected = True
                await self._recover()
                log_event(LOG, "reconnected", attempts=attempt + 1)
                return
            except BrokerUnavailable as exc:
                self.connected = False
                delay = delays[min(attempt, len(delays) - 1)]
                log_event(LOG, "reconnect_retry", level=logging.WARNING, msg=str(exc), delay_s=delay)
                attempt += 1
                try:
                    await asyncio.wait_for(self._stop.wait(), delay)
                except TimeoutError:
                    pass

    def request_stop(self) -> None:
        self._stop.set()

    async def _every(self, interval_s: float, work: Callable[[], Any], name: str) -> None:
        while not self._stop.is_set():
            try:
                await work()
            except (BrokerUnavailable, StateUnavailable, BusError) as exc:
                log_event(LOG, f"{name}_skipped", level=logging.WARNING, msg=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), interval_s)
            except TimeoutError:
                pass

    async def _event_loop(self) -> None:
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(self._events.get(), 0.25)
            except TimeoutError:
                await self.flush_outbox()
                continue
            await self._process_event(event)
            await self.flush_outbox()

    async def _intent_loop(self, consumer: Consumer) -> None:
        while not self._stop.is_set():
            try:
                await self.process_one_intent(consumer)
            except BusError as exc:
                log_event(LOG, "consume_failed", level=logging.WARNING, msg=str(exc))
                await asyncio.sleep(RETRY_DELAY_S)

    async def _reconnect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._reconnect_needed.wait(), 0.5)
            except TimeoutError:
                continue
            self._reconnect_needed.clear()
            await self.reconnect()

    async def _when_connected(self, work: Callable[[], Any]) -> None:
        if self.connected:
            await work()

    async def run(self) -> None:
        await self.startup()
        consumer = self._bus.consumer([self._topics.orders_intents], GROUP)
        await consumer.start()
        hb = self._cfg.heartbeats
        tasks = [
            asyncio.create_task(self._event_loop()),
            asyncio.create_task(self._intent_loop(consumer)),
            asyncio.create_task(self._reconnect_loop()),
            asyncio.create_task(self._every(hb.interval_s, self.beat, "heartbeat")),
            asyncio.create_task(self._every(TICK_FLUSH_S, self.flush_ticks, "ticks")),
            asyncio.create_task(
                self._every(
                    self._acfg.snapshot_interval_s,
                    lambda: self._when_connected(self.publish_snapshot),
                    "snapshot",
                )
            ),
            asyncio.create_task(
                self._every(1.0, lambda: self._when_connected(self.sync_subscriptions), "subs")
            ),
            asyncio.create_task(
                self._every(5.0, lambda: self._when_connected(self.cancel_timed_out_orders), "timeouts")
            ),
            asyncio.create_task(
                self._every(
                    self._acfg.snapshot_interval_s,
                    lambda: self._when_connected(self._resolve_placing),
                    "resolve",
                )
            ),
        ]
        stopper = asyncio.create_task(self._stop.wait())
        try:
            done, _ = await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            self._stop.set()
            for task in [*tasks, stopper]:
                task.cancel()
            await asyncio.gather(*tasks, stopper, return_exceptions=True)
            await consumer.stop()
            await self._broker.disconnect()
