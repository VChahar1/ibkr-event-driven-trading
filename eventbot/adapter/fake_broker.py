"A scripted broker for unit tests, replay and measurements. Never connects to anything."

import asyncio
import datetime as dt
from dataclasses import dataclass, field, replace

from eventbot.adapter.broker import (
    BrokerExecution,
    BrokerHandlers,
    BrokerOrder,
    BrokerPosition,
    BrokerTick,
    BrokerUnavailable,
)

FAKE_ACCOUNT = "DU" + "0000000"
DONE = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})


@dataclass
class FakeOrder:
    order_ref: str
    symbol: str
    side: str
    qty: float
    status: str = "Submitted"
    filled: float = 0.0
    value: float = 0.0
    perm_id: int = 0


@dataclass
class FakeBrokerState:
    accounts: list[str] = field(default_factory=lambda: [FAKE_ACCOUNT])
    positions: dict[str, float] = field(default_factory=dict)
    avg_costs: dict[str, float] = field(default_factory=dict)
    usd_cash: float = 25_000.0
    orders: dict[str, FakeOrder] = field(default_factory=dict)
    executions: list[BrokerExecution] = field(default_factory=list)
    prices: dict[str, float] = field(default_factory=dict)
    market_open: bool = True
    clock_offset_s: float = 0.0
    next_id: int = 1


class FakeBroker:
    def __init__(
        self,
        state: FakeBrokerState | None = None,
        *,
        auto_fill: bool = False,
        fail_before_send: bool = False,
        fail_after_send: bool = False,
        scripted: dict[str, list[BrokerExecution]] | None = None,
    ) -> None:
        self.state = state or FakeBrokerState()
        self.auto_fill = auto_fill
        self.fail_before_send = fail_before_send
        self.fail_after_send = fail_after_send
        self.scripted = scripted or {}
        self.placed: list[str] = []
        self.cancels: list[str] = []
        self.subscriptions: set[str] = set()
        self.connect_failures = 0
        self._connected = False
        self._handlers: BrokerHandlers | None = None

    def set_handlers(self, handlers: BrokerHandlers) -> None:
        self._handlers = handlers

    async def connect(self) -> None:
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise BrokerUnavailable("injected connect failure")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def managed_accounts(self) -> list[str]:
        return list(self.state.accounts)

    def _require_connection(self) -> None:
        if not self._connected:
            raise BrokerUnavailable("not connected")

    async def server_time(self) -> dt.datetime:
        self._require_connection()
        return dt.datetime.now(dt.UTC) + dt.timedelta(seconds=self.state.clock_offset_s)

    async def positions(self, account: str) -> list[BrokerPosition]:
        self._require_connection()
        return [
            BrokerPosition(s, q, self.state.avg_costs.get(s, 0.0))
            for s, q in self.state.positions.items()
            if q != 0
        ]

    async def usd_cash(self, account: str) -> float:
        self._require_connection()
        return self.state.usd_cash

    async def executions(self) -> list[BrokerExecution]:
        self._require_connection()
        return list(self.state.executions)

    def _view(self, o: FakeOrder, **overrides: object) -> BrokerOrder:
        view = BrokerOrder(
            o.order_ref,
            o.symbol,
            o.side,
            o.qty,
            o.status,
            o.filled,
            o.qty - o.filled,
            o.value / o.filled if o.filled else 0.0,
            o.perm_id,
        )
        return replace(view, **overrides)  # type: ignore[arg-type]

    async def open_orders(self) -> list[BrokerOrder]:
        self._require_connection()
        return [self._view(o) for o in self.state.orders.values() if o.status not in DONE]

    async def completed_orders(self) -> list[BrokerOrder]:
        self._require_connection()
        return [self._view(o) for o in self.state.orders.values() if o.status in DONE]

    async def market_open(self) -> bool:
        self._require_connection()
        return self.state.market_open

    async def place_market_order(
        self, symbol: str, side: str, qty: float, order_ref: str, account: str
    ) -> BrokerOrder:
        self._require_connection()
        if account not in self.state.accounts:
            raise AssertionError("order sent to an account this broker does not manage")
        if self.fail_before_send:
            raise BrokerUnavailable("injected failure before the order was sent")
        order = FakeOrder(order_ref, symbol, side, qty, perm_id=self.state.next_id)
        self.state.next_id += 1
        self.state.orders[order_ref] = order
        self.placed.append(order_ref)
        if self.fail_after_send:
            raise BrokerUnavailable("injected failure after the order was sent")
        if self.auto_fill or order_ref in self.scripted:
            asyncio.get_running_loop().call_soon(self._play, order_ref)
        return self._view(order)

    def _play(self, order_ref: str) -> None:
        if order_ref in self.scripted:
            for execution in self.scripted[order_ref]:
                self.report_execution(execution)
        else:
            self.fill(order_ref)

    async def cancel(self, order_ref: str) -> None:
        self._require_connection()
        self.cancels.append(order_ref)
        order = self.state.orders.get(order_ref)
        if order is not None and order.status not in DONE:
            order.status = "Cancelled"
            self._emit_order(order)

    async def set_subscriptions(self, symbols: set[str]) -> None:
        self._require_connection()
        self.subscriptions = set(symbols)

    def _emit_order(self, order: FakeOrder, **overrides: object) -> None:
        if self._handlers is not None and self._connected:
            self._handlers.on_order(self._view(order, **overrides))

    def _record_fill(
        self, order: FakeOrder, qty: float, price: float, exec_id: str | None
    ) -> BrokerExecution:
        order.filled += qty
        order.value += qty * price
        if order.filled >= order.qty:
            order.status = "Filled"
        sign = 1.0 if order.side == "BUY" else -1.0
        self.state.positions[order.symbol] = self.state.positions.get(order.symbol, 0.0) + sign * qty
        self.state.usd_cash -= sign * qty * price
        self.state.next_id += 1
        execution = BrokerExecution(
            exec_id=exec_id or f"0000fake.{self.state.next_id:08x}.01.01",
            order_ref=order.order_ref,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            cum_qty=order.filled,
            avg_price=order.value / order.filled,
            time=dt.datetime.now(dt.UTC),
            perm_id=order.perm_id,
        )
        self.state.executions.append(execution)
        return execution

    def fill(
        self, order_ref: str, qty: float | None = None, price: float | None = None, exec_id: str | None = None
    ) -> BrokerExecution:
        """Fill all or part of an open order, and report it live."""
        order = self.state.orders[order_ref]
        qty = order.qty - order.filled if qty is None else qty
        price = price if price is not None else self.state.prices.get(order.symbol, 100.0)
        execution = self._record_fill(order, qty, price, exec_id)
        if self._handlers is not None and self._connected:
            self._handlers.on_execution(execution)
        self._emit_order(order)
        return execution

    def fill_silently(
        self, order_ref: str, qty: float | None = None, price: float | None = None
    ) -> BrokerExecution:
        """A fill that happens while this client is not listening: no callback fires, as with ib_async
        after a disconnect. Only reqExecutions will reveal it."""
        order = self.state.orders[order_ref]
        qty = order.qty - order.filled if qty is None else qty
        price = price if price is not None else self.state.prices.get(order.symbol, 100.0)
        return self._record_fill(order, qty, price, None)

    def report_execution(self, execution: BrokerExecution) -> None:
        """Report an execution exactly as given. For duplicates and for corrections: it is reported,
        and recorded if new, but does not move this fake's positions."""
        if all(e.exec_id != execution.exec_id for e in self.state.executions):
            self.state.executions.append(execution)
        if self._handlers is not None and self._connected:
            self._handlers.on_execution(execution)

    def reject(
        self,
        order_ref: str,
        code: int = 201,
        reason: str = "Order rejected",
        *,
        from_error: bool = True,
        status: str = "Cancelled",
    ) -> None:
        """A broker rejection. By default the broker really has dropped the order."""
        order = self.state.orders[order_ref]
        order.status = status
        self._emit_order(order, from_error=from_error, error_code=code, reason=reason)

    def spurious_error_cancel(self, order_ref: str, code: int = 321) -> None:
        """ib_async marks an order Cancelled locally on some errors while it stays live at the broker."""
        self._emit_order(
            self.state.orders[order_ref],
            status="Cancelled",
            from_error=True,
            error_code=code,
            reason="validation error",
        )

    def validation_warning(self, order_ref: str, code: int = 399) -> None:
        """ib_async overwrites a status with the synthetic ValidationError on a warning."""
        self._emit_order(
            self.state.orders[order_ref],
            status="ValidationError",
            from_error=True,
            error_code=code,
            reason="warning",
        )

    def tick(self, symbol: str, price: float, md_type: int = 3) -> None:
        self.state.prices[symbol] = price
        if self._handlers is not None and self._connected:
            self._handlers.on_tick(
                BrokerTick(symbol, price, price - 0.01, price + 0.01, price, None, md_type)
            )

    def error(self, code: int, message: str, order_ref: str | None = None) -> None:
        if self._handlers is not None:
            self._handlers.on_error(code, message, order_ref)

    def drop_connection(self) -> None:
        self._connected = False
        if self._handlers is not None:
            self._handlers.on_disconnect()
