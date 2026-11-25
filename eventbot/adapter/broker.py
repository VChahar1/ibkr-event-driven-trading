"""What the adapter needs from a broker, in plain types."""

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class BrokerUnavailable(RuntimeError):
    """A request failed because the broker connection is down or did not answer in time."""


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    avg_cost: float = 0.0


@dataclass(frozen=True)
class BrokerExecution:
    exec_id: str
    order_ref: str
    symbol: str
    side: str
    qty: float
    price: float
    cum_qty: float
    avg_price: float
    time: dt.datetime
    perm_id: int | None = None


@dataclass(frozen=True)
class BrokerOrder:
    order_ref: str
    symbol: str
    side: str
    qty: float
    status: str
    filled: float = 0.0
    remaining: float = 0.0
    avg_fill_price: float = 0.0
    perm_id: int | None = None
    from_error: bool = False
    error_code: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class BrokerTick:
    symbol: str
    price: float
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None
    md_type: int


@dataclass
class BrokerHandlers:
    on_execution: Callable[[BrokerExecution], None]
    on_order: Callable[[BrokerOrder], None]
    on_tick: Callable[[BrokerTick], None]
    on_disconnect: Callable[[], None]
    on_error: Callable[[int, str, str | None], None]


class BrokerClient(Protocol):
    def set_handlers(self, handlers: BrokerHandlers) -> None: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def managed_accounts(self) -> list[str]: ...
    async def server_time(self) -> dt.datetime: ...
    async def positions(self, account: str) -> list[BrokerPosition]: ...
    async def usd_cash(self, account: str) -> float: ...
    async def executions(self) -> list[BrokerExecution]: ...
    async def open_orders(self) -> list[BrokerOrder]: ...
    async def completed_orders(self) -> list[BrokerOrder]: ...
    async def market_open(self) -> bool: ...

    async def place_market_order(
        self, symbol: str, side: str, qty: float, order_ref: str, account: str
    ) -> BrokerOrder: ...

    async def cancel(self, order_ref: str) -> None: ...
    async def set_subscriptions(self, symbols: set[str]) -> None: ...
