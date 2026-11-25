"""BrokerClient over ib_async, reusing the daily bot's pure IB helpers."""

import asyncio
import datetime as dt
import logging
import math
from collections.abc import Awaitable
from typing import Any, TypeVar

from ib_async import IB, MarketOrder, Stock

from bot import smart_routed, usd_cash_balance
from eventbot.adapter.broker import (
    BrokerExecution,
    BrokerHandlers,
    BrokerOrder,
    BrokerPosition,
    BrokerTick,
    BrokerUnavailable,
)
from eventbot.config import AdapterConfig, Settings
from eventbot.logs import log_event
from eventbot.paper_guard import assert_paper_port
from eventbot.schemas import utc_now
from stockbot import in_liquid_hours

LOG = logging.getLogger("eventbot.ib")
T = TypeVar("T")
SIDES = {"BOT": "BUY", "SLD": "SELL", "BUY": "BUY", "SELL": "SELL"}
CALENDAR_CACHE_S = 3600.0


def _number(value: Any) -> float | None:
    """ib_async uses NaN and a very large sentinel for "no value"."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and abs(number) < 1e300 else None


def tick_price(ticker: Any) -> float | None:
    """The same rule as bot.ib_prices: marketPrice(), else close, and only if positive."""
    for candidate in (ticker.marketPrice(), ticker.close):
        price = _number(candidate)
        if price is not None and price > 0:
            return price
    return None


class IbAsyncClient:
    def __init__(
        self,
        settings: Settings,
        cfg: AdapterConfig,
        to_ib: dict[str, str],
        from_ib: dict[str, str],
        primary: dict[str, str],
        ib: Any | None = None,
    ) -> None:
        self._settings = settings
        self._cfg = cfg
        self._to_ib = to_ib
        self._from_ib = from_ib
        self._primary = primary
        self._ib: Any = ib if ib is not None else IB()
        self._handlers: BrokerHandlers | None = None
        self._lock = asyncio.Lock()
        self._contracts: dict[str, Any] = {}
        self._trades: dict[str, Any] = {}
        self._subscribed: dict[str, Any] = {}
        self._hours: tuple[float, str] | None = None
        self._ib.execDetailsEvent += self._on_exec
        self._ib.orderStatusEvent += self._on_status
        self._ib.pendingTickersEvent += self._on_tickers
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.errorEvent += self._on_error

    def _symbol(self, contract: Any) -> str:
        return self._from_ib.get(contract.symbol, contract.symbol.replace(" ", "."))

    def _ib_symbol(self, symbol: str) -> str:
        return self._to_ib.get(symbol, symbol.replace(".", " "))

    async def _contract(self, symbol: str) -> Any:
        if symbol not in self._contracts:
            stock = Stock(
                self._ib_symbol(symbol), "SMART", "USD", primaryExchange=self._primary.get(symbol, "")
            )
            async with self._lock:
                qualified = await self._timed(self._ib.qualifyContractsAsync(stock))
            if not qualified or qualified[0] is None or not getattr(qualified[0], "conId", 0):
                raise BrokerUnavailable(f"Could not qualify {symbol}")
            self._contracts[symbol] = qualified[0]
        return self._contracts[symbol]

    async def _timed(self, request: Awaitable[T]) -> T:
        try:
            return await asyncio.wait_for(request, self._cfg.request_timeout_s)
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise BrokerUnavailable(f"IB request failed: {exc!r}") from exc

    async def _request(self, make_request: Any, *args: Any) -> Any:
        self._require_connection()
        async with self._lock:
            return await self._timed(make_request(*args))

    def _require_connection(self) -> None:
        if not self._ib.isConnected():
            raise BrokerUnavailable("not connected to IB Gateway")

    def set_handlers(self, handlers: BrokerHandlers) -> None:
        self._handlers = handlers

    async def connect(self) -> None:
        assert_paper_port(self._settings.ib_port)
        self._subscribed.clear()
        self._trades.clear()
        try:
            await self._ib.connectAsync(
                self._settings.ib_host,
                self._settings.ib_port,
                clientId=self._settings.ib_client_id,
                timeout=self._cfg.request_timeout_s,
                account=self._settings.ib_account,
            )
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise BrokerUnavailable(f"Could not connect to IB Gateway: {exc!r}") from exc
        self._ib.reqMarketDataType(self._cfg.market_data_type)

    async def disconnect(self) -> None:
        self._ib.disconnect()

    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())

    def managed_accounts(self) -> list[str]:
        return list(self._ib.managedAccounts())

    async def server_time(self) -> dt.datetime:
        value: dt.datetime = await self._request(self._ib.reqCurrentTimeAsync)
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)

    async def positions(self, account: str) -> list[BrokerPosition]:
        rows = await self._request(self._ib.reqPositionsAsync)
        return [
            BrokerPosition(self._symbol(p.contract), float(p.position), float(p.avgCost or 0.0))
            for p in rows
            if p.account == account and p.position and getattr(p.contract, "secType", "STK") != "CASH"
        ]

    async def usd_cash(self, account: str) -> float:
        self._require_connection()
        return float(usd_cash_balance(self._ib, account))

    async def executions(self) -> list[BrokerExecution]:
        fills = await self._request(self._ib.reqExecutionsAsync)
        return [self._execution(f) for f in fills]

    async def open_orders(self) -> list[BrokerOrder]:
        trades = await self._request(self._ib.reqAllOpenOrdersAsync)
        for trade in trades:
            if trade.order.orderRef:
                self._trades.setdefault(trade.order.orderRef, trade)
        return [self._order(t) for t in trades]

    async def completed_orders(self) -> list[BrokerOrder]:
        trades = await self._request(self._ib.reqCompletedOrdersAsync, True)
        return [self._order(t) for t in trades]

    async def market_open(self) -> bool:
        loop = asyncio.get_running_loop()
        if self._hours is None or loop.time() - self._hours[0] > CALENDAR_CACHE_S:
            contract = Stock(self._cfg.base_watchlist[0], "SMART", "USD")
            details = await self._request(self._ib.reqContractDetailsAsync, contract)
            if not details:
                raise BrokerUnavailable("No contract details for the market calendar")
            self._hours = (loop.time(), details[0].liquidHours)
        return bool(in_liquid_hours(utc_now(), self._hours[1]))

    async def place_market_order(
        self, symbol: str, side: str, qty: float, order_ref: str, account: str
    ) -> BrokerOrder:
        contract = await self._contract(symbol)
        self._require_connection()

        order = MarketOrder(side, qty, account=account, tif="DAY", orderRef=order_ref)
        trade = self._ib.placeOrder(smart_routed(contract), order)
        self._trades[order_ref] = trade
        return self._order(trade)

    async def cancel(self, order_ref: str) -> None:
        self._require_connection()
        trade = self._trades.get(order_ref)
        if trade is None:
            for candidate in self._ib.openTrades():
                if candidate.order.orderRef == order_ref:
                    trade = candidate
                    break
        if trade is None:
            raise BrokerUnavailable(f"No open order found for {order_ref}")
        self._ib.cancelOrder(trade.order)

    async def set_subscriptions(self, symbols: set[str]) -> None:
        self._require_connection()
        for symbol in sorted(set(self._subscribed) - symbols):
            self._ib.cancelMktData(self._subscribed.pop(symbol))
        for symbol in sorted(symbols - set(self._subscribed)):
            try:
                contract = await self._contract(symbol)
            except BrokerUnavailable as exc:
                log_event(LOG, "subscribe_failed", level=logging.WARNING, symbol=symbol, error=str(exc))
                continue
            self._ib.reqMktData(contract, "", False, False)
            self._subscribed[symbol] = contract

    def _execution(self, fill: Any) -> BrokerExecution:
        e = fill.execution
        return BrokerExecution(
            exec_id=e.execId,
            order_ref=e.orderRef or "",
            symbol=self._symbol(fill.contract),
            side=SIDES.get(e.side, e.side),
            qty=float(e.shares),
            price=float(e.price),
            cum_qty=float(e.cumQty),
            avg_price=float(e.avgPrice),
            time=e.time,
            perm_id=e.permId or None,
        )

    def _order(self, trade: Any) -> BrokerOrder:
        status = trade.orderStatus
        last_log = trade.log[-1] if trade.log else None
        error_code = getattr(last_log, "errorCode", 0) or None
        return BrokerOrder(
            order_ref=trade.order.orderRef or "",
            symbol=self._symbol(trade.contract),
            side=SIDES.get(trade.order.action, trade.order.action),
            qty=float(trade.order.totalQuantity),
            status=status.status,
            filled=float(status.filled or 0.0),
            remaining=float(status.remaining or 0.0),
            avg_fill_price=float(status.avgFillPrice or 0.0),
            perm_id=trade.order.permId or None,
            from_error=error_code is not None,
            error_code=error_code,
            reason=(getattr(last_log, "message", "") or "") if error_code else "",
        )

    def _on_exec(self, trade: Any, fill: Any) -> None:
        try:
            if self._handlers is not None:
                self._handlers.on_execution(self._execution(fill))
        except Exception:
            LOG.exception("execDetails handler failed")

    def _on_status(self, trade: Any) -> None:
        try:
            if self._handlers is not None:
                self._handlers.on_order(self._order(trade))
        except Exception:
            LOG.exception("orderStatus handler failed")

    def _on_tickers(self, tickers: Any) -> None:
        try:
            for ticker in tickers:
                price = tick_price(ticker)
                if price is None or self._handlers is None:
                    continue
                self._handlers.on_tick(
                    BrokerTick(
                        symbol=self._symbol(ticker.contract),
                        price=price,
                        bid=_number(ticker.bid),
                        ask=_number(ticker.ask),
                        last=_number(ticker.last),
                        close=_number(ticker.close),
                        md_type=int(ticker.marketDataType),
                    )
                )
        except Exception:
            LOG.exception("pendingTickers handler failed")

    def _on_disconnected(self) -> None:
        try:
            if self._handlers is not None:
                self._handlers.on_disconnect()
        except Exception:
            LOG.exception("disconnect handler failed")

    def _on_error(self, req_id: int, code: int, message: str, contract: Any) -> None:
        try:
            if self._handlers is not None:
                self._handlers.on_error(code, message, None)
        except Exception:
            LOG.exception("error handler failed")
