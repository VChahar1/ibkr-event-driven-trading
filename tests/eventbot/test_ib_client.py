"""IbAsyncClient against a stand-in IB, using ib_async's real data classes. Checks conversions, the order
fields and the timeouts. tests/integration/test_ib_paper.py exercises it against the real paper Gateway."""

import asyncio
import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest
from eventkit import Event
from ib_async import (
    CommissionReport,
    Contract,
    Execution,
    Fill,
    Order,
    OrderStatus,
    Position,
    Stock,
    Ticker,
    Trade,
)
from ib_async.objects import TradeLogEntry

from eventbot.adapter.broker import BrokerHandlers, BrokerUnavailable
from eventbot.adapter.ib_client import IbAsyncClient, tick_price
from eventbot.config import Settings, load_config
from eventbot.paper_guard import PaperGuardError

CFG = load_config().adapter
ACCOUNT = "DU" + "0000000"
OTHER = "DU" + "0000009"
EXEC_TIME = dt.datetime(2026, 11, 2, 15, 0, tzinfo=dt.UTC)
REF = "a" * 32


class StubIB:
    def __init__(self) -> None:
        self.execDetailsEvent, self.orderStatusEvent = Event(), Event()
        self.pendingTickersEvent, self.disconnectedEvent, self.errorEvent = Event(), Event(), Event()
        self.connected = True
        self.connect_calls: list[tuple] = []
        self.placed: list[tuple[Any, Order]] = []
        self.cancelled: list[Order] = []
        self.mktdata: list[Any] = []
        self.results: dict[str, Any] = {}
        self.in_flight = 0
        self.max_in_flight = 0

    async def connectAsync(self, host: str, port: int, clientId: int, timeout: float, account: str) -> None:
        self.connect_calls.append((host, port, clientId, account))

    def reqMarketDataType(self, kind: int) -> None:
        self.md_type = kind

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return [ACCOUNT]

    async def _answer(self, name: str) -> Any:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return self.results[name]

    def reqPositionsAsync(self) -> Any:
        return self._answer("positions")

    def reqExecutionsAsync(self) -> Any:
        return self._answer("executions")

    def reqAllOpenOrdersAsync(self) -> Any:
        return self._answer("open")

    def reqCompletedOrdersAsync(self, api_only: bool) -> Any:
        return self._answer("completed")

    def reqCurrentTimeAsync(self) -> Any:
        return self._answer("time")

    def reqContractDetailsAsync(self, contract: Any) -> Any:
        return self._answer("details")

    async def qualifyContractsAsync(self, contract: Any) -> list[Any]:
        contract.conId = 1234
        contract.primaryExchange = contract.primaryExchange or "NASDAQ"
        return [contract]

    def placeOrder(self, contract: Any, order: Order) -> Trade:
        self.placed.append((contract, order))
        return Trade(contract, order, OrderStatus(status="PendingSubmit"), [], [])

    def cancelOrder(self, order: Order) -> None:
        self.cancelled.append(order)

    def openTrades(self) -> list[Trade]:
        return self.results.get("open", [])

    def accountValues(self, account: str) -> list[Any]:
        return [SimpleNamespace(tag="$LEDGER-CashBalance", currency="USD", value="1500.5")]

    def reqMktData(self, contract: Any, *args: Any) -> None:
        self.mktdata.append(contract)

    def cancelMktData(self, contract: Any) -> None:
        self.mktdata.remove(contract)


@pytest.fixture
def ib() -> StubIB:
    return StubIB()


@pytest.fixture
def client(ib: StubIB) -> IbAsyncClient:
    return IbAsyncClient(Settings(), CFG, {"BRK.B": "BRK B"}, {"BRK B": "BRK.B"}, {"AMD": "NASDAQ"}, ib=ib)


def make_handlers() -> tuple[BrokerHandlers, dict[str, list]]:
    seen: dict[str, list] = {"exec": [], "order": [], "tick": [], "disc": [], "err": []}
    return BrokerHandlers(
        on_execution=seen["exec"].append,
        on_order=seen["order"].append,
        on_tick=seen["tick"].append,
        on_disconnect=lambda: seen["disc"].append(1),
        on_error=lambda c, m, r: seen["err"].append((c, m)),
    ), seen


def ib_fill(side: str = "BOT", shares: float = 3.0, symbol: str = "AMD", ref: str = REF) -> Fill:
    execution = Execution(
        execId="0000e0d5.1.01.01",
        time=EXEC_TIME,
        side=side,
        shares=shares,
        price=600.5,
        permId=77,
        cumQty=shares,
        avgPrice=600.5,
        orderRef=ref,
    )
    packet_time = EXEC_TIME + dt.timedelta(minutes=5)
    return Fill(Stock(symbol, "SMART", "USD"), execution, CommissionReport(), packet_time)


def ib_trade(
    status: str = "Submitted", log: list | None = None, ref: str = REF, symbol: str = "AMD"
) -> Trade:
    order = Order(orderRef=ref, action="BUY", totalQuantity=4, permId=77)
    return Trade(
        Stock(symbol, "SMART", "USD"),
        order,
        OrderStatus(status=status, filled=1.0, remaining=3.0, avgFillPrice=600.0),
        [],
        log or [],
    )


async def test_connect_refuses_a_live_port_before_opening_a_socket(ib: StubIB) -> None:
    settings = Settings().model_copy(update={"ib_port": 4001})
    client = IbAsyncClient(settings, CFG, {}, {}, {}, ib=ib)
    with pytest.raises(PaperGuardError):
        await client.connect()
    assert ib.connect_calls == []


async def test_connect_uses_the_settings_and_requests_delayed_data(client: IbAsyncClient, ib: StubIB) -> None:
    await client.connect()
    assert ib.connect_calls == [("127.0.0.1", 4002, 71, "")]
    assert ib.md_type == 3


async def test_the_order_is_the_same_as_the_daily_bots(client: IbAsyncClient, ib: StubIB) -> None:
    result = await client.place_market_order("AMD", "BUY", 4, REF, ACCOUNT)
    contract, order = ib.placed[0]
    assert (order.orderType, order.action, order.totalQuantity) == ("MKT", "BUY", 4)
    assert (order.tif, order.account, order.orderRef) == ("DAY", ACCOUNT, REF)
    assert (contract.exchange, contract.primaryExchange) == ("SMART", "NASDAQ")
    assert result.order_ref == REF and result.status == "PendingSubmit"


async def test_symbols_with_a_class_map_both_ways(client: IbAsyncClient, ib: StubIB) -> None:
    await client.place_market_order("BRK.B", "BUY", 1, REF, ACCOUNT)
    assert ib.placed[0][0].symbol == "BRK B"
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    ib.execDetailsEvent.emit(ib_trade(symbol="BRK B"), ib_fill(symbol="BRK B"))
    assert seen["exec"][0].symbol == "BRK.B"


async def test_a_symbol_outside_the_universe_uses_the_same_rule_as_the_daily_bot(
    client: IbAsyncClient,
) -> None:
    assert client._ib_symbol("SPY") == "SPY" and client._ib_symbol("X.Y") == "X Y"


def test_execution_conversion_uses_the_execution_time_not_packet_time(
    client: IbAsyncClient, ib: StubIB
) -> None:
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    ib.execDetailsEvent.emit(ib_trade(), ib_fill())
    e = seen["exec"][0]
    assert (e.exec_id, e.order_ref, e.symbol, e.side) == ("0000e0d5.1.01.01", REF, "AMD", "BUY")
    assert (e.qty, e.price, e.cum_qty, e.avg_price, e.perm_id) == (3.0, 600.5, 3.0, 600.5, 77)
    assert e.time == EXEC_TIME


def test_sold_maps_to_sell(client: IbAsyncClient, ib: StubIB) -> None:
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    ib.execDetailsEvent.emit(ib_trade(), ib_fill(side="SLD"))
    assert seen["exec"][0].side == "SELL"


def test_order_status_conversion(client: IbAsyncClient, ib: StubIB) -> None:
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    ib.orderStatusEvent.emit(ib_trade("PreSubmitted"))
    o = seen["order"][0]
    assert (o.order_ref, o.status, o.filled, o.remaining, o.avg_fill_price) == (
        REF,
        "PreSubmitted",
        1,
        3,
        600,
    )
    assert not o.from_error and o.error_code is None


def test_a_status_set_by_an_error_is_flagged_so_the_adapter_can_confirm_it(
    client: IbAsyncClient, ib: StubIB
) -> None:
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    log = [TradeLogEntry(EXEC_TIME, "Cancelled", "Error 201: rejected", 201)]
    ib.orderStatusEvent.emit(ib_trade("Cancelled", log=log))
    o = seen["order"][0]
    assert o.from_error and o.error_code == 201 and "rejected" in o.reason


def test_a_raising_handler_does_not_escape_into_ib_async(client: IbAsyncClient, ib: StubIB) -> None:
    def boom(_: Any) -> None:
        raise ValueError("handler bug")

    handlers, _ = make_handlers()
    handlers.on_execution = boom
    client.set_handlers(handlers)
    ib.execDetailsEvent.emit(ib_trade(), ib_fill())


def ticker(symbol: str = "X", **fields: float) -> Ticker:
    """ib_async's Ticker resets its price fields on construction, so they are set afterwards."""
    t = Ticker(contract=Stock(symbol))
    for name, value in fields.items():
        setattr(t, name, value)
    return t


@pytest.mark.parametrize(
    "fields,expected",
    [
        (dict(last=100.0, bid=99.9, ask=100.1, bidSize=1.0, askSize=1.0, close=98.0), 100.0),
        (dict(last=105.0, bid=99.9, ask=100.1, bidSize=1.0, askSize=1.0, close=98.0), 100.0),
        (dict(last=100.0, close=98.0), 100.0),
        (dict(close=98.0), 98.0),
        (dict(), None),
        (dict(last=-1.0, close=0.0), None),
    ],
)
def test_tick_price_follows_the_daily_bots_rule(fields: dict, expected: float | None) -> None:
    assert tick_price(ticker(**fields)) == pytest.approx(expected)


def test_tickers_become_ticks_and_priceless_ones_are_skipped(client: IbAsyncClient, ib: StubIB) -> None:
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    good = ticker("AMD", last=100.0, bid=99.9, ask=100.1, bidSize=1.0, askSize=1.0, marketDataType=3)
    ib.pendingTickersEvent.emit([good, ticker("MU")])
    assert [(t.symbol, t.price, t.md_type) for t in seen["tick"]] == [("AMD", 100.0, 3)]


async def test_positions_keep_this_account_and_drop_cash_pairs(client: IbAsyncClient, ib: StubIB) -> None:
    ib.results["positions"] = [
        Position(ACCOUNT, Stock("AMD"), 4.0, 600.0),
        Position(ACCOUNT, Contract(secType="CASH", symbol="EUR", currency="USD"), -30000.0, 1.1),
        Position(OTHER, Stock("MU"), 9.0, 100.0),
        Position(ACCOUNT, Stock("XOM"), 0.0, 100.0),
    ]
    positions = await client.positions(ACCOUNT)
    assert [(p.symbol, p.qty, p.avg_cost) for p in positions] == [("AMD", 4.0, 600.0)]


async def test_cash_reads_the_ledger_tag_through_the_daily_bots_helper(client: IbAsyncClient) -> None:
    assert await client.usd_cash(ACCOUNT) == 1500.5


async def test_open_orders_are_remembered_so_they_can_be_cancelled(client: IbAsyncClient, ib: StubIB) -> None:
    trade = ib_trade()
    ib.results["open"] = [trade]
    assert [o.order_ref for o in await client.open_orders()] == [REF]
    await client.cancel(REF)
    assert ib.cancelled == [trade.order]


async def test_cancelling_an_unknown_order_is_an_error(client: IbAsyncClient, ib: StubIB) -> None:
    ib.results["open"] = []
    with pytest.raises(BrokerUnavailable):
        await client.cancel(REF)


async def test_requests_are_serialized(client: IbAsyncClient, ib: StubIB) -> None:
    """ib_async keys these by a fixed name: a concurrent second request would orphan the first."""
    ib.results.update(open=[], completed=[], positions=[], executions=[])
    await asyncio.gather(
        client.open_orders(),
        client.open_orders(),
        client.completed_orders(),
        client.positions(ACCOUNT),
        client.executions(),
    )
    assert ib.max_in_flight == 1


async def test_a_request_that_never_answers_times_out_instead_of_hanging(ib: StubIB) -> None:
    fast = CFG.model_copy(update={"request_timeout_s": 0.01})
    client = IbAsyncClient(Settings(), fast, {}, {}, {}, ib=ib)
    ib.reqPositionsAsync = lambda: asyncio.get_running_loop().create_future()  # type: ignore[method-assign]
    with pytest.raises(BrokerUnavailable):
        await client.positions(ACCOUNT)


async def test_requests_while_disconnected_raise(client: IbAsyncClient, ib: StubIB) -> None:
    ib.connected = False
    for call in (
        client.open_orders(),
        client.executions(),
        client.positions(ACCOUNT),
        client.usd_cash(ACCOUNT),
    ):
        with pytest.raises(BrokerUnavailable):
            await call


async def test_subscriptions_add_and_remove(client: IbAsyncClient, ib: StubIB) -> None:
    await client.set_subscriptions({"AMD", "SPY"})
    assert sorted(c.symbol for c in ib.mktdata) == ["AMD", "SPY"]
    await client.set_subscriptions({"SPY"})
    assert [c.symbol for c in ib.mktdata] == ["SPY"]


async def test_market_open_uses_ibkrs_calendar_and_caches_it(client: IbAsyncClient, ib: StubIB) -> None:
    ib.results["details"] = [SimpleNamespace(liquidHours="20000101:CLOSED")]
    assert await client.market_open() is False
    ib.results["details"] = []
    assert await client.market_open() is False


def test_a_disconnect_reaches_the_handler(client: IbAsyncClient, ib: StubIB) -> None:
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    ib.disconnectedEvent.emit()
    assert seen["disc"] == [1]


def test_an_error_reaches_the_handler(client: IbAsyncClient, ib: StubIB) -> None:
    handlers, seen = make_handlers()
    client.set_handlers(handlers)
    ib.errorEvent.emit(-1, 1100, "lost", None)
    assert seen["err"] == [(1100, "lost")]
