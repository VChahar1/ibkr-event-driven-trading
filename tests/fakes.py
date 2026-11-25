"FakeIB: the subset of ib_async.IB that bot.execute_day() uses, scripted per test."

from types import SimpleNamespace

FAKE_ACCOUNT = "DU" + "0000001"
CONIDS = {"CSPX": 101, "EIMI": 102, "IDTL": 103, "IGLN": 104, "IB01": 105}


def conid(symbol: str) -> int:
    """Fixed ids for the ETFs, deterministic ones for any stock symbol."""
    return CONIDS.get(symbol) or 1000 + sum(ord(ch) * 31**i for i, ch in enumerate(symbol)) % 900_000


class FakeTrade:
    def __init__(self, contract, order, status: str, price: float) -> None:
        self.contract, self.order = contract, order
        filled = order.totalQuantity if status == "Filled" else 0.0
        self.orderStatus = SimpleNamespace(
            status=status, filled=filled, avgFillPrice=price if filled else 0.0
        )

    def isDone(self) -> bool:
        return self.orderStatus.status in ("Filled", "Cancelled", "Inactive")


class FakeIB:
    def __init__(
        self,
        prices: dict,
        positions: dict | None = None,
        usd_cash: float = 10_000.0,
        accounts: tuple = (FAKE_ACCOUNT,),
        fill_refs: tuple = (),
        open_trades: tuple = (),
        fill_sides: tuple = ("SELL", "BUY"),
        liquid_hours: str = "",
    ) -> None:
        self.prices = dict(prices)
        self.position_map = dict(positions or {})
        self.usd_cash = usd_cash
        self.accounts = list(accounts)
        self.fill_refs = list(fill_refs)
        self.open = list(open_trades)
        self.fill_sides = fill_sides
        self.placed: list[FakeTrade] = []
        self.cancelled: list = []
        self.liquid_hours = liquid_hours

    def managedAccounts(self) -> list:
        return self.accounts

    def qualifyContracts(self, c):
        c.conId = conid(c.symbol)
        c.primaryExchange = c.primaryExchange or c.exchange
        return [c]

    def positions(self, acct: str) -> list:
        return [
            SimpleNamespace(
                contract=SimpleNamespace(conId=conid(s), symbol=s, secType="STK", currency="USD"),
                position=float(q),
            )
            for s, q in self.position_map.items()
            if q
        ]

    def reqContractDetails(self, contract) -> list:
        return [SimpleNamespace(liquidHours=self.liquid_hours, timeZoneId="US/Eastern")]

    def accountValues(self, acct: str) -> list:
        return [SimpleNamespace(tag="$LEDGER-CashBalance", currency="USD", value=str(self.usd_cash))]

    def reqMarketDataType(self, kind: int) -> None:
        pass

    def reqTickers(self, *contracts) -> list:
        return [
            SimpleNamespace(marketPrice=lambda p=self.prices[c.symbol]: p, close=self.prices[c.symbol])
            for c in contracts
        ]

    def reqExecutions(self) -> list:
        return [SimpleNamespace(execution=SimpleNamespace(orderRef=r)) for r in self.fill_refs]

    def openTrades(self) -> list:
        return self.open

    def placeOrder(self, contract, order) -> FakeTrade:
        status = "Filled" if order.action in self.fill_sides else "Submitted"
        trade = FakeTrade(contract, order, status, self.prices[contract.symbol])
        self.placed.append(trade)
        return trade

    def cancelOrder(self, order) -> None:
        self.cancelled.append(order)

    def sleep(self, seconds: float) -> None:
        pass

    def disconnect(self) -> None:
        pass

    connected = True

    def isConnected(self) -> bool:
        return self.connected


def open_trade(symbol: str, ref: str = "manual") -> SimpleNamespace:
    return SimpleNamespace(
        order=SimpleNamespace(orderRef=ref), contract=SimpleNamespace(conId=conid(symbol), symbol=symbol)
    )
