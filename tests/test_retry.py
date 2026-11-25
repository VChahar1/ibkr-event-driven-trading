"""Connection handling: retry with backoff on startup, abort cleanly on loss mid-run."""

import pytest

from bot import Abort, connect_with_retry, execute_day
from config import load_config
from tests.conftest import make_prices
from tests.fakes import FAKE_ACCOUNT, FakeIB
from tests.test_bot_flow import THU_10_LONDON


class FlakyIB:
    """connect() fails with the given exceptions in order, then succeeds."""

    def __init__(self, *failures: BaseException) -> None:
        self.failures = list(failures)
        self.connects = 0
        self.disconnects = 0

    def connect(self, host: str, port: int, clientId: int, timeout: float) -> None:
        self.connects += 1
        if self.failures:
            raise self.failures.pop(0)

    def disconnect(self) -> None:
        self.disconnects += 1


def test_first_try_does_not_sleep() -> None:
    ib = FlakyIB()
    slept: list[float] = []
    connect_with_retry(ib, "127.0.0.1", 4002, 17, sleep=slept.append)
    assert ib.connects == 1 and slept == []


def test_recovers_after_gateway_restart() -> None:
    ib = FlakyIB(ConnectionRefusedError(111, "refused"), TimeoutError())
    slept: list[float] = []
    connect_with_retry(ib, "127.0.0.1", 4002, 17, sleep=slept.append)
    assert ib.connects == 3 and slept == [15, 30]
    assert ib.disconnects == 2


def test_gives_up_with_abort_after_all_attempts() -> None:
    ib = FlakyIB(*[ConnectionRefusedError(111, "refused")] * 4)
    slept: list[float] = []
    with pytest.raises(Abort, match="unreachable at 127.0.0.1:4002 after 4 attempts"):
        connect_with_retry(ib, "127.0.0.1", 4002, 17, sleep=slept.append)
    assert slept == [15, 30, 60]


def test_non_connection_errors_are_not_retried() -> None:
    ib = FlakyIB(ValueError("bug"))
    with pytest.raises(ValueError):
        connect_with_retry(ib, "127.0.0.1", 4002, 17, sleep=lambda s: None)
    assert ib.connects == 1


def test_lost_connection_before_orders_aborts_without_placing(tmp_path) -> None:
    conf, cfg = load_config(env_file=None)
    conf["ib"]["account"] = FAKE_ACCOUNT
    conf["risk"].update(fill_timeout_s=0, min_trade_usd=0)
    prices_df = make_prices()
    ib = FakeIB(prices_df.iloc[-1].to_dict())
    ib.connected = False
    with pytest.raises(Abort, match="Lost connection"):
        execute_day(ib, conf, cfg, prices_df, "execute", tmp_path, clock=lambda: THU_10_LONDON)
    assert ib.placed == []


def test_connection_drop_mid_run_is_a_clean_abort(tmp_path, monkeypatch) -> None:
    """Today's 09:31 case: the Gateway dropped while state was being rebuilt."""
    import argparse

    import ib_async

    import bot
    from logsetup import RunContext

    class DroppingIB(FakeIB):
        def connect(self, host: str, port: int, clientId: int, timeout: float) -> None:
            pass

        def positions(self, acct: str) -> list:
            raise ConnectionError("Socket disconnect")

    conf, cfg = load_config(env_file=None)
    conf["ib"]["account"] = FAKE_ACCOUNT
    conf["bot"] = {"strategy": "etf"}
    prices_df = make_prices()
    ib = DroppingIB(prices_df.iloc[-1].to_dict())
    monkeypatch.setattr(ib_async, "IB", lambda: ib)
    monkeypatch.setattr(bot, "load_signal_prices", lambda conf, symbols, now: prices_df)
    monkeypatch.setattr(bot, "ROOT", tmp_path)
    args = argparse.Namespace(check=False, execute=False, live=False)
    with pytest.raises(Abort, match="Lost connection to IB Gateway mid-run"):
        bot.run(args, RunContext(mode="dry", git="test"), conf, cfg)
