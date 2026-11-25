"""Read-only tests against the real paper Gateway (make test-ib). No test here places or cancels an order;
a stand-in replaces place_market_order and cancel so a mistake cannot reach IB."""

import asyncio
import socket
from collections.abc import AsyncIterator

import pytest

from eventbot.adapter.broker import BrokerHandlers, BrokerTick
from eventbot.adapter.ib_client import IbAsyncClient
from eventbot.config import ROOT, load_config, load_settings, load_trading_config
from eventbot.paper_guard import assert_paper_account, mask_account
from eventbot.schemas import utc_now
from stockbot import symbol_maps
from universe import load_universe

pytestmark = pytest.mark.ib_paper
TEST_CLIENT_ID = 73


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("read-only test tried to touch an order")


@pytest.fixture
async def client() -> AsyncIterator[IbAsyncClient]:
    settings = load_settings().model_copy(update={"ib_client_id": TEST_CLIENT_ID})
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex((settings.ib_host, settings.ib_port)) != 0:
            pytest.skip("IB Gateway is not listening")
    cfg, trading = load_config(), load_trading_config()
    uni = load_universe(ROOT / trading["stocks"]["universe_file"])
    to_ib, from_ib = symbol_maps(uni)
    primary = dict(zip(uni["symbol"], uni["primary_exchange"], strict=True))
    instance = IbAsyncClient(settings, cfg.adapter, to_ib, from_ib, primary)
    instance.place_market_order = _forbidden  # type: ignore[assignment,method-assign]
    instance.cancel = _forbidden  # type: ignore[assignment,method-assign]
    await instance.connect()
    yield instance
    await instance.disconnect()


def account_of(client: IbAsyncClient) -> str:
    return assert_paper_account(client.managed_accounts(), load_settings().ib_account)


async def test_the_gateway_account_passes_the_paper_guard(client: IbAsyncClient) -> None:
    assert mask_account(account_of(client)).startswith("DU***")


async def test_broker_state_requests_answer(client: IbAsyncClient) -> None:
    account = account_of(client)
    assert isinstance(await client.positions(account), list)
    assert await client.usd_cash(account) >= 0
    assert isinstance(await client.executions(), list)
    assert isinstance(await client.open_orders(), list)
    assert isinstance(await client.completed_orders(), list)
    assert isinstance(await client.market_open(), bool)


async def test_the_clock_is_close_to_ib_server_time(client: IbAsyncClient) -> None:
    skew = (utc_now() - await client.server_time()).total_seconds()
    assert abs(skew) < load_config().adapter.max_clock_skew_s


async def test_market_data_arrives_when_the_market_is_open(client: IbAsyncClient) -> None:
    if not await client.market_open():
        pytest.skip("US market is closed: no ticks are expected")
    ticks: list[BrokerTick] = []
    client.set_handlers(
        BrokerHandlers(
            on_execution=lambda e: None,
            on_order=lambda o: None,
            on_tick=ticks.append,
            on_disconnect=lambda: None,
            on_error=lambda c, m, r: None,
        )
    )
    await client.set_subscriptions({"SPY"})
    for _ in range(60):
        if ticks:
            break
        await asyncio.sleep(0.5)
    await client.set_subscriptions(set())
    assert ticks and ticks[0].symbol == "SPY" and ticks[0].price > 0
