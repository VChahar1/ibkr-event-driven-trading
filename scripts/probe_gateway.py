#!/usr/bin/env python3
"""Read-only check of the paper Gateway through the same client the adapter uses: paper."""

import asyncio
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eventbot.adapter.broker import BrokerHandlers, BrokerTick
from eventbot.adapter.ib_client import IbAsyncClient
from eventbot.config import ROOT, load_config, load_settings, load_trading_config
from eventbot.paper_guard import PaperGuardError, assert_paper_account, mask_account
from stockbot import symbol_maps
from universe import load_universe

PROBE_CLIENT_ID = 72
WATCH_SECONDS = 15


async def probe() -> int:
    cfg, trading = load_config(), load_trading_config()
    settings = load_settings().model_copy(update={"ib_client_id": PROBE_CLIENT_ID})
    uni = load_universe(ROOT / trading["stocks"]["universe_file"])
    to_ib, from_ib = symbol_maps(uni)
    primary = dict(zip(uni["symbol"], uni["primary_exchange"], strict=True))
    client = IbAsyncClient(settings, cfg.adapter, to_ib, from_ib, primary)

    ticks: list[BrokerTick] = []
    errors: list[tuple[int, str]] = []
    client.set_handlers(
        BrokerHandlers(
            on_execution=lambda e: None,
            on_order=lambda o: None,
            on_tick=ticks.append,
            on_disconnect=lambda: None,
            on_error=lambda c, m, r: errors.append((c, m)),
        )
    )
    await client.connect()
    try:
        account = assert_paper_account(client.managed_accounts(), settings.ib_account)
        print(f"account            {mask_account(account)} (paper check passed)")
        positions = await client.positions(account)
        print(f"positions          {len(positions)}")
        for p in positions:
            print(f"  {p.symbol:<10} {p.qty:>10} avg_cost {p.avg_cost:.2f}")
        print(f"usd cash           {await client.usd_cash(account):.2f}")
        executions = await client.executions()
        print(f"executions         {len(executions)} visible")
        for e in executions[:10]:
            print(f"  {e.time:%Y-%m-%d %H:%M} {e.side} {e.qty} {e.symbol} @ {e.price} ref={e.order_ref!r}")
        print(f"open orders        {len(await client.open_orders())}")
        print(f"completed orders   {len(await client.completed_orders())}")
        print(f"market open        {await client.market_open()}")
        skew = (dt.datetime.now(dt.UTC) - await client.server_time()).total_seconds()
        print(f"clock skew         {skew:+.2f} s (local minus IB server)")

        symbols = {"SPY", "AMD", "MSFT"}
        await client.set_subscriptions(symbols)
        await asyncio.sleep(WATCH_SECONDS)
        await client.set_subscriptions(set())
        counts = Counter(t.symbol for t in ticks)
        print(f"ticks in {WATCH_SECONDS} s      {dict(counts)}")
        print(f"market data types  {sorted({t.md_type for t in ticks})} (1 live, 3 delayed)")
        if ticks:
            t = ticks[-1]
            print(f"last tick          {t.symbol} price={t.price} bid={t.bid} ask={t.ask} last={t.last}")
        print(f"error codes seen   {sorted(Counter(c for c, _ in errors).items())}")
    except PaperGuardError as exc:
        print(f"REFUSED: {exc}")
        return 5
    finally:
        await client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(probe()))
