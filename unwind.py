#!/usr/bin/env python3
"unwind.py - one-time sale of the old ETF holdings after the switch to stocks."

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bot import (
    LOG,
    Abort,
    Order,
    broker_guards,
    check_port,
    connect_with_retry,
    execute_orders,
    ib_prices,
    london_session_open,
    qualify,
    resolve_account,
    trade_date,
    utc_now,
)
from logsetup import ev

ROOT = Path(__file__).parent
DEFAULT_SYMBOLS = ("CSPX", "EIMI")


def unwind(
    ib: Any, conf: dict, symbols: list[str], mode: str, root: Path, clock: Callable[[], dt.datetime] = utc_now
) -> list[Order]:
    """Plan (and in execute mode, place) market sells of the full position in `symbols`."""
    unknown = [s for s in symbols if s not in conf["universe"]]
    if unknown:
        raise Abort(f"{unknown} not in config.toml [universe] - refusing to guess contracts")
    if (root / "STOP").exists():
        raise Abort("Kill switch file STOP present")
    if mode == "execute" and not london_session_open(clock()):
        raise Abort("LSE closed: run between 08:15 and 16:15 London (09:15-17:15 Warsaw), Mon-Fri")
    acct = resolve_account(ib, conf, False)
    contracts = qualify(ib, conf, symbols)
    con_to_sym = {c.conId: s for s, c in contracts.items()}
    held = {
        con_to_sym[p.contract.conId]: int(p.position)
        for p in ib.positions(acct)
        if p.contract.conId in con_to_sym and int(p.position) > 0
    }
    if not held:
        LOG.info("Nothing to unwind: no position in %s", symbols, extra=ev("no_op"))
        return []
    prices = ib_prices(ib, {s: contracts[s] for s in held})
    orders: list[Order] = [(s, "SELL", q, q * prices[s]) for s, q in sorted(held.items())]
    for s, _, q, est in orders:
        if not est == est or est > conf["risk"]["max_order_usd"]:
            raise Abort(f"SELL {q} {s} ~{est:.0f} USD: price missing or above max_order_usd")
    ref = f"unwind-{trade_date(clock()):%Y%m%d}"
    broker_guards(ib, ref, lambda c: c.conId in con_to_sym, mode)
    LOG.info("Unwind plan: %s", orders, extra=ev("plan", budget=0.0, orders=orders))
    if mode != "execute":
        LOG.info("DRY RUN - rerun with --execute to sell.", extra=ev("dry_run"))
        return orders
    execute_orders(ib, acct, orders, contracts, ref, conf["risk"], root / "journal.jsonl", clock)
    return orders


def main() -> None:
    from alerts import DEFAULT_SUMMARY_MODES, RunDigest, notify_run
    from config import load_config
    from logsetup import RunContext, code_version, setup_logging

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    ctx = RunContext(mode="execute" if args.execute else "dry", git=code_version(ROOT))
    setup_logging(ROOT / "logs", ctx)
    digest = RunDigest()
    logging.getLogger().addHandler(digest)
    LOG.info(
        "Unwind %s start (%s, code %s): %s", ctx.run_id, ctx.mode, ctx.git, args.symbols, extra=ev("start")
    )
    outcome, status = "OK", 0
    try:
        conf, _ = load_config()
        check_port(conf, False)
        from ib_async import IB

        ib = IB()
        connect_with_retry(ib, conf["ib"]["host"], conf["ib"]["port"], conf["ib"]["client_id"] + 2, (5, 10))
        try:
            unwind(ib, conf, args.symbols, ctx.mode, ROOT)
        finally:
            ib.disconnect()
    except Abort as e:
        LOG.error("ABORT: %s", e, extra=ev("abort", reason=str(e)))
        outcome, status = "ABORT", 2
    except Exception:
        LOG.exception("Unexpected error", extra=ev("crash"))
        outcome, status = "CRASH", 3
    notify_run(ctx, outcome, digest, DEFAULT_SUMMARY_MODES)
    sys.exit(status)


if __name__ == "__main__":
    main()
