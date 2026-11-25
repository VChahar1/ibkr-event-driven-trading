"stockbot.py - the daily run for the S&P 100 momentum strategy (stock_strategy.py)."

import argparse
import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from bot import (
    LOG,
    Abort,
    broker_guards,
    connect_with_retry,
    execute_orders,
    ib_prices,
    plan_orders,
    record_decision,
    resolve_account,
    resolve_mode,
    usd_cash_balance,
    utc_now,
)
from logsetup import RunContext, ev
from stock_strategy import Decision, StockConfig, decide

NEW_YORK = ZoneInfo("America/New_York")
PREFIX = "decided-stocks"
REF_PREFIX = "stockbot"


def ny_session_open(now: dt.datetime) -> bool:
    """Regular NYSE hours on a weekday (holidays are caught later via IBKR's calendar)."""
    t = now.astimezone(NEW_YORK)
    return t.weekday() < 5 and dt.time(9, 30) <= t.time() <= dt.time(15, 55)


def ny_date(now: dt.datetime) -> dt.date:
    return now.astimezone(NEW_YORK).date()


def parse_liquid_hours(text: str, tz: ZoneInfo = NEW_YORK) -> list[tuple[dt.datetime, dt.datetime]]:
    """IBKR liquidHours ("20261005:0930-20261005:1600;20261003:CLOSED") -> sessions."""
    sessions = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        if part.endswith("CLOSED") or "-" not in part:
            continue
        a, b = part.split("-")
        sessions.append(
            (
                dt.datetime.strptime(a, "%Y%m%d:%H%M").replace(tzinfo=tz),
                dt.datetime.strptime(b, "%Y%m%d:%H%M").replace(tzinfo=tz),
            )
        )
    return sessions


def in_liquid_hours(now: dt.datetime, text: str, tz: ZoneInfo = NEW_YORK) -> bool:
    return any(a <= now <= b for a, b in parse_liquid_hours(text, tz))


def month_done(state_dir: Path, today: dt.date) -> bool:
    """A stock decision (trade or no-op) has already been recorded this calendar month."""
    return any(state_dir.glob(f"{PREFIX}-{today:%Y%m}*.json"))


def symbol_maps(uni: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """(our symbol -> IB symbol, IB symbol -> our symbol). Unknown IB symbols map 'BRK B' -> 'BRK.B'."""
    to_ib = dict(zip(uni["symbol"], uni["ib_symbol"], strict=True))
    return to_ib, {v: k for k, v in to_ib.items()}


def bot_owned(journal: Path) -> dict[str, float]:
    """Net shares the stock bot itself bought, per symbol (orphan detection)."""
    owned: dict[str, float] = {}
    if not journal.exists():
        return owned
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = json.loads(line)
        if str(f.get("ref", "")).startswith(REF_PREFIX + "-") and float(f.get("filled") or 0) > 0:
            sign = 1.0 if f["side"] == "BUY" else -1.0
            owned[f["sym"]] = owned.get(f["sym"], 0.0) + sign * float(f["filled"])
    return {s: q for s, q in owned.items() if q > 0}


def affordable_picks(dec: Decision, prices: dict[str, float], slot_usd: float, top_n: int) -> list[str]:
    """Keep the strategy's picks when one share fits in the slot; replace the others with the
    next-ranked affordable, priced stocks (live-only rule: whole shares)."""
    keep = [s for s in dec.picks if prices.get(s, float("nan")) <= slot_usd]
    for s in dec.ranks.index:
        if len(keep) >= min(top_n, len(dec.picks)):
            break
        if s not in keep and s not in dec.picks and prices.get(s, float("nan")) <= slot_usd:
            keep.append(s)
    dropped = [s for s in dec.picks if s not in keep]
    if dropped:
        LOG.warning(
            "Unaffordable at a $%.0f slot, replaced: %s",
            slot_usd,
            dropped,
            extra=ev("unaffordable", dropped=dropped, slot_usd=slot_usd),
        )
    return keep


def load_signal_prices(uni: pd.DataFrame, cfg: StockConfig, conf: dict, now: dt.datetime) -> pd.DataFrame:
    """Two years of completed Yahoo closes (universe + market), columns = our symbols."""
    from data import load_stock_prices

    ny = now.astimezone(NEW_YORK)
    ymap = dict(zip(uni["symbol"], uni["yahoo"], strict=True)) | {cfg.market: cfg.market}
    prices = load_stock_prices(ymap, f"{ny.year - 2}-{ny:%m-%d}")
    if prices.index[-1].date() == ny.date() and ny.time() < dt.time(16, 15):
        prices = prices.iloc[:-1]
    last = prices.index[-1].date()
    age = (ny.date() - last).days
    if age > conf["risk"]["max_data_age_days"]:
        raise Abort(f"Latest close is {age} days old ({last})")
    return prices


def stock_day(
    ib: Any,
    conf: dict,
    cfg: StockConfig,
    uni: pd.DataFrame,
    prices_df: pd.DataFrame,
    mode: str,
    root: Path,
    live_flag: bool = False,
    clock: Callable[[], dt.datetime] = utc_now,
) -> None:
    """One run against a connected IB (real or fake). mode: check | dry | execute."""
    from ib_async import Stock

    risk, scfg = conf["risk"], conf.get("stocks", {})
    today = ny_date(clock())
    state_dir, journal = root / "state", root / "journal.jsonl"
    to_ib, from_ib = symbol_maps(uni)
    primary = dict(zip(uni["symbol"], uni["primary_exchange"], strict=True))

    acct = resolve_account(ib, conf, live_flag)
    orphans = {s for s in bot_owned(journal) if s not in to_ib}
    managed = set(to_ib) | orphans
    positions: dict[str, int] = {}
    for p in ib.positions(acct):
        c = p.contract
        if getattr(c, "secType", "STK") != "STK" or getattr(c, "currency", "USD") != "USD":
            continue
        sym = from_ib.get(c.symbol, c.symbol.replace(" ", "."))
        if sym in managed and int(p.position) != 0:
            positions[sym] = int(p.position)
    usd_cash = usd_cash_balance(ib, acct)
    held = {s for s, q in positions.items() if q > 0}
    LOG.info(
        "Account %s | USD cash %.2f | held %s",
        acct,
        usd_cash,
        sorted(held),
        extra=ev("account", usd_cash=usd_cash, held=sorted(held)),
    )

    due = not month_done(state_dir, today)
    dec = decide(prices_df, cfg, held & set(to_ib), today) if due else None
    if dec is not None:
        top = {s: int(r) for s, r in dec.ranks.head(cfg.top_n).items()}
        LOG.info(
            "Signal date %s | market filter %s | picks %s",
            dec.signal_date.date(),
            "ON" if dec.market_on else "OFF (all cash)",
            dec.picks,
            extra=ev(
                "signal",
                signal_date=dec.signal_date.date(),
                market_on=dec.market_on,
                picks=dec.picks,
                held=sorted(held),
                top_ranks=top,
                target={s: round(1 / cfg.top_n, 3) for s in dec.picks},
            ),
        )
    else:
        LOG.info("Hold: this month's rebalance is already recorded", extra=ev("hold", held=sorted(held)))

    extra = int(scfg.get("candidates", 10))
    wanted = set(held) | (set(dec.picks) | set(dec.ranks.index[: cfg.top_n + extra]) if dec else set())
    contracts: dict[str, Any] = {}
    for s in sorted(wanted):
        ib_sym = to_ib.get(s, s.replace(".", " "))
        q = ib.qualifyContracts(Stock(ib_sym, "SMART", "USD", primaryExchange=primary.get(s, "")))
        if not q or not getattr(q[0], "conId", 0):
            raise Abort(f"Could not qualify {s} ({ib_sym})")
        contracts[s] = q[0]
    iprice = ib_prices(ib, contracts) if contracts else {}
    yprice = prices_df.iloc[-1]
    gap_limit = float(scfg.get("max_price_gap", 0.15))
    bad = set()
    for s in sorted(wanted):
        ip, yp = iprice.get(s, float("nan")), float(yprice.get(s, float("nan")))
        gap = abs(ip / yp - 1) if ip == ip and yp == yp else float("nan")
        LOG.info(
            "  %-6s pos=%-5d yahoo=%.2f ibkr=%.2f",
            s,
            positions.get(s, 0),
            yp,
            ip,
            extra=ev("price_check", sym=s, pos=positions.get(s, 0), yahoo=yp, ibkr=ip, gap=gap),
        )
        if not ip == ip or gap > gap_limit:
            bad.add(s)
    must_trade = held | (set(dec.picks) if dec else set())
    if bad & must_trade:
        raise Abort(
            f"Price check failed for {sorted(bad & must_trade)} "
            f"(missing IBKR price or Yahoo/IBKR gap > {gap_limit:.0%})"
        )
    hours = ib.reqContractDetails(Stock(to_ib[uni["symbol"].iloc[0]], "SMART", "USD"))[0].liquidHours
    open_now = in_liquid_hours(clock(), hours)
    LOG.info(
        "US market %s now (IBKR calendar)",
        "open" if open_now else "CLOSED",
        extra=ev("session", open=open_now),
    )
    if mode == "check":
        LOG.info(
            "CHECK OK - contracts, prices, calendar and account verified. No orders.", extra=ev("check_ok")
        )
        return

    ref = f"{REF_PREFIX}-{today:%Y%m%d}"
    managed_ib = {to_ib.get(s, s.replace(".", " ")) for s in managed}
    broker_guards(ib, ref, lambda c: getattr(c, "symbol", None) in managed_ib, mode)
    if dec is None:
        return

    priced = {s: p for s, p in iprice.items() if s not in bad}
    slot = (
        min(risk["capital_usd"], sum(positions.get(s, 0) * priced.get(s, 0.0) for s in held) + usd_cash)
        * (1 - risk["cash_buffer"])
        / cfg.top_n
    )
    picks = affordable_picks(dec, priced, slot, cfg.top_n) if dec.market_on else []
    tw = pd.Series(0.0, index=sorted(held | set(picks)))
    tw[picks] = 1.0 / cfg.top_n
    orders, budget = plan_orders(tw, positions, priced, usd_cash, risk) if len(tw) else ([], 0.0)
    LOG.info(
        "Budget %.0f USD | plan: %s",
        budget,
        orders or "no orders above min size",
        extra=ev("plan", budget=budget, orders=orders),
    )
    if mode == "dry":
        if orders:
            LOG.info("DRY RUN - rerun with --execute to place these orders.", extra=ev("dry_run"))
        return

    if not open_now:
        LOG.info(
            "Skip: US market closed today per IBKR calendar; retry next trading day",
            extra=ev("skip", reason="market closed (IBKR calendar)"),
        )
        return
    if orders and (root / "STOP").exists():
        raise Abort("Kill switch file STOP appeared - no orders placed")
    record_decision(
        state_dir,
        today,
        {
            "ts": clock().isoformat(),
            "ref": ref,
            "signal_date": dec.signal_date.date(),
            "market_on": dec.market_on,
            "picks": picks,
            "decision": "trade" if orders else "no-op",
            "orders": orders,
        },
        prefix=PREFIX,
    )
    if orders:
        execute_orders(ib, acct, orders, contracts, ref, risk, journal, clock)


def run_stocks(args: argparse.Namespace, ctx: RunContext, conf: dict, root: Path) -> None:
    from backtest_stocks import stock_config
    from universe import load_universe

    uni = load_universe(root / conf.get("stocks", {}).get("universe_file", "universe/sp100.csv"))
    cfg = stock_config(conf, uni)
    requested = ctx.mode
    mode, why = resolve_mode(
        requested,
        utc_now(),
        root / "state",
        session_open=ny_session_open,
        closed_reason="US market closed (09:30-15:55 New York, Mon-Fri)",
        prefix=PREFIX,
        day_of=ny_date,
    )
    ctx.mode = mode
    if why:
        LOG.info(
            "Mode %s -> %s: %s",
            requested,
            mode,
            why,
            extra=ev("skip" if mode == "skip" else "mode", requested=requested, reason=why),
        )
    if mode == "skip":
        return
    prices_df = load_signal_prices(uni, cfg, conf, utc_now())
    from ib_async import IB

    ib = IB()
    connect_with_retry(
        ib,
        conf["ib"]["host"],
        conf["ib"]["port"],
        conf["ib"]["client_id"],
        tuple(conf["ib"].get("connect_retry_delays_s", (15, 30, 60))),
    )
    try:
        stock_day(ib, conf, cfg, uni, prices_df, mode, root, live_flag=args.live)
    except (ConnectionError, TimeoutError) as e:
        raise Abort(
            f"Lost connection to IB Gateway mid-run ({e!r}) - "
            "check TWS for open orders/fills before rerunning"
        ) from e
    finally:
        ib.disconnect()
