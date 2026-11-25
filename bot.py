#!/usr/bin/env python3
"""bot.py - once-a-day IBKR execution."""

import argparse
import copy
import datetime as dt
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from alerts import DEFAULT_SUMMARY_MODES, RunDigest, notify_run
from config import is_live_port, load_config
from logsetup import RunContext, code_version, ev, setup_logging
from strategy import StrategyConfig, held_set, needs_rebalance, target_weights

ROOT = Path(__file__).parent
LOG = logging.getLogger("dailybot")
LONDON = ZoneInfo("Europe/London")


class Abort(Exception):
    """Any safety check failure: log it and place nothing."""


Order = tuple[str, str, int, float]


def budget_usd(symbols: Iterable[str], positions: dict, prices: dict, usd_cash: float, risk: dict) -> float:
    """USD the bot manages: bot holdings + USD cash, capped at capital_usd, minus the buffer.
    Every weight the bot computes (current or target) is a fraction of this number."""
    held_value = sum(positions.get(s, 0) * prices[s] for s in symbols)
    return min(risk["capital_usd"], held_value + usd_cash) * (1 - risk["cash_buffer"])


def target_shares(target_w: pd.Series, prices: dict, budget: float) -> dict[str, int]:
    return {s: math.floor(budget * w / prices[s]) for s, w in target_w.items()}


def current_weights(symbols: Iterable[str], positions: dict, prices: dict, budget: float) -> pd.Series:
    return pd.Series({s: positions.get(s, 0) * prices[s] / budget for s in symbols}, dtype=float)


def plan_orders(
    target_w: pd.Series, positions: dict, prices: dict, usd_cash: float, risk: dict
) -> tuple[list[Order], float]:
    """
    positions: {symbol: shares held}, prices: {symbol: USD price}
    Returns ([(symbol, side, qty, est_usd), ...] sells first, budget_usd).
    """
    budget = budget_usd(target_w.index, positions, prices, usd_cash, risk)
    if budget <= 0:
        raise Abort(f"No budget: usd_cash={usd_cash:.0f} USD, capital_usd={risk['capital_usd']}")

    sells: list[list[Any]] = []
    buys: list[list[Any]] = []
    for s, tgt_sh in target_shares(target_w, prices, budget).items():
        delta = tgt_sh - positions.get(s, 0)
        value = abs(delta) * prices[s]
        if delta == 0 or value < risk["min_trade_usd"]:
            continue
        (buys if delta > 0 else sells).append([s, "BUY" if delta > 0 else "SELL", abs(delta), value])

    available = usd_cash + sum(o[3] for o in sells)
    need = sum(o[3] for o in buys)
    if need > available and need > 0:
        scale = available / need
        LOG.warning(
            "Scaling buys by %.3f (need %.0f, have %.0f USD)",
            scale,
            need,
            available,
            extra=ev("scale_buys", scale=scale, need=need, available=available),
        )
        for o in buys:
            o[2] = math.floor(o[2] * scale)
            o[3] = o[2] * prices[o[0]]
        buys = [o for o in buys if o[2] > 0 and o[3] >= risk["min_trade_usd"]]

    orders = [tuple(o) for o in sells + buys]
    for s, side, qty, val in orders:
        if val > risk["max_order_usd"]:
            raise Abort(f"{side} {qty} {s} = {val:.0f} USD exceeds max_order_usd")
    if len(orders) > risk["max_orders_per_run"]:
        raise Abort(f"{len(orders)} orders exceeds max_orders_per_run")
    return orders, budget


def rebalance_due(
    target_w: pd.Series, positions: dict, prices: dict, usd_cash: float, risk: dict, band: float
) -> bool:
    "True if holdings differ from target by more than the band (or in which ETFs are held)."
    budget = budget_usd(target_w.index, positions, prices, usd_cash, risk)
    if budget <= 0:
        return True
    cur_w = current_weights(target_w.index, positions, prices, budget)
    shares = target_shares(target_w, prices, budget)
    achievable = pd.Series({s: shares[s] * prices[s] / budget for s in target_w.index}, dtype=float)
    return needs_rebalance(cur_w, achievable, band)


def london_session_open(now: dt.datetime) -> bool:
    t = now.astimezone(LONDON)
    return t.weekday() < 5 and dt.time(8, 15) <= t.time() <= dt.time(16, 15)


def check_port(conf: dict, allow_live_flag: bool) -> None:
    """Runs BEFORE connecting: a live port is refused unless both live switches are on."""
    port = conf["ib"]["port"]
    if is_live_port(port) and not (conf["ib"]["allow_live"] and allow_live_flag):
        raise Abort(f"Port {port} is a LIVE port - refusing to connect (paper only).")


def resolve_account(ib: Any, conf: dict, allow_live_flag: bool) -> str:
    accounts = ib.managedAccounts()
    acct = conf["ib"]["account"] or (accounts[0] if len(accounts) == 1 else "")
    if not acct or acct not in accounts:
        raise Abort(f"Set IB_ACCOUNT in .env; gateway manages {len(accounts)} account(s)")
    is_paper = acct.startswith("DU")
    if not is_paper and not (conf["ib"]["allow_live"] and allow_live_flag):
        raise Abort(f"{acct} is a LIVE account. Needs allow_live=true AND --live.")
    return acct


def qualify(ib: Any, conf: dict, symbols: Iterable[str]) -> dict[str, Any]:
    from ib_async import Stock

    contracts = {}
    for s in symbols:
        ib_sym, exch, ccy, _ = conf["universe"][s]
        c = Stock(ib_sym, exch, ccy)
        q = ib.qualifyContracts(c)
        if not q or not getattr(q[0], "conId", 0):
            raise Abort(f"Could not qualify {s} ({ib_sym}/{exch}/{ccy})")
        contracts[s] = q[0]
    return contracts


def smart_routed(contract: Any) -> Any:
    """Same instrument, routed via SMART with the listing exchange as primary.
    Direct-routed orders trip IBKR's API precaution (Error 10311)."""
    oc = copy.copy(contract)
    oc.primaryExchange = contract.primaryExchange or contract.exchange
    oc.exchange = "SMART"
    return oc


def ib_prices(ib: Any, contracts: dict[str, Any]) -> dict[str, float]:
    ib.reqMarketDataType(3)
    tickers = ib.reqTickers(*contracts.values())
    out = {}
    for s, t in zip(contracts, tickers, strict=True):
        p = t.marketPrice()
        if not (p and p == p and p > 0):
            p = t.close
        out[s] = float(p) if p and p == p and p > 0 else float("nan")
    return out


def usd_cash_balance(ib: Any, acct: str) -> float:

    for v in ib.accountValues(acct):
        if v.tag in ("CashBalance", "$LEDGER-CashBalance") and v.currency == "USD":
            return float(v.value)
    return 0.0


def connect_with_retry(
    ib: Any,
    host: str,
    port: int,
    client_id: int,
    delays_s: Sequence[float] = (15, 30, 60),
    sleep: Callable[[float], object] = time.sleep,
) -> None:
    """Connect, retrying with backoff (covers the Gateway's daily restart window).
    Only connection errors are retried; anything else propagates."""
    attempts = len(delays_s) + 1
    for i in range(attempts):
        try:
            ib.connect(host, port, clientId=client_id, timeout=20)
        except (OSError, TimeoutError) as e:
            ib.disconnect()
            if i == attempts - 1:
                raise Abort(
                    f"IB Gateway unreachable at {host}:{port} after {attempts} attempts "
                    f"({e!r}) - is it running and logged in?"
                ) from e
            LOG.warning(
                "Connect attempt %d/%d failed (%r); retrying in %ds",
                i + 1,
                attempts,
                e,
                delays_s[i],
                extra=ev("connect_retry", attempt=i + 1, error=repr(e)),
            )
            sleep(delays_s[i])
        else:
            if i:
                LOG.info("Connected on attempt %d", i + 1, extra=ev("connected", attempt=i + 1))
            return


def ensure_connected(ib: Any, stage: str) -> None:
    """A dropped connection mid-run aborts; the next run rebuilds state from IBKR."""
    if not ib.isConnected():
        raise Abort(f"Lost connection to IB Gateway {stage} - check fills in TWS before rerunning")


def wait_filled(ib: Any, trades: list[Any], timeout_s: float) -> bool:
    deadline = dt.datetime.now() + dt.timedelta(seconds=timeout_s)
    while dt.datetime.now() < deadline:
        if all(t.isDone() for t in trades):
            break
        ib.sleep(1)
    ok = all(t.orderStatus.status == "Filled" for t in trades)
    for t in trades:
        LOG.info(
            "  %s %s %s -> %s filled=%s avg=%.4f",
            t.order.action,
            t.order.totalQuantity,
            t.contract.symbol,
            t.orderStatus.status,
            t.orderStatus.filled,
            t.orderStatus.avgFillPrice or 0.0,
            extra=ev(
                "fill",
                sym=t.contract.symbol,
                side=t.order.action,
                qty=t.order.totalQuantity,
                status=t.orderStatus.status,
                filled=t.orderStatus.filled,
                avg=t.orderStatus.avgFillPrice,
            ),
        )
        if not t.isDone():
            ib.cancelOrder(t.order)
    return ok


def trade_date(now: dt.datetime) -> dt.date:
    """The trading day a run belongs to (London date; orderRef and decision marker use it)."""
    return now.astimezone(LONDON).date()


def decision_path(state_dir: Path, day: dt.date, prefix: str = "decided") -> Path:
    return state_dir / f"{prefix}-{day:%Y%m%d}.json"


def resolve_mode(
    requested: str,
    now: dt.datetime,
    state_dir: Path,
    session_open: Callable[[dt.datetime], bool] = london_session_open,
    closed_reason: str = "LSE closed (08:15-16:15 London, Mon-Fri)",
    prefix: str = "decided",
    day_of: Callable[[dt.datetime], dt.date] | None = None,
) -> tuple[str, str]:
    "Turn the requested mode into what this run may do: (mode, reason)."
    if requested != "execute":
        return requested, ""
    if not session_open(now):
        return "skip", closed_reason
    marker = decision_path(state_dir, (day_of or trade_date)(now), prefix)
    if marker.exists():
        return "dry", f"{marker.name} exists - today's decision is made; dry run only"
    return "execute", ""


def record_decision(state_dir: Path, day: dt.date, payload: dict, prefix: str = "decided") -> None:
    """Exclusive create: if two runs race, the second aborts instead of trading too."""
    state_dir.mkdir(exist_ok=True)
    path = decision_path(state_dir, day, prefix)
    try:
        with open(path, "x") as f:
            json.dump(payload, f, default=str)
    except FileExistsError as e:
        raise Abort(f"{path.name} appeared during this run - another run decided first") from e


def load_signal_prices(conf: dict, symbols: list[str], now: dt.datetime) -> pd.DataFrame:
    """Completed Yahoo daily closes up to the last finished London session."""
    from data import load_prices

    prices_df = load_prices({s: conf["universe"][s][3] for s in symbols}, period="3y")

    now_ldn = now.astimezone(LONDON)
    if prices_df.index[-1].date() == now_ldn.date() and now_ldn.time() < dt.time(16, 45):
        prices_df = prices_df.iloc[:-1]
    last = prices_df.index[-1].date()
    prev_bday = (pd.Timestamp(now_ldn.date()) - pd.offsets.BDay(1)).date()
    if last < prev_bday:
        LOG.warning(
            "Latest completed close is %s, expected %s - Yahoo may be lagging",
            last,
            prev_bday,
            extra=ev("data_lag", last_close=last, expected=prev_bday),
        )
    age = (now_ldn.date() - last).days
    if age > conf["risk"]["max_data_age_days"]:
        raise Abort(f"Latest close is {age} days old ({last})")
    return prices_df


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def broker_guards(ib: Any, ref: str, is_bot_contract: Callable[[Any], bool], mode: str) -> None:
    """Fills or live orders already carrying today's ref, or any open order on a bot
    instrument: abort in execute mode, warn in dry runs."""
    filled_today = [f for f in ib.reqExecutions() if getattr(f.execution, "orderRef", "") == ref]
    live_today = [t for t in ib.openTrades() if t.order.orderRef == ref]
    problems = []
    if filled_today or live_today:
        problems.append(
            f"Ref {ref} already has {len(filled_today)} fill(s) / "
            f"{len(live_today)} live order(s) - already traded today"
        )
    if any(is_bot_contract(t.contract) for t in ib.openTrades()):
        problems.append("Open orders exist on bot symbols - resolve them manually first")
    if problems and mode == "execute":
        raise Abort("; ".join(problems))
    for p in problems:
        LOG.warning("(dry run, would abort) %s", p, extra=ev("guard", problem=p))


def execute_orders(
    ib: Any,
    acct: str,
    orders: list[Order],
    contracts: dict[str, Any],
    ref: str,
    risk: dict,
    journal: Path,
    clock: Callable[[], dt.datetime],
) -> None:
    """Sells first; buys only after every sell filled. Each batch is journaled."""
    from ib_async import MarketOrder

    for side in ("SELL", "BUY"):
        batch = [o for o in orders if o[1] == side]
        if not batch:
            continue
        ensure_connected(ib, f"before placing {side} orders")
        trades = []
        for s, sd, qty, est in batch:
            order = MarketOrder(sd, qty, account=acct, tif="DAY", orderRef=ref)
            trades.append(ib.placeOrder(smart_routed(contracts[s]), order))
            LOG.info(
                "Placed %s %d %s (~%.0f USD)",
                sd,
                qty,
                s,
                est,
                extra=ev("order", sym=s, side=sd, qty=qty, est_usd=est, ref=ref),
            )
        ok = wait_filled(ib, trades, risk["fill_timeout_s"])
        ensure_connected(ib, f"while waiting for {side} fills")
        with open(journal, "a") as f:
            for t in trades:
                f.write(
                    json.dumps(
                        {
                            "ts": clock().isoformat(),
                            "ref": ref,
                            "sym": t.contract.symbol,
                            "side": t.order.action,
                            "qty": t.order.totalQuantity,
                            "status": t.orderStatus.status,
                            "filled": t.orderStatus.filled,
                            "avg": t.orderStatus.avgFillPrice,
                        }
                    )
                    + "\n"
                )
        if not ok:
            raise Abort(f"{side} orders not fully filled - stopping before next step")
    LOG.info("Done.", extra=ev("done"))


def execute_day(
    ib: Any,
    conf: dict,
    cfg: StrategyConfig,
    prices_df: pd.DataFrame,
    mode: str,
    root: Path,
    live_flag: bool = False,
    clock: Callable[[], dt.datetime] = utc_now,
) -> None:
    """One day's run against a connected IB (real or fake). mode: check | dry | execute."""
    risk = conf["risk"]
    symbols = list(cfg.assets)
    day = trade_date(clock())

    acct = resolve_account(ib, conf, live_flag)
    contracts = qualify(ib, conf, symbols)
    con_to_sym = {c.conId: s for s, c in contracts.items()}
    positions = {s: 0 for s in symbols}
    for p in ib.positions(acct):
        if p.contract.conId in con_to_sym:
            positions[con_to_sym[p.contract.conId]] = int(p.position)
    usd_cash = usd_cash_balance(ib, acct)

    yprice = prices_df.iloc[-1].to_dict()
    iprice = ib_prices(ib, contracts)
    for s in symbols:
        ip, yp = iprice[s], yprice[s]
        gap = abs(ip / yp - 1) if ip == ip else float("inf")
        LOG.info(
            "  %-5s conId=%-10s pos=%-6d yahoo=%.3f ibkr=%.3f",
            s,
            contracts[s].conId,
            positions[s],
            yp,
            ip,
            extra=ev("price_check", sym=s, pos=positions[s], yahoo=yp, ibkr=ip, gap=gap),
        )
        if gap > risk["max_price_gap"]:
            raise Abort(f"{s}: Yahoo {yp:.3f} vs IBKR {ip:.3f} differ > {risk['max_price_gap']:.0%}")
    LOG.info("Account %s | USD cash %.2f", acct, usd_cash, extra=ev("account", usd_cash=usd_cash))

    budget = budget_usd(symbols, positions, iprice, usd_cash, risk)
    held = (
        held_set(current_weights(symbols, positions, iprice, budget), cfg.risky)
        if budget > 0
        else frozenset()
    )
    tw = target_weights(prices_df, cfg, held)
    target = {k: round(v, 3) for k, v in tw.items() if v > 0}
    LOG.info(
        "Signal date %s | held %s | target weights: %s",
        prices_df.index[-1].date(),
        sorted(held),
        target,
        extra=ev("signal", signal_date=prices_df.index[-1].date(), held=sorted(held), target=target),
    )
    if mode == "check":
        LOG.info("CHECK OK - contracts, prices and account verified. No orders.", extra=ev("check_ok"))
        return

    today_ref = f"dailybot-{day:%Y%m%d}"
    broker_guards(ib, today_ref, lambda c: c.conId in con_to_sym, mode)

    orders: list[Order] = []
    if rebalance_due(tw, positions, iprice, usd_cash, risk, cfg.band):
        orders, budget = plan_orders(tw, positions, iprice, usd_cash, risk)
        LOG.info(
            "Budget %.0f USD | plan: %s",
            budget,
            orders or "no orders above min size",
            extra=ev("plan", budget=budget, orders=orders),
        )
    else:
        LOG.info("Within band - nothing to do.", extra=ev("no_op"))
    if mode == "dry":
        if orders:
            LOG.info("DRY RUN - rerun with --execute to place these orders.", extra=ev("dry_run"))
        return

    if orders:
        if (root / "STOP").exists():
            raise Abort("Kill switch file STOP appeared - no orders placed")
        if not london_session_open(clock()):
            raise Abort("LSE not in session (08:15-16:15 London, Mon-Fri)")
    record_decision(
        root / "state",
        day,
        {
            "ts": clock().isoformat(),
            "ref": today_ref,
            "signal_date": prices_df.index[-1].date(),
            "decision": "trade" if orders else "no-op",
            "orders": orders,
        },
    )
    if not orders:
        return
    execute_orders(ib, acct, orders, contracts, today_ref, risk, root / "journal.jsonl", clock)


def run(args: argparse.Namespace, ctx: RunContext, conf: dict, cfg: StrategyConfig) -> None:
    check_port(conf, args.live)
    if (ROOT / "STOP").exists():
        raise Abort("Kill switch file STOP present")
    strategy = conf.get("bot", {}).get("strategy", "etf")
    LOG.info("Strategy: %s", strategy, extra=ev("strategy", name=strategy))
    if strategy == "stocks":
        from stockbot import run_stocks

        run_stocks(args, ctx, conf, ROOT)
        return
    requested = ctx.mode
    mode, why = resolve_mode(requested, dt.datetime.now(dt.UTC), ROOT / "state")
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

    prices_df = load_signal_prices(conf, list(cfg.assets), dt.datetime.now(dt.UTC))
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
        execute_day(ib, conf, cfg, prices_df, mode, ROOT, live_flag=args.live)
    except (ConnectionError, TimeoutError) as e:
        raise Abort(
            f"Lost connection to IB Gateway mid-run ({e!r}) - "
            "check TWS for open orders/fills before rerunning"
        ) from e
    finally:
        ib.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify setup, no orders")
    ap.add_argument("--execute", action="store_true", help="actually place orders")
    ap.add_argument("--live", action="store_true", help="permit a live (non-DU) account")
    args = ap.parse_args()
    ctx = RunContext(
        mode="check" if args.check else "execute" if args.execute else "dry", git=code_version(ROOT)
    )
    setup_logging(ROOT / "logs", ctx)
    digest = RunDigest()
    logging.getLogger().addHandler(digest)
    LOG.info("Run %s start (%s, code %s)", ctx.run_id, ctx.mode, ctx.git, extra=ev("start"))
    summary_modes = DEFAULT_SUMMARY_MODES
    try:
        conf, cfg = load_config()
        summary_modes = tuple(conf.get("alerts", {}).get("summary_modes", summary_modes))
        run(args, ctx, conf, cfg)
        outcome, status = "OK", 0
    except Abort as e:
        LOG.error("ABORT: %s", e, extra=ev("abort", reason=str(e)))
        outcome, status = "ABORT", 2
    except Exception:
        LOG.exception("Unexpected error", extra=ev("crash"))
        outcome, status = "CRASH", 3
    notify_run(ctx, outcome, digest, summary_modes)
    sys.exit(status)


if __name__ == "__main__":
    main()
