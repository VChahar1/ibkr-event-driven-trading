"""The stock strategy's decision logic, as stock_day() applies it, with broker calls replaced by
plain inputs."""

import datetime as dt
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bot import Abort, Order, plan_orders
from eventbot.schemas import OrderIntent, Side
from stock_strategy import Decision, StockConfig
from stockbot import affordable_picks


def managed_positions(
    raw: Mapping[str, float], universe: Iterable[str], orphans: Iterable[str]
) -> dict[str, int]:
    """Lines 148-157: positions in universe stocks plus orphans; everything else is not ours to touch."""
    managed = set(universe) | set(orphans)
    return {s: int(q) for s, q in raw.items() if s in managed and int(q) != 0}


def held_symbols(positions: Mapping[str, int]) -> set[str]:
    """Line 159."""
    return {s for s, q in positions.items() if q > 0}


def wanted_symbols(held: set[str], dec: Decision | None, cfg: StockConfig, candidates: int) -> set[str]:
    """Line 178: prices are only needed for holdings, picks and a few replacement candidates."""
    return set(held) | (set(dec.picks) | set(dec.ranks.index[: cfg.top_n + candidates]) if dec else set())


def failed_price_checks(
    wanted: Iterable[str], ibkr: Mapping[str, float], yahoo: Mapping[str, float], gap_limit: float
) -> set[str]:
    """Lines 190-196: missing IBKR price, or too far from the Yahoo close."""
    bad = set()
    for s in sorted(wanted):
        ip, yp = ibkr.get(s, float("nan")), float(yahoo.get(s, float("nan")))
        gap = abs(ip / yp - 1) if ip == ip and yp == yp else float("nan")
        if not ip == ip or gap > gap_limit:
            bad.add(s)
    return bad


def require_tradable(bad: set[str], held: set[str], dec: Decision | None, gap_limit: float) -> None:
    """Lines 197-200: abort if a stock that must trade failed its price check."""
    must_trade = held | (set(dec.picks) if dec else set())
    if bad & must_trade:
        raise Abort(
            f"Price check failed for {sorted(bad & must_trade)} "
            f"(missing IBKR price or Yahoo/IBKR gap > {gap_limit:.0%})"
        )


@dataclass(frozen=True)
class StockPlan:
    orders: list[Order]
    budget: float
    picks: list[str]


def plan_stock_orders(
    dec: Decision,
    positions: dict[str, int],
    ibkr: Mapping[str, float],
    bad: set[str],
    usd_cash: float,
    risk: dict[str, Any],
    cfg: StockConfig,
) -> StockPlan:
    """Lines 218-224: slot size, affordability, target weights, whole-share orders."""
    held = held_symbols(positions)
    priced = {s: p for s, p in ibkr.items() if s not in bad}
    slot = (
        min(risk["capital_usd"], sum(positions.get(s, 0) * priced.get(s, 0.0) for s in held) + usd_cash)
        * (1 - risk["cash_buffer"])
        / cfg.top_n
    )
    picks = affordable_picks(dec, priced, slot, cfg.top_n) if dec.market_on else []
    tw = pd.Series(0.0, index=sorted(held | set(picks)))
    tw[picks] = 1.0 / cfg.top_n
    orders, budget = plan_orders(tw, positions, priced, usd_cash, risk) if len(tw) else ([], 0.0)
    return StockPlan(orders=list(orders), budget=budget, picks=picks)


def make_intents(
    orders: Iterable[Order],
    positions: Mapping[str, int],
    strategy: str,
    trading_date: dt.date,
    decision_id: str,
) -> list[OrderIntent]:
    """One intent per order, sells first as plan_orders returned them. target_qty is the position after
    the order, so a half-filled decision keeps its ID when re-run."""
    intents = []
    for symbol, side, qty, est_usd in orders:
        position = positions.get(symbol, 0)
        target = position + qty if side == Side.BUY.value else position - qty
        intents.append(
            OrderIntent.build(
                strategy=strategy,
                trading_date=trading_date,
                symbol=symbol,
                side=Side(side),
                target_qty=target,
                qty=qty,
                est_price=est_usd / qty,
                decision_id=decision_id,
            )
        )
    return intents


def price_frame_hash(prices: pd.DataFrame) -> str:
    """Yahoo re-adjusts history over time, so a decision is only reproducible from what it actually saw."""
    return hashlib.sha256(prices.to_csv().encode()).hexdigest()
