"Pre-trade checks: a pure function from an intent and a context to a list of rejections."

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum

from eventbot.config import AdapterConfig, CapsConfig
from eventbot.schemas import OrderIntent, Side

QTY_TOLERANCE = 1e-6


class RejectCode(StrEnum):
    KILL_SWITCH = "kill_switch"
    STOP_FILE = "stop_file"
    BROKER_DISCONNECTED = "broker_disconnected"
    STALE_MARKET_DATA = "stale_market_data"
    RECONCILER_NOT_RUNNING = "reconciler_not_running"
    MARKET_CLOSED = "market_closed"
    CLOCK_SKEW = "clock_skew"
    STALE_INTENT = "stale_intent"
    NO_PRICE = "no_price"
    ORDER_NOTIONAL = "order_notional"
    GROSS_NOTIONAL = "gross_notional"
    POSITION_NOTIONAL = "position_notional"
    ORDERS_PER_DAY = "orders_per_day"
    OPEN_ORDER_ON_SYMBOL = "open_order_on_symbol"
    WOULD_SHORT = "would_short"
    TARGET_MISMATCH = "target_mismatch"


BLOCKING = frozenset(
    {
        RejectCode.KILL_SWITCH,
        RejectCode.STOP_FILE,
        RejectCode.BROKER_DISCONNECTED,
        RejectCode.STALE_MARKET_DATA,
        RejectCode.RECONCILER_NOT_RUNNING,
        RejectCode.MARKET_CLOSED,
        RejectCode.CLOCK_SKEW,
        RejectCode.STALE_INTENT,
    }
)


@dataclass(frozen=True)
class Rejection:
    code: RejectCode
    message: str

    @property
    def blocking(self) -> bool:
        return self.code in BLOCKING


@dataclass(frozen=True)
class PreTradeContext:
    now: dt.datetime
    today: dt.date
    connected: bool
    kill_switch: str | None
    stop_file: bool
    md_heartbeat_age_s: float | None
    reconciler_heartbeat_age_s: float | None
    market_open: bool
    clock_skew_s: float | None
    price: float | None
    positions: dict[str, float] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    avg_costs: dict[str, float] = field(default_factory=dict)
    open_order_symbols: frozenset[str] = frozenset()
    orders_today: int = 0


def gross_notional(
    positions: dict[str, float], marks: dict[str, float], avg_costs: dict[str, float]
) -> float | None:
    """Sum of |position| x price. None if some position has neither a mark nor an average cost."""
    total = 0.0
    for symbol, qty in positions.items():
        price = marks.get(symbol) or avg_costs.get(symbol)
        if not price:
            return None
        total += abs(qty) * price
    return total


def check(intent: OrderIntent, ctx: PreTradeContext, caps: CapsConfig, cfg: AdapterConfig) -> list[Rejection]:
    out: list[Rejection] = []

    def reject(code: RejectCode, message: str) -> None:
        out.append(Rejection(code, message))

    if ctx.kill_switch is not None:
        reject(RejectCode.KILL_SWITCH, f"Kill switch is on: {ctx.kill_switch}")
    if ctx.stop_file:
        reject(RejectCode.STOP_FILE, f"{cfg.stop_file} file is present")
    if not ctx.connected:
        reject(RejectCode.BROKER_DISCONNECTED, "Not connected to IB Gateway")
    if ctx.md_heartbeat_age_s is None:
        reject(RejectCode.STALE_MARKET_DATA, "Market data heartbeat is stale")
    if ctx.reconciler_heartbeat_age_s is None:
        reject(RejectCode.RECONCILER_NOT_RUNNING, "No reconciler heartbeat: positions are not verified")
    if not ctx.market_open:
        reject(RejectCode.MARKET_CLOSED, "US market is closed per IBKR's calendar")
    if ctx.clock_skew_s is None or abs(ctx.clock_skew_s) > cfg.max_clock_skew_s:
        reject(RejectCode.CLOCK_SKEW, f"Local clock vs IB server skew is {ctx.clock_skew_s} s")
    age_s = (ctx.now - intent.ts_utc).total_seconds()
    if intent.trading_date != ctx.today or age_s > cfg.intent_max_age_s:
        reject(RejectCode.STALE_INTENT, f"Intent for {intent.trading_date} is {age_s:.0f} s old")

    position = ctx.positions.get(intent.symbol, 0.0)
    signed_qty = intent.qty if intent.side is Side.BUY else -intent.qty
    after = position + signed_qty
    if after < -QTY_TOLERANCE:
        reject(RejectCode.WOULD_SHORT, f"Selling {intent.qty} of a {position} position would go short")
    if abs(after - intent.target_qty) > QTY_TOLERANCE:
        reject(
            RejectCode.TARGET_MISMATCH,
            f"Position {position} {intent.side.value} {intent.qty} gives {after}, not {intent.target_qty}",
        )
    if intent.symbol in ctx.open_order_symbols:
        reject(RejectCode.OPEN_ORDER_ON_SYMBOL, f"An order is already open on {intent.symbol}")
    if ctx.orders_today >= caps.max_orders_per_day:
        reject(RejectCode.ORDERS_PER_DAY, f"{ctx.orders_today} orders today, cap {caps.max_orders_per_day}")

    if ctx.price is None or ctx.price <= 0:
        reject(RejectCode.NO_PRICE, f"No current price for {intent.symbol}")
        return out

    notional = intent.qty * ctx.price
    if notional > caps.max_order_notional_usd:
        reject(
            RejectCode.ORDER_NOTIONAL,
            f"Order notional {notional:.2f} exceeds cap {caps.max_order_notional_usd:.2f}",
        )
    position_after = abs(after) * ctx.price
    if abs(after) > abs(position) and position_after > caps.max_position_notional_usd:
        reject(
            RejectCode.POSITION_NOTIONAL,
            f"{intent.symbol} would be {position_after:.2f}, cap {caps.max_position_notional_usd:.2f}",
        )
    marks = {**ctx.marks, intent.symbol: ctx.price}
    gross_before = gross_notional(ctx.positions, marks, ctx.avg_costs)
    if gross_before is None:
        reject(RejectCode.GROSS_NOTIONAL, "A position has no price, so gross notional is unknown")
    else:
        gross_after = gross_before + (abs(after) - abs(position)) * ctx.price
        if gross_after > gross_before and gross_after > caps.max_gross_notional_usd:
            reject(
                RejectCode.GROSS_NOTIONAL,
                f"Gross notional would be {gross_after:.2f}, cap {caps.max_gross_notional_usd:.2f}",
            )
    return out
