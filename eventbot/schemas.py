"""Message schemas. Every message is a pydantic model serialised as JSON, carrying schema_version,
ts_utc and a unique msg_id. Unknown fields are rejected, so a producer/consumer mismatch shows up as
an invalid-message alert rather than as silently dropped data.
"""

import datetime as dt
import hashlib
import json
import math
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, model_validator

from eventbot.paper_guard import ACCOUNT_RE

SCHEMA_VERSION: Literal[1] = 1
INTENT_ID_LEN = 32
TERMINAL_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"})


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _must_be_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(dt.UTC)


def _must_be_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("must be finite")
    return value


UtcDatetime = Annotated[dt.datetime, AfterValidator(_must_be_utc)]
Finite = Annotated[float, AfterValidator(_must_be_finite)]
PositiveFinite = Annotated[float, Field(gt=0), AfterValidator(_must_be_finite)]
NonNegFinite = Annotated[float, Field(ge=0), AfterValidator(_must_be_finite)]
Symbol = Annotated[str, Field(min_length=1, max_length=16)]


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class IntentStatus(StrEnum):
    PLACING = "placing"
    PLACED = "placed"
    REJECTED = "rejected"
    BLOCKED = "blocked"


def make_intent_id(
    strategy: str, trading_date: dt.date, symbol: str, side: Side | str, target_qty: float
) -> str:
    "Deterministic: the same decision always hashes to the same ID."
    qty: int | float = int(target_qty) if float(target_qty).is_integer() else float(target_qty)
    payload = json.dumps([strategy, trading_date.isoformat(), symbol, str(side), qty], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:INTENT_ID_LEN]


def is_intent_ref(order_ref: str) -> bool:
    """Whether an IB orderRef has the shape of one of our intent IDs."""
    return len(order_ref) == INTENT_ID_LEN and all(c in "0123456789abcdef" for c in order_ref)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    msg_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    ts_utc: UtcDatetime = Field(default_factory=utc_now)


class Tick(Message):
    symbol: Symbol
    price: PositiveFinite
    bid: Finite | None = None
    ask: Finite | None = None
    last: Finite | None = None
    close: Finite | None = None
    md_type: int = Field(ge=1, le=4)


class OrderIntent(Message):
    intent_id: str
    strategy: str = Field(min_length=1)
    trading_date: dt.date
    symbol: Symbol
    side: Side
    target_qty: NonNegFinite
    qty: PositiveFinite
    est_price: PositiveFinite
    decision_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def id_matches_content(self) -> "OrderIntent":
        expected = make_intent_id(self.strategy, self.trading_date, self.symbol, self.side, self.target_qty)
        if self.intent_id != expected:
            raise ValueError("intent_id does not match the hash of its fields")
        return self

    @classmethod
    def build(
        cls,
        *,
        strategy: str,
        trading_date: dt.date,
        symbol: str,
        side: Side,
        target_qty: float,
        qty: float,
        est_price: float,
        decision_id: str,
    ) -> "OrderIntent":
        return cls(
            intent_id=make_intent_id(strategy, trading_date, symbol, side, target_qty),
            strategy=strategy,
            trading_date=trading_date,
            symbol=symbol,
            side=side,
            target_qty=target_qty,
            qty=qty,
            est_price=est_price,
            decision_id=decision_id,
        )


class OrderEvent(Message):
    order_ref: str = Field(min_length=1)
    symbol: Symbol
    side: Side
    status: str
    qty: NonNegFinite
    filled: NonNegFinite = 0.0
    remaining: NonNegFinite = 0.0
    avg_fill_price: NonNegFinite = 0.0
    perm_id: int | None = None
    error_code: int | None = None
    reason: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class Fill(Message):
    exec_id: str = Field(min_length=1)
    order_ref: str = Field(min_length=1)
    symbol: Symbol
    side: Side
    qty: PositiveFinite
    price: PositiveFinite
    cum_qty: NonNegFinite = 0.0
    avg_price: NonNegFinite = 0.0
    exec_time_utc: UtcDatetime
    perm_id: int | None = None
    late: bool = False


def _masked_account(value: str) -> str:
    if ACCOUNT_RE.search(value):
        raise ValueError("account must be masked before it goes on the bus")
    return value


class PositionSnapshot(Message):
    account: Annotated[str, AfterValidator(_masked_account)]
    kind: Literal["startup", "periodic", "baseline"]
    positions: dict[Symbol, Finite]
    usd_cash: Finite | None = None
    exec_ids: list[str] = Field(default_factory=list)
    foreign_exec_ids: list[str] = Field(default_factory=list)
    open_order_refs: list[str] = Field(default_factory=list)
    market_open: bool = False
    clock_skew_s: Finite | None = None

    baseline_exec_ids: list[str] = Field(default_factory=list)


class AlertType(StrEnum):
    INVALID_MESSAGE = "invalid_message"
    PRE_TRADE_REJECT = "pre_trade_reject"
    DUPLICATE_INTENT = "duplicate_intent"
    CONFLICTING_INTENT = "conflicting_intent"
    RECOVERY_BLOCKED = "recovery_blocked"
    RECOVERY_ADOPTED = "recovery_adopted"
    FOREIGN_EXECUTION = "foreign_execution"
    EXEC_CORRECTION = "exec_correction"
    ORDER_REJECTED = "order_rejected"
    ORDER_UNCERTAIN = "order_uncertain"
    BROKER_DISCONNECT = "broker_disconnect"
    BROKER_ERROR = "broker_error"
    RECONCILE_MISMATCH = "reconcile_mismatch"
    RECONCILE_STALE = "reconcile_stale"
    STRATEGY_ABORT = "strategy_abort"
    KILL_SWITCH = "kill_switch"
    SERVICE_ERROR = "service_error"
    SMOKE = "smoke"


class Alert(Message):
    type: AlertType
    severity: Literal["info", "warning", "critical"]
    service: str
    message: str
    intent_id: str | None = None
    order_ref: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


def alert(
    type_: AlertType,
    severity: Literal["info", "warning", "critical"],
    service: str,
    message: str,
    intent_id: str | None = None,
    order_ref: str | None = None,
    **details: Any,
) -> Alert:
    return Alert(
        type=type_,
        severity=severity,
        service=service,
        message=message,
        intent_id=intent_id,
        order_ref=order_ref,
        details=json.loads(json.dumps(details, default=str)),
    )
