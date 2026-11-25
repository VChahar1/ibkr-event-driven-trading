"""The reconciler: compares internal positions, built only from fills, with the broker's, at
startup and every N minutes."""

import asyncio
import datetime as dt
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eventbot.bus import Bus, BusError, Consumer, TopicMap, next_message, publish
from eventbot.config import EventbotConfig
from eventbot.ledger import ApplyResult, FillLedger, diff_positions
from eventbot.logs import log_event
from eventbot.schemas import AlertType, Fill, PositionSnapshot, alert, utc_now
from eventbot.state import StateStore, StateUnavailable

LOG = logging.getLogger("eventbot.reconciler")
SERVICE = "reconciler"
POLL_S = 0.25
RETRY_S = 1.0


class Halted(RuntimeError):
    """A divergence was found. The kill switch is set; a human takes it from here."""


@dataclass(frozen=True)
class Comparison:
    status: str
    diffs: dict[str, dict[str, float]] = field(default_factory=dict)
    missing_exec_ids: list[str] = field(default_factory=list)


def build_baseline(snapshot: PositionSnapshot, known_exec_ids: set[str]) -> PositionSnapshot:
    """A baseline adopts the broker's current positions as the starting point. Every fill the positions
    already include is listed, so the ledger will not count it again."""
    included = sorted(known_exec_ids | set(snapshot.exec_ids) | set(snapshot.foreign_exec_ids))
    return PositionSnapshot(
        account=snapshot.account,
        kind="baseline",
        positions=dict(snapshot.positions),
        usd_cash=snapshot.usd_cash,
        baseline_exec_ids=included,
    )


class Reconciler:
    def __init__(
        self,
        cfg: EventbotConfig,
        bus: Bus,
        state: StateStore,
        topics: TopicMap,
        *,
        clock: Callable[[], dt.datetime] = utc_now,
        on_position: Callable[[Fill, int], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._rcfg = cfg.reconciler
        self._bus = bus
        self._state = state
        self._topics = topics
        self._clock = clock
        self._on_position = on_position
        self.ledger = FillLedger()
        self.snapshot: PositionSnapshot | None = None
        self.verified = False
        self.halted = False
        self._consumer: Consumer | None = None
        self._last_fill_ts: dt.datetime | None = None
        self._stale_alerted = False
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def start(self) -> None:
        """No consumer group: the full fill history is read on every start, by design."""
        self._consumer = self._bus.consumer([self._topics.fills, self._topics.positions_snapshot], None)
        await self._consumer.start()

    async def stop(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()

    async def apply_available(self, timeout_s: float = 0.0) -> None:
        assert self._consumer is not None
        delivery = await next_message(
            self._consumer, self._bus, self._topics, service=SERVICE, timeout_s=timeout_s
        )
        while delivery is not None:
            await self._apply(delivery.message)
            delivery = await next_message(
                self._consumer, self._bus, self._topics, service=SERVICE, timeout_s=0.0
            )

    async def _apply(self, message: Any) -> None:
        if isinstance(message, Fill):
            await self._apply_fill(message)
        elif isinstance(message, PositionSnapshot):
            if message.kind == "baseline":
                self.ledger.rebase(message.positions, message.baseline_exec_ids)
                log_event(LOG, "baseline_adopted", positions=message.positions)
            else:
                self.snapshot = message

    async def _apply_fill(self, fill: Fill) -> None:
        result = self.ledger.apply(fill)
        fields: dict[str, Any] = {
            "intent_id": fill.order_ref,
            "order_ref": fill.order_ref,
            "exec_id": fill.exec_id,
            "symbol": fill.symbol,
            "result": result.value,
        }
        if result in (ApplyResult.DUPLICATE, ApplyResult.SUPERSEDED):
            log_event(LOG, "fill_ignored", **fields)
            return
        self._last_fill_ts = max(self._last_fill_ts or fill.ts_utc, fill.ts_utc)
        if result is ApplyResult.CORRECTION:
            log_event(LOG, "exec_correction", level=logging.WARNING, **fields)
            await publish(
                self._bus,
                self._topics.alerts,
                AlertType.EXEC_CORRECTION.value,
                alert(
                    AlertType.EXEC_CORRECTION,
                    "warning",
                    SERVICE,
                    f"IB corrected an execution on {fill.symbol}",
                    order_ref=fill.order_ref,
                    exec_id=fill.exec_id,
                ),
            )
        position = self.ledger.position(fill.symbol)
        try:
            await self._state.set_pos(fill.symbol, position)
        except StateUnavailable as exc:
            log_event(LOG, "pos_cache_failed", level=logging.WARNING, msg=str(exc), **fields)
        if self._on_position is not None:
            self._on_position(fill, time.time_ns())
        log_event(LOG, "position_updated", position=position, **fields)

    def compare(self) -> Comparison:
        snapshot = self.snapshot
        if snapshot is None:
            return Comparison("stale")
        if (self._clock() - snapshot.ts_utc).total_seconds() > self._rcfg.max_snapshot_age_s:
            return Comparison("stale")
        if self._last_fill_ts is not None:
            since_fill = (snapshot.ts_utc - self._last_fill_ts).total_seconds()
            if since_fill < self._rcfg.settle_s:
                return Comparison("unsettled")

        missing = sorted(set(snapshot.exec_ids) - self.ledger.exec_ids)
        diffs = diff_positions(self.ledger.positions, snapshot.positions)
        if diffs or missing:
            return Comparison("mismatch", diffs, missing)
        return Comparison("ok")

    async def check(self) -> Comparison:
        """Catch up on fills and snapshots, then compare. Mismatch raises Halted, after setting the switch."""
        await self.apply_available()
        result = self.compare()
        if result.status == "ok":
            self.verified = True
            self._stale_alerted = False
            log_event(LOG, "reconcile_ok", positions=len(self.ledger.positions))
        elif result.status == "mismatch":
            await self._halt(result)
        elif result.status == "stale":
            self.verified = False
            log_event(LOG, "reconcile_stale", level=logging.WARNING)
            if not self._stale_alerted:
                self._stale_alerted = True
                await publish(
                    self._bus,
                    self._topics.alerts,
                    AlertType.RECONCILE_STALE.value,
                    alert(
                        AlertType.RECONCILE_STALE,
                        "warning",
                        SERVICE,
                        "No recent broker snapshot, so positions cannot be verified. Orders stay blocked "
                        "until one arrives.",
                    ),
                )
        return result

    async def _halt(self, result: Comparison) -> None:
        self.halted = True
        self.verified = False
        reason: dict[str, Any] = {"diffs": result.diffs, "missing_exec_ids": result.missing_exec_ids}
        log_event(LOG, "reconcile_mismatch", level=logging.CRITICAL, **reason)
        try:
            await publish(
                self._bus,
                self._topics.alerts,
                AlertType.RECONCILE_MISMATCH.value,
                alert(
                    AlertType.RECONCILE_MISMATCH,
                    "critical",
                    SERVICE,
                    "Positions built from fills disagree with the broker. Trading halted, positions left "
                    "as they are. Investigate, then clear the kill switch by hand.",
                    **reason,
                ),
            )
        except BusError as exc:
            log_event(LOG, "halt_alert_failed", level=logging.CRITICAL, msg=str(exc))
        while True:
            try:
                await self._state.set_kill_switch(f"reconcile mismatch: {json.dumps(reason)}")
                break
            except StateUnavailable as exc:
                log_event(LOG, "kill_switch_retry", level=logging.CRITICAL, msg=str(exc))
                await asyncio.sleep(RETRY_S)
        raise Halted(str(reason))

    async def cold_start(self) -> None:
        """Rebuild from history, wait for a broker snapshot, compare. No vouching until it passes."""
        await self.apply_available()
        for symbol, qty in self.ledger.positions.items():
            await self._guarded_set_pos(symbol, qty)
        log_event(LOG, "history_loaded", fills=len(self.ledger.exec_ids), positions=self.ledger.positions)
        while not self._stop.is_set():
            result = await self.check()
            if result.status == "ok":
                await self._state.beat(SERVICE)
                return
            await self._wait(RETRY_S)

    async def _guarded_set_pos(self, symbol: str, qty: float) -> None:
        try:
            await self._state.set_pos(symbol, qty)
        except StateUnavailable as exc:
            log_event(LOG, "pos_cache_failed", level=logging.WARNING, msg=str(exc), symbol=symbol)

    async def _wait(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), seconds)
        except TimeoutError:
            pass

    async def run(self) -> None:
        await self.start()
        try:
            await self.cold_start()
            interval = self._rcfg.interval_minutes * 60
            beat_every = self._cfg.heartbeats.interval_s
            next_check = time.monotonic() + interval
            next_beat = 0.0
            while not self._stop.is_set():
                await self.apply_available(POLL_S)
                now = time.monotonic()
                if now >= next_check:
                    result = await self.check()

                    next_check = now + (RETRY_S if result.status == "unsettled" else interval)
                if self.verified and now >= next_beat:
                    next_beat = now + beat_every
                    try:
                        await self._state.beat(SERVICE)
                    except StateUnavailable as exc:
                        log_event(LOG, "heartbeat_skipped", level=logging.WARNING, msg=str(exc))
        finally:
            await self.stop()


async def amain() -> int:
    """Run the fill-ledger reconciliation service against broker snapshots."""
    from eventbot.runtime import EXIT_HALTED, build_runtime, install_stop_handlers

    rt = build_runtime(SERVICE)
    service = Reconciler(rt.cfg, rt.bus, rt.state, rt.topics)
    install_stop_handlers(service.request_stop)
    await rt.bus.start()
    try:
        await service.run()
        return 0
    except Halted as exc:
        log_event(LOG, "halted", level=logging.CRITICAL, msg=str(exc))
        return EXIT_HALTED
    finally:
        await rt.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
