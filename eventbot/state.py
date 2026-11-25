"The Redis keyspace, in one place."

import time
from collections.abc import Awaitable
from typing import Any, TypeVar, cast

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from eventbot.schemas import IntentStatus

T = TypeVar("T")
KILL_SWITCH = "kill_switch"
WATCHLIST = "md:watchlist"
INTENT_PREFIX = "intent:"
ORDERS_TODAY_TTL_S = 7 * 24 * 3600


class StateUnavailable(RuntimeError):
    """Redis cannot be reached. Treat as "not safe to trade" and retry without committing."""


async def _guard(call: Awaitable[T]) -> T:
    try:
        return await call
    except (RedisError, OSError) as exc:
        raise StateUnavailable(f"Redis unavailable: {exc}") from exc


class StateStore:
    def __init__(self, client: aioredis.Redis, heartbeat_ttl_s: int = 5) -> None:
        self._r: Any = client
        self._heartbeat_ttl_s = heartbeat_ttl_s

    async def kill_switch(self) -> str | None:
        return cast("str | None", await _guard(self._r.get(KILL_SWITCH)))

    async def set_kill_switch(self, reason: str) -> None:
        await _guard(self._r.set(KILL_SWITCH, reason))

    async def clear_kill_switch(self) -> None:
        await _guard(self._r.delete(KILL_SWITCH))

    async def claim_intent(self, intent_id: str) -> bool:
        """SET NX: True for exactly one caller, however many times the intent is delivered."""
        return bool(
            await _guard(self._r.set(f"{INTENT_PREFIX}{intent_id}", IntentStatus.PLACING.value, nx=True))
        )

    async def intent_status(self, intent_id: str) -> IntentStatus | None:
        value = await _guard(self._r.get(f"{INTENT_PREFIX}{intent_id}"))
        return IntentStatus(value) if value is not None else None

    async def set_intent_status(self, intent_id: str, status: IntentStatus) -> None:
        await _guard(self._r.set(f"{INTENT_PREFIX}{intent_id}", status.value))

    async def intents_with_status(self, status: IntentStatus) -> list[str]:
        async def scan() -> list[str]:
            found = []
            async for key in self._r.scan_iter(match=f"{INTENT_PREFIX}*", count=200):
                if await self._r.get(key) == status.value:
                    found.append(key.removeprefix(INTENT_PREFIX))
            return found

        return await _guard(scan())

    async def status_counts(self) -> dict[str, int]:
        async def scan() -> dict[str, int]:
            counts: dict[str, int] = {}
            async for key in self._r.scan_iter(match=f"{INTENT_PREFIX}*", count=200):
                value = await self._r.get(key)
                counts[value] = counts.get(value, 0) + 1
            return counts

        return await _guard(scan())

    async def claim_intent_day(self, strategy: str, day: str, symbol: str, intent_id: str) -> str | None:
        """Allow one intent per strategy, day and symbol. Returns the other intent's ID on a conflict.
        This stops a re-sized decision (AMD 3 vs AMD 4 on 2026-10-09) hashing to a second order."""
        key = f"intent_day:{strategy}:{day}:{symbol}"
        if await _guard(self._r.set(key, intent_id, nx=True)):
            return None
        holder = await _guard(self._r.get(key))
        return None if holder == intent_id else holder

    async def orders_today(self, day: str) -> int:
        return int(await _guard(self._r.get(f"orders_today:{day}")) or 0)

    async def count_order(self, day: str) -> int:
        key = f"orders_today:{day}"
        count = int(await _guard(self._r.incr(key)))
        await _guard(self._r.expire(key, ORDERS_TODAY_TTL_S))
        return count

    async def beat(self, process: str) -> None:
        await _guard(self._r.set(f"hb:{process}", str(time.time()), ex=self._heartbeat_ttl_s))

    async def heartbeat_age_s(self, process: str) -> float | None:
        """Seconds since the last beat, or None if the key has expired or never existed."""
        value = await _guard(self._r.get(f"hb:{process}"))
        return None if value is None else max(0.0, time.time() - float(value))

    async def set_last(self, symbol: str, price: float, ts: float) -> None:
        await _guard(self._r.hset(f"last:{symbol}", mapping={"price": price, "ts": ts}))

    async def get_last(self, symbol: str) -> tuple[float, float] | None:
        value = await _guard(self._r.hgetall(f"last:{symbol}"))
        return (float(value["price"]), float(value["ts"])) if value else None

    async def set_pos(self, symbol: str, qty: float) -> None:
        await _guard(self._r.set(f"pos:{symbol}", qty))

    async def get_pos(self, symbol: str) -> float | None:
        value = await _guard(self._r.get(f"pos:{symbol}"))
        return None if value is None else float(value)

    async def all_pos(self) -> dict[str, float]:
        async def scan() -> dict[str, float]:
            return {
                key.removeprefix("pos:"): float(await self._r.get(key))
                async for key in self._r.scan_iter(match="pos:*", count=200)
            }

        return await _guard(scan())

    async def all_last(self) -> dict[str, tuple[float, float]]:
        async def scan() -> dict[str, tuple[float, float]]:
            out: dict[str, tuple[float, float]] = {}
            async for key in self._r.scan_iter(match="last:*", count=200):
                value = await self._r.hgetall(key)
                if value:
                    out[key.removeprefix("last:")] = (float(value["price"]), float(value["ts"]))
            return out

        return await _guard(scan())

    async def add_to_watchlist(self, symbols: list[str]) -> None:
        if symbols:
            await _guard(self._r.sadd(WATCHLIST, *symbols))

    async def watchlist(self) -> set[str]:
        return set(await _guard(self._r.smembers(WATCHLIST)))
