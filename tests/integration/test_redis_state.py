"""Exercise coordination against a real isolated Redis instance."""

import asyncio

import pytest

from eventbot.state import StateStore

pytestmark = pytest.mark.integration


async def test_intent_claim_is_atomic_across_concurrent_requests(real_redis) -> None:
    state = StateStore(real_redis)
    claims = await asyncio.gather(*(state.claim_intent("integration-fixture") for _ in range(12)))
    assert sum(claims) == 1


async def test_kill_switch_survives_client_reconnection(real_redis) -> None:
    state = StateStore(real_redis)
    await state.set_kill_switch("integration-fixture")
    await real_redis.connection_pool.disconnect()
    assert await StateStore(real_redis).kill_switch() == "integration-fixture"


async def test_heartbeat_expires_on_the_server(real_redis) -> None:
    state = StateStore(real_redis, heartbeat_ttl_s=1)
    await state.beat("integration-fixture")
    assert await state.heartbeat_age_s("integration-fixture") is not None
    await asyncio.sleep(1.2)
    assert await state.heartbeat_age_s("integration-fixture") is None
