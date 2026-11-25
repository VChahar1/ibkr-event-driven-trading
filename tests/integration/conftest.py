"Fixtures for tests against real Kafka and Redis (`make up`, then `make test-integration`)."

import uuid
from collections.abc import AsyncIterator

import pytest
import redis.asyncio as aioredis
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaError

from eventbot.config import load_settings

REDIS_TEST_DB = 15


@pytest.fixture
def bootstrap() -> str:
    return load_settings().kafka_bootstrap


@pytest.fixture
async def make_topic(bootstrap: str) -> AsyncIterator:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    try:
        await admin.start()
    except KafkaError as exc:
        pytest.skip(f"Kafka not reachable at {bootstrap} ({exc}): run `make up`")
    created: list[str] = []

    async def _make(partitions: int = 1) -> str:
        name = f"it.{uuid.uuid4().hex[:12]}"
        await admin.create_topics([NewTopic(name=name, num_partitions=partitions, replication_factor=1)])
        created.append(name)
        return name

    yield _make
    await admin.delete_topics(created)
    await admin.close()


@pytest.fixture
async def real_redis() -> AsyncIterator[aioredis.Redis]:
    url = load_settings().redis_url.rsplit("/", 1)[0] + f"/{REDIS_TEST_DB}"
    client = aioredis.Redis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except aioredis.RedisError as exc:
        pytest.skip(f"Redis not reachable ({exc}): run `make up`")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()
