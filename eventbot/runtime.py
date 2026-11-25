"""Shared startup for the service entry points: config, logging, Redis, Kafka, signal handling."""

import asyncio
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis

from eventbot.bus import TopicMap
from eventbot.bus.kafka import KafkaBus
from eventbot.config import (
    EventbotConfig,
    Settings,
    check_caps_against_toml,
    load_config,
    load_settings,
    load_trading_config,
)
from eventbot.logs import setup_logging
from eventbot.state import StateStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HALTED = 4
EXIT_NOT_PAPER = 5


@dataclass
class Runtime:
    cfg: EventbotConfig
    settings: Settings
    trading: dict[str, Any]
    topics: TopicMap
    bus: KafkaBus
    redis: aioredis.Redis
    state: StateStore

    async def close(self) -> None:
        await self.bus.stop()
        await self.redis.aclose()


def build_runtime(service: str) -> Runtime:
    cfg, settings, trading = load_config(), load_settings(), load_trading_config()
    check_caps_against_toml(cfg, trading)
    setup_logging(service, cfg.logging.dir)
    client = aioredis.Redis.from_url(
        settings.redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
    )
    return Runtime(
        cfg,
        settings,
        trading,
        TopicMap.from_config(cfg.topics),
        KafkaBus(settings.kafka_bootstrap, f"eventbot-{service}"),
        client,
        StateStore(client, cfg.heartbeats.ttl_s),
    )


def install_stop_handlers(stop: Callable[[], None]) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop)
