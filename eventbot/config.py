"Configuration: config/eventbot.yaml validated by pydantic, plus connection settings from .env."

import os
import tomllib
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eventbot.paper_guard import assert_paper_port

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "eventbot.yaml"
DEFAULT_TOML = ROOT / "config.toml"
DEFAULT_ENV = ROOT / ".env"


RESERVED_CLIENT_IDS = frozenset({17, 18, 19, 27, 28, 40, 42, 47})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TopicConfig(StrictModel):
    name: str
    partitions: int = Field(ge=1)
    retention_ms: int = Field(ge=-1)
    durable: bool


class TopicsConfig(StrictModel):
    md_ticks: TopicConfig
    orders_intents: TopicConfig
    orders_events: TopicConfig
    fills: TopicConfig
    positions_snapshot: TopicConfig
    alerts: TopicConfig

    def all(self) -> list[TopicConfig]:
        return [
            self.md_ticks,
            self.orders_intents,
            self.orders_events,
            self.fills,
            self.positions_snapshot,
            self.alerts,
        ]

    @model_validator(mode="after")
    def unique_names(self) -> "TopicsConfig":
        names = [t.name for t in self.all()]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate topic names: {names}")
        return self


class CapsConfig(StrictModel):
    max_order_notional_usd: float = Field(gt=0)
    max_gross_notional_usd: float = Field(gt=0)
    max_position_notional_usd: float = Field(gt=0)
    max_orders_per_day: int = Field(gt=0)


class HeartbeatConfig(StrictModel):
    interval_s: float = Field(gt=0)
    ttl_s: int = Field(gt=0)
    md_max_tick_age_s: float = Field(gt=0)

    @model_validator(mode="after")
    def interval_inside_ttl(self) -> "HeartbeatConfig":
        if self.interval_s >= self.ttl_s:
            raise ValueError("heartbeats.interval_s must be below ttl_s, or heartbeats would flap")
        return self


class AdapterConfig(StrictModel):
    market_data_type: int = Field(ge=1, le=4)
    base_watchlist: list[str]
    max_market_data_lines: int = Field(gt=0, le=100)
    request_timeout_s: float = Field(gt=0)
    reconnect_delays_s: list[float] = Field(min_length=1)
    snapshot_interval_s: float = Field(gt=0)
    intent_max_age_s: float = Field(gt=0)
    max_clock_skew_s: float = Field(gt=0)
    stop_file: str


class ReconcilerConfig(StrictModel):
    interval_minutes: float = Field(gt=0)
    settle_s: float = Field(ge=0)
    max_snapshot_age_s: float = Field(gt=0)


class StrategyServiceConfig(StrictModel):
    name: str
    mode: str
    run_times_ny: list[str] = Field(min_length=1)
    state_dir: str
    price_wait_s: float = Field(gt=0)
    snapshot_max_age_s: float = Field(gt=0)
    fill_wait_margin_s: float = Field(ge=0)

    @field_validator("mode")
    @classmethod
    def known_mode(cls, v: str) -> str:
        if v not in ("dry", "execute"):
            raise ValueError(f"strategy.mode must be dry or execute, not {v!r}")
        return v

    @field_validator("run_times_ny")
    @classmethod
    def hhmm(cls, v: list[str]) -> list[str]:
        for t in v:
            h, _, m = t.partition(":")
            if not (h.isdigit() and m.isdigit() and int(h) < 24 and int(m) < 60):
                raise ValueError(f"Bad HH:MM time {t!r}")
        return sorted(v)


class RecorderConfig(StrictModel):
    out_dir: str
    flush_records: int = Field(gt=0)
    flush_seconds: float = Field(gt=0)


class LoggingConfig(StrictModel):
    dir: str


class EventbotConfig(StrictModel):
    topics: TopicsConfig
    caps: CapsConfig
    heartbeats: HeartbeatConfig
    adapter: AdapterConfig
    reconciler: ReconcilerConfig
    strategy: StrategyServiceConfig
    recorder: RecorderConfig
    logging: LoggingConfig


class Settings(StrictModel):
    """Connection details from .env. The port check runs here, before any connection exists."""

    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_account: str = ""
    ib_client_id: int = 71
    kafka_bootstrap: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379/0"

    @field_validator("ib_port")
    @classmethod
    def paper_port(cls, v: int) -> int:
        return assert_paper_port(v)

    @field_validator("ib_client_id")
    @classmethod
    def free_client_id(cls, v: int) -> int:
        if v in RESERVED_CLIENT_IDS:
            raise ValueError(f"IB_CLIENT_ID {v} is already used by another tool on this Gateway")
        return v


def load_config(path: Path = DEFAULT_CONFIG) -> EventbotConfig:
    with open(path, encoding="utf-8") as f:
        return EventbotConfig.model_validate(yaml.safe_load(f))


def load_settings(env_file: Path | None = DEFAULT_ENV) -> Settings:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    mapping = {
        "ib_host": "IB_HOST",
        "ib_port": "IB_PORT",
        "ib_account": "IB_ACCOUNT",
        "ib_client_id": "IB_CLIENT_ID",
        "kafka_bootstrap": "KAFKA_BOOTSTRAP",
        "redis_url": "REDIS_URL",
    }
    values: dict[str, Any] = {k: os.environ[v] for k, v in mapping.items() if os.environ.get(v)}
    return Settings.model_validate(values)


def load_trading_config(path: Path = DEFAULT_TOML) -> dict[str, Any]:
    """The daily bot's config.toml, read as data: the strategy and sizing parameters stay there."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def check_caps_against_toml(cfg: EventbotConfig, trading: dict[str, Any]) -> None:
    """Placement caps may be tighter than the planning limits, never looser."""
    risk = trading["risk"]
    problems = []
    if cfg.caps.max_order_notional_usd > risk["max_order_usd"]:
        problems.append("caps.max_order_notional_usd exceeds config.toml max_order_usd")
    if cfg.caps.max_gross_notional_usd > risk["capital_usd"]:
        problems.append("caps.max_gross_notional_usd exceeds config.toml capital_usd")
    if problems:
        raise ValueError("; ".join(problems))
