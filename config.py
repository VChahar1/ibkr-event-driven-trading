"config.py - the one place config.toml and .env are read."

import os
import tomllib
from pathlib import Path
from typing import Any

from strategy import StrategyConfig

ROOT = Path(__file__).parent
LIVE_PORTS = frozenset({4001, 7496})


def load_config(
    path: Path = ROOT / "config.toml", env_file: Path | None = ROOT / ".env"
) -> tuple[dict, StrategyConfig]:
    if env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    with open(path, "rb") as f:
        conf = tomllib.load(f)
    conf["ib"]["account"] = os.environ.get("IB_ACCOUNT", conf["ib"]["account"]).strip()
    params: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in conf["strategy"].items()}
    cfg = StrategyConfig(**params)
    return conf, cfg


def is_live_port(port: int) -> bool:
    return port in LIVE_PORTS
