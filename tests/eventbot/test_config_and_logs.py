import copy
import json
import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eventbot.config import (
    DEFAULT_CONFIG,
    EventbotConfig,
    Settings,
    check_caps_against_toml,
    load_config,
    load_trading_config,
)
from eventbot.logs import JsonFormatter, log_event, setup_logging

PAPER = "DU" + "1234567"


def raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def test_the_committed_config_is_valid_and_consistent_with_config_toml() -> None:
    cfg = load_config()
    check_caps_against_toml(cfg, load_trading_config())
    assert [t.name for t in cfg.topics.all()] == [
        "md.ticks",
        "orders.intents",
        "orders.events",
        "fills",
        "positions.snapshot",
        "alerts",
    ]
    assert cfg.strategy.mode == "dry", "the committed default must not publish intents"


def test_durable_topics_are_the_ones_cold_start_needs() -> None:
    topics = load_config().topics
    assert topics.fills.durable and topics.orders_intents.durable and topics.orders_events.durable
    assert topics.fills.retention_ms == -1 and topics.positions_snapshot.retention_ms == -1


@pytest.mark.parametrize(
    "path,value",
    [
        (("caps", "max_order_notional_usd"), -1),
        (("caps", "not_a_cap"), 1),
        (("strategy", "mode"), "live"),
        (("strategy", "run_times_ny"), ["25:00"]),
        (("heartbeats", "interval_s"), 9),
        (("adapter", "max_market_data_lines"), 500),
        (("topics", "fills", "name"), "md.ticks"),
    ],
)
def test_invalid_config_fails_at_startup(path: tuple, value: object) -> None:
    data = copy.deepcopy(raw())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        EventbotConfig.model_validate(data)


def test_yaml_caps_may_not_be_looser_than_config_toml() -> None:
    data = raw()
    data["caps"]["max_order_notional_usd"] = 999_999
    with pytest.raises(ValueError, match="max_order_notional_usd"):
        check_caps_against_toml(EventbotConfig.model_validate(data), load_trading_config())
    data = raw()
    data["caps"]["max_gross_notional_usd"] = 999_999
    with pytest.raises(ValueError, match="max_gross_notional_usd"):
        check_caps_against_toml(EventbotConfig.model_validate(data), load_trading_config())


@pytest.mark.parametrize("client_id", [17, 27, 40])
def test_client_ids_used_by_other_tools_are_refused(client_id: int) -> None:
    with pytest.raises(ValidationError):
        Settings(ib_client_id=client_id)


def capture(service: str = "adapter") -> tuple[logging.Logger, list[str]]:
    lines: list[str] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(self.format(record))

    logger = logging.getLogger(f"t.{service}.{id(lines)}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = ListHandler()
    handler.setFormatter(JsonFormatter(service))
    logger.addHandler(handler)
    return logger, lines


def test_log_lines_are_json_with_intent_and_order_ref() -> None:
    logger, lines = capture()
    log_event(logger, "order_placed", intent_id="abc", order_ref="abc", symbol="AMD")
    entry = json.loads(lines[0])
    assert entry["event"] == "order_placed" and entry["service"] == "adapter"
    assert entry["intent_id"] == "abc" and entry["order_ref"] == "abc" and entry["symbol"] == "AMD"
    assert entry["ts"].endswith("+00:00")


def test_log_lines_never_contain_an_account_id() -> None:
    logger, lines = capture()
    log_event(logger, "x", msg=f"account {PAPER}", account=PAPER, nested={"a": f"pre_{PAPER}"})
    try:
        raise ValueError(f"boom {PAPER}")
    except ValueError:
        logger.exception("failed")
    for line in lines:
        assert PAPER not in line


def test_setup_logging_writes_a_service_file(tmp_path: Path) -> None:
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    try:
        setup_logging("recorder", tmp_path)
        log_event(logging.getLogger("x"), "hello", intent_id="i")
        for handler in root.handlers:
            handler.flush()
        entry = json.loads((tmp_path / "recorder.jsonl").read_text().splitlines()[-1])
        assert entry["service"] == "recorder" and entry["intent_id"] == "i"
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = saved
        root.setLevel(level)
