"Structured JSON logging, one record per event, with account IDs redacted from every line."

import datetime as dt
import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from eventbot.paper_guard import redact

MAX_BYTES = 20_000_000
BACKUPS = 3


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "event": getattr(record, "event", record.name),
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(entry, default=str))


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, msg: str = "", **fields: Any
) -> None:
    """Pass intent_id and order_ref as fields wherever they apply, so a single order can be followed
    through every service with one grep."""
    logger.log(level, msg or event, extra={"event": event, "fields": fields})


def setup_logging(service: str, log_dir: Path | str | None = None) -> None:
    """JSON to stdout, and to a rotated file when log_dir is given. Replaces existing root handlers."""
    formatter = JsonFormatter(service)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                path / f"{service}.jsonl", maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"
            )
        )
    root = logging.getLogger()
    root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("aiokafka", "ib_async"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
