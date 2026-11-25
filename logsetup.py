"logsetup.py - JSON-lines run log + readable console, with account IDs redacted."

import datetime as dt
import json
import logging
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ACCOUNT_ID = re.compile(r"\bD?U[A-Z]?\d{5,}\b")


def redact(text: str) -> str:
    return ACCOUNT_ID.sub(lambda m: m.group(0)[: 2 if m.group(0).startswith("DU") else 1] + "***", text)


def ev(event: str, **data: object) -> dict:
    """`extra=` payload for a structured log record."""
    return {"event": event, "data": data}


def code_version(repo: Path) -> str:
    """Short git hash, with "-dirty" if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


@dataclass
class RunContext:
    mode: str
    git: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class JsonFormatter(logging.Formatter):
    def __init__(self, ctx: RunContext) -> None:
        super().__init__()
        self.ctx = ctx

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": self.ctx.run_id,
            "git": self.ctx.git,
            "mode": self.ctx.mode,
            "event": getattr(record, "event", None),
            "msg": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if data:
            out["data"] = data
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(out, default=str))


def setup_logging(log_dir: Path, ctx: RunContext) -> None:
    """Console (human) + logs/bot.jsonl (machine). Replaces any existing root handlers."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(console)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = logging.FileHandler(log_dir / "bot.jsonl", encoding="utf-8")
    jsonl.setFormatter(JsonFormatter(ctx))
    root.addHandler(jsonl)

    logging.getLogger("ib_async").setLevel(logging.WARNING)
