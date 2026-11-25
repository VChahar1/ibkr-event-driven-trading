"""Structured run log: valid JSON lines with run context, and no account IDs anywhere."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from logsetup import RunContext, code_version, ev, redact, setup_logging

PAPER = "DU" + "1234567"
PAPER_LETTER = "DU" + "Q000042"
LIVE = "U" + "7654321"


@pytest.fixture
def logdir(tmp_path: Path) -> Iterator[tuple[Path, RunContext]]:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, logging.getLogger("ib_async").level)
    ctx = RunContext(mode="dry", git="abc1234")
    setup_logging(tmp_path, ctx)
    yield tmp_path, ctx
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in saved[0]:
        root.addHandler(h)
    root.setLevel(saved[1])
    logging.getLogger("ib_async").setLevel(saved[2])


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "bot.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("acct", [PAPER, PAPER_LETTER, LIVE])
def test_redact_masks_paper_and_live_ids(acct: str) -> None:
    out = redact(f"Position(account='{acct}', position=3.0)")
    assert acct not in out and "***" in out


def test_redact_leaves_ordinary_text_alone() -> None:
    text = "BUY 89 EIMI (~4881 USD) conId=153454131 USD cash 24356.39"
    assert redact(text) == text


def test_records_carry_run_context_event_and_data(logdir) -> None:
    path, ctx = logdir
    logging.getLogger("dailybot").info("Budget %d", 9800, extra=ev("plan", budget=9800, orders=[]))
    ctx.mode = "execute"
    logging.getLogger("dailybot").info("plain")
    first, second = records(path)
    assert first["event"] == "plan" and first["data"] == {"budget": 9800, "orders": []}
    assert first["msg"] == "Budget 9800" and first["run_id"] == ctx.run_id
    assert first["git"] == "abc1234" and first["mode"] == "dry"
    assert second["mode"] == "execute" and second["event"] is None


def test_account_ids_never_reach_the_log_file(logdir, capsys: pytest.CaptureFixture) -> None:
    path, _ = logdir
    log = logging.getLogger("dailybot")
    log.info("Account %s | USD cash %.2f", PAPER, 100.0, extra=ev("account", account=PAPER))
    log.warning("ib_async says: Position(account='%s')", PAPER_LETTER)
    try:
        raise RuntimeError(f"broker error for {LIVE}")
    except RuntimeError:
        log.exception("crash")
    text = (path / "bot.jsonl").read_text() + capsys.readouterr().err
    for acct in (PAPER, PAPER_LETTER, LIVE):
        assert acct not in text


def test_ib_async_info_chatter_is_dropped(logdir) -> None:
    path, _ = logdir
    logging.getLogger("ib_async.wrapper").info("position: Position(...)")
    logging.getLogger("ib_async.wrapper").error("Error 10311, reqId 13")
    msgs = [r["msg"] for r in records(path)]
    assert msgs == ["Error 10311, reqId 13"]


def test_code_version_reports_git_hash() -> None:
    v = code_version(Path(__file__).parent.parent)
    assert v == "unknown" or len(v.split("-")[0]) >= 7
