"""Hard rule 1: paper only, and no way to bypass it."""

import ast
import inspect
from pathlib import Path

import pytest

from eventbot import paper_guard
from eventbot.config import Settings, load_settings
from eventbot.paper_guard import (
    PaperGuardError,
    assert_paper_account,
    assert_paper_port,
    mask_account,
    redact,
)

PACKAGE = Path(paper_guard.__file__).parent
SCRIPTS = PACKAGE.parent / "scripts"

PAPER = "DU" + "1234567"
LIVE = "U" + "1234567"


@pytest.mark.parametrize("port", [4002, 7497])
def test_paper_ports_allowed(port: int) -> None:
    assert assert_paper_port(port) == port


@pytest.mark.parametrize("port", [4001, 7496, 0, 4003, 8888])
def test_any_other_port_refused(port: int) -> None:
    with pytest.raises(PaperGuardError):
        assert_paper_port(port)


def test_settings_refuse_a_live_port_before_any_connection() -> None:
    with pytest.raises(PaperGuardError):
        Settings(ib_port=4001)


def test_load_settings_refuses_a_live_port_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IB_PORT", "7496")
    with pytest.raises(PaperGuardError):
        load_settings(env_file=None)


def test_paper_account_accepted() -> None:
    assert assert_paper_account([PAPER]) == PAPER
    assert assert_paper_account([PAPER, "DU" + "7654321"], PAPER) == PAPER


@pytest.mark.parametrize(
    "accounts,configured",
    [
        ([LIVE], ""),
        ([LIVE], LIVE),
        ([LIVE, PAPER], LIVE),
        ([], ""),
        ([PAPER], "DU" + "9999999"),
        ([PAPER, "DU" + "7654321"], ""),
        (["DF" + "1234567"], ""),
    ],
)
def test_account_refused(accounts: list[str], configured: str) -> None:
    with pytest.raises(PaperGuardError):
        assert_paper_account(accounts, configured)


def test_guard_functions_take_no_override_arguments() -> None:
    assert list(inspect.signature(assert_paper_port).parameters) == ["port"]
    assert list(inspect.signature(assert_paper_account).parameters) == ["managed_accounts", "configured"]


def _python_files() -> list[Path]:
    return [*PACKAGE.rglob("*.py"), *SCRIPTS.glob("*.py")]


def test_no_bypass_vocabulary_in_new_code() -> None:
    """No allow_live, --live or similar anywhere in eventbot/ or scripts/. Comments and strings are
    not exempt: this reads the raw text."""
    banned = ("allow_live", "--live", "live_flag", "skip_guard", "bypass_guard", "force_live")
    offenders = [
        f"{p}: {word}" for p in _python_files() for word in banned if word in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_new_code_does_not_use_the_daily_bots_bypassable_guards() -> None:
    """bot.check_port and bot.resolve_account accept a live override, so they must not be called."""
    banned = {"check_port", "resolve_account", "is_live_port"}
    offenders = []
    for path in _python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module in ("bot", "config", "stockbot"):
                offenders += [f"{path}: {a.name}" for a in node.names if a.name in banned]
    assert offenders == []


def test_redaction_and_mask() -> None:
    masked = mask_account(PAPER)
    assert masked == "DU***567" and PAPER not in masked
    for text in (f"acct={PAPER}", f"acct_{PAPER}", f"{PAPER}_x", f"U{'9' * 7} here"):
        assert not paper_guard.ACCOUNT_RE.search(redact(text)), text
