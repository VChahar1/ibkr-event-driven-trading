"""Config loading, secrets handling, and the paper-only guards."""

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot import Abort, check_port, resolve_account
from config import ROOT, load_config

FAKE_PAPER = "DU" + "1234567"
FAKE_LIVE = "U" + "7654321"
ACCOUNT_ID = re.compile(r"\bDU[A-Z]?\d{5,}\b")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:

    monkeypatch.setenv("IB_ACCOUNT", "x")
    monkeypatch.delenv("IB_ACCOUNT")
    return monkeypatch


def test_account_comes_from_env_file(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(f"IB_ACCOUNT={FAKE_PAPER}\n")
    conf, cfg = load_config(env_file=env)
    assert conf["ib"]["account"] == FAKE_PAPER
    assert cfg.top_n == 2 and cfg.safe == "IB01"


def test_process_environment_beats_env_file(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("IB_ACCOUNT=from-file\n")
    clean_env.setenv("IB_ACCOUNT", FAKE_PAPER)
    assert load_config(env_file=env)[0]["ib"]["account"] == FAKE_PAPER


def test_committed_config_has_no_account() -> None:
    import tomllib

    assert tomllib.loads((ROOT / "config.toml").read_text())["ib"]["account"] == ""


@pytest.mark.parametrize("port", [4001, 7496])
def test_live_ports_refused_before_connecting(port: int) -> None:
    conf = {"ib": {"port": port, "allow_live": False}}
    with pytest.raises(Abort, match="LIVE port"):
        check_port(conf, allow_live_flag=True)
    conf["ib"]["allow_live"] = True
    with pytest.raises(Abort, match="LIVE port"):
        check_port(conf, allow_live_flag=False)


def test_paper_ports_allowed() -> None:
    for port in (4002, 7497):
        check_port({"ib": {"port": port, "allow_live": False}}, allow_live_flag=False)


def test_live_account_refused() -> None:
    ib = SimpleNamespace(managedAccounts=lambda: [FAKE_LIVE])
    conf = {"ib": {"account": "", "allow_live": False, "port": 4002}}
    with pytest.raises(Abort, match="LIVE account"):
        resolve_account(ib, conf, allow_live_flag=True)


def test_paper_account_resolved_when_single() -> None:
    ib = SimpleNamespace(managedAccounts=lambda: [FAKE_PAPER])
    conf = {"ib": {"account": "", "allow_live": False, "port": 4002}}
    assert resolve_account(ib, conf, allow_live_flag=False) == FAKE_PAPER


def git_files(*args: str) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", *args], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.decode().split("\0") if p]


def test_no_account_ids_in_tracked_or_addable_files() -> None:
    files = git_files() + git_files("--others", "--exclude-standard")
    hits = []
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            continue
        for n, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if ACCOUNT_ID.search(line):
                hits.append(f"{rel}:{n}")
    assert hits == [], f"account IDs found: {hits}"


def test_real_account_digits_appear_in_no_tracked_or_addable_file() -> None:
    """Catches the real ID even when split up (e.g. "DU" + "R..."), which the regex can't."""
    from dotenv import dotenv_values

    env = ROOT / ".env"
    account = (dotenv_values(env).get("IB_ACCOUNT") or "") if env.exists() else ""
    digits = re.sub(r"\D", "", account)
    if len(digits) < 5:
        pytest.skip("no real IB_ACCOUNT in .env to check against")
    hits = [
        rel
        for rel in git_files() + git_files("--others", "--exclude-standard")
        if (ROOT / rel).is_file() and digits in (ROOT / rel).read_text(errors="ignore")
    ]
    assert hits == [], f"real account digits found in: {hits}"


def test_secrets_and_logs_are_git_ignored() -> None:
    for name in (".env", "logs/x.log", "bot.log", "journal.jsonl"):
        r = subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT)
        assert r.returncode == 0, f"{name} is not git-ignored"
