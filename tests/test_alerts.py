"""Email alerts: content, redaction, and that a failing mail server never breaks a run."""

import datetime as dt
import logging

import pytest

from alerts import RunDigest, email_settings, notify_run, send_alert
from logsetup import RunContext, ev

ENV = {
    "SMTP_HOST": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_USER": "bot@example.com",
    "SMTP_PASSWORD": "app-password",
    "ALERT_TO": "me@example.com",
}
ACCT = "DU" + "1234567"


class FakeSMTP:
    sent: list = []
    calls: list = []

    def __init__(self, host: str, port: int, timeout: float) -> None:
        FakeSMTP.calls.append(("connect", host, port))

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def starttls(self, context: object) -> None:
        FakeSMTP.calls.append(("starttls",))

    def login(self, user: str, password: str) -> None:
        FakeSMTP.calls.append(("login", user))

    def send_message(self, msg: object) -> None:
        FakeSMTP.sent.append(msg)


class BrokenSMTP(FakeSMTP):
    def login(self, user: str, password: str) -> None:
        raise OSError("535 bad credentials")


@pytest.fixture(autouse=True)
def reset_fake() -> None:
    FakeSMTP.sent, FakeSMTP.calls = [], []


def test_settings_require_all_fields() -> None:
    assert email_settings(ENV) is not None
    assert email_settings({**ENV, "SMTP_PASSWORD": ""}) is None
    assert email_settings({}) is None


def test_sends_redacted_mail_over_starttls() -> None:
    ok = send_alert(f"ABORT for {ACCT}", f"Account {ACCT} | USD cash 1.00", env=ENV, smtp_factory=FakeSMTP)
    assert ok
    assert [c[0] for c in FakeSMTP.calls] == ["connect", "starttls", "login"]
    msg = FakeSMTP.sent[0]
    assert ACCT not in msg["Subject"] and ACCT not in msg.get_content()
    assert msg["To"] == "me@example.com" and msg["From"] == "bot@example.com"


def test_unconfigured_smtp_is_a_logged_noop(caplog: pytest.LogCaptureFixture) -> None:
    assert send_alert("x", "y", env={}, smtp_factory=FakeSMTP) is False
    assert FakeSMTP.calls == [] and "not configured" in caplog.text


def test_mail_server_failure_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    assert send_alert("x", "y", env=ENV, smtp_factory=BrokenSMTP) is False
    assert "Alert email failed" in caplog.text


def digest_with(*events: tuple[str, dict]) -> RunDigest:
    d = RunDigest()
    log = logging.getLogger("dailybot.test_alerts")
    log.addHandler(d)
    log.setLevel(logging.INFO)
    try:
        for name, data in events:
            log.info("%s happened", name, extra=ev(name, **data))
    finally:
        log.removeHandler(d)
    return d


@pytest.mark.parametrize(
    ("events", "headline"),
    [
        ([("abort", {"reason": "SELL orders not fully filled"})], "SELL orders not fully filled"),
        (
            [
                ("plan", {"orders": [1, 2]}),
                ("order", {}),
                ("order", {}),
                ("fill", {"status": "Filled"}),
                ("fill", {"status": "Cancelled"}),
            ],
            "1/2 orders filled",
        ),
        ([("plan", {"orders": [1, 2, 3]}), ("dry_run", {})], "plan only, 3 orders"),
        ([("no_op", {})], "within band, no trade"),
        ([("plan", {"orders": []})], "no orders above min size"),
    ],
)
def test_headline(events: list, headline: str) -> None:
    assert digest_with(*events).headline() == headline


def test_problems_always_mail_but_summaries_follow_summary_modes() -> None:
    sent: list[tuple[str, str]] = []

    def send(subject: str, body: str) -> bool:
        sent.append((subject, body))
        return True

    day = dt.date(2026, 10, 2)
    digest = digest_with(("no_op", {}))
    assert not notify_run(RunContext(mode="check", git="g"), "OK", digest, send=send, today=day)
    assert not notify_run(RunContext(mode="skip", git="g"), "OK", digest, send=send, today=day)
    assert notify_run(RunContext(mode="execute", git="g"), "OK", digest, send=send, today=day)
    assert notify_run(RunContext(mode="check", git="g"), "ABORT", digest, send=send, today=day)
    assert sent[0][0] == "OK execute 2026-10-02: within band, no trade"
    assert sent[1][0].startswith("ABORT check")
    assert "no_op happened" in sent[0][1]
