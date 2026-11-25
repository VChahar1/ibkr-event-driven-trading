#!/usr/bin/env python3
"alerts.py - best-effort email alerts (stdlib smtplib, STARTTLS on port 587)."

import datetime as dt
import logging
import os
import smtplib
import ssl
import sys
from collections.abc import Callable, Iterable, Mapping
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from logsetup import RedactingFormatter, RunContext, ev, redact

ROOT = Path(__file__).parent
LOG = logging.getLogger("dailybot.alerts")
DEFAULT_SUMMARY_MODES = ("execute", "dry")


def email_settings(env: Mapping[str, str]) -> dict | None:
    needed = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_TO")
    if not all(env.get(k, "").strip() for k in needed):
        return None
    return {
        "host": env["SMTP_HOST"].strip(),
        "port": int(env.get("SMTP_PORT", "587") or 587),
        "user": env["SMTP_USER"].strip(),
        "password": env["SMTP_PASSWORD"],
        "to": env["ALERT_TO"].strip(),
        "from": env.get("ALERT_FROM", "").strip() or env["SMTP_USER"],
    }


def send_alert(
    subject: str,
    body: str,
    env: Mapping[str, str] | None = None,
    smtp_factory: Callable[..., Any] = smtplib.SMTP,
) -> bool:
    """Email subject/body (both redacted). Returns True if the server accepted it."""
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
        env = os.environ
    settings = email_settings(env)
    subject = redact(f"[ibkr-bot] {subject}")
    if settings is None:
        LOG.warning(
            "Alert not sent (SMTP not configured in .env): %s",
            subject,
            extra=ev("alert_skipped", subject=subject),
        )
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, settings["from"], settings["to"]
    msg.set_content(redact(body))
    try:
        with smtp_factory(settings["host"], settings["port"], timeout=20) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(settings["user"], settings["password"])
            smtp.send_message(msg)
    except Exception as e:
        LOG.error("Alert email failed: %s", e, extra=ev("alert_failed", subject=subject, error=str(e)))
        return False
    LOG.info("Alert sent: %s", subject, extra=ev("alert_sent", subject=subject))
    return True


class RunDigest(logging.Handler):
    """Collects this run's log lines and events, to build the email."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(message)s"))
        self.lines: list[str] = []
        self.events: list[tuple[str, dict]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == LOG.name:
            return
        self.lines.append(self.format(record))
        event = getattr(record, "event", None)
        if event:
            self.events.append((event, getattr(record, "data", None) or {}))

    def _all(self, name: str) -> list[dict]:
        return [d for e, d in self.events if e == name]

    def headline(self) -> str:
        if aborts := self._all("abort"):
            return aborts[-1].get("reason", "aborted")
        if self._all("crash"):
            return "unexpected error"
        if orders := self._all("order"):
            filled = [f for f in self._all("fill") if f.get("status") == "Filled"]
            return f"{len(filled)}/{len(orders)} orders filled"
        if self._all("dry_run"):
            plan = self._all("plan")[-1]
            return f"plan only, {len(plan.get('orders', []))} orders"
        if self._all("no_op"):
            return "within band, no trade"
        if self._all("plan"):
            return "no orders above min size"
        if self._all("check_ok"):
            return "check OK"
        return "finished"


def notify_run(
    ctx: RunContext,
    outcome: str,
    digest: RunDigest,
    summary_modes: Iterable[str] = DEFAULT_SUMMARY_MODES,
    send: Callable[[str, str], bool] = send_alert,
    today: dt.date | None = None,
) -> bool:
    """Problems always email; a normal run emails a summary only in summary_modes."""
    if outcome == "OK" and ctx.mode not in summary_modes:
        return False
    day = today or dt.date.today()
    subject = f"{outcome} {ctx.mode} {day}: {digest.headline()}"
    body = f"run {ctx.run_id} | code {ctx.git} | mode {ctx.mode} | outcome {outcome}\n\n" + "\n".join(
        digest.lines
    )
    return send(subject, body)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sent = send_alert(sys.argv[1] if len(sys.argv) > 1 else "alert", sys.stdin.read())
    sys.exit(0 if sent else 1)
