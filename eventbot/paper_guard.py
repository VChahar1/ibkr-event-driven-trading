"""The paper-only guard."""

import re
from collections.abc import Iterable

PAPER_PORTS = frozenset({4002, 7497})
PAPER_ACCOUNT_PREFIX = "DU"


ACCOUNT_RE = re.compile(r"(?<![A-Z0-9])D?U[A-Z]?\d{5,}(?![0-9])")


class PaperGuardError(RuntimeError):
    """Refusing to run: this is not provably a paper connection."""


def assert_paper_port(port: int) -> int:
    if port not in PAPER_PORTS:
        raise PaperGuardError(f"Port {port} is not a paper port {sorted(PAPER_PORTS)} - refusing to connect.")
    return port


def assert_paper_account(managed_accounts: Iterable[str], configured: str = "") -> str:
    """Return the one paper account to trade, or raise. Called after every (re)connect, because
    the Gateway can log back in between connections."""
    accounts = [a.strip() for a in managed_accounts if a.strip()]
    configured = configured.strip()
    if not accounts:
        raise PaperGuardError("The Gateway reports no managed accounts - refusing to run.")
    if configured:
        if configured not in accounts:
            raise PaperGuardError("IB_ACCOUNT is not among the accounts this Gateway manages.")
        account = configured
    elif len(accounts) == 1:
        account = accounts[0]
    else:
        raise PaperGuardError(f"The Gateway manages {len(accounts)} accounts - set IB_ACCOUNT in .env.")
    if not account.startswith(PAPER_ACCOUNT_PREFIX):
        raise PaperGuardError(
            f"{mask_account(account)} is not a paper account (must start with "
            f"{PAPER_ACCOUNT_PREFIX}) - refusing to run."
        )
    return account


def mask_account(account: str) -> str:
    """Keep the first two and last three characters (DU***567). Used for Kafka keys and payloads,
    logs and status output, so the audit trail never holds the real ID."""
    return f"{account[:2]}***{account[-3:]}" if len(account) > 5 else "***"


def redact(text: str) -> str:
    return ACCOUNT_RE.sub(lambda m: mask_account(m.group(0)), text)
