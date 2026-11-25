"""Structural rules that a code review could miss."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "eventbot"
IB_MODULE = PACKAGE / "adapter" / "ib_client.py"

UNCHANGED = [
    "bot.py",
    "stockbot.py",
    "stock_strategy.py",
    "strategy.py",
    "config.py",
    "config.toml",
    "data.py",
    "universe.py",
    "backtest_stocks.py",
    "logsetup.py",
    "alerts.py",
    "run_daily.sh",
    "universe/sp100.csv",
]


def imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def package_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_only_one_module_imports_ib_libraries() -> None:
    offenders = [
        p
        for p in package_files()
        if p != IB_MODULE and imports(p) & {"ib_async", "ib_insync", "ibapi", "eventkit"}
    ]
    assert offenders == []


def test_the_adapter_main_is_the_only_entry_point_that_builds_the_real_client() -> None:
    allowed = {IB_MODULE, PACKAGE / "adapter" / "main.py"}
    offenders = [
        p
        for p in package_files()
        if p not in allowed and "adapter.ib_client" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "module",
    [
        "eventbot.schemas",
        "eventbot.bus",
        "eventbot.bus.kafka",
        "eventbot.ledger",
        "eventbot.risk",
        "eventbot.state",
        "eventbot.adapter.broker",
        "eventbot.adapter.fake_broker",
        "eventbot.adapter.service",
    ],
)
def test_importing_a_module_does_not_load_ib_async(module: str) -> None:
    code = f"import sys, {module}; sys.exit(1 if 'ib_async' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True).returncode == 0


def test_no_code_can_flatten_positions() -> None:
    """Halt, never flatten: nothing in the package closes positions by itself. Only the strategy's plan
    produces orders, and only the adapter's intent path sends them."""
    suspicious = ("flatten", "close_all", "liquidate", "exit_all")
    offenders = []
    for path in package_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                s in node.name for s in suspicious
            ):
                offenders.append(f"{path}:{node.name}")
    assert offenders == []


def test_only_the_adapter_service_calls_place_market_order() -> None:
    offenders = []
    for path in package_files():
        if path.parent.name == "adapter":
            continue
        if "place_market_order" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert offenders == []


def test_the_live_bots_files_are_unchanged_from_main() -> None:
    try:
        known = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "main"], cwd=ROOT, capture_output=True
        )
    except OSError:
        pytest.skip("git is not available")
    if known.returncode != 0:
        pytest.skip("no main branch in this checkout")
    result = subprocess.run(
        ["git", "diff", "--stat", "main", "--", *UNCHANGED], cwd=ROOT, capture_output=True, text=True
    )
    assert result.stdout == "", f"Files that must stay byte-identical have changed:\n{result.stdout}"
