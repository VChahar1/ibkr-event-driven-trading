"""One-time ETF unwind: sells exactly the listed ETFs, only on paper, only during LSE hours."""

import datetime as dt
from pathlib import Path

import pytest

from bot import Abort
from config import load_config
from tests.fakes import FAKE_ACCOUNT, FakeIB, open_trade
from unwind import unwind

THU_10_LONDON = dt.datetime(2026, 10, 1, 9, 0, tzinfo=dt.UTC)
SAT = dt.datetime(2026, 10, 3, 9, 0, tzinfo=dt.UTC)
PRICES = {"CSPX": 830.0, "EIMI": 55.0, "IDTL": 2.9, "IGLN": 81.0, "IB01": 122.0, "AAPL": 330.0}


@pytest.fixture
def conf() -> dict:
    c, _ = load_config(env_file=None)
    c["ib"]["account"] = FAKE_ACCOUNT
    c["risk"].update(max_order_usd=5_000, fill_timeout_s=0)
    return c


def go(
    conf: dict,
    ib: FakeIB,
    root: Path,
    mode: str = "execute",
    clock: dt.datetime = THU_10_LONDON,
    symbols: tuple = ("CSPX", "EIMI"),
) -> list:
    return unwind(ib, conf, list(symbols), mode, root, clock=lambda: clock)


def test_dry_run_plans_full_sells_and_places_nothing(conf: dict, tmp_path: Path) -> None:
    ib = FakeIB(PRICES, positions={"CSPX": 5, "EIMI": 88, "AAPL": 3})
    orders = go(conf, ib, tmp_path, "dry")
    assert [(s, side, q) for s, side, q, _ in orders] == [("CSPX", "SELL", 5), ("EIMI", "SELL", 88)]
    assert ib.placed == []


def test_execute_sells_exactly_the_listed_etfs(conf: dict, tmp_path: Path) -> None:
    ib = FakeIB(PRICES, positions={"CSPX": 5, "EIMI": 88, "IGLN": 13, "AAPL": 3})
    go(conf, ib, tmp_path)
    assert sorted((t.contract.symbol, t.order.action, t.order.totalQuantity) for t in ib.placed) == [
        ("CSPX", "SELL", 5),
        ("EIMI", "SELL", 88),
    ]
    assert {t.order.orderRef for t in ib.placed} == {"unwind-20261001"}
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) == 2


def test_refuses_outside_lse_hours(conf: dict, tmp_path: Path) -> None:
    ib = FakeIB(PRICES, positions={"CSPX": 5})
    with pytest.raises(Abort, match="LSE closed"):
        go(conf, ib, tmp_path, clock=SAT)
    assert ib.placed == []


def test_refuses_symbols_outside_the_etf_universe(conf: dict, tmp_path: Path) -> None:
    with pytest.raises(Abort, match="not in config.toml"):
        go(conf, FakeIB(PRICES, positions={"AAPL": 3}), tmp_path, symbols=("AAPL",))


def test_nothing_held_is_a_no_op(conf: dict, tmp_path: Path) -> None:
    ib = FakeIB(PRICES, positions={"AAPL": 3})
    assert go(conf, ib, tmp_path) == [] and ib.placed == []


def test_cap_and_same_day_guard(conf: dict, tmp_path: Path) -> None:
    conf["risk"]["max_order_usd"] = 1_000
    with pytest.raises(Abort, match="max_order_usd"):
        go(conf, FakeIB(PRICES, positions={"CSPX": 5}), tmp_path)
    conf["risk"]["max_order_usd"] = 5_000
    with pytest.raises(Abort, match="already traded today"):
        go(conf, FakeIB(PRICES, positions={"CSPX": 5}, fill_refs=("unwind-20261001",)), tmp_path)
    with pytest.raises(Abort, match="Open orders"):
        go(conf, FakeIB(PRICES, positions={"CSPX": 5}, open_trades=(open_trade("CSPX"),)), tmp_path)


def test_kill_switch(conf: dict, tmp_path: Path) -> None:
    (tmp_path / "STOP").touch()
    with pytest.raises(Abort, match="STOP"):
        go(conf, FakeIB(PRICES, positions={"CSPX": 5}), tmp_path)


def test_live_account_refused(conf: dict, tmp_path: Path) -> None:
    conf["ib"]["account"] = ""
    with pytest.raises(Abort, match="LIVE account"):
        go(conf, FakeIB(PRICES, positions={"CSPX": 5}, accounts=("U" + "7654321",)), tmp_path)
