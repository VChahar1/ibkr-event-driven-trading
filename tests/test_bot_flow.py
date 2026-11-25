"""execute_day() end to end against FakeIB: every safety rule in bot.py's docstring."""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from bot import Abort, decision_path, execute_day, resolve_mode, target_weights
from config import load_config
from tests.conftest import make_prices
from tests.fakes import FAKE_ACCOUNT, FakeIB, open_trade

THU_10_LONDON = dt.datetime(2026, 10, 1, 9, 0, tzinfo=dt.UTC)
SAT_10_LONDON = dt.datetime(2026, 10, 3, 9, 0, tzinfo=dt.UTC)
THU_20_LONDON = dt.datetime(2026, 10, 1, 19, 0, tzinfo=dt.UTC)
TODAY_REF = "dailybot-20261001"


@pytest.fixture
def setup(tmp_path: Path):
    conf, cfg = load_config(env_file=None)
    conf["ib"]["account"] = FAKE_ACCOUNT
    conf["risk"].update(fill_timeout_s=0, min_trade_usd=0)
    conf["risk"].update(capital_usd=10_000, max_order_usd=10_000, max_orders_per_run=6)
    conf["bot"] = {"strategy": "etf"}
    prices_df = make_prices()
    last = prices_df.iloc[-1].to_dict()
    tw = target_weights(prices_df, cfg)
    assert tw[list(cfg.risky)].sum() > 0, "fixture should want risky ETFs"
    return conf, cfg, prices_df, last, tmp_path


def run_day(setup, ib: FakeIB, mode: str = "execute", clock=THU_10_LONDON) -> None:
    conf, cfg, prices_df, _, root = setup
    execute_day(ib, conf, cfg, prices_df, mode, root, clock=lambda: clock)


def placed(ib: FakeIB) -> list[tuple]:
    return [(t.contract.symbol, t.order.action, t.order.totalQuantity) for t in ib.placed]


def test_check_and_dry_run_place_nothing_and_record_nothing(setup) -> None:
    for mode in ("check", "dry"):
        ib = FakeIB(setup[3])
        run_day(setup, ib, mode)
        assert ib.placed == []
    assert not (setup[4] / "state").exists()


def test_execute_from_cash_buys_and_records_decision(setup) -> None:
    ib = FakeIB(setup[3])
    run_day(setup, ib)
    assert ib.placed and all(t.order.action == "BUY" for t in ib.placed)
    assert all(t.order.orderRef == TODAY_REF for t in ib.placed)
    assert all(t.contract.exchange == "SMART" for t in ib.placed)
    marker = json.loads(decision_path(setup[4] / "state", THU_10_LONDON.date()).read_text())
    assert marker["decision"] == "trade" and marker["ref"] == TODAY_REF
    journal = [json.loads(x) for x in (setup[4] / "journal.jsonl").read_text().splitlines()]
    assert len(journal) == len(ib.placed) and all(j["status"] == "Filled" for j in journal)


def test_sells_are_placed_before_buys(setup) -> None:
    last = setup[3]
    ib = FakeIB(last, positions={"IB01": int(9000 / last["IB01"])}, usd_cash=1000)
    run_day(setup, ib)
    sides = [t.order.action for t in ib.placed]
    assert sides[0] == "SELL" and "BUY" in sides
    assert sides == sorted(sides, key=lambda s: s != "SELL")


def test_unfilled_sells_block_all_buys(setup) -> None:
    last = setup[3]
    ib = FakeIB(last, positions={"IB01": int(9000 / last["IB01"])}, usd_cash=1000, fill_sides=("BUY",))
    with pytest.raises(Abort, match="SELL orders not fully filled"):
        run_day(setup, ib)
    assert {t.order.action for t in ib.placed} == {"SELL"}
    assert len(ib.cancelled) == len(ib.placed)


def test_fills_already_today_abort_execute_but_only_warn_in_dry_run(setup) -> None:
    with pytest.raises(Abort, match="already traded today"):
        run_day(setup, FakeIB(setup[3], fill_refs=(TODAY_REF,)))
    ib = FakeIB(setup[3], fill_refs=(TODAY_REF,))
    run_day(setup, ib, "dry")
    assert ib.placed == []


def test_open_order_on_bot_symbol_aborts(setup) -> None:
    ib = FakeIB(setup[3], open_trades=(open_trade("EIMI"),))
    with pytest.raises(Abort, match="Open orders exist"):
        run_day(setup, ib)
    assert ib.placed == []


def test_kill_switch_appearing_mid_run_blocks_orders(setup) -> None:
    (setup[4] / "STOP").touch()
    ib = FakeIB(setup[3])
    with pytest.raises(Abort, match="STOP"):
        run_day(setup, ib)
    assert ib.placed == [] and not (setup[4] / "state").exists()


def test_price_gap_aborts_before_any_order(setup) -> None:
    prices = {**setup[3], "CSPX": setup[3]["CSPX"] * 1.10}
    ib = FakeIB(prices)
    with pytest.raises(Abort, match="differ"):
        run_day(setup, ib)
    assert ib.placed == []


def test_live_account_aborts_before_any_order(setup) -> None:
    setup[0]["ib"]["account"] = ""
    ib = FakeIB(setup[3], accounts=("U" + "7654321",))
    with pytest.raises(Abort, match="LIVE account"):
        run_day(setup, ib)
    assert ib.placed == []


def test_outside_session_aborts_execute_even_if_mode_resolution_was_bypassed(setup) -> None:
    ib = FakeIB(setup[3])
    with pytest.raises(Abort, match="not in session"):
        run_day(setup, ib, clock=THU_20_LONDON)
    assert ib.placed == []


def test_within_band_execute_records_noop_and_places_nothing(setup) -> None:
    conf, cfg, prices_df, last, root = setup
    ib = FakeIB(last)
    run_day(setup, ib)
    settled: dict[str, int] = {}
    for t in ib.placed:
        settled[t.contract.symbol] = settled.get(t.contract.symbol, 0) + t.order.totalQuantity
    spent = sum(q * last[s] for s, q in settled.items())
    decision_path(root / "state", THU_10_LONDON.date()).unlink()
    ib2 = FakeIB(last, positions=settled, usd_cash=10_000 - spent)
    run_day(setup, ib2)
    assert ib2.placed == []
    marker = json.loads(decision_path(root / "state", THU_10_LONDON.date()).read_text())
    assert marker["decision"] == "no-op"


def test_second_execute_same_day_is_downgraded_to_dry(setup) -> None:
    root = setup[4]
    run_day(setup, FakeIB(setup[3]))
    mode, why = resolve_mode("execute", THU_10_LONDON, root / "state")
    assert mode == "dry" and "decision is made" in why


def test_racing_second_decision_aborts(setup) -> None:
    run_day(setup, FakeIB(setup[3]))
    ib = FakeIB(setup[3])
    with pytest.raises(Abort, match="another run decided first"):
        run_day(setup, ib)
    assert ib.placed == []


@pytest.mark.parametrize(
    ("requested", "now", "expected"),
    [
        ("execute", THU_10_LONDON, "execute"),
        ("execute", SAT_10_LONDON, "skip"),
        ("execute", THU_20_LONDON, "skip"),
        ("dry", SAT_10_LONDON, "dry"),
        ("check", SAT_10_LONDON, "check"),
    ],
)
def test_resolve_mode(tmp_path: Path, requested: str, now: dt.datetime, expected: str) -> None:
    assert resolve_mode(requested, now, tmp_path / "state")[0] == expected


def test_fake_prices_match_signal_prices(setup) -> None:

    pd.testing.assert_series_equal(pd.Series(setup[3]), setup[2].iloc[-1], check_names=False)
