"""Stock bot daily run against FakeIB: monthly rebalance, hold days, calendar, orphans, safety."""

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

import bot
from bot import Abort
from config import load_config
from stock_strategy import StockConfig, decide
from stockbot import (
    affordable_picks,
    bot_owned,
    in_liquid_hours,
    month_done,
    ny_session_open,
    parse_liquid_hours,
    stock_day,
)
from tests.conftest import make_stock_prices
from tests.fakes import FAKE_ACCOUNT, FakeIB

SYMS = [f"S{i:02d}" for i in range(1, 31)]
CFG = StockConfig(universe=tuple(SYMS), min_eligible=25)


def universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": SYMS,
            "yahoo": SYMS,
            "ib_symbol": SYMS,
            "name": SYMS,
            "sector": "X",
            "primary_exchange": "NYSE",
            "con_id": 0,
            "as_of": "2026-10-03",
        }
    )


def hours(day: dt.date, closed: bool = False) -> str:
    return f"{day:%Y%m%d}:CLOSED" if closed else f"{day:%Y%m%d}:0930-{day:%Y%m%d}:1600"


@pytest.fixture
def env(tmp_path: Path) -> dict:
    conf, _ = load_config(env_file=None)
    conf["ib"]["account"] = FAKE_ACCOUNT
    conf["risk"].update(
        capital_usd=25_000,
        max_order_usd=5_000,
        max_orders_per_run=24,
        min_trade_usd=100,
        cash_buffer=0.02,
        fill_timeout_s=0,
    )
    conf["stocks"] = {"max_price_gap": 0.15, "candidates": 10}
    prices = make_stock_prices(market_drift=1.5e-3)
    day = (prices.index[-1] + pd.offsets.BDay(1)).date()
    clock = dt.datetime.combine(day, dt.time(15, 0), tzinfo=dt.UTC)
    return {
        "conf": conf,
        "prices": prices,
        "day": day,
        "clock": clock,
        "root": tmp_path,
        "last": prices.iloc[-1].to_dict(),
    }


def run(
    env: dict,
    ib: FakeIB,
    mode: str = "execute",
    prices: pd.DataFrame | None = None,
    clock: dt.datetime | None = None,
) -> None:
    stock_day(
        ib,
        env["conf"],
        CFG,
        universe(),
        env["prices"] if prices is None else prices,
        mode,
        env["root"],
        clock=lambda: clock or env["clock"],
    )


def fake(env: dict, **kw: object) -> FakeIB:
    kw.setdefault("usd_cash", 30_000.0)
    kw.setdefault("liquid_hours", hours(env["day"]))
    return FakeIB(env["last"], **kw)  # type: ignore[arg-type]


def placed(ib: FakeIB) -> list[tuple]:
    return [(t.contract.symbol, t.order.action, t.order.totalQuantity) for t in ib.placed]


def test_liquid_hours_parsing_handles_closed_and_half_days() -> None:
    text = "20261126:CLOSED;20261127:0930-20261127:1300;20261130:0930-20261130:1600"
    sessions = parse_liquid_hours(text)
    assert len(sessions) == 2 and sessions[0][1].hour == 13
    ny = sessions[0][0].tzinfo
    assert in_liquid_hours(dt.datetime(2026, 11, 27, 12, 0, tzinfo=ny), text)
    assert not in_liquid_hours(dt.datetime(2026, 11, 27, 14, 0, tzinfo=ny), text)
    assert not in_liquid_hours(dt.datetime(2026, 11, 26, 11, 0, tzinfo=ny), text)


def test_ny_session_survives_the_us_eu_daylight_saving_gap() -> None:

    assert ny_session_open(dt.datetime(2026, 3, 9, 14, 45, tzinfo=dt.UTC))
    assert not ny_session_open(dt.datetime(2026, 3, 9, 13, 15, tzinfo=dt.UTC))
    assert not ny_session_open(dt.datetime(2026, 3, 7, 15, 0, tzinfo=dt.UTC))


def test_bot_owned_counts_only_stock_bot_fills(tmp_path: Path) -> None:
    j = tmp_path / "journal.jsonl"
    rows = [
        {"ref": "stockbot-20261005", "sym": "AAA", "side": "BUY", "filled": 10.0},
        {"ref": "stockbot-20261102", "sym": "AAA", "side": "SELL", "filled": 4.0},
        {"ref": "stockbot-20261005", "sym": "BBB", "side": "BUY", "filled": 0.0},
        {"ref": "dailybot-20261002", "sym": "CSPX", "side": "BUY", "filled": 5.0},
    ]
    j.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert bot_owned(j) == {"AAA": 6.0}


def test_month_done_reads_stock_markers_only(tmp_path: Path) -> None:
    (tmp_path / "decided-20261002.json").write_text("{}")
    assert not month_done(tmp_path, dt.date(2026, 10, 20))
    (tmp_path / "decided-stocks-20261005.json").write_text("{}")
    assert month_done(tmp_path, dt.date(2026, 10, 20)) and not month_done(tmp_path, dt.date(2026, 11, 2))


def test_affordable_picks_replace_expensive_names_with_next_ranked() -> None:
    p = make_stock_prices(market_drift=1.5e-3)
    d = decide(p, CFG)
    prices = {s: 50.0 for s in d.ranks.index}
    prices[d.picks[0]] = 3_000.0
    out = affordable_picks(d, prices, slot_usd=2_450.0, top_n=10)
    nxt = [s for s in d.ranks.index if s not in d.picks][0]
    assert d.picks[0] not in out and nxt in out and len(out) == 10


def test_check_and_dry_place_nothing_and_record_nothing(env: dict) -> None:
    for mode in ("check", "dry"):
        ib = fake(env)
        run(env, ib, mode)
        assert ib.placed == []
    assert not (env["root"] / "state").exists()


def test_first_execute_of_the_month_builds_ten_positions(env: dict) -> None:
    ib = fake(env)
    run(env, ib)
    buys = placed(ib)
    assert len(buys) == 10 and all(side == "BUY" for _, side, _ in buys)
    assert {t.order.orderRef for t in ib.placed} == {f"stockbot-{env['day']:%Y%m%d}"}
    spend = sum(q * env["last"][s] for s, _, q in buys)
    assert spend <= 25_000 * 0.98
    marker = json.loads((env["root"] / "state" / f"decided-stocks-{env['day']:%Y%m%d}.json").read_text())
    assert marker["decision"] == "trade" and len(marker["picks"]) == 10
    assert len((env["root"] / "journal.jsonl").read_text().splitlines()) == 10


def test_later_run_in_the_same_month_holds(env: dict, caplog: pytest.LogCaptureFixture) -> None:
    run(env, fake(env))
    later = env["clock"] + dt.timedelta(days=1)
    ib = fake(env, liquid_hours=hours(later.date()))
    with caplog.at_level(logging.INFO, logger="dailybot"):
        run(env, ib, clock=later)
    assert ib.placed == [] and "Hold" in caplog.text


def test_market_filter_off_sells_every_holding(env: dict) -> None:
    down = make_stock_prices(market_drift=-2e-3)
    held = {s: 10 for s in SYMS[:5]}
    ib = FakeIB(down.iloc[-1].to_dict(), positions=held, usd_cash=1_000.0, liquid_hours=hours(env["day"]))
    run(env, ib, prices=down)
    assert sorted(placed(ib)) == sorted((s, "SELL", 10) for s in SYMS[:5])


def test_orphans_from_the_journal_are_sold(env: dict) -> None:
    (env["root"] / "journal.jsonl").write_text(
        json.dumps({"ref": "stockbot-20230801", "sym": "OLD", "side": "BUY", "filled": 7.0, "avg": 20.0})
        + "\n"
    )
    ib = fake(env, positions={"OLD": 7})
    ib.prices["OLD"] = 21.0
    run(env, ib)
    assert ("OLD", "SELL", 7) in placed(ib)


def test_etf_positions_are_left_alone(env: dict) -> None:
    ib = fake(env, positions={"CSPX": 5, "EIMI": 88})
    run(env, ib)
    assert not any(s in ("CSPX", "EIMI") for s, _, _ in placed(ib))


def test_holiday_skips_without_recording_a_decision(env: dict) -> None:
    ib = fake(env, liquid_hours=hours(env["day"], closed=True))
    run(env, ib)
    assert ib.placed == [] and not (env["root"] / "state").exists()


def test_price_gap_on_a_pick_aborts_before_any_order(env: dict) -> None:
    pick = decide(env["prices"], CFG, today=env["day"]).picks[0]
    ib = fake(env)
    ib.prices[pick] *= 1.5
    with pytest.raises(Abort, match="Price check failed"):
        run(env, ib)
    assert ib.placed == []


def test_unaffordable_pick_is_replaced_live(env: dict) -> None:
    prices = env["prices"].copy()
    pick = decide(prices, CFG, today=env["day"]).picks[0]
    prices.iloc[-1, prices.columns.get_loc(pick)] = 4_000.0
    ib = FakeIB(prices.iloc[-1].to_dict(), usd_cash=30_000.0, liquid_hours=hours(env["day"]))
    run(env, ib, prices=prices)
    syms = [s for s, _, _ in placed(ib)]
    assert pick not in syms and len(syms) == 10


def test_kill_switch_blocks_orders(env: dict) -> None:
    (env["root"] / "STOP").touch()
    ib = fake(env)
    with pytest.raises(Abort, match="STOP"):
        run(env, ib)
    assert ib.placed == []


def test_bot_run_dispatches_to_the_stock_bot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse

    import stockbot
    from logsetup import RunContext

    calls = []
    monkeypatch.setattr(stockbot, "run_stocks", lambda *a: calls.append(a))
    monkeypatch.setattr(bot, "ROOT", tmp_path)
    conf, cfg = load_config(env_file=None)
    conf["bot"] = {"strategy": "stocks"}
    bot.run(
        argparse.Namespace(check=True, execute=False, live=False),
        RunContext(mode="check", git="t"),
        conf,
        cfg,
    )
    assert len(calls) == 1
