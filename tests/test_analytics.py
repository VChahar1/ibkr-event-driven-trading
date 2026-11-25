"""Metrics and live-log parsing behind report.py."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics import (
    drawdown,
    load_fills,
    load_runs,
    max_drawdown_days,
    monthly_table,
    perf_stats,
    rebalance_turnover,
    regime_table,
    sharpe,
    shortfall,
    signal_parity,
    sleeve_value,
    time_held,
)
from backtest import run_backtest
from strategy import StrategyConfig, target_weights
from tests.conftest import make_prices


def series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_drawdown_and_duration() -> None:
    r = series([0.10, -0.50, 0.20, 0.0])
    np.testing.assert_allclose(drawdown(r).to_numpy(), [0.0, -0.5, -0.4, -0.4])
    assert max_drawdown_days(r) == 3


def test_sharpe_uses_excess_return_volatility() -> None:

    rng = np.random.default_rng(0)
    r = series(list(0.0005 + 0.01 * rng.standard_normal(500)))
    rf = series(list(0.0002 + 0.004 * rng.standard_normal(500)))
    ex = r - rf
    assert sharpe(r, rf) == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252))
    assert sharpe(r, rf) != pytest.approx((ex.mean() / r.std(ddof=1)) * math.sqrt(252), rel=1e-3)


def test_perf_stats_known_values() -> None:
    r = series([0.01, -0.02, 0.015, 0.0] * 63)
    s = perf_stats(r)
    assert s["Years"] == pytest.approx(1.0)
    assert s["CAGR"] == pytest.approx(float((1 + r).prod() - 1))
    assert s["Max drawdown"] == pytest.approx(float(drawdown(r).min()))
    assert s["Positive days"] == pytest.approx(0.5)
    assert s["Calmar"] == pytest.approx(s["CAGR"] / abs(s["Max drawdown"]))


def test_monthly_table_compounds_within_months() -> None:
    idx = pd.bdate_range("2024-01-01", "2025-12-31")
    r = pd.Series(0.001, index=idx)
    t = monthly_table(r)
    jan = r.loc["2024-01"]
    assert t.loc[2024, "Jan"] == pytest.approx(float((1 + jan).prod() - 1))
    assert t.loc[2025, "Year"] == pytest.approx(float((1 + t.loc[2025, "Jan":"Dec"]).prod() - 1))
    assert list(t.columns)[-1] == "Year" and len(t.columns) == 13


def test_regime_table_marks_uncovered_windows() -> None:
    r = series([0.01] * 300, start="2020-01-01")
    t = regime_table(
        {"A": r}, {"early": ("2019-01-01", "2019-06-01"), "inside": ("2020-02-03", "2020-02-07")}
    )
    early, inside = t.set_index("Regime").loc["early"], t.set_index("Regime").loc["inside"]
    assert math.isnan(early["Return"])
    assert inside["Return"] == pytest.approx(1.01**5 - 1)


def test_rebalance_turnover_matches_backtest_bookkeeping() -> None:
    """Independent re-derivation of the backtest's trades from its weights alone."""
    cfg = StrategyConfig()
    prices = make_prices(n_days=700, seed=11)
    _, w, total_to, n_reb = run_backtest(prices, cfg, cost_bps=0.0)
    rets = prices.pct_change().fillna(0.0)
    initial = pd.Series(0.0, index=w.columns)
    initial[cfg.safe] = 1.0
    to = rebalance_turnover(w, rets, initial)
    assert int((to > 1e-9).sum()) == n_reb
    assert float(to.sum()) == pytest.approx(total_to, rel=1e-9)


def test_time_held_uses_dust_threshold() -> None:
    w = pd.DataFrame({"A": [0.5, 0.01, 0.0, 0.5], "B": [0.5, 0.99, 1.0, 0.5]})
    held = time_held(w)
    assert held["A"] == 0.5 and held["B"] == 1.0


def write_log(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def rec(run: str, ts: str, mode: str, event: str | None, data: dict | None = None, msg: str = "") -> dict:
    return {
        "ts": ts,
        "level": "INFO",
        "logger": "dailybot",
        "run_id": run,
        "git": "abc1234",
        "mode": mode,
        "event": event,
        "msg": msg,
        **({"data": data} if data else {}),
    }


LOG = [
    rec("r1", "2026-10-02T09:12:00+00:00", "execute", "start"),
    rec(
        "r1",
        "2026-10-02T09:12:01+00:00",
        "execute",
        "price_check",
        {"sym": "CSPX", "pos": 3, "yahoo": 830.0, "ibkr": 831.0, "gap": 0.0012},
    ),
    rec(
        "r1",
        "2026-10-02T09:12:01+00:00",
        "execute",
        "price_check",
        {"sym": "IB01", "pos": 16, "yahoo": 122.0, "ibkr": 122.0, "gap": 0.0},
    ),
    rec("r1", "2026-10-02T09:12:01+00:00", "execute", "account", {"usd_cash": 5000.0}),
    rec(
        "r1",
        "2026-10-02T09:12:02+00:00",
        "execute",
        "signal",
        {"signal_date": "2026-09-30", "held": ["CSPX"], "target": {"CSPX": 0.5, "EIMI": 0.5}},
    ),
    rec(
        "r1",
        "2026-10-02T09:12:03+00:00",
        "execute",
        "order",
        {"sym": "IB01", "side": "SELL", "qty": 16, "est_usd": 1952.0, "ref": "dailybot-20261002"},
    ),
    rec(
        "r1",
        "2026-10-02T09:12:03+00:00",
        "execute",
        "order",
        {"sym": "CSPX", "side": "BUY", "qty": 2, "est_usd": 1662.0, "ref": "dailybot-20261002"},
    ),
    rec("r2", "2026-10-02T12:00:00+00:00", "dry", "start"),
    rec("r2", "2026-10-02T12:00:30+00:00", "dry", "abort", {"reason": "Gateway unreachable"}),
]


def test_load_runs_summarises_each_run(tmp_path: Path) -> None:
    runs = load_runs(write_log(tmp_path / "bot.jsonl", LOG))
    assert list(runs["run_id"]) == ["r1", "r2"]
    r1, r2 = runs.iloc[0], runs.iloc[1]
    assert r1["outcome"] == "OK" and r1["ref"] == "dailybot-20261002" and len(r1["orders"]) == 2
    assert r1["holdings_usd"] == pytest.approx(3 * 831.0 + 16 * 122.0)
    assert r1["staleness_bdays"] == 2.0
    assert r2["outcome"] == "ABORT" and r2["reason"] == "Gateway unreachable"
    assert load_runs(tmp_path / "missing.jsonl").empty


def test_shortfall_sign_convention(tmp_path: Path) -> None:
    runs = load_runs(write_log(tmp_path / "bot.jsonl", LOG))
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps(
            {
                "ts": "2026-10-02T09:12:05+00:00",
                "ref": "dailybot-20261002",
                "sym": "CSPX",
                "side": "BUY",
                "qty": 2,
                "status": "Filled",
                "filled": 2.0,
                "avg": 838.3,
            }
        )
        + "\n"
        + json.dumps(
            {
                "ts": "2026-10-02T09:12:04+00:00",
                "ref": "dailybot-20261002",
                "sym": "IB01",
                "side": "SELL",
                "qty": 16,
                "status": "Filled",
                "filled": 16.0,
                "avg": 123.22,
            }
        )
        + "\n"
        + json.dumps(
            {
                "ts": "2026-09-29T12:21:26",
                "ref": "dailybot-20260929",
                "sym": "CSPX",
                "side": "BUY",
                "qty": 5,
                "status": "Cancelled",
                "filled": 0.0,
                "avg": 0.0,
            }
        )
        + "\n"
    )
    fills = load_fills(journal)
    assert str(fills["ts"].iloc[2].tz) == "UTC"
    sf = shortfall(fills, runs).set_index("sym")
    assert sf.loc["CSPX", "bps_vs_signal_close"] == pytest.approx(100.0)
    assert sf.loc["IB01", "bps_vs_signal_close"] == pytest.approx(-100.0)
    assert len(sf) == 2


def test_signal_parity_flags_irreproducible_signals() -> None:
    cfg = StrategyConfig()
    prices = make_prices()
    day = prices.index[-1]
    tw = target_weights(prices, cfg)
    good = {k: round(float(v), 3) for k, v in tw.items() if v > 0}
    runs = pd.DataFrame(
        [
            {"start": day, "signal_date": day, "held": [], "target": good},
            {"start": day, "signal_date": prices.index[-2], "held": [], "target": {"IB01": 1.0, "X": 0.0}},
        ]
    )
    parity = signal_parity(runs, prices, cfg)
    assert list(parity["match"]) == [True, False]


def test_sleeve_value_counts_trade_cash_so_rebalancing_is_not_a_loss(tmp_path: Path) -> None:
    """Selling 16 IB01 for cash must leave the sleeve value unchanged at unchanged prices."""
    log = LOG[:5] + [
        rec("r3", "2026-10-02T12:30:00+00:00", "dry", "start"),
        rec(
            "r3",
            "2026-10-02T12:30:01+00:00",
            "dry",
            "price_check",
            {"sym": "CSPX", "pos": 3, "yahoo": 830.0, "ibkr": 831.0, "gap": 0.0012},
        ),
        rec(
            "r3",
            "2026-10-02T12:30:01+00:00",
            "dry",
            "price_check",
            {"sym": "IB01", "pos": 0, "yahoo": 122.0, "ibkr": 122.0, "gap": 0.0},
        ),
    ]
    runs = load_runs(write_log(tmp_path / "bot.jsonl", log))
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps(
            {
                "ts": "2026-10-02T09:12:04+00:00",
                "ref": "dailybot-20261002",
                "sym": "IB01",
                "side": "SELL",
                "qty": 16,
                "status": "Filled",
                "filled": 16.0,
                "avg": 122.0,
            }
        )
        + "\n"
    )
    sv = sleeve_value(runs, load_fills(journal))
    assert list(sv["trade_cash"]) == [0.0, 16 * 122.0]
    assert sv["value"].iloc[1] == pytest.approx(sv["value"].iloc[0])
    assert sv["pnl"].iloc[1] == pytest.approx(0.0)


def test_runs_are_classified_by_strategy(tmp_path: Path) -> None:
    log = LOG + [
        rec("s1", "2026-10-05T14:30:00+00:00", "dry", "start"),
        rec("s1", "2026-10-05T14:30:00+00:00", "dry", "strategy", {"name": "stocks"}),
        rec("s2", "2026-10-05T15:00:00+00:00", "check", "start"),
        rec(
            "s2",
            "2026-10-05T15:00:01+00:00",
            "check",
            "signal",
            {
                "signal_date": "2026-09-30",
                "held": [],
                "market_on": True,
                "picks": ["AAPL"],
                "target": {"AAPL": 0.1},
            },
        ),
    ]
    kinds = load_runs(write_log(tmp_path / "bot.jsonl", log)).set_index("run_id")["kind"]
    assert kinds["r1"] == "etf" and kinds["r2"] == "etf"
    assert kinds["s1"] == "stocks" and kinds["s2"] == "stocks"
