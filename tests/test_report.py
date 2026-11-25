"""report.py builds a self-contained, secret-free page from synthetic inputs (no network)."""

import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from analytics import load_fills, load_runs
from config import load_config
from report import bar_gap, build_report, clean, log_ticks, short_labels
from tests.conftest import make_prices
from tests.test_analytics import LOG, rec, write_log

FAKE_ACCT = "DU" + "7777777"
META = {"Code": "abc1234", "Generated": "now", "live": "on"}


def figures(page: str) -> list[dict]:
    m = re.search(r'<script type="application/json" id="figs">(.*?)</script>', page, re.S)
    assert m, "figure data block missing"
    return json.loads(m.group(1))


def build(tmp_path: Path, live: bool = True) -> str:
    conf, cfg = load_config(env_file=None)
    prices = make_prices(n_days=800, seed=3)
    log = LOG + [
        rec("r9", "2026-10-02T13:00:00+00:00", "dry", "start"),
        rec(
            "r9", "2026-10-02T13:00:05+00:00", "dry", "abort", {"reason": f"Account {FAKE_ACCT} not managed"}
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
                "avg": 122.1,
            }
        )
        + "\n"
    )
    meta = META | {"live": "on" if live else "off"}
    return build_report(
        prices, cfg, conf, 10.0, "Synthetic", runs, load_fills(journal), prices if live else None, meta
    )


@pytest.fixture(scope="module")
def page(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Built once per module: the backtest inside is the slow part."""
    return build(tmp_path_factory.mktemp("report"))


def test_full_report_has_all_sections_and_valid_figure_json(page: str) -> None:
    for sid in ("backtest", "live", "notes"):
        assert f'<section id="{sid}">' in page
    figs = figures(page)
    assert len(figs) >= 20
    assert all(f["id"] and isinstance(f["data"], list) for f in figs)
    assert "NaN" not in page.split('id="figs">')[1]
    assert page.count('<div class="plot"') == len(figs)


def test_report_never_contains_account_ids(page: str) -> None:
    assert FAKE_ACCT not in page


def test_backtest_only_mode_skips_live_section(tmp_path: Path) -> None:
    page = build(tmp_path, live=False)
    assert '<section id="live">' not in page and '<section id="backtest">' in page


def test_preregistered_strategies_are_not_plotted(page: str) -> None:
    names = {t.get("name", "") for f in figures(page) for t in f["data"]}
    assert not any(n.startswith(("H1", "H2")) or "vol-target" in n.lower() for n in names)


def test_helpers() -> None:
    assert 0.25 <= bar_gap(3, 4, wide=True) <= 0.85
    assert bar_gap(40, 1) == 0.25
    times = pd.Series(
        pd.to_datetime(["2026-10-02 09:43:30", "2026-10-02 09:43:46", "2026-10-02 10:00:00"], utc=True)
    )
    labels = short_labels(times)
    assert len(set(labels)) == 3 and labels[2] == "10-02 12:00"
    ticks = log_ticks(pd.DataFrame({"a": [0.6, 7.5]}))
    assert ticks[0] >= 0.5 and ticks[-1] <= 8.25 and 1 in ticks
    assert clean({"x": [math.nan, 1.5], "t": pd.Timestamp("2026-10-02")}) == {
        "x": [None, 1.5],
        "t": "2026-10-02T00:00:00",
    }


def test_figure_colors_use_theme_tokens_not_raw_hex(page: str) -> None:
    """Dark mode works by resolving @tokens in the browser; a raw hex would not switch."""
    blob = json.dumps(figures(page))
    assert not re.search(r'"color": "#[0-9a-fA-F]{3,6}"', blob)


@pytest.mark.parametrize("name", ["report.css", "report.js"])
def test_templates_exist(name: str) -> None:
    assert (Path(__file__).parent.parent / "templates" / name).stat().st_size > 500


@pytest.fixture(scope="module")
def stock_page(tmp_path_factory: pytest.TempPathFactory) -> str:
    from backtest_stocks import run_all
    from report import build_stock_report, stock_cost_sensitivity
    from stock_strategy import StockConfig
    from tests.conftest import make_stock_prices

    p = make_stock_prices(market_drift=1e-3)
    cash = pd.Series(0.0001, index=p.index)
    syms = [f"S{i:02d}" for i in range(1, 31)]
    cfg = StockConfig(universe=tuple(syms), min_eligible=25)
    uni = pd.DataFrame(
        {
            "symbol": syms,
            "yahoo": syms,
            "ib_symbol": syms,
            "name": syms,
            "sector": ["Tech", "Health", "Energy"] * 10,
            "primary_exchange": "NYSE",
            "con_id": 0,
            "as_of": "2026-10-03",
        }
    )
    res = run_all(p, cash, cfg)
    cost = stock_cost_sensitivity(p, cash, cfg, res["strategy"])
    tmp = tmp_path_factory.mktemp("stock")
    runs = load_runs(write_log(tmp / "bot.jsonl", LOG))
    conf, _ = load_config(env_file=None)
    return build_stock_report(res, p, cfg, uni, conf, runs, pd.DataFrame(), cost, META)


def test_stock_report_sections_banner_and_figures(stock_page: str) -> None:
    for sid in ("backtest", "live", "notes"):
        assert f'<section id="{sid}">' in stock_page
    assert "Read first: survivorship bias" in stock_page
    figs = figures(stock_page)
    assert len(figs) >= 18
    blob = json.dumps(figs)
    assert not re.search(r'"color": "#[0-9a-fA-F]{3,6}"', blob)
    assert FAKE_ACCT not in stock_page
    names = {t.get("name", "") for f in figs for t in f["data"]}
    assert {"Momentum", "Equal-weight", "SPY"} <= names
    assert not any(n.startswith(("H1", "H2")) for n in names)
