"""Universe snapshot: parsing, ticker mapping, one share class per company."""

import datetime as dt
from pathlib import Path

import pytest

from universe import COLUMNS, UNIVERSE_FILE, build_universe, load_universe, parse_constituents, tickers

HTML = """
<table><tr><th>Index</th><th>Value</th></tr><tr><td>S&P 100</td><td>1</td></tr></table>
<table>
  <tr><th>Symbol</th><th>Name</th><th>Sector</th></tr>
  <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
  <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
  <tr><td>GOOG</td><td>Alphabet Inc. (Class C)</td><td>Communication Services</td></tr>
  <tr><td>GOOGL</td><td>Alphabet Inc. (Class A)</td><td>Communication Services</td></tr>
  <tr><td>JPM</td><td>JPMorgan Chase</td><td>Financials</td></tr>
</table>
"""


def test_ticker_mapping_for_share_classes() -> None:
    assert tickers("BRK.B") == ("BRK-B", "BRK B")
    assert tickers("AAPL") == ("AAPL", "AAPL")


def test_parse_finds_the_constituents_table() -> None:
    df = parse_constituents(HTML)
    assert list(df.columns) == ["symbol", "name", "sector"] and len(df) == 5


def test_build_keeps_one_class_per_company() -> None:
    u = build_universe(parse_constituents(HTML), dt.date(2026, 10, 3))
    assert list(u.columns) == COLUMNS
    assert list(u["symbol"]) == ["AAPL", "BRK.B", "GOOGL", "JPM"]
    brk = u.set_index("symbol").loc["BRK.B"]
    assert brk["yahoo"] == "BRK-B" and brk["ib_symbol"] == "BRK B"
    assert set(u["as_of"]) == {"2026-10-03"}


def test_parse_rejects_pages_without_a_constituents_table() -> None:
    with pytest.raises(ValueError, match="No constituents table"):
        parse_constituents("<table><tr><th>A</th></tr><tr><td>1</td></tr></table>")


def test_load_universe_validates_columns(tmp_path: Path) -> None:
    bad = tmp_path / "u.csv"
    bad.write_text("symbol,name\nAAPL,Apple\n")
    with pytest.raises(ValueError, match="lacks columns"):
        load_universe(bad)


@pytest.mark.skipif(not UNIVERSE_FILE.exists(), reason="no committed snapshot yet")
def test_committed_snapshot_is_sane() -> None:
    u = load_universe()
    assert 90 <= len(u) <= 105 and u["symbol"].is_unique and u["yahoo"].is_unique
    assert not {"GOOG", "GOOGL"} <= set(u["symbol"])
    assert u["as_of"].nunique() == 1
