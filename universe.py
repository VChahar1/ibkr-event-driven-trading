#!/usr/bin/env python3
"universe.py - the stock bot's universe: a dated snapshot of the S&P 100."

import argparse
import datetime as dt
import io
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
UNIVERSE_FILE = ROOT / "universe" / "sp100.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"
COLUMNS = ["symbol", "yahoo", "ib_symbol", "name", "sector", "primary_exchange", "con_id", "as_of"]


def tickers(symbol: str) -> tuple[str, str]:
    """Index symbol -> (Yahoo ticker, IBKR symbol). Share classes: BRK.B -> BRK-B / 'BRK B'."""
    return symbol.replace(".", "-"), symbol.replace(".", " ")


def parse_constituents(html: str) -> pd.DataFrame:
    """The constituents table: the one with Symbol, Name/Company and Sector columns."""
    for table in pd.read_html(io.StringIO(html)):
        cols = {str(c).strip().lower(): c for c in table.columns}
        name_col = cols.get("name") or cols.get("company")
        if "symbol" in cols and "sector" in cols and name_col is not None:
            out = table[[cols["symbol"], name_col, cols["sector"]]].copy()
            out.columns = ["symbol", "name", "sector"]
            out["symbol"] = out["symbol"].astype(str).str.strip()
            return out
    raise ValueError("No constituents table (Symbol/Name/Sector) found on the page")


def one_class_per_company(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one share class per company (e.g. GOOGL, not also GOOG): two classes of the same
    company would double one bet. Within a company the alphabetically last symbol is kept,
    which is the class-A line for GOOG/GOOGL, NWS/NWSA, FOX/FOXA."""
    stem = df["name"].str.replace(r"\s*\(?class [a-z]\)?\s*$", "", regex=True, flags=re.I).str.strip()
    keep = df.assign(_stem=stem).sort_values("symbol").groupby("_stem", sort=False).tail(1)
    return keep.drop(columns="_stem").sort_values("symbol").reset_index(drop=True)


def build_universe(raw: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
    df = one_class_per_company(raw)
    mapped = df["symbol"].map(tickers)
    df["yahoo"] = [y for y, _ in mapped]
    df["ib_symbol"] = [i for _, i in mapped]
    df["primary_exchange"] = ""
    df["con_id"] = 0
    df["as_of"] = as_of.isoformat()
    return df[COLUMNS]


def verify_with_ibkr(df: pd.DataFrame, conf: dict, client_id: int = 42) -> pd.DataFrame:
    """Read-only: qualify each symbol (SMART/USD) and store its primary exchange and conId."""
    from ib_async import IB, Contract, Stock

    from config import is_live_port

    if is_live_port(conf["ib"]["port"]):
        raise SystemExit("Refusing a live port.")
    ib = IB()
    ib.connect(conf["ib"]["host"], conf["ib"]["port"], clientId=client_id, timeout=20, readonly=True)
    try:
        out = df.copy()
        for i, row in out.iterrows():
            q = ib.qualifyContracts(Stock(row["ib_symbol"], "SMART", "USD"))
            c = q[0] if q else None
            if isinstance(c, Contract) and c.conId:
                out.at[i, "primary_exchange"] = c.primaryExchange
                out.at[i, "con_id"] = int(c.conId)
            else:
                print(f"  could not qualify {row['symbol']} ({row['ib_symbol']})")
        return out
    finally:
        ib.disconnect()


def load_universe(path: Path = UNIVERSE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"primary_exchange": str}, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-fetch the list from Wikipedia")
    ap.add_argument("--no-ibkr", action="store_true", help="skip the IBKR contract check")
    args = ap.parse_args()
    if not args.refresh:
        print(load_universe().to_string(index=False))
        return
    import requests

    html = requests.get(WIKI_URL, timeout=30, headers={"User-Agent": "ibkr-bot-universe/1.0"}).text
    df = build_universe(parse_constituents(html), dt.date.today())
    if not args.no_ibkr:
        from config import load_config

        try:
            df = verify_with_ibkr(df, load_config()[0])
        except OSError as e:
            print(f"IBKR check skipped ({e!r}); primary exchange left blank")
    UNIVERSE_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(UNIVERSE_FILE, index=False)
    print(
        f"Wrote {UNIVERSE_FILE.relative_to(ROOT)}: {len(df)} stocks, "
        f"{(df['con_id'] > 0).sum()} verified with IBKR, as of {df['as_of'].iloc[0]}"
    )


if __name__ == "__main__":
    main()
