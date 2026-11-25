"""data.py - daily closes from Yahoo Finance (free; used for signals only)."""

from pathlib import Path

import pandas as pd


def load_prices(yahoo_map: dict, period: str = "3y", start: str | None = None) -> pd.DataFrame:
    """yahoo_map: {our_symbol: yahoo_ticker}. Returns adjusted closes,
    columns renamed to our symbols, oldest row first."""
    import yfinance as yf

    tickers = list(yahoo_map.values())
    kw = {"start": start} if start else {"period": period}
    raw = yf.download(tickers, auto_adjust=True, progress=False, **kw)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    missing = [t for t in tickers if t not in raw.columns]
    if missing:
        raise RuntimeError(f"Yahoo returned no data for {missing}")
    df = raw.rename(columns={v: k for k, v in yahoo_map.items()})[list(yahoo_map)]
    return df.dropna(how="all").ffill().dropna()


def load_stock_prices(
    yahoo_map: dict, start: str, cache: Path | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Adjusted closes for many stocks, keeping each one's own history: NaN before a stock
    existed (load_prices would drop every row where any ticker is missing). Short gaps
    inside a stock's life (<= 5 days) are forward-filled. Cached as CSV when `cache` is set."""
    if cache is not None and cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        covers = set(yahoo_map) <= set(df.columns) and df.index[0] <= pd.Timestamp(start) + pd.Timedelta(
            days=7
        )
        if covers:
            print(
                f"Using cached prices {cache.name} (through {df.index[-1].date()}; --refresh to re-download)"
            )
            return df[list(yahoo_map)]
    import yfinance as yf

    tickers = list(yahoo_map.values())
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    missing = [t for t in tickers if t not in raw.columns or raw[t].notna().sum() == 0]
    if missing:
        raise RuntimeError(f"Yahoo returned no data for {missing}")
    df = raw.rename(columns={v: k for k, v in yahoo_map.items()})[list(yahoo_map)]
    df = df.dropna(how="all").ffill(limit=5)
    df = df.where(df.bfill().notna())
    if cache is not None:
        cache.parent.mkdir(exist_ok=True)
        df.to_csv(cache)
    return df


def load_prices_ibkr(
    conf: dict, symbols: list, proxy: bool = False, duration: str = "20 Y", client_id: int = 40
) -> pd.DataFrame:
    """Daily dividend/split-adjusted closes from IBKR via the running IB Gateway.
    proxy=True uses the US-listed twins from [proxy] (SMART/USD)."""
    from ib_async import IB, Stock, util

    from config import is_live_port

    ibc = conf["ib"]
    if is_live_port(ibc["port"]):
        raise RuntimeError(f"Port {ibc['port']} is a LIVE port - refusing to connect.")
    ib = IB()
    ib.connect(ibc["host"], ibc["port"], clientId=client_id, timeout=20)
    frames = {}
    try:
        for s in symbols:
            if proxy:
                c = Stock(conf["proxy"][s], "SMART", "USD")
            else:
                sym, exch, ccy, _ = conf["universe"][s]
                c = Stock(sym, exch, ccy)
            if not ib.qualifyContracts(c):
                raise RuntimeError(f"Could not qualify {s}")
            bars = ib.reqHistoricalData(
                c,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="ADJUSTED_LAST",
                useRTH=True,
                formatDate=1,
                timeout=180,
            )
            if not bars:
                raise RuntimeError(
                    f"IBKR returned no history for {s} ({c.symbol}). "
                    "Usually a market-data permission issue (Error 162/354)."
                )
            df = util.df(bars)
            frames[s] = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
            print(
                f"  {s:<5} ({c.symbol}): {len(df)} bars "
                f"{frames[s].index[0].date()} -> {frames[s].index[-1].date()}"
            )
    finally:
        ib.disconnect()
    return pd.DataFrame(frames).sort_index().ffill().dropna()
