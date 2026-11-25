#!/usr/bin/env python3
"audit/pit_experiment.py - how much does survivorship bias inflate the momentum backtest?"

import datetime as dt
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics import perf_stats  # noqa: E402
from backtest_stocks import DecideFn, evaluation_start, run_monthly  # noqa: E402
from stock_strategy import StockConfig, decide, momentum  # noqa: E402

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HIST = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
CACHE = ROOT / "data_cache" / "audit_sp500_all.csv"
START = "2006-01-01"


def fetch_current() -> set[str]:
    """Today's S&P 500 members (Wikipedia) - the BIASED universe."""
    import requests

    html = requests.get(WIKI, timeout=30, headers={"User-Agent": "ibkr-bot-audit/1.0"}).text
    current = next(
        t for t in pd.read_html(io.StringIO(html)) if "Symbol" in t.columns and "GICS Sector" in t.columns
    )
    return set(current["Symbol"].astype(str).str.strip())


def fetch_history() -> pd.DataFrame:
    """Full member list at every change date since 1996 (github.com/fja05680/sp500)."""
    path = ROOT / "data_cache" / "sp500_hist.csv"
    if not path.exists():
        import requests

        path.write_bytes(requests.get(HIST, timeout=60).content)
    hist = pd.read_csv(path)
    hist["date"] = pd.to_datetime(hist["date"])
    hist["members"] = hist["tickers"].str.split(",").map(lambda xs: {x.strip() for x in xs})
    return hist.sort_values("date").reset_index(drop=True)


def membership(hist: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[pd.Timestamp, set[str]]:
    """Members on each date = the latest published list on or before it."""
    pos = hist["date"].searchsorted(dates, side="right") - 1
    return {d: hist["members"].iloc[i] for d, i in zip(dates, pos, strict=True)}


def download(tickers: list[str]) -> pd.DataFrame:
    miss_file = CACHE.with_suffix(".missing.json")
    if CACHE.exists() and miss_file.exists():
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        if set(tickers) <= set(df.columns) | set(json.loads(miss_file.read_text())):
            return df
    import yfinance as yf

    frames = []
    for k in range(0, len(tickers), 100):
        batch = tickers[k : k + 100]
        raw = yf.download(
            [t.replace(".", "-") for t in batch], start=START, auto_adjust=True, progress=False, threads=True
        )["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(batch[0].replace(".", "-"))
        raw.columns = [c.replace("-", ".") for c in raw.columns]
        frames.append(raw)
        print(f"  downloaded {min(k + 100, len(tickers))}/{len(tickers)}", flush=True)
    df = pd.concat(frames, axis=1)
    df = df.loc[:, df.notna().any()].dropna(how="all").ffill(limit=5)
    missing = sorted(set(tickers) - set(df.columns))
    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE)
    CACHE.with_suffix(".missing.json").write_text(json.dumps(missing))
    return df


def main() -> None:
    t0 = time.time()
    cur = fetch_current()
    hist = fetch_history()
    since = hist[hist["date"] >= pd.Timestamp(START) - pd.Timedelta(days=400)]
    ever = sorted(cur | set().union(*since["members"]))
    print(
        f"Today's members (Wikipedia): {len(cur)}; membership history {hist['date'].min().date()} -> "
        f"{hist['date'].max().date()} ({len(since)} lists since {START}); {len(ever)} tickers ever in the index"
    )
    prices = download(ever + ["SPY", "BIL"])
    cash = prices["BIL"] / prices["BIL"].shift(1) - 1
    px = prices.drop(columns=["BIL"])
    have = [t for t in ever if t in px.columns]
    missing = sorted(set(ever) - set(have))
    print(f"Yahoo history for {len(have)}/{len(ever)} tickers; missing {len(missing)} (mostly delisted)")

    ends = pd.DatetimeIndex(list(px.index[:-1][px.index[1:].to_period("M") != px.index[:-1].to_period("M")]))
    mem = membership(hist, ends)

    cov = pd.Series({d: np.mean([t in px.columns and pd.notna(px.at[d, t]) for t in mem[d]]) for d in ends})
    print(
        f"Point-in-time members with a Yahoo price at month-end: mean {cov.mean():.1%}, "
        f"min {cov.min():.1%} ({cov.idxmin().date()}), latest {cov.iloc[-1]:.1%}"
    )

    biased_cfg = StockConfig(universe=tuple(sorted(cur & set(px.columns))))
    pit_cfg = StockConfig(universe=tuple(have))

    def pit_fn(hist: pd.DataFrame, held: set, today: dt.date) -> tuple[pd.Series, dict]:
        s = hist.index[-1]
        cols = [c for c in hist.columns if c in mem[s]] + ["SPY"]
        d = decide(hist[cols], pit_cfg, held & mem[s], today)
        return d.weights, {"market_on": d.market_on, "picks": d.picks, "n_members_priced": len(d.ranks)}

    def biased_fn(hist: pd.DataFrame, held: set, today: dt.date) -> tuple[pd.Series, dict]:
        d = decide(hist, biased_cfg, held, today)
        return d.weights, {"market_on": d.market_on, "picks": d.picks, "n_members_priced": len(d.ranks)}

    def ew_fn(cfg: StockConfig, pit: bool) -> DecideFn:
        def fn(hist: pd.DataFrame, held: set, today: dt.date) -> tuple[pd.Series, dict]:
            s = hist.index[-1]
            cols = [c for c in cfg.universe if c in hist.columns and (not pit or c in mem[s])]
            mom = momentum(hist[cols], cfg).iloc[-1]
            ok = mom.notna() & hist[cols].iloc[-1].notna()
            w = pd.Series(0.0, index=cols)
            if ok.any():
                w[ok] = 1.0 / int(ok.sum())
            return w, {}

        return fn

    start = max(evaluation_start(px, cash, biased_cfg), evaluation_start(px, cash, pit_cfg))
    rf = cash.loc[start:].fillna(0.0)
    res = {}
    for name, fn, cfg in (
        ("momentum_biased", biased_fn, biased_cfg),
        ("momentum_pit", pit_fn, pit_cfg),
        ("equal_weight_biased", ew_fn(biased_cfg, False), biased_cfg),
        ("equal_weight_pit", ew_fn(pit_cfg, True), pit_cfg),
    ):
        bt = run_monthly(
            px, cash, fn, list(cfg.universe), start, min_trade=0.0 if name.startswith("equal") else 100.0
        )
        st = perf_stats(bt.returns, rf.reindex(bt.returns.index).fillna(0.0))
        res[name] = {
            k: round(float(st[k]), 4) for k in ("CAGR", "Volatility", "Sharpe", "Max drawdown", "Calmar")
        }
        res[name]["final_nav"] = round(float(bt.nav.iloc[-1]), 0)
        if "picks" in bt.decisions.columns:
            picks = bt.decisions["picks"].explode().dropna()
            res[name]["most_picked"] = picks.value_counts().head(8).to_dict()
        print(f"{name:22s} {res[name]}", flush=True)
    spy = px["SPY"].pct_change().loc[start:].fillna(0.0)
    st = perf_stats(spy, rf)
    res["spy"] = {
        k: round(float(st[k]), 4) for k in ("CAGR", "Volatility", "Sharpe", "Max drawdown", "Calmar")
    }
    print(f"{'spy':22s} {res['spy']}")
    out = {
        "evaluation_start": str(start.date()),
        "evaluation_end": str(px.index[-1].date()),
        "current_members": len(cur),
        "membership_lists_since_start": len(since),
        "membership_history": [str(hist["date"].min().date()), str(hist["date"].max().date())],
        "tickers_ever": len(ever),
        "tickers_with_yahoo_history": len(have),
        "missing_tickers": missing,
        "pit_price_coverage_mean": round(float(cov.mean()), 4),
        "pit_price_coverage_min": round(float(cov.min()), 4),
        "results": res,
    }
    (ROOT / "audit" / "pit_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote audit/pit_results.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
