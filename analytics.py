"analytics.py - performance metrics and live-log parsing (pure: no plotting, no network)."

import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategy import HELD_MIN_WEIGHT, StrategyConfig, target_weights

TRADING_DAYS = 252
WARSAW = ZoneInfo("Europe/Warsaw")


REGIMES: dict[str, tuple[str, str]] = {
    "GFC: Lehman to trough": ("2008-09-01", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
}


def equity(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    eq = equity(r)
    return eq / eq.cummax() - 1


def max_drawdown_days(r: pd.Series) -> int:
    """Longest stretch (trading days) spent below a previous equity high."""
    under = (drawdown(r) < 0).to_numpy()
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return longest


def excess(r: pd.Series, rf: pd.Series | None) -> pd.Series:
    return r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)


def sharpe(r: pd.Series, rf: pd.Series | None = None) -> float:
    """Annualised Sharpe of returns in excess of rf (excess std in the denominator)."""
    ex = excess(r, rf)
    sd = float(ex.std(ddof=1))
    return float(ex.mean()) / sd * math.sqrt(TRADING_DAYS) if sd > 0 else float("nan")


def sharpe_se(sr_annual: float, years: float) -> float:
    """Lo (2002) iid standard error of an annualised Sharpe ratio."""
    if years <= 0 or not math.isfinite(sr_annual):
        return float("nan")
    sr_daily = sr_annual / math.sqrt(TRADING_DAYS)
    return math.sqrt((1 + 0.5 * sr_daily**2) / years)


def perf_stats(r: pd.Series, rf: pd.Series | None = None) -> dict[str, float]:
    r = r.dropna()
    years = len(r) / TRADING_DAYS
    eq = equity(r)
    cagr = float(eq.iloc[-1]) ** (1 / years) - 1 if years > 0 else float("nan")
    ex = excess(r, rf)
    downside = math.sqrt(float((ex.clip(upper=0) ** 2).mean()))
    mdd = float(drawdown(r).min())
    sr = sharpe(r, rf)
    return {
        "CAGR": cagr,
        "Volatility": float(r.std(ddof=1)) * math.sqrt(TRADING_DAYS),
        "Sharpe": sr,
        "Sharpe SE": sharpe_se(sr, years),
        "Sortino": float(ex.mean()) / downside * math.sqrt(TRADING_DAYS) if downside > 0 else float("nan"),
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "Longest drawdown (days)": float(max_drawdown_days(r)),
        "Positive days": float((r > 0).mean()),
        "Skew": float(r.skew()),
        "Excess kurtosis": float(r.kurt()),
        "Best day": float(r.max()),
        "Worst day": float(r.min()),
        "Years": years,
    }


def rolling_sharpe(r: pd.Series, rf: pd.Series | None, window: int = TRADING_DAYS) -> pd.Series:
    ex = excess(r, rf)
    return ex.rolling(window).mean() / ex.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def rolling_vol(r: pd.Series, window: int = 63) -> pd.Series:
    return r.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def period_returns(r: pd.Series, freq: str) -> pd.Series:
    """Compounded returns per period ("ME" month, "YE" year)."""
    return (1 + r).resample(freq).prod() - 1


def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Years x months matrix of compounded monthly returns, plus a full-year column."""
    m = period_returns(r, "ME")
    table = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.to_numpy()})
    out = table.pivot(index="year", columns="month", values="ret")
    out = out.reindex(columns=range(1, 13))
    out.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    yearly = period_returns(r, "YE")
    out["Year"] = pd.Series(yearly.to_numpy(), index=yearly.index.year)
    return out


def regime_table(series: dict[str, pd.Series], regimes: dict[str, tuple[str, str]] = REGIMES) -> pd.DataFrame:
    """Total return and max drawdown of each series inside each stress window.
    NaN where the window starts before the series has data."""
    rows = []
    for name, (start, end) in regimes.items():
        for label, r in series.items():
            covered = len(r) > 0 and r.index[0] <= pd.Timestamp(start)
            win = r.loc[start:end] if covered else r.iloc[0:0]
            rows.append(
                {
                    "Regime": name,
                    "Window": f"{start} to {end}",
                    "Series": label,
                    "Return": float(equity(win).iloc[-1] - 1) if len(win) else float("nan"),
                    "Max drawdown": float(drawdown(win).min()) if len(win) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def rebalance_turnover(
    weights: pd.DataFrame, rets: pd.DataFrame, initial: pd.Series | None = None
) -> pd.Series:
    "Turnover (sum of |weight change|) traded each day, inferred from end-of-day weights."
    w = weights.to_numpy(dtype=float)
    r = rets.reindex(index=weights.index, columns=weights.columns).fillna(0.0).to_numpy(dtype=float)
    prev = np.vstack(
        [
            initial.reindex(weights.columns).fillna(0.0).to_numpy(dtype=float)
            if initial is not None
            else w[0],
            w[:-1],
        ]
    )
    port = (prev * r).sum(axis=1)
    drift = prev * (1 + r) / (1 + port)[:, None]
    return pd.Series(np.abs(w - drift).sum(axis=1), index=weights.index)


def time_held(weights: pd.DataFrame) -> pd.Series:
    """Fraction of days each asset was held (weight above the dust threshold)."""
    return (weights > HELD_MIN_WEIGHT).mean()


def _ts(value: str) -> pd.Timestamp:
    """Journal timestamps before 2026-10-02 are naive local (Warsaw) times."""
    t = pd.Timestamp(value)
    return t.tz_localize(WARSAW).tz_convert("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_runs(path: Path) -> pd.DataFrame:
    """One row per bot run from logs/bot.jsonl (records carry run_id + event)."""
    if not path.exists():
        return pd.DataFrame()
    runs: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        run = runs.setdefault(
            rec["run_id"],
            {
                "run_id": rec["run_id"],
                "start": _ts(rec["ts"]),
                "git": rec.get("git"),
                "requested": rec.get("mode"),
                "mode": rec.get("mode"),
                "outcome": "OK",
                "reason": "",
                "signal_date": None,
                "target": {},
                "held": [],
                "usd_cash": float("nan"),
                "positions": {},
                "ibkr": {},
                "yahoo": {},
                "gap": {},
                "orders": [],
                "fills": [],
                "ref": None,
                "skipped": False,
                "kind": None,
            },
        )
        run["mode"] = rec.get("mode", run["mode"])
        event, data = rec.get("event"), rec.get("data") or {}
        if event == "start":
            run["requested"] = rec.get("mode")
        elif event == "strategy":
            run["kind"] = data.get("name")
        elif event in ("hold", "session", "unaffordable") and run["kind"] is None:
            run["kind"] = "stocks"
        elif event == "signal":
            if run["kind"] is None:
                run["kind"] = "stocks" if "market_on" in data else "etf"
            run["signal_date"] = pd.Timestamp(data["signal_date"])
            run["target"], run["held"] = data.get("target", {}), data.get("held", [])
        elif event == "account":
            run["usd_cash"] = float(data["usd_cash"])
        elif event == "price_check":
            s = data["sym"]
            run["positions"][s] = int(data["pos"])
            run["ibkr"][s], run["yahoo"][s] = float(data["ibkr"]), float(data["yahoo"])
            run["gap"][s] = float(data["gap"]) if data.get("gap") is not None else float("nan")
        elif event == "order":
            run["orders"].append(data)
            run["ref"] = data.get("ref")
        elif event == "fill":
            run["fills"].append(data)
        elif event == "abort":
            run["outcome"], run["reason"] = "ABORT", data.get("reason", rec.get("msg", ""))
        elif event == "crash":
            run["outcome"], run["reason"] = "CRASH", rec.get("msg", "")
        elif event == "skip":
            run["outcome"], run["reason"], run["skipped"] = "SKIP", data.get("reason", ""), True
    for run in runs.values():
        run["kind"] = run["kind"] or "etf"
    df = pd.DataFrame(list(runs.values()))
    df["holdings_usd"] = [
        sum(p * run["ibkr"].get(s, float("nan")) for s, p in run["positions"].items())
        if run["positions"]
        else float("nan")
        for run in runs.values()
    ]
    df["staleness_bdays"] = [
        float(np.busday_count(run["signal_date"].date(), run["start"].tz_convert(WARSAW).date()))
        if pd.notna(run["signal_date"])
        else float("nan")
        for run in runs.values()
    ]
    return df.sort_values("start").reset_index(drop=True)


def load_fills(path: Path) -> pd.DataFrame:
    """journal.jsonl -> one row per order outcome (filled or not)."""
    if not path.exists():
        return pd.DataFrame(columns=["ts", "ref", "sym", "side", "qty", "status", "filled", "avg"])
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    df["ts"] = [_ts(t) for t in df["ts"]]
    return df


def shortfall(fills: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    "Implementation shortfall of each filled order, in bps (positive = cost)."
    cols = [
        "ts",
        "ref",
        "sym",
        "side",
        "filled",
        "avg",
        "decision_px",
        "signal_close",
        "bps_vs_decision",
        "bps_vs_signal_close",
        "usd_vs_signal_close",
    ]
    if fills.empty or runs.empty:
        return pd.DataFrame(columns=cols)
    placing = runs[runs["ref"].notna()].set_index("ref")
    rows = []
    for f in fills[fills["filled"] > 0].itertuples():
        if f.ref not in placing.index:
            continue
        run = placing.loc[f.ref]
        run = run.iloc[-1] if isinstance(run, pd.DataFrame) else run
        dec, close = run["ibkr"].get(f.sym), run["yahoo"].get(f.sym)
        if dec is None or close is None:
            continue
        sign = 1.0 if f.side == "BUY" else -1.0
        rows.append(
            {
                "ts": f.ts,
                "ref": f.ref,
                "sym": f.sym,
                "side": f.side,
                "filled": f.filled,
                "avg": f.avg,
                "decision_px": dec,
                "signal_close": close,
                "bps_vs_decision": sign * (f.avg / dec - 1) * 1e4,
                "bps_vs_signal_close": sign * (f.avg / close - 1) * 1e4,
                "usd_vs_signal_close": sign * (f.avg - close) * f.filled,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def sleeve_value(runs: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    "Mark-to-market value of the bot's sleeve at each run that saw positions."
    cols = ["start", "holdings", "trade_cash", "value", "pnl"]
    snaps = runs[runs["positions"].map(bool)] if not runs.empty else runs
    if snaps.empty:
        return pd.DataFrame(columns=cols)
    t0 = snaps["start"].iloc[0]
    flows = fills[(fills["filled"] > 0) & (fills["ts"] >= t0)] if not fills.empty else fills
    cash = (
        pd.Series(
            [
                (1.0 if s == "SELL" else -1.0) * a * q
                for s, a, q in zip(flows["side"], flows["avg"], flows["filled"], strict=True)
            ],
            index=flows.index,
            dtype=float,
        )
        if len(flows)
        else pd.Series(dtype=float)
    )
    rows = []
    for run in snaps.itertuples():
        tc = float(cash[flows["ts"] < run.start].sum()) if len(flows) else 0.0
        rows.append(
            {
                "start": run.start,
                "holdings": run.holdings_usd,
                "trade_cash": tc,
                "value": run.holdings_usd + tc,
            }
        )
    out = pd.DataFrame(rows, columns=cols[:-1])
    out["pnl"] = out["value"] - out["value"].iloc[0]
    return out


def signal_parity(runs: pd.DataFrame, prices: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    "Recompute each logged signal from today's copy of the price history."
    cols = ["start", "signal_date", "logged", "recomputed", "match"]
    if runs.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    done: set[tuple] = set()
    for run in runs[runs["signal_date"].notna()].itertuples():
        key = (run.signal_date, tuple(sorted(run.held)))
        if key in done:
            continue
        done.add(key)
        hist = prices.loc[: run.signal_date]
        if len(hist) < cfg.warmup or hist.index[-1] != run.signal_date:
            continue
        tw = target_weights(hist, cfg, frozenset(run.held))
        recomputed = {k: round(float(v), 3) for k, v in tw.items() if v > 0}
        logged = {k: round(float(v), 3) for k, v in run.target.items()}
        rows.append(
            {
                "start": run.start,
                "signal_date": run.signal_date,
                "logged": logged,
                "recomputed": recomputed,
                "match": logged == recomputed,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def stock_signal_parity(runs: pd.DataFrame, prices: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Stock-bot version of signal_parity: re-run stock_strategy.decide() for each logged
    month-end signal (same held set, same run date) and compare the picks."""
    from stock_strategy import decide

    cols = ["start", "signal_date", "logged", "recomputed", "match"]
    if runs.empty:
        return pd.DataFrame(columns=cols)
    rows, done = [], set()
    for run in runs[runs["signal_date"].notna()].itertuples():
        held = tuple(sorted(s for s in run.held if s in cfg.universe))
        key = (run.signal_date, held)
        if key in done or run.signal_date not in prices.index:
            continue
        done.add(key)
        today = run.start.tz_convert("America/New_York").date()
        d = decide(prices.loc[: run.signal_date], cfg, held, today)
        logged, recomputed = sorted(run.target), sorted(d.picks)
        rows.append(
            {
                "start": run.start,
                "signal_date": run.signal_date,
                "logged": logged,
                "recomputed": recomputed,
                "match": logged == recomputed,
            }
        )
    return pd.DataFrame(rows, columns=cols)
