#!/usr/bin/env python3
"report.py - interactive HTML report: backtest diagnostics + live (paper) results."

import argparse
import datetime as dt
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analytics import (
    REGIMES,
    WARSAW,
    load_fills,
    load_runs,
    monthly_table,
    perf_stats,
    period_returns,
    rebalance_turnover,
    regime_table,
    rolling_sharpe,
    rolling_vol,
    sharpe,
    shortfall,
    signal_parity,
    sleeve_value,
    stock_signal_parity,
    time_held,
)
from analytics import drawdown as dd_series
from analytics import equity as eq_series
from backtest import benchmark_returns, load_backtest_prices, run_backtest
from backtest_stocks import CAPITAL, DEFAULT_COSTS, CostModel, run_all, run_monthly, strategy_fn
from bot import budget_usd, current_weights
from config import load_config
from logsetup import code_version, redact
from stock_strategy import StockConfig
from strategy import StrategyConfig

ROOT = Path(__file__).parent
TEMPLATES = ROOT / "templates"
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"
COST_GRID = (0.0, 5.0, 10.0, 20.0, 50.0)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


STRATEGY_SLOT = {
    "Dual momentum": "series-1",
    "60/40": "series-2",
    "Equal-weight": "series-3",
    "Buy & hold equities": "series-4",
}
OUTCOME_STYLE = {
    "OK": ("good", "circle"),
    "SKIP": ("muted", "diamond"),
    "ABORT": ("serious", "triangle-up"),
    "CRASH": ("critical", "x"),
}


def clean(obj: Any) -> Any:
    """JSON-safe: NaN/inf -> null, floats rounded, numpy/pandas scalars -> Python, timestamps -> ISO."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, pd.Timestamp | dt.date):
        return obj.isoformat()
    if hasattr(obj, "item") and not isinstance(obj, str):
        obj = obj.item()
    if isinstance(obj, float):
        return float(f"{obj:.6g}") if math.isfinite(obj) else None
    return obj


def dates(index: pd.Index) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in index]


def pct(x: float, digits: int = 1) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{x:,.{digits}f}"


def esc(x: Any) -> str:
    return html.escape(str(x))


def table_html(
    df: pd.DataFrame, fmt: dict[str, Any] | None = None, index: bool = True, max_rows: int = 400
) -> str:
    """Escaped HTML table; `fmt` maps column -> formatter. Numbers right-aligned (CSS)."""
    fmt = fmt or {}
    df = df.tail(max_rows)
    head = ("<th></th>" if index else "") + "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for idx, row in df.iterrows():
        cells = f"<th>{esc(idx)}</th>" if index else ""
        for c in df.columns:
            v = row[c]
            text = fmt[c](v) if c in fmt else (num(v) if isinstance(v, float) else str(v))
            cells += f"<td>{esc(text)}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.resample("ME").last()
    out.index = [d.strftime("%Y-%m") for d in out.index]
    return out


def line(
    x: list[str], y: pd.Series | list, name: str, slot: str, fmt: str = ".2f", mode: str = "lines"
) -> dict:
    return {
        "type": "scatter",
        "mode": mode,
        "x": x,
        "y": list(y),
        "name": name,
        "line": {"color": f"@{slot}", "width": 2},
        "marker": {"color": f"@{slot}", "size": 9, "line": {"color": "@surface-1", "width": 2}},
        "hovertemplate": f"%{{y:{fmt}}}<extra>{esc(name)}</extra>",
    }


def short_labels(times: pd.Series) -> list[str]:
    """Compact run labels for category axes; seconds only where two runs share a minute."""
    local = [t.tz_convert(WARSAW) for t in times]
    minutes = [f"{t:%m-%d %H:%M}" for t in local]
    return [f"{t:%m-%d %H:%M:%S}" if minutes.count(m) > 1 else m for t, m in zip(local, minutes, strict=True)]


def log_ticks(df: pd.DataFrame) -> list[float]:
    """Readable ticks for a log growth axis (Plotly's own log ticks crowd or drop the decade):
    dense 1-1.5-2-3-4-5-6-8 within ~1.5 decades, 1-2-5 per decade beyond that."""
    lo, hi = float(df.min().min()), float(df.max().max())
    span = math.log10(hi / lo)
    mantissas = (1, 1.5, 2, 3, 4, 5, 6, 8) if span <= 1.5 else (1, 2, 5)
    k0, k1 = math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1
    steps = sorted(m * 10.0**k for k in range(k0, k1) for m in mantissas)
    return [v for v in steps if lo * 0.9 <= v <= hi * 1.1]


def bar_gap(
    n_categories: int, n_series: int, wide: bool = False, plot_px: float | None = None, max_bar_px: float = 24
) -> float:
    """bargap that keeps each bar at most ~max_bar_px thick; the rest of the band stays air."""
    slot = (plot_px or (1100 if wide else 520)) / max(n_categories, 1)
    return min(0.85, max(0.25, 1 - n_series * max_bar_px * 1.15 / slot))


def diverging_heatmap(
    z: pd.DataFrame, fmt: str, cap: float | None = None, text: bool = True, hover_fmt: str | None = None
) -> dict:
    vals = z.to_numpy(dtype=float)
    finite = vals[pd.notna(vals)]
    a = cap or (float(abs(finite).max()) if finite.size else 1.0) or 1.0
    trace = {
        "type": "heatmap",
        "z": vals.tolist(),
        "x": [str(c) for c in z.columns],
        "y": [str(i) for i in z.index],
        "colorscale": "@diverging",
        "zmid": 0,
        "zmin": -a,
        "zmax": a,
        "xgap": 2,
        "ygap": 2,
        "hoverongaps": False,
        "hovertemplate": f"%{{y}} · %{{x}}: %{{z:{hover_fmt or fmt}}}<extra></extra>",
        "colorbar": {"thickness": 10, "outlinewidth": 0, "tickformat": ".0%" if "%" in fmt else fmt},
    }
    if text:
        trace.update(texttemplate=f"%{{z:{fmt}}}", textfont={"size": 10, "color": "@text-primary"})
    return trace


def short_names(cfg: StrategyConfig, bench_names: list[str]) -> dict[str, str]:
    eq = cfg.risky[0]
    mapping = {
        f"Buy&hold {eq}": "Buy & hold equities",
        "Static equal-weight (5 ETFs)": "Equal-weight",
        f"Static 60/40 {eq}/IDTL": "60/40",
    }
    return {b: mapping.get(b, b) for b in bench_names}


def compute_backtest(prices: pd.DataFrame, cfg: StrategyConfig, cost_bps: float) -> dict[str, Any]:
    prices = prices[list(cfg.assets)].dropna()
    r, w, to_total, n_reb = run_backtest(prices, cfg, cost_bps)
    rets = prices.pct_change().fillna(0.0)
    rf = rets[cfg.safe].loc[r.index]
    benches = benchmark_returns(prices, cfg, cost_bps, r.index)
    names = short_names(cfg, list(benches))
    series = {"Dual momentum": r} | {names[k]: v for k, v in benches.items()}
    series = {k: series[k] for k in STRATEGY_SLOT if k in series}
    initial = pd.Series(0.0, index=w.columns)
    initial[cfg.safe] = 1.0
    turnover = rebalance_turnover(w, rets, initial)
    years = len(r) / 252
    cost_rows = []
    for c in COST_GRID:
        rc = r if c == cost_bps else run_backtest(prices, cfg, c)[0]
        cost_rows.append({"Cost (bps)": c, "CAGR": perf_stats(rc, rf)["CAGR"], "Sharpe": sharpe(rc, rf)})
    return {
        "prices": prices,
        "r": r,
        "weights": w,
        "rets": rets.loc[r.index],
        "rf": rf,
        "series": series,
        "stats": {k: perf_stats(v, rf) for k, v in series.items()},
        "turnover": turnover,
        "n_reb": n_reb,
        "turnover_per_year": to_total / years,
        "reb_per_year": n_reb / years,
        "monthly": monthly_table(r),
        "regimes": regime_table(series),
        "cost": pd.DataFrame(cost_rows),
        "corr_assets": rets.loc[r.index].corr(),
        "corr_strats": pd.DataFrame(series).corr(),
    }


class Report:
    def __init__(self) -> None:
        self.sections: list[str] = []
        self.figs: list[dict] = []

    def chart(
        self,
        title: str,
        subtitle: str,
        traces: list[dict],
        layout: dict | None = None,
        table: str = "",
        height: int = 340,
        wide: bool = False,
    ) -> str:
        fid = f"fig{len(self.figs)}"
        self.figs.append({"id": fid, "data": traces, "layout": layout or {}})
        tv = (
            f'<details><summary>Table view</summary><div class="tbl">{table}</div></details>' if table else ""
        )
        return (
            f'<figure class="card{" wide" if wide else ""}"><figcaption><h3>{esc(title)}</h3>'
            f"<p>{esc(subtitle)}</p></figcaption>"
            f'<div class="plot" id="{fid}" style="height:{height}px"></div>{tv}</figure>'
        )

    @staticmethod
    def card(title: str, subtitle: str, body: str, wide: bool = True) -> str:
        return (
            f'<figure class="card{" wide" if wide else ""}"><figcaption><h3>{esc(title)}</h3>'
            f'<p>{esc(subtitle)}</p></figcaption><div class="tbl open">{body}</div></figure>'
        )

    @staticmethod
    def tiles(items: list[tuple[str, str, str, str]]) -> str:
        """(label, value, delta text, delta class: good|bad|'')"""
        out = []
        for label, value, delta, cls in items:
            d = f'<div class="delta {cls}">{esc(delta)}</div>' if delta else ""
            out.append(
                f'<div class="tile"><div class="label">{esc(label)}</div>'
                f'<div class="value">{esc(value)}</div>{d}</div>'
            )
        return f'<div class="kpis">{"".join(out)}</div>'

    def section(self, sid: str, title: str, intro: str, body: list[str]) -> None:
        self.sections.append(
            f'<section id="{sid}"><h2>{esc(title)}</h2><p class="intro">{intro}</p>'
            f'<div class="grid">{"".join(body)}</div></section>'
        )


def delta(
    a: float,
    b: float,
    higher_is_better: bool,
    unit: str = "pts",
    scale: float = 100,
    words: tuple[str, str] = ("above", "below"),
) -> tuple[str, str]:
    """Stat-tile delta vs 60/40: arrow + words carry the meaning, color only reinforces it."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return "", ""
    d = (a - b) * scale
    good = (d > 0) == higher_is_better
    arrow, word = ("▲", words[0]) if d > 0 else ("▼", words[1])
    size = f"{abs(d):.{2 if scale == 1 else 1}f}" + (f" {unit}" if unit else "")
    return f"{arrow} {size} {word} 60/40", "good" if good else "bad"


def backtest_section(
    rep: Report, res: dict[str, Any], cfg: StrategyConfig, label: str, asset_labels: dict[str, str]
) -> None:
    series, stats = res["series"], res["stats"]
    dm, bm = stats["Dual momentum"], stats.get("60/40", {})
    x = dates(res["r"].index)
    tiles = [
        ("CAGR", pct(dm["CAGR"]), *delta(dm["CAGR"], bm.get("CAGR", math.nan), True)),
        (
            "Sharpe (vs T-bills)",
            f"{dm['Sharpe']:.2f} ± {dm['Sharpe SE']:.2f}",
            *delta(dm["Sharpe"], bm.get("Sharpe", math.nan), True, "", 1),
        ),
        (
            "Max drawdown",
            pct(dm["Max drawdown"]),
            *delta(
                dm["Max drawdown"],
                bm.get("Max drawdown", math.nan),
                True,
                words=("shallower than", "deeper than"),
            ),
        ),
        ("Calmar", num(dm["Calmar"]), *delta(dm["Calmar"], bm.get("Calmar", math.nan), True, "", 1)),
        ("Rebalances / year", num(res["reb_per_year"], 1), "", ""),
        ("Turnover / year", f"{res['turnover_per_year']:.1f}×", "", ""),
    ]
    body = [rep.tiles(tiles)]
    eq = pd.DataFrame({k: eq_series(v) for k, v in series.items()})
    dd = pd.DataFrame({k: dd_series(v) for k, v in series.items()})
    body.append(
        rep.chart(
            "Growth of 1 (log scale)",
            f"{label}, after costs. Hover for every series at a date.",
            [line(x, eq[k], k, STRATEGY_SLOT[k]) for k in eq],
            {
                "yaxis": {"type": "log", "tickformat": ".2~f", "tickvals": log_ticks(eq)},
                "hovermode": "x unified",
            },
            table_html(month_end(eq), {c: num for c in eq}),
            height=380,
            wide=True,
        )
    )
    body.append(
        rep.chart(
            "Drawdown",
            "Fall from the previous peak. Shallower is better.",
            [line(x, dd[k], k, STRATEGY_SLOT[k], ".1%") for k in dd],
            {"yaxis": {"tickformat": ".0%"}, "hovermode": "x unified"},
            table_html(month_end(dd), {c: pct for c in dd}),
            wide=True,
        )
    )

    focus = [k for k in ("Dual momentum", "60/40") if k in series]
    rs = pd.DataFrame({k: rolling_sharpe(series[k], res["rf"]) for k in focus}).dropna()
    rv = pd.DataFrame({k: rolling_vol(series[k]) for k in focus}).dropna()
    body.append(
        rep.chart(
            "Rolling 1-year Sharpe",
            "252-day window, excess of T-bills. Shows how unstable a Sharpe ratio is.",
            [line(dates(rs.index), rs[k], k, STRATEGY_SLOT[k]) for k in focus],
            {"hovermode": "x unified", "yaxis": {"zeroline": True}},
            table_html(month_end(rs), {c: num for c in focus}),
        )
    )
    body.append(
        rep.chart(
            "Rolling 3-month volatility",
            "63-day window, annualised.",
            [line(dates(rv.index), rv[k], k, STRATEGY_SLOT[k], ".1%") for k in focus],
            {"hovermode": "x unified", "yaxis": {"tickformat": ".0%", "rangemode": "tozero"}},
            table_html(month_end(rv), {c: pct for c in focus}),
        )
    )

    yearly = pd.DataFrame({k: period_returns(series[k], "YE") for k in focus})
    yearly.index = yearly.index.year
    body.append(
        rep.chart(
            "Calendar-year returns",
            "First and last years are partial.",
            [
                {
                    "type": "bar",
                    "x": [str(y) for y in yearly.index],
                    "y": list(yearly[k]),
                    "name": k,
                    "marker": {"color": f"@{STRATEGY_SLOT[k]}", "cornerradius": 4},
                    "hovertemplate": f"%{{x}}: %{{y:.1%}}<extra>{esc(k)}</extra>",
                }
                for k in focus
            ],
            {
                "barmode": "group",
                "bargap": bar_gap(len(yearly), len(focus), wide=True),
                "bargroupgap": 0.08,
                "yaxis": {"tickformat": ".0%"},
            },
            table_html(yearly, {c: pct for c in focus}),
            wide=True,
        )
    )

    mt = res["monthly"]
    body.append(
        rep.chart(
            "Monthly returns: dual momentum",
            "Blue = gain, red = loss, gray = flat. Year totals in the table view.",
            [diverging_heatmap(mt[MONTHS], ".1%", cap=0.08)],
            {
                "yaxis": {"autorange": "reversed", "type": "category"},
                "xaxis": {"side": "top"},
                "margin": {"t": 56},
            },
            table_html(mt, {c: pct for c in mt.columns}),
            height=max(300, 26 * len(mt) + 60),
            wide=True,
        )
    )

    w = res["weights"].rename(columns=asset_labels)
    wm = w.resample("ME").mean()
    body.append(
        rep.chart(
            "Allocation over time",
            "Average monthly weight per ETF. Stronger blue = larger share of the portfolio.",
            [
                {
                    "type": "heatmap",
                    "z": wm.T.to_numpy().tolist(),
                    "x": dates(wm.index),
                    "y": list(wm.columns),
                    "colorscale": "@sequential",
                    "zmin": 0,
                    "zmax": 1,
                    "ygap": 2,
                    "hovertemplate": "%{y} · %{x|%b %Y}: %{z:.0%}<extra></extra>",
                    "colorbar": {"thickness": 10, "outlinewidth": 0, "tickformat": ".0%"},
                }
            ],
            {"yaxis": {"autorange": "reversed"}},
            table_html(month_end(w), {c: pct for c in w.columns}),
            height=260,
            wide=True,
        )
    )

    to = res["turnover"]
    reb = (to > 1e-9).groupby(to.index.year).sum()
    tby = to.groupby(to.index.year).sum()
    body.append(
        rep.chart(
            "Rebalances per year",
            "Days the strategy traded. Whipsaw shows up as tall bars.",
            [
                {
                    "type": "bar",
                    "x": [str(y) for y in reb.index],
                    "y": list(reb),
                    "name": "Rebalances",
                    "marker": {"color": "@series-1", "cornerradius": 4},
                    "hovertemplate": "%{x}: %{y} rebalances<extra></extra>",
                }
            ],
            {"bargap": bar_gap(len(reb), 1)},
            table_html(
                pd.DataFrame({"Rebalances": reb, "Turnover (×)": tby}),
                {"Rebalances": lambda v: f"{int(v)}", "Turnover (×)": num},
            ),
        )
    )
    held = time_held(w)
    body.append(
        rep.chart(
            "Time held per ETF",
            "Share of days each ETF was in the portfolio (weight > 2%).",
            [
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": list(held.index),
                    "x": list(held),
                    "name": "Held",
                    "marker": {"color": "@series-1", "cornerradius": 4},
                    "hovertemplate": "%{y}: %{x:.0%} of days<extra></extra>",
                }
            ],
            {
                "bargap": bar_gap(len(held), 1, plot_px=200),
                "xaxis": {"tickformat": ".0%", "range": [0, 1]},
                "yaxis": {"autorange": "reversed"},
            },
            table_html(held.to_frame("Share of days"), {"Share of days": pct}),
            height=260,
        )
    )

    body.append(
        rep.chart(
            "Daily return distribution",
            "Share of days per 0.1% bucket, clipped at ±4%. Fat tails sit at the edges.",
            [
                {
                    "type": "histogram",
                    "x": list(series[k]),
                    "name": k,
                    "histnorm": "probability",
                    "xbins": {"start": -0.04, "end": 0.04, "size": 0.001},
                    "opacity": 0.6,
                    "marker": {"color": f"@{STRATEGY_SLOT[k]}"},
                    "hovertemplate": f"%{{x}}: %{{y:.1%}} of days<extra>{esc(k)}</extra>",
                }
                for k in focus
            ],
            {
                "barmode": "overlay",
                "bargap": 0.05,
                "xaxis": {"tickformat": ".0%", "range": [-0.04, 0.04]},
                "yaxis": {"tickformat": ".0%"},
            },
            table_html(
                pd.DataFrame(
                    {
                        k: {
                            m: stats[k][m]
                            for m in ("Skew", "Excess kurtosis", "Best day", "Worst day", "Positive days")
                        }
                        for k in focus
                    }
                ),
                {k: num for k in focus},
            ),
        )
    )
    ca = res["corr_assets"].rename(index=asset_labels, columns=asset_labels)
    cs = res["corr_strats"]
    body.append(
        rep.chart(
            "Correlation matrix: ETFs",
            "Daily returns over the backtest period.",
            [diverging_heatmap(ca, ".2f", cap=1.0)],
            {"yaxis": {"autorange": "reversed"}},
            table_html(ca, {c: num for c in ca.columns}),
            height=340,
        )
    )
    body.append(
        rep.chart(
            "Correlation matrix: strategies",
            "Daily returns of the strategy and the benchmarks.",
            [diverging_heatmap(cs, ".2f", cap=1.0)],
            {"yaxis": {"autorange": "reversed"}},
            table_html(cs, {c: num for c in cs.columns}),
            height=340,
        )
    )

    reg = res["regimes"]
    body.append(
        rep.chart(
            "Stress periods",
            "Total return inside each window (windows chosen after the fact; descriptive only).",
            [
                {
                    "type": "bar",
                    "x": list(REGIMES),
                    "y": list(reg[reg["Series"] == k]["Return"]),
                    "name": k,
                    "marker": {"color": f"@{STRATEGY_SLOT[k]}", "cornerradius": 4},
                    "hovertemplate": f"%{{x}}: %{{y:.1%}}<extra>{esc(k)}</extra>",
                }
                for k in series
            ],
            {
                "barmode": "group",
                "bargap": bar_gap(len(REGIMES), len(series), wide=True),
                "bargroupgap": 0.08,
                "yaxis": {"tickformat": ".0%"},
            },
            table_html(
                reg.set_index(["Regime", "Series"]).drop(columns="Window"),
                {"Return": pct, "Max drawdown": pct},
            ),
            wide=True,
        )
    )

    st = pd.DataFrame(stats)
    fmts: dict[str, Any] = {
        "CAGR": pct,
        "Volatility": pct,
        "Max drawdown": pct,
        "Positive days": pct,
        "Best day": lambda v: pct(v, 2),
        "Worst day": lambda v: pct(v, 2),
        "Longest drawdown (days)": lambda v: f"{v:,.0f}",
        "Years": lambda v: f"{v:.1f}",
    }
    rows = []
    for metric, vals in st.iterrows():
        f = fmts.get(str(metric), num)
        rows.append([metric] + [f(v) for v in vals])
    body.append(
        rep.card(
            "Summary statistics",
            "All after costs; Sharpe and Sortino on returns in excess of T-bills.",
            table_html(pd.DataFrame(rows, columns=["Metric", *st.columns]).set_index("Metric")),
        )
    )
    cost = res["cost"]
    body.append(
        rep.chart(
            "Cost sensitivity",
            "Dual momentum Sharpe at different one-way trading costs.",
            [
                line(
                    [f"{c:g} bps" for c in cost["Cost (bps)"]],
                    cost["Sharpe"],
                    "Sharpe",
                    "series-1",
                    mode="lines+markers",
                )
            ],
            {"yaxis": {"rangemode": "tozero"}},
            table_html(cost.set_index("Cost (bps)"), {"CAGR": pct, "Sharpe": num}),
        )
    )
    rep.section(
        "backtest",
        "Backtest",
        esc(
            f"{label}. Dual momentum vs the three benchmarks already seen before pre-registration. "
            f"{len(res['r'])} trading days, "
            f"{res['r'].index[0]:%Y-%m-%d} to {res['r'].index[-1]:%Y-%m-%d}."
        ),
        body,
    )


def live_section(
    rep: Report,
    runs: pd.DataFrame,
    fills: pd.DataFrame,
    parity: pd.DataFrame,
    conf: dict,
    monthly_signal: bool = False,
) -> None:
    """Live (paper) results from the run log and journal; works for either strategy.
    monthly_signal: the stock bot trades on month-end signals by design, so signal age
    is not a defect there and the staleness chart is skipped."""
    body: list[str] = []
    if runs.empty:
        rep.section("live", "Live (paper)", "No structured runs in logs/bot.jsonl yet.", [])
        return
    local = runs["start"].dt.tz_convert(WARSAW)
    xs = [t.strftime("%Y-%m-%d %H:%M:%S") for t in local]
    labels_short = short_labels(runs["start"])
    one_x = {"xaxis": {"type": "category"}} if len(set(xs)) < 2 else {}
    close_label = "previous close" if monthly_signal else "signal-date close"
    with_pos = runs[runs["positions"].map(bool)]
    sf = shortfall(fills, runs)
    sv = sleeve_value(runs, fills)
    counts = runs["outcome"].value_counts()
    latest_sig = runs[runs["signal_date"].notna()].iloc[-1] if runs["signal_date"].notna().any() else None
    n_match = int(parity["match"].sum()) if not parity.empty else 0
    body.append(
        rep.tiles(
            [
                (
                    "Live since",
                    f"{local.iloc[0]:%Y-%m-%d}",
                    f"{runs['start'].dt.date.nunique()} day(s) with runs",
                    "",
                ),
                (
                    "Runs",
                    f"{len(runs)}",
                    " · ".join(f"{k} {v}" for k, v in counts.items()),
                    "bad" if counts.get("CRASH", 0) or counts.get("ABORT", 0) else "",
                ),
                (
                    "Sleeve value",
                    f"${sv['value'].iloc[-1]:,.0f}" if len(sv) else "–",
                    (
                        f"{'▲' if sv['pnl'].iloc[-1] >= 0 else '▼'} "
                        f"${abs(sv['pnl'].iloc[-1]):,.0f} since first run"
                        if len(sv)
                        else ""
                    ),
                    ("good" if sv["pnl"].iloc[-1] >= 0 else "bad") if len(sv) else "",
                ),
                (
                    "Latest signal",
                    f"{latest_sig['signal_date']:%Y-%m-%d}" if latest_sig is not None else "–",
                    (
                        "month-end signal (by design)"
                        if monthly_signal
                        else f"{latest_sig['staleness_bdays']:.0f} business days old (expected 1)"
                    )
                    if latest_sig is not None
                    else "",
                    "bad"
                    if latest_sig is not None and latest_sig["staleness_bdays"] > 1 and not monthly_signal
                    else "",
                ),
                (
                    "Fills vs backtest price",
                    f"${sf['usd_vs_signal_close'].sum():,.2f}" if len(sf) else "–",
                    f"{len(sf)} fills; positive = cost",
                    "bad" if len(sf) and sf["usd_vs_signal_close"].sum() > 0 else "good",
                ),
                (
                    "Signals reproducible",
                    f"{n_match}/{len(parity)}" if len(parity) else "–",
                    "re-computed from today's data",
                    "bad" if len(parity) and n_match < len(parity) else "",
                ),
            ]
        )
    )

    if len(with_pos):
        last = with_pos.iloc[-1]
        risk = conf["risk"]
        symbols = [s for s in last["ibkr"] if last["positions"].get(s, 0) or s in last["target"]]
        budget = budget_usd(symbols, last["positions"], last["ibkr"], last["usd_cash"], risk)
        cur = (
            current_weights(symbols, last["positions"], last["ibkr"], budget)
            if budget > 0
            else pd.Series(0.0, index=symbols)
        )
        tgt = pd.Series({s: float(last["target"].get(s, 0.0)) for s in symbols})
        body.append(
            rep.chart(
                "Holdings vs target",
                f"Latest run {last['start'].tz_convert(WARSAW):%Y-%m-%d %H:%M}: weights "
                f"as a share of the ${budget:,.0f} budget.",
                [
                    {
                        "type": "bar",
                        "orientation": "h",
                        "y": symbols,
                        "x": list(cur),
                        "name": "Current",
                        "marker": {"color": "@series-1", "cornerradius": 4},
                        "hovertemplate": "%{y}: %{x:.1%}<extra>Current</extra>",
                    },
                    {
                        "type": "bar",
                        "orientation": "h",
                        "y": symbols,
                        "x": list(tgt),
                        "name": "Target",
                        "marker": {"color": "@muted", "cornerradius": 4},
                        "hovertemplate": "%{y}: %{x:.1%}<extra>Target</extra>",
                    },
                ],
                {
                    "barmode": "group",
                    "bargap": bar_gap(len(symbols), 2, plot_px=260),
                    "bargroupgap": 0.1,
                    "xaxis": {"tickformat": ".0%"},
                    "yaxis": {"autorange": "reversed"},
                },
                table_html(
                    pd.DataFrame(
                        {
                            "Shares": pd.Series(last["positions"]),
                            "Price": pd.Series(last["ibkr"]),
                            "Current": cur,
                            "Target": tgt,
                        }
                    ),
                    {"Shares": lambda v: f"{v:,.0f}", "Price": num, "Current": pct, "Target": pct},
                ),
            )
        )
        hx = [t.strftime("%Y-%m-%d %H:%M:%S") for t in sv["start"].dt.tz_convert(WARSAW)]
        body.append(
            rep.chart(
                "Sleeve value per run",
                "Bot holdings + net cash from the bot's own fills since the first run (mark-to-market; "
                "excludes commissions and manual trades).",
                [line(hx, sv["value"], "Sleeve value", "series-1", ",.0f", mode="lines+markers")],
                {"yaxis": {"tickprefix": "$"}, **({"xaxis": {"type": "category"}} if len(hx) < 2 else {})},
                table_html(
                    sv.assign(start=hx).set_index("start"),
                    {c: lambda v: f"{v:,.2f}" for c in ("holdings", "trade_cash", "value", "pnl")},
                ),
            )
        )

    traces = []
    for outcome, (tok, symbol) in OUTCOME_STYLE.items():
        sel = runs["outcome"] == outcome
        if not sel.any():
            continue
        sub = runs[sel]
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": outcome,
                "x": [xs[i] for i in sub.index],
                "y": list(sub["requested"]),
                "marker": {
                    "color": f"@{tok}",
                    "size": 12,
                    "symbol": symbol,
                    "line": {"color": "@surface-1", "width": 2},
                },
                "customdata": [[r["mode"], redact(r["reason"])[:120], r["git"]] for _, r in sub.iterrows()],
                "hovertemplate": "%{x}<br>requested %{y} → ran %{customdata[0]}<br>%{customdata[1]}"
                "<br>code %{customdata[2]}<extra>" + outcome + "</extra>",
            }
        )
    body.append(
        rep.chart(
            "Run timeline",
            "Every bot run by requested mode; symbol and color give the outcome.",
            traces,
            {
                "yaxis": {
                    "type": "category",
                    "categoryorder": "array",
                    "categoryarray": ["check", "dry", "execute"],
                },
                **one_x,
            },
            table_html(
                pd.DataFrame(
                    {
                        "Requested": list(runs["requested"]),
                        "Ran as": list(runs["mode"]),
                        "Outcome": list(runs["outcome"]),
                        "Reason": [redact(r)[:90] for r in runs["reason"]],
                        "Code": list(runs["git"]),
                    },
                    index=xs,
                )
            ),
            wide=True,
        )
    )

    sig = runs[runs["signal_date"].notna()]
    if len(sig) and not monthly_signal:
        sx = [labels_short[i] for i in sig.index]
        body.append(
            rep.chart(
                "Signal staleness",
                "Business days between the signal's close and the run. 1 = as the backtest assumes.",
                [
                    {
                        "type": "bar",
                        "x": sx,
                        "y": list(sig["staleness_bdays"]),
                        "name": "Staleness",
                        "marker": {"color": "@series-1", "cornerradius": 4},
                        "hovertemplate": "%{x}: %{y} business days<extra></extra>",
                    }
                ],
                {
                    "bargap": bar_gap(len(sig), 1),
                    "xaxis": {"type": "category"},
                    "yaxis": {"dtick": 1, "rangemode": "tozero"},
                    "shapes": [
                        {
                            "type": "line",
                            "xref": "paper",
                            "x0": 0,
                            "x1": 1,
                            "y0": 1,
                            "y1": 1,
                            "line": {"color": "@text-secondary", "width": 1},
                        }
                    ],
                    "annotations": [
                        {
                            "xref": "paper",
                            "x": 1,
                            "y": 1,
                            "xanchor": "right",
                            "yanchor": "bottom",
                            "text": "expected",
                            "showarrow": False,
                            "font": {"color": "@text-secondary"},
                        }
                    ],
                },
                table_html(
                    pd.DataFrame(
                        {
                            "Signal date": [f"{d:%Y-%m-%d}" for d in sig["signal_date"]],
                            "Business days": list(sig["staleness_bdays"]),
                        },
                        index=sx,
                    ),
                    {"Business days": lambda v: f"{v:.0f}"},
                ),
            )
        )

    if len(with_pos):
        gap_cols = [labels_short[i] for i in with_pos.index]
        gaps = pd.DataFrame(
            {
                x: {s: r["ibkr"][s] / r["yahoo"][s] - 1 for s in r["yahoo"]}
                for x, (_, r) in zip(gap_cols, with_pos.iterrows(), strict=True)
            }
        )
        body.append(
            rep.chart(
                "Price cross-check: IBKR vs Yahoo",
                (
                    "IBKR (delayed) price / latest Yahoo close − 1, per run. "
                    "The stock bot aborts beyond ±15% "
                    "for names it will trade (overnight earnings moves are real)."
                    if monthly_signal
                    else "IBKR (delayed) price / Yahoo signal-date close − 1, per run. "
                    "That close is 1–2 days old, "
                    "so steady gaps mostly reflect market moves since then. The bot aborts beyond ±5%."
                ),
                [
                    diverging_heatmap(gaps * 1e4, ".0f", text=len(gaps.columns) <= 12, hover_fmt=".1f")
                    | {"hovertemplate": "%{y} · %{x}: %{z:.1f} bps<extra></extra>"}
                ],
                {"yaxis": {"autorange": "reversed"}, "xaxis": {"type": "category"}},
                table_html((gaps * 1e4).T, {c: lambda v: f"{v:.1f} bps" for c in gaps.index}),
                height=max(280, 22 * len(gaps.index) + 140),
                wide=True,
            )
        )

    if len(sf):
        labels = [
            f"{t.tz_convert(WARSAW):%m-%d} {s} {sym}"
            for t, s, sym in zip(sf["ts"], sf["side"], sf["sym"], strict=True)
        ]
        body.append(
            rep.chart(
                "Implementation shortfall per fill",
                "Basis points; positive = worse than the reference price. "
                "Paper fills are optimistic (no queue position, no market impact).",
                [
                    {
                        "type": "bar",
                        "x": labels,
                        "y": list(sf["bps_vs_signal_close"]),
                        "name": f"vs {close_label}",
                        "marker": {"color": "@series-1", "cornerradius": 4},
                        "hovertemplate": f"%{{x}}: %{{y:.1f}} bps<extra>vs {close_label}</extra>",
                    },
                    {
                        "type": "bar",
                        "x": labels,
                        "y": list(sf["bps_vs_decision"]),
                        "name": "vs decision price",
                        "marker": {"color": "@series-2", "cornerradius": 4},
                        "hovertemplate": "%{x}: %{y:.1f} bps<extra>vs decision price</extra>",
                    },
                ],
                {
                    "barmode": "group",
                    "bargap": bar_gap(len(sf), 2),
                    "bargroupgap": 0.08,
                    "yaxis": {"zeroline": True},
                },
                table_html(
                    sf.assign(ts=[f"{t.tz_convert(WARSAW):%Y-%m-%d %H:%M}" for t in sf["ts"]])
                    .set_index("ts")
                    .drop(columns="ref"),
                    {
                        "filled": lambda v: f"{v:,.0f}",
                        "avg": num,
                        "decision_px": num,
                        "signal_close": num,
                        "bps_vs_decision": lambda v: f"{v:.1f}",
                        "bps_vs_signal_close": lambda v: f"{v:.1f}",
                        "usd_vs_signal_close": num,
                    },
                ),
            )
        )

    if len(fills):
        f = fills.assign(ts=[f"{t.tz_convert(WARSAW):%Y-%m-%d %H:%M}" for t in fills["ts"]]).set_index("ts")
        body.append(
            rep.card(
                "Order journal",
                "Every order outcome from journal.jsonl (Warsaw time).",
                table_html(f, {"qty": lambda v: f"{v:,.0f}", "filled": lambda v: f"{v:,.0f}", "avg": num}),
                wide=False,
            )
        )
    if len(parity):
        p = parity.assign(
            start=[f"{t.tz_convert(WARSAW):%Y-%m-%d %H:%M}" for t in parity["start"]],
            signal_date=[f"{d:%Y-%m-%d}" for d in parity["signal_date"]],
            match=["✓ same" if m else "✗ differs" for m in parity["match"]],
        ).set_index("start")
        body.append(
            rep.card(
                "Signal reproducibility",
                "Each logged target re-computed with today's code and data. "
                "A difference means a code change or a Yahoo data revision.",
                table_html(p),
                wide=False,
            )
        )
    rep.section(
        "live",
        "Live (paper)",
        esc(
            f"From logs/bot.jsonl and journal.jsonl: {len(runs)} runs since {local.iloc[0]:%Y-%m-%d}. "
            "Small samples: read these as operations checks, not as performance evidence."
        ),
        body,
    )


def build_report(
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    conf: dict,
    cost_bps: float,
    label: str,
    runs: pd.DataFrame,
    fills: pd.DataFrame,
    live_prices: pd.DataFrame | None,
    meta: dict[str, str],
    asset_labels: dict[str, str] | None = None,
) -> str:
    rep = Report()
    res = compute_backtest(prices, cfg, cost_bps)
    backtest_section(rep, res, cfg, label, asset_labels or {a: a for a in cfg.assets})
    if meta.get("live") != "off":
        parity = (
            signal_parity(runs, live_prices, cfg)
            if live_prices is not None
            else pd.DataFrame(columns=["start", "signal_date", "logged", "recomputed", "match"])
        )
        live_section(rep, runs, fills, parity, conf)
    notes = [
        "The pre-registered hypotheses (H1 vol-targeted 60/40, H2 trend-filtered 60/40) are deliberately "
        "absent: they are tested exactly once, by prereg_test.py.",
        "US-listed proxies (SPY, EEM, TLT, GLD, BIL) stand in for the UCITS ETFs before ~2019; fees, "
        "withholding tax and tracking differ.",
        "Backtest timing: signal on the close of day t, earning from t+1, i.e. trading at that close. "
        "The live bot trades mid-session one or two days later (Yahoo lag), so the live signal is older.",
        "Costs: a flat charge of the stated bps × turnover; no commission tiers, spread or market "
        "impact yet.",
        "Stress windows were picked after the fact and are descriptive only.",
        "IBKR paper fills are optimistic: no queue position, no market impact. IBKR prices in the logs are "
        "15-minute delayed.",
    ]
    return render_page(rep, "Dual momentum: backtest and live report", notes, meta)


def render_page(rep: Report, title: str, notes: list[str], meta: dict[str, str]) -> str:
    """Notes section + header + figure data -> one self-contained, redacted HTML page."""
    rep.section(
        "notes",
        "Method notes and caveats",
        "",
        [
            rep.card(
                "Read before drawing conclusions",
                "",
                '<ul class="notes">' + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>",
            )
        ],
    )
    meta_html = "".join(f"<div>{esc(k)}: <b>{esc(v)}</b></div>" for k, v in meta.items() if k != "live")
    figs = json.dumps(clean(rep.figs), allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    css, js = (TEMPLATES / "report.css").read_text(), (TEMPLATES / "report.js").read_text()
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Strategy Report</title>
<style>{css}</style><script src="{PLOTLY}" charset="utf-8"></script></head><body><main>
<header><h1>{esc(title)}</h1>
<p>Generated by report.py. Every number on this page comes from code in this repository.</p>
<nav><a href="#backtest">Backtest</a><a href="#live">Live</a><a href="#notes">Notes</a></nav>
<div class="meta">{meta_html}</div></header>
{"".join(rep.sections)}
</main><script type="application/json" id="figs">{figs}</script><script>{js}</script></body></html>"""
    return redact(page)


STOCK_SLOT = {"Momentum": "series-1", "Equal-weight": "series-2", "SPY": "series-3"}
DIST_STATS = ("Skew", "Excess kurtosis", "Best day", "Worst day", "Positive days")
SLIPPAGE_GRID = (0.0, 5.0, 10.0, 20.0, 50.0)


def stock_cost_sensitivity(
    prices: pd.DataFrame, cash: pd.Series, cfg: StockConfig, base: Any
) -> pd.DataFrame:
    """Momentum Sharpe/CAGR across slippage assumptions (commissions unchanged)."""
    rows = []
    for slip in SLIPPAGE_GRID:
        bt = (
            base
            if slip == DEFAULT_COSTS.slippage_bps
            else run_monthly(
                prices,
                cash,
                strategy_fn(cfg),
                list(cfg.universe),
                base.returns.index[0],
                CostModel(slippage_bps=slip),
            )
        )
        st = perf_stats(bt.returns, cash.loc[bt.returns.index])
        rows.append({"Slippage (bps)": slip, "CAGR": st["CAGR"], "Sharpe": st["Sharpe"]})
    return pd.DataFrame(rows)


def stock_backtest_section(
    rep: Report,
    res: dict[str, Any],
    prices: pd.DataFrame,
    cfg: StockConfig,
    uni: pd.DataFrame,
    cost: pd.DataFrame,
) -> None:
    strat, ew = res["strategy"], res["equal_weight"]
    rf = res["cash"]
    series = {"Momentum": strat.returns, "Equal-weight": ew.returns, "SPY": res["spy"]}
    stats = {k: perf_stats(v, rf) for k, v in series.items()}
    m, e = stats["Momentum"], stats["Equal-weight"]
    years = len(rf) / 252
    turnover = float(strat.trades["value"].sum()) / float(strat.nav.mean()) / years
    drag = float(strat.costs.sum()) / float(strat.nav.mean()) / years
    on_share = float(strat.decisions["market_on"].mean()) if len(strat.decisions) else math.nan
    vs = ("above", "below")

    def d(
        a: float, b: float, unit: str = "pts", scale: float = 100, words: tuple[str, str] = vs
    ) -> tuple[str, str]:
        text, cls = delta(a, b, True, unit, scale, words)
        return text.replace("60/40", "equal-weight"), cls

    body = [
        rep.card(
            "Read first: survivorship bias",
            "The universe is today's S&P 100 applied to the past.",
            '<p class="notes">Stocks join the index <b>because</b> they rose, and losers leave, so every '
            "series built from this list is flattered: absolute returns are an <b>upper bound</b>. "
            "Momentum chases exactly the winners whose inclusion was decided after the fact. Compare "
            "momentum with the equal-weight portfolio of the same list (same bias) rather than with SPY, "
            "and treat even that gap with caution. "
            "Rules were fixed in STOCK_STRATEGY.md before this ran.</p>",
        )
    ]
    body.append(
        rep.tiles(
            [
                ("CAGR", pct(m["CAGR"]), *d(m["CAGR"], e["CAGR"])),
                (
                    "Sharpe (vs T-bills)",
                    f"{m['Sharpe']:.2f} ± {m['Sharpe SE']:.2f}",
                    *d(m["Sharpe"], e["Sharpe"], "", 1),
                ),
                (
                    "Max drawdown",
                    pct(m["Max drawdown"]),
                    *d(m["Max drawdown"], e["Max drawdown"], words=("shallower than", "deeper than")),
                ),
                ("Calmar", num(m["Calmar"]), *d(m["Calmar"], e["Calmar"], "", 1)),
                ("Months in the market", pct(on_share, 0), "SPY above its 200-day average", ""),
                ("Turnover / year", f"{turnover:.1f}×", f"cost drag {pct(drag, 2)} / year", ""),
            ]
        )
    )
    x = dates(strat.returns.index)
    nav = pd.DataFrame(
        {"Momentum": strat.nav, "Equal-weight": ew.nav, "SPY": CAPITAL * eq_series(res["spy"])}
    )
    body.append(
        rep.chart(
            "Growth of $25,000 (log scale)",
            "After commissions and slippage; cash earns T-bills (BIL).",
            [line(x, nav[k], k, STOCK_SLOT[k], ",.0f") for k in nav],
            {
                "yaxis": {"type": "log", "tickformat": "$,.0f", "tickvals": log_ticks(nav)},
                "hovermode": "x unified",
            },
            table_html(month_end(nav), {c: lambda v: f"{v:,.0f}" for c in nav}),
            height=380,
            wide=True,
        )
    )
    dd = pd.DataFrame({k: dd_series(v) for k, v in series.items()})
    body.append(
        rep.chart(
            "Drawdown",
            "Fall from the previous peak.",
            [line(x, dd[k], k, STOCK_SLOT[k], ".1%") for k in dd],
            {"yaxis": {"tickformat": ".0%"}, "hovermode": "x unified"},
            table_html(month_end(dd), {c: pct for c in dd}),
            wide=True,
        )
    )

    sma = prices[cfg.market].rolling(cfg.sma_window).mean().loc[strat.returns.index[0] :]
    spy = prices[cfg.market].loc[strat.returns.index[0] :]
    shapes = []
    dec = strat.decisions.reset_index(drop=True)
    for i, row in dec.iterrows():
        if not row["market_on"]:
            end = dec.loc[i + 1, "exec_date"] if i + 1 < len(dec) else spy.index[-1]
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": f"{row['exec_date']:%Y-%m-%d}",
                    "x1": f"{end:%Y-%m-%d}",
                    "y0": 0,
                    "y1": 1,
                    "line": {"width": 0},
                    "fillcolor": "@muted",
                    "opacity": 0.18,
                    "layer": "below",
                }
            )
    sx = dates(spy.index)
    body.append(
        rep.chart(
            "Market filter",
            "SPY vs its 200-day average. Shaded months: filter off, portfolio in cash.",
            [line(sx, spy, "SPY", "series-3"), line(sx, sma, "200-day average", "series-1")],
            {
                "yaxis": {"type": "log", "tickformat": "$,.0f", "tickvals": log_ticks(spy.to_frame())},
                "hovermode": "x unified",
                "shapes": shapes,
            },
            table_html(
                dec.assign(exec_date=[f"{t:%Y-%m-%d}" for t in dec["exec_date"]])
                .set_index("exec_date")[["market_on"]]
                .tail(36)
            ),
            height=340,
            wide=True,
        )
    )

    focus = ["Momentum", "Equal-weight"]
    rs = pd.DataFrame({k: rolling_sharpe(series[k], rf) for k in focus}).dropna()
    body.append(
        rep.chart(
            "Rolling 1-year Sharpe",
            "252-day window, excess of T-bills.",
            [line(dates(rs.index), rs[k], k, STOCK_SLOT[k]) for k in focus],
            {"hovermode": "x unified"},
            table_html(month_end(rs), {c: num for c in focus}),
        )
    )
    weights = strat.weights
    cash_share = weights["CASH"].resample("ME").mean()
    body.append(
        rep.chart(
            "Cash share",
            "Average monthly cash weight: filter-off months and unfilled slots.",
            [line(dates(cash_share.index), cash_share, "Cash", "series-1", ".0%")],
            {"yaxis": {"tickformat": ".0%", "range": [0, 1]}},
            table_html(month_end(weights[["CASH"]]), {"CASH": pct}),
        )
    )

    yearly = pd.DataFrame({k: period_returns(v, "YE") for k, v in series.items()})
    yearly.index = yearly.index.year
    body.append(
        rep.chart(
            "Calendar-year returns",
            "First and last years are partial.",
            [
                {
                    "type": "bar",
                    "x": [str(y) for y in yearly.index],
                    "y": list(yearly[k]),
                    "name": k,
                    "marker": {"color": f"@{STOCK_SLOT[k]}", "cornerradius": 4},
                    "hovertemplate": f"%{{x}}: %{{y:.1%}}<extra>{esc(k)}</extra>",
                }
                for k in yearly
            ],
            {
                "barmode": "group",
                "bargap": bar_gap(len(yearly), 3, wide=True),
                "bargroupgap": 0.08,
                "yaxis": {"tickformat": ".0%"},
            },
            table_html(yearly, {c: pct for c in yearly}),
            wide=True,
        )
    )
    mt = monthly_table(strat.returns)
    body.append(
        rep.chart(
            "Monthly returns: momentum",
            "Blue = gain, red = loss, gray = flat. Year totals in the table view.",
            [diverging_heatmap(mt[MONTHS], ".1%", cap=0.10)],
            {
                "yaxis": {"autorange": "reversed", "type": "category"},
                "xaxis": {"side": "top"},
                "margin": {"t": 56},
            },
            table_html(mt, {c: pct for c in mt.columns}),
            height=max(300, 26 * len(mt) + 60),
            wide=True,
        )
    )

    sector = dict(zip(uni["symbol"], uni["sector"], strict=True))
    stock_w = weights.drop(columns="CASH")
    by_sector = stock_w.T.groupby(stock_w.columns.map(lambda s: sector.get(s, "Other"))).sum().T
    sm = by_sector.resample("ME").mean()
    sm = sm[sm.mean().sort_values(ascending=False).index]
    body.append(
        rep.chart(
            "Sector exposure over time",
            "Average monthly weight per GICS sector. Stronger blue = larger share.",
            [
                {
                    "type": "heatmap",
                    "z": sm.T.to_numpy().tolist(),
                    "x": dates(sm.index),
                    "y": list(sm.columns),
                    "colorscale": "@sequential",
                    "zmin": 0,
                    "zmax": max(0.5, float(sm.max().max())),
                    "ygap": 2,
                    "hovertemplate": "%{y} · %{x|%b %Y}: %{z:.0%}<extra></extra>",
                    "colorbar": {"thickness": 10, "outlinewidth": 0, "tickformat": ".0%"},
                }
            ],
            {"yaxis": {"autorange": "reversed"}},
            table_html(month_end(by_sector), {c: pct for c in by_sector.columns}),
            height=max(300, 26 * len(sm.columns) + 80),
            wide=True,
        )
    )

    held_share = (stock_w > 0).mean().sort_values(ascending=False).head(20)
    body.append(
        rep.chart(
            "Most-held stocks",
            "Share of days in the portfolio (top 20).",
            [
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": list(held_share.index),
                    "x": list(held_share),
                    "name": "Held",
                    "marker": {"color": "@series-1", "cornerradius": 4},
                    "hovertemplate": "%{y}: %{x:.0%} of days<extra></extra>",
                }
            ],
            {
                "bargap": bar_gap(20, 1, plot_px=560),
                "xaxis": {"tickformat": ".0%"},
                "yaxis": {"autorange": "reversed"},
            },
            table_html(held_share.to_frame("Share of days"), {"Share of days": pct}),
            height=620,
        )
    )
    rets = prices[list(stock_w.columns)].pct_change().reindex(stock_w.index)
    contrib = (stock_w.shift(1) * rets).sum().sort_values()
    picks = pd.concat([contrib.head(10), contrib.tail(10)]).drop_duplicates()
    body.append(
        rep.chart(
            "Return contribution by stock",
            "Sum of daily contributions (weight × return), best and worst 10.",
            [
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": list(picks.index),
                    "x": list(picks),
                    "name": "Contribution",
                    "marker": {"color": ["@pos" if v >= 0 else "@neg" for v in picks], "cornerradius": 4},
                    "hovertemplate": "%{y}: %{x:.1%}<extra></extra>",
                }
            ],
            {"bargap": bar_gap(len(picks), 1, plot_px=560), "xaxis": {"tickformat": ".0%", "zeroline": True}},
            table_html(picks.to_frame("Contribution"), {"Contribution": pct}),
            height=620,
        )
    )

    to = strat.trades.assign(year=[t.year for t in strat.trades["date"]])
    avg_nav = strat.nav.groupby(strat.nav.index.year).mean()
    ty = to.groupby("year")["value"].sum() / avg_nav.reindex(to["year"].unique()).to_numpy()
    cy = strat.costs.groupby(strat.costs.index.year).sum() / avg_nav
    body.append(
        rep.chart(
            "Turnover per year",
            "Traded value / average portfolio value.",
            [
                {
                    "type": "bar",
                    "x": [str(y) for y in ty.index],
                    "y": list(ty),
                    "name": "Turnover",
                    "marker": {"color": "@series-1", "cornerradius": 4},
                    "hovertemplate": "%{x}: %{y:.1f}×<extra></extra>",
                }
            ],
            {"bargap": bar_gap(len(ty), 1)},
            table_html(
                pd.DataFrame({"Turnover (×)": ty, "Cost drag": cy.reindex(ty.index)}),
                {"Turnover (×)": num, "Cost drag": lambda v: pct(v, 2)},
            ),
        )
    )
    body.append(
        rep.chart(
            "Daily return distribution",
            "Share of days per 0.1% bucket, clipped at ±5%.",
            [
                {
                    "type": "histogram",
                    "x": list(series[k]),
                    "name": k,
                    "histnorm": "probability",
                    "xbins": {"start": -0.05, "end": 0.05, "size": 0.001},
                    "opacity": 0.6,
                    "marker": {"color": f"@{STOCK_SLOT[k]}"},
                    "hovertemplate": f"%{{x}}: %{{y:.1%}} of days<extra>{esc(k)}</extra>",
                }
                for k in focus
            ],
            {
                "barmode": "overlay",
                "bargap": 0.05,
                "xaxis": {"tickformat": ".0%", "range": [-0.05, 0.05]},
                "yaxis": {"tickformat": ".0%"},
            },
            table_html(
                pd.DataFrame({k: {s: stats[k][s] for s in DIST_STATS} for k in focus}),
                {k: num for k in focus},
            ),
        )
    )
    corr = pd.DataFrame(series).corr()
    body.append(
        rep.chart(
            "Correlation matrix",
            "Daily returns.",
            [diverging_heatmap(corr, ".2f", cap=1.0)],
            {"yaxis": {"autorange": "reversed"}},
            table_html(corr, {c: num for c in corr.columns}),
            height=300,
        )
    )
    reg = regime_table(series)
    body.append(
        rep.chart(
            "Stress periods",
            "Total return inside each window (chosen after the fact; descriptive only).",
            [
                {
                    "type": "bar",
                    "x": list(REGIMES),
                    "y": list(reg[reg["Series"] == k]["Return"]),
                    "name": k,
                    "marker": {"color": f"@{STOCK_SLOT[k]}", "cornerradius": 4},
                    "hovertemplate": f"%{{x}}: %{{y:.1%}}<extra>{esc(k)}</extra>",
                }
                for k in series
            ],
            {
                "barmode": "group",
                "bargap": bar_gap(len(REGIMES), 3, wide=True),
                "bargroupgap": 0.08,
                "yaxis": {"tickformat": ".0%"},
            },
            table_html(
                reg.set_index(["Regime", "Series"]).drop(columns="Window"),
                {"Return": pct, "Max drawdown": pct},
            ),
            wide=True,
        )
    )
    st = pd.DataFrame(stats)
    fmts: dict[str, Any] = {
        "CAGR": pct,
        "Volatility": pct,
        "Max drawdown": pct,
        "Positive days": pct,
        "Best day": lambda v: pct(v, 2),
        "Worst day": lambda v: pct(v, 2),
        "Longest drawdown (days)": lambda v: f"{v:,.0f}",
        "Years": lambda v: f"{v:.1f}",
    }
    rows = [[metric] + [fmts.get(str(metric), num)(v) for v in vals] for metric, vals in st.iterrows()]
    body.append(
        rep.card(
            "Summary statistics",
            "After costs; Sharpe and Sortino on returns in excess of T-bills.",
            table_html(pd.DataFrame(rows, columns=["Metric", *st.columns]).set_index("Metric")),
        )
    )
    body.append(
        rep.chart(
            "Slippage sensitivity",
            "Momentum Sharpe for different slippage assumptions (commissions unchanged).",
            [
                line(
                    [f"{c:g} bps" for c in cost["Slippage (bps)"]],
                    cost["Sharpe"],
                    "Sharpe",
                    "series-1",
                    mode="lines+markers",
                )
            ],
            {"yaxis": {"rangemode": "tozero"}},
            table_html(cost.set_index("Slippage (bps)"), {"CAGR": pct, "Sharpe": num}),
        )
    )
    rep.section(
        "backtest",
        "Backtest",
        esc(
            f"S&P 100 momentum (top {cfg.top_n}, keep while ranked ≤ {cfg.hold_rank}, "
            f"SPY {cfg.sma_window}-day "
            f"filter) vs an equal-weight portfolio of the same {len(cfg.universe)} stocks and SPY. "
            f"{len(strat.returns)} trading days, {strat.returns.index[0]:%Y-%m-%d} to "
            f"{strat.returns.index[-1]:%Y-%m-%d}, {len(strat.decisions)} monthly rebalances."
        ),
        body,
    )


def build_stock_report(
    res: dict[str, Any],
    prices: pd.DataFrame,
    cfg: StockConfig,
    uni: pd.DataFrame,
    conf: dict,
    runs: pd.DataFrame,
    fills: pd.DataFrame,
    cost: pd.DataFrame,
    meta: dict[str, str],
) -> str:
    rep = Report()
    stock_backtest_section(rep, res, prices, cfg, uni, cost)
    if meta.get("live") != "off":
        live_section(rep, runs, fills, stock_signal_parity(runs, prices, cfg), conf, monthly_signal=True)
    notes = [
        "Survivorship bias: the universe is today's S&P 100 applied to the past; absolute returns are an "
        "upper bound and momentum is flattered most. Point-in-time membership is a planned data upgrade.",
        "Rules and parameters were fixed in STOCK_STRATEGY.md before the first backtest; the backtest was "
        "run once and nothing was tuned on it.",
        "Timing: signal on each month-end close, trade at the next day's close (live: about 10:30 New York).",
        "Costs: IBKR fixed commissions ($0.005/share, $1 minimum, 1% maximum) plus 5 bps slippage; "
        "shares are estimated from adjusted prices.",
        "Momentum concentrates by sector (see sector exposure); there is no sector cap.",
        "The pre-registered H1/H2 ETF strategies are deliberately absent (evaluated once by prereg_test.py).",
        "Paper fills are optimistic: no queue position, no market impact; IBKR prices in the logs "
        "are delayed.",
    ]
    return render_page(rep, "S&P 100 momentum: backtest and live report", notes, meta)


def live_logs(kind: str, skip: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs and fills of one bot only: the stock report shows stockbot-/unwind- fills and
    stock runs; the ETF report the rest. Mixing them made the signal re-check meaningless."""
    if skip:
        return pd.DataFrame(), pd.DataFrame()
    runs = load_runs(ROOT / "logs" / "bot.jsonl")
    fills = load_fills(ROOT / "journal.jsonl")
    if len(runs):
        runs = runs[runs["kind"] == kind].reset_index(drop=True)
    if len(fills):
        stock_fill = fills["ref"].str.startswith(("stockbot-", "unwind-"))
        fills = fills[stock_fill if kind == "stocks" else ~stock_fill].reset_index(drop=True)
    return runs, fills


def main_stocks(args: argparse.Namespace, conf: dict) -> None:
    from backtest_stocks import CACHE, START, stock_config
    from data import load_stock_prices
    from universe import load_universe

    uni = load_universe(ROOT / conf.get("stocks", {}).get("universe_file", "universe/sp100.csv"))
    cfg = stock_config(conf, uni)
    runs, fills = live_logs("stocks", args.no_live)
    ymap = dict(zip(uni["symbol"], uni["yahoo"], strict=True)) | {cfg.market: cfg.market, "BIL": "BIL"}
    print("Loading stock prices...")
    prices = load_stock_prices(ymap, START, CACHE, args.refresh)
    newest_signal = (
        runs["signal_date"].dropna().max() if len(runs) and runs["signal_date"].notna().any() else None
    )
    if newest_signal is not None and newest_signal > prices.index[-1]:
        prices = load_stock_prices(ymap, START, CACHE, refresh=True)
    cash = prices.pop("BIL").pct_change()
    print("Running backtests...")
    res = run_all(prices, cash, cfg)
    cost = stock_cost_sensitivity(prices, cash, cfg, res["strategy"])
    params = ", ".join(f"{k}={v}" for k, v in vars(cfg).items() if k != "universe")
    meta = {
        "Code": code_version(ROOT),
        "Generated": f"{dt.datetime.now(WARSAW):%Y-%m-%d %H:%M} Warsaw",
        "Data": f"Yahoo, {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}",
        "Universe": f"S&P 100 as of {uni['as_of'].iloc[0]} ({len(uni)} stocks; survivorship-biased)",
        "Parameters": params,
        "Cost model": "IBKR fixed + 5 bps slippage, $25k start",
        "live": "off" if args.no_live else "on",
    }
    page = build_stock_report(res, prices, cfg, uni, conf, runs, fills, cost, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"Wrote {args.out} ({len(page) / 1e6:.1f} MB). Open it in a browser.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--strategy", choices=["stocks", "etf"], default=None, help="default: [bot] strategy in config.toml"
    )
    ap.add_argument("--universe", choices=["proxy", "ucits"], default="proxy", help="ETF mode only")
    ap.add_argument("--source", choices=["yahoo", "ibkr"], default="yahoo")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--refresh", action="store_true", help="re-download cached prices")
    ap.add_argument("--no-live", action="store_true", help="skip the live section")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "report.html")
    args = ap.parse_args()

    conf, cfg = load_config()
    if (args.strategy or conf.get("bot", {}).get("strategy", "etf")) == "stocks":
        main_stocks(args, conf)
        return
    proxy = args.universe == "proxy"
    print("Loading backtest prices...")
    prices = load_backtest_prices(conf, cfg, proxy, args.source, args.start, args.refresh)
    runs, fills = live_logs("etf", args.no_live)
    live_prices = None
    if not args.no_live and not runs.empty:
        from data import load_prices

        print("Loading UCITS prices for the signal re-check...")
        live_prices = load_prices({s: conf["universe"][s][3] for s in cfg.assets}, period="3y")
    label = ("US proxies " + "/".join(conf["proxy"][a] for a in cfg.assets)) if proxy else "UCITS ETFs"
    params = ", ".join(f"{k}={v}" for k, v in vars(cfg).items() if k not in ("risky", "safe"))
    meta = {
        "Code": code_version(ROOT),
        "Generated": f"{dt.datetime.now(WARSAW):%Y-%m-%d %H:%M} Warsaw",
        "Data": f"{label} via {args.source}, {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}",
        "Universe": ", ".join(cfg.risky) + f" + {cfg.safe} (T-bills)",
        "Parameters": params,
        "Cost model": f"{args.cost_bps:g} bps × turnover",
        "live": "off" if args.no_live else "on",
    }
    asset_labels = {a: f"{a} (as {conf['proxy'][a]})" if proxy else a for a in cfg.assets}
    page = build_report(prices, cfg, conf, args.cost_bps, label, runs, fills, live_prices, meta, asset_labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"Wrote {args.out} ({len(page) / 1e6:.1f} MB). Open it in a browser.")


if __name__ == "__main__":
    main()
