#!/usr/bin/env python3
"audit/run_audit.py - the backtest audit checklist (see audit/AUDIT_REPORT.md), end to end."

import argparse
import datetime as dt
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402
from analytics import perf_stats  # noqa: E402
from backtest_stocks import (  # noqa: E402
    CACHE,
    CAPITAL,
    DEFAULT_COSTS,
    MIN_TRADE,
    START,
    CostModel,
    StockBacktest,
    equal_weight_fn,
    evaluation_start,
    run_monthly,
    stock_config,
    strategy_fn,
)
from config import load_config  # noqa: E402
from data import load_stock_prices  # noqa: E402
from stock_strategy import StockConfig, decide, market_states, momentum, month_ends, select  # noqa: E402
from strategy import held_set, needs_rebalance, signal_frames, weights_from_signals  # noqa: E402
from universe import load_universe  # noqa: E402

ZERO = CostModel(per_share=0.0, min_order=0.0, max_pct=0.0, slippage_bps=0.0)
DOUBLE = CostModel(per_share=0.010, min_order=2.0, max_pct=0.02, slippage_bps=10.0)
R: dict[str, Any] = {}


def head(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}", flush=True)


def m(r: pd.Series, rf: pd.Series) -> dict[str, float]:
    s = perf_stats(r, rf.reindex(r.index).fillna(0.0))
    return {
        k: round(float(s[k]), 4)
        for k in ("CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar")
    }


def run_monthly_lag(
    prices: pd.DataFrame,
    cash_ret: pd.Series,
    decide_fn: Any,
    symbols: list[str],
    start: pd.Timestamp,
    costs: CostModel = DEFAULT_COSTS,
    capital: float = CAPITAL,
    min_trade: float = MIN_TRADE,
    lag: int = 1,
) -> StockBacktest:
    """Copy of backtest_stocks.run_monthly; only difference: execute `lag` trading days after
    the month-end (original: lag=1). Verified below to reproduce the original at lag=1."""
    px = prices[symbols]
    days = prices.index[prices.index >= start]
    signal_for = {}
    for s in month_ends(prices.index):
        after = prices.index[prices.index > s]
        if len(after) >= lag and after[lag - 1] >= start:
            signal_for[after[lag - 1]] = s
    h = np.zeros(len(symbols))
    cash = capital
    prev: np.ndarray | None = None
    navs, wrows, cost_days, trades, decisions = [], [], [], [], []
    for d in days:
        p = px.loc[d].to_numpy(dtype=float)
        if prev is not None:
            ok = np.isfinite(p) & np.isfinite(prev) & (prev > 0)
            h = h * np.where(ok, p / np.where(ok, prev, 1.0), 1.0)
            r = cash_ret.get(d, 0.0)
            cash *= 1 + (0.0 if pd.isna(r) else float(r))
        day_cost = 0.0
        if d in signal_for:
            held = {symbols[i] for i in np.flatnonzero(h > 0)}
            w, info = decide_fn(prices.loc[: signal_for[d]], held, d.date())
            nav = h.sum() + cash
            target = w.reindex(symbols).fillna(0.0).to_numpy(dtype=float) * nav
            delta = target - h
            doable = np.isfinite(p) & (np.abs(delta) >= min_trade)
            for i in np.flatnonzero(doable):
                c = costs.order_cost(abs(delta[i]), p[i])
                day_cost += c
                trades.append(
                    {
                        "date": d,
                        "symbol": symbols[i],
                        "side": "BUY" if delta[i] > 0 else "SELL",
                        "value": abs(delta[i]),
                        "cost": c,
                    }
                )
            h[doable] = target[doable]
            cash = nav - h.sum() - day_cost
            decisions.append({"signal_date": signal_for[d], "exec_date": d, **info})
        nav = h.sum() + cash
        navs.append(nav)
        wrows.append(np.append(h / nav, cash / nav))
        cost_days.append(day_cost)
        prev = np.where(np.isfinite(p), p, prev if prev is not None else p)
    nav_s = pd.Series(navs, index=days)
    rets = nav_s.pct_change()
    rets.iloc[0] = nav_s.iloc[0] / capital - 1
    return StockBacktest(
        returns=rets,
        nav=nav_s,
        weights=pd.DataFrame(wrows, index=days, columns=[*symbols, "CASH"]),
        trades=pd.DataFrame(trades, columns=["date", "symbol", "side", "value", "cost"]),
        decisions=pd.DataFrame(decisions),
        costs=pd.Series(cost_days, index=days),
    )


def run_backtest_lag(
    prices: pd.DataFrame, cfg: Any, cost_bps: float = 10.0, lag: int = 0, weight_fn: Any = None
) -> tuple[pd.Series, pd.DataFrame, float, int]:
    """Copy of backtest.run_backtest; differences: signals can be lagged `lag` extra bars
    (original: 0 = trade on the signal bar's close) and the weight rule can be swapped
    (for sanity tests). Verified below to reproduce the original at lag=0."""
    weight_fn = weight_fn or weights_from_signals
    prices = prices[list(cfg.assets)].dropna()
    rets = prices.pct_change().fillna(0.0)
    mom, ratio = signal_frames(prices, cfg)
    if lag:
        mom, ratio = mom.shift(lag), ratio.shift(lag)
    w = pd.Series(0.0, index=list(cfg.assets))
    w[cfg.safe] = 1.0
    port, weights, turnover, n_reb = [], [], [], 0
    for i in range(len(prices)):
        r = rets.iloc[i]
        day_ret = float((w * r).sum())
        w = w * (1 + r) / (1 + day_ret)
        cost = 0.0
        if i >= cfg.warmup + lag:
            held = held_set(w, cfg.risky)
            tgt = weight_fn(mom.iloc[i], ratio.iloc[i], cfg, held)
            if needs_rebalance(w, tgt, cfg.band):
                to = float((tgt - w).abs().sum())
                cost = to * cost_bps / 1e4
                turnover.append(to)
                n_reb += 1
                w = tgt.copy()
        port.append(day_ret - cost)
        weights.append(w.copy())
    idx = prices.index
    live = idx[cfg.warmup :]
    r_s = pd.Series(port, index=idx).loc[live]
    return r_s, pd.DataFrame(weights, index=idx).loc[live], sum(turnover), n_reb


def data_integrity(prices: pd.DataFrame, name: str, events_limit: int = 60) -> dict:
    idx = prices.index
    first = prices.apply(lambda c: c.first_valid_index())
    last = prices.apply(lambda c: c.last_valid_index())
    interior_gaps = int(sum(prices[c].loc[first[c] : last[c]].isna().sum() for c in prices.columns))
    trailing = [c for c in prices.columns if last[c] < idx[-1]]
    ret = prices / prices.shift(1) - 1
    big = ret.stack()
    big = big[big.abs() > 0.20].sort_values()
    events = [
        {
            "symbol": s,
            "date": str(d.date()),
            "move": round(float(v), 4),
            "prev": round(float(prices[s].shift(1).loc[d]), 4),
            "close": round(float(prices[s].loc[d]), 4),
        }
        for (d, s), v in big.items()
    ]

    alive = prices.notna().sum(axis=1)
    expected = pd.Series([int(((first <= d) & (last >= d)).sum()) for d in idx], index=idx)
    misaligned = int(((alive < 0.9 * expected) & (expected > 5)).sum())
    out = {
        "rows": len(idx),
        "columns": prices.shape[1],
        "start": str(idx[0].date()),
        "end": str(idx[-1].date()),
        "sorted": bool(idx.is_monotonic_increasing),
        "unique": bool(idx.is_unique),
        "tz": str(idx.tz),
        "interior_nan_cells": interior_gaps,
        "trailing_nan_columns": trailing,
        "leading_nan_columns": int((first > idx[0]).sum()),
        "n_moves_gt_20pct": len(events),
        "moves_gt_20pct": events[:events_limit],
        "misaligned_rows": misaligned,
    }
    print(
        f"[{name}] {out['rows']} rows x {out['columns']} cols {out['start']}..{out['end']} | sorted "
        f"{out['sorted']} unique {out['unique']} tz {out['tz']} | interior NaN cells {interior_gaps} | "
        f"trailing-NaN cols {trailing} | late listings {out['leading_nan_columns']} | "
        f">20% moves {len(events)} | misaligned rows {misaligned}"
    )
    return out


def ibkr_crosscheck(events: list[dict], conf: dict, uni: pd.DataFrame) -> list[dict]:
    """Recompute flagged daily moves from IBKR's own adjusted daily bars (read-only)."""
    from ib_async import IB, Stock, util

    to_ib = dict(zip(uni["symbol"], uni["ib_symbol"], strict=True)) | {"SPY": "SPY", "BIL": "BIL"}
    ib = IB()
    ib.connect(conf["ib"]["host"], conf["ib"]["port"], clientId=43, timeout=20, readonly=True)
    out: list[dict] = []
    cache: dict[str, pd.Series | None] = {}
    try:
        for e in events:
            s = e["symbol"]
            if s not in cache:
                c = Stock(to_ib.get(s, s), "SMART", "USD")
                if not ib.qualifyContracts(c):
                    cache[s] = None
                    continue
                bars = ib.reqHistoricalData(
                    c,
                    endDateTime="",
                    durationStr="20 Y",
                    barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,
                    formatDate=1,
                    timeout=120,
                )
                df = util.df(bars) if bars else None
                cache[s] = (
                    None if df is None else pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
                )
                ib.sleep(1)
            ser = cache[s]
            d = pd.Timestamp(e["date"])
            if ser is None or d not in ser.index:
                out.append({**e, "ibkr_move": None})
                continue
            i = ser.index.get_loc(d)
            mv = float(ser.iloc[i] / ser.iloc[i - 1] - 1) if i > 0 else None
            out.append({**e, "ibkr_move": None if mv is None else round(mv, 4)})
    finally:
        ib.disconnect()
    return out


def independent_metrics(nav: pd.Series, capital: float, rf: pd.Series) -> dict[str, float]:
    """From the equity curve only, numpy, written separately from analytics.perf_stats.
    The starting capital counts as the first peak."""
    v = np.concatenate([[capital], nav.to_numpy(dtype=float)])
    r = v[1:] / v[:-1] - 1
    years = len(r) / 252
    cagr = (v[-1] / v[0]) ** (1 / years) - 1
    ex = r - rf.reindex(nav.index).fillna(0.0).to_numpy()
    sharpe = ex.mean() / ex.std(ddof=1) * math.sqrt(252)
    sortino = ex.mean() / math.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * math.sqrt(252)
    mdd = float(np.min(v / np.maximum.accumulate(v) - 1))
    return {
        "CAGR": round(cagr, 4),
        "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4),
        "Max drawdown": round(mdd, 4),
        "Calmar": round(cagr / abs(mdd), 4),
    }


def compare(pipe: dict, indep: dict) -> dict[str, float]:
    return {k: round(abs(pipe[k] - indep[k]), 6) for k in indep}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibkr", action="store_true", help="cross-check >20%% moves against IBKR bars")
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    t0 = time.time()
    conf, etf_cfg = load_config()
    uni = load_universe()
    cfg = stock_config(conf, uni)
    ymap = dict(zip(uni["symbol"], uni["yahoo"], strict=True)) | {"SPY": "SPY", "BIL": "BIL"}
    allp = load_stock_prices(ymap, START, CACHE)
    cash = allp["BIL"] / allp["BIL"].shift(1) - 1
    prices = allp.drop(columns="BIL")
    symbols = list(cfg.universe)
    start = evaluation_start(prices, cash, cfg)
    etf = backtest.load_backtest_prices(conf, etf_cfg, proxy=True, source="yahoo", start="2005-01-01")
    etf_rf = etf[etf_cfg.safe].pct_change()
    R["meta"] = {
        "run": dt.datetime.now().isoformat(timespec="seconds"),
        "stock_start": str(start.date()),
        "stock_data_end": str(prices.index[-1].date()),
        "etf_data": [str(etf.index[0].date()), str(etf.index[-1].date())],
    }

    head("1. Data integrity")
    R["data"] = {
        "stocks": data_integrity(allp, "stocks+SPY+BIL"),
        "etf_proxies": data_integrity(etf, "ETF proxies"),
    }
    if args.ibkr:
        ev = R["data"]["stocks"]["moves_gt_20pct"]
        print(f"Cross-checking {len(ev)} flagged stock moves against IBKR adjusted bars...")
        R["data"]["ibkr_crosscheck"] = ibkr_crosscheck(ev, conf, uni)
        agree = [
            e
            for e in R["data"]["ibkr_crosscheck"]
            if e["ibkr_move"] is not None and abs(e["ibkr_move"] - e["move"]) < 0.02
        ]
        print(f"  IBKR agrees within 2 pts on {len(agree)}/{len(R['data']['ibkr_crosscheck'])}")

    head("Base runs (original code)")
    base = run_monthly(prices, cash, strategy_fn(cfg), symbols, start)
    ew = run_monthly(prices, cash, equal_weight_fn(cfg), symbols, start)
    spy = prices["SPY"].pct_change().loc[base.returns.index].fillna(0.0)
    spy.iloc[0] = 0.0
    rf = cash.loc[base.returns.index].fillna(0.0)
    R["base"] = {"momentum": m(base.returns, rf), "equal_weight": m(ew.returns, rf), "spy": m(spy, rf)}
    print(
        "momentum",
        R["base"]["momentum"],
        "\nequal-wt",
        R["base"]["equal_weight"],
        "\nSPY     ",
        R["base"]["spy"],
    )
    e_r, e_w, e_to, e_n = backtest.run_backtest(etf, etf_cfg, 10.0)
    R["base"]["etf_dual_momentum"] = m(e_r, etf_rf)
    print("ETF dual momentum (proxies)", R["base"]["etf_dual_momentum"])

    lag1 = run_monthly_lag(prices, cash, strategy_fn(cfg), symbols, start, lag=1)
    l0 = run_backtest_lag(etf, etf_cfg, 10.0, lag=0)
    R["scratch_copies_identical"] = {
        "stock_engine_lag1": bool(np.allclose(lag1.nav.to_numpy(), base.nav.to_numpy(), rtol=0, atol=1e-9)),
        "etf_engine_lag0": bool(np.allclose(l0[0].to_numpy(), e_r.to_numpy(), rtol=0, atol=1e-15)),
    }
    print("scratch copies reproduce originals:", R["scratch_copies_identical"])

    head("2. Look-ahead: truncation and future-perturbation tests")
    idx = base.weights.index
    cuts = [idx[int(f * (len(idx) - 1))] for f in (0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.98)]
    trunc = []
    for t in cuts:
        tr = run_monthly(prices.loc[:t], cash.loc[:t], strategy_fn(cfg), symbols, start)
        diff = (base.weights.loc[:t] - tr.weights).abs().max(axis=1)
        bad = diff[diff > 1e-12]
        trunc.append(
            {
                "t": str(t.date()),
                "max_abs_diff": float(diff.max()),
                "first_diff": None if bad.empty else str(bad.index[0].date()),
            }
        )
    R["lookahead"] = {"stock_truncation": trunc}
    print("stock truncation:", [(x["t"], x["max_abs_diff"], x["first_diff"]) for x in trunc])
    rng = np.random.default_rng(20261003)
    pert = []
    for t in cuts[1::2]:
        shocked = prices.copy()
        after = shocked.index > t
        noise = np.exp(np.cumsum(rng.normal(0, 0.03, (after.sum(), shocked.shape[1])), axis=0))
        shocked.loc[after] = shocked.loc[after].to_numpy() * noise
        c2 = cash.copy()
        c2.loc[c2.index > t] = rng.normal(0, 0.01, (c2.index > t).sum())
        alt = run_monthly(shocked, c2, strategy_fn(cfg), symbols, start)
        diff = (base.weights.loc[:t] - alt.weights.loc[:t]).abs().max(axis=1)
        bad = diff[diff > 1e-12]
        pert.append(
            {
                "t": str(t.date()),
                "max_abs_diff": float(diff.max()),
                "first_diff": None if bad.empty else str(bad.index[0].date()),
            }
        )
    R["lookahead"]["stock_perturbation"] = pert
    print("stock perturbation:", [(x["t"], x["max_abs_diff"], x["first_diff"]) for x in pert])

    e_idx = e_w.index
    e_trunc, e_pert = [], []
    for f in (0.1, 0.35, 0.6, 0.85, 0.97):
        t = e_idx[int(f * (len(e_idx) - 1))]
        _, w_t, _, _ = backtest.run_backtest(etf.loc[:t], etf_cfg, 10.0)
        diff = (e_w.loc[:t] - w_t).abs().max(axis=1)
        e_trunc.append({"t": str(t.date()), "max_abs_diff": float(diff.max())})
        shocked = etf.copy()
        after = shocked.index > t
        shocked.loc[after] = shocked.loc[after].to_numpy() * np.exp(
            np.cumsum(rng.normal(0, 0.02, (after.sum(), shocked.shape[1])), axis=0)
        )
        _, w_p, _, _ = backtest.run_backtest(shocked, etf_cfg, 10.0)
        e_pert.append(
            {"t": str(t.date()), "max_abs_diff": float((e_w.loc[:t] - w_p.loc[:t]).abs().max().max())}
        )
    R["lookahead"]["etf_truncation"], R["lookahead"]["etf_perturbation"] = e_trunc, e_pert
    print("ETF truncation:", [(x["t"], x["max_abs_diff"]) for x in e_trunc])
    print("ETF perturbation:", [(x["t"], x["max_abs_diff"]) for x in e_pert])

    import prereg_test as pr

    srng = np.random.default_rng(7)
    sd = pd.bdate_range("2010-01-01", periods=900)
    syn = pd.DataFrame(
        {
            "EQ": 100 * np.exp(np.cumsum(3e-4 + 0.011 * srng.standard_normal(900))),
            "BD": 100 * np.exp(np.cumsum(1e-4 + 0.007 * srng.standard_normal(900))),
            "CASH": 100 * np.exp(np.cumsum(np.full(900, 5e-5))),
        },
        index=sd,
    )
    srets = syn.pct_change().fillna(0.0)
    vt_full, tf_full = pr.targets_vol_target(srets), pr.targets_trend_filter(syn, pr.WARMUP - 1)
    pre = []
    for t in (400, 600, 800):
        vt = pr.targets_vol_target(srets.iloc[:t])
        tf = pr.targets_trend_filter(syn.iloc[:t], pr.WARMUP - 1)
        sh = syn.copy()
        sh.iloc[t:] *= np.exp(np.cumsum(srng.normal(0, 0.03, (900 - t, 3)), axis=0))
        vt_p = pr.targets_vol_target(sh.pct_change().fillna(0.0))
        tf_p = pr.targets_trend_filter(sh, pr.WARMUP - 1)
        pre.append(
            {
                "t": t,
                "vol_target_trunc": float((vt_full.iloc[:t] - vt).abs().max().max()),
                "trend_trunc": float((tf_full.iloc[:t] - tf).abs().max().max()),
                "vol_target_perturb": float((vt_full.iloc[:t] - vt_p.iloc[:t]).abs().max().max()),
                "trend_perturb": float((tf_full.iloc[:t] - tf_p.iloc[:t]).abs().max().max()),
            }
        )
    R["lookahead"]["prereg_synthetic"] = pre
    print("prereg H1/H2 rules (synthetic):", pre)

    head("3. Execution and costs")
    tr = base.trades
    sw = base.weights.drop(columns="CASH")
    R["execution"] = {
        "n_orders": len(tr),
        "n_buy": int((tr["side"] == "BUY").sum()),
        "n_sell": int((tr["side"] == "SELL").sum()),
        "cost_on_both_sides": bool(
            (tr.loc[tr.side == "BUY", "cost"] > 0).all() and (tr.loc[tr.side == "SELL", "cost"] > 0).all()
        ),
        "total_costs_usd": round(float(tr["cost"].sum()), 2),
        "min_cash_weight": round(float(base.weights["CASH"].min()), 6),
        "max_gross_exposure": round(float(sw.sum(axis=1).max()), 6),
        "max_single_weight": round(float(sw.max().max()), 4),
        "median_order_usd": round(float(tr["value"].median()), 2),
        "max_order_usd": round(float(tr["value"].max()), 2),
        "final_nav_usd": round(float(base.nav.iloc[-1]), 2),
    }

    R["execution"]["etf_cost_formula_two_legs"] = (
        "turnover = sum|target - drifted weights| (both legs); cost = turnover x bps"
    )
    print(R["execution"])

    head("4. Metrics: independent recomputation from the equity curve")
    ind = independent_metrics(base.nav, CAPITAL, cash)
    pipe = m(base.returns, rf)
    eq_nav = (1 + e_r).cumprod()
    ind_e = independent_metrics(eq_nav, 1.0, etf_rf)
    R["metrics"] = {
        "stock_pipeline": pipe,
        "stock_independent": ind,
        "stock_abs_diff": compare(pipe, ind),
        "etf_pipeline": R["base"]["etf_dual_momentum"],
        "etf_independent": ind_e,
        "etf_abs_diff": compare(R["base"]["etf_dual_momentum"], ind_e),
        "annualization": 252,
    }
    print("stock |pipeline - independent|:", R["metrics"]["stock_abs_diff"])
    print("ETF   |pipeline - independent|:", R["metrics"]["etf_abs_diff"])

    head("5. Sanity tests")
    S: dict[str, Any] = {}
    one = run_monthly(
        prices, cash, lambda h, held, d: (pd.Series({"SPY": 1.0}), {}), ["SPY"], start, ZERO, min_trade=0.0
    )
    spy_bh = prices["SPY"].loc[start:] / prices["SPY"].loc[start]
    S["stock_always_long_spy_vs_bh_max_abs_diff"] = float((one.nav / CAPITAL - spy_bh).abs().max())
    zero = run_monthly(
        prices, cash * 0.0, lambda h, held, d: (pd.Series(dtype=float), {}), symbols, start, ZERO
    )
    S["stock_zero_position_nav_range"] = float(zero.nav.max() - zero.nav.min())

    def all_in(asset: str) -> Any:
        def fn(mom_row: pd.Series, ratio_row: pd.Series, c: Any, held: Any = frozenset()) -> pd.Series:
            w = pd.Series(0.0, index=list(c.assets))
            w[asset] = 1.0
            return w

        return fn

    r_long, _, _, _ = run_backtest_lag(etf, etf_cfg, 0.0, weight_fn=all_in("CSPX"))
    bh = etf["CSPX"].pct_change().loc[r_long.index]
    S["etf_always_long_vs_bh_max_abs_diff"] = float((r_long.iloc[1:] - bh.iloc[1:]).abs().max())
    flat = etf.copy()
    flat[etf_cfg.safe] = 100.0
    r_flat, _, _, _ = run_backtest_lag(flat, etf_cfg, 0.0, weight_fn=all_in(etf_cfg.safe))
    S["etf_zero_position_max_abs_return"] = float(r_flat.abs().max())
    print({k: v for k, v in S.items()})

    mom_all = momentum(prices[symbols], cfg)
    elig = mom_all.notna() & prices[symbols].notna()
    ends = month_ends(prices.index)
    states = market_states(prices[cfg.market], ends, cfg)

    def random_fn(seed: int, use_filter: bool) -> Any:
        g = np.random.default_rng(seed)

        def fn(hist: pd.DataFrame, held: set, today: dt.date) -> tuple[pd.Series, dict]:
            s = hist.index[-1]
            w = pd.Series(0.0, index=symbols)
            if use_filter and not bool(states.get(s, True)):
                return w, {"market_on": False}
            row = elig.loc[s]
            cand = row[row].index.to_numpy()
            w[g.choice(cand, size=min(10, len(cand)), replace=False)] = 0.1
            return w, {"market_on": True}

        return fn

    ew_sharpe = R["base"]["equal_weight"]["Sharpe"]
    rand = {}
    for use_filter in (False, True):
        for label, costs in (("gross", ZERO), ("net", DEFAULT_COSTS)):
            sh = []
            for seed in range(args.seeds):
                bt = run_monthly(prices, cash, random_fn(seed, use_filter), symbols, start, costs)
                sh.append(float(perf_stats(bt.returns, rf)["Sharpe"]))
            key = f"{'filter' if use_filter else 'always_in'}_{label}"
            rand[key] = {
                "mean": round(float(np.mean(sh)), 3),
                "std": round(float(np.std(sh)), 3),
                "p05": round(float(np.percentile(sh, 5)), 3),
                "p95": round(float(np.percentile(sh, 95)), 3),
                "max": round(float(np.max(sh)), 3),
            }
            print(f"random {key}: {rand[key]}", flush=True)
    mom_net, mom_gross = R["base"]["momentum"]["Sharpe"], None
    gross_base = run_monthly(prices, cash, strategy_fn(cfg), symbols, start, ZERO)
    mom_gross = float(perf_stats(gross_base.returns, rf)["Sharpe"])
    S["random_stock"] = rand
    S["random_reference"] = {
        "equal_weight_sharpe_net": ew_sharpe,
        "momentum_sharpe_net": mom_net,
        "momentum_sharpe_gross": round(mom_gross, 4),
    }

    e_sh: dict[str, list[float]] = {"gross": [], "net": []}
    for seed in range(min(args.seeds, 30)):
        g = np.random.default_rng(1000 + seed)

        def rnd(
            mom_row: pd.Series, ratio_row: pd.Series, c: Any, held: Any = frozenset(), g: Any = g
        ) -> pd.Series:
            w = pd.Series(0.0, index=list(c.assets))
            picks = g.choice(list(c.risky), size=c.top_n, replace=False)
            w[list(picks)] = 1.0 / c.top_n
            return w

        for label, bps in (("gross", 0.0), ("net", 10.0)):
            g2 = np.random.default_rng(1000 + seed)

            def rnd2(
                mom_row: pd.Series, ratio_row: pd.Series, c: Any, held: Any = frozenset(), g: Any = g2
            ) -> pd.Series:
                return rnd(mom_row, ratio_row, c, held, g)

            rr, _, _, _ = run_backtest_lag(etf, etf_cfg, bps, weight_fn=rnd2)
            e_sh[label].append(float(perf_stats(rr, etf_rf)["Sharpe"]))
    S["random_etf"] = {
        k: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)} for k, v in e_sh.items()
    }
    print("random ETF:", S["random_etf"], flush=True)

    def inverted_fn(invert_selection: bool, invert_filter: bool) -> Any:
        def fn(hist: pd.DataFrame, held: set, today: dt.date) -> tuple[pd.Series, dict]:
            d = decide(hist, cfg, held, today)
            on = (not d.market_on) if invert_filter else d.market_on
            uni_cols = [s for s in cfg.universe if s in hist.columns]
            mom = momentum(hist[uni_cols], cfg).iloc[-1]
            mom = mom[hist[uni_cols].iloc[-1].notna()]
            w = pd.Series(0.0, index=uni_cols)
            if on:
                picks, _ = select(-mom if invert_selection else mom, held, cfg)
                w[picks] = 1.0 / cfg.top_n
            return w, {"market_on": on}

        return fn

    inv = {}
    for name, a, b in (("bottom10_same_filter", True, False), ("top10_inverted_filter", False, True)):
        bt = run_monthly(prices, cash, inverted_fn(a, b), symbols, start, ZERO)
        inv[name] = m(bt.returns, rf)
    ew_gross = run_monthly(prices, cash, equal_weight_fn(cfg), symbols, start, ZERO)
    inv["momentum_gross"] = m(gross_base.returns, rf)
    inv["equal_weight_gross"] = m(ew_gross.returns, rf)
    S["inverted_stock"] = inv
    print("inverted (gross):", inv)

    dbl = run_monthly(prices, cash, strategy_fn(cfg), symbols, start, DOUBLE)
    e_dbl, _, _, _ = backtest.run_backtest(etf, etf_cfg, 20.0)
    S["double_costs"] = {
        "stock_base": R["base"]["momentum"],
        "stock_double": m(dbl.returns, rf),
        "stock_gross": inv["momentum_gross"],
        "etf_10bps": R["base"]["etf_dual_momentum"],
        "etf_20bps": m(e_dbl, etf_rf),
    }
    print("double costs:", S["double_costs"])

    lag2 = run_monthly_lag(prices, cash, strategy_fn(cfg), symbols, start, lag=2)
    S["extra_lag"] = {"stock_lag1_original": R["base"]["momentum"], "stock_lag2": m(lag2.returns, rf)}
    for lg in (1, 2):
        rr, _, _, _ = run_backtest_lag(etf, etf_cfg, 10.0, lag=lg)
        S["extra_lag"][f"etf_lag{lg}"] = m(rr, etf_rf)
    S["extra_lag"]["etf_lag0_original"] = R["base"]["etf_dual_momentum"]
    print("extra lag:", S["extra_lag"])
    R["sanity"] = S

    head("6. Overfitting and validation")
    O: dict[str, Any] = {}
    r = base.returns
    O["subperiods"] = {
        "2007-2016": m(r.loc[:"2016-12-31"], rf),
        "2017-2026": m(r.loc["2017-01-01":], rf),
        "ew_2007-2016": m(ew.returns.loc[:"2016-12-31"], rf),
        "ew_2017-2026": m(ew.returns.loc["2017-01-01":], rf),
    }
    grid = []
    gstart = evaluation_start(prices, cash, StockConfig(**{**vars(cfg), "mom_lookback": 315}))
    O["grid_start"] = str(gstart.date())
    for L in (126, 189, 252, 315):
        for N in (5, 10, 15, 20):
            c = StockConfig(**{**vars(cfg), "mom_lookback": L, "top_n": N, "hold_rank": 2 * N})
            bt = run_monthly(prices, cash, strategy_fn(c), symbols, gstart)
            st = m(bt.returns, rf)
            grid.append(
                {
                    "mom_lookback": L,
                    "top_n": N,
                    "Sharpe": st["Sharpe"],
                    "CAGR": st["CAGR"],
                    "Max drawdown": st["Max drawdown"],
                }
            )
        print(f"grid lookback {L} done", flush=True)
    O["grid_lookback_x_topn"] = grid
    one_d = []
    for key, vals in (("sma_window", (100, 150, 200, 250)), ("hold_rank", (10, 15, 20, 30, 40))):
        for v in vals:
            c = StockConfig(**{**vars(cfg), key: v})
            bt = run_monthly(
                prices, cash, strategy_fn(c), symbols, max(start, evaluation_start(prices, cash, c))
            )
            st = m(bt.returns, rf)
            one_d.append(
                {
                    "param": key,
                    "value": v,
                    "Sharpe": st["Sharpe"],
                    "CAGR": st["CAGR"],
                    "Max drawdown": st["Max drawdown"],
                }
            )
    c = StockConfig(**{**vars(cfg), "exit_ratio": 0.0, "enter_ratio": 0.0})
    bt = run_monthly(prices, cash, strategy_fn(c), symbols, start)
    st = m(bt.returns, rf)
    one_d.append(
        {
            "param": "market_filter",
            "value": "off",
            "Sharpe": st["Sharpe"],
            "CAGR": st["CAGR"],
            "Max drawdown": st["Max drawdown"],
        }
    )
    O["one_dimensional"] = one_d
    print("1-D sensitivity:", one_d)
    O["n_rebalances"] = len(base.decisions)
    O["n_orders"] = len(base.trades)
    O["etf_n_rebalances"] = e_n
    stock_w = base.weights.drop(columns="CASH")
    rets = prices[symbols].pct_change().reindex(stock_w.index)
    contrib = (stock_w.shift(1) * rets).sum().sort_values(ascending=False)
    O["top_contributors"] = {k: round(float(v), 4) for k, v in contrib.head(10).items()}
    O["top5_share_of_total_contribution"] = round(float(contrib.head(5).sum() / contrib.sum()), 4)
    top5 = list(contrib.head(5).index)
    c = StockConfig(**{**vars(cfg), "universe": tuple(s for s in cfg.universe if s not in top5)})
    bt = run_monthly(prices, cash, strategy_fn(c), list(c.universe), start)
    O["without_top5_contributors"] = {"excluded": top5, **m(bt.returns, rf)}
    ew_wo = run_monthly(prices, cash, equal_weight_fn(c), list(c.universe), start)
    O["ew_without_top5"] = m(ew_wo.returns, rf)
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    best = int(yearly.idxmax())
    keep = r[r.index.year != best]
    O["yearly"] = {int(k): round(float(v), 4) for k, v in yearly.items()}
    O["cagr_excluding_best_year"] = {
        "best_year": best,
        "CAGR": round(float((1 + keep).prod() ** (252 / len(keep)) - 1), 4),
    }
    print(
        "concentration:",
        O["top5_share_of_total_contribution"],
        O["without_top5_contributors"],
        O["cagr_excluding_best_year"],
    )
    R["overfit"] = O

    head("7. Visualization correctness")
    V: dict[str, Any] = {}
    page_path = ROOT / "reports" / "report.html"
    if page_path.exists():
        page = page_path.read_text()
        found = re.search(r'id="figs">(.*?)</script>', page, re.S)
        assert found, "figure data block missing from report.html"
        figs = json.loads(found.group(1).replace("<\\/", "</"))
        growth = next(
            f
            for f in figs
            if any(t.get("name") == "Momentum" for t in f["data"])
            and f["layout"].get("yaxis", {}).get("type") == "log"
        )
        mom_tr = next(t for t in growth["data"] if t["name"] == "Momentum")
        V["plotted_final_nav"] = mom_tr["y"][-1]
        V["backtest_final_nav"] = round(float(base.nav.iloc[-1]), 2)
        V["plotted_vs_backtest_rel_diff"] = abs(mom_tr["y"][-1] / float(base.nav.iloc[-1]) - 1)
        dd_fig = next(
            f
            for f in figs
            if f["layout"].get("yaxis", {}).get("tickformat") == ".0%"
            and any(t.get("name") == "Momentum" for t in f["data"])
            and f["data"][0]["type"] == "scatter"
            and min(v for v in f["data"][0]["y"] if v is not None) < 0
        )
        V["plotted_max_dd"] = min(v for v in dd_fig["data"][0]["y"] if v is not None)
        V["metric_max_dd"] = R["base"]["momentum"]["Max drawdown"]
        tile = re.search(r'CAGR</div><div class="value">([0-9.\-]+)%', page)
        V["tile_cagr_pct"] = float(tile.group(1)) if tile else None
        V["metric_cagr_pct"] = round(100 * R["base"]["momentum"]["CAGR"], 1)
    V["net_vs_gross_final_nav"] = {
        "net": round(float(base.nav.iloc[-1]), 2),
        "gross": round(float(gross_base.nav.iloc[-1]), 2),
    }
    R["visual"] = V
    print(V)

    out = ROOT / "audit" / "results.json"
    out.write_text(json.dumps(R, indent=2, default=str))
    print(f"\nWrote {out.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
