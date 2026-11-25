#!/usr/bin/env python3
"prereg_test.py - run the pre-registered tests in HYPOTHESES.md exactly once."

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DOC, LOCK, OUT = ROOT / "HYPOTHESES.md", ROOT / "HYPOTHESES.lock", ROOT / "PREREG_RESULTS.md"


TICKERS = {"EQ": "SPY", "BD": "TLT", "CASH": "BIL"}
START = "2006-01-01"
WARMUP = 252
BASE = {"EQ": 0.6, "BD": 0.4}
BAND = 0.05
COST_BPS = 10.0
VT_WINDOW, VT_TARGET = 63, 0.10
SMA, EXIT_RATIO, ENTER_RATIO = 200, 0.98, 1.02
BLOCK, N_BOOT, SEED, ALPHA = 21, 5000, 20260929, 0.025
N_TRIALS = 3
ASSETS = ["EQ", "BD", "CASH"]


def targets_vol_target(rets: pd.DataFrame) -> pd.DataFrame:
    w = np.array([BASE["EQ"], BASE["BD"]])
    rows = []
    for i in range(len(rets)):
        if i + 1 < VT_WINDOW:
            rows.append((np.nan, np.nan, np.nan))
            continue
        cov = np.cov(rets[["EQ", "BD"]].iloc[i + 1 - VT_WINDOW : i + 1].values, rowvar=False)
        sigma = math.sqrt(252 * float(w @ cov @ w))
        scale = min(1.0, VT_TARGET / sigma) if sigma > 0 else 1.0
        rows.append((BASE["EQ"] * scale, BASE["BD"] * scale, 1 - scale))
    return pd.DataFrame(rows, index=rets.index, columns=ASSETS)


def targets_trend_filter(prices: pd.DataFrame, first_eval: int) -> pd.DataFrame:
    ratio = prices[["EQ", "BD"]] / prices[["EQ", "BD"]].rolling(SMA).mean()
    state: dict[str, bool | None] = {a: None for a in ("EQ", "BD")}
    rows = []
    for i in range(len(prices)):
        if i < first_eval:
            rows.append((np.nan, np.nan, np.nan))
            continue
        for a in ("EQ", "BD"):
            r = ratio[a].iloc[i]
            if state[a] is None:
                state[a] = bool(r > 1)
            elif state[a] and r < EXIT_RATIO:
                state[a] = False
            elif not state[a] and r > ENTER_RATIO:
                state[a] = True
        eq = BASE["EQ"] if state["EQ"] else 0.0
        bd = BASE["BD"] if state["BD"] else 0.0
        rows.append((eq, bd, 1 - eq - bd))
    return pd.DataFrame(rows, index=prices.index, columns=ASSETS)


def simulate(
    rets: pd.DataFrame, targets: pd.DataFrame, first_eval: int, cost_bps: float, monthly: bool = False
) -> tuple[pd.Series, float]:
    """Signal at close t -> weights earn from t+1. Returns (daily returns, turnover/yr)."""
    w = pd.Series({"EQ": 0.0, "BD": 0.0, "CASH": 1.0})
    out, turnover, month = [], 0.0, None
    for i in range(len(rets)):
        r = rets.iloc[i]
        day = float((w * r).sum())
        w = w * (1 + r) / (1 + day)
        cost = 0.0
        if i >= first_eval - 1:
            tgt = targets.iloc[i]
            if monthly:
                trade = rets.index[i].month != month
                month = rets.index[i].month
            else:
                trade = bool((w - tgt).abs().max() > BAND)
            if trade:
                to = float((tgt - w).abs().sum())
                cost = to * cost_bps / 1e4
                if i >= first_eval:
                    turnover += to
                w = tgt.copy()
        out.append(day - (cost if i >= first_eval else 0.0))
    s = pd.Series(out, index=rets.index).iloc[first_eval:]
    return s, turnover / (len(s) / 252)


def sharpe(ex: np.ndarray) -> float:
    return float(ex.mean() / ex.std(ddof=1) * math.sqrt(252))


def describe(r: pd.Series, cash: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "CAGR": cagr,
        "vol": r.std() * math.sqrt(252),
        "Sharpe": sharpe((r - cash).values),
        "maxDD": dd,
        "Calmar": cagr / abs(dd),
    }


def block_bootstrap_dsharpe(ex_s: np.ndarray, ex_b: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    T = len(ex_s)
    n_blocks = math.ceil(T / BLOCK)
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        starts = rng.integers(0, T, n_blocks)
        idx = ((starts[:, None] + np.arange(BLOCK)[None, :]) % T).ravel()[:T]
        out[k] = sharpe(ex_s[idx]) - sharpe(ex_b[idx])
    return out


def deflated_sharpe(ex: np.ndarray, sr_trials: list) -> float:
    sr = ex.mean() / ex.std(ddof=1)
    z = (ex - ex.mean()) / ex.std(ddof=0)
    g3, g4 = float(np.mean(z**3)), float(np.mean(z**4))
    nd, g = NormalDist(), 0.5772156649015329
    n = len(sr_trials)
    sr0 = float(np.std(sr_trials, ddof=1)) * (
        (1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e))
    )
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return nd.cdf((sr - sr0) * math.sqrt(len(ex) - 1) / denom)


def run_all(prices: pd.DataFrame) -> dict:
    prices = prices[ASSETS].dropna()
    rets = prices.pct_change().fillna(0.0)
    fe = WARMUP
    static_t = pd.DataFrame([[BASE["EQ"], BASE["BD"], 0.0]] * len(rets), index=rets.index, columns=ASSETS)
    tg = {
        "H1 vol-target 60/40": targets_vol_target(rets),
        "H2 trend-filter 60/40": targets_trend_filter(prices, fe - 1),
    }
    res: dict[str, Any] = {"sample": (str(rets.index[fe].date()), str(rets.index[-1].date()), len(rets) - fe)}
    cash = rets["CASH"].iloc[fe:]
    series = {}
    for cost in (COST_BPS, 20.0):
        bench, bto = simulate(rets, static_t, fe, cost, monthly=True)
        series[("Static 60/40", cost)] = (bench, bto)
        for name, t in tg.items():
            series[(name, cost)] = simulate(rets, t, fe, cost)
    res["series"] = series

    bench = series[("Static 60/40", COST_BPS)][0]
    ex_b = (bench - cash).values

    decisions = {}
    for name in tg:
        r = series[(name, COST_BPS)][0]
        ex_s = (r - cash).values
        boot = block_bootstrap_dsharpe(ex_s, ex_b)
        lo = float(np.quantile(boot, ALPHA))
        d_sr = sharpe(ex_s) - sharpe(ex_b)
        dd_s, dd_b = describe(r, cash)["maxDD"], describe(bench, cash)["maxDD"]
        ok = lo > 0 and abs(dd_s) < abs(dd_b)
        decisions[name] = {
            "dSharpe": d_sr,
            "boot_lo": lo,
            "boot_hi": float(np.quantile(boot, 1 - ALPHA)),
            "p_one_sided": float(np.mean(boot <= 0)),
            "dd_better": abs(dd_s) < abs(dd_b),
            "verdict": "SUPPORTED" if ok else "NOT SUPPORTED",
        }
    res["decisions"] = decisions
    return res


def fmt_table(
    res: dict, cash: pd.Series, cost: float, start: str | None = None, end: str | None = None
) -> str:
    rows = []
    for (name, c), (r, to) in res["series"].items():
        if c != cost:
            continue
        rr = r.loc[start:end]
        d = describe(rr, cash.loc[rr.index])
        rows.append(
            f"| {name} | {d['CAGR'] * 100:.1f}% | {d['vol'] * 100:.1f}% | {d['Sharpe']:.2f} | "
            f"{d['maxDD'] * 100:.1f}% | {d['Calmar']:.2f} | {to:.2f}x |"
        )
    head = "| Strategy | CAGR | Vol | Sharpe | Max DD | Calmar | Turnover/yr (full sample) |\n|---|---|---|---|---|---|---|\n"
    return head + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--rerun-reason")
    ap.add_argument(
        "--dm-sharpe",
        type=float,
        default=0.543,
        help="annualised excess Sharpe of the earlier dual-momentum trial (for DSR)",
    )
    args = ap.parse_args()
    digest = hashlib.sha256(DOC.read_bytes()).hexdigest()

    if args.lock:
        if LOCK.exists():
            raise SystemExit(f"Already locked: {LOCK.read_text()}")
        LOCK.write_text(
            json.dumps(
                {"sha256": digest, "locked_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
            )
        )
        print(f"Locked HYPOTHESES.md sha256={digest[:16]}... Commit both files to git now.")
        return
    if not LOCK.exists():
        raise SystemExit("Run --lock first (and commit HYPOTHESES.md + HYPOTHESES.lock to git).")
    lock = json.loads(LOCK.read_text())
    if lock["sha256"] != digest:
        raise SystemExit("HYPOTHESES.md changed after locking - the pre-registration is void.")
    if OUT.exists() and not args.rerun_reason:
        raise SystemExit("Already run once. A re-run needs --rerun-reason and will be flagged.")

    from data import load_prices

    prices = load_prices({k: v for k, v in TICKERS.items()}, start=START)
    res = run_all(prices)
    cash = prices["CASH"].pct_change().fillna(0.0)

    sr_dm = args.dm_sharpe / math.sqrt(252)
    lines = [
        "# Pre-registered test results",
        f"Run {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"hypotheses locked {lock['locked_utc']} · sha256 {digest[:16]}…",
        f"Sample (evaluation): {res['sample'][0]} → {res['sample'][1]} ({res['sample'][2]} days)",
    ]
    if args.rerun_reason:
        lines.append(f"\n**RE-RUN** (results already existed). Reason given: {args.rerun_reason}")
    lines += [
        "\n## Decisions (primary, 10 bps)",
        "| Hypothesis | ΔSharpe vs 60/40 | 2.5th–97.5th pct (bootstrap) | one-sided p | Max DD better? | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name, d in res["decisions"].items():
        lines.append(
            f"| {name} | {d['dSharpe']:+.2f} | [{d['boot_lo']:+.2f}, {d['boot_hi']:+.2f}] | "
            f"{d['p_one_sided']:.3f} | {'yes' if d['dd_better'] else 'no'} | **{d['verdict']}** |"
        )
    lines += [
        "\n## Statistics, 10 bps",
        fmt_table(res, cash, COST_BPS),
        "\n## Statistics, 20 bps (descriptive)",
        fmt_table(res, cash, 20.0),
        "\n## Sub-period to 2016-12-31 (descriptive)",
        fmt_table(res, cash, COST_BPS, None, "2016-12-31"),
        "\n## Sub-period from 2017-01-01 (descriptive)",
        fmt_table(res, cash, COST_BPS, "2017-01-01", None),
        "\n## Deflated Sharpe ratio (N = 3 trials, descriptive)",
    ]
    trial_srs = [sr_dm]
    for name in res["decisions"]:
        r = res["series"][(name, COST_BPS)][0]
        trial_srs.append(float((r - cash.loc[r.index]).mean() / (r - cash.loc[r.index]).std(ddof=1)))
    for name in res["decisions"]:
        r = res["series"][(name, COST_BPS)][0]
        dsr = deflated_sharpe((r - cash.loc[r.index]).values, trial_srs)
        lines.append(f"- {name}: DSR = {dsr:.3f}")
    text = "\n".join(lines) + "\n"
    with open(OUT, "a" if args.rerun_reason else "w") as f:
        f.write(("\n---\n" if args.rerun_reason else "") + text)
    print(text)


if __name__ == "__main__":
    main()
