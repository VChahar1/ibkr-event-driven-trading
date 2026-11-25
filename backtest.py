#!/usr/bin/env python3
"backtest.py - daily backtest of strategy.py with costs."

import argparse
from pathlib import Path

import pandas as pd

from analytics import perf_stats, sharpe_se
from config import load_config
from strategy import StrategyConfig, held_set, needs_rebalance, signal_frames, weights_from_signals

ROOT = Path(__file__).parent


def run_backtest(
    prices: pd.DataFrame, cfg: StrategyConfig, cost_bps: float = 10.0
) -> tuple[pd.Series, pd.DataFrame, float, int]:
    prices = prices[list(cfg.assets)].dropna()
    rets = prices.pct_change().fillna(0.0)
    mom, ratio = signal_frames(prices, cfg)

    w = pd.Series(0.0, index=list(cfg.assets))
    w[cfg.safe] = 1.0
    port, weights, turnover, n_reb = [], [], [], 0
    for i in range(len(prices)):
        r = rets.iloc[i]
        day_ret = float((w * r).sum())

        w = w * (1 + r) / (1 + day_ret)
        cost = 0.0
        if i >= cfg.warmup:
            held = held_set(w, cfg.risky)
            tgt = weights_from_signals(mom.iloc[i], ratio.iloc[i], cfg, held)
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
    r = pd.Series(port, index=idx).loc[live]
    return r, pd.DataFrame(weights, index=idx).loc[live], sum(turnover), n_reb


def static_mix(prices: pd.DataFrame, weights: dict, cost_bps: float = 10.0) -> pd.Series:
    """Fixed-weight portfolio, rebalanced on the first trading day of each month."""
    cols = list(weights)
    rets = prices[cols].pct_change().fillna(0.0)
    target = pd.Series(weights, dtype=float)
    w = target.copy()
    out, month = [], None
    for dt_, r in rets.iterrows():
        day = float((w * r).sum())
        w = w * (1 + r) / (1 + day)
        cost = 0.0
        if dt_.month != month:
            cost = float((target - w).abs().sum()) * cost_bps / 1e4
            w, month = target.copy(), dt_.month
        out.append(day - cost)
    return pd.Series(out, index=rets.index)


def stats(r: pd.Series, name: str, rf: pd.Series | None = None) -> dict:
    """Summary row; metrics come from analytics.perf_stats (excess-return Sharpe)."""
    s = perf_stats(r, rf)
    return {
        "strategy": name,
        "CAGR": s["CAGR"],
        "vol": s["Volatility"],
        "Sharpe": s["Sharpe"],
        "maxDD": s["Max drawdown"],
        "Calmar": s["Calmar"],
        "years": s["Years"],
    }


def load_backtest_prices(
    conf: dict,
    cfg: StrategyConfig,
    proxy: bool,
    source: str = "yahoo",
    start: str = "2005-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily closes for the backtest: UCITS tickers or US proxies, from Yahoo or IBKR (cached)."""
    if source == "ibkr":
        from data import load_prices_ibkr

        cache = ROOT / "data_cache" / f"ibkr_{'proxy' if proxy else 'ucits'}.csv"
        if cache.exists() and not refresh:
            prices = pd.read_csv(cache, index_col=0, parse_dates=True)
            print(f"Using cached IBKR data {cache.name} (--refresh to re-download)")
        else:
            print("Downloading daily history from IBKR...")
            prices = load_prices_ibkr(conf, list(cfg.assets), proxy=proxy)
            cache.parent.mkdir(exist_ok=True)
            prices.to_csv(cache)
        return prices.loc[start:]
    from data import load_prices

    ymap = dict(conf["proxy"]) if proxy else {k: v[3] for k, v in conf["universe"].items()}
    return load_prices(ymap, start=start)


def benchmark_returns(
    prices: pd.DataFrame, cfg: StrategyConfig, cost_bps: float, index: pd.Index
) -> dict[str, pd.Series]:
    """The already-observed comparison portfolios, aligned to the strategy's dates."""
    eq = cfg.risky[0]
    return {
        f"Buy&hold {eq}": prices[eq].pct_change().loc[index].fillna(0),
        "Static equal-weight (5 ETFs)": static_mix(
            prices, {a: 1 / len(cfg.assets) for a in cfg.assets}, cost_bps
        ).loc[index],
        f"Static 60/40 {eq}/IDTL": static_mix(prices, {eq: 0.6, "IDTL": 0.4}, cost_bps).loc[index],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", action="store_true", help="use US-listed proxies (longer history)")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument(
        "--source",
        choices=["yahoo", "ibkr"],
        default="yahoo",
        help="price source; ibkr needs IB Gateway running",
    )
    ap.add_argument("--refresh", action="store_true", help="ignore cached IBKR data")
    args = ap.parse_args()

    conf, cfg = load_config()
    prices = load_backtest_prices(conf, cfg, args.proxy, args.source, args.start, args.refresh)
    print(f"Data: {prices.index[0].date()} -> {prices.index[-1].date()} ({len(prices)} days)")

    r, w, to, n_reb = run_backtest(prices, cfg, args.cost_bps)
    rf = prices[cfg.safe].pct_change().loc[r.index].fillna(0)
    benches = benchmark_returns(prices, cfg, args.cost_bps, r.index)
    bench, ew = benches[f"Buy&hold {cfg.risky[0]}"], benches["Static equal-weight (5 ETFs)"]
    table = pd.DataFrame(
        [stats(r, "Dual momentum", rf)] + [stats(b, name, rf) for name, b in benches.items()]
    )
    print(f"Sharpe uses {cfg.safe} (T-bills) as the risk-free rate. Calmar = CAGR / |maxDD|.")
    print(table.round(3).to_string(index=False))
    n_yrs = len(r) / 252
    print(
        f"Sharpe standard error (Lo 2002) over {n_yrs:.1f} yrs: "
        f"+/-{sharpe_se(float(table['Sharpe'].iloc[0]), n_yrs):.2f}"
    )
    print(
        f"Rebalances: {n_reb} ({n_reb / (len(r) / 252):.1f}/yr), "
        f"turnover {to / (len(r) / 252):.2f}x/yr, cost {args.cost_bps} bps"
    )
    print("Average weights:\n", w.mean().round(3).to_string())

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        (1 + r).cumprod().plot(ax=ax[0], label="Dual momentum")
        (1 + bench).cumprod().plot(ax=ax[0], label=f"Buy&hold {cfg.risky[0]}", alpha=0.7)
        (1 + ew).cumprod().plot(ax=ax[0], label="Static equal-weight", alpha=0.7, ls="--")
        ax[0].set_yscale("log")
        ax[0].legend()
        ax[0].set_title("Equity (log)")
        w.plot.area(ax=ax[1], linewidth=0)
        ax[1].set_ylim(0, 1)
        ax[1].set_title("Weights")
        plt.tight_layout()
        plt.savefig(ROOT / "backtest.png", dpi=120)
        print("Saved backtest.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
