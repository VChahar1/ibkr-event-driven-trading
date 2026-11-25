"""Shared fixtures: synthetic prices and default config/risk settings."""

import numpy as np
import pandas as pd
import pytest

from strategy import StrategyConfig


def make_prices(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic closes for the default universe: distinct drifts
    plus noise, so momentum ranks and SMA crossings both occur."""
    cfg = StrategyConfig()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    drifts = {"CSPX": 6e-4, "EIMI": 2e-4, "IDTL": -3e-4, "IGLN": 4e-4, "IB01": 1.5e-4}
    vols = {"CSPX": 0.010, "EIMI": 0.012, "IDTL": 0.009, "IGLN": 0.008, "IB01": 0.0002}
    data = {a: 100 * np.exp(np.cumsum(drifts[a] + vols[a] * rng.standard_normal(n_days))) for a in cfg.assets}
    return pd.DataFrame(data, index=dates)


def make_stock_prices(
    n_stocks: int = 30, n_days: int = 700, seed: int = 5, market_drift: float = 4e-4
) -> pd.DataFrame:
    """Synthetic closes for S01..Snn plus SPY. Each stock has its own drift, so momentum
    ranks are persistent but noisy; SPY trends with `market_drift`."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    drifts = rng.normal(3e-4, 6e-4, n_stocks)
    data = {
        f"S{i + 1:02d}": 50 * np.exp(np.cumsum(drifts[i] + 0.015 * rng.standard_normal(n_days)))
        for i in range(n_stocks)
    }
    data["SPY"] = 400 * np.exp(np.cumsum(market_drift + 0.01 * rng.standard_normal(n_days)))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def cfg() -> StrategyConfig:
    return StrategyConfig()


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def risk() -> dict:
    return {
        "capital_usd": 10000,
        "max_order_usd": 10000,
        "min_trade_usd": 0,
        "max_orders_per_run": 6,
        "cash_buffer": 0.02,
    }
