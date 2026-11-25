"""The live rebalance gate must match the backtest's band, whatever the USD cash pile."""

import pandas as pd

from bot import plan_orders, rebalance_due

OCT1_PRICES = {"CSPX": 829.31, "EIMI": 54.84, "IDTL": 2.897, "IGLN": 81.009, "IB01": 122.02}
OCT1_USD_CASH = 24356.39
TARGET = pd.Series({"CSPX": 0.5, "EIMI": 0.5, "IDTL": 0.0, "IGLN": 0.0, "IB01": 0.0})
BAND = 0.05


def risk() -> dict:
    return {
        "capital_usd": 10000,
        "max_order_usd": 10000,
        "min_trade_usd": 250,
        "max_orders_per_run": 6,
        "cash_buffer": 0.02,
    }


def settled_positions() -> dict:
    """Positions after executing the Oct 1 plan: exactly the share-rounded target."""
    return {"CSPX": 5, "EIMI": 89, "IDTL": 0, "IGLN": 0, "IB01": 0}


def test_at_target_with_large_usd_cash_does_not_rebalance() -> None:

    assert not rebalance_due(TARGET, settled_positions(), OCT1_PRICES, OCT1_USD_CASH, risk(), BAND)


def test_whole_share_rounding_alone_does_not_rebalance() -> None:

    cash = 10000 - (5 * 829.31 + 89 * 54.84)
    assert not rebalance_due(TARGET, settled_positions(), OCT1_PRICES, cash, risk(), BAND)


def test_real_drift_beyond_band_rebalances() -> None:
    drifted = {**OCT1_PRICES, "EIMI": OCT1_PRICES["EIMI"] * 1.30}
    assert rebalance_due(TARGET, settled_positions(), drifted, OCT1_USD_CASH, risk(), BAND)


def test_change_of_holdings_rebalances() -> None:
    target = pd.Series({"CSPX": 0.5, "EIMI": 0.0, "IDTL": 0.0, "IGLN": 0.5, "IB01": 0.0})
    assert rebalance_due(target, settled_positions(), OCT1_PRICES, OCT1_USD_CASH, risk(), BAND)


def test_settled_positions_really_are_the_planned_target() -> None:

    orders, _ = plan_orders(TARGET, {}, OCT1_PRICES, OCT1_USD_CASH, risk())
    assert {s: q for s, _, q, _ in orders} == {"CSPX": 5, "EIMI": 89}
