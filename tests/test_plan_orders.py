"""Order planning (pure, no IBKR): caps, sell-first, never overspend."""

import datetime as dt

import pandas as pd
import pytest

from bot import Abort, london_session_open, plan_orders

TARGET = pd.Series({"CSPX": 0.4, "EIMI": 0.4, "IB01": 0.2})
PRICES = {"CSPX": 100.0, "EIMI": 100.0, "IB01": 100.0}


def test_sells_come_first_and_buys_never_exceed_cash_plus_proceeds(risk: dict) -> None:
    positions = {"CSPX": 50, "EIMI": 0, "IB01": 0}
    orders, budget = plan_orders(TARGET, positions, PRICES, 2000, risk)
    sells = [o for o in orders if o[1] == "SELL"]
    buys = [o for o in orders if o[1] == "BUY"]
    assert orders[0][1] == "SELL"
    assert sum(o[3] for o in buys) <= 2000 + sum(o[3] for o in sells)
    assert budget <= risk["capital_usd"]


def test_budget_is_capped_by_capital_and_cash_buffer(risk: dict) -> None:
    _, budget = plan_orders(TARGET, {}, PRICES, 50_000, risk)
    assert budget == pytest.approx(10_000 * 0.98)
    _, budget = plan_orders(TARGET, {}, PRICES, 5_000, risk)
    assert budget == pytest.approx(5_000 * 0.98)


def test_whole_shares_only(risk: dict) -> None:
    orders, budget = plan_orders(pd.Series({"CSPX": 1.0}), {}, {"CSPX": 650.0}, 10_000, risk)
    assert orders == [("CSPX", "BUY", 15, 9750.0)]


def test_trades_below_min_size_are_skipped(risk: dict) -> None:
    positions = {"CSPX": 37, "EIMI": 37, "IB01": 19}
    orders, _ = plan_orders(TARGET, positions, PRICES, 700, risk)
    assert [o[0] for o in orders] == ["CSPX", "EIMI"]
    risk["min_trade_usd"] = 250
    orders, _ = plan_orders(TARGET, positions, PRICES, 700, risk)
    assert orders == []


def test_buys_are_scaled_down_when_a_skipped_sell_leaves_cash_short(risk: dict) -> None:

    risk["min_trade_usd"] = 500
    orders, _ = plan_orders(
        pd.Series({"CSPX": 0.5, "EIMI": 0.5}), {"CSPX": 52}, {"CSPX": 100.0, "EIMI": 100.0}, 4600, risk
    )
    assert [o[:2] for o in orders] == [("EIMI", "BUY")]
    assert orders[0][3] <= 4600


def test_oversized_order_aborts_instead_of_trimming(risk: dict) -> None:
    risk["max_order_usd"] = 1000
    with pytest.raises(Abort, match="max_order_usd"):
        plan_orders(TARGET, {"CSPX": 50}, PRICES, 2000, risk)


def test_too_many_orders_aborts(risk: dict) -> None:
    risk["max_orders_per_run"] = 2
    with pytest.raises(Abort, match="max_orders_per_run"):
        plan_orders(TARGET, {}, PRICES, 10_000, risk)


def test_no_budget_aborts(risk: dict) -> None:
    with pytest.raises(Abort, match="No budget"):
        plan_orders(TARGET, {}, PRICES, 0, risk)


@pytest.mark.parametrize(
    ("utc", "expected"),
    [
        (dt.datetime(2026, 10, 1, 9, 0), True),
        (dt.datetime(2026, 10, 1, 7, 0), False),
        (dt.datetime(2026, 10, 1, 15, 30), False),
        (dt.datetime(2026, 10, 3, 9, 0), False),
        (dt.datetime(2026, 12, 1, 10, 0), True),
    ],
)
def test_london_session_window(utc: dt.datetime, expected: bool) -> None:
    assert london_session_open(utc.replace(tzinfo=dt.UTC)) is expected
