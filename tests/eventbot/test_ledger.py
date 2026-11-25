import datetime as dt
import random

from eventbot.ledger import ApplyResult, FillLedger, diff_positions, split_exec_id
from eventbot.schemas import Fill, Side

NOW = dt.datetime(2026, 11, 2, 15, 0, tzinfo=dt.UTC)


def fill(exec_id: str, qty: float = 1, side: Side = Side.BUY, symbol: str = "AMD") -> Fill:
    return Fill(
        exec_id=exec_id, order_ref="r", symbol=symbol, side=side, qty=qty, price=100.0, exec_time_utc=NOW
    )


def test_the_same_exec_id_delivered_twice_is_counted_once() -> None:
    ledger = FillLedger()
    assert ledger.apply(fill("a.01.01", 4)) is ApplyResult.APPLIED
    assert ledger.apply(fill("a.01.01", 4)) is ApplyResult.DUPLICATE
    assert ledger.position("AMD") == 4


def test_a_sequence_of_partial_fills_gives_the_correct_position() -> None:
    ledger = FillLedger()
    for n, qty in enumerate([3, 1, 2, 4]):
        ledger.apply(fill(f"a.{n}.01", qty))
    assert ledger.position("AMD") == 10
    ledger.apply(fill("b.0.01", 6, Side.SELL))
    ledger.apply(fill("b.1.01", 4, Side.SELL))
    assert ledger.position("AMD") == 0 and "AMD" not in ledger.positions


def test_final_position_does_not_depend_on_arrival_order_or_on_duplicates() -> None:
    """Fills are keyed by execId across partitions, so they can arrive in any order, and at-least-once
    delivery can repeat them."""
    fills = [
        fill(f"a.{n}.01", qty, side)
        for n, (qty, side) in enumerate(
            [(5, Side.BUY), (2, Side.SELL), (7, Side.BUY), (1, Side.SELL), (3, Side.BUY)]
        )
    ]
    expected = 12.0
    rng = random.Random(7)
    for _ in range(50):
        stream = fills + rng.choices(fills, k=10)
        rng.shuffle(stream)
        ledger = FillLedger()
        for f in stream:
            ledger.apply(f)
        assert ledger.position("AMD") == expected


def test_positions_are_tracked_per_symbol() -> None:
    ledger = FillLedger()
    ledger.apply(fill("a.1.01", 2, symbol="AMD"))
    ledger.apply(fill("b.1.01", 3, symbol="MU"))
    assert ledger.positions == {"AMD": 2, "MU": 3}


def test_split_exec_id() -> None:
    assert split_exec_id("0000e0d5.66f.01.02") == ("0000e0d5.66f.01", 2)
    assert split_exec_id("plain") == ("plain", 0)
    assert split_exec_id("a.b") == ("a.b", 0)


def test_a_correction_replaces_the_earlier_execution_rather_than_adding_to_it() -> None:
    ledger = FillLedger()
    ledger.apply(fill("e.1.01", 5))
    assert ledger.apply(fill("e.1.02", 3)) is ApplyResult.CORRECTION
    assert ledger.position("AMD") == 3


def test_an_older_version_arriving_after_its_correction_is_ignored() -> None:
    ledger = FillLedger()
    ledger.apply(fill("e.1.02", 3))
    assert ledger.apply(fill("e.1.01", 5)) is ApplyResult.SUPERSEDED
    assert ledger.position("AMD") == 3


def test_a_correction_delivered_twice_is_applied_once() -> None:
    ledger = FillLedger()
    ledger.apply(fill("e.1.01", 5))
    ledger.apply(fill("e.1.02", 3))
    assert ledger.apply(fill("e.1.02", 3)) is ApplyResult.DUPLICATE
    assert ledger.position("AMD") == 3


def test_baseline_plus_fills() -> None:
    ledger = FillLedger(baseline={"AMD": 10.0})
    ledger.apply(fill("a.1.01", 2, Side.SELL))
    assert ledger.position("AMD") == 8


def test_fills_already_in_the_baseline_are_not_counted_again() -> None:
    ledger = FillLedger(baseline={"AMD": 10.0}, excluded=["old.1.01"])
    assert ledger.apply(fill("old.1.01", 10)) is ApplyResult.EXCLUDED
    assert ledger.position("AMD") == 10


def test_rebase_recomputes_from_the_new_baseline() -> None:
    ledger = FillLedger()
    ledger.apply(fill("a.1.01", 5))
    ledger.apply(fill("b.1.01", 1))
    ledger.rebase({"AMD": 100.0}, excluded=["a.1.01"])
    assert ledger.position("AMD") == 101


def test_bought_net_ignores_the_baseline_and_drops_sold_out_names() -> None:
    ledger = FillLedger(baseline={"XOM": 50.0})
    ledger.apply(fill("a.1.01", 5, symbol="AMD"))
    ledger.apply(fill("b.1.01", 5, Side.SELL, symbol="AMD"))
    ledger.apply(fill("c.1.01", 7, symbol="MU"))
    assert ledger.bought_net() == {"MU": 7}


def test_diff_reports_each_disagreement() -> None:
    assert diff_positions({"A": 1, "B": 2}, {"A": 1, "B": 2}) == {}
    assert diff_positions({"A": 1}, {"A": 2, "C": 3}) == {
        "A": {"internal": 1, "broker": 2},
        "C": {"internal": 0.0, "broker": 3},
    }
