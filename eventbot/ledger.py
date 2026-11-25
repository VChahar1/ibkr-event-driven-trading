"Positions built from fills, counting each execution exactly once."

from collections.abc import Iterable
from enum import Enum

from eventbot.schemas import Fill, Side

ZERO_TOLERANCE = 1e-9


class ApplyResult(Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    CORRECTION = "correction"
    SUPERSEDED = "superseded"
    EXCLUDED = "excluded"


def split_exec_id(exec_id: str) -> tuple[str, int]:
    """'0000e0d5.66f.01.02' -> ('0000e0d5.66f.01', 2). An ID without a numeric suffix is its own base."""
    base, dot, version = exec_id.rpartition(".")
    if dot and version.isdigit():
        return base, int(version)
    return exec_id, 0


def signed(fill: Fill) -> float:
    return fill.qty if fill.side is Side.BUY else -fill.qty


class FillLedger:
    def __init__(self, baseline: dict[str, float] | None = None, excluded: Iterable[str] = ()) -> None:
        self._baseline = dict(baseline or {})
        self._excluded = set(excluded)
        self._current: dict[str, tuple[int, Fill]] = {}
        self._seen: set[str] = set()
        self._positions = dict(self._baseline)

    def apply(self, fill: Fill) -> ApplyResult:
        if fill.exec_id in self._seen:
            return ApplyResult.DUPLICATE
        self._seen.add(fill.exec_id)
        base, version = split_exec_id(fill.exec_id)
        prior = self._current.get(base)
        if prior is not None and version < prior[0]:
            return ApplyResult.SUPERSEDED
        self._current[base] = (version, fill)
        if fill.exec_id in self._excluded or (prior is not None and prior[1].exec_id in self._excluded):
            return ApplyResult.EXCLUDED
        if prior is not None:
            self._add(prior[1].symbol, -signed(prior[1]))
            self._add(fill.symbol, signed(fill))
            return ApplyResult.CORRECTION
        self._add(fill.symbol, signed(fill))
        return ApplyResult.APPLIED

    def _add(self, symbol: str, delta: float) -> None:
        value = self._positions.get(symbol, 0.0) + delta
        if abs(value) < ZERO_TOLERANCE:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = value

    def rebase(self, baseline: dict[str, float], excluded: Iterable[str]) -> None:
        """Adopt a new baseline and recompute from every fill seen so far."""
        self._baseline = dict(baseline)
        self._excluded = set(excluded)
        self._positions = dict(self._baseline)
        for _, fill in self._current.values():
            if fill.exec_id not in self._excluded:
                self._add(fill.symbol, signed(fill))

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    @property
    def exec_ids(self) -> set[str]:
        return set(self._seen)

    def bought_net(self) -> dict[str, float]:
        """Net shares bought through fills, per symbol, ignoring the baseline. The strategy uses this
        to spot holdings it bought that have since left the universe (stockbot's "orphans")."""
        net: dict[str, float] = {}
        for _, fill in self._current.values():
            net[fill.symbol] = net.get(fill.symbol, 0.0) + signed(fill)
        return {s: q for s, q in net.items() if q > ZERO_TOLERANCE}


def diff_positions(
    internal: dict[str, float], broker: dict[str, float], tolerance: float = 1e-6
) -> dict[str, dict[str, float]]:
    """Every symbol on which the two views disagree."""
    out = {}
    for symbol in sorted(set(internal) | set(broker)):
        a, b = internal.get(symbol, 0.0), broker.get(symbol, 0.0)
        if abs(a - b) > tolerance:
            out[symbol] = {"internal": a, "broker": b}
    return out
