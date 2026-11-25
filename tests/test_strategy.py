"""Signal logic in strategy.py: eligibility rules, hysteresis, no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from strategy import StrategyConfig, needs_rebalance, signal_frames, target_weights, weights_from_signals


def rows(mom: dict, ratio: dict) -> tuple[pd.Series, pd.Series]:
    return pd.Series(mom, dtype=float), pd.Series(ratio, dtype=float)


BASE_MOM = {"CSPX": 0.20, "EIMI": 0.15, "IDTL": 0.10, "IGLN": 0.05, "IB01": 0.03}
ALL_IN_TREND = {"CSPX": 1.10, "EIMI": 1.10, "IDTL": 1.10, "IGLN": 1.10, "IB01": 1.0}


def test_holds_top_n_by_momentum(cfg: StrategyConfig) -> None:
    w = weights_from_signals(*rows(BASE_MOM, ALL_IN_TREND), cfg)
    assert w["CSPX"] == 0.5 and w["EIMI"] == 0.5
    assert w[["IDTL", "IGLN", "IB01"]].sum() == 0.0


def test_weights_sum_to_one_and_unused_slots_go_to_tbills(cfg: StrategyConfig) -> None:
    mom = {**BASE_MOM, "EIMI": 0.0, "IDTL": 0.0, "IGLN": 0.0}
    w = weights_from_signals(*rows(mom, ALL_IN_TREND), cfg)
    assert w["CSPX"] == 0.5 and w["IB01"] == 0.5
    assert w.sum() == pytest.approx(1.0)


def test_nothing_eligible_means_all_tbills(cfg: StrategyConfig) -> None:
    below = {a: 0.9 for a in cfg.risky} | {"IB01": 1.0}
    w = weights_from_signals(*rows(BASE_MOM, below), cfg)
    assert w["IB01"] == 1.0


def test_tbill_hurdle_excludes_weak_momentum(cfg: StrategyConfig) -> None:
    mom = {**BASE_MOM, "CSPX": 0.02}
    w = weights_from_signals(*rows(mom, ALL_IN_TREND), cfg)
    assert w["CSPX"] == 0.0


def test_trend_filter_excludes_asset_below_sma(cfg: StrategyConfig) -> None:
    ratio = {**ALL_IN_TREND, "CSPX": 0.95}
    w = weights_from_signals(*rows(BASE_MOM, ratio), cfg)
    assert w["CSPX"] == 0.0 and w["EIMI"] == 0.5 and w["IDTL"] == 0.5


def test_non_finite_momentum_is_never_eligible(cfg: StrategyConfig) -> None:
    mom = {**BASE_MOM, "CSPX": np.nan}
    w = weights_from_signals(*rows(mom, ALL_IN_TREND), cfg)
    assert w["CSPX"] == 0.0


def test_trend_buffer_hysteresis(cfg: StrategyConfig) -> None:

    ratio = {**ALL_IN_TREND, "CSPX": 1.01}
    mom_row, ratio_row = rows(BASE_MOM, ratio)
    assert weights_from_signals(mom_row, ratio_row, cfg)["CSPX"] == 0.0
    assert weights_from_signals(mom_row, ratio_row, cfg, frozenset({"CSPX"}))["CSPX"] == 0.5

    ratio_row["CSPX"] = 0.97
    assert weights_from_signals(mom_row, ratio_row, cfg, frozenset({"CSPX"}))["CSPX"] == 0.0


def test_momentum_margin_protects_holder() -> None:
    cfg = StrategyConfig(top_n=1)
    mom = {**BASE_MOM, "CSPX": 0.11, "EIMI": 0.10}
    mom_row, ratio_row = rows(mom, ALL_IN_TREND)
    assert weights_from_signals(mom_row, ratio_row, cfg, frozenset({"EIMI"}))["EIMI"] == 1.0
    mom_row["CSPX"] = 0.13
    assert weights_from_signals(mom_row, ratio_row, cfg, frozenset({"EIMI"}))["CSPX"] == 1.0


def test_signal_frames_have_no_look_ahead(prices: pd.DataFrame, cfg: StrategyConfig) -> None:
    t = 300
    mom, ratio = signal_frames(prices, cfg)
    shocked = prices.copy()
    shocked.iloc[t + 1 :] *= 3.0
    mom2, ratio2 = signal_frames(shocked, cfg)
    pd.testing.assert_frame_equal(mom.iloc[: t + 1], mom2.iloc[: t + 1])
    pd.testing.assert_frame_equal(ratio.iloc[: t + 1], ratio2.iloc[: t + 1])


def test_target_weights_uses_last_row_only(prices: pd.DataFrame, cfg: StrategyConfig) -> None:
    mom, ratio = signal_frames(prices, cfg)
    expected = weights_from_signals(mom.iloc[-1], ratio.iloc[-1], cfg)
    pd.testing.assert_series_equal(target_weights(prices, cfg), expected)


def test_target_weights_requires_warmup(prices: pd.DataFrame, cfg: StrategyConfig) -> None:
    with pytest.raises(ValueError, match="need >="):
        target_weights(prices.iloc[: cfg.warmup - 1], cfg)


def test_needs_rebalance_on_set_change_or_drift() -> None:
    tgt = pd.Series({"CSPX": 0.5, "EIMI": 0.5, "IB01": 0.0})
    assert not needs_rebalance(pd.Series({"CSPX": 0.52, "EIMI": 0.48, "IB01": 0.0}), tgt, 0.05)
    assert needs_rebalance(pd.Series({"CSPX": 0.56, "EIMI": 0.44, "IB01": 0.0}), tgt, 0.05)
    assert needs_rebalance(pd.Series({"CSPX": 0.5, "EIMI": 0.0, "IB01": 0.5}), tgt, 0.05)
