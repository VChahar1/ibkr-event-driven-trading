"""One code path: the bot's signal call reproduces the backtest's decisions exactly."""

import pandas as pd

from backtest import run_backtest
from strategy import StrategyConfig, held_set, needs_rebalance, target_weights
from tests.conftest import make_prices


def test_bot_style_replay_matches_backtest_weights() -> None:
    """Replays the backtest day by day the way the bot decides: target_weights() on the
    history up to t only, held from current weights. Any look-ahead in the backtest's
    precomputed signals, or any held/band mismatch, shows up as a weight difference."""
    cfg = StrategyConfig()
    prices = make_prices(n_days=700, seed=11)
    _, bt_weights, _, n_reb = run_backtest(prices, cfg, cost_bps=0.0)
    assert n_reb >= 3, "fixture too quiet to test anything"

    rets = prices.pct_change().fillna(0.0)
    w = pd.Series(0.0, index=list(cfg.assets))
    w[cfg.safe] = 1.0
    replay = []
    for i in range(len(prices)):
        r = rets.iloc[i]
        w = w * (1 + r) / (1 + float((w * r).sum()))
        if i >= cfg.warmup:
            tgt = target_weights(prices.iloc[: i + 1], cfg, held_set(w, cfg.risky))
            if needs_rebalance(w, tgt, cfg.band):
                w = tgt.copy()
        replay.append(w.copy())
    replay_df = pd.DataFrame(replay, index=prices.index).loc[bt_weights.index]
    pd.testing.assert_frame_equal(replay_df, bt_weights, check_exact=False, rtol=1e-12)
