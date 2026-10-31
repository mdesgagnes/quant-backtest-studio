"""Next-rebalance checks. Synthetic data only, no network.

Each test pins a consequence of "replay the engine over a few flat sessions"
to a number worked out independently of the engine.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt import fx as FX                                    # noqa: E402
from qbt.config import CostConfig, EngineConfig             # noqa: E402
from qbt.engine import run_backtest                         # noqa: E402
from qbt.nextrade import next_rebalance                     # noqa: E402

ZERO = CostConfig(commission_bps=0, slippage_bps=0, fx_spread_bps=0)


def _world(end="2026-10-30", n_assets=4, seed=5, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    r = rng.normal(0.0004, 0.011, (len(idx), n_assets))
    px = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), idx, list("ABCDEFG")[:n_assets])
    return px


def _rotating(px, k=2):
    """Equal weight in the k best 60-day performers, a plain causal signal."""
    mom = px.pct_change(60)
    rank = mom.rank(axis=1, ascending=False)
    w = (rank <= k).astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def _cell(nx, a, col):
    """A table value; an instrument that is neither held nor wanted has no row."""
    t = nx.table.set_index("Instrument")
    return float(t.loc[a, col]) if a in t.index else 0.0


def _run(px, w, eng, costs=ZERO, **kw):
    res = run_backtest(px, w, eng, costs, None, "x", **kw)
    return res, next_rebalance(px, w, eng, costs, reference=res, **kw)


def test_replay_reproduces_history_and_reaches_the_latest_target():
    px = _world()
    w = _rotating(px)
    eng = EngineConfig(rebalance="M", execution_lag=1, min_trade_weight=0.0)
    res, nx = _run(px, w, eng)
    assert nx.replay_diff < 1e-12
    tgt = w.iloc[-1]
    # No frictions, no minimum: the book lands exactly on the latest signal.
    for a in px.columns:
        assert abs(_cell(nx, a, "Weight after") - tgt[a]) < 1e-9
    # Units bought are (target value / price) minus what is held, worked out
    # here from the result's own end-of-run state.
    value = float(res.equity.iloc[-1])
    for a in px.columns:
        want = tgt[a] * value / px[a].iloc[-1] - res.shares[a].iloc[-1]
        assert abs(_cell(nx, a, "Units to trade") - want) < 1e-6 * max(1.0, abs(want)) + 1e-9
    # Buys and sells net to nothing without costs.
    assert abs(nx.table[nx.table["Instrument"] != "Cash"]["Amount"].sum()) < 1e-6


def test_friday_month_end_is_a_scheduled_signal_traded_on_monday():
    px = _world(end="2026-10-30")                      # Oct 31 is a Saturday
    _, nx = _run(px, _rotating(px), EngineConfig(rebalance="M", execution_lag=1))
    assert nx.is_scheduled
    assert nx.scheduled_signal == pd.Timestamp("2026-10-30")
    assert nx.scheduled_exec == pd.Timestamp("2026-11-02")
    assert nx.exec_date == pd.Timestamp("2026-11-02")


def test_mid_month_data_is_provisional_and_dated_to_month_end():
    px = _world(end="2026-10-20")
    _, nx = _run(px, _rotating(px), EngineConfig(rebalance="M", execution_lag=1))
    assert not nx.is_scheduled
    assert nx.scheduled_signal == pd.Timestamp("2026-10-30")     # last weekday of October
    assert nx.scheduled_exec == pd.Timestamp("2026-11-02")
    assert nx.exec_date == pd.Timestamp("2026-10-21")            # if traded at once


def test_quarterly_and_first_day_rules_find_the_right_future_date():
    px = _world(end="2026-10-30")
    _, q = _run(px, _rotating(px), EngineConfig(rebalance="Q", execution_lag=1))
    assert q.scheduled_signal == pd.Timestamp("2026-12-31")
    _, f = _run(px, _rotating(px), EngineConfig(rebalance="M", day_rule="first",
                                                execution_lag=1))
    assert f.scheduled_signal == pd.Timestamp("2026-11-02")      # first weekday of November
    assert f.scheduled_exec == pd.Timestamp("2026-11-03")


def test_two_session_lag_trades_two_sessions_after_the_signal():
    px = _world(end="2026-10-30")
    _, nx = _run(px, _rotating(px), EngineConfig(rebalance="M", execution_lag=2,
                                                 min_trade_weight=0.0))
    assert nx.exec_date == pd.Timestamp("2026-11-03")
    assert nx.scheduled_exec == pd.Timestamp("2026-11-03")
    w = _rotating(px).iloc[-1]
    assert all(abs(_cell(nx, a, "Weight after") - w[a]) < 1e-9 for a in px.columns)


def test_execution_at_the_open_is_sized_on_the_last_close():
    px = _world()
    opens = px * 0.995
    w = _rotating(px)
    eng = EngineConfig(rebalance="M", execution_lag=1, min_trade_weight=0.0)
    res, nx = _run(px, w, eng, open_prices=opens)
    assert nx.at_open and nx.replay_diff < 1e-12
    for a in px.columns:
        if _cell(nx, a, "Units to trade") != 0:
            assert abs(_cell(nx, a, "Price") - px[a].iloc[-1]) < 1e-9


def test_trades_below_the_minimum_size_are_left_out_like_the_engine_does():
    px = _world()
    eng = EngineConfig(rebalance="M", execution_lag=1, min_trade_weight=0.005)
    w = pd.DataFrame(0.0, px.index, px.columns)
    w.iloc[:, :2] = 0.5
    held = run_backtest(px, w, eng, ZERO, None, "x").weights.iloc[-1]
    w.iloc[-1, 0] = held.iloc[0] + 0.002               # a 0.2-point nudge
    w.iloc[-1, 1] = held.iloc[1] - 0.002
    _, nx = _run(px, w, eng)
    assert not nx.has_trades
    assert (nx.table["Action"].isin(["Hold", ""])).all()


def test_whole_units_are_whole():
    px = _world()
    eng = EngineConfig(rebalance="M", execution_lag=1, whole_shares=True,
                       min_trade_weight=0.0)
    _, nx = _run(px, _rotating(px), eng)
    u = nx.table.dropna(subset=["Units to trade"])["Units to trade"]
    assert np.allclose(u, np.round(u))


def test_multi_currency_run_replays_and_lands_on_target():
    px_local = _world(n_assets=3)
    px_local.columns = ["CAD1", "US1", "US2"]
    rng = np.random.default_rng(9)
    rate = pd.Series(1.30 * np.exp(np.cumsum(rng.normal(0, 0.004, len(px_local)))),
                     px_local.index)
    book = FX.FxBook(base="CAD", ccy={"CAD1": "CAD", "US1": "USD", "US2": "USD"},
                     scale={}, source={}, rates=pd.DataFrame({"USD": rate}))
    px = px_local.copy()
    for c in ("US1", "US2"):
        px[c] = px_local[c] * rate
    w = _rotating(px, k=2)
    eng = EngineConfig(rebalance="M", execution_lag=1, min_trade_weight=0.0)
    res = run_backtest(px, w, eng, ZERO, None, "x", fx=book)
    nx = next_rebalance(px, w, eng, ZERO, fx=book, reference=res)
    assert nx.replay_diff < 1e-12
    for a in px.columns:
        assert abs(_cell(nx, a, "Weight after") - w.iloc[-1][a]) < 1e-6


def test_leverage_cap_is_applied_to_the_target_like_the_engine_does():
    px = _world()
    w = pd.DataFrame(0.0, px.index, px.columns)
    w.iloc[:, 0] = 1.0
    w.iloc[:, 1] = 1.0                                  # asks for 200% gross
    eng = EngineConfig(rebalance="M", execution_lag=1, max_leverage=1.0,
                       min_trade_weight=0.0)
    _, nx = _run(px, w, eng)
    assert abs(_cell(nx, "A", "Target weight") - 0.5) < 1e-12
    assert abs(_cell(nx, "A", "Weight after") - 0.5) < 1e-9


def test_imported_weight_dates_are_used_and_future_ones_are_unknown():
    px = _world()
    w = _rotating(px)
    dates = pd.DatetimeIndex(["2026-08-31", "2026-09-30"])
    eng = EngineConfig(rebalance="M", execution_lag=1)
    res = run_backtest(px, w, eng, ZERO, None, "x", dates)
    nx = next_rebalance(px, w, eng, ZERO, rebalance_dates=dates, reference=res)
    assert nx.scheduled_signal is None and not nx.is_scheduled
    assert nx.replay_diff < 1e-12
    assert any("Imported weights" in s for s in nx.notes)
