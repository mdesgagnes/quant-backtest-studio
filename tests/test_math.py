"""Math checks: every figure the app reports against an independent
calculation. Run with `python -m pytest tests` or `python tests/test_math.py`.

Synthetic data only, no network: prices are random walks with dividends,
open prices and volume, so the checks exercise the same code paths as a
real run.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt import attribution as A                     # noqa: E402
from qbt import metrics as M                         # noqa: E402
from qbt import returns_input as RS                  # noqa: E402
from qbt import robustness as R                      # noqa: E402
from qbt import stress as S                          # noqa: E402
from qbt import tax as T                             # noqa: E402
from qbt.config import CostConfig, EngineConfig      # noqa: E402
from qbt.engine import rebalance_calendar, run_backtest, trim_warmup  # noqa: E402

ZERO = CostConfig(commission_bps=0, slippage_bps=0)


def _market(n=900, m=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    r = rng.normal(0.0004, 0.012, (n, m)) + rng.normal(0, 0.006, (n, 1))
    px = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), idx, [f"A{i}" for i in range(m)])
    op = px.shift(1) * np.exp(rng.normal(0, 0.004, (n, m)))
    op.iloc[0] = px.iloc[0]
    div = pd.DataFrame(0.0, idx, px.columns)
    q = idx[::63][1:]
    for c in px.columns:
        div.loc[q, c] = px.loc[q, c] * 0.006
    return px, op, div


def _weights(px, seed=1):
    rng = np.random.default_rng(seed)
    w = pd.DataFrame(rng.random(px.shape), px.index, px.columns)
    w = w.div(w.sum(1), axis=0)
    w.iloc[:40] = 0.0
    return w


# ----------------------------------------------------------------------
def test_metrics_match_reference_formulas():
    rng = np.random.default_rng(7)
    n = 1500
    idx = pd.bdate_range("2018-01-01", periods=n)
    b = pd.Series(rng.normal(0.0004, 0.011, n), idx)
    r = 0.8 * b + pd.Series(rng.normal(0.0002, 0.006, n), idx)
    s = M.summary(r, M.to_equity(r, 100.0), b, None, None, 0.02, 252)
    x = r.to_numpy()
    E = np.concatenate([[1.0], np.cumprod(1 + x)])
    ex = x - 0.02 / 252
    dd = E / np.maximum.accumulate(E) - 1
    q = np.quantile(x, 0.05)
    beta, a0 = np.polyfit(b.to_numpy() - 0.02 / 252, ex, 1)
    act = x - b.to_numpy()
    te = np.std(act, ddof=1) * np.sqrt(252)
    ref = {
        "Total Return": E[-1] - 1,
        "CAGR": E[-1] ** (252 / n) - 1,
        "Volatility": np.std(x, ddof=1) * np.sqrt(252),
        "Sharpe": ex.mean() / np.std(ex, ddof=1) * np.sqrt(252),
        "Sortino": ex.mean() / np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(252),
        "Max Drawdown": dd.min(),
        "Ulcer Index": np.sqrt(np.mean((dd * 100) ** 2)),
        "VaR 95% (daily)": q,
        "CVaR 95% (daily)": x[x <= q].mean(),
        "Beta": beta,
        "Alpha (ann.)": a0 * 252,
        "Tracking Error": te,
        "Information Ratio": act.mean() * 252 / te,
    }
    for k, v in ref.items():
        assert np.isclose(s[k], v, rtol=1e-9, atol=1e-12), (k, s[k], v)


def test_first_period_is_not_dropped():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    r = pd.Series([-0.20] + [0.01] * 11, idx)
    eq = RS.equity_from_returns(r, 100)
    s = M.summary(r, eq, None, None, None, 0.0, 12)
    assert eq.index[0] == pd.Timestamp("2019-12-31")
    assert np.isclose(s["Total Return"], (1 + r).prod() - 1)
    assert np.isclose(s["CAGR"], (1 + r).prod() - 1)          # exactly one year
    assert np.isclose(s["Max Drawdown"], -0.20)
    one = S.evaluate_periods(r, [S.StressPeriod("m", "2020-01-01", "2020-01-31", "Crash")])
    assert np.isclose(one["Max Drawdown"].iloc[0], -0.20)


def test_drawdown_episode_starts_at_peak():
    eq = M.to_equity(pd.Series([0.1, -0.2, 0.05, 0.2, -0.1, 0.3],
                               pd.bdate_range("2020-01-01", periods=6)), 100)
    row = M.drawdown_table(eq, 1, 252).iloc[0]
    assert str(row["Start"]) == "2020-01-01" and row["Sessions to Trough"] == 1


def test_engine_matches_reference_60_40():
    px, _, _ = _market(m=2)
    tw = np.array([0.6, 0.4])
    w = pd.DataFrame([tw] * len(px), px.index, px.columns)
    dates = rebalance_calendar(px.index, "M").union(pd.DatetimeIndex([px.index[0]]))
    res = run_backtest(px, w, EngineConfig(execution_lag=0, min_trade_weight=0), ZERO,
                       rebalance_dates=dates)
    R_ = px.pct_change().fillna(0).to_numpy()
    V, wt, reb = [1.0], tw.copy(), set(dates)
    for t in range(1, len(px)):
        g = wt @ R_[t]
        V.append(V[-1] * (1 + g))
        wt = wt * (1 + R_[t]) / (1 + g)
        if px.index[t] in reb:
            wt = tw.copy()
    assert np.allclose(res.equity / res.equity.iloc[0], V, rtol=1e-10)


def test_cash_rate_compounds_to_stated_rate():
    px, _, _ = _market(m=1)
    res = run_backtest(px, pd.DataFrame(0.0, px.index, px.columns), EngineConfig(),
                       CostConfig(cash_rate_pa=0.03))
    assert np.isclose(res.equity.iloc[252] / res.equity.iloc[0] - 1, 0.03, atol=1e-12)


def test_contributions_reconcile_every_day_and_period():
    px, op, div = _market()
    for at_open in (False, True):
        res = run_backtest(px, _weights(px), EngineConfig(execute_at_open=at_open, max_leverage=1.3),
                           CostConfig(commission_bps=5, slippage_bps=10, management_fee_pa=0.01,
                                      cash_rate_pa=0.02, borrow_rate_pa=0.05),
                           open_prices=op if at_open else None, dividends=div)
        res = trim_warmup(res)
        assert (res.contributions.sum(axis=1) - res.returns).abs().max() < 1e-12
        yearly = A.by_period(res, "Year")["Total"]
        comp = (1 + res.returns).groupby(res.returns.index.year).prod() - 1
        assert np.allclose(yearly.values, comp.values, atol=1e-12)
        s = A.summary(res, {"A0": "Equities", "A1": "Equities", "A2": "Fixed income"})
        assert np.isclose(s["Contribution"].sum(), A.total_return(res))
        assert np.isclose(s["Share of risk"].sum(), 1.0)


def test_robustness_reruns_reproduce_headline():
    px, op, div = _market(n=1200)
    w = _weights(px)
    eng = EngineConfig(execute_at_open=True)
    cst = CostConfig(commission_bps=5, slippage_bps=10, management_fee_pa=0.01, cash_rate_pa=0.02)
    head = trim_warmup(run_backtest(px, w, eng, cst, open_prices=op, dividends=div),
                       None, eng.initial_capital)
    kw = dict(weights=w, dividends=div, open_prices=op, start=head.equity.index[0])
    cs = R.cost_sensitivity(px, None, {}, eng, cst, [15], **kw)
    assert np.isclose(cs["CAGR"].iloc[0], M.cagr(head.equity))


def test_tax_zero_rate_leaves_curve_unchanged_and_acb_includes_costs():
    px, _, div = _market()
    res = run_backtest(px, _weights(px), EngineConfig(), CostConfig(), dividends=div)
    rep = T.evaluate(res.trades, res.shares, div, res.equity,
                     T.TaxSettings(marginal_rate=0.0, eligible_dividend_rate=0.0,
                                   foreign_dividend_rate=0.0))
    assert (rep.after_tax_equity - res.equity).abs().max() < 1e-6
    tr = pd.DataFrame({"Date": pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"]),
                       "Instrument": "X", "Change": [10.0, 10.0, -5.0],
                       "Price": [100.0, 120.0, 130.0], "Effective Cost (bps)": [10.0] * 3})
    gain = T.realized_gains(tr)["Realized Gain"].iloc[0]
    assert np.isclose(gain, 650 * 0.999 - (1000 + 1200) * 1.001 / 20 * 5)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
