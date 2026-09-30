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


# ----------------------------------------------------------------------
def test_fee_liquidation_sells_whole_units_and_covers_the_fee():
    px, _, _ = _market()
    px = px * 4                       # prices around $200: a unit is lumpy
    w = pd.DataFrame(1 / 3, px.index, px.columns)
    eng = EngineConfig(whole_shares=True, min_trade_weight=0.0, rebalance="Q")
    cost = CostConfig(commission_bps=5, slippage_bps=10, management_fee_pa=0.02)
    res = run_backtest(px, w, eng, cost)
    assert np.allclose(res.shares.to_numpy(), np.round(res.shares.to_numpy()))
    fee_sales = res.trades[res.trades.get("Reason", "") == "Fee liquidation"]
    assert len(fee_sales) > 0
    assert np.allclose(fee_sales["Change"], np.round(fee_sales["Change"]))
    # The fee was paid in full, from whole-unit proceeds: cash never goes
    # negative after a month-end deduction.
    cash = res.cash_weight * res.equity
    assert cash.min() > -1e-6
    assert res.fees.sum() > 0


def test_fee_sales_are_logged_even_without_frictions():
    px, _, _ = _market()
    w = pd.DataFrame(1 / 3, px.index, px.columns)
    cost = CostConfig(commission_bps=0, slippage_bps=0, management_fee_pa=0.02,
                      apply_frictions_to_fee_liquidation=False)
    res = run_backtest(px, w, EngineConfig(min_trade_weight=0.0, rebalance="Q"), cost)
    held = res.shares.diff().fillna(res.shares)
    logged = (res.trades.groupby(["Date", "Instrument"])["Change"].sum()
              .unstack().reindex(index=held.index, columns=held.columns).fillna(0.0))
    assert np.allclose(held.to_numpy(), logged.to_numpy(), atol=1e-8)


def test_square_root_impact_matches_formula():
    px, _, _ = _market()
    vol = pd.DataFrame(20_000.0, px.index, px.columns)
    w = pd.DataFrame(1 / 3, px.index, px.columns)
    cost = CostConfig(commission_bps=5, slippage_bps=10,
                      impact_model="sqrt_vol", impact_coef=0.8)
    eng = EngineConfig(min_trade_weight=0.0, rebalance="Q", initial_capital=1e6)
    res = run_backtest(px, w, eng, cost, volume=vol)
    t = res.trades.iloc[-1]
    i = px.index.get_loc(t["Date"])
    sig = np.log(px[t["Instrument"]]).diff().iloc[i - 20:i].std()
    adv = (vol[t["Instrument"]] * px[t["Instrument"]]).iloc[i - 20:i].mean()
    ref = 5e-4 + 10e-4 + 0.8 * sig * np.sqrt(t["Notional"] / adv)
    # The engine sizes the rate on the pre-cash-check order: allow for that.
    assert abs(t["Effective Cost (bps)"] / 1e4 - ref) < 0.02 * ref


def test_fee_is_paid_from_cash_before_any_sale():
    px, _, _ = _market()
    eng = EngineConfig(min_trade_weight=0.0, rebalance="Q")
    cost = CostConfig(commission_bps=5, slippage_bps=10, management_fee_pa=0.02)
    # 10% held in cash covers every monthly fee: nothing is sold for it.
    res = run_backtest(px, pd.DataFrame(0.3, px.index, px.columns), eng, cost)
    assert "Reason" not in res.trades or not (res.trades["Reason"] == "Fee liquidation").any()
    assert res.fees.sum() > 0
    # Fully invested apart from a sliver: the sale raises only what the
    # cash cannot cover, so the account ends each fee date at zero cash
    # rather than holding the proceeds of a full-fee sale.
    res = run_backtest(px, pd.DataFrame(0.3333, px.index, px.columns), eng, cost)
    sales = res.trades[res.trades.get("Reason", "") == "Fee liquidation"]
    assert len(sales) > 0
    used = 0.0
    for d in sales["Date"].unique():
        cash_before = res.cash_weight.shift(1)[d] * res.equity.shift(1)[d]
        fee = res.fees[d] * res.equity.shift(1)[d]
        assert -1e-6 <= cash_before < fee                 # cash first, then a sale
        assert abs(res.cash_weight[d] * res.equity[d]) < 1e-6
        used += cash_before
    assert used > 0                                       # some months drew on cash


# ----------------------------------------------------------------------
def _layouts(px, op, div, vol):
    """The same market data written in every layout the importer reads."""
    import io
    long_ = pd.concat([pd.DataFrame({"Date": px.index.strftime("%Y-%m-%d"), "Ticker": c,
                                     "Open": op[c].values, "Close": px[c].values,
                                     "Volume": vol[c].values, "Dividends": div[c].values})
                       for c in px.columns])
    out = {"long": [("prices.csv", long_.to_csv(index=False).encode())]}
    out["per field"] = [(f"{n}.csv", fr.rename_axis("Date").to_csv().encode())
                        for n, fr in (("close", px), ("open", op),
                                      ("dividends", div), ("volume", vol))]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        pd.DataFrame({"Notes": ["exported for a test"]}).to_excel(xw, sheet_name="Read me")
        for c in px.columns:
            pd.DataFrame({"Open": op[c], "Close": px[c], "Volume": vol[c],
                          "Dividends": div[c]}).rename_axis("Date").to_excel(xw, sheet_name=c)
    out["sheet per ticker"] = [("book.xlsx", buf.getvalue())]
    hdr = ["Price"] + ["Close"] * 3 + ["Open"] * 3 + ["Dividends"] * 3
    tick = ["Ticker"] + list(px.columns) * 3
    rows = [hdr, tick, ["Date"] + [""] * 9]
    for d in px.index:
        rows.append([d.strftime("%Y-%m-%d")] + list(px.loc[d]) + list(op.loc[d]) + list(div.loc[d]))
    out["yahoo csv"] = [("yf.csv", "\n".join(",".join(map(str, r)) for r in rows).encode())]
    return out


def test_imported_files_run_exactly_like_the_source_data():
    from qbt import price_import as PI
    px, op, div = _market(n=400)
    vol = pd.DataFrame(1e5, px.index, px.columns)
    w = _weights(px)
    eng = EngineConfig(execute_at_open=True)
    ref = run_backtest(px, w, eng, CostConfig(), open_prices=op, dividends=div)
    for name, files in _layouts(px, op, div, vol).items():
        imp = PI.read_files(files)
        mkt = PI.to_market(imp, "auto")
        assert not mkt.adjusted, name                   # dividends -> price return
        cols = list(px.columns)
        got = run_backtest(mkt.close[cols], w, eng, CostConfig(),
                           open_prices=mkt.open[cols],
                           dividends=mkt.dividends.reindex(columns=cols).fillna(0.0))
        assert np.allclose(got.equity.values, ref.equity.values, rtol=1e-10), name
        assert set(imp.instruments) == set(cols), name


def test_total_return_import_scales_ohlc_and_drops_dividends():
    from qbt import price_import as PI
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"Date": idx, "Ticker": "X", "Open": 99.0, "Close": 100.0,
                       "Adj Close": 95.0, "Dividends": [0, 0, 1.0, 0, 0]})
    imp = PI.read_files([("x.csv", df.to_csv(index=False).encode())])
    m = PI.to_market(imp, "total")
    assert m.adjusted and m.dividends is None
    assert np.allclose(m.close["X"], 95.0) and np.allclose(m.open["X"], 99.0 * 0.95)
    assert any("ignored" in n for n in m.notes)
    m = PI.to_market(imp, "price")                     # adj close kept for a benchmark
    assert not m.adjusted and np.allclose(m.close["X"], 100.0)
    assert m.adj_close is not None and m.dividends["X"].sum() == 1.0


# ----------------------------------------------------------------------
def test_fee_ledger_dollars_match_an_independent_accrual():
    px, _, div = _market()
    cost = CostConfig(commission_bps=5, slippage_bps=10, management_fee_pa=0.015)
    res = run_backtest(px, pd.DataFrame(0.3333, px.index, px.columns),
                       EngineConfig(min_trade_weight=0.0, rebalance="Q"), cost,
                       dividends=div)
    cf = res.cash_flows
    fees = cf[cf["Type"] == "Management fee"].set_index("Date")
    assert len(fees) > 0
    # Value on which each day accrues: the close, before that day's fee and
    # the costs of any sale made to pay it.
    pre = res.equity.copy()
    pre[fees.index] += -fees["Amount"] + fees["Trading costs"]
    last = res.equity.index[0]
    for d, row in fees.iterrows():
        span = pre.loc[last:d].iloc[1:] if last != res.equity.index[0] else pre.loc[:d].iloc[1:]
        expected = 0.015 / 252 * span.sum()
        assert abs(-row["Amount"] - expected) < 1e-6 * max(1.0, expected), d
        assert row["Sessions accrued"] == len(span)
        assert abs(row["Paid from cash"] + row["Raised by selling"] - (-row["Amount"])) < 1e-6             or row["Raised by selling"] == 0
        last = d
    # The ledger and the daily fee series agree to the cent.
    prev = res.equity.shift(1).fillna(1e5)
    assert abs(fees["Amount"].sum() + (res.fees * prev).sum()) < 1e-6
    divs = cf[cf["Type"] == "Dividend"]
    assert abs(divs["Amount"].sum() - (res.dividend_income * prev).sum()) < 1e-6


def test_cash_reinvestment_between_rebalances():
    px, _, div = _market(n=1200)
    w = pd.DataFrame(0.3333, px.index, px.columns)
    base = EngineConfig(min_trade_weight=0.0, rebalance="A")
    off = run_backtest(px, w, base, ZERO, dividends=div)
    eng = EngineConfig(min_trade_weight=0.0, rebalance="A", cash_sweep="M",
                       cash_buffer=0.01)
    on = run_backtest(px, w, eng, ZERO, dividends=div)
    sw = on.trades[on.trades["Reason"] == "Cash reinvestment"]
    assert len(sw) > 0 and (sw["Change"] > 0).all()           # buys only
    assert on.cash_weight.mean() < off.cash_weight.mean()
    for d in sw["Date"].unique():
        assert on.cash_weight[d] >= 0.01 - 1e-9                # buffer kept
        bought = sw.loc[sw["Date"] == d, "Instrument"]
        assert (on.weights.loc[d, bought] <= 0.3333 + 1e-9).all()  # never above target
        assert d not in set(on.rebalance_dates)
    # Still reconciles: contributions add up to the return every day.
    assert np.allclose(on.contributions.sum(axis=1), on.returns, atol=1e-12)
    # Threshold only: fires only after a close with cash above it.
    thr = EngineConfig(min_trade_weight=0.0, rebalance="A",
                       cash_sweep_threshold=0.015, cash_buffer=0.005)
    t = run_backtest(px, w, thr, ZERO, dividends=div)
    tdays = t.trades.loc[t.trades["Reason"] == "Cash reinvestment", "Date"].unique()
    assert len(tdays) > 0
    prev_cash = t.cash_weight.shift(1)
    assert (prev_cash[tdays] > 0.015).all()
    # Whole units stay whole.
    wu = run_backtest(px * 4, w, EngineConfig(min_trade_weight=0.0, rebalance="A",
                                              cash_sweep="M", whole_shares=True),
                      ZERO, dividends=div * 4)
    assert np.allclose(wu.shares.to_numpy(), np.round(wu.shares.to_numpy()))


# ----------------------------------------------------------------------
def test_semi_annual_rebalance():
    from qbt.schedule import RebalanceSpec, build_calendar, month_variants
    idx = pd.bdate_range("2015-01-01", "2019-10-15")
    cal = build_calendar(idx, RebalanceSpec(frequency="S"))
    ends = cal[:-1]                                   # the last half is incomplete
    assert set(ends.month) == {6, 12} and len(ends) == 9
    for d in ends:                                    # last session of its month
        same = idx[(idx.year == d.year) & (idx.month == d.month)]
        assert d == same[-1]
    assert cal[-1] == idx[-1]                          # like the other frequencies
    assert list(rebalance_calendar(idx, "S")) == list(cal)
    jj = build_calendar(idx, RebalanceSpec(frequency="S", anchor_month=1, day_rule="first"))
    assert set(jj.month) == {1, 7} and len(jj) == 10
    assert all(d == idx[(idx.year == d.year) & (idx.month == d.month)][0] for d in jj)
    assert len({v.label() for v in month_variants(RebalanceSpec(frequency="S"))}) == 6
    # The engine trades twice a year, one session after each signal date.
    px, _, _ = _market(n=1300)
    res = run_backtest(px, _weights(px), EngineConfig(rebalance="S"), ZERO)
    reb = res.rebalance_dates
    assert len(reb) >= 9
    years = pd.Series(1, index=reb[:-1]).groupby(reb[:-1].year).size()
    assert years.max() <= 2
    traded = set(pd.to_datetime(res.trades["Date"]))
    assert traded <= set(reb)


def test_rebalance_frequency_as_a_sweep_axis():
    from qbt.strategies import REGISTRY
    px, _, _ = _market(n=1300)
    strat = REGISTRY["risk_parity"]
    params = {q.key: q.default for q in strat.params}
    calls = []
    orig = strat.generate
    strat.generate = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        sw = R.parameter_sweep(px, strat, params, {R.REBALANCE_KEY: list(R.SWEEP_FREQUENCIES)},
                               EngineConfig(), ZERO)
        assert len(calls) == 1                          # signals built once
        assert list(sw[R.REBALANCE_KEY].cat.categories) == list(R.SWEEP_FREQUENCIES.values())
        w = orig(px, params, None, None)
        for code, label in R.SWEEP_FREQUENCIES.items():
            direct = R._stats(R._run(px, w, EngineConfig(rebalance=code), ZERO), 252)
            row = sw[sw[R.REBALANCE_KEY] == label].iloc[0]
            assert np.isclose(row["CAGR"], direct["CAGR"]) and np.isclose(row["Sharpe"], direct["Sharpe"])
        num = next(q.key for q in strat.params if q.kind in ("int", "float"))
        calls.clear()
        sw2 = R.parameter_sweep(px, strat, params,
                                {num: [20, 40], R.REBALANCE_KEY: ["M", "S"]}, EngineConfig(), ZERO)
        assert len(sw2) == 4 and len(calls) == 2 and "error" not in sw2
    finally:
        strat.generate = orig


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
