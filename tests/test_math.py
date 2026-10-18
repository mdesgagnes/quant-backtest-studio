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
    ex = x - ((1.02) ** (1 / 252) - 1)
    dd = E / np.maximum.accumulate(E) - 1
    q = np.quantile(x, 0.05)
    beta, a0 = np.polyfit(b.to_numpy() - ((1.02) ** (1 / 252) - 1), ex, 1)
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
        "Alpha (ann.)": E[-1] ** (252 / n) - np.prod(1 + b.to_numpy()) ** (252 / n),
        "Risk-adjusted alpha (ann.)": a0 * 252,
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


# ----------------------------------------------------------------------
def test_trailing_windows_are_month_end_to_month_end():
    idx = pd.date_range("2019-12-31", "2026-09-30", freq="ME")
    r = pd.Series(np.linspace(-0.02, 0.03, len(idx)), idx)
    tr = M.trailing_returns(M.to_equity(r), 12).set_index("Period")
    assert str(tr.loc["1M", "From"]) == "2026-08-31"          # not 2026-07-31
    assert str(tr.loc["6M", "From"]) == "2026-03-31"          # not 2026-02-28
    assert np.isclose(tr.loc["1M", "Return"], r.iloc[-1])
    assert np.isclose(tr.loc["6M", "Return"], (1 + r.iloc[-6:]).prod() - 1)
    assert np.isclose(tr.loc["Since inception", "Return"], M.cagr(M.to_equity(r), 12))


def test_risk_free_rate_in_sharpe_sortino_alpha():
    rng = np.random.default_rng(11)
    idx = pd.date_range("1990-01-31", periods=360, freq="ME")
    b = pd.Series(rng.normal(0.007, 0.04, 360), idx)
    r = 0.5 * b + pd.Series(rng.normal(0.003, 0.01, 360), idx)
    y = pd.Series(np.linspace(8.0, 1.0, 400), pd.date_range("1989-12-01", periods=400, freq="21D"))
    rf = M.rf_from_yield(y, idx)
    # Each month earns the yield known at its start, for its calendar days.
    prev = idx[4] - pd.offsets.MonthEnd(1)
    y0 = y[:prev].iloc[-1] / 100
    assert np.isclose(rf.iloc[4], (1 + y0) ** ((idx[4] - prev).days / 365.25) - 1)
    s = M.summary(r, None, b, None, None, rf, 12)
    ex, bx = (r - rf).to_numpy(), (b - rf).to_numpy()
    beta, a0 = np.polyfit(bx, ex, 1)
    assert np.isclose(s["Beta"], beta) and np.isclose(s["Risk-adjusted alpha (ann.)"], a0 * 12)
    # Alpha is the outperformance: CAGR minus the benchmark's CAGR, whatever
    # the risk-free rate.
    sb = M.summary(b, None, None, None, None, rf, 12)
    assert np.isclose(s["Alpha (ann.)"], s["CAGR"] - sb["CAGR"])
    assert np.isclose(M.summary(r, None, b, None, None, 0.0, 12)["Alpha (ann.)"],
                      s["Alpha (ann.)"])
    assert np.isclose(s["Sharpe"], ex.mean() / ex.std(ddof=1) * np.sqrt(12))
    assert np.isclose(s["Sortino"], ex.mean() / np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(12))
    # A fixed rate compounds to itself, like the engine's cash.
    assert np.isclose((1 + M.rf_per_period(idx, 0.03, 12).iloc[0]) ** 12, 1.03)
    # Zero risk-free credits a low-beta portfolio with the cash yield.
    assert (M.summary(r, None, b, None, None, 0.0, 12)["Risk-adjusted alpha (ann.)"]
            > s["Risk-adjusted alpha (ann.)"])


def test_benchmark_frictions_are_opt_in():
    from qbt.engine import (align_start, benchmark_costs, benchmark_result,
                            blended_benchmark, common_start)
    px, _, _ = _market(n=700, m=2, seed=5)
    eng = EngineConfig(initial_capital=100_000.0)
    cost = CostConfig(commission_bps=5, slippage_bps=25, management_fee_pa=0.01)
    s = px["A0"]

    # Nothing selected: no costs object, and the benchmark is the bare index.
    assert benchmark_costs(cost, []) is None
    bare = benchmark_result(s, eng)
    assert np.isclose(bare.equity.iloc[-1] / bare.equity.iloc[0], s.iloc[-1] / s.iloc[0])
    assert float(bare.costs.sum()) == 0.0

    # Each choice carries only its own friction.
    only = benchmark_costs(cost, ["commission"])
    assert (only.commission_bps, only.slippage_bps, only.management_fee_pa) == (5, 0, 0)
    only = benchmark_costs(cost, ["slippage"])
    assert (only.commission_bps, only.slippage_bps, only.management_fee_pa) == (0, 25, 0)
    only = benchmark_costs(cost, ["fee"])
    assert (only.commission_bps, only.slippage_bps, only.management_fee_pa) == (0, 0, 0.01)

    # One purchase, then drift: the entry costs commission + slippage once,
    # and the rest of the path is the price path.
    paid = benchmark_result(s, eng, costs=benchmark_costs(cost, ["commission", "slippage"]))
    rate = (5 + 25) / 10_000.0
    assert np.isclose(paid.equity.iloc[0], 100_000.0 / (1 + rate), rtol=1e-6)
    assert np.isclose(paid.equity.iloc[-1] / paid.equity.iloc[0], s.iloc[-1] / s.iloc[0])
    # The statistics read the returns, so they must reconcile to the value
    # curve: the entry cost is in both.
    assert np.isclose((1 + paid.returns).prod(), paid.equity.iloc[-1] / 100_000.0)
    assert np.isclose(paid.returns.iloc[0], -rate / (1 + rate))
    fee_only = benchmark_result(s, eng, costs=benchmark_costs(cost, ["fee"]))
    assert np.isclose((1 + fee_only.returns).prod(), fee_only.equity.iloc[-1] / 100_000.0)
    assert fee_only.returns.iloc[0] == 0.0
    assert np.isclose(float(paid.costs.sum()), rate / (1 + rate), rtol=1e-6)   # fraction of NAV

    # The fee is a drag of about its rate per year, on top of the index.
    fee = benchmark_result(s, eng, costs=benchmark_costs(cost, ["fee"]))
    years = (s.index[-1] - s.index[0]).days / 365.25
    ratio = fee.equity.iloc[-1] / bare.equity.iloc[-1]
    assert 0.98 ** years < ratio < 1.0
    assert abs(np.log(ratio) / years + 0.01) < 0.0015

    # A blend pays on its entry and on every rebalance back to target.
    w = {"A0": 0.6, "A1": 0.4}
    free = blended_benchmark(px, w, eng, rebalance="Q")
    dear = blended_benchmark(px, w, eng, rebalance="Q",
                             costs=benchmark_costs(cost, ["commission", "slippage"]))
    assert float(free.costs.sum()) == 0.0
    assert float(dear.costs.sum()) > rate / (1 + rate)
    assert dear.equity.iloc[-1] < free.equity.iloc[-1]


def test_benchmark_entry_cost_survives_the_warmup_trim():
    from qbt.engine import (align_start, benchmark_costs, benchmark_result,
                            common_start)
    px, _, _ = _market(n=700, m=3, seed=6)
    eng = EngineConfig(initial_capital=100_000.0)
    cost = CostConfig(commission_bps=5, slippage_bps=25)
    strat = run_backtest(px, _weights(px), eng, ZERO)     # flat for 40 sessions
    start = common_start(strat)
    assert start > px.index[0]
    bc = benchmark_costs(cost, ["commission", "slippage"])
    rate = (5 + 25) / 10_000.0

    # Bought on its own first session, the entry falls in the warm-up and
    # is trimmed away with it: nothing is charged.
    early = benchmark_result(px["A0"], eng, costs=bc)
    _, early = align_start(strat, early, initial_capital=100_000.0)
    assert float(early.costs.sum()) == 0.0

    # Bought on the shared start date, the entry is part of the record.
    late = benchmark_result(px["A0"], eng, costs=bc, start=start)
    assert late.equity.index[0] == start
    _, late = align_start(strat, late, initial_capital=100_000.0)
    assert np.isclose(float(late.costs.sum()), rate / (1 + rate), rtol=1e-6)
    assert late.equity.index[0] == start


def test_strategy_and_benchmark_both_pay_their_entry():
    from qbt.engine import (align_start, benchmark_costs, benchmark_result,
                            common_start, window)
    px, _, _ = _market(n=700, m=3, seed=6)
    eng = EngineConfig(initial_capital=100_000.0)
    cost = CostConfig(commission_bps=5, slippage_bps=25)
    rate = 30 / 10_000.0
    entry = rate / (1 + rate)

    # A strategy that is flat for a warm-up, then holds the one asset.
    w = pd.DataFrame(0.0, px.index, px.columns)
    w.iloc[40:, 0] = 1.0
    raw = run_backtest(px, w, eng, cost)
    strat = trim_warmup(raw, None, 100_000.0)

    # Its first buy is in the record: the first return is the entry cost,
    # the starting value is capital less it, and the two reconcile.
    assert np.isclose(strat.returns.iloc[0], -entry)
    assert np.isclose(strat.equity.iloc[0], 100_000.0 * (1 - entry))
    assert np.isclose((1 + strat.returns).prod(), strat.equity.iloc[-1] / 100_000.0)
    assert (strat.contributions.sum(axis=1) - strat.returns).abs().max() < 1e-12

    # The benchmark, bought the same day with the same costs, starts level.
    start = common_start(raw)
    bench = benchmark_result(px["A0"], eng, costs=benchmark_costs(
        cost, ["commission", "slippage"]), start=start)
    s2, b2 = align_start(raw, bench, initial_capital=100_000.0)
    assert np.isclose(s2.equity.iloc[0], b2.equity.iloc[0])
    assert np.isclose(s2.returns.iloc[0], b2.returns.iloc[0])
    # Frictionless, the strategy and the benchmark are the same series.
    free = run_backtest(px, w, eng, ZERO)
    s3, b3 = align_start(free, benchmark_result(px["A0"], eng, start=start),
                         initial_capital=100_000.0)
    assert np.allclose(s3.equity.values, b3.equity.values)

    # A report window opening later is a plain origin: no entry is charged.
    win = window(raw, px.index[300])
    assert win.returns.iloc[0] == 0.0


def test_compared_strategies_enter_on_the_shared_start():
    from qbt import compare as CMP
    from qbt.engine import align_start, common_start, first_active_date
    px, _, _ = _market(n=700, m=3, seed=8)
    eng = EngineConfig(initial_capital=100_000.0)
    cost = CostConfig(commission_bps=5, slippage_bps=25)
    rate = 30 / 10_000.0
    entry = rate / (1 + rate)

    # Two strategies hold the same single asset; one warms up for 40
    # sessions, the other for 200.
    def holds_from(n):
        w = pd.DataFrame(0.0, px.index, px.columns)
        w.iloc[n:, 0] = 1.0
        return w
    early_w, late_w = holds_from(40), holds_from(200)
    early = run_backtest(px, early_w, eng, cost)
    late = run_backtest(px, late_w, eng, cost)
    start = common_start(early, late)
    assert start == first_active_date(late) > first_active_date(early)

    # Merely trimmed, the earlier one is cut down to that date without ever
    # paying to enter on it.
    e_trim, l_trim = align_start(early, late, initial_capital=100_000.0)
    assert e_trim.returns.iloc[0] == 0.0 and np.isclose(l_trim.returns.iloc[0], -entry)

    # Run again with its weights held at zero until then, it enters on the
    # shared date like the other, and the two start level.
    again = run_backtest(px, CMP.masked_until(early_w, start, eng.execution_lag), eng, cost)
    assert first_active_date(again) == start
    e2, l2 = align_start(again, late, initial_capital=100_000.0)
    assert np.isclose(e2.returns.iloc[0], -entry)
    assert np.isclose(e2.equity.iloc[0], l2.equity.iloc[0])
    assert np.allclose(e2.equity.values, l2.equity.values)    # same holding, same path
    # Nothing before `start` is touched but the weights themselves.
    m = CMP.masked_until(early_w, start, eng.execution_lag)
    assert (m.loc[:px.index[150]].sum().sum() == 0.0) and m.loc[start].equals(early_w.loc[start])

    # Tables over the same results.
    allr = {"A": early, "B": late}
    assert list(CMP.unique_labels(["A"], ["A", "A", "C"])) == ["A (2)", "A (3)", "C"]
    t = CMP.comparison_table({"A": e2, "B": l2}, None, 0.0, 252)
    assert np.isclose(t.loc["CAGR", "A"], t.loc["CAGR", "B"])
    h = CMP.head_to_head("A", {"A": e2, "B": l2}, 0.0, 252)
    assert abs(h.loc["B", "Excess CAGR"]) < 1e-9
    assert CMP.calendar_year_table(allr).shape[1] == 2
    assert np.isclose(CMP.return_correlation({"A": e2, "B": l2}).loc["A", "B"], 1.0)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)


def test_executive_summary_lists_every_position_and_cash():
    """The current-weights table must show every held or targeted position
    plus cash. It once appended Cash with `.loc[len(table) + 1]`, which
    overwrote a real position whenever that label already existed in the
    filtered table: a book of five positions drawn from eight names, with
    the sixth name held, lost it."""
    from qbt.exec_summary import current_weights_table
    px, _, _ = _market(m=8)
    w = pd.DataFrame(0.0, px.index, px.columns)
    held = ["A0", "A1", "A3", "A4", "A6"]               # labels 0,1,3,4,6: len + 1 == 6
    for c in held:
        w[c] = 0.2
    w.iloc[:40] = 0.0
    res = run_backtest(px, w, EngineConfig(min_trade_weight=0.0, rebalance="M"), ZERO)
    table = current_weights_table(res)
    assert set(table["Instrument"]) == set(held) | {"Cash"}, list(table["Instrument"])
    assert table["Instrument"].is_unique and table["Instrument"].iloc[-1] == "Cash"
    # nothing is lost: the weights shown add up to the whole portfolio
    assert abs(table["Weight"].sum() - 1.0) < 1e-9
    assert abs(table["Value"].sum() - float(res.equity.iloc[-1])) < 1e-6
    # every sizing of book, from one name to all of them
    for k in range(1, 9):
        w2 = pd.DataFrame(0.0, px.index, px.columns)
        for c in px.columns[:k]:
            w2[c] = 1.0 / k
        w2.iloc[:40] = 0.0
        r2 = run_backtest(px, w2, EngineConfig(min_trade_weight=0.0, rebalance="M"), ZERO)
        got = current_weights_table(r2)
        assert set(got["Instrument"]) == set(px.columns[:k]) | {"Cash"}, k
        # and in any order: the original labels are what used to collide
        px3 = px[list(px.columns[::-1])]
        w3 = w2[px3.columns]
        r3 = run_backtest(px3, w3, EngineConfig(min_trade_weight=0.0, rebalance="M"), ZERO)
        assert set(current_weights_table(r3)["Instrument"]) == set(px.columns[:k]) | {"Cash"}, k


def test_risk_parity_score_with_downside_volatility():
    """The downside option used to raise NameError (downside_vol was not imported)."""
    from qbt.strategies import REGISTRY
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (300, 3)), 0)),
                      idx, list("ABC"))
    s = REGISTRY["risk_parity"].score(px, {"downside": True})
    assert s is not None and s.iloc[-1].notna().all()
