"""Golden tests: the CIPM / Bacon (3rd ed.) worked examples.

If any of these fail, the implementation no longer matches the standard.
"""
import numpy as np
import pandas as pd
import pytest

from qbt import cipm as C
from qbt import metrics as M

# ----------------------------------------------------------------------
# Data (percent -> decimal)
P = np.array([0.3, 2.6, 1.1, -0.9, 1.4, 2.4, 1.5, 6.6, -1.4, 3.9, -0.5, 8.1, 4.0, -3.7, -6.1, 1.4,
              -4.9, -2.1, 6.2, 5.8, -6.4, 1.7, -0.4, -0.2, -2.1, 1.1, 4.7, 2.4, 3.3, -0.7, 4.7,
              0.6, 1.0, -0.2, 3.4, 1.0]) / 100
B = np.array([0.2, 2.5, 1.8, -1.1, 1.4, 2.3, 1.4, 6.5, -1.5, 4.2, -0.3, 8.3, 3.9, -3.8, -6.2, 1.5,
              -4.8, -2.0, 6.0, 5.6, -6.7, 1.9, -0.3, -0.1, -2.6, 0.7, 4.3, 2.9, 3.8, -0.2, 5.1,
              1.4, 1.3, 0.3, 3.4, 2.1]) / 100
RF = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.2,
               0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2,
               0.2, 0.2, 0.2]) / 100
IDX = pd.date_range("2021-01-31", periods=36, freq="ME")
p, b, rf = (pd.Series(x, IDX) for x in (P, B, RF))


# ----------------------------------------------------------------------
# Returns
def test_returns_golden():
    d = dict(v0=74.2, v1=104.4, flows=[37.1], days=[14], td=31)
    assert C.simple_dietz(74.2, 104.4, [37.1]) == pytest.approx(-0.0744, abs=5e-5)
    assert C.modified_dietz(**d) == pytest.approx(-0.0730, abs=5e-5)
    assert C.modified_dietz(**d, timing="bod") == pytest.approx(-0.0721, abs=5e-5)
    assert C.irr_expost(**d) == pytest.approx(-0.0727, abs=5e-5)
    assert C.twr([66.0], [37.1], 74.2, 104.4) == pytest.approx(-0.0993, abs=5e-5)


def test_annualise_refuses_under_a_year():
    with pytest.raises(ValueError):
        C.annualise([0.01] * 6, 12)
    assert C.annualise(list(P), 12) == pytest.approx(0.1329, abs=5e-5)


# ----------------------------------------------------------------------
# Risk and appraisal (36 monthly observations)
def test_risk_golden():
    assert M.ann_return(p, 12) == pytest.approx(0.1329, abs=5e-5)
    assert M.ann_return(b, 12) == pytest.approx(0.1462, abs=5e-5)
    assert M.rf_annual(IDX, rf, 12) == pytest.approx(0.0243, abs=5e-5)
    assert M.annual_vol(p, 12) == pytest.approx(0.1149, abs=5e-5)
    assert M.annual_vol(b, 12) == pytest.approx(0.1165, abs=5e-5)
    assert M.pop_skew(p) == pytest.approx(-0.24, abs=5e-3)
    assert M.pop_kurt(p, excess=False) == pytest.approx(2.98, abs=5e-3)
    assert M.sharpe(p, rf, 12) == pytest.approx(0.94, abs=1e-2)   # 0.9453; Bacon prints 0.94
    assert C.adjusted_sharpe(p, rf, 12) == pytest.approx(0.91, abs=1e-2)
    assert C.skew_adjusted_sharpe(p, rf, 12) == pytest.approx(0.91, abs=1e-2)
    assert C.m2(p, b, rf, 12) == pytest.approx(0.1344, abs=5e-5)
    assert C.m2(p, b, rf, 12, adjusted=True) == pytest.approx(0.1343, abs=5e-5)


def test_relative_golden():
    ba = M.beta_alpha(p, b, rf, 12)
    assert ba["tracking_error"] == pytest.approx(0.0119, abs=5e-5)           # geometric TE
    assert ba["information_ratio"] == pytest.approx(-0.98, abs=5e-3)
    arith_te = M.pop_std(p - b) * np.sqrt(12)
    assert arith_te == pytest.approx(0.0121, abs=5e-5)
    assert (1.1329 / 1.1462 - 1) == pytest.approx(-0.0116, abs=5e-5)
    assert ba["beta"] == pytest.approx(0.982, abs=5e-4)
    monthly_alpha = ((p - rf).mean() - ba["beta"] * (b - rf).mean())
    assert monthly_alpha == pytest.approx(-0.00082, abs=5e-6)
    assert ba["r2_reliable"]                                                   # R2 is high here


def test_downside_golden():
    mar = 0.005
    assert M.downside_deviation(p, 0.0, 12, mar) * np.sqrt(12) == pytest.approx(0.0733, abs=5e-5)
    assert M.sortino(p, 0.0, 12, mar) == pytest.approx(0.97, abs=5e-3)
    assert C.omega(P, mar) == pytest.approx(1.6, abs=0.05)


def test_alpha_is_withheld_when_r2_is_low():
    rng = np.random.default_rng(3)
    noise = pd.Series(rng.normal(0.005, 0.04, 60), pd.date_range("2018-01-31", periods=60, freq="ME"))
    bench = pd.Series(rng.normal(0.005, 0.04, 60), noise.index)
    t = C.appraisal_table(noise, bench, 0.0, 12)
    row = t[t.Measure == "Jensen alpha (annualised)"].iloc[0]
    assert np.isnan(row.Value) and "withheld" in row.Note
    assert np.isnan(t[t.Measure == "Beta"].Value.iloc[0])


def test_strict_annualisation_below_one_year():
    short = p.iloc[:6]
    assert np.isnan(M.sharpe(short, rf.iloc[:6], 12, strict=True))
    assert np.isnan(M.cagr(M.to_equity(short), 12, strict=True))
    assert not np.isnan(M.sharpe(p, rf, 12, strict=True))


def test_skill_t_stat_rules():
    assert C.years_for_significance(1.0) == pytest.approx(4.0)
    assert C.years_for_significance(0.5) == pytest.approx(16.0)


# ----------------------------------------------------------------------
# Attribution
UK_JP_US = ["UK", "Japan", "US"]
Q = [  # w, W, r, b  (Q1..Q4)
    ([.40, .30, .30], [.40, .20, .40], [.20, -.05, .06], [.10, -.04, .08]),
    ([.70, .20, .10], [.40, .30, .30], [-.05, .03, -.05], [-.07, .04, -.10]),
    ([.30, .50, .20], [.50, .40, .10], [-.20, .08, -.15], [-.25, .05, -.20]),
    ([.30, .50, .20], [.40, .40, .20], [.10, -.07, .25], [.05, -.05, .10]),
]


def test_single_period_attribution_golden():
    w, W, r, bb = (np.array(x) for x in Q[0])
    assert w @ r == pytest.approx(0.083, abs=5e-4)
    assert W @ bb == pytest.approx(0.064, abs=5e-4)
    assert C.semi_notional_return(w, bb) == pytest.approx(0.052, abs=5e-4)
    bf = C.brinson_fachler(w, W, r, bb)
    assert bf["allocation"].round(4).tolist() == pytest.approx([0.0, -0.0104, -0.0016], abs=5e-5)
    assert bf["selection"].round(4).tolist() == pytest.approx([0.04, -0.003, -0.006], abs=5e-5)
    assert bf["allocation"].sum() == pytest.approx(-0.012, abs=5e-5)
    assert bf["selection"].sum() == pytest.approx(0.031, abs=5e-5)
    assert C.bhb(w, W, r, bb)["interaction"].sum() == pytest.approx(0.001, abs=5e-5)
    g = C.geometric_attribution(w, W, r, bb)
    assert g["allocation"].sum() == pytest.approx(-0.0113, abs=5e-5)
    assert g["selection"].sum() == pytest.approx(0.0295, abs=5e-5)
    assert (1 + w @ r) / (1 + W @ bb) - 1 == pytest.approx(0.0179, abs=5e-5)
    # the two geometric totals compound exactly to the geometric excess
    assert abs(C.reconciliation_error(w, W, r, bb, "geometric")) < 1e-12
    assert abs(C.reconciliation_error(w, W, r, bb, "brinson_fachler")) < 1e-12
    assert abs(C.reconciliation_error(w, W, r, bb, "bhb")) < 1e-12


def test_off_benchmark_category_needs_an_explicit_return():
    with pytest.raises(ValueError):
        C.geometric_attribution([.5, .5], [1.0, 0.0], [.1, .2], [.08, np.nan])


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        C.brinson_fachler([.5, .4], [.5, .5], [.1, .1], [.1, .1])


def _periods():
    rs = [np.array(q[0]) @ np.array(q[2]) for q in Q]
    bs = [np.array(q[1]) @ np.array(q[3]) for q in Q]
    return rs, bs


def test_linking_golden():
    rs, bs = _periods()
    assert rs == pytest.approx([.083, -.034, -.05, .045], abs=5e-4)
    assert bs == pytest.approx([.064, -.046, -.125, .02], abs=5e-4)
    R, Bm = C.link(rs), C.link(bs)
    assert R == pytest.approx(0.0386, abs=5e-4)
    assert Bm == pytest.approx(-0.0941, abs=5e-4)
    E = [np.column_stack([C.brinson_fachler(*map(np.array, (w, W, r, bb)))[k] for k in ("allocation", "selection")])
         for (w, W, r, bb) in Q]
    # Q1 UK selection 4.0 -> Carino 3.62, Menchero 3.96, GRAP 3.41, Frongello 4.00
    expect = {"carino": 0.0362, "menchero": 0.0396, "grap": 0.0341, "frongello": 0.0400}
    for name, fn in C.LINKERS.items():
        adj = fn(E, rs, bs)
        assert sum(a.sum() for a in adj) == pytest.approx(R - Bm, abs=1e-9), name
        assert adj[0][0, 1] == pytest.approx(expect[name], abs=6e-5), name


def test_geometric_multi_period_totals():
    out = C.multi_period_attribution(
        [dict(w=q[0], W=q[1], r=q[2], b=q[3]) for q in Q], UK_JP_US, model="geometric")
    t = out["linked"].loc["Total"]
    assert t["Allocation"] == pytest.approx(0.0129, abs=6e-5)
    assert t["Selection"] == pytest.approx(0.1319, abs=6e-5)
    assert abs(out["reconciliation_error"]) < 1e-12
    # category rows add to the linked totals
    cats = out["linked"].drop("Total").sum()
    assert cats["Allocation"] == pytest.approx(t["Allocation"], abs=1e-12)
    assert cats["Selection"] == pytest.approx(t["Selection"], abs=1e-12)


@pytest.mark.parametrize("method", ["carino", "menchero", "grap", "frongello"])
def test_arithmetic_multi_period_reconciles(method):
    out = C.multi_period_attribution(
        [dict(w=q[0], W=q[1], r=q[2], b=q[3]) for q in Q], UK_JP_US,
        model="brinson_fachler", linking=method)
    assert abs(out["reconciliation_error"]) < 1e-9
    assert out["linked"].loc["Total"].sum() == pytest.approx(out["R"] - out["B"], abs=1e-9)


# ----------------------------------------------------------------------
def test_currency_naive_compounds_to_geometric_excess():
    x = C.currency_attribution_naive(r=0.083, r_local=0.06, b=0.064, b_local=0.05, b_semi_local=0.055)
    assert x["total"] == pytest.approx((1.083) / (1.064) - 1, abs=1e-12)


def test_composite_and_dispersion():
    assert C.composite_return([0.10, 0.02], [300.0, 100.0]) == pytest.approx(0.08)      # not 6%
    assert C.internal_dispersion([0.01, 0.03]) == pytest.approx(0.01)                    # population


def test_three_year_sd_needs_36_months():
    assert np.isnan(C.three_year_ex_post_sd(p.iloc[:30]))
    assert C.three_year_ex_post_sd(p) == pytest.approx(M.annual_vol(p, 12))


# ----------------------------------------------------------------------
# Benchmark-relative attribution on a real (synthetic-market) backtest
def _bt(leverage=1.0):
    from qbt.config import CostConfig, EngineConfig
    from qbt.engine import run_backtest, trim_warmup
    from test_math import _market, _weights
    px, op, div = _market(n=900)
    res = run_backtest(px, _weights(px), EngineConfig(max_leverage=leverage),
                       CostConfig(commission_bps=5, slippage_bps=10, management_fee_pa=0.01,
                                  cash_rate_pa=0.02, borrow_rate_pa=0.05), dividends=div)
    return trim_warmup(res), px


@pytest.mark.parametrize("linking", ["frongello", "carino", "menchero", "grap"])
@pytest.mark.parametrize("leverage", [1.0, 1.3])
def test_benchmark_relative_reconciles(linking, leverage):
    from qbt import attribution as A
    res, px = _bt(leverage)
    classes = {"A0": "Equity", "A1": "Equity", "A2": "Bonds"}
    cash = pd.Series(0.02 / 252, px.index)
    out = A.benchmark_relative(res, classes, {"Equity": 0.5, "Bonds": 0.4, A.CASH_CLASS: 0.1},
                               px.pct_change(), cash, period="Month", linking=linking)
    assert abs(out["reconciliation_error"]) < 1e-9
    lk = out["linked"]
    assert lk.loc["Total", "Total effect"] == pytest.approx(out["excess"], abs=1e-9)
    assert lk.loc["Total", "Allocation"] + lk.loc["Total", "Selection"] + lk.loc[A.COSTS_FEES, "Total effect"] \
        == pytest.approx(out["excess"], abs=1e-9)
    # each period closes on its own: allocation + selection + costs = portfolio - benchmark
    per = out["periods"]
    assert (per["Allocation"] + per["Selection"] + per[A.COSTS_FEES] - per["Excess"]).abs().max() < 1e-9
    # the portfolio return is the engine's own
    assert out["R"] == pytest.approx(A.total_return(res), abs=1e-9)


def test_benchmark_weights_must_sum_to_one():
    from qbt import attribution as A
    res, px = _bt()
    with pytest.raises(ValueError):
        A.benchmark_relative(res, {"A0": "Equity", "A1": "Equity", "A2": "Bonds"},
                             {"Equity": 0.5, "Bonds": 0.4}, px.pct_change(), pd.Series(0.0, px.index))


def test_headline_summary_withholds_beta_and_alpha_below_r2_floor():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2018-01-31", periods=60, freq="ME")
    noise = pd.Series(rng.normal(0.005, 0.04, 60), idx)
    bench = pd.Series(rng.normal(0.005, 0.04, 60), idx)
    s = M.summary(noise, None, bench, None, None, 0.0, 12)
    assert s["R²"] < 0.7
    assert np.isnan(s["Beta"]) and np.isnan(s["Risk-adjusted alpha (ann.)"])
    assert not np.isnan(s["Alpha (ann.)"])            # plain outperformance stays
    good = M.summary(0.9 * bench + 0.002 * pd.Series(rng.normal(size=60), idx), None, bench, None, None, 0.0, 12)
    assert good["R²"] > 0.7 and not np.isnan(good["Beta"])


def test_benchmark_relative_each_year_reconciles():
    from qbt import attribution as A
    res, px = _bt()
    out = A.benchmark_relative(res, {"A0": "Equity", "A1": "Equity", "A2": "Bonds"},
                               {"Equity": 0.5, "Bonds": 0.4, A.CASH_CLASS: 0.1},
                               px.pct_change(), pd.Series(0.02 / 252, px.index))
    yy = out["by_year"]
    assert len(yy) >= 3 and yy["Check"].abs().max() < 1e-9


# ----------------------------------------------------------------------
# Excel export
def _book(**kw):
    import io
    import openpyxl
    from qbt import analysis_sheet as AS, attribution as A, excel_export as XL
    from qbt.config import RunConfig
    res, px = _bt()
    res.label = "Strategy"
    import copy
    bench = copy.copy(res)                                   # any result will do as a benchmark
    bench.label = "Bench"
    rf = pd.Series(0.02 / 252, res.equity.index)
    stats = M.summary(res.returns, res.equity, bench.returns, res.turnover, res.exposure, rf, 252)
    bstats = M.summary(bench.returns, bench.equity, None, None, None, rf, 252)
    an = AS.build_analysis(
        res, bench, rf, 252, title="T", subtitle="s", footer="f", rf_label="2%",
        bench_label="Bench",
        attribution=dict(classes={"A0": "Equity", "A1": "Equity", "A2": "Bonds"},
                         bench_weights={"Equity": 0.5, "Bonds": 0.4, A.CASH_CLASS: 0.1},
                         asset_returns=px.pct_change(), proxies={}, period="Month",
                         linking="frongello"), **kw)
    data = XL.workbook_from_backtest(res, bench, stats, bstats, RunConfig(label="Strategy"),
                                     analysis=an)
    return openpyxl.load_workbook(io.BytesIO(data)), an


def test_excel_report_carries_the_new_analysis():
    wb, an = _book()
    assert "Attribution & Analysis" in wb.sheetnames
    ws = wb["Attribution & Analysis"]
    text = " ".join(str(c.value) for r in ws.iter_rows() for c in r if c.value is not None)
    for needle in ("BENCHMARK-RELATIVE ATTRIBUTION", "PERFORMANCE APPRAISAL", "Adjusted Sharpe",
                   "Methodology", "Brinson-Fachler", "RECONCILES"):
        assert needle.lower() in text.lower(), needle
    assert len(ws._charts) == 2
    perf = wb["Performance"]
    titles = [c.value for r in perf.iter_rows(max_col=1) for c in r if c.value]
    assert "CIPM Appraisal" in titles and "Statistics" in titles
    # the Statistics sheet formats each row by what it is, not by the column's name
    rows = {r[0].value: r[1] for r in perf.iter_rows(min_row=3, max_row=40, max_col=3) if r[0].value}
    assert "%" in rows["CAGR"].number_format and rows["Sharpe"].number_format == "0.00"
    assert "%" in rows["VaR 95% (daily)"].number_format
    # sheet is listed on About
    about = " ".join(str(c.value) for r in wb["About"].iter_rows() for c in r if c.value)
    assert "Attribution & Analysis" in about
    assert an["battr"]["reconciliation_error"] == pytest.approx(0, abs=1e-9)


def test_return_stream_workbook_gets_the_performance_analysis_sheet():
    import io
    import openpyxl
    from qbt import analysis_sheet as AS, excel_export as XL
    r = p.copy()
    eq = M.to_equity(r, 100.0)
    st = M.summary(r, eq, b, None, None, rf, 12)
    an = AS.build_analysis(r, b, rf, 12, title="T", rf_label="rf", returns_only=True,
                           label="P", bench_label="B")
    data = XL.workbook_from_returns(r, eq, st, b, M.to_equity(b, 100.0), None, 12, "P", "B",
                                    analysis=an)
    wb = openpyxl.load_workbook(io.BytesIO(data))
    assert "Performance Analysis" in wb.sheetnames
    text = " ".join(str(c.value) for row in wb["Performance Analysis"].iter_rows()
                    for c in row if c.value is not None)
    assert "Sharpe" in text and "Information ratio" in text
