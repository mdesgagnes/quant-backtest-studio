"""Security review: metrics against hand-computed values, alignment edge
cases, profiles and rebalancing, splicing, regimes, objectives, verdict
rules, export parity, and calibration on synthetic data with known answers.

Everything runs offline: prices come from a fake downloader, the cache goes
to a temporary folder, and the macro series (T-bill, CPI) are switched off.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import pytest

from qbt import metrics as M
from qbt.security_review import config as CF
from qbt.security_review import data as D
from qbt.security_review import engine as E
from qbt.security_review import export as X
from qbt.security_review import impact as I
from qbt.security_review import portfolio as P
from qbt.security_review import report as RP
from qbt.security_review import scenarios as SC
from qbt.security_review import stats as S
from qbt.security_review import verdict as V
from qbt.security_review.bundle import load_bundle

AS_OF = pd.Timestamp("2026-11-15")


# ----------------------------------------------------------------------
# Hand-computed metrics
# ----------------------------------------------------------------------
def _m(vals, start="2020-01-31"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="ME"), dtype=float)


def test_underwater_hand_computed():
    # value: 1.1, 0.88, 0.924, 1.1088 -> under water in months 2 and 3
    u = S.underwater(_m([0.10, -0.20, 0.05, 0.20]))
    assert u["max_underwater"] == 2
    assert u["underwater_share"] == pytest.approx(0.5)
    assert u["recovery_months"] == 2      # trough in month 2, back above the peak in month 4


def test_underwater_not_recovered():
    u = S.underwater(_m([0.1, -0.2, 0.01]))
    assert np.isnan(u["recovery_months"]) and u["max_underwater"] == 2


def test_worst_12m_hand_computed():
    r = _m([0.01] * 12 + [-0.10])
    # windows: 1.01^12 - 1 and 1.01^11 * 0.9 - 1
    assert S.worst_12m(r) == pytest.approx(1.01 ** 11 * 0.9 - 1)
    assert np.isnan(S.worst_12m(_m([0.01] * 11)))


def test_ols_exact_line():
    x = _m(np.linspace(-0.05, 0.05, 40))
    res = S.ols(2 + 3 * x, x.rename("x").to_frame())
    assert res["coef"]["alpha"] == pytest.approx(2)
    assert res["coef"]["x"] == pytest.approx(3)
    assert res["r2"] == pytest.approx(1)


def test_standalone_matches_existing_metrics():
    rng = np.random.default_rng(1)
    r = _m(rng.normal(0.006, 0.04, 60))
    rf = pd.Series(0.001, index=r.index)
    s = S.standalone(r, rf)
    assert s["CAGR"] == pytest.approx((1 + r).prod() ** (12 / 60) - 1)
    assert s["Volatility"] == pytest.approx(r.std(ddof=0) * np.sqrt(12))
    assert s["Worst month"] == pytest.approx(r.min())
    assert s["Max drawdown"] == pytest.approx(M.max_drawdown(M.to_equity(r)))
    assert s["CVaR 95% (monthly)"] == pytest.approx(M.var_cvar(r)["cvar"])
    assert s["Sharpe"] == pytest.approx(M.sharpe(r, rf, 12))


def test_numpy_window_stats_match_pandas_versions():
    rng = np.random.default_rng(7)
    r = _m(rng.normal(0.005, 0.03, 36))
    rf = pd.Series(0.002, index=r.index)
    fast = I._np_stats(r.to_numpy(), rf.to_numpy())
    eq = M.to_equity(r)
    assert fast["Volatility"] == pytest.approx(M.annual_vol(r, 12))
    assert fast["CAGR"] == pytest.approx(M.ann_return(r, 12))
    assert fast["Sharpe"] == pytest.approx(M.sharpe(r, rf, 12))
    assert fast["Sortino"] == pytest.approx(M.sortino(r, rf, 12))
    assert fast["Max drawdown"] == pytest.approx(M.max_drawdown(eq))
    assert fast["Ulcer index"] == pytest.approx(M.ulcer_index(eq))
    assert fast["Calmar"] == pytest.approx(M.calmar(eq, 12))
    assert fast["CVaR 95% (monthly)"] == pytest.approx(M.var_cvar(r)["cvar"])


def test_relative_scaled_benchmark_calibration():
    """0.7 x benchmark: beta 0.7, lower volatility, wins every down month."""
    rng = np.random.default_rng(3)
    b = _m(rng.normal(0.007, 0.045, 120))
    r = 0.7 * b
    rel = S.relative(r, b, pd.Series(0.0, index=b.index))
    assert rel["Beta"] == pytest.approx(0.7, abs=1e-9)
    assert rel["Downside beta"] == pytest.approx(0.7, abs=1e-9)
    assert rel["Hit rate in benchmark down months"] == 1.0
    assert rel["Downside capture"] < 1 and rel["Upside capture"] < 1
    assert S.standalone(r, b * 0)["Volatility"] < S.standalone(b, b * 0)["Volatility"]


def test_relative_drawdown_hand_computed():
    b = _m([0.0] * 12 + [0.10, 0.0, 0.0])
    r = _m([0.0] * 15)
    # relative wealth: 1 ... 1, 1/1.1 -> drawdown 1/1.1 - 1
    assert S.relative(r, b, b * 0)["Relative drawdown"] == pytest.approx(1 / 1.1 - 1)


def test_equivalents_flags_near_identical_only():
    rng = np.random.default_rng(5)
    b = _m(rng.normal(0.006, 0.04, 120))
    clone = b + rng.normal(0, 0.0005, 120)
    other = 0.5 * b + rng.normal(0, 0.02, 120)
    rets = pd.DataFrame({"A": b, "CLONE": clone, "OTHER": other})
    eq = S.equivalents(rets, {"A": b, "CLONE": b, "OTHER": b})
    pairs = {(e["A"], e["B"]) for e in eq}
    assert ("A", "CLONE") in pairs
    assert not any("OTHER" in p for p in pairs)


# ----------------------------------------------------------------------
# Portfolio construction
# ----------------------------------------------------------------------
def test_monthly_rebalance_is_weighted_sum():
    R = pd.DataFrame({"A": _m([0.1, -0.05, 0.02]), "B": _m([0.0, 0.03, -0.01])})
    p = P.portfolio_returns(R, {"A": 0.6, "B": 0.4}, "monthly")
    assert np.allclose(p, 0.6 * R["A"] + 0.4 * R["B"])


def test_buy_and_hold_matches_value_ratio():
    R = pd.DataFrame({"A": _m([0.1, -0.05, 0.02, 0.04]), "B": _m([0.0, 0.03, -0.01, 0.0])})
    p = P.portfolio_returns(R, {"A": 0.5, "B": 0.5}, "none")
    value = 0.5 * (1 + R["A"]).cumprod() + 0.5 * (1 + R["B"]).cumprod()
    assert (1 + p).prod() == pytest.approx(value.iloc[-1])


def test_quarterly_rebalance_hand_computed():
    # Jan, Feb, Mar (rebalance at end of March), Apr
    R = pd.DataFrame({"A": _m([1.0, 0.0, 0.0, 0.0], "2021-01-31"),
                      "B": _m([0.0, 0.0, 0.0, 0.0], "2021-01-31")})
    R.loc[R.index[3], "A"] = 0.10
    p = P.portfolio_returns(R, {"A": 0.5, "B": 0.5}, "quarterly")
    assert p.iloc[0] == pytest.approx(0.5)
    # April starts from 50/50 again (rebalanced in March): 0.5 * 10%
    assert p.iloc[3] == pytest.approx(0.05)
    q = P.portfolio_returns(R, {"A": 0.5, "B": 0.5}, "annual")
    # no rebalance: A is 2/3 of the book in April
    assert q.iloc[3] == pytest.approx(2 / 3 * 0.10)


def test_add_and_replace_weights():
    base = {"X": 0.6, "Y": 0.4}
    a = P.add_weights(base, "C", 0.10)
    assert a == pytest.approx({"X": 0.54, "Y": 0.36, "C": 0.10})
    assert sum(P.add_weights(base, "X", 0.2).values()) == pytest.approx(1)
    r = P.replace_weights(base, "Y", "C")
    assert r == {"X": 0.6, "C": 0.4}
    with pytest.raises(KeyError):
        P.replace_weights(base, "Z", "C")


def test_diversification_ratio():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.04, 5000)
    b = rng.normal(0, 0.04, 5000)
    df = pd.DataFrame({"A": a, "B": b, "A2": 2 * a})
    assert P.diversification_ratio(df, {"A": 0.5, "A2": 0.5}) == pytest.approx(1.0)
    assert P.diversification_ratio(df, {"A": 0.5, "B": 0.5}) == pytest.approx(np.sqrt(2), rel=0.03)


# ----------------------------------------------------------------------
# Data layer: alignment, currency, splicing
# ----------------------------------------------------------------------
def test_month_end_drops_current_and_keeps_gaps():
    idx = pd.bdate_range("2024-01-02", "2024-06-14")
    px = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    px = px[~((px.index.month == 3))]                         # no March prices at all
    r = D.monthly_returns(px, pd.Timestamp("2024-06-14"))
    assert r.index[-1] == pd.Timestamp("2024-05-31")          # June in progress: dropped
    assert np.isnan(r[pd.Timestamp("2024-03-31")])            # gap month
    assert np.isnan(r[pd.Timestamp("2024-04-30")])            # never spans two months
    assert r[pd.Timestamp("2024-02-29")] == r[pd.Timestamp("2024-02-29")]


def test_to_cad_converts_usd_only():
    idx = pd.bdate_range("2024-01-01", periods=5)
    px = pd.Series([10.0, 11, 12, 13, 14], index=idx)
    fx = pd.Series([1.3, 1.31, 1.32, 1.33, 1.34], index=idx)
    assert D.to_cad(px, "SPY", fx).tolist() == pytest.approx((px * fx).tolist())
    assert D.to_cad(px, "XIC.TO", fx).equals(px)
    # rate carried forward at most FX_LIMIT sessions, never back
    fx2 = fx.iloc[:1]
    out = D.to_cad(px, "SPY", fx2)
    assert len(out) == 5 and out.iloc[-1] == pytest.approx(14 * 1.3)


def test_splice_flags_proxy_dates_only():
    prim = _m([0.01, 0.02], "2020-03-31")
    prox = _m([0.05, 0.06, 0.07, 0.08], "2020-01-31")
    s, f = D.splice(prim, prox)
    assert s.tolist() == [0.05, 0.06, 0.01, 0.02]               # primary wins where it exists
    assert f.tolist() == [True, True, False, False]
    s2, f2 = D.splice(prim, None)
    assert s2.equals(prim.dropna()) and not f2.any()


def test_quality_flags_detects_jump_and_stale():
    idx = pd.bdate_range("2024-01-01", periods=40)
    px = pd.Series(100.0, index=idx)
    px.iloc[20:] = 140.0
    fl = " ".join(D.quality_flags("T", px))
    assert "daily move" in fl and "unchanged" in fl


def test_distribution_yield():
    idx = pd.bdate_range("2019-01-01", "2024-12-31")
    fr = pd.DataFrame({"close": 100.0, "dividends": 0.0, "adj_close": 100.0}, index=idx)
    for y in range(2019, 2025):
        for m in (3, 6, 9, 12):
            fr.loc[fr.index[(fr.index.year == y) & (fr.index.month == m)][0], "dividends"] = 1.0
    out = D.distribution_yield(fr)
    assert out["dist_yield"] == pytest.approx(0.04)
    assert out["yield_stability"] == pytest.approx(0.0, abs=1e-9)


def test_parsers():
    s = D.parse_fred_csv("DATE,CPIAUCSL\n2020-01-01,100\n2020-02-01,.\n2020-03-01,101\n")
    assert s.tolist() == [100.0, 101.0]
    j = '{"observations":[{"d":"2020-01-01","V1":{"v":"1.5"}},{"d":"2020-02-01","V1":{"v":"x"}}]}'
    assert D.parse_valet_json(j, "V1").tolist() == [1.5]


# ----------------------------------------------------------------------
# Regimes and stress windows
# ----------------------------------------------------------------------
def test_regime_rules():
    y = _m([2.0, 2.1, 2.0, 2.0, 2.5])
    lab = SC.label_yield_change(y)
    assert lab.tolist()[1:] == [True, False, False, True] and np.isnan(lab.iloc[0])
    cpi = _m(list(np.linspace(100, 101, 13)) + [110.0])
    inf = SC.label_cpi_yoy(cpi, 0.03)
    assert inf.iloc[12] == False and inf.iloc[13] == True  # noqa: E712
    px = _m([10] * 10 + [5, 20])
    below = SC.label_below_ma(px, ma=10, lag=1)
    # month 11 (price 5) is below its average, but is only labelled in month 12
    assert below.iloc[11] == True and below.iloc[10] == False  # noqa: E712
    a, b = _m([-0.01, 0.02, -0.03]), _m([-0.02, -0.01, 0.01])
    assert SC.label_both_negative(a, b).tolist() == [True, False, False]


def test_expanding_median_has_no_look_ahead():
    s = _m(list(range(1, 81)))
    lab1 = SC.label_above_expanding_median(s, min_hist=60)
    s2 = s.copy()
    s2.iloc[-1] = -1e6                       # change only the future
    lab2 = SC.label_above_expanding_median(s2, min_hist=60)
    pd.testing.assert_series_equal(lab1.iloc[:-1], lab2.iloc[:-1])
    assert lab1.iloc[:60].isna().all()


def test_stress_window_no_data_and_buy_and_hold():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    a = pd.Series(np.linspace(100, 80, len(idx)), index=idx)
    b = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    late = b[b.index > "2020-03-01"]
    w = [CF.StressWindow("w", "W", "2020-02-03", "2020-04-30")]
    t = SC.stress_table("LATE", {"A": a, "LATE": late}, {"A": 1.0}, {"A": 0.9, "LATE": 0.1}, w)
    assert t["Status"].iloc[0].startswith("no data")
    t = SC.stress_table("B", {"A": a, "B": b}, {"A": 1.0}, {"A": 0.5, "B": 0.5}, w)
    ra = a.loc["2020-04-30"] / a.loc["2020-02-03"] - 1
    rb = b.loc["2020-04-30"] / b.loc["2020-02-03"] - 1
    assert t["Portfolio return, without"].iloc[0] == pytest.approx(ra)
    assert t["Portfolio return, with"].iloc[0] == pytest.approx(0.5 * ra + 0.5 * rb)


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
def test_bootstrap_is_seeded_and_bounded():
    a = I.bootstrap_index(50, 120, 100, 12, 42)
    b = I.bootstrap_index(50, 120, 100, 12, 42)
    assert (a == b).all() and a.shape == (100, 120) and a.max() < 50
    assert (np.diff(a[0, :12]) % 50 == 1).all()          # a block is consecutive (circular)


def test_ruin_and_decumulation_hand_cases():
    idx = I.bootstrap_index(10, 24, 5, 12, 1)
    assert I.prob_drawdown(np.full(10, -0.02), idx, [0.2, 0.5]) == {0.2: 1.0, 0.5: 0.0}
    # zero return, 4% a year, no indexation: money lasts exactly 25 years
    idx = I.bootstrap_index(10, 30 * 12, 3, 12, 1)
    out = I.decumulate(np.zeros(10), idx, 0.04, 0.0)
    assert out["depletion_prob"] == 1.0
    idx = I.bootstrap_index(10, 24 * 12, 3, 12, 1)
    out = I.decumulate(np.zeros(10), idx, 0.04, 0.0)
    assert out["depletion_prob"] == 0.0 and out["p50"] == pytest.approx(0.04, abs=1e-9)


# ----------------------------------------------------------------------
# Objectives, scores and verdict rules
# ----------------------------------------------------------------------
def test_config_files_are_complete():
    cfg = CF.load_all()
    assert len([p for p in cfg.profiles.profiles.values() if not p.custom]) == 7
    for p in cfg.profiles.profiles.values():
        assert sum(p.weights.values()) == pytest.approx(1.0)
    bal = cfg.profiles.profiles["balanced"].weights
    assert bal == pytest.approx({"XIC.TO": 0.18, "VFV.TO": 0.27, "XEF.TO": 0.12,
                                 "XEC.TO": 0.03, "XBB.TO": 0.40})
    assert cfg.profiles.profiles["bonds_100"].weights == {"XBB.TO": 1.0}
    expected = {"lower_volatility", "reduce_drawdown", "reduce_risk_of_ruin", "tail_protection",
                "upside_participation", "diversify_return_sources", "improve_risk_adjusted_return",
                "faster_recovery", "inflation_protection", "rising_rate_protection",
                "stock_bond_correlation_hedge", "income_generation",
                "currency_and_home_bias_diversification", "reduce_concentration",
                "sequence_risk_decumulation"}
    assert set(cfg.objectives.objectives) == expected
    for o in cfg.objectives.objectives.values():
        assert o.rolling_metric in cfg.objectives.metrics
        for g in o.guardrails:
            assert g["metric"] in cfg.objectives.metrics
    assert len(cfg.scenarios["windows"]) == 9
    assert set(cfg.scenarios["regimes"]) == {"rising_yields", "high_inflation", "equity_bear",
                                             "high_vol", "stock_bond_down"}
    assert sum(len(s.tickers) for s in cfg.universe.sleeves.values()) == 44


def test_custom_profile_roundtrip(tmp_path):
    for name in ("profiles.yaml",):
        (tmp_path / name).write_text(open(os.path.join(CF.CONFIG_DIR, name), encoding="utf-8").read(),
                                     encoding="utf-8")
    p = CF.custom_profile("My mix", {"xic.to": 30, "XBB.TO": 70}, "annual")
    assert p.weights == pytest.approx({"XIC.TO": 0.3, "XBB.TO": 0.7})
    CF.save_custom_profile(p, str(tmp_path))
    loaded = CF.load_profiles(str(tmp_path)).profiles[p.key]
    assert loaded.custom and loaded.rebalance == "annual"
    assert loaded.weights == pytest.approx(p.weights)
    with pytest.raises(CF.ConfigError):
        CF.custom_profile("empty", {"A": 0})


def test_metric_score_direction_and_clip():
    lower = CF.MetricSpec("d_vol", "lower", 0.0, 0.01, "")
    higher = CF.MetricSpec("x", "higher", 1.0, 0.2, "")
    assert V.metric_score(lower, -0.005) == pytest.approx(0.5)
    assert V.metric_score(lower, 0.05) == -1.0
    assert V.metric_score(higher, 1.1) == pytest.approx(0.5)
    assert np.isnan(V.metric_score(higher, np.nan))


def _obj(**kw):
    base = dict(key="o", label="O", description="", weights={"a": 1.0, "b": 1.0},
                rolling_metric="a", guardrails=[{"metric": "g", "min": -0.01, "text": "costs"}],
                add=0.3, consider=0.05, candidate_groups=[])
    base.update(kw)
    return CF.Objective(**base)


CAT = {k: CF.MetricSpec(k, "higher", 0.0, 1.0, k) for k in ("a", "b", "g")}


def test_score_renormalises_missing_metrics():
    sc = V.score(_obj(), {"a": 0.5, "b": np.nan}, CAT)
    assert sc["composite"] == pytest.approx(0.5) and sc["missing_share"] == pytest.approx(0.5)


def test_verdict_thresholds_and_caps():
    o = _obj()
    assert V.decide(o, {"a": 0.5, "b": 0.5, "g": 0}, CAT, 120, 0.9, 0.6)["verdict"] == "Add"
    assert V.decide(o, {"a": 0.1, "b": 0.1, "g": 0}, CAT, 120, 0.9, 0.6)["verdict"] == "Consider"
    assert V.decide(o, {"a": -0.1, "b": 0, "g": 0}, CAT, 120, 0.9, 0.6)["verdict"] == "Do not add"
    d = V.decide(o, {"a": 0.5, "b": 0.5, "g": -0.05}, CAT, 120, 0.9, 0.6)
    assert d["verdict"] == "Consider" and d["raw_verdict"] == "Add" and d["guardrails_failed"]
    d = V.decide(o, {"a": 0.5, "b": 0.5, "g": 0}, CAT, 120, 0.4, 0.6)
    assert d["verdict"] == "Consider"


def test_confidence_rules():
    assert V.confidence(120, 0.9, 0.6, {}, 0)["level"] == "High"
    assert V.confidence(50, 0.9, 0.6, {}, 0)["level"] == "Moderate"
    assert V.confidence(30, 0.9, 0.6, {}, 0)["level"] == "Low"
    assert V.confidence(120, 0.3, 0.6, {}, 0)["level"] == "Moderate"
    assert V.confidence(120, 0.9, 0.6, {0.05: "Add", 0.2: "Do not add"}, 0)["level"] == "Moderate"
    assert V.confidence(120, 0.9, 0.6, {}, 0.6)["level"] == "Low"
    assert V.confidence(120, 0.9, 0.6, {}, 0, stress_coverage=4 / 9)["level"] == "Moderate"


# ----------------------------------------------------------------------
# Synthetic world: the whole pipeline offline
# ----------------------------------------------------------------------
def _fake_world():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2001-01-02", "2026-11-13")
    n = len(idx)
    mkt = rng.normal(0.0003, 0.011, n)
    bond = rng.normal(0.00015, 0.003, n) - 0.05 * mkt
    rets = {}

    def walk(r):
        return 100 * np.cumprod(1 + r)
    eq_like = ["SPY", "IWM", "IWD", "IWF", "MTUM", "QUAL", "QQQ", "RSP", "VTV", "VUG", "SCHD",
               "SPLV", "EFA", "EFAV", "EEM", "EWC", "XIC.TO", "XIU.TO", "ZLB.TO", "XEI.TO",
               "ZWC.TO", "XEF.TO", "XEC.TO", "VFV.TO", "VNQ", "XRE.TO", "RSST", "RSSB"]
    bond_like = ["XBB.TO", "ZAG.TO", "XSB.TO", "XLB.TO", "XRB.TO", "ZCS.TO", "AGG", "SHY",
                 "IEF", "TLT", "TIP", "HYG", "LQD"]
    for t in eq_like:
        rets[t] = 0.9 * mkt + rng.normal(0, 0.004, n)
    for t in bond_like:
        rets[t] = bond + rng.normal(0, 0.001, n)
    for t in ["GLD", "DBC", "DBMF", "KMLM", "BTAL", "TAIL"]:
        rets[t] = rng.normal(0.0002, 0.008, n)
    rets["SPY"] = mkt
    rets["USMV"] = 0.7 * mkt + rng.normal(0, 0.002, n)          # calibration: low beta
    rets["NEGC"] = -0.5 * mkt + rng.normal(0.0002, 0.003, n)    # calibration: hedge
    rets["SPYCLONE"] = mkt + rng.normal(0, 0.0001, n)            # calibration: near-identical
    frames = {}
    starts = {"VFV.TO": "2012-11-01", "XEF.TO": "2013-05-01", "XEC.TO": "2013-05-01",
              "USMV": "2011-10-01", "DBMF": "2019-05-08"}
    for t, r in rets.items():
        px = pd.Series(walk(r), index=idx)
        if t in starts:
            px = px[px.index >= starts[t]]
        frames[t] = pd.DataFrame({"adj_close": px, "close": px, "dividends": 0.0})
    for t, level in (("CAD=X", 1.3), ("^TNX", 3.0), ("^VIX", 18.0)):
        lv = level * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        frames[t] = pd.DataFrame({"adj_close": lv, "close": lv, "dividends": 0.0}, index=idx)
    return frames


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    frames = _fake_world()

    def downloader(tickers, start):
        return {t: frames[t] for t in tickers if t in frames}
    cfg = CF.load_all()
    cache = str(tmp_path_factory.mktemp("cache"))
    b = load_bundle(cfg, ["USMV", "NEGC", "SPYCLONE", "DBMF", "NOSUCH"], downloader=downloader,
                    cache_dir=cache, as_of=AS_OF, macro=False)
    return cfg, b, E.regime_labels(cfg, b)


def test_bundle_flags_failed_and_drops_current_month(world):
    cfg, b, _ = world
    assert "NOSUCH" in b.failed
    assert b.monthly.index[-1] == pd.Timestamp("2026-10-31")


def test_profiles_build_and_flag_splices(world):
    cfg, b, _ = world
    for k, p in cfg.profiles.profiles.items():
        if p.custom:
            continue
        s = E.profile_series(cfg, b, k)
        assert len(s) > 250 and s.notna().all()
    _, h = E.holding_returns(cfg, b, cfg.profiles.profiles["balanced"].weights)
    vfv = h.set_index("Holding").loc["VFV.TO"]
    assert vfv["Proxy"] == "SPY" and vfv["Spliced to"] == "2012-11" and vfv["Spliced months"] > 100
    assert h.set_index("Holding").loc["XBB.TO", "Spliced months"] == 0


def test_calibration_low_beta_lowers_vol(world):
    cfg, b, reg = world
    rv = E.run(E.Request("equities_100", "lower_volatility", ["USMV", "SPYCLONE"]), cfg, b, reg)
    u, c = rv.results["USMV"], rv.results["SPYCLONE"]
    assert u.relative["Beta"] == pytest.approx(0.7 / 1.0, abs=0.05)
    assert u.metrics["d_vol"] < 0 and u.metrics["down_capture"] < 1
    assert u.decision["composite"] > c.decision["composite"]


def test_calibration_negative_correlation_lowers_portfolio_vol(world):
    cfg, b, reg = world
    rv = E.run(E.Request("balanced", "diversify_return_sources", ["NEGC"]), cfg, b, reg)
    m = rv.results["NEGC"].metrics
    assert m["d_vol"] < 0 and m["corr_to_portfolio"] < 0 and m["d_div_ratio"] > 0
    assert rv.results["NEGC"].decision["verdict"] == "Add"


def test_calibration_near_identical_flagged_equivalent(world):
    cfg, b, reg = world
    rv = E.run(E.Request("balanced", "lower_volatility", ["SPY", "SPYCLONE", "USMV"]), cfg, b, reg)
    pairs = {frozenset((e["A"], e["B"])) for e in rv.equivalents}
    assert frozenset(("SPY", "SPYCLONE")) in pairs
    assert frozenset(("SPY", "USMV")) not in pairs


def test_objective_and_profile_can_change_the_verdict(world):
    cfg, b, reg = world
    rv = E.run(E.Request("balanced", "lower_volatility", ["NEGC"]), cfg, b, reg)
    verdicts = set(rv.results["NEGC"].objectives["Verdict"])
    assert len(verdicts) >= 2


def test_replace_mode_and_short_history(world):
    cfg, b, reg = world
    rv = E.run(E.Request("growth", "reduce_drawdown", ["USMV"], mode="replace",
                         replace_holding="VFV.TO"), cfg, b, reg)
    r = rv.results["USMV"]
    assert r.weights_new["USMV"] == pytest.approx(0.70 * 0.45) and "VFV.TO" not in r.weights_new
    assert r.weight_grid.empty                          # no add-weight grid in Replace mode
    with pytest.raises(CF.ConfigError):
        E.run(E.Request("growth", "reduce_drawdown", ["USMV"], mode="replace",
                        replace_holding="SPY"), cfg, b, reg)
    rv = E.run(E.Request("balanced", "stock_bond_correlation_hedge", ["DBMF"]), cfg, b, reg)
    r = rv.results["DBMF"]
    st = r.stress.set_index("Key")
    assert st.loc["gfc", "Status"].startswith("no data")     # DBMF starts in 2019: never filled
    assert st.loc["rate_shock_2022", "Status"] == "ok"
    assert r.months < 96 and r.decision["confidence"] in ("High", "Moderate", "Low")


def test_common_alignment_uses_shared_months(world):
    cfg, b, reg = world
    rv = E.run(E.Request("balanced", "lower_volatility", ["USMV", "DBMF"]), cfg, b, reg)
    assert rv.results["USMV"].months == rv.results["DBMF"].months
    rv2 = E.run(E.Request("balanced", "lower_volatility", ["USMV", "DBMF"], alignment="each"),
                cfg, b, reg)
    assert rv2.results["USMV"].months > rv2.results["DBMF"].months


def test_sweep_covers_all_profiles(world):
    cfg, b, reg = world
    sw = E.sweep(cfg, b, "NEGC", "lower_volatility", regimes=reg, robustness=False)
    assert len(sw) == 7 and set(sw["Verdict"]) <= {"Add", "Consider", "Do not add"}


def test_export_has_every_sheet_and_every_metric(world):
    import openpyxl
    cfg, b, reg = world
    rv = E.run(E.Request("balanced", "lower_volatility", ["USMV", "SPYCLONE"]), cfg, b, reg)
    sw = E.sweep(cfg, b, "USMV", "lower_volatility", regimes=reg, robustness=False)
    secs = RP.sections(rv, cfg, sw)
    wb = openpyxl.load_workbook(io.BytesIO(X.write_sections(secs)))
    required = {"Verdict", "Fund Review", "Sleeve Comparison", "Diversification", "Portfolio Impact",
                "Scenarios", "Regimes", "Factor Exposures", "Robustness", "Profile Sweep", "Ranking",
                "Methodology"}
    assert required <= set(wb.sheetnames)
    # the page and the workbook use the same section list: same sheets
    assert set(s[0] for s in secs) == set(wb.sheetnames)
    ws = wb["Portfolio Impact"]
    keys = {ws.cell(row=r, column=2).value for r in range(1, ws.max_row + 1)}
    for c, r in rv.results.items():
        assert set(r.metrics) <= keys
    # styling of the backtest export: red bold title, black header
    assert ws["A1"].font.bold and ws["A2"].fill.fgColor.rgb.endswith("1A1A1A")
