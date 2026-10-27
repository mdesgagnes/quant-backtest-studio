"""Plain-language layer and the event study on strategy return streams."""
import numpy as np
import pandas as pd
import pytest

from qbt.lab import eventstudy as E, explain as X, factors as F
from qbt.lab.validate import grade_of, Check, THRESHOLDS


def _series(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    m = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
    return m, 1.1 * m + rng.normal(0, 0.004, n)


def test_every_validation_check_has_a_question_and_advice():
    keys = {"length", "psr", "dsr", "bootstrap", "sign_flip", "timing", "walk_forward", "oos", "costs", "pbo", "cpcv"}
    assert keys <= set(X.CHECK_HELP)
    for k in keys:
        q, advice = X.CHECK_HELP[k]
        assert q.endswith("?") and len(advice) > 20


def test_every_screen_column_has_hover_help():
    scr = F.screen(pd.DataFrame(np.cumprod(1 + np.random.default_rng(0).normal(0, .01, (900, 6)), axis=0) * 100,
                                index=pd.bdate_range("2018-01-01", periods=900), columns=list("ABCDEF")),
                   ["mom_126", "vol_63"], 21)
    shown = ["Mean IC", "ICIR", "t-stat", "p", "p_bonferroni", "p_bh", "Spread Sharpe", "Monotonicity",
             "Top-bucket turnover", "n"]
    for c in shown:
        assert c in scr.columns and c in X.COLUMN_HELP


def test_glossary_is_complete_and_well_formed():
    assert {"Factor Lab", "Strategy validation", "Event study", "Risk x-ray", "Monitor"} <= set(X.GLOSSARY)
    for rows in X.GLOSSARY.values():
        for term, meaning, good, watch in rows:
            assert term and len(meaning) > 20


@pytest.mark.parametrize("row,expect", [
    ({"p": 0.001, "p_bh": 0.01, "Mean IC": 0.1}, "Strong evidence"),
    ({"p": 0.001, "p_bh": 0.01, "Mean IC": -0.1}, "Works in reverse (strong)"),
    ({"p": 0.01, "p_bh": 0.07, "Mean IC": 0.05}, "Probable signal"),
    ({"p": 0.03, "p_bh": 0.4, "Mean IC": 0.05}, "Likely luck (fails correction)"),
    ({"p": 0.4, "p_bh": 0.9, "Mean IC": 0.01}, "No evidence"),
    ({"p": float("nan"), "p_bh": 0.9, "Mean IC": 0.01}, "Not enough data"),
])
def test_factor_verdict_rules(row, expect):
    assert X.factor_verdict(row) == expect


def test_plain_summary_mentions_failures_and_missing_deflation():
    checks = [Check("psr", "Probabilistic Sharpe (vs 0)", "pass", .99, "x"),
              Check("timing", "Random-timing test", "fail", .4, "x"),
              Check("dsr", "Deflated Sharpe (1 trial)", "na", None, "x")]
    txt = X.plain_summary({"checks": checks, "grade": grade_of(checks)})
    assert "1 of 2 checks pass" in txt and "random-timing" in txt.lower() and "hypothesis" in txt
    clean = [Check("psr", "PSR", "pass", .99, "x")]
    assert "Nothing in the battery argues against it" in X.plain_summary({"checks": clean, "grade": grade_of(clean)})


def test_worst_days_are_declustered_and_sorted():
    m, _ = _series()
    m.iloc[500:505] = [-0.09, -0.08, -0.07, -0.06, -0.05]            # one crash, five terrible days
    d = E.worst_days(m, 10, min_gap=10)
    assert len(d) == 10 and d == sorted(d)
    assert sum(pd.Timestamp(m.index[500]) <= x <= pd.Timestamp(m.index[504]) for x in d) == 1


def test_parse_dates_ignores_junk():
    assert E.parse_dates("2018-01-05\n2019-02-01, bad\n\n  2020-03-16 crash") == [
        pd.Timestamp("2018-01-05"), pd.Timestamp("2019-02-01"), pd.Timestamp("2020-03-16")]


def test_series_event_study_recovers_planted_effect_and_works_without_market():
    m, r = _series(seed=3)
    dates = [m.index[p] for p in range(300, 1300, 100)]
    # Each event's planted effect also sits inside the next event's estimation
    # window, so the measured CAR is a little under the planted 6%.
    for p in range(300, 1300, 100):
        r.iloc[p:p + 6] += 0.01                                       # +6% over the event's first six days
    with_mkt = E.series_event_study({"S": r}, m, dates, pre=5, post=15)["S"]
    assert with_mkt["n_events"] == 10 and with_mkt["mean_car"] == pytest.approx(0.05, abs=0.02)
    assert with_mkt["t_car"] > 3
    no_mkt = E.series_event_study({"S": r}, None, dates, pre=5, post=15)["S"]
    assert no_mkt["n_events"] == 10 and no_mkt["mean_car"] > 0.03
    assert E.series_event_study({"short": r.iloc[:30]}, m, dates) == {}


def test_event_panel_renders_every_source_in_streamlit():
    from streamlit.testing.v1 import AppTest

    def app():
        import numpy as np
        import pandas as pd
        from qbt.lab import ui
        from qbt.stress import DEFAULT_PERIODS
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2005-01-03", periods=4500)
        m = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
        r = pd.Series(0.8 * m.to_numpy() + rng.normal(0, 0.004, len(idx)), index=idx)
        ui.event_panel({"Strat": r}, m, periods=DEFAULT_PERIODS, rebalances=idx[::21], key="t")

    at = AppTest.from_function(app, default_timeout=120).run()
    assert not at.exception
    sources = at.selectbox(key="t_src").options
    assert len(sources) == 4
    for s in sources:
        at.selectbox(key="t_src").set_value(s).run()
        assert not at.exception, (s, at.exception)
        if s == "My own dates":
            at.text_area(key="t_txt").set_value("2012-03-01\n2015-08-24\n2018-02-05").run()
            assert not at.exception
    assert any("Average CAR" in m.value for m in at.markdown)
