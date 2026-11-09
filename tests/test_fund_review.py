"""Fund review: the replica test must find no alpha in a product that IS its
replica, find planted alpha, and every export must build."""
import numpy as np
import pandas as pd
import pytest

from qbt import boc
from qbt import fund_review as FR
from qbt import fund_review_export as FX


def _data(alpha_daily: float = 0.0, beta: float = 0.4, noise: float = 0.0005, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", "2024-12-31")
    fac = pd.Series(rng.normal(0.0004, 0.011, len(idx)), idx)
    rf = pd.Series(0.02 / 252, idx)
    bond = pd.Series(rng.normal(0.0001, 0.003, len(idx)), idx)
    prod = beta * fac + (1 - beta) * rf + alpha_daily + rng.normal(0, noise, len(idx))
    eq2 = pd.Series(rng.normal(0.0003, 0.009, len(idx)), idx)
    df = pd.DataFrame({"Fund": prod, "Index": fac, "Bonds": bond, "Other equity": eq2, "rf": rf})
    df.loc[: idx[300], "Fund"] = np.nan        # the fund starts later than the index
    return df


def _cfg(**kw):
    base = dict(product="Fund", factor="Index", bond="Bonds", split_date="2021-01-01",
                base_portfolio={"Index": 0.3, "Other equity": 0.3, "Bonds": 0.4},
                equity_sleeves=["Index", "Other equity"], bond_sleeves=["Bonds"], mer=0.011,
                distribution=FR.Distribution(nav=10, income=0.25, capital_gains=0.4))
    base.update(kw)
    return FR.FundReviewConfig(**base)


def test_product_equal_to_replica_has_no_alpha():
    df = _data()
    rv = FR.run_review(df.drop(columns="rf"), df["rf"], _cfg(), 252)
    assert rv.beta == pytest.approx(0.4, abs=0.02)
    assert abs(rv.regression["alpha"]) < 0.01
    assert abs(rv.regression["alpha_t"]) < 2
    assert rv.verdict["checks"][0]["status"] in ("fail", "borderline")
    assert not rv.verdict["verdict"].startswith("Add")


def test_planted_alpha_is_found():
    df = _data(alpha_daily=0.03 / 252)        # +3%/yr, low noise
    rv = FR.run_review(df.drop(columns="rf"), df["rf"], _cfg(), 252)
    assert rv.regression["alpha"] == pytest.approx(0.03, abs=0.01)
    assert rv.regression["alpha_t"] > 2
    assert rv.verdict["checks"][0]["status"] == "pass"


def test_newey_west_coefficients_match_ols():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 0.5 + 2.0 * x + rng.normal(scale=0.1, size=200)
    out = FR.newey_west_ols(y, x)
    slope, icpt = np.polyfit(x, y, 1)
    assert out["coef"][0] == pytest.approx(icpt)
    assert out["coef"][1] == pytest.approx(slope)


def test_to_monthly_drops_unfinished_month():
    idx = pd.bdate_range("2024-01-01", "2024-03-15")
    m = FR.to_monthly(pd.Series(0.001, idx), 252)
    assert list(m.index.month) == [1, 2]


def test_replica_is_exact_mix_at_monthly_frequency():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    f, c = pd.Series(0.02, idx), pd.Series(0.001, idx)
    assert np.allclose(FR.replica_returns(f, c, 0.6, 12), 0.6 * 0.02 + 0.4 * 0.001)


def test_funding_weights():
    base = {"A": 0.3, "B": 0.3, "C": 0.4}
    w = FR.funding_weights(base, "P", 0.1, "beta-proportional", 0.4, ["A", "B"], ["C"])
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["A"] == pytest.approx(0.28) and w["C"] == pytest.approx(0.34)
    w = FR.funding_weights(base, "P", 0.1, "C", 0.4, [], [])
    assert w["C"] == pytest.approx(0.3) and w["P"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        FR.funding_weights({"A": 0.05}, "P", 0.1, "A", 0.4, [], [])


def test_config_yaml_round_trip_and_unknown_keys():
    cfg = _cfg(extra_periods=[{"name": "X", "start": "2020-02-19", "end": "2020-03-23"}])
    back = FR.config_from_yaml(FR.config_to_yaml(cfg))
    assert back == cfg
    with pytest.raises(ValueError):
        FR.config_from_dict({"product": "a", "factor": "b", "typo": 1})


def test_parse_valet_csv():
    text = ('"TERMS AND CONDITIONS"\n"https://www.bankofcanada.ca/terms/"\n\n"SERIES"\n'
            '"id","label","description"\n"V39065","3 month","Treasury bills"\n\n"OBSERVATIONS"\n'
            '"date","V39065"\n"2024-01-02","4.95"\n"2024-01-03",""\n"2024-01-04","4.97"\n')
    s = boc.parse_valet_csv(text, "V39065")
    assert list(s.values) == [4.95, 4.97]
    with pytest.raises(boc.BoCError):
        boc.parse_valet_csv("nothing here")


def test_exports_build():
    df = _data()
    rv = FR.run_review(df.drop(columns="rf"), df["rf"], _cfg(), 252)
    imgs = FX.charts(rv)
    assert {"growth", "relative", "calendar", "funding"} <= set(imgs)
    assert FX.render_pdf(rv, imgs)[:4] == b"%PDF"
    assert FX.render_pptx(rv, imgs)[:2] == b"PK"
    assert FX.render_excel(rv)[:2] == b"PK"
