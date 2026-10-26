"""Factor library: no look-ahead, correct research maths, planted-effect recovery."""
import numpy as np
import pandas as pd
import pytest

from qbt.lab import factors as F
from qbt.lab.run import run_strategy, synthetic_dataset


@pytest.fixture(scope="module")
def px():
    return synthetic_dataset(n_assets=6, years=4.5, seed=11).prices


@pytest.mark.parametrize("key", list(F.FACTORS))
def test_every_factor_is_causal(px, key):
    """A factor's value at t must not change when the future is removed."""
    t = 900
    full = F.compute(key, px, oriented=False).iloc[t]
    cut = F.compute(key, px.iloc[:t + 1], oriented=False).iloc[-1]
    both = pd.concat([full, cut], axis=1).dropna()
    assert len(both) > 0 or full.isna().all()
    np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-9, atol=1e-12)


def test_every_factor_returns_aligned_panel(px):
    for key in F.FACTORS:
        out = F.compute(key, px)
        assert out.shape == px.shape and not np.isinf(out.to_numpy(dtype=float)).any()


def test_perfect_predictor_has_ic_one(px):
    fwd = F.forward_returns(px, 21, 1)
    ic = F.information_coefficient(fwd, px, 21)
    assert ic["mean_ic"] == pytest.approx(1.0, abs=1e-9)


def test_random_scores_have_no_ic(px):
    rnd = pd.DataFrame(np.random.default_rng(0).normal(size=px.shape), index=px.index, columns=px.columns)
    ic = F.information_coefficient(rnd, px, 21)
    assert abs(ic["mean_ic"]) < 0.08 and ic["p_value"] > 0.01


def test_planted_momentum_is_found_and_null_is_not():
    edge = synthetic_dataset(edge=0.05, seed=3).prices
    out = F.screen(edge, ["mom_12_1", "mom_252"], 21)
    assert out.loc["mom_12_1", "Mean IC"] > 0.1 and out.loc["mom_12_1", "p_bh"] < 0.01
    null = synthetic_dataset(edge=0.0, seed=5).prices
    scr = F.screen(null, None, 21)
    # 43 looks at pure noise: the corrected p-values must not hand out discoveries freely
    assert (scr["p_bh"] < 0.05).sum() <= 3


def test_multiple_testing_adjustment():
    p = pd.Series([0.001, 0.01, 0.02, 0.2, 0.5], index=list("abcde"))
    adj = F.multiple_testing(p)
    assert (adj["p_bonferroni"] >= adj["p"]).all() and (adj["p_bh"] >= adj["p"]).all()
    assert adj.loc["a", "p_bonferroni"] == pytest.approx(0.005)
    assert (adj["p_bh"] <= adj["p_bonferroni"]).all()
    assert adj["p_bh"].is_monotonic_increasing


def test_newey_west_is_more_conservative_than_naive_on_autocorrelated():
    rng = np.random.default_rng(1)
    e = rng.normal(size=400)
    x = np.convolve(e, np.ones(10) / 10, mode="same") + 0.05
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(F._newey_west_t(x, 9)) < abs(naive)


def test_quantile_buckets_and_spread(px):
    sc = F.compute("mom_126", px)
    q = F.quantile_returns(sc, px, 3, 21)
    assert set(q["table"].columns) == {1, 2, 3} and q["n_dates"] > 20
    # the bucket averages are consistent with the spread
    assert q["spread"].iloc[0] == pytest.approx(q["table"][3].iloc[0] - q["table"][1].iloc[0])
    perfect = F.quantile_returns(F.forward_returns(px, 21, 1), px, 3, 21)
    assert perfect["monotonicity"] == pytest.approx(1.0) and perfect["spread_mean"] > 0


def test_factor_to_weights_shapes(px):
    sc = F.compute("mom_126", px)
    w = F.factor_to_weights(sc, 2)
    held = w[w.sum(axis=1) > 0]
    assert (held.sum(axis=1).round(9) == 1.0).all() and ((held > 0).sum(axis=1) <= 2).all()
    iv = F.factor_to_weights(sc, 2, weighting="inverse_vol", prices=px)
    assert (iv >= 0).all().all()
    floor = F.factor_to_weights(sc, 2, min_score=10.0)           # nothing qualifies
    assert floor.sum().sum() == 0


def test_zoo_factor_runs_through_engine_and_registry():
    from qbt.strategies import REGISTRY
    assert "zoo_factor" in REGISTRY
    ds = synthetic_dataset(edge=0.05, seed=3)
    res = run_strategy(ds, "zoo_factor", {"factor": "mom_12_1", "top_n": 3})
    assert len(res.returns) > 500 and np.isfinite(res.equity.iloc[-1])
