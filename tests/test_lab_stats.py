"""Statistical validation: known values, null behaviour, and planted effects."""
import math

import numpy as np
import pandas as pd
import pytest

from qbt.lab import stats as S


def _noise(n=1500, mu=0.0, sd=0.01, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n),
                     index=pd.bdate_range("2015-01-01", periods=n))


def test_normal_cdf_ppf_roundtrip():
    for p in (0.01, 0.2, 0.5, 0.9, 0.999):
        assert S.norm_cdf(S.norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_psr_matches_closed_form():
    r = _noise(1000, 0.0006, seed=1)
    x = r.to_numpy()
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std(ddof=0)
    g3, g4 = (z ** 3).mean(), (z ** 4).mean()
    expect = S.norm_cdf(sr / math.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (len(x) - 1)))
    assert S.probabilistic_sharpe(r)["psr"] == pytest.approx(expect, rel=1e-9)


def test_psr_pure_noise_is_not_confident():
    assert S.probabilistic_sharpe(_noise(1500, 0.0, seed=2))["psr"] < 0.95


def test_expected_max_sharpe_known_value_and_monotone():
    # Expected maximum of 100 standard normals is about 2.51
    assert S.expected_max_sharpe(100, 1.0) == pytest.approx(2.51, abs=0.1)
    assert S.expected_max_sharpe(10, 1) < S.expected_max_sharpe(100, 1) < S.expected_max_sharpe(1000, 1)
    assert S.expected_max_sharpe(1, 1.0) == 0.0


def test_dsr_equals_psr_with_one_trial_and_falls_with_trials():
    r = _noise(1200, 0.0007, seed=3)
    assert S.deflated_sharpe(r, 1)["dsr"] == pytest.approx(S.probabilistic_sharpe(r)["psr"], rel=1e-9)
    assert S.deflated_sharpe(r, 200)["dsr"] < S.deflated_sharpe(r, 5)["dsr"] < S.deflated_sharpe(r, 1)["dsr"]


def test_dsr_uses_cross_trial_dispersion_when_given():
    r = _noise(1200, 0.0007, seed=3)
    tight = S.deflated_sharpe(r, 50, trial_sharpes_pa=[0.5] * 25 + [0.52] * 25)
    wide = S.deflated_sharpe(r, 50, trial_sharpes_pa=np.linspace(-2, 3, 50))
    assert wide["expected_max_sharpe_pa"] > tight["expected_max_sharpe_pa"]


def test_min_track_record_length():
    strong = S.min_track_record_length(_noise(2000, 0.0012, seed=4))
    weak = S.min_track_record_length(_noise(2000, 0.0004, seed=4))
    assert strong["min_obs"] < weak["min_obs"]
    losing = S.min_track_record_length(_noise(2000, -0.001, seed=4))
    assert math.isnan(losing["min_obs"]) and not losing["enough"]


def test_bootstrap_interval_and_block_widens_on_autocorrelation():
    rng = np.random.default_rng(5)
    e = rng.normal(0, 0.01, 2000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.5 * x[i - 1] + e[i] + 0.0003
    r = pd.Series(x)
    iid = S.bootstrap_sharpe(r, 600, mean_block=1.0)
    blk = S.bootstrap_sharpe(r, 600, mean_block=15.0)
    assert iid["ci_low"] < iid["observed"] < iid["ci_high"]
    assert (blk["ci_high"] - blk["ci_low"]) > (iid["ci_high"] - iid["ci_low"])
    assert S.bootstrap_sharpe(r.iloc[:10])["observed"] != S.bootstrap_sharpe(r.iloc[:10])["observed"]  # NaN, error path


def test_bootstrap_is_reproducible():
    r = _noise(500, 0.0005, seed=6)
    a, b = S.bootstrap_sharpe(r, 200, seed=9), S.bootstrap_sharpe(r, 200, seed=9)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


def test_sign_flip_null_and_effect():
    assert S.sign_flip_test(_noise(1500, 0.0, seed=7))["p_value"] > 0.05
    assert S.sign_flip_test(_noise(1500, 0.002, seed=7))["p_value"] < 0.01


def test_random_timing_constant_book_is_all_beta():
    idx = pd.bdate_range("2018-01-01", periods=600)
    px = pd.DataFrame(100 * np.cumprod(1 + np.random.default_rng(1).normal(0.0004, 0.01, (600, 3)), axis=0),
                      index=idx, columns=list("ABC"))
    w = pd.DataFrame(1 / 3, index=idx, columns=px.columns)
    assert S.random_timing_test(w, px, 200)["p_value"] == pytest.approx(1.0)


def test_random_timing_detects_foresight():
    idx = pd.bdate_range("2018-01-01", periods=900)
    rets = pd.DataFrame(np.random.default_rng(2).normal(0.0, 0.01, (900, 2)), index=idx, columns=["A", "B"])
    px = 100 * (1 + rets).cumprod()
    # weights at close t chosen with knowledge of t+1's return: cheating, and obvious
    w = (rets.shift(-1) > 0).astype(float).fillna(0.0)
    assert S.random_timing_test(w, px, 300)["p_value"] < 0.02


def test_trade_order_conserves_total_and_bounds_percentile():
    pnls = np.random.default_rng(3).normal(50, 400, 120)
    out = S.trade_order_test(pnls, 100_000, 300)
    assert 0 <= out["dd_percentile"] <= 100 and out["n_trades"] == 120
    assert "error" in S.trade_order_test([1, 2, 3])


def test_walk_forward_windows():
    wf = S.walk_forward_windows(_noise(1000, 0.0005, seed=8), 5)
    assert len(wf) == 5 and wf["Start"].is_monotonic_increasing


def test_pbo_noise_is_about_half_and_real_edge_is_low():
    # One dataset's PBO is a noisy estimate (the splits share blocks); the
    # average over independent pure-noise datasets sits just above one half.
    vals = [S.pbo(pd.DataFrame(np.random.default_rng(s).normal(0, 0.01, (1200, 20))), 12)["pbo"]
            for s in range(15)]
    assert 0.4 < np.mean(vals) < 0.7
    noise = pd.DataFrame(np.random.default_rng(10).normal(0, 0.01, (1200, 20)))
    edge = noise.copy()
    edge[3] = edge[3] + 0.0015                       # one configuration with a genuine edge
    assert S.pbo(edge, 12)["pbo"] < 0.15
    assert "error" in S.pbo(noise.iloc[:, :2])


def test_purged_kfold_leaves_gap():
    for train, test in S.purged_kfold(1000, 5, purge=10, embargo=5):
        assert not set(train) & set(test)
        gap_lo = test.min() - 10
        gap_hi = test.max() + 5
        assert not any(gap_lo <= t <= gap_hi for t in train)


def test_cpcv_counts_and_isolation():
    splits = S.cpcv_splits(600, n_groups=6, k_test=2, purge=5)
    assert len(splits) == math.comb(6, 2)
    for tr, te, _ in splits:
        assert not set(tr) & set(te)
    out = S.cpcv_selection(pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (800, 6))), 8, 2)
    assert out["n_paths"] == math.comb(7, 1) and np.isfinite(out["median_oos_sharpe"])
