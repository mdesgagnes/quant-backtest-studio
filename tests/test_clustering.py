"""Return-stream clustering (qbt/clustering.py)."""
from __future__ import annotations

import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from qbt import clustering as CL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _weekly(n=260, start="2015-01-02"):
    return pd.date_range(start, periods=n, freq="W-FRI")


def _blocks(n=260, per_block=3, blocks=2, rho_in=0.9, seed=0):
    """Blocks of streams correlated rho_in inside, independent between."""
    rng = np.random.default_rng(seed)
    cols = {}
    for b in range(blocks):
        f = rng.standard_normal(n)
        for i in range(per_block):
            e = rng.standard_normal(n)
            cols[f"B{b}S{i}"] = 0.02 * (np.sqrt(rho_in) * f + np.sqrt(1 - rho_in) * e)
    return pd.DataFrame(cols, index=_weekly(n))


def _cfg(**kw):
    return CL.ClusteringConfig(**kw)


# ---------------------------------------------------------------- config
def test_yaml_file_matches_defaults():
    with open(os.path.join(ROOT, "configs", "clustering_default.yaml")) as fh:
        cfg = CL.config_from_yaml(fh.read())
    assert asdict(cfg) == asdict(CL.ClusteringConfig())


def test_yaml_round_trip_and_validation():
    cfg = _cfg(threshold=0.7, linkage="ward", cut="clusters", n_clusters=3)
    cfg.stress.percentile = 0.1
    back = CL.config_from_yaml(CL.config_to_yaml(cfg))
    assert asdict(back) == asdict(cfg)
    with pytest.raises(CL.ClusteringError):
        CL.config_from_yaml("clustering:\n  linkage: single\n")
    with pytest.raises(CL.ClusteringError):
        CL.config_from_yaml("clustering:\n  stress: {percentile: 1.5}\n")


# ---------------------------------------------------------------- distance
def test_distance_formula():
    corr = pd.DataFrame([[1.0, 0.0, -1.0], [0.0, 1.0, 0.5], [-1.0, 0.5, 1.0]],
                        index=list("abc"), columns=list("abc"))
    d = CL.distance(corr)
    assert d.loc["a", "a"] == 0.0
    assert d.loc["a", "b"] == pytest.approx(np.sqrt(0.5))
    assert d.loc["a", "c"] == pytest.approx(1.0)
    assert d.loc["b", "c"] == pytest.approx(0.5)
    assert CL.distance_to_rho(CL.rho_to_distance(0.8)) == pytest.approx(0.8)


# ---------------------------------------------------------------- edge cases
def test_two_streams():
    df = _blocks(per_block=1, blocks=2)
    d = CL.analyse(df)
    assert d.n_streams == 2
    assert d.clusters.n_clusters == 2           # independent: two bets
    assert np.isnan(d.clusters.silhouette)      # undefined with singletons only
    assert 1.5 < d.effective_bets <= 2.0


def test_stream_below_minimum_is_dropped_with_reason():
    df = _blocks()
    df["SHORT"] = np.nan
    df.iloc[-20:, df.columns.get_loc("SHORT")] = 0.01 * np.arange(20)
    al = CL.align(df, _cfg())
    assert "SHORT" not in al.returns.columns
    reason = al.dropped.set_index("Stream").at["SHORT", "Reason"]
    assert "only 20 weekly returns" in reason and "52" in reason


def test_identical_streams():
    df = _blocks()
    df["COPY"] = df["B0S0"]
    d = CL.analyse(df)
    assert d.corr.loc["B0S0", "COPY"] == pytest.approx(1.0)
    assert CL.distance(d.corr).loc["B0S0", "COPY"] == pytest.approx(0.0)
    assert d.clusters.labels["COPY"] == d.clusters.labels["B0S0"]
    top = d.redundant.iloc[0]
    assert {top["Stream"], top["Same bet as"]} == {"B0S0", "COPY"}
    two = pd.DataFrame({"a": df["B0S0"], "b": df["B0S0"]})
    assert CL.effective_bets(CL.correlation(two), "entropy") == pytest.approx(1.0)
    assert CL.effective_bets(CL.correlation(two), "participation") == pytest.approx(1.0)


def test_zero_variance_dropped():
    df = _blocks()
    df["FLAT"] = 0.0
    d = CL.analyse(df)
    assert "FLAT" not in d.corr.columns
    assert "zero variance" in d.alignment.dropped.set_index("Stream").at["FLAT", "Reason"]
    assert not d.corr.isna().any().any()


def test_nan_gaps_removed_and_counted():
    df = _blocks()
    df.iloc[[50, 51, 120], 0] = np.nan
    al = CL.align(df, _cfg())
    assert al.gap_rows == 3
    assert len(al.returns) == len(df) - 3
    assert not al.returns.isna().any().any()


def test_mixed_start_dates_use_shared_window():
    df = _blocks(n=300)
    df.iloc[:100, 1] = np.nan                     # starts 100 weeks late
    df.iloc[-30:, 2] = np.nan                     # stops 30 weeks early
    al = CL.align(df, _cfg())
    assert al.returns.index[0] == df.index[100]
    assert al.returns.index[-1] == df.index[-31]
    # every stream with more history than the window is reported
    assert set(al.truncated["Stream"]) == set(df.columns)


def test_late_starter_dropped_when_window_too_short():
    df = _blocks(n=260)
    df.iloc[:200, 0] = np.nan                     # 60 weeks: enough alone
    df.iloc[231:, 1] = np.nan                     # ends early: shared window only 31
    al = CL.align(df, _cfg())
    assert list(al.dropped["Stream"]) == [df.columns[0]]
    assert "only 31 shared weekly returns" in al.dropped["Reason"].iloc[0]
    assert len(al.returns) == 231


def test_fewer_than_two_usable_streams_raises():
    df = _blocks(per_block=1, blocks=2)
    df["B1S0"] = np.nan
    with pytest.raises(CL.ClusteringError):
        CL.analyse(df)


# ---------------------------------------------------------------- clustering
def test_threshold_cut_finds_blocks():
    d = CL.analyse(_blocks(), _cfg(threshold=0.8))
    lab = d.clusters.labels
    assert d.clusters.n_clusters == 2
    assert lab["B0S0"] == lab["B0S1"] == lab["B0S2"]
    assert lab["B1S0"] == lab["B1S1"] == lab["B1S2"]
    assert lab["B0S0"] != lab["B1S0"]
    assert d.within > 0.8 and abs(d.between) < 0.2
    assert d.clusters.silhouette > 0.5
    assert set(d.table.columns) >= {"Stream", "Cluster", "Closest to", "Correlation",
                                    "Volatility", "Max drawdown"}
    # numbered left to right in dendrogram order
    assert list(d.table["Cluster"]) == sorted(d.table["Cluster"])


def test_fixed_cluster_count_and_ward():
    for linkage in ("average", "ward"):
        d = CL.analyse(_blocks(blocks=3), _cfg(cut="clusters", n_clusters=3, linkage=linkage))
        assert d.clusters.n_clusters == 3
        # the drawn cut line separates exactly three groups
        h = d.clusters.cut_height
        Z = d.clusters.linkage
        assert (Z[:, 2] > h).sum() == 2


def test_spearman_option():
    df = _blocks()
    d = CL.analyse(df, _cfg(correlation="spearman"))
    assert d.corr.loc["B0S0", "B0S1"] == pytest.approx(
        df["B0S0"].corr(df["B0S1"], method="spearman"))


def test_effective_bets_uncorrelated():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.standard_normal((5000, 5)), columns=list("abcde"))
    for m in ("entropy", "participation"):
        assert CL.effective_bets(CL.correlation(df), m) == pytest.approx(5.0, rel=0.02)


# ---------------------------------------------------------------- stress
def test_down_market_mask_takes_worst_share():
    b = pd.Series(np.arange(100, dtype=float) - 50, index=_weekly(100))
    m = CL.down_market_mask(b, 0.2)
    assert m.sum() == 20
    assert b[m].max() < b[~m].min()


def test_stress_matrix_only_uses_down_periods():
    df = _blocks()
    bench = df.mean(axis=1)
    # Make B1S0 move with B0S0 only when the benchmark is down.
    mask = CL.down_market_mask(bench, 0.2)
    df.loc[mask, "B1S0"] = df.loc[mask, "B0S0"]
    d = CL.analyse(df, _cfg(), bench=bench, bench_label="Mkt")
    assert d.down_label == "Mkt"
    assert d.down_n == mask.sum()
    assert d.down_corr.loc["B0S0", "B1S0"] == pytest.approx(1.0)
    assert d.corr.loc["B0S0", "B1S0"] < 0.6


def test_stress_ignores_benchmark_outside_window():
    df = _blocks()
    bench = df.mean(axis=1)
    early = CL.analyse(df, _cfg(), bench=bench, bench_label="Mkt",
                       end=df.index[150])
    assert early.alignment.returns.index[-1] <= df.index[150]
    assert early.down_n == int(np.ceil(0.2 * len(early.alignment.returns)))


def test_equal_weight_benchmark_by_default():
    d = CL.analyse(_blocks())
    assert d.down_label.startswith("equal-weight")


# ---------------------------------------------------------------- candidate
def test_candidate_duplicate_and_new():
    df = _blocks()
    holdings = ["B0S0", "B0S1", "B1S0"]
    dup = CL.candidate_test(df, holdings, "B0S2")
    assert dup.verdict == "duplicates"
    assert dup.nearest in ("B0S0", "B0S1")
    assert "duplicates an existing bet" in dup.sentence
    assert dup.bets_after - dup.bets_before < 0.3

    rng = np.random.default_rng(9)
    df["NEW"] = 0.02 * rng.standard_normal(len(df))
    new = CL.candidate_test(df, holdings, "NEW")
    assert new.verdict == "new"
    assert new.bets_after - new.bets_before > 0.7
    assert new.sentence.startswith("NEW adds a new bet")
    # the change is put against what a fully independent stream would add
    assert new.bets_after <= new.bets_if_independent + 0.05
    assert dup.bets_after < dup.bets_if_independent
    assert "fully independent stream would take them to" in dup.sentence


def test_candidate_without_enough_history():
    df = _blocks()
    df["YOUNG"] = np.nan
    df.iloc[-10:, df.columns.get_loc("YOUNG")] = 0.01
    v = CL.candidate_test(df, ["B0S0", "B1S0"], "YOUNG")
    assert v.verdict == "untestable"
    assert "cannot be tested" in v.sentence


# ---------------------------------------------------------------- frequency
def test_daily_compounded_to_weekly_on_own_dates():
    idx = pd.bdate_range("2018-01-01", periods=800)
    rng = np.random.default_rng(1)
    df = pd.DataFrame(0.01 * rng.standard_normal((800, 3)), index=idx, columns=list("abc"))
    df.iloc[5, 0] = np.nan                       # a holiday on one exchange only
    al = CL.align(df, _cfg())
    assert al.frequency == "weekly" and al.ppy == 52
    wk = al.returns.index[1]
    raw = df.loc[(df.index > al.returns.index[0]) & (df.index <= wk), "b"]
    assert al.returns.at[wk, "b"] == pytest.approx((1 + raw).prod() - 1)


def test_monthly_data_is_not_made_weekly():
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    df = _blocks(n=120).set_axis(idx)
    al = CL.align(df, _cfg())
    assert al.frequency == "monthly"
    assert any("used as is" in n for n in al.notes)


# ---------------------------------------------------------------- export, speed
def test_export_sections():
    df = _blocks()
    df["FLAT"] = 0.0
    secs = CL.export_sections(CL.analyse(df))
    titles = [s[1] for s in secs]
    assert titles[:4] == ["Diversification Summary", "Cluster Assignment",
                          "Redundant Pairs", "Correlation, Cluster Order"]
    assert "Streams Left Out" in titles
    assert all(s[0] == "Diversification" for s in secs)


def test_hundred_streams_fast():
    df = _blocks(n=520, per_block=10, blocks=10)
    t = time.perf_counter()
    d = CL.analyse(df)
    assert time.perf_counter() - t < 3.0
    assert d.n_streams == 100
    assert d.clusters.n_clusters == 10


def test_workbook_sheet_is_opt_in():
    """The sheet appears only when asked for, and asking for it adds no row
    to the About index of an export that does not ask."""
    import io
    import openpyxl
    from qbt import excel_export as XL

    df = _blocks()
    r = df["B0S0"]
    eq = (1 + r).cumprod() * 100
    stats = {"CAGR": 0.05}
    base = dict(all_series=df, include=list(XL.RETURNS_MODULES), style=False)

    def book(**kw):
        b = XL.workbook_from_returns(r, eq, stats, None, None, None, 52, "B0S0",
                                     **{**base, **kw})
        return openpyxl.load_workbook(io.BytesIO(b))

    plain = book()
    assert "Diversification" not in plain.sheetnames
    about = [c.value for row in plain["About"].iter_rows() for c in row]
    assert "Diversification" not in about

    secs = CL.export_sections(CL.analyse(df))
    wb = book(sections=secs, include=list(XL.RETURNS_MODULES) + ["Diversification"])
    assert "Diversification" in wb.sheetnames
    cells = [c.value for row in wb["Diversification"].iter_rows() for c in row]
    assert "Cluster Assignment" in cells and "Effective number of bets" in cells
    assert set(plain.sheetnames) == set(wb.sheetnames) - {"Diversification"}
