"""Tests for qbt/snapshot.py (Markets Snapshot and Screener numbers)."""
import numpy as np
import pandas as pd
import pytest

from qbt import snapshot as S


@pytest.fixture
def prices():
    idx = pd.bdate_range("2021-01-04", "2026-11-09")
    rng = np.random.default_rng(7)
    out = {}
    for i, n in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        out[n] = 100 * np.cumprod(1 + rng.normal(0.0004 * (i + 1), 0.01, len(idx)))
    df = pd.DataFrame(out, index=idx)
    # A late lister: no history before mid-2025.
    df.loc[:"2025-06-30", "FFF"] = np.nan
    return df


def test_window_ytd_starts_at_prior_year_end(prices):
    w = S.slice_window(prices[["AAA"]], "YTD")
    assert w.index[0] == pd.Timestamp("2025-12-31")
    assert w.index[-1] == prices.index[-1]


def test_window_blanks_short_history(prices):
    w = S.slice_window(prices, "3Y")
    assert "FFF" not in w.columns          # lists 2025: no 3Y window
    assert "AAA" in w.columns


def test_window_max_keeps_everything(prices):
    w = S.slice_window(prices, "Max")
    assert w["FFF"].dropna().index[0] == pd.Timestamp("2025-07-01")


def test_explicit_dates_override_window(prices):
    w = S.slice_window(prices, "1Y", start="2024-01-02", end="2024-06-28")
    assert w.index[0] <= pd.Timestamp("2024-01-02")
    assert w.index[-1] == pd.Timestamp("2024-06-28")


def test_stats_match_hand_calculation(prices):
    w = S.slice_window(prices[["AAA"]], "1Y")
    st = S.window_stats(w).iloc[0]
    s = w["AAA"]
    assert st["Cumulative Return"] == pytest.approx(s.iloc[-1] / s.iloc[0] - 1)
    # A one-year window: annualized equals cumulative to within the day count.
    assert st["Ann. Return"] == pytest.approx(st["Cumulative Return"], abs=0.01)
    assert st["Max DD"] <= 0


def test_no_annualizing_under_a_year(prices):
    st = S.window_stats(S.slice_window(prices[["AAA"]], "3M")).iloc[0]
    assert np.isnan(st["Ann. Return"])
    assert not np.isnan(st["Cumulative Return"])


def test_percentile_needs_enough_peers():
    assert S.percentile_rank(pd.Series([1.0, 2.0, 3.0])).isna().all()
    p = S.percentile_rank(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert p.iloc[-1] == 100 and p.iloc[0] == 20


def test_percentile_lower_is_better():
    p = S.percentile_rank(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
                          higher_is_better=False)
    assert p.iloc[0] == 100


def test_curve_modes(prices):
    w = S.slice_window(prices[["AAA"]], "1Y")
    cum = S.rebased_curves(w, "Cumulative")
    assert cum["AAA"].iloc[0] == 0
    dd = S.rebased_curves(w, "Drawdown")
    assert dd["AAA"].max() <= 0
    roll = S.rebased_curves(prices[["AAA"]], "Rolling", 60)
    assert len(roll) == len(prices) - 60
    with pytest.raises(ValueError):
        S.rebased_curves(w, "nope")


def test_annual_returns_flags_partial_years(prices):
    tbl, part = S.annual_returns(prices)
    assert tbl.loc["AAA", 2022] == pytest.approx(
        prices["AAA"].loc["2022-12-30"] / prices["AAA"].loc["2021-12-31"] - 1)
    assert part.loc["AAA", 2026]            # current year in progress
    assert not part.loc["AAA", 2022]
    assert part.loc["FFF", 2025]            # listed mid-year
    assert pd.isna(tbl.loc["FFF", 2022])


def test_monthly_grid_sums_to_annual(prices):
    g = S.monthly_grid(prices["AAA"])
    yr = (1 + g.loc[2023]).prod() - 1
    s = prices["AAA"]
    assert yr == pytest.approx(s.loc["2023-12-29"] / s.loc["2022-12-30"] - 1)


def test_trend_points_rebased_and_capped(prices):
    pts = S.trend_points(prices["AAA"], 40)
    assert len(pts) == 40 and pts[0] == pytest.approx(100.0)
    assert S.trend_points(pd.Series([1.0])) == []


def test_screener_table(prices):
    t = S.screener_table(prices)
    assert set(t["Instrument"]) == set(prices.columns)
    row = t.set_index("Instrument").loc["AAA"]
    assert len(row["Trend"]) > 10
    assert row["From 52w High"] <= 0
    best = t.loc[t["1Y"].idxmax(), "1Y %ile"]
    assert best == 100
    med = S.summary_row(t, ["1Y", "Volatility"])
    assert med["1Y"] == pytest.approx(t["1Y"].median())
