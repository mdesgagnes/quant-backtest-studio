"""Tests for qbt/groups.py (Markets Groups tab numbers)."""
import numpy as np
import pandas as pd
import pytest

from qbt import groups as G


@pytest.fixture
def prices():
    idx = pd.bdate_range("2021-01-04", "2026-11-09")
    rng = np.random.default_rng(11)
    out = {}
    for i, n in enumerate(["A", "B", "C", "D", "E", "F", "G"]):
        out[n] = 100 * np.cumprod(1 + rng.normal(0.0003 * (i + 1), 0.01, len(idx)))
    df = pd.DataFrame(out, index=idx)
    df.loc[:"2025-06-30", "G"] = np.nan          # late lister
    return df


def test_parse_groups_handles_noise():
    txt = "# comment\nGrowth: aapl, msft  nvda\n\nBad line\nValue: xom;jpm\nGrowth: amzn\nEmpty:\n"
    g = G.parse_groups(txt)
    assert g == {"Growth": ["AAPL", "MSFT", "NVDA", "AMZN"], "Value": ["XOM", "JPM"]}


def test_groups_from_presets_and_classes():
    names = G.PRESETS.names()
    gp = G.groups_from_presets(names[:2])
    assert list(gp) == names[:2] and all(gp.values())
    gc = G.groups_by_class(["Canadian asset classes"])
    assert set(gc) == {"Equities", "Fixed income", "Real assets"}
    assert "XBB.TO" in gc["Fixed income"]
    # counted once even when two presets tag it
    both = G.groups_by_class(["Canadian asset classes", "Canadian ETFs: broad"])
    flat = [t for v in both.values() for t in v]
    assert len(flat) == len(set(flat))


def test_group_index_equal_weights_daily(prices):
    idx = G.group_index(prices, ["A", "B"])
    r = prices[["A", "B"]].pct_change().mean(axis=1).dropna()
    assert idx.iloc[-1] == pytest.approx(100 * (1 + r).prod())
    assert G.group_index(prices, ["ZZZ"]).empty


def test_group_index_uses_members_present(prices):
    idx = G.group_index(prices, ["A", "G"])
    day = pd.Timestamp("2024-03-15")           # G not listed yet: A alone
    prev = idx.index[idx.index.get_loc(day) - 1]
    assert idx[day] / idx[prev] - 1 == pytest.approx(prices["A"].pct_change()[day])


def test_group_table_stat_and_counts(prices):
    groups = {"Low": ["A", "B", "C"], "High": ["D", "E", "F", "G"], "Ghost": ["ZZZ"]}
    med = G.group_table(prices, groups).set_index("Group")
    assert "Ghost" not in med.index
    assert med.loc["Low", "Members"] == 3 and bool(med.loc["Low", "Small"])
    one_y = G.group_table(prices[["A", "B", "C"]], {"Low": ["A", "B", "C"]}, "Max")
    grid = G.MON.performance_grid(prices[["A", "B", "C"]]).set_index("Instrument")
    assert one_y.iloc[0]["1Y"] == pytest.approx(grid["1Y"].max())
    assert med.loc["Low", "1Y"] == pytest.approx(grid["1Y"].median())
    # late lister has no 3Y: the statistic is over the members that do
    g3 = G.MON.performance_grid(prices[["D", "E", "F", "G"]]).set_index("Instrument")["3Y"]
    assert med.loc["High", "3Y"] == pytest.approx(g3.dropna().median())
    assert med.loc["High", "Best (1Y)"] in ["D", "E", "F", "G"]
    assert len(med.loc["Low", "Trend"]) > 10


def test_group_table_rejects_unknown_stat(prices):
    with pytest.raises(ValueError):
        G.group_table(prices, {"x": ["A"]}, "Mode")


def test_group_curves_common_window(prices):
    groups = {"Low": ["A", "B"], "High": ["D", "G"]}
    c = G.group_curves(prices, groups, ["Low", "High"], "1Y", "Cumulative")
    assert list(c.columns) == ["Low", "High"]
    assert (c.iloc[0] == 0).all()
    assert G.group_curves(prices, groups, ["nope"], "1Y").empty
