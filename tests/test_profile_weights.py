"""Investor profiles offered to the Backtest as weight text."""
import pytest

from qbt import allocation as ALLOC
from qbt import profile_weights as PW
from qbt.strategies.library import _fixed_weights
import pandas as pd


def test_every_preset_profile_is_offered():
    opts = PW.profiles()
    labels = [p.label for p in opts.values()]
    for name in ("100% Equities", "Balanced", "Income", "100% Bonds"):
        assert name in labels


def test_spec_round_trips_through_both_backtest_readers():
    p = PW.profiles()["Balanced (60% equity / 40% bonds)"]
    spec = PW.weights_spec(p)
    assert spec == "XIC.TO:0.18, VFV.TO:0.27, XEF.TO:0.12, XEC.TO:0.03, XBB.TO:0.4"
    # the Fixed Weights model reads fractions as they are (no rescaling)
    px = pd.DataFrame(1.0, index=pd.bdate_range("2024-01-01", periods=3), columns=list(p.weights))
    w = _fixed_weights(px, {"weights": spec}).iloc[-1]
    assert w.to_dict() == pytest.approx(p.weights) and w.sum() == pytest.approx(1.0)
    # the benchmark blend parser gives the same ratios
    b = ALLOC.parse_fixed(spec, list(p.weights))
    tot = sum(b.values())
    assert {k: v / tot for k, v in b.items()} == pytest.approx(p.weights)
    assert PW.weights_words(p).startswith("XIC.TO 18%, VFV.TO 27%")


def test_merge_tickers_keeps_existing_symbols():
    out = PW.merge_tickers("XIC.TO, ZEB.TO", ["XIC.TO", "XBB.TO"]).split("\n")
    assert out == ["XIC.TO", "ZEB.TO", "XBB.TO"]
