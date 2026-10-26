"""Ledger, run cards, decay monitor, x-ray, event study, validation suite."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from qbt.config import CostConfig, EngineConfig
from qbt.lab import decay, eventstudy, ledger as L, runcard, xray
from qbt.lab.run import run_strategy, synthetic_dataset
from qbt.lab.validate import validate_run


@pytest.fixture(autouse=True)
def lab_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QBT_LAB_DIR", str(tmp_path / "lab"))
    return tmp_path


@pytest.fixture(scope="module")
def ds():
    return synthetic_dataset(edge=0.05, seed=3)


# ----------------------------------------------------------------- ledger
def test_ledger_counts_trials_and_dedupes():
    hid = L.add_hypothesis("Momentum")
    assert L.get_hypothesis(hid)["status"] == "idea"
    cfg = {"s": "x", "p": {"a": 1}}
    L.log_experiment(hid, "backtest", "a", cfg)
    L.log_experiment(hid, "backtest", "a again", cfg)               # identical config: not a new trial
    L.log_experiment(hid, "backtest", "b", {"s": "x", "p": {"a": 2}})
    L.log_experiment(hid, "sweep", "grid", {"grid": 1}, n_variants=40)
    assert L.trial_count(hid) == 42
    assert L.get_hypothesis(hid)["status"] == "testing"             # first experiment moves it on
    assert L.trial_count(None) == 1 and L.trial_count(9999) == 1
    row = L.list_hypotheses()[0]
    assert row["n_experiments"] == 4


def test_ledger_validation_and_json_safety():
    with pytest.raises(ValueError):
        L.add_hypothesis("  ")
    hid = L.add_hypothesis("x")
    with pytest.raises(ValueError):
        L.update_hypothesis(hid, status="bogus")
    L.log_experiment(hid, "k", "n", {}, metrics={"Sharpe": float("nan"), "n": np.int64(3)})
    m = L.list_experiments(hid)[0]["metrics"]
    assert m["Sharpe"] is None and m["n"] == 3
    assert json.loads(L.export_json())["hypotheses"]


def test_strategy_registry_rejects_duplicates():
    spec = {"strategy_key": "sma_trend", "params": {}, "tickers": ["A"], "registered": "2024-01-01"}
    L.register_strategy("s1", spec, {"sharpe": 1.0})
    with pytest.raises(ValueError):
        L.register_strategy("s1", spec, {"sharpe": 1.0})
    L.retire_strategy("s1")
    assert L.list_strategies(active_only=True) == []


# ---------------------------------------------------------------- runcard
def _run(ds):
    eng, cst = EngineConfig(), CostConfig()
    res = run_strategy(ds, "zoo_factor", {"factor": "mom_12_1", "top_n": 3}, eng, cst)
    return res, eng, cst


def test_run_card_verifies_detects_tampering_and_reproduces(ds, tmp_path):
    res, eng, cst = _run(ds)
    v = validate_run(ds, res, "zoo_factor", {"factor": "mom_12_1", "top_n": 3}, eng, cst, n_boot=100, n_timing=100)
    d = str(tmp_path / "card")
    card = runcard.write_run_card(d, ds, res, "zoo_factor", {"factor": "mom_12_1", "top_n": 3}, eng, cst,
                                  "t", v, 1)
    assert os.path.isfile(os.path.join(d, "run_card.md"))
    assert card["warnings"], "a run card with no warnings would be a claim, not a record"
    assert any("hindsight" in w or "survivorship" in w for w in card["warnings"]) or ds.source != "yfinance"
    assert runcard.verify_run_card(d)["ok"]
    assert runcard.reproduce_run_card(d)["ok"]
    # tamper with a stored artifact
    with open(os.path.join(d, "artifacts", "returns.csv"), "a") as fh:
        fh.write("2030-01-01,0.5\n")
    bad = runcard.verify_run_card(d)
    assert not bad["ok"] and any("returns.csv" in p for p in bad["problems"])


def test_run_card_catches_edited_metric(ds, tmp_path):
    res, eng, cst = _run(ds)
    d = str(tmp_path / "card2")
    runcard.write_run_card(d, ds, res, "zoo_factor", {"factor": "mom_12_1", "top_n": 3}, eng, cst)
    p = os.path.join(d, "run_card.json")
    card = json.load(open(p))
    card["metrics"]["Sharpe"] = 9.99
    json.dump(card, open(p, "w"))
    assert any("Sharpe" in x for x in runcard.verify_run_card(d)["problems"])


def test_dataset_hash_is_stable_and_sensitive(ds):
    h = ds.content_hash()
    assert h == synthetic_dataset(edge=0.05, seed=3).content_hash()
    assert h != synthetic_dataset(edge=0.05, seed=4).content_hash()


def test_warning_for_lookahead_execution(ds, tmp_path):
    eng = EngineConfig(execution_lag=0)
    res = run_strategy(ds, "zoo_factor", {"factor": "mom_12_1", "top_n": 3}, eng, CostConfig())
    card = runcard.write_run_card(str(tmp_path / "c"), ds, res, "zoo_factor", {"factor": "mom_12_1"}, eng, CostConfig())
    assert any("look-ahead" in w for w in card["warnings"])


# ------------------------------------------------------------------ decay
def _stream(mu, n, seed, start="2020-01-01", sd=0.01):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=pd.bdate_range(start, periods=n))


def test_decay_statuses():
    base = {"sharpe": 1.0, "vol": 0.16}
    reg = "2021-12-31"
    fresh = pd.concat([_stream(0.0006, 500, 1), _stream(0.0006, 20, 2, "2022-01-03")])
    assert decay.assess(fresh, reg, base)["status"] == "fresh"
    good = pd.concat([_stream(0.0006, 500, 1), _stream(0.0006, 300, 2, "2022-01-03")])
    assert decay.assess(good, reg, base)["status"] in ("healthy", "aging")
    dead = pd.concat([_stream(0.0006, 500, 1), _stream(-0.002, 300, 3, "2022-01-03")])
    out = decay.assess(dead, reg, base)
    assert out["status"] == "stale" and out["cusum_breached"] and out["z_vs_baseline"] < -2


def test_cusum_resets_on_recovery():
    c = decay.cusum_down(np.array([-1.0] * 5 + [3.0] * 5))
    assert c["current"] == 0.0 and not c["breached"]
    assert decay.cusum_down(np.array([-2.0] * 25))["breached"]


# ------------------------------------------------------------------- xray
def test_xray_equal_weights_known_values():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=500)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (500, 4)), axis=0), index=idx, columns=list("ABCD"))
    out = xray.risk_xray(px, {c: 0.25 for c in "ABCD"})
    assert out["concentration"]["hhi"] == pytest.approx(0.25)
    assert out["concentration"]["effective_n"] == pytest.approx(4.0)
    assert out["table"]["Risk contribution"].sum() == pytest.approx(1.0)
    assert out["diversification"]["effective_bets"] > 3.0                 # near-independent assets
    assert out["diversification"]["ratio"] > 1.5


def test_xray_concentrated_correlated_and_errors():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=500)
    common = rng.normal(0, 0.01, 500)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + common), "B": 100 * np.cumprod(1 + common + rng.normal(0, 0.001, 500)),
                       "C": 100 * np.cumprod(1 + rng.normal(0, 0.01, 500))}, index=idx)
    out = xray.risk_xray(px, {"A": 0.4, "B": 0.4, "C": 0.2})
    assert out["diversification"]["effective_bets"] < 2.2                 # A and B are one bet
    assert out["table"].loc[["A", "B"], "Risk contribution"].sum() > 0.7
    assert out["cash"] if False else out["inputs"]["cash"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        xray.risk_xray(px, {"ZZZ": 1.0})
    with pytest.raises(ValueError):
        xray.risk_xray(px, {})
    short = xray.risk_xray(px, {"A": 0.6, "C": -0.4})
    assert short["inputs"]["net"] == pytest.approx(0.2)


# ------------------------------------------------------------ event study
def test_event_study_recovers_planted_abnormal_return():
    rng = np.random.default_rng(2)
    n = 1500
    idx = pd.bdate_range("2015-01-01", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    r = 1.2 * m + rng.normal(0, 0.008, n)
    events = []
    for pos in range(300, 1300, 100):
        r[pos] += 0.03
        events.append(("X", str(idx[pos].date())))
    px = pd.DataFrame({"X": 100 * np.cumprod(1 + r)}, index=idx)
    mk = pd.Series(100 * np.cumprod(1 + m), index=idx)
    out = eventstudy.event_study(px, mk, events, pre=5, post=10)
    assert out["n_events"] == 10
    assert out["mean_car"] == pytest.approx(0.03, abs=0.02) and out["t_car"] > 2
    assert out["average"].loc[out["average"]["Relative day"] == 0, "Mean AR"].iloc[0] == pytest.approx(0.03, abs=0.01)
    null = eventstudy.event_study(px, mk, [("X", str(idx[p].date())) for p in (350, 450, 550, 750)], 5, 10)
    assert abs(null["mean_car"]) < 0.03


def test_event_study_skips_bad_events_and_parses():
    idx = pd.bdate_range("2020-01-01", periods=400)
    px = pd.DataFrame({"X": np.linspace(100, 120, 400)}, index=idx)
    out = eventstudy.event_study(px, px["X"], [("NOPE", "2020-06-01"), ("X", "2020-01-10"), ("X", "2030-01-01")])
    assert out["n_events"] == 0 and len(out["skipped"]) == 3
    assert eventstudy.parse_events("aapl 2024-05-02\nbad\nMSFT, 2023-01-05") == [("AAPL", "2024-05-02"), ("MSFT", "2023-01-05")]


# --------------------------------------------------------------- validate
def test_validation_battery_structure_and_overfit_checks(ds):
    res, eng, cst = _run(ds)
    v = validate_run(ds, res, "zoo_factor", {"factor": "mom_12_1", "top_n": 3}, eng, cst, n_trials=30,
                     grid={"top_n": [1, 2, 3, 4]}, n_boot=100, n_timing=100)
    keys = {c.key for c in v["checks"]}
    assert {"psr", "dsr", "bootstrap", "timing", "walk_forward", "costs", "pbo", "cpcv"} <= keys
    assert v["grade"]["label"] in ("Robust", "Promising", "Fragile", "Unsupported")
    assert v["n_trials"] == 30
    assert all(c.status in ("pass", "warn", "fail", "na") for c in v["checks"])


def test_validation_rejects_noise_strategy():
    noise = synthetic_dataset(edge=0.0, seed=21, n_assets=6, years=6)
    eng, cst = EngineConfig(), CostConfig()
    res = run_strategy(noise, "zoo_factor", {"factor": "rsi_14", "top_n": 2}, eng, cst)
    v = validate_run(noise, res, "zoo_factor", {"factor": "rsi_14", "top_n": 2}, eng, cst, n_trials=60,
                     n_boot=100, n_timing=100)
    assert v["grade"]["label"] not in ("Robust",)
    dsr = next(c for c in v["checks"] if c.key == "dsr")
    assert dsr.status in ("warn", "fail")
