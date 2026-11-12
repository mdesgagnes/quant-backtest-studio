"""Golden-output tests for the redesigned app's service layer (`server/`).

The same reference files as test_golden_legacy.py, reached without
Streamlit: `server.pipeline.execute` must reproduce the recorded statistics
and simulation output, and each rebuilt screen must reproduce the recorded
content of the Streamlit tab it replaces.

    pytest tests/golden/test_golden_service.py -q      # about 20 seconds

Adding a screen: write `server/views/<tab>.py`, add its name to SCREENS
below with the name of the recorded tab, and this test checks it on all four
backtest scenarios.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import harness as H                                              # noqa: E402
from server.pipeline import BLEND, RunFiles, RunSpec, execute    # noqa: E402
from server.views import results                                 # noqa: E402
from server.views.common import RunView                          # noqa: E402

# screen name -> (recorded tab, builder)
SCREENS = {
    "results": ("tab: Results", results.build),
}


def _f(name: str):
    return (name, H.fixture(name))


PRICES = dict(source="upload", tickers=[], start=None)

SPECS = {
    "momentum_file": lambda: (
        RunSpec(**PRICES),
        RunFiles(prices=[_f("prices_long.csv")])),
    "imported_weights": lambda: (
        RunSpec(**PRICES, mode="external_weights"),
        RunFiles(prices=[_f("prices_long.csv")], weights=_f("target_weights.csv"))),
    "factor_signal": lambda: (
        RunSpec(**PRICES, strategy="factor_rank", exog_lag=45,
                params={"direction": "Low value = favorable", "top_n": 2},
                execute_at_open=True, impact_model="sqrt_vol",
                cash_rate_pa=0.02, management_fee_pa=0.005),
        RunFiles(prices=[_f("prices_long.csv")], exog=[_f("fundamental_factor.csv")])),
    "macro_blend": lambda: (
        RunSpec(**PRICES, strategy="macro_gate", exog_lag=30,
                params={"series": "ISM_MANUFACTURING"}, rebalance="Q",
                benchmark=BLEND, bench_blend="XIC.TO:60, ZLB.TO:40",
                compare=[{"key": "accelerating_momentum"}]),
        RunFiles(prices=[_f("prices_long.csv")], exog=[_f("economic_data.csv")])),
}

_RUNS: Dict[str, Dict[str, Any]] = {}


def _run(name: str) -> Dict[str, Any]:
    if name not in _RUNS:
        spec, files = SPECS[name]()
        _RUNS[name] = execute(spec, files)
    return _RUNS[name]


def blocks_record(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Screen blocks in the form the legacy transcript was recorded in."""
    out: List[Dict[str, Any]] = []
    for b in blocks:
        k = b["kind"]
        if k in ("note", "flag"):
            out.append({"kind": k, "text": H.plain(b["html"])})
        elif k in ("heading", "caption"):
            out.append({"kind": k, "text": H.plain(b["text"])})
        elif k == "kpi":
            out.append({"kind": "kpi", "label": b["label"], "value": b["value"],
                        "sub": b["sub"], "tone": b["tone"]})
        elif k == "table":
            out.append({"kind": "table", **H.table_record(b["frame"], index=True)})
        elif k == "chart":
            if b.get("log_toggle"):
                out.append({"kind": "toggle", "label": "Log scale",
                            "value": json.dumps(bool(b["log"]))})
            out.append({"kind": "chart",
                        **H.chart_record(results.legacy_figure(b).to_json())})
        elif k == "panel":
            out.append({"kind": "<expander>", "label": b["label"]})
            out += blocks_record(b["blocks"])
            out.append({"kind": "</expander>"})
        else:
            raise AssertionError(f"no record form for block kind {k!r}")
    return out


def _fail(name: str, what: str, diffs: List[str]) -> None:
    pytest.fail(f"{name} / {what}: {len(diffs)} difference(s) from the recorded "
                f"output:\n  " + "\n  ".join(diffs[:40]), pytrace=False)


@pytest.mark.golden
@pytest.mark.parametrize("name", list(SPECS))
def test_pipeline_matches_recorded_run(name):
    expected = H.load(os.path.join(H.EXPECTED, name))
    run = _run(name)
    actual = {"metrics": H.backtest_metrics(run), "engine": H.engine_record(run)}
    diffs = H.compare(expected, actual, parts=("metrics", "engine"))
    if diffs:
        _fail(name, "pipeline", diffs)


@pytest.mark.golden
@pytest.mark.parametrize("name", list(SPECS))
def test_context_strip_matches(name):
    expected = H.load(os.path.join(H.EXPECTED, name))["screens"]["page"]
    strip = next(b for b in expected if b["kind"] == "context")
    assert RunView(_run(name)).context() == strip["items"]


@pytest.mark.golden
@pytest.mark.parametrize("screen", list(SCREENS))
@pytest.mark.parametrize("name", list(SPECS))
def test_screen_matches_recorded_tab(name, screen):
    tab, build = SCREENS[screen]
    expected = H.load(os.path.join(H.EXPECTED, name))["screens"][tab]
    actual = json.loads(json.dumps(blocks_record(build(RunView(_run(name)))), default=str))
    diffs: List[str] = []
    H._diff(expected, actual, tab, diffs, numeric=False)
    if diffs:
        _fail(name, screen, diffs)


@pytest.mark.golden
@pytest.mark.parametrize("name", list(SPECS))
def test_chart_data_is_what_the_legacy_chart_plots(name):
    """The series sent to the browser are the series the Streamlit chart
    drew: same count, same sum."""
    import numpy as np
    for b in results.build(RunView(_run(name))):
        if b["kind"] != "chart" or "series" not in b["data"]:
            continue
        fig = json.loads(results.legacy_figure(b).to_json())
        for sent, drawn in zip(b["data"]["series"], fig["data"]):
            y = H._array(drawn["y"])
            mine = np.asarray(sent["y"], dtype="float64")
            assert len(mine) == len(y), (b["title"], sent["name"])
            assert np.isclose(np.nansum(mine), np.nansum(y), rtol=1e-12), (b["title"], sent["name"])
