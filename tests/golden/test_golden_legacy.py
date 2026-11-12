"""Golden-output tests for the current Streamlit app (`app.py`).

Each scenario runs the real app on frozen input files and must reproduce the
recorded reference exactly: raw statistics, the simulation's value, holdings
and trades, every control, every screen (text, tables, what each chart
plots) and every export. See tests/golden/README.md.

    pytest tests/golden -q
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import harness as H                      # noqa: E402
from scenarios import SCENARIOS          # noqa: E402


@pytest.mark.golden
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_legacy_app_matches_recorded_output(name):
    folder = os.path.join(H.EXPECTED, name)
    expected = H.load(folder)
    assert expected, (f"No reference recorded for '{name}'. "
                      f"Run: python tests/golden/record.py {name}")
    with H.LegacyApp() as app:
        SCENARIOS[name](app)
        actual = H.record(app)
    diffs = H.compare(expected, actual)
    if diffs:
        H.save(actual, os.path.join(H.ACTUAL, name))
        pytest.fail(f"{name}: {len(diffs)} difference(s) from the recorded "
                    f"output (actual saved to tests/golden/_actual/{name}):\n  "
                    + "\n  ".join(diffs[:40]), pytrace=False)


def test_every_scenario_has_a_reference():
    missing = [n for n in SCENARIOS
               if not os.path.exists(os.path.join(H.EXPECTED, n, "screens.json"))]
    assert not missing, f"Scenarios without a recorded reference: {missing}"
