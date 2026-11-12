"""Golden-output tests for the redesigned app.

The same five scenarios and the same reference files as
test_golden_legacy.py, driven through the new server's rendering instead of
Streamlit. Every control, every screen, every export and the simulation's
own output must equal what the classic app produced.

    pytest tests/golden/test_golden_app.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import harness as H                      # noqa: E402
import shim_driver as D                  # noqa: E402
from scenarios import SCENARIOS          # noqa: E402


@pytest.mark.golden
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_new_app_matches_recorded_output(name):
    expected = H.load(os.path.join(H.EXPECTED, name))
    with D.ShimApp() as app:
        SCENARIOS[name](app)
        actual = D.record(app)
    diffs = H.compare(expected, actual)
    if diffs:
        H.save(actual, os.path.join(H.ACTUAL, "app_" + name))
        pytest.fail(f"{name}: {len(diffs)} difference(s) from the recorded output "
                    f"(actual saved to tests/golden/_actual/app_{name}):\n  "
                    + "\n  ".join(diffs[:40]), pytrace=False)
