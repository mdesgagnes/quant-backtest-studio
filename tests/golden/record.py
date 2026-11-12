"""Records the reference outputs. Deliberate, manual, and reviewed in a diff.

    python tests/golden/record.py                 # every scenario
    python tests/golden/record.py momentum_file   # one

Only run this when a change in results is intended and understood. During
the interface redesign it should never be needed: the redesign must
reproduce what is already recorded.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import harness as H                      # noqa: E402
from scenarios import SCENARIOS          # noqa: E402


def main(names) -> None:
    for name in names or list(SCENARIOS):
        t0 = time.perf_counter()
        with H.LegacyApp() as app:
            SCENARIOS[name](app)
            rec = H.record(app)
            steps = ", ".join(f"{w} {s:.1f}s" for w, s in app.timings)
        H.save(rec, os.path.join(H.EXPECTED, name))
        print(f"{name}: recorded in {time.perf_counter() - t0:.0f}s ({steps})")


if __name__ == "__main__":
    main(sys.argv[1:])
