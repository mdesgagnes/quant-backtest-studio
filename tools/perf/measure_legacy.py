"""Measures the current Streamlit app's server-side script time.

    python tools/perf/measure_legacy.py            # offline, golden fixture
    python tools/perf/measure_legacy.py --yahoo    # also a live Yahoo run

Every Streamlit interaction re-executes `app.py` from the top, so the time a
person waits after touching a control is the script time below plus the
network round trip and the browser's own rendering. The script time is
measured by running the real app headlessly (the golden-test harness); the
browser side is measured separately (see docs/HANDOFF.md once it exists, and
the Phase 1 report).

The profile wraps the engine functions the page calls and reports how much of
one rerun each accounts for. Nothing in the app is modified.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "golden"))
sys.path.insert(0, ROOT)

import harness as H                                   # noqa: E402

PROFILE = [
    ("qbt.robustness", ["walk_forward", "in_out_sample", "cost_sensitivity",
                        "monte_carlo", "parameter_sweep", "rebalance_day_sweep"]),
    ("qbt.nextrade", ["next_rebalance"]),
    ("qbt.report", ["render_tearsheet"]),
    ("qbt.tax", ["evaluate"]),
    ("qbt.attribution", ["summary", "cumulative", "by_period", "benchmark_relative",
                         "group_columns"]),
    ("qbt.cipm", ["appraisal_table"]),
    ("qbt.stress", ["evaluate_periods"]),
    ("qbt.metrics", ["summary", "period_table", "drawdown_table", "rolling_sharpe"]),
    ("qbt.formula", ["describe", "evaluate_frame"]),
    ("qbt.analysis_sheet", ["build_analysis"]),
    ("qbt.excel_export", ["tax_sections", "robustness_sections", "stress_sections",
                          "regime_sections"]),
    ("qbt.exec_summary", ["current_weights_table"]),
    ("qbt.lab.eventstudy", ["series_event_study"]),
    ("qbt.engine", ["run_backtest", "benchmark_result", "blended_benchmark"]),
    ("qbt.price_import", ["read_files", "to_market"]),
    ("qbt.data", ["load_market_data", "load_yfinance", "clean_prices"]),
]


class Profiler:
    def __init__(self):
        self.total = defaultdict(float)
        self.calls = defaultdict(int)
        self._undo = []
        self._depth = 0

    def install(self):
        import importlib
        for mod_name, names in PROFILE:
            mod = importlib.import_module(mod_name)
            for n in names:
                if not hasattr(mod, n):
                    continue
                self._wrap(mod, n, f"{mod_name.split('.')[-1]}.{n}")
        # Chart construction, as one line.
        from qbt import charts
        for n in dir(charts):
            f = getattr(charts, n)
            if callable(f) and not n.startswith("_") and getattr(f, "__module__", "") == charts.__name__:
                self._wrap(charts, n, "charts.*")

    def _wrap(self, mod, name, label):
        orig = getattr(mod, name)
        prof = self

        def timed(*a, **k):
            # Only the outermost wrapped call is charged, so nested engine
            # calls (walk_forward -> run_backtest) are not counted twice.
            outer = prof._depth == 0
            prof._depth += 1
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                prof._depth -= 1
                if outer:
                    prof.total[label] += time.perf_counter() - t0
                    prof.calls[label] += 1
        timed.__wrapped__ = orig
        setattr(mod, name, timed)
        self._undo.append((mod, name, orig))

    def reset(self):
        self.total.clear()
        self.calls.clear()

    def remove(self):
        for mod, name, orig in self._undo:
            setattr(mod, name, orig)


def _last(app):
    return app.timings[-1][1]


def offline_run():
    print("\n== Offline: golden fixture (4 instruments, 2012-2025, uploaded file) ==")
    prof = Profiler()
    rows = []
    with H.LegacyApp() as app:
        rows.append(("Open the app (script only, no result)", _last(app)))
        app.upload("price_files", "prices_long.csv")
        app.set("radio", "File (CSV / Excel)", label="Source").run()
        rows.append(("Switch source, parse upload (no result)", _last(app)))
        app.set("number_input", 10.0, key="slip_bps").run()
        rows.append(("Change a control before any run", _last(app)))
        app.set("number_input", 25.0, key="slip_bps").run()

        prof.install()
        app.click("Run backtest")
        rows.append(("Run backtest and render every tab", _last(app)))
        run_profile = dict(prof.total)

        idle = []
        for _ in range(3):
            prof.reset()
            app.run()
            idle.append(_last(app))
        rows.append(("Rerun with a result on screen, nothing changed (median of 3)",
                     statistics.median(idle)))
        idle_profile, idle_calls = dict(prof.total), dict(prof.calls)

        app.widget("toggle", "Log scale", sidebar=False).set_value(False)
        app.run()
        rows.append(("Toggle Log scale on the equity chart", _last(app)))
        app.set("slider", 7, key="wf", sidebar=False).run()
        rows.append(("Move the folds slider (Robustness)", _last(app)))
        app.set("selectbox", "Quarter", key="attr_period", sidebar=False).run()
        rows.append(("Change the attribution period", _last(app)))
        app.set("number_input", 10.0, key="slip_bps").run()
        rows.append(("Change a sidebar control (result unchanged)", _last(app)))
        app.click("Run backtest")
        rows.append(("Run the identical backtest again", _last(app)))
        t0 = time.perf_counter()
        app.download("dlxl")
        rows.append(("Build the Excel workbook (on click)", time.perf_counter() - t0))
        prof.remove()

    for label, secs in rows:
        print(f"  {secs:7.2f} s  {label}")
    total = statistics.median(idle)
    print(f"\n  Where one rerun goes ({total:.2f} s, nothing changed):")
    for label, secs in sorted(idle_profile.items(), key=lambda kv: -kv[1])[:14]:
        print(f"  {secs:7.2f} s  {secs / total * 100:5.1f}%  {label} (x{idle_calls[label]})")
    rest = total - sum(idle_profile.values())
    print(f"  {rest:7.2f} s  {rest / total * 100:5.1f}%  everything else "
          f"(Streamlit, formatting, page assembly)")
    return rows, idle_profile, run_profile


def yahoo_run():
    print("\n== Live: Yahoo Finance, default universe and a 32-ETF preset ==")
    rows = []
    with H.LegacyApp(offline=False) as app:
        t0 = time.perf_counter()
        app.click("Run backtest")
        rows.append(("Default universe (3 ETFs from 2010): download + run + render",
                     time.perf_counter() - t0))
        app.run()
        rows.append(("Default universe: rerun, nothing changed", _last(app)))
        n = len(app.at.session_state["run"]["result"].equity)
        app.click("Run backtest")
        rows.append(("Default universe: identical run again (prices cached)", _last(app)))
        app.set("selectbox", "Global multi-asset ETF universe", label="Preset universe").run()
        app.click("Run backtest")
        rows.append(("32-ETF preset: download + run + render", _last(app)))
        app.run()
        rows.append(("32-ETF preset: rerun, nothing changed", _last(app)))
        n2 = len(app.at.session_state["run"]["result"].equity)
    for label, secs in rows:
        print(f"  {secs:7.2f} s  {label}")
    print(f"  ({n:,} sessions in the default run, {n2:,} in the 32-ETF run)")
    return rows


if __name__ == "__main__":
    offline_run()
    if "--yahoo" in sys.argv:
        yahoo_run()
