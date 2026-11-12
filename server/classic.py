"""Runs the classic app's own screen logic under the recording layer.

`app.py` is read once, adjusted in memory (never on disk), compiled, and
executed for each request with `st` bound to `server.shim.st`. The
adjustments keep every line on its original line number, so a traceback
points at the real line of `app.py`:

1. `import streamlit as st` becomes the recording proxy.
2. `with tabs[i]:` becomes `for _ in tabs[i].body():`, which lets a tab that
   is not on screen be skipped instead of computed.
3. Three expensive calls are remembered for the life of a run (walk-forward,
   in/out-of-sample, cost sensitivity), the tearsheet is rendered only when
   downloaded, and the Excel analysis sheet is remembered.
4. The TradingView chart goes to a sandboxed frame.
5. The Research workspace is rendered by `_lab_render` below, which is
   `qbt.lab.ui.render` with skippable tabs.

Nothing here computes a figure. If `app.py` changes, the checks below fail
loudly rather than run a half-adjusted script.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List

from . import shim
from .shim import st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")

# Which other tabs a tab needs to have run, keyed "<tabs in the set>:<tab>".
# The Export tabs read values the other tabs compute (app.py:5851, 2028).
TAB_NEEDS = {
    "10:Export": ["Signals", "Positions", "Attribution", "Tax", "Robustness"],
    "6:Export": ["Results", "Stress tests", "Market regimes", "Robustness"],
}

_TAB = re.compile(r"^(\s*)with (tabs|rs_tabs|mtabs)\[(\d+)\]:\s*$")


class Lazy:
    """Text or bytes produced only when asked for (a download)."""

    def __init__(self, fn: Callable[[], Any]):
        self._fn, self._done, self._value = fn, False, None

    def __call__(self) -> Any:
        if not self._done:
            self._value, self._done = self._fn(), True
        return self._value

    def encode(self, *args) -> "Lazy":
        return Lazy(lambda: self().encode(*args))


class _Module:
    """A module with a few functions replaced."""

    def __init__(self, module: Any, overrides: Dict[str, Callable[..., Any]]):
        self.__dict__["_module"] = module
        self.__dict__["_overrides"] = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._module, name)


def _remembered(module: Any, name: str) -> Callable[..., Any]:
    fn = getattr(module, name)

    def call(*args, **kwargs):
        key = (module.__name__, name, shim.arg_key(args), shim.arg_key(kwargs))
        return shim.memo(key, lambda: fn(*args, **kwargs))
    return call


def _wrap_heavy(R: Any, REPORT: Any, ANALYSIS: Any, TAX: Any, ATTR: Any, STRESS: Any, XL: Any, C: Any):
    """Results that depend only on the run are computed once per run. Each
    is keyed by the content of its arguments, so a changed setting (a fold
    count, a tax rate) computes afresh. None of these results is modified
    in place by the code that uses them."""
    def remember(module, *names):
        return _Module(module, {n: _remembered(module, n) for n in names})
    return (
        remember(R, "walk_forward", "in_out_sample", "cost_sensitivity"),
        _Module(REPORT, {"render_tearsheet":
                         lambda *a, **k: Lazy(lambda: REPORT.render_tearsheet(*a, **k))}),
        remember(ANALYSIS, "build_analysis"),
        remember(TAX, "evaluate"),
        remember(ATTR, "summary", "cumulative", "by_period", "benchmark_relative"),
        remember(STRESS, "evaluate_periods"),
        remember(XL, "regime_sections"),
        _Module(C, {n: _unless_muted(getattr(C, n)) for n in dir(C)
                    if not n.startswith("_") and callable(getattr(C, n))
                    and getattr(getattr(C, n), "__module__", "") == C.__name__}),
    )


def _unless_muted(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A chart is not built for a tab that is only run for the values it
    leaves behind (nothing of it is shown)."""
    def call(*args, **kwargs):
        if shim.recorder().muted:
            return None
        return fn(*args, **kwargs)
    return call


def _lab_render(lab_ui: Any) -> None:
    """qbt.lab.ui.render, with tabs that are skipped when not on screen."""
    S_ = lab_ui._state()
    lab_ui._sidebar(S_)
    st.markdown('<div class="masthead"><div><h1>Research Lab</h1>'
                '<div class="sub">Find an idea, test it honestly, decide. Research only: nothing here trades.'
                '</div></div></div>', unsafe_allow_html=True)
    ds = lab_ui._dataset()
    tabs = st.tabs(["Start here", "Ledger", "Factor Lab", "Strategy Lab", "Risk X-Ray", "Event Study",
                    "Monitor", "Assistant"])
    for _ in tabs[0].body():
        lab_ui._tab_start(S_, ds)
    for _ in tabs[1].body():
        lab_ui._tab_ledger(S_, ds)
    needs_data = [lab_ui._tab_factors, lab_ui._tab_strategy, lab_ui._tab_xray, lab_ui._tab_events]
    for i, fn in enumerate(needs_data, start=2):
        for _ in tabs[i].body():
            if ds is None:
                st.info("Load a universe from the sidebar first (left): pick the instruments you want to research "
                        "and press **Load universe**.")
            else:
                fn(S_, ds)
    for _ in tabs[6].body():
        lab_ui._tab_monitor(S_)
    for _ in tabs[7].body():
        lab_ui._tab_assistant(S_)


def _adjust(lines: List[str]) -> List[str]:
    out = list(lines)

    def swap(old: str, new: str, count: int = 1) -> None:
        hits = [i for i, ln in enumerate(out) if old in ln]
        if len(hits) != count:
            raise RuntimeError(
                f"app.py no longer matches server/classic.py: expected {count} line(s) "
                f"containing {old!r}, found {len(hits)}.")
        for i in hits:
            out[i] = out[i].replace(old, new)

    swap("import streamlit as st", "from server.shim import st")
    swap("from qbt import fund_review_export as FREVX",
         "from qbt import fund_review_export as FREVX; "
         "R, REPORT, ANALYSIS, TAX, ATTR, STRESS, XL, C = "
         "_wrap_heavy(R, REPORT, ANALYSIS, TAX, ATTR, STRESS, XL, C)")
    swap("_components.html(html, height=height + 6, scrolling=False)",
         "st.html_frame(html, height + 6)")
    swap("    lab_ui.render()", "    _lab_render(lab_ui)")
    tabs = 0
    for i, ln in enumerate(out):
        m = _TAB.match(ln)
        if m:
            out[i] = f"{m.group(1)}for _tab_ in {m.group(2)}[{m.group(3)}].body():"
            tabs += 1
    if tabs != 25:
        raise RuntimeError(f"app.py no longer matches server/classic.py: {tabs} tab blocks, expected 25.")
    return out


def _compile():
    with open(APP, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    return compile("\n".join(_adjust(lines)), APP, "exec")


_CODE = None


def _patch_views() -> None:
    """The view modules inside qbt import streamlit themselves; point them
    at the recording layer too. (Only in this process: the classic app runs
    in its own.)"""
    from qbt import groups_ui, snapshot_ui
    from qbt.lab import ui as lab_ui
    for mod in (snapshot_ui, groups_ui, lab_ui):
        mod.st = st
    # The classic charts are built as Plotly figures and only their data is
    # read here (the new front end styles them itself). Plotly's built-in
    # themes cost about 0.2 s per figure to apply, so in this process they
    # are empty. Exported documents draw with matplotlib and are unaffected.
    import plotly.graph_objects as go
    import plotly.io as pio
    for name in ("plotly_dark", "plotly_white"):
        pio.templates[name] = go.layout.Template()


def script() -> None:
    """One pass of the classic app. Call inside `shim.render`."""
    global _CODE
    if _CODE is None:
        _patch_views()
        _CODE = _compile()
    session = shim.recorder().session
    run = session.state.get("run")
    stamp = id(run) if run is not None else None
    if session.memo_stamp != stamp:              # a new run: forget the old one's results
        session.memo.clear()
        session.memo_stamp = stamp
    loaded = id(session.state.get("loaded_cfg"))
    try:
        exec(_CODE, {"__name__": "qbt_classic", "__file__": APP,
                     "_wrap_heavy": _wrap_heavy, "_lab_render": _lab_render})
    finally:
        _apply_loaded_config(session, loaded)


# Session entries that are not control values and must survive a load.
_KEEP = {"workspace", "loaded_cfg", "loaded_rules", "run", "auth_ok", "lab", "mk_data", "gr_data",
         "fr_out", "rule_list"}


def _apply_loaded_config(session: shim.Session, before: int) -> None:
    """In the classic app a loaded preset or YAML file only reached the
    controls that had no value yet, so the symbols, the model's parameters,
    slippage, cash rate and fee stayed as they were (checklist U1). Here a
    load clears what the Backtest controls hold, so every one of them takes
    the loaded configuration on the pass that follows."""
    cfg = session.state.get("loaded_cfg")
    if cfg is None or id(cfg) == before:
        return
    rec = shim.recorder()
    for wid, meta in rec.registry.items():
        key = meta.get("key")
        if key is None:
            session.auto.pop(wid, None)
        elif key not in _KEEP and not key.startswith(("mk_", "lab_", "gr_", "sn_", "sc_", "rs")):
            session.state.pop(key, None)
    session.state["_preset_applied"] = "Custom"
    for k in ("_preset_bench", "_preset_cash", "tickers_box", "benchspec", "corespec",
              "ccy_override_text"):
        session.state.pop(k, None)
    session.state.pop("rule_list", None)          # the Builder takes the loaded rules
    raise shim.Rerun()


def render(session: shim.Session, tab: str = None, all_tabs: bool = False,
           skip: set = None) -> shim.Recorder:
    with session.lock:
        return shim.render(session, script, active_tab=tab, all_tabs=all_tabs,
                           tab_needs=TAB_NEEDS, skip=skip)
