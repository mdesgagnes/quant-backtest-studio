"""Backtest > Results. A port of app.py:3861-3993, block for block.

The reference screen: every later screen follows this file's pattern. Take
the tab's code from app.py, keep its expressions and its wording, and emit
blocks instead of Streamlit calls.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from qbt import charts as C
from qbt import cipm as CIPM
from qbt import metrics as M
from qbt.engine import align_results

from .common import (Block, RunView, caption, chart, heading, kpi, lines, note,
                     panel, table)

KPI_KEYS = ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar", "Sortino"]


def _fmt_period_tables(tables, main_label, bench_label=None):
    """Formats the trailing / calendar tables for display (app.py:525)."""
    out = {}
    for key in ("trailing", "calendar"):
        t = tables[key].copy()
        if t.empty:
            out[key] = t
            continue
        for c in [main_label] + ([bench_label] if bench_label else []) + ["Excess"]:
            if c in t.columns:
                t[c] = t[c].map(lambda v: "n/a" if pd.isna(v) else f"{v*100:+.2f}%")
        out[key] = t
    return out


def kpis(v: RunView) -> List[Block]:
    out = []
    for k in KPI_KEYS:
        val = v.stats.get(k, np.nan)
        tone = ""
        if k in ("CAGR", "Sharpe", "Calmar", "Sortino"):
            tone = "pos" if (val == val and val > 0) else "neg"
        elif k == "Max Drawdown":
            tone = "neg"
        sub = ""
        if v.bstats:
            bv = v.bstats.get(k, np.nan)
            if bv == bv:
                sub = f"bench. {M.format_metric(k, bv)}"
        out.append(kpi(k, M.format_metric(k, val), sub, tone))
    return out


def build(v: RunView) -> List[Block]:
    res, bench, cfg, ppy, run = v.res, v.bench, v.cfg, v.ppy, v.run
    blocks: List[Block] = []

    if run.get("trimmed"):
        dropped = run["raw_start"].date()
        why = ("the first day the model took a position"
               if (v.sleeves is not None and v.sleeves.signal_start is not None)
               else "the first day the strategy held a position")
        blocks.append(note(
            f"Warm-up trimmed: the record starts on "
            f"{res.equity.index[0].date()}, {why}, rather than {dropped}. "
            f"Anything held before that (a fixed core, cash) is "
            f"excluded, and the benchmark is measured over the same window."))
    nx, _err = v.next_rebalance()
    if nx is not None:
        blocks.append(note("<b>Next rebalance.</b> " + nx.headline()
                           + " The trade list is in Positions."))

    blocks += kpis(v)

    curves = {res.label: res.equity}
    if bench is not None:
        curves[bench.label] = bench.equity
    # Both curves rebased to 100 at their common start.
    rebased = align_results(curves)
    blocks.append(chart(
        "equity", "Portfolio value (base 100)", lines(rebased),
        legacy=("equity_curve", (rebased, True), {}),
        log=True, log_toggle=True,
        log_help="A log scale makes relative changes comparable across the whole period."))

    dd = pd.DataFrame({name: M.drawdown(eq) * 100 for name, eq in curves.items()})
    blocks.append(chart(
        "drawdown", "Drawdown from prior peak", lines(dd),
        legacy=("underwater", ({k: e for k, e in curves.items()},), {}),
        suffix="%"))

    mtbl = M.monthly_returns(res.returns) if ppy >= 12 else pd.DataFrame()
    blocks.append(chart(
        "monthly_heatmap", "Monthly returns",
        {"years": [str(y) for y in mtbl.index],
         "z": (mtbl.reindex(columns=range(1, 13)).values * 100) if not mtbl.empty
         else np.zeros((0, 12))},
        legacy=("monthly_heatmap", (res.returns,), {"ppy": ppy})))

    r_pct = res.returns.dropna() * 100
    blocks.append(chart(
        "distribution", f"{M.freq_words(ppy)[0].capitalize()} return distribution",
        {"values": r_pct, "var95": float(np.percentile(r_pct, 5)), "bins": 90},
        legacy=("return_distribution", (res.returns,), {"ppy": ppy}), suffix="%"))

    win = min(252, max(63, len(res.returns) // 6))
    roll = M.rolling_sharpe(res.returns, win, ppy, v.rf)
    blocks.append(chart(
        "line", f"Rolling Sharpe over {win} sessions",
        lines(roll.to_frame("Rolling Sharpe")),
        legacy=("rolling_metric", (roll, "Rolling Sharpe"),
                {"ref": 0.0, "title": f"Rolling Sharpe over {win} sessions"}),
        ref=0.0))

    blocks.append(heading("Full statistics"))
    order = list(v.stats.keys())
    tbl = pd.DataFrame({
        "Metric": order,
        cfg.label: [M.format_metric(k, v.stats[k]) for k in order],
    })
    if v.bstats:
        tbl[bench.label] = [M.format_metric(k, v.bstats.get(k, np.nan)) for k in order]
    blocks.append(table(tbl, key="statistics"))

    ap = CIPM.appraisal_table(res.returns, v.bench_r, v.rf, ppy)
    blocks.append(panel(
        "CIPM appraisal: risk-adjusted, relative and drawdown measures", [
            table(CIPM.format_appraisal(ap), key="cipm"),
            note("One consistent set: same dates, same risk-free, population "
                 "standard deviation, geometric annualisation and geometric excess "
                 "return. Alpha, beta, Treynor and the appraisal ratio are "
                 "withheld when R² against the benchmark is below 0.70, since "
                 "the regression then explains too little to mean anything. "
                 "Annualised figures need a full year of history."),
            caption(CIPM.methodology_block(
                ppy, rf=v.rf_label,
                benchmark=(bench.label if bench is not None else "none"),
                n_obs=len(res.returns))),
        ]))

    blocks.append(heading("Trailing periods"))
    ptabs = M.period_table(res.equity, res.returns,
                           bench.equity if bench is not None else None,
                           bench.returns if bench is not None else None,
                           ppy, cfg.label, bench.label if bench is not None else "Benchmark")
    fmt = _fmt_period_tables(ptabs, cfg.label, bench.label if bench is not None else None)
    if not fmt["trailing"].empty:
        cols = [c for c in fmt["trailing"].columns
                if c not in ("Period", "Annualized", "From", "To")]
        blocks.append(table(fmt["trailing"], signed=cols, emphasise=["Excess"],
                            key="trailing"))
        blocks.append(note(
            "Periods longer than one year are annualized; shorter ones are "
            "cumulative, as the <b>Annualized</b> column records. A window "
            "the history does not cover is omitted rather than measured "
            "over a shorter span and labelled as though it were complete."))

    blocks.append(heading("Calendar years"))
    if not fmt["calendar"].empty:
        cols = [c for c in fmt["calendar"].columns if c not in ("Year", "Partial")]
        blocks.append(table(fmt["calendar"], signed=cols, emphasise=["Excess"],
                            key="calendar"))
        cy = ptabs["calendar"]
        if cfg.label in cy.columns:
            labels = [str(y) for y in cy["Year"]]
            values = (cy[cfg.label] * 100).tolist()
            blocks.append(chart(
                "bars", "Return by calendar year", {"labels": labels, "values": values},
                legacy=("bar_series", (labels, values, "Return by calendar year", "%"), {}),
                suffix="%"))
        if (cy["Partial"].astype(str) != "").any():
            blocks.append(note("A partial first or last year is flagged in the "
                               "<b>Partial</b> column: those are not full-year figures."))

    blocks.append(heading("Main drawdown episodes"))
    dd_tbl = M.drawdown_table(res.equity, 6, ppy)
    if not dd_tbl.empty:
        dd_tbl["Drawdown"] = dd_tbl["Drawdown"].map(lambda x: f"{x*100:.2f}%")
        dd_tbl["Recovery"] = dd_tbl["Recovery"].astype(str)
    blocks.append(table(dd_tbl, signed=["Drawdown"], key="drawdowns"))
    return blocks


def legacy_figure(block: Block):
    """The Plotly figure the Streamlit app draws for a chart block."""
    name, args, kwargs = block["legacy"]
    return getattr(C, name)(*args, **kwargs)
