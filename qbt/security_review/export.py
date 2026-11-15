"""Excel export of a Security review.

`full_report` builds the same workbook the Backtest's return-stream export
builds (`qbt.excel_export.workbook_from_returns`): a two-page executive
summary, an About sheet indexing every table with links, the statistics,
CIPM appraisal, trailing periods, calendar years, drawdowns, monthly returns
and the in-depth performance analysis -- for the portfolio WITH the security
measured against the portfolio WITHOUT it as the benchmark. Every Security
review table (`report.all_sections`, the same list the page renders) follows
on its own sheets.

`write_sections` writes only the review tables (used by the scenario matrix).
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from .. import analysis_sheet as ANALYSIS
from .. import excel_export as XL
from .. import metrics as M
from ..excel_export import _TITLE_FONT, _style_block, _widths
from . import report as RP
from .config import AllConfig
from .engine import Review

PPY = 12
CAPITAL = 100_000.0


def write_sections(secs: List[RP.Section]) -> bytes:
    buf = io.BytesIO()
    order = [s for s in RP.SHEETS if any(x[0] == s for x in secs)]
    order += [s for s in dict.fromkeys(x[0] for x in secs) if s not in order]
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for sheet in order:
            r = 0
            titles = set()
            for sh, title, df, note in secs:
                if sh != sheet or df is None or df.empty:
                    continue
                df.to_excel(xw, sheet_name=sheet, startrow=r + 1, index=False)
                ws = xw.sheets[sheet]
                ws.cell(row=r + 1, column=1, value=title).font = _TITLE_FONT
                titles.add(r + 1)
                _style_block(ws, r + 2, len(df), df.shape[1], df, False, title)
                end = r + 2 + len(df)
                if note:
                    ws.cell(row=end + 1, column=1, value=note)
                    titles.add(end + 1)
                r = end + 3
            if sheet in xw.sheets:
                _widths(xw.sheets[sheet], titles)
                if sheet == "Series":
                    xw.sheets[sheet].freeze_panes = "B3"
    return buf.getvalue()


def workbook(rv: Review, cfg: AllConfig, sweep: Optional[pd.DataFrame] = None) -> bytes:
    return write_sections(RP.sections(rv, cfg, sweep))


# ----------------------------------------------------------------------
# Full report, in the Backtest's format
# ----------------------------------------------------------------------
def labels(rv: Review, c: str) -> Tuple[str, str]:
    """(portfolio with the security, portfolio without it)."""
    base = RP.base_label(rv)
    if rv.request.mode == "add":
        return f"{base} + {rv.weight:.0%} {c}", base
    return f"{base}, {rv.request.replace_holding} replaced by {c}", base


def _stress_for_summary(rv: Review, c: str) -> pd.DataFrame:
    st = rv.results[c].stress
    if st is None or st.empty or "Status" not in st:
        return pd.DataFrame()
    ok = st[st["Status"] == "ok"]
    return pd.DataFrame({"Period": ok["Window"], "Start": ok["Start"], "Coverage": "Full",
                         "Return": ok["Portfolio return, with"],
                         "Benchmark Return": ok["Portfolio return, without"],
                         "Excess vs Benchmark": ok["Change in portfolio return"]})


def _regimes_for_summary(rv: Review, c: str, cfg: AllConfig, label: str, blabel: str) -> pd.DataFrame:
    r = rv.results[c]
    rg = r.regimes
    if rg is None or rg.empty:
        return pd.DataFrame()
    rules = cfg.scenarios["regimes"]
    rows = []
    for _, x in rg.iterrows():
        key = x["Key"][4:] if x["Key"].startswith("not_") else x["Key"]
        dim = rules.get(key, {}).get("label", key)
        share = x["Months"] / r.months if r.months else float("nan")
        a, b = x["Portfolio ann. return, with"], x["Portfolio ann. return, without"]
        ex = (1 + a) / (1 + b) - 1 if a == a and b == b else float("nan")
        rows.append({"Dimension": dim, "Regime": x["Regime"], "Series": label, "Ann. return": a,
                     "% of time": share, "Excess ann. return": ex})
        rows.append({"Dimension": dim, "Regime": x["Regime"], "Series": blabel, "Ann. return": b,
                     "% of time": share, "Excess ann. return": float("nan")})
    return pd.DataFrame(rows)


def full_report(rvs: List[Review], cfg: AllConfig, focus: Optional[Tuple[Review, str]] = None,
                sweep: Optional[pd.DataFrame] = None) -> bytes:
    """The Backtest-style workbook for one security added to one portfolio
    (`focus`, default the first result), with every review table after it."""
    pairs = [(rv, c) for rv in rvs for c in rv.results]
    if not pairs:
        return write_sections(RP.all_sections(rvs, cfg, sweep))
    rv, c = focus or pairs[0]
    r = rv.results[c]
    label, blabel = labels(rv, c)
    rw = r.series["Portfolio with"].dropna()
    rb = r.series["Portfolio without"].reindex(rw.index)
    rf = (rv.rf if rv.rf is not None else pd.Series(0.0, index=rw.index)).reindex(rw.index).fillna(0.0)
    eq, beq = M.to_equity(rw, CAPITAL), M.to_equity(rb, CAPITAL)
    stats = M.summary(rw, eq, rb, None, None, rf, PPY)
    bstats = M.summary(rb, beq, None, None, None, rf, PPY)
    per = M.period_table(eq, rw, beq, rb, PPY, label, blabel)
    period = f"{rw.index[0]:%Y-%m} to {rw.index[-1]:%Y-%m}"
    now = datetime.now()
    try:
        analysis = ANALYSIS.build_analysis(
            rw, rb, rf, PPY, title=label, subtitle=f"{period} · vs {blabel} (without {c})",
            footer=f"{label} · {now:%Y-%m-%d}", rf_label=rv.rf_label, returns_only=True,
            label=label, bench_label=blabel)
    except Exception:                     # the review tables still go out
        analysis = None
    curve = pd.DataFrame({label: eq, blabel: beq}).resample("ME").last()
    weights = pd.DataFrame([{"Instrument": t, "Weight": w, "Target": w, "Value": w * CAPITAL}
                            for t, w in sorted(r.weights_new.items(), key=lambda x: -x[1])])
    comments = RP.headline(rv, c)
    if r.flags:
        comments += "\n\nCaveats: " + " ".join(r.flags)
    summary = {
        "title": label,
        "subtitle": " · ".join([period, f"vs {blabel} (the portfolio without {c})",
                                "monthly returns, CAD", f"generated {now:%Y-%m-%d}"]),
        "label": label, "bench_label": blabel, "stats": stats, "bench_stats": bstats,
        "trailing": per["trailing"], "calendar": per["calendar"],
        "drawdowns": M.drawdown_table(eq, 5, PPY), "curve": curve,
        "capital_label": f"${CAPITAL:,.0f}", "weights": weights, "as_of": "target",
        "stress": _stress_for_summary(rv, c),
        "regimes": _regimes_for_summary(rv, c, cfg, label, blabel),
        "years": len(rw) / PPY, "comments": comments,
        "assumptions": {
            "Question": f"What adding {c} does to {blabel}",
            "Change": RP.change_label(rv) + (", funded pro rata from every holding"
                                              if rv.request.mode == "add" and not rv.request.fund_from
                                              else ""),
            "Benchmark": f"{blabel}, the portfolio without {c}",
            "Rebalancing": rv.request.rebalance or rv.profile.rebalance,
            "Period": period,
            "Returns": "Total returns (Yahoo Finance adjusted close), month end to month end, "
                       "in CAD; uploaded streams as supplied",
            "Risk-free rate": rv.rf_label,
            "Performance standard": "CIPM: population SD, geometric annualisation, geometric "
                                    "excess; nothing annualised under 12 months; alpha and beta "
                                    "withheld below R² 0.70",
            "Objective": rv.objective_label,
        },
        "footer": f"{label} · {now:%Y-%m-%d}",
    }
    sections = [(sh, t, df, False, note) for sh, t, df, note in RP.all_sections(rvs, cfg, sweep)]
    omitted = [("Data", "Holdings history, trades, tax, attribution",
                "The portfolio is simulated from monthly returns at target weights; there are no "
                "trades or tax lots to report."),
               ("Robustness", "Monte Carlo and walk-forward of the Backtest",
                "The review's own robustness (rolling windows, add-weight grid, block bootstrap) "
                "is on the Robustness and Portfolio Impact sheets.")]
    notes = {"Analysis": "Security review", "Security": c, "Portfolio": blabel,
             "Period": period, "Risk-free rate": rv.rf_label,
             "Other results in this workbook": ", ".join(f"{x} + {RP.base_label(v)}"
                                                          for v, x in pairs if not (v is rv and x == c))
             or "none"}
    return XL.workbook_from_returns(rw, eq, stats, rb, beq, bstats, PPY, label, blabel,
                                    report_notes=notes, sections=sections, omitted=omitted,
                                    summary=summary, analysis=analysis)
