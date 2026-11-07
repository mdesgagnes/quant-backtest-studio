"""Attribution and in-depth performance analysis: a formatted sheet for the
Excel report, in the same house style as the Executive Summary.

It carries what is too detailed for the two-page summary and is usually
presented on its own:

- the benchmark-relative attribution (Brinson-Fachler, linked): headline
  effects, the category table with the proxy behind every benchmark return,
  each calendar year reconciled to its own excess return, and two charts;
- what each holding contributed to the return and to the risk;
- the full CIPM appraisal, strategy beside benchmark: total-risk ratios and
  the skew-and-kurtosis adjusted ones, downside, drawdown, relative risk and
  skill, CAPM (withheld below R-squared 0.70), up and down capture;
- the methodology, since every number on the sheet is defined by it.

Written with openpyxl into the workbook the export already builds.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.page import PageMargins

from . import cipm as CIPM
from .exec_summary import (BRAND_RED, GAIN, INK, LOSS, MUTED, NUM, PCT, PCT2, SAND,
                           _Page, _f, _isnum, _lines_needed, _pct, _pts)

SHEET_ATTR = "Attribution & Analysis"
SHEET_PERF = "Performance Analysis"          # imported return streams: no attribution
_DATA = "_Analysis Data"

# One wide label column, then eight figure columns.
WIDTHS = {1: 1.5, 2: 32, **{c: 11.5 for c in range(3, 11)}, 11: 1.5}
FIRST, LAST = 2, 10

STANDARD = (
    "Standard: CIPM program and Carl Bacon, Practical Portfolio Performance Measurement and "
    "Attribution (3rd ed.). Returns are time-weighted from daily valuation, total return, net of "
    "trading costs and fees. Dispersion statistics use population moments (divide by n). Returns "
    "are annualised geometrically and never for a record shorter than one year. Excess return is "
    "geometric, (1+r)/(1+b)-1, for tracking error and the information ratio; the attribution effects "
    "below are arithmetic Brinson-Fachler (interaction folded into selection), calculated per "
    "period from the daily ledger and linked across periods so that they add up to the excess return "
    "exactly. Alpha, beta, Treynor and the appraisal ratio are withheld when R-squared against the "
    "benchmark is below 0.70. The same risk-free series is used for every measure and every portfolio."
)


def _na(v):
    """A figure, or the text 'n/a' (right-aligned by the page helper)."""
    return v if _isnum(v) else "n/a"


def _blank(v):
    return v if _isnum(v) else ""


# ----------------------------------------------------------------------
def _val(table: pd.DataFrame, measure: str, col: str = "Strategy"):
    if table is None or table.empty or col not in table:
        return np.nan
    m = table[table["Measure"] == measure]
    return m[col].iloc[0] if len(m) else np.nan


def observations(d: Dict[str, Any]) -> List[str]:
    """Plain statements of what the figures show; never an opinion they
    do not support."""
    out: List[str] = []
    ba, ap, bl = d.get("battr"), d.get("appraisal"), d.get("bench_label") or "the benchmark"
    if ba:
        lk, by = ba["linked"], ba.get("by_year")
        tot = lk.loc["Total"]
        costs = lk.loc["Costs & fees", "Total effect"] if "Costs & fees" in lk.index else 0.0
        out.append(
            f"The portfolio returned {_pct(ba['R'])} against {_pct(ba['B'])} for {bl}, "
            f"{_pts(ba['excess'])} of excess return: allocation {_pts(tot['Allocation'])}, "
            f"selection {_pts(tot['Selection'])}, costs and fees {_pts(costs)}.")
        cats = lk.drop(index=[i for i in ("Total", "Costs & fees") if i in lk.index])
        if len(cats):
            top = cats["Total effect"].abs().idxmax()
            out.append(f"The largest single effect came from {top} "
                       f"({_pts(cats.loc[top, 'Total effect'])}: allocation "
                       f"{_pts(cats.loc[top, 'Allocation'])}, selection {_pts(cats.loc[top, 'Selection'])}).")
        if by is not None and len(by) > 1:
            b, w = by.loc[by["Excess"].idxmax()], by.loc[by["Excess"].idxmin()]
            out.append(f"Best year against the benchmark {int(b['Year'])} ({_pts(b['Excess'])}), "
                       f"weakest {int(w['Year'])} ({_pts(w['Excess'])}).")
        if abs(ba["reconciliation_error"]) > 1e-6:
            out.append("WARNING: the effects do not reconcile to the excess return; "
                       "this is an error to investigate.")
    if ap is not None and not ap.empty:
        s, bs_ = _val(ap, "Sharpe"), _val(ap, "Sharpe", "Benchmark")
        if _isnum(s):
            t = f"Sharpe ratio {s:.2f}" + (f" against {bs_:.2f} for {bl}" if _isnum(bs_) else "")
            a = _val(ap, "Adjusted Sharpe (Pezier-White)")
            if _isnum(a):
                t += f"; adjusted for skewness and kurtosis {a:.2f}"
            out.append(t + ".")
        k = _val(ap, "Kurtosis (normal = 3)")
        sk = _val(ap, "Skewness (population)")
        if _isnum(k) and (k > 4 or (_isnum(sk) and sk < -0.5)):
            out.append(f"The return distribution has fat tails or a negative skew (kurtosis {k:.1f} "
                       f"against 3 for a normal distribution, skewness {sk:.2f}): the plain Sharpe "
                       f"ratio understates the risk.")
        ir, ts, yrs = (_val(ap, "Information ratio (geometric)"),
                       _val(ap, "Skill t-statistic (IR x sqrt years)"),
                       _val(ap, "Years needed for t = 2"))
        if _isnum(ir) and _isnum(ts):
            t = f"Information ratio {ir:.2f}, a skill t-statistic of {ts:.1f} (about 2 is needed)"
            if _isnum(yrs):
                t += f"; at this information ratio about {yrs:.0f} years of track record would be required"
            out.append(t + ".")
        r2 = _val(ap, "R² vs benchmark")
        if _isnum(r2) and r2 < 0.70:
            out.append(f"R² against the benchmark is {r2:.2f}, below 0.70: alpha, beta, Treynor "
                       f"and the appraisal ratio are not quoted, as the regression explains too little.")
        dn, up = _val(ap, "Down capture"), _val(ap, "Up capture")
        if _isnum(dn) and _isnum(up):
            out.append(f"Captured {up * 100:.0f}% of the benchmark's rising periods and "
                       f"{dn * 100:.0f}% of its falling ones.")
        n = ap[ap["Measure"] == "Sample length"]
        if len(n):
            out.append(f"Short track record: {n['Note'].iloc[0]}.")
    return out


# ----------------------------------------------------------------------
def write_analysis(wb, d: Dict[str, Any]) -> None:
    """Adds the sheet to `wb`. `d` keys: label, bench_label, title,
    subtitle, battr (attribution.benchmark_relative result or None),
    proxies (category -> proxy label), appraisal (cipm.appraisal_compare),
    contributions (attribution.summary), standard (optional extra text),
    footer."""
    ba = d.get("battr")
    name = SHEET_ATTR if ba else SHEET_PERF
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws.sheet_format.defaultRowHeight = 12
    ws.sheet_format.customHeight = True
    for c, w in WIDTHS.items():
        ws.column_dimensions[get_column_letter(c)].width = w
    p = _Page(ws)
    chart_rows: set = set()
    label = d.get("label", "Strategy")
    bl = d.get("bench_label")

    # --- Masthead ----------------------------------------------------------
    ws.row_dimensions[1].height = 6
    ws.merge_cells("B2:G2")
    p.cell(2, 2, d.get("title", label), font=_f(16, True))
    ws.merge_cells("H2:J2")
    p.cell(2, 8, "ATTRIBUTION & PERFORMANCE ANALYSIS" if ba else "PERFORMANCE ANALYSIS",
           font=_f(8.5, True, BRAND_RED), align=Alignment(horizontal="right", vertical="center"))
    ws.row_dimensions[2].height = 24
    ws.merge_cells("B3:J3")
    p.cell(3, 2, d.get("subtitle", ""), font=_f(8.5, color=MUTED))
    p.rule(3, FIRST, LAST, BRAND_RED, "medium")
    r = 5

    # --- Observations ------------------------------------------------------
    obs = observations(d)
    if obs:
        p.heading(r, FIRST, LAST, "What the figures show")
        r += 1
        for o in obs:
            txt = f"•  {o}"
            p.para(r, FIRST, LAST, txt, height=12 * _lines_needed(txt, 130))
            r += 1
        r += 1

    # --- Attribution -------------------------------------------------------
    if ba:
        lk, per, yy = ba["linked"], ba["periods"], ba.get("by_year")
        tot = lk.loc["Total"]
        costs = lk.loc["Costs & fees", "Total effect"] if "Costs & fees" in lk.index else 0.0
        p.heading(r, FIRST, LAST, "Benchmark-relative attribution (Brinson-Fachler, linked)")
        r += 1
        tiles = [("PORTFOLIO", ba["R"], PCT, False), ("BENCHMARK", ba["B"], PCT, False),
                 ("EXCESS", ba["excess"], PCT, True), ("ALLOCATION", tot["Allocation"], PCT, True),
                 ("SELECTION", tot["Selection"], PCT, True), ("COSTS & FEES", costs, PCT, True),
                 ("PERIODS", float(len(per)), "0", False),
                 ("RECONCILES", abs(ba["reconciliation_error"]) < 1e-6, None, False)]
        for (k, v, fmt, tone), c in zip(tiles, range(FIRST, FIRST + len(tiles))):
            p.cell(r, c, k, font=_f(7, True, MUTED), align=Alignment(horizontal="center"), fill=SAND)
            if fmt is None:
                p.cell(r + 1, c, "yes" if v else "NO", font=_f(14, True, GAIN if v else LOSS),
                       align=Alignment(horizontal="center", vertical="center"), fill=SAND)
            else:
                col = (GAIN if v > 0 else LOSS if v < 0 else INK) if tone and _isnum(v) else INK
                p.cell(r + 1, c, v, fmt=fmt, font=_f(14, True, col),
                       align=Alignment(horizontal="center", vertical="center"), fill=SAND)
            sub = {"PORTFOLIO": "linked, net", "BENCHMARK": "weights x proxies",
                   "EXCESS": "arithmetic", "ALLOCATION": "BF", "SELECTION": "incl. interaction",
                   "COSTS & FEES": "own line", "PERIODS": ba["period"].lower() + "ly" if
                   ba["period"] == "Month" else "quarterly",
                   "RECONCILES": "to the excess"}[k]
            p.cell(r + 2, c, sub, font=_f(7, color=MUTED), align=Alignment(horizontal="center"), fill=SAND)
        ws.row_dimensions[r + 1].height = 22
        r += 4

        # Category table
        prox = d.get("proxies") or {}
        header = ["Category", "Avg weight", "Bench. weight", "Allocation", "Selection",
                  "Total effect", "Benchmark return proxy"]
        ws.merge_cells(start_row=r, start_column=FIRST + 6, end_row=r, end_column=LAST)
        for j, h in enumerate(header):
            p.cell(r, FIRST + j, h, font=_f(8, True, MUTED),
                   align=Alignment(horizontal="left" if j in (0, 6) else "right", vertical="center"))
        p.rule(r, FIRST, LAST)
        r += 1
        for k, (cat, row) in enumerate(lk.iterrows()):
            fill = SAND if k % 2 == 1 else None
            bold = cat == "Total"
            f_ = _f(9, bold)
            p.cell(r, FIRST, cat, font=f_, fill=fill)
            _aw, _bw = row.get("Avg portfolio weight"), row.get("Benchmark weight")
            p.cell(r, FIRST + 1, _aw if _isnum(_aw) else "", fmt=PCT, font=f_, fill=fill)
            p.cell(r, FIRST + 2, _bw if _isnum(_bw) else "", fmt=PCT, font=f_, fill=fill)
            p.cell(r, FIRST + 3, row["Allocation"], fmt=PCT2, font=f_, fill=fill, signed=True)
            p.cell(r, FIRST + 4, row["Selection"], fmt=PCT2, font=f_, fill=fill, signed=True)
            p.cell(r, FIRST + 5, row["Total effect"], fmt=PCT2, font=f_, fill=fill, signed=True)
            ws.merge_cells(start_row=r, start_column=FIRST + 6, end_row=r, end_column=LAST)
            pr = prox.get(cat, "" if cat in ("Total", "Costs & fees") else "")
            p.cell(r, FIRST + 6, pr, font=_f(8.5, color=MUTED), fill=fill,
                   align=Alignment(horizontal="left", vertical="center"))
            r += 1
        _real = [c for c in lk.index if c not in ("Total", "Costs & fees", "Cash balance")]
        if len(_real) <= 1:
            _t = ("All holdings sit in one asset class, so allocation can only show the shift between "
                  "that class and cash. Assign each instrument its asset class on the Attribution tab "
                  "to see allocation across classes.")
            p.para(r, FIRST, LAST, _t, font=_f(8, italic=True, color=MUTED), height=22)
            r += 1
        r += 1

        # Each calendar year
        if yy is not None and not yy.empty:
            p.heading(r, FIRST, LAST, "Attribution by calendar year (each year linked on its own)")
            r += 1
            rows = []
            for _, x in yy.iterrows():
                partial = (ba["period"] == "Month" and x["Periods"] < 12) or \
                          (ba["period"] == "Quarter" and x["Periods"] < 4)
                rows.append([f"{int(x['Year'])}" + (" (partial)" if partial else ""),
                             x["Portfolio"], x["Benchmark"], x["Excess"], x["Allocation"],
                             x["Selection"], x["Costs & fees"], round(float(x["Check"]), 10) + 0.0])
            r = p.table(r, FIRST, ["Year", "Portfolio", "Benchmark", "Excess", "Allocation",
                                   "Selection", "Costs & fees", "Check"], rows,
                        [None, PCT2, PCT2, PCT2, PCT2, PCT2, PCT2, "0.0000%"],
                        [False, True, True, True, True, True, True, False])
            r += 1

            # Charts: effects by category and by year, from a hidden sheet.
            if _DATA in wb.sheetnames:
                del wb[_DATA]
            ds = wb.create_sheet(_DATA)
            ds.sheet_state = "hidden"
            cats = lk.drop(index=[i for i in ("Total", "Costs & fees") if i in lk.index])
            ds.append(["Category", "Allocation", "Selection"])
            for cat, row in cats.iterrows():
                ds.append([str(cat), float(row["Allocation"]), float(row["Selection"])])
            n1 = len(cats) + 1
            ds.cell(row=1, column=6, value="Year")
            for j, h in enumerate(["Allocation", "Selection", "Costs & fees"], start=7):
                ds.cell(row=1, column=j, value=h)
            for i, (_, x) in enumerate(yy.iterrows(), start=2):
                ds.cell(row=i, column=6, value=str(int(x["Year"])))
                ds.cell(row=i, column=7, value=float(x["Allocation"]))
                ds.cell(row=i, column=8, value=float(x["Selection"]))
                ds.cell(row=i, column=9, value=float(x["Costs & fees"]))
            n2 = len(yy) + 1
            p.heading(r, FIRST, LAST, "Where the excess return came from")
            r += 1
            for anchor_col, title, cmin, cmax, cat_col, nrows_, grouping in (
                    ("B", "Linked effect by category", 2, 3, 1, n1, "clustered"),
                    ("F", "Effect by calendar year", 7, 9, 6, n2, "stacked")):
                ch = BarChart()
                ch.type = "col"
                ch.grouping = grouping
                if grouping == "stacked":
                    ch.overlap = 100
                ch.title = None
                ch.height, ch.width = 6.6, 12.8 if anchor_col == "B" else 12.2
                ch.legend.position = "t"
                ch.y_axis.number_format = "0%"
                ch.y_axis.majorGridlines.spPr = None
                ch.x_axis.delete = False
                ch.y_axis.delete = False
                ch.add_data(Reference(ds, min_col=cmin, max_col=cmax, min_row=1, max_row=nrows_),
                            titles_from_data=True)
                ch.set_categories(Reference(ds, min_col=cat_col, min_row=2, max_row=nrows_))
                for s, col in zip(ch.series, [BRAND_RED, "3F3A34", "B8AE9C"]):
                    s.graphicalProperties.solidFill = col
                    s.graphicalProperties.line.solidFill = col
                ws.add_chart(ch, f"{anchor_col}{r}")
            chart_rows |= set(range(r, r + 15))
            r += 15

        # Methodology of the benchmark
        bm = ba.get("methodology", "")
        bw_txt = "; ".join(f"{c} {w * 100:.1f}%" for c, w in (d.get("bench_weights") or {}).items())
        txt = (bm + (f". Benchmark weights: {bw_txt}." if bw_txt else "."))
        p.para(r, FIRST, LAST, txt, font=_f(8, italic=True, color=MUTED),
               height=11 * _lines_needed(txt, 140))
        r += 2

    # --- Contribution ------------------------------------------------------
    con = d.get("contributions")
    if con is not None and not con.empty:
        p.heading(r, FIRST, LAST, "Contribution to return and risk")
        r += 1
        c2 = con.copy()
        c2["_abs"] = c2["Contribution"].abs()
        c2 = c2.sort_values("_abs", ascending=False).head(12).drop(columns="_abs")
        rows = [[str(x["Name"])[:34], _blank(x["Average weight"]), _blank(x["Time held"]),
                 x["Contribution"], _na(x["Share of result"]), _na(x["Share of risk"])]
                for _, x in c2.iterrows()]
        r = p.table(r, FIRST, ["", "Avg weight", "Time held", "Contribution", "Share of result",
                               "Share of risk"], rows, [None, PCT, PCT, PCT2, PCT, PCT],
                    [False, False, False, True, False, False])
        p.para(r, FIRST, LAST, "Contributions are linked in currency, so they add up to the portfolio's "
               "return exactly. Share of risk is each line's share of the variance of returns; a "
               "negative share means the holding reduced volatility.",
               font=_f(8, italic=True, color=MUTED), height=22)
        r += 2

    # --- Appraisal ---------------------------------------------------------
    ap = d.get("appraisal")
    if ap is not None and not ap.empty:
        p.heading(r, FIRST, LAST, "Performance appraisal (CIPM)")
        r += 1
        has_b = "Benchmark" in ap
        for j, h in enumerate(["", "Strategy", "Benchmark" if has_b else ""]):
            p.cell(r, FIRST + j, h, font=_f(8, True, MUTED),
                   align=Alignment(horizontal="left" if j == 0 else "right"))
        p.cell(r, FIRST + 3, "Comment", font=_f(8, True, MUTED))
        p.rule(r, FIRST, LAST)
        r += 1
        fm = list(ap.attrs.get("row_fmt") or [None] * len(ap))
        k_row = 0
        last_group = None
        for i, (_, x) in enumerate(ap.iterrows()):
            if x["Group"] != last_group:
                last_group = x["Group"]
                p.cell(r, FIRST, str(last_group).upper(), font=_f(7.5, True, BRAND_RED))
                r += 1
                k_row = 0
            fill = SAND if k_row % 2 == 1 else None
            p.cell(r, FIRST, x["Measure"], fill=fill)
            p.cell(r, FIRST + 1, _na(x["Strategy"]), fmt=fm[i], fill=fill,
                   signed=("ratio" in x["Measure"].lower() or x["Measure"] in (
                       "Sharpe", "Sortino", "Burke", "Treynor", "Jensen alpha (annualised)",
                       "Calmar (rf-adjusted)")))
            p.cell(r, FIRST + 2, _blank(x["Benchmark"]) if has_b else None, fmt=fm[i], fill=fill)
            ws.merge_cells(start_row=r, start_column=FIRST + 3, end_row=r, end_column=LAST)
            p.cell(r, FIRST + 3, x["Note"] if isinstance(x["Note"], str) else "",
                   font=_f(8.5, color=MUTED), fill=fill,
                   align=Alignment(horizontal="left", vertical="center"))
            r += 1
            k_row += 1
        r += 1
        p.para(r, FIRST, LAST,
               "Withheld measures read n/a with the reason beside them. Benchmark values are shown "
               "where the measure exists for the benchmark on its own; relative measures (tracking error, "
               "information ratio, M², CAPM, capture) compare the strategy with it.",
               font=_f(8, italic=True, color=MUTED), height=22)
        r += 2

    # --- Methodology -------------------------------------------------------
    p.heading(r, FIRST, LAST, "Methodology")
    r += 1
    txt = d.get("method_line") or ""
    if txt:
        p.para(r, FIRST, LAST, txt, font=_f(8.5), height=12 * _lines_needed(txt, 135))
        r += 1
    p.para(r, FIRST, LAST, STANDARD, font=_f(8, italic=True, color=MUTED),
           height=11 * _lines_needed(STANDARD, 140))
    r += 1
    last = r + 1

    for rr in range(4, last):
        if rr in chart_rows or ws.row_dimensions[rr].height:
            continue
        if all(ws.cell(row=rr, column=c).value in (None, "") for c in range(1, 12)):
            ws.row_dimensions[rr].height = 5

    # --- Print: portrait, one page wide ---------------------------------------
    ws.print_area = f"A1:K{last}"
    ws.page_setup.orientation = "portrait"
    ws.page_setup.paperSize = ws.PAPERSIZE_LETTER
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.page_margins = PageMargins(left=0.35, right=0.35, top=0.4, bottom=0.45)
    ws.print_options.horizontalCentered = True
    ws.oddFooter.left.text = "&8" + d.get("footer", "")
    ws.oddFooter.right.text = "&8Page &P of &N"


# ----------------------------------------------------------------------
def build_analysis(res, bench, rf, ppy: int, *, title: str, subtitle: str = "",
                   footer: str = "", rf_label: str = "model risk-free",
                   attribution: Dict[str, Any] | None = None,
                   returns_only: bool = False, label: str | None = None,
                   bench_label: str | None = None) -> Dict[str, Any]:
    """Everything `write_analysis` and the workbook builders need.

    `attribution`: {classes, bench_weights, proxies, asset_returns,
    period, linking} as chosen on the Attribution tab (None: no
    attribution section). Failures of the attribution (a category with no
    benchmark return, weights that do not sum to 100%) are returned as
    `battr_error` rather than raised, so the rest of the workbook is built.
    """
    from . import attribution as ATTR

    returns = res.returns if hasattr(res, "returns") else res
    bret = bench.returns if (bench is not None and hasattr(bench, "returns")) else bench
    d: Dict[str, Any] = {
        "title": title, "subtitle": subtitle, "footer": footer,
        "label": label or getattr(res, "label", "Strategy"),
        "bench_label": bench_label or (getattr(bench, "label", None) if bench is not None else None),
        "appraisal": CIPM.appraisal_compare(returns, bret, rf, ppy),
        "method_line": CIPM.methodology_block(
            ppy, rf=rf_label,
            benchmark=(bench_label or getattr(bench, "label", "benchmark")) if bench is not None else "none",
            n_obs=len(returns)),
    }
    if returns_only or attribution is None:
        return d
    if getattr(res, "contributions", None) is not None:
        cls = attribution.get("classes")
        # One asset class would make the table a single row: by security then.
        d["contributions"] = ATTR.summary(
            res, cls if cls and len(set(cls.values())) > 1 else None, ppy)
        try:
            ba = ATTR.benchmark_relative(
                res, attribution["classes"], attribution["bench_weights"],
                attribution["asset_returns"], rf, attribution.get("proxies"),
                attribution.get("period", "Month"), attribution.get("linking", "frongello"))
            d["battr"] = ba
            d["bench_weights"] = {k: v for k, v in attribution["bench_weights"].items() if v}
            prox = {c: attribution.get("proxies", {}).get(c, ATTR.EQUAL_WEIGHT)
                    for c in ba["linked"].index if c not in ("Total", ATTR.COSTS_FEES)}
            prox[ATTR.CASH_CLASS] = "Cash rate (risk-free series)"
            d["proxies"] = prox
        except ValueError as e:
            d["battr_error"] = str(e)
    return d
