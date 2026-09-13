"""The detailed report.

One document covering every module in the app, built from whatever has
already been computed rather than re-running the expensive parts. Results
and Positions come straight from the backtest already on screen. Signals,
Stress test periods and Tax are cheap enough to compute fresh every time
this is generated, so they always appear. Walk-forward, in/out-of-sample
and cost sensitivity are also cheap and always included. The parameter
surface, the trading-day sweep and Monte Carlo are each expensive enough
that the app only runs them on request -- this report includes whichever
of those the person has already run in this session, and says plainly
when one has not been, rather than silently omitting it or forcing an
expensive recompute just to build a document.

Every chart is a static PNG from `qbt/report_charts.py`, embedded as base64
in the HTML and as an Image flowable in the PDF -- the same bytes in both,
so the two documents cannot show different pictures of the same result.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from . import report_charts as RC
from .brand import BRAND

CSS_DARK = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;600;700&display=swap');
*{box-sizing:border-box;}
body{margin:0;padding:2.4rem 3rem 4rem;background:%(bg)s;color:%(text)s;
  font-family:'Inter',Arial,sans-serif;font-size:13px;line-height:1.6;
  max-width:1100px;margin-left:auto;margin-right:auto;
  font-variant-numeric:tabular-nums;}
h1,h2,h3{font-weight:700;letter-spacing:-.01em;}
.masthead{border-bottom:2px solid %(red)s;padding-bottom:1rem;margin-bottom:1.6rem;
  display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:.6rem;}
.masthead h1{font-size:1.7rem;margin:0;}
.masthead .meta{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:%(muted)s;
  text-align:right;line-height:1.7;}
.section{margin-top:2.2rem;padding-top:1.1rem;border-top:1px solid %(rule)s;}
.section:first-of-type{border-top:none;margin-top:0;}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.14em;
  text-transform:uppercase;color:%(muted)s;font-weight:600;margin-bottom:.7rem;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.6rem;
  margin-bottom:1rem;}
.kpi{background:%(panel)s;border:1px solid %(rule)s;border-top:2px solid %(red)s;
  padding:.6rem .7rem;border-radius:2px;}
.kpi .k{font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.08em;
  text-transform:uppercase;color:%(muted)s;display:block;margin-bottom:.28rem;}
.kpi .v{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:600;}
.kpi .v.pos{color:%(gain)s;} .kpi .v.neg{color:%(loss)s;}
table{border-collapse:collapse;width:100%%;font-size:.78rem;margin-bottom:1rem;}
th,td{border-bottom:1px solid %(rule)s;padding:.34rem .55rem;text-align:right;
  font-family:'IBM Plex Mono',monospace;}
th:first-child,td:first-child{text-align:left;}
th{font-size:.6rem;letter-spacing:.06em;text-transform:uppercase;color:%(muted)s;
  background:%(panel)s;}
.note{color:%(muted)s;font-size:.79rem;margin:.4rem 0 1rem;}
.flag{background:rgba(228,28,35,.10);border-left:2px solid %(red)s;border-radius:0 2px 2px 0;
  padding:.5rem .8rem;color:%(text)s;font-size:.79rem;margin:.5rem 0 1rem;}
.chart{border:1px solid %(rule)s;border-radius:3px;overflow:hidden;margin-bottom:1rem;
  background:%(panel)s;padding:.3rem;}
.chart img{display:block;width:100%%;height:auto;}
.pos{color:%(gain)s;} .neg{color:%(loss)s;}
.footer{margin-top:2.4rem;padding-top:1rem;border-top:1px solid %(rule)s;
  font-size:.72rem;color:%(muted)s;}
@media print{ body{background:#fff;color:#000;} }
"""


def _img_tag(png: bytes) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return f'<div class="chart"><img src="data:image/png;base64,{b64}"/></div>'


def _kpi(key: str, value: str, tone: str = "") -> str:
    return (f'<div class="kpi"><span class="k">{key}</span>'
            f'<span class="v {tone}">{value}</span></div>')


def _df_to_html(df: pd.DataFrame, pct_cols: Optional[List[str]] = None,
                money_cols: Optional[List[str]] = None) -> str:
    if df is None or df.empty:
        return '<p class="note">No data.</p>'
    d = df.copy()
    for c in (pct_cols or []):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "\u2014" if pd.isna(v) else f"{v*100:+.2f}%")
    for c in (money_cols or []):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "\u2014" if pd.isna(v) else f"${v:,.0f}")
    return d.to_html(index=False, border=0, escape=True, na_rep="\u2014")


def render_full_report(bundle: Dict[str, Any], dark: bool = True) -> str:
    """Assembles the complete HTML document from a bundle dict.

    Expected keys (all optional except `label`/`equity`): label, equity,
    returns, trades, benchmark_label, benchmark_equity, drawdown_table,
    trailing_table, calendar_table, scores (DataFrame), walk_forward,
    in_out_sample, cost_sensitivity, parameter_sweep (df, x, y, z),
    day_sweep (df), monte_carlo (dict), stress (evaluate_periods df),
    tax_report (TaxReport), tax_benchmark_report, engine_summary (dict of
    config strings).
    """
    pal = {
        "bg": BRAND["bg"] if dark else "#FFFFFF",
        "panel": BRAND["panel"] if dark else "#FAFAF8",
        "rule": BRAND["rule"] if dark else "#DDDAD2",
        "text": BRAND["text"] if dark else "#1A1A1A",
        "muted": BRAND["muted"] if dark else "#6B655C",
        "red": BRAND["red"], "gain": BRAND["gain"], "loss": BRAND["loss"],
    }
    css = CSS_DARK % pal
    parts: List[str] = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                        f"<title>{bundle.get('label','Backtest')} \u2014 Detailed Report</title>"
                        f"<style>{css}</style></head><body>"]

    label = bundle.get("label", "Backtest")
    equity: pd.Series = bundle.get("equity")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts.append(
        f'<div class="masthead"><div><h1>{label}</h1>'
        f'<div class="note" style="margin:0;">Detailed report \u2014 every module '
        f'in one document</div></div>'
        f'<div class="meta">Generated {stamp}<br>'
        f'{bundle.get("period","")}<br>'
        f'{bundle.get("engine_line","")}</div></div>')

    # ---------------- Results ----------------
    if equity is not None and len(equity):
        parts.append('<div class="section"><div class="eyebrow">Results</div>')
        st = bundle.get("stats", {})
        cols = ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar", "Sortino"]
        tones = {"CAGR": "pos", "Sharpe": "pos", "Calmar": "pos", "Sortino": "pos",
                "Max Drawdown": "neg"}
        kpis = "".join(_kpi(c, M.format_metric(c, st.get(c, float("nan"))), tones.get(c, ""))
                       for c in cols if c in st)
        parts.append(f'<div class="kpi-grid">{kpis}</div>')

        curves = {label: equity}
        if bundle.get("benchmark_equity") is not None:
            curves[bundle.get("benchmark_label", "Benchmark")] = bundle["benchmark_equity"]
        parts.append(_img_tag(RC.equity_chart(curves, "Equity curve", dark=dark)))
        parts.append(_img_tag(RC.drawdown_chart(equity, "Drawdown", dark=dark)))

        if bundle.get("trailing_table") is not None:
            parts.append('<div class="eyebrow" style="margin-top:1rem;">Trailing periods</div>')
            parts.append(_df_to_html(bundle["trailing_table"]))
        if bundle.get("calendar_table") is not None:
            parts.append('<div class="eyebrow">Calendar years</div>')
            parts.append(_df_to_html(bundle["calendar_table"]))
        parts.append("</div>")

    # ---------------- Signals ----------------
    scores = bundle.get("scores")
    if scores is not None and not scores.empty:
        parts.append('<div class="section"><div class="eyebrow">Signals</div>')
        parts.append('<p class="note">The score the model ranks on, through time. '
                     'Score decides ranking; separate filters decide eligibility, so a '
                     'top-scored name can still sit unheld.</p>')
        parts.append(_img_tag(RC.score_chart(scores, "Score by instrument", dark=dark)))
        parts.append("</div>")

    # ---------------- Positions ----------------
    trades: pd.DataFrame = bundle.get("trades")
    if trades is not None and not trades.empty:
        parts.append('<div class="section"><div class="eyebrow">Positions</div>')
        n_liq = int((trades.get("Reason", pd.Series(dtype=str)) == "Fee liquidation").sum())
        parts.append(f'<p class="note">{len(trades):,} fills'
                     + (f", {n_liq} of them fee liquidations" if n_liq else "")
                     + ".</p>")
        recent = trades.tail(30).copy()
        if "Date" in recent.columns:
            recent["Date"] = pd.to_datetime(recent["Date"]).dt.date
        parts.append(_df_to_html(recent))
        parts.append("</div>")

    # ---------------- Robustness ----------------
    parts.append('<div class="section"><div class="eyebrow">Robustness</div>')
    wf = bundle.get("walk_forward")
    if wf is not None and not wf.empty:
        parts.append('<h3 style="font-size:.85rem;margin:.8rem 0 .3rem;">Successive folds</h3>')
        parts.append(_df_to_html(wf))
    ios = bundle.get("in_out_sample")
    if ios is not None and not ios.empty:
        parts.append('<h3 style="font-size:.85rem;margin:.8rem 0 .3rem;">In-sample / out-of-sample</h3>')
        parts.append(_df_to_html(ios))
    cs = bundle.get("cost_sensitivity")
    if cs is not None and not cs.empty and "CAGR" in cs.columns:
        parts.append('<h3 style="font-size:.85rem;margin:.8rem 0 .3rem;">Cost sensitivity</h3>')
        xcol = cs.columns[0]
        parts.append(_img_tag(RC.bar_chart(cs[xcol].astype(str), (cs["CAGR"] * 100).tolist(),
                                           "CAGR by cost level", "%", dark=dark)))
    psw = bundle.get("parameter_sweep")
    if psw and psw.get("df") is not None and not psw["df"].empty:
        df_, x_, y_, z_ = psw["df"], psw["x"], psw["y"], psw["z"]
        parts.append(f'<h3 style="font-size:.85rem;margin:.8rem 0 .3rem;">Parameter surface ({z_})</h3>')
        if y_ and y_ in df_.columns:
            piv = df_.pivot_table(index=y_, columns=x_, values=z_, aggfunc="mean")
            parts.append(_img_tag(RC.heatmap_chart(piv, f"{z_} by {x_} and {y_}", dark=dark)))
        elif z_ in df_.columns:
            d = df_.groupby(x_)[z_].mean().sort_index()
            parts.append(_img_tag(RC.bar_chart(d.index.astype(str), d.values,
                                               f"{z_} by {x_}", dark=dark)))
    else:
        parts.append('<p class="note">Parameter surface: not yet computed for this run '
                     '(run it from the Robustness tab to include it here).</p>')
    dsw = bundle.get("day_sweep")
    if dsw is not None and not dsw.empty and "CAGR" in dsw.columns:
        parts.append('<h3 style="font-size:.85rem;margin:.8rem 0 .3rem;">Trading-day sensitivity</h3>')
        d = dsw.dropna(subset=["CAGR"]).sort_values("CAGR", ascending=False)
        parts.append(_img_tag(RC.bar_chart(d["Trading day"], (d["CAGR"] * 100).tolist(),
                                           "CAGR by trading day", "%", dark=dark)))
    else:
        parts.append('<p class="note">Trading-day sensitivity: not yet computed for this run.</p>')
    mc = bundle.get("monte_carlo")
    if mc is not None:
        parts.append('<h3 style="font-size:.85rem;margin:.8rem 0 .3rem;">Block-resampled Monte Carlo</h3>')
        parts.append(f'<p class="note">Median simulated CAGR {mc.get("median_cagr", float("nan"))*100:.2f}%, '
                     f'5th-95th percentile [{mc.get("p5", float("nan"))*100:.2f}%, '
                     f'{mc.get("p95", float("nan"))*100:.2f}%].</p>')
    else:
        parts.append('<p class="note">Monte Carlo: not yet computed for this run.</p>')

    stress = bundle.get("stress")
    if stress is not None and not stress.empty:
        parts.append('<h3 style="font-size:.85rem;margin:.8rem 0 .3rem;">Stress test periods</h3>')
        shown = stress[stress["Coverage"] != "No data"].copy()
        if not shown.empty:
            order = shown.sort_values("Start")["Period"]
            parts.append(_img_tag(RC.bar_chart(
                order, (shown.set_index("Period").loc[order, "Return"] * 100).tolist(),
                "Return by historical episode", "%", dark=dark)))
        parts.append(_df_to_html(stress, pct_cols=["Return", "Max Drawdown", "Best Day",
                                                   "Worst Day", "Benchmark Return",
                                                   "Excess vs Benchmark"]))
    parts.append("</div>")

    # ---------------- Tax ----------------
    tax_rep = bundle.get("tax_report")
    if tax_rep is not None and not tax_rep.by_year.empty:
        parts.append('<div class="section"><div class="eyebrow">Tax friendliness</div>')
        from . import tax as TAX
        eff_label, tone = TAX.efficiency_label(tax_rep.tax_cost_ratio)
        kpis = "".join([
            _kpi("Total tax", f"${tax_rep.total_tax:,.0f}", "neg"),
            _kpi("Tax Cost Ratio", f"{tax_rep.tax_cost_ratio*100:.2f}%/yr", tone),
            _kpi("Realized gains", f"${tax_rep.total_pretax_gain:,.0f}"),
        ])
        parts.append(f'<div class="kpi-grid">{kpis}</div>')
        parts.append(f'<p class="note">{eff_label}, on Morningstar\'s Tax Cost Ratio scale.</p>')

        curves = {label: equity, f"{label} (after tax)": tax_rep.after_tax_equity}
        bench_tax = bundle.get("tax_benchmark_report")
        if bench_tax is not None and bundle.get("benchmark_equity") is not None:
            curves[bundle.get("benchmark_label", "Benchmark")] = bundle["benchmark_equity"]
            curves[f'{bundle.get("benchmark_label", "Benchmark")} (after tax)'] = bench_tax.after_tax_equity
        parts.append(_img_tag(RC.equity_chart(curves, "Pretax vs. after-tax", dark=dark)))

        comp = tax_rep.by_year[["Year", "Capital Gains Tax", "Eligible Dividend Tax",
                                "Foreign Dividend Tax"]]
        if comp.drop(columns="Year").to_numpy().sum() > 0:
            parts.append(_img_tag(RC.grouped_bar_chart(
                comp["Year"].astype(str),
                {"Capital gains": comp["Capital Gains Tax"].tolist(),
                 "Eligible dividends": comp["Eligible Dividend Tax"].tolist(),
                 "Foreign dividends": comp["Foreign Dividend Tax"].tolist()},
                "Tax by source and year", "$", dark=dark)))

        parts.append(_df_to_html(tax_rep.by_year, money_cols=[
            c for c in tax_rep.by_year.columns if c != "Year"]))
        for w in tax_rep.warnings:
            parts.append(f'<div class="flag">{w}</div>')
        parts.append("</div>")
    else:
        parts.append('<div class="section"><div class="eyebrow">Tax friendliness</div>'
                     '<p class="note">Not available for this run -- needs the Price '
                     'return + cash dividends convention and at least one calendar '
                     'year of data.</p></div>')

    parts.append(
        f'<div class="footer">Quant Backtest Studio \u2014 detailed report \u2014 '
        f'generated {stamp}. Every figure above comes from the backtest already run; '
        f'this document adds no new calculation of its own.</div>')
    parts.append("</body></html>")
    return "".join(parts)


# ----------------------------------------------------------------------
# PDF, via reportlab rather than converting the HTML through a browser
# engine. Weasyprint/wkhtmltopdf need system libraries (cairo, pango) that
# are a fragile, slow addition to a free-tier deploy's build step;
# reportlab is pure Python, already a dependency, and is what this
# environment's own PDF guidance recommends for generating a PDF from
# scratch. The same PNG chart bytes from `qbt/report_charts.py` are
# embedded directly as Image flowables, so the PDF shows the identical
# pictures the HTML does.
# ----------------------------------------------------------------------
def render_full_report_pdf(bundle: Dict[str, Any]) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image as RLImage, PageBreak,
                                    KeepTogether)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT

    bg = rl_colors.HexColor(BRAND["bg"])
    panel = rl_colors.HexColor(BRAND["panel"])
    rule = rl_colors.HexColor(BRAND["rule"])
    text_c = rl_colors.HexColor(BRAND["text"])
    muted = rl_colors.HexColor(BRAND["muted"])
    red = rl_colors.HexColor(BRAND["red"])
    gain = rl_colors.HexColor(BRAND["gain"])
    loss = rl_colors.HexColor(BRAND["loss"])

    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18,
                                textColor=text_c, spaceAfter=4),
        "meta": ParagraphStyle("meta", fontName="Courier", fontSize=7.5,
                               textColor=muted, alignment=TA_RIGHT, leading=11),
        "eyebrow": ParagraphStyle("eyebrow", fontName="Helvetica-Bold", fontSize=9,
                                  textColor=muted, spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=9.5,
                             textColor=text_c, spaceBefore=8, spaceAfter=4),
        "note": ParagraphStyle("note", fontName="Helvetica", fontSize=8,
                               textColor=muted, leading=11, spaceAfter=6),
        "flag": ParagraphStyle("flag", fontName="Helvetica", fontSize=8,
                               textColor=text_c, leading=11, spaceAfter=6,
                               leftIndent=6, borderColor=red, borderWidth=0,
                               backColor=rl_colors.HexColor("#2A1416")),
        "footer": ParagraphStyle("footer", fontName="Helvetica", fontSize=7,
                                 textColor=muted, leading=10),
    }

    def _paint_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(bg)
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        canvas.restoreState()

    def _kpi_table(items):
        """A small dark card grid, built as a single-row table per item
        rather than one wide table, so items wrap onto a new line rather
        than being squeezed illegibly narrow."""
        cells = []
        for k, v, tone in items:
            color = gain if tone == "pos" else (loss if tone == "neg" else text_c)
            cells.append([Paragraph(f'<font size=6 color="#{BRAND["muted"][1:]}">{k.upper()}'
                                    f'</font><br/><font size=13 color="#{tone_hex(color)}">'
                                    f'{v}</font>', styles["note"])])
        t = Table([cells], colWidths=[1.55 * inch] * len(cells))
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), panel),
            ("BOX", (0, 0), (-1, -1), 0.5, rule),
            ("LINEABOVE", (0, 0), (-1, 0), 1.5, red),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    def tone_hex(color) -> str:
        return color.hexval()[2:] if hasattr(color, "hexval") else "EFE9DE"

    def _df_table(df: pd.DataFrame, pct_cols=None, money_cols=None, max_rows=40):
        if df is None or df.empty:
            return Paragraph("No data.", styles["note"])
        d = df.tail(max_rows).copy()
        for c in (pct_cols or []):
            if c in d.columns:
                d[c] = d[c].map(lambda v: "\u2014" if pd.isna(v) else f"{v*100:+.2f}%")
        for c in (money_cols or []):
            if c in d.columns:
                d[c] = d[c].map(lambda v: "\u2014" if pd.isna(v) else f"${v:,.0f}")
        d = d.astype(str)
        data = [list(d.columns)] + d.values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), panel),
            ("TEXTCOLOR", (0, 0), (-1, 0), muted),
            ("TEXTCOLOR", (0, 1), (-1, -1), text_c),
            ("FONTNAME", (0, 0), (-1, -1), "Courier"),
            ("FONTSIZE", (0, 0), (-1, -1), 6.5),
            ("GRID", (0, 0), (-1, -1), 0.4, rule),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    story: List[Any] = []
    label = bundle.get("label", "Backtest")
    equity = bundle.get("equity")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    story.append(Paragraph(label, styles["title"]))
    story.append(Paragraph(f"Detailed report \u2014 generated {stamp}<br/>"
                           f'{bundle.get("period","")}<br/>{bundle.get("engine_line","")}',
                           styles["meta"]))
    story.append(Spacer(1, 10))

    def add_png(png: bytes, width=6.6 * inch):
        img = RLImage(io.BytesIO(png))
        ratio = img.imageHeight / float(img.imageWidth)
        img.drawWidth = width
        img.drawHeight = width * ratio
        story.append(img)
        story.append(Spacer(1, 8))

    if equity is not None and len(equity):
        story.append(Paragraph("RESULTS", styles["eyebrow"]))
        st = bundle.get("stats", {})
        tones = {"CAGR": "pos", "Sharpe": "pos", "Calmar": "pos", "Sortino": "pos",
                "Max Drawdown": "neg"}
        items = [(c, M.format_metric(c, st.get(c, float("nan"))), tones.get(c, ""))
                for c in ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar", "Sortino"]
                if c in st]
        if items:
            story.append(_kpi_table(items[:4]))
            story.append(Spacer(1, 4))
            if len(items) > 4:
                story.append(_kpi_table(items[4:]))
            story.append(Spacer(1, 8))

        curves = {label: equity}
        if bundle.get("benchmark_equity") is not None:
            curves[bundle.get("benchmark_label", "Benchmark")] = bundle["benchmark_equity"]
        add_png(RC.equity_chart(curves, "Equity curve"))
        add_png(RC.drawdown_chart(equity, "Drawdown"))
        if bundle.get("trailing_table") is not None:
            story.append(Paragraph("Trailing periods", styles["h3"]))
            story.append(_df_table(bundle["trailing_table"]))
            story.append(Spacer(1, 6))
        if bundle.get("calendar_table") is not None:
            story.append(Paragraph("Calendar years", styles["h3"]))
            story.append(_df_table(bundle["calendar_table"]))

    scores = bundle.get("scores")
    if scores is not None and not scores.empty:
        story.append(PageBreak())
        story.append(Paragraph("SIGNALS", styles["eyebrow"]))
        story.append(Paragraph("The score the model ranks on, through time.", styles["note"]))
        add_png(RC.score_chart(scores, "Score by instrument"))

    trades = bundle.get("trades")
    if trades is not None and not trades.empty:
        story.append(Paragraph("POSITIONS", styles["eyebrow"]))
        n_liq = int((trades.get("Reason", pd.Series(dtype=str)) == "Fee liquidation").sum())
        story.append(Paragraph(f"{len(trades):,} fills"
                               + (f", {n_liq} fee liquidations" if n_liq else "") + ".",
                               styles["note"]))
        story.append(_df_table(trades, max_rows=25))

    story.append(PageBreak())
    story.append(Paragraph("ROBUSTNESS", styles["eyebrow"]))
    wf = bundle.get("walk_forward")
    if wf is not None and not wf.empty:
        story.append(Paragraph("Successive folds", styles["h3"]))
        story.append(_df_table(wf))
    ios = bundle.get("in_out_sample")
    if ios is not None and not ios.empty:
        story.append(Paragraph("In-sample / out-of-sample", styles["h3"]))
        story.append(_df_table(ios))
    cs = bundle.get("cost_sensitivity")
    if cs is not None and not cs.empty and "CAGR" in cs.columns:
        story.append(Paragraph("Cost sensitivity", styles["h3"]))
        add_png(RC.bar_chart(cs[cs.columns[0]].astype(str), (cs["CAGR"] * 100).tolist(),
                             "CAGR by cost level", "%"))
    psw = bundle.get("parameter_sweep")
    if psw and psw.get("df") is not None and not psw["df"].empty:
        df_, x_, y_, z_ = psw["df"], psw["x"], psw["y"], psw["z"]
        story.append(Paragraph(f"Parameter surface ({z_})", styles["h3"]))
        if y_ and y_ in df_.columns:
            piv = df_.pivot_table(index=y_, columns=x_, values=z_, aggfunc="mean")
            add_png(RC.heatmap_chart(piv, f"{z_} by {x_} and {y_}"))
        elif z_ in df_.columns:
            d = df_.groupby(x_)[z_].mean().sort_index()
            add_png(RC.bar_chart(d.index.astype(str), d.values, f"{z_} by {x_}"))
    else:
        story.append(Paragraph("Parameter surface: not yet computed for this run.",
                               styles["note"]))
    dsw = bundle.get("day_sweep")
    if dsw is not None and not dsw.empty and "CAGR" in dsw.columns:
        story.append(Paragraph("Trading-day sensitivity", styles["h3"]))
        d = dsw.dropna(subset=["CAGR"]).sort_values("CAGR", ascending=False)
        add_png(RC.bar_chart(d["Trading day"], (d["CAGR"] * 100).tolist(),
                             "CAGR by trading day", "%"))
    else:
        story.append(Paragraph("Trading-day sensitivity: not yet computed for this run.",
                               styles["note"]))
    mc = bundle.get("monte_carlo")
    if mc is not None:
        story.append(Paragraph("Block-resampled Monte Carlo", styles["h3"]))
        story.append(Paragraph(
            f'Median simulated CAGR {mc.get("median_cagr", float("nan"))*100:.2f}%, '
            f'5th-95th percentile [{mc.get("p5", float("nan"))*100:.2f}%, '
            f'{mc.get("p95", float("nan"))*100:.2f}%].', styles["note"]))
    else:
        story.append(Paragraph("Monte Carlo: not yet computed for this run.", styles["note"]))

    stress = bundle.get("stress")
    if stress is not None and not stress.empty:
        story.append(Paragraph("Stress test periods", styles["h3"]))
        shown = stress[stress["Coverage"] != "No data"].copy()
        if not shown.empty:
            order = shown.sort_values("Start")["Period"]
            add_png(RC.bar_chart(order, (shown.set_index("Period").loc[order, "Return"] * 100).tolist(),
                                 "Return by historical episode", "%"))
        story.append(_df_table(stress, pct_cols=["Return", "Max Drawdown", "Best Day",
                                                 "Worst Day", "Benchmark Return",
                                                 "Excess vs Benchmark"]))

    tax_rep = bundle.get("tax_report")
    if tax_rep is not None and not tax_rep.by_year.empty:
        from . import tax as TAX
        story.append(PageBreak())
        story.append(Paragraph("TAX FRIENDLINESS", styles["eyebrow"]))
        eff_label, tone = TAX.efficiency_label(tax_rep.tax_cost_ratio)
        story.append(_kpi_table([
            ("Total tax", f"${tax_rep.total_tax:,.0f}", "neg"),
            ("Tax Cost Ratio", f"{tax_rep.tax_cost_ratio*100:.2f}%/yr", tone),
            ("Realized gains", f"${tax_rep.total_pretax_gain:,.0f}", ""),
        ]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"{eff_label}, on Morningstar's Tax Cost Ratio scale.",
                               styles["note"]))
        curves = {label: equity, f"{label} (after tax)": tax_rep.after_tax_equity}
        bench_tax = bundle.get("tax_benchmark_report")
        if bench_tax is not None and bundle.get("benchmark_equity") is not None:
            curves[bundle.get("benchmark_label", "Benchmark")] = bundle["benchmark_equity"]
            curves[f'{bundle.get("benchmark_label", "Benchmark")} (after tax)'] = bench_tax.after_tax_equity
        add_png(RC.equity_chart(curves, "Pretax vs. after-tax"))
        comp = tax_rep.by_year[["Year", "Capital Gains Tax", "Eligible Dividend Tax",
                                "Foreign Dividend Tax"]]
        if comp.drop(columns="Year").to_numpy().sum() > 0:
            add_png(RC.grouped_bar_chart(
                comp["Year"].astype(str),
                {"Capital gains": comp["Capital Gains Tax"].tolist(),
                 "Eligible dividends": comp["Eligible Dividend Tax"].tolist(),
                 "Foreign dividends": comp["Foreign Dividend Tax"].tolist()},
                "Tax by source and year", "$"))
        story.append(_df_table(tax_rep.by_year, money_cols=[
            c for c in tax_rep.by_year.columns if c != "Year"]))
        for w in tax_rep.warnings:
            story.append(Paragraph(w, styles["flag"]))
    else:
        story.append(Paragraph("TAX FRIENDLINESS", styles["eyebrow"]))
        story.append(Paragraph("Not available for this run.", styles["note"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        f"Quant Backtest Studio \u2014 detailed report \u2014 generated {stamp}. "
        f"Every figure above comes from the backtest already run.", styles["footer"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                            title=f"{label} - Detailed Report")
    doc.build(story, onFirstPage=_paint_bg, onLaterPages=_paint_bg)
    return buf.getvalue()
