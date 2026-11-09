"""Exports for a fund review: charts, PDF report, PowerPoint deck, Excel.

All three documents are built from the same `FundReview` object and the
same PNG charts, so they cannot disagree. Charts use the print palette from
`qbt.brand` (light ground, product in the brand red), matching the other
exported reports. Nothing here computes a statistic: `qbt.fund_review`
does that.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

from .brand import PRINT, SERIES_PRINT
from .fund_review import FundReview, config_to_yaml

DPI = 160
C_PRODUCT = SERIES_PRINT[0]
C_REPLICA = PRINT["text"]
C_FACTOR = SERIES_PRINT[3]
C_ALT = SERIES_PRINT[2]


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def pct(x, d: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def _date(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %d, %Y")


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def _style():
    plt.rcParams.update({
        "figure.facecolor": PRINT["bg"], "axes.facecolor": PRINT["bg"],
        "axes.edgecolor": PRINT["rule"], "axes.labelcolor": PRINT["muted"],
        "text.color": PRINT["text"], "xtick.color": PRINT["muted"], "ytick.color": PRINT["muted"],
        "grid.color": PRINT["rule_soft"], "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def charts(rv: FundReview) -> Dict[str, bytes]:
    _style()
    cfg, nm = rv.config, rv.config.name
    out: Dict[str, bytes] = {}
    pfmt = mt.PercentFormatter(1.0, decimals=0)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    eq = rv.equity
    ax.plot(eq[cfg.factor], color=C_FACTOR, lw=1.4, label=cfg.factor)
    ax.plot(eq["Beta replica"], color=C_REPLICA, lw=1.8, ls="--",
            label=f"Beta replica ({rv.beta:.2f} factor / {1 - rv.beta:.2f} cash)")
    ax.plot(eq[nm], color=C_PRODUCT, lw=2.0, label=nm)
    ax.set_title("Growth of $100: product vs. its beta-matched replica")
    ax.legend(loc="upper left")
    out["growth"] = _png(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(rv.relative, color=C_PRODUCT, lw=1.8)
    ax.axhline(0, color=PRINT["text"], lw=0.8)
    if cfg.split_date:
        ax.axvline(pd.Timestamp(cfg.split_date), color=PRINT["muted"], lw=1, ls=":")
        ax.text(pd.Timestamp(cfg.split_date), ax.get_ylim()[1], f" split {cfg.split_date}",
                va="top", fontsize=8, color=PRINT["muted"])
    ax.yaxis.set_major_formatter(pfmt)
    ax.set_title("Cumulative excess return over the beta replica")
    out["relative"] = _png(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for col, c, lw in ((cfg.factor, C_FACTOR, 1.2), ("Beta replica", C_REPLICA, 1.2), (nm, C_PRODUCT, 1.6)):
        e = eq[col].dropna()
        ax.plot(e / e.cummax() - 1, color=c, lw=lw, label=col)
    ax.yaxis.set_major_formatter(pfmt)
    ax.set_title("Drawdowns")
    ax.legend(loc="lower left")
    out["drawdowns"] = _png(fig)

    cal = rv.calendar.drop(columns=["Partial"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(cal))
    w = 0.27
    for i, (col, c) in enumerate(((nm, C_PRODUCT), ("Beta replica", C_REPLICA), (cfg.factor, C_FACTOR))):
        ax.bar(x + (i - 1) * w, cal[col].values, w * 0.92, color=c, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}*" if p else str(y) for y, p in zip(cal.index, rv.calendar["Partial"])])
    ax.axhline(0, color=PRINT["text"], lw=0.8)
    ax.yaxis.set_major_formatter(pfmt)
    ax.set_title("Calendar-year returns (* partial year)")
    ax.legend(loc="upper left", ncol=3)
    out["calendar"] = _png(fig)

    q = rv.quintiles
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(q))
    for i, (col, c) in enumerate(((nm, C_PRODUCT), ("Beta replica", C_REPLICA), (cfg.factor, C_FACTOR))):
        ax.bar(x + (i - 1) * w, q[col].values, w * 0.92, color=c, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in q.index])
    ax.axhline(0, color=PRINT["text"], lw=0.8)
    ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1))
    ax.set_title("Average monthly return by factor-month quintile")
    ax.legend(loc="upper left")
    out["payoff"] = _png(fig)

    if not rv.rolling.empty:
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.plot(rv.rolling["Beta"], color=C_PRODUCT, lw=1.8, label="Beta")
        ax.plot(rv.rolling["Correlation"], color=C_FACTOR, lw=1.4, label="Correlation")
        ax.axhline(rv.beta, color=PRINT["muted"], lw=0.8, ls=":")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Rolling {rv.config.rolling_window}-month beta and correlation to {cfg.factor}")
        ax.legend(loc="lower left")
        out["rolling"] = _png(fig)

    if rv.funding is not None:
        f = rv.funding
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        y = np.arange(len(f))[::-1]
        cols = [PRINT["muted"]] + [C_PRODUCT] * (len(f) - 1)
        for ax, m, fmt in ((axs[0], "Sharpe", "{:.2f}"), (axs[1], "Max Drawdown", "{:.1%}")):
            v = f[m].astype(float).values
            ax.barh(y, v, color=cols, height=0.7)
            ax.axvline(float(f.loc["Base portfolio", m]), color=PRINT["text"], lw=0.8, ls="--")
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            pad = (hi - lo) * 0.35 or abs(hi) * 0.05
            ax.set_xlim(lo - pad, hi + pad)
            for yi, vi in zip(y, v):
                ax.text(vi, yi, " " + fmt.format(vi), va="center", fontsize=8)
            ax.set_title(m)
            ax.grid(axis="y", visible=False)
        axs[0].set_yticks(y)
        axs[0].set_yticklabels(list(f.index))
        out["funding"] = _png(fig)
    return out


# ----------------------------------------------------------------------
# Tables shared by every format
# ----------------------------------------------------------------------
def headline_table(rv: FundReview) -> pd.DataFrame:
    h = rv.headline
    return pd.DataFrame({
        "CAGR": h["CAGR"].map(pct), "Volatility": h["Volatility"].map(pct),
        "Sharpe": h["Sharpe"].map(num), "Max drawdown": h["Max Drawdown"].map(pct),
        "Worst month": h["Worst month"].map(pct), "Growth of $100": h["Growth of 100"].map(lambda v: f"${v:,.0f}"),
    })


def regression_table(rv: FundReview) -> pd.DataFrame:
    r, s = rv.regression, rv.split or {}
    rows = [("Months", f"{r['months']:.0f}"),
            ("Alpha, annualized (Newey-West t)", f"{pct(r['alpha'], 1, True)} ({num(r['alpha_t'])})"),
            ("Beta (t)", f"{num(r['beta'])} ({num(r['beta_t'], 1)})"),
            ("R²", num(r["r2"])),
            ("Up-market / down-market beta", f"{num(r['beta_up'])} / {num(r['beta_down'])}"),
            ("Up / down capture", f"{pct(r['up_capture'], 0)} / {pct(r['down_capture'], 0)}"),
            ("Treynor-Mazuy convexity γ (t)", f"{num(r['tm_gamma'])} ({num(r['tm_gamma_t'])})"),
            ("Correlation with factor", num(r["corr_factor"]))]
    if "beta_bond" in r:
        rows += [("Bond beta, 2-factor (t)", f"{num(r['beta_bond'])} ({num(r['beta_bond_t'], 1)})"),
                 ("Correlation with bond factor", num(r["corr_bond"]))]
    if s:
        sd = rv.config.split_date
        rows += [(f"Alpha before {sd} (t)", f"{pct(s.get('alpha_before'), 1, True)} ({num(s.get('alpha_before_t'))})"),
                 (f"Alpha since {sd} (t)", f"{pct(s.get('alpha_after'), 1, True)} ({num(s.get('alpha_after_t'))})")]
    return pd.DataFrame(rows, columns=["Measure", "Value"])


def stress_table(rv: FundReview) -> pd.DataFrame:
    t = rv.stress.copy()
    if t.empty:
        return t
    for c in t.columns[3:]:
        t[c] = t[c].map(lambda v: pct(v, 1, True))
    t["Episode"] = t["Episode"] + " (" + t["Start"].str[:7] + ")"
    return t.drop(columns=["Start", "End"])


def funding_table(rv: FundReview) -> Optional[pd.DataFrame]:
    f = rv.funding
    if f is None:
        return None
    return pd.DataFrame({"CAGR": f["CAGR"].map(lambda v: pct(v, 2)), "Volatility": f["Volatility"].map(pct),
                         "Sharpe": f["Sharpe"].map(num), "Max drawdown": f["Max Drawdown"].map(pct),
                         "CVaR 95% (per period)": f["CVaR 95%"].map(lambda v: pct(v, 2))})


def key_points(rv: FundReview) -> List[str]:
    cfg, h, r = rv.config, rv.headline, rv.regression
    nm = cfg.name
    rep = [i for i in h.index if i.startswith("Beta replica")][0]
    pts = [f"{nm} returned {pct(h.loc[nm, 'CAGR'])}/yr vs. {pct(h.loc[rep, 'CAGR'])} for "
           f"{rv.beta:.2f} x {cfg.factor} + {1 - rv.beta:.2f} x cash, with volatility "
           f"{pct(h.loc[nm, 'Volatility'])} vs. {pct(h.loc[rep, 'Volatility'])} and max drawdown "
           f"{pct(h.loc[nm, 'Max Drawdown'])} vs. {pct(h.loc[rep, 'Max Drawdown'])}.",
           f"Alpha {pct(r['alpha'], 1, True)}/yr (t = {num(r['alpha_t'])}); only the beta of "
           f"{num(r['beta'])} is reliably estimated." if abs(r["alpha_t"]) < 2 else
           f"Alpha {pct(r['alpha'], 1, True)}/yr (t = {num(r['alpha_t'])}).",]
    if rv.split:
        pts.append(f"Since {cfg.split_date}: alpha {pct(rv.split.get('alpha_after'), 1, True)}/yr "
                   f"vs. {pct(rv.split.get('alpha_before'), 1, True)}/yr before.")
    if rv.big_days:
        b = rv.big_days
        pts.append(f"On {cfg.big_day_threshold:.0%}+ factor down days it lost {pct(b['down_share'], 0)} "
                   f"of the move; on up days it kept {pct(b['up_share'], 0)}.")
    if rv.costs.get("fee_gap") is not None:
        pts.append(f"MER {pct(cfg.mer, 2)} vs. about {pct(cfg.replica_cost, 2)} for the replica.")
    return pts


def method_text(rv: FundReview) -> List[str]:
    cfg = rv.config
    return [
        f"Data: {cfg.name} from {_date(rv.start)} to {_date(rv.end)}, {rv.ppy} periods a year; "
        f"factor {cfg.factor}" + (f", bond factor {cfg.bond}" if cfg.bond else "") + ".",
        "Beta replica: beta x factor + (1 - beta) x the risk-free series, rebalanced monthly, no costs; "
        "beta from the full-sample monthly regression of excess returns.",
        "Regressions on monthly excess returns with Newey-West (3 lags) standard errors; a final partial "
        "month is dropped.",
        "Stress episodes: the app's standard list inside the history, the config's own windows, and the "
        "current drawdown if deeper than 3%.",
        f"Funding test: base portfolio plus {cfg.funding_amount:.0%} of the product, funded from each sleeve "
        f"and beta-proportionally, {cfg.rebalance} rebalance, no frictions, through the backtest engine.",
        "Limitations: beta is in-sample; currency exposure is captured only through the factor's currency; "
        "tax figures are illustrative and use one year of distributions.",
    ]


# ----------------------------------------------------------------------
# PDF (reportlab)
# ----------------------------------------------------------------------
def render_pdf(rv: FundReview, imgs: Optional[Dict[str, bytes]] = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)
    imgs = imgs or charts(rv)
    cfg = rv.config
    ink, muted, rule = (colors.HexColor(PRINT[k]) for k in ("text", "muted", "rule"))
    accent = colors.HexColor(C_PRODUCT)
    S = {
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=ink),
        "meta": ParagraphStyle("meta", fontName="Helvetica", fontSize=8.5, textColor=muted, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, textColor=ink, spaceBefore=12,
                             spaceAfter=5),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=12.5, textColor=ink,
                               alignment=TA_LEFT, spaceAfter=4),
        "verdict": ParagraphStyle("verdict", fontName="Helvetica-Bold", fontSize=14, leading=18,
                                  textColor=accent, spaceAfter=4),
        "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8, textColor=muted, spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10, textColor=ink),
    }
    W = letter[0] - 1.3 * inch

    def table(df: pd.DataFrame, index: bool = True, first: float = 0.34) -> Table:
        d = df.reset_index() if index else df
        data = [[Paragraph(f"<b>{c}</b>", S["cell"]) for c in d.columns]]
        data += [[Paragraph(str(v), S["cell"]) for v in row] for row in d.astype(str).values]
        n = len(d.columns)
        widths = [W * first] + [W * (1 - first) / max(n - 1, 1)] * (n - 1)
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, ink),
                               ("LINEBELOW", (0, 1), (-1, -1), 0.3, rule),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def img(key: str, caption: str):
        if key not in imgs:
            return []
        from reportlab.lib.utils import ImageReader
        ir = ImageReader(io.BytesIO(imgs[key]))
        iw, ih = ir.getSize()
        return [KeepTogether([Image(io.BytesIO(imgs[key]), width=W, height=W * ih / iw),
                              Paragraph(caption, S["cap"])])]

    v = rv.verdict
    story = [Paragraph(f"Fund review: {cfg.name}", S["h1"]),
             Paragraph(f"{_date(rv.start)} to {_date(rv.end)} · factor {cfg.factor} · generated "
                       f"{datetime.now():%Y-%m-%d} by Quant Backtest Studio", S["meta"]),
             Paragraph(f"Verdict: {v['verdict']}", S["verdict"]), Paragraph(v["summary"], S["body"]),
             table(pd.DataFrame([(c["check"], c["status"].upper(), c["detail"]) for c in v["checks"]],
                                columns=["Check", "Result", "Detail"]), index=False, first=0.2),
             Paragraph("Key points", S["h2"])]
    story += [Paragraph("• " + p, S["body"]) for p in key_points(rv)]
    story += [Paragraph("Headline statistics", S["h2"]), table(headline_table(rv)),
              Paragraph("Volatility from monthly returns; Sharpe = (CAGR − risk-free) ÷ volatility; "
                        "max drawdown from the native frequency.", S["cap"])]
    story += img("growth", "Growth of $100.") + img("relative", "Cumulative excess over the beta replica.")
    story += [Paragraph("Is there alpha?", S["h2"]), table(regression_table(rv), index=False, first=0.5)]
    story += [Paragraph("Calendar years", S["h2"]),
              table(rv.calendar.drop(columns=["Partial"]).map(lambda x: pct(x, 1))
                    .rename(index=lambda y: f"{y}*" if rv.calendar.loc[y, "Partial"] else str(y)))]
    story += img("calendar", "Calendar-year returns.")
    story += [Paragraph("Stress and recovery episodes", S["h2"]), table(stress_table(rv), index=False, first=0.4)]
    if not rv.crash.empty:
        c = rv.crash.copy()
        c["Share absorbed"] = c["Share absorbed"].map(lambda x: pct(x, 0))
        c["Share / beta"] = c["Share / beta"].map(lambda x: f"{x:.2f}x")
        for col in (cfg.name, cfg.factor):
            c[col] = c[col].map(lambda x: pct(x, 1, True))
        story += [Paragraph("Share of each factor sell-off absorbed (vs. a beta of "
                            f"{rv.beta:.2f})", S["body"]), table(c, index=False, first=0.36)]
    story += img("drawdowns", "Drawdowns.") + img("payoff", "Payoff profile by factor-month quintile.")
    story += img("rolling", "Rolling beta and correlation.")
    ft = funding_table(rv)
    if ft is not None:
        story += [Paragraph("Portfolio funding test", S["h2"]),
                  Paragraph(f"{rv.funding.attrs.get('start')} to {rv.funding.attrs.get('end')}. Base: "
                            + ", ".join(f"{k} {w:.0%}" for k, w in cfg.base_portfolio.items()) + ".", S["body"]),
                  table(ft, first=0.36)]
        story += img("funding", "Sharpe and max drawdown by funding source.")
    story += [Paragraph("Costs and taxes", S["h2"])]
    if cfg.mer is not None:
        story.append(Paragraph(f"MER {pct(cfg.mer, 2)} vs. about {pct(cfg.replica_cost, 2)} for the replica: "
                               f"a {pct(rv.costs['fee_gap'], 2)} annual hurdle before any alpha reaches the client.",
                               S["body"]))
    tx = rv.costs.get("tax")
    if tx:
        story.append(Paragraph(
            f"Illustrative, non-registered ({cfg.distribution.source or 'one year of distributions'}): "
            f"income {pct(tx['income_pct'])} + dividends {pct(tx['dividends_pct'])} + realized gains "
            f"{pct(tx['capital_gains_pct'])} of NAV → about {pct(tx['product_tax'])} of NAV in tax a year, vs. "
            f"{pct(tx['replica_tax'])} for the replica (cash yield {pct(tx['cash_yield'])}, equity gains deferred). "
            "Much of the gap is timing; in registered accounts it disappears.", S["body"]))
    story += [Paragraph("Method and limitations", S["h2"])] + [Paragraph("• " + m, S["body"]) for m in method_text(rv)]
    for n in rv.notes:
        story.append(Paragraph("Note: " + n, S["body"]))

    def _foot(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawString(0.65 * inch, 0.45 * inch, f"Fund review · {cfg.name}")
        canvas.drawRightString(letter[0] - 0.65 * inch, 0.45 * inch, f"{doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                      topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                      title=f"Fund review: {cfg.name}").build(story, onFirstPage=_foot, onLaterPages=_foot)
    return buf.getvalue()


# ----------------------------------------------------------------------
# PowerPoint (python-pptx)
# ----------------------------------------------------------------------
def render_pptx(rv: FundReview, imgs: Optional[Dict[str, bytes]] = None) -> bytes:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Emu, Inches, Pt

    imgs = imgs or charts(rv)
    cfg = rv.config
    rgb = lambda h: RGBColor.from_string(h.lstrip("#"))
    INK, MUTED, ACC, RULE = rgb(PRINT["text"]), rgb(PRINT["muted"]), rgb(C_PRODUCT), rgb(PRINT["rule"])
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    n = [0]

    def text(slide, x, y, w, h, s, size=16, bold=False, color=INK):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        lines = s if isinstance(s, list) else [s]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
            p.space_after = Pt(6)
        return tb

    def slide(title: str):
        s = prs.slides.add_slide(blank)
        n[0] += 1
        text(s, 0.6, 0.35, 12.1, 0.9, title, 28, True)
        text(s, 0.6, 7.0, 10, 0.3, f"Fund review · {cfg.name} · {_date(rv.start)} – {_date(rv.end)}", 10,
             color=MUTED)
        text(s, 12.2, 7.0, 0.6, 0.3, str(n[0]), 10, color=MUTED)
        return s

    def picture(s, key, y=1.9, h=4.9):
        if key in imgs:
            pic = s.shapes.add_picture(io.BytesIO(imgs[key]), Inches(0.6), Inches(y), height=Inches(h))
            if pic.width > Inches(12.1):
                ratio = Inches(12.1) / pic.width
                pic.width, pic.height = Inches(12.1), int(pic.height * ratio)

    def grid(s, df: pd.DataFrame, y=1.6, size=12, first=3.6):
        rows, cols = df.shape[0] + 1, df.shape[1]
        sh = s.shapes.add_table(rows, cols, Inches(0.6), Inches(y), Inches(12.1), Inches(0.38 * rows))
        t = sh.table
        t.columns[0].width = Inches(first)
        for j in range(1, cols):
            t.columns[j].width = Emu(int((Inches(12.1) - Inches(first)) / (cols - 1)))
        for j, c in enumerate(df.columns):
            t.cell(0, j).text = str(c)
        for i, row in enumerate(df.astype(str).values, start=1):
            for j, v in enumerate(row):
                t.cell(i, j).text = v
        for i in range(rows):
            for j in range(cols):
                cell = t.cell(i, j)
                cell.fill.background()
                for p in cell.text_frame.paragraphs:
                    p.font.size, p.font.color.rgb = Pt(size), INK
                    p.font.bold = i == 0
        return sh

    v = rv.verdict
    s = prs.slides.add_slide(blank)
    n[0] += 1
    text(s, 0.8, 2.2, 11.5, 1.5, f"Fund review: {cfg.name}", 40, True)
    text(s, 0.8, 3.7, 11.5, 0.8, f"Does it earn its place? Replica test vs. {cfg.factor}", 20, color=MUTED)
    text(s, 0.8, 6.3, 11.5, 0.5, f"{_date(rv.start)} – {_date(rv.end)} · Quant Backtest Studio", 12, color=MUTED)

    s = slide(f"Verdict: {v['verdict']}")
    text(s, 0.6, 1.3, 12.1, 0.8, v["summary"], 18, color=ACC, bold=True)
    grid(s, pd.DataFrame([(c["check"], c["status"].upper(), c["detail"]) for c in v["checks"]],
                         columns=["Check", "Result", "Detail"]), y=2.3, size=13, first=2.6)
    text(s, 0.6, 4.6, 12.1, 2.3, ["• " + p for p in key_points(rv)[:4]], 14)

    s = slide("Headline numbers")
    grid(s, headline_table(rv).reset_index().rename(columns={"index": ""}), size=13, first=4.4)
    for key, title in (("growth", "Growth of $100: product vs. its replica"),
                       ("relative", "Cumulative excess over the replica"),
                       ("calendar", "Calendar years")):
        picture(slide(title), key, y=1.4, h=5.4)
    st = stress_table(rv)
    if not st.empty:
        grid(slide("Episode by episode"), st.head(12), y=1.3, size=11, first=4.6)
    s = slide("Is there alpha?")
    grid(s, regression_table(rv), y=1.3, size=12, first=6.0)
    for key, title in (("drawdowns", "Drawdowns"), ("payoff", "Payoff by factor-month quintile"),
                       ("rolling", "Rolling beta and correlation")):
        if key in imgs:
            picture(slide(title), key, y=1.4, h=5.4)
    ft = funding_table(rv)
    if ft is not None:
        s = slide(f"Funding test: adding {cfg.funding_amount:.0%}")
        grid(s, ft.reset_index().rename(columns={"index": "Portfolio"}), y=1.3, size=11, first=4.2)
        if "funding" in imgs:
            picture(slide("Funding test: what you sell decides the outcome"), "funding", y=1.4, h=5.2)
    s = slide("Method and limitations")
    text(s, 0.6, 1.4, 12.1, 5.4, ["• " + m for m in method_text(rv)], 14)
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# ----------------------------------------------------------------------
# Excel
# ----------------------------------------------------------------------
def render_excel(rv: FundReview) -> bytes:
    buf = io.BytesIO()
    v = rv.verdict
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([("Verdict", v["verdict"]), ("Summary", v["summary"])]
                     + [(c["check"], f"{c['status'].upper()}: {c['detail']}") for c in v["checks"]]
                     + [("Key point", p) for p in key_points(rv)],
                     columns=["Item", "Value"]).to_excel(xw, sheet_name="Summary", index=False)
        rv.headline.to_excel(xw, sheet_name="Headline")
        pd.DataFrame({**rv.regression, **(rv.split or {})}, index=["Value"]).T.to_excel(xw, sheet_name="Regression")
        rv.calendar.to_excel(xw, sheet_name="Calendar")
        rv.trailing.to_excel(xw, sheet_name="Trailing")
        rv.stress.to_excel(xw, sheet_name="Stress", index=False)
        if not rv.crash.empty:
            rv.crash.to_excel(xw, sheet_name="Sell-offs", index=False)
        rv.quintiles.to_excel(xw, sheet_name="Payoff quintiles")
        rv.rolling.to_excel(xw, sheet_name="Rolling beta")
        if rv.funding is not None:
            rv.funding.to_excel(xw, sheet_name="Funding test")
        rv.equity.to_excel(xw, sheet_name="Growth of 100")
        pd.DataFrame({"config": config_to_yaml(rv.config).splitlines()}).to_excel(xw, sheet_name="Config", index=False)
        for ws in xw.book.worksheets:
            ws.column_dimensions["A"].width = 44
            for col in "BCDEFGH":
                ws.column_dimensions[col].width = 18
    return buf.getvalue()
