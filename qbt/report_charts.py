"""Chart rendering for the detailed report.

Every other chart in this app is Plotly, drawn live in the browser. This
report needs something different: a chart that is the same static image
whether it ends up in the HTML file or the PDF, and that renders without a
browser, since the PDF cannot execute JavaScript and a downloaded HTML file
should not need an internet connection to show its own charts.

Plotly's own static-image path (`fig.to_image()`) goes through Kaleido,
which as of v1 requires downloading a full Chrome binary at first use --
a slow, large, and failure-prone step to add to a free-tier deployment's
build process. Matplotlib is already a dependency of this project's own
dependencies, renders to PNG natively with no browser and no network
access, and is the standard tool for exactly this job. Every chart here
returns raw PNG bytes, which the HTML renderer base64-encodes and the PDF
renderer hands directly to a reportlab Image flowable -- one rendering
path, two destinations, so the two documents cannot drift apart visually.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")   # no display backend needed or wanted here
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .brand import BRAND, SERIES

DPI = 150


def _style(dark: bool = True) -> Dict[str, str]:
    pal = {
        "bg": BRAND["bg"] if dark else "#FFFFFF",
        "panel": BRAND["panel"] if dark else "#FAFAF8",
        "rule": BRAND["rule"] if dark else "#DDDAD2",
        "text": BRAND["text"] if dark else "#1A1A1A",
        "muted": BRAND["muted"] if dark else "#6B655C",
        "red": BRAND["red"],
        "gain": BRAND["gain"] if dark else "#1F6F5C",
        "loss": BRAND["loss"] if dark else "#B3322B",
    }
    plt.rcParams.update({
        "figure.facecolor": pal["bg"], "axes.facecolor": pal["panel"],
        "savefig.facecolor": pal["bg"], "axes.edgecolor": pal["rule"],
        "axes.labelcolor": pal["text"], "text.color": pal["text"],
        "xtick.color": pal["muted"], "ytick.color": pal["muted"],
        "grid.color": pal["rule"], "grid.alpha": 0.5,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return pal


def _finish(fig, pal: Dict[str, str]) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=pal["bg"],
               bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def equity_chart(curves: Dict[str, pd.Series], title: str = "",
                 log: bool = True, dark: bool = True) -> bytes:
    """One or more equity curves. Used for Results (strategy vs benchmark)
    and Tax (pretax vs after-tax, up to four lines)."""
    pal = _style(dark)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (name, s) in enumerate(curves.items()):
        s = s.dropna()
        if s.empty:
            continue
        dashed = "after tax" in name.lower()
        ax.plot(s.index, s.values, label=name,
               color=SERIES[i % len(SERIES)], linewidth=1.6,
               linestyle="--" if dashed else "-")
    if log:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=pal["text"])
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def drawdown_chart(equity: pd.Series, title: str = "", dark: bool = True) -> bytes:
    pal = _style(dark)
    eq = equity.dropna()
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(7.5, 2.0))
    ax.fill_between(dd.index, dd.values * 100, 0, color=pal["loss"], alpha=0.35)
    ax.plot(dd.index, dd.values * 100, color=pal["loss"], linewidth=1.0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("%")
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def bar_chart(labels: Sequence[str], values: Sequence[float], title: str = "",
             suffix: str = "", horizontal: bool = False, dark: bool = True) -> bytes:
    pal = _style(dark)
    fig, ax = plt.subplots(figsize=(7.5, max(2.2, 0.32 * len(labels)) if horizontal else 2.6))
    colors = [pal["gain"] if v >= 0 else pal["loss"] for v in values]
    if horizontal:
        ax.barh(list(labels), values, color=colors)
        ax.axvline(0, color=pal["rule"], linewidth=0.8)
        ax.set_xlabel(suffix)
    else:
        ax.bar(list(labels), values, color=colors)
        ax.axhline(0, color=pal["rule"], linewidth=0.8)
        ax.set_ylabel(suffix)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def grouped_bar_chart(labels: Sequence[str], series: Dict[str, Sequence[float]],
                      title: str = "", suffix: str = "", dark: bool = True) -> bytes:
    pal = _style(dark)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
              label=name, color=SERIES[i % len(SERIES)])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(suffix)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=pal["text"])
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def heatmap_chart(pivot: pd.DataFrame, title: str = "", fmt: str = "{:.2f}",
                  dark: bool = True) -> bytes:
    pal = _style(dark)
    fig, ax = plt.subplots(figsize=(7.5, max(2.2, 0.35 * len(pivot))))
    vals = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                       fontsize=6, color=pal["bg"] if abs(v) > vmax * 0.4 else pal["text"])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7)
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def score_chart(scores: pd.DataFrame, title: str = "", dark: bool = True) -> bytes:
    """Up to six series, for the Signals section."""
    pal = _style(dark)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    cols = list(scores.columns)[:6]
    for i, c in enumerate(cols):
        s = scores[c].dropna()
        ax.plot(s.index, s.values, label=str(c), color=SERIES[i % len(SERIES)], linewidth=1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2, labelcolor=pal["text"])
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def underwater_chart(equities: Dict[str, pd.Series], title: str = "",
                     dark: bool = True) -> bytes:
    """Drawdown from prior peak for one or more series; the first is filled."""
    pal = _style(dark)
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    for i, (name, eq) in enumerate(equities.items()):
        eq = eq.dropna()
        if eq.empty:
            continue
        dd = (eq / eq.cummax() - 1.0) * 100
        color = pal["loss"] if i == 0 else SERIES[i % len(SERIES)]
        if i == 0:
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25)
        ax.plot(dd.index, dd.values, color=color, linewidth=1.1, label=name)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if len(equities) > 1:
        ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=pal["text"])
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def distribution_chart(returns: pd.Series, title: str = "",
                       dark: bool = True) -> bytes:
    """Histogram of period returns with the 95% VaR marked."""
    pal = _style(dark)
    r = returns.dropna() * 100
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    if len(r):
        ax.hist(r, bins=min(90, max(10, len(r) // 3)), color=pal["red"],
                alpha=0.85)
        v = float(np.percentile(r, 5))
        ax.axvline(v, color=pal["loss"], linewidth=1.2, linestyle="--")
        ax.text(v, ax.get_ylim()[1] * 0.95, f" VaR 95%: {v:.2f}%",
                color=pal["loss"], fontsize=7, va="top")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)


def composition_chart(weights: pd.DataFrame, cash: Optional[pd.Series] = None,
                      title: str = "", dark: bool = True,
                      max_series: int = 10) -> bytes:
    """Stacked portfolio weights. Beyond `max_series` holdings, the smallest
    by average weight are grouped as "Other" so the legend stays legible."""
    pal = _style(dark)
    w = weights.fillna(0.0).clip(lower=0.0)
    if w.shape[1] > max_series:
        keep = w.mean().sort_values(ascending=False).index[:max_series - 1]
        w = pd.concat([w[keep], w.drop(columns=keep).sum(axis=1).rename("Other")],
                      axis=1)
    layers, labels, colors = [], [], []
    if cash is not None:
        layers.append(cash.reindex(w.index).fillna(0.0).clip(lower=0.0) * 100)
        labels.append("Cash"); colors.append(pal["muted"])
    for i, c in enumerate(w.columns):
        layers.append(w[c] * 100)
        labels.append(str(c)); colors.append(SERIES[i % len(SERIES)])
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    if layers:
        ax.stackplot(w.index, *[l.values for l in layers], labels=labels,
                     colors=colors, alpha=0.85)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
              fontsize=7, labelcolor=pal["text"])
    if title:
        ax.set_title(title, color=pal["text"], fontsize=10, loc="left")
    fig.tight_layout()
    return _finish(fig, pal)
