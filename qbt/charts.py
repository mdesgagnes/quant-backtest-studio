"""Plotly charts.

"Ink and brass" visual identity: dark ink-blue background, brass traces for
the strategy, teal for the benchmark, rust for risk. One accent color per
chart, everything else in working gray.

A light "print" theme is also available for the exportable tearsheet report,
where a dark background would be wrong on paper.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import metrics as M

PALETTE = {
    "ink": "#0E1116",
    "panel": "#161B22",
    "rule": "#2A323D",
    "text": "#E3E8EF",
    "muted": "#7D8A9C",
    "brass": "#C9A227",
    "teal": "#4C9A8F",
    "rust": "#B4553F",
    "slate": "#5B6B80",
}

PALETTE_PRINT = {
    "ink": "#FFFFFF",
    "panel": "#FFFFFF",
    "rule": "#DDE2E8",
    "text": "#1B2430",
    "muted": "#6B7684",
    "brass": "#A9791E",
    "teal": "#2E7A6E",
    "rust": "#A2432F",
    "slate": "#4E5C6E",
}

SERIES_COLORS = [PALETTE["brass"], PALETTE["teal"], PALETTE["slate"],
                 PALETTE["rust"], "#8E7CC3", "#D08C4E", "#6FA8C7"]

SERIES_COLORS_PRINT = [PALETTE_PRINT["brass"], PALETTE_PRINT["teal"],
                       PALETTE_PRINT["slate"], PALETTE_PRINT["rust"],
                       "#6E5FA3", "#B06E30", "#3E7FA0"]

FONT = "IBM Plex Sans, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, SFMono-Regular, Consolas, monospace"


def _colors(theme: str):
    return (PALETTE_PRINT, SERIES_COLORS_PRINT) if theme == "print" else (PALETTE, SERIES_COLORS)


def _base(fig: go.Figure, height: int = 380, title: str = "", theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    fig.update_layout(
        template="plotly_white" if theme == "print" else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=pal["text"]),
        title=dict(text=title, font=dict(family=FONT, size=14,
                                         color=pal["muted"]), x=0, xanchor="left"),
        height=height,
        margin=dict(l=8, r=8, t=38 if title else 12, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=pal["rule"],
                   tickfont=dict(family=MONO, size=10, color=pal["muted"])),
        yaxis=dict(gridcolor=pal["rule"], zeroline=False,
                   tickfont=dict(family=MONO, size=10, color=pal["muted"])),
    )
    return fig


# ----------------------------------------------------------------------
def equity_curve(curves: pd.DataFrame, log: bool = True,
                 title: str = "Portfolio value (base 100)",
                 theme: str = "dark") -> go.Figure:
    pal, colors = _colors(theme)
    fig = go.Figure()
    for i, col in enumerate(curves.columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=curves.index, y=curves[col], name=str(col), mode="lines",
            line=dict(color=color, width=2.2 if i == 0 else 1.4),
            hovertemplate="%{y:,.1f}<extra>" + str(col) + "</extra>",
        ))
    _base(fig, 420, title, theme)
    fig.update_yaxes(type="log" if log else "linear",
                     tickformat=",.0f" if not log else None)
    return fig


def underwater(equities: Dict[str, pd.Series],
               title: str = "Drawdown from prior peak",
               theme: str = "dark") -> go.Figure:
    pal, colors = _colors(theme)
    fig = go.Figure()
    for i, (name, eq) in enumerate(equities.items()):
        dd = M.drawdown(eq) * 100
        color = pal["rust"] if i == 0 else colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=name, mode="lines",
            line=dict(color=color, width=1.4),
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(180,85,63,0.22)",
            hovertemplate="%{y:.2f}%<extra>" + name + "</extra>",
        ))
    _base(fig, 260, title, theme)
    fig.update_yaxes(ticksuffix="%")
    return fig


def monthly_heatmap(returns: pd.Series,
                    title: str = "Monthly returns",
                    theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    tbl = M.monthly_returns(returns)
    if tbl.empty:
        return _base(go.Figure(), 260, title, theme)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z = tbl.reindex(columns=range(1, 13)).values * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=months, y=[str(y) for y in tbl.index],
        colorscale=[[0.0, pal["rust"]], [0.5, pal["panel"]], [1.0, pal["teal"]]],
        zmid=0, showscale=False,
        text=np.round(z, 1), texttemplate="%{text}",
        textfont=dict(family=MONO, size=9),
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        xgap=2, ygap=2,
    ))
    _base(fig, max(220, 26 * len(tbl) + 80), title, theme)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def weights_area(weights: pd.DataFrame, cash: Optional[pd.Series] = None,
                 title: str = "Portfolio composition",
                 theme: str = "dark") -> go.Figure:
    pal, colors = _colors(theme)
    fig = go.Figure()
    if cash is not None:
        fig.add_trace(go.Scatter(
            x=cash.index, y=cash * 100, name="Cash", mode="lines",
            stackgroup="w", line=dict(width=0), fillcolor="rgba(125,138,156,0.35)",
        ))
    for i, col in enumerate(weights.columns):
        fig.add_trace(go.Scatter(
            x=weights.index, y=weights[col] * 100, name=str(col), mode="lines",
            stackgroup="w", line=dict(width=0),
            fillcolor=colors[i % len(colors)],
            hovertemplate="%{y:.1f}%<extra>" + str(col) + "</extra>",
        ))
    _base(fig, 320, title, theme)
    fig.update_yaxes(ticksuffix="%", range=[0, 100])
    return fig


def rolling_metric(series: pd.Series, label: str, ref: Optional[float] = None,
                   title: str = "", theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    fig = go.Figure(go.Scatter(
        x=series.index, y=series, name=label, mode="lines",
        line=dict(color=pal["brass"], width=1.6),
    ))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=pal["muted"], width=1, dash="dot"))
    return _base(fig, 260, title or label, theme)


def return_distribution(returns: pd.Series,
                        title: str = "Daily return distribution",
                        theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    r = returns.dropna() * 100
    fig = go.Figure(go.Histogram(
        x=r, nbinsx=90, marker=dict(color=pal["brass"], line=dict(width=0)),
        opacity=0.85, name="Observed",
    ))
    v = np.percentile(r, 5)
    fig.add_vline(x=v, line=dict(color=pal["rust"], width=1.5, dash="dash"),
                  annotation_text=f"VaR 95%: {v:.2f}%",
                  annotation_font=dict(family=MONO, size=10, color=pal["rust"]))
    _base(fig, 280, title, theme)
    fig.update_layout(hovermode="closest", bargap=0.02)
    fig.update_xaxes(ticksuffix="%")
    return fig


def monte_carlo_fan(bands: pd.DataFrame, actual: pd.Series,
                    title: str = "Simulated paths (block resampling)",
                    theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    fig = go.Figure()
    if not bands.empty:
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p95"], name="95th percentile",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p5"], name="5th-95th percentile",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(76,154,143,0.18)"))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["p50"], name="Simulated median",
                                 line=dict(color=pal["teal"], width=1.4, dash="dot")))
    base = actual / actual.iloc[0]
    fig.add_trace(go.Scatter(x=base.index, y=base, name="Realized path",
                             line=dict(color=pal["brass"], width=2.2)))
    _base(fig, 360, title, theme)
    fig.update_yaxes(type="log")
    return fig


def sweep_heatmap(df: pd.DataFrame, x: str, y: str, z: str,
                  title: str = "", theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    piv = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[str(c) for c in piv.columns], y=[str(i) for i in piv.index],
        colorscale=[[0, pal["rust"]], [0.5, pal["panel"]], [1, pal["brass"]]],
        zmid=float(np.nanmedian(piv.values)) if np.isfinite(piv.values).any() else 0,
        text=np.round(piv.values, 2), texttemplate="%{text}",
        textfont=dict(family=MONO, size=9),
        colorbar=dict(thickness=8, outlinewidth=0,
                      tickfont=dict(family=MONO, size=9)),
        xgap=2, ygap=2,
    ))
    _base(fig, max(260, 30 * len(piv) + 90), title or f"{z} by {x} and {y}", theme)
    fig.update_xaxes(title=x, showgrid=False)
    fig.update_yaxes(title=y, showgrid=False)
    return fig


def sweep_line(df: pd.DataFrame, x: str, z: str, title: str = "",
              theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    d = df.groupby(x)[z].mean().sort_index()
    fig = go.Figure(go.Scatter(x=d.index, y=d.values, mode="lines+markers",
                               line=dict(color=pal["brass"], width=2),
                               marker=dict(size=6, color=pal["brass"])))
    _base(fig, 280, title or f"{z} by {x}", theme)
    fig.update_xaxes(title=x)
    return fig


def bar_series(labels, values, title: str = "", suffix: str = "",
              theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    colors = [pal["teal"] if v >= 0 else pal["rust"] for v in values]
    fig = go.Figure(go.Bar(x=list(labels), y=list(values),
                           marker=dict(color=colors, line=dict(width=0))))
    _base(fig, 280, title, theme)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(ticksuffix=suffix)
    return fig


def correlation_matrix(corr: pd.DataFrame,
                       title: str = "Correlation of daily returns",
                       theme: str = "dark") -> go.Figure:
    """Correlation matrix. Native Plotly rendering: no matplotlib dependency,
    which pandas' `.style` gradient would otherwise require."""
    pal, _ = _colors(theme)
    labels = [str(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0.0, pal["teal"]], [0.5, pal["panel"]], [1.0, pal["rust"]]],
        zmin=-1, zmax=1, zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(family=MONO, size=10),
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=8, outlinewidth=0,
                      tickfont=dict(family=MONO, size=9)),
        xgap=2, ygap=2,
    ))
    _base(fig, max(260, 32 * len(labels) + 110), title, theme)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig
