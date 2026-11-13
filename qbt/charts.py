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

from .brand import BRAND, PRINT, SERIES, SERIES_PRINT

# Every colour comes from qbt/brand.py. Correcting the three brand values
# there restyles the interface, the charts and the exported report together.
PALETTE = {
    "ink": BRAND["bg"],
    "panel": BRAND["panel"],
    "rule": BRAND["rule"],
    "text": BRAND["text"],
    "muted": BRAND["muted"],
    "brass": BRAND["red"],
    "teal": BRAND["gain"],
    "rust": BRAND["loss"],
    "slate": "#8C93A8",
}

# The report is printed and shared, so it stays light regardless of the
# interface theme.
PALETTE_PRINT = {
    "ink": PRINT["bg"],
    "panel": PRINT["panel"],
    "rule": PRINT["rule"],
    "text": PRINT["text"],
    "muted": PRINT["muted"],
    "brass": PRINT["red"],
    "teal": PRINT["gain"],
    "rust": PRINT["loss"],
    "slate": "#3F3A34",
}

SERIES_COLORS = list(SERIES)
SERIES_COLORS_PRINT = list(SERIES_PRINT)

FONT = "Inter, Segoe UI, Helvetica Neue, sans-serif"
MONO = "IBM Plex Mono, SFMono-Regular, Consolas, monospace"


def _colors(theme: str):
    return (PALETTE_PRINT, SERIES_COLORS_PRINT) if theme == "print" else (PALETTE, SERIES_COLORS)


def _base(fig: go.Figure, height: int = 380, title: str = "", theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    fig.update_layout(
        template="plotly_white" if theme == "print" else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=(SERIES_COLORS_PRINT if theme == "print" else SERIES_COLORS),
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
            line=dict(color=color, width=2.2 if i == 0 else 1.4,
                      shape="linear"),
            hovertemplate="%{y:,.1f}<extra>" + str(col) + "</extra>",
        ))
    _base(fig, 420, title, theme)
    fig.update_yaxes(type="log" if log else "linear",
                     tickformat=",.0f" if not log else None)
    # A vertical spike on hover: reading two curves against each other at the
    # same date is the whole point of putting them on one chart.
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor=pal["muted"], spikethickness=1, spikedash="dot")
    fig.update_layout(hoverdistance=60, spikedistance=-1)
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
            fillcolor="rgba(232,112,92,0.18)",
            hovertemplate="%{y:.2f}%<extra>" + name + "</extra>",
        ))
    _base(fig, 260, title, theme)
    fig.update_yaxes(ticksuffix="%")
    return fig


def monthly_heatmap(returns: pd.Series,
                    title: str = "Monthly returns",
                    theme: str = "dark",
                    ppy: int = 252) -> go.Figure:
    pal, _ = _colors(theme)
    # A quarterly or annual stream cannot fill a month-by-month grid.
    if ppy < 12:
        return _base(go.Figure(), 200, title, theme)
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
        texttemplate="%{z:.1f}",
        textfont=dict(family=MONO, size=9),
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        xgap=2, ygap=2,
    ))
    _base(fig, max(220, 26 * len(tbl) + 80), title, theme)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def annual_heatmap(table: pd.DataFrame, partial: Optional[pd.DataFrame] = None,
                   title: str = "Calendar-year returns",
                   theme: str = "dark") -> go.Figure:
    """Instruments by calendar year, shaded by sign and size. A partial year
    (the first year of a series, or the year in progress) carries a star."""
    pal, _ = _colors(theme)
    if table is None or table.empty:
        return _base(go.Figure(), 200, title, theme)
    z = table.values * 100
    text = []
    for i in range(z.shape[0]):
        row = []
        for j in range(z.shape[1]):
            v = z[i, j]
            if v != v:
                row.append("")
            else:
                star = ("*" if partial is not None and bool(partial.iat[i, j])
                        else "")
                row.append(f"{v:.1f}{star}")
        text.append(row)
    cap = float(np.nanpercentile(np.abs(z), 95)) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in table.columns], y=[str(i) for i in table.index],
        colorscale=[[0.0, pal["rust"]], [0.5, pal["panel"]], [1.0, pal["teal"]]],
        zmid=0, zmin=-cap, zmax=cap, showscale=False,
        text=text, texttemplate="%{text}", textfont=dict(family=MONO, size=9),
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>", xgap=2, ygap=2))
    _base(fig, max(220, 30 * len(table) + 90), title, theme)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top")
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
                        title: Optional[str] = None,
                        theme: str = "dark",
                        ppy: int = 252) -> go.Figure:
    pal, _ = _colors(theme)
    title = title or f"{M.freq_words(ppy)[0].capitalize()} return distribution"
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
                                 fillcolor="rgba(201,184,150,0.16)"))
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
    piv = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean", observed=True)
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


def sweep_surface(df: pd.DataFrame, x: str, y: str, z: str,
                  title: str = "", theme: str = "dark") -> go.Figure:
    """A true 3D surface over two swept parameters and the resulting
    metric, for the case a flat heatmap can leave ambiguous: whether a
    strong reading sits on a ridge running the length of one parameter
    (robust to the other) or right at an isolated peak (overfit to both
    at once). Tilting the surface shows that shape directly; a heatmap's
    colour scale can hide it behind two cells that look similarly warm.
    """
    pal, _ = _colors(theme)
    piv = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    x_vals = [float(c) for c in piv.columns]
    y_vals = [float(i) for i in piv.index]
    fig = go.Figure(go.Surface(
        z=piv.values, x=x_vals, y=y_vals,
        colorscale=[[0, pal["rust"]], [0.5, pal["panel"]], [1, pal["brass"]]],
        showscale=True,
        colorbar=dict(thickness=10, outlinewidth=0,
                      tickfont=dict(family=MONO, size=9)),
        contours=dict(z=dict(show=True, usecolormap=True,
                             highlightcolor="rgba(255,255,255,0.5)",
                             project=dict(z=True))),
    ))
    fig.update_layout(
        template="plotly_white" if theme == "print" else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title or f"{z} across {x} and {y}", font=dict(size=13)),
        height=480, margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family=FONT, size=11, color=pal["text"]),
        scene=dict(
            xaxis=dict(title=x, backgroundcolor="rgba(0,0,0,0)",
                      gridcolor=pal["rule"], showbackground=True),
            yaxis=dict(title=y, backgroundcolor="rgba(0,0,0,0)",
                      gridcolor=pal["rule"], showbackground=True),
            zaxis=dict(title=z, backgroundcolor="rgba(0,0,0,0)",
                      gridcolor=pal["rule"], showbackground=True),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
        ),
    )
    return fig


def sweep_line(df: pd.DataFrame, x: str, z: str, title: str = "",
              theme: str = "dark") -> go.Figure:
    pal, _ = _colors(theme)
    d = df.groupby(x, observed=True)[z].mean().sort_index()
    xs = [str(v) for v in d.index] if isinstance(d.index, pd.CategoricalIndex) else d.index
    fig = go.Figure(go.Scatter(x=xs, y=d.values, mode="lines+markers",
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


def bar_compare(labels, series: Dict[str, list], title: str = "",
                suffix: str = "", theme: str = "dark") -> go.Figure:
    """Grouped bars, for a strategy set against a benchmark per label.

    A single-series bar can only ever say "the strategy fell here" -- it
    cannot say whether that was worse or better than the alternative it is
    actually judged against. Two colours per label side by side answers
    the question a single excess-return number collapses away: whether a
    strategy that looks defensive was actually positive during the stress,
    or simply lost less than a benchmark that fell even further.
    """
    pal, colors = _colors(theme)
    fig = go.Figure()
    palette = ([pal["brass"], pal["slate"], pal["teal"], pal["rust"]]
               if len(series) <= 4 else colors)
    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=list(labels), y=list(values), name=str(name),
            marker=dict(color=palette[i % len(palette)], line=dict(width=0)),
        ))
    _base(fig, 320, title, theme)
    fig.update_layout(barmode="group", hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0))
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


def scatter_points(df: pd.DataFrame, x: str, y: str, label: str,
                   title: str = "", x_suffix: str = "%", y_suffix: str = "%",
                   theme: str = "dark") -> go.Figure:
    """Labelled scatter, for risk against return.

    Each point is named on the chart rather than in a legend: with a dozen
    instruments a legend forces the eye to travel back and forth, and the
    whole value of this view is seeing which name sits where.
    """
    pal, colors = _colors(theme)
    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], mode="markers+text",
        text=df[label], textposition="top center",
        textfont=dict(family=MONO, size=10, color=pal["muted"]),
        marker=dict(size=11, color=pal["brass"],
                    line=dict(width=1, color=pal["ink"])),
        hovertemplate=("%{text}<br>" + x + ": %{x:.2f}" + x_suffix +
                       "<br>" + y + ": %{y:.2f}" + y_suffix + "<extra></extra>"),
    ))
    _base(fig, 420, title, theme)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title=x, ticksuffix=x_suffix, zeroline=True,
                     zerolinecolor=pal["rule"])
    fig.update_yaxes(title=y, ticksuffix=y_suffix, zeroline=True,
                     zerolinecolor=pal["rule"])
    return fig


def multi_line(df: pd.DataFrame, title: str = "", suffix: str = "",
               ref: Optional[float] = None, theme: str = "dark") -> go.Figure:
    """Several series on one axis, for rolling correlation or beta."""
    pal, colors = _colors(theme)
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=str(col), mode="lines",
            line=dict(color=colors[i % len(colors)], width=1.5),
            hovertemplate="%{y:.2f}<extra>" + str(col) + "</extra>",
        ))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=pal["muted"], width=1, dash="dot"))
    _base(fig, 340, title, theme)
    fig.update_yaxes(ticksuffix=suffix)
    return fig


def regime_timeline(price: pd.Series, segments: pd.DataFrame, order: list,
                    title: str = "", theme: str = "dark") -> go.Figure:
    """A price line over shaded regime bands, one colour per regime.

    Colours run from calm to stressed in the order the regimes are listed,
    so the eye reads danger as red whichever dimension is on screen.
    """
    pal, _ = _colors(theme)
    ramp = {2: [pal["teal"], pal["rust"]],
            3: [pal["rust"], "#C9B896", pal["teal"]],
            4: [pal["teal"], "#C9B896", "#D08C4E", pal["rust"]]}
    shades = ramp.get(len(order), SERIES_COLORS)
    color = {lab: shades[i % len(shades)] for i, lab in enumerate(order)}
    # The price line goes first: the legend placeholders below carry no
    # x values, and as the first trace they would stop Plotly from
    # recognising the axis as dates.
    fig = go.Figure(go.Scatter(x=price.index, y=price, name=str(price.name or "Index"),
                               mode="lines", line=dict(color=pal["text"], width=1.3),
                               hovertemplate="%{y:,.0f}<extra></extra>"))
    seg = segments.reset_index(drop=True)
    for i, row in seg.iterrows():
        x1 = seg.loc[i + 1, "start"] if i + 1 < len(seg) else row["end"]
        fig.add_vrect(x0=row["start"], x1=x1, fillcolor=color.get(row["label"], pal["muted"]),
                      opacity=0.22, line_width=0, layer="below")
    for lab in order:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lab,
                                 marker=dict(size=11, symbol="square",
                                             color=color[lab], opacity=0.6)))
    _base(fig, 360, title, theme)
    fig.update_xaxes(type="date")
    fig.update_yaxes(type="log")
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0))
    return fig


# ----------------------------------------------------------------------
# Diversification (qbt/clustering.py)
# ----------------------------------------------------------------------
def dendrogram(linkage: np.ndarray, labels: list, cut_height: float,
               cut_label: str = "", clusters: Optional[pd.Series] = None,
               correlation_axis: bool = True, title: str = "",
               theme: str = "dark") -> go.Figure:
    """Cluster tree, leaves listed top to bottom, with the cut line drawn.

    Each branch starts beside its label (correlation 1 on the left) and
    joins grow to the right. Branches inside a cluster of two or more take
    that cluster's colour; streams that stand alone and joins beyond the cut
    are grey. With `correlation_axis` the distance axis is labelled in the
    equivalent correlation, rho = 1 - 2 d^2.
    """
    from scipy.cluster.hierarchy import dendrogram as _dg
    pal, colors = _colors(theme)
    dg = _dg(linkage, labels=[str(x) for x in labels], no_plot=True,
             color_threshold=0, above_threshold_color="x")
    leaves = dg["ivl"]
    # Colour each link by the cluster of the leaves under it, when below the cut.
    leaf_y = {nm: 5.0 + 10.0 * i for i, nm in enumerate(leaves)}
    # Distinct colours for clusters of two or more, in tree order. The grey
    # and blue series colours are left out: grey is for streams that stand
    # alone, and the palette has no blue accent.
    cluster_color: Dict[int, str] = {}
    if clusters is not None:
        # Sand sits too close to the grey of a lone stream, so it comes last.
        hues = [c for c in colors if c not in ("#8C93A8", "#5B9EC7", "#3F3A34", "#5B7A8C")]
        hues = [c for c in hues if c not in ("#C9B896",)] + [c for c in hues if c == "#C9B896"]
        sizes = clusters.value_counts()
        for nm in leaves:
            k = int(clusters.get(nm, 0))
            if k and sizes.get(k, 0) > 1 and k not in cluster_color:
                cluster_color[k] = hues[len(cluster_color) % len(hues)]
    fig = go.Figure()
    for ic, dc in zip(dg["icoord"], dg["dcoord"]):
        color = pal["muted"]
        if clusters is not None and max(dc) <= cut_height:
            mid = (ic[0] + ic[3]) / 2.0
            nearest_leaf = min(leaves, key=lambda nm: abs(leaf_y[nm] - mid))
            k = int(clusters.get(nearest_leaf, 0))
            color = cluster_color.get(k, pal["muted"])
        fig.add_trace(go.Scatter(x=dc, y=ic, mode="lines", hoverinfo="skip",
                                 line=dict(color=color, width=1.4), showlegend=False))
    top = max([max(d) for d in dg["dcoord"]] + [cut_height]) * 1.05 + 1e-9
    fig.add_vline(x=cut_height, line=dict(color=pal["brass"], width=1.5, dash="dash"))
    if cut_label:
        fig.add_annotation(x=cut_height, y=1.0, yref="paper", text=cut_label,
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(family=MONO, size=10, color=pal["brass"]))
    _base(fig, max(300, 16 * len(leaves) + 90), title, theme)
    fig.update_layout(hovermode=False, margin=dict(l=8, r=8, t=48, b=8))
    fig.update_yaxes(tickvals=[leaf_y[nm] for nm in leaves], ticktext=leaves,
                     range=[10.0 * len(leaves), 0], showgrid=False,
                     tickfont=dict(family=MONO, size=10, color=pal["text"]))
    fig.update_xaxes(range=[0, top], showgrid=True, gridcolor=pal["rule"])
    if correlation_axis:
        low = 1.0 - 2.0 * top * top            # lowest correlation on the axis
        step = next(s for s in (0.05, 0.1, 0.2, 0.5) if (1.0 - low) / s <= 10)
        rhos = [round(1.0 - i * step, 2) for i in range(int(2.0 / step) + 1)]
        rhos = [r for r in rhos if r >= -1.0 and np.sqrt(0.5 * (1 - r)) <= top]
        fig.update_xaxes(tickvals=[float(np.sqrt(0.5 * (1 - r))) for r in rhos],
                         ticktext=[f"{r:.2f}" if step < 0.1 else f"{r:.1f}" for r in rhos],
                         title=dict(text="Correlation at which branches join",
                                    font=dict(size=11, color=pal["muted"])))
    else:
        fig.update_xaxes(title=dict(text="Ward merge distance",
                                    font=dict(size=11, color=pal["muted"])))
    return fig


def cluster_heatmap(corr: pd.DataFrame, clusters: Optional[pd.Series] = None,
                    title: str = "", theme: str = "dark") -> go.Figure:
    """Correlation matrix in the order given, with cluster blocks outlined.
    Values are printed in the cells up to 25 streams."""
    pal, _ = _colors(theme)
    labels = [str(c) for c in corr.columns]
    n = len(labels)
    show_text = n <= 25
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0.0, pal["teal"]], [0.5, pal["panel"]], [1.0, pal["rust"]]],
        zmin=-1, zmax=1, zmid=0,
        text=np.round(corr.values, 2) if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(family=MONO, size=10 if n <= 15 else 8),
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(family=MONO, size=9)),
        xgap=1 if n > 40 else 2, ygap=1 if n > 40 else 2,
    ))
    if clusters is not None:
        lab = [int(clusters.get(c, 0)) for c in labels]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and lab[j + 1] == lab[i]:
                j += 1
            if j > i:
                fig.add_shape(type="rect", x0=i - 0.5, x1=j + 0.5, y0=i - 0.5, y1=j + 0.5,
                              line=dict(color=pal["text"], width=1.5))
            i = j + 1
    _base(fig, int(min(900, max(300, (30 if n <= 25 else 12) * n + 120))), title, theme)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=False, tickangle=-45 if n > 8 else 0)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig
