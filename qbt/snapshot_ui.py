"""Streamlit views for the Markets "Snapshot" and "Screener" tabs.

Kept out of app.py (which is already five thousand lines) and driven only by
a price frame, so the numbers come from qbt/snapshot.py and the file holds
layout and formatting. Colours come from the CSS variables defined in app.py
and from qbt/brand.py; there are no hex codes here.
"""
from __future__ import annotations

import io
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import charts as C
from . import snapshot as S
from .brand import BRAND

PLOT_CFG = {"displaylogo": False}


# ----------------------------------------------------------------------
# small formatting helpers
# ----------------------------------------------------------------------
def _pct(v: float, signed: bool = True, dp: int = 1) -> str:
    if v is None or v != v:
        return "n/a"
    return f"{v * 100:+.{dp}f}%" if signed else f"{v * 100:.{dp}f}%"


def _tone(v: float) -> str:
    if v is None or v != v or v == 0:
        return ""
    return "pos" if v > 0 else "neg"


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _spark_svg(points: List[float], w: int = 150, h: int = 44) -> str:
    """A bare inline trend line, coloured by whether it ended up or down."""
    if len(points) < 2:
        return ""
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    xs = [i * (w - 2) / (len(points) - 1) + 1 for i in range(len(points))]
    ys = [h - 3 - (p - lo) / span * (h - 6) for p in points]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    col = BRAND["gain"] if points[-1] >= points[0] else BRAND["loss"]
    return (f'<svg class="hero-spark" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}"><polyline fill="none" stroke="{col}" '
            f'stroke-width="1.6" points="{d}"/></svg>')


def _pill_pick(label: str, options: List[str], default: str, key: str) -> str:
    """st.pills that cannot end up empty (clicking the active pill clears it)."""
    got = st.pills(label, options, selection_mode="single", default=default,
                   key=key, label_visibility="collapsed")
    return got or default


def _section(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Snapshot
# ----------------------------------------------------------------------
def render_snapshot(prices: pd.DataFrame, base: str,
                    label: Callable[[str], str], rf_pa: float = 0.0) -> None:
    cols = list(prices.columns)
    if not cols:
        st.info("Load a watchlist first.")
        return

    # -- selector row -------------------------------------------------
    sel = st.selectbox("Instrument", cols, key="sn_pick", format_func=label,
                       index=cols.index(base) if base in cols else 0)

    # -- one window drives the header, the chart and both tables ------
    c1, c2 = st.columns([5, 2])
    with c1:
        window = _pill_pick("Window", S.WINDOWS, "1Y", "sn_window")
    with c2:
        custom = st.checkbox("Custom dates", key="sn_custom")
    start = end = None
    if custom:
        d1, d2 = st.columns(2)
        start = d1.date_input("From", value=prices.index[-1].date()
                              - pd.Timedelta(days=365), key="sn_from",
                              min_value=prices.index[0].date(),
                              max_value=prices.index[-1].date())
        end = d2.date_input("To", value=prices.index[-1].date(), key="sn_to",
                            min_value=prices.index[0].date(),
                            max_value=prices.index[-1].date())

    win_px = S.slice_window(prices, window, start, end)
    stats = S.window_stats(win_px, rf_pa)
    if sel not in win_px.columns or stats.empty or sel not in set(stats["Instrument"]):
        st.markdown(
            f'<div class="flag">{label(sel)} does not have enough history for '
            f'this window. Pick a shorter one.</div>', unsafe_allow_html=True)
        return

    # -- hero card ----------------------------------------------------
    s_all = prices[sel].dropna()
    me = stats.set_index("Instrument").loc[sel]
    pct = {
        "Return": S.percentile_rank(stats.set_index("Instrument")["Cumulative Return"]),
        "Vol": S.percentile_rank(stats.set_index("Instrument")["Ann. Vol"], False),
        "Sharpe": S.percentile_rank(stats.set_index("Instrument")["Sharpe"]),
    }
    n_peers = len(stats)
    rank_txt = ""
    if pct["Return"].get(sel, np.nan) == pct["Return"].get(sel, np.nan):
        rank_txt = (f'<span class="rank">{_ordinal(pct["Return"][sel])} %ile '
                    f'<span class="vs">vs {n_peers} on this watchlist</span></span>')
    day = s_all.iloc[-1] / s_all.iloc[-2] - 1.0 if len(s_all) > 1 else np.nan
    hi52 = s_all.loc[s_all.index[-1] - pd.DateOffset(months=12):].max()
    off_hi = s_all.iloc[-1] / hi52 - 1.0 if hi52 else np.nan
    trend = S.trend_points(win_px[sel])
    ann_txt = _pct(me["Ann. Return"]) if me["Ann. Return"] == me["Ann. Return"] else "n/a"

    def kv(k: str, v: str, tone: str = "") -> str:
        return (f'<div class="kv"><span class="k">{k}</span>'
                f'<span class="v {tone}">{v}</span></div>')

    st.markdown(
        '<div class="hero">'
        '<div class="hero-main">'
        f'<div class="hero-tkr">{sel}</div>'
        f'<div class="hero-name">{label(sel)}</div>'
        f'<div class="hero-row">'
        f'{kv("Last", f"{s_all.iloc[-1]:,.2f}")}'
        f'{kv("1 day", _pct(day), _tone(day))}'
        f'{kv(window + " return", _pct(me["Cumulative Return"]), _tone(me["Cumulative Return"]))}'
        f'{kv("From 52w high", _pct(off_hi), _tone(off_hi))}'
        f'{kv("History", f"{s_all.index[0].date()}")}'
        f'</div>{rank_txt}</div>'
        f'<div class="hero-side">{_spark_svg(trend)}'
        f'<div class="hero-cap">{window} trend</div></div>'
        '</div>', unsafe_allow_html=True)

    # -- chart --------------------------------------------------------
    _section("Performance")
    m1, m2, m3 = st.columns([3, 2, 3])
    mode = m1.segmented_control("Mode", S.CHART_MODES, default="Cumulative",
                                key="sn_mode", label_visibility="collapsed") \
        or "Cumulative"
    scale = m2.segmented_control("Scale", ["Linear", "Log"], default="Linear",
                                 key="sn_scale", label_visibility="collapsed") \
        or "Linear"
    others = m3.multiselect("Compare with", [c for c in cols if c != sel],
                            default=[base] if base in cols and base != sel else [],
                            key="sn_cmp", format_func=label,
                            label_visibility="collapsed",
                            placeholder="Compare with...")
    series = [sel] + [c for c in others if c in win_px.columns]
    # Compare on the period every selection shares, as the rest of Markets does.
    shared = win_px[series].dropna()
    if len(shared) < 3:
        st.markdown('<div class="flag">The selected instruments share no '
                    'history in this window.</div>', unsafe_allow_html=True)
    else:
        if mode == "Cumulative" and scale == "Log":
            reb = shared / shared.iloc[0] * 100.0
            fig = C.equity_curve(reb.rename(columns={c: label(c) for c in reb}),
                                 True, f"Growth of 100, since {shared.index[0].date()}")
        else:
            curves = S.rebased_curves(shared, mode)
            curves = curves.rename(columns={c: label(c) for c in curves})
            title = {"Cumulative": f"Cumulative return, since {shared.index[0].date()}",
                     "Drawdown": "Drawdown from prior peak",
                     "Rolling": "Rolling 126-session return"}[mode]
            fig = C.multi_line(curves, title, "%", ref=0.0)
            fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    # -- summary table with percentile bars ---------------------------
    _section(f"Summary, {window}")
    tbl = stats.copy()
    tbl["Return %ile"] = S.percentile_rank(tbl.set_index("Instrument")["Cumulative Return"]).values
    tbl["Vol %ile"] = S.percentile_rank(tbl.set_index("Instrument")["Ann. Vol"], False).values
    tbl["Sharpe %ile"] = S.percentile_rank(tbl.set_index("Instrument")["Sharpe"]).values
    tbl = tbl.sort_values("Cumulative Return", ascending=False)
    show = pd.DataFrame({
        "Instrument": tbl["Instrument"].map(label),
        "Cumulative": tbl["Cumulative Return"] * 100,
        "Annualized": tbl["Ann. Return"] * 100,
        "Volatility": tbl["Ann. Vol"] * 100,
        "Sharpe": tbl["Sharpe"],
        "Max DD": tbl["Max DD"] * 100,
        "Return %ile": tbl["Return %ile"],
        "Vol %ile": tbl["Vol %ile"],
        "Sharpe %ile": tbl["Sharpe %ile"],
    })
    pct_cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100,
                                                  format="%d")
               for c in ("Return %ile", "Vol %ile", "Sharpe %ile")}
    st.dataframe(
        show, hide_index=True, use_container_width=True,
        height=min(520, 60 + 36 * len(show)),
        column_config={
            "Cumulative": st.column_config.NumberColumn(format="%+.1f%%"),
            "Annualized": st.column_config.NumberColumn(format="%+.1f%%"),
            "Volatility": st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Max DD": st.column_config.NumberColumn(format="%.1f%%"),
            **pct_cfg})
    if len(tbl) < S.MIN_PEERS:
        st.caption(f"Percentiles appear once the watchlist has at least "
                   f"{S.MIN_PEERS} instruments with history in the window; "
                   f"a rank among {len(tbl)} is not a percentile.")
    else:
        st.caption("Percentile is among the instruments on this watchlist "
                   "over the same window; 100 is best (for volatility, "
                   "lowest). Annualized is blank for windows under a year.")

    # -- calendar returns ---------------------------------------------
    _section("Calendar returns")
    view = st.segmented_control("Calendar view", ["Table", "Heatmap", "Monthly"],
                                default="Heatmap", key="sn_cal",
                                label_visibility="collapsed") or "Heatmap"
    if view == "Monthly":
        g = S.monthly_grid(prices[sel])
        if g.empty:
            st.caption("Not enough history for a monthly grid.")
        else:
            r = prices[sel].dropna().pct_change().dropna()
            st.plotly_chart(C.monthly_heatmap(r, f"{label(sel)}, monthly returns"),
                            use_container_width=True, config=PLOT_CFG)
    else:
        tab, part = S.annual_returns(prices)
        if tab.empty:
            st.caption("Not enough history for calendar years.")
        else:
            tab = tab.rename(index=label)
            part = part.rename(index=label)
            if view == "Heatmap":
                st.plotly_chart(C.annual_heatmap(tab, part, "Calendar-year returns, %"),
                                use_container_width=True, config=PLOT_CFG)
            else:
                disp = (tab * 100).copy()
                disp.columns = [str(c) for c in disp.columns]
                disp = disp.reset_index().rename(columns={"index": "Instrument"})
                st.dataframe(disp, hide_index=True, use_container_width=True,
                             column_config={c: st.column_config.NumberColumn(
                                 format="%+.1f%%") for c in disp.columns[1:]})
            st.caption("* A partial year: the first year of the series, or the "
                       "year in progress.")


# ----------------------------------------------------------------------
# Screener
# ----------------------------------------------------------------------
SCREEN_COLUMNS = ["Trend", "1D", "1M", "3M", "YTD", "1Y", "3Y (ann.)",
                  "Volatility", "Sharpe", "Max DD", "From 52w High",
                  "1Y %ile", "Sharpe %ile", "Vol %ile"]
DEFAULT_COLUMNS = ["Trend", "1M", "3M", "YTD", "1Y", "Volatility", "Sharpe",
                   "From 52w High", "1Y %ile"]
_PCT_COLS = {"1D", "1M", "3M", "YTD", "1Y", "3Y (ann.)", "Volatility",
             "Max DD", "From 52w High"}
_PERCENTILE_COLS = {"1Y %ile", "Sharpe %ile", "Vol %ile"}


def _screen_view(df: pd.DataFrame, label: Callable[[str], str],
                 columns: List[str]) -> pd.DataFrame:
    out = pd.DataFrame({"Instrument": df["Instrument"], "Name": df["Instrument"].map(label)})
    for c in columns:
        out[c] = df[c] * 100 if c in _PCT_COLS else df[c]
    return out


def _excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.drop(columns=[c for c in df if c == "Trend"]).to_excel(
            xw, sheet_name="Screener", index=False)
    return buf.getvalue()


def render_screener(prices: pd.DataFrame, label: Callable[[str], str],
                    rf_pa: float = 0.0, key_prefix: str = "sc") -> None:
    if prices.empty:
        st.info("Load a watchlist first.")
        return

    trend_win = _pill_pick("Trend window", ["3M", "6M", "1Y", "3Y", "Max"],
                           "1Y", f"{key_prefix}_trend")
    df = S.screener_table(prices, trend_win, rf_pa)
    if df.empty:
        st.markdown('<div class="flag">No instrument has enough history to '
                    'screen (at least 30 observations).</div>',
                    unsafe_allow_html=True)
        return

    f1, f2, f3 = st.columns([3, 3, 2])
    q = f1.text_input("Filter", key=f"{key_prefix}_q", placeholder="Filter by ticker or name...",
                      label_visibility="collapsed")
    picked = f2.multiselect("Columns", SCREEN_COLUMNS, default=DEFAULT_COLUMNS,
                            key=f"{key_prefix}_cols", label_visibility="collapsed",
                            placeholder="Columns")
    sort_by = f3.selectbox("Sort by", [c for c in SCREEN_COLUMNS if c != "Trend"],
                           index=SCREEN_COLUMNS.index("1Y") - 1, key=f"{key_prefix}_sort",
                           label_visibility="collapsed")
    s1, s2, s3 = st.columns([2, 3, 3])
    desc = s1.checkbox("Descending", value=True, key=f"{key_prefix}_desc")
    min_1y = s2.number_input("Minimum 1Y return (%)", value=-100.0, step=5.0,
                             key=f"{key_prefix}_min1y")
    max_vol = s3.number_input("Maximum volatility (%)", value=500.0, step=5.0,
                              key=f"{key_prefix}_maxvol")

    view = df.copy()
    if q.strip():
        qq = q.strip().lower()
        view = view[view["Instrument"].str.lower().str.contains(qq, regex=False)
                    | view["Instrument"].map(label).str.lower().str.contains(qq, regex=False)]
    view = view[(view["1Y"].fillna(-9) * 100 >= min_1y)
                & (view["Volatility"].fillna(0) * 100 <= max_vol)]
    view = view.sort_values(sort_by, ascending=not desc, na_position="last")

    cols = [c for c in SCREEN_COLUMNS if c in picked]
    st.markdown(f'<div class="runbar"><span class="lead">Showing '
                f'{len(view)} of {len(df)}</span>'
                f'<span class="item">sorted by {sort_by}</span>'
                f'<span class="item">trend: {trend_win}</span></div>',
                unsafe_allow_html=True)
    if view.empty:
        st.caption("Nothing matches those filters.")
        return

    # Pinned median row, as Sparkline shows above its table.
    med = S.summary_row(view, [c for c in cols if c != "Trend"])
    chips = "".join(
        f'<span class="item">{c} <b>{_pct(med[c], dp=1) if c in _PCT_COLS else f"{med[c]:.2f}" if med[c]==med[c] else "n/a"}</b></span>'
        for c in cols if c not in ("Trend",) and c not in _PERCENTILE_COLS)
    if chips:
        st.markdown(f'<div class="runbar"><span class="lead">Median of shown</span>'
                    f'{chips}</div>', unsafe_allow_html=True)

    show = _screen_view(view, label, cols)
    cfg = {"Instrument": st.column_config.TextColumn("Ticker", width="small"),
           "Name": st.column_config.TextColumn(width="medium")}
    for c in cols:
        if c == "Trend":
            cfg[c] = st.column_config.LineChartColumn(f"Trend ({trend_win})",
                                                      width="small")
        elif c in _PERCENTILE_COLS:
            cfg[c] = st.column_config.ProgressColumn(c, min_value=0, max_value=100,
                                                     format="%d")
        elif c in _PCT_COLS:
            cfg[c] = st.column_config.NumberColumn(c, format="%+.1f%%"
                                                   if c not in ("Volatility", "Max DD")
                                                   else "%.1f%%")
        else:
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f")
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config=cfg, height=min(640, 60 + 36 * len(show)))
    st.caption("Returns are trailing; 3Y is annualized and blank when the history "
               "is shorter. Volatility, Sharpe and Max DD cover the whole loaded "
               "period, so they differ from the window figures on the Snapshot "
               "tab. Percentiles rank each instrument among this watchlist.")
    if S.MIN_PEERS > len(df):
        st.caption(f"Percentile columns stay blank below {S.MIN_PEERS} "
                   f"instruments with history.")

    d1, d2 = st.columns(2)
    d1.download_button("Screen (CSV)",
                       _screen_view(view, label, [c for c in cols if c != "Trend"])
                       .to_csv(index=False).encode("utf-8"),
                       "watchlist_screen.csv", "text/csv", key=f"{key_prefix}_csv")
    try:
        d2.download_button("Screen (Excel)", _excel(_screen_view(view, label, cols)),
                           "watchlist_screen.xlsx",
                           "application/vnd.openxmlformats-officedocument."
                           "spreadsheetml.sheet", key=f"{key_prefix}_xlsx")
    except Exception:
        pass

    # -- distribution -------------------------------------------------
    with st.expander("Distribution", expanded=False):
        metric_opts = [c for c in SCREEN_COLUMNS if c not in ("Trend",)]
        dm = st.selectbox("Metric", metric_opts,
                          index=metric_opts.index("1Y"), key=f"{key_prefix}_dist")
        vals = pd.to_numeric(df[dm], errors="coerce").dropna()
        if dm in _PCT_COLS:
            vals = vals * 100
        if len(vals) < 3:
            st.caption("Too few values for a distribution.")
        else:
            fig = go.Figure(go.Histogram(
                x=vals, nbinsx=min(30, max(6, len(vals) // 2)),
                marker=dict(color=BRAND["muted"], line=dict(color=BRAND["rule"], width=1)),
                hovertemplate="%{x}: %{y}<extra></extra>"))
            fig.add_vline(x=float(vals.median()),
                          line=dict(color=BRAND["red"], width=1.5, dash="dot"),
                          annotation_text="median", annotation_font_color=BRAND["red"])
            C._base(fig, 280, f"{dm}, {len(vals)} instruments"
                    + (" (%)" if dm in _PCT_COLS else ""))
            fig.update_layout(bargap=0.05, hovermode="closest", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
