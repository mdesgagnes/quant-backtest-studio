"""Streamlit view for the Markets "Groups" tab (numbers are in qbt/groups.py).

Driven by a loader callable, so this file holds layout and formatting only:
`loader(tickers)` returns (prices, tickers_not_loaded). The dates come from
the Markets sidebar, which the loader closes over.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from . import charts as C
from . import groups as G
from . import presets as PRESETS
from . import snapshot as S
from .snapshot_ui import PLOT_CFG, _pill_pick, _section, render_screener

DEFAULT_MY_GROUPS = (
    "Canadian equity: XIC.TO, XEI.TO, XDV.TO, ZEB.TO\n"
    "U.S. equity: SPY, QQQ, IWM, DIA\n"
    "International: EFA, EEM, VGK\n"
    "Bonds: XBB.TO, TLT, HYG, LQD\n"
    "Real assets: GLD, VNQ, XRE.TO, DBC"
)


def _signature(src, defn) -> tuple:
    return (src, tuple((k, tuple(v)) for k, v in defn.items()))


def render_groups(loader: Callable[[tuple], tuple],
                  label: Callable[[str], str], rf_pa: float = 0.0) -> None:
    # -- 1. define ----------------------------------------------------
    src = _pill_pick("Groups from", ["Presets", "Asset classes", "My groups"],
                     "Presets", "gr_src")
    names = PRESETS.names()
    if src == "Presets":
        chosen = st.multiselect("Presets to include", names, default=names[:4],
                                key="gr_presets")
        defn = G.groups_from_presets(chosen)
        st.caption("Each preset is one group.")
    elif src == "Asset classes":
        chosen = st.multiselect(
            "Presets to pool", names,
            default=[n for n in ("Global asset classes", "Canadian asset classes")
                     if n in names], key="gr_pool")
        defn = G.groups_by_class(chosen)
        st.caption("The pooled instruments are regrouped by the asset-class tag "
                   "each preset gives them. Untagged instruments are left out "
                   "rather than guessed at.")
    else:
        txt = st.text_area("One group per line, as  Name: ticker, ticker, ...",
                           value=DEFAULT_MY_GROUPS, height=150, key="gr_text")
        defn = G.parse_groups(txt)

    tickers = G.union_tickers(defn)
    c1, c2 = st.columns([2, 5])
    go_load = c1.button("Load groups", key="gr_load", disabled=not tickers)
    c2.caption(f"{len(defn)} groups, {len(tickers)} instruments. Prices cover the "
               f"dates set in the sidebar.")

    if go_load and tickers:
        with st.spinner(f"Loading {len(tickers)} instruments..."):
            try:
                px, missing = loader(tuple(tickers))
                st.session_state["gr_data"] = {
                    "prices": px, "groups": defn, "missing": list(missing),
                    "sig": _signature(src, defn)}
            except Exception as exc:
                st.error(f"Could not load: {exc}")

    data = st.session_state.get("gr_data")
    if not data:
        st.markdown('<div class="note">Choose the groups and load them. Each '
                    'group is summarised the way a peer group is on Sparkline: '
                    'how many members, a statistic of their returns over each '
                    'horizon, and a trend, so groups can be ranked against each '
                    'other.</div>', unsafe_allow_html=True)
        return

    px, loaded = data["prices"], data["groups"]
    if data["sig"] != _signature(src, defn):
        st.markdown('<div class="flag">The group definition has changed since '
                    'it was loaded. Press Load groups to refresh; what is '
                    'shown below is the earlier selection.</div>',
                    unsafe_allow_html=True)
    if data["missing"]:
        st.markdown('<div class="flag">No usable prices for: '
                    + ", ".join(data["missing"][:20])
                    + ("..." if len(data["missing"]) > 20 else "")
                    + ". They are left out of the group figures.</div>",
                    unsafe_allow_html=True)

    # -- 2. the taxonomy table ----------------------------------------
    o1, o2 = st.columns(2)
    stat = o1.selectbox("Statistic", list(G.STATS), key="gr_stat")
    sort_opts = ["1M", "3M", "YTD", "1Y", "3Y", "5Y", "Volatility", "Sharpe",
                 "Members"]
    sort_by = o2.selectbox("Sort by", sort_opts, index=sort_opts.index("1Y"),
                           key="gr_sort")
    tw = _pill_pick("Trend window", ["3M", "6M", "1Y", "3Y", "Max"], "1Y",
                    "gr_trend")
    tbl = G.group_table(px, loaded, stat, tw, rf_pa)
    if tbl.empty:
        st.markdown('<div class="flag">None of these groups has an instrument '
                    'with enough history.</div>', unsafe_allow_html=True)
        return
    tbl = tbl.sort_values(sort_by, ascending=False, na_position="last")

    show = pd.DataFrame({
        "Group": [g + (" †" if sm else "")
                  for g, sm in zip(tbl["Group"], tbl["Small"])],
        "Members": tbl["Members"],
        "Trend": tbl["Trend"],
        **{h: tbl[h] * 100 for h in G.HORIZON_COLS},
        "Volatility": tbl["Volatility"] * 100,
        "Sharpe": tbl["Sharpe"],
        "Max DD": tbl["Max DD"] * 100,
        "1Y spread": tbl["1Y spread"] * 100,
        "Best (1Y)": tbl["Best (1Y)"],
        "Worst (1Y)": tbl["Worst (1Y)"],
        "Largest": tbl["Largest"],
    })
    cfg = {"Group": st.column_config.TextColumn(width="medium"),
           "Members": st.column_config.NumberColumn(format="%d", width="small"),
           "Trend": st.column_config.LineChartColumn(f"Trend ({tw})", width="small"),
           "Sharpe": st.column_config.NumberColumn(format="%.2f")}
    for c in G.HORIZON_COLS:
        cfg[c] = st.column_config.NumberColumn(
            f"{c} ann." if c in ("3Y", "5Y") else c, format="%+.1f%%")
    for c in ("Volatility", "Max DD", "1Y spread"):
        cfg[c] = st.column_config.NumberColumn(format="%.1f%%")

    st.markdown(f'<div class="runbar"><span class="lead">{len(tbl)} groups</span>'
                f'<span class="item">{stat.lower()} of members</span>'
                f'<span class="item">sorted by {sort_by}</span></div>',
                unsafe_allow_html=True)
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config=cfg, height=min(560, 60 + 36 * len(show)))
    st.caption(f"Each figure is the {stat.lower()} over the members that have "
               f"history for that horizon; 3Y and 5Y are annualized and blank "
               f"when no member reaches back that far. † marks a group "
               f"with fewer than {G.SMALL_SAMPLE} members, where a statistic "
               f"of the group is not robust. Volatility, Sharpe and Max DD "
               f"are the median member over the whole loaded period.")
    st.download_button("Groups (CSV)",
                       show.drop(columns=["Trend"]).to_csv(index=False).encode("utf-8"),
                       "groups.csv", "text/csv", key="gr_csv")

    # -- 3. groups against each other ---------------------------------
    _section("Group performance")
    gnames = list(tbl["Group"])
    cc1, cc2 = st.columns([4, 3])
    pick = cc1.multiselect("Groups", gnames, default=gnames[:4], key="gr_cmp",
                           label_visibility="collapsed", placeholder="Groups")
    mode = cc2.segmented_control("Mode", S.CHART_MODES, default="Cumulative",
                                 key="gr_mode",
                                 label_visibility="collapsed") or "Cumulative"
    win = _pill_pick("Window", S.WINDOWS, "1Y", "gr_win")
    if pick:
        curves = G.group_curves(px, loaded, pick, win, mode)
        if curves.empty:
            st.markdown('<div class="flag">These groups share no history in '
                        'this window.</div>', unsafe_allow_html=True)
        else:
            title = {"Cumulative": "Equal-weight group index, cumulative return",
                     "Drawdown": "Equal-weight group index, drawdown",
                     "Rolling": "Equal-weight group index, rolling 126-session "
                                "return"}[mode]
            fig = C.multi_line(curves, title, "%", ref=0.0)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
            st.caption("Each line is the group's members equally weighted and "
                       "rebalanced daily, over the period they all share. It "
                       "describes the group; it is not a portfolio.")

    # -- 4. one group's members, as a screen --------------------------
    _section("Inside a group")
    inside = st.selectbox("Group", gnames, key="gr_inside",
                          label_visibility="collapsed")
    members = [m for m in loaded.get(inside, []) if m in px.columns]
    if not members:
        st.caption("No members with data.")
    else:
        # Its own prefix: the group table above already owns gr_trend.
        render_screener(px[members], label, rf_pa, key_prefix="grm")
