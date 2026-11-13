"""Streamlit view for the Diversification tab (numbers are in qbt/clustering.py).

Used twice: Return stream > Diversification and Backtest > Diversification.
`prefix` keeps the two sets of controls apart. Every control's value lives
in `st.session_state`, so the Export tab, which runs before this one, can
rebuild the same configuration with `config_from_state` and add the same
tables to the workbook.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from . import charts as C
from . import clustering as CL

PLOT_CFG = {"displaylogo": False}
_FREQ = {"Weekly": "weekly", "Daily": "daily", "Monthly": "monthly"}
_LINK = {"Average": "average", "Ward": "ward"}
_ENB = {"Entropy": "entropy", "Participation ratio": "participation"}
_CORR = {"Pearson": "pearson", "Spearman (ranks)": "spearman"}
EQUAL = "Equal-weight average of the streams"


# ----------------------------------------------------------------------
# Markup, matching the app's own classes
# ----------------------------------------------------------------------
def _eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def _note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def _flag(text: str) -> None:
    st.markdown(f'<div class="flag">{text}</div>', unsafe_allow_html=True)


def _dial(label: str, value: str, sub: str = "") -> None:
    st.markdown(f'<div class="dial"><span class="k">{label}</span>'
                f'<span class="v">{value}</span>'
                f'{f"<div class=d>{sub}</div>" if sub else ""}</div>',
                unsafe_allow_html=True)


def _f2(v: float) -> str:
    return "n/a" if v != v else f"{v:.2f}"


# ----------------------------------------------------------------------
# Controls and configuration
# ----------------------------------------------------------------------
def _cols_key(cols: List[str]) -> str:
    return hashlib.md5("|".join(map(str, cols)).encode("utf-8")).hexdigest()[:8]


def _k(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def _inverse(d: Dict[str, str], v: str, fallback: str) -> str:
    return next((k for k, x in d.items() if x == v), fallback)


def _apply_config(prefix: str, cfg: CL.ClusteringConfig) -> None:
    """Writes a loaded config into the controls (before they are drawn)."""
    s = st.session_state
    s[_k(prefix, "freq")] = _inverse(_FREQ, cfg.frequency, "Weekly")
    s[_k(prefix, "cut")] = ("Correlation threshold" if cfg.cut == "threshold"
                            else "Number of clusters")
    s[_k(prefix, "thr")] = float(cfg.threshold)
    s[_k(prefix, "k")] = int(cfg.n_clusters)
    s[_k(prefix, "link")] = _inverse(_LINK, cfg.linkage, "Average")
    s[_k(prefix, "corr")] = _inverse(_CORR, cfg.correlation, "Pearson")
    s[_k(prefix, "minobs")] = int(cfg.min_observations)
    s[_k(prefix, "enb")] = _inverse(_ENB, cfg.effective_bets, "Entropy")
    s[_k(prefix, "pct")] = int(round(cfg.stress.percentile * 100))
    s[_k(prefix, "stressmin")] = int(cfg.stress.min_observations)
    s[_k(prefix, "roll")] = int(cfg.rolling_window)
    s[_k(prefix, "yaml_bench")] = cfg.stress.benchmark


def config_from_state(prefix: str) -> CL.ClusteringConfig:
    """The configuration the controls describe; defaults where a control
    has not been drawn yet."""
    s = st.session_state
    d = CL.ClusteringConfig()
    cut = s.get(_k(prefix, "cut"), "Correlation threshold")
    bench = s.get(_k(prefix, "bench"))
    cfg = CL.ClusteringConfig(
        frequency=_FREQ.get(s.get(_k(prefix, "freq"), ""), d.frequency),
        min_observations=int(s.get(_k(prefix, "minobs"), d.min_observations)),
        correlation=_CORR.get(s.get(_k(prefix, "corr"), ""), d.correlation),
        linkage=_LINK.get(s.get(_k(prefix, "link"), ""), d.linkage),
        cut="threshold" if cut == "Correlation threshold" else "clusters",
        threshold=float(s.get(_k(prefix, "thr"), d.threshold)),
        n_clusters=int(s.get(_k(prefix, "k"), d.n_clusters)),
        effective_bets=_ENB.get(s.get(_k(prefix, "enb"), ""), d.effective_bets),
        rolling_window=int(s.get(_k(prefix, "roll"), d.rolling_window)),
    )
    cfg.stress.percentile = int(s.get(_k(prefix, "pct"),
                                      round(d.stress.percentile * 100))) / 100.0
    cfg.stress.min_observations = int(s.get(_k(prefix, "stressmin"),
                                            d.stress.min_observations))
    cfg.stress.benchmark = None if bench in (None, EQUAL) or str(bench).startswith(
        "Screen benchmark") else bench
    if bench is None:
        cfg.stress.benchmark = s.get(_k(prefix, "yaml_bench"))
    return cfg.validate()


def selected_streams(prefix: str, cols: List[str],
                     default: Optional[List[str]] = None) -> List[str]:
    key = _k(prefix, f"streams_{_cols_key(cols)}")
    if key in st.session_state:
        return [c for c in st.session_state[key] if c in cols]
    return list(default) if default else list(cols)


# ----------------------------------------------------------------------
# Computation (cached by content)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def _analyse(rets: pd.DataFrame, cfg_yaml: str, bench: Optional[pd.Series],
             bench_label: Optional[str], start: Any, end: Any) -> CL.Diversification:
    return CL.analyse(rets, CL.config_from_yaml(cfg_yaml), bench, bench_label, start, end)


def _bench_choice(prefix: str, cfg: CL.ClusteringConfig, streams: List[str],
                  bench: Optional[pd.Series], bench_label: Optional[str]
                  ) -> Tuple[Optional[pd.Series], Optional[str]]:
    """The down-market reference the controls point to."""
    if cfg.stress.benchmark:
        return None, None              # named in the config: clustering resolves it
    choice = st.session_state.get(_k(prefix, "bench"))
    if choice == EQUAL:
        return None, None
    return bench, bench_label


def compute(rets: pd.DataFrame, prefix: str, *, bench: Optional[pd.Series] = None,
            bench_label: Optional[str] = None, start: Any = None, end: Any = None,
            default: Optional[List[str]] = None) -> CL.Diversification:
    """The analysis the tab shows, from the current controls."""
    cols = [str(c) for c in rets.columns]
    rets = rets.copy()
    rets.columns = cols
    streams = selected_streams(prefix, cols, default)
    cfg = config_from_state(prefix)
    b, bl = _bench_choice(prefix, cfg, streams, bench, bench_label)
    data = rets[streams]
    if cfg.stress.benchmark and cfg.stress.benchmark in cols \
            and cfg.stress.benchmark not in streams:
        b, bl = rets[cfg.stress.benchmark], cfg.stress.benchmark
    return _analyse(data, CL.config_to_yaml(cfg), b, bl,
                    None if start is None else pd.Timestamp(start),
                    None if end is None else pd.Timestamp(end))


def export_sections(rets: pd.DataFrame, prefix: str, **kw) -> List[Tuple[Any, ...]]:
    """Workbook sections for the Export tab, with the Diversification tab's settings."""
    return CL.export_sections(compute(rets, prefix, **kw))


def backtest_streams(run: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[pd.Series],
                                                   Optional[str], List[str]]:
    """A backtest's holdings and strategies as return streams: (returns,
    benchmark returns, benchmark label, default selection)."""
    px = run["prices"]
    rets = px.pct_change(fill_method=None).iloc[1:]
    rets.columns = [str(c) for c in rets.columns]
    holdings = list(rets.columns)
    res = run["result"]
    extra = {f"Strategy: {res.label or 'this run'}": res.returns}
    for name, r in (run.get("compare") or {}).items():
        extra[f"Strategy: {name}"] = r.returns
    bench = run.get("bench")
    if bench is not None:
        extra[f"Benchmark: {bench.label}"] = bench.returns
    for k, v in extra.items():
        rets[k] = v.reindex(rets.index)
    b = bench.returns if bench is not None else None
    bl = f"Benchmark: {bench.label}" if bench is not None else None
    return rets, b, bl, holdings


# ----------------------------------------------------------------------
# The tab
# ----------------------------------------------------------------------
def _config_files(prefix: str) -> None:
    with st.expander("Settings file (YAML)"):
        up = st.file_uploader("Load a clustering config (YAML)", type=["yaml", "yml"],
                              key=_k(prefix, "yaml"))
        if up is not None:
            digest = hashlib.md5(up.getvalue()).hexdigest()
            if st.session_state.get(_k(prefix, "yaml_done")) != digest:
                try:
                    cfg = CL.config_from_yaml(up.getvalue().decode("utf-8"))
                except (CL.ClusteringError, UnicodeDecodeError, TypeError) as exc:
                    st.error(str(exc))
                else:
                    _apply_config(prefix, cfg)
                    st.session_state[_k(prefix, "yaml_done")] = digest
                    st.rerun()
        st.download_button("Current settings (YAML)",
                           CL.config_to_yaml(config_from_state(prefix)).encode("utf-8"),
                           "clustering.yaml", "text/yaml", key=_k(prefix, "yaml_dl"))


def _controls(prefix: str, cols: List[str], default: Optional[List[str]],
              bench_label: Optional[str]) -> None:
    d = CL.ClusteringConfig()
    s = st.session_state
    # Values live in session state only (a YAML load writes them there), so
    # no control is given a default as well.
    seed = {"freq": "Weekly", "cut": "Correlation threshold", "thr": d.threshold,
            "k": d.n_clusters, "link": "Average", "corr": "Pearson",
            "minobs": d.min_observations, "enb": "Entropy", "roll": d.rolling_window,
            "pct": int(round(d.stress.percentile * 100)),
            "stressmin": d.stress.min_observations}
    for name, v in seed.items():
        s.setdefault(_k(prefix, name), v)
    s[_k(prefix, "thr")] = min(0.99, max(0.0, float(s[_k(prefix, "thr")])))
    s[_k(prefix, "k")] = min(max(1, len(cols)), max(1, int(s[_k(prefix, "k")])))
    s[_k(prefix, "pct")] = min(50, max(5, int(s[_k(prefix, "pct")])))
    s[_k(prefix, "roll")] = min(520, max(3, int(s[_k(prefix, "roll")])))
    st.multiselect(
        "Streams", cols, default=default or cols,
        key=_k(prefix, f"streams_{_cols_key(cols)}"),
        help="Every loaded stream can be included, not only the ones analysed "
             "on the other tabs.")
    c1, c2, c3, c4 = st.columns([1, 1.4, 1.2, 1])
    c1.selectbox("Frequency", list(_FREQ), key=_k(prefix, "freq"),
                 help="Weekly by default: Canadian and US markets close at "
                      "different times, which pushes daily correlations "
                      "toward zero. Never finer than the data.")
    cut = c2.radio("Group streams by", ["Correlation threshold", "Number of clusters"],
                   key=_k(prefix, "cut"), horizontal=True)
    if cut == "Correlation threshold":
        c3.slider("Same bet at or above", 0.0, 0.99, step=0.01, key=_k(prefix, "thr"),
                  help="Streams whose average correlation is at or above this "
                       "are treated as one bet.")
    else:
        c3.number_input("Clusters", 1, max(1, len(cols)), key=_k(prefix, "k"))
    c4.selectbox("Linkage", list(_LINK), key=_k(prefix, "link"),
                 help="Average: the cut line reads as a correlation. Ward: "
                      "compact groups, best with a fixed number of clusters.")
    with st.expander("More settings"):
        m1, m2, m3 = st.columns(3)
        m1.selectbox("Correlation", list(_CORR), key=_k(prefix, "corr"))
        m1.number_input("Minimum observations", 3, 5000, key=_k(prefix, "minobs"),
                        help="Per stream, and for the window they all share.")
        m2.selectbox("Effective bets measure", list(_ENB), key=_k(prefix, "enb"))
        m2.number_input("Rolling window (periods)", 3, 520, key=_k(prefix, "roll"))
        opts = ([f"Screen benchmark ({bench_label})"] if bench_label else []) + \
            [EQUAL] + cols
        yb = s.get(_k(prefix, "yaml_bench"))
        if yb in cols and _k(prefix, "bench") not in s:
            s[_k(prefix, "bench")] = yb
        if s.get(_k(prefix, "bench")) not in opts:
            s.pop(_k(prefix, "bench"), None)
        m3.selectbox("Down markets defined by", opts, key=_k(prefix, "bench"))
        m3.number_input("Down markets: worst % of periods", 5, 50, key=_k(prefix, "pct"))
        m3.number_input("Fewest down-market periods to trust", 3, 500, key=_k(prefix, "stressmin"))


def render(rets: pd.DataFrame, prefix: str, *, bench: Optional[pd.Series] = None,
           bench_label: Optional[str] = None, start: Any = None, end: Any = None,
           default: Optional[List[str]] = None, source_note: str = ""
           ) -> Optional[CL.Diversification]:
    """Draws the Diversification tab and returns the analysis shown."""
    rets = rets.copy()
    rets.columns = [str(c) for c in rets.columns]
    cols = list(rets.columns)
    default = [c for c in (default or cols) if c in cols]
    _config_files(prefix)
    # A YAML benchmark name that is a loaded column becomes the selected option.
    _controls(prefix, cols, default, bench_label)
    if len(selected_streams(prefix, cols, default)) < 2:
        _note("Pick at least two streams.")
        return None
    try:
        with st.spinner("Clustering..."):
            d = compute(rets, prefix, bench=bench, bench_label=bench_label,
                        start=start, end=end, default=default)
    except CL.ClusteringError as exc:
        st.error(str(exc))
        return None
    cfg = d.cfg
    al = d.alignment

    # -- what was used --------------------------------------------------
    for n in al.notes:
        _flag(n)
    if not al.dropped.empty:
        _flag("Left out: " + "; ".join(f"<b>{r.Stream}</b> ({r.Reason})"
                                       for r in al.dropped.itertuples()) + ".")
    _note(f"Measured on {d.window_text}, the window all {d.n_streams} streams "
          f"share. {source_note}")
    if not al.truncated.empty:
        with st.expander(f"{len(al.truncated)} streams have more history than the "
                         f"shared window"):
            st.dataframe(al.truncated, hide_index=True, use_container_width=True)

    # -- headline -------------------------------------------------------
    n_red = len(d.redundant)
    thr = cfg.threshold
    a, b, c, e, f = st.columns(5)
    with a:
        _dial("Effective bets", f"{d.effective_bets:.1f}", f"of {d.n_streams} streams")
    with b:
        _dial("Clusters", f"{d.clusters.n_clusters}",
              f"cut at {thr:.2f}" if cfg.cut == "threshold" else "fixed count")
    with c:
        _dial("Avg. correlation within", _f2(d.within), "same cluster")
    with e:
        _dial("Avg. correlation between", _f2(d.between), "different clusters")
    with f:
        _dial("Silhouette", _f2(d.clusters.silhouette), "guide only, 1 = clean split")
    singles = int((d.table["Cluster size"] == 1).sum())
    k = d.clusters.n_clusters
    lead = (f"<b>{d.n_streams} streams carry about {d.effective_bets:.1f} independent "
            f"bets in {k} cluster{'s' if k != 1 else ''}.</b> ")
    lead += (f"{n_red} pair{'s are' if n_red != 1 else ' is'} correlated at "
             f"{thr:.2f} or more" if n_red else f"No pair is correlated at {thr:.2f} or more")
    lead += (f"; {singles} stream{'s stand' if singles != 1 else ' stands'} alone."
             if singles else ".")
    if d.down_corr is not None and d.down_n >= 3:
        dc = CL.cluster_correlation(d.down_corr, d.clusters.labels)
        if dc["between"] == dc["between"] and d.between == d.between:
            lead += (f" In down markets the average correlation between clusters "
                     f"moves from {d.between:.2f} to {dc['between']:.2f}.")
    st.markdown(f'<div class="note" style="font-size:0.95rem;">{lead}</div>',
                unsafe_allow_html=True)

    # -- charts ---------------------------------------------------------
    order = d.clusters.order
    left, right = st.columns([1, 1.25])
    with left:
        _eyebrow("Cluster tree")
        cut_label = (f"same bet at {thr:.2f}" if cfg.cut == "threshold"
                     else f"{cfg.n_clusters} clusters")
        st.plotly_chart(C.dendrogram(d.clusters.linkage, list(d.corr.columns),
                                     d.clusters.cut_height, cut_label, d.clusters.labels,
                                     correlation_axis=cfg.linkage == "average"),
                        use_container_width=True, config=PLOT_CFG)
    with right:
        _eyebrow("Correlation, ordered by cluster")
        view = st.radio("Period", ["Full period", "Down markets"], horizontal=True,
                        key=_k(prefix, "view"), label_visibility="collapsed")
        if view == "Down markets":
            m = d.down_corr.loc[order, order]
            title = (f"Worst {cfg.stress.percentile:.0%} of {al.frequency} periods "
                     f"for {d.down_label}: {d.down_n} periods")
            if d.down_n < cfg.stress.min_observations:
                _flag(f"Only {d.down_n} down-market periods: below "
                      f"{cfg.stress.min_observations}, these correlations are rough.")
        else:
            m = d.corr.loc[order, order]
            title = d.window_text
        st.plotly_chart(C.cluster_heatmap(m, d.clusters.labels, title),
                        use_container_width=True, config=PLOT_CFG)

    # -- tables ---------------------------------------------------------
    _eyebrow("Cluster assignment")
    tbl = d.table.copy()
    tbl["Volatility"] = tbl["Volatility"] * 100
    tbl["Max drawdown"] = tbl["Max drawdown"] * 100
    st.dataframe(
        tbl, hide_index=True, use_container_width=True,
        column_config={
            "Correlation": st.column_config.NumberColumn("Correlation to closest",
                                                         format="%.2f"),
            "Volatility": st.column_config.NumberColumn("Volatility (ann.)",
                                                        format="%.1f%%"),
            "Max drawdown": st.column_config.NumberColumn(format="%.1f%%"),
        })
    st.download_button("Cluster table (CSV)", d.table.to_csv(index=False).encode("utf-8"),
                       "clusters.csv", "text/csv", key=_k(prefix, "dl"))
    _note(f"Volatility and drawdown are measured on the same {al.frequency} returns "
          f"and window as the correlations, so they can differ slightly from the "
          f"other tabs.")

    _eyebrow(f"Same bet: pairs correlated at {thr:.2f} or more")
    if d.redundant.empty:
        _note("None. Every pair is below the line.")
    else:
        st.dataframe(d.redundant, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ("Correlation", "Down-market correlation")})

    # -- candidate ------------------------------------------------------
    _eyebrow("Does a candidate add a new bet?")
    picked = selected_streams(prefix, cols, default)
    k1, k2 = st.columns([1, 2.2])
    others = [c for c in cols if c not in picked]
    cand = k1.selectbox("Candidate", others + [c for c in cols if c not in others],
                        key=_k(prefix, "cand"))
    base = k2.multiselect("Compare with (current holdings)", cols,
                          default=[c for c in picked if c != cand],
                          key=_k(prefix, f"base_{_cols_key(cols)}_{cand}"))
    if base:
        try:
            v = CL.candidate_test(rets, base, cand, cfg, start, end)
        except CL.ClusteringError as exc:
            st.error(str(exc))
        else:
            tone = {"duplicates": "flag", "overlaps": "flag"}.get(v.verdict, "note")
            st.markdown(f'<div class="{tone}" style="font-size:0.95rem;">'
                        f'<b>{v.sentence}</b>'
                        f'{f"<br>Measured on {v.window}." if v.window else ""}</div>',
                        unsafe_allow_html=True)
    else:
        _note("Pick at least one holding to compare with.")

    # -- pair stability -------------------------------------------------
    with st.expander("Is a relationship stable? Rolling correlation of a pair"):
        top = (list(d.redundant.iloc[0][["Stream", "Same bet as"]])
               if not d.redundant.empty else order[:2])
        p1, p2 = st.columns(2)
        sa = p1.selectbox("First", order, index=order.index(top[0]), key=_k(prefix, "pa"))
        sb = p2.selectbox("Second", order, index=order.index(top[1]), key=_k(prefix, "pb"))
        if sa != sb:
            try:
                pair = CL.align(rets[[sa, sb]], cfg, start, end)
            except CL.ClusteringError as exc:
                st.error(str(exc))
            else:
                rc = CL.rolling_correlation(pair.returns[sa], pair.returns[sb],
                                            cfg.rolling_window)
                if rc.empty:
                    _note("Not enough shared history for one rolling window.")
                else:
                    st.plotly_chart(
                        C.multi_line(rc.to_frame(f"{sa} / {sb}"),
                                     f"Rolling {cfg.rolling_window}-period correlation, "
                                     f"{al.frequency} returns", ref=thr),
                        use_container_width=True, config=PLOT_CFG)
                    _note(f"The dotted line is the {thr:.2f} threshold. A pair that "
                          f"crosses it often is not reliably the same bet, or "
                          f"reliably different.")
    return d
