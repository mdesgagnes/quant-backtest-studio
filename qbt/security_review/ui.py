"""Streamlit page for the Security review. Thin by design: it collects the
inputs, calls `engine.run` / `engine.sweep`, and renders `report.sections`
(the same tables the Excel export writes). No calculation lives here.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..brand import BRAND, RED
from . import config as CF
from . import engine as E
from . import export as X
from . import report as RP
from .bundle import load_bundle

# Red, then the sand / gold family: no blue (house colour scheme).
PALETTE = [RED, "#C9B896", "#D08C4E", "#EFE9DE", "#9A9389", "#B01419", "#CBBFA8", "#8C6D4F"]
VERDICT_COLOUR = {"Add": BRAND["gain"], "Consider": "#C9B896", "Do not add": BRAND["loss"]}
CUSTOM = "Custom…"


@st.cache_resource(show_spinner=False)
def _config() -> CF.AllConfig:
    return CF.load_all()


@st.cache_resource(show_spinner="Loading prices (cached on disk after the first run)…", ttl=6 * 3600)
def _bundle(tickers: tuple):
    cfg = _config()
    b = load_bundle(cfg, list(tickers))
    return b, E.regime_labels(cfg, b)


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
_PY = {RP.PCT: "{:.2%}", RP.PTS: "{:+.2%}", RP.RATIO: "{:.2f}", RP.INT: "{:.0f}",
       RP.MONEY: "{:,.0f}"}


def _text(df: pd.DataFrame) -> pd.DataFrame:
    """The numbers of a report table as the text the workbook shows."""
    out = df.astype(object).copy()
    col_fmt = df.attrs.get("col_fmt") or {}
    row_fmt = df.attrs.get("row_fmt")

    def f(v, fmt):
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            return "n/a" if v != v else _PY.get(fmt, "{:.4g}").format(v)
        return v

    if row_fmt:
        cols = [df.columns[i - 1] for i in (df.attrs.get("fmt_cols") or range(2, df.shape[1] + 1))]
        for i, fmt in enumerate(row_fmt):
            for c in cols:
                out.iat[i, df.columns.get_loc(c)] = f(df.iat[i, df.columns.get_loc(c)], fmt)
    for c in df.columns:
        if c in col_fmt or (not row_fmt and pd.api.types.is_float_dtype(df[c])):
            out[c] = [f(v, col_fmt.get(c, RP.RATIO)) for v in df[c]]
    return out.astype(object).where(out.notna(), "")


def _show(secs: List[RP.Section]) -> None:
    for _, title, df, note in secs:
        if df is None or df.empty:
            continue
        st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)
        st.dataframe(_text(df), hide_index=True, use_container_width=True)
        if note:
            st.caption(note)


def _fig(height=340, title="") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      paper_bgcolor=BRAND["bg"], plot_bgcolor=BRAND["bg"],
                      font=dict(color=BRAND["text"], size=11),
                      margin=dict(l=40, r=20, t=40 if title else 10, b=30),
                      legend=dict(orientation="h", y=-0.15))
    return fig


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def _parse_weights(text: str) -> Dict[str, float]:
    out = {}
    for line in text.replace(";", "\n").splitlines():
        parts = line.replace(",", " ").replace(":", " ").replace("\t", " ").split()
        if len(parts) >= 2:
            try:
                w = float(parts[1].rstrip("%"))
            except ValueError:
                continue
            out[parts[0].upper()] = w / 100 if parts[1].endswith("%") or w > 1.0 else w
    return out


def _weights_file(up) -> Dict[str, float]:
    raw = up.getvalue()
    df = (pd.read_excel(io.BytesIO(raw)) if up.name.lower().endswith((".xlsx", ".xls"))
          else pd.read_csv(io.BytesIO(raw)))
    df = df.iloc[:, :2].dropna()
    w = {str(t).strip().upper(): float(x) for t, x in zip(df.iloc[:, 0], df.iloc[:, 1])}
    if sum(w.values()) > 1.5:          # given in percent
        w = {t: x / 100 for t, x in w.items()}
    return w


def _sidebar(cfg: CF.AllConfig):
    sb = st.sidebar
    sb.markdown('<div class="eyebrow">1. Investor profile</div>', unsafe_allow_html=True)
    presets = [k for k, p in cfg.profiles.profiles.items() if not p.custom]
    saved = [k for k, p in cfg.profiles.profiles.items() if p.custom]
    keys = presets + saved + [CUSTOM]
    label = {k: cfg.profiles.profiles[k].describe() if k in cfg.profiles.profiles else k for k in keys}
    label[CUSTOM] = "Custom (enter or upload weights)"
    pk = sb.selectbox("Profile", keys, index=keys.index("balanced") if "balanced" in keys else 0,
                      format_func=lambda k: label[k], key="sr_profile")
    profile: Optional[CF.Profile] = cfg.profiles.profiles.get(pk)
    if pk == CUSTOM:
        how = sb.radio("Weights from", ["Type them", "Upload a file"], horizontal=True, key="sr_cust_how")
        w: Dict[str, float] = {}
        if how == "Type them":
            txt = sb.text_area("One holding per line: ticker weight", "XIC.TO 30\nVFV.TO 30\nXBB.TO 40",
                               key="sr_cust_txt", height=110)
            w = _parse_weights(txt)
        else:
            up = sb.file_uploader("CSV or Excel: ticker, weight", type=["csv", "xlsx", "xls"],
                                  key="sr_cust_file")
            if up is not None:
                try:
                    w = _weights_file(up)
                except Exception as exc:
                    sb.error(f"Could not read the file: {exc}")
        name = sb.text_input("Name", "My portfolio", key="sr_cust_name")
        try:
            profile = CF.custom_profile(name, w)
            sb.caption(" · ".join(f"{t} {x:.0%}" for t, x in profile.weights.items()))
            if sb.button("Save as a profile", key="sr_cust_save"):
                path = CF.save_custom_profile(profile)
                _config.clear()
                sb.success(f"Saved to {path}.")
        except CF.ConfigError as exc:
            sb.warning(str(exc))
            profile = None
    reb_default = profile.rebalance if profile else cfg.profiles.rebalance_default
    reb = sb.selectbox("Rebalancing", list(CF.REBALANCE_OPTIONS),
                       index=list(CF.REBALANCE_OPTIONS).index(reb_default), key="sr_reb")

    sb.markdown('<div class="eyebrow">2. Objective</div>', unsafe_allow_html=True)
    objs = list(cfg.objectives.objectives)
    ok = sb.selectbox("Objective", objs, format_func=lambda k: cfg.objectives.objectives[k].label,
                      key="sr_obj")
    sb.caption(cfg.objectives.objectives[ok].description)

    sb.markdown('<div class="eyebrow">3. Candidates</div>', unsafe_allow_html=True)
    sleeves = cfg.universe.sleeves
    fill = sb.selectbox("Fill with a sleeve", ["(keep my picks)"] + list(sleeves),
                        format_func=lambda k: sleeves[k].label if k in sleeves else k, key="sr_fill")
    if fill != st.session_state.get("_sr_fill_applied"):
        st.session_state["_sr_fill_applied"] = fill
        if fill in sleeves:
            st.session_state["sr_cands"] = list(sleeves[fill].tickers)
    st.session_state.setdefault("sr_cands", ["USMV", "SPLV", "QUAL", "SPY"])
    universe = cfg.universe.all_tickers()
    extra_now = [t for t in st.session_state["sr_cands"] if t not in universe]
    cands = sb.multiselect("From the universe", universe + extra_now, key="sr_cands",
                           format_func=lambda t: f"{t} ({cfg.universe.sleeve_of(t).label})"
                           if cfg.universe.sleeve_of(t) else t)
    typed = sb.text_input("Or type tickers (comma-separated)", "", key="sr_typed",
                          help="Any Yahoo Finance symbol: .TO for Toronto listings.")
    cands = list(dict.fromkeys(cands + [t.strip().upper() for t in typed.split(",") if t.strip()]))

    sb.markdown('<div class="eyebrow">4. Sleeve benchmark</div>', unsafe_allow_html=True)
    auto = sorted({cfg.universe.benchmark_for(c) or "profile" for c in cands})
    sb.caption("Auto: " + (", ".join(auto) if auto else "n/a") + " (each candidate's sleeve).")
    bench = sb.text_input("Override for every candidate (blank = auto)", "", key="sr_bench").strip().upper()

    sb.markdown('<div class="eyebrow">5. Add or replace</div>', unsafe_allow_html=True)
    mode = sb.radio("Mode", ["Add", "Replace"], horizontal=True, key="sr_mode")
    weight, holding = None, None
    if mode == "Add":
        weight = sb.slider("Add weight (funded pro rata)", 1, 40,
                           int(round(cfg.objectives.add_weight * 100)), key="sr_w") / 100
    elif profile:
        hs = list(profile.weights)
        holding = sb.selectbox("Holding to replace", hs, key="sr_hold",
                               format_func=lambda t: f"{t} ({profile.weights[t]:.0%})")
    with sb.expander("Advanced"):
        align = st.radio("Analysis period", ["common", "each"], key="sr_align",
                         format_func=lambda a: "Common to all candidates" if a == "common"
                         else "Each candidate's own history",
                         help="Common: every candidate is judged on the same months (fair ranking). "
                              "Each: a long-history candidate keeps its full record.")
        sweep_on = st.checkbox("Also run a profile sweep", key="sr_sweep_on",
                               help="The first candidate across all 7 preset profiles.")
    run = sb.button("Run review", type="primary", use_container_width=True, key="sr_run")
    return dict(profile=profile, rebalance=reb, objective=ok, candidates=cands,
                benchmark=bench or None, mode=mode.lower(), weight=weight, holding=holding,
                alignment=align, sweep=sweep_on, run=run)


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
def _card(c: str, r: E.CandidateResult) -> str:
    d = r.decision
    col = VERDICT_COLOUR.get(d["verdict"], BRAND["muted"])
    facts = "".join(f"<li>{f}</li>" for f in r.rationale["facts"])
    return (f'<div style="border:1px solid {BRAND["rule"]};border-top:3px solid {col};'
            f'background:{BRAND["panel"]};padding:.7rem .9rem;margin-bottom:.6rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-size:1.15rem;font-weight:600;">{c}</span>'
            f'<span style="color:{col};font-weight:700;font-size:1.05rem;">{d["verdict"]}</span></div>'
            f'<div style="color:{BRAND["muted"]};font-size:.8rem;">Score {d["composite"]:+.2f} · '
            f'Confidence {d["confidence"]} · {r.months} months · vs {r.benchmark_label}</div>'
            f'<ul style="margin:.4rem 0 .3rem 1rem;font-size:.85rem;">{facts}</ul>'
            f'<div style="font-size:.85rem;"><b>Trade-off.</b> {r.rationale["tradeoff"]}</div></div>')


def _charts(rv: E.Review) -> None:
    if not rv.results:
        return
    fig = _fig(360, "Growth of 1: profile with each candidate (CAD)")
    first = next(iter(rv.results.values()))
    base = first.series["Portfolio without"].dropna()
    fig.add_scatter(x=base.index, y=(1 + base).cumprod(), name="Profile without",
                    line=dict(color=BRAND["muted"], width=2, dash="dot"))
    for i, (c, r) in enumerate(rv.results.items()):
        s = r.series["Portfolio with"].dropna()
        fig.add_scatter(x=s.index, y=(1 + s).cumprod(), name=f"with {c}",
                        line=dict(color=PALETTE[i % len(PALETTE)], width=1.6))
    st.plotly_chart(fig, use_container_width=True)
    fig = _fig(300, "Drawdown of the profile with each candidate")
    eq = (1 + base).cumprod()
    fig.add_scatter(x=eq.index, y=eq / eq.cummax() - 1, name="Profile without",
                    line=dict(color=BRAND["muted"], dash="dot"))
    for i, (c, r) in enumerate(rv.results.items()):
        e = (1 + r.series["Portfolio with"].dropna()).cumprod()
        fig.add_scatter(x=e.index, y=e / e.cummax() - 1, name=f"with {c}",
                        line=dict(color=PALETTE[i % len(PALETTE)], width=1.4))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st_rows = [(c, r.stress) for c, r in rv.results.items()]
    fig = _fig(320, "Stress windows: change in profile return from adding each candidate")
    for i, (c, t) in enumerate(st_rows):
        ok = t[t["Status"] == "ok"] if "Status" in t else t
        if len(ok):
            fig.add_bar(x=ok["Window"], y=ok["Change in portfolio return"], name=c,
                        marker_color=PALETTE[i % len(PALETTE)])
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)
    obj = _config().objectives.objectives[rv.objective_key]
    if obj.rolling_metric:
        fig = _fig(300, f"Rolling 36-month {obj.rolling_metric} (with minus without)")
        for i, (c, r) in enumerate(rv.results.items()):
            s = r.rolling.get(obj.rolling_metric, {}).get("series")
            if s is not None and len(s):
                fig.add_scatter(x=s.index, y=s, name=c, line=dict(color=PALETTE[i % len(PALETTE)]))
        st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    cfg = _config()
    st.markdown('<div class="masthead"><div><h1>Security review</h1>'
                '<div class="sub">Does this security solve a real problem for this investor, '
                'and which candidate solves it best?</div></div></div>', unsafe_allow_html=True)
    inp = _sidebar(cfg)
    if inp["run"]:
        if not inp["profile"]:
            st.error("Define the custom profile's weights first.")
            st.stop()
        if not inp["candidates"]:
            st.error("Pick at least one candidate.")
            st.stop()
        extra = list(inp["profile"].weights) + ([inp["benchmark"]] if inp["benchmark"] else [])
        try:
            b, reg = _bundle(tuple(sorted(set(inp["candidates"] + extra + cfg.universe.all_tickers()))))
            req = E.Request(inp["profile"] if inp["profile"].custom else inp["profile"].key,
                            inp["objective"], inp["candidates"], inp["benchmark"], inp["mode"],
                            inp["weight"], inp["holding"], inp["rebalance"], inp["alignment"])
            with st.spinner("Running the review…"):
                rv = E.run(req, cfg, b, reg)
                sw = E.sweep(cfg, b, next(iter(rv.results)), inp["objective"], inp["mode"],
                             inp["weight"], inp["holding"], reg) if inp["sweep"] and rv.results else None
        except (CF.ConfigError, KeyError, ValueError) as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["sr_result"] = (rv, sw)
    res = st.session_state.get("sr_result")
    if not res:
        st.markdown('<div class="note">Pick a profile, an objective and one or more candidates in the '
                    'sidebar, then press <b>Run review</b>. The first run downloads prices for the '
                    'whole universe (about 30 seconds); later runs read the disk cache.</div>',
                    unsafe_allow_html=True)
        return
    rv, sw = res
    secs = RP.sections(rv, cfg, sw)
    st.markdown(f'<div class="eyebrow">{rv.profile.describe()} · {rv.objective_label} · '
                f'{rv.period}</div>', unsafe_allow_html=True)
    if rv.skipped:
        st.markdown('<div class="flag">Skipped: ' + "; ".join(f"{k}: {v}" for k, v in rv.skipped.items())
                    + '</div>', unsafe_allow_html=True)
    ranked = [c for c in rv.ranking["Candidate"]] if len(rv.ranking) else []
    cols = st.columns(min(3, max(1, len(ranked))))
    for i, c in enumerate(ranked):
        with cols[i % len(cols)]:
            st.markdown(_card(c, rv.results[c]), unsafe_allow_html=True)
    st.download_button("Download the full comparison (Excel)", X.write_sections(secs),
                       f"security_review_{rv.profile.key}_{rv.objective_key}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="sr_xlsx")
    by_sheet: Dict[str, List[RP.Section]] = {}
    for s in secs:
        by_sheet.setdefault(s[0], []).append(s)
    tabs = ["Verdict", "Ranking", "Fund Review", "Sleeve Comparison", "Portfolio Impact",
            "Scenarios", "Regimes", "Robustness", "Factor Exposures", "Diversification",
            "Profile Sweep", "Charts", "Methodology"]
    tabs = [t for t in tabs if t in by_sheet or t == "Charts"]
    for t, tab in zip(tabs, st.tabs(tabs)):
        with tab:
            if t == "Charts":
                _charts(rv)
                st.caption("Chart data: the Series sheet of the Excel export.")
                _show(by_sheet.get("Series", []))
            else:
                _show(by_sheet[t])
