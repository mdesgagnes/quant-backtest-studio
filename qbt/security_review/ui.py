"""Streamlit page for the Security review. Thin by design: it collects the
inputs, calls `engine.run_many` / `engine.sweep`, and renders
`report.all_sections` (the same tables the Excel export writes). No
calculation lives here.

The page answers one question: what does adding this security (or return
stream) do to a portfolio? The portfolio can be a benchmark ticker, a blend
of tickers, an uploaded portfolio return stream or an investor profile; the
summary comes first and every detail table sits behind "Details".
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import returns_input as RI
from ..brand import BRAND, RED
from . import config as CF
from . import engine as E
from . import export as X
from . import report as RP
from .bundle import load_bundle, load_macro, with_streams

# Red, then the sand / gold family: no blue (house colour scheme).
PALETTE = [RED, "#C9B896", "#D08C4E", "#EFE9DE", "#9A9389", "#B01419", "#CBBFA8", "#8C6D4F"]
BASE_KINDS = ["Benchmark", "Blend", "Uploaded portfolio", "Investor profile"]
NO_OBJECTIVE = "(none)"
BENCH_MODES = ["The portfolio (default)", "Each security's sleeve benchmark", "A ticker"]


@st.cache_resource(show_spinner=False)
def _config() -> CF.AllConfig:
    return CF.load_all()


@st.cache_resource(show_spinner="Loading T-bill, CPI and factor data…", ttl=6 * 3600)
def _macro():
    return load_macro(_config())


@st.cache_resource(show_spinner="Loading prices (cached on disk after the first run)…", ttl=6 * 3600,
                   max_entries=8)
def _bundle(tickers: tuple):
    """Prices for exactly the tickers a run needs (the securities, the
    portfolio's holdings, and the benchmark, factor, regime and FX series),
    not the whole universe."""
    cfg = _config()
    b = load_bundle(cfg, list(tickers), macro_data=_macro())
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


# Colours of the Backtest's tables (gain / loss / faint), from the brand file.
_GOOD, _BAD, _DIM = BRAND["gain"], BRAND["loss"], BRAND["faint"]
# A rise in these is bad for the investor; everything else, a rise is good.
_LOWER_IS_BETTER = ("volatility", "ulcer", "cvar", "var 95", "tracking error", "underwater",
                    "downside capture", "probability", "depletion")


def _lower_better(name: str) -> bool:
    n = str(name).lower()
    return any(w in n for w in _LOWER_IS_BETTER)


def _tone(v, lower_better: bool) -> str:
    """Teal when the figure helps the investor, rust when it hurts; works on
    the formatted text ("+0.42%", "-0.60%")."""
    t = str(v).strip()
    if t in ("", "n/a", "nan", "None"):
        return f"color:{_DIM}"
    try:
        x = float(t.replace("%", "").replace(",", "").replace("+", ""))
    except ValueError:
        return ""
    if abs(x) < 1e-12:
        return f"color:{_DIM}"
    good = (x < 0) if lower_better else (x > 0)
    return f"color:{_GOOD if good else _BAD}"


def _is_change(col: str) -> bool:
    c = str(col)
    return c.startswith(("Change in", "Excess return")) or c == "Change"


def _styled(df: pd.DataFrame):
    """The table as the workbook shows it, coloured the way the Backtest
    colours its tables: every change is teal when it helps the investor and
    rust when it hurts (a fall in volatility is teal, a fall in return rust)."""
    txt = _text(df)
    sty = txt.style
    if df.attrs.get("row_fmt") and "Change" in txt.columns:
        names = txt.iloc[:, 0].astype(str).tolist()
        sty = sty.apply(lambda col: [_tone(v, _lower_better(n)) for v, n in zip(col, names)],
                        subset=["Change"])
    for c in txt.columns:
        if _is_change(c) and c != "Change":
            lb = _lower_better(c)
            sty = sty.map(lambda v, lb=lb: _tone(v, lb), subset=[c])
    first = txt.columns[0] if len(txt.columns) else None
    if first is not None and first in ("Security", "Candidate", "Portfolio", "Measure"):
        sty = sty.set_properties(subset=[first], **{"font-weight": "600"})
    return sty


def _show(secs: List[RP.Section]) -> None:
    for _, title, df, note in secs:
        if df is None or df.empty:
            continue
        st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)
        st.dataframe(_styled(df), hide_index=True, width="stretch")
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
# Inputs
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


def _streams(up) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Monthly return streams from an uploaded file: one per numeric column.
    Daily or weekly returns are compounded to calendar months."""
    raw, meta = RI.load_return_stream(up.getvalue(), up.name)
    df, rep = RI.prepare_returns(raw, "auto", meta)
    msgs = list(rep.warnings)
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        if rep.frequency != "monthly":
            s = RI.compound_to(s.to_frame(), "Monthly").iloc[:, 0]
            msgs.append(f"{c}: {rep.frequency} returns compounded to months.")
        out[str(c).strip().upper()] = s
    return out, msgs


def _ticker_list(text: str) -> List[str]:
    return [t.strip().upper() for t in text.replace(";", ",").replace(" ", ",").split(",") if t.strip()]


def _sidebar(cfg: CF.AllConfig):
    sb = st.sidebar
    uploads: Dict[str, pd.Series] = {}
    msgs: List[str] = []

    # 1. What to analyse ------------------------------------------------
    sb.markdown('<div class="eyebrow">1. Securities to analyse</div>', unsafe_allow_html=True)
    sleeves = cfg.universe.sleeves
    universe = cfg.universe.all_tickers()
    st.session_state.setdefault("sr_cands", ["USMV"])
    with sb.expander("Pick a whole sleeve"):
        fill = st.selectbox("Sleeve", ["(keep my picks)"] + list(sleeves),
                            format_func=lambda k: sleeves[k].label if k in sleeves else k, key="sr_fill")
    if fill != st.session_state.get("_sr_fill_applied"):
        st.session_state["_sr_fill_applied"] = fill
        if fill in sleeves:
            st.session_state["sr_cands"] = list(sleeves[fill].tickers)
    extra_now = [t for t in st.session_state["sr_cands"] if t not in universe]
    cands = sb.multiselect("From the universe", universe + extra_now, key="sr_cands",
                           format_func=lambda t: f"{t} ({cfg.universe.sleeve_of(t).label})"
                           if cfg.universe.sleeve_of(t) else t)
    typed = sb.text_input("Or type tickers", "", key="sr_typed",
                          help="Any Yahoo Finance symbol, comma-separated: .TO for Toronto listings.")
    cands = list(dict.fromkeys(cands + _ticker_list(typed)))
    up_c = sb.file_uploader("Or upload return streams (CSV or Excel)", type=["csv", "xlsx", "xls"],
                            key="sr_cand_file",
                            help="A date column and one column of returns per stream; each column "
                                 "is analysed as a security.")
    if up_c is not None:
        try:
            s, m = _streams(up_c)
            uploads.update(s)
            msgs += m
            cands += [k for k in s if k not in cands]
        except Exception as exc:
            sb.error(f"Could not read the returns file: {exc}")

    # 2. What to add it to ------------------------------------------------
    sb.markdown('<div class="eyebrow">2. Add it to</div>', unsafe_allow_html=True)
    kind = sb.radio("Portfolio", BASE_KINDS, key="sr_kind", horizontal=False,
                    captions=["One or more tickers, each compared separately",
                              "Tickers and weights", "Your portfolio's own return history",
                              "The firm's preset profiles"])
    bases: List[CF.Profile] = []
    presets = list(cfg.profiles.profiles)
    if kind == "Benchmark":
        picks = sb.multiselect("Benchmark(s)", universe, default=["XIC.TO"], key="sr_bench_pick")
        picks = list(dict.fromkeys(picks + _ticker_list(sb.text_input("Or type tickers", "",
                                                                      key="sr_bench_typed"))))
        bases = [CF.custom_profile(t, {t: 1.0}) for t in picks]
    elif kind == "Blend":
        how = sb.radio("Weights from", ["Type them", "Upload a file"], horizontal=True, key="sr_cust_how")
        w: Dict[str, float] = {}
        if how == "Type them":
            w = _parse_weights(sb.text_area("One holding per line: ticker weight",
                                            "XIC.TO 30\nVFV.TO 30\nXBB.TO 40",
                                            key="sr_cust_txt", height=110))
        else:
            up = sb.file_uploader("CSV or Excel: ticker, weight", type=["csv", "xlsx", "xls"],
                                  key="sr_cust_file")
            if up is not None:
                try:
                    w = _weights_file(up)
                except Exception as exc:
                    sb.error(f"Could not read the file: {exc}")
        name = sb.text_input("Name", "My blend", key="sr_cust_name")
        try:
            p = CF.custom_profile(name, w)
            sb.caption(" · ".join(f"{t} {x:.0%}" for t, x in p.weights.items()))
            if sb.button("Save as a profile", key="sr_cust_save"):
                path = CF.save_custom_profile(p)
                _config.clear()
                sb.success(f"Saved to {path}.")
            bases = [p]
        except CF.ConfigError as exc:
            sb.warning(str(exc))
    elif kind == "Uploaded portfolio":
        up_b = sb.file_uploader("Portfolio returns (CSV or Excel)", type=["csv", "xlsx", "xls"],
                                key="sr_base_file",
                                help="A date column and one column of returns per portfolio; each "
                                     "column is compared separately.")
        if up_b is not None:
            try:
                s, m = _streams(up_b)
                uploads.update(s)
                msgs += m
                bases = [CF.custom_profile(k, {k: 1.0}) for k in s]
            except Exception as exc:
                sb.error(f"Could not read the returns file: {exc}")
    else:
        keys = sb.multiselect("Profile(s)", presets, default=["balanced"] if "balanced" in presets else [],
                              format_func=lambda k: cfg.profiles.profiles[k].describe(), key="sr_profiles")
        bases = [cfg.profiles.profiles[k] for k in keys]

    # 3. How much ---------------------------------------------------------
    sb.markdown('<div class="eyebrow">3. How much</div>', unsafe_allow_html=True)
    can_replace = len(bases) == 1 and len(bases[0].weights) > 1
    mode = sb.radio("Change", ["Add", "Replace a holding"] if can_replace else ["Add"],
                    horizontal=True, key="sr_mode")
    weight, holding = None, None
    if mode == "Add":
        weight = sb.slider("Add weight, funded pro rata (%)", 1, 40,
                           int(round(cfg.objectives.add_weight * 100)), key="sr_w") / 100
    else:
        hs = list(bases[0].weights)
        holding = sb.selectbox("Holding to replace", hs, key="sr_hold",
                               format_func=lambda t: f"{t} ({bases[0].weights[t]:.0%})")

    # 4. Objective (optional) ----------------------------------------------
    sb.markdown('<div class="eyebrow">4. Objective (optional)</div>', unsafe_allow_html=True)
    objs = [NO_OBJECTIVE] + list(cfg.objectives.objectives)
    ok = sb.selectbox("Judge it against", objs, key="sr_obj",
                      format_func=lambda k: "None: just show what changes" if k == NO_OBJECTIVE
                      else cfg.objectives.objectives[k].label)
    if ok != NO_OBJECTIVE:
        sb.caption(cfg.objectives.objectives[ok].description + " Adds a verdict to each result.")

    with sb.expander("Advanced"):
        reb = st.selectbox("Rebalancing", ["(each portfolio's own)"] + list(CF.REBALANCE_OPTIONS),
                           key="sr_reb", help="Blends, benchmarks and uploads default to quarterly.")
        align = st.radio("Analysis period", ["common", "each"], key="sr_align",
                         format_func=lambda a: "Common to all securities" if a == "common"
                         else "Each security's own history",
                         help="Common: every security is judged on the same months (fair "
                              "comparison). Each: a long-history security keeps its full record.")
        currency = st.radio("Uploaded returns are in", ["CAD", "USD"], horizontal=True, key="sr_ccy")
        bmode = st.radio("Benchmark for the relative statistics", BENCH_MODES, key="sr_bench_mode",
                         help="Capture, tracking error, beta, factor regression and the Fund Review "
                              "comparison. Default: the portfolio the security is added to, so "
                              "every figure is measured against the same thing.")
        bench = None
        if bmode == BENCH_MODES[1]:
            bench = "sleeve"
        elif bmode == BENCH_MODES[2]:
            bench = st.text_input("Ticker", "", key="sr_bench").strip().upper() or None
        sweep_on = st.checkbox("Also run a profile sweep", key="sr_sweep_on",
                               help="The first security across all 7 preset profiles.")
    for m in msgs:
        sb.caption(m)
    run = sb.button("Run", type="primary", width="stretch", key="sr_run")
    return dict(bases=bases, kind=kind, rebalance=None if reb.startswith("(") else reb,
                objective=None if ok == NO_OBJECTIVE else ok, candidates=cands, uploads=uploads,
                currency=currency, benchmark=bench, mode="add" if mode == "Add" else "replace",
                weight=weight, holding=holding, alignment=align, sweep=sweep_on, run=run)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
def _charts(rvs: List[E.Review]) -> None:
    pairs = [(rv, c) for rv in rvs for c in rv.results]
    if not pairs:
        return
    if len(rvs) == 1 and len(pairs) <= 5:
        chosen = pairs
    else:
        lab = {f"{c} added to {RP.base_label(rv)}": (rv, c) for rv, c in pairs}
        pick = st.selectbox("Chart", list(lab), key="sr_chart_pick")
        chosen = [lab[pick]]
    fig = _fig(340, "Growth of 1 (CAD)")
    fig2 = _fig(260, "Drawdown")
    seen = set()
    for i, (rv, c) in enumerate(chosen):
        r = rv.results[c]
        base = r.series["Portfolio without"].dropna()
        if id(rv) not in seen:
            seen.add(id(rv))
            eq = (1 + base).cumprod()
            name = RP.base_label(rv)
            fig.add_scatter(x=eq.index, y=eq, name=name, line=dict(color=BRAND["muted"], width=2, dash="dot"))
            fig2.add_scatter(x=eq.index, y=eq / eq.cummax() - 1, name=name,
                             line=dict(color=BRAND["muted"], dash="dot"))
        e = (1 + r.series["Portfolio with"].dropna()).cumprod()
        lbl = f"with {c}" if len(rvs) == 1 else f"{RP.base_label(rv)} with {c}"
        fig.add_scatter(x=e.index, y=e, name=lbl, line=dict(color=PALETTE[i % len(PALETTE)], width=1.8))
        fig2.add_scatter(x=e.index, y=e / e.cummax() - 1, name=lbl,
                         line=dict(color=PALETTE[i % len(PALETTE)], width=1.4))
    fig2.update_yaxes(tickformat=".0%")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    c2.plotly_chart(fig2, width="stretch")


def _detail_charts(rv: E.Review) -> None:
    fig = _fig(320, "Stress windows: change in portfolio return")
    for i, (c, r) in enumerate(rv.results.items()):
        t = r.stress
        ok = t[t["Status"] == "ok"] if "Status" in t else t
        if len(ok):
            fig.add_bar(x=ok["Window"], y=ok["Change in portfolio return"], name=c,
                        marker_color=PALETTE[i % len(PALETTE)])
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    obj = _config().objectives.objectives[rv.objective_key]
    if rv.objective_chosen and obj.rolling_metric:
        fig = _fig(300, f"Rolling 36-month {obj.rolling_metric} (with minus without)")
        for i, (c, r) in enumerate(rv.results.items()):
            s = r.rolling.get(obj.rolling_metric, {}).get("series")
            if s is not None and len(s):
                fig.add_scatter(x=s.index, y=s, name=c, line=dict(color=PALETTE[i % len(PALETTE)]))
        st.plotly_chart(fig, width="stretch")


def _run(inp, cfg):
    if not inp["bases"]:
        st.error("Choose the portfolio to add the securities to.")
        st.stop()
    if not inp["candidates"]:
        st.error("Pick at least one security.")
        st.stop()
    uploads = inp["uploads"]
    tickers = [t for t in inp["candidates"] if t not in uploads]
    extra = [t for p in inp["bases"] for t in p.weights if t not in uploads]
    if inp["benchmark"] and inp["benchmark"] != "sleeve":
        extra.append(inp["benchmark"])
    try:
        b, reg = _bundle(tuple(sorted(set(tickers + extra))))
        b = with_streams(b, uploads, inp["currency"])
        bases = [p if p.custom else p.key for p in inp["bases"]]
        req = E.Request(bases[0], inp["objective"], inp["candidates"], inp["benchmark"], inp["mode"],
                        inp["weight"], inp["holding"], inp["rebalance"], inp["alignment"])
        with st.spinner("Running…"):
            rvs = E.run_many(req, bases, cfg, b, reg)
            first = next((c for rv in rvs for c in rv.results), None)
            sw = (E.sweep(cfg, b, first, inp["objective"] or next(iter(cfg.objectives.objectives)),
                          inp["mode"], inp["weight"], inp["holding"], reg)
                  if inp["sweep"] and first else None)
    except (CF.ConfigError, KeyError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["sr_result"] = (rvs, sw)


def _deferred(build):
    """Builds the workbook only when the button is clicked (as the Backtest
    does); a failure gives a one-line workbook saying why, never a broken
    download."""
    def _make():
        try:
            return build()
        except Exception as exc:
            buf = io.BytesIO()
            pd.DataFrame({"The workbook could not be built": [str(exc)]}).to_excel(buf, index=False)
            return buf.getvalue()
    return _make


def _downloads(rvs, cfg, sw, pairs) -> None:
    if not pairs:
        return
    focus = pairs[0]
    c1, c2 = st.columns([2, 3])
    if len(pairs) > 1:
        lab = {f"{c} added to {RP.base_label(rv)}": (rv, c) for rv, c in pairs}
        focus = lab[c2.selectbox("Report on", list(lab), key="sr_report_focus",
                                 help="The executive summary and performance sheets are for this "
                                      "result; every result's tables follow.")]
    rv, c = focus
    name = "".join(ch if ch.isalnum() else "_" for ch in f"{c}_in_{RP.base_label(rv)}")[:60]
    c1.download_button("Full report (Excel)",
                       _deferred(lambda _r=rvs, _f=focus, _s=sw: X.full_report(_r, cfg, _f, _s)),
                       f"security_review_{name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="sr_xlsx", on_click="ignore", type="primary")
    st.caption("The workbook opens on a two-page executive summary, then an About sheet indexing "
               "every table: statistics, CIPM appraisal, trailing periods, calendar years, "
               "drawdowns, monthly returns and the performance analysis of the portfolio with the "
               "security against the portfolio without it, followed by every table on this page.")


def render() -> None:
    cfg = _config()
    st.markdown('<div class="masthead"><div><h1>Security review</h1>'
                '<div class="sub">What does adding this security do to the portfolio?</div>'
                '</div></div>', unsafe_allow_html=True)
    inp = _sidebar(cfg)
    if inp["run"]:
        _run(inp, cfg)
    res = st.session_state.get("sr_result")
    if not res:
        st.markdown('<div class="note">Pick one or more securities (or upload return streams), '
                    'choose what to add them to (a benchmark, a blend, your portfolio\'s returns '
                    'or a profile), then press <b>Run</b>. The first run downloads prices '
                    '(a few seconds); later runs read the disk cache.</div>', unsafe_allow_html=True)
        return
    rvs, sw = res
    secs = RP.all_sections(rvs, cfg, sw)
    summ = [s for s in secs if s[0] == "Summary"]
    pairs = [(rv, c) for rv in rvs for c in rv.results]
    rv0 = rvs[0]
    ctx = [RP.change_label(rv0) if len({RP.change_label(r) for r in rvs}) == 1 else "",
           rv0.objective_label if rv0.objective_chosen else "",
           rv0.period if rv0.request.alignment == "common" else "each security's own history"]
    st.markdown(f'<div class="eyebrow">{" · ".join(x for x in ctx if x)}</div>', unsafe_allow_html=True)

    if pairs and len(pairs) <= 4:
        for rv, c in pairs:
            st.markdown(f'<div class="note" style="margin-bottom:.5rem;">{RP.headline(rv, c)}</div>',
                        unsafe_allow_html=True)
    elif pairs:
        st.caption(f"{len(pairs)} results: one row each below. Click a column header to sort.")

    for _, title, df, note in summ:
        if title == "In one sentence":
            continue
        st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)
        st.dataframe(_styled(df), hide_index=True, width="stretch")
        if note:
            st.caption(note)

    _charts(rvs)
    _downloads(rvs, cfg, sw, pairs)

    with st.expander("Details: stress windows, regimes, factors, robustness, fund data, methodology"):
        rv = rv0
        if len(rvs) > 1:
            lab = {RP.base_label(r): r for r in rvs}
            rv = lab[st.selectbox("Portfolio", list(lab), key="sr_detail_base")]
        by_sheet: Dict[str, List[RP.Section]] = {}
        for s in RP.sections(rv, cfg, sw if rv is rv0 else None):
            by_sheet.setdefault(s[0], []).append(s)
        tabs = ["Scenarios", "Regimes", "Portfolio Impact", "Diversification", "Factor Exposures",
                "Fund Review", "Relative to Benchmark", "Robustness", "Verdict", "Ranking",
                "Profile Sweep", "Charts", "Methodology"]
        names = {"Verdict": "Objectives" if not rv.objective_chosen else "Verdict"}
        tabs = [t for t in tabs if t in by_sheet or t == "Charts"]
        for t, tab in zip(tabs, st.tabs([names.get(t, t) for t in tabs])):
            with tab:
                if t == "Charts":
                    _detail_charts(rv)
                    st.caption("Chart data: the Series sheet of the Excel export.")
                    _show(by_sheet.get("Series", []))
                else:
                    _show(by_sheet[t])
