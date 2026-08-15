"""Quant Backtest Studio -- Streamlit interface.

Run locally:   streamlit run app.py
Deployment:    see README.md
"""
from __future__ import annotations

import io
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from qbt.config import (RunConfig, DataConfig, EngineConfig, CostConfig,
                        StrategyConfig, ExogConfig, REBALANCE_RULES)
from qbt.data import (load_yfinance, load_market_data, load_file, clean_prices,
                      excel_sheet_names, MarketData)
from qbt.exog import load_exog, prepare_exog, exog_report, split_roles
from qbt.external import (load_target_weights, prepare_target_weights,
                          weights_template)
from qbt.engine import (run_backtest, benchmark_result, blended_benchmark,
                        align_results, align_start, first_active_date)
from qbt.strategies import REGISTRY, get as get_strategy
from qbt import metrics as M
from qbt import charts as C
from qbt import robustness as R
from qbt import report as REPORT
from qbt import returns_input as RS
from qbt import presets as PRESETS
from qbt import formula as FORMULA
from qbt import excel_export as XL
from qbt import allocation as ALLOC

st.set_page_config(page_title="Quant Backtest Studio",
                   page_icon="\u25e7", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@600;700&display=swap');

/* ---------------------------------------------------------------
   Ink and brass. Ink carries the page, brass marks what matters,
   teal and rust are reserved for sign -- never decoration.
   --------------------------------------------------------------- */
:root{
  --ink:#0E1116; --panel:#161B22; --panel-2:#12171F;
  --rule:#232B35; --rule-soft:#1C222B;
  --text:#E3E8EF; --muted:#7D8A9C; --dim:#5B6675;
  --brass:#C9A227; --brass-lift:#DDB63A; --brass-dim:#8A701C;
  --teal:#4C9A8F; --rust:#B4553F;
  --mono:'IBM Plex Mono',SFMono-Regular,Consolas,monospace;
  --sans:'IBM Plex Sans',-apple-system,Segoe UI,sans-serif;
  --cond:'IBM Plex Sans Condensed',sans-serif;
}

html, body, [class*="css"] { font-family:var(--sans); }
.stApp { background:var(--ink); }

/* Tabular figures everywhere numbers appear: columns of returns must
   align on the decimal or they cannot be scanned down. */
.dial .v,.dial .d,[data-testid="stDataFrame"],.stTabs [data-baseweb="tab"],
table,code { font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1; }

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"]{ background:var(--panel-2); border-right:1px solid var(--rule); }
section[data-testid="stSidebar"] > div{ padding-top:.7rem; }
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stTextArea,
section[data-testid="stSidebar"] .stRadio{ margin-bottom:-.35rem; }
section[data-testid="stSidebar"] label p{ font-size:.79rem !important; color:#B9C3D0 !important; }

/* Sidebar sections fold away: the panel holds sixty controls and only a
   handful change between runs. */
section[data-testid="stSidebar"] [data-testid="stExpander"]{
  border:0; border-top:1px solid var(--rule-soft); background:transparent; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary{
  font-family:var(--mono); font-size:.66rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted); padding:.5rem 0; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{ color:var(--brass); }

/* ---------------- Masthead ---------------- */
.masthead{ border-bottom:1px solid var(--rule); padding:.1rem 0 .85rem; margin-bottom:.4rem; }
.masthead h1{ font-family:var(--cond); font-weight:700; font-size:1.9rem;
  letter-spacing:-.015em; color:var(--text); margin:0; line-height:1.1; }
.masthead .sub{ font-family:var(--mono); font-size:.7rem; letter-spacing:.17em;
  text-transform:uppercase; color:var(--muted); margin-top:.32rem; }

/* ---------------- Section rules ---------------- */
.eyebrow{ font-family:var(--mono); font-size:.67rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); margin:1.7rem 0 .6rem;
  display:flex; align-items:center; gap:.7rem; }
.eyebrow::after{ content:""; flex:1; height:1px; background:var(--rule-soft); }

/* ---------------- Dials ---------------- */
.dial{ background:var(--panel); border:1px solid var(--rule);
  border-left:2px solid var(--brass); padding:.6rem .8rem; height:100%;
  transition:border-color .16s ease, background .16s ease; }
.dial:hover{ background:#1A2029; border-left-color:var(--brass-lift); }
.dial .k{ font-family:var(--mono); font-size:.62rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--muted); display:block; margin-bottom:.26rem; }
.dial .v{ font-family:var(--mono); font-size:1.24rem; font-weight:600;
  color:var(--text); line-height:1.1; }
.dial .d{ font-family:var(--mono); font-size:.67rem; color:var(--dim); margin-top:.1rem; }
.dial.pos{ border-left-color:var(--teal); } .dial.pos .v{ color:var(--teal); }
.dial.neg{ border-left-color:var(--rust); } .dial.neg .v{ color:var(--rust); }

/* ---------------- Notes and flags ---------------- */
.note{ border-left:2px solid var(--rule); padding:.2rem 0 .2rem .8rem;
  color:var(--muted); font-size:.83rem; line-height:1.55; }
.flag{ border-left:2px solid var(--brass); padding:.35rem 0 .35rem .8rem;
  color:var(--brass); font-size:.81rem; font-family:var(--mono);
  line-height:1.5; margin:.3rem 0; }

/* ---------------- Tabs: stay reachable ---------------- */
.stTabs [data-baseweb="tab-list"]{
  gap:1.7rem; border-bottom:1px solid var(--rule);
  position:sticky; top:0; z-index:99;
  background:rgba(14,17,22,.94); backdrop-filter:blur(8px);
  padding-top:.45rem; }
.stTabs [data-baseweb="tab"]{ font-family:var(--mono); font-size:.71rem;
  letter-spacing:.13em; text-transform:uppercase; color:var(--muted);
  padding:.42rem 0; transition:color .14s ease; }
.stTabs [data-baseweb="tab"]:hover{ color:#A8B4C4; }
.stTabs [aria-selected="true"]{ color:var(--brass) !important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--brass); height:2px; }
.stTabs [data-baseweb="tab-border"]{ background:transparent; }

/* ---------------- Controls ---------------- */
.stButton>button{ font-family:var(--mono); font-size:.74rem; letter-spacing:.1em;
  text-transform:uppercase; background:var(--brass); color:var(--ink);
  border:0; border-radius:2px; font-weight:600; width:100%; padding:.56rem;
  transition:background .16s ease, transform .1s ease; }
.stButton>button:hover{ background:var(--brass-lift); color:var(--ink); }
.stButton>button:active{ transform:translateY(1px); }
.stDownloadButton>button{ background:transparent; color:var(--brass);
  border:1px solid var(--brass-dim); font-weight:500; }
.stDownloadButton>button:hover{ background:rgba(201,162,39,.1);
  border-color:var(--brass); color:var(--brass-lift); }

/* Keyboard focus must stay visible: brass ring, never the browser default. */
:focus-visible{ outline:2px solid var(--brass) !important; outline-offset:2px !important; }

/* ---------------- Tables ---------------- */
[data-testid="stDataFrame"]{ font-family:var(--mono); }
[data-testid="stDataFrame"] [role="columnheader"]{
  font-size:.66rem !important; letter-spacing:.07em; text-transform:uppercase;
  color:var(--muted) !important; }

/* ---------------- Charts ---------------- */
.js-plotly-plot{ border:1px solid var(--rule-soft); border-radius:2px;
  background:rgba(22,27,34,.35); }

/* ---------------- Expanders ---------------- */
[data-testid="stExpander"]{ border:1px solid var(--rule); border-radius:2px;
  background:var(--panel); }
[data-testid="stExpander"] summary{ font-family:var(--mono); font-size:.72rem;
  letter-spacing:.06em; color:#A8B4C4; }
[data-testid="stExpander"] summary:hover{ color:var(--brass); }

/* ---------------- Run context strip ---------------- */
/* Six tabs deep, it stops being obvious which run is on screen. */
.runbar{ display:flex; flex-wrap:wrap; gap:.15rem 1.6rem; align-items:baseline;
  border-left:2px solid var(--brass); padding:.4rem 0 .4rem .8rem;
  margin:.2rem 0 .3rem; background:linear-gradient(90deg,rgba(201,162,39,.06),transparent 55%); }
.runbar .item{ font-family:var(--mono); font-size:.72rem; color:var(--muted); }
.runbar .item b{ color:var(--text); font-weight:600; }
.runbar .lead{ font-family:var(--cond); font-size:.98rem; font-weight:700;
  color:var(--text); letter-spacing:-.01em; margin-right:.3rem; }

/* ---------------- Empty state ---------------- */
.startcard{ border:1px solid var(--rule); border-left:2px solid var(--brass);
  background:var(--panel); padding:.85rem 1rem; height:100%; }
.startcard .n{ font-family:var(--mono); font-size:.62rem; letter-spacing:.14em;
  color:var(--brass); text-transform:uppercase; }
.startcard .t{ font-family:var(--cond); font-size:1.02rem; font-weight:700;
  color:var(--text); margin:.3rem 0 .25rem; }
.startcard .b{ font-size:.81rem; color:var(--muted); line-height:1.5; }

.strat{ border-top:1px solid var(--rule-soft); padding:.55rem 0; }
.strat .nm{ font-family:var(--mono); font-size:.79rem; color:var(--brass); }
.strat .ds{ font-size:.81rem; color:var(--muted); line-height:1.5; margin-top:.15rem; }

/* ---------------- Scrollbars ---------------- */
::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-track{ background:var(--ink); }
::-webkit-scrollbar-thumb{ background:#2A323D; border-radius:0; }
::-webkit-scrollbar-thumb:hover{ background:#3A4553; }

/* ---------------- Chrome ---------------- */
[data-testid="stHeader"]{ background:transparent; }
.stSpinner > div{ border-top-color:var(--brass) !important; }
[data-testid="stToolbar"]{ right:.6rem; }
hr{ border-color:var(--rule-soft); }
#MainMenu, footer{ visibility:hidden; }

/* A person who has asked the system to stop moving things gets that. */
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{ animation-duration:.001ms !important;
    transition-duration:.001ms !important; }
}

/* ---------------- Narrow screens ---------------- */
@media (max-width: 640px){
  .masthead h1{ font-size:1.45rem; }
  .dial .v{ font-size:1.05rem; }
  .stTabs [data-baseweb="tab-list"]{ gap:1rem; overflow-x:auto; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Access
# ----------------------------------------------------------------------
def require_password() -> None:
    """Optional protection. Only active if a password is set in the host's
    secrets (Settings -> Secrets):

        password = "..."

    Locally, with no secret set, the app opens normally.
    """
    try:
        expected = st.secrets.get("password", "")
    except Exception:
        expected = ""
    if not expected:
        return
    if st.session_state.get("auth_ok"):
        return

    st.markdown(
        '<div class="masthead"><h1>Quant Backtest Studio</h1>'
        '<div class="sub">Restricted access</div></div>', unsafe_allow_html=True)
    with st.form("auth"):
        entry = st.text_input("Password", type="password",
                              label_visibility="collapsed",
                              placeholder="Password")
        ok = st.form_submit_button("Open")
    if ok:
        if entry == expected:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.markdown('<div class="flag">Incorrect password.</div>',
                        unsafe_allow_html=True)
    st.stop()


TEAL, RUST, DIM = "#4C9A8F", "#B4553F", "#7D8A9C"


def _sign_color(v) -> str:
    """Colour a value by its sign. Works on the pre-formatted strings the
    tables already carry ("+3.45%", "-2.10%") as well as raw numbers, so
    formatting and colouring stay independent of each other."""
    if v is None:
        return f"color:{DIM}"
    if isinstance(v, str):
        t = v.strip()
        if t in ("", "\u2014", "nan", "None"):
            return f"color:{DIM}"
        if t.startswith("-"):
            return f"color:{RUST}"
        if t.startswith("+") or (t[:1].isdigit() and float(
                t.replace("%", "").replace(",", "") or 0) > 0):
            return f"color:{TEAL}"
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:
        return f"color:{DIM}"
    return f"color:{TEAL}" if f > 0 else (f"color:{RUST}" if f < 0 else f"color:{DIM}")


def signed(df, cols=None, emphasise=None):
    """Returns a Styler that colours the given columns by sign.

    Scanning a column of returns is the single most common thing done with
    these tables, and sign is what the eye is looking for. Colour carries
    it without adding a glyph or a second column.
    """
    if df is None or getattr(df, "empty", True):
        return df
    use = [c for c in (cols or df.columns) if c in df.columns]
    if not use:
        return df
    st_ = df.style.map(_sign_color, subset=use)
    if emphasise:
        keep = [c for c in emphasise if c in df.columns]
        if keep:
            st_ = st_.set_properties(subset=keep, **{"font-weight": "600"})
    return st_


def dial(label: str, value: str, sub: str = "", tone: str = ""):
    cls = f"dial {tone}".strip()
    st.markdown(
        f'<div class="{cls}"><span class="k">{label}</span>'
        f'<span class="v">{value}</span>'
        f'{f"<div class=d>{sub}</div>" if sub else ""}</div>',
        unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def _pseudo_result(returns, equity, label):
    """Wraps a bare return stream in the structure the report expects."""
    from qbt.engine import BacktestResult
    idx = returns.index
    one = pd.DataFrame(1.0, index=idx, columns=[label])
    zero = pd.Series(0.0, index=idx)
    return BacktestResult(
        equity=equity, returns=returns, gross_returns=returns,
        weights=one, target_weights=one, turnover=zero, costs=zero,
        exposure=pd.Series(1.0, index=idx), cash_weight=zero,
        rebalance_dates=pd.DatetimeIndex([]),
        trades=pd.DataFrame(columns=["Date", "Instrument", "Weight Before",
                                     "Weight After", "Change"]),
        label=label)


def _fmt_period_tables(tables, main_label, bench_label=None):
    """Formats the trailing / calendar tables for display."""
    out = {}
    for key in ("trailing", "calendar"):
        t = tables[key].copy()
        if t.empty:
            out[key] = t
            continue
        for c in [main_label] + ([bench_label] if bench_label else []) + ["Excess"]:
            if c in t.columns:
                t[c] = t[c].map(lambda v: "\u2014" if pd.isna(v) else f"{v*100:+.2f}%")
        out[key] = t
    return out


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


require_password()


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf(tickers: tuple, start: str, end: Optional[str], field: str) -> pd.DataFrame:
    return load_yfinance(list(tickers), start, end, field)


@st.cache_data(show_spinner=False)
def parse_upload(content: bytes, name: str, sheet: Optional[str]) -> pd.DataFrame:
    buf = io.BytesIO(content)
    buf.name = name
    return load_file(buf, sheet)


@st.cache_data(show_spinner=False)
def parse_exog(content: bytes, name: str, sheet: Optional[str]) -> pd.DataFrame:
    buf = io.BytesIO(content)
    buf.name = name
    return load_exog(buf, sheet)


@st.cache_data(show_spinner=False)
def parse_weights(content: bytes, name: str, sheet: Optional[str]) -> pd.DataFrame:
    buf = io.BytesIO(content)
    buf.name = name
    return load_target_weights(buf, sheet)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="font-family:IBM Plex Mono,monospace;font-size:.68rem;'
    'letter-spacing:.18em;text-transform:uppercase;color:#C9A227;'
    'padding:.2rem 0 .8rem;">Settings</div>', unsafe_allow_html=True)

with st.sidebar.expander("Import a configuration", expanded=False):
    up_cfg = st.file_uploader("YAML file", type=["yaml", "yml"], key="cfgup")
    if up_cfg is not None and st.button("Apply configuration"):
        try:
            st.session_state["loaded_cfg"] = RunConfig.from_yaml(
                up_cfg.read().decode("utf-8"))
            st.success("Configuration loaded.")
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

loaded: Optional[RunConfig] = st.session_state.get("loaded_cfg")
d0 = loaded.data if loaded else DataConfig()
e0 = loaded.engine if loaded else EngineConfig()
c0 = loaded.costs if loaded else CostConfig()
s0 = loaded.strategy if loaded else StrategyConfig()
x0 = loaded.exog if loaded else ExogConfig()

# --- Data ---
st.sidebar.markdown('<div class="eyebrow">Data</div>', unsafe_allow_html=True)
SOURCES = ["Yahoo Finance", "File (CSV / Excel)", "Return stream"]
_src_idx = {"yfinance": 0, "upload": 1, "returns": 2}.get(d0.source, 0)
source = st.sidebar.radio("Source", SOURCES, index=_src_idx,
                          label_visibility="collapsed",
                          help="\u201cReturn stream\u201d skips prices and signals "
                               "entirely: upload periodic returns and get the "
                               "statistics on that track record.")

# ======================================================================
# Return stream mode: no prices, no signals, no simulation.
# ======================================================================
if source == "Return stream":
    st.sidebar.markdown('<div class="eyebrow">Return stream</div>',
                        unsafe_allow_html=True)
    rfile = st.sidebar.file_uploader("Returns file",
                                     type=["csv", "xlsx", "xls", "txt"], key="rsup")
    rsheet = None
    if rfile is not None and rfile.name.lower().endswith((".xlsx", ".xls")):
        names = excel_sheet_names(io.BytesIO(rfile.getvalue()))
        if len(names) > 1:
            rsheet = st.sidebar.selectbox("Sheet", names, key="rssheet")

    scale_label = st.sidebar.selectbox(
        "Value scale", ["Detect automatically", "Decimals (0.0213)",
                        "Percentages (2.13)"],
        help="Only override if the automatic reading is wrong.")
    scale = {"Detect automatically": "auto", "Decimals (0.0213)": "decimal",
             "Percentages (2.13)": "percentage"}[scale_label]
    rs_capital = st.sidebar.number_input("Base value ($)", 1_000, 1_000_000_000,
                                         100_000, 10_000, key="rscap")

    st.markdown(
        '<div class="masthead"><h1>Quant Backtest Studio</h1>'
        '<div class="sub">Return stream &nbsp;\u00b7&nbsp; Statistics</div></div>',
        unsafe_allow_html=True)

    if rfile is None:
        note("Upload a file of periodic returns: one date column, then one "
             "column per series. Daily, weekly, monthly or quarterly \u2014 the "
             "frequency is inferred from the dates and drives the "
             "annualization. Values may be decimals (0.0213) or percentages "
             "(2.13).<br><br>Long format (date, name, return) is also "
             "recognized. This mode analyses the track record directly: there "
             "is no portfolio to simulate, so positions, frictions and "
             "parameter tests do not apply.")
        st.download_button("Monthly CSV template", RS.template("monthly"),
                           "returns_template.csv", "text/csv")
        st.stop()

    try:
        rs_raw = RS.load_return_stream(io.BytesIO(rfile.getvalue()), rsheet)
        rets, rrep = RS.prepare_returns(rs_raw, scale)
    except Exception as exc:
        st.error(f"The file could not be read: {exc}")
        st.stop()

    cols = list(rets.columns)
    c1, c2 = st.columns(2)
    main_col = c1.selectbox("Series analysed", cols)
    bench_opts = ["\u2014 none \u2014"] + [c for c in cols if c != main_col]
    bench_col = c2.selectbox("Compare against", bench_opts)
    bench_col = None if bench_col.startswith("\u2014") else bench_col

    ppy = rrep.periods_per_year
    r_main = rets[main_col]
    eq_main = RS.equity_from_returns(r_main, rs_capital)
    r_bench = rets[bench_col] if bench_col else None
    eq_bench = RS.equity_from_returns(r_bench, rs_capital) if bench_col else None

    stats = M.summary(r_main, eq_main, r_bench, None, None, 0.0, ppy)
    bstats = M.summary(r_bench, eq_bench, None, None, None, 0.0, ppy) if bench_col else {}

    a, b, c, d = st.columns(4)
    with a:
        dial("Frequency", rrep.frequency.capitalize(), f"{ppy} periods/year")
    with b:
        dial("Observations", f"{rrep.n_periods:,}")
    with c:
        dial("Period", str(rrep.start.date()) if rrep.start is not None else "\u2014",
             f"to {rrep.end.date()}" if rrep.end is not None else "")
    with d:
        dial("Scale read", rrep.scale.capitalize())
    for w in rrep.warnings:
        st.markdown(f'<div class="flag">{w}</div>', unsafe_allow_html=True)

    _rb = [f'<span class="lead">{main_col}</span>',
           f'<span class="item">{rrep.start.date()} &rarr; {rrep.end.date()}</span>'
           if rrep.start is not None else "",
           f'<span class="item">{rrep.n_periods:,} {rrep.frequency} periods</span>']
    if bench_col:
        _rb.append(f'<span class="item">vs <b>{bench_col}</b></span>')
    _rc = stats.get("CAGR", float("nan"))
    if _rc == _rc:
        _rb.append(f'<span class="item">CAGR <b>{M.format_metric("CAGR", _rc)}</b></span>')
    st.markdown(f'<div class="runbar">{"".join(_rb)}</div>', unsafe_allow_html=True)

    rs_tabs = st.tabs(["Results", "Robustness", "Export"])

    with rs_tabs[0]:
        keys = ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar", "Sortino"]
        kcols = st.columns(len(keys))
        for col, k in zip(kcols, keys):
            with col:
                v = stats.get(k, np.nan)
                tone = ""
                if k in ("CAGR", "Sharpe", "Calmar", "Sortino"):
                    tone = "pos" if (v == v and v > 0) else "neg"
                elif k == "Max Drawdown":
                    tone = "neg"
                sub = ""
                if bstats:
                    bv = bstats.get(k, np.nan)
                    if bv == bv:
                        sub = f"bench. {M.format_metric(k, bv)}"
                dial(k, M.format_metric(k, v), sub, tone)

        st.write("")
        curves = {main_col: eq_main}
        if bench_col:
            curves[bench_col] = eq_bench
        logs = st.toggle("Log scale", value=True, key="rslog")
        st.plotly_chart(C.equity_curve(align_results(curves), logs),
                        use_container_width=True, config={"displaylogo": False})
        st.plotly_chart(C.underwater(curves), use_container_width=True,
                        config={"displaylogo": False})

        left, right = st.columns([1.15, 1])
        with left:
            st.plotly_chart(C.monthly_heatmap(r_main, ppy=ppy),
                            use_container_width=True,
                            config={"displaylogo": False})
        with right:
            st.plotly_chart(C.return_distribution(r_main, ppy=ppy),
                            use_container_width=True,
                            config={"displaylogo": False})
            win = min(ppy, max(6, len(r_main) // 6))
            _u = M.freq_words(ppy)[2]
            st.plotly_chart(
                C.rolling_metric(M.rolling_sharpe(r_main, win, ppy),
                                 "Rolling Sharpe", ref=0.0,
                                 title=f"Rolling Sharpe over {win} {_u}"),
                use_container_width=True, config={"displaylogo": False})

        eyebrow("Full statistics")
        order = list(stats.keys())
        tbl = pd.DataFrame({"Metric": order,
                            main_col: [M.format_metric(k, stats[k]) for k in order]})
        if bstats:
            tbl[bench_col] = [M.format_metric(k, bstats.get(k, np.nan)) for k in order]
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=560)

        eyebrow("Trailing periods")
        rp = M.period_table(eq_main, r_main, eq_bench, r_bench, ppy,
                            str(main_col), str(bench_col or "Benchmark"))
        rfmt = _fmt_period_tables(rp, str(main_col),
                                  str(bench_col) if bench_col else None)
        if not rfmt["trailing"].empty:
            _rc = [c for c in rfmt["trailing"].columns
                   if c not in ("Period", "Annualized", "From", "To")]
            st.dataframe(signed(rfmt["trailing"], _rc, emphasise=["Excess"]),
                         use_container_width=True, hide_index=True)
            note("Periods longer than one year are annualized; shorter ones "
                 "are cumulative.")

        eyebrow("Calendar years")
        if not rfmt["calendar"].empty:
            _rc = [c for c in rfmt["calendar"].columns
                   if c not in ("Year", "Partial")]
            st.dataframe(signed(rfmt["calendar"], _rc, emphasise=["Excess"]),
                         use_container_width=True, hide_index=True)
            cyr = rp["calendar"]
            if str(main_col) in cyr.columns:
                st.plotly_chart(
                    C.bar_series([str(y) for y in cyr["Year"]],
                                 (cyr[str(main_col)] * 100).tolist(),
                                 "Return by calendar year", "%"),
                    use_container_width=True, config={"displaylogo": False})

        eyebrow("Main drawdown episodes")
        dd = M.drawdown_table(eq_main, 6, ppy)
        if not dd.empty:
            dd["Drawdown"] = dd["Drawdown"].map(lambda v: f"{v*100:.2f}%")
            dd["Recovery"] = dd["Recovery"].astype(str)
        st.dataframe(signed(dd, ["Drawdown"]), use_container_width=True,
                     hide_index=True)

        eyebrow("Period returns")
        pr = M.monthly_returns(r_main, ppy)
        if not pr.empty:
            st.dataframe(signed((pr * 100).round(2)), use_container_width=True)

    with rs_tabs[1]:
        note("A track record is one sample. These tests ask how much of it "
             "survives being cut up or reshuffled. Parameter and cost tests "
             "do not apply: there is no model here to re-run, only the "
             "realized stream.")
        eyebrow("Stability over sub-periods")
        nf = st.slider("Number of folds", 3, 10, 5, key="rsfold")
        wf = R.fold_stats(r_main, nf, ppy)
        if not wf.empty:
            disp = wf.copy()
            for cc in ("CAGR", "Volatility", "Max Drawdown"):
                disp[cc] = disp[cc].map(lambda v: f"{v*100:.2f}%")
            disp["Sharpe"] = disp["Sharpe"].map(lambda v: f"{v:.2f}")
            st.dataframe(signed(disp, ["CAGR", "Sharpe", "Max Drawdown"]),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(C.bar_series(wf["Fold"], wf["Sharpe"], "Sharpe by fold"),
                            use_container_width=True, config={"displaylogo": False})

        eyebrow("Sampling uncertainty")
        s1, s2 = st.columns(2)
        nsim = s1.slider("Simulations", 100, 2000, 500, 100, key="rsmc")
        blk = s2.slider("Block size (periods)", 2, max(3, min(63, len(r_main) // 8)),
                        min(21, max(3, len(r_main) // 20)), 1, key="rsblk")
        if st.button("Run Monte Carlo simulation", key="rsmcbtn"):
            st.session_state["rsmc_res"] = R.monte_carlo(r_main, nsim, blk, ppy)
        if "rsmc_res" in st.session_state:
            mc = st.session_state["rsmc_res"]
            if not mc["paths"].empty:
                st.plotly_chart(C.monte_carlo_fan(mc["paths"], eq_main),
                                use_container_width=True, config={"displaylogo": False})
                m1, m2, m3 = st.columns(3)
                with m1:
                    dial("Simulated median CAGR", f"{mc['median_cagr']*100:.2f}%")
                with m2:
                    dial("Probability of loss", f"{mc['prob_loss']*100:.1f}%")
                with m3:
                    dial("Probability of a drawdown > 20%", f"{mc['prob_dd_20']*100:.1f}%")
                sdf = mc["stats"].copy()
                for cc in ("CAGR", "Max Drawdown"):
                    sdf[cc] = sdf[cc].map(lambda v: f"{v*100:.2f}%")
                sdf["Sharpe"] = sdf["Sharpe"].map(lambda v: f"{v:.2f}")
                st.dataframe(signed(sdf, ["CAGR", "Max Drawdown", "Sharpe"]),
                             use_container_width=True, hide_index=True)
            else:
                note("Not enough observations to resample meaningfully.")

    with rs_tabs[2]:
        note("Exports reproduce the statistics shown above for the selected "
             "series.")
        rs_cfg = RunConfig(label=str(main_col),
                           engine=EngineConfig(initial_capital=float(rs_capital),
                                               periods_per_year=ppy))
        pseudo = _pseudo_result(r_main, eq_main, str(main_col))
        pseudo_b = _pseudo_result(r_bench, eq_bench, str(bench_col)) if bench_col else None
        ts = REPORT.render_tearsheet(pseudo, pseudo_b, stats, bstats, rs_cfg,
                                     simple=True)
        st.download_button("Download tearsheet report (HTML)", ts.encode("utf-8"),
                           f"tearsheet_{main_col}.html", "text/html", key="rsts")

        out = pd.DataFrame({"return": r_main, "value": eq_main})
        if bench_col:
            out["benchmark_return"] = r_bench
            out["benchmark_value"] = eq_bench
        e1, e2, e3 = st.columns(3)
        e1.download_button("Series (CSV)", out.to_csv().encode("utf-8"),
                           "return_stream.csv", "text/csv", key="rscsv")
        stat_df = pd.DataFrame({"Metric": list(stats.keys()),
                                main_col: list(stats.values())})
        if bstats:
            stat_df[bench_col] = [bstats.get(k, np.nan) for k in stats]
        e2.download_button("Statistics (CSV)", stat_df.to_csv(index=False).encode("utf-8"),
                           "statistics.csv", "text/csv", key="rsstat")
        try:
            rbook = XL.workbook_from_returns(
                r_main, eq_main, stats, r_bench, eq_bench, bstats or None,
                ppy, str(main_col), str(bench_col or "Benchmark"),
                all_series=rets,
                report_notes={"Scale read": rrep.scale,
                              "Source file": rfile.name})
            e3.download_button("Full report (Excel)", rbook,
                               f"return_stream_{main_col}.xlsx",
                               "application/vnd.openxmlformats-officedocument."
                               "spreadsheetml.sheet", key="rsxl")
        except Exception as exc:
            st.markdown(f'<div class="flag">Excel export unavailable: {exc}</div>',
                        unsafe_allow_html=True)
        note("The workbook carries every table behind this report: "
             "statistics, trailing periods, calendar years, drawdown "
             "episodes, monthly returns, the full series, and every column "
             "from the uploaded file \u2014 plus a Notes sheet recording the "
             "frequency and scale that were read.")
    st.stop()


prices_raw: Optional[pd.DataFrame] = None
upload_error = None

if source == "Yahoo Finance":
    preset_names = ["\u2014 custom \u2014"] + PRESETS.names()
    preset = st.sidebar.selectbox(
        "Preset universe", preset_names,
        help="Loads a ready-made set of symbols. Everything stays editable "
             "afterwards, and the benchmark and cash proxy are filled in to "
             "match.")
    if preset != st.session_state.get("_preset_applied"):
        st.session_state["_preset_applied"] = preset
        if not preset.startswith("\u2014"):
            info = PRESETS.get(preset)
            st.session_state["tickers_box"] = "\n".join(info["tickers"])
            st.session_state["_preset_bench"] = info.get("benchmark")
            st.session_state["_preset_cash"] = info.get("cash")
    if not preset.startswith("\u2014"):
        st.sidebar.markdown(
            f'<div class="note">{PRESETS.get(preset).get("note","")}</div>',
            unsafe_allow_html=True)

    st.session_state.setdefault("tickers_box", "\n".join(d0.tickers))
    tickers_txt = st.sidebar.text_area(
        "Symbols (one per line or comma-separated)",
        key="tickers_box", height=110,
        help="Add .TO for Toronto, .V for TSX-V, no suffix for U.S. tickers.")
    tickers = [t.strip().upper() for t in
               tickers_txt.replace(",", "\n").replace(";", "\n").split("\n") if t.strip()]
    col1, col2 = st.sidebar.columns(2)
    start = col1.date_input("Start", value=pd.to_datetime(d0.start).date(),
                            min_value=date(1970, 1, 1), max_value=date.today())
    end = col2.date_input("End", value=date.today(),
                          min_value=date(1971, 1, 1), max_value=date.today())
else:
    up = st.sidebar.file_uploader("Price file", type=["csv", "xlsx", "xls", "txt"])
    sheet = None
    tickers = []
    start, end = None, None
    if up is not None:
        content = up.getvalue()
        if up.name.lower().endswith((".xlsx", ".xls")):
            names = excel_sheet_names(io.BytesIO(content))
            if len(names) > 1:
                sheet = st.sidebar.selectbox("Sheet", names)
        try:
            prices_raw = parse_upload(content, up.name, sheet)
            tickers = [str(c) for c in prices_raw.columns]
        except Exception as exc:
            upload_error = str(exc)
    else:
        st.sidebar.markdown(
            '<div class="note">Expected columns: a date column, then one '
            'price column per instrument. Long format (date, symbol, '
            'price) is also recognized.</div>',
            unsafe_allow_html=True)

univ_options = tickers or []
sel_universe = st.sidebar.multiselect(
    "Investable universe", univ_options,
    default=[t for t in (d0.tickers if source == "Yahoo Finance" else univ_options)
             if t in univ_options] or univ_options)

BLEND = "\u2014 blend of several \u2014"
bench_choices = ["\u2014 none \u2014"] + univ_options + [BLEND]
_pb = st.session_state.get("_preset_bench")
bench_default = _pb if _pb in univ_options else (
    d0.benchmark if d0.benchmark in univ_options else None)
benchmark = st.sidebar.selectbox(
    "Comparison benchmark", bench_choices,
    index=bench_choices.index(bench_default) if bench_default else 0)

bench_blend: Dict[str, float] = {}
bench_blend_rule = "A"
bench_extra: List[str] = []
if benchmark == BLEND:
    bench_spec = st.sidebar.text_area(
        "Benchmark weights", value=st.session_state.get(
            "benchspec", "XIC.TO:60, XBB.TO:40"),
        height=68, key="benchspec",
        help="TICKER:WEIGHT, comma or line separated. Percentages or "
             "fractions both work. Instruments outside the investable "
             "universe are downloaded alongside it.")
    bench_blend = ALLOC.parse_fixed(
        bench_spec, univ_options + [t.strip().upper() for t in
                                    bench_spec.replace(":", " ").replace(",", " ")
                                    .replace(";", " ").split()
                                    if any(ch.isalpha() for ch in t)])
    bench_blend_rule = st.sidebar.selectbox(
        "Benchmark rebalance", list(REBALANCE_RULES.keys()),
        index=list(REBALANCE_RULES.keys()).index("A"),
        format_func=lambda k: REBALANCE_RULES[k],
        help="How often the blend returns to its target weights. An "
             "unrebalanced 60/40 drifts toward equities over a long window, "
             "which quietly changes the bar being measured against.")
    if bench_blend:
        _tot = sum(abs(v) for v in bench_blend.values()) or 1.0
        st.sidebar.markdown(
            '<div class="note">' + ", ".join(
                f"{k} {abs(v)/_tot*100:.0f}%" for k, v in bench_blend.items())
            + "</div>", unsafe_allow_html=True)
        bench_extra = [t for t in bench_blend if t not in univ_options]
    else:
        st.sidebar.markdown('<div class="flag">No valid weight parsed. Use '
                            'TICKER:WEIGHT.</div>', unsafe_allow_html=True)
    benchmark = BLEND if bench_blend else None
else:
    benchmark = None if benchmark == "\u2014 none \u2014" else benchmark

BENCH_MODES = ["Total return (adjusted close)",
               "Price return + dividends reinvested at rebalance",
               "Price return only"]
bench_mode = BENCH_MODES[0]
if benchmark:
    bench_mode = st.sidebar.selectbox(
        "Benchmark convention", BENCH_MODES,
        help="Set independently of the strategy. A benchmark measured on "
             "price return only is not the index anyone actually tracks and "
             "understates the bar by roughly its dividend yield each year. "
             "Adjusted close reinvests continuously; the middle option lets "
             "dividends sit in cash until the rebalance, which is closer to "
             "how a real account behaves.")

if source == "Yahoo Finance":
    price_mode = st.sidebar.selectbox(
        "Price convention",
        ["Total return (dividends reinvested)", "Price return + cash dividends"],
        index=1 if (not d0.adjusted) else 0,
        help="Total return folds dividends into the price series, so they "
             "compound inside the position from the instant they are paid. "
             "The second option keeps prices ex-dividend and credits each "
             "payment as cash on its ex-date, where it sits until the next "
             "rebalance. Same cash in, different timing.")
    adjusted = price_mode.startswith("Total")
    use_divs = not adjusted
else:
    adjusted, use_divs = True, False

cash_choices = ["Fixed rate"] + univ_options
_pc = st.session_state.get("_preset_cash")
_cash_idx = cash_choices.index(_pc) if _pc in cash_choices else 0
cash_proxy = st.sidebar.selectbox("Cash remuneration", cash_choices,
                                  index=_cash_idx,
                                  help="A cash-equivalent ETF (e.g. PSA.TO) gives a "
                                       "realistic opportunity cost for staying out "
                                       "of the market.")
cash_proxy = None if cash_proxy == "Fixed rate" else cash_proxy

# --- Exogenous series ---
st.sidebar.markdown('<div class="eyebrow">Exogenous series</div>', unsafe_allow_html=True)
exog_files = st.sidebar.file_uploader(
    "Economic, fundamental, or signal data",
    type=["csv", "xlsx", "xls", "txt"], accept_multiple_files=True, key="exogup",
    help="One column per series for macro data; one column per symbol for a "
         "cross-sectional factor. The first sheet of Excel workbooks is used.")

exog_raw: Optional[pd.DataFrame] = None
exog_lag = int(x0.publication_lag_days)
if exog_files:
    frames = []
    for f in exog_files:
        try:
            fr = parse_exog(f.getvalue(), f.name, None)
            fr.columns = [f"{c}" if c not in
                          [x for fr2 in frames for x in fr2.columns]
                          else f"{c} ({f.name})" for c in fr.columns]
            frames.append(fr)
        except Exception as exc:
            st.sidebar.error(f"{f.name}: {exc}")
    if frames:
        exog_raw = pd.concat(frames, axis=1).sort_index()
        exog_raw = exog_raw.loc[:, ~exog_raw.columns.duplicated(keep="first")]
        exog_lag = st.sidebar.slider(
            "Publication lag (calendar days)", 0, 120, exog_lag, 1,
            help="Delay between a data point's reference date and its release. "
                 "A monthly series typically needs 20 to 45 days; an in-house "
                 "price or ratio, 0 or 1. This lag is what keeps the backtest "
                 "from using a data point before it was published.")
        st.sidebar.markdown(
            f'<div class="note">{exog_raw.shape[1]} series loaded, '
            f'from {exog_raw.index.min().date()} to {exog_raw.index.max().date()}.</div>',
            unsafe_allow_html=True)

exog_columns = list(exog_raw.columns) if exog_raw is not None else []

# --- Signal origin ---
st.sidebar.markdown('<div class="eyebrow">Signal</div>', unsafe_allow_html=True)
mode_labels = ["Built-in model", "Imported target weights"]
mode_label = st.sidebar.radio(
    "Signal origin", mode_labels,
    index=1 if s0.mode == "external_weights" else 0,
    label_visibility="collapsed",
    help="\u201cImported target weights\u201d replaces the signal generator "
         "with an allocation file: the engine applies drift, frictions, and "
         "the execution lag exactly as it would for a built-in model.")
mode = "external_weights" if mode_label == mode_labels[1] else "builtin"

weights_raw: Optional[pd.DataFrame] = None
w_normalize = s0.weights_normalize
w_calendar = s0.weights_calendar
w_source = ""

if mode == "external_weights":
    wf = st.sidebar.file_uploader("Target weights file",
                                  type=["csv", "xlsx", "xls", "txt"], key="wup")
    if wf is not None:
        w_source = wf.name
        wsheet = None
        if wf.name.lower().endswith((".xlsx", ".xls")):
            names = excel_sheet_names(io.BytesIO(wf.getvalue()))
            if len(names) > 1:
                wsheet = st.sidebar.selectbox("Sheet", names, key="wsheet")
        try:
            weights_raw = parse_weights(wf.getvalue(), wf.name, wsheet)
        except Exception as exc:
            st.sidebar.error(f"Could not read the file: {exc}")
    else:
        st.sidebar.markdown(
            '<div class="note">A date column, then one column per symbol '
            'holding the target weight. Long format (date, symbol, weight) '
            'is also recognized. Fractions or percentages, either works.</div>',
            unsafe_allow_html=True)
        st.sidebar.download_button(
            "CSV template", weights_template(sel_universe or ["XIC.TO", "ZEB.TO"]),
            "target_weights_template.csv", "text/csv")

    w_normalize = st.sidebar.selectbox(
        "Row handling", ["None", "Scale to 100%"],
        index=0 if w_normalize == "None" else 1,
        help="\u201cNone\u201d respects the file: a row at 80% leaves 20% in "
             "cash. \u201cScale to 100%\u201d rescales the weights.")
    w_calendar = st.sidebar.selectbox(
        "Rebalance calendar", ["File dates", "Engine calendar"],
        index=0 if w_calendar == "File dates" else 1,
        help="\u201cFile dates\u201d only trades on the supplied dates. "
             "\u201cEngine calendar\u201d additionally resets the portfolio "
             "to the last known weights at every engine checkpoint, "
             "correcting drift.")
    strategy = None
    strat_key = "external_weights"
    params = {}

# --- Strategy ---
strat_keys = sorted(REGISTRY.keys(), key=lambda k: REGISTRY[k].label)
if mode == "builtin":
    strat_key = st.sidebar.selectbox(
        "Model", strat_keys,
        index=strat_keys.index(s0.name) if s0.name in strat_keys else 0,
        format_func=lambda k: REGISTRY[k].label, label_visibility="collapsed")
    strategy = REGISTRY[strat_key]
    st.sidebar.markdown(f'<div class="note">{strategy.description}</div>',
                        unsafe_allow_html=True)
    if strategy.needs_exog and not exog_columns:
        st.sidebar.markdown(
            '<div class="flag">This model reads exogenous series. Upload a '
            'file above, or it will have no signal.</div>',
            unsafe_allow_html=True)
    st.sidebar.write("")

# --- Portfolio construction -------------------------------------------
# Sleeves sit above the model: the model still picks holdings, the sleeve
# decides how much of the portfolio it gets to pick for.
construction = "Single strategy"
core_exclusive = True
class_map: Dict[str, str] = {}
class_budgets: Dict[str, float] = {}
core_spec, core_budget = "", 0.5

if mode == "builtin":
    construction = st.sidebar.selectbox(
        "Portfolio construction", ["Single strategy", "By asset class",
                                   "Core + strategy"],
        help="\u201cBy asset class\u201d gives each class a fixed budget and "
             "runs the model inside each one, so it picks the best bonds "
             "among bonds. \u201cCore + strategy\u201d holds a fixed sleeve "
             "permanently and runs the model on the rest.")

    if construction == "By asset class":
        with st.sidebar.expander("Asset classes", expanded=True):
            _preset_name = st.session_state.get("_preset_applied", "")
            _default_cls = PRESETS.classes_for(_preset_name, sel_universe)
            _opts = ALLOC.DEFAULT_CLASSES + ["Unclassified"]
            st.markdown('<div class="note">Assign each instrument, then set '
                        'a budget per class.</div>', unsafe_allow_html=True)
            for t in sel_universe:
                d = _default_cls.get(t, "Unclassified")
                class_map[t] = st.selectbox(
                    t, _opts, index=_opts.index(d) if d in _opts else len(_opts) - 1,
                    key=f"cls_{t}")

            _used = sorted({c for c in class_map.values() if c != "Unclassified"})
            if _used:
                st.markdown('<div class="eyebrow" style="margin:.7rem 0 .3rem;">'
                            'Budgets</div>', unsafe_allow_html=True)
                _even = round(100.0 / len(_used))
                for c in _used:
                    class_budgets[c] = st.number_input(
                        f"{c} (%)", 0.0, 100.0, float(_even), 5.0,
                        key=f"bud_{c}") / 100.0
                _tot = sum(class_budgets.values())
                _msg = (f'<div class="flag">Budgets total {_tot*100:.0f}%. '
                        f'The remainder stays in cash.</div>' if _tot < 0.999
                        else f'<div class="note">Budgets total {_tot*100:.0f}%.</div>')
                st.markdown(_msg, unsafe_allow_html=True)
            else:
                st.markdown('<div class="flag">Assign at least one instrument '
                            'to a class.</div>', unsafe_allow_html=True)

    elif construction == "Core + strategy":
        with st.sidebar.expander("Core sleeve", expanded=True):
            core_budget = st.slider("Core share (%)", 0, 100, 50, 5,
                                    key="corebud") / 100.0
            core_spec = st.text_area(
                "Core holdings", value=st.session_state.get(
                    "corespec", ", ".join(f"{t}:{round(100/max(len(sel_universe[:2]),1))}"
                                          for t in sel_universe[:2])),
                height=70, key="corespec",
                help="One per line or comma separated, as TICKER:WEIGHT. "
                     "Percentages or fractions both work; only the ratios "
                     "matter, since the sleeve is scaled to its share.")
            core_exclusive = st.checkbox(
                "Keep core holdings out of the strategy universe",
                value=True, key="coreexcl",
                help="On, the model never picks a name the core already "
                     "holds, so the core position is exactly the core share. "
                     "Off, the model may add to it, and the combined weight "
                     "can exceed the core share.")
            _parsed = ALLOC.parse_fixed(core_spec, sel_universe)
            if _parsed:
                _t = sum(abs(v) for v in _parsed.values()) or 1.0
                st.markdown(
                    '<div class="note">Core: ' + ", ".join(
                        f"{k} {abs(v)/_t*core_budget*100:.1f}%"
                        for k, v in _parsed.items())
                    + f' \u00b7 strategy gets {(1-core_budget)*100:.0f}%.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown('<div class="flag">No valid holding parsed. Use '
                            'TICKER:WEIGHT.</div>', unsafe_allow_html=True)

params: Dict[str, Any] = {}
for p in (strategy.params if mode == "builtin" else []):
    key = f"p_{strat_key}_{p.key}"
    default = s0.params.get(p.key, p.default) if s0.name == strat_key else p.default
    if p.kind == "int":
        params[p.key] = st.sidebar.slider(p.label, int(p.min), int(p.max),
                                          int(default), int(p.step),
                                          help=p.help or None, key=key)
    elif p.kind == "float":
        params[p.key] = st.sidebar.slider(p.label, float(p.min), float(p.max),
                                          float(default), float(p.step),
                                          help=p.help or None, key=key)
    elif p.kind == "bool":
        params[p.key] = st.sidebar.checkbox(p.label, bool(default),
                                            help=p.help or None, key=key)
    elif p.kind == "series":
        opts = exog_columns or ["\u2014 no series imported \u2014"]
        idx = opts.index(default) if default in opts else 0
        choice = st.sidebar.selectbox(p.label, opts, index=idx,
                                      help=p.help or None, key=key)
        params[p.key] = "" if choice.startswith("\u2014") else choice
    elif p.kind == "formula":
        params[p.key] = st.sidebar.text_area(
            p.label, str(default or ""), height=80,
            help=(p.help or "") + " Edit and test in the Builder tab.",
            key=key)
    elif p.kind == "choice" and p.choices:
        opts = list(p.choices)
        idx = opts.index(default) if default in opts else 0
        params[p.key] = st.sidebar.selectbox(p.label, opts, index=idx,
                                             help=p.help or None, key=key)
    else:
        params[p.key] = st.sidebar.text_input(p.label, str(default or ""),
                                              help=p.help or None, key=key)

# --- Execution and frictions ---
# Folded by default: these hold steady across runs, while the universe and
# the signal are what actually get changed between one backtest and the next.
with st.sidebar.expander("Execution", expanded=False):
    rb_keys = list(REBALANCE_RULES.keys())
    rebalance = st.selectbox(
        "Rebalance", rb_keys, index=rb_keys.index(e0.rebalance),
        format_func=lambda k: REBALANCE_RULES[k])
    lag = st.slider("Execution lag (days)", 0, 5, int(e0.execution_lag),
                    help="1 = signal at the close, executed the next session. "
                         "0 assumes execution at the price that produced the signal.")
    exec_price = st.selectbox(
        "Execution price", ["Close", "Open (marked at the close)"],
        index=1 if e0.execute_at_open else 0,
        help="Trading at the open splits the day in two: the overnight move is "
             "earned on the old weights, the intraday move on the new ones. It is "
             "the more realistic assumption for an order placed after a "
             "prior-close signal. Only available with Yahoo Finance data.")
    exec_at_open = exec_price.startswith("Open")
    if exec_at_open and source != "Yahoo Finance":
        st.markdown('<div class="flag">Opening prices come from Yahoo '
                    'Finance only. Uploaded files fall back to close '
                    'execution.</div>', unsafe_allow_html=True)
    trim_warm = st.checkbox(
        "Trim the warm-up period", value=bool(e0.trim_warmup),
        help="Indicators are blind until they have enough history. Those early "
             "sessions sit in cash and still earn the cash rate, which lifts the "
             "reported return and dilutes volatility. Trimming starts the record "
             "on the first day capital is actually at risk, for the strategy and "
             "the benchmark alike.")
    capital = st.number_input("Initial capital ($)", 1_000, 1_000_000_000,
                              int(e0.initial_capital), 10_000)
    max_lev = st.slider("Max leverage", 0.5, 2.0, float(e0.max_leverage), 0.1)

with st.sidebar.expander("Frictions", expanded=False):
    comm = st.number_input("Commission (bps)", 0.0, 200.0, float(c0.commission_bps), 1.0)
    slip = st.number_input("Slippage (bps)", 0.0, 500.0, float(c0.slippage_bps), 5.0)
    cash_rate = st.number_input("Cash rate (annual %)", 0.0, 15.0,
                                float(c0.cash_rate_pa * 100), 0.25) / 100.0

st.sidebar.markdown("")

run_clicked = st.sidebar.button("Run backtest")

# ----------------------------------------------------------------------
# Configuration assembly
# ----------------------------------------------------------------------
_suffix = {"By asset class": " by class",
           "Core + strategy": " + core"}.get(construction, "")
run_label = ((strategy.label + _suffix) if mode == "builtin"
             else (f"Imported weights \u2014 {w_source}" if w_source else "Imported weights"))

cfg = RunConfig(
    label=run_label,
    data=DataConfig(
        source="yfinance" if source == "Yahoo Finance" else "upload",
        tickers=sel_universe,
        start=str(start) if start else "1990-01-01",
        end=str(end) if end else None,
        benchmark=(None if benchmark == BLEND else benchmark),
        cash_proxy=cash_proxy,
        adjusted=bool(adjusted), use_dividends=bool(use_divs),
    ),
    exog=ExogConfig(
        enabled=exog_raw is not None,
        publication_lag_days=int(exog_lag),
        columns=exog_columns,
        note=", ".join(f.name for f in exog_files) if exog_files else "",
    ),
    strategy=StrategyConfig(mode=mode, name=strat_key, params=params,
                            weights_normalize=w_normalize,
                            weights_calendar=w_calendar,
                            weights_source=w_source),
    engine=EngineConfig(initial_capital=float(capital), rebalance=rebalance,
                        execution_lag=int(lag), max_leverage=float(max_lev),
                        execute_at_open=bool(exec_at_open and source == "Yahoo Finance"),
                        trim_warmup=bool(trim_warm)),
    costs=CostConfig(commission_bps=comm, slippage_bps=slip, cash_rate_pa=cash_rate),
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    '<div class="masthead"><h1>Quant Backtest Studio</h1>'
    '<div class="sub">Signal &nbsp;\u00b7&nbsp; Simulation &nbsp;\u00b7&nbsp; Robustness</div></div>',
    unsafe_allow_html=True)

if upload_error:
    st.error(f"The file could not be read: {upload_error}")

problems = cfg.validate()
blocking = [p for p in problems if not p.startswith("WARNING")]
for p in problems:
    (st.error if p in blocking else st.warning)(p)

# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_market(tickers: tuple, start: str, end: Optional[str],
                 adjusted: bool, want_open: bool, want_div: bool) -> MarketData:
    return load_market_data(list(tickers), start, end, adjusted=adjusted,
                            want_open=want_open, want_dividends=want_div)


def build_market() -> MarketData:
    """Prices, plus opens and dividends when the settings call for them."""
    if source == "Yahoo Finance":
        needed = list(dict.fromkeys(
            sel_universe
            + [t for t in (benchmark, cash_proxy) if t and t != BLEND]
            + list(bench_blend.keys())))
        # Adjusted close is fetched whenever the benchmark is measured on
        # total return while the strategy runs on price-return prices.
        need_div = bool(use_divs) or (
            benchmark is not None
            and bench_mode.startswith("Price return + dividends"))
        return fetch_market(tuple(needed), str(start), str(end),
                            bool(adjusted), bool(exec_at_open), need_div)
    if prices_raw is None:
        raise RuntimeError("No file loaded.")
    return MarketData(close=prices_raw, adjusted=True)


if run_clicked and not blocking:
    try:
        with st.spinner("Loading prices..."):
            market = build_market()
            prices, quality = clean_prices(market.close, cfg.data)
            for msg in market.notes:
                quality.warnings.append(msg)

        cash_px = prices[cash_proxy] if (cash_proxy and cash_proxy in prices.columns) else None
        drop = [c for c in [cash_proxy] if c and c in prices.columns and c not in sel_universe]
        universe = prices.drop(columns=drop)
        universe = universe[[c for c in sel_universe if c in universe.columns]]

        if universe.empty or universe.shape[1] == 0:
            st.error("No usable instrument in the selected universe.")
            st.stop()

        # Exogenous series: publication lag, then alignment
        exog_aligned, ex_rep = None, None
        if exog_raw is not None and not exog_raw.empty:
            exog_aligned = prepare_exog(exog_raw, universe.index, exog_lag)
            ex_rep = exog_report(exog_raw, exog_aligned, exog_lag, universe.index)

        w_report, rebal_dates, sleeve_report = None, None, None
        with st.spinner("Generating signals and running simulation..."):
            if mode == "external_weights":
                if weights_raw is None or weights_raw.empty:
                    st.error("No target weights file was imported.")
                    st.stop()
                weights, rebal_dates, w_report = prepare_target_weights(
                    weights_raw, universe.index, list(universe.columns),
                    w_normalize, float(max_lev))
                if w_calendar == "Engine calendar":
                    # Weights stay those from the file, but the portfolio is
                    # reset onto them at every engine checkpoint: drift
                    # between two file rows is corrected.
                    rebal_dates = None
            else:
                # Sleeves resolve to an ordinary weight frame; from here the
                # engine cannot tell whether sleeves were involved.
                if construction == "By asset class" and class_budgets:
                    sleeves = ALLOC.sleeves_from_classes(
                        {t: c for t, c in class_map.items() if t in universe.columns},
                        class_budgets, strat_key, params)
                    weights, sleeve_report = ALLOC.resolve(
                        universe, sleeves, REGISTRY, exog_aligned,
                        float(cfg.engine.max_leverage))
                elif construction == "Core + strategy":
                    core = ALLOC.parse_fixed(core_spec, list(universe.columns))
                    sleeves = []
                    if core and core_budget > 0:
                        sleeves.append(ALLOC.Sleeve(
                            "Core", float(core_budget), list(core),
                            mode="fixed", fixed=core))
                    sat = 1.0 - float(core_budget)
                    if sat > 1e-9:
                        pool = [c for c in universe.columns
                                if not (core_exclusive and c in core)]
                        if not pool:
                            st.warning(
                                "The core covers the whole universe, leaving "
                                "the strategy nothing to pick from. Its share "
                                "stays in cash.")
                        else:
                            sleeves.append(ALLOC.Sleeve(
                                "Strategy", sat, pool,
                                mode="strategy", strategy_key=strat_key,
                                params=dict(params)))
                    weights, sleeve_report = ALLOC.resolve(
                        universe, sleeves, REGISTRY, exog_aligned,
                        float(cfg.engine.max_leverage))
                else:
                    weights = strategy.generate(universe, params, exog_aligned)

            cols = list(universe.columns)
            open_px = None
            if market.open is not None and cfg.engine.execute_at_open:
                open_px = market.open.reindex(index=universe.index,
                                              columns=cols)
            div_px = None
            if market.dividends is not None and cfg.data.use_dividends:
                div_px = market.dividends.reindex(index=universe.index,
                                                  columns=cols).fillna(0.0)

            result = run_backtest(universe, weights, cfg.engine, cfg.costs,
                                  cash_px, run_label, rebal_dates,
                                  open_prices=open_px, dividends=div_px)
            bench = None
            if benchmark == BLEND and bench_blend:
                have = {k: v for k, v in bench_blend.items() if k in prices.columns}
                missing = [k for k in bench_blend if k not in prices.columns]
                if missing:
                    quality.warnings.append(
                        "Benchmark component(s) unavailable and dropped: "
                        + ", ".join(missing))
                if have:
                    bsrc, bdiv = prices, None
                    if bench_mode.startswith("Total return") and \
                            market.adj_close is not None:
                        bsrc = market.adj_close.reindex(prices.index)
                    elif bench_mode.startswith("Price return + dividends"):
                        if market.dividends is not None:
                            bdiv = market.dividends.reindex(prices.index)
                        else:
                            quality.warnings.append(
                                "No dividend data for the benchmark blend: it "
                                "is shown on price return only, which "
                                "understates it.")
                    _tot = sum(abs(v) for v in have.values()) or 1.0
                    blabel = " / ".join(
                        f"{k} {abs(v)/_tot*100:.0f}%" for k, v in have.items())
                    try:
                        bench = blended_benchmark(
                            bsrc, have, cfg.engine, blabel,
                            dividends=bdiv, rebalance=bench_blend_rule)
                    except Exception as exc:
                        quality.warnings.append(f"Benchmark blend failed: {exc}")
            elif benchmark and benchmark in prices.columns:
                bseries, bdiv = prices[benchmark], None
                if bench_mode.startswith("Total return"):
                    # If the strategy runs on price-return prices, the
                    # total-return series is the separate adjusted close.
                    if market.adj_close is not None and \
                            benchmark in market.adj_close.columns:
                        bseries = market.adj_close[benchmark].reindex(prices.index)
                elif bench_mode.startswith("Price return + dividends"):
                    if market.dividends is not None and \
                            benchmark in market.dividends.columns:
                        bdiv = market.dividends[benchmark].reindex(prices.index)
                    else:
                        quality.warnings.append(
                            f"No dividend data available for {benchmark}: the "
                            f"benchmark is shown on price return only, which "
                            f"understates it.")
                bench = benchmark_result(bseries.dropna(), cfg.engine, benchmark,
                                         dividends=bdiv,
                                         reinvest_rule=cfg.engine.rebalance)

            # Warm-up: both series must start on the same day or the
            # benchmark is credited with a stretch the strategy sat out.
            raw_start = result.equity.index[0]
            if cfg.engine.trim_warmup:
                # A permanently held core keeps total exposure above zero from
                # session one, hiding the model's warm-up. Start the record
                # where the model itself first takes a position.
                sig_start = (sleeve_report.signal_start
                             if sleeve_report is not None else None)
                result, bench = align_start(
                    result, bench, initial_capital=cfg.engine.initial_capital,
                    signal_start=sig_start)

        st.session_state["run"] = {
            "result": result, "bench": bench, "prices": universe,
            "cash": cash_px, "quality": quality, "cfg": cfg,
            "params": dict(params), "strategy_key": strat_key, "mode": mode,
            "exog": exog_aligned, "exog_raw": exog_raw, "exog_report": ex_rep,
            "weights_report": w_report, "rebalance_dates": rebal_dates,
            "weights": weights,
            "sleeves": sleeve_report,
            "construction": construction,
            "market": market,
            "bench_mode": bench_mode,
            "raw_start": raw_start,
            "trimmed": bool(cfg.engine.trim_warmup
                            and result.equity.index[0] > raw_start),
            "stamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        for k in ("sweep", "mc"):
            st.session_state.pop(k, None)
    except Exception as exc:
        st.error(f"The backtest stopped: {exc}")

run = st.session_state.get("run")

if run is None:
    st.markdown(
        '<div class="note">Pick a source on the left, then run the backtest. '
        'Everything below fills in from that one action.</div>',
        unsafe_allow_html=True)

    eyebrow("Three ways in")
    routes = [
        ("Prices", "Simulate a strategy",
         "Pull ETFs from Yahoo Finance or upload a price file, choose a model, "
         "and the engine handles drift, frictions and execution."),
        ("Weights", "Test an allocation you already have",
         "Upload target weights from a spreadsheet or committee and measure "
         "them under the same frictions as any built-in model."),
        ("Returns", "Analyse a track record",
         "Upload a stream of periodic returns and get the full statistics with "
         "no simulation at all."),
    ]
    cols = st.columns(3)
    for col, (tag, title, body) in zip(cols, routes):
        with col:
            st.markdown(
                f'<div class="startcard"><div class="n">{tag}</div>'
                f'<div class="t">{title}</div><div class="b">{body}</div></div>',
                unsafe_allow_html=True)

    eyebrow("Models available")
    for k in sorted(REGISTRY, key=lambda x: REGISTRY[x].label):
        sx = REGISTRY[k]
        st.markdown(
            f'<div class="strat"><div class="nm">{sx.label}</div>'
            f'<div class="ds">{sx.description}</div></div>',
            unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
res = run["result"]
bench = run["bench"]
universe = run["prices"]
quality = run["quality"]
rcfg: RunConfig = run["cfg"]
ppy = rcfg.engine.periods_per_year
run_mode = run.get("mode", "builtin")
exog_used = run.get("exog")
ex_rep = run.get("exog_report")
w_report = run.get("weights_report")
sleeve_report = run.get("sleeves")
run_rebal = run.get("rebalance_dates")

bench_r = bench.returns if bench is not None else None
stats = M.summary(res.returns, res.equity, bench_r, res.turnover,
                  res.exposure, rcfg.costs.cash_rate_pa, ppy)
bstats = M.summary(bench.returns, bench.equity, None, None, None,
                   rcfg.costs.cash_rate_pa, ppy) if bench is not None else {}

# Context strip: which run is on screen, visible from every tab.
_bits = [f'<span class="lead">{rcfg.label}</span>']
_bits.append(f'<span class="item">{res.equity.index[0].date()} '
             f'&rarr; {res.equity.index[-1].date()}</span>')
_bits.append(f'<span class="item">{len(universe.columns)} instrument'
             f'{"s" if len(universe.columns) != 1 else ""}</span>')
_bits.append(f'<span class="item">{REBALANCE_RULES.get(rcfg.engine.rebalance, "").lower()}</span>')
if bench is not None:
    _bits.append(f'<span class="item">vs <b>{bench.label}</b></span>')
_cagr = stats.get("CAGR", float("nan"))
if _cagr == _cagr:
    _bits.append(f'<span class="item">CAGR <b>{M.format_metric("CAGR", _cagr)}</b></span>')
if run.get("trimmed"):
    _bits.append('<span class="item">warm-up trimmed</span>')
st.markdown(f'<div class="runbar">{"".join(_bits)}</div>', unsafe_allow_html=True)

tabs = st.tabs(["Results", "Positions", "Robustness", "Data", "Builder", "Export"])

# --------------------------- RESULTS -----------------------------------
with tabs[0]:
    if run.get("trimmed"):
        dropped = run["raw_start"].date()
        _why = ("the first day the model took a position"
                if (sleeve_report is not None
                    and sleeve_report.signal_start is not None)
                else "the first day the strategy held a position")
        note(f"Warm-up trimmed: the record starts on "
             f"{res.equity.index[0].date()}, {_why}, rather than {dropped}. "
             f"Anything held before that \u2014 a fixed core, cash \u2014 is "
             f"excluded, and the benchmark is measured over the same window.")
    keys = ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar", "Sortino"]
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        with col:
            v = stats.get(k, np.nan)
            tone = ""
            if k in ("CAGR", "Sharpe", "Calmar", "Sortino"):
                tone = "pos" if (v == v and v > 0) else "neg"
            elif k == "Max Drawdown":
                tone = "neg"
            sub = ""
            if bstats:
                bv = bstats.get(k, np.nan)
                if bv == bv:
                    sub = f"bench. {M.format_metric(k, bv)}"
            dial(k, M.format_metric(k, v), sub, tone)

    st.write("")
    c1, c2 = st.columns([3, 1])
    log_scale = c2.toggle("Log scale", value=True,
                          help="A log scale makes relative changes comparable "
                               "across the whole period.")
    curves = {res.label: res.equity}
    if bench is not None:
        curves[bench.label] = bench.equity
    rebased = align_results(curves)
    st.plotly_chart(C.equity_curve(rebased, log_scale), use_container_width=True,
                    config={"displaylogo": False})

    st.plotly_chart(C.underwater({k: v for k, v in curves.items()}),
                    use_container_width=True, config={"displaylogo": False})

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(C.monthly_heatmap(res.returns, ppy=ppy),
                        use_container_width=True,
                        config={"displaylogo": False})
    with right:
        st.plotly_chart(C.return_distribution(res.returns, ppy=ppy),
                        use_container_width=True,
                        config={"displaylogo": False})
        win = min(252, max(63, len(res.returns) // 6))
        st.plotly_chart(
            C.rolling_metric(M.rolling_sharpe(res.returns, win, ppy),
                             "Rolling Sharpe", ref=0.0,
                             title=f"Rolling Sharpe over {win} sessions"),
            use_container_width=True, config={"displaylogo": False})

    eyebrow("Full statistics")
    order = list(stats.keys())
    tbl = pd.DataFrame({
        "Metric": order,
        rcfg.label: [M.format_metric(k, stats[k]) for k in order],
    })
    if bstats:
        tbl[bench.label] = [M.format_metric(k, bstats.get(k, np.nan)) for k in order]
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=560)

    eyebrow("Trailing periods")
    ptabs = M.period_table(res.equity, res.returns,
                           bench.equity if bench is not None else None,
                           bench.returns if bench is not None else None,
                           ppy, rcfg.label, bench.label if bench is not None else "Benchmark")
    fmt = _fmt_period_tables(ptabs, rcfg.label,
                             bench.label if bench is not None else None)
    if not fmt["trailing"].empty:
        _cols = [c for c in fmt["trailing"].columns
                 if c not in ("Period", "Annualized", "From", "To")]
        st.dataframe(signed(fmt["trailing"], _cols, emphasise=["Excess"]),
                     use_container_width=True, hide_index=True)
        note("Periods longer than one year are annualized; shorter ones are "
             "cumulative, as the <b>Annualized</b> column records. A window "
             "the history does not cover is omitted rather than measured "
             "over a shorter span and labelled as though it were complete.")

    eyebrow("Calendar years")
    if not fmt["calendar"].empty:
        _cols = [c for c in fmt["calendar"].columns
                 if c not in ("Year", "Partial")]
        st.dataframe(signed(fmt["calendar"], _cols, emphasise=["Excess"]),
                     use_container_width=True, hide_index=True)
        cy = ptabs["calendar"]
        if rcfg.label in cy.columns:
            st.plotly_chart(
                C.bar_series([str(y) for y in cy["Year"]],
                             (cy[rcfg.label] * 100).tolist(),
                             "Return by calendar year", "%"),
                use_container_width=True, config={"displaylogo": False})
        if (cy["Partial"].astype(str) != "").any():
            note("A partial first or last year is flagged in the "
                 "<b>Partial</b> column: those are not full-year figures.")

    eyebrow("Main drawdown episodes")
    dd_tbl = M.drawdown_table(res.equity, 6, ppy)
    if not dd_tbl.empty:
        dd_tbl["Drawdown"] = dd_tbl["Drawdown"].map(lambda v: f"{v*100:.2f}%")
        # "Recovery" mixes date objects with the string "ongoing"; Arrow
        # (Streamlit's serialization layer) rejects a mixed-type column.
        dd_tbl["Recovery"] = dd_tbl["Recovery"].astype(str)
    st.dataframe(signed(dd_tbl, ["Drawdown"]), use_container_width=True,
                 hide_index=True)

# --------------------------- POSITIONS ----------------------------------
with tabs[1]:
    if w_report is not None:
        eyebrow("Target weights file check")
        a, b, c, d = st.columns(4)
        with a:
            dial("Rebalance dates", f"{w_report.n_dates:,}")
        with b:
            dial("Weighted instruments", f"{w_report.n_instruments}")
        with c:
            dial("Average exposure", f"{w_report.mean_gross*100:,.1f}%",
                 f"max {w_report.max_gross*100:,.1f}%")
        with d:
            dial("Detected scale", w_report.scale.capitalize(),
                 "short positions" if w_report.has_shorts else "long only")
        for wmsg in w_report.warnings:
            st.markdown(f'<div class="flag">{wmsg}</div>', unsafe_allow_html=True)
        with st.expander("First rows retained, after calendaring onto trading days"):
            st.dataframe((w_report.preview * 100).round(2),
                         use_container_width=True)

    if sleeve_report is not None and not sleeve_report.rows.empty:
        eyebrow("Sleeves")
        _sr = sleeve_report.rows.copy()
        for _c in ("Budget", "Average weight", "Cash within sleeve"):
            _sr[_c] = _sr[_c].map(lambda v: f"{v*100:.1f}%")
        st.dataframe(_sr, use_container_width=True, hide_index=True)
        for _w in sleeve_report.warnings:
            st.markdown(f'<div class="flag">{_w}</div>', unsafe_allow_html=True)
        note("A sleeve holds at most its budget. Whatever its model leaves "
             "in cash stays inside that sleeve rather than being handed to "
             "another, so a defensive signal in one class cannot quietly "
             "become extra risk in another.")
        if sleeve_report.by_sleeve is not None and not sleeve_report.by_sleeve.empty:
            st.plotly_chart(
                C.weights_area(sleeve_report.by_sleeve,
                               (1.0 - sleeve_report.by_sleeve.sum(axis=1)).clip(lower=0),
                               "Weight held by sleeve"),
                use_container_width=True, config={"displaylogo": False})

    eyebrow("Composition over time")
    st.plotly_chart(C.weights_area(res.weights, res.cash_weight),
                    use_container_width=True, config={"displaylogo": False})

    c1, c2, c3 = st.columns(3)
    with c1:
        dial("Annual turnover", f"{stats.get('Annual Turnover', float('nan')):.2f}x",
             "sum of weight changes")
    with c2:
        dial("Average exposure",
             M.format_metric("Average Exposure", stats.get("Average Exposure", np.nan)),
             "share invested outside cash")
    with c3:
        drag = float(res.costs.sum())
        dial("Cumulative friction cost", f"{drag*100:,.2f}%",
             "compounded as a percentage of value", "neg")

    if res.dividend_income is not None and float(res.dividend_income.sum()) > 0:
        eyebrow("Dividends")
        di = res.dividend_income
        years = max(len(di) / ppy, 1e-9)
        d1, d2, d3 = st.columns(3)
        with d1:
            dial("Cumulative dividends", f"{float(di.sum())*100:,.2f}%",
                 "of portfolio value, summed", "pos")
        with d2:
            dial("Average per year", f"{float(di.sum())/years*100:,.2f}%",
                 "cash yield on the portfolio")
        with d3:
            dial("Payment days", f"{int((di > 0).sum()):,}",
                 "ex-dates with a credit")
        note("Dividends are credited as cash on their ex-date and stay "
             "uninvested until the next rebalance, which is what actually "
             "happens in an account. Prices are ex-dividend, so nothing is "
             "counted twice.")
        st.plotly_chart(
            C.rolling_metric((di * 100).cumsum(), "Cumulative dividends (%)",
                             title="Dividend income, cumulative"),
            use_container_width=True, config={"displaylogo": False})

    eyebrow("Average weight by instrument")
    avg = (res.weights.mean() * 100).sort_values(ascending=False)
    st.plotly_chart(C.bar_series(avg.index, avg.values,
                                 "Average weight over the period", "%"),
                    use_container_width=True, config={"displaylogo": False})

    eyebrow("Current position as of the last date")
    last = res.weights.iloc[-1]
    cur = pd.DataFrame({
        "Instrument": last.index,
        "Weight": [f"{v*100:.2f}%" for v in last.values],
        "Value ($)": [f"{v * res.equity.iloc[-1]:,.0f}" for v in last.values],
    })
    cur.loc[len(cur)] = ["Cash", f"{res.cash_weight.iloc[-1]*100:.2f}%",
                         f"{res.cash_weight.iloc[-1] * res.equity.iloc[-1]:,.0f}"]
    st.dataframe(cur, use_container_width=True, hide_index=True)
    st.download_button("Download current holdings (CSV)",
                       cur.to_csv(index=False).encode("utf-8"),
                       "current_holdings.csv", "text/csv", key="dlholdings")

    eyebrow("Trade log")
    if res.trades.empty:
        note("No trades were generated.")
    else:
        t = res.trades.copy()
        for c in ("Weight Before", "Weight After", "Change"):
            t[c] = t[c].map(lambda v: f"{v*100:+.2f}%")
        st.dataframe(t.tail(400), use_container_width=True, hide_index=True, height=380)
        st.download_button("Download full trade log (CSV)",
                           res.trades.to_csv(index=False).encode("utf-8"),
                           "trades.csv", "text/csv")

# --------------------------- ROBUSTNESS ----------------------------------
with tabs[2]:
    note("A single backtest is only one observation. These four tests probe "
         "whether the result holds up beyond the exact parameter set chosen.")

    is_external = run_mode == "external_weights"
    strategy_obj = None if is_external else REGISTRY[run["strategy_key"]]
    params_run = run["params"]
    fixed_w = run["weights"] if is_external else None
    kw = dict(cash_prices=run["cash"], exog=exog_used,
              weights=fixed_w, rebalance_dates=run_rebal)

    eyebrow("1. Stability over time")
    n_folds = st.slider("Number of folds", 3, 10, 5, key="wf")
    wf = R.walk_forward(universe, strategy_obj, params_run, rcfg.engine,
                        rcfg.costs, n_folds, **kw)
    if not wf.empty:
        disp = wf.copy()
        for c in ("CAGR", "Volatility", "Max Drawdown"):
            disp[c] = disp[c].map(lambda v: f"{v*100:.2f}%")
        disp["Sharpe"] = disp["Sharpe"].map(lambda v: f"{v:.2f}")
        st.dataframe(signed(disp, ["CAGR", "Sharpe", "Max Drawdown"]),
                     use_container_width=True, hide_index=True)
        st.plotly_chart(C.bar_series(wf["Fold"], wf["Sharpe"],
                                     "Sharpe by fold"),
                        use_container_width=True, config={"displaylogo": False})
        disp_sd = wf.attrs.get("sharpe_dispersion", np.nan)
        if disp_sd == disp_sd and disp_sd > 0.6:
            st.markdown('<div class="flag">High Sharpe dispersion across '
                        'folds: the result depends heavily on the market '
                        'regime.</div>', unsafe_allow_html=True)

    ios = R.in_out_sample(universe, strategy_obj, params_run, rcfg.engine,
                          rcfg.costs, 0.6, **kw)
    if not ios.empty:
        d = ios.copy()
        for c in ("CAGR", "Max Drawdown"):
            d[c] = d[c].map(lambda v: f"{v*100:.2f}%")
        d["Sharpe"] = d["Sharpe"].map(lambda v: f"{v:.2f}")
        st.dataframe(signed(d, ["CAGR", "Sharpe", "Max Drawdown"]),
                     use_container_width=True, hide_index=True)

    eyebrow("2. Parameter stability")
    numeric = [] if is_external else [p for p in strategy_obj.params
                                      if p.kind in ("int", "float")]
    if is_external:
        note("With no parameter to vary, this test does not apply to "
             "imported weights. The file is taken as-is: the question of "
             "overfitting plays out upstream, where the weights were "
             "produced.")
    if len(numeric) >= 1:
        cols = st.columns(2)
        px_ = cols[0].selectbox("First parameter", [p.key for p in numeric],
                                format_func=lambda k: next(p.label for p in numeric if p.key == k))
        others = [p.key for p in numeric if p.key != px_]
        py_ = cols[1].selectbox("Second parameter", ["\u2014 none \u2014"] + others,
                                format_func=lambda k: k if k == "\u2014 none \u2014"
                                else next(p.label for p in numeric if p.key == k))
        metric_choice = st.selectbox("Metric", ["Sharpe", "CAGR", "Calmar",
                                                "Max Drawdown", "Annual Turnover"])

        def _grid(pk: str) -> List[Any]:
            spec = next(p for p in numeric if p.key == pk)
            lo, hi = float(spec.min), float(spec.max)
            base = float(params_run.get(pk, spec.default))
            lo = max(lo, base * 0.35)
            hi = min(hi, base * 2.0 if base > 0 else hi)
            vals = np.linspace(lo, hi, 7)
            return sorted({int(round(v)) if spec.kind == "int" else round(float(v), 2)
                           for v in vals})

        if st.button("Compute parameter surface", key="sweepbtn"):
            grid = {px_: _grid(px_)}
            if py_ != "\u2014 none \u2014":
                grid[py_] = _grid(py_)
            with st.spinner("Sweeping..."):
                sw = R.parameter_sweep(universe, strategy_obj, params_run, grid,
                                       rcfg.engine, rcfg.costs, run["cash"],
                                       exog=exog_used)
            st.session_state["sweep"] = (sw, px_, py_, metric_choice)

        if "sweep" in st.session_state:
            sw, sx, sy, sz = st.session_state["sweep"]
            if sz in sw.columns:
                if sy != "\u2014 none \u2014" and sy in sw.columns:
                    st.plotly_chart(C.sweep_heatmap(sw, sx, sy, sz),
                                    use_container_width=True,
                                    config={"displaylogo": False})
                else:
                    st.plotly_chart(C.sweep_line(sw, sx, sz),
                                    use_container_width=True,
                                    config={"displaylogo": False})
                good = sw[sz].dropna()
                if len(good) > 2:
                    spread = float(good.max() - good.min())
                    st.markdown(
                        f'<div class="note">Gap between the best and worst '
                        f'combination: {spread:.2f}. A flat surface is a good '
                        f'sign; a lone spike on the chosen parameter set is '
                        f'not.</div>', unsafe_allow_html=True)
                    dsr = R.deflated_sharpe_note(stats.get("Sharpe", np.nan),
                                                 len(sw), len(res.returns))
                    if dsr["expected_max_sharpe"] == dsr["expected_max_sharpe"]:
                        st.markdown(
                            f'<div class="flag">Across {len(sw)} trials, a Sharpe '
                            f'of {dsr["expected_max_sharpe"]:.2f} would be expected '
                            f'by pure chance. Net edge of the model: '
                            f'{dsr["haircut"]:+.2f}.</div>', unsafe_allow_html=True)

    eyebrow("3. Cost sensitivity")
    cs = R.cost_sensitivity(universe, strategy_obj, params_run, rcfg.engine,
                            rcfg.costs, None, **kw)
    st.plotly_chart(C.sweep_line(cs, "Costs (bps round-trip)", "CAGR",
                                 "CAGR by level of frictions"),
                    use_container_width=True, config={"displaylogo": False})
    cd = cs.copy()
    for c in ("CAGR", "Max Drawdown"):
        cd[c] = cd[c].map(lambda v: f"{v*100:.2f}%")
    cd["Sharpe"] = cd["Sharpe"].map(lambda v: f"{v:.2f}")
    st.dataframe(signed(cd, ["CAGR", "Sharpe", "Max Drawdown"]),
                 use_container_width=True, hide_index=True)

    eyebrow("4. Sampling uncertainty")
    c1, c2 = st.columns(2)
    n_sims = c1.slider("Simulations", 100, 2000, 500, 100)
    blk = c2.slider("Block size (days)", 5, 63, 21, 1)
    if st.button("Run Monte Carlo simulation", key="mcbtn"):
        with st.spinner("Resampling..."):
            st.session_state["mc"] = R.monte_carlo(res.returns, n_sims, blk, ppy)
    if "mc" in st.session_state:
        mc = st.session_state["mc"]
        if not mc["paths"].empty:
            st.plotly_chart(C.monte_carlo_fan(mc["paths"], res.equity),
                            use_container_width=True, config={"displaylogo": False})
            a, b, c = st.columns(3)
            with a:
                dial("Simulated median CAGR", f"{mc['median_cagr']*100:.2f}%")
            with b:
                dial("Probability of loss", f"{mc['prob_loss']*100:.1f}%",
                     "final value below initial capital")
            with c:
                dial("Probability of a drawdown > 20%", f"{mc['prob_dd_20']*100:.1f}%")
            s = mc["stats"].copy()
            for cname in ("CAGR", "Max Drawdown"):
                s[cname] = s[cname].map(lambda v: f"{v*100:.2f}%")
            s["Sharpe"] = s["Sharpe"].map(lambda v: f"{v:.2f}")
            st.dataframe(signed(s, ["CAGR", "Max Drawdown", "Sharpe"]),
                         use_container_width=True, hide_index=True)

# --------------------------- DATA ----------------------------------------
with tabs[3]:
    a, b, c = st.columns(3)
    with a:
        dial("Sessions", f"{quality.rows:,}")
    with b:
        dial("Start", str(quality.start.date()) if quality.start is not None else "\u2014")
    with c:
        dial("End", str(quality.end.date()) if quality.end is not None else "\u2014")

    eyebrow("Per-instrument diagnostic")
    st.dataframe(quality.per_asset, use_container_width=True, hide_index=True)

    if quality.warnings:
        eyebrow("Flags")
        for w in quality.warnings[:20]:
            st.markdown(f'<div class="flag">{w}</div>', unsafe_allow_html=True)

    if ex_rep is not None:
        eyebrow("Exogenous series")
        a, b, c = st.columns(3)
        with a:
            dial("Imported series", f"{len(ex_rep.columns)}")
        with b:
            dial("Detected frequency", ex_rep.frequency.capitalize())
        with c:
            dial("Publication lag", f"{ex_rep.lag_days}d",
                 "applied before any alignment")
        st.dataframe(ex_rep.per_series, use_container_width=True, hide_index=True)
        for wmsg in ex_rep.warnings:
            st.markdown(f'<div class="flag">{wmsg}</div>', unsafe_allow_html=True)

        roles = split_roles(run["exog_raw"], list(universe.columns))
        if roles["factor"]:
            note("Columns recognized as a cross-sectional factor (matches a "
                 "universe symbol): " + ", ".join(map(str, roles["factor"])))
        if roles["macro"]:
            note("Columns treated as macro series (no matching symbol): "
                 + ", ".join(map(str, roles["macro"][:15])))

        if exog_used is not None and not exog_used.empty:
            show = st.multiselect("Plot one or more series",
                                  list(exog_used.columns),
                                  default=list(exog_used.columns)[:2])
            if show:
                sub = exog_used[show].dropna(how="all")
                fig = C.equity_curve(sub, False, "Exogenous series, after "
                                                 "publication lag")
                st.plotly_chart(fig, use_container_width=True,
                                config={"displaylogo": False})
            st.download_button("Download aligned series (CSV)",
                               exog_used.to_csv().encode("utf-8"),
                               "exogenous_series_aligned.csv", "text/csv")

    eyebrow("Correlation of daily returns")
    corr = universe.pct_change().corr()
    st.plotly_chart(C.correlation_matrix(corr), use_container_width=True,
                    config={"displaylogo": False})

    eyebrow("Normalized prices (base 100)")
    norm = universe.dropna(how="all")
    norm = norm.div(norm.bfill().iloc[0]) * 100
    st.plotly_chart(C.equity_curve(norm, True, "Adjusted prices, base 100"),
                    use_container_width=True, config={"displaylogo": False})

    st.download_button("Download prices used (CSV)",
                       universe.to_csv().encode("utf-8"), "prices.csv", "text/csv")

# --------------------------- BUILDER --------------------------------------
with tabs[4]:
    note("Write a strategy as an expression over the same indicators the "
         "packaged models use. Test it here against the loaded universe, "
         "then select <b>Custom Formula</b> as the model in the sidebar and "
         "paste the expressions in to run a full backtest.")

    ref = FORMULA.available_names(list(universe.columns),
                                  list(exog_used.columns) if exog_used is not None
                                  else None)
    with st.expander("Available indicators", expanded=False):
        rc = st.columns(3)
        for i, (group, items) in enumerate(ref.items()):
            with rc[i % 3]:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;'
                    f'font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;'
                    f'color:#C9A227;margin-top:.5rem;">{group}</div>',
                    unsafe_allow_html=True)
                for it in items:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;'
                        f'font-size:.74rem;color:#E3E8EF;">{it}</div>',
                        unsafe_allow_html=True)
        st.markdown(
            '<div class="note" style="margin-top:.8rem;">'
            '<b>x</b> is any series, usually <code>price</code>. '
            '<b>n</b> is a window in sessions. Expressions are applied to the '
            'whole universe at once and produce one value per instrument per '
            'day. Arithmetic, comparisons, <code>and</code>/<code>or</code> '
            'and <code>ifelse</code> are available; nothing else is, by '
            'design.</div>', unsafe_allow_html=True)

    EXAMPLES = {
        "Momentum rank": "pctrank(mom(price, 126))",
        "Trend filter": "price > sma(price, 200)",
        "Momentum, trend-filtered":
            "ifelse(price > sma(price, 200), pctrank(mom(price, 126)), 0)",
        "Momentum and low volatility":
            "0.7 * pctrank(mom(price, 126)) + 0.3 * pctrank(-vol(price, 60))",
        "Distance from the average": "price / sma(price, 100) - 1",
        "Trend quality":
            "pctrank(mom(price, 126)) * er(price, 20)",
        "Oversold in an uptrend":
            "ifelse(rsi(price, 14) < 35 and price > sma(price, 200), 1, 0)",
        "Breakout": "price > mmax(shift(price, 1), 60)",
    }
    pick = st.selectbox("Start from an example", ["\u2014 blank \u2014"] + list(EXAMPLES))
    if pick != st.session_state.get("_bld_pick"):
        st.session_state["_bld_pick"] = pick
        if not pick.startswith("\u2014"):
            st.session_state["bld_expr"] = EXAMPLES[pick]

    st.session_state.setdefault("bld_expr", "pctrank(mom(price, 126))")
    expr = st.text_area("Expression", key="bld_expr", height=90)

    if expr and expr.strip():
        try:
            d = FORMULA.describe(expr, universe, exog_used)
            frame = d["frame"]
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                dial("Type", "Boolean" if d["is_boolean"] else "Continuous",
                     "0 / 1 filter" if d["is_boolean"] else "score to rank")
            with b2:
                dial("Coverage", f"{d['coverage']*100:.0f}%",
                     "of the price grid")
            with b3:
                dial("First value",
                     str(d["first_valid"].date()) if d["first_valid"] is not None
                     else "\u2014", "warm-up needed")
            with b4:
                dial("Median", f"{d['median']:.3f}",
                     f"range {d['min']:.2f} to {d['max']:.2f}")

            eyebrow("Value over time")
            show = st.multiselect("Instruments", list(frame.columns),
                                  default=list(frame.columns)[:3], key="bld_show")
            if show:
                st.plotly_chart(
                    C.equity_curve(frame[show].dropna(how="all"), False,
                                   "Expression value"),
                    use_container_width=True, config={"displaylogo": False})

            eyebrow("Latest values")
            last = frame.dropna(how="all")
            if not last.empty:
                tail = last.iloc[-1].sort_values(ascending=False)
                st.dataframe(
                    pd.DataFrame({"Instrument": tail.index,
                                  "Value": [f"{v:,.4f}" for v in tail.values]}),
                    use_container_width=True, hide_index=True)

            if d["coverage"] < 0.5:
                st.markdown(
                    f'<div class="flag">The expression only produces a value '
                    f'on {d["coverage"]*100:.0f}% of the grid. A long window '
                    f'costs history: the backtest cannot start until it has '
                    f'one.</div>', unsafe_allow_html=True)

            eyebrow("Use it")
            note("Select <b>Custom Formula</b> as the model in the sidebar, "
                 "then paste this into <b>Score expression</b> (higher is "
                 "better) or into <b>Filter expression</b> if it is a "
                 "boolean test.")
            st.code(expr, language="text")
        except FORMULA.FormulaError as exc:
            st.markdown(f'<div class="flag">{exc}</div>', unsafe_allow_html=True)
    else:
        note("Enter an expression above, or pick an example.")


# --------------------------- EXPORT ---------------------------------------
with tabs[5]:
    note("Every export below reproduces the backtest currently on screen. "
         "The tearsheet is the fastest way to share a result; the workbook "
         "and CSVs are for further analysis elsewhere.")

    eyebrow("Tearsheet report")
    note("A single printable HTML page: key statistics, equity curve, "
         "drawdown, monthly returns, return distribution, full stats table, "
         "top drawdown episodes, and current holdings. Open it in a browser "
         "and use Print \u2192 Save as PDF for a shareable PDF.")
    tearsheet_html = REPORT.render_tearsheet(res, bench, stats, bstats, rcfg)
    st.download_button("Download tearsheet report (HTML)",
                       tearsheet_html.encode("utf-8"),
                       f"tearsheet_{run['strategy_key']}.html", "text/html",
                       key="dltearsheet")

    eyebrow("Holdings")
    hc1, hc2 = st.columns(2)
    hc1.download_button("Current holdings (CSV)", cur.to_csv(index=False).encode("utf-8"),
                        "current_holdings.csv", "text/csv", key="dlholdings2")
    hc2.download_button("Holdings history (CSV)",
                        res.weights.to_csv().encode("utf-8"),
                        "holdings_history.csv", "text/csv", key="dlw")

    eyebrow("Configuration")
    note("This configuration exactly reproduces the backtest shown. Keep it "
         "with the result; re-import it to replay the test. Imported files "
         "are not included: the YAML retains their name, settings, and "
         "publication lag, but the files themselves must be re-uploaded.")
    y = rcfg.to_yaml()
    st.code(y, language="yaml")

    eyebrow("Other downloads")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Configuration (YAML)", y.encode("utf-8"),
                       f"config_{run['strategy_key']}.yaml", "text/yaml")

    series = pd.DataFrame({
        "value": res.equity, "return": res.returns,
        "exposure": res.exposure, "cash": res.cash_weight,
        "turnover": res.turnover, "costs": res.costs,
    })
    if bench is not None:
        series["benchmark"] = bench.equity
    c2.download_button("Daily series (CSV)",
                       series.to_csv().encode("utf-8"), "series.csv", "text/csv")

    try:
        book = XL.workbook_from_backtest(
            res, bench, stats, bstats, rcfg,
            prices=universe, exog=exog_used,
            quality=quality.per_asset if quality is not None else None)
        c3.download_button("Full report (Excel)", book,
                           f"backtest_{run['strategy_key']}.xlsx",
                           "application/vnd.openxmlformats-officedocument."
                           "spreadsheetml.sheet", key="dlxl")
        note("The workbook carries every table behind this report: "
             "statistics, trailing periods, calendar years, drawdown "
             "episodes, monthly returns, the full daily series, current and "
             "historical holdings, target weights, the trade log, rebalance "
             "dates, parameters, prices, any imported series, and the data "
             "diagnostic \u2014 plus a Notes sheet recording the settings "
             "these figures depend on.")
    except Exception as exc:
        st.markdown(f'<div class="flag">Excel export unavailable: {exc}</div>',
                    unsafe_allow_html=True)

    st.markdown(
        f'<div class="note" style="margin-top:1.5rem;">Run on {run["stamp"]} \u00b7 '
        f'execution lag {rcfg.engine.execution_lag}d \u00b7 '
        f'{REBALANCE_RULES[rcfg.engine.rebalance].lower()} \u00b7 '
        f'{rcfg.costs.commission_bps + rcfg.costs.slippage_bps:.0f} bps of '
        f'frictions per weight round-trip.</div>', unsafe_allow_html=True)
