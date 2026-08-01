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
from qbt.data import load_yfinance, load_file, clean_prices, excel_sheet_names
from qbt.exog import load_exog, prepare_exog, exog_report, split_roles
from qbt.external import (load_target_weights, prepare_target_weights,
                          weights_template)
from qbt.engine import run_backtest, benchmark_result, align_results
from qbt.strategies import REGISTRY, get as get_strategy
from qbt import metrics as M
from qbt import charts as C
from qbt import robustness as R
from qbt import report as REPORT

st.set_page_config(page_title="Quant Backtest Studio",
                   page_icon="\u25e7", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@600;700&display=swap');

html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:#0E1116; }
section[data-testid="stSidebar"] { background:#12171F; border-right:1px solid #2A323D; }

.masthead { border-bottom:1px solid #2A323D; padding:.2rem 0 .9rem; margin-bottom:1.1rem; }
.masthead h1 { font-family:'IBM Plex Sans Condensed',sans-serif; font-weight:700;
  font-size:1.85rem; letter-spacing:-.015em; color:#E3E8EF; margin:0; }
.masthead .sub { font-family:'IBM Plex Mono',monospace; font-size:.72rem;
  letter-spacing:.16em; text-transform:uppercase; color:#7D8A9C; margin-top:.35rem; }

.eyebrow { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.18em;
  text-transform:uppercase; color:#7D8A9C; margin:1.5rem 0 .55rem; }

.dial { background:#161B22; border:1px solid #232B35; border-left:2px solid #C9A227;
  padding:.62rem .8rem; height:100%; }
.dial .k { font-family:'IBM Plex Mono',monospace; font-size:.63rem; letter-spacing:.11em;
  text-transform:uppercase; color:#7D8A9C; display:block; margin-bottom:.28rem; }
.dial .v { font-family:'IBM Plex Mono',monospace; font-size:1.22rem; font-weight:600;
  color:#E3E8EF; line-height:1.1; }
.dial .d { font-family:'IBM Plex Mono',monospace; font-size:.68rem; color:#7D8A9C; }
.dial.pos { border-left-color:#4C9A8F; } .dial.pos .v { color:#4C9A8F; }
.dial.neg { border-left-color:#B4553F; } .dial.neg .v { color:#B4553F; }

.note { border-left:2px solid #2A323D; padding:.15rem 0 .15rem .75rem;
  color:#7D8A9C; font-size:.83rem; }
.flag { border-left:2px solid #C9A227; padding:.3rem 0 .3rem .75rem;
  color:#C9A227; font-size:.82rem; font-family:'IBM Plex Mono',monospace; }

.stTabs [data-baseweb="tab-list"] { gap:1.6rem; border-bottom:1px solid #2A323D; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace; font-size:.72rem;
  letter-spacing:.13em; text-transform:uppercase; color:#7D8A9C; padding:.4rem 0; }
.stTabs [aria-selected="true"] { color:#C9A227 !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#C9A227; }

.stButton>button { font-family:'IBM Plex Mono',monospace; font-size:.75rem;
  letter-spacing:.1em; text-transform:uppercase; background:#C9A227; color:#0E1116;
  border:0; border-radius:2px; font-weight:600; width:100%; padding:.55rem; }
.stButton>button:hover { background:#DDB63A; color:#0E1116; }
.stDownloadButton>button { background:transparent; color:#C9A227; border:1px solid #C9A227; }

[data-testid="stDataFrame"] { font-family:'IBM Plex Mono',monospace; }
#MainMenu, footer { visibility:hidden; }
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


def dial(label: str, value: str, sub: str = "", tone: str = ""):
    cls = f"dial {tone}".strip()
    st.markdown(
        f'<div class="{cls}"><span class="k">{label}</span>'
        f'<span class="v">{value}</span>'
        f'{f"<div class=d>{sub}</div>" if sub else ""}</div>',
        unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


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
source = st.sidebar.radio("Source", ["Yahoo Finance", "File (CSV / Excel)"],
                          index=0 if d0.source == "yfinance" else 1,
                          horizontal=True, label_visibility="collapsed")

prices_raw: Optional[pd.DataFrame] = None
upload_error = None

if source == "Yahoo Finance":
    tickers_txt = st.sidebar.text_area(
        "Symbols (one per line or comma-separated)",
        value="\n".join(d0.tickers), height=110,
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

bench_choices = ["\u2014 none \u2014"] + univ_options
bench_default = d0.benchmark if d0.benchmark in univ_options else None
benchmark = st.sidebar.selectbox(
    "Comparison benchmark", bench_choices,
    index=bench_choices.index(bench_default) if bench_default else 0)
benchmark = None if benchmark == "\u2014 none \u2014" else benchmark

cash_choices = ["Fixed rate"] + univ_options
cash_proxy = st.sidebar.selectbox("Cash remuneration", cash_choices, index=0,
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
    elif p.kind == "choice" and p.choices:
        opts = list(p.choices)
        idx = opts.index(default) if default in opts else 0
        params[p.key] = st.sidebar.selectbox(p.label, opts, index=idx,
                                             help=p.help or None, key=key)
    else:
        params[p.key] = st.sidebar.text_input(p.label, str(default or ""),
                                              help=p.help or None, key=key)

# --- Execution ---
st.sidebar.markdown('<div class="eyebrow">Execution</div>', unsafe_allow_html=True)
rb_keys = list(REBALANCE_RULES.keys())
rebalance = st.sidebar.selectbox(
    "Rebalance", rb_keys, index=rb_keys.index(e0.rebalance),
    format_func=lambda k: REBALANCE_RULES[k])
lag = st.sidebar.slider("Execution lag (days)", 0, 5, int(e0.execution_lag),
                        help="1 = signal at the close, executed the next session. "
                             "0 assumes execution at the price that produced the signal.")
capital = st.sidebar.number_input("Initial capital ($)", 1_000, 1_000_000_000,
                                  int(e0.initial_capital), 10_000)
max_lev = st.sidebar.slider("Max leverage", 0.5, 2.0, float(e0.max_leverage), 0.1)

st.sidebar.markdown('<div class="eyebrow">Frictions</div>', unsafe_allow_html=True)
comm = st.sidebar.number_input("Commission (bps)", 0.0, 200.0, float(c0.commission_bps), 1.0)
slip = st.sidebar.number_input("Slippage (bps)", 0.0, 500.0, float(c0.slippage_bps), 5.0)
cash_rate = st.sidebar.number_input("Cash rate (annual %)", 0.0, 15.0,
                                    float(c0.cash_rate_pa * 100), 0.25) / 100.0

run_clicked = st.sidebar.button("Run backtest")

# ----------------------------------------------------------------------
# Configuration assembly
# ----------------------------------------------------------------------
run_label = (strategy.label if mode == "builtin"
             else (f"Imported weights \u2014 {w_source}" if w_source else "Imported weights"))

cfg = RunConfig(
    label=run_label,
    data=DataConfig(
        source="yfinance" if source == "Yahoo Finance" else "upload",
        tickers=sel_universe,
        start=str(start) if start else "1990-01-01",
        end=str(end) if end else None,
        benchmark=benchmark, cash_proxy=cash_proxy,
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
                        execution_lag=int(lag), max_leverage=float(max_lev)),
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
def build_prices() -> pd.DataFrame:
    if source == "Yahoo Finance":
        needed = list(dict.fromkeys(
            sel_universe + [t for t in (benchmark, cash_proxy) if t]))
        return fetch_yf(tuple(needed), str(start), str(end), "Close")
    if prices_raw is None:
        raise RuntimeError("No file loaded.")
    return prices_raw


if run_clicked and not blocking:
    try:
        with st.spinner("Loading prices..."):
            raw = build_prices()
            prices, quality = clean_prices(raw, cfg.data)

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

        w_report, rebal_dates = None, None
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
                weights = strategy.generate(universe, params, exog_aligned)

            result = run_backtest(universe, weights, cfg.engine, cfg.costs,
                                  cash_px, run_label, rebal_dates)
            bench = None
            if benchmark and benchmark in prices.columns:
                bench = benchmark_result(prices[benchmark], cfg.engine, benchmark)

        st.session_state["run"] = {
            "result": result, "bench": bench, "prices": universe,
            "cash": cash_px, "quality": quality, "cfg": cfg,
            "params": dict(params), "strategy_key": strat_key, "mode": mode,
            "exog": exog_aligned, "exog_raw": exog_raw, "exog_report": ex_rep,
            "weights_report": w_report, "rebalance_dates": rebal_dates,
            "weights": weights,
            "stamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        for k in ("sweep", "mc"):
            st.session_state.pop(k, None)
    except Exception as exc:
        st.error(f"The backtest stopped: {exc}")

run = st.session_state.get("run")

if run is None:
    st.markdown(
        '<div class="note">Choose a data source and a strategy in the panel '
        'on the left, then run the backtest. Results, data diagnostics, and '
        'robustness tests will appear here.<br><br>'
        'Two optional inputs: <b>exogenous series</b> (economic data, '
        'fundamental ratios, signals computed elsewhere) that feed certain '
        'models, and a <b>target weights</b> file that entirely replaces '
        'the signal generator.</div>',
        unsafe_allow_html=True)
    eyebrow("Available strategies")
    for k in sorted(REGISTRY, key=lambda x: REGISTRY[x].label):
        s = REGISTRY[k]
        st.markdown(
            f'<div style="margin-bottom:.7rem;"><span style="font-family:IBM Plex Mono,'
            f'monospace;color:#C9A227;font-size:.8rem;">{s.label}</span>'
            f'<div class="note" style="margin-top:.2rem;">{s.description}</div></div>',
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
run_rebal = run.get("rebalance_dates")

bench_r = bench.returns if bench is not None else None
stats = M.summary(res.returns, res.equity, bench_r, res.turnover,
                  res.exposure, rcfg.costs.cash_rate_pa, ppy)
bstats = M.summary(bench.returns, bench.equity, None, None, None,
                   rcfg.costs.cash_rate_pa, ppy) if bench is not None else {}

tabs = st.tabs(["Results", "Positions", "Robustness", "Data", "Export"])

# --------------------------- RESULTS -----------------------------------
with tabs[0]:
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
        st.plotly_chart(C.monthly_heatmap(res.returns), use_container_width=True,
                        config={"displaylogo": False})
    with right:
        st.plotly_chart(C.return_distribution(res.returns), use_container_width=True,
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

    eyebrow("Main drawdown episodes")
    dd_tbl = M.drawdown_table(res.equity, 6)
    if not dd_tbl.empty:
        dd_tbl["Drawdown"] = dd_tbl["Drawdown"].map(lambda v: f"{v*100:.2f}%")
        # "Recovery" mixes date objects with the string "ongoing"; Arrow
        # (Streamlit's serialization layer) rejects a mixed-type column.
        dd_tbl["Recovery"] = dd_tbl["Recovery"].astype(str)
    st.dataframe(dd_tbl, use_container_width=True, hide_index=True)

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
        st.dataframe(disp, use_container_width=True, hide_index=True)
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
        st.dataframe(d, use_container_width=True, hide_index=True)

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
    st.dataframe(cd, use_container_width=True, hide_index=True)

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
            st.dataframe(s, use_container_width=True, hide_index=True)

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

# --------------------------- EXPORT ---------------------------------------
with tabs[4]:
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

    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame({"Metric": list(stats.keys()),
                          "Value": list(stats.values())}).to_excel(
                xw, sheet_name="Statistics", index=False)
            series.to_excel(xw, sheet_name="Series")
            res.weights.to_excel(xw, sheet_name="Holdings")
            cur.to_excel(xw, sheet_name="Current Holdings", index=False)
            mt = M.monthly_returns(res.returns)
            if not mt.empty:
                mt.to_excel(xw, sheet_name="Monthly Returns")
            if not res.trades.empty:
                res.trades.to_excel(xw, sheet_name="Trades", index=False)
        c3.download_button("Full report (Excel)", buf.getvalue(),
                           "backtest.xlsx",
                           "application/vnd.openxmlformats-officedocument."
                           "spreadsheetml.sheet")
    except Exception:
        c3.markdown('<div class="note">Excel export unavailable '
                    '(openpyxl missing).</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="note" style="margin-top:1.5rem;">Run on {run["stamp"]} \u00b7 '
        f'execution lag {rcfg.engine.execution_lag}d \u00b7 '
        f'{REBALANCE_RULES[rcfg.engine.rebalance].lower()} \u00b7 '
        f'{rcfg.costs.commission_bps + rcfg.costs.slippage_bps:.0f} bps of '
        f'frictions per weight round-trip.</div>', unsafe_allow_html=True)
