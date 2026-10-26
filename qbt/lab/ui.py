"""Streamlit front end of the Research workspace.

A thin layer: every number comes from `qbt.lab.tools` / the pure modules, so
what the page shows is what the MCP server and the command line return.
Imported only when the workspace is selected.
"""
from __future__ import annotations

import io
import os
import zipfile
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import factors as F
from . import ledger as L
from . import stats as S
from . import tools as T
from .. import charts, presets as PRESETS
from ..brand import BRAND
from ..strategies import REGISTRY

STATUS_COLOR = {"PASS": BRAND["gain"], "WARN": "#D9A441", "FAIL": BRAND["loss"], "NA": BRAND["muted"]}
GRADE_COLOR = {"Robust": BRAND["gain"], "Promising": "#8DBA6B", "Fragile": "#D9A441",
               "Unsupported": BRAND["loss"], "Unrated": BRAND["muted"]}


# ----------------------------------------------------------------------
def _chip(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#111;border-radius:3px;padding:1px 8px;'
            f'font-size:.72rem;font-weight:700">{text}</span>')


def _section(title: str) -> None:
    st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)


def _fig(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1%}"


def _num(v, d=2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def _kpis(items: List[tuple]) -> None:
    cols = st.columns(len(items))
    for c, (k, v) in zip(cols, items):
        c.metric(k, v)


def _state() -> Dict[str, Any]:
    return st.session_state.setdefault("lab", {"did": None, "run_id": None, "screen": None, "xray": None,
                                               "events": None, "scan": None, "chat": []})


def _dataset():
    did = _state()["did"]
    return T.DATASETS.get(did) if did else None


# ----------------------------------------------------------------------
def render() -> None:
    S_ = _state()
    _sidebar(S_)
    st.markdown('<div class="masthead"><div><h1>Research Lab</h1>'
                '<div class="sub">Hypothesis &rarr; factor evidence &rarr; backtest &rarr; validation &rarr; '
                'run card. Research only: nothing here trades.</div></div></div>', unsafe_allow_html=True)
    ds = _dataset()
    tabs = st.tabs(["Ledger", "Factor Lab", "Strategy Lab", "Risk X-Ray", "Event Study", "Monitor", "Assistant"])
    with tabs[0]:
        _tab_ledger(S_, ds)
    if ds is None:
        for t in tabs[1:6]:
            with t:
                st.info("Load a universe from the sidebar first.")
    else:
        with tabs[1]:
            _tab_factors(S_, ds)
        with tabs[2]:
            _tab_strategy(S_, ds)
        with tabs[3]:
            _tab_xray(S_, ds)
        with tabs[4]:
            _tab_events(S_, ds)
        with tabs[5]:
            _tab_monitor(S_)
    with tabs[6]:
        _tab_assistant(S_)


# ----------------------------------------------------------------------
def _sidebar(S_: Dict[str, Any]) -> None:
    sb = st.sidebar
    sb.markdown('<div class="eyebrow">Universe</div>', unsafe_allow_html=True)
    names = ["Custom"] + PRESETS.names()
    pick = sb.selectbox("Preset", names, key="lab_preset")
    if pick != st.session_state.get("_lab_applied"):
        st.session_state["_lab_applied"] = pick
        if pick != "Custom":
            st.session_state["lab_tickers"] = "\n".join(PRESETS.get(pick)["tickers"])
            st.session_state["lab_bench"] = PRESETS.get(pick).get("benchmark", "")
    st.session_state.setdefault("lab_tickers", "SPY\nEFA\nEEM\nTLT\nIEF\nGLD\nVNQ\nDBC")
    st.session_state.setdefault("lab_bench", "SPY")
    txt = sb.text_area("Symbols", key="lab_tickers", height=140)
    tickers = [t.strip().upper() for t in txt.replace(",", "\n").split("\n") if t.strip()]
    c1, c2 = sb.columns(2)
    start = c1.date_input("From", value=date(date.today().year - 12, 1, 1), min_value=date(1970, 1, 1),
                          max_value=date.today(), key="lab_start")
    end = c2.date_input("To", value=date.today(), min_value=date(1971, 1, 1), max_value=date.today(), key="lab_end")
    bench = sb.text_input("Benchmark (also the event-study market)", key="lab_bench")
    synth = sb.checkbox("Use synthetic demo data (offline)", value=False, key="lab_synth",
                        help="Reproducible fake prices with a weak planted momentum effect. For trying the tools.")
    if sb.button("Load universe", key="lab_load", type="primary"):
        with st.spinner("Loading prices..."):
            out = (T.load_data(synthetic=True, edge=0.05) if synth else
                   T.load_data(tickers=tickers, start=str(start), end=str(end), benchmark=bench or None))
        if "error" in out:
            sb.error(out["error"])
        else:
            S_.update(did=out["dataset_id"], run_id=None, screen=None, xray=None, events=None)
            for w in out.get("warnings", [])[:3]:
                sb.warning(w)

    sb.markdown('<div class="eyebrow">Active hypothesis</div>', unsafe_allow_html=True)
    hyps = L.list_hypotheses()
    opts = {0: "None (runs are not logged)"} | {h["id"]: f"#{h['id']} {h['title']}" for h in hyps}
    S_["hid"] = sb.selectbox("Log experiments under", list(opts), format_func=lambda k: opts[k],
                             key="lab_hid") or None
    if S_["hid"]:
        sb.caption(f"{L.trial_count(S_['hid'])} trial(s) counted so far. The deflated Sharpe uses this number.")


# ----------------------------------------------------------------------
def _tab_ledger(S_: Dict[str, Any], ds) -> None:
    if ds is not None:
        _section("Dataset")
        q = ds.prices.notna().sum()
        _kpis([("Instruments", f"{ds.prices.shape[1]}"), ("From", ds.start), ("To", ds.end),
               ("Sessions", f"{len(ds.prices):,}"), ("Source", ds.source)])
        shortest = q.idxmin()
        st.caption(f"Shortest history: {shortest} ({q.min():,} sessions). A backtest can only start once its "
                   f"signal has history; short members cap what every strategy here can be tested on. "
                   f"Data hash `{ds.content_hash()[:12]}`.")
    _section("Hypotheses")
    st.caption("Write the idea down before testing it. Every experiment run under a hypothesis is counted, and "
               "that count is what the deflated Sharpe ratio charges you for.")
    with st.expander("New hypothesis", expanded=not L.list_hypotheses()):
        with st.form("lab_new_h", clear_on_submit=True):
            title = st.text_input("Title", placeholder="Trend filter reduces drawdown on a global ETF set")
            stmt = st.text_area("Statement", placeholder="What exactly do you expect to see, and by how much?", height=70)
            why = st.text_area("Why it should work", placeholder="Economic reason, not a pattern in the data", height=70)
            if st.form_submit_button("Add hypothesis"):
                try:
                    hid = L.add_hypothesis(title, stmt, why, ", ".join(ds.tickers) if ds else "")
                    st.success(f"Hypothesis #{hid} added.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    hyps = L.list_hypotheses()
    if hyps:
        df = pd.DataFrame(hyps)[["id", "title", "status", "n_experiments", "n_trials", "updated"]]
        st.dataframe(df, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns([2, 2, 1])
        hid = c1.selectbox("Hypothesis", [h["id"] for h in hyps], format_func=lambda i: f"#{i}", key="lab_edit_h")
        new = c2.selectbox("Status", L.STATUSES, key="lab_edit_s")
        if c3.button("Set status", key="lab_set_s"):
            L.update_hypothesis(hid, status=new)
            st.rerun()
        ex = L.list_experiments(hid)
        if ex:
            st.markdown(f"**Experiments under #{hid}** &mdash; {L.trial_count(hid)} trials counted")
            rows = [{"id": e["id"], "when": e["created"], "kind": e["kind"], "name": e["name"],
                     "variants": e["n_variants"], "Sharpe": e["metrics"].get("Sharpe"),
                     "verdict": e["verdict"]} for e in ex]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Export ledger (JSON)", L.export_json(), "qbt_lab_ledger.json", "application/json",
                       help="The ledger lives on this machine. On an ephemeral host, export it before it is lost.")


# ----------------------------------------------------------------------
def _tab_factors(S_: Dict[str, Any], ds) -> None:
    _section("Screen the factor library")
    st.caption(f"{len(F.FACTORS)} price-based factors. Each is scored on how well its cross-sectional rank predicts "
               f"the forward return rank (information coefficient). With this many factors, some look good by "
               f"chance, so the table adds Benjamini-Hochberg and Bonferroni adjusted p-values: trust those.")
    c1, c2, c3 = st.columns([1, 2, 1])
    horizon = c1.select_slider("Forward horizon (days)", [5, 10, 21, 42, 63], value=21, key="lab_h")
    cats = sorted({f.category for f in F.FACTORS.values()})
    chosen = c2.multiselect("Categories", cats, default=cats, key="lab_cats")
    run = c3.button("Run screen", key="lab_screen", type="primary")
    if ds.prices.shape[1] < 6:
        st.warning("Fewer than 6 instruments: cross-sectional factors have almost nothing to rank, and ICs from "
                   "so few names are very noisy.")
    if run:
        keys = [k for k, f in F.FACTORS.items() if f.category in chosen]
        with st.spinner(f"Scoring {len(keys)} factors..."):
            out = T.factor_screen(dataset_id=S_["did"], horizon=int(horizon), factors=keys, top=len(keys),
                                  hypothesis_id=S_.get("hid"))
        S_["screen"] = out
    out = S_.get("screen")
    if out and "error" not in out:
        rows = pd.DataFrame(out["top"])
        k1, k2 = st.columns(2)
        k1.metric("Factors screened", out["n_factors"])
        k2.metric("Survive FDR at 10%", out["survivors_after_fdr_10pct"],
                  help="Significant after correcting for having looked at every factor in the screen.")
        if "key" in rows:
            rows = rows.rename(columns={"key": "Key"})
        fmt = {c: "{:+.3f}" for c in ("Mean IC", "ICIR")} | {"t-stat": "{:+.2f}", "p": "{:.3f}",
                                                           "p_bonferroni": "{:.3f}", "p_bh": "{:.3f}",
                                                           "Spread Sharpe": "{:+.2f}"}
        st.dataframe(rows.style.format(fmt, na_rep="n/a").background_gradient(subset=["t-stat"], cmap="RdYlGn", vmin=-4, vmax=4),
                     hide_index=True, use_container_width=True, height=420)
        _section("Inspect one factor")
        key_map = {r["Factor"]: r.get("Key") for r in rows.to_dict("records")}
        pick = st.selectbox("Factor", list(key_map), key="lab_f_pick")
        fkey = key_map.get(pick)
        if fkey:
            _factor_detail(ds, fkey, int(out["horizon"]))
    elif out:
        st.error(out["error"])


def _factor_detail(ds, fkey: str, horizon: int) -> None:
    f = F.FACTORS[fkey]
    sc = F.compute(fkey, ds.prices)
    ic = F.information_coefficient(sc, ds.prices, horizon)
    st.caption(f"{f.note} Literature sign: {f.expected} (flipped so higher is better)." if f.note else
               f"Literature sign: {f.expected}.")
    _kpis([("Mean IC", _num(ic["mean_ic"], 3)), ("ICIR", _num(ic["icir"])), ("t-stat (NW)", _num(ic["t_stat"])),
           ("IC > 0", _pct(ic["hit_rate"])), ("Samples", f"{ic['n']}")])
    c1, c2 = st.columns(2)
    if "ic" in ic and len(ic["ic"]):
        s = ic["ic"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=s.index, y=s.values, marker_color=BRAND["faint"], name="IC", opacity=.55))
        fig.add_trace(go.Scatter(x=s.index, y=s.rolling(126, min_periods=40).mean(), name="126d mean",
                                 line=dict(color=BRAND["red"], width=2)))
        charts._base(fig, 300, "Daily rank IC")
        c1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    dec = F.ic_decay(sc, ds.prices)
    c2.plotly_chart(charts.bar_series([str(h) for h in dec["Horizon (days)"]], dec["Mean IC"].fillna(0).tolist(),
                                      "Mean IC by horizon (days): how fast the signal goes stale"),
                    use_container_width=True, config={"displayModeBar": False})
    qr = F.quantile_returns(sc, ds.prices, 3 if ds.prices.shape[1] < 15 else 5, horizon)
    if qr.get("table") is not None and len(qr["table"]):
        c3, c4 = st.columns(2)
        mq = qr["mean_by_q"]
        c3.plotly_chart(charts.bar_series([f"Q{int(i)}" for i in mq.index], (mq * 100).tolist(),
                                          f"Mean {horizon}-day forward return by score bucket (%)", "%"),
                        use_container_width=True, config={"displayModeBar": False})
        sp = (1 + qr["spread"]).cumprod() * 100
        c4.plotly_chart(charts.equity_curve(pd.DataFrame({"Top minus bottom": sp}), log=False,
                                            title=f"Long-short spread (Sharpe {_num(qr['spread_sharpe'])}, "
                                                  f"monotonicity {_num(qr['monotonicity'])})"),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Top-bucket turnover per rebalance: {_pct(qr['top_turnover'])}. Buckets are sampled every "
                   f"{horizon} sessions so forward windows do not overlap.")
    if st.button("Backtest this factor in the Strategy Lab", key=f"lab_to_strat_{fkey}"):
        st.session_state["lab_strat"] = "zoo_factor"
        st.session_state["lab_p_factor"] = fkey
        st.info("Open the Strategy Lab tab: the factor is preselected.")


# ----------------------------------------------------------------------
def _param_form(strategy_key: str, prefix: str) -> Dict[str, Any]:
    s = REGISTRY[strategy_key]
    params: Dict[str, Any] = {}
    cols = st.columns(3)
    visible = [p for p in s.params if p.kind in ("int", "float", "choice", "bool")]
    for i, p in enumerate(visible):
        col = cols[i % 3]
        k = f"{prefix}_{strategy_key}_{p.key}"
        if p.kind == "int":
            params[p.key] = int(col.number_input(p.label, int(p.min), int(max(p.max, p.default)), int(p.default),
                                                 int(p.step), key=k, help=p.help or None))
        elif p.kind == "float":
            params[p.key] = float(col.number_input(p.label, float(p.min), float(max(p.max, p.default)),
                                                   float(p.default), float(p.step), key=k, help=p.help or None))
        elif p.kind == "choice":
            idx = p.choices.index(p.default) if p.default in p.choices else 0
            params[p.key] = col.selectbox(p.label, p.choices, index=idx, key=k, help=p.help or None)
        else:
            params[p.key] = col.checkbox(p.label, value=bool(p.default), key=k, help=p.help or None)
    skipped = [p.label for p in s.params if p.kind not in ("int", "float", "choice", "bool")]
    if skipped:
        st.caption(f"Not editable here (use the Backtest workspace): {', '.join(skipped)}. Defaults apply.")
    return params


def _tab_strategy(S_: Dict[str, Any], ds) -> None:
    usable = {k: s for k, s in REGISTRY.items() if not s.needs_exog and k not in ("fixed_weights", "custom_formula")}
    st.session_state.setdefault("lab_strat", "zoo_factor" if "zoo_factor" in usable else next(iter(usable)))
    _section("Strategy")
    c1, c2 = st.columns([2, 3])
    key = c1.selectbox("Strategy", list(usable), format_func=lambda k: usable[k].label, key="lab_strat")
    c2.caption(usable[key].description)
    params = _param_form(key, "lab_p")
    with st.expander("Execution and costs"):
        e1, e2, e3, e4 = st.columns(4)
        reb = e1.selectbox("Rebalance", ["M", "W", "Q", "S", "A", "D"], key="lab_reb")
        lag = e2.number_input("Execution lag (days)", 0, 5, 1, key="lab_lag",
                              help="0 trades at the price that produced the signal: look-ahead bias.")
        com = e3.number_input("Commission (bps)", 0.0, 100.0, 5.0, 0.5, key="lab_com")
        slp = e4.number_input("Slippage (bps)", 0.0, 200.0, 10.0, 1.0, key="lab_slp")
    if st.button("Run backtest", key="lab_run", type="primary"):
        with st.spinner("Simulating..."):
            out = T.run_backtest(dataset_id=S_["did"], strategy=key, params=params, rebalance=reb,
                                 execution_lag=int(lag), commission_bps=com, slippage_bps=slp,
                                 hypothesis_id=S_.get("hid"))
        if "error" in out:
            st.error(out["error"])
        else:
            S_["run_id"] = out["run_id"]
    rid = S_.get("run_id")
    if not rid or rid not in T.RUNS:
        return
    r = T.RUNS[rid]
    res, bench, met = r["res"], r["bench"], r["metrics"]
    _section(f"Result: {r['label']}")
    _kpis([("CAGR", _pct(met.get("CAGR"))), ("Volatility", _pct(met.get("Volatility"))),
           ("Sharpe", _num(met.get("Sharpe"))), ("Max drawdown", _pct(met.get("Max Drawdown"))),
           ("Turnover / yr", _pct(met.get("Annual Turnover")))])
    curves = pd.DataFrame({r["label"]: res.equity / res.equity.iloc[0] * 100})
    if bench is not None:
        b = bench.equity.reindex(curves.index).ffill()
        curves["Benchmark"] = b / b.dropna().iloc[0] * 100
    _fig(charts.equity_curve(curves))
    _fig(charts.underwater({k: (res.equity if k == r["label"] else bench.equity.reindex(res.equity.index).ffill())
                            for k in curves.columns}))

    _section("Validate")
    st.caption("Runs the statistical battery: is this evidence or luck? Add a parameter grid to also measure "
               "overfitting across the configurations you might have chosen between.")
    grid = _grid_form(key, params)
    n_trials = L.trial_count(S_.get("hid")) if S_.get("hid") else 1
    st.caption(f"Trials counted for the deflated Sharpe: **{n_trials}**"
               + (" (from the ledger)" if S_.get("hid") else " (no hypothesis selected, so nothing is deflated)"))
    if st.button("Run validation", key="lab_validate"):
        with st.spinner("Bootstrapping, rotating positions, sweeping the grid..."):
            out = T.validate_run_tool(run_id=rid, grid=grid or None, n_trials=n_trials)
        if "error" in out:
            st.error(out["error"])
        else:
            r["validation_out"] = out
    out = r.get("validation_out")
    if out:
        _validation_view(r, out)

    _section("Keep")
    k1, k2, k3 = st.columns(3)
    if k1.button("Save run card + dossier", key="lab_save"):
        with st.spinner("Writing run card..."):
            r["card"] = T.save_run_card(run_id=rid)
    card = r.get("card")
    if card and "error" not in card:
        k1.success(f"Saved to {card['run_dir']}")
        if os.path.isfile(card["dossier"]):
            with open(card["dossier"], "rb") as fh:
                k2.download_button("Download dossier (HTML)", fh.read(), "dossier.html", "text/html", key="lab_dl_d")
        k3.download_button("Download run card (ZIP)", _zip_dir(card["run_dir"]), "run_card.zip", "application/zip", key="lab_dl_z")
        for w in card["warnings"]:
            st.warning(w)
    elif card:
        st.error(card["error"])
    with st.expander("Register for decay monitoring"):
        nm = st.text_input("Name", value=r["label"], key="lab_reg_name")
        if st.button("Freeze strategy as of today", key="lab_reg"):
            o = T.register_strategy(run_id=rid, name=nm)
            st.error(o["error"]) if "error" in o else st.success(f"Registered. Out-of-sample starts after {o['registered']}.")


def _zip_dir(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(path):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, path))
    return buf.getvalue()


def _grid_form(key: str, params: Dict[str, Any]) -> Dict[str, List[Any]]:
    s = REGISTRY[key]
    nums = [p for p in s.params if p.kind in ("int", "float", "choice")]
    with st.expander("Parameter grid for overfitting checks (optional)"):
        pick = st.multiselect("Parameters to vary (up to 2)", [p.key for p in nums], max_selections=2, key="lab_grid_pick")
        grid: Dict[str, List[Any]] = {}
        for pk in pick:
            p = next(x for x in nums if x.key == pk)
            if p.kind == "choice":
                vals = st.multiselect(f"{p.label}: values", p.choices, default=p.choices[:4], key=f"lab_g_{pk}")
            else:
                cur = params[pk]
                step = max(p.step, abs(cur) * 0.25 if p.kind == "float" else int(max(p.step, abs(cur) * 0.25)))
                raw = st.text_input(f"{p.label}: values (comma separated)",
                                    ", ".join(str(round(cur + i * step, 4) if p.kind == "float" else int(cur + i * step))
                                              for i in (-2, -1, 0, 1, 2) if cur + i * step >= p.min),
                                    key=f"lab_g_{pk}")
                try:
                    vals = [float(v) if p.kind == "float" else int(float(v)) for v in raw.split(",") if v.strip()]
                except ValueError:
                    vals = []
            if vals:
                grid[pk] = vals
        n = int(np.prod([len(v) for v in grid.values()])) if grid else 0
        if n:
            st.caption(f"{n} configurations will be simulated and counted as trials.")
        return grid if n >= 3 else {}


def _validation_view(r: Dict[str, Any], out: Dict[str, Any]) -> None:
    g = out["grade"]
    st.markdown(f"### Grade: {_chip(g['label'], GRADE_COLOR.get(g['label'], BRAND['muted']))}",
                unsafe_allow_html=True)
    st.caption(f"{g['passes']} passed, {g['warnings']} warnings, {g['fails']} failed, "
               f"{out['n_trials_counted']} trial(s) counted. The grade counts independent lines of evidence; "
               f"it is not a probability of making money.")
    rows = ""
    for c in out["checks"]:
        stt = c["status"].upper()
        rows += (f"<tr><td style='padding:5px 10px;white-space:nowrap'><b>{c['check']}</b></td>"
                 f"<td style='padding:5px 10px'>{_chip(stt, STATUS_COLOR.get(stt, BRAND['muted']))}</td>"
                 f"<td style='padding:5px 10px;color:{BRAND['muted']}'>{c['evidence']}"
                 f"<div style='font-size:.72rem;color:{BRAND['faint']}'>{c['threshold']}</div></td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{rows}</table>", unsafe_allow_html=True)

    ppy = r["engine"].periods_per_year
    rets = r["res"].returns
    c1, c2 = st.columns(2)
    bs = S.bootstrap_sharpe(rets, 500, 0.95, None, ppy)
    if "samples" in bs:
        fig = go.Figure(go.Histogram(x=bs["samples"], nbinsx=40, marker_color=BRAND["faint"]))
        fig.add_vline(x=0, line_color=BRAND["loss"], line_dash="dot")
        fig.add_vline(x=bs["observed"], line_color=BRAND["red"])
        charts._base(fig, 260, "Bootstrap distribution of the Sharpe")
        c1.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    tt = S.random_timing_test(r["res"].weights, r["ds"].prices, 500, 21, ppy)
    if "samples" in tt:
        fig = go.Figure(go.Histogram(x=tt["samples"], nbinsx=40, marker_color=BRAND["faint"]))
        fig.add_vline(x=tt["observed_gross"], line_color=BRAND["red"])
        charts._base(fig, 260, "Same positions at random times (gross Sharpe) vs actual")
        c2.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    wf = S.walk_forward_windows(rets, 5, ppy)
    if len(wf):
        _fig(charts.bar_series([f"{a}" for a in wf["Start"]], wf["Sharpe"].fillna(0).tolist(),
                               "Sharpe by period (equal windows)"))


# ----------------------------------------------------------------------
def _tab_xray(S_: Dict[str, Any], ds) -> None:
    from . import xray as X
    _section("What is this portfolio made of?")
    src_opts = ["Equal weight, whole universe", "Custom weights"]
    rid = S_.get("run_id")
    if rid in T.RUNS:
        src_opts = ["Latest run: current weights", "Latest run: average weights"] + src_opts
    src = st.radio("Weights from", src_opts, horizontal=True, key="lab_x_src")
    window = st.number_input("Trailing window (sessions, 0 = all)", 0, 5000, 756, 21, key="lab_x_win")
    weights: Optional[Dict[str, float]] = None
    px = ds.prices
    if src.startswith("Latest run") and rid in T.RUNS:
        w = T.RUNS[rid]["res"].weights
        weights = X.current_weights(w) if "current" in src else X.average_weights(w)
        px = T.RUNS[rid]["ds"].prices
    elif src.startswith("Equal"):
        weights = {t: 1.0 / ds.prices.shape[1] for t in ds.prices.columns}
    else:
        txt = st.text_area("Weights (TICKER weight, one per line)", "\n".join(f"{t} {1 / min(4, len(ds.tickers)):.3f}"
                                                                           for t in ds.tickers[:4]), key="lab_x_txt")
        weights = {}
        for line in txt.splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    weights[parts[0].upper()] = float(parts[1])
                except ValueError:
                    pass
    if not weights:
        st.info("No weights to analyse.")
        return
    try:
        x = X.risk_xray(px, weights, window=int(window) or None)
    except ValueError as exc:
        st.error(str(exc))
        return
    c, rk, d, tl = x["concentration"], x["risk"], x["diversification"], x["tail"]
    _kpis([("Volatility", _pct(rk["vol"])), ("Max drawdown", _pct(rk["max_drawdown"])),
           ("Effective N (weights)", _num(c["effective_n"], 1)), ("Effective bets (risk)", _num(d["effective_bets"], 1)),
           ("Diversification ratio", _num(d["ratio"]))])
    _kpis([("VaR 95% (1d)", _pct(tl["95"]["var"])), ("ES 95% (1d)", _pct(tl["95"]["es"])),
           ("VaR 99% (1d)", _pct(tl["99"]["var"])), ("ES 99% (1d)", _pct(tl["99"]["es"])),
           ("Avg pair corr", _num(d["avg_pair_corr"]))])
    t = x["table"].sort_values("Risk contribution", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t.index, y=t["Weight"] / t["Weight"].abs().sum(), name="Share of weight",
                         marker_color=BRAND["faint"]))
    fig.add_trace(go.Bar(x=t.index, y=t["Risk contribution"], name="Share of risk", marker_color=BRAND["red"]))
    charts._base(fig, 320, "Where the weight sits vs where the risk sits")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    _fig(fig)
    st.caption(f"A holding whose share of risk far exceeds its share of weight is driving the portfolio. "
               f"Window {x['inputs']['start']} to {x['inputs']['end']} ({x['inputs']['sessions']} sessions). "
               + " ".join(x["notes"]))
    st.dataframe(t.style.format({"Weight": "{:+.1%}", "Stand-alone vol": "{:.1%}", "Risk contribution": "{:.1%}",
                                 "Return contribution": "{:+.1%}", "Risk / weight": "{:.2f}"}, na_rep="n/a"),
                 use_container_width=True)
    if len(x["corr"]) > 1:
        _fig(charts.correlation_matrix(x["corr"]))


# ----------------------------------------------------------------------
def _tab_events(S_: Dict[str, Any], ds) -> None:
    from .eventstudy import event_study, parse_events
    _section("Event study")
    st.caption("Abnormal return around dated events against a market-model baseline estimated before the "
               "event window. The benchmark is the market proxy.")
    if ds.benchmark is None:
        st.warning("This dataset has no benchmark. Set one in the sidebar and reload.")
        return
    default = "\n".join(f"{ds.tickers[0]} {d}" for d in ds.prices.index[400::180][:8].strftime("%Y-%m-%d"))
    txt = st.text_area("Events (TICKER YYYY-MM-DD per line)", default, height=130, key="lab_ev_txt")
    c1, c2 = st.columns(2)
    pre = c1.number_input("Days before", 0, 60, 5, key="lab_ev_pre")
    post = c2.number_input("Days after", 1, 120, 20, key="lab_ev_post")
    if st.button("Run event study", key="lab_ev_run"):
        S_["events"] = event_study(ds.prices, ds.benchmark, parse_events(txt), int(pre), int(post))
    ev = S_.get("events")
    if not ev:
        return
    if ev["n_events"] == 0:
        st.warning("No usable events. " + "; ".join(f"{a} {b}: {c}" for a, b, c in ev["skipped"][:5]))
        return
    _kpis([("Events", f"{ev['n_events']}"), ("Mean CAR", _pct(ev["mean_car"])), ("Median CAR", _pct(ev["median_car"])),
           ("t-stat", _num(ev["t_car"])), ("Share positive", _pct(ev["share_positive"]))])
    a = ev["average"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["Relative day"], y=a["CAR 75th"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=a["Relative day"], y=a["CAR 25th"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(228,28,35,0.15)", name="25th-75th percentile"))
    fig.add_trace(go.Scatter(x=a["Relative day"], y=a["Mean CAR"], name="Mean CAR", line=dict(color=BRAND["red"], width=2.4)))
    fig.add_vline(x=0, line_dash="dot", line_color=BRAND["muted"])
    charts._base(fig, 340, "Cumulative abnormal return by trading day relative to the event")
    fig.update_yaxes(tickformat=".1%")
    _fig(fig)
    st.dataframe(ev["table"], hide_index=True, use_container_width=True)
    if ev["skipped"]:
        st.caption(f"{len(ev['skipped'])} event(s) skipped: " + "; ".join(f"{a} {b} ({c})" for a, b, c in ev["skipped"][:6]))


# ----------------------------------------------------------------------
def _tab_monitor(S_: Dict[str, Any]) -> None:
    _section("Registered strategies")
    st.caption("A registered strategy is frozen as it was on the day. Everything after that date is out-of-sample "
               "for that exact rule: the only evidence a backtest cannot manufacture.")
    strat = L.list_strategies()
    if not strat:
        st.info("None yet. Register one from the Strategy Lab after a run.")
        return
    st.dataframe(pd.DataFrame([{"name": s["name"], "strategy": s["strategy_key"], "registered": s["registered"],
                                "status": s["status"], "baseline Sharpe": (s["baseline"] or {}).get("sharpe")}
                               for s in strat]), hide_index=True, use_container_width=True)
    if st.button("Scan for decay (downloads current prices)", key="lab_scan"):
        with st.spinner("Re-running registered strategies on fresh data..."):
            S_["scan"] = T.decay_scan()
    for a in (S_.get("scan") or {}).get("strategies", []):
        color = {"healthy": BRAND["gain"], "fresh": BRAND["muted"], "aging": "#D9A441", "stale": BRAND["loss"],
                 "error": BRAND["loss"]}.get(a.get("status"), BRAND["muted"])
        st.markdown(f"**{a['name']}** {_chip(str(a.get('status')).upper(), color)} &nbsp; {a.get('detail', '')}",
                    unsafe_allow_html=True)
        if a.get("forward_sessions"):
            _kpis([("Forward sessions", f"{a['forward_sessions']}"), ("Forward Sharpe", _num(a.get("forward_sharpe"))),
                   ("Baseline Sharpe", _num(a.get("baseline_sharpe"))), ("z vs baseline", _num(a.get("z_vs_baseline"))),
                   ("Forward return", _pct(a.get("forward_return")))])
    with st.expander("Retire a strategy"):
        nm = st.selectbox("Strategy", [s["name"] for s in strat], key="lab_retire")
        if st.button("Retire", key="lab_retire_b"):
            L.retire_strategy(nm)
            st.rerun()


# ----------------------------------------------------------------------
def _tab_assistant(S_: Dict[str, Any]) -> None:
    from . import assistant as A
    _section("Assistant")
    ok, why = A.available()
    st.caption("Ask for research in plain language. The assistant can only call the lab's tools "
               "(the same ones the MCP server exposes) and cannot trade.")
    if not ok:
        st.info(why)
        st.markdown("**Use Claude directly instead.** The lab is an MCP server. Register it once:")
        st.code("claude mcp add qbt-lab -- C:/Users/marcd/miniconda3/envs/quant/python.exe -m qbt.lab mcp",
                language="bash")
        return
    for m in S_["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
    q = st.chat_input("e.g. Screen momentum factors on my ETF universe and test the best one")
    if q:
        S_["chat"].append({"role": "user", "text": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"), st.spinner("Working..."):
            reply, trace = A.ask(q, S_.get("did"), S_.get("hid"), history=S_["chat"][:-1])
            st.markdown(reply)
            if trace:
                with st.expander(f"{len(trace)} tool call(s)"):
                    for t in trace:
                        st.code(f"{t['name']}({t['args']})", language="python")
        S_["chat"].append({"role": "assistant", "text": reply})
