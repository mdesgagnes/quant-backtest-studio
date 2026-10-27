"""Streamlit front end of the Research workspace.

A thin layer: every number comes from `qbt.lab.tools` / the pure modules, so
what the page shows is what the MCP server and the command line return.
Imported only when the workspace is selected, except `event_panel`, which the
Backtest workspace also uses.

Layout rules, because this screen has to be readable by someone who does not
live in quant jargon:
- figures sit in wrapping cards (never squeezed into narrow columns where a
  value gets cut off);
- every tab opens with what it is for, and every statistic has a hover
  explanation and an entry in the Guide;
- results come with a sentence saying what they mean and what to do next.
"""
from __future__ import annotations

import html
import io
import os
import zipfile
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import explain as X
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
VERDICT_COLOR = {"Strong evidence": BRAND["gain"], "Probable signal": "#8DBA6B",
                 "Likely luck (fails correction)": "#D9A441", "No evidence": BRAND["muted"]}


# ----------------------------------------------------------------------
# Small building blocks
# ----------------------------------------------------------------------
def _e(text: Any) -> str:
    return html.escape(str(text))


def _chip(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#111;border-radius:3px;padding:2px 9px;'
            f'font-size:.72rem;font-weight:700;white-space:nowrap">{_e(text)}</span>')


def _section(title: str) -> None:
    st.markdown(f'<div class="eyebrow">{_e(title)}</div>', unsafe_allow_html=True)


def _note(text: str) -> None:
    """Grey explanatory note. `text` may contain simple inline HTML (<b>, <code>)."""
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def _callout(text: str, tone: str = "info") -> None:
    colors = {"info": BRAND["rule"], "good": BRAND["gain"], "warn": "#D9A441", "bad": BRAND["loss"]}
    st.markdown(f'<div style="border-left:3px solid {colors[tone]};background:{BRAND["panel"]};'
                f'padding:.65rem .9rem;border-radius:0 4px 4px 0;margin:.5rem 0;line-height:1.6;'
                f'font-size:.88rem">{text}</div>', unsafe_allow_html=True)


def _cards(items: List[tuple], min_px: int = 165) -> None:
    """Figures as a wrapping grid of cards: (label, value[, sub[, tone]]).

    A grid with auto-fit columns wraps on a narrow screen instead of cutting
    the value off, which is what a row of st.metric columns does."""
    cells = ""
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else ""
        tone = it[3] if len(it) > 3 else ""
        cells += (f'<div class="dial {tone}"><span class="k">{_e(label)}</span><span class="v">{_e(value)}</span>'
                  f'{f"<div class=d>{_e(sub)}</div>" if sub else ""}</div>')
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax({min_px}px,1fr));'
                f'gap:.6rem;margin:.4rem 0 .9rem">{cells}</div>', unsafe_allow_html=True)


def _fig(fig: go.Figure, caption: str = "") -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if caption:
        _note(caption)


def _pct(v, d: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}%}"


def _num(v, d=2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def _tone(v, good_positive: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return ("pos" if v > 0 else "neg") if good_positive else ""


def _guide(group: str, label: str = "Column and term guide") -> None:
    """The glossary entries for one area, folded away under the tab's content."""
    with st.expander(f"{label}: what these numbers mean"):
        _glossary_table(X.GLOSSARY[group])


def _glossary_table(rows: List[tuple]) -> None:
    body = ""
    for term, meaning, good, watch in rows:
        body += (f'<tr><td style="padding:.45rem .6rem;vertical-align:top;white-space:nowrap"><b>{_e(term)}</b></td>'
                 f'<td style="padding:.45rem .6rem;vertical-align:top">{_e(meaning)}'
                 + (f'<div style="color:{BRAND["gain"]};font-size:.78rem;margin-top:.2rem">Good: {_e(good)}</div>' if good else "")
                 + (f'<div style="color:{BRAND["loss"]};font-size:.78rem">Watch out: {_e(watch)}</div>' if watch else "")
                 + "</td></tr>")
    st.markdown(f'<table style="width:100%;border-collapse:collapse;font-size:.85rem">{body}</table>',
                unsafe_allow_html=True)


def _state() -> Dict[str, Any]:
    return st.session_state.setdefault("lab", {"did": None, "run_id": None, "screen": None, "xray": None,
                                               "events": None, "scan": None, "chat": [], "screened": False,
                                               "saved": False})


def _dataset():
    did = _state()["did"]
    return T.DATASETS.get(did) if did else None


# ----------------------------------------------------------------------
def render() -> None:
    S_ = _state()
    _sidebar(S_)
    st.markdown('<div class="masthead"><div><h1>Research Lab</h1>'
                '<div class="sub">Find an idea, test it honestly, decide. Research only: nothing here trades.'
                '</div></div></div>', unsafe_allow_html=True)
    ds = _dataset()
    tabs = st.tabs(["Start here", "Ledger", "Factor Lab", "Strategy Lab", "Risk X-Ray", "Event Study",
                    "Monitor", "Assistant"])
    with tabs[0]:
        _tab_start(S_, ds)
    with tabs[1]:
        _tab_ledger(S_, ds)
    if ds is None:
        for t in tabs[2:6]:
            with t:
                st.info("Load a universe from the sidebar first (left): pick the instruments you want to research "
                        "and press **Load universe**.")
    else:
        with tabs[2]:
            _tab_factors(S_, ds)
        with tabs[3]:
            _tab_strategy(S_, ds)
        with tabs[4]:
            _tab_xray(S_, ds)
        with tabs[5]:
            _tab_events(S_, ds)
    with tabs[6]:
        _tab_monitor(S_)
    with tabs[7]:
        _tab_assistant(S_)


# ----------------------------------------------------------------------
def _sidebar(S_: Dict[str, Any]) -> None:
    sb = st.sidebar
    sb.markdown('<div class="eyebrow">1. Universe</div>', unsafe_allow_html=True)
    sb.caption("The instruments you are researching. More names give the factor screen more to rank: "
               "aim for 8 or more.")
    names = ["Custom"] + PRESETS.names()
    pick = sb.selectbox("Preset", names, key="lab_preset")
    if pick != st.session_state.get("_lab_applied"):
        st.session_state["_lab_applied"] = pick
        if pick != "Custom":
            st.session_state["lab_tickers"] = "\n".join(PRESETS.get(pick)["tickers"])
            st.session_state["lab_bench"] = PRESETS.get(pick).get("benchmark", "")
    st.session_state.setdefault("lab_tickers", "SPY\nEFA\nEEM\nTLT\nIEF\nGLD\nVNQ\nDBC")
    st.session_state.setdefault("lab_bench", "SPY")
    txt = sb.text_area("Symbols (one per line)", key="lab_tickers", height=140)
    tickers = [t.strip().upper() for t in txt.replace(",", "\n").split("\n") if t.strip()]
    c1, c2 = sb.columns(2)
    start = c1.date_input("From", value=date(date.today().year - 12, 1, 1), min_value=date(1970, 1, 1),
                          max_value=date.today(), key="lab_start")
    end = c2.date_input("To", value=date.today(), min_value=date(1971, 1, 1), max_value=date.today(), key="lab_end")
    bench = sb.text_input("Benchmark", key="lab_bench",
                          help="The market you compare against. Also the baseline for the event study.")
    synth = sb.checkbox("Use synthetic demo data (offline)", value=False, key="lab_synth",
                        help="Reproducible fake prices with a weak planted momentum effect. For trying the tools.")
    if sb.button("Load universe", key="lab_load", type="primary"):
        with st.spinner("Loading prices..."):
            out = (T.load_data(synthetic=True, edge=0.05) if synth else
                   T.load_data(tickers=tickers, start=str(start), end=str(end), benchmark=bench or None))
        if "error" in out:
            sb.error(out["error"])
        else:
            S_.update(did=out["dataset_id"], run_id=None, screen=None, xray=None, events=None,
                      screened=False, saved=False)
            for w in out.get("warnings", [])[:3]:
                sb.warning(w)

    sb.markdown('<div class="eyebrow">2. Active hypothesis</div>', unsafe_allow_html=True)
    sb.caption("Optional but important: every test you run is logged under it, and the deflated Sharpe "
               "uses that count to stay honest.")
    hyps = L.list_hypotheses()
    opts = {0: "None (runs are not logged)"} | {h["id"]: f"#{h['id']} {h['title']}" for h in hyps}
    S_["hid"] = sb.selectbox("Log experiments under", list(opts), format_func=lambda k: opts[k],
                             key="lab_hid") or None
    if S_["hid"]:
        sb.caption(f"{L.trial_count(S_['hid'])} trial(s) counted so far.")


# ----------------------------------------------------------------------
# Start here
# ----------------------------------------------------------------------
def _tab_start(S_: Dict[str, Any], ds) -> None:
    _section("What this workspace is for")
    st.markdown(
        "A backtest tells you what a rule **would have** returned. This workspace helps with the two questions "
        "around it: *is there anything here worth testing?* and *is the result evidence or luck?* "
        "It is built to stop the most common way systematic research goes wrong: trying many things, keeping "
        "the best one, and believing it.")

    _section("The workflow, with your progress")
    run_ok = bool(S_.get("run_id") and S_["run_id"] in T.RUNS)
    val_ok = bool(run_ok and T.RUNS[S_["run_id"]].get("validation_out"))
    steps = [
        (ds is not None, "Load a universe", "Sidebar, step 1. The instruments you want to research."),
        (bool(S_.get("hid")), "Pick or write a hypothesis (recommended)",
         "Ledger tab. Say what you expect and why, before testing. Select it in the sidebar so every test is counted."),
        (bool(S_.get("screened")), "Screen the factors",
         "Factor Lab. Which simple signals have predictive power here, after correcting for how many you tried?"),
        (run_ok, "Backtest a strategy", "Strategy Lab. Turn the best idea into rules and run it with trading costs."),
        (val_ok, "Validate it", "Strategy Lab. The evidence-or-luck battery. Add a parameter grid for the overfitting checks."),
        (bool(S_.get("saved")), "Save the run card",
         "Strategy Lab. A folder with the numbers, the data fingerprint, and the warnings, so it can be checked later."),
    ]
    for done, title, text in steps:
        mark = "&#10003;" if done else "&#9675;"
        color = BRAND["gain"] if done else BRAND["faint"]
        st.markdown(f'<div style="display:flex;gap:.7rem;margin:.35rem 0"><span style="color:{color};font-size:1.1rem">'
                    f'{mark}</span><div><b>{_e(title)}</b><div style="color:{BRAND["muted"]};font-size:.84rem">{_e(text)}'
                    f'</div></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{BRAND["muted"]};font-size:.84rem;margin-top:.4rem">Afterwards: '
                f'<b>Risk X-Ray</b> shows what a portfolio is really made of, <b>Event Study</b> shows what prices do '
                f'around dated events, and <b>Monitor</b> tracks a frozen strategy as new data arrives.</div>',
                unsafe_allow_html=True)

    _section("How to decide: what to do with a result")
    rows = [
        ("The factor screen shows no factor that survives the correction",
         "Nothing here beats noise on this universe and period. Do not build a strategy on the best raw winner. "
         "Try a larger universe, a different horizon, or accept that there is no edge."),
        ("A factor is 'Strong evidence' or 'Probable signal'",
         "Worth a backtest. Check the IC-by-horizon chart: rebalance at about the horizon where it peaks."),
        ("A backtest looks good but the grade is Fragile or Unsupported",
         "Believe the grade over the headline. Read which checks failed and the advice under each."),
        ("The deflated Sharpe check says N/A",
         "You have no hypothesis selected, so the app cannot count your trials. Select one and re-run: "
         "this is the most common way a result looks better than it is."),
        ("The random-timing test fails",
         "The return is mostly market exposure, not timing. Compare it to simply holding the same assets."),
        ("PBO above 50%, or a big in-sample/out-of-sample gap",
         "Picking the 'best' parameters is overfitting. Prefer a setting from the middle of a broad good zone, "
         "or fix the parameters before looking."),
        ("Everything passes", "Good. Save the run card, register the strategy, and let the Monitor tell you if "
                              "the forward record disagrees. A backtest is still not a forecast."),
    ]
    body = "".join(f'<tr><td style="padding:.5rem .6rem;vertical-align:top;width:34%"><b>{_e(a)}</b></td>'
                   f'<td style="padding:.5rem .6rem;vertical-align:top;color:{BRAND["muted"]}">{_e(b)}</td></tr>'
                   for a, b in rows)
    st.markdown(f'<table style="width:100%;border-collapse:collapse;font-size:.86rem">{body}</table>',
                unsafe_allow_html=True)

    _section("Glossary")
    st.caption("Every unconventional term, in plain words, with what good looks like and what to watch out for. "
               "The same entries sit at the bottom of each tab.")
    for group, rows_ in X.GLOSSARY.items():
        with st.expander(group):
            _glossary_table(rows_)

    _section("Limits to keep in mind")
    _note("Factors use price data only. Universes built from today's tickers carry survivorship bias. "
          "A small universe gives noisy statistics. A pass is evidence, not proof, and nothing here is a forecast "
          "or investment advice.")


# ----------------------------------------------------------------------
def _tab_ledger(S_: Dict[str, Any], ds) -> None:
    _section("Hypotheses")
    st.markdown("Write the idea down **before** testing it. Every experiment run under a hypothesis is counted, "
                "and that count is what the deflated Sharpe ratio charges you for: a Sharpe found on the first try "
                "and the same Sharpe found as the best of 80 variants are very different facts.")
    with st.expander("New hypothesis", expanded=not L.list_hypotheses()):
        with st.form("lab_new_h", clear_on_submit=True):
            title = st.text_input("Title", placeholder="Trend filter reduces drawdown on a global ETF set")
            stmt = st.text_area("Statement", placeholder="What exactly do you expect to see, and by how much?", height=70)
            why = st.text_area("Why it should work", placeholder="An economic reason, not a pattern in the data",
                               height=70)
            if st.form_submit_button("Add hypothesis"):
                try:
                    hid = L.add_hypothesis(title, stmt, why, ", ".join(ds.tickers) if ds else "")
                    st.success(f"Hypothesis #{hid} added. Select it in the sidebar to log tests under it.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    hyps = L.list_hypotheses()
    if hyps:
        df = pd.DataFrame(hyps)[["id", "title", "status", "n_experiments", "n_trials", "updated"]]
        st.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "n_experiments": st.column_config.NumberColumn("Experiments", help="Number of logged runs."),
            "n_trials": st.column_config.NumberColumn("Trials", help="Distinct configurations tried. "
                                                      "This is the number the deflated Sharpe uses."),
        })
        c1, c2, c3 = st.columns([2, 2, 1])
        hid = c1.selectbox("Hypothesis", [h["id"] for h in hyps], format_func=lambda i: f"#{i}", key="lab_edit_h")
        new = c2.selectbox("Status", L.STATUSES, key="lab_edit_s",
                           help="idea > testing > validated or rejected > promoted > retired")
        if c3.button("Set status", key="lab_set_s"):
            L.update_hypothesis(hid, status=new)
            st.rerun()
        ex = L.list_experiments(hid)
        if ex:
            st.markdown(f"**Experiments under #{hid}**: {L.trial_count(hid)} trials counted")
            rows = [{"id": e["id"], "when": e["created"], "kind": e["kind"], "name": e["name"],
                     "variants": e["n_variants"], "Sharpe": e["metrics"].get("Sharpe"),
                     "verdict": e["verdict"]} for e in ex]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("Export ledger (JSON)", L.export_json(), "qbt_lab_ledger.json", "application/json",
                       help="The ledger lives on this machine. On a hosted app it is wiped when the app "
                            "restarts, so export it first.")


# ----------------------------------------------------------------------
# Factor Lab
# ----------------------------------------------------------------------
def _tab_factors(S_: Dict[str, Any], ds) -> None:
    _section("Which simple signals have predictive power?")
    st.markdown(
        f"A **factor** is a number worked out for every instrument every day from its price history, for example "
        f"its return over the last 12 months. This screen asks, for each of {len(F.FACTORS)} factors: *did the "
        f"instruments it ranked highest tend to do better afterwards than the ones it ranked lowest?*")
    _note("Because {n} factors are tested at once, some will look good by chance alone. The table therefore "
          "shows a <b>Verdict</b> that already corrects for that. Trust the verdict, not the raw ranking.".format(
              n=len(F.FACTORS)))
    c1, c2, c3 = st.columns([1.2, 2.6, 1])
    horizon = c1.select_slider("Look-ahead (days)", [5, 10, 21, 42, 63], value=21, key="lab_h",
                               help="How far ahead each factor is asked to predict. 21 days is about a month.")
    cats = sorted({f.category for f in F.FACTORS.values()})
    chosen = c2.multiselect("Factor families", cats, default=cats, key="lab_cats")
    c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    run = c3.button("Run screen", key="lab_screen", type="primary")
    if ds.prices.shape[1] < 6:
        st.warning("Fewer than 6 instruments: a ranking of so few names is very noisy.")
    if run:
        keys = [k for k, f in F.FACTORS.items() if f.category in chosen]
        with st.spinner(f"Scoring {len(keys)} factors..."):
            out = T.factor_screen(dataset_id=S_["did"], horizon=int(horizon), factors=keys, top=len(keys),
                                  hypothesis_id=S_.get("hid"))
        S_["screen"] = out
        S_["screened"] = "error" not in out
    out = S_.get("screen")
    if out and "error" not in out and (not out["top"] or "t-stat" not in out["top"][0]):
        # Every factor failed: show why, rather than an empty or broken table.
        msgs = sorted({str(r.get("error")) for r in out["top"] if r.get("error")})
        out = {"error": "No factor could be scored. " + ("; ".join(msgs) or "No data.")}
    if not out:
        _guide("Factor Lab")
        return
    if "error" in out:
        st.error(out["error"])
        return

    rows = pd.DataFrame(out["top"]).rename(columns={"key": "Key"})
    rows["Verdict"] = [X.factor_verdict(r) for r in rows.to_dict("records")]
    n_strong = int((rows["Verdict"] == "Strong evidence").sum())
    n_prob = int((rows["Verdict"] == "Probable signal").sum())
    n_rev = int(rows["Verdict"].str.startswith("Works in reverse").sum())
    n_luck = int((rows["Verdict"] == "Likely luck (fails correction)").sum())
    _cards([("Factors tested", f"{out['n_factors']}", f"{out['horizon']}-day look-ahead"),
            ("Strong evidence", f"{n_strong}", "survive a strict correction", "pos" if n_strong else ""),
            ("Probable", f"{n_prob}", "survive a looser correction"),
            ("Work in reverse", f"{n_rev}", "significant, but opposite to the literature"),
            ("Likely luck", f"{n_luck}", "looked good, then failed the correction")])
    best = rows[rows["Verdict"].isin(["Strong evidence", "Probable signal"])].head(3)
    rev = rows[rows["Verdict"].str.startswith("Works in reverse")].head(3)
    rev_names = ", ".join(_e(r["Factor"]) for r in rev.to_dict("records"))
    if len(best):
        names = ", ".join(f"<b>{_e(r['Factor'])}</b> (IC {r['Mean IC']:+.3f})" for r in best.to_dict("records"))
        extra = (f" {len(rev)} more are significant in reverse ({rev_names}); see below." if len(rev) else "")
        _callout(f"{len(best)} factor(s) show evidence of predicting returns on this universe: {names}. "
                 f"Pick one below to see why, then backtest it.{extra}", "good")
    elif len(rev):
        _callout(f"No factor works in the direction the literature expects, but <b>{len(rev)}</b> are significant "
                 f"the <i>opposite</i> way ({rev_names}): on this universe and period the instruments the factor "
                 f"favoured did worse. That can be a real feature of this universe but is more often a quirk of "
                 f"the sample. Treat it as a lead to investigate, not a ready-made strategy.", "warn")
    elif n_luck:
        _callout(f"No factor survives the correction. {n_luck} looked promising on their own but that is what "
                 f"chance produces when you test {out['n_factors']}. <b>Do not build a strategy on them.</b>", "warn")
    else:
        _callout("No factor shows predictive power on this universe and period.", "warn")

    show_all = st.checkbox("Show every statistic", value=False, key="lab_f_all",
                           help="Off: the columns that matter. On: also the raw p-values, hit rate and turnover.")
    cols = ["Factor", "Category", "Verdict", "Mean IC", "t-stat", "p_bh", "Spread Sharpe"]
    if show_all:
        cols = ["Factor", "Category", "Verdict", "Mean IC", "ICIR", "t-stat", "p", "p_bonferroni", "p_bh",
                "IC hit rate", "Spread Sharpe", "Monotonicity", "Top-bucket turnover", "n"]
    view = rows[[c for c in cols if c in rows]]
    H = X.COLUMN_HELP
    cfg = {
        "Mean IC": st.column_config.NumberColumn("Predictive power (IC)", format="%+.3f", help=H["Mean IC"]),
        "ICIR": st.column_config.NumberColumn("Consistency (ICIR)", format="%+.2f", help=H["ICIR"]),
        "t-stat": st.column_config.NumberColumn("t-stat", format="%+.2f", help=H["t-stat"]),
        "p": st.column_config.NumberColumn("Raw p", format="%.3f", help=H["p"]),
        "p_bonferroni": st.column_config.NumberColumn("Bonferroni p", format="%.3f", help=H["p_bonferroni"]),
        "p_bh": st.column_config.NumberColumn("Adjusted p (trust this)", format="%.3f", help=H["p_bh"]),
        "IC hit rate": st.column_config.NumberColumn("Days IC > 0", format="%.2f",
                                                     help="Share of sampled dates on which the ranking pointed the right way."),
        "Spread Sharpe": st.column_config.NumberColumn("Long-short Sharpe", format="%+.2f", help=H["Spread Sharpe"]),
        "Monotonicity": st.column_config.NumberColumn("Monotonicity", format="%+.2f", help=H["Monotonicity"]),
        "Top-bucket turnover": st.column_config.NumberColumn("Top-bucket turnover", format="%.2f",
                                                             help=H["Top-bucket turnover"]),
        "n": st.column_config.NumberColumn("Samples", format="%d", help=H["n"]),
        "Verdict": st.column_config.TextColumn("Verdict", help=H["Verdict"]),
    }
    st.dataframe(view, hide_index=True, use_container_width=True, column_config=cfg,
                 height=min(640, 35 * (len(view) + 1) + 3))

    _section("Look at one factor in detail")
    key_map = {r["Factor"]: r.get("Key") for r in rows.to_dict("records")}
    pick = st.selectbox("Factor", list(key_map), key="lab_f_pick")
    fkey = key_map.get(pick)
    if fkey:
        _factor_detail(ds, fkey, int(out["horizon"]))
    _guide("Factor Lab")


def _factor_detail(ds, fkey: str, horizon: int) -> None:
    f = F.FACTORS[fkey]
    sc = F.compute(fkey, ds.prices)
    ic = F.information_coefficient(sc, ds.prices, horizon)
    st.markdown(f"**{_e(f.label)}** ({_e(f.category)}). {_e(f.note)} "
                f"{'The literature expects the opposite sign, so the score is flipped: higher always means expected better.' if f.expected == '-' else ''}")
    _cards([("Predictive power (mean IC)", _num(ic["mean_ic"], 3), "0 = none, +1 = perfect", _tone(ic["mean_ic"])),
            ("Consistency (ICIR)", _num(ic["icir"]), "higher = steadier"),
            ("t-stat", _num(ic["t_stat"]), "about 3+ is solid when many are tested"),
            ("Days the ranking was right", _pct(ic["hit_rate"], 0), "of sampled dates"),
            ("Samples", f"{ic['n']}", "non-overlapping windows")])
    c1, c2 = st.columns(2)
    if "ic" in ic and len(ic["ic"]):
        s = ic["ic"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=s.index, y=s.values, marker_color=BRAND["faint"], name="Daily IC", opacity=.5))
        fig.add_trace(go.Scatter(x=s.index, y=s.rolling(126, min_periods=40).mean(), name="6-month average",
                                 line=dict(color=BRAND["red"], width=2)))
        charts._base(fig, 320, "Predictive power over time")
        with c1:
            _fig(fig, "Above zero = the ranking pointed the right way. A good factor spends most of its time "
                      "above zero; one that is up for two years and down for five is not stable.")
    dec = F.ic_decay(sc, ds.prices)
    with c2:
        _fig(charts.bar_series([f"{h}d" for h in dec["Horizon (days)"]], dec["Mean IC"].fillna(0).tolist(),
                               "How long the signal lasts (IC by look-ahead)"),
             "Where the bars peak is roughly how often to rebalance. A signal that has faded by 5 days is no "
             "use to a monthly strategy.")
    qr = F.quantile_returns(sc, ds.prices, 3 if ds.prices.shape[1] < 15 else 5, horizon)
    if qr.get("table") is not None and len(qr["table"]):
        c3, c4 = st.columns(2)
        mq = qr["mean_by_q"]
        with c3:
            _fig(charts.bar_series([f"Bucket {int(i)}" for i in mq.index], (mq * 100).tolist(),
                                   f"Average {horizon}-day return by score bucket (%)", "%"),
                 "Bucket 1 holds the lowest-scored instruments, the last bucket the highest. If the factor works "
                 "the bars should climb from left to right.")
        sp = (1 + qr["spread"]).cumprod() * 100
        with c4:
            _fig(charts.equity_curve(pd.DataFrame({"Top minus bottom": sp}), log=False,
                                     title=f"Buy the top bucket, short the bottom (Sharpe {_num(qr['spread_sharpe'])})"),
                 f"Rising steadily = the ranking turns into money. Monotonicity {_num(qr['monotonicity'])} "
                 f"(1.0 = perfectly ordered). About {_pct(qr['top_turnover'], 0)} of the top bucket changes each "
                 f"rebalance, which is what trading costs are paid on.")
    if st.button("Backtest this factor in the Strategy Lab", key=f"lab_to_strat_{fkey}"):
        st.session_state["lab_strat"] = "zoo_factor"
        st.session_state["lab_p_factor"] = fkey
        st.success("Done. Open the Strategy Lab tab: this factor is preselected.")


# ----------------------------------------------------------------------
# Strategy Lab
# ----------------------------------------------------------------------
def _param_form(strategy_key: str, prefix: str) -> Dict[str, Any]:
    s = REGISTRY[strategy_key]
    params: Dict[str, Any] = {}
    cols = st.columns(2)
    visible = [p for p in s.params if p.kind in ("int", "float", "choice", "bool")]
    for i, p in enumerate(visible):
        col = cols[i % 2]
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
    _section("1. Choose and run a strategy")
    _note("Pick a rule, set its parameters, and run it with trading costs through the same engine as the Backtest "
          "workspace. To test a factor from the Factor Lab, choose <b>Factor Zoo</b> and select the factor.")
    key = st.selectbox("Strategy", list(usable), format_func=lambda k: usable[k].label, key="lab_strat")
    st.caption(usable[key].description)
    params = _param_form(key, "lab_p")
    with st.expander("Execution and costs"):
        e1, e2 = st.columns(2)
        reb = e1.selectbox("Rebalance", ["M", "W", "Q", "S", "A", "D"], key="lab_reb",
                           format_func=lambda c: {"M": "Monthly", "W": "Weekly", "Q": "Quarterly", "S": "Semi-annual",
                                                  "A": "Annual", "D": "Daily"}[c])
        lag = e2.number_input("Execution lag (days)", 0, 5, 1, key="lab_lag",
                              help="0 trades at the price that produced the signal: look-ahead bias.")
        e3, e4 = st.columns(2)
        com = e3.number_input("Commission (bps)", 0.0, 100.0, 5.0, 0.5, key="lab_com",
                              help="1 bp = 0.01%. Charged on every trade.")
        slp = e4.number_input("Slippage (bps)", 0.0, 200.0, 10.0, 1.0, key="lab_slp",
                              help="The price you get versus the price you saw. Realistic ETF values: 2 to 25.")
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
    bm = {}
    if bench is not None:
        from .. import metrics as M
        bm = M.summary(bench.returns, bench.equity, None, None, None, 0.0, r["engine"].periods_per_year)
    _cards([("CAGR", _pct(met.get("CAGR")), f"benchmark {_pct(bm.get('CAGR'))}" if bm else "", _tone(met.get("CAGR"))),
            ("Volatility", _pct(met.get("Volatility")), f"benchmark {_pct(bm.get('Volatility'))}" if bm else ""),
            ("Sharpe", _num(met.get("Sharpe")), f"benchmark {_num(bm.get('Sharpe'))}" if bm else "",
             _tone(met.get("Sharpe"))),
            ("Worst drawdown", _pct(met.get("Max Drawdown")), f"benchmark {_pct(bm.get('Max Drawdown'))}" if bm else ""),
            ("Turnover / year", _pct(met.get("Annual Turnover"), 0), "share of the portfolio traded")])
    curves = pd.DataFrame({r["label"]: res.equity / res.equity.iloc[0] * 100})
    if bench is not None:
        b = bench.equity.reindex(curves.index).ffill()
        curves["Benchmark"] = b / b.dropna().iloc[0] * 100
    _fig(charts.equity_curve(curves), "Growth of 100, log scale so equal percentage moves look equal.")
    _fig(charts.underwater({k: (res.equity if k == r["label"] else bench.equity.reindex(res.equity.index).ffill())
                            for k in curves.columns}),
         "How far below its previous peak each line was. Ask yourself whether you could have sat through the "
         "deepest trough.")

    _section("2. Validate: evidence or luck?")
    st.markdown("A good-looking backtest proves little. This runs a battery of tests that ask whether the "
                "result could plausibly be luck, flattered by trying many variants, or just market exposure.")
    grid = _grid_form(key, params)
    n_trials = L.trial_count(S_.get("hid")) if S_.get("hid") else 1
    if S_.get("hid"):
        st.caption(f"Trials counted for the deflated Sharpe: **{n_trials}** (from the ledger).")
    else:
        st.warning("No hypothesis is selected (sidebar, step 2), so the app cannot count how many things you "
                   "tried and the deflated Sharpe check will show N/A. That makes the grade more flattering "
                   "than it should be.")
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

    _section("3. Keep it")
    k1, k2, k3 = st.columns(3)
    if k1.button("Save run card + dossier", key="lab_save"):
        with st.spinner("Writing run card..."):
            r["card"] = T.save_run_card(run_id=rid)
        S_["saved"] = "error" not in r["card"]
    card = r.get("card")
    if card and "error" not in card:
        k1.success(f"Saved to {card['run_dir']}")
        if os.path.isfile(card["dossier"]):
            with open(card["dossier"], "rb") as fh:
                k2.download_button("Download dossier (HTML)", fh.read(), "dossier.html", "text/html", key="lab_dl_d")
        k3.download_button("Download run card (ZIP)", _zip_dir(card["run_dir"]), "run_card.zip", "application/zip",
                           key="lab_dl_z")
        for w in card["warnings"]:
            st.warning(w)
    elif card:
        st.error(card["error"])
    st.caption("The run card is the folder that lets anyone (including you in a year) check the numbers and "
               "re-run it on the same prices. The dossier is its one-page readable summary.")
    with st.expander("Register for decay monitoring"):
        st.caption("Freezes this strategy as of the last date in the data. Everything after that date is genuinely "
                   "out-of-sample for it; the Monitor tab then tells you if it stays consistent.")
        nm = st.text_input("Name", value=r["label"], key="lab_reg_name")
        if st.button("Freeze strategy as of today", key="lab_reg"):
            o = T.register_strategy(run_id=rid, name=nm)
            st.error(o["error"]) if "error" in o else st.success(f"Registered. Out-of-sample starts after {o['registered']}.")
    _guide("Strategy validation")


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
    with st.expander("Optional: test neighbouring parameter values (adds the overfitting checks)"):
        st.caption("If you tuned a parameter, try values around it. If performance collapses a little way from "
                   "your choice, you found a lucky spike rather than a robust setting. Every value tried counts as "
                   "a trial.")
        pick = st.multiselect("Parameters to vary (up to 2)", [p.key for p in nums], max_selections=2,
                              key="lab_grid_pick", format_func=lambda k: next(p.label for p in nums if p.key == k))
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
    st.markdown(f'<div style="margin:.8rem 0 .3rem;font-size:1.15rem"><b>Grade</b> &nbsp;'
                f'{_chip(g["label"], GRADE_COLOR.get(g["label"], BRAND["muted"]))}</div>', unsafe_allow_html=True)
    v = r.get("validation")
    if v:
        _callout(_e(X.plain_summary(v)), {"Robust": "good", "Promising": "good", "Fragile": "warn"}.get(g["label"], "bad"))
    st.caption(f"{g['passes']} passed, {g['warnings']} borderline, {g['fails']} failed, "
               f"{out['n_trials_counted']} trial(s) counted. The grade counts independent lines of evidence; "
               f"it is not a probability of making money.")
    for c in out["checks"]:
        stt = c["status"].upper()
        q = X.check_question(c.get("key", ""))
        advice = X.CHECK_HELP.get(c.get("key", ""), ("", ""))[1] if stt in ("FAIL", "WARN") else ""
        faint, warn = BRAND["faint"], "#D9A441"
        parts = [f"<b>{_e(c['check'])}</b>"]
        if q:
            parts.append(f'<div style="color:{faint};font-size:.78rem">{_e(q)}</div>')
        parts.append(f'<div style="margin-top:.25rem;font-size:.86rem">{_e(c["evidence"])}</div>')
        if advice:
            parts.append(f'<div style="color:{warn};font-size:.82rem;margin-top:.2rem">What to do: {_e(advice)}</div>')
        parts.append(f'<div style="color:{faint};font-size:.72rem;margin-top:.2rem">{_e(c["threshold"])}</div>')
        st.markdown(
            f'<div style="display:grid;grid-template-columns:78px 1fr;gap:.8rem;padding:.7rem 0;'
            f'border-bottom:1px solid {BRAND["rule"]}">'
            f'<div>{_chip(stt, STATUS_COLOR.get(stt, BRAND["muted"]))}</div><div>{"".join(parts)}</div></div>',
            unsafe_allow_html=True)

    ppy = r["engine"].periods_per_year
    rets = r["res"].returns
    c1, c2 = st.columns(2)
    bs = S.bootstrap_sharpe(rets, 500, 0.95, None, ppy)
    if "samples" in bs:
        fig = go.Figure(go.Histogram(x=bs["samples"], nbinsx=40, marker_color=BRAND["faint"]))
        fig.add_vline(x=0, line_color=BRAND["loss"], line_dash="dot")
        fig.add_vline(x=bs["observed"], line_color=BRAND["red"])
        charts._base(fig, 280, "How much the Sharpe could vary")
        with c1:
            _fig(fig, "Each bar counts re-drawn versions of your history. Red line = your Sharpe; dotted line = "
                      "zero. If much of the histogram sits left of zero, the edge is not established.")
    tt = S.random_timing_test(r["res"].weights, r["ds"].prices, 500, 21, ppy)
    if "samples" in tt:
        fig = go.Figure(go.Histogram(x=tt["samples"], nbinsx=40, marker_color=BRAND["faint"]))
        fig.add_vline(x=tt["observed_gross"], line_color=BRAND["red"])
        charts._base(fig, 280, "Same positions, held at random times")
        with c2:
            _fig(fig, "The histogram is what the same positions earned when moved to the wrong dates. If your "
                      "result (red) sits inside it, the timing added nothing.")
    wf = S.walk_forward_windows(rets, 5, ppy)
    if len(wf):
        _fig(charts.bar_series([f"{a}" for a in wf["Start"]], wf["Sharpe"].fillna(0).tolist(),
                               "Sharpe in each fifth of the history"),
             "A real edge shows up in most periods, not just one or two.")


# ----------------------------------------------------------------------
# Risk X-ray
# ----------------------------------------------------------------------
def _tab_xray(S_: Dict[str, Any], ds) -> None:
    from . import xray as XR
    _section("What is this portfolio really made of?")
    st.markdown("Weights show where the **money** is. This shows where the **risk** is: which holdings drive "
                "the ups and downs, and how many truly independent bets you have.")
    src_opts = ["Equal weight, whole universe", "Custom weights"]
    rid = S_.get("run_id")
    if rid in T.RUNS:
        src_opts = ["Latest run: current weights", "Latest run: average weights"] + src_opts
    src = st.radio("Weights from", src_opts, horizontal=True, key="lab_x_src")
    window = st.number_input("Trailing window (sessions, 0 = all)", 0, 5000, 756, 21, key="lab_x_win",
                             help="756 sessions is about three years.")
    weights: Optional[Dict[str, float]] = None
    px = ds.prices
    if src.startswith("Latest run") and rid in T.RUNS:
        w = T.RUNS[rid]["res"].weights
        weights = XR.current_weights(w) if "current" in src else XR.average_weights(w)
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
        x = XR.risk_xray(px, weights, window=int(window) or None)
    except ValueError as exc:
        st.error(str(exc))
        return
    c, rk, d, tl = x["concentration"], x["risk"], x["diversification"], x["tail"]
    n_hold = len(x["table"])
    _cards([("Volatility", _pct(rk["vol"]), "yearly ups and downs"),
            ("Worst drawdown", _pct(rk["max_drawdown"]), f"{rk['dd_peak']} to {rk['dd_trough']}"),
            ("Holdings that matter (by weight)", _num(c["effective_n"], 1), f"you hold {n_hold}"),
            ("Independent bets (by risk)", _num(d["effective_bets"], 1), f"you hold {n_hold}; fewer = overlap"),
            ("Diversification ratio", _num(d["ratio"]), "1.0 = no benefit; higher is better")])
    _cards([("Bad day (95%)", _pct(tl["95"]["var"]), "loss exceeded on 1 day in 20"),
            ("Average loss on those days", _pct(tl["95"]["es"]), "expected shortfall"),
            ("Very bad day (99%)", _pct(tl["99"]["var"]), "loss exceeded on 1 day in 100"),
            ("Average loss on those days", _pct(tl["99"]["es"]), "expected shortfall"),
            ("Average pair correlation", _num(d["avg_pair_corr"]), "how alike the holdings move")])
    t = x["table"].sort_values("Risk contribution", ascending=False)
    top = t.index[0]
    wshare = t.loc[top, "Weight"] / t["Weight"].abs().sum()
    rshare = t.loc[top, "Risk contribution"]
    if rshare > 1.5 * abs(wshare) and rshare > 0.25:
        _callout(f"<b>{_e(top)}</b> is {wshare:.0%} of the money but {rshare:.0%} of the risk. It drives the "
                 f"portfolio far more than its weight suggests.", "warn")
    elif d["effective_bets"] is not None and d["effective_bets"] < 0.6 * n_hold:
        _callout(f"You hold {n_hold} positions but they behave like about {d['effective_bets']:.1f} independent bets: "
                 f"several of them overlap.", "warn")
    else:
        _callout(f"Risk is spread roughly in line with the weights. The biggest risk driver is <b>{_e(top)}</b> at "
                 f"{rshare:.0%} of the risk.", "good")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t.index, y=t["Weight"] / t["Weight"].abs().sum(), name="Share of the money",
                         marker_color=BRAND["faint"]))
    fig.add_trace(go.Bar(x=t.index, y=t["Risk contribution"], name="Share of the risk", marker_color=BRAND["red"]))
    charts._base(fig, 340, "Where the money sits versus where the risk sits")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    _fig(fig, f"A red bar much taller than its grey bar is a holding that punches above its weight in risk. "
              f"Measured {x['inputs']['start']} to {x['inputs']['end']} ({x['inputs']['sessions']} sessions). "
              + " ".join(x["notes"]))
    st.dataframe(t, use_container_width=True, column_config={
        "Weight": st.column_config.NumberColumn("Weight", format="%.3f", help=X.COLUMN_HELP["Weight"]),
        "Stand-alone vol": st.column_config.NumberColumn("Stand-alone volatility", format="%.3f",
                                                         help=X.COLUMN_HELP["Stand-alone vol"]),
        "Risk contribution": st.column_config.NumberColumn("Share of risk", format="%.3f",
                                                           help=X.COLUMN_HELP["Risk contribution"]),
        "Return contribution": st.column_config.NumberColumn("Return contribution (yearly)", format="%.3f",
                                                             help=X.COLUMN_HELP["Return contribution"]),
        "Risk / weight": st.column_config.NumberColumn("Risk per unit of weight", format="%.2f",
                                                       help=X.COLUMN_HELP["Risk / weight"]),
    })
    if len(x["corr"]) > 1:
        _fig(charts.correlation_matrix(x["corr"]), "Red = move together, green = move opposite. Mostly red means "
                                                   "holding more of them adds little diversification.")
    _guide("Risk x-ray")


# ----------------------------------------------------------------------
# Event study (used here and in the Backtest workspace)
# ----------------------------------------------------------------------
EVENT_INTRO = ("An <b>event study</b> asks what a price (or a strategy) does <i>around</i> a specific date, compared "
               "with what it normally does given the market. Day 0 is the event. The grey baseline is learned from "
               "the ~200 days before the window, then the extra return (<b>abnormal return</b>) is added up day by "
               "day (<b>CAR</b>).")


def _event_results(results: Dict[str, Dict[str, Any]], pre: int, post: int, who: str) -> None:
    """Render one or more event-study results (name -> event_study output)."""
    usable = {k: v for k, v in results.items() if v.get("n_events", 0) > 0}
    if not usable:
        sk = [s for v in results.values() for s in v.get("skipped", [])]
        st.warning("No usable events. " + ("; ".join(f"{a} {b}: {c}" for a, b, c in sk[:5]) if sk else
                                           "Check the dates and that the data covers them."))
        return
    first_name = next(iter(usable))
    ev = usable[first_name]
    n = ev["n_events"]
    sig = ev["t_car"] == ev["t_car"] and abs(ev["t_car"]) >= 2
    _cards([("Events used", f"{n}", "fewer than ~10 = anecdote, not evidence"),
            ("Average CAR", _pct(ev["mean_car"], 2), f"extra return over days -{pre} to +{post}", _tone(ev["mean_car"])),
            ("Median CAR", _pct(ev["median_car"], 2), "the typical event"),
            ("t-stat", _num(ev["t_car"]), "about 2+ is notable"),
            ("Share positive", _pct(ev["share_positive"], 0), "events with extra return > 0")])
    if n < 10:
        _callout(f"Only {n} event(s). Treat the average as a description of those events, not a statistical finding.",
                 "warn")
    elif sig:
        direction = "more" if ev["mean_car"] > 0 else "less"
        _callout(f"Around these dates, <b>{_e(first_name)}</b> earned on average {abs(ev['mean_car']):.2%} {direction} "
                 f"than the baseline predicted, over {pre + post + 1} sessions (t = {ev['t_car']:.2f}, {n} events). "
                 f"That is unlikely to be chance.", "good")
    else:
        _callout(f"On average {_e(first_name)} earned {ev['mean_car']:+.2%} versus the baseline (t = {ev['t_car']:.2f}, "
                 f"{n} events). That is within what chance produces: no reliable pattern around these dates.", "info")
    fig = go.Figure()
    if len(usable) == 1:
        a = ev["average"]
        fig.add_trace(go.Scatter(x=a["Relative day"], y=a["CAR 75th"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=a["Relative day"], y=a["CAR 25th"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(228,28,35,0.15)", name="Middle half of events"))
        fig.add_trace(go.Scatter(x=a["Relative day"], y=a["Mean CAR"], name="Average CAR",
                                 line=dict(color=BRAND["red"], width=2.4)))
    else:
        for name, e in usable.items():
            a = e["average"]
            fig.add_trace(go.Scatter(x=a["Relative day"], y=a["Mean CAR"], name=str(name), mode="lines"))
    fig.add_vline(x=0, line_dash="dot", line_color=BRAND["muted"])
    fig.add_hline(y=0, line_color=BRAND["rule"])
    charts._base(fig, 360, "Cumulative extra return by trading day around the event (day 0 = the event)")
    fig.update_yaxes(tickformat=".1%")
    fig.update_xaxes(title="Trading days from the event")
    _fig(fig, "Flat near zero before day 0 means nothing was leaking ahead of the event. A step at day 0 is the "
              "immediate reaction; a slope afterwards is drift. The shaded band shows where the middle half of "
              "individual events fell, so a narrow band means the events agree.")
    if len(usable) > 1:
        rows = [{who: k, "Events": v["n_events"], "Average CAR": v["mean_car"], "Median CAR": v["median_car"],
                 "t-stat": v["t_car"], "Share positive": v["share_positive"]} for k, v in usable.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    tbl = ev["table"]
    if len(tbl):
        with st.expander(f"Every event ({first_name})"):
            st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
                "CAR": st.column_config.NumberColumn("CAR", format="%.4f", help=X.COLUMN_HELP["CAR"]),
                "CAR t": st.column_config.NumberColumn("CAR t", format="%.2f", help=X.COLUMN_HELP["CAR t"]),
                "Beta": st.column_config.NumberColumn("Beta", format="%.2f", help=X.COLUMN_HELP["Beta"]),
                "Day-0 AR": st.column_config.NumberColumn("Day-0 AR", format="%.4f", help=X.COLUMN_HELP["Day-0 AR"]),
                "Alpha (daily)": st.column_config.NumberColumn("Alpha (daily)", format="%.5f"),
            })
    sk = [s for v in usable.values() for s in v.get("skipped", [])]
    if sk:
        st.caption(f"{len(sk)} event(s) skipped (outside the data, or too little history before them): "
                   + "; ".join(f"{b} ({c})" for _, b, c in sk[:6]))


def event_panel(returns: Dict[str, pd.Series], market_returns: Optional[pd.Series] = None,
                periods: Optional[list] = None, rebalances: Optional[pd.DatetimeIndex] = None,
                key: str = "ev", market_label: str = "the benchmark") -> None:
    """Event study on strategy return streams, for the Backtest workspace.

    `returns`: name -> daily return series. `market_returns` is the benchmark's
    (without one the baseline is each series' own average). `periods` are the
    stress periods in use (their start dates are one event source) and
    `rebalances` the strategy's rebalance dates (another)."""
    from . import eventstudy as ES
    st.markdown(f'<div class="note">{EVENT_INTRO} Stress periods above show what happened <i>inside</i> a named '
                f'episode; this shows what the strategy does <i>around</i> a date, relative to '
                f'{_e(market_label)}.</div>', unsafe_allow_html=True)
    sources = ["Start of each stress period", f"{market_label.capitalize()}'s worst days"]
    if rebalances is not None and len(rebalances):
        sources.append("The strategy's rebalance dates")
    sources.append("My own dates")
    c1, c2, c3 = st.columns([2, 1, 1])
    src = c1.selectbox("Events", sources, key=f"{key}_src",
                       help="Which dates to study. Stress-period starts are the 'what happens as a crisis begins' "
                            "question; worst days is 'what happens after the market's biggest shocks'.")
    pre = int(c2.number_input("Days before", 0, 60, 5, key=f"{key}_pre"))
    post = int(c3.number_input("Days after", 1, 120, 20, key=f"{key}_post"))
    dates: List[pd.Timestamp] = []
    if src.startswith("Start of"):
        dates = sorted({pd.Timestamp(p.start) for p in (periods or [])})
    elif "worst days" in src:
        if market_returns is None:
            st.info("Needs a benchmark: the worst days are the benchmark's. Set one, or choose another source.")
            return
        n = st.slider("How many worst days", 5, 40, 15, key=f"{key}_n")
        dates = ES.worst_days(market_returns, n)
    elif "rebalance" in src:
        dates = list(pd.DatetimeIndex(rebalances))
    else:
        txt = st.text_area("Dates, one per line (YYYY-MM-DD)", key=f"{key}_txt", height=90,
                           placeholder="2020-03-16\n2022-06-15")
        dates = ES.parse_dates(txt)
    if len(dates) < 2:
        st.info("Add at least two dates to run the study.")
        return
    if market_returns is None:
        st.caption("No benchmark set: the baseline is each series' own average return, a weaker comparison.")
    res = ES.series_event_study(returns, market_returns, dates, pre, post)
    if not res:
        st.warning("Not enough return history to run the study.")
        return
    _event_results(res, pre, post, "Series")


def _tab_events(S_: Dict[str, Any], ds) -> None:
    from .eventstudy import event_study, parse_events
    _section("What does a price do around a dated event?")
    st.markdown(EVENT_INTRO, unsafe_allow_html=True)
    if ds.benchmark is None:
        st.warning("This dataset has no benchmark, which the study needs as its market baseline. Set one in the "
                   "sidebar and reload.")
        return
    default = "\n".join(f"{ds.tickers[0]} {d}" for d in ds.prices.index[400::180][:8].strftime("%Y-%m-%d"))
    txt = st.text_area("Events (TICKER YYYY-MM-DD, one per line)", default, height=140, key="lab_ev_txt",
                       help="For example an earnings date for a stock, a rate decision for an ETF.")
    c1, c2 = st.columns(2)
    pre = int(c1.number_input("Days before the event", 0, 60, 5, key="lab_ev_pre"))
    post = int(c2.number_input("Days after the event", 1, 120, 20, key="lab_ev_post"))
    if st.button("Run event study", key="lab_ev_run"):
        S_["events"] = {"res": {"Events": event_study(ds.prices, ds.benchmark, parse_events(txt), pre, post)},
                        "pre": pre, "post": post}
    saved = S_.get("events")
    if saved:
        _event_results(saved["res"], saved["pre"], saved["post"], "Group")
    _guide("Event study")


# ----------------------------------------------------------------------
def _tab_monitor(S_: Dict[str, Any]) -> None:
    _section("Is a frozen strategy still working?")
    st.markdown("Registering a strategy freezes it on a date. Everything after that date is genuinely "
                "out-of-sample for that exact rule: the one kind of evidence a backtest cannot manufacture. "
                "The monitor re-runs it on fresh prices and compares the new record with what it promised.")
    strat = L.list_strategies()
    if not strat:
        st.info("No registered strategies yet. After a run in the Strategy Lab, open 'Register for decay "
                "monitoring'.")
        _guide("Monitor")
        return
    st.dataframe(pd.DataFrame([{"Name": s["name"], "Strategy": s["strategy_key"], "Registered": s["registered"],
                                "Status": s["status"], "Baseline Sharpe": (s["baseline"] or {}).get("sharpe")}
                               for s in strat]), hide_index=True, use_container_width=True)
    if st.button("Scan for decay (downloads current prices)", key="lab_scan"):
        with st.spinner("Re-running registered strategies on fresh data..."):
            S_["scan"] = T.decay_scan()
    for a in (S_.get("scan") or {}).get("strategies", []):
        color = {"healthy": BRAND["gain"], "fresh": BRAND["muted"], "aging": "#D9A441", "stale": BRAND["loss"],
                 "error": BRAND["loss"]}.get(a.get("status"), BRAND["muted"])
        st.markdown(f"**{_e(a['name'])}** {_chip(str(a.get('status')).upper(), color)} &nbsp; {_e(a.get('detail', ''))}",
                    unsafe_allow_html=True)
        if a.get("forward_sessions"):
            _cards([("Days since registration", f"{a['forward_sessions']}"), ("Forward Sharpe", _num(a.get("forward_sharpe")),
                    f"promised {_num(a.get('baseline_sharpe'))}"),
                    ("z vs baseline", _num(a.get("z_vs_baseline")), "below -2 is flagged"),
                    ("Forward return", _pct(a.get("forward_return")), "", _tone(a.get("forward_return")))])
    with st.expander("Retire a strategy"):
        nm = st.selectbox("Strategy", [s["name"] for s in strat], key="lab_retire")
        if st.button("Retire", key="lab_retire_b"):
            L.retire_strategy(nm)
            st.rerun()
    _guide("Monitor")


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
