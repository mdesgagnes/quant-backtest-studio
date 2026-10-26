"""Research dossier: a single self-contained HTML page for one strategy.

Everything a reviewer needs on one page and nothing they have to take on
trust: headline metrics, equity and drawdown against the benchmark, the full
validation table with the evidence and threshold for every check, the
warnings the run card raised, and the hashes that identify the data and the
configuration. Light, print-friendly palette from `qbt.brand.PRINT`.

The page embeds plotly.js by default so it opens offline; pass
`offline=False` to load it from the CDN instead and keep the file small.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import metrics as M
from ..brand import PRINT, SERIES_PRINT
from ..engine import BacktestResult

_STATUS_COLOR = {"PASS": "#1F6F5C", "WARN": "#B0791A", "FAIL": "#B3322B", "NA": "#6B655C"}
_GRADE_COLOR = {"Robust": "#1F6F5C", "Promising": "#4C7F2A", "Fragile": "#B0791A",
                "Unsupported": "#B3322B", "Unrated": "#6B655C"}


def _fig_layout(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=PRINT["text"])), height=height,
                      margin=dict(l=50, r=20, t=44, b=36), paper_bgcolor=PRINT["bg"],
                      plot_bgcolor=PRINT["bg"], font=dict(color=PRINT["text"], size=11),
                      legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
                      xaxis=dict(gridcolor=PRINT["rule_soft"]), yaxis=dict(gridcolor=PRINT["rule_soft"]))
    return fig


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1%}"


def _num(v, d=2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def build_dossier(card: Dict[str, Any], res: BacktestResult, bench: Optional[BacktestResult] = None,
                  extra_sections: Optional[Dict[str, str]] = None, offline: bool = True) -> str:
    m = card["metrics"]
    v = card.get("validation")
    grade = v["grade"] if v else {"label": "Unrated", "passes": 0, "warnings": 0, "fails": 0}

    eq = res.equity / res.equity.iloc[0] * 100
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=eq.index, y=eq, name=card["label"], line=dict(color=SERIES_PRINT[0], width=2)))
    if bench is not None:
        be = bench.equity.reindex(eq.index).ffill()
        be = be / be.dropna().iloc[0] * 100
        fig1.add_trace(go.Scatter(x=be.index, y=be, name="Benchmark", line=dict(color=SERIES_PRINT[3], width=1.5)))
    fig1.update_yaxes(type="log")
    _fig_layout(fig1, "Growth of 100 (log scale)")

    dd = res.equity / res.equity.cummax() - 1
    fig2 = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=PRINT["loss"], width=1),
                                name="Drawdown"))
    fig2.update_yaxes(tickformat=".0%")
    _fig_layout(fig2, "Drawdown", 260)

    figs = [fig1, fig2]
    wf = next((c for c in (v or {}).get("checks", []) if "Consistency" in c["check"]), None)
    div = []
    for i, f in enumerate(figs):
        div.append(f.to_html(full_html=False, include_plotlyjs=("cdn" if not offline else (True if i == 0 else False)),
                             config={"displayModeBar": False}))

    yearly = res.returns.groupby(res.returns.index.year).apply(lambda s: (1 + s).prod() - 1)
    yr_rows = "".join(f"<tr><td>{y}</td><td class='n'>{r:+.1%}</td></tr>" for y, r in yearly.items())

    kpis = [("CAGR", _pct(m.get("CAGR"))), ("Volatility", _pct(m.get("Volatility"))),
            ("Sharpe", _num(m.get("Sharpe"))), ("Max drawdown", _pct(m.get("Max Drawdown"))),
            ("Calmar", _num(m.get("Calmar"))), ("Turnover / yr", _pct(m.get("Annual Turnover")))]
    kpi_html = "".join(f"<div class='kpi'><div class='k'>{k}</div><div class='v'>{x}</div></div>" for k, x in kpis)

    check_rows = ""
    for c in (v or {}).get("checks", []):
        st = c["status"].upper()
        check_rows += (f"<tr><td><b>{html.escape(c['check'])}</b></td>"
                       f"<td><span class='chip' style='background:{_STATUS_COLOR.get(st, '#6B655C')}'>{st}</span></td>"
                       f"<td>{html.escape(c['evidence'])}</td></tr>")
    warn_html = "".join(f"<li>{html.escape(w)}</li>" for w in card["warnings"]) or "<li>none raised</li>"
    r = card["reproducibility"]
    extra = "".join(f"<h2>{html.escape(k)}</h2>{body}" for k, body in (extra_sections or {}).items())
    hyp = f" &middot; hypothesis #{card['hypothesis_id']}" if card.get("hypothesis_id") else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(card['label'])} | research dossier</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--ink:{PRINT['text']};--muted:{PRINT['muted']};--rule:{PRINT['rule']};--panel:{PRINT['panel']};--red:{PRINT['red']}}}
*{{box-sizing:border-box}} body{{font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);
background:#fff;margin:0}} .wrap{{max-width:980px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
border-bottom:1px solid var(--rule);padding-bottom:6px;margin:32px 0 12px}}
.sub{{color:var(--muted)}} .grade{{display:inline-block;color:#fff;border-radius:4px;padding:3px 12px;font-weight:600;
background:{_GRADE_COLOR.get(grade['label'], '#6B655C')};margin-left:10px;font-size:15px;vertical-align:middle}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:18px 0}}
.kpi{{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:10px 12px}}
.kpi .k{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}} .kpi .v{{font-size:21px;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid var(--rule);padding:7px 8px;
text-align:left;vertical-align:top}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.chip{{color:#fff;border-radius:3px;padding:1px 8px;font-size:11px;font-weight:700}}
.warn{{background:#FFF7E6;border-left:3px solid #B0791A;padding:8px 14px;border-radius:3px}} .warn ul{{margin:4px 0;padding-left:18px}}
code{{background:var(--panel);padding:1px 5px;border-radius:3px;font-size:12px}}
.small{{font-size:12px;color:var(--muted)}} .cols{{display:grid;grid-template-columns:2fr 1fr;gap:24px}}
@media(max-width:700px){{.cols{{grid-template-columns:1fr}}}} @media print{{.wrap{{padding:0}}}}
</style></head><body><div class="wrap">
<h1>{html.escape(card['label'])}<span class="grade">{grade['label']}</span></h1>
<div class="sub">{card['backtest']['start']} to {card['backtest']['end']} &middot; {card['backtest']['sessions']:,} sessions &middot;
{html.escape(', '.join(card['backtest']['tickers']))}{hyp}</div>
<div class="kpis">{kpi_html}</div>
{div[0]}{div[1]}
<h2>Validation</h2>
<p class="small">{grade['passes']} passed &middot; {grade['warnings']} warnings &middot; {grade['fails']} failed. The grade summarises
how many independent lines of evidence agree. It is not a probability of profit.</p>
<table><tr><th style="width:24%">Check</th><th style="width:8%">Status</th><th>Evidence</th></tr>{check_rows or '<tr><td colspan=3>Validation was not run.</td></tr>'}</table>
<h2>Warnings</h2><div class="warn"><ul>{warn_html}</ul></div>
<div class="cols"><div><h2>Calendar years</h2><table><tr><th>Year</th><th style="text-align:right">Return</th></tr>{yr_rows}</table></div>
<div><h2>Strategy</h2><p><code>{html.escape(card['strategy']['key'])}</code></p>
<pre class="small" style="white-space:pre-wrap">{html.escape(json.dumps(card['strategy']['params'], indent=1))}</pre></div></div>
{extra}
<h2>Reproducibility</h2>
<table><tr><td>Config hash</td><td><code>{r['config_hash']}</code></td></tr>
<tr><td>Data hash</td><td><code>{r['data_hash']}</code></td></tr>
<tr><td>Data</td><td>{html.escape(str(r['data_source']))}: {html.escape(str(r['data_note']))}</td></tr>
<tr><td>Software</td><td>python {r['python']} &middot; pandas {r['pandas']} &middot; numpy {r['numpy']} &middot; lab {r['qbt_lab']}
{('&middot; commit ' + r['git_commit']) if r['git_commit'] else ''}</td></tr>
<tr><td>Generated</td><td>{card['generated_at']}</td></tr></table>
<p class="small">Research output, not investment advice. Past performance in a backtest does not predict future results.</p>
</div></body></html>"""
