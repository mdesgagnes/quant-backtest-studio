"""Tearsheet report.

Assembles a single, self-contained HTML document: KPI grid, equity curve,
drawdown, monthly returns, return distribution, full stats table, top
drawdown episodes, current holdings, and the engine assumptions that produced
the numbers. Built for printing or sharing with a portfolio manager, so it
uses a light "print" theme distinct from the dark in-app interface.

No extra dependency: Plotly renders as interactive HTML via a single CDN
script tag, and everything else is plain HTML/CSS. Opening the file in a
browser and using Print -> Save as PDF produces a clean PDF without any
server-side rendering step.
"""
from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from . import charts as C
from . import metrics as M
from .config import RunConfig, REBALANCE_RULES
from .engine import BacktestResult

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Condensed:wght@600;700&display=swap');
  :root{
    --ink:#1B2430; --muted:#6B7684; --rule:#DDE2E8; --panel:#F7F8FA;
    --brass:#A9791E; --teal:#2E7A6E; --rust:#A2432F;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; padding:2.2rem 2.6rem 3rem; background:#fff; color:var(--ink);
    font-family:'IBM Plex Sans',Arial,sans-serif; font-size:13px; line-height:1.5;
    max-width:1080px; margin-left:auto; margin-right:auto;
  }
  .masthead{border-bottom:2px solid var(--ink); padding-bottom:.9rem; margin-bottom:1.4rem;
    display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:.5rem;}
  .masthead h1{font-family:'IBM Plex Sans Condensed',sans-serif; font-weight:700;
    font-size:1.5rem; letter-spacing:-.01em; margin:0;}
  .masthead .meta{font-family:'IBM Plex Mono',monospace; font-size:.72rem;
    color:var(--muted); text-align:right; line-height:1.6;}
  .eyebrow{font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--muted); margin:1.6rem 0 .6rem;
    border-top:1px solid var(--rule); padding-top:.9rem;}
  .eyebrow:first-of-type{border-top:none; margin-top:0;}
  .kpi-grid{display:grid; grid-template-columns:repeat(6,1fr); gap:.6rem;}
  .kpi{background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--brass);
    padding:.55rem .65rem;}
  .kpi .k{font-family:'IBM Plex Mono',monospace; font-size:.6rem; letter-spacing:.08em;
    text-transform:uppercase; color:var(--muted); display:block; margin-bottom:.25rem;}
  .kpi .v{font-family:'IBM Plex Mono',monospace; font-size:1.05rem; font-weight:600;}
  .kpi .b{font-family:'IBM Plex Mono',monospace; font-size:.62rem; color:var(--muted); margin-top:.15rem;}
  table{border-collapse:collapse; width:100%; font-size:.78rem;}
  th,td{border-bottom:1px solid var(--rule); padding:.32rem .5rem; text-align:right;}
  th:first-child,td:first-child{text-align:left;}
  th{font-family:'IBM Plex Mono',monospace; font-size:.62rem; letter-spacing:.06em;
    text-transform:uppercase; color:var(--muted); font-weight:600;}
  td{font-family:'IBM Plex Mono',monospace;}
  .note{color:var(--muted); font-size:.78rem; margin:.3rem 0 .8rem;}
  .two-col{display:grid; grid-template-columns:1fr 1fr; gap:1.4rem;}
  .chart{border:1px solid var(--rule); padding:.4rem; margin-bottom:.9rem;}
  .footer{margin-top:2rem; padding-top:1rem; border-top:1px solid var(--rule);
    font-size:.72rem; color:var(--muted);}
  .assumptions{font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:var(--muted);
    line-height:1.8;}
  @media print{
    body{padding:0 .3in;}
    .chart{break-inside:avoid;}
    .kpi-grid{break-inside:avoid;}
    a{color:inherit; text-decoration:none;}
  }
</style>
"""


def _esc(x: Any) -> str:
    return html.escape(str(x)) if x is not None else ""


def _kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><span class="k">{_esc(label)}</span>'
            f'<span class="v">{_esc(value)}</span>'
            f'{f"<div class=b>{_esc(sub)}</div>" if sub else ""}</div>')


def _stats_table(stats: Dict[str, float], bench_stats: Optional[Dict[str, float]],
                 strategy_label: str, bench_label: str) -> str:
    keys = list(stats.keys())
    head = f"<tr><th>Metric</th><th>{_esc(strategy_label)}</th>"
    if bench_stats:
        head += f"<th>{_esc(bench_label)}</th>"
    head += "</tr>"
    rows = []
    for k in keys:
        row = f"<tr><td>{_esc(k)}</td><td>{_esc(M.format_metric(k, stats[k]))}</td>"
        if bench_stats:
            row += f"<td>{_esc(M.format_metric(k, bench_stats.get(k, float('nan'))))}</td>"
        row += "</tr>"
        rows.append(row)
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _drawdown_table_html(dd: pd.DataFrame) -> str:
    if dd.empty:
        return '<div class="note">No drawdown episode over the period.</div>'
    dd = dd.copy()
    dd["Drawdown"] = dd["Drawdown"].map(lambda v: f"{v*100:.2f}%")
    head = "".join(f"<th>{_esc(c)}</th>" for c in dd.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in dd.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _holdings_table_html(res: BacktestResult, top: int = 30) -> str:
    last = res.weights.iloc[-1].sort_values(ascending=False)
    last = last[last.abs() > 1e-6].head(top)
    val = res.equity.iloc[-1]
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{v*100:.2f}%</td><td>{v*val:,.0f}</td></tr>"
        for k, v in last.items()
    )
    cash = res.cash_weight.iloc[-1]
    rows += (f"<tr><td>Cash</td><td>{cash*100:.2f}%</td>"
            f"<td>{cash*val:,.0f}</td></tr>")
    return (
        "<table><thead><tr><th>Instrument</th><th>Weight</th>"
        f"<th>Value</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "displayModeBar": False})


def render_tearsheet(res: BacktestResult,
                     bench: Optional[BacktestResult],
                     stats: Dict[str, float],
                     bench_stats: Optional[Dict[str, float]],
                     cfg: RunConfig,
                     currency: str = "$") -> str:
    """Builds the full tearsheet as a self-contained HTML string."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    period = f"{res.equity.index[0].date()} to {res.equity.index[-1].date()}"
    bench_label = bench.label if bench is not None else None

    curves = {res.label: res.equity}
    if bench is not None:
        curves[bench.label] = bench.equity
    rebased = pd.DataFrame({k: (v / v.iloc[0] * 100) for k, v in curves.items()})

    eq_fig = C.equity_curve(rebased, True, "Portfolio value (base 100)", theme="print")
    dd_fig = C.underwater(
        {res.label: res.equity, **({bench.label: bench.equity} if bench is not None else {})},
        theme="print")
    mh_fig = C.monthly_heatmap(res.returns, theme="print")
    rd_fig = C.return_distribution(res.returns, theme="print")
    wa_fig = C.weights_area(res.weights, res.cash_weight,
                            "Portfolio composition", theme="print")

    dd_table = M.drawdown_table(res.equity, 6)
    kpi_keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown"]
    kpis = "".join(
        _kpi(k, M.format_metric(k, stats.get(k, float("nan"))),
             f"benchmark {M.format_metric(k, bench_stats.get(k, float('nan')))}"
             if bench_stats else "")
        for k in kpi_keys
    )

    assumptions = (
        f"Rebalance: {REBALANCE_RULES.get(cfg.engine.rebalance, cfg.engine.rebalance)} &middot; "
        f"Execution lag: {cfg.engine.execution_lag} session(s) &middot; "
        f"Costs: {cfg.costs.commission_bps + cfg.costs.slippage_bps:.0f} bps round-trip &middot; "
        f"Initial capital: {currency}{cfg.engine.initial_capital:,.0f} &middot; "
        f"Max leverage: {cfg.engine.max_leverage:.2f}x"
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tearsheet - {_esc(res.label)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{CSS}
</head>
<body>

<div class="masthead">
  <div>
    <h1>{_esc(res.label)}</h1>
    <div class="note" style="margin-top:.3rem;">Backtest tearsheet{f" &middot; benchmark: {_esc(bench_label)}" if bench_label else ""}</div>
  </div>
  <div class="meta">
    Generated {_esc(generated)}<br>
    Period {_esc(period)}
  </div>
</div>

<div class="eyebrow">Key statistics</div>
<div class="kpi-grid">{kpis}</div>

<div class="eyebrow">Portfolio value</div>
<div class="chart">{_fig_html(eq_fig)}</div>

<div class="eyebrow">Drawdown from prior peak</div>
<div class="chart">{_fig_html(dd_fig)}</div>

<div class="two-col">
  <div>
    <div class="eyebrow" style="border-top:none; margin-top:0;">Monthly returns</div>
    <div class="chart">{_fig_html(mh_fig)}</div>
  </div>
  <div>
    <div class="eyebrow" style="border-top:none; margin-top:0;">Daily return distribution</div>
    <div class="chart">{_fig_html(rd_fig)}</div>
  </div>
</div>

<div class="eyebrow">Portfolio composition</div>
<div class="chart">{_fig_html(wa_fig)}</div>

<div class="two-col">
  <div>
    <div class="eyebrow" style="border-top:none; margin-top:0;">Full statistics</div>
    {_stats_table(stats, bench_stats, res.label, bench_label or "Benchmark")}
  </div>
  <div>
    <div class="eyebrow" style="border-top:none; margin-top:0;">Top drawdown episodes</div>
    {_drawdown_table_html(dd_table)}
  </div>
</div>

<div class="eyebrow">Current holdings (as of {_esc(res.equity.index[-1].date())})</div>
{_holdings_table_html(res)}

<div class="footer">
  <div class="assumptions">{assumptions}</div>
  <div style="margin-top:.6rem;">
    Hypothetical backtested results. Past performance, simulated or actual,
    is not indicative of future results. This document is for internal
    analysis and does not constitute investment advice.
  </div>
</div>

</body>
</html>"""
    return html_doc
