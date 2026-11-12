"""What every Backtest screen shares: the run's statistics and the block
vocabulary screens are written in.

A screen is a list of blocks. A block is a plain dict with a `kind`:

    note     html                      explanatory text (may contain <b>, <code>)
    flag     html                      a warning attached to what it concerns
    heading  text
    kpi      label, value, sub, tone   tone: "pos" | "neg" | ""
    table    frame, signed, emphasise  a DataFrame exactly as the app shows it
    chart    chart, title, data, ...   series to draw; `legacy` rebuilds the
                                       Plotly figure of the Streamlit app, for
                                       the parity tests only
    panel    label, open, blocks       a collapsible group
    caption  text

Blocks hold Python objects (DataFrames, Series). `server.serialize` turns
them into JSON for the browser. Formatting happens here, in Python, with the
same expressions `app.py` uses, so the browser never formats a figure.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qbt import metrics as M
from qbt import nextrade as NR
from qbt.config import REBALANCE_RULES, RunConfig
from qbt.engine import _cash_returns

Block = Dict[str, Any]


class RunView:
    """The figures app.py computes once after a run (app.py:3814-3838)."""

    def __init__(self, run: Dict[str, Any]):
        self.run = run
        self.res = run["result"]
        self.bench = run["bench"]
        self.universe = run["prices"]
        self.cfg: RunConfig = run["cfg"]
        self.ppy = self.cfg.engine.periods_per_year
        self.sleeves = run.get("sleeves")
        res, bench, cfg = self.res, self.bench, self.cfg
        self.bench_r = bench.returns if bench is not None else None
        # The risk-free rate is what the portfolio's own cash earned.
        self.rf = _cash_returns(res.equity.index, run.get("cash"),
                                cfg.costs.cash_rate_pa, self.ppy)
        self.rf_label = (f"cash proxy {cfg.data.cash_proxy}" if cfg.data.cash_proxy
                         else f"{cfg.costs.cash_rate_pa * 100:.2f}% a year (cash rate)")
        self.stats = M.summary(res.returns, res.equity, self.bench_r, res.turnover,
                               res.exposure, self.rf, self.ppy)
        self.bstats = (M.summary(bench.returns, bench.equity, None, None, None,
                                 self.rf, self.ppy) if bench is not None else {})
        self._next = None

    def next_rebalance(self):
        """(result, error text), computed once per run."""
        if self._next is None:
            nx, err = None, ""
            try:
                run, c = self.run, self.cfg
                nx = NR.next_rebalance(
                    run["prices"], run["weights"], c.engine, c.costs, run["cash"],
                    run.get("rebalance_dates"), run.get("open"), run.get("dividends"),
                    run.get("volume"), run.get("fx"), reference=run["result"])
            except Exception as exc:
                err = str(exc)
            self._next = (nx, err)
        return self._next

    def context(self) -> List[str]:
        """The strip that says which run is on screen (app.py:3840)."""
        res, cfg = self.res, self.cfg
        n = len(self.universe.columns)
        bits = [cfg.label,
                f"{res.equity.index[0].date()} → {res.equity.index[-1].date()}",
                f"{n} instrument{'s' if n != 1 else ''}",
                REBALANCE_RULES.get(cfg.engine.rebalance, "").lower()]
        if self.bench is not None:
            bits.append(f"vs {self.bench.label}")
        cagr = self.stats.get("CAGR", float("nan"))
        if cagr == cagr:
            bits.append(f"CAGR {M.format_metric('CAGR', cagr)}")
        if self.run.get("trimmed"):
            bits.append("warm-up trimmed")
        return bits


# ----------------------------------------------------------------------
# Block constructors
# ----------------------------------------------------------------------
def note(html: str) -> Block:
    return {"kind": "note", "html": html}


def flag(html: str) -> Block:
    return {"kind": "flag", "html": html}


def heading(text: str) -> Block:
    return {"kind": "heading", "text": text}


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> Block:
    return {"kind": "kpi", "label": label, "value": value, "sub": sub, "tone": tone}


def table(frame: Optional[pd.DataFrame], signed: Optional[List[str]] = None,
          emphasise: Optional[List[str]] = None, hide_index: bool = True,
          key: str = "") -> Block:
    """`signed`: columns coloured by sign. None colours nothing; an empty
    frame is passed through as the app does."""
    cols = list(frame.columns) if frame is not None else []
    return {"kind": "table", "frame": frame, "key": key,
            "signed": [c for c in (signed or []) if c in cols],
            "emphasise": [c for c in (emphasise or []) if c in cols],
            "hide_index": hide_index}


def chart(kind: str, title: str, data: Dict[str, Any], legacy: tuple, **options) -> Block:
    return {"kind": "chart", "chart": kind, "title": title, "data": data,
            "legacy": legacy, **options}


def panel(label: str, blocks: List[Block], open: bool = False) -> Block:
    return {"kind": "panel", "label": label, "open": open, "blocks": blocks}


def caption(text: str) -> Block:
    return {"kind": "caption", "text": text}


def lines(frame: pd.DataFrame) -> Dict[str, Any]:
    """Time series for the browser: one date axis, one array per column."""
    return {"x": frame.index, "series": [{"name": str(c), "y": frame[c]}
                                         for c in frame.columns]}


def tone_of(value: float, good_positive: bool = True) -> str:
    if not good_positive:
        return ""
    return "pos" if (value == value and value > 0) else "neg"
