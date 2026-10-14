"""Strategies side by side.

Every strategy here is run over the same universe, calendar, execution
rules and frictions as the main one, then cut to the window they all share,
so a difference between two columns is a difference in the model and
nothing else. The functions take finished `BacktestResult`s and return
plain tables; the interface only formats them.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from .engine import BacktestResult


def unique_labels(reserved: Iterable[str], labels: Iterable[str]) -> List[str]:
    """`labels`, with any that repeat each other or a reserved name given a
    numeric suffix, so each can key a table column and a chart trace."""
    taken = set(reserved)
    out: List[str] = []
    for lab in labels:
        cand, n = lab, 2
        while cand in taken:
            cand = f"{lab} ({n})"
            n += 1
        taken.add(cand)
        out.append(cand)
    return out


def masked_until(weights: pd.DataFrame, start: pd.Timestamp,
                 execution_lag: int = 1) -> pd.DataFrame:
    """Target weights with everything before `start` set to zero, so that a
    strategy which was already invested earlier buys on `start` instead.

    The engine delays weights by the execution lag, so the rows to clear are
    those more than `lag` sessions before `start`: the target that governs
    `start` itself is left as the model produced it. Used to put strategies
    with different warm-ups on a level footing: each one enters, and pays
    for entering, on the date they share.
    """
    out = weights.copy()
    pos = int(out.index.get_indexer([pd.Timestamp(start)])[0])
    if pos > 0:
        out.iloc[:max(0, pos - max(0, int(execution_lag)))] = 0.0
    return out


def comparison_table(results: Dict[str, BacktestResult],
                     benchmark: Optional[BacktestResult] = None,
                     rf: M.RiskFree = 0.0,
                     ppy: int = M.TRADING_DAYS) -> pd.DataFrame:
    """Full statistics, one column per strategy (and one for the benchmark
    when it is not already among them). Relative statistics (beta, alpha,
    tracking error) are measured against the benchmark; the benchmark's own
    column has none."""
    cols: Dict[str, Dict[str, float]] = {}
    for label, r in results.items():
        is_bench = benchmark is not None and r is benchmark
        cols[label] = M.summary(
            r.returns, r.equity,
            None if benchmark is None or is_bench else benchmark.returns,
            r.turnover, r.exposure, rf, ppy)
    if benchmark is not None and not any(r is benchmark for r in results.values()):
        cols[benchmark.label] = M.summary(
            benchmark.returns, benchmark.equity, None, None, None, rf, ppy)
    order: List[str] = []
    for c in cols.values():
        order += [k for k in c if k not in order]
    return pd.DataFrame(cols).reindex(order)


def head_to_head(base: str, results: Dict[str, BacktestResult],
                 rf: M.RiskFree = 0.0,
                 ppy: int = M.TRADING_DAYS) -> pd.DataFrame:
    """`base` against each of the others: how far ahead it is, how closely
    the two move together, and how often it wins a month.

    Excess return is the CAGR difference (what the reports call alpha);
    beta and the information ratio are of `base` on the other series.
    """
    a = results[base]
    rows = {}
    for label, b in results.items():
        if label == base:
            continue
        rel = M.beta_alpha(a.returns, b.returns, rf, ppy)
        am = (1 + a.returns).resample("ME").prod() - 1 if ppy > 12 else a.returns
        bm = (1 + b.returns).resample("ME").prod() - 1 if ppy > 12 else b.returns
        both = pd.concat([am, bm], axis=1).dropna()
        rows[label] = {
            "Excess CAGR": rel["outperformance"],
            "Tracking error": rel["tracking_error"],
            "Information ratio": rel["information_ratio"],
            "Beta": rel["beta"],
            "Correlation": rel["corr"],
            "Months ahead": (float((both.iloc[:, 0] > both.iloc[:, 1]).mean())
                             if len(both) else np.nan),
        }
    return pd.DataFrame(rows).T


def calendar_year_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Return by calendar year, one column per strategy. A partial first or
    last year is a partial figure for every column alike, since all share
    one window."""
    out = {}
    for label, r in results.items():
        g = (1.0 + r.returns.dropna()).groupby(r.returns.dropna().index.year).prod() - 1.0
        out[label] = g
    df = pd.DataFrame(out)
    df.index.name = "Year"
    return df


def return_correlation(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Correlation of periodic returns between the strategies."""
    df = pd.concat({k: r.returns for k, r in results.items()}, axis=1).dropna()
    return df.corr()
