"""Return contribution: where the performance came from.

The engine records, for every day, how much each instrument added to the
portfolio's return (its price move plus dividends, on the units actually
held), alongside cash and financing, trading costs and the management fee.
Each day's contributions sum to that day's net return exactly.

**Linking days into periods.** Daily contributions cannot simply be added
across days: returns compound, so the sum of daily returns is not the
period return. Nor can each line be compounded on its own: a position's
compounded "return" ignores how large the rest of the book was. The linking
used here is the dollar one. Each day's contribution is turned back into
currency (multiplied by the previous day's value), summed over the period,
and divided by the value at the start of the period:

    contribution_i(period) = sum over days of P&L_i / value at period start

Because every day's P&L lines add up to that day's change in value, the
period contributions add up to the period return exactly, for any period:
a month, a year, or the whole backtest. There is no residual to hide.

**Contribution to risk** uses the same ledger: an instrument's share of the
variance of daily returns is cov(its daily contribution, portfolio return)
divided by the variance of the portfolio return. The shares add up to 100%
exactly; a negative share is a diversifier.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult, CASH_LINE, COST_LINE, FEE_LINE

NON_ASSET_LINES = [CASH_LINE, COST_LINE, FEE_LINE]

PERIODS = {"Month": "ME", "Quarter": "QE", "Year": "YE"}


def _pnl(res: BacktestResult) -> pd.DataFrame:
    """Contributions back in currency, relative to a base of 1 at the start.

    Scaled so the first day's opening value is 1: that makes cumulative
    sums read directly as a share of starting capital.
    """
    c = res.contributions
    eq = res.equity
    prev = eq.shift(1)
    prev.iloc[0] = eq.iloc[0] / (1.0 + res.returns.iloc[0])
    return c.mul(prev / prev.iloc[0], axis=0)


def group_columns(frame: pd.DataFrame, classes: Optional[Dict[str, str]] = None,
                  keep_lines: bool = True) -> pd.DataFrame:
    """Sums instrument columns into their asset class. Cash, cost and fee
    lines are kept as they are (or dropped with `keep_lines=False`)."""
    assets = [c for c in frame.columns if c not in NON_ASSET_LINES]
    lines = [c for c in frame.columns if c in NON_ASSET_LINES]
    if classes is None:
        out = frame[assets].copy()
    else:
        mapping = {a: (classes.get(a) or "Unclassified") for a in assets}
        out = frame[assets].T.groupby(mapping, sort=False).sum().T
    if keep_lines:
        for c in lines:
            out[c] = frame[c]
    return out


def cumulative(res: BacktestResult, classes: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Running contribution since the start, as a share of starting capital.
    The columns add up to the portfolio's cumulative return on every date."""
    return group_columns(_pnl(res), classes).cumsum()


def by_period(res: BacktestResult, period: str = "Year",
              classes: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Contribution per calendar period; each row adds up to that period's
    return, which is included as the "Total" column."""
    pnl = group_columns(_pnl(res), classes)
    eq = res.equity / (res.equity.iloc[0] / (1.0 + res.returns.iloc[0]))
    start_val = eq.shift(1)
    start_val.iloc[0] = 1.0
    rule = PERIODS[period]
    summed = pnl.resample(rule).sum()
    base = start_val.resample(rule).first()
    out = summed.div(base, axis=0)
    out["Total"] = out.sum(axis=1)
    out = out[base.notna()]
    return out


def summary(res: BacktestResult, classes: Optional[Dict[str, str]] = None,
            ppy: int = 252) -> pd.DataFrame:
    """One row per instrument (or class): average weight, total
    contribution, share of the result, and share of the risk."""
    pnl = group_columns(_pnl(res), classes)
    total = pnl.sum()
    r = res.returns
    daily = group_columns(res.contributions, classes)
    var = float(r.var(ddof=1))
    risk = (daily.apply(lambda s: s.cov(r)) / var) if var > 0 else daily.iloc[0] * np.nan

    w = res.weights
    if classes is not None:
        w = w.T.groupby({a: (classes.get(a) or "Unclassified") for a in w.columns},
                        sort=False).sum().T
    avg_w = w.mean()
    held = (w.abs() > 1e-9).mean()

    port_total = float(res.equity.iloc[-1] / (res.equity.iloc[0] / (1.0 + r.iloc[0])) - 1.0)
    rows = []
    for name in pnl.columns:
        rows.append({
            "Name": name,
            "Average weight": float(avg_w.get(name, np.nan)),
            "Time held": float(held.get(name, np.nan)),
            "Contribution": float(total[name]),
            "Share of result": (float(total[name]) / port_total
                                if abs(port_total) > 1e-12 else np.nan),
            "Share of risk": float(risk.get(name, np.nan)),
        })
    df = pd.DataFrame(rows)
    return df


def total_return(res: BacktestResult) -> float:
    r = res.returns
    return float(res.equity.iloc[-1] / (res.equity.iloc[0] / (1.0 + r.iloc[0])) - 1.0)
