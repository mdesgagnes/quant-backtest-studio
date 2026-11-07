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

from typing import Dict, Optional

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


# ----------------------------------------------------------------------
# Benchmark-relative attribution (CIPM: Brinson-Fachler, linked)
# ----------------------------------------------------------------------
# The contribution analysis above explains the ABSOLUTE return. This
# explains the EXCESS return over a policy benchmark through the two
# decisions an asset allocator takes -- how much to hold in each asset class
# (allocation) and which holdings to use within it (selection, interaction
# included, as no one sets out to earn interaction). Fees and trading costs
# are shown as their own line, never hidden in selection or a residual.
#
# Periods are attributed one by one (monthly by default, the GIPS floor; the
# daily ledger underneath makes the category returns Dietz-style: contribution
# over average weight) and linked with Frongello,
# Carino, Menchero or GRAP so that the effects add up to R - B over the whole
# span. The reconciliation error is returned: anything but ~0 is a bug.
COSTS_FEES = "Costs & fees"
CASH_CLASS = "Cash balance"
EQUAL_WEIGHT = "Equal-weight class"


def _proxy_returns(asset_returns: pd.DataFrame, classes: Dict[str, str],
                   proxies: Optional[Dict[str, str]]) -> pd.DataFrame:
    """Daily benchmark return of each asset class: a chosen instrument, or
    the equal-weight average of the class's instruments."""
    proxies = proxies or {}
    cols: Dict[str, pd.Series] = {}
    members: Dict[str, list] = {}
    for a in asset_returns.columns:
        members.setdefault(classes.get(a) or "Unclassified", []).append(a)
    for cls, assets in members.items():
        pick = proxies.get(cls, EQUAL_WEIGHT)
        if pick in asset_returns.columns:
            cols[cls] = asset_returns[pick]
        else:
            cols[cls] = asset_returns[assets].mean(axis=1)
    return pd.DataFrame(cols)


def benchmark_relative(res: BacktestResult, classes: Dict[str, str],
                       bench_weights: Dict[str, float],
                       asset_returns: pd.DataFrame, cash_returns: pd.Series,
                       proxies: Optional[Dict[str, str]] = None,
                       period: str = "Month", linking: str = "frongello") -> Dict[str, object]:
    from . import cipm

    if res.contributions is None:
        raise ValueError("this result has no contribution ledger; run the backtest again")
    rule = PERIODS[period]
    bw = {k: float(v) for k, v in bench_weights.items() if abs(float(v)) > 0}
    if not np.isclose(sum(bw.values()), 1.0, atol=1e-6):
        raise ValueError(f"benchmark weights must sum to 100% (cash is its own category), "
                         f"got {sum(bw.values()) * 100:.2f}%")

    bp = by_period(res, period, classes)                       # contribution / period-start value
    cls_cols = [c for c in bp.columns if c not in NON_ASSET_LINES and c != "Total"]
    # Portfolio weights for the period: the average of the start-of-day
    # weights (yesterday's close) over its days, cash as the remainder. An
    # average rather than the opening weight, so that a class bought
    # part-way through the month keeps its contribution (Dietz-style:
    # category return = contribution / average weight), and the category
    # contributions add up to the period return exactly.
    w_daily = res.weights.T.groupby({a: (classes.get(a) or "Unclassified")
                                     for a in res.weights.columns}, sort=False).sum().T
    start = w_daily.shift(1).fillna(0.0).resample(rule).mean().reindex(bp.index).fillna(0.0)
    start[CASH_CLASS] = 1.0 - start.sum(axis=1)

    idx = res.equity.index
    rets = asset_returns.reindex(idx).fillna(0.0)
    cls_ret = _proxy_returns(rets, classes, proxies)
    cls_ret[CASH_CLASS] = cash_returns.reindex(idx).fillna(0.0)

    def comp(s: pd.Series) -> pd.Series:
        return (1.0 + s).resample(rule).prod().sub(1.0).reindex(bp.index)

    b_p = pd.DataFrame({c: comp(cls_ret[c]) for c in cls_ret.columns})

    cats = [c for c in dict.fromkeys(list(cls_cols) + [CASH_CLASS] + list(bw))]
    for c in cats:
        if c not in start.columns:
            start[c] = 0.0
        if c not in b_p.columns:
            b_p[c] = np.nan
    cats = sorted(cats, key=lambda c: (c == CASH_CLASS, c))

    per_rows, E, rs, bs = [], [], [], []
    for t in bp.index:
        w = start.loc[t, cats].to_numpy(float)
        W = np.array([bw.get(c, 0.0) for c in cats])
        b = b_p.loc[t, cats].to_numpy(float)
        contrib = np.array([bp.loc[t, CASH_LINE] if (c == CASH_CLASS and CASH_LINE in bp.columns)
                            else (bp.loc[t, c] if c in bp.columns else 0.0) for c in cats])
        r = np.where(np.abs(w) > 1e-9, contrib / np.where(np.abs(w) > 1e-9, w, 1.0), b)
        used = (np.abs(w) > 1e-9) | (np.abs(W) > 0)
        bad = [c for c, u, x in zip(cats, used, b) if u and not np.isfinite(x)]
        if bad:
            raise ValueError(f"no benchmark return for {', '.join(bad)}: give each category a proxy "
                             f"that matches how the decision is made, never an implicit 0%")
        b = np.where(np.isfinite(b), b, 0.0)
        bf = cipm.brinson_fachler(w, W, r, b)
        drag = float(sum(bp.loc[t, c] for c in (COST_LINE, FEE_LINE) if c in bp.columns))
        gross = float(w @ r)
        total_r = float(bp.loc[t, "Total"])
        if abs(gross + drag - total_r) > 1e-9:               # the ledger must close
            raise ValueError(f"contributions do not add up to the period return in {t.date()} "
                             f"({gross + drag:.8f} vs {total_r:.8f})")
        rs.append(total_r)
        bs.append(float(W @ b))
        E.append(np.vstack([np.column_stack([bf["allocation"], bf["selection"]]), [0.0, drag]]))
        per_rows.append({"Period": t, "Portfolio": total_r, "Benchmark": float(W @ b),
                         "Excess": total_r - float(W @ b),
                         "Allocation": float(bf["allocation"].sum()),
                         "Selection": float(bf["selection"].sum()), COSTS_FEES: drag})
    if not E:
        raise ValueError("no complete period to attribute")

    adj = cipm.link_arithmetic(E, rs, bs, linking) if len(E) > 1 else E
    total = np.sum(adj, axis=0)
    linked = pd.DataFrame(total, index=cats + [COSTS_FEES], columns=["Allocation", "Selection"])
    linked["Total effect"] = linked.sum(axis=1)
    # Costs and fees are neither allocation nor selection: their effect sits
    # in Total effect only.
    linked.loc[COSTS_FEES, ["Allocation", "Selection"]] = 0.0
    w_avg = start[cats].mean()
    linked.insert(0, "Avg portfolio weight", list(w_avg) + [np.nan])
    linked.insert(1, "Benchmark weight", [bw.get(c, 0.0) for c in cats] + [np.nan])
    linked.loc["Total"] = linked.sum(numeric_only=True)
    linked.loc["Total", ["Avg portfolio weight", "Benchmark weight"]] = [w_avg.sum(), sum(bw.values())]
    R, B = cipm.link(rs), cipm.link(bs)
    err = float(linked.loc["Total", "Total effect"] - (R - B))

    # Each calendar year on its own, linked within the year, so that every
    # year reconciles to its own excess return.
    plist = list(bp.index)
    yrows = []
    for y in sorted({t.year for t in plist}):
        ii = [i for i, t in enumerate(plist) if t.year == y]
        Ey, ry, by = [E[i] for i in ii], [rs[i] for i in ii], [bs[i] for i in ii]
        ay = cipm.link_arithmetic(Ey, ry, by, linking) if len(Ey) > 1 else Ey
        ty = np.sum(ay, axis=0)
        Ry, By = cipm.link(ry), cipm.link(by)
        alloc, sel, cost = float(ty[:-1, 0].sum()), float(ty[:-1, 1].sum()), float(ty[-1, 1])
        yrows.append({"Year": y, "Periods": len(ii), "Portfolio": Ry, "Benchmark": By,
                      "Excess": Ry - By, "Allocation": alloc, "Selection": sel,
                      COSTS_FEES: cost, "Check": alloc + sel + cost - (Ry - By)})
    return {"linked": linked, "periods": pd.DataFrame(per_rows).set_index("Period"),
            "by_year": pd.DataFrame(yrows),
            "R": R, "B": B, "excess": R - B, "reconciliation_error": err,
            "linking": linking, "period": period,
            "methodology": (f"Brinson-Fachler (interaction in selection), {period.lower()} periods "
                            f"from the daily ledger, linked with {linking.capitalize()}; costs and "
                            f"fees shown apart; benchmark = fixed class weights x class proxies, "
                            f"rebalanced each period")}
