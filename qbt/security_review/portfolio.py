"""Portfolio construction: rebalanced returns, Add / Replace weights.

A portfolio is a dict of ticker -> target weight. Its monthly return is
simulated with drift: weights move with each holding's return during the
month and are reset to target at each rebalance date (end of the month for
"monthly", end of March / June / September / December for "quarterly",
end of December for "annual", never for "none").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REBALANCE_MONTHS = {"monthly": set(range(1, 13)), "quarterly": {3, 6, 9, 12},
                    "annual": {12}, "none": set()}


def portfolio_returns(rets: pd.DataFrame, weights: Dict[str, float],
                      rebalance: str = "quarterly") -> pd.Series:
    """Monthly portfolio returns from holdings' returns.

    Uses only rows where every holding has a return; raises if a holding is
    missing from `rets`. Weights are normalised to sum to 1."""
    if rebalance not in REBALANCE_MONTHS:
        raise ValueError(f"Unknown rebalancing rule: {rebalance}")
    w = {t: float(x) for t, x in weights.items() if abs(float(x)) > 0}
    missing = [t for t in w if t not in rets.columns]
    if missing:
        raise KeyError(f"No returns for {missing}")
    cols = list(w)
    tot = sum(w.values())
    target = np.array([w[c] / tot for c in cols])
    df = rets[cols].dropna()
    if df.empty:
        return pd.Series(dtype=float)
    R = df.to_numpy(dtype=float)
    months = REBALANCE_MONTHS[rebalance]
    cur = target.copy()
    out = np.empty(len(R))
    for i, (r, ts) in enumerate(zip(R, df.index)):
        rp = float(cur @ r)
        out[i] = rp
        grown = cur * (1.0 + r)
        cur = grown / grown.sum() if grown.sum() != 0 else target.copy()
        if ts.month in months:
            cur = target.copy()
    return pd.Series(out, index=df.index, name="portfolio")


def add_weights(base: Dict[str, float], candidate: str, weight: float) -> Dict[str, float]:
    """Add `candidate` at `weight`, funded pro rata from every holding."""
    if not 0 < weight < 1:
        raise ValueError("The add weight must be between 0 and 1.")
    out = {t: w * (1.0 - weight) for t, w in base.items()}
    out[candidate] = out.get(candidate, 0.0) + weight
    return out


def fund_weights(base: Dict[str, float], candidate: str, weight: float,
                 sources: List[str]) -> Dict[str, float]:
    """Add `candidate` at `weight` (a share of the whole portfolio), taken
    from `sources` only, pro rata to their weights; every other holding keeps
    its weight. Raises if the sources hold less than `weight`."""
    if not 0 < weight < 1:
        raise ValueError("The add weight must be between 0 and 1.")
    src = [t for t in dict.fromkeys(sources) if base.get(t, 0) > 0]
    if not src:
        raise KeyError("None of the holdings to reduce is in this portfolio.")
    avail = sum(base[t] for t in src)
    if weight > avail + 1e-12:
        raise ValueError(f"Only {avail:.0%} is held in {', '.join(src)}; cannot take {weight:.0%}.")
    out = {t: w * (1.0 - weight / avail) if t in src else w for t, w in base.items()}
    out = {t: w for t, w in out.items() if w > 1e-12}
    out[candidate] = out.get(candidate, 0.0) + weight
    return out


def split_fund_weights(base: Dict[str, float], candidate: str, weight: float,
                       parts: List[Tuple[List[str], float]]) -> Dict[str, float]:
    """Add `candidate` at `weight`, paid for by several parts: each part is
    (holdings, share of the add weight) and takes share x weight from its
    holdings, pro rata to their weights. Shares must sum to 1. With 6% gold
    paid 50 / 50 by bonds and equities, bonds give up 3 points and equities
    3 points, whatever the portfolio's own split."""
    if not 0 < weight < 1:
        raise ValueError("The add weight must be between 0 and 1.")
    if abs(sum(s for _, s in parts) - 1.0) > 1e-6:
        raise ValueError("The funding shares must add up to 100%.")
    cut: Dict[str, float] = {}
    for holdings, share in parts:
        src = [t for t in dict.fromkeys(holdings) if base.get(t, 0) > 0]
        amount = share * weight
        if amount <= 0:
            continue
        if not src:
            raise KeyError("A funding part holds nothing in this portfolio.")
        avail = sum(base[t] for t in src)
        if amount > avail + 1e-12:
            raise ValueError(f"Only {avail:.1%} is held in {', '.join(src)}; cannot take {amount:.1%}.")
        for t in src:
            cut[t] = cut.get(t, 0.0) + amount * base[t] / avail
    over = [t for t, x in cut.items() if x > base[t] + 1e-12]
    if over:
        raise ValueError(f"The funding takes more than is held in {', '.join(over)}.")
    out = {t: w - cut.get(t, 0.0) for t, w in base.items()}
    out = {t: w for t, w in out.items() if w > 1e-12}
    out[candidate] = out.get(candidate, 0.0) + weight
    return out


def replace_weights(base: Dict[str, float], holding: str, candidate: str) -> Dict[str, float]:
    """Give `holding`'s whole weight to `candidate`."""
    if holding not in base:
        raise KeyError(f"{holding} is not a holding of this profile.")
    out = {t: w for t, w in base.items() if t != holding}
    out[candidate] = out.get(candidate, 0.0) + base[holding]
    return out


def buy_and_hold_path(daily_px: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Value of a portfolio bought at the first date at target weights and
    left to drift (used inside short stress windows). Prices are levels on a
    common calendar; the first row is the start."""
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    rel = daily_px[w.index] / daily_px[w.index].iloc[0]
    return (rel * w).sum(axis=1)


def diversification_ratio(rets: pd.DataFrame, weights: Dict[str, float]) -> float:
    """Weighted average of holdings' volatilities over the portfolio's
    volatility (constant target weights). 1 = no diversification."""
    cols = [t for t, w in weights.items() if w]
    df = rets[cols].dropna()
    if len(df) < 3:
        return np.nan
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    cov = np.cov(df.to_numpy().T, ddof=0)
    cov = np.atleast_2d(cov)
    sd = np.sqrt(np.diag(cov))
    port = float(np.sqrt(w @ cov @ w))
    return float(w @ sd / port) if port > 0 else np.nan
