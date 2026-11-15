"""Portfolio construction: rebalanced returns, Add / Replace weights.

A portfolio is a dict of ticker -> target weight. Its monthly return is
simulated with drift: weights move with each holding's return during the
month and are reset to target at each rebalance date (end of the month for
"monthly", end of March / June / September / December for "quarterly",
end of December for "annual", never for "none").
"""
from __future__ import annotations

from typing import Dict, Optional

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
