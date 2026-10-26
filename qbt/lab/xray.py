"""Risk x-ray: what a set of weights is actually made of.

A pure function of prices and weights. Given a basket (long-only or long/short)
it reports concentration, volatility, drawdown, tail risk, and -- the part a
weight table hides -- where the *risk* sits: each holding's contribution to
portfolio volatility, the effective number of independent bets, and the
diversification ratio.

Nothing here looks forward: every statistic uses the sample passed in, so
feed it a trailing window when the question is "what am I exposed to now".
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def risk_xray(prices: pd.DataFrame, weights: Dict[str, float], ppy: int = TRADING_DAYS,
              var_levels=(0.95, 0.99), min_history: int = 60,
              window: Optional[int] = None) -> Dict[str, object]:
    """Risk report for a static weight vector.

    `weights` need not sum to one (the remainder is cash earning nothing); a
    negative weight is a short. `window` restricts to the last N sessions.
    """
    if not weights:
        raise ValueError("No weights given.")
    unknown = [k for k in weights if k not in prices.columns]
    if unknown:
        raise ValueError(f"Not in the price panel: {unknown}")
    w = pd.Series({k: float(v) for k, v in weights.items()})
    if not np.isfinite(w).all():
        raise ValueError("Weights must be finite numbers.")

    px = prices[list(w.index)]
    if window:
        px = px.iloc[-int(window):]
    notes = []
    thin = [c for c in px.columns if px[c].notna().sum() < min_history]
    if thin:
        notes.append(f"Dropped for too little history: {', '.join(thin)}.")
        px = px.drop(columns=thin)
        w = w.drop(index=thin)
    if w.empty or float(w.abs().sum()) == 0:
        raise ValueError("Nothing left to analyse.")

    rets = px.pct_change(fill_method=None).dropna(how="any")
    if len(rets) < min_history:
        raise ValueError(f"Only {len(rets)} common sessions; need {min_history}.")
    w = w[rets.columns]

    port = rets.mul(w, axis=1).sum(axis=1)
    cov = rets.cov().to_numpy() * ppy
    wv = w.to_numpy()
    var_p = float(wv @ cov @ wv)
    vol_p = math.sqrt(max(var_p, 0.0))

    # Risk contributions (Euler): share of portfolio variance from each name.
    marginal = cov @ wv
    contrib = wv * marginal
    pct_contrib = contrib / var_p if var_p > 0 else np.full_like(contrib, np.nan)
    asset_vol = np.sqrt(np.diag(cov))

    corr = rets.corr().to_numpy()

    gross = float(w.abs().sum())
    hhi = float(((w.abs() / gross) ** 2).sum())
    eq = (1 + port).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()

    tails = {}
    for lv in var_levels:
        q = float(np.quantile(port, 1 - lv))
        tail = port[port <= q]
        tails[f"{int(round(lv * 100))}"] = {
            "var": _finite(-q), "es": _finite(-tail.mean()) if len(tail) else None}
    # Cornish-Fisher adjusts the normal VaR for skew and kurtosis
    z = 1.6448536269514722
    sk, ku = float(port.skew()), float(port.kurt())
    zcf = (z + (z * z - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24 - (2 * z ** 3 - 5 * z) * sk ** 2 / 36)
    cf_var = -(port.mean() - zcf * port.std(ddof=1))

    offdiag = corr[np.triu_indices_from(corr, k=1)]
    # Diversification ratio (Choueifaty & Coignard 2008): the weighted average
    # stand-alone volatility over the portfolio's. Its square is an effective
    # number of independent bets: N for N uncorrelated equal-risk holdings,
    # 1 when everything moves together. (A principal-component version is
    # ill-defined when assets are near-uncorrelated, because the eigenvectors
    # of a near-identity matrix are arbitrary.)
    div_ratio = float(np.abs(wv * asset_vol).sum() / vol_p) if vol_p > 0 else None
    enb = div_ratio ** 2 if div_ratio else None
    pair = None
    if corr.shape[0] > 1:
        c = pd.DataFrame(corr, index=rets.columns, columns=rets.columns)
        iu = np.triu_indices_from(corr, k=1)
        j = int(np.argmax(np.abs(corr[iu])))
        pair = {"a": c.index[iu[0][j]], "b": c.columns[iu[1][j]], "corr": _finite(corr[iu][j])}

    table = pd.DataFrame({
        "Weight": w, "Stand-alone vol": asset_vol,
        "Risk contribution": pct_contrib,
        "Return contribution": rets.mean().mul(w) * ppy,
    })
    table["Risk / weight"] = table["Risk contribution"] / (w.abs() / gross)

    return {
        "inputs": {"symbols": list(w.index), "start": str(rets.index[0].date()),
                   "end": str(rets.index[-1].date()), "sessions": int(len(rets)),
                   "gross": gross, "net": float(w.sum()), "cash": float(1 - w.sum())},
        "concentration": {"hhi": hhi, "effective_n": 1 / hhi if hhi > 0 else None,
                          "top1": float(w.abs().max() / gross),
                          "top3": float(w.abs().sort_values(ascending=False).head(3).sum() / gross)},
        "risk": {"vol": _finite(vol_p),
                 "downside_dev": _finite(math.sqrt(float((port.clip(upper=0) ** 2).mean())) * math.sqrt(ppy)),
                 "max_drawdown": _finite(dd.min()), "dd_peak": str(peak.date()),
                 "dd_trough": str(trough.date()), "skew": _finite(sk), "excess_kurtosis": _finite(ku),
                 "cornish_fisher_var95_daily": _finite(cf_var)},
        "tail": tails,
        "diversification": {"ratio": _finite(div_ratio), "effective_bets": _finite(enb),
                            "avg_pair_corr": _finite(np.mean(offdiag)) if len(offdiag) else None,
                            "avg_abs_pair_corr": _finite(np.mean(np.abs(offdiag))) if len(offdiag) else None,
                            "most_correlated": pair},
        "table": table,
        "corr": pd.DataFrame(corr, index=rets.columns, columns=rets.columns),
        "portfolio_returns": port,
        "notes": notes,
    }


def average_weights(weights: pd.DataFrame, since: Optional[str] = None) -> Dict[str, float]:
    """Mean held weight per name from a backtest's weight panel, for x-raying
    what a strategy typically held (or `weights.iloc[-1]` for what it holds now)."""
    w = weights.loc[since:] if since else weights
    m = w.mean()
    return {k: float(v) for k, v in m.items() if abs(v) > 1e-6}


def current_weights(weights: pd.DataFrame) -> Dict[str, float]:
    last = weights.iloc[-1]
    return {k: float(v) for k, v in last.items() if abs(v) > 1e-6}
