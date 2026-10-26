"""Strategy decay monitor.

A registered strategy is frozen on the registration date (rule, parameters,
costs, universe) together with its in-sample statistics. Whatever the data
does after that is out-of-sample *for that exact rule* -- the only kind of
evidence a backtest cannot manufacture. This module re-runs the frozen
strategy on fresh data, splits the record at the registration date, and asks
whether the forward record is consistent with the baseline.

Three signals, deliberately simple:

- z-score of the forward mean return against what the baseline Sharpe implies
  (how many standard errors below expectation is it?);
- the forward rolling 6- and 12-month Sharpe against the baseline Sharpe;
- a CUSUM of standardised excess returns over expectation, the standard
  sequential change detector: it accumulates evidence of a downward drift
  and resets when performance recovers.

Status: `fresh` (fewer than ~60 forward sessions: too early to say), `healthy`,
`aging` (forward results are weak but inside what chance explains) and
`stale` (the forward record is improbably poor given the baseline, or the
CUSUM has crossed its limit). It is a prompt to look, never an instruction
to stop trading.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import stats as S

MIN_FORWARD = 60


def cusum_down(excess: np.ndarray, k: float = 0.05, h: float = 40.0) -> Dict[str, Any]:
    """One-sided lower CUSUM on standardised daily values.

    The textbook settings (k = 0.5, h = 5) are for a process measured in
    whole standard deviations per observation; a daily return stream moves by
    a few hundredths of a standard deviation, so they would never fire. Here
    `k` is half the daily shift worth detecting (0.1 sd a day is a Sharpe
    ratio about 1.6 lower) and `h` is set so that a strategy that is still on
    its baseline crosses the limit about 2% of the time over any horizon
    (exp(-2 k h) = exp(-4)). A shift of 0.2 sd a day is flagged in about
    nine months."""
    c, path, breach = 0.0, [], None
    for i, z in enumerate(excess):
        c = min(0.0, c + z + k)
        path.append(c)
        if breach is None and c <= -h:
            breach = i
    return {"path": np.array(path), "breached": breach is not None, "breach_index": breach,
            "current": float(path[-1]) if path else 0.0}


def assess(returns: pd.Series, registered: str, baseline: Dict[str, Any], ppy: int = 252
           ) -> Dict[str, Any]:
    """Compare the post-registration record with the baseline statistics.

    `baseline` needs `sharpe` (annualised) and `vol` (annualised) from the
    registration-time backtest.
    """
    r = returns.dropna()
    cut = pd.Timestamp(registered)
    fwd = r[r.index > cut]
    n = len(fwd)
    base_sr = float(baseline.get("sharpe") or 0.0)
    base_vol = float(baseline.get("vol") or float(r[r.index <= cut].std() * math.sqrt(ppy)) or 0.0)
    out: Dict[str, Any] = {"registered": registered, "forward_sessions": n,
                           "baseline_sharpe": base_sr, "baseline_vol": base_vol}
    if n < 5:
        out.update(status="fresh", detail="No forward data yet.")
        return out

    daily_vol = base_vol / math.sqrt(ppy) if base_vol > 0 else float(fwd.std())
    exp_mean = base_sr / math.sqrt(ppy) * daily_vol          # per-day mean the baseline implies
    z_mean = (fwd.mean() - exp_mean) / (daily_vol / math.sqrt(n)) if daily_vol > 0 else float("nan")
    fwd_sr = S.annualised_sharpe(fwd, ppy)
    std = ((fwd - exp_mean) / daily_vol).to_numpy() if daily_vol > 0 else np.zeros(n)
    cs = cusum_down(std)
    roll = {}
    for label, win in (("6m", ppy // 2), ("12m", ppy)):
        if n >= win:
            roll[label] = S.annualised_sharpe(fwd.iloc[-win:], ppy)
    out.update({
        "forward_sharpe": fwd_sr, "forward_return": float((1 + fwd).prod() - 1),
        "forward_max_drawdown": float(((1 + fwd).cumprod() / (1 + fwd).cumprod().cummax() - 1).min()),
        "z_vs_baseline": float(z_mean), "rolling_sharpe": roll,
        "cusum_current": cs["current"], "cusum_breached": cs["breached"],
        "cusum_breach_date": str(fwd.index[cs["breach_index"]].date()) if cs["breached"] else None,
        "cusum_path": pd.Series(cs["path"], index=fwd.index),
    })
    if n < MIN_FORWARD:
        out.update(status="fresh", detail=f"Only {n} forward sessions; at least {MIN_FORWARD} are needed to judge.")
    elif cs["breached"] or z_mean < -2.0:
        out.update(status="stale", detail="Forward performance is improbably weak against the baseline "
                   f"(z = {z_mean:.2f}{', CUSUM limit crossed' if cs['breached'] else ''}).")
    elif z_mean < -1.0:
        out.update(status="aging", detail=f"Forward results trail the baseline (z = {z_mean:.2f}) but chance can explain it.")
    else:
        out.update(status="healthy", detail=f"Forward results are consistent with the baseline (z = {z_mean:.2f}).")
    return out
