"""Event study: what does a price do around dated events?

Standard market-model design (MacKinlay 1997):

1. For each event, estimate alpha and beta of the security against a market
   proxy over an estimation window that ends before the event window starts.
2. Inside the event window, abnormal return = actual - (alpha + beta x market).
3. Cumulate to CAR, average over events, and test.

Events are (ticker, date) pairs: earnings dates, index inclusions, rate
decisions, a signal firing in a backtest. Event day 0 is the first trading
session on or after the date. Everything is measured in trading days.

The estimation window is cut off `gap` sessions before the event window opens
so that a pre-event run-up does not contaminate the baseline it is judged
against.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def event_study(prices: pd.DataFrame, market: Optional[pd.Series],
                events: Iterable[Tuple[str, str]],
                pre: int = 5, post: int = 20, est_len: int = 200, gap: int = 10,
                min_est: int = 60) -> Dict[str, object]:
    """Run the study. `events` is an iterable of (ticker, date) pairs.

    With `market=None` there is no market model: the baseline is the
    security's own average return over the estimation window (beta = 0).
    Weaker, but it lets a backtest with no benchmark be studied at all."""
    rets = prices.pct_change(fill_method=None)
    no_mkt = market is None
    mret = (pd.Series(0.0, index=rets.index) if no_mkt else market.pct_change(fill_method=None))
    rel = np.arange(-pre, post + 1)
    paths, rows, skipped = [], [], []

    for tk, d in events:
        if tk not in rets.columns:
            skipped.append((tk, str(d), "ticker not in data"))
            continue
        d = pd.Timestamp(d)
        pos = rets.index.searchsorted(d)
        if pos >= len(rets.index):
            skipped.append((tk, str(d.date()), "after the data"))
            continue
        lo, hi = pos - pre, pos + post
        if lo < 0 or hi >= len(rets.index):
            skipped.append((tk, str(d.date()), "event window outside the data"))
            continue
        e_hi = lo - gap
        e_lo = e_hi - est_len
        if e_lo < 1:
            skipped.append((tk, str(d.date()), "too little history to estimate"))
            continue
        est = pd.concat([rets[tk].iloc[e_lo:e_hi], mret.iloc[e_lo:e_hi]], axis=1,
                        keys=["r", "m"]).dropna()
        if len(est) < min_est:
            skipped.append((tk, str(d.date()), "estimation window too thin"))
            continue
        if no_mkt:
            beta, alpha = 0.0, float(est["r"].mean())
            resid_sd = float(est["r"].std(ddof=1))
        else:
            var_m = est["m"].var(ddof=1)
            if var_m <= 0:
                skipped.append((tk, str(d.date()), "market has no variance in the window"))
                continue
            beta = float(est["r"].cov(est["m"]) / var_m)
            alpha = float(est["r"].mean() - beta * est["m"].mean())
            resid_sd = float((est["r"] - alpha - beta * est["m"]).std(ddof=2))
        win = pd.concat([rets[tk].iloc[lo:hi + 1], mret.iloc[lo:hi + 1]], axis=1,
                        keys=["r", "m"])
        if win.isna().any().any():
            skipped.append((tk, str(d.date()), "missing prices in the event window"))
            continue
        ar = (win["r"] - alpha - beta * win["m"]).to_numpy()
        car = ar.cumsum()
        paths.append(ar)
        # standardised: CAR over its own standard error under the estimation noise
        se = resid_sd * math.sqrt(len(ar)) if resid_sd > 0 else float("nan")
        rows.append({"Ticker": tk, "Event date": str(d.date()),
                     "Day 0": str(rets.index[pos].date()), "Alpha (daily)": alpha, "Beta": beta,
                     "CAR": float(car[-1]), "CAR t": float(car[-1] / se) if se == se and se > 0 else float("nan"),
                     "Day-0 AR": float(ar[pre])})

    if not paths:
        return {"n_events": 0, "skipped": skipped, "table": pd.DataFrame(), "relative_day": rel}

    A = np.vstack(paths)
    n = A.shape[0]
    mean_ar = A.mean(axis=0)
    cum = A.cumsum(axis=1)
    car_final = cum[:, -1]
    mean_car = cum.mean(axis=0)
    # Cross-sectional t of the average CAR: robust to event-induced variance
    # changes, unlike a test that assumes the estimation-window noise level.
    sd_car = car_final.std(ddof=1) if n > 1 else float("nan")
    t_car = float(car_final.mean() / (sd_car / math.sqrt(n))) if n > 1 and sd_car > 0 else float("nan")
    sd_ar = A.std(axis=0, ddof=1) if n > 1 else np.full(A.shape[1], np.nan)
    t_ar = mean_ar / (sd_ar / math.sqrt(n)) if n > 1 else np.full(A.shape[1], np.nan)
    avg = pd.DataFrame({"Relative day": rel, "Mean AR": mean_ar, "t (AR)": t_ar,
                        "Mean CAR": mean_car,
                        "CAR 25th": np.percentile(cum, 25, axis=0),
                        "CAR 75th": np.percentile(cum, 75, axis=0)})
    return {"n_events": n, "table": pd.DataFrame(rows), "average": avg,
            "mean_car": float(car_final.mean()), "median_car": float(np.median(car_final)),
            "t_car": t_car, "p_car": float(math.erfc(abs(t_car) / math.sqrt(2))) if t_car == t_car else float("nan"),
            "share_positive": float((car_final > 0).mean()),
            "skipped": skipped, "relative_day": rel, "pre": pre, "post": post}


def parse_events(text: str) -> List[Tuple[str, str]]:
    """'AAPL 2024-05-02' per line (comma, tab or space separated)."""
    out = []
    for line in text.replace(",", " ").replace("\t", " ").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out.append((parts[0].strip().upper(), parts[1].strip()))
    return out


# ----------------------------------------------------------------------
# Studying a backtest's own return stream
# ----------------------------------------------------------------------
def series_event_study(returns: Dict[str, pd.Series], market_returns: Optional[pd.Series],
                       dates: Sequence, pre: int = 5, post: int = 20, est_len: int = 200,
                       gap: int = 10) -> Dict[str, Dict[str, object]]:
    """Event study on one or more strategy return streams.

    Each stream is turned back into an index (the cumulative product of its
    returns) so the ordinary security-level study applies, with the benchmark's
    returns as the market. Result: one `event_study` result per series name.
    """
    mkt = None
    if market_returns is not None and len(market_returns.dropna()):
        mkt = (1.0 + market_returns.fillna(0.0)).cumprod()
    out: Dict[str, Dict[str, object]] = {}
    for name, r in returns.items():
        r = r.dropna()
        if len(r) < 60:
            continue
        idx = (1.0 + r).cumprod().to_frame(name)
        m = mkt.reindex(idx.index).ffill() if mkt is not None else None
        out[name] = event_study(idx, m, [(name, d) for d in dates], pre, post, est_len, gap)
    return out


def worst_days(market_returns: pd.Series, n: int = 20, min_gap: int = 10) -> List[pd.Timestamp]:
    """Dates of the market's `n` worst days, de-clustered: a crash is one event,
    not twelve, so a day within `min_gap` sessions of a worse one is skipped."""
    clean = market_returns.dropna()
    pos = {d: i for i, d in enumerate(clean.index)}
    chosen: List[pd.Timestamp] = []
    for d in clean.sort_values().index:
        if all(abs(pos[d] - pos[c]) > min_gap for c in chosen):
            chosen.append(d)
        if len(chosen) >= n:
            break
    return sorted(chosen)


def parse_dates(text: str) -> List[pd.Timestamp]:
    """One date per line (commas also separate); anything after the first token is ignored."""
    out = []
    for line in text.replace(",", "\n").splitlines():
        tok = line.strip().split()
        if not tok:
            continue
        try:
            out.append(pd.Timestamp(tok[0]))
        except (ValueError, TypeError):
            continue
    return out
