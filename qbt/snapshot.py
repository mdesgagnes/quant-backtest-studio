"""Instrument snapshot and watchlist screener: the numbers behind the
Markets "Snapshot" and "Screener" tabs.

Both are modelled on the analysis and screener pages of Sparkline Analytics
(window pills, a one-row summary per instrument, annual and monthly return
grids, a sortable table with an inline trend column and a percentile among
peers). Everything here is price-only and pure pandas, so it works for any
instrument the backtester can load and can be unit-tested without Streamlit.

House rules carried over from qbt/monitor.py: a window the history does not
cover is left blank rather than measured over a shorter span under a longer
label, and an annualized figure is never produced for a window under a year.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import metrics as M

WINDOWS: List[str] = ["YTD", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "Max"]
CHART_MODES: List[str] = ["Cumulative", "Drawdown", "Rolling"]

# Fewer funds than this and a percentile says nothing.
MIN_PEERS = 5


def window_start(end: pd.Timestamp, window: str,
                 first: Optional[pd.Timestamp] = None) -> Optional[pd.Timestamp]:
    """Calendar cutoff for a window label, or None for the whole history."""
    if window == "Max":
        return first
    if window == "YTD":
        return pd.Timestamp(year=end.year, month=1, day=1) - pd.Timedelta(days=1)
    months = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "3Y": 36, "5Y": 60}.get(window)
    if months is None:
        raise ValueError(f"unknown window {window!r}")
    return end - pd.DateOffset(months=months)


def slice_window(prices: pd.DataFrame, window: str,
                 start: Optional[pd.Timestamp] = None,
                 end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Prices from the base observation of the window to its end.

    The base is the last price at or before the cutoff (as in
    monitor._window_return), so the first return of the window is a real
    one. Explicit start/end override the window label (the date pickers).
    Instruments that do not reach back to the cutoff are blanked, not
    quietly measured over a shorter span.
    """
    if prices.empty:
        return prices
    last = prices.index[-1] if end is None else pd.Timestamp(end)
    px = prices.loc[:last]
    if px.empty:
        return px
    if start is not None:
        cut = pd.Timestamp(start)
    else:
        cut = window_start(px.index[-1], window, px.index[0])
    out = {}
    for c in px.columns:
        s = px[c].dropna()
        if s.empty:
            continue
        if window == "Max" and start is None:
            out[c] = s
            continue
        prior = s.index[s.index <= cut]
        if len(prior) == 0:
            # Falls inside the series' own first day: allow when the series
            # starts within a week after the cutoff (weekends, holidays).
            if s.index[0] - cut <= pd.Timedelta(days=7):
                out[c] = s
            continue
        out[c] = s.loc[prior[-1]:]
    return pd.DataFrame(out) if out else px.iloc[0:0]


def _span_years(s: pd.Series) -> float:
    return (s.index[-1] - s.index[0]).days / 365.25 if len(s) > 1 else 0.0


def window_stats(prices: pd.DataFrame, rf_pa: float = 0.0,
                 ppy: int = 252) -> pd.DataFrame:
    """One row per instrument over the prices passed in.

    Cumulative return, annualized return (only when the span is a year or
    more), annualized volatility, Sharpe and maximum drawdown.
    """
    rows = []
    for name in prices.columns:
        s = prices[name].dropna()
        if len(s) < 3 or s.iloc[0] <= 0:
            continue
        r = s.pct_change().dropna()
        cum = float(s.iloc[-1] / s.iloc[0] - 1.0)
        yrs = _span_years(s)
        ann = (1.0 + cum) ** (1.0 / yrs) - 1.0 if yrs >= 0.99 else np.nan
        rows.append({
            "Instrument": name,
            "Cumulative Return": cum,
            "Ann. Return": ann,
            "Ann. Vol": M.annual_vol(r, ppy),
            "Sharpe": M.sharpe(r, rf_pa, ppy=ppy),
            "Max DD": M.max_drawdown(s),
        })
    return pd.DataFrame(rows)


def percentile_rank(values: pd.Series, higher_is_better: bool = True,
                    min_peers: int = MIN_PEERS) -> pd.Series:
    """Percentile (1-100) of each value among the others, 100 = best.

    Sparkline shows each fund against its peer group; here the peers are the
    other instruments on the watchlist. With fewer than `min_peers` values
    the answer is blank: a rank among three is not a percentile.
    """
    v = pd.to_numeric(values, errors="coerce")
    ok = v.dropna()
    out = pd.Series(np.nan, index=v.index, dtype=float)
    if len(ok) < min_peers:
        return out
    ranks = ok.rank(pct=True, method="average", ascending=higher_is_better)
    out.loc[ok.index] = (ranks * 100.0).round()
    return out


def rebased_curves(prices: pd.DataFrame, mode: str = "Cumulative",
                   rolling_window: int = 126) -> pd.DataFrame:
    """Series for the chart, in percent, for one of the chart modes.

    Cumulative: return since the first observation of each series.
    Drawdown: distance below the running peak.
    Rolling: return over the trailing `rolling_window` sessions.
    """
    if mode not in CHART_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    cols = {}
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) < 2:
            continue
        if mode == "Cumulative":
            cols[c] = (s / s.iloc[0] - 1.0) * 100.0
        elif mode == "Drawdown":
            cols[c] = M.drawdown(s) * 100.0
        else:
            cols[c] = (s / s.shift(rolling_window) - 1.0).dropna() * 100.0
    return pd.DataFrame(cols)


def annual_returns(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar-year returns (decimal) and a mask of partial years.

    Rows are instruments, columns are years. A year is complete only when the
    series has a price on or before its own first trading days of the prior
    year-end; the first year of a series and the current year are partial
    and flagged in the second frame, the way Sparkline stars them.
    """
    vals: Dict[str, pd.Series] = {}
    part: Dict[str, pd.Series] = {}
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) < 2:
            continue
        ye = s.groupby(s.index.year).last()
        prev = ye.shift(1)
        first_px = s.iloc[0]
        prev.iloc[0] = first_px          # first year: from the first price
        ret = ye / prev - 1.0
        flag = pd.Series(False, index=ret.index)
        starts_late = s.index[0] > pd.Timestamp(year=s.index[0].year, month=1, day=10)
        if starts_late:
            flag.iloc[0] = True
        if s.index[-1] < pd.Timestamp(year=s.index[-1].year, month=12, day=24):
            flag.iloc[-1] = True
        vals[c], part[c] = ret, flag
    if not vals:
        return pd.DataFrame(), pd.DataFrame()
    table = pd.DataFrame(vals).T.sort_index(axis=1)
    mask = pd.DataFrame(part).T.reindex(index=table.index,
                                        columns=table.columns).fillna(False)
    return table, mask.astype(bool)


def monthly_grid(series: pd.Series) -> pd.DataFrame:
    """Year x month returns (decimal) for one price series."""
    s = series.dropna()
    if len(s) < 3:
        return pd.DataFrame()
    me = s.resample("ME").last().dropna()
    r = me.pct_change()
    # First month is measured from the first price, not dropped.
    if len(me):
        r.iloc[0] = me.iloc[0] / s.iloc[0] - 1.0
    df = pd.DataFrame({"y": r.index.year, "m": r.index.month, "r": r.values})
    return df.pivot(index="y", columns="m", values="r").reindex(columns=range(1, 13))


def trend_points(series: pd.Series, n: int = 60) -> List[float]:
    """At most `n` evenly spaced values rebased to 100, for an inline chart."""
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return []
    if len(s) > n:
        idx = np.linspace(0, len(s) - 1, n).round().astype(int)
        s = s.iloc[idx]
    return [float(x) for x in (s / s.iloc[0] * 100.0).values]


def screener_table(prices: pd.DataFrame, trend_window: str = "1Y",
                   rf_pa: float = 0.0) -> pd.DataFrame:
    """The watchlist as a screen: one row per instrument.

    Columns: a trend over `trend_window`, trailing returns, risk, distance
    from the 52-week high, and each return and Sharpe as a percentile on the
    watchlist. Returns are decimal; the app formats them.
    """
    from . import monitor as MON
    grid = MON.performance_grid(prices).set_index("Instrument")
    rows = []
    for name in prices.columns:
        s = prices[name].dropna()
        if len(s) < 30:
            continue
        r = s.pct_change().dropna()
        last = s.index[-1]
        yr = s.loc[last - pd.DateOffset(months=12):]
        hi = float(yr.max())
        win = slice_window(prices[[name]], trend_window)
        trend = trend_points(win[name]) if name in win.columns else []
        rows.append({
            "Instrument": name,
            "Trend": trend,
            "1D": grid.at[name, "1D"], "1M": grid.at[name, "1M"],
            "3M": grid.at[name, "3M"], "YTD": grid.at[name, "YTD"],
            "1Y": grid.at[name, "1Y"], "3Y (ann.)": grid.at[name, "3Y"],
            "Volatility": M.annual_vol(r),
            "Sharpe": M.sharpe(r, rf_pa),
            "Max DD": M.max_drawdown(s),
            "From 52w High": float(s.iloc[-1] / hi - 1.0) if hi > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["1Y %ile"] = percentile_rank(df["1Y"])
    df["Sharpe %ile"] = percentile_rank(df["Sharpe"])
    df["Vol %ile"] = percentile_rank(df["Volatility"], higher_is_better=False)
    return df


def summary_row(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """Median of each numeric column (Sparkline's pinned 'Median' row)."""
    out = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna() if c in df else []
        out[c] = float(np.median(v)) if len(v) else np.nan
    return out
