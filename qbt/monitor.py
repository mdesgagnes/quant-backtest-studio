"""Market monitor.

A research surface alongside the backtester: what a watchlist has done
lately, how its members move together, where the risk sits. Everything here
is derived from prices.

That last point is a deliberate limit rather than an oversight. Free
fundamental endpoints are inconsistent between tickers, change shape without
warning, and return empty for anything outside large-cap US equity. A
dashboard that silently shows a blank P/E for half a watchlist is worse than
one that never promised it. Price-derived analytics are computable for every
instrument the backtester can already load, and they either work for all of
them or fail visibly for all of them.

All functions are pure: prices in, frames out. Nothing here fetches, caches
or renders.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import metrics as M

# (label, calendar offset). None means year-to-date, "ITD" means all of it.
HORIZONS: List[Tuple[str, Optional[str]]] = [
    ("1D", "1D"), ("1W", "1W"), ("1M", "1M"), ("3M", "3M"),
    ("6M", "6M"), ("YTD", None), ("1Y", "12M"), ("3Y", "36M"),
    ("5Y", "60M"), ("ITD", "ITD"),
]


def _window_return(series: pd.Series, spec: Optional[str]) -> float:
    """Return over one horizon, measured between two observations.

    The base is the last price at or before the cutoff, never the first one
    after it -- on a weekly or monthly series the difference is a whole
    period, and it is the period most likely to matter.
    """
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    end = s.index[-1]

    if spec == "ITD":
        base_val = s.iloc[0]
    else:
        if spec is None:
            cutoff = pd.Timestamp(year=end.year, month=1, day=1) - pd.Timedelta(days=1)
        elif spec.endswith("D"):
            cutoff = end - pd.Timedelta(days=int(spec[:-1]))
        elif spec.endswith("W"):
            cutoff = end - pd.Timedelta(weeks=int(spec[:-1]))
        else:
            cutoff = end - pd.DateOffset(months=int(spec[:-1]))
        prior = s.index[s.index <= cutoff]
        if len(prior) == 0:
            return np.nan          # not enough history: absent, not guessed
        base_val = s.loc[prior[-1]]

    if base_val <= 0:
        return np.nan
    return float(s.iloc[-1] / base_val - 1.0)


def performance_grid(prices: pd.DataFrame,
                     annualize_beyond_1y: bool = True) -> pd.DataFrame:
    """Return for every instrument over every standard horizon.

    Windows longer than a year are annualized, shorter ones left cumulative,
    because annualizing a three-month figure implies the quarter repeats
    four times. A horizon the history does not cover is left blank rather
    than measured over a shorter span under a longer label.
    """
    rows = []
    for name in prices.columns:
        s = prices[name].dropna()
        row: Dict[str, object] = {"Instrument": name}
        for label, spec in HORIZONS:
            val = _window_return(s, spec)
            if (annualize_beyond_1y and val == val and spec not in (None, "ITD")
                    and spec.endswith("M") and int(spec[:-1]) > 12):
                years = int(spec[:-1]) / 12.0
                val = (1.0 + val) ** (1.0 / years) - 1.0
            elif (annualize_beyond_1y and val == val and spec == "ITD"
                  and len(s) > 1):
                years = (s.index[-1] - s.index[0]).days / 365.25
                if years > 1.0:
                    val = (1.0 + val) ** (1.0 / years) - 1.0
            row[label] = val
        rows.append(row)
    return pd.DataFrame(rows)


def risk_grid(prices: pd.DataFrame, ppy: int = 252,
              window: int = 252) -> pd.DataFrame:
    """Volatility, drawdown and shape, per instrument."""
    rows = []
    for name in prices.columns:
        s = prices[name].dropna()
        if len(s) < 30:
            continue
        r = s.pct_change().dropna()
        recent = r.tail(window)
        rows.append({
            "Instrument": name,
            "Volatility": M.annual_vol(r, ppy),
            "Vol (recent)": M.annual_vol(recent, ppy) if len(recent) > 20 else np.nan,
            "Max Drawdown": M.max_drawdown(s),
            "Current Drawdown": float(M.drawdown(s).iloc[-1]),
            "Sharpe": M.sharpe(r, ppy=ppy),
            "Sortino": M.sortino(r, ppy=ppy),
            "Skew": float(r.skew()),
            "Kurtosis": float(r.kurtosis()),
            "% Positive Days": float((r > 0).mean()),
        })
    return pd.DataFrame(rows)


def relative_strength(prices: pd.DataFrame, base: str) -> pd.DataFrame:
    """Each instrument divided by the base, rebased to 100.

    A rising line means outperformance, which is a different question from
    whether the instrument itself rose.
    """
    if base not in prices.columns:
        return pd.DataFrame(index=prices.index)
    denom = prices[base].replace(0, np.nan)
    rel = prices.div(denom, axis=0).dropna(how="all")
    if rel.empty:
        return rel
    first = rel.bfill().iloc[0].replace(0, np.nan)
    return rel.div(first, axis=1) * 100.0


def rolling_correlation(prices: pd.DataFrame, base: str,
                        window: int = 126) -> pd.DataFrame:
    """Correlation of each instrument with the base, through time."""
    if base not in prices.columns:
        return pd.DataFrame(index=prices.index)
    r = prices.pct_change()
    b = r[base]
    out = {c: r[c].rolling(window, min_periods=max(20, window // 3)).corr(b)
           for c in r.columns if c != base}
    return pd.DataFrame(out).dropna(how="all")


def rolling_beta(prices: pd.DataFrame, base: str,
                 window: int = 126) -> pd.DataFrame:
    """Beta of each instrument to the base, through time."""
    if base not in prices.columns:
        return pd.DataFrame(index=prices.index)
    r = prices.pct_change()
    b = r[base]
    var = b.rolling(window, min_periods=max(20, window // 3)).var()
    out = {}
    for c in r.columns:
        if c == base:
            continue
        cov = r[c].rolling(window, min_periods=max(20, window // 3)).cov(b)
        out[c] = cov / var.replace(0, np.nan)
    return pd.DataFrame(out).dropna(how="all")


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns, with months the instrument did not trade left blank.

    `resample().prod()` returns 1.0 for an empty group, which becomes a 0%
    return. On a name that listed halfway through the sample that invents a
    decade of flat months, and every average computed from them is wrong.
    Masking on the observation count is what keeps a short history short.
    """
    r = prices.pct_change()
    grown = (1 + r).resample("ME").prod() - 1.0
    observed = r.notna().resample("ME").sum()
    return grown.where(observed > 0)


def seasonality(prices: pd.DataFrame) -> pd.DataFrame:
    """Average calendar-month return per instrument.

    Read with suspicion. Twenty years of data gives twenty observations per
    month, and a single crash lands in one of them. This describes what
    happened; it does not forecast. The sample count per instrument is
    reported alongside so the thinness is visible rather than implied.
    """
    monthly = _monthly_returns(prices)
    if monthly.empty:
        return pd.DataFrame()
    out = {}
    for c in monthly.columns:
        s = monthly[c].dropna()
        if s.empty:
            continue
        out[c] = s.groupby(s.index.month).mean()
    frame = pd.DataFrame(out).T
    if frame.empty:
        return frame
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
             "Aug", "Sep", "Oct", "Nov", "Dec"]
    frame = frame.reindex(columns=range(1, 13))
    frame.columns = names
    return frame


def seasonality_counts(prices: pd.DataFrame) -> pd.Series:
    """Whole years of observation behind each instrument's averages."""
    monthly = _monthly_returns(prices)
    if monthly.empty:
        return pd.Series(dtype=int)
    return (monthly.notna().sum() / 12.0).round(1)


def risk_return_points(prices: pd.DataFrame, ppy: int = 252) -> pd.DataFrame:
    """Annualized return against annualized volatility, per instrument."""
    rows = []
    for name in prices.columns:
        s = prices[name].dropna()
        if len(s) < 60:
            continue
        years = (s.index[-1] - s.index[0]).days / 365.25
        if years <= 0:
            continue
        rows.append({
            "Instrument": name,
            "Return": float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1),
            "Volatility": M.annual_vol(s.pct_change().dropna(), ppy),
            "Max Drawdown": M.max_drawdown(s),
        })
    return pd.DataFrame(rows)


def correlation_pairs(prices: pd.DataFrame, top: int = 10) -> Dict[str, pd.DataFrame]:
    """The most and least correlated pairs in the watchlist."""
    corr = prices.pct_change().corr()
    if corr.empty or len(corr) < 2:
        return {"highest": pd.DataFrame(), "lowest": pd.DataFrame()}
    pairs = []
    names = list(corr.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = corr.iloc[i, j]
            if v == v:
                pairs.append({"Pair": f"{names[i]} / {names[j]}",
                              "Correlation": float(v)})
    df = pd.DataFrame(pairs)
    if df.empty:
        return {"highest": df, "lowest": df}
    df = df.sort_values("Correlation", ascending=False)
    return {"highest": df.head(top).reset_index(drop=True),
            "lowest": df.tail(top).sort_values("Correlation")
                        .reset_index(drop=True)}


def drawdown_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """How far each instrument sits below its own high, and for how long."""
    rows = []
    for name in prices.columns:
        s = prices[name].dropna()
        if len(s) < 5:
            continue
        dd = M.drawdown(s)
        peak_idx = s.idxmax()
        cur = float(dd.iloc[-1])
        since_peak = int(len(s.loc[s.index > peak_idx]))
        rows.append({
            "Instrument": name,
            "Current Drawdown": cur,
            "Max Drawdown": float(dd.min()),
            "Peak Date": s.loc[:s.index[-1]].idxmax().date(),
            "Sessions Since Peak": since_peak,
            "At High": "Yes" if cur > -1e-6 else "No",
        })
    return pd.DataFrame(rows)
