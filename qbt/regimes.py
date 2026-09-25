"""Market regimes.

Stress periods (`qbt/stress.py`) ask what a series did inside a handful of
named episodes. This module asks the broader, continuous question: how did
it behave *on average* across different states of the world -- rising versus
falling rates, recessions versus expansions, calm versus panicked markets --
using every period in the sample, not just the dramatic ones.

Each regime dimension is a daily label series. The labels are built from
public market data (Yahoo Finance: S&P 500, VIX, 3-month T-bill, 10-year
Treasury yield) plus the NBER recession chronology, which is short and
changes rarely enough to keep inline. Like the stress module this is
descriptive, post-hoc analysis: it cannot change a number the engine
reports, and a label uses only information up to its own date.

**Aligning labels to returns.** Returns can be daily, weekly, monthly or
quarterly. Each return covers the days since the previous observation, so
it is assigned the regime that prevailed on most of those days, rather than
whatever the label happened to be on its last day.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# NBER US business-cycle recessions (peak month to trough month).
# Source: NBER Business Cycle Dating Committee. Recessions are dated
# months after the fact; add a row here when a new one is announced.
# ----------------------------------------------------------------------
NBER_RECESSIONS = [
    ("1960-04-01", "1961-02-28"),
    ("1969-12-01", "1970-11-30"),
    ("1973-11-01", "1975-03-31"),
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

# Trailing window, in trading days, over which a change in yield counts as
# "rising" or "falling", and the size of move that qualifies.
RATE_WINDOW = 126          # about six months
RATE_THRESHOLD = 0.25      # percentage points

MARKET_TICKERS = ["^GSPC", "^VIX", "^IRX", "^TNX"]


@dataclass
class Dimension:
    key: str
    title: str
    description: str
    order: List[str]                              # display order of labels
    build: Callable[[pd.DataFrame], pd.Series]    # market frame -> labels


def _equity_state(mkt: pd.DataFrame) -> pd.Series:
    spx = mkt["^GSPC"].dropna()
    dd = spx / spx.cummax() - 1.0
    return pd.cut(dd, [-np.inf, -0.20, -0.10, -0.05, np.inf],
                  labels=["Bear market (< -20%)", "Correction (-10 to -20%)",
                          "Pullback (-5 to -10%)", "Near high (> -5%)"],
                  right=False).astype(str)


def _vix_state(mkt: pd.DataFrame) -> pd.Series:
    vix = mkt["^VIX"].dropna()
    return pd.cut(vix, [0, 15, 25, 35, np.inf],
                  labels=["Calm (VIX < 15)", "Normal (15-25)",
                          "Stressed (25-35)", "Panic (VIX > 35)"],
                  right=False).astype(str)


def _rate_direction(series: pd.Series) -> pd.Series:
    s = series.dropna()
    chg = s - s.shift(RATE_WINDOW)
    out = pd.Series("Stable", index=s.index)
    out[chg > RATE_THRESHOLD] = "Rising"
    out[chg < -RATE_THRESHOLD] = "Falling"
    return out[chg.notna()]


def _short_rates(mkt: pd.DataFrame) -> pd.Series:
    return _rate_direction(mkt["^IRX"]).map(
        {"Rising": "Short rates rising", "Falling": "Short rates falling",
         "Stable": "Short rates stable"})


def _long_rates(mkt: pd.DataFrame) -> pd.Series:
    return _rate_direction(mkt["^TNX"]).map(
        {"Rising": "10Y yield rising", "Falling": "10Y yield falling",
         "Stable": "10Y yield stable"})


def _curve(mkt: pd.DataFrame) -> pd.Series:
    spread = (mkt["^TNX"] - mkt["^IRX"]).dropna()
    out = pd.Series("Normal curve (10Y > 3M)", index=spread.index)
    out[spread < 0] = "Inverted curve (10Y < 3M)"
    return out


def _economy(mkt: pd.DataFrame) -> pd.Series:
    idx = mkt.index
    out = pd.Series("Expansion", index=idx)
    for a, b in NBER_RECESSIONS:
        out[(idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))] = "Recession"
    return out


DIMENSIONS: Dict[str, Dimension] = {d.key: d for d in [
    Dimension("economy", "Economy (NBER recessions)",
              "US recessions as dated by the NBER, against expansions.",
              ["Expansion", "Recession"], _economy),
    Dimension("short_rates", "Short rates (3-month T-bill)",
              f"Direction of the 3-month T-bill yield, a proxy for Fed policy: "
              f"a move of more than {RATE_THRESHOLD:.2f} pp over the previous "
              f"~6 months counts as rising or falling.",
              ["Short rates rising", "Short rates stable", "Short rates falling"],
              _short_rates),
    Dimension("long_rates", "Long rates (10-year Treasury)",
              f"Direction of the 10-year Treasury yield: a move of more than "
              f"{RATE_THRESHOLD:.2f} pp over the previous ~6 months counts as "
              f"rising or falling.",
              ["10Y yield rising", "10Y yield stable", "10Y yield falling"],
              _long_rates),
    Dimension("curve", "Yield curve (10Y minus 3M)",
              "Whether the 10-year yield is above or below the 3-month bill. "
              "An inverted curve has preceded every US recession since the 1960s.",
              ["Normal curve (10Y > 3M)", "Inverted curve (10Y < 3M)"], _curve),
    Dimension("equity", "Equity market (S&P 500 drawdown)",
              "How far the S&P 500 sits below its previous all-time high.",
              ["Near high (> -5%)", "Pullback (-5 to -10%)",
               "Correction (-10 to -20%)", "Bear market (< -20%)"], _equity_state),
    Dimension("volatility", "Volatility (VIX)",
              "Level of the CBOE VIX index (available from 1990).",
              ["Calm (VIX < 15)", "Normal (15-25)", "Stressed (25-35)",
               "Panic (VIX > 35)"], _vix_state),
]}


def load_market(start: str = "1960-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Daily closes for the regime inputs, one column per ticker."""
    from .data import load_yfinance
    px = load_yfinance(MARKET_TICKERS, start, end, "Close")
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    return px.sort_index()


def build_labels(mkt: pd.DataFrame) -> Dict[str, pd.Series]:
    """Daily labels for every dimension the market data can support."""
    out: Dict[str, pd.Series] = {}
    for key, dim in DIMENSIONS.items():
        try:
            lab = dim.build(mkt).dropna()
        except Exception:
            continue
        lab = lab[lab.isin(dim.order)]
        if len(lab):
            out[key] = lab
    return out


def align_labels(labels: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Regime for each return date: the label that prevailed on most days
    of the period the return covers (previous date, this date]."""
    index = pd.DatetimeIndex(index).sort_values()
    lab = labels.dropna()
    if lab.empty or len(index) == 0:
        return pd.Series(index=index, dtype=object)
    # Bucket each labelled day into the first return date on or after it.
    pos = index.searchsorted(lab.index, side="left")
    ok = pos < len(index)
    first = index[0]
    # Days before the first return date belong to no period we can see,
    # except the first return's own day.
    days = pd.DataFrame({"period": index[pos[ok]], "label": lab.values[ok]},
                        index=lab.index[ok])
    if len(index) > 1:
        gap = index[1] - index[0]
        days = days[(days["period"] > first) | (days.index > first - gap)]
    mode = (days.groupby(["period", "label"]).size().reset_index(name="n")
                .sort_values(["period", "n"], ascending=[True, False])
                .drop_duplicates("period").set_index("period")["label"])
    out = mode.reindex(index)
    # A date that fell outside label coverage (e.g. before the VIX existed)
    # stays unlabelled rather than inheriting a neighbour.
    return out


def regime_table(returns: Dict[str, pd.Series], labels: pd.Series,
                 order: List[str], ppy: int,
                 benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per (regime, series): how the series behaved in that regime.

    Annualized return compounds only the periods spent in the regime and
    scales to a year, so regimes of different lengths are comparable.
    """
    rows = []
    for name, r in returns.items():
        r = r.dropna()
        if r.empty:
            continue
        lab = align_labels(labels, r.index)
        total = int(lab.notna().sum())
        b = benchmark.reindex(r.index) if benchmark is not None else None
        for reg in order:
            m = (lab == reg).to_numpy()
            x = r[m]
            n = len(x)
            row = {"Regime": reg, "Series": name, "Periods": n,
                   "% of time": n / total if total else np.nan}
            if n:
                growth = float((1 + x).prod())
                row["Ann. return"] = growth ** (ppy / n) - 1 if growth > 0 else -1.0
                row["Ann. volatility"] = float(x.std(ddof=1) * np.sqrt(ppy)) if n > 1 else np.nan
                vol = row["Ann. volatility"]
                row["Sharpe"] = (float(x.mean() * ppy / vol)
                                 if vol and vol == vol and vol > 0 else np.nan)
                row["Hit rate"] = float((x > 0).mean())
                row["Worst period"] = float(x.min())
                if b is not None:
                    bx = b[m].dropna()
                    if len(bx):
                        bg = float((1 + bx).prod())
                        bann = bg ** (ppy / len(bx)) - 1 if bg > 0 else -1.0
                        row["Benchmark ann. return"] = bann
                        row["Excess ann. return"] = row["Ann. return"] - bann
            rows.append(row)
    cols = ["Regime", "Series", "% of time", "Periods", "Ann. return",
            "Ann. volatility", "Sharpe", "Hit rate", "Worst period"]
    if benchmark is not None:
        cols += ["Benchmark ann. return", "Excess ann. return"]
    df = pd.DataFrame(rows)
    return df.reindex(columns=cols) if not df.empty else pd.DataFrame(columns=cols)


def segments(labels: pd.Series) -> pd.DataFrame:
    """Contiguous runs of one label: start, end, label. For chart shading."""
    lab = labels.dropna()
    if lab.empty:
        return pd.DataFrame(columns=["start", "end", "label"])
    change = lab.ne(lab.shift()).cumsum()
    g = lab.groupby(change)
    return pd.DataFrame({"start": g.apply(lambda s: s.index[0]).values,
                         "end": g.apply(lambda s: s.index[-1]).values,
                         "label": g.first().values})
