"""Exogenous data: economic, fundamental, or any other imported signal that
is not a price.

Two uses, distinguished by the shape of the file:

- **Macro series** -- one column, one value per date, applied to the whole
  portfolio (policy rate, ISM, credit spreads, surprise index...). Used as a
  regime filter.
- **Cross-sectional factor** -- one column per instrument, named exactly like
  the symbol (P/E ratio, ROE, earnings revisions, ESG score...). Used to rank
  the universe.

The critical point is the **publication lag**. A data point dated January 31
is not known until several weeks later; using it on its reference date
produces a backtest that cannot be reproduced live. The lag is applied to the
index before any alignment, which makes look-ahead structurally impossible
rather than dependent on the user's vigilance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .data import load_file, _normalize_index

VALUE_HINTS = ["value", "level", "score", "obs_value", "close", "price",
              "valeur", "niveau", "prix"]
KEY_HINTS = ["series", "variable", "indicator", "id", "ticker", "symbol",
            "instrument", "serie", "indicateur", "symbole"]


@dataclass
class ExogReport:
    columns: List[str]
    frequency: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    lag_days: int
    per_series: pd.DataFrame
    warnings: List[str]


def detect_frequency(idx: pd.DatetimeIndex) -> str:
    if len(idx) < 3:
        return "undetermined"
    d = float(np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int)))
    if d <= 1.5:
        return "daily"
    if d <= 4:
        return "business-daily"
    if d <= 10:
        return "weekly"
    if d <= 45:
        return "monthly"
    if d <= 135:
        return "quarterly"
    if d <= 250:
        return "semi-annual"
    return "annual"


def load_exog(file_obj, sheet: Optional[str] = None) -> pd.DataFrame:
    """Reads an exogenous-series file (wide or long format)."""
    df = load_file(file_obj, sheet, value_hints=VALUE_HINTS, key_hints=KEY_HINTS)
    df = _normalize_index(df)
    return df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


def prepare_exog(raw: pd.DataFrame, price_index: pd.DatetimeIndex,
                 publication_lag_days: int = 1,
                 ffill_limit: Optional[int] = None) -> pd.DataFrame:
    """Shifts by the publication lag, then aligns to the price calendar.

    The lag is in calendar days: that is how economic releases are announced
    ("roughly three weeks after month-end").
    """
    if raw is None or raw.empty:
        return pd.DataFrame(index=price_index)

    df = raw.sort_index().copy()
    lag = max(0, int(publication_lag_days))
    df.index = df.index + pd.Timedelta(days=lag)

    full = df.reindex(df.index.union(price_index)).sort_index()
    full = full.ffill(limit=ffill_limit)
    return full.reindex(price_index)


def exog_report(raw: pd.DataFrame, aligned: pd.DataFrame,
                lag_days: int, price_index: pd.DatetimeIndex) -> ExogReport:
    warnings: List[str] = []
    rows = []
    for c in raw.columns:
        s = raw[c].dropna()
        a = aligned[c].dropna() if c in aligned.columns else pd.Series(dtype=float)
        cov = 100 * len(a) / max(len(price_index), 1)
        rows.append({
            "Series": c,
            "Observations": len(s),
            "Start": s.index.min() if len(s) else None,
            "End": s.index.max() if len(s) else None,
            "Backtest Coverage": round(cov, 1),
            "First Usable Date": a.index.min() if len(a) else None,
        })
        if len(s) and s.index.max() < price_index.max() - pd.Timedelta(days=180):
            warnings.append(
                f"{c}: last observation on {s.index.max().date()}, "
                f"the series ends well before the end of the backtest.")
        if cov < 50:
            warnings.append(f"{c}: covers only {cov:.0f}% of the tested period.")

    freq = detect_frequency(raw.index)
    if freq in ("monthly", "quarterly", "semi-annual", "annual") and lag_days < 15:
        warnings.append(
            f"{freq.capitalize()} frequency with a {lag_days}-day publication lag. "
            f"Economic data is typically released 20 to 45 days after its "
            f"reference date: too short a lag creates look-ahead bias.")

    return ExogReport(
        columns=list(raw.columns), frequency=freq,
        start=raw.index.min() if len(raw) else None,
        end=raw.index.max() if len(raw) else None,
        lag_days=lag_days,
        per_series=pd.DataFrame(rows), warnings=warnings,
    )


def split_roles(exog: pd.DataFrame, universe: List[str]) -> dict:
    """Separates columns that match an instrument (cross-sectional factor)
    from those that don't (macro series)."""
    up = {str(c).upper(): c for c in exog.columns}
    matched = [up[t.upper()] for t in universe if t.upper() in up]
    macro = [c for c in exog.columns if c not in matched]
    return {"factor": matched, "macro": macro}
