"""Imported return streams.

Analyses a series of periodic returns directly, with no prices, no signals
and no simulation: the return stream *is* the input. Use it for a track
record, a fund's monthly history, a composite, or the output of an engine
that lives elsewhere.

Accepted shapes:

    Wide                              Long
    Date,Strategy,Benchmark           date,name,return
    2020-01-31,0.0213,0.0185          2020-01-31,Strategy,0.0213
    2020-02-29,-0.0154,-0.0210        2020-01-31,Benchmark,0.0185

Daily, weekly, monthly or quarterly: the frequency is inferred from the
spacing of the dates and drives the annualization. Values may be decimals
(0.0213) or percentages (2.13); the scale is inferred and reported, and can
be overridden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import load_file, _normalize_index

VALUE_HINTS = ["return", "returns", "ret", "performance", "pnl", "value",
               "rendement", "rendements", "perf"]
KEY_HINTS = ["name", "series", "strategy", "fund", "portfolio", "ticker",
             "symbol", "nom", "serie", "strategie", "fonds"]

# Median spacing in calendar days -> (label, periods per year)
_FREQ_TABLE = [
    (4, "daily", 252),
    (10, "weekly", 52),
    (45, "monthly", 12),
    (135, "quarterly", 4),
    (250, "semi-annual", 2),
]


@dataclass
class ReturnStreamReport:
    columns: List[str]
    frequency: str
    periods_per_year: int
    scale: str                       # "decimal" or "percentage"
    n_periods: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    warnings: List[str] = field(default_factory=list)


def detect_frequency(idx: pd.DatetimeIndex) -> tuple:
    """Returns (label, periods per year) from the spacing of the dates."""
    if len(idx) < 3:
        return "undetermined", 252
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    med = float(np.median(gaps))
    for limit, label, ppy in _FREQ_TABLE:
        if med <= limit:
            return label, ppy
    return "annual", 1


def load_return_stream(file_obj, sheet: Optional[str] = None) -> pd.DataFrame:
    """Reads a returns file in wide or long format."""
    df = load_file(file_obj, sheet, value_hints=VALUE_HINTS, key_hints=KEY_HINTS)
    df = _normalize_index(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(axis=1, how="all").dropna(how="all")


def prepare_returns(raw: pd.DataFrame,
                    scale: str = "auto") -> tuple:
    """Normalizes an imported return stream to decimals.

    Returns (returns, report).
    """
    warnings: List[str] = []
    df = raw.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.empty or df.shape[1] == 0:
        raise ValueError("No numeric return column found in the file.")

    label, ppy = detect_frequency(df.index)

    # --- Scale -----------------------------------------------------------
    finite = df.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    typical = float(np.nanmedian(np.abs(finite))) if finite.size else 0.0
    detected = "percentage" if typical > 0.05 else "decimal"
    chosen = detected if scale == "auto" else scale
    if chosen == "percentage":
        df = df / 100.0
        if scale == "auto":
            warnings.append(
                f"Values read as percentages (median magnitude {typical:.2f}) "
                f"and divided by 100.")

    # --- Sanity checks ----------------------------------------------------
    if finite.size and not (df.to_numpy() < 0).any():
        warnings.append(
            "No negative value anywhere in the file. If these are index "
            "levels or cumulative values rather than periodic returns, the "
            "statistics below will be meaningless.")

    extreme = df.abs() > 1.0
    if extreme.to_numpy().any():
        n = int(extreme.to_numpy().sum())
        warnings.append(
            f"{n} value(s) beyond +/-100% for a single {label} period. "
            f"Check the scale of the file.")

    if df.isna().to_numpy().any():
        n = int(df.isna().to_numpy().sum())
        warnings.append(f"{n} missing value(s) treated as zero.")
        df = df.fillna(0.0)

    if label in ("daily", "weekly") and len(df) < 60:
        warnings.append(
            f"Only {len(df)} {label} observations: statistics will be very "
            f"imprecise.")
    if label == "monthly" and len(df) < 24:
        warnings.append(
            f"Only {len(df)} monthly observations: two years is a thin basis "
            f"for annualized figures.")

    report = ReturnStreamReport(
        columns=[str(c) for c in df.columns],
        frequency=label, periods_per_year=ppy, scale=chosen,
        n_periods=len(df),
        start=df.index.min() if len(df) else None,
        end=df.index.max() if len(df) else None,
        warnings=warnings,
    )
    return df, report


def equity_from_returns(returns: pd.Series, initial: float = 100_000.0) -> pd.Series:
    """Compounds a return stream into a value curve."""
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def template(freq: str = "monthly") -> str:
    """Fillable CSV template."""
    if freq == "daily":
        idx = pd.bdate_range("2020-01-01", periods=6)
    else:
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"Strategy": np.round(rng.normal(0.006, 0.03, len(idx)), 4),
         "Benchmark": np.round(rng.normal(0.005, 0.035, len(idx)), 4)},
        index=idx.normalize())
    df.index.name = "Date"
    return df.to_csv()
