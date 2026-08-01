"""Imported target weights.

Lets you test an allocation produced elsewhere -- a manager's spreadsheet,
the output of another engine, a committee's allocation policy -- in the same
simulator, with the same frictions and execution lag as the built-in
strategies.

The file supplies the weights; the engine supplies the drift between dates,
turnover costs, cash remuneration, and statistics. Dates present in the file
become the rebalance calendar.

Accepted formats:

    Wide format                          Long format
    Date,XIC.TO,ZEB.TO,PSA.TO            date,ticker,weight
    2020-01-31,0.4,0.3,0.3               2020-01-31,XIC.TO,0.4
    2020-02-28,0.5,0.2,0.3               2020-01-31,ZEB.TO,0.3

Values can be fractions (0.4) or percentages (40): the scale is detected and
reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import load_file, _normalize_index

VALUE_HINTS = ["weight", "target", "allocation", "value",
              "poids", "poids cible", "ponderation", "valeur"]
KEY_HINTS = ["ticker", "symbol", "instrument", "asset", "name",
            "symbole", "actif", "titre"]


@dataclass
class WeightsReport:
    n_dates: int
    n_instruments: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    scale: str                       # "fraction" or "percentage"
    mean_gross: float
    max_gross: float
    has_shorts: bool
    unknown: List[str] = field(default_factory=list)
    absent: List[str] = field(default_factory=list)
    dropped_dates: int = 0
    warnings: List[str] = field(default_factory=list)
    preview: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_target_weights(file_obj, sheet: Optional[str] = None) -> pd.DataFrame:
    df = load_file(file_obj, sheet, value_hints=VALUE_HINTS, key_hints=KEY_HINTS)
    df = _normalize_index(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(axis=1, how="all").dropna(how="all")


def prepare_target_weights(raw: pd.DataFrame,
                           price_index: pd.DatetimeIndex,
                           universe: List[str],
                           normalize: str = "None",
                           max_leverage: float = 1.0
                           ) -> Tuple[pd.DataFrame, pd.DatetimeIndex, WeightsReport]:
    """Validates, rescales, and calendars weights onto trading days.

    Returns the weights aligned to the price index, the rebalance calendar
    inferred from the file, and the validation report.
    """
    warnings: List[str] = []
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    # --- Scale -----------------------------------------------------------
    gross = raw.abs().sum(axis=1)
    med = float(gross.median()) if len(gross) else 0.0
    if 40.0 <= med <= 160.0:
        raw = raw / 100.0
        scale = "percentage"
        warnings.append("Values interpreted as percentages and divided by 100.")
    else:
        scale = "fraction"

    # --- Column matching ---------------------------------------------------
    up_uni = {u.upper(): u for u in universe}
    rename, unknown = {}, []
    for c in raw.columns:
        key = str(c).strip().upper()
        if key in up_uni:
            rename[c] = up_uni[key]
        else:
            unknown.append(str(c))
    raw = raw.rename(columns=rename)
    if unknown:
        warnings.append(
            "Columns outside the universe, ignored: " + ", ".join(unknown[:12]))
        raw = raw.drop(columns=[c for c in raw.columns if str(c) in unknown])

    absent = [u for u in universe if u not in raw.columns]
    if absent:
        warnings.append(
            "Universe instruments with no weight in the file, treated as zero: "
            + ", ".join(absent[:12]))

    frame = raw.reindex(columns=universe).astype(float).fillna(0.0)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(
            "No column in the file matches the selected universe. "
            "Headers must reproduce the exact symbols."
        )

    # --- Calendar onto trading days ---------------------------------------
    pidx = pd.DatetimeIndex(price_index)
    pos = pidx.get_indexer(frame.index, method="pad")   # last trading day <= date
    keep = pos >= 0
    dropped = int((~keep).sum())
    if dropped:
        warnings.append(
            f"{dropped} date(s) earlier than the start of the price history, ignored.")
    frame = frame.loc[keep]
    eff_dates = pidx[pos[keep]]

    tmp = frame.copy()
    tmp.index = eff_dates
    tmp = tmp[~tmp.index.duplicated(keep="last")]
    rebal = pd.DatetimeIndex(sorted(set(tmp.index)))

    # Dense matrix: target weights stay in effect until the next row.
    # Without this, the engine's execution lag would read an empty row on
    # the trade day itself.
    aligned = pd.DataFrame(np.nan, index=pidx, columns=universe)
    aligned.loc[tmp.index, :] = tmp.values
    aligned = aligned.ffill().fillna(0.0)

    # --- Consistency checks -------------------------------------------------
    gross_after = aligned.loc[rebal].abs().sum(axis=1)
    has_shorts = bool((aligned.values < -1e-9).any())
    if has_shorts:
        warnings.append("Negative weights detected: short positions simulated.")

    over = gross_after > max_leverage + 1e-9
    if over.any():
        if normalize == "Scale to 100%":
            f = aligned.loc[rebal]
            sc = gross_after.where(gross_after > 0, np.nan)
            aligned.loc[rebal] = f.div(sc, axis=0).fillna(0.0).values
            warnings.append(
                f"{int(over.sum())} date(s) exceeded budget: rows scaled to 100%.")
        else:
            warnings.append(
                f"{int(over.sum())} date(s) exceed budget "
                f"({float(gross_after.max()):.2f} at the maximum). The engine will "
                f"scale them back to the allowed leverage of {max_leverage:.2f}.")
    elif normalize == "Scale to 100%":
        f = aligned.loc[rebal]
        sc = gross_after.where(gross_after > 0, np.nan)
        aligned.loc[rebal] = f.div(sc, axis=0).fillna(0.0).values
        warnings.append("Every row was scaled to 100% invested.")

    under = gross_after < 0.99
    if under.any() and normalize == "None":
        warnings.append(
            f"{int(under.sum())} date(s) under 100%: the remainder sits in cash.")

    if len(rebal):
        idle = int(pidx.get_indexer([rebal.min()])[0])
        if idle > 60:
            warnings.append(
                f"Weights only start on {rebal.min().date()}, {idle} sessions "
                f"after the start of the price history. The portfolio sits in "
                f"cash until then: narrow the start date so statistics only "
                f"cover the invested period.")

    preview = aligned.loc[rebal].head(12).copy()

    report = WeightsReport(
        n_dates=len(rebal), n_instruments=int((aligned.abs().sum() > 0).sum()),
        start=rebal.min() if len(rebal) else None,
        end=rebal.max() if len(rebal) else None,
        scale=scale,
        mean_gross=float(gross_after.mean()) if len(gross_after) else np.nan,
        max_gross=float(gross_after.max()) if len(gross_after) else np.nan,
        has_shorts=has_shorts, unknown=unknown, absent=absent,
        dropped_dates=dropped, warnings=warnings, preview=preview,
    )
    return aligned, rebal, report


def weights_template(universe: List[str], dates: Optional[pd.DatetimeIndex] = None) -> str:
    """Fillable CSV template, sized to the current universe."""
    dates = dates if dates is not None else pd.date_range("2020-01-31", periods=3, freq="ME")
    n = max(len(universe), 1)
    df = pd.DataFrame(round(1.0 / n, 4), index=dates.normalize(), columns=universe)
    df.index.name = "Date"
    return df.to_csv()
