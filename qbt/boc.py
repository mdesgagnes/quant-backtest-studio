"""Bank of Canada rates, for a Canadian risk-free series.

The rest of the app measures excess returns over the US 3-month T-bill
(Yahoo ^IRX). For a Canadian investor holding CAD cash that is the wrong
hurdle: from 2022 to 2025 US bills paid one to two points more than
Canadian bills, which flatters the alpha of any low-beta CAD product by
the same amount. The Bank of Canada publishes its own series through the
free Valet API, no key required.

Two series are offered:

- ``V39065``  -- 3-month Treasury bill yield, daily (Wednesday auctions
  carried forward between them).
- ``V122531`` -- 3-month Treasury bill yield, monthly average.

Both are annual yields in percent, the same convention as ^IRX, so the
result goes straight into `qbt.metrics.rf_from_yield`.

Parsing is a pure function kept apart from the network call, so it can be
tested offline and reused on a CSV saved from the Bank's website.
"""
from __future__ import annotations

import csv
import io
import urllib.request
from typing import Optional

import pandas as pd

VALET_CSV = "https://www.bankofcanada.ca/valet/observations/{series}/csv?start_date={start}"
TBILL_DAILY = "V39065"
TBILL_MONTHLY = "V122531"
SERIES_LABELS = {
    TBILL_DAILY: "Canada 3-month T-bill (Bank of Canada, daily)",
    TBILL_MONTHLY: "Canada 3-month T-bill (Bank of Canada, monthly average)",
}


class BoCError(ValueError):
    """The Valet answer could not be read as a dated series."""


def parse_valet_csv(text: str, series: Optional[str] = None) -> pd.Series:
    """A Valet CSV export (or any two-column ``date,value`` CSV) as a float
    series indexed by date. Valet prefixes the data with metadata blocks;
    the observations start at the first row whose first cell is ``date``.
    Blank and non-numeric values (Valet leaves holidays empty) are dropped.
    """
    rows = list(csv.reader(io.StringIO(text)))
    start = next((i for i, r in enumerate(rows)
                  if r and r[0].strip().strip('"').lower() == "date"), None)
    if start is None:
        # Plain two-column file with no header.
        body, header = rows, None
    else:
        header, body = rows[start], rows[start + 1:]
    col = 1
    if header is not None and series:
        names = [h.strip().strip('"') for h in header]
        if series in names:
            col = names.index(series)
    dates, vals = [], []
    for r in body:
        if len(r) <= col or not r[0].strip():
            continue
        try:
            d = pd.Timestamp(r[0].strip().strip('"'))
            v = float(r[col].strip().strip('"'))
        except (ValueError, TypeError):
            continue
        dates.append(d)
        vals.append(v)
    if not dates:
        raise BoCError("No dated observations found in the Bank of Canada answer.")
    out = pd.Series(vals, index=pd.DatetimeIndex(dates), name=series or "yield").sort_index()
    return out[~out.index.duplicated(keep="last")]


def fetch_series(series: str = TBILL_DAILY, start: str = "1990-01-01",
                 timeout: float = 20.0) -> pd.Series:
    """Download one Valet series. Raises on any network or parse failure;
    the interface decides what to fall back to."""
    url = VALET_CSV.format(series=series, start=start)
    req = urllib.request.Request(url, headers={"User-Agent": "quant-backtest-studio"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return parse_valet_csv(text, series)
