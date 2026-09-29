"""Market data from files, in the same shape Yahoo Finance delivers it.

The engine never knows where its prices came from: it receives a
`MarketData` (close, and optionally open, high, low, volume, dividends and
a total-return close) and simulates from that. This module builds exactly
that object from CSV or Excel files, so a run on imported data follows the
same code path, with the same assumptions, as a run on downloaded data.

Accepted layouts, detected from the content -- mixing them across files is
fine:

- **Long**: one row per date and instrument, one column per field --
  ``Date, Ticker, Open, High, Low, Close, Adj Close, Volume, Dividends``.
  The layout most data vendors export, and the recommended one.
- **One instrument per file or sheet**: ``Date, Open, High, Low, Close...``
  with the instrument named by the sheet or the file (``SPY.csv``).
- **One field per file or sheet**: wide, dates down, one column per
  instrument, with the field named by the sheet or the file
  (``open.csv``, a sheet called ``Dividends``). A wide table named after
  no field is read as closing prices.
- **Yahoo Finance's own CSV**, with its two header rows (Price / Ticker).

An Excel workbook is read sheet by sheet; sheets without a date column
(notes, a cover page) are skipped and reported.

The one decision that matters is the price convention, because getting it
wrong counts dividends twice or not at all:

- *Total return*: the close already includes dividends. Dividends in the
  file are then ignored. When the file carries both a close and an
  adjusted close, the adjusted close becomes the price and open, high and
  low are scaled by the same factor -- what Yahoo's auto-adjust does.
- *Price return*: the close excludes dividends, which the engine credits
  as cash on their ex-date. An adjusted close, if present, is kept for a
  benchmark measured on total return.

`convention="auto"` reads it from the data: price return when dividends
are supplied, total return otherwise.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import MarketData
from .returns_input import (_as_bytes, _best_date_column, _clean_numeric_column,
                            _looks_like_excel, _read_csv_bytes)

FIELDS = ["close", "adj_close", "open", "high", "low", "volume", "dividends"]
FIELD_LABELS = {"close": "Close", "adj_close": "Adj close", "open": "Open",
                "high": "High", "low": "Low", "volume": "Volume",
                "dividends": "Dividends"}

# Checked in this order, so "adj close" is never read as "close".
_ALIASES: List[Tuple[str, List[str]]] = [
    ("adj_close", ["adj close", "adjusted close", "adjclose", "adj price",
                   "total return", "total return index", "tr index",
                   "cours ajuste", "cours ajusté"]),
    ("dividends", ["dividends", "dividend", "div", "divs", "dividend amount",
                   "cash dividend", "cash dividends", "distribution",
                   "distributions", "dividende", "dividendes"]),
    ("open", ["open", "px open", "opening price", "open price", "ouverture"]),
    ("high", ["high", "px high", "high price", "haut", "plus haut"]),
    ("low", ["low", "px low", "low price", "bas", "plus bas"]),
    ("volume", ["volume", "vol", "px volume", "shares traded", "volumes"]),
    ("close", ["close", "px last", "last", "last price", "close price",
               "closing price", "price", "prices", "prix", "cours",
               "fermeture", "cloture", "clôture"]),
]
_TICKER_NAMES = {"ticker", "tickers", "symbol", "symbols", "instrument",
                 "asset", "security", "code", "ric", "isin", "symbole",
                 "actif", "titre"}
_GENERIC_SHEETS = {"sheet", "sheet1", "feuil1", "feuille1", "data", "donnees",
                   "données", "export", "prices", "values"}


class PriceImportError(ValueError):
    pass


def _norm(label) -> str:
    s = str(label).strip().lower()
    s = re.sub(r"\.\d+$", "", s)                   # pandas de-duplication
    s = re.sub(r"[_\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def field_of(label) -> Optional[str]:
    """The field a column, sheet or file name refers to, if any. Exact
    match first; then, for a longer name such as ``prices_open`` or
    ``XIC dividends``, a whole-word match."""
    n = _norm(label)
    for f, names in _ALIASES:
        if n in names:
            return f
    words = n.split()
    for f, names in _ALIASES:
        for a in names:
            aw = a.split()
            if len(aw) <= len(words) and any(words[i:i + len(aw)] == aw
                                             for i in range(len(words) - len(aw) + 1)):
                return f
    return None


def _stem(filename: str) -> str:
    base = re.split(r"[\\/]", filename)[-1]
    return re.sub(r"\.(csv|txt|tsv|xlsx|xlsm|xls)$", "", base, flags=re.I).strip()


# ----------------------------------------------------------------------
@dataclass
class ImportResult:
    fields: Dict[str, pd.DataFrame]              # field -> dates x instruments
    sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def instruments(self) -> List[str]:
        cols: List[str] = []
        for f in ("close", "adj_close") + tuple(FIELDS):
            for c in self.fields.get(f, pd.DataFrame()).columns:
                if c not in cols:
                    cols.append(c)
        return cols

    def summary(self) -> pd.DataFrame:
        """What was found, per instrument: dates and which fields."""
        rows = []
        for t in self.instruments:
            px = None
            for f in ("close", "adj_close"):
                fr = self.fields.get(f)
                if fr is not None and t in fr.columns:
                    px = fr[t].dropna()
                    break
            row = {"Instrument": t,
                   "From": px.index[0].date() if px is not None and len(px) else None,
                   "To": px.index[-1].date() if px is not None and len(px) else None,
                   "Sessions": int(len(px)) if px is not None else 0}
            for f in FIELDS:
                fr = self.fields.get(f)
                has = fr is not None and t in fr.columns and fr[t].notna().any()
                if f == "dividends" and has:
                    row[FIELD_LABELS[f]] = int((fr[t].fillna(0) != 0).sum())
                else:
                    row[FIELD_LABELS[f]] = "yes" if has else ""
            rows.append(row)
        return pd.DataFrame(rows)

    def has(self, f: str) -> bool:
        fr = self.fields.get(f)
        return fr is not None and not fr.empty and bool(fr.notna().any().any())


# ----------------------------------------------------------------------
def _yahoo_two_header(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Yahoo's CSV: a 'Price' header of fields, a 'Ticker' row of symbols,
    then a 'Date' row. Returned as a long table."""
    if not len(df) or _norm(df.columns[0]) != "price":
        return None
    if _norm(df.iloc[0, 0]) != "ticker":
        return None
    fields = [field_of(c) for c in df.columns[1:]]
    tickers = [str(t).strip() for t in df.iloc[0, 1:]]
    body = df.iloc[1:]
    body = body[~body.iloc[:, 0].astype(str).str.strip().str.lower().isin(["date", "nan", ""])]
    parts = []
    for j, (f, t) in enumerate(zip(fields, tickers), start=1):
        if f is None:
            continue
        parts.append(pd.DataFrame({"Date": body.iloc[:, 0].values, "Ticker": t,
                                   "__f": f, "__v": body.iloc[:, j].values}))
    if not parts:
        return None
    long_ = pd.concat(parts, ignore_index=True)
    return long_.pivot_table(index=["Date", "Ticker"], columns="__f", values="__v",
                             aggfunc="last").reset_index().rename(
        columns={f: FIELD_LABELS[f] for f in FIELDS})


def _numeric(col: pd.Series, where: str, warnings: List[str]) -> pd.Series:
    vals, why = _clean_numeric_column(col)
    if why:
        warnings.append(f"{where}: {why}; ignored.")
    return vals


def parse_table(df: pd.DataFrame, context: str, fallback: str = "",
                warnings: Optional[List[str]] = None
                ) -> Dict[str, pd.DataFrame]:
    """One table -> {field: dates x instruments}. `context` is the sheet
    or file name, used to name the instrument or the field when the table
    itself does not."""
    warnings = warnings if warnings is not None else []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    yahoo = _yahoo_two_header(df)
    if yahoo is not None:
        df = yahoo

    date_col, _, parsed = _best_date_column(df)
    if date_col is None:
        raise PriceImportError("no date column found")
    keep = parsed.notna().to_numpy()
    df = df.loc[keep]
    dates = pd.DatetimeIndex(parsed[keep]).normalize()
    others = [c for c in df.columns if c != date_col]

    tick_col = next((c for c in others if _norm(c) in _TICKER_NAMES), None)
    fmap = {c: field_of(c) for c in others if c != tick_col}
    fcols = {c: f for c, f in fmap.items() if f is not None}

    out: Dict[str, pd.DataFrame] = {}

    def _put(f: str, frame: pd.DataFrame) -> None:
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        out[f] = frame if f not in out else out[f].combine_first(frame)

    if tick_col is not None and fcols:                          # long
        ticks = df[tick_col].astype(str).str.strip()
        for c, f in fcols.items():
            vals = _numeric(df[c], f"{context} / {c}", warnings)
            tbl = pd.DataFrame({"d": dates, "t": ticks.values, "v": vals.values})
            wide = tbl.pivot_table(index="d", columns="t", values="v", aggfunc="last")
            _put(f, wide.reindex(columns=[t for t in ticks.unique() if t in wide.columns]))
        return out

    name_field = field_of(context) or (field_of(fallback) if fallback else None)
    if fcols and len(fcols) >= max(1, len(fmap) // 2):        # one instrument
        inst = context if (_norm(context) not in _GENERIC_SHEETS
                           and field_of(context) is None) else fallback
        inst = (inst or context).strip()
        for c, f in fcols.items():
            vals = _numeric(df[c], f"{context} / {c}", warnings)
            _put(f, pd.DataFrame({inst: vals.values}, index=dates))
        return out

    f = name_field or "close"                                  # wide
    cols = {}
    for c in others:
        vals = _numeric(df[c], f"{context} / {c}", warnings)
        if vals.notna().any():
            cols[c] = vals.values
    if cols:
        _put(f, pd.DataFrame(cols, index=dates))
    return out


def read_files(files: Sequence[Tuple[str, bytes]]) -> ImportResult:
    """Every file, and every sheet of every workbook, merged by field.
    Where two sources give the same instrument and field, the first file
    listed wins and any disagreement is reported."""
    res = ImportResult(fields={})
    for name, content in files:
        data = _as_bytes(content)
        stem = _stem(name)
        tables: List[Tuple[str, pd.DataFrame]] = []
        try:
            if _looks_like_excel(data, name):
                xl = pd.ExcelFile(io.BytesIO(data))
                for sh in xl.sheet_names:
                    tables.append((sh, pd.read_excel(xl, sheet_name=sh)))
            else:
                tables.append((stem, _read_csv_bytes(data)))
        except Exception as exc:
            res.warnings.append(f"{name}: could not be read ({exc}).")
            continue
        for ctx, tbl in tables:
            where = name if ctx == stem else f"{name} [{ctx}]"
            if tbl is None or tbl.dropna(how="all").empty:
                continue
            try:
                got = parse_table(tbl, ctx, stem, res.warnings)
            except PriceImportError as exc:
                res.warnings.append(f"{where}: skipped, {exc}.")
                continue
            if not got:
                res.warnings.append(f"{where}: no numeric data found.")
                continue
            res.sources.append(where)
            for f, fr in got.items():
                _merge(res, f, fr, where)
    return res


def _merge(res: ImportResult, f: str, fr: pd.DataFrame, where: str) -> None:
    cur = res.fields.get(f)
    if cur is None:
        res.fields[f] = fr
        return
    for c in fr.columns.intersection(cur.columns):
        a, b = cur[c].dropna(), fr[c].dropna()
        both = a.index.intersection(b.index)
        if len(both):
            diff = (a[both] - b[both]).abs() / a[both].abs().clip(lower=1e-12)
            if f != "dividends" and float(diff.max()) > 0.005:
                res.warnings.append(
                    f"{c} {FIELD_LABELS[f].lower()}: {where} disagrees with an "
                    f"earlier file on {int((diff > 0.005).sum())} date(s); the "
                    f"earlier file is kept.")
    res.fields[f] = cur.combine_first(fr)


# ----------------------------------------------------------------------
CONVENTIONS = {"auto": "Detect from the file",
               "total": "Total return (dividends already in the prices)",
               "price": "Price return (dividends paid as cash)"}


def resolve_convention(imp: ImportResult, convention: str = "auto") -> str:
    if convention in ("total", "price"):
        return convention
    return "price" if imp.has("dividends") else "total"


def to_market(imp: ImportResult, convention: str = "auto") -> MarketData:
    """The `MarketData` a Yahoo download would have produced."""
    conv = resolve_convention(imp, convention)
    f = {k: v.sort_index() for k, v in imp.fields.items() if v is not None and not v.empty}
    notes: List[str] = []
    close, adj = f.get("close"), f.get("adj_close")
    op, hi, lo = f.get("open"), f.get("high"), f.get("low")
    div = f.get("dividends")

    if close is None and adj is None:
        raise PriceImportError("No closing prices found. Name a column Close "
                               "(or Price, Last), or name the file or sheet "
                               "after the field.")
    if close is None:
        close, adj = adj, None
        if conv == "price":
            notes.append("Only an adjusted close was supplied: prices are "
                         "total return, so the run uses that convention.")
            conv = "total"
    elif adj is not None:
        missing = [c for c in adj.columns if c not in close.columns]
        if missing:
            close = close.join(adj[missing], how="outer")

    if conv == "total":
        if adj is not None:
            # As Yahoo's auto-adjust: the adjusted close is the price, and
            # open, high and low move by the same factor on each day.
            cols = [c for c in adj.columns if c in close.columns]
            factor = (adj[cols] / close[cols]).where(lambda x: np.isfinite(x) & (x > 0))
            close = close.copy()
            close[cols] = adj[cols].combine_first(close[cols])

            def _scale(fr):
                if fr is None:
                    return None
                c2 = [c for c in cols if c in fr.columns]
                fr = fr.copy()
                fr[c2] = fr[c2] * factor[c2].reindex(fr.index)
                return fr

            op, hi, lo = _scale(op), _scale(hi), _scale(lo)
            adj = None
        if div is not None and bool((div.fillna(0) != 0).any().any()):
            notes.append("Dividends in the file are ignored: with total-return "
                         "prices they are already inside the price, and "
                         "crediting them again would count them twice.")
        div = None
    else:
        if div is None or not bool((div.fillna(0) != 0).any().any()):
            notes.append("Price-return prices with no dividends supplied: "
                         "the run earns no income from them.")
            div = None
        else:
            div = div.fillna(0.0)
            div = div.loc[:, (div != 0).any()]

    return MarketData(close=close, open=op, dividends=div, adj_close=adj,
                      high=hi, low=lo, volume=f.get("volume"),
                      adjusted=(conv == "total"),
                      notes=list(imp.warnings) + notes)


def template_csv() -> str:
    """A small example of the recommended layout."""
    return (
        "Date,Ticker,Open,High,Low,Close,Volume,Dividends\n"
        "2024-03-27,XIC.TO,36.02,36.21,35.98,36.18,812400,0\n"
        "2024-03-27,ZAG.TO,13.71,13.76,13.70,13.74,1203100,0\n"
        "2024-03-28,XIC.TO,36.20,36.41,36.15,36.37,951200,0.2263\n"
        "2024-03-28,ZAG.TO,13.74,13.79,13.72,13.75,987600,0.0420\n"
    )
