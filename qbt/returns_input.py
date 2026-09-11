"""Imported return streams.

Analyses a series of periodic returns directly, with no prices, no signals
and no simulation: the return stream *is* the input. Use it for a track
record, a fund's monthly history, a composite, or the output of an engine
that lives elsewhere.

This module is deliberately self-contained. It does not call
`qbt.data.load_file`, which the price loader, the exogenous-series loader
and the target-weights loader all share -- so nothing here can change what
the backtesting engine sees. The two pipelines are independent by
construction, not by convention.

That independence is also why this loader can afford to be considerably
more forgiving than the shared one. A return-stream file is rarely a clean
export: it is a fund fact sheet, a spreadsheet someone has been hand-editing
for years, or a report copied out of a PDF. This reader is built to survive
that:

- **Excel or CSV, detected from the file's actual bytes**, not just its
  name -- so it works even if a wrapper somewhere lost the filename.
- **Several encodings and separators tried in turn** for CSV, since a
  French-locale export is often semicolon-separated in CP-1252 or Latin-1,
  not comma-separated UTF-8.
- **The date column is found by content, not by name.** Every column is
  tested by actually parsing it as dates; whichever parses the most
  successfully wins, with a small preference for a column whose header
  looks date-like. A column named "Period", "As of" or "Mois" is found the
  same way a column literally named "Date" is.
- **Both date conventions are tried and compared.** "05/01/2020" is
  ambiguous -- month/day or day/month -- and if every day in the file is
  12 or under, nothing in the data itself resolves it. Both readings are
  scored by how regularly the resulting dates space out, since a real
  return series (monthly, weekly, daily) is far more regular than the same
  dates reinterpreted the wrong way, and the more regular reading is kept.
  Which convention was used is reported, not left to guesswork.
- **Numbers are cleaned before being given up on.** Percent signs, currency
  symbols, parenthesised negatives, and both the North American
  (1,234.56) and European (1 234,56) number conventions are recognised.
  Convention is decided once per column from the punctuation actually
  present, not guessed row by row.
- **A column that still won't parse is named in the error**, with a
  sample of the values that defeated it, rather than a bare "no numeric
  column found".
"""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Frequency vocabulary
# ----------------------------------------------------------------------
_FREQ_TABLE = [
    (2, "daily", 252),
    (5.5, "business-daily", 252),
    (10, "weekly", 52),
    (50, "monthly", 12),
    (135, "quarterly", 4),
    (250, "semi-annual", 2),
]

DATE_NAME_HINTS = {
    "date", "dates", "datetime", "time", "period", "periods", "month",
    "months", "month end", "monthend", "as of", "as at", "effective date",
    "period end", "période", "periode", "mois", "date de fin de période",
    "date de la période",
}

_NAME_HINTS = {"name", "series", "strategy", "fund", "portfolio", "ticker",
               "symbol", "nom", "serie", "strategie", "fonds"}
_VALUE_HINTS = {"return", "returns", "ret", "performance", "pnl", "value",
                "rendement", "rendements", "perf"}

_NULL_TOKENS = {"", "-", "--", "n/a", "na", "n.a.", "nan", "none", "null",
                "#n/a", "#value!", ".", "…"}

_CURRENCY_CHARS = "$€£¥"


@dataclass
class ReturnStreamReport:
    columns: List[str]
    frequency: str
    periods_per_year: int
    scale: str
    n_periods: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    date_column: str = ""
    date_convention: str = ""
    warnings: List[str] = field(default_factory=list)
    dropped_columns: Dict[str, str] = field(default_factory=dict)


class ReturnStreamError(ValueError):
    """Raised with a message meant to be read by the person, not a trace."""


# ----------------------------------------------------------------------
# Format sniffing and raw table reading
# ----------------------------------------------------------------------
def _as_bytes(source: Union[bytes, bytearray, Any]) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):
        return bytes(source.getvalue())
    if hasattr(source, "read"):
        return bytes(source.read())
    raise ReturnStreamError("Could not read this file's contents.")


def _looks_like_excel(data: bytes, filename: str = "") -> bool:
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return True
    if name.endswith((".csv", ".txt", ".tsv")):
        return False
    return data[:4] == b"PK\x03\x04" or data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    text = _decode_text(data)
    attempts = [dict(sep=None, engine="python"), dict(sep=","),
                dict(sep=";"), dict(sep="\t")]
    best, best_cols = None, -1
    last_exc: Optional[Exception] = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), **kwargs)
        except Exception as exc:
            last_exc = exc
            continue
        if df.shape[1] > best_cols:
            best, best_cols = df, df.shape[1]
        if df.shape[1] > 1:
            return df
    if best is not None:
        return best
    raise ReturnStreamError(
        f"Could not read this as a CSV file. ({last_exc})" if last_exc
        else "Could not read this as a CSV file.")


def sheet_names(source: Union[bytes, Any]) -> List[str]:
    data = _as_bytes(source)
    try:
        return pd.ExcelFile(io.BytesIO(data)).sheet_names
    except Exception:
        return []


def _read_excel_bytes(data: bytes,
                      sheet: Optional[Union[str, int]] = None) -> pd.DataFrame:
    """Reads one sheet, or -- when none is specified -- the first sheet
    that actually yields a parseable date column and a numeric one."""
    xl = pd.ExcelFile(io.BytesIO(data))
    if sheet is not None:
        return pd.read_excel(xl, sheet_name=sheet)

    names = xl.sheet_names
    if len(names) <= 1:
        return pd.read_excel(xl, sheet_name=0)

    first_attempt = pd.read_excel(xl, sheet_name=names[0])
    if _has_usable_data(first_attempt):
        return first_attempt
    for nm in names[1:]:
        candidate = pd.read_excel(xl, sheet_name=nm)
        if _has_usable_data(candidate):
            return candidate
    return first_attempt


def _has_usable_data(df: pd.DataFrame) -> bool:
    if df is None or df.empty or df.shape[1] < 2:
        return False
    col, _, _ = _best_date_column(df)
    return col is not None


def _read_raw_table(source: Union[bytes, Any], filename: str = "",
                    sheet: Optional[Union[str, int]] = None) -> pd.DataFrame:
    data = _as_bytes(source)
    filename = filename or getattr(source, "name", "") or ""
    if _looks_like_excel(data, filename):
        df = _read_excel_bytes(data, sheet)
    else:
        df = _read_csv_bytes(data)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


# ----------------------------------------------------------------------
# Date column: found by content, not by name
# ----------------------------------------------------------------------
_STANDARD_PERIODS = (1.0, 7.0, 30.4, 91.3, 182.6, 365.25)


def _regularity(sorted_dates: pd.DatetimeIndex) -> float:
    """Higher is better. How closely the spacing matches a real calendar
    cadence (daily, weekly, monthly, quarterly...), tempered by how
    consistent that spacing is.

    This is a tie-break for a genuinely ambiguous date, used only when both
    day-first and month-first parse the same number of rows. It resolves
    the common real case -- a file of monthly dates that happens to have
    every day-of-month at 12 or under -- because true monthly data lands
    close to 30.4 days apart with modest month-to-month variation, while
    the wrong reading usually does not land near any calendar cadence at
    all. It will not always win against data built specifically to defeat
    it; nothing statistical can, since the two readings are informationally
    equivalent in that case. The date convention actually used is always
    reported, and can be overridden, precisely because this is a judgment
    call and not a computation with one right answer.
    """
    if len(sorted_dates) < 4:
        return 0.0
    gaps = np.diff(sorted_dates.values).astype("timedelta64[D]").astype(float)
    gaps = gaps[gaps > 0]
    if len(gaps) < 2 or gaps.mean() <= 0:
        return 0.0
    med = float(np.median(gaps))
    rel_dist = min(abs(med - p) / p for p in _STANDARD_PERIODS)
    cv = float(np.std(gaps) / gaps.mean())
    return -(rel_dist + 0.3 * cv)


def _best_date_column(df: pd.DataFrame,
                      day_first_override: Optional[bool] = None
                      ) -> Tuple[Optional[str], bool, Optional[pd.Series]]:
    """Picks the column that is most plausibly the date axis, by content.

    `day_first_override` forces the convention for whichever column is
    chosen, for the case -- day/month order in a file with every day at 12
    or under -- that cannot be resolved from the dates alone. Column
    selection itself still runs both ways and picks whichever parses more
    of the file, since that part is never ambiguous.
    """
    best: Optional[Tuple[float, float, str, bool, pd.Series]] = None
    for col in df.columns:
        if str(col).strip().lower().startswith("unnamed"):
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        name_bonus = 0.12 if str(col).strip().lower() in DATE_NAME_HINTS else 0.0
        orders = [day_first_override] if day_first_override is not None else [False, True]
        for day_first in orders:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                parsed = pd.to_datetime(series, errors="coerce",
                                        dayfirst=bool(day_first))
            ok = int(parsed.notna().sum())
            if ok == 0:
                continue
            ratio = ok / max(len(series), 1)
            score = round(ratio + name_bonus, 4)
            reg = _regularity(parsed.dropna().sort_values())
            candidate = (score, reg, col, bool(day_first), parsed)
            if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate
    if best is None:
        return None, False, None
    return best[2], best[3], best[4]


# ----------------------------------------------------------------------
# Numeric cleaning
# ----------------------------------------------------------------------
def _strip_invisible(text: str) -> str:
    return "".join(ch for ch in text
                   if unicodedata.category(ch) not in ("Zs",) or ch == " ")


def _comma_convention(raw_values: List[str]) -> str:
    has_period = any("." in v for v in raw_values)
    has_comma = any("," in v for v in raw_values)
    if not has_comma:
        return "none"
    if has_period:
        return "thousands"
    after = []
    for v in raw_values:
        if "," in v:
            tail = re.sub(r"[^0-9]", "", v.rsplit(",", 1)[-1])
            if tail:
                after.append(len(tail))
    if after and all(n == 3 for n in after) and len(after) > 1:
        return "thousands"
    return "decimal"


def _clean_numeric_column(series: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    """Coerces one column to floats. Returns (values, failure_reason)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), None

    raw = series.astype(str).map(_strip_invisible).str.strip()
    is_null = raw.str.lower().isin(_NULL_TOKENS)
    working = raw.where(~is_null, other=np.nan)

    is_paren = working.str.match(r"^\(.*\)$", na=False)
    working = working.where(~is_paren,
                            working.str.replace(r"^\((.*)\)$", r"-\1", regex=True))

    working = working.str.replace(f"[{re.escape(_CURRENCY_CHARS)}%]", "",
                                  regex=True).str.strip()

    convention = _comma_convention([v for v in working.dropna().tolist()])
    if convention == "thousands":
        cleaned = working.str.replace(",", "", regex=False)
    elif convention == "decimal":
        cleaned = working.str.replace(" ", "", regex=False)
        cleaned = cleaned.str.replace(r"\.(?=.*,)", "", regex=True)
        cleaned = cleaned.str.replace(",", ".", regex=False)
    else:
        cleaned = working

    numeric = pd.to_numeric(cleaned, errors="coerce")
    ok = numeric.notna()
    if not is_null.all() and ok.sum() == 0:
        bad = raw[~is_null].dropna().unique()[:3]
        reason = ("not numeric" + (f" (example values: {', '.join(map(repr, bad))})"
                                   if len(bad) else ""))
        return numeric, reason
    return numeric, None


# ----------------------------------------------------------------------
# Public loading
# ----------------------------------------------------------------------
def load_return_stream(source: Union[bytes, Any], filename: str = "",
                       sheet: Optional[Union[str, int]] = None,
                       day_first: Optional[bool] = None
                       ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reads a returns file, wide or long, in whatever shape it arrives.

    Accepts raw bytes, or any object with `.getvalue()` or `.read()`.
    `filename` drives Excel-vs-CSV detection together with the file's own
    bytes, which removes the failure mode where a wrapper without a
    `.name` attribute made every Excel upload get read as text.

    Returns `(frame, meta)`.
    """
    df = _read_raw_table(source, filename, sheet)
    if df.empty or df.shape[1] == 0:
        raise ReturnStreamError("The file appears to be empty.")

    date_col, resolved_day_first, parsed = _best_date_column(df, day_first)
    if date_col is None:
        sample_cols = ", ".join(map(str, df.columns[:6]))
        raise ReturnStreamError(
            "No column in this file could be read as dates. Columns found: "
            f"{sample_cols}. A date column is required, in any common "
            "format (2020-01-31, 31/01/2020, Jan-2020, ...).")

    rest = df.drop(columns=[date_col])
    rest = rest.set_index(pd.DatetimeIndex(parsed.values))
    rest = rest[rest.index.notna()].sort_index()
    rest.index.name = "Date"

    # Long format is detected, and pivoted, BEFORE any date-level
    # deduplication. In long format, several rows legitimately share one
    # date -- one per series name -- and collapsing "duplicate" dates at
    # this point would keep only the last name's row for each date and
    # silently discard every other series. `pivot_table`'s own
    # `aggfunc="last"` is where duplicate (date, name) pairs are meant to
    # be resolved, once the frame is actually in that shape.
    lower = {str(c).strip().lower(): c for c in rest.columns}
    key_col = next((lower[k] for k in _NAME_HINTS if k in lower), None)
    val_col = next((lower[k] for k in _VALUE_HINTS if k in lower), None)
    if key_col and val_col and key_col != val_col:
        cleaned, reason = _clean_numeric_column(rest[val_col])
        # Build fresh, positionally, rather than assigning into `rest` by
        # index: the index is duplicated by construction here (one date
        # per name), and assignment-by-index-alignment against a
        # duplicated index is exactly the kind of operation pandas
        # resolves ambiguously. Values lines up with `rest` by position,
        # which duplicate dates cannot disturb.
        pivot_src = pd.DataFrame({
            "__date": rest.index.values,
            "__key": rest[key_col].to_numpy(),
            "__v": cleaned.to_numpy(),
        })
        wide = pivot_src.pivot_table(index="__date", columns="__key",
                                     values="__v", aggfunc="last")
        dropped: Dict[str, str] = {}
        if reason:
            dropped[str(val_col)] = reason
        wide.index = pd.DatetimeIndex(wide.index, name="Date")
        wide.columns.name = None
        return wide, {"date_column": str(date_col), "day_first": resolved_day_first,
                     "dropped": dropped}

    # Wide format: one row is one date's full set of series, so a repeated
    # date here really is a duplicate observation to collapse.
    rest = rest[~rest.index.duplicated(keep="last")]

    candidates = [c for c in rest.columns
                  if not str(c).strip().lower().startswith("unnamed")]
    if not candidates:
        candidates = list(rest.columns)

    cleaned_cols: Dict[str, pd.Series] = {}
    dropped = {}
    for c in candidates:
        vals, reason = _clean_numeric_column(rest[c])
        if reason:
            dropped[str(c)] = reason
        elif vals.notna().any():
            cleaned_cols[str(c)] = vals

    if not cleaned_cols:
        detail = "; ".join(f"{k}: {v}" for k, v in dropped.items()) or "no usable columns"
        raise ReturnStreamError(
            f"Found dates in \u201c{date_col}\u201d, but no column of numbers "
            f"next to them ({detail}).")

    out = pd.DataFrame(cleaned_cols, index=rest.index)
    return out, {"date_column": str(date_col), "day_first": resolved_day_first,
                "dropped": dropped}


def detect_frequency(idx: pd.DatetimeIndex) -> Tuple[str, int]:
    """Returns (label, periods per year) from the spacing of the dates.

    Sorts and de-duplicates defensively regardless of what the caller
    already did, so this is never the reason a result depends on upload
    order.
    """
    idx = pd.DatetimeIndex(idx).sort_values().unique()
    if len(idx) < 3:
        return "undetermined", 252
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(float)
    gaps = gaps[gaps > 0]
    if len(gaps) < 2:
        return "undetermined", 252
    if len(gaps) >= 10:
        trim = max(1, len(gaps) // 10)
        gaps = np.sort(gaps)[trim:-trim] if len(gaps) - 2 * trim > 0 else gaps
    med = float(np.median(gaps))
    for limit, label, ppy in _FREQ_TABLE:
        if med <= limit:
            return label, ppy
    return "annual", 1


def prepare_returns(raw: pd.DataFrame, scale: str = "auto",
                    meta: Optional[Dict[str, Any]] = None
                    ) -> Tuple[pd.DataFrame, ReturnStreamReport]:
    """Normalizes an imported, already-parsed return stream to decimals."""
    warnings_out: List[str] = []
    meta = meta or {}
    df = raw.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.empty or df.shape[1] == 0:
        raise ReturnStreamError("No numeric return column found in the file.")

    label, ppy = detect_frequency(df.index)

    finite = df.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    typical = float(np.nanmedian(np.abs(finite))) if finite.size else 0.0
    detected = "percentage" if typical > 0.05 else "decimal"
    chosen = detected if scale == "auto" else scale
    if chosen == "percentage":
        df = df / 100.0
        if scale == "auto":
            warnings_out.append(
                f"Values read as percentages (median magnitude {typical:.2f}) "
                f"and divided by 100.")

    if finite.size and not (df.to_numpy() < 0).any():
        warnings_out.append(
            "No negative value anywhere in the file. If these are index "
            "levels or cumulative values rather than periodic returns, the "
            "statistics below will be meaningless.")

    extreme = df.abs() > 1.0
    if extreme.to_numpy().any():
        n = int(extreme.to_numpy().sum())
        warnings_out.append(
            f"{n} value(s) beyond +/-100% for a single {label} period. "
            f"Check the scale of the file.")

    if df.isna().to_numpy().any():
        n = int(df.isna().to_numpy().sum())
        warnings_out.append(f"{n} missing value(s) treated as zero.")
        df = df.fillna(0.0)

    if label in ("daily", "business-daily", "weekly") and len(df) < 60:
        warnings_out.append(
            f"Only {len(df)} {label} observations: statistics will be very "
            f"imprecise.")
    if label == "monthly" and len(df) < 24:
        warnings_out.append(
            f"Only {len(df)} monthly observations: two years is a thin basis "
            f"for annualized figures.")

    dc = meta.get("date_column")
    if dc:
        warnings_out.append(
            f"Dates read from \u201c{dc}\u201d"
            + (", day-first" if meta.get("day_first") else "")
            + ".")
    for col, reason in (meta.get("dropped") or {}).items():
        warnings_out.append(f"Column \u201c{col}\u201d could not be used: {reason}.")

    report = ReturnStreamReport(
        columns=[str(c) for c in df.columns],
        frequency=label, periods_per_year=ppy, scale=chosen,
        n_periods=len(df),
        start=df.index.min() if len(df) else None,
        end=df.index.max() if len(df) else None,
        date_column=str(dc or ""),
        date_convention=("day-first" if meta.get("day_first") else "month-first"),
        warnings=warnings_out,
        dropped_columns=dict(meta.get("dropped") or {}),
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
