"""Data layer: yfinance or uploaded file -> clean price matrix.

Canonical output: DataFrame indexed by date (tz-naive, ascending order),
one column per instrument, adjusted prices (dividends + splits).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class DataQuality:
    """Diagnostic shown on screen before any backtest."""
    rows: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    per_asset: pd.DataFrame          # first/last point, % missing, gaps
    warnings: List[str]
    dropped: List[str]


# ----------------------------------------------------------------------
# Common cleaning
# ----------------------------------------------------------------------
def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index, errors="coerce", utc=True)
    df = df.loc[~idx.isna()].copy()
    idx = idx[~idx.isna()].tz_convert(None).normalize()
    df.index = idx
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def clean_prices(raw: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, DataQuality]:
    """Normalizes, validates and documents the price matrix."""
    warnings: List[str] = []
    dropped: List[str] = []

    df = _normalize_index(raw)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Unusable columns
    for col in list(df.columns):
        s = df[col].dropna()
        if s.empty:
            dropped.append(col)
            df = df.drop(columns=col)
        elif (s <= 0).any():
            n = int((s <= 0).sum())
            df[col] = df[col].where(df[col] > 0)
            warnings.append(f"{col}: {n} null or negative price(s) neutralized.")

    # Bounded forward-fill: bridges shifted holidays, not long gaps
    df = df.ffill(limit=max(0, int(cfg.fill_limit)))

    # Per-asset diagnostic
    rows = []
    for col in df.columns:
        s = df[col]
        first, last = s.first_valid_index(), s.last_valid_index()
        span = s.loc[first:last] if first is not None else s
        gaps = int(span.isna().sum())
        rows.append({
            "Instrument": col,
            "Start": first,
            "End": last,
            "Observations": int(span.notna().sum()),
            "Internal Gaps": gaps,
            "% Missing": round(100 * gaps / max(len(span), 1), 2),
        })
        if first is not None and first > df.index[0]:
            warnings.append(f"{col}: shorter history, starts on {first.date()}.")
        if gaps > 0:
            warnings.append(f"{col}: {gaps} missing day(s) within the history.")

    per_asset = pd.DataFrame(rows)

    if len(df) < cfg.min_history:
        warnings.append(
            f"Total history of {len(df)} days, below the minimum of {cfg.min_history}."
        )
    if dropped:
        warnings.append("Empty columns removed: " + ", ".join(dropped))

    q = DataQuality(
        rows=len(df),
        start=df.index[0] if len(df) else None,
        end=df.index[-1] if len(df) else None,
        per_asset=per_asset,
        warnings=warnings,
        dropped=dropped,
    )
    return df, q


# ----------------------------------------------------------------------
# Source: yfinance
# ----------------------------------------------------------------------
def load_yfinance(tickers: List[str], start: str, end: Optional[str] = None,
                  field: str = "Close") -> pd.DataFrame:
    """Downloads adjusted prices. Raises an explicit exception if empty."""
    import yfinance as yf

    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("No valid ticker.")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,       # dividends and splits baked in
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "Yahoo Finance returned no data. Check the symbols "
            "(.TO suffix for Canada) and the date range."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        use = field if field in set(lvl0) else "Close"
        px = raw[use].copy()
    else:
        px = raw[[field]].copy() if field in raw.columns else raw[["Close"]].copy()
        px.columns = tickers[:1]

    px = px.reindex(columns=[t for t in tickers if t in px.columns])
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        px = px.reindex(columns=list(px.columns) + missing)
    return px


@dataclass
class MarketData:
    """Everything the engine may need about the instruments.

    `adjusted` records which convention `close` follows, and it matters:

    - True  -- total-return prices, dividends already reinvested inside the
               price series. `dividends` must stay empty, otherwise every
               payment is counted twice.
    - False -- price-return prices (split-adjusted only). Dividends are
               separate cash flows the engine credits on their ex-date.

    Total return over a long window comes out close either way. What differs
    is the timing: reinvested instantly and compounding inside the position,
    versus sitting in cash until the next rebalance. For a strategy that is
    often partly in cash, or rebalances rarely, that gap is real.
    """
    close: pd.DataFrame
    open: Optional[pd.DataFrame] = None
    dividends: Optional[pd.DataFrame] = None
    adjusted: bool = True
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def _extract(raw: pd.DataFrame, field: str, tickers: List[str]) -> Optional[pd.DataFrame]:
    """Pulls one field out of a yfinance frame, single or multi-ticker."""
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        if field not in lvl0:
            return None
        out = raw[field].copy()
    else:
        if field not in raw.columns:
            return None
        out = raw[[field]].copy()
        out.columns = tickers[:1]
    keep = [t for t in tickers if t in out.columns]
    return out.reindex(columns=keep) if keep else None


def load_market_data(tickers: List[str], start: str, end: Optional[str] = None,
                     adjusted: bool = True,
                     want_open: bool = False,
                     want_dividends: bool = False) -> MarketData:
    """Downloads prices, and optionally opens and dividends, in one call.

    `adjusted=False` is required for dividends to mean anything: Yahoo's
    adjusted close already folds them in.
    """
    import yfinance as yf

    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("No valid ticker.")

    notes: List[str] = []
    need_actions = want_dividends and not adjusted

    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=bool(adjusted),
        actions=bool(need_actions),
        progress=False, threads=True, group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "Yahoo Finance returned no data. Check the symbols "
            "(.TO suffix for Canada) and the date range."
        )

    close = _extract(raw, "Close", tickers)
    if close is None:
        raise RuntimeError("No price column found in the downloaded data.")

    op = None
    if want_open:
        op = _extract(raw, "Open", tickers)
        if op is None:
            notes.append("No opening prices available: execution falls back "
                         "to the close.")

    div = None
    if want_dividends:
        if adjusted:
            notes.append("Dividends ignored: adjusted prices already include "
                         "them. Switch to price-return prices to model them "
                         "as cash.")
        else:
            div = _extract(raw, "Dividends", tickers)
            if div is None or float(np.nansum(div.to_numpy())) == 0.0:
                div = None
                notes.append("No dividend was reported over this period for "
                             "these symbols.")

    return MarketData(close=close, open=op, dividends=div,
                      adjusted=bool(adjusted), notes=notes)


# ----------------------------------------------------------------------
# Source: uploaded file
# ----------------------------------------------------------------------
def load_file(file_obj, sheet: Optional[str] = None,
              date_col: Optional[str] = None,
              value_hints: Optional[List[str]] = None,
              key_hints: Optional[List[str]] = None) -> pd.DataFrame:
    """Reads a CSV/XLSX in wide format (dates as rows, series as columns)
    or long format (date, key, value). Format is auto-detected.

    `value_hints` and `key_hints` let the exogenous-data and target-weights
    modules reuse this reader with their own column vocabulary.
    """
    name = getattr(file_obj, "name", "file").lower()
    data = file_obj.read() if hasattr(file_obj, "read") else file_obj
    buf = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(buf, sheet_name=sheet or 0)
    else:
        try:
            df = pd.read_csv(buf, sep=None, engine="python")
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf)

    df.columns = [str(c).strip() for c in df.columns]

    # Date column
    if date_col and date_col in df.columns:
        dc = date_col
    else:
        cand = [c for c in df.columns
                if str(c).lower() in ("date", "dates", "datetime", "time",
                                      "période", "periode")]
        dc = cand[0] if cand else df.columns[0]

    # Long format?
    lower = {str(c).lower(): c for c in df.columns}
    keys = key_hints or ["ticker", "symbol", "instrument", "asset",
                         "symbole", "symbole boursier", "actif"]
    vals = value_hints or ["close", "price", "adj close", "value", "nav",
                           "prix", "valeur"]
    tick_col = next((lower[k] for k in keys if k in lower), None)
    val_col = next((lower[k] for k in vals if k in lower), None)

    if tick_col and val_col and tick_col != dc and val_col != dc:
        wide = df.pivot_table(index=dc, columns=tick_col, values=val_col, aggfunc="last")
    else:
        wide = df.set_index(dc)
        wide = wide.select_dtypes(include=[np.number, "object"])

    wide.index.name = "Date"
    return wide


def excel_sheet_names(file_obj) -> List[str]:
    try:
        return pd.ExcelFile(file_obj).sheet_names
    except Exception:
        return []


# ----------------------------------------------------------------------
def align_universe(prices: pd.DataFrame, benchmark: Optional[str],
                   cash_proxy: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Separates the investable universe from the benchmark and cash proxy."""
    bench = prices[[benchmark]].copy() if benchmark and benchmark in prices.columns else None
    cash = prices[[cash_proxy]].copy() if cash_proxy and cash_proxy in prices.columns else None
    universe = prices.drop(columns=[c for c in (cash_proxy,) if c and c in prices.columns])
    return {"universe": universe, "benchmark": bench, "cash": cash}
