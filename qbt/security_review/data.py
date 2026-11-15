"""Data for the Security review: download, disk cache, CAD conversion,
monthly returns, proxy splicing and data-quality flags.

Conventions (also in docs/METHODOLOGY.md):

* **Total return.** Every return comes from Yahoo Finance's *Adj Close*
  (`yf.download(..., auto_adjust=False)["Adj Close"]`): splits and
  distributions reinvested. *Close* (split-adjusted only) and *Dividends*
  are kept for the distribution yield.
* **Currency.** The base currency is CAD. A US-listed security's adjusted
  price is multiplied by the USD/CAD rate (CAD per USD; Yahoo `CAD=X`,
  extended before 2003 with the stored Fed H.10 history in `data/fx`). The
  rate is carried forward at most 5 sessions to meet a price date. CAD-listed
  securities are untouched.
* **No filled returns.** Returns are computed on each security's own trading
  dates. Months (or days) a security did not exist are NaN, never 0. When a
  portfolio needs prices on a common daily calendar, the last *price level*
  is carried over an exchange holiday (the market was shut, so the return
  really was 0); a missing month is never filled.
* **Months.** A monthly return is the change in month-end adjusted price.
  The month in progress is dropped. A security's first, partial month is
  dropped too (there is no previous month-end price).
* **Splicing.** A building block can name a proxy. Before the block's first
  monthly (or daily) return, the proxy's CAD return is used; each spliced
  date is returned as a flag.

The network is only touched by `Downloader` callables, so every function is
testable offline with a fake downloader.
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import fx as FX
from .. import metrics as M
from .config import ROOT

CACHE_DIR = os.path.join(ROOT, "data", "cache")
PRICE_FIELD = "Adj Close"
MAX_AGE_DAYS = 3.0
FX_LIMIT = 5
SUSPICIOUS_DAILY = 0.25
STALE_RUN = 10

# downloader(tickers, start) -> {ticker: DataFrame[adj_close, close, dividends]}
Downloader = Callable[[List[str], str], Dict[str, pd.DataFrame]]


# ----------------------------------------------------------------------
# Download and cache
# ----------------------------------------------------------------------
def yahoo_downloader(tickers: List[str], start: str) -> Dict[str, pd.DataFrame]:
    """Adjusted close, close and dividends from Yahoo Finance."""
    import yfinance as yf
    raw = yf.download(tickers=tickers, start=start, auto_adjust=False, actions=True,
                      progress=False, threads=True, group_by="column")
    out: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                adj, close = raw["Adj Close"][t], raw["Close"][t]
                div = raw["Dividends"][t] if "Dividends" in raw.columns.get_level_values(0) \
                    else pd.Series(0.0, index=raw.index)
            else:
                adj, close = raw["Adj Close"], raw["Close"]
                div = raw.get("Dividends", pd.Series(0.0, index=raw.index))
        except KeyError:
            continue
        df = pd.DataFrame({"adj_close": adj, "close": close, "dividends": div})
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df = df[df["adj_close"].notna() & (df["adj_close"] > 0)]
        if len(df):
            out[t] = df
    return out


def _path(ticker: str, folder: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in ticker)
    return os.path.join(folder, "prices", f"{safe}.csv")


def _read_cached(ticker: str, folder: str) -> Optional[pd.DataFrame]:
    p = _path(ticker, folder)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df if len(df) else None


def _age_days(ticker: str, folder: str) -> float:
    p = _path(ticker, folder)
    return (time.time() - os.path.getmtime(p)) / 86400 if os.path.exists(p) else np.inf


@dataclass
class PriceSet:
    """Daily prices per ticker, plus what went wrong fetching them."""
    frames: Dict[str, pd.DataFrame]
    failed: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def adj(self, t: str) -> pd.Series:
        return self.frames[t]["adj_close"].dropna() if t in self.frames else pd.Series(dtype=float)


def fetch_prices(tickers: Iterable[str], start: str = "1990-01-01",
                 downloader: Optional[Downloader] = None,
                 cache_dir: Optional[str] = CACHE_DIR,
                 max_age_days: float = MAX_AGE_DAYS) -> PriceSet:
    """Daily prices for `tickers`, from the disk cache when it is fresh.

    A ticker the download does not return is listed in `failed` (never
    filled). If the download fails but a stale cached copy exists, the stale
    copy is used and a note says so."""
    tickers = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    downloader = downloader or yahoo_downloader
    frames: Dict[str, pd.DataFrame] = {}
    notes: List[str] = []
    todo = []
    for t in tickers:
        cached = _read_cached(t, cache_dir) if cache_dir else None
        if cached is not None and _age_days(t, cache_dir) <= max_age_days:
            frames[t] = cached
        else:
            todo.append(t)
    got: Dict[str, pd.DataFrame] = {}
    if todo:
        try:
            got = downloader(todo, start) or {}
        except Exception as exc:          # network down: fall back to stale cache
            notes.append(f"Download failed ({exc}); cached prices used where available.")
            got = {}
    failed = []
    for t in todo:
        if t in got and len(got[t]):
            frames[t] = got[t]
            if cache_dir:
                os.makedirs(os.path.dirname(_path(t, cache_dir)), exist_ok=True)
                got[t].to_csv(_path(t, cache_dir))
        else:
            stale = _read_cached(t, cache_dir) if cache_dir else None
            if stale is not None:
                frames[t] = stale
                notes.append(f"{t}: download failed, cached prices from "
                             f"{stale.index[-1].date()} used.")
            else:
                failed.append(t)
    if failed:
        notes.append("No data for " + ", ".join(failed) + ": skipped.")
    return PriceSet(frames, failed, notes)


def fetch_url_cached(url: str, name: str, cache_dir: Optional[str] = CACHE_DIR,
                     max_age_days: float = 7.0, timeout: float = 20.0) -> Optional[str]:
    """Text from a URL, cached on disk. None when unreachable and not cached."""
    p = os.path.join(cache_dir, "macro", name) if cache_dir else None
    if p and os.path.exists(p) and (time.time() - os.path.getmtime(p)) / 86400 <= max_age_days:
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        text = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text
    except Exception:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                return fh.read()
        return None


# ----------------------------------------------------------------------
# Currency
# ----------------------------------------------------------------------
def currency_of(ticker: str) -> str:
    return FX.parse_currency(FX.guess_currency(ticker))[0]


def usdcad_series(prices: PriceSet) -> Tuple[pd.Series, List[str]]:
    """CAD per USD, daily: Yahoo CAD=X, extended back with the stored history."""
    notes: List[str] = []
    live = prices.adj("CAD=X")
    rates = pd.DataFrame({"USD": live}) if len(live) else pd.DataFrame({"USD": pd.Series(dtype=float)})
    rates, hist_notes = FX.extend_with_history(rates, "CAD")
    notes.extend(hist_notes)
    s = rates["USD"].dropna().sort_index()
    return s, notes


def to_cad(px: pd.Series, ticker: str, usdcad: pd.Series) -> pd.Series:
    """A price series expressed in CAD (unchanged when quoted in CAD)."""
    ccy = currency_of(ticker)
    px = px.dropna()
    if ccy == "CAD":
        return px
    if ccy != "USD":
        raise ValueError(f"{ticker} is quoted in {ccy}; only CAD and USD are supported "
                         f"by the Security review.")
    r = usdcad.reindex(usdcad.index.union(px.index)).ffill(limit=FX_LIMIT).reindex(px.index)
    return (px * r).dropna()


# ----------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------
def daily_returns(px: pd.Series) -> pd.Series:
    px = px.dropna()
    return px.pct_change().iloc[1:]


def month_end_prices(px: pd.Series, drop_current: Optional[pd.Timestamp] = None) -> pd.Series:
    """Last price of each calendar month, indexed by month end. The month in
    progress (the month of `drop_current`, default the last price's month if
    it is the current calendar month) is dropped."""
    px = px.dropna()
    if px.empty:
        return px
    m = px.groupby(px.index.to_period("M")).last()
    m.index = m.index.to_timestamp("M")
    today = pd.Timestamp(drop_current) if drop_current is not None else pd.Timestamp.today()
    cur = today.to_period("M").to_timestamp("M")
    return m[m.index < cur]


def monthly_returns(px: pd.Series, drop_current: Optional[pd.Timestamp] = None) -> pd.Series:
    m = month_end_prices(px, drop_current)
    return m.pct_change().iloc[1:]


def splice(primary: pd.Series, proxy: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """Primary returns, with the proxy's returns before the primary's first
    valid date. Returns (spliced series, boolean flag of spliced dates)."""
    p = primary.dropna()
    if proxy is None or proxy.dropna().empty:
        return p, pd.Series(False, index=p.index)
    q = proxy.dropna()
    early = q[q.index < p.index[0]] if len(p) else q
    out = pd.concat([early, p]).sort_index()
    flag = pd.Series(out.index.isin(early.index), index=out.index)
    return out, flag


# ----------------------------------------------------------------------
# Data quality
# ----------------------------------------------------------------------
def quality_flags(ticker: str, px: pd.Series) -> List[str]:
    """Gaps, stale prices and jumps in a daily price series."""
    px = px.dropna()
    out: List[str] = []
    if len(px) < 2:
        return [f"{ticker}: fewer than two prices."]
    r = px.pct_change().dropna()
    big = r[r.abs() > SUSPICIOUS_DAILY]
    if len(big):
        out.append(f"{ticker}: {len(big)} daily move(s) above {SUSPICIOUS_DAILY:.0%} "
                   f"(first {big.index[0].date()}, {big.iloc[0]:+.1%}); check for a bad print.")
    gaps = px.index.to_series().diff().dt.days
    long_gaps = gaps[gaps > 10]
    if len(long_gaps):
        out.append(f"{ticker}: {len(long_gaps)} gap(s) longer than 10 calendar days "
                   f"(first ends {long_gaps.index[0].date()}).")
    zero = (r == 0).astype(int)
    runs = zero.groupby((zero != zero.shift()).cumsum()).cumsum()
    if len(runs) and runs.max() >= STALE_RUN:
        out.append(f"{ticker}: price unchanged for {int(runs.max())} sessions in a row "
                   f"(stale or illiquid quotes).")
    return out


def distribution_yield(frame: pd.DataFrame, years: int = 5) -> Dict[str, float]:
    """Trailing 12-month distributions over the last close, and the
    coefficient of variation of calendar-year yields over the last `years`
    full years (lower = steadier)."""
    if frame is None or "dividends" not in frame or "close" not in frame:
        return {"dist_yield": np.nan, "yield_stability": np.nan}
    close = frame["close"].dropna()
    div = frame["dividends"].fillna(0.0)
    if close.empty:
        return {"dist_yield": np.nan, "yield_stability": np.nan}
    end = close.index[-1]
    ttm = float(div[div.index > end - pd.DateOffset(years=1)].sum())
    dy = ttm / float(close.iloc[-1]) if close.iloc[-1] > 0 else np.nan
    yearly = div.groupby(div.index.year).sum()
    yclose = close.groupby(close.index.year).mean()
    full = [y for y in yearly.index if y < end.year and y > close.index[0].year]
    full = full[-years:]
    ys = (yearly.reindex(full) / yclose.reindex(full)).dropna()
    cv = float(ys.std(ddof=0) / ys.mean()) if len(ys) >= 3 and ys.mean() > 0 else np.nan
    return {"dist_yield": dy, "yield_stability": cv}


# ----------------------------------------------------------------------
# Macro series
# ----------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
VALET_JSON = "https://www.bankofcanada.ca/valet/observations/{series}/json?start_date=1950-01-01"


def parse_fred_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0], errors="coerce"))
    return s[s.index.notna()].dropna().sort_index()


def parse_valet_json(text: str, series: str) -> pd.Series:
    obs = json.loads(text).get("observations", [])
    vals = {}
    for o in obs:
        try:
            vals[pd.Timestamp(o["d"])] = float(o[series]["v"])
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(vals, dtype=float).sort_index()


def cpi_series(sources: List[Dict[str, str]], cache_dir: Optional[str] = CACHE_DIR
               ) -> Tuple[Optional[pd.Series], str]:
    """The first CPI source that answers; (series, label) or (None, reason)."""
    tried = []
    for src in sources:
        kind, ser = src.get("kind"), src.get("series")
        if kind == "fred":
            text = fetch_url_cached(FRED_CSV.format(series=ser), f"fred_{ser}.csv", cache_dir)
            s = parse_fred_csv(text) if text else None
        elif kind == "boc":
            text = fetch_url_cached(VALET_JSON.format(series=ser), f"boc_{ser}.json", cache_dir)
            s = parse_valet_json(text, ser) if text else None
        else:
            s = None
        if s is not None and len(s) > 24:
            return s, src.get("label", f"{kind}:{ser}")
        tried.append(src.get("label", f"{kind}:{ser}"))
    return None, "No CPI source reachable (tried: " + "; ".join(tried) + ")"


def cad_tbill_monthly(cache_dir: Optional[str] = CACHE_DIR) -> Tuple[Optional[pd.Series], str]:
    """Canadian 3-month T-bill yield in percent, monthly average (Bank of Canada V122531)."""
    text = fetch_url_cached(VALET_JSON.format(series="V122531"), "boc_V122531.json", cache_dir)
    if not text:
        return None, "Bank of Canada T-bill unreachable: risk-free set to 0."
    s = parse_valet_json(text, "V122531")
    return (s, "Canada 3-month T-bill (Bank of Canada V122531, monthly average)") if len(s) \
        else (None, "Bank of Canada T-bill empty: risk-free set to 0.")


def rf_monthly(index: pd.DatetimeIndex, tbill_pct: Optional[pd.Series]) -> pd.Series:
    """Per-month risk-free return: each month earns the yield known at its start."""
    if tbill_pct is None or tbill_pct.empty:
        return pd.Series(0.0, index=index)
    y = tbill_pct.copy()
    y.index = y.index.to_period("M").to_timestamp("M")
    return M.rf_from_yield(y, index)
