"""Currencies: which one each instrument trades in, the exchange rates, and
the conversion of every price series into the strategy's base currency.

A Canadian investor holding XIC.TO and SPY in one account owns two things
priced in different units. Adding them up as if both were dollars is wrong
in two ways: the US position earns the stock's return *plus* the move in
USD/CAD, and getting between the two currencies costs real money.

**How this is built.** The engine still sees one price matrix, in the base
currency. This module converts every foreign series (close, open, high, low,
adjusted close and dividends) into that currency *before* anything else
runs, so signals, benchmarks, attribution and tax all work in the currency
the investor actually lives in. Only two things need more than a price
conversion, and the engine handles those itself with the book kept here:

- **Foreign cash.** A US stock bought or sold is settled in USD. Proceeds,
  dividends and purchases move a USD cash balance; the spread is paid only
  when money crosses between currencies, on the *net* amount. Selling one
  US ETF to buy another costs no conversion at all.
- **The conversion cost** itself, a spread on the amount converted.

**Currency of an instrument means the currency it is quoted in**, not the
currency of what it holds. XUU.TO holds US stocks but trades in CAD, so for
a CAD investor it needs no conversion; the US exposure inside it is already
reflected in its CAD price.

Nothing here touches the network at import time. Exchange rates and the
per-ticker currency lookup go through injectable callables, so the whole
module is testable offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import MarketData

# Currencies quoted in a fraction of the unit: (ISO code, multiplier into it).
# London lists in pence, Tel Aviv in agorot, Johannesburg in cents. A price
# left in the minor unit is wrong by a factor of 100 and nothing downstream
# would notice, so these are normalised here and nowhere else.
_SUBUNITS = {
    "GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01),
    "ILA": ("ILS", 0.01), "ILa": ("ILS", 0.01),
    "ZAc": ("ZAR", 0.01), "ZAC": ("ZAR", 0.01),
}

# What the ticker suffix says, used only when Yahoo cannot be asked.
_SUFFIX_CCY = {
    ".TO": "CAD", ".V": "CAD", ".NE": "CAD", ".CN": "CAD",
    ".L": "GBp", ".IL": "USD",
    ".PA": "EUR", ".DE": "EUR", ".F": "EUR", ".AS": "EUR", ".MI": "EUR",
    ".MC": "EUR", ".BR": "EUR", ".LS": "EUR", ".HE": "EUR", ".VI": "EUR",
    ".IR": "EUR", ".AT": "EUR",
    ".SW": "CHF", ".ST": "SEK", ".OL": "NOK", ".CO": "DKK",
    ".T": "JPY", ".HK": "HKD", ".SS": "CNY", ".SZ": "CNY",
    ".AX": "AUD", ".NZ": "NZD", ".SI": "SGD", ".KS": "KRW", ".KQ": "KRW",
    ".TW": "TWD", ".NS": "INR", ".BO": "INR", ".SA": "BRL", ".MX": "MXN",
    ".TA": "ILA", ".JO": "ZAc",
}
_INDEX_CCY = {
    "^GSPTSE": "CAD", "^FTSE": "GBP", "^N225": "JPY", "^GDAXI": "EUR",
    "^FCHI": "EUR", "^STOXX50E": "EUR", "^HSI": "HKD", "^AXJO": "AUD",
}

# The currencies offered in the interface; any 3-letter code is accepted.

FX_FILL_LIMIT = 5            # sessions an exchange rate may be carried forward
JUMP_WARN = 0.07             # a one-day move in a rate worth flagging


def parse_currency(code: str) -> Tuple[str, float]:
    """(ISO code, multiplier) for whatever a data source calls a currency."""
    code = str(code).strip()
    if code in _SUBUNITS:
        return _SUBUNITS[code]
    return code.upper(), 1.0


def guess_currency(ticker: str) -> str:
    """The quote currency a ticker's shape suggests. A guess, and labelled as
    one wherever it is shown: no suffix is read as a US listing."""
    t = str(ticker).strip().upper()
    if t in _INDEX_CCY:
        return _INDEX_CCY[t]
    if t.startswith("^"):
        return "USD"
    if t.endswith("=X") or t.endswith("=F"):
        return "USD"
    if "-" in t:                                   # crypto: BTC-CAD
        tail = t.rsplit("-", 1)[1]
        if len(tail) == 3 and tail.isalpha():
            return tail
    for suf in sorted(_SUFFIX_CCY, key=len, reverse=True):
        if t.endswith(suf):
            return _SUFFIX_CCY[suf]
    return "USD"


def yahoo_currency(ticker: str) -> Optional[str]:
    """The currency Yahoo Finance reports for a ticker, or None."""
    try:
        import yfinance as yf
        tk = yf.Ticker(str(ticker).strip().upper())
        try:
            code = tk.fast_info["currency"]
        except Exception:
            code = tk.info.get("currency")
        return str(code) if code else None
    except Exception:
        return None


def detect_currencies(tickers: Iterable[str],
                      lookup: Optional[Callable[[str], Optional[str]]] = None
                      ) -> Dict[str, Tuple[str, str]]:
    """{ticker: (currency as the source spells it, where it came from)}."""
    lookup = lookup or yahoo_currency
    out: Dict[str, Tuple[str, str]] = {}
    for t in dict.fromkeys(str(x).strip().upper() for x in tickers if str(x).strip()):
        code = None
        try:
            code = lookup(t)
        except Exception:
            code = None
        if code:
            out[t] = (str(code), "Yahoo Finance")
        else:
            out[t] = (guess_currency(t), "guessed from ticker")
    return out


# ----------------------------------------------------------------------
# Exchange rates
# ----------------------------------------------------------------------
Downloader = Callable[[Sequence[str], Optional[str], Optional[str]], pd.DataFrame]


def _yahoo_download(tickers: Sequence[str], start: Optional[str],
                    end: Optional[str]) -> pd.DataFrame:
    """Daily closes for Yahoo FX tickers, one column per ticker found."""
    import yfinance as yf
    raw = yf.download(tickers=list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False, threads=True,
                      group_by="column")
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in set(raw.columns.get_level_values(0)):
            return pd.DataFrame()
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].copy() if "Close" in raw.columns else pd.DataFrame()
        if not px.empty:
            px.columns = list(tickers)[:1]
    return px


def _clean_rate(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    idx = pd.to_datetime(s.index, utc=True, errors="coerce")
    s = s[~idx.isna()]
    s.index = idx[~idx.isna()].tz_convert(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.where(s > 0).dropna()


def load_rates(currencies: Iterable[str], base: str,
               start: Optional[str] = None, end: Optional[str] = None,
               download: Optional[Downloader] = None
               ) -> Tuple[pd.DataFrame, List[str]]:
    """Daily rates, one column per currency, each in units of `base` per one
    unit of that currency (so a USD amount times the USD column is the CAD
    amount). Tries the direct pair, then the inverse, then a cross through
    USD; the route used is reported."""
    download = download or _yahoo_download
    base = base.upper()
    wanted = [c for c in dict.fromkeys(x.upper() for x in currencies) if c != base]
    notes: List[str] = []
    if not wanted:
        return pd.DataFrame(), notes

    first = (pd.Timestamp(start) - pd.Timedelta(days=15)).strftime("%Y-%m-%d") \
        if start else None
    cache: Dict[str, pd.Series] = {}

    def fetch(names: Sequence[str]) -> None:
        todo = [n for n in names if n not in cache]
        if not todo:
            return
        try:
            df = download(todo, first, end)
        except Exception:
            df = pd.DataFrame()
        for n in todo:
            if df is not None and n in df.columns:
                s = _clean_rate(df[n])
                cache[n] = s
            else:
                cache[n] = pd.Series(dtype=float)

    def pair(a: str, b: str) -> Optional[pd.Series]:
        """Units of b per one unit of a, from Yahoo's a+b=X or its inverse."""
        if a == b:
            return None
        s = cache.get(f"{a}{b}=X")
        if s is not None and len(s):
            return s
        s = cache.get(f"{b}{a}=X")
        if s is not None and len(s):
            return 1.0 / s
        return None

    fetch([f"{c}{base}=X" for c in wanted])
    missing = [c for c in wanted if pair(c, base) is None]
    if missing:
        fetch([f"{base}{c}=X" for c in missing])
        missing = [c for c in wanted if pair(c, base) is None]
    if missing:
        legs = [f"{c}USD=X" for c in missing] + [f"USD{c}=X" for c in missing]
        if base != "USD":
            legs += [f"USD{base}=X", f"{base}USD=X"]
        fetch(legs)

    out: Dict[str, pd.Series] = {}
    for c in wanted:
        direct = pair(c, base)
        if direct is not None:
            out[c] = direct
            continue
        to_usd = pair(c, "USD") if c != "USD" else None
        usd_to_base = pair("USD", base) if base != "USD" else None
        if c == "USD":
            cross = usd_to_base
        elif base == "USD":
            cross = to_usd
        elif to_usd is not None and usd_to_base is not None:
            both = pd.concat([to_usd, usd_to_base], axis=1, join="inner")
            cross = both.iloc[:, 0] * both.iloc[:, 1] if len(both) else None
        else:
            cross = None
        if cross is not None and len(cross):
            out[c] = cross
            notes.append(f"{c}/{base} was built through USD (no direct "
                         f"{c}{base}=X series).")
        else:
            notes.append(f"No exchange rate found for {c} to {base}: "
                         f"instruments priced in {c} cannot be converted.")

    rates = pd.DataFrame(out).sort_index()
    for c in rates.columns:
        s = rates[c].dropna()
        if len(s) > 2:
            jumps = np.log(s).diff().abs()
            bad = jumps[jumps > JUMP_WARN]
            if len(bad):
                when = ", ".join(str(d.date()) for d in bad.index[:3])
                notes.append(f"{c}/{base}: {len(bad)} one-day move(s) above "
                             f"{JUMP_WARN * 100:.0f}% (first on {when}). "
                             f"Check the series for a bad print.")
    return rates, notes


# ----------------------------------------------------------------------
@dataclass
class FxBook:
    """Everything the run needs to know about currencies.

    `rates` holds base-per-foreign-unit, daily. `ccy` maps an instrument to
    its ISO currency, `scale` to the multiplier that takes a minor-unit
    quote (pence) into the major unit, and `source` records how the
    currency was decided so the interface can show it.
    """
    base: str
    ccy: Dict[str, str] = field(default_factory=dict)
    scale: Dict[str, float] = field(default_factory=dict)
    source: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, str] = field(default_factory=dict)       # as spelled ("GBp"), for saving
    rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: List[str] = field(default_factory=list)

    # -- building -------------------------------------------------------
    @classmethod
    def resolve(cls, tickers: Iterable[str], base: str,
                overrides: Optional[Dict[str, str]] = None,
                detected: Optional[Dict[str, Tuple[str, str]]] = None
                ) -> "FxBook":
        """Decides each instrument's currency, with no exchange rates yet.
        An override beats Yahoo, which beats the ticker's shape."""
        base = parse_currency(base)[0]
        overrides = {str(k).strip().upper(): v for k, v in (overrides or {}).items()}
        detected = detected or {}
        book = cls(base=base)
        for t in dict.fromkeys(str(x).strip().upper() for x in tickers
                               if str(x).strip()):
            if t in overrides and str(overrides[t]).strip():
                raw, src = str(overrides[t]).strip(), "set by you"
            elif t in detected:
                raw, src = detected[t]
            else:
                raw, src = guess_currency(t), "guessed from ticker"
            iso, mult = parse_currency(raw)
            book.ccy[t], book.scale[t], book.source[t] = iso, mult, src
            book.raw[t] = raw
        return book

    @classmethod
    def build(cls, tickers: Iterable[str], base: str,
              overrides: Optional[Dict[str, str]] = None,
              start: Optional[str] = None, end: Optional[str] = None,
              detect: Optional[Callable[[str], Optional[str]]] = None,
              download: Optional[Downloader] = None) -> "FxBook":
        """Currencies and exchange rates, ready to convert and to simulate."""
        tickers = list(dict.fromkeys(str(x).strip().upper() for x in tickers
                                     if str(x).strip()))
        det = detect_currencies(
            [t for t in tickers if t not in {str(k).strip().upper()
                                             for k in (overrides or {})}],
            detect)
        book = cls.resolve(tickers, base, overrides, det)
        need = sorted({c for c in book.ccy.values() if c != book.base})
        book.rates, rate_notes = load_rates(need, book.base, start, end, download)
        book.notes.extend(rate_notes)
        return book

    # -- queries --------------------------------------------------------
    def currency_of(self, instrument: str) -> str:
        return self.ccy.get(str(instrument).strip().upper(),
                            self.ccy.get(instrument, self.base))

    def foreign_currencies(self, instruments: Iterable[str]) -> List[str]:
        """Currencies other than the base among these instruments, sorted."""
        return sorted({self.currency_of(a) for a in instruments} - {self.base})

    def is_active(self, instruments: Iterable[str]) -> bool:
        return bool(self.foreign_currencies(instruments))

    def aligned(self, index: pd.DatetimeIndex,
                currencies: Sequence[str]) -> np.ndarray:
        """(len(index), len(currencies)) rates: carried forward from the last
        observation, never from a later one. Leading dates before a series
        starts take its first value; no position can exist there, since the
        prices it would convert are missing too."""
        cols = []
        for c in currencies:
            if c not in self.rates.columns:
                raise ValueError(f"No exchange rate available for {c} "
                                 f"to {self.base}.")
            s = self.rates[c].dropna()
            s = s.reindex(s.index.union(index)).ffill().reindex(index)
            s = s.bfill()
            if s.isna().any():
                raise ValueError(f"The {c}/{self.base} rate is empty over "
                                 f"this period.")
            cols.append(s.to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.ones((len(index), 0))

    def summary(self, instruments: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """One row per instrument, for display."""
        rows = []
        keys = [str(i).strip().upper() for i in instruments] if instruments is not None \
            else list(self.ccy)
        for t in keys:
            c = self.ccy.get(t, self.base)
            mult = self.scale.get(t, 1.0)
            rows.append({
                "Instrument": t,
                "Quoted in": c + (" (minor units, x%g)" % mult if mult != 1.0 else ""),
                "How decided": self.source.get(t, "assumed base"),
                "Converted": "yes" if c != self.base or mult != 1.0 else "no",
            })
        return pd.DataFrame(rows)

    # -- conversion -----------------------------------------------------
    def _factor(self, instrument: str, index: pd.DatetimeIndex, lag: int = 0):
        """Base per one unit of the instrument's quote, on each date of
        `index`. `lag` > 0 takes the rate of that many sessions earlier: the
        only rate that is known when a trade is placed at the open."""
        key = str(instrument).strip().upper()
        code = self.ccy.get(key, self.base)
        mult = self.scale.get(key, 1.0)
        if code == self.base:
            return mult
        if code not in self.rates.columns:
            return np.nan
        s = self.rates[code].dropna()
        s = s.reindex(s.index.union(index)).ffill(limit=FX_FILL_LIMIT).reindex(index)
        if lag:
            late = s.shift(lag)
            late.iloc[:lag] = s.iloc[:lag]      # no earlier session exists: nothing to trade yet
            s = late
        return s * mult

    def convert_frame(self, df: Optional[pd.DataFrame],
                      skip: Iterable[str] = (), lag: int = 0
                      ) -> Optional[pd.DataFrame]:
        """A price (or dividend) frame expressed in the base currency."""
        if df is None:
            return None
        skip = {str(s).strip().upper() for s in skip}
        out = df.copy()
        for col in out.columns:
            key = str(col).strip().upper()
            if key in skip:
                continue
            f = self._factor(key, out.index, lag)
            if isinstance(f, float) and f == 1.0:
                continue                         # base currency: untouched, bit for bit
            out[col] = out[col] * f
        return out

    def convert_market(self, market: MarketData,
                       skip: Iterable[str] = (),
                       at_open: bool = False) -> MarketData:
        """A MarketData with every price field in the base currency.

        Volume is a count of shares and is left alone. `skip` names series
        to leave in their own currency: a foreign cash proxy is a *rate*,
        whose own-currency return is what cash in that currency earns.

        `at_open` says trades execute at the open. Opening prices, and the
        dividends that go ex at the open, are then converted at the previous
        session's closing rate -- the latest one that exists when the order
        is placed -- instead of the same day's close. Closing prices always
        use the same day's close."""
        skip = list(skip)
        conv = lambda f: self.convert_frame(f, skip)
        conv_open = lambda f: self.convert_frame(f, skip, 1 if at_open else 0)
        notes = list(market.notes or [])
        foreign = sorted({c for t, c in self.ccy.items()
                          if c != self.base and t in set(
                              str(x).strip().upper() for x in market.close.columns)})
        if foreign:
            notes.append(
                f"Prices in {', '.join(foreign)} are converted to {self.base} "
                f"at each day's exchange rate"
                + ("; fills at the open use the previous close's rate, the "
                   "only one known then." if at_open else "."))
        notes.extend(self.notes)
        return MarketData(
            close=conv(market.close), open=conv_open(market.open),
            dividends=conv_open(market.dividends), adj_close=conv(market.adj_close),
            high=conv(market.high), low=conv(market.low),
            volume=market.volume, adjusted=market.adjusted, notes=notes)
