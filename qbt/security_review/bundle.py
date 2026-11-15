"""Everything the decision engine reads, fetched once and reused.

`load_bundle` downloads (or reads from the disk cache) every ticker a run
can need: candidates, sleeve benchmarks, profile building blocks and their
proxies, factor ETFs, regime series, USD/CAD. The scenario matrix calls it
once for the whole universe; the app calls it per run (cached).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data as D
from .config import AllConfig


@dataclass
class Bundle:
    monthly: pd.DataFrame                    # CAD monthly total returns, raw (not spliced)
    daily_px: Dict[str, pd.Series]           # CAD daily adjusted price levels, raw
    usd_monthly: pd.DataFrame                # own-currency monthly returns (factor spreads)
    usdcad_m: pd.Series                      # monthly % change in CAD per USD
    rf: pd.Series                            # monthly risk-free return (CAD T-bill)
    rf_label: str
    raw_px: Dict[str, pd.Series]             # own-currency daily adjusted prices
    yields: Dict[str, Dict[str, float]]
    flags: Dict[str, List[str]]
    failed: List[str]
    notes: List[str]
    cpi: Optional[pd.Series] = None
    cpi_label: str = ""
    as_of: Optional[pd.Timestamp] = None
    spliced_monthly: Dict[str, Tuple[pd.Series, pd.Series, str]] = field(default_factory=dict)
    spliced_daily: Dict[str, pd.Series] = field(default_factory=dict)
    kf: Optional[pd.DataFrame] = None        # Ken French factors (decimals), if reachable
    kf_label: str = ""
    cache_dir: Optional[str] = None
    info_fetcher: Optional[D.InfoFetcher] = None
    fetch_info: bool = True
    _info: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Uploaded return streams: monthly only, no daily prices, no fund data.
    streams: Dict[str, str] = field(default_factory=dict)        # name -> description

    def is_monthly_only(self, t: str) -> bool:
        return t in self.streams

    def info(self, t: str) -> Dict[str, float]:
        """Expense ratio, AUM and average trading volume (yfinance, cached)."""
        if t in self.streams:
            return {}
        if t not in self._info:
            self._info[t] = D.fund_info(t, self.cache_dir, self.info_fetcher)                 if (self.fetch_info or self.info_fetcher) else {}
        return self._info[t]

    def has(self, t: str) -> bool:
        return t in self.monthly.columns and self.monthly[t].notna().sum() > 0

    # -- building blocks with their proxies --------------------------------
    def block_monthly(self, ticker: str, proxy: Optional[str]) -> Tuple[pd.Series, pd.Series, str]:
        """(returns, spliced flag, proxy used) for a holding."""
        key = f"{ticker}|{proxy}"
        if key not in self.spliced_monthly:
            prim = self.monthly[ticker] if ticker in self.monthly else pd.Series(dtype=float)
            prox = self.monthly[proxy] if proxy and proxy in self.monthly else None
            s, f = D.splice(prim, prox)
            self.spliced_monthly[key] = (s, f, proxy or "")
        return self.spliced_monthly[key]

    def block_daily(self, ticker: str, proxy: Optional[str]) -> pd.Series:
        """Daily CAD price level for a holding, spliced onto the proxy's path."""
        key = f"{ticker}|{proxy}"
        if key not in self.spliced_daily:
            prim = D.daily_returns(self.daily_px.get(ticker, pd.Series(dtype=float)))
            prox = D.daily_returns(self.daily_px[proxy]) if proxy and proxy in self.daily_px else None
            r, _ = D.splice(prim, prox)
            self.spliced_daily[key] = (1 + r).cumprod() if len(r) else r
        return self.spliced_daily[key]


def stream_returns(r: pd.Series, currency: str, usdcad_m: pd.Series) -> pd.Series:
    """An uploaded monthly return series in CAD, indexed by calendar month end.
    USD streams are converted with the month's change in CAD per USD."""
    r = pd.Series(r, dtype=float).dropna().sort_index()
    r.index = pd.DatetimeIndex(r.index).to_period("M").to_timestamp("M")
    r = r[~r.index.duplicated(keep="last")]
    if currency.upper() == "USD":
        fx = usdcad_m.reindex(r.index)
        if fx.isna().any():
            raise ValueError("USD/CAD is missing for part of the uploaded period.")
        r = (1 + r) * (1 + fx) - 1
    elif currency.upper() != "CAD":
        raise ValueError("Uploaded returns must be in CAD or USD.")
    return r


def with_streams(b: Bundle, streams: Dict[str, pd.Series], currency: str = "CAD") -> Bundle:
    """A copy of `b` that also holds uploaded monthly return streams, as if
    they were tickers. The original bundle (shared by the app's cache) is
    not modified. A stream's "daily" price is its month-end value path, so
    stress windows fall back to month-end points for it."""
    if not streams:
        return b
    monthly = b.monthly.copy()
    usd_m = b.usd_monthly.copy()
    daily, flags, yields, descr = dict(b.daily_px), dict(b.flags), dict(b.yields), dict(b.streams)
    for name, r in streams.items():
        cad = stream_returns(r, currency, b.usdcad_m)
        if len(cad) < 2:
            raise ValueError(f"{name}: fewer than two monthly returns.")
        monthly = monthly.reindex(monthly.index.union(cad.index))
        monthly[name] = cad
        usd_m = usd_m.reindex(usd_m.index.union(cad.index))
        usd_m[name] = stream_returns(r, "CAD", b.usdcad_m)
        start = cad.index[0] - pd.offsets.MonthEnd(1)
        lvl = pd.concat([pd.Series([1.0], index=[start]), (1 + cad).cumprod()])
        daily[name] = lvl
        flags[name] = [f"{name} is an uploaded monthly return stream: stress windows use month-end "
                       f"points and windows shorter than two months are skipped."]
        yields[name] = {}
        descr[name] = f"uploaded, {currency.upper()}, {cad.index[0]:%Y-%m} to {cad.index[-1]:%Y-%m}"
    return replace(b, monthly=monthly.sort_index(), usd_monthly=usd_m.sort_index(), daily_px=daily,
                   flags=flags, yields=yields, streams=descr, spliced_monthly={}, spliced_daily={},
                   _info=dict(b._info))


def needed_tickers(cfg: AllConfig, candidates: Iterable[str] = (),
                   extra: Iterable[str] = ()) -> List[str]:
    out = list(candidates) + list(extra)
    for b in cfg.profiles.blocks.values():
        out.append(b.ticker)
        if b.proxy:
            out.append(b.proxy)
    for p in cfg.profiles.profiles.values():
        out.extend(p.weights)
    for s in cfg.universe.sleeves.values():
        if s.benchmark:
            out.append(s.benchmark)
    f = cfg.factors
    for spec in list((f.get("factors") or {}).values()) + [f.get("megacap_spread") or {}]:
        out.extend([spec.get("long"), spec.get("short")])
    out.extend([f.get("home_equity"), f.get("usd_market"), "CAD=X"])
    for r in cfg.scenarios["regimes"].values():
        if r.get("series"):
            out.append(r["series"])
    return [str(t).upper() for t in dict.fromkeys(out) if t]


def load_macro(cfg: AllConfig, cache_dir: Optional[str] = D.CACHE_DIR) -> Dict[str, object]:
    """T-bill, CPI and Ken French factors: the same for every run, so the app
    loads them once per server instead of with every set of tickers."""
    tb, rf_label = D.cad_tbill_monthly(cache_dir)
    src = (cfg.scenarios["regimes"].get("high_inflation") or {}).get("sources") or []
    cpi, cpi_label = D.cpi_series(src, cache_dir)
    kf, kf_label = (None, "ETF factor proxies (config/factors.yaml: source)")
    if str(cfg.factors.get("source", "auto")) in ("auto", "ken_french"):
        kf, kf_label = D.ken_french_factors(cache_dir)
    return {"tbill": tb, "rf_label": rf_label, "cpi": cpi, "cpi_label": cpi_label,
            "kf": kf, "kf_label": kf_label}


def load_bundle(cfg: AllConfig, candidates: Iterable[str] = (), extra: Iterable[str] = (),
                downloader: Optional[D.Downloader] = None,
                cache_dir: Optional[str] = D.CACHE_DIR,
                as_of: Optional[pd.Timestamp] = None,
                macro: bool = True,
                info_fetcher: Optional[D.InfoFetcher] = None,
                macro_data: Optional[Dict[str, object]] = None) -> Bundle:
    """Prices for the tickers a run needs, plus the macro series. Pass
    `macro_data` (from `load_macro`) to reuse macro series already loaded."""
    tickers = needed_tickers(cfg, candidates, extra)
    ps = D.fetch_prices(tickers, downloader=downloader, cache_dir=cache_dir)
    notes = list(ps.notes)
    usdcad, fx_notes = D.usdcad_series(ps)
    notes.extend(fx_notes)
    last = max((f.index[-1] for f in ps.frames.values() if len(f)), default=pd.Timestamp.today())
    as_of = pd.Timestamp(as_of) if as_of is not None else last
    monthly, usd_monthly, daily_px, raw_px, flags, yields = {}, {}, {}, {}, {}, {}
    for t, fr in ps.frames.items():
        adj = fr["adj_close"].dropna()
        raw_px[t] = adj
        if t == "CAD=X" or t.startswith("^"):
            continue
        try:
            cad = D.to_cad(adj, t, usdcad)
        except ValueError as exc:
            notes.append(str(exc))
            continue
        daily_px[t] = cad
        monthly[t] = D.monthly_returns(cad, as_of)
        usd_monthly[t] = D.monthly_returns(adj, as_of)
        flags[t] = D.quality_flags(t, adj)
        yields[t] = D.distribution_yield(fr)
    mdf = pd.DataFrame(monthly).sort_index()
    udf = pd.DataFrame(usd_monthly).sort_index()
    fxm = D.monthly_returns(usdcad, as_of) if len(usdcad) else pd.Series(dtype=float)
    if macro and macro_data is None:
        macro_data = load_macro(cfg, cache_dir)
    if macro:
        tb, rf_label = macro_data["tbill"], macro_data["rf_label"]
        cpi, cpi_label = macro_data["cpi"], macro_data["cpi_label"]
        kf, kf_label = macro_data["kf"], macro_data["kf_label"]
        if cpi is None:
            notes.append(cpi_label + ": the inflation regime is unavailable.")
    else:
        tb, rf_label = None, "risk-free set to 0 (offline)"
        cpi, cpi_label = None, "offline"
        kf, kf_label = None, "offline: ETF factor proxies used"
    rf = D.rf_monthly(mdf.index, tb)
    return Bundle(mdf, daily_px, udf, fxm, rf, rf_label, raw_px, yields, flags,
                  ps.failed, notes, cpi, cpi_label, as_of, kf=kf, kf_label=kf_label,
                  cache_dir=cache_dir, info_fetcher=info_fetcher, fetch_info=macro)
