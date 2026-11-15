"""Everything the decision engine reads, fetched once and reused.

`load_bundle` downloads (or reads from the disk cache) every ticker a run
can need: candidates, sleeve benchmarks, profile building blocks and their
proxies, factor ETFs, regime series, USD/CAD. The scenario matrix calls it
once for the whole universe; the app calls it per run (cached).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


def load_bundle(cfg: AllConfig, candidates: Iterable[str] = (), extra: Iterable[str] = (),
                downloader: Optional[D.Downloader] = None,
                cache_dir: Optional[str] = D.CACHE_DIR,
                as_of: Optional[pd.Timestamp] = None,
                macro: bool = True) -> Bundle:
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
    tb, rf_label = D.cad_tbill_monthly(cache_dir) if macro else (None, "risk-free set to 0 (offline)")
    rf = D.rf_monthly(mdf.index, tb)
    cpi, cpi_label = (None, "offline")
    if macro:
        src = (cfg.scenarios["regimes"].get("high_inflation") or {}).get("sources") or []
        cpi, cpi_label = D.cpi_series(src, cache_dir)
        if cpi is None:
            notes.append(cpi_label + ": the inflation regime is unavailable.")
    return Bundle(mdf, daily_px, udf, fxm, rf, rf_label, raw_px, yields, flags,
                  ps.failed, notes, cpi, cpi_label, as_of)
