"""Datasets and strategy runs for the lab.

A `Dataset` is the frozen input of a study: one price panel (adjusted,
total-return, cleaned by the same `clean_prices` the app uses), an optional
benchmark and cash proxy, and a content hash. The hash is what makes a run
reproducible -- a result is only comparable to another if both saw exactly the
same prices, and Yahoo revises history.

Runs go through the unmodified engine: `Strategy.generate` for weights,
`run_backtest` for the simulation, `trim_warmup` for the record. The same
three calls `qbt.robustness` makes, so a lab run and a sweep point agree.
"""
from __future__ import annotations

import hashlib
import io
import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import CostConfig, DataConfig, EngineConfig
from ..data import clean_prices, load_yfinance
from ..engine import BacktestResult, benchmark_result, run_backtest, trim_warmup
from ..strategies import get as get_strategy


@dataclass
class Dataset:
    prices: pd.DataFrame                      # the investable universe
    benchmark: Optional[pd.Series] = None
    cash: Optional[pd.Series] = None
    source: str = "yfinance"
    note: str = ""
    quality_warnings: List[str] = field(default_factory=list)

    @property
    def tickers(self) -> List[str]:
        return list(self.prices.columns)

    @property
    def start(self) -> str:
        return str(self.prices.index[0].date())

    @property
    def end(self) -> str:
        return str(self.prices.index[-1].date())

    def content_hash(self) -> str:
        """SHA-256 of the numbers actually used. Rounded to 8 decimals so a
        float formatting difference between machines cannot change it."""
        h = hashlib.sha256()
        frames = [self.prices]
        if self.benchmark is not None:
            frames.append(self.benchmark.to_frame("__benchmark__"))
        if self.cash is not None:
            frames.append(self.cash.to_frame("__cash__"))
        for f in frames:
            buf = io.StringIO()
            f.round(8).to_csv(buf, date_format="%Y-%m-%d")
            h.update(buf.getvalue().encode("utf-8"))
        return h.hexdigest()

    def to_frame(self) -> pd.DataFrame:
        """One frame holding everything (for snapshotting to disk)."""
        out = self.prices.copy()
        if self.benchmark is not None:
            out["__benchmark__"] = self.benchmark
        if self.cash is not None:
            out["__cash__"] = self.cash
        return out

    @classmethod
    def from_frame(cls, df: pd.DataFrame, source: str = "snapshot", note: str = "") -> "Dataset":
        bench = df["__benchmark__"] if "__benchmark__" in df else None
        cash = df["__cash__"] if "__cash__" in df else None
        px = df.drop(columns=[c for c in ("__benchmark__", "__cash__") if c in df])
        return cls(prices=px, benchmark=bench, cash=cash, source=source, note=note)


def load_dataset(tickers: List[str], start: str, end: Optional[str] = None,
                 benchmark: Optional[str] = None, cash_proxy: Optional[str] = None,
                 min_history: int = 60) -> Dataset:
    """Downloads and cleans total-return prices for a universe."""
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    extras = [x.strip().upper() for x in (benchmark, cash_proxy) if x and x.strip()]
    want = list(dict.fromkeys(tickers + extras))
    raw = load_yfinance(want, start, end, "Close")
    cfg = DataConfig(tickers=want, start=start, end=end, min_history=min_history,
                     adjusted=True, use_dividends=False)
    px, quality = clean_prices(raw, cfg)
    bench = px[benchmark.strip().upper()] if benchmark and benchmark.strip().upper() in px else None
    cash = px[cash_proxy.strip().upper()] if cash_proxy and cash_proxy.strip().upper() in px else None
    universe = px[[t for t in tickers if t in px.columns]]
    return Dataset(prices=universe, benchmark=bench, cash=cash, source="yfinance",
                   note="total-return (dividend-adjusted) closes",
                   quality_warnings=list(getattr(quality, "warnings", []) or []))


def synthetic_dataset(n_assets: int = 8, years: float = 8.0, seed: int = 7,
                      edge: float = 0.0) -> Dataset:
    """Reproducible fake prices, for tests and the offline demo.

    A common market factor plus idiosyncratic noise. `edge` > 0 plants a
    genuine 12-month momentum effect (winners keep a small drift), so tools
    that should detect an edge can be tested against one that exists.
    """
    rng = np.random.default_rng(seed)
    n = int(252 * years)
    idx = pd.bdate_range("2014-01-02", periods=n)
    mkt = rng.normal(0.0003, 0.009, n)
    rets = np.empty((n, n_assets))
    drift = np.zeros(n_assets)
    for t in range(n):
        if edge and t > 252 and t % 21 == 0:
            past = rets[t - 252:t].sum(axis=0)
            drift = edge * (past.argsort().argsort() - (n_assets - 1) / 2) / n_assets
        rets[t] = 0.8 * mkt[t] + rng.normal(0.0001, 0.011, n_assets) + drift / 21.0
    px = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                      columns=[f"S{i + 1}" for i in range(n_assets)])
    bench = pd.Series(100 * np.cumprod(1 + mkt), index=idx, name="MKT")
    return Dataset(prices=px, benchmark=bench, source="synthetic",
                   note=f"synthetic, seed={seed}, edge={edge}")


# ----------------------------------------------------------------------
def run_strategy(ds: Dataset, key: str, params: Optional[Dict[str, Any]] = None,
                 engine: Optional[EngineConfig] = None, costs: Optional[CostConfig] = None,
                 label: Optional[str] = None, weights: Optional[pd.DataFrame] = None,
                 ) -> BacktestResult:
    """One backtest on a dataset, recorded the way the app records it."""
    engine = engine or EngineConfig()
    costs = costs or CostConfig()
    if weights is None:
        weights = get_strategy(key).generate(ds.prices, params or {}, None, ds.cash)
    res = run_backtest(ds.prices, weights, engine, costs, ds.cash, label or key)
    if engine.trim_warmup:
        res = trim_warmup(res, None, engine.initial_capital)
    return res


def run_benchmark(ds: Dataset, engine: Optional[EngineConfig] = None,
                  start: Optional[pd.Timestamp] = None) -> Optional[BacktestResult]:
    if ds.benchmark is None:
        return None
    return benchmark_result(ds.benchmark, engine or EngineConfig(), "Benchmark", start=start)


def sweep_returns(ds: Dataset, key: str, base_params: Dict[str, Any],
                  grid: Dict[str, List[Any]], engine: Optional[EngineConfig] = None,
                  costs: Optional[CostConfig] = None, max_runs: int = 200) -> pd.DataFrame:
    """Daily net returns of every parameter combination, on a common index.

    The same simulation `robustness.parameter_sweep` runs, but keeping the
    return streams rather than a summary row, which is what overfitting
    statistics (PBO, CPCV) need. All streams are cut to the latest first
    invested day so they cover identical dates.
    """
    engine = engine or EngineConfig()
    costs = costs or CostConfig()
    strat = get_strategy(key)
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))[:max_runs]
    cols: Dict[str, pd.Series] = {}
    for combo in combos:
        p = dict(base_params)
        p.update(dict(zip(keys, combo)))
        name = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
        try:
            w = strat.generate(ds.prices, p, None, ds.cash)
            res = run_backtest(ds.prices, w, engine, costs, ds.cash, name)
            res = trim_warmup(res, None, engine.initial_capital) if engine.trim_warmup else res
            cols[name] = res.returns
        except Exception:
            continue
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    first = df.apply(lambda s: s.first_valid_index()).max()
    return df.loc[first:].dropna(how="any")


def expand_grid(grid: Dict[str, List[Any]]) -> int:
    n = 1
    for v in grid.values():
        n *= max(1, len(v))
    return n
