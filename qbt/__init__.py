"""qbt -- modular backtesting engine.

Usable without the interface:

    from qbt import RunConfig, run_from_config
    cfg = RunConfig.from_yaml(open("configs/dual_momentum.yaml").read())
    res, bench, prices = run_from_config(cfg)
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .config import RunConfig, DataConfig, EngineConfig, CostConfig, StrategyConfig
from .data import load_yfinance, load_file, clean_prices
from .engine import run_backtest, benchmark_result, BacktestResult, align_results
from .strategies import REGISTRY, get as get_strategy, list_strategies
from . import metrics, charts, robustness, report

__version__ = "1.0.0"

__all__ = [
    "RunConfig", "DataConfig", "EngineConfig", "CostConfig", "StrategyConfig",
    "load_yfinance", "load_file", "clean_prices", "run_backtest",
    "benchmark_result", "BacktestResult", "align_results", "REGISTRY",
    "get_strategy", "list_strategies", "metrics", "charts", "robustness",
    "report", "run_from_config",
]


def run_from_config(cfg: RunConfig) -> Tuple[BacktestResult, Optional[BacktestResult], pd.DataFrame]:
    """Full chain driven by YAML: data -> signals -> simulation."""
    tickers = list(cfg.data.tickers)
    for extra in (cfg.data.benchmark, cfg.data.cash_proxy):
        if extra and extra not in tickers:
            tickers.append(extra)

    raw = load_yfinance(tickers, cfg.data.start, cfg.data.end, cfg.data.price_field)
    prices, _ = clean_prices(raw, cfg.data)

    cash_px = prices[cfg.data.cash_proxy] if (
        cfg.data.cash_proxy and cfg.data.cash_proxy in prices.columns) else None
    drop = [c for c in {cfg.data.cash_proxy} if c and c in prices.columns]
    universe = prices.drop(columns=drop)
    universe = universe[[c for c in cfg.data.tickers if c in universe.columns]]

    strat = get_strategy(cfg.strategy.name)
    weights = strat.generate(universe, cfg.strategy.params)
    res = run_backtest(universe, weights, cfg.engine, cfg.costs, cash_px, cfg.label)

    bench = None
    if cfg.data.benchmark and cfg.data.benchmark in prices.columns:
        bench = benchmark_result(prices[cfg.data.benchmark], cfg.engine,
                                 cfg.data.benchmark)
    return res, bench, universe
