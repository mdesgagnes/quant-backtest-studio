"""Robustness tests.

A single backtest proves nothing. This module attacks the result from four
angles: stability over time, stability across parameters, sensitivity to
frictions, and sampling uncertainty.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CostConfig, EngineConfig
from .engine import run_backtest, BacktestResult
from . import metrics as M


RunFn = Callable[[pd.DataFrame, Dict[str, Any], EngineConfig, CostConfig], BacktestResult]


def _weights_for(strategy, prices, params, exog, weights):
    """Weights supplied directly (imported mode) or produced by the strategy."""
    if weights is not None:
        return weights
    return strategy.generate(prices, params, exog)


def fold_stats(returns: pd.Series, n_folds: int, ppy: int) -> pd.DataFrame:
    """Splits a return series into folds and summarizes each one."""
    idx = returns.index
    rows = []
    for i, seg in enumerate(np.array_split(np.arange(len(idx)), max(2, int(n_folds))), 1):
        if len(seg) < 30:
            continue
        sl = idx[seg]
        r = returns.loc[sl]
        eq = M.to_equity(r)
        vol = M.annual_vol(r, ppy)
        rows.append({
            "Fold": f"{i}", "Start": sl[0].date(), "End": sl[-1].date(),
            "CAGR": M.cagr(eq, ppy), "Volatility": vol,
            # A riskless fold (portfolio in cash) has no meaningful ratio:
            # NaN instead of a misleading number.
            "Sharpe": M.sharpe(r, ppy=ppy) if vol > 0.002 else np.nan,
            "Max Drawdown": M.max_drawdown(eq),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["sharpe_dispersion"] = float(df["Sharpe"].std(ddof=1))
        df.attrs["sharpe_min"] = float(df["Sharpe"].min())
    return df


def _stats(res: BacktestResult, ppy: int) -> Dict[str, float]:
    return {
        "CAGR": M.cagr(res.equity, ppy),
        "Volatility": M.annual_vol(res.returns, ppy),
        "Sharpe": M.sharpe(res.returns, ppy=ppy),
        "Sortino": M.sortino(res.returns, ppy=ppy),
        "Calmar": M.calmar(res.equity, ppy),
        "Max Drawdown": M.max_drawdown(res.equity),
        "Annual Turnover": float(res.turnover.sum() / max(len(res.returns) / ppy, 1e-9)),
    }


# ----------------------------------------------------------------------
def parameter_sweep(prices: pd.DataFrame, strategy, base_params: Dict[str, Any],
                    grid: Dict[str, List[Any]], engine: EngineConfig,
                    costs: CostConfig, cash_prices: Optional[pd.Series] = None,
                    max_runs: int = 400,
                    exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Sweeps a parameter grid. A flat surface is worth more than a sharp peak."""
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))[:max_runs]
    rows = []
    for combo in combos:
        p = dict(base_params)
        p.update(dict(zip(keys, combo)))
        try:
            w = strategy.generate(prices, p, exog)
            res = run_backtest(prices, w, engine, costs, cash_prices)
            row = dict(zip(keys, combo))
            row.update(_stats(res, engine.periods_per_year))
            rows.append(row)
        except Exception as exc:  # an invalid combination must not halt the sweep
            row = dict(zip(keys, combo))
            row["error"] = str(exc)[:120]
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def walk_forward(prices: pd.DataFrame, strategy, params: Dict[str, Any],
                 engine: EngineConfig, costs: CostConfig,
                 n_folds: int = 5,
                 cash_prices: Optional[pd.Series] = None,
                 exog: Optional[pd.DataFrame] = None,
                 weights: Optional[pd.DataFrame] = None,
                 rebalance_dates=None) -> pd.DataFrame:
    """Splits the history into successive folds and measures stability.

    Signals are generated over the full history, then evaluated fold by
    fold: what is tested here is the stability of the behavior, not
    re-optimization. A sharp gap between folds signals a dependency on
    market regime.
    """
    w = _weights_for(strategy, prices, params, exog, weights)
    res = run_backtest(prices, w, engine, costs, cash_prices,
                       rebalance_dates=rebalance_dates)
    return fold_stats(res.returns, n_folds, engine.periods_per_year)


def in_out_sample(prices: pd.DataFrame, strategy, params: Dict[str, Any],
                  engine: EngineConfig, costs: CostConfig, split: float = 0.6,
                  cash_prices: Optional[pd.Series] = None,
                  exog: Optional[pd.DataFrame] = None,
                  weights: Optional[pd.DataFrame] = None,
                  rebalance_dates=None) -> pd.DataFrame:
    """Compares the first portion of the history to the last."""
    w = _weights_for(strategy, prices, params, exog, weights)
    res = run_backtest(prices, w, engine, costs, cash_prices,
                       rebalance_dates=rebalance_dates)
    cut = int(len(res.returns) * split)
    parts = {"In-sample": res.returns.iloc[:cut],
             "Out-of-sample": res.returns.iloc[cut:]}
    rows = []
    for name, r in parts.items():
        if len(r) < 30:
            continue
        eq = M.to_equity(r)
        rows.append({
            "Period": name, "Start": r.index[0].date(), "End": r.index[-1].date(),
            "CAGR": M.cagr(eq, engine.periods_per_year),
            "Sharpe": M.sharpe(r, ppy=engine.periods_per_year),
            "Max Drawdown": M.max_drawdown(eq),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def cost_sensitivity(prices: pd.DataFrame, strategy, params: Dict[str, Any],
                     engine: EngineConfig, costs: CostConfig,
                     levels_bps: Optional[List[float]] = None,
                     cash_prices: Optional[pd.Series] = None,
                     exog: Optional[pd.DataFrame] = None,
                     weights: Optional[pd.DataFrame] = None,
                     rebalance_dates=None) -> pd.DataFrame:
    """At what level of frictions does the strategy stop paying off?"""
    levels = levels_bps or [0, 5, 10, 20, 30, 50, 75, 100]
    w = _weights_for(strategy, prices, params, exog, weights)
    rows = []
    for lv in levels:
        c = CostConfig(commission_bps=0.0, slippage_bps=float(lv),
                       cash_rate_pa=costs.cash_rate_pa,
                       borrow_rate_pa=costs.borrow_rate_pa)
        res = run_backtest(prices, w, engine, c, cash_prices,
                           rebalance_dates=rebalance_dates)
        rows.append({
            "Costs (bps round-trip)": lv,
            "CAGR": M.cagr(res.equity, engine.periods_per_year),
            "Sharpe": M.sharpe(res.returns, ppy=engine.periods_per_year),
            "Max Drawdown": M.max_drawdown(res.equity),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def monte_carlo(returns: pd.Series, n_sims: int = 500, block: int = 21,
                ppy: int = 252, seed: int = 42) -> Dict[str, Any]:
    """Block resampling: preserves short-term autocorrelation.

    Answers the question: how much of this result depends on the exact
    order in which returns occurred?
    """
    r = returns.dropna().to_numpy()
    if len(r) < block * 3:
        return {"paths": pd.DataFrame(), "stats": pd.DataFrame()}
    rng = np.random.default_rng(seed)
    n = len(r)
    n_blocks = int(np.ceil(n / block))

    paths = np.zeros((n_sims, n))
    for s in range(n_sims):
        starts = rng.integers(0, n - block, size=n_blocks)
        sim = np.concatenate([r[st:st + block] for st in starts])[:n]
        paths[s] = sim

    equity = np.cumprod(1 + paths, axis=1)
    finals = equity[:, -1]
    years = n / ppy
    cagrs = finals ** (1 / years) - 1
    peaks = np.maximum.accumulate(equity, axis=1)
    mdds = (equity / peaks - 1).min(axis=1)
    sharpes = paths.mean(axis=1) / paths.std(axis=1, ddof=1) * np.sqrt(ppy)

    pct = [5, 25, 50, 75, 95]
    stats = pd.DataFrame({
        "Percentile": [f"{p}th" for p in pct],
        "CAGR": np.percentile(cagrs, pct),
        "Max Drawdown": np.percentile(mdds, pct),
        "Sharpe": np.percentile(sharpes, pct),
    })

    bands = pd.DataFrame(
        {f"p{p}": np.percentile(equity, p, axis=0) for p in pct},
        index=returns.dropna().index,
    )
    return {
        "paths": bands,
        "stats": stats,
        "prob_loss": float((finals < 1).mean()),
        "prob_dd_20": float((mdds < -0.20).mean()),
        "median_cagr": float(np.median(cagrs)),
    }


# ----------------------------------------------------------------------
def deflated_sharpe_note(sharpe_obs: float, n_trials: int, n_obs: int) -> Dict[str, float]:
    """Approximate Sharpe adjustment for the number of trials (Bailey &
    Lopez de Prado). A reminder that a Sharpe reached after 200 trials is
    not comparable to a Sharpe reached on the first try."""
    if n_trials < 2 or n_obs < 30 or not np.isfinite(sharpe_obs):
        return {"expected_max_sharpe": np.nan, "haircut": np.nan}
    euler = 0.5772156649
    e_max_z = ((1 - euler) * _norm_ppf(1 - 1 / n_trials)
               + euler * _norm_ppf(1 - 1 / (n_trials * np.e)))
    exp_max_sr = e_max_z / np.sqrt(n_obs) * np.sqrt(252)
    return {"expected_max_sharpe": float(exp_max_sr),
            "haircut": float(sharpe_obs - exp_max_sr)}


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam algorithm), no scipy dependency."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
