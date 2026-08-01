"""Backtest engine.

Simulation model, explicit and verifiable:

1. The strategy produces target weights for each day, from information
   available at that day's close.
2. The engine shifts these weights by `execution_lag` business days. With the
   default of 1, a Monday-evening signal executes on Tuesday. No look-ahead
   bias is possible.
3. Between rebalances, weights **drift** with asset returns: an implicit
   daily rebalance is never assumed, the classic mistake that inflates
   backtested results.
4. On rebalance dates, turnover is charged:
   cost = sum(|target weight - current weight|) x (commission + slippage).
5. The uninvested portion earns the cash rate, or tracks the return of a
   cash proxy asset (e.g. PSA.TO) if supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig, CostConfig

_RESAMPLE = {"D": None, "W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}


@dataclass
class BacktestResult:
    equity: pd.Series             # portfolio value
    returns: pd.Series            # net daily returns
    gross_returns: pd.Series      # before frictions
    weights: pd.DataFrame         # effective weights held (post-drift)
    target_weights: pd.DataFrame  # weights targeted at execution
    turnover: pd.Series           # one-way turnover per day
    costs: pd.Series              # daily cost, as a fraction of value
    exposure: pd.Series           # sum of risky weights
    cash_weight: pd.Series
    rebalance_dates: pd.DatetimeIndex
    trades: pd.DataFrame
    label: str = "Backtest"

    @property
    def nav(self) -> pd.Series:
        return self.equity


# ----------------------------------------------------------------------
def rebalance_calendar(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """Last available business day of each period."""
    if rule == "D" or _RESAMPLE.get(rule) is None:
        return pd.DatetimeIndex(index)
    s = pd.Series(index, index=index)
    dates = s.resample(_RESAMPLE[rule]).last().dropna()
    return pd.DatetimeIndex(sorted(set(dates.values))).intersection(index)


def _cash_returns(index: pd.DatetimeIndex, cash_prices: Optional[pd.Series],
                  rate_pa: float, ppy: int) -> pd.Series:
    if cash_prices is not None and len(cash_prices.dropna()) > 1:
        return cash_prices.reindex(index).ffill().pct_change().fillna(0.0)
    return pd.Series(rate_pa / ppy, index=index)


# ----------------------------------------------------------------------
def run_backtest(prices: pd.DataFrame,
                 target_weights: pd.DataFrame,
                 engine: EngineConfig,
                 costs: CostConfig,
                 cash_prices: Optional[pd.Series] = None,
                 label: str = "Backtest",
                 rebalance_dates: Optional[pd.DatetimeIndex] = None) -> BacktestResult:
    """Simulates the portfolio day by day. Returns all diagnostics.

    `rebalance_dates` overrides the periodic calendar when dates are imposed
    externally (e.g. an imported target-weights file). These dates are
    interpreted as *signal* dates and shifted by `execution_lag`, exactly
    like a built-in strategy's signals.
    """
    prices = prices.sort_index()
    idx = prices.index
    assets = list(prices.columns)

    w_target = target_weights.reindex(index=idx, columns=assets).fillna(0.0)

    # --- Leverage control ---------------------------------------------
    gross = w_target.abs().sum(axis=1)
    over = gross > engine.max_leverage
    if over.any():
        w_target.loc[over] = w_target.loc[over].div(gross[over], axis=0) * engine.max_leverage

    # --- Execution lag: this is where the future is neutralized -------
    w_exec = w_target.shift(max(0, int(engine.execution_lag))).fillna(0.0)

    if rebalance_dates is not None and len(rebalance_dates):
        sig = pd.DatetimeIndex(rebalance_dates).intersection(idx)
        pos = idx.get_indexer(sig) + max(0, int(engine.execution_lag))
        rebal = idx[pos[pos < len(idx)]]
    else:
        rebal = rebalance_calendar(idx, engine.rebalance)
    rebal_set = set(rebal)

    asset_ret = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cash_ret = _cash_returns(idx, cash_prices, costs.cash_rate_pa, engine.periods_per_year)
    borrow_daily = costs.borrow_rate_pa / engine.periods_per_year
    fric = (costs.commission_bps + costs.slippage_bps) / 10_000.0
    min_trade = float(engine.min_trade_weight)

    n = len(idx)
    m = len(assets)
    R = asset_ret.to_numpy(dtype=float)
    T = w_exec.to_numpy(dtype=float)
    Rc = cash_ret.to_numpy(dtype=float)

    nav = np.zeros(n)
    gross_r = np.zeros(n)
    net_r = np.zeros(n)
    turn = np.zeros(n)
    cost_arr = np.zeros(n)
    W = np.zeros((n, m))
    cash_w = np.zeros(n)

    w = np.zeros(m)          # current risky weights
    c = 1.0                  # current cash weight
    value = float(engine.initial_capital)
    trades = []

    for i, date in enumerate(idx):
        # 1) Value evolves over the day
        if i == 0:
            r_port = 0.0
        else:
            grown = w * (1.0 + R[i])
            cash_grown = c * (1.0 + Rc[i])
            lev = max(0.0, np.abs(w).sum() - 1.0)
            total = grown.sum() + cash_grown - lev * borrow_daily
            r_port = total - (w.sum() + c)
            w, c = grown, cash_grown - lev * borrow_daily

        gross_r[i] = r_port
        value *= (1.0 + r_port)

        # Renormalize: weights are expressed as a fraction of value
        base = w.sum() + c
        if base > 0:
            w, c = w / base, c / base

        # 2) Rebalance, if any, at the close
        cost_i = 0.0
        if date in rebal_set and i >= int(engine.execution_lag):
            tgt = T[i]
            delta = tgt - w
            # Micro-adjustments are not worth their execution cost
            delta[np.abs(delta) < min_trade] = 0.0
            if np.any(delta):
                new_w = w + delta
                # The filter above can leave a residual position and push the
                # trade over budget: scale back to the allowed leverage.
                gross_new = np.abs(new_w).sum()
                if gross_new > engine.max_leverage:
                    new_w = new_w / gross_new * engine.max_leverage
                    delta = new_w - w
                tr = float(np.abs(delta).sum())
                cost_i = tr * fric
                turn[i] = tr
                value *= (1.0 - cost_i)
                for j, a in enumerate(assets):
                    if delta[j] != 0.0:
                        trades.append({
                            "Date": date, "Instrument": a,
                            "Weight Before": w[j], "Weight After": new_w[j],
                            "Change": delta[j],
                        })
                w = new_w
                c = 1.0 - w.sum()

        cost_arr[i] = cost_i
        net_r[i] = (1.0 + r_port) * (1.0 - cost_i) - 1.0
        nav[i] = value
        W[i] = w
        cash_w[i] = c

    weights = pd.DataFrame(W, index=idx, columns=assets)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Date", "Instrument", "Weight Before", "Weight After", "Change"])

    return BacktestResult(
        equity=pd.Series(nav, index=idx, name=label),
        returns=pd.Series(net_r, index=idx, name=label),
        gross_returns=pd.Series(gross_r, index=idx, name=label),
        weights=weights,
        target_weights=w_exec,
        turnover=pd.Series(turn, index=idx),
        costs=pd.Series(cost_arr, index=idx),
        exposure=weights.abs().sum(axis=1),
        cash_weight=pd.Series(cash_w, index=idx),
        rebalance_dates=rebal,
        trades=trades_df,
        label=label,
    )


# ----------------------------------------------------------------------
def benchmark_result(bench_prices: pd.Series, engine: EngineConfig,
                     label: str = "Benchmark") -> BacktestResult:
    """Buy-and-hold on the index, no frictions, for comparison."""
    px = bench_prices.dropna().to_frame()
    w = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    zero_costs = CostConfig(commission_bps=0, slippage_bps=0,
                            cash_rate_pa=0, borrow_rate_pa=0)
    eng = EngineConfig(initial_capital=engine.initial_capital, rebalance="A",
                       execution_lag=0, max_leverage=1.0,
                       periods_per_year=engine.periods_per_year)
    return run_backtest(px, w, eng, zero_costs, label=label)


def align_results(results: Dict[str, "BacktestResult | pd.Series"]) -> pd.DataFrame:
    """Value curves rebased to 100 over the common period.

    Accepts either BacktestResult objects or raw value series.
    """
    eq = pd.DataFrame({
        k: (v.equity if isinstance(v, BacktestResult) else v)
        for k, v in results.items()
    }).dropna()
    if eq.empty:
        return eq
    return eq.div(eq.iloc[0]) * 100.0
