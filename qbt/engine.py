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
6. Optionally, trades execute at the **open** and the portfolio is marked to
   market at the **close**. The day then has two legs: the overnight move
   from the prior close to the open is earned on the old weights, and the
   intraday move from open to close on the new ones.
7. Optionally, **cash dividends** are credited on their ex-date and held in
   the cash bucket until the next rebalance, rather than being assumed
   instantly reinvested. This requires price-return (non dividend-adjusted)
   prices, otherwise the dividend would be counted twice.
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
    dividend_income: Optional[pd.Series] = None   # daily, as a fraction of value
    warmup_start: Optional[pd.Timestamp] = None   # first day actually invested

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
                 rebalance_dates: Optional[pd.DatetimeIndex] = None,
                 open_prices: Optional[pd.DataFrame] = None,
                 dividends: Optional[pd.DataFrame] = None) -> BacktestResult:
    """Simulates the portfolio day by day. Returns all diagnostics.

    `rebalance_dates` overrides the periodic calendar when dates are imposed
    externally (e.g. an imported target-weights file). These dates are
    interpreted as *signal* dates and shifted by `execution_lag`, exactly
    like a built-in strategy's signals.

    `open_prices`, when supplied, switches execution to the open: the
    overnight leg is earned on the pre-trade weights and the intraday leg on
    the post-trade weights. Without it, the whole day is earned on the
    pre-trade weights and the trade happens at the close.

    `dividends` holds per-share cash amounts on their ex-date, in the same
    units as `prices`. They are credited to the cash bucket, where they sit
    until the next rebalance reinvests them. Supply these only alongside
    price-return prices: with dividend-adjusted prices the payment is
    already inside the price series and would be counted twice.
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

    # --- Return legs ---------------------------------------------------
    # Without open prices the overnight leg is empty and the full day is
    # earned before trading, which reproduces close-to-close execution.
    trade_at_open = open_prices is not None
    prev_close = prices.shift(1)
    if trade_at_open:
        op = open_prices.reindex(index=idx, columns=assets)
        # A missing open falls back to the prior close, which collapses that
        # day to close-to-close rather than inventing a price.
        op = op.where(op.notna() & (op > 0), prev_close)
        r_overnight = (op / prev_close - 1.0)
        r_intraday = (prices / op - 1.0)
    else:
        r_overnight = pd.DataFrame(0.0, index=idx, columns=assets)
        r_intraday = (prices / prev_close - 1.0)

    clean = lambda d: d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    R_on = clean(r_overnight).to_numpy(dtype=float)
    R_id = clean(r_intraday).to_numpy(dtype=float)

    # --- Dividends ------------------------------------------------------
    # Expressed as a fraction of the prior close, so the cash credited is
    # weight x (dividend per share / price per share).
    if dividends is not None:
        div_frac = dividends.reindex(index=idx, columns=assets).fillna(0.0) / prev_close
        D = clean(div_frac).to_numpy(dtype=float)
    else:
        D = np.zeros((len(idx), len(assets)))

    cash_ret = _cash_returns(idx, cash_prices, costs.cash_rate_pa, engine.periods_per_year)
    borrow_daily = costs.borrow_rate_pa / engine.periods_per_year
    fric = (costs.commission_bps + costs.slippage_bps) / 10_000.0
    min_trade = float(engine.min_trade_weight)

    n = len(idx)
    m = len(assets)
    T = w_exec.to_numpy(dtype=float)
    Rc = cash_ret.to_numpy(dtype=float)

    nav = np.zeros(n)
    gross_r = np.zeros(n)
    net_r = np.zeros(n)
    turn = np.zeros(n)
    cost_arr = np.zeros(n)
    div_arr = np.zeros(n)
    W = np.zeros((n, m))
    cash_w = np.zeros(n)

    w = np.zeros(m)          # current risky weights
    c = 1.0                  # current cash weight
    value = float(engine.initial_capital)
    trades = []

    def _charge(w_now, tgt, i, date):
        """Applies the target, returns (new weights, cost, turnover)."""
        delta = tgt - w_now
        delta[np.abs(delta) < min_trade] = 0.0
        if not np.any(delta):
            return w_now, 0.0, 0.0
        new_w = w_now + delta
        # The min-trade filter can leave a residual position that pushes the
        # book over budget: scale back to the allowed leverage.
        gross_new = np.abs(new_w).sum()
        if gross_new > engine.max_leverage:
            new_w = new_w / gross_new * engine.max_leverage
            delta = new_w - w_now
        tr = float(np.abs(delta).sum())
        for j, a in enumerate(assets):
            if delta[j] != 0.0:
                trades.append({
                    "Date": date, "Instrument": a,
                    "Weight Before": w_now[j], "Weight After": new_w[j],
                    "Change": delta[j],
                })
        return new_w, tr * fric, tr

    for i, date in enumerate(idx):
        cost_i = 0.0
        div_i = 0.0

        if i == 0:
            r_port = 0.0
        else:
            # 1) Overnight leg, earned on the weights held overnight
            w_start = w.copy()
            w = w * (1.0 + R_on[i])
            # Dividends go ex overnight: credited on the position held, and
            # parked in cash rather than reinvested in the paying asset.
            if D[i].any():
                div_i = float((w_start * D[i]).sum())
            cash_grown = c * (1.0 + Rc[i]) + div_i
            lev = max(0.0, np.abs(w_start).sum() - 1.0)
            cash_grown -= lev * borrow_daily

            # 2) Trade at the open, before the intraday leg
            if trade_at_open and date in rebal_set and i >= int(engine.execution_lag):
                base_open = w.sum() + cash_grown
                if base_open > 0:
                    w_n = w / base_open
                    c_n = cash_grown / base_open
                    new_w, cost_i, tr = _charge(w_n, T[i], i, date)
                    turn[i] = tr
                    w = new_w * base_open * (1.0 - cost_i)
                    cash_grown = (1.0 - new_w.sum()) * base_open * (1.0 - cost_i)

            # 3) Intraday leg, earned on the post-trade weights
            w = w * (1.0 + R_id[i])
            total = w.sum() + cash_grown
            r_port = total - (w_start.sum() + c)
            c = cash_grown

        gross_r[i] = r_port + cost_i if i else 0.0
        div_arr[i] = div_i
        value *= (1.0 + r_port)

        # Renormalize: weights are expressed as a fraction of value
        base = w.sum() + c
        if base > 0:
            w, c = w / base, c / base

        # 4) Close-price execution, when trading at the open is off
        if not trade_at_open and date in rebal_set and i >= int(engine.execution_lag):
            new_w, cost_i, tr = _charge(w, T[i], i, date)
            if tr:
                turn[i] = tr
                value *= (1.0 - cost_i)
                w = new_w
                c = 1.0 - w.sum()

        cost_arr[i] = cost_i
        net_r[i] = (1.0 + r_port) * (1.0 - (cost_i if not trade_at_open else 0.0)) - 1.0
        nav[i] = value
        W[i] = w
        cash_w[i] = c

    weights = pd.DataFrame(W, index=idx, columns=assets)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Date", "Instrument", "Weight Before", "Weight After", "Change"])

    exposure = weights.abs().sum(axis=1)
    active = exposure[exposure > 1e-9]

    return BacktestResult(
        equity=pd.Series(nav, index=idx, name=label),
        returns=pd.Series(net_r, index=idx, name=label),
        gross_returns=pd.Series(gross_r, index=idx, name=label),
        weights=weights,
        target_weights=w_exec,
        turnover=pd.Series(turn, index=idx),
        costs=pd.Series(cost_arr, index=idx),
        exposure=exposure,
        cash_weight=pd.Series(cash_w, index=idx),
        rebalance_dates=rebal,
        trades=trades_df,
        label=label,
        dividend_income=pd.Series(div_arr, index=idx),
        warmup_start=active.index[0] if len(active) else None,
    )


# ----------------------------------------------------------------------
def first_active_date(res: BacktestResult,
                      threshold: float = 1e-9) -> Optional[pd.Timestamp]:
    """First day the portfolio actually holds something."""
    active = res.exposure[res.exposure > threshold]
    return active.index[0] if len(active) else None


def trim_warmup(res: BacktestResult, start: Optional[pd.Timestamp] = None,
                initial_capital: Optional[float] = None) -> BacktestResult:
    """Drops the leading stretch where the strategy had no position yet.

    Indicators need history before they can produce a first signal: a
    200-day moving average is blind for its first 200 sessions. Those
    sessions are not a flat, neutral prologue. The portfolio sits in cash
    and *earns the cash rate*, which quietly lifts the reported return,
    lengthens the measured period, and dilutes volatility and drawdown --
    all of it manufactured by the warm-up rather than by the strategy.

    Trimming starts the record on the first day capital is actually at
    risk. Only the leading run is removed: a strategy that deliberately
    moves to cash mid-period keeps that stretch, because there the cash
    position is a decision rather than an artefact.
    """
    start = start or first_active_date(res)
    if start is None or start <= res.equity.index[0]:
        return res

    idx = res.equity.index
    keep = idx[idx >= start]
    if len(keep) < 2:
        return res

    base = float(res.equity.loc[start])
    cap = float(initial_capital) if initial_capital else base
    equity = res.equity.loc[keep] / base * cap

    returns = res.returns.loc[keep].copy()
    returns.iloc[0] = 0.0          # the first retained day is the new origin
    gross = res.gross_returns.loc[keep].copy()
    gross.iloc[0] = 0.0

    trades = res.trades
    if not trades.empty and "Date" in trades.columns:
        trades = trades[trades["Date"] >= start].reset_index(drop=True)

    return BacktestResult(
        equity=equity, returns=returns, gross_returns=gross,
        weights=res.weights.loc[keep], target_weights=res.target_weights.loc[keep],
        turnover=res.turnover.loc[keep], costs=res.costs.loc[keep],
        exposure=res.exposure.loc[keep], cash_weight=res.cash_weight.loc[keep],
        rebalance_dates=pd.DatetimeIndex([d for d in res.rebalance_dates if d >= start]),
        trades=trades, label=res.label,
        dividend_income=(res.dividend_income.loc[keep]
                         if res.dividend_income is not None else None),
        warmup_start=start,
    )


def align_start(*results: Optional[BacktestResult],
                initial_capital: Optional[float] = None):
    """Trims a set of results to their common first active day.

    The benchmark is invested from day one, so left alone it would be
    credited with the whole warm-up while the strategy sat in cash. Both
    sides have to start on the same date for the comparison to mean
    anything.
    """
    live = [r for r in results if r is not None]
    if not live:
        return list(results)
    starts = [first_active_date(r) for r in live]
    starts = [s for s in starts if s is not None]
    if not starts:
        return list(results)
    common = max(starts)
    return [None if r is None else trim_warmup(r, common, initial_capital)
            for r in results]


# ----------------------------------------------------------------------
def benchmark_result(bench_prices: pd.Series, engine: EngineConfig,
                     label: str = "Benchmark",
                     dividends: Optional[pd.Series] = None) -> BacktestResult:
    """Buy-and-hold on the index, no frictions, for comparison.

    The purchase happens on the first available session. Leaving it to a
    periodic calendar would park the benchmark in cash until the first
    scheduled rebalance -- up to a full year on an annual calendar -- and
    quietly understate the bar the strategy is measured against. After that
    single purchase the weight simply drifts, which is what buy-and-hold
    means.
    """
    px = bench_prices.dropna().to_frame()
    if px.empty:
        raise ValueError("Benchmark series is empty.")
    w = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    zero_costs = CostConfig(commission_bps=0, slippage_bps=0,
                            cash_rate_pa=0, borrow_rate_pa=0)
    eng = EngineConfig(initial_capital=engine.initial_capital, rebalance="D",
                       execution_lag=0, max_leverage=1.0,
                       min_trade_weight=0.0,
                       periods_per_year=engine.periods_per_year)
    div = dividends.dropna().to_frame() if dividends is not None else None
    if div is not None:
        div.columns = px.columns
    return run_backtest(px, w, eng, zero_costs, label=label,
                        rebalance_dates=pd.DatetimeIndex([px.index[0]]),
                        dividends=div)


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
