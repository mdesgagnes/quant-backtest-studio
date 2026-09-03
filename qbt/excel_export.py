"""Complete Excel export.

Writes every table behind the on-screen report into one workbook, so the
numbers can be checked, charted or merged elsewhere without re-deriving
anything. Two entry points, one for a simulated backtest and one for an
imported return stream, both producing the same shape of file wherever the
underlying data allows it.

The guiding rule is that nothing shown on screen should be missing here,
and nothing here should require the app to interpret it: every sheet
carries its own headers, percentages stay as real numbers rather than
pre-formatted strings, and a Notes sheet records the settings the figures
depend on.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from .config import RunConfig, REBALANCE_RULES
from .engine import BacktestResult


def _safe(name: str) -> str:
    """Excel sheet names: 31 chars, no []:*?/\\ ."""
    out = "".join(c for c in str(name) if c not in "[]:*?/\\")
    return (out[:31] or "Sheet")


def _write(xw, df: Optional[pd.DataFrame], sheet: str, index: bool = False) -> None:
    if df is None or (hasattr(df, "empty") and df.empty):
        return
    df.to_excel(xw, sheet_name=_safe(sheet), index=index)


def _stats_frame(stats: Dict[str, float],
                 bench_stats: Optional[Dict[str, float]],
                 label: str, bench_label: str) -> pd.DataFrame:
    rows = []
    for k, v in stats.items():
        row = {"Metric": k, label: v}
        if bench_stats:
            row[bench_label] = bench_stats.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _notes_frame(cfg: RunConfig, extra: Dict[str, Any]) -> pd.DataFrame:
    base = {
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Label": cfg.label,
        "Periods per year": cfg.engine.periods_per_year,
        "Initial capital": cfg.engine.initial_capital,
    }
    base.update(extra)
    return pd.DataFrame({"Setting": list(base.keys()),
                         "Value": [str(v) for v in base.values()]})


# ----------------------------------------------------------------------
def workbook_from_backtest(res: BacktestResult,
                           bench: Optional[BacktestResult],
                           stats: Dict[str, float],
                           bench_stats: Optional[Dict[str, float]],
                           cfg: RunConfig,
                           prices: Optional[pd.DataFrame] = None,
                           exog: Optional[pd.DataFrame] = None,
                           quality: Optional[pd.DataFrame] = None) -> bytes:
    """Every table behind a simulated backtest, in one workbook."""
    ppy = cfg.engine.periods_per_year
    label = res.label
    blabel = bench.label if bench is not None else "Benchmark"

    periods = M.period_table(
        res.equity, res.returns,
        bench.equity if bench is not None else None,
        bench.returns if bench is not None else None,
        ppy, label, blabel)

    series = pd.DataFrame({
        "Value": res.equity,
        "Return": res.returns,
        "Gross return": res.gross_returns,
        "Exposure": res.exposure,
        "Cash weight": res.cash_weight,
        "Turnover": res.turnover,
        "Cost": res.costs,
    })
    if res.dividend_income is not None:
        series["Dividend income"] = res.dividend_income
    if res.fees is not None and float(res.fees.sum()) > 0:
        series["Management fee"] = res.fees
    if bench is not None:
        series["Benchmark value"] = bench.equity
        series["Benchmark return"] = bench.returns
    series["Drawdown"] = M.drawdown(res.equity)

    last = res.weights.iloc[-1]
    value = float(res.equity.iloc[-1])
    holdings = pd.DataFrame({
        "Instrument": list(last.index) + ["Cash"],
        "Weight": list(last.values) + [float(res.cash_weight.iloc[-1])],
    })
    holdings["Value"] = holdings["Weight"] * value

    rebal = pd.DataFrame({"Rebalance date": [d.date() for d in res.rebalance_dates]})

    notes = _notes_frame(cfg, {
        "Rebalance": REBALANCE_RULES.get(cfg.engine.rebalance, cfg.engine.rebalance),
        "Execution lag (sessions)": cfg.engine.execution_lag,
        "Execution price": "Open, marked at close" if cfg.engine.execute_at_open else "Close",
        "Warm-up trimmed": "Yes" if cfg.engine.trim_warmup else "No",
        "Dividends": "Credited as cash" if cfg.data.use_dividends else "Inside adjusted prices",
        "Commission (bps)": cfg.costs.commission_bps,
        "Slippage (bps)": cfg.costs.slippage_bps,
        "Cash rate (annual)": cfg.costs.cash_rate_pa,
        "Management fee (annual)": getattr(cfg.costs, "management_fee_pa", 0.0),
        "Whole units only": "Yes" if getattr(cfg.engine, "whole_shares", False) else "No",
        "Max leverage": cfg.engine.max_leverage,
        "Strategy": cfg.strategy.name,
        "Period start": res.equity.index[0].date(),
        "Period end": res.equity.index[-1].date(),
        "Observations": len(res.equity),
    })

    params = pd.DataFrame({
        "Parameter": list(cfg.strategy.params.keys()),
        "Value": [str(v) for v in cfg.strategy.params.values()],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        _write(xw, notes, "Notes")
        _write(xw, _stats_frame(stats, bench_stats, label, blabel), "Statistics")
        _write(xw, periods["trailing"], "Trailing Periods")
        _write(xw, periods["calendar"], "Calendar Years")
        _write(xw, M.drawdown_table(res.equity, 25, ppy), "Drawdowns")
        mr = M.monthly_returns(res.returns, ppy)
        _write(xw, mr, "Monthly Returns", index=True)
        _write(xw, series, "Daily Series", index=True)
        _write(xw, holdings, "Current Holdings")
        _write(xw, res.weights, "Holdings History", index=True)
        _write(xw, res.target_weights, "Target Weights", index=True)
        _write(xw, res.trades, "Trades")
        if res.shares is not None:
            _write(xw, res.shares, "Share Counts", index=True)
        _write(xw, rebal, "Rebalance Dates")
        if params is not None and not params.empty:
            _write(xw, params, "Parameters")
        if prices is not None:
            _write(xw, prices, "Prices", index=True)
        if exog is not None and not exog.empty:
            _write(xw, exog, "Exogenous Series", index=True)
        if quality is not None:
            _write(xw, quality, "Data Quality")
    return buf.getvalue()


# ----------------------------------------------------------------------
def workbook_from_returns(returns: pd.Series, equity: pd.Series,
                          stats: Dict[str, float],
                          bench_returns: Optional[pd.Series],
                          bench_equity: Optional[pd.Series],
                          bench_stats: Optional[Dict[str, float]],
                          ppy: int, label: str,
                          bench_label: str = "Benchmark",
                          all_series: Optional[pd.DataFrame] = None,
                          report_notes: Optional[Dict[str, Any]] = None) -> bytes:
    """The same workbook for an imported return stream.

    Sheets that need position data are absent, since none exists; every
    sheet derived purely from the return series is present and identical in
    shape to the backtest export.
    """
    periods = M.period_table(equity, returns, bench_equity, bench_returns,
                             ppy, label, bench_label)

    series = pd.DataFrame({"Return": returns, "Value": equity})
    if bench_returns is not None:
        series["Benchmark return"] = bench_returns
        series["Benchmark value"] = bench_equity
    series["Drawdown"] = M.drawdown(equity)

    cfg = RunConfig(label=label)
    cfg.engine.periods_per_year = ppy
    cfg.engine.initial_capital = float(equity.iloc[0]) if len(equity) else 0.0
    notes = _notes_frame(cfg, {
        "Source": "Imported return stream",
        "Frequency": M.freq_words(ppy)[0],
        "Period start": returns.index[0].date(),
        "Period end": returns.index[-1].date(),
        "Observations": len(returns),
        **(report_notes or {}),
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        _write(xw, notes, "Notes")
        _write(xw, _stats_frame(stats, bench_stats, label, bench_label), "Statistics")
        _write(xw, periods["trailing"], "Trailing Periods")
        _write(xw, periods["calendar"], "Calendar Years")
        _write(xw, M.drawdown_table(equity, 25, ppy), "Drawdowns")
        _write(xw, M.monthly_returns(returns, ppy), "Monthly Returns", index=True)
        _write(xw, series, "Series", index=True)
        if all_series is not None and not all_series.empty:
            _write(xw, all_series, "All Imported Series", index=True)
    return buf.getvalue()
