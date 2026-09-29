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

Beyond the core results, the caller passes every other module it has on
hand -- signals, attribution, tax, robustness, stress tests, regimes, data
diagnostics -- as `sections`, and names whatever it could not include in
`omitted`. A Contents sheet at the front lists every sheet by module and
says which modules are absent and why, so a missing tab is never mistaken
for a missing result.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import metrics as M
from .config import RunConfig, REBALANCE_RULES
from .engine import BacktestResult


def _safe(name: str) -> str:
    """Excel sheet names: 31 chars, no []:*?/\\ ."""
    out = "".join(c for c in str(name) if c not in "[]:*?/\\")
    return (out[:31] or "Sheet")


# The modules each workbook can carry, in the order they appear. The
# Export tab offers these as a checklist; Notes and Contents always come.
BACKTEST_MODULES = ["Executive summary", "Results", "Signals", "Positions", "Attribution", "Tax",
                    "Robustness", "Stress tests", "Market regimes",
                    "Configuration", "Data"]
RETURNS_MODULES = ["Executive summary", "Results", "Stress tests", "Market regimes", "Robustness",
                   "Data"]

# (module, sheet name, frame, write the index[, note for the Contents sheet])
Section = Tuple[Any, ...]
# (module, what is missing, why)
Omission = Tuple[str, str, str]


def _empty(df) -> bool:
    return df is None or (hasattr(df, "empty") and df.empty)


class _Book:
    """Writes sheets while recording each one for the Contents sheet."""

    def __init__(self, xw, include: Optional[Iterable[str]] = None,
                 modules: Optional[List[str]] = None):
        self.xw = xw
        self.modules = modules or []
        self.module = "Results"
        self.include = set(include) if include is not None else None
        self.rows: List[Dict[str, Any]] = []
        self.used: set = set()
        self.summary = False

    def wants(self, module: str) -> bool:
        return self.include is None or module in self.include

    def _name(self, sheet: str) -> str:
        base = _safe(sheet)
        name, i = base, 2
        while name.lower() in self.used or name.lower() == "contents":
            tail = f" ({i})"
            name = base[:31 - len(tail)] + tail
            i += 1
        self.used.add(name.lower())
        return name

    def write(self, df, sheet: str, index: bool = False,
              module: Optional[str] = None, always: bool = False,
              note: str = "") -> None:
        if _empty(df) or not (always or self.wants(module or self.module)):
            return
        if isinstance(df, pd.Series):
            df = df.to_frame()
        name = self._name(sheet)
        df.to_excel(self.xw, sheet_name=name, index=index)
        self.rows.append({"Module": module or self.module, "Sheet": name,
                          "Rows": int(len(df)), "Note": note})

    def sections(self, sections: Optional[List[Section]],
                 module: Optional[str] = None) -> None:
        """Writes the pending sections of one module, or all that remain,
        so each module's extra sheets sit beside its core ones."""
        keep = []
        for sec in sections or []:
            if module is None or sec[0] == module:
                self.write(sec[2], sec[1], sec[3], sec[0],
                           note=sec[4] if len(sec) > 4 else "")
            else:
                keep.append(sec)
        if sections is not None:
            sections[:] = keep

    def omit(self, omitted: Optional[List[Omission]]) -> None:
        for module, what, why in omitted or []:
            if not self.wants(module):
                continue
            self.rows.append({"Module": module, "Sheet": f"(not included) {what}",
                              "Rows": None, "Note": why})

    def contents(self) -> None:
        rows = list(self.rows)
        if self.summary:
            rows.insert(0, {"Module": "Executive summary", "Sheet": "Executive Summary",
                            "Rows": None, "Note": "One-page CIO report, fitted to "
                                                  "a printed page."})
        if self.include is not None:
            left_out = [m for m in self.modules if m not in self.include]
            rows += [{"Module": m, "Sheet": "(excluded)", "Rows": None,
                      "Note": "Left out of this export by choice."}
                     for m in left_out]
        df = pd.DataFrame(rows, columns=["Module", "Sheet", "Rows", "Note"])
        df.to_excel(self.xw, sheet_name="Contents", index=False)
        wb = self.xw.book
        wb.move_sheet("Contents", offset=-(len(wb.sheetnames) - 1))
        wb.active = 0

    def finish(self, summary: Optional[Dict[str, Any]], style: bool) -> None:
        """Contents, then formatting, then the summary in front of both."""
        from . import exec_summary as ES
        self.summary = bool(summary) and self.wants("Executive summary")
        self.contents()
        if style:
            ES.style_workbook(self.xw.book)
        if self.summary:
            ES.write_summary(self.xw.book, summary)


def _write(xw, df: Optional[pd.DataFrame], sheet: str, index: bool = False) -> None:
    if _empty(df):
        return
    df.to_excel(xw, sheet_name=_safe(sheet), index=index)


def _kv(pairs: Dict[str, Any], key: str = "Measure", value: str = "Value") -> pd.DataFrame:
    return pd.DataFrame({key: list(pairs.keys()), value: list(pairs.values())})


# ----------------------------------------------------------------------
# Module helpers: each turns one module's output into ready-to-write
# sections, so both entry points shape them identically.
def stress_sections(ev: Optional[pd.DataFrame], module: str = "Stress tests",
                    prefix: str = "") -> List[Section]:
    """The per-period table plus its headline summary."""
    if _empty(ev):
        return []
    from . import stress as STRESS
    names = {
        "n_covered": "Periods covered", "n_total": "Periods listed",
        "n_positive": "Positive periods", "median_return": "Median return",
        "best_period": "Best period", "best_return": "Best return",
        "worst_period": "Worst period", "worst_return": "Worst return",
        "worst_drawdown": "Deepest drawdown",
        "n_vs_benchmark": "Periods with a benchmark",
        "n_beat_benchmark": "Periods beating the benchmark",
        "median_excess": "Median excess vs benchmark",
    }
    summ = STRESS.summary_stats(ev)
    return [(module, f"{prefix}Stress Periods", ev, False),
            (module, f"{prefix}Stress Summary",
             _kv({names.get(k, k): v for k, v in summ.items()}), False)]


def monte_carlo_sections(mc: Optional[Dict[str, Any]],
                         module: str = "Robustness") -> List[Section]:
    if not isinstance(mc, dict) or _empty(mc.get("stats")):
        return []
    head = _kv({"Simulated median CAGR": mc.get("median_cagr"),
                "Probability of loss": mc.get("prob_loss"),
                "Probability of a drawdown > 20%": mc.get("prob_dd_20")})
    return [(module, "Monte Carlo Summary", head, False),
            (module, "Monte Carlo Percentiles", mc["stats"], False),
            (module, "Monte Carlo Bands", mc.get("paths"), True)]


def _attrs_frame(df: Optional[pd.DataFrame],
                 labels: Dict[str, str]) -> Optional[pd.DataFrame]:
    """The summary figures a robustness function stores in `.attrs`."""
    if _empty(df):
        return None
    got = {labels[k]: v for k, v in df.attrs.items() if k in labels}
    return _kv(got) if got else None


def robustness_sections(walk_forward: Optional[pd.DataFrame] = None,
                        in_out_sample: Optional[pd.DataFrame] = None,
                        cost_sensitivity: Optional[pd.DataFrame] = None,
                        parameter_sweep: Optional[pd.DataFrame] = None,
                        sweep_setup: Optional[Dict[str, Any]] = None,
                        day_sweep: Optional[pd.DataFrame] = None,
                        monte_carlo: Optional[Dict[str, Any]] = None,
                        module: str = "Robustness") -> List[Section]:
    out: List[Section] = [
        (module, "Walk-Forward", walk_forward, False),
        (module, "Walk-Forward Summary", _attrs_frame(walk_forward, {
            "sharpe_dispersion": "Sharpe dispersion across folds",
            "sharpe_min": "Lowest fold Sharpe"}), False),
        (module, "In-Out of Sample", in_out_sample, False),
        (module, "Parameter Sweep", parameter_sweep, False),
    ]
    if sweep_setup and not _empty(parameter_sweep):
        out.append((module, "Parameter Sweep Setup", _kv(sweep_setup), False))
    out += [
        (module, "Trading-Day Sweep", day_sweep, False),
        (module, "Trading-Day Summary", _attrs_frame(day_sweep, {
            "cagr_spread": "CAGR spread (best minus worst day)",
            "cagr_std": "CAGR standard deviation",
            "cagr_median": "Median CAGR",
            "sharpe_spread": "Sharpe spread",
            "sharpe_std": "Sharpe standard deviation"}), False),
        (module, "Cost Sensitivity", cost_sensitivity, False),
    ]
    return out + monte_carlo_sections(monte_carlo, module)


def regime_sections(returns: Dict[str, pd.Series],
                    labels: Dict[str, pd.Series], ppy: int,
                    benchmark: Optional[pd.Series] = None,
                    module: str = "Market regimes") -> List[Section]:
    """Every regime dimension: an overview of annualized return by regime
    for each series, then the full table (volatility, Sharpe, hit rate,
    time spent, excess over the benchmark)."""
    from . import regimes as RG
    tables = []
    for k, dim in RG.DIMENSIONS.items():
        if k not in labels:
            continue
        t = RG.regime_table(returns, labels[k], dim.order, ppy, benchmark)
        t = t[t["Periods"] > 0]
        if not t.empty:
            t.insert(0, "Dimension", dim.title)
            tables.append(t)
    if not tables:
        return []
    full = pd.concat(tables, ignore_index=True)
    order = list(dict.fromkeys(full["Series"]))
    overview = (full.pivot_table(index=["Dimension", "Regime"], columns="Series",
                                 values="Ann. return", sort=False)
                .reindex(columns=order).reset_index())
    return [(module, "Regime Overview", overview, False),
            (module, "Market Regimes", full, False)]


def tax_sections(rep, bench_rep, label: str, bench_label: str,
                 equity: pd.Series, bench_equity: Optional[pd.Series] = None,
                 module: str = "Tax", windowed: bool = False) -> List[Section]:
    """Everything on the Tax tab, strategy and benchmark side by side.

    `windowed` means `equity` covers only part of the run the reports were
    computed on. The cost base still carries the full history -- it has to,
    a sale's gain depends on every earlier purchase -- but the sheets keep
    only the tax years and sales inside the window, and the summary is
    re-totalled over them. Tax is assessed per calendar year, so a window
    that starts or ends mid-year carries that whole year's bill.
    """
    if rep is None:
        return []
    from . import tax as TAX
    t0, t1 = equity.index[0], equity.index[-1]

    def _years(r) -> pd.DataFrame:
        y = r.by_year
        if windowed and not y.empty:
            y = y[(y["Year"] >= t0.year) & (y["Year"] <= t1.year)]
        return y.reset_index(drop=True)

    def _sales(r) -> pd.DataFrame:
        x = r.realized_trades
        if windowed and not x.empty and "Date" in x.columns:
            d = pd.to_datetime(x["Date"])
            x = x[(d >= t0) & (d <= t1)]
        return x.reset_index(drop=True)

    def _tcr(r, pretax: pd.Series) -> float:
        if not windowed:
            return r.tax_cost_ratio
        after = r.after_tax_equity.reindex(pretax.index).ffill().bfill()
        yrs = (t1 - t0).days / 365.25
        if yrs <= 0 or pretax.iloc[0] <= 0 or after.iloc[0] <= 0:
            return np.nan
        g = lambda s_: (float(s_.iloc[-1] / s_.iloc[0])) ** (1 / yrs) - 1
        return g(pretax) - g(after)

    def _head(r, pretax: pd.Series) -> Dict[str, Any]:
        yrs, sales = _years(r), _sales(r)
        tcr = _tcr(r, pretax)
        return {
            "Total tax": (float(yrs["Total Tax"].sum()) if not yrs.empty else 0.0)
                         if windowed else r.total_tax,
            "Tax Cost Ratio (per year)": tcr,
            "Tax efficiency": (TAX.efficiency_label(tcr)[0]
                               if tcr == tcr else "n/a"),
            "Realized gains, net of losses": (
                (float(sales["Realized Gain"].sum()) if not sales.empty else 0.0)
                if windowed else r.total_pretax_gain),
            "Years with tax owed": (int((yrs["Total Tax"] > 0).sum())
                                    if not yrs.empty else 0),
            "Tax years": len(yrs),
        }

    h = _head(rep, equity)
    head = pd.DataFrame({"Measure": list(h.keys()), label: list(h.values())})
    if bench_rep is not None and bench_equity is not None:
        head[bench_label] = list(_head(
            bench_rep, bench_equity.reindex(equity.index).ffill().bfill()).values())

    # Real dollars on the strategy's own scale, as on the Tax tab.
    curves = pd.DataFrame({label: equity})
    curves[f"{label} (after tax)"] = rep.after_tax_equity.reindex(
        equity.index).ffill().bfill()
    if bench_rep is not None and bench_equity is not None:
        curves[bench_label] = bench_equity.reindex(curves.index).ffill().bfill()
        curves[f"{bench_label} (after tax)"] = bench_rep.after_tax_equity.reindex(
            curves.index).ffill().bfill()

    out: List[Section] = [
        (module, "Tax Summary", head, False),
        (module, "Tax by Year", _years(rep), False),
        (module, "Tax Realized Sales (ACB)", _sales(rep), False),
        (module, "After-Tax Equity", curves, True),
    ]
    if bench_rep is not None:
        out.append((module, "Benchmark Tax by Year", _years(bench_rep), False))
        out.append((module, "Benchmark Realized Sales", _sales(bench_rep), False))
    warn = list(rep.warnings) + (list(bench_rep.warnings)
                                 if bench_rep is not None else [])
    if warn:
        out.append((module, "Tax Warnings", pd.DataFrame({"Warning": warn}), False))
    return out


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
                           quality: Optional[pd.DataFrame] = None,
                           sections: Optional[List[Section]] = None,
                           omitted: Optional[List[Omission]] = None,
                           include: Optional[Iterable[str]] = None,
                           extra_notes: Optional[Dict[str, Any]] = None,
                           summary: Optional[Dict[str, Any]] = None,
                           style: bool = True) -> bytes:
    """Every table behind a simulated backtest, in one workbook.

    `sections` carries the other modules (signals, attribution, tax,
    robustness, stress tests, data diagnostics) in the order they should
    appear; `omitted` names the modules that could not be included.
    `include` limits the workbook to the named modules (all when None).
    """
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
        **(extra_notes or {}),
    })

    params = pd.DataFrame({
        "Parameter": list(cfg.strategy.params.keys()),
        "Value": [str(v) for v in cfg.strategy.params.values()],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        bk = _Book(xw, include, BACKTEST_MODULES)
        bk.write(notes, "Notes", module="Notes", always=True)
        bk.write(_stats_frame(stats, bench_stats, label, blabel), "Statistics")
        bk.write(periods["trailing"], "Trailing Periods")
        bk.write(periods["calendar"], "Calendar Years")
        bk.write(M.drawdown_table(res.equity, 25, ppy), "Drawdowns")
        bk.write(M.monthly_returns(res.returns, ppy), "Monthly Returns", index=True)
        bk.write(series, "Daily Series", index=True)
        sections = list(sections or [])
        bk.sections(sections, "Results")
        bk.sections(sections, "Signals")
        bk.module = "Positions"
        bk.write(holdings, "Current Holdings")
        bk.write(res.weights, "Holdings History", index=True)
        bk.write(res.target_weights, "Target Weights", index=True)
        bk.write(res.trades, "Trades")
        if res.shares is not None:
            bk.write(res.shares, "Share Counts", index=True)
        bk.write(rebal, "Rebalance Dates")
        bk.sections(sections, "Positions")
        if res.contributions is not None:
            from . import attribution as ATTR
            bk.module = "Attribution"
            bk.write(ATTR.summary(res, None, ppy), "Contribution")
            bk.write(ATTR.by_period(res, "Year"), "Contribution by Year", index=True)
            bk.write(res.contributions, "Daily Contributions", index=True)
        bk.sections(sections, "Attribution")
        for m in dict.fromkeys(s_[0] for s_ in sections if s_[0] != "Data"):
            bk.sections(sections, m)
        bk.module = "Configuration"
        if params is not None and not params.empty:
            bk.write(params, "Parameters")
        bk.module = "Data"
        if prices is not None:
            bk.write(prices, "Prices", index=True)
        if exog is not None and not exog.empty:
            bk.write(exog, "Exogenous Series", index=True)
        if quality is not None:
            bk.write(quality, "Data Quality")
        bk.sections(sections)
        bk.omit(omitted)
        bk.finish(summary, style)
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
                          report_notes: Optional[Dict[str, Any]] = None,
                          sections: Optional[List[Section]] = None,
                          omitted: Optional[List[Omission]] = None,
                          include: Optional[Iterable[str]] = None,
                          summary: Optional[Dict[str, Any]] = None,
                          style: bool = True) -> bytes:
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
        bk = _Book(xw, include, RETURNS_MODULES)
        bk.write(notes, "Notes", module="Notes", always=True)
        bk.write(_stats_frame(stats, bench_stats, label, bench_label), "Statistics")
        bk.write(periods["trailing"], "Trailing Periods")
        bk.write(periods["calendar"], "Calendar Years")
        bk.write(M.drawdown_table(equity, 25, ppy), "Drawdowns")
        bk.write(M.monthly_returns(returns, ppy), "Monthly Returns", index=True)
        bk.write(series, "Series", index=True)
        bk.sections(sections)
        if all_series is not None and not all_series.empty:
            bk.write(all_series, "All Imported Series", index=True, module="Data")
        bk.omit(omitted)
        bk.finish(summary, style)
    return buf.getvalue()
