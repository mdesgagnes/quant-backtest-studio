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
`omitted`. Tables are grouped rather than given a sheet each: small tables
are stacked on one sheet per module, daily series are joined side by side
on their dates. An About sheet at the front records the settings and
indexes every table, with a link to it, and says which modules are absent
and why, so a missing table is never mistaken for a missing result.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import metrics as M
from .config import RunConfig, REBALANCE_RULES
from .engine import BacktestResult


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
    """Collects every table, then lays them out on a handful of sheets.

    Small tables are stacked, one under another with a title, on their
    module's sheet (all performance tables on "Performance", everything
    about the book on "Positions"...). Daily time series are joined side by
    side on their dates instead: one row per session, one column per
    instrument and field, as in "Daily Positions". An "About" sheet lists
    the run's settings and every table with a link to where it sits.
    """

    def __init__(self, xw, include: Optional[Iterable[str]] = None,
                 modules: Optional[List[str]] = None):
        self.xw = xw
        self.modules = modules or []
        self.module = "Results"
        self.include = set(include) if include is not None else None
        self.blocks: List[Dict[str, Any]] = []
        self.missing: List[Dict[str, Any]] = []
        self.notes: Optional[pd.DataFrame] = None
        self.summary = False

    def wants(self, module: str) -> bool:
        return self.include is None or module in self.include

    def write(self, df, sheet: str, index: bool = False,
              module: Optional[str] = None, always: bool = False,
              note: str = "") -> None:
        """Records one table; `sheet` is its title."""
        mod = module or self.module
        if _empty(df) or not (always or self.wants(mod)):
            return
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if sheet == "Notes":
            self.notes = df
            return
        self.blocks.append({"title": sheet, "df": df, "index": index,
                            "module": mod, "note": note})

    def sections(self, sections: Optional[List[Section]],
                 module: Optional[str] = None) -> None:
        """Records the pending sections of one module, or all that remain."""
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
            if self.wants(module):
                self.missing.append({"Module": module, "Table": f"(not included) {what}",
                                     "Sheet": "", "Rows": None, "Note": why})

    # ------------------------------------------------------------------
    def _place(self, title: str, module: str) -> Tuple[str, Optional[str]]:
        """(sheet, None) for a stacked table, (sheet, prefix) for a joined one."""
        if title in _JOINED:
            return _JOINED[title]
        if title in _OWN_SHEET:
            return _OWN_SHEET[title], None
        return _MODULE_SHEET.get(module, module[:31]), None

    def finish(self, summary: Optional[Dict[str, Any]], style: bool) -> None:
        from . import exec_summary as ES
        self.summary = bool(summary) and self.wants("Executive summary")
        wb = self.xw.book
        stacked: Dict[str, List[Dict[str, Any]]] = {}
        joined: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for b in self.blocks:
            sheet, prefix = self._place(b["title"], b["module"])
            if prefix is None:
                stacked.setdefault(sheet, []).append(b)
            else:
                joined.setdefault(sheet, []).append((prefix, b))

        index_rows: List[Dict[str, Any]] = []
        titles: Dict[str, set] = {}
        for sheet in [s for s in SHEET_ORDER if s in stacked] + \
                     [s for s in stacked if s not in SHEET_ORDER]:
            r = 0
            for b in stacked[sheet]:
                df = b["df"]
                df.to_excel(self.xw, sheet_name=sheet, startrow=r + 1, index=b["index"])
                ws = self.xw.sheets[sheet]
                ws.cell(row=r + 1, column=1, value=b["title"]).font = _TITLE_FONT
                titles.setdefault(sheet, set()).add(r + 1)
                ncols = df.shape[1] + (df.index.nlevels if b["index"] else 0)
                if style:
                    _style_block(ws, r + 2, len(df), ncols, df, b["index"], b["title"])
                index_rows.append({"Module": b["module"], "Table": b["title"],
                                   "Sheet": sheet, "Cell": f"A{r + 1}",
                                   "Rows": int(len(df)), "Note": b["note"]})
                r += len(df) + 4
            if style:
                _widths(self.xw.sheets[sheet], titles.get(sheet, set()))
        for sheet in [s for s in SHEET_ORDER if s in joined] + \
                     [s for s in joined if s not in SHEET_ORDER]:
            parts = []
            for prefix, b in joined[sheet]:
                df = b["df"].copy()
                if prefix:
                    df.columns = [f"{prefix} {c}" for c in df.columns]
                parts.append(df)
                index_rows.append({"Module": b["module"], "Table": b["title"],
                                   "Sheet": sheet, "Cell": "",
                                   "Rows": int(len(df)),
                                   "Note": (f"Columns headed “{prefix} ...”"
                                            if prefix else "") + (
                                       ("; " if prefix and b["note"] else "") + b["note"])})
            wide = pd.concat(parts, axis=1).sort_index()
            wide.index.name = "Date"
            wide.to_excel(self.xw, sheet_name=sheet, index=True)
            ws = self.xw.sheets[sheet]
            if style:
                _style_block(ws, 1, len(wide), wide.shape[1] + 1, wide, True, sheet)
                _widths(ws, set())
            ws.freeze_panes = "B2"

        self._about(index_rows)
        order = ["About"] + [s for s in SHEET_ORDER if s in wb.sheetnames and s != "About"]
        order += [s for s in wb.sheetnames if s not in order]
        wb._sheets = [wb[s] for s in order]
        if self.summary:
            ES.write_summary(wb, summary)
        wb.active = 0

    def _about(self, index_rows: List[Dict[str, Any]]) -> None:
        """Settings, then an index of every table with a link to it."""
        r = 0
        if self.notes is not None:
            self.notes.to_excel(self.xw, sheet_name="About", startrow=1, index=False)
            ws = self.xw.sheets["About"]
            ws.cell(row=1, column=1, value="Settings").font = _TITLE_FONT
            _style_block(ws, 2, len(self.notes), 2, self.notes, False, "Settings")
            r = len(self.notes) + 4
        skip = {1, r + 1}
        rows = list(index_rows)
        if self.summary:
            rows.insert(0, {"Module": "Executive summary", "Table": "Executive summary",
                            "Sheet": "Executive Summary", "Cell": "A1", "Rows": None,
                            "Note": "Two-page CIO report: performance, then risk and robustness."})
        rows += self.missing
        if self.include is not None:
            rows += [{"Module": m, "Table": "(excluded)", "Sheet": "", "Cell": "",
                      "Rows": None, "Note": "Left out of this export by choice."}
                     for m in self.modules if m not in self.include]
        idx = pd.DataFrame(rows, columns=["Module", "Table", "Sheet", "Cell", "Rows", "Note"])
        idx.to_excel(self.xw, sheet_name="About", startrow=r + 1, index=False)
        ws = self.xw.sheets["About"]
        ws.cell(row=r + 1, column=1, value="Contents").font = _TITLE_FONT
        if self.notes is None:
            skip = {r + 1}
        _style_block(ws, r + 2, len(idx), idx.shape[1], idx, False, "Contents")
        for k, row in enumerate(rows):
            if row.get("Sheet") and row["Sheet"] != "":
                cell = ws.cell(row=r + 3 + k, column=2)
                target = f"'{row['Sheet']}'!{row['Cell'] or 'A1'}"
                cell.hyperlink = f"#{target}"
                cell.font = Font(name="Calibri", size=9, color="B01419", underline="single")
        _widths(ws, skip)


# Where each table goes. Tables not named here are stacked on their
# module's sheet.
SHEET_ORDER = ["About", "Performance", "Positions", "Trades & Cash Flows",
               "Attribution", "Tax", "Robustness", "Stress & Regimes", "Data",
               "Daily Series", "Daily Positions", "Daily Signals", "Market Data"]
_MODULE_SHEET = {"Results": "Performance", "Signals": "Positions",
                 "Positions": "Positions", "Attribution": "Attribution",
                 "Tax": "Tax", "Robustness": "Robustness",
                 "Stress tests": "Stress & Regimes", "Market regimes": "Stress & Regimes",
                 "Configuration": "Data", "Data": "Data"}
_OWN_SHEET = {"Trades": "Trades & Cash Flows", "Cash Flows": "Trades & Cash Flows",
              "Rebalance Dates": "Trades & Cash Flows"}
# title -> (sheet, column prefix): daily series joined on their dates.
_JOINED = {
    "Daily Series": ("Daily Series", ""), "Series": ("Daily Series", ""),
    "After-Tax Equity": ("Daily Series", ""),
    "Monte Carlo Bands": ("Daily Series", "Monte Carlo"),
    "All Imported Series": ("Daily Series", "Return"),
    "Holdings History": ("Daily Positions", "Weight"),
    "Target Weights": ("Daily Positions", "Target"),
    "Share Counts": ("Daily Positions", "Units"),
    "Weight by Sleeve": ("Daily Positions", "Sleeve"),
    "Daily Contributions": ("Daily Positions", "Contribution"),
    "Signal Scores": ("Daily Signals", "Score"),
    "Signal Ranks": ("Daily Signals", "Rank"),
    "Prices": ("Market Data", "Price"),
    "Exogenous Series": ("Market Data", ""),
}


# ----------------------------------------------------------------------
# Formatting: header rows, number formats and widths. Values never change.
_TITLE_FONT = Font(name="Calibri", size=11, bold=True, color="B01419")
_HEAD_FONT = Font(name="Calibri", size=9, bold=True, color="FFFFFF")
_HEAD_FILL = PatternFill("solid", fgColor="1A1A1A")
_PCT2 = '0.00%;-0.00%;0.00%'
_MONEY2 = '#,##0.00'
_PCT_COL_WORDS = ("return", "cagr", "volatility", "drawdown", "weight", "excess",
                  "exposure", "turnover", "contribution", "share of", "% of",
                  "hit rate", "annual rate", "alpha", "tracking error", "worst",
                  "best", "yield", "target", "sleeve", "monte carlo")
_NOT_PCT_WORDS = ("sessions", "days", "periods", "rows", "count", "fold", "rank",
                  "score", "deductions", "bps", "units", "price")
_MONEY_WORDS = ("value", "notional", "amount", "tax", "gain", "proceeds", "acb",
                "paid", "raised", "cash before", "cash after", "costs")
# Tables whose every numeric column after the first is a fraction.
_PCT_TABLES = ("Monthly Returns", "Calendar", "Trailing", "Regime Overview",
               "Contribution by", "Class Contribution", "Income and Costs",
               "Stress Returns")


def _col_format(name: str) -> Optional[str]:
    n = str(name).lower()
    if any(w in n for w in _MONEY_WORDS) and not any(w in n for w in ("rate", "ratio", "bps")):
        return _MONEY2
    if any(w in n for w in _PCT_COL_WORDS) and not any(w in n for w in _NOT_PCT_WORDS):
        return _PCT2
    return None


def _style_block(ws, head_row: int, nrows: int, ncols: int, df: pd.DataFrame,
                 index: bool, title: str, max_cells: int = 250_000) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=head_row, column=c)
        cell.font, cell.fill = _HEAD_FONT, _HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    if nrows * ncols > max_cells:
        return
    heads = [ws.cell(row=head_row, column=c).value for c in range(1, ncols + 1)]
    whole = title.startswith(_PCT_TABLES)
    fmts = {}
    for c in range(1, ncols + 1):
        f = _col_format(heads[c - 1] or "")
        if f is None and whole and c > 1:
            f = _PCT2
        if f:
            fmts[c] = f
    if not fmts:
        return
    for row in ws.iter_rows(min_row=head_row + 1, max_row=head_row + nrows,
                            max_col=ncols):
        for cell in row:
            f = fmts.get(cell.column)
            if f and isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = f


def _widths(ws, title_rows: set, sample: int = 400) -> None:
    """Column widths from the headers and a sample of the values; the
    tables' title rows do not count."""
    widths: Dict[int, int] = {}
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, sample)):
        for c in row:
            if c.value is None or c.row in title_rows:
                continue
            v = c.value
            n = len(f"{v:,.4f}") if isinstance(v, float) else len(str(v))
            widths[c.column] = max(widths.get(c.column, 0), n)
    for col, n in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = max(9, min(40, n * 0.95 + 2))


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
                    module: str = "Market regimes",
                    rf=0.0) -> List[Section]:
    """Every regime dimension: an overview of annualized return by regime
    for each series, then the full table (volatility, Sharpe, hit rate,
    time spent, excess over the benchmark)."""
    from . import regimes as RG
    tables = []
    for k, dim in RG.DIMENSIONS.items():
        if k not in labels:
            continue
        t = RG.regime_table(returns, labels[k], dim.order, ppy, benchmark, rf)
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
    with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd",
                        date_format="yyyy-mm-dd") as xw:
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
    with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd",
                        date_format="yyyy-mm-dd") as xw:
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
