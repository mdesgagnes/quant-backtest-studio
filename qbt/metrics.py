"""Performance and risk statistics. Pure, stateless functions."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def to_equity(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(equity: pd.Series, ppy: int = TRADING_DAYS) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return np.nan
    years = len(equity) / ppy
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan


def annual_vol(returns: pd.Series, ppy: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(ppy))


# Below this daily volatility, a period is effectively riskless (portfolio
# fully in cash): a return/risk ratio has no meaning there and would read in
# the hundreds. NaN instead of a flattering, misleading number.
RISKLESS_SD = 1e-5


def sharpe(returns: pd.Series, rf_pa: float = 0.0, ppy: int = TRADING_DAYS) -> float:
    ex = returns - rf_pa / ppy
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ppy)) if sd and sd > RISKLESS_SD else np.nan


def sortino(returns: pd.Series, rf_pa: float = 0.0, ppy: int = TRADING_DAYS) -> float:
    ex = returns - rf_pa / ppy
    down = ex[ex < 0].std(ddof=1)
    return float(ex.mean() / down * np.sqrt(ppy)) if down and down > RISKLESS_SD else np.nan


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """Worst drawdown episodes, with duration and recovery time."""
    dd = drawdown(equity)
    in_dd = dd < 0
    episodes = []
    start = None
    for date, flag in in_dd.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            seg = dd.loc[start:date]
            episodes.append((start, seg.idxmin(), date, float(seg.min())))
            start = None
    if start is not None:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), None, float(seg.min())))

    rows = []
    for s, trough, rec, depth in sorted(episodes, key=lambda x: x[3])[:top]:
        rows.append({
            "Start": s.date(),
            "Trough": trough.date(),
            "Recovery": rec.date() if rec is not None else "ongoing",
            "Drawdown": depth,
            "Days to Trough": int(len(equity.loc[s:trough])),
            "Total Days": int(len(equity.loc[s:rec])) if rec is not None
                          else int(len(equity.loc[s:])),
        })
    return pd.DataFrame(rows)


def calmar(equity: pd.Series, ppy: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    return float(cagr(equity, ppy) / mdd) if mdd > 0 else np.nan


def ulcer_index(equity: pd.Series) -> float:
    dd = drawdown(equity) * 100
    return float(np.sqrt((dd ** 2).mean()))


def var_cvar(returns: pd.Series, level: float = 0.05) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {"var": np.nan, "cvar": np.nan}
    v = float(np.quantile(r, level))
    tail = r[r <= v]
    return {"var": v, "cvar": float(tail.mean()) if len(tail) else v}


def beta_alpha(returns: pd.Series, bench: pd.Series,
               rf_pa: float = 0.0, ppy: int = TRADING_DAYS) -> Dict[str, float]:
    df = pd.concat([returns, bench], axis=1).dropna()
    if len(df) < 30:
        return {"beta": np.nan, "alpha": np.nan, "r2": np.nan, "corr": np.nan,
                "tracking_error": np.nan, "information_ratio": np.nan}
    y = df.iloc[:, 0] - rf_pa / ppy
    x = df.iloc[:, 1] - rf_pa / ppy
    var_x = x.var(ddof=1)
    b = float(np.cov(y, x, ddof=1)[0, 1] / var_x) if var_x > 0 else np.nan
    a = float((y.mean() - b * x.mean()) * ppy)
    corr = float(np.corrcoef(y, x)[0, 1])
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = float(active.std(ddof=1) * np.sqrt(ppy))
    return {
        "beta": b, "alpha": a, "r2": corr ** 2, "corr": corr,
        "tracking_error": te,
        "information_ratio": float(active.mean() * ppy / te) if te > 0 else np.nan,
    }


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Year x month table, in decimal."""
    m = (1 + returns.fillna(0)).resample("ME").prod() - 1
    if m.empty:
        return pd.DataFrame()
    t = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "r": m.values})
    return t.pivot(index="Year", columns="Month", values="r")


def rolling_sharpe(returns: pd.Series, window: int = 252,
                   ppy: int = TRADING_DAYS) -> pd.Series:
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    return (mu / sd) * np.sqrt(ppy)


# ----------------------------------------------------------------------
# Full summary
# ----------------------------------------------------------------------
def summary(returns: pd.Series,
            equity: Optional[pd.Series] = None,
            bench_returns: Optional[pd.Series] = None,
            turnover: Optional[pd.Series] = None,
            exposure: Optional[pd.Series] = None,
            rf_pa: float = 0.0,
            ppy: int = TRADING_DAYS) -> Dict[str, float]:
    returns = returns.dropna()
    eq = equity if equity is not None else to_equity(returns)
    vc = var_cvar(returns)
    monthly = (1 + returns).resample("ME").prod() - 1

    out: Dict[str, float] = {
        "Total Return": float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else np.nan,
        "CAGR": cagr(eq, ppy),
        "Volatility": annual_vol(returns, ppy),
        "Sharpe": sharpe(returns, rf_pa, ppy),
        "Sortino": sortino(returns, rf_pa, ppy),
        "Calmar": calmar(eq, ppy),
        "Max Drawdown": max_drawdown(eq),
        "Ulcer Index": ulcer_index(eq),
        "VaR 95% (daily)": vc["var"],
        "CVaR 95% (daily)": vc["cvar"],
        "Skew": float(returns.skew()),
        "Kurtosis": float(returns.kurtosis()),
        "% Positive Months": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "Best Month": float(monthly.max()) if len(monthly) else np.nan,
        "Worst Month": float(monthly.min()) if len(monthly) else np.nan,
    }
    if turnover is not None and len(turnover):
        out["Annual Turnover"] = float(turnover.sum() / (len(returns) / ppy))
    if exposure is not None and len(exposure):
        out["Average Exposure"] = float(exposure.mean())
    if bench_returns is not None:
        ba = beta_alpha(returns, bench_returns, rf_pa, ppy)
        out.update({
            "Beta": ba["beta"], "Alpha (ann.)": ba["alpha"], "R\u00b2": ba["r2"],
            "Tracking Error": ba["tracking_error"],
            "Information Ratio": ba["information_ratio"],
        })
    return out


FORMATS = {
    "Total Return": "pct", "CAGR": "pct", "Volatility": "pct",
    "Max Drawdown": "pct", "VaR 95% (daily)": "pct", "CVaR 95% (daily)": "pct",
    "% Positive Months": "pct", "Best Month": "pct", "Worst Month": "pct",
    "Alpha (ann.)": "pct", "Tracking Error": "pct", "Average Exposure": "pct",
}


def format_metric(key: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "\u2014"
    if FORMATS.get(key) == "pct":
        return f"{value * 100:,.2f}%"
    return f"{value:,.2f}"


# ----------------------------------------------------------------------
# Period reporting
# ----------------------------------------------------------------------
# (label, months back). None = year-to-date, -1 = since inception.
TRAILING_PERIODS = [
    ("1M", 1), ("3M", 3), ("6M", 6), ("YTD", None), ("1Y", 12),
    ("2Y", 24), ("3Y", 36), ("5Y", 60), ("10Y", 120), ("15Y", 180),
    ("20Y", 240), ("Since inception", -1),
]


def trailing_returns(equity: pd.Series, ppy: int = TRADING_DAYS,
                     as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Cumulative returns over standard trailing windows.

    Anything longer than a year is annualized, anything shorter is left
    cumulative -- annualizing a three-month number implies the quarter
    repeats four times, which is exactly the extrapolation that makes short
    windows look impressive. The `Annualized` column records which
    convention each row uses so the table can never be misread.

    A window is reported only if the history actually covers it. A period
    that reaches back further than the data is left out rather than
    silently measured over whatever is available and labelled as though it
    were the full span.
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return pd.DataFrame()

    end = as_of or eq.index[-1]
    eq = eq.loc[:end]
    start_all = eq.index[0]
    end_val = float(eq.iloc[-1])

    rows = []
    for label, months in TRAILING_PERIODS:
        if months is None:
            start = pd.Timestamp(year=end.year, month=1, day=1)
            if start <= start_all:
                continue
        elif months < 0:
            start = start_all
        else:
            start = end - pd.DateOffset(months=months)
            if start < start_all:
                continue        # not enough history: omit rather than mislead

        window = eq.loc[eq.index >= start]
        if len(window) < 2:
            continue
        base = float(window.iloc[0])
        if base <= 0:
            continue

        total = end_val / base - 1.0
        years = max((window.index[-1] - window.index[0]).days / 365.25, 1e-9)
        annualized = years > 1.0000001
        value = ((1.0 + total) ** (1.0 / years) - 1.0) if annualized else total

        rows.append({
            "Period": label,
            "Return": value,
            "Annualized": "Yes" if annualized else "No",
            "From": window.index[0].date(),
            "To": window.index[-1].date(),
        })
    return pd.DataFrame(rows)


def calendar_years(returns: pd.Series) -> pd.DataFrame:
    """Return for each calendar year, with the partial first and last year
    flagged rather than quietly presented as full years."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()

    grouped = (1.0 + r).groupby(r.index.year).prod() - 1.0
    first_year, last_year = r.index[0].year, r.index[-1].year
    rows = []
    for year, value in grouped.items():
        partial = ""
        if year == first_year and (r.index[0].month, r.index[0].day) > (1, 5):
            partial = f"from {r.index[0].date()}"
        if year == last_year and (r.index[-1].month, r.index[-1].day) < (12, 26):
            partial = f"to {r.index[-1].date()}"
        rows.append({"Year": int(year), "Return": float(value),
                     "Partial": partial})
    return pd.DataFrame(rows)


def period_table(equity: pd.Series, returns: pd.Series,
                 bench_equity: Optional[pd.Series] = None,
                 bench_returns: Optional[pd.Series] = None,
                 ppy: int = TRADING_DAYS,
                 label: str = "Strategy",
                 bench_label: str = "Benchmark") -> Dict[str, pd.DataFrame]:
    """Trailing and calendar-year tables, strategy against benchmark."""
    tr = trailing_returns(equity, ppy)
    cy = calendar_years(returns)

    if bench_equity is not None and not tr.empty:
        btr = trailing_returns(bench_equity, ppy)
        merged = tr.merge(btr[["Period", "Return"]], on="Period", how="left",
                          suffixes=("", "_bench"))
        merged = merged.rename(columns={"Return": label,
                                        "Return_bench": bench_label})
        merged["Excess"] = merged[label] - merged[bench_label]
        tr = merged
    elif not tr.empty:
        tr = tr.rename(columns={"Return": label})

    if bench_returns is not None and not cy.empty:
        bcy = calendar_years(bench_returns)
        merged = cy.merge(bcy[["Year", "Return"]], on="Year", how="left",
                          suffixes=("", "_bench"))
        merged = merged.rename(columns={"Return": label,
                                        "Return_bench": bench_label})
        merged["Excess"] = merged[label] - merged[bench_label]
        cy = merged
    elif not cy.empty:
        cy = cy.rename(columns={"Return": label})

    return {"trailing": tr, "calendar": cy}
