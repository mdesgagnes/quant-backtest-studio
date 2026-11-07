"""Performance and risk statistics. Pure, stateless functions."""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Frequency vocabulary
# ----------------------------------------------------------------------
# Statistics computed on a monthly stream are monthly statistics. The
# arithmetic is identical, but a label reading "daily" on a monthly VaR is
# simply false, and "18 days to trough" for an 18-month drawdown is worse:
# it understates the episode by a factor of thirty.
_FREQ_WORDS = {
    252: ("daily", "day", "days", "Month", "Months"),
    52: ("weekly", "week", "weeks", "Month", "Months"),
    12: ("monthly", "month", "months", "Month", "Months"),
    4: ("quarterly", "quarter", "quarters", "Quarter", "Quarters"),
    2: ("semi-annual", "half-year", "half-years", "Half-year", "Half-years"),
    1: ("annual", "year", "years", "Year", "Years"),
}


def freq_words(ppy: int = TRADING_DAYS) -> tuple:
    """(adjective, singular, plural, period-noun, period-plural)."""
    if ppy in _FREQ_WORDS:
        return _FREQ_WORDS[ppy]
    nearest = min(_FREQ_WORDS, key=lambda k: abs(k - ppy))
    return _FREQ_WORDS[nearest]


def var_label(ppy: int = TRADING_DAYS) -> str:
    return f"VaR 95% ({freq_words(ppy)[0]})"


def cvar_label(ppy: int = TRADING_DAYS) -> str:
    return f"CVaR 95% ({freq_words(ppy)[0]})"


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def _base_date(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """The date one period before the first observation. Month-, quarter-
    and year-end series step back a whole calendar period (31 Jan -> 31
    Dec), since their day gaps vary; anything finer steps back the typical
    gap between observations."""
    first = idx[0]
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(float) if len(idx) > 1 else np.array([1.0])
    med = float(np.median(gaps)) if len(gaps) else 1.0
    if first.is_month_end and 25 <= med <= 35:
        return first - pd.offsets.MonthEnd(1)
    if first.is_quarter_end and 85 <= med <= 95:
        return first - pd.offsets.QuarterEnd(1)
    if first.is_year_end and 360 <= med <= 370:
        return first - pd.offsets.YearEnd(1)
    return first - pd.Timedelta(days=max(1.0, round(med)))


def to_equity(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Value curve from returns, starting at `initial` one period BEFORE
    the first return.

    The base point matters. Without it the curve's first value already
    includes the first period's return, so every measure read off the curve
    -- total return, CAGR, max drawdown, "since inception" -- silently
    starts one period late and never sees the first return at all. On a
    monthly track record that opens with a bad month, the loss simply
    vanished from the headline numbers.
    """
    r = returns.fillna(0.0)
    if r.empty:
        return pd.Series(dtype=float)
    idx = r.index
    base_date = _base_date(idx) if isinstance(idx, pd.DatetimeIndex) else -1
    eq = initial * (1.0 + r).cumprod()
    base = pd.Series([float(initial)], index=[base_date])
    out = pd.concat([base, eq])
    out.name = returns.name
    return out


# CIPM convention (Bacon, ch. on risk): every dispersion statistic uses the
# POPULATION moments (divide by n, not n-1), and a return is annualised
# geometrically -- never as an arithmetic mean times periods-per-year -- and
# only when the track record covers a full year.
MIN_YEAR_FRACTION = 0.95        # a "year" of daily data can hold 240-252 rows


def pop_std(x) -> float:
    """Population standard deviation (divide by n)."""
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    return float(np.sqrt(np.mean((a - a.mean()) ** 2))) if len(a) else np.nan


def pop_skew(x) -> float:
    """Population skewness: sum((x-m)^3) / (n * sd^3)."""
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(a) < 3:
        return np.nan
    sd = pop_std(a)
    return float(np.mean((a - a.mean()) ** 3) / sd ** 3) if sd > 0 else np.nan


def pop_kurt(x, excess: bool = True) -> float:
    """Population kurtosis (a normal distribution gives 3). Excess kurtosis,
    the default, is that minus 3. Not Excel's KURT or pandas' kurtosis(),
    which are sample-adjusted."""
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(a) < 4:
        return np.nan
    sd = pop_std(a)
    if not sd > 0:
        return np.nan
    k = float(np.mean((a - a.mean()) ** 4) / sd ** 4)
    return k - 3.0 if excess else k


def long_enough(n_periods: int, ppy: int) -> bool:
    """True when `n_periods` of data cover (about) a full year."""
    return n_periods >= MIN_YEAR_FRACTION * max(1, int(ppy))


def ann_return(returns: pd.Series, ppy: int = TRADING_DAYS, strict: bool = False) -> float:
    """Annualised geometric return, [prod(1+r)]^(ppy/n) - 1. With
    `strict`, a record shorter than a year gives NaN: the extrapolation of
    a short window to a year is exactly what the standard forbids."""
    r = returns.dropna()
    if r.empty or (strict and not long_enough(len(r), ppy)):
        return np.nan
    g = float((1.0 + r).prod())
    return g ** (ppy / len(r)) - 1.0 if g > 0 else np.nan


def cagr(equity: pd.Series, ppy: int = TRADING_DAYS, strict: bool = False) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return np.nan
    # N values span N - 1 periods of growth.
    if strict and not long_enough(len(equity) - 1, ppy):
        return np.nan
    years = (len(equity) - 1) / ppy
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan


def annual_vol(returns: pd.Series, ppy: int = TRADING_DAYS) -> float:
    return pop_std(returns) * float(np.sqrt(ppy))


# Below this daily volatility, a period is effectively riskless (portfolio
# fully in cash): a return/risk ratio has no meaning there and would read in
# the hundreds. NaN instead of a flattering, misleading number.
RISKLESS_SD = 1e-5


# The risk-free rate: an annual rate, or a series of per-period risk-free
# returns (a T-bill, the cash the portfolio actually earned). Sharpe,
# Sortino and alpha are all measured in excess of it. Leaving it at zero
# credits a portfolio with the cash yield as though it were skill: over a
# long history that adds the whole T-bill return to a low-beta portfolio's
# alpha and inflates every Sharpe ratio.
RiskFree = Union[float, pd.Series, None]


def rf_per_period(index: pd.Index, rf: RiskFree, ppy: int = TRADING_DAYS) -> pd.Series:
    """Per-period risk-free returns aligned to `index`. An annual rate is
    converted to the per-period rate that compounds to it -- the engine's
    own convention for cash -- not divided by the number of periods."""
    if isinstance(rf, pd.Series):
        return rf.reindex(index).astype(float).ffill().fillna(0.0)
    rate = float(rf or 0.0)
    per = (1.0 + rate) ** (1.0 / max(1, int(ppy))) - 1.0 if rate > -1 else 0.0
    return pd.Series(per, index=index)


def excess_returns(returns: pd.Series, rf: RiskFree, ppy: int = TRADING_DAYS) -> pd.Series:
    return returns - rf_per_period(returns.index, rf, ppy)


def rf_from_yield(yield_pct: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Per-period risk-free returns from a quoted annual yield in percent
    (a T-bill rate such as Yahoo's ^IRX). Each period earns the yield
    known at its start -- the previous observation's date -- for the
    calendar days it spans, so a monthly, weekly or daily stream gets the
    right amount whatever its spacing."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    y = (yield_pct.dropna().sort_index() / 100.0)
    y.index = pd.DatetimeIndex(y.index).tz_localize(None).normalize()
    y = y[~y.index.duplicated(keep="last")]
    prev = pd.DatetimeIndex([_base_date(idx)] + list(idx[:-1]))
    known = y.reindex(y.index.union(prev)).ffill().reindex(prev)
    days = np.asarray((idx - prev).days, dtype=float)
    out = (1.0 + known.to_numpy()) ** (days / 365.25) - 1.0
    return pd.Series(out, index=idx).fillna(0.0)


def rf_annual(index: pd.Index, rf: RiskFree, ppy: int = TRADING_DAYS) -> float:
    """The risk-free rate as an annualised geometric return over `index`."""
    return ann_return(rf_per_period(index, rf, ppy), ppy)


def sharpe(returns: pd.Series, rf_pa: RiskFree = 0.0, ppy: int = TRADING_DAYS,
           strict: bool = False) -> float:
    """Sharpe ratio, CIPM form: (annualised geometric return - annualised
    geometric risk-free) / (population SD of returns x sqrt(periods)).

    The same risk-free series must be used for every portfolio compared;
    a negative rate is kept as it is. With `strict` it is NaN below a year.
    """
    r = returns.dropna()
    sd = pop_std(r)
    if not (sd and sd > RISKLESS_SD):
        return np.nan
    ra = ann_return(r, ppy, strict)
    if ra != ra:
        return np.nan
    return float((ra - rf_annual(r.index, rf_pa, ppy)) / (sd * np.sqrt(ppy)))


def _target_series(index: pd.Index, rf_pa: RiskFree, mar: Optional[float],
                   ppy: int) -> pd.Series:
    """The per-period return the downside measures compare against: a
    minimum acceptable return when one is given, else the risk-free."""
    if mar is not None:
        return pd.Series(float(mar), index=index)
    return rf_per_period(index, rf_pa, ppy)


def downside_deviation(returns: pd.Series, rf_pa: RiskFree = 0.0,
                       ppy: int = TRADING_DAYS, mar: Optional[float] = None) -> float:
    """Per-period downside deviation below the target (the risk-free rate,
    or the per-period minimum acceptable return `mar`): the root mean square
    of shortfalls, taken over ALL periods (a period above the target counts
    as a zero shortfall). Not the standard deviation of the losing periods
    around their own mean, which measures how much the losses vary rather
    than how large they are."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    ex = r - _target_series(r.index, rf_pa, mar, ppy)
    return float(np.sqrt((np.minimum(ex, 0.0) ** 2).mean()))


def sortino(returns: pd.Series, rf_pa: RiskFree = 0.0, ppy: int = TRADING_DAYS,
            mar: Optional[float] = None, strict: bool = False) -> float:
    """Sortino ratio: (annualised geometric return - annualised target) /
    (downside deviation x sqrt(periods)). The target is the risk-free rate
    unless a per-period minimum acceptable return `mar` is given."""
    r = returns.dropna()
    down = downside_deviation(r, rf_pa, ppy, mar)
    if not (down and down > RISKLESS_SD):
        return np.nan
    ra = ann_return(r, ppy, strict)
    if ra != ra:
        return np.nan
    tgt = ann_return(_target_series(r.index, rf_pa, mar, ppy), ppy)
    return float((ra - tgt) / (down * np.sqrt(ppy)))


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def drawdown_table(equity: pd.Series, top: int = 5,
                   ppy: int = TRADING_DAYS) -> pd.DataFrame:
    """Worst drawdown episodes, with duration and recovery time.

    Durations are counted in observations and labelled with the actual
    frequency, so a monthly series reports months rather than "days".
    Calendar days are given alongside, which is the figure that does not
    depend on the sampling at all.
    """
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

    # The observation unit, not the aggregation bucket: a daily series is
    # measured in sessions even though its Best/Worst figures are monthly.
    _, _, obs_plural, _, _ = freq_words(ppy)
    obs_plural = "Sessions" if ppy >= 200 else obs_plural.capitalize()
    to_trough = f"{obs_plural} to Trough"
    total = f"Total {obs_plural}"

    # An episode starts at the peak it falls from -- the last observation
    # before the first one under water -- not on that first underwater
    # observation, which would make a one-session crash take "0 sessions
    # to trough" and shorten every duration by one period.
    pos = {d: i for i, d in enumerate(equity.index)}
    episodes = [(equity.index[pos[s] - 1] if pos[s] > 0 else s, t, rec, depth)
                for s, t, rec, depth in episodes]

    rows = []
    for s, trough, rec, depth in sorted(episodes, key=lambda x: x[3])[:top]:
        end = rec if rec is not None else equity.index[-1]
        rows.append({
            "Start": s.date(),
            "Trough": trough.date(),
            "Recovery": rec.date() if rec is not None else "ongoing",
            "Drawdown": depth,
            to_trough: max(int(len(equity.loc[s:trough])) - 1, 0),
            total: max(int(len(equity.loc[s:end])) - 1, 0),
            "Calendar Days": int((end - s).days),
        })
    return pd.DataFrame(rows)


def calmar(equity: pd.Series, ppy: int = TRADING_DAYS, rf_ann: float = 0.0,
           strict: bool = False) -> float:
    """Calmar ratio: (annualised return - annualised risk-free) / |max
    drawdown|. Bacon's version nets off the risk-free rate; with the default
    `rf_ann=0` it is the classic CAGR / |max drawdown|."""
    mdd = abs(max_drawdown(equity))
    c = cagr(equity, ppy, strict)
    return float((c - rf_ann) / mdd) if mdd > 0 and c == c else np.nan


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


# Below this R-squared the regression explains too little of the portfolio
# for alpha, beta, Treynor or the appraisal ratio to mean anything.
R2_FLOOR = 0.70


def beta_alpha(returns: pd.Series, bench: pd.Series,
               rf_pa: RiskFree = 0.0, ppy: int = TRADING_DAYS,
               strict: bool = False) -> Dict[str, float]:
    """Relative statistics against a benchmark, over the dates both cover.

    - `outperformance`: the strategy's CAGR minus the benchmark's -- by how
      much it beat the benchmark each year. This is what the reports call
      Alpha.
    - `alpha`: Jensen's alpha, annualised geometrically as the CIPM text
      does: r~ - r~F - beta (b~ - r~F). It compares the strategy with the
      benchmark scaled to the same beta, so a less risky strategy can trail
      the benchmark and still show a positive risk-adjusted alpha.
    - `tracking_error`: population SD of the GEOMETRIC excess return
      (1+r)/(1+b)-1 per period, times sqrt(periods). `information_ratio`
      is the annualised geometric excess return over that.
    - `r2_reliable`: False when R-squared < 0.70, in which case alpha and
      beta (and everything derived from them) should not be quoted.
    """
    df = pd.concat([returns, bench], axis=1).dropna()
    outperf = np.nan
    if len(df) and not (strict and not long_enough(len(df), ppy)):
        g = (1.0 + df).prod()
        if (g > 0).all():
            outperf = float(g.iloc[0] ** (ppy / len(df)) - g.iloc[1] ** (ppy / len(df)))
    if len(df) < 30:
        return {"beta": np.nan, "alpha": np.nan, "r2": np.nan, "corr": np.nan,
                "tracking_error": np.nan, "information_ratio": np.nan,
                "outperformance": outperf, "r2_reliable": False}
    rf = rf_per_period(df.index, rf_pa, ppy)
    y = df.iloc[:, 0] - rf
    x = df.iloc[:, 1] - rf
    var_x = x.var(ddof=1)
    b = float(np.cov(y, x, ddof=1)[0, 1] / var_x) if var_x > 0 else np.nan
    corr = float(np.corrcoef(y, x)[0, 1])
    rp = ann_return(df.iloc[:, 0], ppy, strict)
    rb = ann_return(df.iloc[:, 1], ppy, strict)
    rfa = ann_return(rf, ppy)
    ok = rp == rp and rb == rb
    a = float(rp - rfa - b * (rb - rfa)) if ok else np.nan
    geo = (1.0 + df.iloc[:, 0]) / (1.0 + df.iloc[:, 1]) - 1.0
    te = pop_std(geo) * float(np.sqrt(ppy))
    ir = float(((1.0 + rp) / (1.0 + rb) - 1.0) / te) if ok and te > 0 else np.nan
    return {
        "beta": b, "alpha": a, "r2": corr ** 2, "corr": corr,
        "outperformance": outperf, "tracking_error": te,
        "information_ratio": ir, "r2_reliable": bool(corr ** 2 >= R2_FLOOR),
    }


def monthly_returns(returns: pd.Series, ppy: int = TRADING_DAYS) -> pd.DataFrame:
    """Year x month table, in decimal.

    Empty for streams coarser than monthly: a quarterly series has no
    month-by-month breakdown to show, and resampling one into months would
    invent buckets that were never observed.
    """
    if ppy < 12:
        return pd.DataFrame()
    m = (1 + returns.fillna(0)).resample("ME").prod() - 1
    if m.empty:
        return pd.DataFrame()
    t = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "r": m.values})
    return t.pivot(index="Year", columns="Month", values="r")


def rolling_sharpe(returns: pd.Series, window: int = 252,
                   ppy: int = TRADING_DAYS, rf_pa: RiskFree = 0.0) -> pd.Series:
    """Rolling Sharpe, same definition as `sharpe`: geometric return in
    excess of the (geometric) risk-free, over the population SD."""
    lr = np.log1p(returns)
    ann = np.expm1(lr.rolling(window).sum() * ppy / window)
    rf = rf_per_period(returns.index, rf_pa, ppy)
    ann_rf = np.expm1(np.log1p(rf).rolling(window).sum() * ppy / window)
    sd = returns.rolling(window).std(ddof=0)
    return (ann - ann_rf) / (sd * np.sqrt(ppy))


# ----------------------------------------------------------------------
# Full summary
# ----------------------------------------------------------------------
def summary(returns: pd.Series,
            equity: Optional[pd.Series] = None,
            bench_returns: Optional[pd.Series] = None,
            turnover: Optional[pd.Series] = None,
            exposure: Optional[pd.Series] = None,
            rf_pa: RiskFree = 0.0,
            ppy: int = TRADING_DAYS) -> Dict[str, float]:
    returns = returns.dropna()
    eq = equity if equity is not None else to_equity(returns)
    vc = var_cvar(returns)
    # Only aggregate when the stream is finer than a month. Resampling a
    # monthly series to months is a no-op; resampling a quarterly one to
    # months would invent empty buckets.
    monthly = ((1 + returns).resample("ME").prod() - 1) if ppy > 12 else returns

    adj, _, _, unit, units = freq_words(ppy)
    out: Dict[str, float] = {
        "Total Return": float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else np.nan,
        # Annualised ratios need a full year of history (CIPM: never
        # annualise a shorter record); they read "n/a" below it.
        "CAGR": cagr(eq, ppy, strict=True),
        "Volatility": annual_vol(returns, ppy),
        "Sharpe": sharpe(returns, rf_pa, ppy, strict=True),
        "Sortino": sortino(returns, rf_pa, ppy, strict=True),
        "Calmar": calmar(eq, ppy, rf_annual(returns.index, rf_pa, ppy), strict=True),
        "Max Drawdown": max_drawdown(eq),
        "Ulcer Index": ulcer_index(eq),
        # VaR / CVaR are reported as positive losses (CIPM convention);
        # horizon and confidence are in the label.
        var_label(ppy): -vc["var"],
        cvar_label(ppy): -vc["cvar"],
        "Skew": pop_skew(returns),
        "Excess Kurtosis": pop_kurt(returns),
        f"% Positive {units}": float((monthly > 0).mean()) if len(monthly) else np.nan,
        f"Best {unit}": float(monthly.max()) if len(monthly) else np.nan,
        f"Worst {unit}": float(monthly.min()) if len(monthly) else np.nan,
    }
    if turnover is not None and len(turnover):
        out["Annual Turnover"] = float(turnover.sum() / (len(returns) / ppy))
    if exposure is not None and len(exposure):
        out["Average Exposure"] = float(exposure.mean())
    if bench_returns is not None:
        ba = beta_alpha(returns, bench_returns, rf_pa, ppy, strict=True)
        # CIPM: with R-squared under 0.70 the regression explains too little
        # of the portfolio, so beta and the risk-adjusted alpha are withheld
        # (the R-squared stays in the table, as the reason). The plain
        # outperformance, which is not a regression result, is unaffected.
        ok = bool(ba["r2_reliable"]) or ba["r2"] != ba["r2"]
        out.update({
            "Beta": ba["beta"] if ok else np.nan,
            "Alpha (ann.)": ba["outperformance"],
            "Risk-adjusted alpha (ann.)": ba["alpha"] if ok else np.nan,
            "R\u00b2": ba["r2"],
            "Tracking Error": ba["tracking_error"],
            "Information Ratio": ba["information_ratio"],
        })
    return out


FORMATS = {
    "Total Return": "pct", "CAGR": "pct", "Volatility": "pct",
    "Max Drawdown": "pct", "VaR 95% (daily)": "pct", "CVaR 95% (daily)": "pct",
    "% Positive Months": "pct", "Best Month": "pct", "Worst Month": "pct",
    "Alpha (ann.)": "pct", "Risk-adjusted alpha (ann.)": "pct",
    "Tracking Error": "pct", "Average Exposure": "pct",
}


_PCT_PREFIXES = ("VaR 95%", "CVaR 95%", "% Positive", "Best ", "Worst ")


def is_pct_metric(key: str) -> bool:
    """Whether a summary() statistic is a fraction shown as a percentage."""
    return FORMATS.get(key) == "pct" or key.startswith(_PCT_PREFIXES)


def format_metric(key: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if is_pct_metric(key):
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
    end_val = float(eq.iloc[-1])

    # A return is measured *between* two observations, so the base is the
    # last point at or before the cutoff -- not the first point after it.
    #
    # With daily data the distinction is nearly invisible: the last close
    # before Jan 1 and the first close after it are a day apart. On a
    # monthly series it is a whole period. Taking the first point inside
    # the window makes YTD start at the January close, silently discarding
    # January's own return -- the single month most likely to matter.
    positions = eq.index
    rows = []
    for label, months in TRAILING_PERIODS:
        if months is None:
            # Strictly before Jan 1: the base is the prior year's final
            # observation. Using <= would select Jan 1 itself whenever the
            # index happens to contain it, discarding that day's return.
            cutoff = pd.Timestamp(year=end.year, month=1, day=1) - pd.Timedelta(days=1)
        elif months < 0:
            cutoff = None                      # since inception
        else:
            cutoff = end - pd.DateOffset(months=months)
            # Month-end to month-end. 30 Sep less one month is 30 Aug, and
            # the last month-end on or before that is 31 Jul: the "1M"
            # window would span two months, "6M" seven.
            if end.is_month_end:
                cutoff = cutoff + pd.offsets.MonthEnd(0)

        if cutoff is None:
            base_pos = 0
        else:
            prior = positions[positions <= cutoff]
            if len(prior) == 0:
                # The window reaches back beyond the data. Reporting it
                # would measure a shorter span under a longer label.
                continue
            base_pos = len(prior) - 1

        if base_pos >= len(eq) - 1:
            continue                            # no completed period yet

        window = eq.iloc[base_pos:]
        base = float(window.iloc[0])
        if base <= 0:
            continue

        total = end_val / base - 1.0
        # Whether to annualize is a calendar question (more than a year);
        # how many years is counted in periods, exactly as CAGR counts
        # them, so "Since inception" equals the CAGR to the decimal.
        span_years = (window.index[-1] - window.index[0]).days / 365.25
        annualized = span_years > 1.02
        years = max((len(window) - 1) / ppy, 1e-9)
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
