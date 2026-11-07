"""CIPM / GIPS-grade performance measurement, appraisal and attribution.

Pure functions, no I/O, decimals throughout (0.05 = 5%). The conventions
follow the CIPM program and Carl Bacon, *Practical Portfolio Performance
Measurement and Attribution* (3rd ed.):

- returns: true time-weighted, Modified Dietz, ex-post IRR, linking, strict
  geometric annualisation (never under a year);
- appraisal: population moments, geometric excess return, Sharpe family,
  M2, Jensen alpha with the R-squared floor, tracking error and information
  ratio, drawdown ratios (Calmar, Burke, Pain, Martin), partial moments;
- attribution: Brinson-Hood-Beebower, Brinson-Fachler and Bacon's geometric
  model, with Carino, Menchero, GRAP and Frongello multi-period linking and a
  naive-geometric currency split;
- GIPS composite helpers: beginning-value weighting, equal-weighted
  dispersion, 3-year ex-post standard deviation.

The headline Sharpe / Sortino / Calmar / alpha / IR in `qbt.metrics` are the
same definitions; this module adds the rest of the toolkit and the golden
tests (tests/test_cipm.py) pin both to Bacon's worked examples.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics as M

# ----------------------------------------------------------------------
# Methodology disclosure
# ----------------------------------------------------------------------
DEFAULT_METHOD = {
    "returns": "Daily valuation, time-weighted (engine NAV); dividends and fees internal",
    "excess": "Geometric (1+r)/(1+b)-1",
    "risk": "Population SD (divide by n), annualised by sqrt(periods)",
    "annualise": "Geometric; never for periods under one year",
    "alpha": "Jensen, geometric annualised; withheld when R-squared < 0.70",
    "attribution": "Brinson-Fachler (interaction in selection) / Bacon geometric",
}


def methodology_block(ppy: int, rf: str = "model risk-free", benchmark: str = "benchmark",
                      n_obs: Optional[int] = None, extra: Optional[Dict[str, str]] = None) -> str:
    """The one-paragraph disclosure the standard asks every deliverable to carry."""
    freq = M.freq_words(ppy)[0]
    obs = f", {n_obs} obs" if n_obs else ""
    parts = [
        f"Methodology: {DEFAULT_METHOD['returns']}",
        f"Benchmark: {benchmark}",
        f"Excess: geometric",
        f"Risk: {freq}{obs}, population SD, annualised sqrt({ppy})",
        f"rf: {rf}",
        f"Annualisation: geometric, full years only",
    ]
    for k, v in (extra or {}).items():
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


# ----------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------
def day_weight(td: int, d: int, timing: str = "eod") -> float:
    """Weight of an external flow on day `d` of a `td`-day period.
    End-of-day: (TD-D)/TD. Start-of-day: (TD-D+1)/TD."""
    return (td - d) / td if timing == "eod" else (td - d + 1) / td


def simple_dietz(v0: float, v1: float, flows: Sequence[float]) -> float:
    c = float(np.sum(flows))
    return (v1 - v0 - c) / (v0 + c / 2.0)


def modified_dietz(v0: float, v1: float, flows: Sequence[float], days: Sequence[int],
                   td: int, timing: str = "eod") -> float:
    """Money-weighted approximation of the return: gain over start value
    plus day-weighted flows. A money-weighted measure, not a time-weighted
    one. The denominator is NOT the average of start and end values."""
    f = np.asarray(flows, float)
    w = np.array([day_weight(td, d, timing) for d in days])
    return float((v1 - v0 - f.sum()) / (v0 + (f * w).sum()))


def irr_expost(v0: float, v1: float, flows: Sequence[float], days: Sequence[int],
               td: int, timing: str = "eod") -> float:
    """Ex-post IRR over the period (daily-dated flows). Money-weighted: it
    cannot be disaggregated, so it cannot be attributed."""
    from scipy.optimize import brentq
    w = [day_weight(td, d, timing) for d in days]
    return float(brentq(lambda r: v0 * (1 + r) + sum(c * (1 + r) ** x for c, x in zip(flows, w)) - v1,
                        -0.99, 10))


def twr(values_before_flow: Sequence[float], flows: Sequence[float], v0: float, v_end: float) -> float:
    """True time-weighted return: revalue at every external flow.
    values_before_flow[k] is the portfolio value just before flow k."""
    wr, start = 1.0, v0
    for vb, c in zip(values_before_flow, flows):
        wr *= vb / start
        start = vb + c
    return float(wr * v_end / start - 1.0)


def link(r: Iterable[float]) -> float:
    return float(np.prod(1.0 + np.asarray(list(r), float)) - 1.0)


def annualise(r: Sequence[float], t: int) -> float:
    """Annualise a series of `t`-per-year returns. Refuses under one year."""
    r = np.asarray(r, float)
    if len(r) < t:
        raise ValueError("never annualise a period shorter than one year")
    return float(np.prod(1.0 + r) ** (t / len(r)) - 1.0)


def geometric_excess(r, b):
    return (1.0 + np.asarray(r, float)) / (1.0 + np.asarray(b, float)) - 1.0


# ----------------------------------------------------------------------
# Appraisal
# ----------------------------------------------------------------------
def _s(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, float))


def adjusted_sharpe(returns, rf_pa=0.0, ppy: int = M.TRADING_DAYS, strict: bool = False) -> float:
    """Pezier-White: SR [1 + (skew/6) SR - ((kurt-3)/24) SR^2]."""
    r = _s(returns)
    sr = M.sharpe(r, rf_pa, ppy, strict)
    if sr != sr:
        return np.nan
    sk, ek = M.pop_skew(r), M.pop_kurt(r, excess=True)
    return float(sr * (1 + sk / 6.0 * sr - ek / 24.0 * sr ** 2))


def skew_adjusted_sharpe(returns, rf_pa=0.0, ppy: int = M.TRADING_DAYS, strict: bool = False) -> float:
    r = _s(returns)
    sr = M.sharpe(r, rf_pa, ppy, strict)
    return float(sr * (1 + M.pop_skew(r) / 6.0 * sr)) if sr == sr else np.nan


def m2(returns, bench, rf_pa=0.0, ppy: int = M.TRADING_DAYS, adjusted: bool = False) -> float:
    """Modigliani-Modigliani: the return the portfolio would have had at the
    benchmark's risk. r + SR (sigma_b - sigma_p), all annualised."""
    r, b = _s(returns), _s(bench)
    sr = adjusted_sharpe(r, rf_pa, ppy) if adjusted else M.sharpe(r, rf_pa, ppy)
    return float(M.ann_return(r, ppy) + sr * (M.annual_vol(b, ppy) - M.annual_vol(r, ppy)))


def differential_return(returns, bench, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> float:
    """r - [rF + (b - rF) sigma_p / sigma_b]."""
    r, b = _s(returns), _s(bench)
    rfa = M.rf_annual(r.index, rf_pa, ppy)
    return float(M.ann_return(r, ppy) - (rfa + (M.ann_return(b, ppy) - rfa)
                                         * M.annual_vol(r, ppy) / M.annual_vol(b, ppy)))


def capm_beta(returns, bench, rf=0.0) -> float:
    """Beta of excess returns over a per-period risk-free (scalar or series)."""
    pe, be = np.asarray(returns, float) - rf, np.asarray(bench, float) - rf
    return float(np.sum((pe - pe.mean()) * (be - be.mean())) / np.sum((be - be.mean()) ** 2))


def jensen_alpha(returns, bench, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> float:
    """Annualised geometric Jensen alpha: r - rF - beta (b - rF)."""
    r, b = _s(returns), _s(bench)
    return M.beta_alpha(r, b, rf_pa, ppy)["alpha"]


def treynor(returns, bench, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> float:
    r, b = _s(returns), _s(bench)
    rf = M.rf_per_period(r.index, rf_pa, ppy)
    beta = capm_beta(r.to_numpy(), b.to_numpy(), rf.to_numpy())
    return float((M.ann_return(r, ppy) - M.rf_annual(r.index, rf_pa, ppy)) / beta)


def appraisal_ratio(returns, bench, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> float:
    """Annualised Jensen alpha over the annualised specific (residual) risk.
    Not the information ratio."""
    r, b = _s(returns), _s(bench)
    rf = M.rf_per_period(r.index, rf_pa, ppy).to_numpy()
    y, x = r.to_numpy() - rf, b.to_numpy() - rf
    beta = capm_beta(r.to_numpy(), b.to_numpy(), rf)
    resid = y - beta * x
    resid = resid - resid.mean()                       # remove the intercept
    sig = float(np.sqrt(np.mean(resid ** 2)) * np.sqrt(ppy))
    a = M.beta_alpha(r, b, rf_pa, ppy)["alpha"]
    return float(a / sig) if sig > 0 else np.nan


def bull_bear_betas(returns, bench, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> Dict[str, float]:
    """Betas estimated separately in up and down benchmark periods; the
    timing ratio is bull / bear (above 1 is desirable)."""
    r, b = _s(returns), _s(bench)
    df = pd.concat([r, b], axis=1).dropna()
    rf = M.rf_per_period(df.index, rf_pa, ppy)
    y, x = df.iloc[:, 0] - rf, df.iloc[:, 1] - rf
    out = {}
    for name, mask in (("bull_beta", x > 0), ("bear_beta", x <= 0)):
        xs, ys = x[mask], y[mask]
        v = float(((xs - xs.mean()) ** 2).sum())
        out[name] = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / v) if len(xs) > 2 and v > 0 else np.nan
    bear = out["bear_beta"]
    out["timing_ratio"] = out["bull_beta"] / bear if bear == bear and bear != 0 else np.nan
    return out


def capture_ratios(returns, bench, ppy: int = M.TRADING_DAYS) -> Dict[str, float]:
    """Up / down capture: the strategy's compound return in the benchmark's
    up (down) periods over the benchmark's, as a ratio."""
    df = pd.concat([_s(returns), _s(bench)], axis=1).dropna()
    out = {}
    for name, mask in (("up_capture", df.iloc[:, 1] > 0), ("down_capture", df.iloc[:, 1] < 0)):
        sub = df[mask]
        if len(sub) < 2:
            out[name] = np.nan
            continue
        n = len(sub)
        a = (1 + sub.iloc[:, 0]).prod() ** (1.0 / n) - 1
        c = (1 + sub.iloc[:, 1]).prod() ** (1.0 / n) - 1
        out[name] = float(a / c) if c != 0 else np.nan
    return out


def active_share(w, W) -> float:
    """1/2 sum |w_i - W_i| (weights aligned by position)."""
    return float(0.5 * np.abs(np.asarray(w, float) - np.asarray(W, float)).sum())


def information_ratio_tstat(ir: float, years: float) -> float:
    """t-statistic of skill: IR x sqrt(years). About 2 is needed."""
    return float(ir * np.sqrt(years))


def years_for_significance(ir: float, t: float = 2.0) -> float:
    """Years of track record at which an information ratio reaches a t-stat
    of `t` (IR 1.0 -> 4 years, 0.75 -> ~7, 0.5 -> 16)."""
    return float((t / ir) ** 2) if ir and ir > 0 else np.inf


def _drawdown_series(returns) -> np.ndarray:
    w = np.cumprod(1.0 + np.asarray(returns, float))
    w = np.concatenate([[1.0], w])
    return w / np.maximum.accumulate(w) - 1.0


def pain_index(returns) -> float:
    return float(np.mean(np.abs(_drawdown_series(returns))))


def ulcer(returns) -> float:
    return float(np.sqrt(np.mean(_drawdown_series(returns) ** 2)))


def _drawdown_episodes(returns) -> List[float]:
    dd, depth, out = _drawdown_series(returns), 0.0, []
    for x in dd:
        if x < 0:
            depth = min(depth, x)
        elif depth < 0:
            out.append(depth)
            depth = 0.0
    if depth < 0:
        out.append(depth)
    return out


def burke_ratio(returns, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> float:
    """(r - rF) / sqrt(sum of squared drawdown depths, one per episode)."""
    r = _s(returns)
    eps = _drawdown_episodes(r.to_numpy())
    den = float(np.sqrt(np.sum(np.square(eps))))
    return float((M.ann_return(r, ppy) - M.rf_annual(r.index, rf_pa, ppy)) / den) if den > 0 else np.nan


def pain_ratio(returns, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> float:
    r = _s(returns)
    pi = pain_index(r.to_numpy())
    return float((M.ann_return(r, ppy) - M.rf_annual(r.index, rf_pa, ppy)) / pi) if pi > 0 else np.nan


def martin_ratio(returns, rf_pa=0.0, ppy: int = M.TRADING_DAYS) -> float:
    r = _s(returns)
    u = ulcer(r.to_numpy())
    return float((M.ann_return(r, ppy) - M.rf_annual(r.index, rf_pa, ppy)) / u) if u > 0 else np.nan


def omega(returns, mar: float = 0.0) -> float:
    """Sum of gains over the target / sum of shortfalls below it."""
    p = np.asarray(returns, float)
    loss = np.mean(np.maximum(mar - p, 0))
    return float(np.mean(np.maximum(p - mar, 0)) / loss) if loss > 0 else np.nan


def upside_potential_ratio(returns, mar: float = 0.0) -> float:
    p = np.asarray(returns, float)
    dd = np.sqrt(np.mean(np.minimum(p - mar, 0) ** 2))
    return float(np.mean(np.maximum(p - mar, 0)) / dd) if dd > 0 else np.nan


def expected_shortfall(returns, level: float = 0.05) -> float:
    """CVaR as a positive loss (mean of the worst `level` of periods)."""
    r = np.asarray(pd.Series(returns).dropna(), float)
    if not len(r):
        return np.nan
    q = np.quantile(r, level)
    tail = r[r <= q]
    return float(-(tail.mean() if len(tail) else q))


# Minimum observations before a statistic is worth quoting (Bacon).
MIN_OBS = {12: 36, 4: 20, 52: 156, 252: 756}
ABSOLUTE_FLOOR = {12: 24, 4: 12, 52: 104, 252: 504}


def data_sufficiency(n: int, ppy: int) -> Dict[str, object]:
    """'ok', 'short' (below the recommended length) or 'insufficient'."""
    rec = MIN_OBS.get(ppy, int(3 * ppy))
    floor = ABSOLUTE_FLOOR.get(ppy, int(2 * ppy))
    status = "ok" if n >= rec else ("short" if n >= floor else "insufficient")
    return {"status": status, "n": n, "recommended": rec, "floor": floor}


def appraisal_table(returns: pd.Series, bench: Optional[pd.Series] = None,
                    rf_pa: M.RiskFree = 0.0, ppy: int = M.TRADING_DAYS,
                    mar_pa: float = 0.0) -> pd.DataFrame:
    """One consistent appraisal: every measure on the same dates, the same
    rf, the same frequency. Measures that depend on a regression are
    withheld (value NaN, the reason in `Note`) when R-squared < 0.70 --
    never shown with a caveat the reader can skip."""
    r = returns.dropna()
    rows: List[Dict[str, object]] = []

    def add(group, name, val, note=""):
        rows.append({"Group": group, "Measure": name, "Value": val, "Note": note})

    suff = data_sufficiency(len(r), ppy)
    if suff["status"] != "ok":
        add("Data", "Sample length", float(suff["n"]),
            f"{suff['status']}: {suff['recommended']} observations recommended at this frequency")
    ra = M.ann_return(r, ppy, strict=True)
    rfa = M.rf_annual(r.index, rf_pa, ppy)
    add("Return", "Annualised return (geometric)", ra, "n/a under one year" if ra != ra else "")
    add("Return", "Annualised risk-free (geometric)", rfa)
    add("Total risk", "Volatility (population, annualised)", M.annual_vol(r, ppy))
    add("Total risk", "Skewness (population)", M.pop_skew(r))
    add("Total risk", "Kurtosis (normal = 3)", M.pop_kurt(r, excess=False))
    add("Total risk", "Sharpe", M.sharpe(r, rf_pa, ppy, strict=True))
    add("Total risk", "Adjusted Sharpe (Pezier-White)", adjusted_sharpe(r, rf_pa, ppy, strict=True),
        "corrects Sharpe for skew and fat tails")
    add("Total risk", "Skew-adjusted Sharpe", skew_adjusted_sharpe(r, rf_pa, ppy, strict=True))
    mar = (1 + mar_pa) ** (1.0 / ppy) - 1
    add("Downside", "Downside risk (annualised)", M.downside_deviation(r, rf_pa, ppy, mar) * np.sqrt(ppy),
        f"target {mar_pa:.2%} a year")
    add("Downside", "Sortino", M.sortino(r, rf_pa, ppy, mar, strict=True), f"target {mar_pa:.2%} a year")
    add("Downside", "Omega", omega(r, mar), "gains over shortfalls vs target")
    add("Downside", "Upside potential ratio", upside_potential_ratio(r, mar))
    add("Downside", "Expected shortfall 95% (per period)", expected_shortfall(r, 0.05))
    eq = M.to_equity(r)
    mdd = M.max_drawdown(eq)
    add("Drawdown", "Max drawdown", mdd)
    add("Drawdown", "Calmar (rf-adjusted)", M.calmar(eq, ppy, rfa, strict=True))
    add("Drawdown", "Burke", burke_ratio(r, rf_pa, ppy))
    add("Drawdown", "Pain ratio", pain_ratio(r, rf_pa, ppy))
    add("Drawdown", "Martin (Ulcer) ratio", martin_ratio(r, rf_pa, ppy))

    if bench is not None:
        b = bench.reindex(r.index).dropna()
        r2 = r.reindex(b.index)
        if len(b) >= 30:
            ba = M.beta_alpha(r2, b, rf_pa, ppy, strict=True)
            reliable = ba["r2_reliable"]
            why = "" if reliable else f"withheld: R² {ba['r2']:.2f} is below {M.R2_FLOOR:.2f}"
            nan = np.nan
            add("Relative", "Tracking error (geometric)", ba["tracking_error"],
                "low is not good in itself")
            ir = ba["information_ratio"]
            add("Relative", "Information ratio (geometric)", ir)
            yrs = len(b) / ppy
            if ir == ir:
                add("Relative", "Skill t-statistic (IR x sqrt years)", information_ratio_tstat(ir, yrs),
                    "about 2 is needed")
                add("Relative", "Years needed for t = 2",
                    years_for_significance(ir) if ir > 0 else nan,
                    "" if ir > 0 else "no positive skill signal")
            add("Relative", "M² (return at benchmark risk)", m2(r2, b, rf_pa, ppy))
            add("Relative", "Differential return", differential_return(r2, b, rf_pa, ppy))
            add("Relative", "R² vs benchmark", ba["r2"], "regression fit")
            add("CAPM", "Beta", ba["beta"] if reliable else nan, why)
            add("CAPM", "Jensen alpha (annualised)", ba["alpha"] if reliable else nan, why)
            add("CAPM", "Treynor", treynor(r2, b, rf_pa, ppy) if reliable else nan, why)
            add("CAPM", "Appraisal ratio (alpha / specific risk)",
                appraisal_ratio(r2, b, rf_pa, ppy) if reliable else nan, why)
            bb = bull_bear_betas(r2, b, rf_pa, ppy)
            add("CAPM", "Bull beta", bb["bull_beta"] if reliable else nan, why)
            add("CAPM", "Bear beta", bb["bear_beta"] if reliable else nan, why)
            add("CAPM", "Timing ratio (bull / bear)", bb["timing_ratio"] if reliable else nan,
                why or "above 1 is desirable")
            cap = capture_ratios(r2, b, ppy)
            add("Capture", "Up capture", cap["up_capture"])
            add("Capture", "Down capture", cap["down_capture"], "below 1 is desirable")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Single-period attribution (category arrays: w, W weights; r, b returns)
# ----------------------------------------------------------------------
def _arrays(w, W, r, b):
    w, W, r, b = (np.asarray(x, float) for x in (w, W, r, b))
    if not (len(w) == len(W) == len(r) == len(b)):
        raise ValueError("w, W, r, b must have one entry per category")
    for name, x in (("portfolio weights", w), ("benchmark weights", W)):
        if not np.isclose(x.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{name} must sum to 100% (cash is its own category), got {x.sum():.6f}")
    if np.isnan(b).any():
        raise ValueError("every category held needs a benchmark return; an off-benchmark category "
                         "must be given the proxy return that matches the decision process, "
                         "never an implicit 0%")
    return w, W, r, b


def bhb(w, W, r, b) -> Dict[str, np.ndarray]:
    """Brinson-Hood-Beebower: allocation, selection and interaction apart."""
    w, W, r, b = _arrays(w, W, r, b)
    return {"allocation": (w - W) * b, "selection": W * (r - b),
            "interaction": (w - W) * (r - b)}


def brinson_fachler(w, W, r, b) -> Dict[str, np.ndarray]:
    """Allocation rewards overweighting categories that beat the whole
    benchmark; selection includes interaction (the stock picker works with
    the weight he is given)."""
    w, W, r, b = _arrays(w, W, r, b)
    return {"allocation": (w - W) * (b - W @ b), "selection": w * (r - b)}


def geometric_attribution(w, W, r, b) -> Dict[str, np.ndarray]:
    """Bacon's geometric model. Totals compound exactly to the geometric
    excess return: (1+A)(1+S) = (1+r)/(1+b)."""
    w, W, r, b = _arrays(w, W, r, b)
    B, bS = W @ b, w @ b
    return {"allocation": (w - W) * ((1 + b) / (1 + B) - 1),
            "selection": w * ((1 + r) / (1 + b) - 1) * (1 + b) / (1 + bS)}


def semi_notional_return(w, b) -> float:
    """b_S = sum(w_i b_i): the portfolio's allocation at benchmark returns."""
    return float(np.asarray(w, float) @ np.asarray(b, float))


def attribution_table(categories: Sequence[str], w, W, r, b, model: str = "geometric") -> pd.DataFrame:
    """Categories x effects, with a Total row that reconciles to the excess
    return. `model` is 'geometric', 'brinson_fachler' or 'bhb'."""
    w, W, r, b = _arrays(w, W, r, b)
    fn = {"geometric": geometric_attribution, "brinson_fachler": brinson_fachler, "bhb": bhb}[model]
    eff = fn(w, W, r, b)
    df = pd.DataFrame({"Portfolio weight": w, "Benchmark weight": W,
                       "Portfolio return": r, "Benchmark return": b},
                      index=list(categories))
    for k, v in eff.items():
        df[k.capitalize()] = v
    df["Total effect"] = sum(eff.values())
    total = df[[c.capitalize() for c in eff] + ["Total effect"]].sum()
    tot_row = {c: np.nan for c in df.columns}
    tot_row.update({"Portfolio weight": w.sum(), "Benchmark weight": W.sum(),
                    "Portfolio return": float(w @ r), "Benchmark return": float(W @ b)})
    tot_row.update(total.to_dict())
    df.loc["Total"] = pd.Series(tot_row)
    return df


def reconciliation_error(w, W, r, b, model: str = "geometric") -> float:
    """Total effect minus the excess return it should explain. Must be ~0;
    a non-zero value is an error to find, never a line to plug."""
    w, W, r, b = _arrays(w, W, r, b)
    eff = {"geometric": geometric_attribution, "brinson_fachler": brinson_fachler, "bhb": bhb}[model](w, W, r, b)
    if model == "geometric":
        tot = (1 + eff["allocation"].sum()) * (1 + eff["selection"].sum()) - 1
        return float(tot - ((1 + w @ r) / (1 + W @ b) - 1))
    return float(sum(e.sum() for e in eff.values()) - (w @ r - W @ b))


# ----------------------------------------------------------------------
# Multi-period linking (E: list over periods of equal-shaped effect arrays)
# ----------------------------------------------------------------------
def _k(r, b):
    return 1.0 / (1.0 + r) if np.isclose(r, b) else (np.log1p(r) - np.log1p(b)) / (r - b)


def carino(E, rs, bs):
    k = _k(link(rs), link(bs))
    return [np.asarray(e, float) * _k(r, b) / k for e, r, b in zip(E, rs, bs)]


def menchero(E, rs, bs):
    R, B, T = link(rs), link(bs), len(rs)
    d = np.asarray(rs, float) - np.asarray(bs, float)
    M_ = (R - B) / (T * ((1 + R) ** (1 / T) - (1 + B) ** (1 / T)))
    C = ((R - B) - M_ * d.sum()) / np.sum(d ** 2)
    return [np.asarray(e, float) * (M_ + C * dt) for e, dt in zip(E, d)]


def grap(E, rs, bs):
    rs, bs = np.asarray(rs, float), np.asarray(bs, float)
    return [np.asarray(e, float) * np.prod(1 + rs[:t]) * np.prod(1 + bs[t + 1:]) for t, e in enumerate(E)]


def frongello(E, rs, bs):
    out, cum = [], np.zeros_like(np.asarray(E[0], float))
    for t, e in enumerate(E):
        x = np.asarray(e, float) * np.prod(1 + np.asarray(rs[:t], float)) + bs[t] * cum
        out.append(x)
        cum = cum + x
    return out


LINKERS = {"carino": carino, "menchero": menchero, "grap": grap, "frongello": frongello}


def link_arithmetic(E, rs, bs, method: str = "frongello"):
    """Adjust arithmetic effects so they sum to R - B over the whole span.
    A presentational convenience, not an economic model; always disclose it."""
    return LINKERS[method](E, rs, bs)


def multi_period_attribution(periods: Sequence[Dict[str, Sequence[float]]], categories: Sequence[str],
                             model: str = "geometric", linking: str = "frongello") -> Dict[str, object]:
    """Attribute several periods and link them.

    `periods` is a list of dicts with keys w, W, r, b (category arrays).
    Geometric effects compound with no adjustment; arithmetic ones (BF) are
    linked with `linking`. Returns the per-period table, the linked
    category x effect table and the reconciliation.
    """
    rs, bs, per = [], [], []
    for p in periods:
        w, W, r, b = _arrays(p["w"], p["W"], p["r"], p["b"])
        rs.append(float(w @ r))
        bs.append(float(W @ b))
        fn = {"geometric": geometric_attribution, "brinson_fachler": brinson_fachler, "bhb": bhb}[model]
        eff = fn(w, W, r, b)
        per.append(eff)
    names = list(per[0].keys())
    R, B = link(rs), link(bs)
    if model == "geometric":
        # totals compound; category rows are shown as linked contributions
        tot = {n: np.prod([1 + e[n].sum() for e in per]) - 1 for n in names}
        # Category rows: each period's effect carried forward by the compounded
        # totals of the periods before it, so the rows add to the linked total
        # exactly (prod(1+x_t) - 1 = sum_t x_t prod_{j<t}(1+x_j)).
        cat = {}
        for n in names:
            acc = np.zeros(len(categories))
            for t, e in enumerate(per):
                prior = np.prod([1 + per[j][n].sum() for j in range(t)]) if t else 1.0
                acc += e[n] * prior
            cat[n] = acc
        out = pd.DataFrame(cat, index=list(categories))
        out.columns = [c.capitalize() for c in out.columns]
        out.loc["Total"] = [tot[n] for n in names]
        recon = (1 + tot["allocation"]) * (1 + tot["selection"]) - 1 - ((1 + R) / (1 + B) - 1)
        return {"linked": out, "R": R, "B": B, "excess_geometric": (1 + R) / (1 + B) - 1,
                "reconciliation_error": float(recon), "linking": "geometric (compounded)"}
    E = [np.column_stack([e[n] for n in names]) for e in per]
    adj = link_arithmetic(E, rs, bs, linking)
    total = np.sum(adj, axis=0)
    out = pd.DataFrame(total, index=list(categories), columns=[n.capitalize() for n in names])
    out.loc["Total"] = out.sum()
    recon = float(total.sum() - (R - B))
    return {"linked": out, "R": R, "B": B, "excess_arithmetic": R - B,
            "reconciliation_error": recon, "linking": linking}


# ----------------------------------------------------------------------
# Currency (naive geometric)
# ----------------------------------------------------------------------
def currency_attribution_naive(r: float, r_local: float, b: float, b_local: float,
                               b_semi_local: float) -> Dict[str, float]:
    """Selection and allocation in local currency, currency as the rest.
    The three compound to (1+r)/(1+b) - 1."""
    sel = (1 + r_local) / (1 + b_semi_local) - 1
    alloc = (1 + b_semi_local) / (1 + b_local) - 1
    cur = ((1 + r) / (1 + r_local)) * ((1 + b_local) / (1 + b)) - 1
    return {"selection": sel, "allocation": alloc, "currency": cur,
            "total": (1 + sel) * (1 + alloc) * (1 + cur) - 1}


# ----------------------------------------------------------------------
# GIPS composite helpers
# ----------------------------------------------------------------------
def composite_return(returns: Sequence[float], beginning_values: Sequence[float]) -> float:
    """Beginning-value-weighted composite period return. Never equal-weight."""
    r, v = np.asarray(returns, float), np.asarray(beginning_values, float)
    return float((r * v).sum() / v.sum())


def internal_dispersion(returns: Sequence[float]) -> float:
    """Equal-weighted standard deviation of portfolio returns (population),
    Bacon's preferred GIPS dispersion measure. Not required for 5 or fewer
    portfolios."""
    return M.pop_std(np.asarray(returns, float))


def three_year_ex_post_sd(monthly_returns: pd.Series, as_of: Optional[pd.Timestamp] = None) -> float:
    """GIPS: 3-year annualised ex-post SD from the 36 months to `as_of`
    (NaN with fewer than 36 observations)."""
    s = monthly_returns.dropna()
    if as_of is not None:
        s = s.loc[:as_of]
    s = s.tail(36)
    return M.annual_vol(s, 12) if len(s) == 36 else np.nan


def gips_annual_table(composite: pd.Series, benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
    """Annual composite (and benchmark) returns from monthly returns, with
    partial years flagged, and the 3-year ex-post SD at each year-end."""
    c = composite.dropna()
    rows = []
    for yr, g in c.groupby(c.index.year):
        row = {"Year": int(yr), "Months": len(g), "Composite": float((1 + g).prod() - 1),
               "Partial": "yes" if len(g) < 12 else ""}
        if benchmark is not None:
            bg = benchmark.reindex(g.index)
            row["Benchmark"] = float((1 + bg).prod() - 1) if bg.notna().all() else np.nan
        row["3Y SD composite"] = three_year_ex_post_sd(c, g.index[-1]) if len(g) == 12 else np.nan
        if benchmark is not None:
            row["3Y SD benchmark"] = (three_year_ex_post_sd(benchmark.dropna(), g.index[-1])
                                       if len(g) == 12 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------
_PCT_MEASURES = {
    "Annualised return (geometric)", "Annualised risk-free (geometric)",
    "Volatility (population, annualised)", "Downside risk (annualised)",
    "Expected shortfall 95% (per period)", "Max drawdown", "Tracking error (geometric)",
    "M\u00b2 (return at benchmark risk)", "Differential return", "Jensen alpha (annualised)",
}


def format_appraisal(table: pd.DataFrame) -> pd.DataFrame:
    """Appraisal table with the Value column as display text."""
    def f(row):
        v = row["Value"]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if row["Measure"] in _PCT_MEASURES:
            return f"{v * 100:,.2f}%"
        if row["Measure"].startswith("Years needed"):
            return "never" if np.isinf(v) else f"{v:,.1f}"
        return f"{v:,.2f}"
    out = table.copy()
    out["Value"] = table.apply(f, axis=1)
    return out


def appraisal_compare(returns: pd.Series, bench: Optional[pd.Series] = None,
                      rf_pa: M.RiskFree = 0.0, ppy: int = M.TRADING_DAYS,
                      mar_pa: float = 0.0) -> pd.DataFrame:
    """The appraisal table with the benchmark's own value beside the
    strategy's, for the measures that exist for both (relative measures are
    blank in the benchmark column). Carries per-row number formats for the
    Excel export in `attrs`."""
    a = appraisal_table(returns, bench, rf_pa, ppy, mar_pa).rename(columns={"Value": "Strategy"})
    if bench is not None:
        b = bench.dropna()
        bt = appraisal_table(b, None, rf_pa, ppy, mar_pa).set_index("Measure")["Value"]
        bt = bt[~bt.index.duplicated()]
        a.insert(3, "Benchmark", a["Measure"].map(bt))
    fmts = []
    for m in a["Measure"]:
        if m in _PCT_MEASURES:
            fmts.append("0.00%;-0.00%;0.00%")
        elif m == "Sample length":
            fmts.append("0")
        elif m.startswith("Years needed"):
            fmts.append("0.0")
        else:
            fmts.append("0.00")
    a.attrs["row_fmt"] = fmts
    a.attrs["fmt_cols"] = [3, 4] if bench is not None else [3]
    return a
