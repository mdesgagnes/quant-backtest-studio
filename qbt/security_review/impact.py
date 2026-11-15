"""Portfolio-level impact of a candidate, risk of ruin, decumulation and the
cheap metrics re-computed on rolling windows.

All inputs are monthly CAD returns on the analysis period. "base" is the
profile without the candidate, "new" is the profile with it (Add or
Replace), both rebalanced by the same rule from the same start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import metrics as M
from . import stats as S
from .portfolio import diversification_ratio

PPY = 12


# ----------------------------------------------------------------------
# Block bootstrap (common random numbers for base and new)
# ----------------------------------------------------------------------
def bootstrap_index(n_obs: int, horizon: int, n_paths: int, block: int, seed: int) -> np.ndarray:
    """(n_paths, horizon) row indices: circular blocks of `block` months
    starting at uniformly drawn months. The same indices are applied to the
    portfolio with and without the candidate, so the comparison is not
    blurred by simulation noise."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n_obs, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_obs
    return idx.reshape(n_paths, -1)[:, :horizon]


def prob_drawdown(r: np.ndarray, idx: np.ndarray, thresholds: List[float]) -> Dict[float, float]:
    paths = r[idx]
    eq = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(eq), 1)), eq], axis=1), axis=1)[:, 1:]
    mdd = (eq / peak - 1).min(axis=1)
    return {t: float((mdd <= -t).mean()) for t in thresholds}


def decumulate(r: np.ndarray, idx: np.ndarray, rate: float, indexation: float) -> Dict[str, float]:
    """Wealth starts at 1; each month pays rate/12 of initial wealth, grown
    by `indexation` a year. A path that reaches 0 stays at 0."""
    paths = r[idx]
    n, h = paths.shape
    w = np.ones(n)
    depleted = np.zeros(n, dtype=bool)
    for m in range(h):
        pay = rate / 12 * (1 + indexation) ** (m // 12)
        w = w * (1 + paths[:, m]) - pay
        hit = w <= 0
        depleted |= hit
        w = np.where(depleted, 0.0, w)
    return {"depletion_prob": float(depleted.mean()),
            "p10": float(np.percentile(w, 10)), "p50": float(np.percentile(w, 50)),
            "p90": float(np.percentile(w, 90))}


def simulate(base: pd.Series, new: pd.Series, sim: Dict) -> Dict[str, object]:
    """Risk of ruin and decumulation for base and new, same random paths."""
    b, n = base.to_numpy(float), new.to_numpy(float)
    seed = int(sim.get("seed", 42))
    paths = int(sim.get("n_paths", 2000))
    block = int(sim.get("block_months", 12))
    ruin = sim.get("ruin") or {}
    th = [float(x) for x in ruin.get("drawdown_thresholds", [0.2, 0.3])]
    hr = int(round(float(ruin.get("horizon_years", 10)) * 12))
    idx = bootstrap_index(len(b), hr, paths, block, seed)
    pb, pn = prob_drawdown(b, idx, th), prob_drawdown(n, idx, th)
    dec = sim.get("decumulation") or {}
    hd = int(round(float(dec.get("horizon_years", 30)) * 12))
    idx2 = bootstrap_index(len(b), hd, paths, block, seed + 1)
    rate, infl = float(dec.get("withdrawal_rate", 0.04)), float(dec.get("inflation_indexation", 0.02))
    db, dn = decumulate(b, idx2, rate, infl), decumulate(n, idx2, rate, infl)
    rows = [{"Measure": f"Probability of a {t:.0%} drawdown within {hr // 12} years",
             "Without": pb[t], "With": pn[t], "Change": pn[t] - pb[t]} for t in th]
    for k, lab in (("depletion_prob", "Probability of running out of money"),
                   ("p10", "Ending wealth, 10th percentile (x initial)"),
                   ("p50", "Ending wealth, median (x initial)"),
                   ("p90", "Ending wealth, 90th percentile (x initial)")):
        rows.append({"Measure": f"{lab}, {hd // 12}y at {rate:.1%} withdrawal",
                     "Without": db[k], "With": dn[k], "Change": dn[k] - db[k]})
    return {"table": pd.DataFrame(rows), "ruin_base": pb, "ruin_new": pn,
            "dec_base": db, "dec_new": dn, "thresholds": th,
            "settings": {"seed": seed, "paths": paths, "block": block, "ruin_months": hr,
                         "decum_months": hd, "withdrawal": rate, "indexation": infl,
                         "history_months": len(b)}}


# ----------------------------------------------------------------------
# Impact metrics
# ----------------------------------------------------------------------
@dataclass
class Context:
    rf: pd.Series
    regimes: Dict[str, pd.Series]
    home: Optional[pd.Series]
    usd_market: Optional[pd.Series]          # own-currency (USD) returns
    usdcad: Optional[pd.Series]
    mkt_cad: Optional[pd.Series]             # SPY in CAD, control for the mega-cap tilt
    spread: Optional[pd.Series]              # mega-cap spread
    yields: Dict[str, Dict[str, float]]
    sim: Dict


def _tail_mask(base: pd.Series) -> pd.Series:
    q = base.quantile(0.10)
    return base <= q


def _coef(y: pd.Series, X: pd.DataFrame, name: str, min_n: int = 24) -> float:
    res = S.ols(y, X)
    if res["n"] < min_n or name not in res["coef"]:
        return np.nan
    return float(res["coef"][name])


def megacap_beta(r: pd.Series, ctx: Context) -> float:
    if ctx.mkt_cad is None or ctx.spread is None:
        return np.nan
    X = pd.concat([ctx.mkt_cad.rename("mkt"), ctx.spread.rename("spread")], axis=1, sort=True)
    return _coef(r, X, "spread")


def portfolio_yield(w: Dict[str, float], yields: Dict[str, Dict[str, float]]) -> float:
    tot = 0.0
    for t, x in w.items():
        y = (yields.get(t) or {}).get("dist_yield", np.nan)
        if y != y:
            return np.nan
        tot += x * y
    return tot


def regime_values(cand: pd.Series, base: pd.Series, new: pd.Series,
                  regimes: Dict[str, pd.Series], min_months: int = 3) -> Dict[str, float]:
    out = {}
    for key, lab in regimes.items():
        lab = lab.reindex(cand.index)
        mask = (lab == True) & lab.notna()  # noqa: E712  (object dtype with NaN)
        if int(mask.sum()) >= min_months:
            out[f"cand_regime_{key}"] = float(cand[mask].mean())
            out[f"d_regime_{key}"] = float((new[mask] - base[mask]).mean())
        else:
            out[f"cand_regime_{key}"] = np.nan
            out[f"d_regime_{key}"] = np.nan
    return out


def impact_metrics(cand: pd.Series, base: pd.Series, new: pd.Series,
                   rets: pd.DataFrame, base_w: Dict[str, float], new_w: Dict[str, float],
                   bench: Optional[pd.Series], ctx: Context, candidate: str,
                   simulate_paths: bool = True) -> Dict[str, object]:
    """Every catalogue metric for one candidate on one period, plus the
    tables behind them."""
    sb, sn = S.standalone(base, ctx.rf), S.standalone(new, ctx.rf)
    m: Dict[str, float] = {}
    for key, name in (("d_vol", "Volatility"), ("d_sharpe", "Sharpe"), ("d_sortino", "Sortino"),
                      ("d_calmar", "Calmar"), ("d_cagr", "CAGR"), ("d_mdd", "Max drawdown"),
                      ("d_ulcer", "Ulcer index"), ("d_cvar", "CVaR 95% (monthly)"),
                      ("d_worst_12m", "Worst 12 months")):
        m[key] = sn.get(name, np.nan) - sb.get(name, np.nan)
    # Return added: geometric excess of the portfolio with the candidate over
    # the portfolio without it (CIPM), not the difference of the two CAGRs.
    m["d_cagr"] = geo_excess(sn.get("CAGR", np.nan), sb.get("CAGR", np.nan))
    m["d_div_ratio"] = diversification_ratio(rets, new_w) - diversification_ratio(rets, base_w)
    m["corr_to_portfolio"] = float(cand.corr(base))
    tail = _tail_mask(base)
    m["worst10_corr"] = float(cand[tail].corr(base[tail])) if tail.sum() >= 6 else np.nan
    m["cand_tail_return"] = float(cand[tail].mean())
    m["d_tail_return"] = float((new[tail] - base[tail]).mean())
    up = base > 0
    m["d_up_month_return"] = float((new[up] - base[up]).mean()) if up.any() else np.nan
    ub, un = S.underwater(base), S.underwater(new)
    m["d_max_underwater"] = un["max_underwater"] - ub["max_underwater"]
    m["d_underwater_share"] = un["underwater_share"] - ub["underwater_share"]
    rel = S.relative(cand, bench, ctx.rf) if bench is not None else {}
    m["up_capture"] = rel.get("Upside capture", np.nan)
    m["down_capture"] = rel.get("Downside capture", np.nan)
    m.update(regime_values(cand, base, new, ctx.regimes))
    y = ctx.yields.get(candidate) or {}
    m["dist_yield"] = y.get("dist_yield", np.nan)
    m["yield_stability"] = y.get("yield_stability", np.nan)
    m["d_portfolio_yield"] = portfolio_yield(new_w, ctx.yields) - portfolio_yield(base_w, ctx.yields)
    m["corr_to_home_equity"] = float(cand.corr(ctx.home)) if ctx.home is not None else np.nan
    if ctx.usd_market is not None and ctx.usdcad is not None:
        m["usd_beta"] = _coef(cand, pd.concat([ctx.usd_market.rename("mkt"),
                                               ctx.usdcad.rename("usdcad")], axis=1, sort=True), "usdcad")
    else:
        m["usd_beta"] = np.nan
    if bench is not None and ctx.spread is not None:
        m["megacap_tilt"] = _coef((cand - bench).dropna(), ctx.spread.rename("spread").to_frame(),
                                  "spread")
    else:
        m["megacap_tilt"] = np.nan
    m["d_megacap_beta"] = megacap_beta(new, ctx) - megacap_beta(base, ctx)
    sim = None
    if simulate_paths and len(base) >= 24:
        sim = simulate(base, new, ctx.sim)
        for t in sim["thresholds"]:
            m[f"d_prob_dd{int(round(t * 100))}"] = sim["ruin_new"][t] - sim["ruin_base"][t]
        m["d_depletion_prob"] = sim["dec_new"]["depletion_prob"] - sim["dec_base"]["depletion_prob"]
        m["d_p10_wealth"] = sim["dec_new"]["p10"] - sim["dec_base"]["p10"]
        m["d_median_wealth"] = sim["dec_new"]["p50"] - sim["dec_base"]["p50"]
    return {"metrics": m, "base_stats": sb, "new_stats": sn, "relative": rel, "sim": sim}


# ----------------------------------------------------------------------
# Rolling windows
# ----------------------------------------------------------------------
_WINDOW_STATS = {"d_vol": "Volatility", "d_sharpe": "Sharpe", "d_cagr": "CAGR",
                 "d_mdd": "Max drawdown", "d_ulcer": "Ulcer index", "d_cvar": "CVaR 95% (monthly)",
                 "d_sortino": "Sortino", "d_calmar": "Calmar"}


def geo_excess(a: float, b: float) -> float:
    """(1 + a) / (1 + b) - 1: the geometric excess of return a over b."""
    return float((1 + a) / (1 + b) - 1) if a == a and b == b and b > -1 else np.nan


def _np_stats(r: np.ndarray, rf: np.ndarray) -> Dict[str, float]:
    """The window statistics of `qbt.metrics`, same formulas, on arrays
    (population moments, geometric annualisation, base point included in
    the drawdown path). Tested against the pandas versions."""
    n = len(r)
    g = float(np.prod(1 + r))
    ann = g ** (PPY / n) - 1 if g > 0 else np.nan
    rfa = float(np.prod(1 + rf)) ** (PPY / n) - 1
    sd = float(r.std())
    eq = np.r_[1.0, np.cumprod(1 + r)]
    dd = eq / np.maximum.accumulate(eq) - 1
    mdd = float(dd.min())
    down = float(np.sqrt((np.minimum(r - rf, 0) ** 2).mean()))
    v = float(np.quantile(r, 0.05))
    tail = r[r <= v]
    return {"Volatility": sd * np.sqrt(PPY), "CAGR": ann,
            "Sharpe": (ann - rfa) / (sd * np.sqrt(PPY)) if sd > M.RISKLESS_SD else np.nan,
            "Sortino": (ann - rfa) / (down * np.sqrt(PPY)) if down > M.RISKLESS_SD else np.nan,
            "Max drawdown": mdd, "Calmar": (ann - rfa) / abs(mdd) if mdd < 0 else np.nan,
            "Ulcer index": float(np.sqrt(((dd * 100) ** 2).mean())),
            # a positive loss, as in qbt.metrics.summary
            "CVaR 95% (monthly)": -(float(tail.mean()) if len(tail) else v)}


def _np_coef(y: np.ndarray, X: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 24:
        return np.nan
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return float(beta[-1])


def rolling_share(key: Optional[str], better: str, center: float, cand: pd.Series,
                  base: pd.Series, new: pd.Series, ctx: Context, window: int = 36,
                  step: int = 1) -> Dict[str, object]:
    """Share of rolling windows (step one month) in which the metric is on
    the good side of its centre."""
    if not key or len(base) < window:
        return {"metric": key, "windows": 0, "share": np.nan, "series": pd.Series(dtype=float)}
    idx = base.index
    c, b, n = (x.reindex(idx).to_numpy(float) for x in (cand, base, new))
    rf = ctx.rf.reindex(idx).fillna(0.0).to_numpy(float)
    lab = None
    if key.startswith(("d_regime_", "cand_regime_")):
        reg = key.split("_regime_", 1)[1]
        if reg not in ctx.regimes:
            return {"metric": key, "windows": 0, "share": np.nan, "series": pd.Series(dtype=float)}
        lab = (ctx.regimes[reg].reindex(idx) == True).to_numpy(bool)  # noqa: E712
    X = None
    if key == "d_megacap_beta" and ctx.mkt_cad is not None and ctx.spread is not None:
        X = np.column_stack([ctx.mkt_cad.reindex(idx).to_numpy(float),
                             ctx.spread.reindex(idx).to_numpy(float)])
    vals = {}
    for end in range(window, len(idx) + 1, step):
        sl = slice(end - window, end)
        cw, bw, nw = c[sl], b[sl], n[sl]
        if key == "d_cagr":
            v = geo_excess(_np_stats(nw, rf[sl])["CAGR"], _np_stats(bw, rf[sl])["CAGR"])
        elif key in _WINDOW_STATS:
            name = _WINDOW_STATS[key]
            v = _np_stats(nw, rf[sl])[name] - _np_stats(bw, rf[sl])[name]
        elif key == "corr_to_portfolio":
            v = float(np.corrcoef(cw, bw)[0, 1])
        elif key == "d_tail_return":
            t = bw <= np.quantile(bw, 0.10)
            v = float((nw[t] - bw[t]).mean())
        elif key == "d_megacap_beta":
            v = _np_coef(nw, X[sl]) - _np_coef(bw, X[sl]) if X is not None else np.nan
        elif lab is not None:
            m = lab[sl]
            v = (float((nw[m] - bw[m]).mean()) if key.startswith("d_") else float(cw[m].mean()))                 if m.sum() >= 2 else np.nan
        else:
            v = np.nan
        vals[idx[end - 1]] = v
    s = pd.Series(vals, dtype=float).dropna()
    if s.empty:
        return {"metric": key, "windows": 0, "share": np.nan, "series": s}
    d = 1.0 if better == "higher" else -1.0
    good = (d * (s - center) > 0)
    return {"metric": key, "windows": int(len(s)), "share": float(good.mean()), "series": s}
