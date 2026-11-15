"""Standalone, within-sleeve and factor statistics, monthly.

Every figure reuses the app's existing CIPM-convention functions
(`qbt.metrics`, `qbt.cipm`): population moments, geometric annualisation.
The helpers here only add what did not exist yet (time underwater, worst
12 months, hit rate in down months, relative drawdown, OLS with t-stats).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import cipm as C
from .. import metrics as M

PPY = 12


# ----------------------------------------------------------------------
# Small building blocks
# ----------------------------------------------------------------------
def underwater(r: pd.Series) -> Dict[str, float]:
    """Longest run of months below the previous peak, the share of months
    spent below it, and the months from the max drawdown's trough to
    recovery (NaN if not recovered)."""
    r = r.dropna()
    if r.empty:
        return {"max_underwater": np.nan, "underwater_share": np.nan, "recovery_months": np.nan}
    eq = (1 + r).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
    under = eq.to_numpy() < peak - 1e-12
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    dd = eq.to_numpy() / peak - 1
    t = int(np.argmin(dd))
    rec = np.nan
    if dd[t] < 0:
        after = np.where(~under[t:])[0]
        rec = float(after[0]) if len(after) else np.nan
    return {"max_underwater": float(longest), "underwater_share": float(under.mean()),
            "recovery_months": rec}


def worst_12m(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 12:
        return np.nan
    return float(((1 + r).rolling(12).apply(np.prod, raw=True) - 1).min())


def ols(y: pd.Series, X: pd.DataFrame) -> Dict[str, object]:
    """OLS with an intercept: coefficients, t-stats (classical) and R2."""
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    n, k = len(df), X.shape[1] + 1
    if n <= k + 2:
        return {"n": n, "coef": pd.Series(dtype=float), "t": pd.Series(dtype=float), "r2": np.nan}
    Y = df["_y"].to_numpy()
    A = np.column_stack([np.ones(n), df[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    s2 = float(resid @ resid) / (n - k)
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    names = ["alpha"] + list(X.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = beta / se
    tss = float(((Y - Y.mean()) ** 2).sum())
    return {"n": n, "coef": pd.Series(beta, index=names),
            "t": pd.Series(tstat, index=names),
            "r2": 1 - float(resid @ resid) / tss if tss > 0 else np.nan}


# ----------------------------------------------------------------------
# Standalone profile
# ----------------------------------------------------------------------
def standalone(r: pd.Series, rf: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if len(r) < 2:
        return {}
    eq = M.to_equity(r)
    rfa = M.rf_annual(r.index, rf.reindex(r.index).fillna(0.0), PPY)
    uw = underwater(r)
    return {
        "Months": len(r),
        "Start": r.index[0].strftime("%Y-%m"),
        "End": r.index[-1].strftime("%Y-%m"),
        "CAGR": M.ann_return(r, PPY),
        "Volatility": M.annual_vol(r, PPY),
        "Sharpe": M.sharpe(r, rf.reindex(r.index).fillna(0.0), PPY),
        "Sortino": M.sortino(r, rf.reindex(r.index).fillna(0.0), PPY),
        "Max drawdown": M.max_drawdown(eq),
        "Calmar": M.calmar(eq, PPY, rf_ann=0.0),
        "Ulcer index": M.ulcer_index(eq),
        "Longest underwater (months)": uw["max_underwater"],
        "Recovery from max drawdown (months)": uw["recovery_months"],
        "Skew": M.pop_skew(r),
        "Excess kurtosis": M.pop_kurt(r),
        "CVaR 95% (monthly)": M.var_cvar(r, 0.05)["cvar"],
        "Worst month": float(r.min()),
        "Worst 12 months": worst_12m(r),
        "Risk-free (annual)": rfa,
    }


# ----------------------------------------------------------------------
# Within-sleeve, against the sleeve benchmark
# ----------------------------------------------------------------------
def relative(r: pd.Series, b: pd.Series, rf: pd.Series) -> Dict[str, float]:
    df = pd.concat([r, b], axis=1).dropna()
    if len(df) < 12:
        return {}
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    rfx = rf.reindex(df.index).fillna(0.0)
    ba = M.beta_alpha(x, y, rfx, PPY)
    cap = C.capture_ratios(x, y, PPY)
    bb = C.bull_bear_betas(x, y, rfx, PPY)
    ax, ay = M.ann_return(x, PPY), M.ann_return(y, PPY)
    down = y < 0
    rel = (1 + x).cumprod() / (1 + y).cumprod()
    rel_dd = float((rel / np.maximum.accumulate(np.r_[1.0, rel.to_numpy()])[1:] - 1).min())
    return {
        "Active return": (1 + ax) / (1 + ay) - 1 if ax == ax and ay == ay else np.nan,
        "Tracking error": ba["tracking_error"],
        "Information ratio": ba["information_ratio"],
        "Beta": ba["beta"],
        "Correlation": ba["corr"],
        "Upside capture": cap["up_capture"],
        "Downside capture": cap["down_capture"],
        "Upside beta": bb["bull_beta"],
        "Downside beta": bb["bear_beta"],
        "Hit rate in benchmark down months": float((x[down] > y[down]).mean()) if down.any() else np.nan,
        "Relative drawdown": rel_dd,
    }


def excess_correlation(rets: pd.DataFrame, bench: Dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation of candidates' returns in excess of their own sleeve benchmark."""
    ex = {}
    for c in rets.columns:
        b = bench.get(c)
        if b is not None:
            ex[c] = (rets[c] - b.reindex(rets.index))
    return pd.DataFrame(ex).corr() if ex else pd.DataFrame()


def equivalents(rets: pd.DataFrame, bench: Dict[str, pd.Series],
                max_te: float = 0.01, min_ex_corr: float = 0.95,
                min_raw_corr: float = 0.95) -> List[Dict[str, object]]:
    """Pairs that are functionally equivalent.

    Rule: tracking error between the two < `max_te` (1% a year) AND either
    their excess returns over the sleeve benchmark correlate above
    `min_ex_corr`, or one of them hardly departs from the benchmark (its own
    tracking error below `max_te`), in which case excess correlation is
    meaningless and the raw correlation must exceed `min_raw_corr`."""
    out = []
    cols = list(rets.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            df = rets[[a, b]].dropna()
            if len(df) < 24:
                continue
            te = M.pop_std((1 + df[a]) / (1 + df[b]) - 1) * np.sqrt(PPY)
            if not te < max_te:
                continue
            ba, bb = bench.get(a), bench.get(b)
            ok, why = False, ""
            if ba is not None and bb is not None:
                xa = (df[a] - ba.reindex(df.index)).dropna()
                xb = (df[b] - bb.reindex(df.index)).dropna()
                tea = M.pop_std(xa) * np.sqrt(PPY)
                teb = M.pop_std(xb) * np.sqrt(PPY)
                if min(tea, teb) < max_te:
                    rc = float(df[a].corr(df[b]))
                    ok, why = rc > min_raw_corr, f"raw correlation {rc:.3f}"
                else:
                    ec = float(xa.corr(xb))
                    ok, why = ec > min_ex_corr, f"excess-return correlation {ec:.3f}"
            else:
                rc = float(df[a].corr(df[b]))
                ok, why = rc > min_raw_corr, f"raw correlation {rc:.3f}"
            if ok:
                out.append({"A": a, "B": b, "Tracking error between them": te,
                            "Evidence": why, "Months": len(df)})
    return out


# ----------------------------------------------------------------------
# Factor / style exposures
# ----------------------------------------------------------------------
def factor_exposure(r: pd.Series, market: pd.Series, factors: pd.DataFrame,
                    rf: pd.Series, min_months: int = 36) -> Dict[str, object]:
    """Excess return on market excess return and style spreads."""
    rfx = rf.reindex(r.index).fillna(0.0)
    y = (r - rfx).rename("y")
    X = pd.concat([(market - rf.reindex(market.index).fillna(0.0)).rename("market"),
                   factors], axis=1)
    res = ols(y, X)
    if res["n"] < min_months:
        return {"n": res["n"], "ok": False, "coef": pd.Series(dtype=float),
                "t": pd.Series(dtype=float), "r2": np.nan}
    res["ok"] = True
    res["coef"] = res["coef"].copy()
    res["coef"]["alpha"] = (1 + res["coef"]["alpha"]) ** PPY - 1   # annualised
    return res
