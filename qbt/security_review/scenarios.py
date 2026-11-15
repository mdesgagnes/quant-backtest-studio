"""Historical stress windows and monthly regimes, both driven by
`config/scenarios.yaml`.

Regime rules (one boolean per month; True = the named regime):

* `yield_change`         month-end yield minus the yield `lookback_months` earlier > 0
* `cpi_yoy`              CPI year-on-year above `threshold_yoy`
* `below_moving_average` price below its `ma_months` average, read at the
                         close `lag_months` earlier
* `above_expanding_median` monthly average above the median of all earlier
                         months (from `min_history_months` on; earlier = no label)
* `both_negative`        two given return series both below 0 in the month

A month with no input data has no label (NaN), so it counts in neither state.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .data import month_end_prices
from .config import StressWindow
from .portfolio import buy_and_hold_path


def _me(s: pd.Series) -> pd.Series:
    """Month-end index for any dated series (last value of each month)."""
    s = s.dropna()
    m = s.groupby(s.index.to_period("M")).last()
    m.index = m.index.to_timestamp("M")
    return m


def label_yield_change(yld: pd.Series, lookback: int = 1) -> pd.Series:
    m = _me(yld)
    d = m - m.shift(lookback)
    return (d > 0).where(d.notna())


def label_cpi_yoy(cpi: pd.Series, threshold: float) -> pd.Series:
    m = _me(cpi)
    yoy = m / m.shift(12) - 1
    return (yoy > threshold).where(yoy.notna())


def label_below_ma(px: pd.Series, ma: int = 10, lag: int = 1) -> pd.Series:
    m = _me(px)
    avg = m.rolling(ma).mean()
    below = (m < avg).where(avg.notna())
    return below.shift(lag)


def label_above_expanding_median(level: pd.Series, min_hist: int = 60) -> pd.Series:
    s = level.dropna()
    m = s.groupby(s.index.to_period("M")).mean()
    m.index = m.index.to_timestamp("M")
    med = m.expanding(min_periods=min_hist).median().shift(1)
    return (m > med).where(med.notna())


def label_both_negative(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([a, b], axis=1).dropna()
    return (df.iloc[:, 0] < 0) & (df.iloc[:, 1] < 0)


def build_regimes(rules: Dict[str, Dict], inputs: Dict[str, Optional[pd.Series]]
                  ) -> Dict[str, pd.Series]:
    """Regime labels from the YAML rules. `inputs` holds the series each
    rule needs, by its config key (and `equity`/`bonds` for both_negative).
    A rule whose input is missing is skipped."""
    out = {}
    for key, r in rules.items():
        rule = r.get("rule")
        s = inputs.get(key)
        if rule == "yield_change" and s is not None:
            out[key] = label_yield_change(s, int(r.get("lookback_months", 1)))
        elif rule == "cpi_yoy" and s is not None:
            out[key] = label_cpi_yoy(s, float(r.get("threshold_yoy", 0.03)))
        elif rule == "below_moving_average" and s is not None:
            out[key] = label_below_ma(s, int(r.get("ma_months", 10)), int(r.get("lag_months", 1)))
        elif rule == "above_expanding_median" and s is not None:
            out[key] = label_above_expanding_median(s, int(r.get("min_history_months", 60)))
        elif rule == "both_negative" and inputs.get("equity") is not None \
                and inputs.get("bonds") is not None:
            out[key] = label_both_negative(inputs["equity"], inputs["bonds"])
    return out


def regime_table(cand: pd.Series, base: pd.Series, new: pd.Series, weight: float,
                 regimes: Dict[str, pd.Series], rules: Dict[str, Dict]) -> pd.DataFrame:
    """For each regime and its opposite: months, the candidate's average
    monthly return and annualised volatility, its contribution to the new
    portfolio (weight x average return) and the change in the portfolio's
    average monthly return."""
    rows = []
    idx = cand.dropna().index.intersection(base.dropna().index).intersection(new.dropna().index)
    for key, lab in regimes.items():
        lab = lab.reindex(idx)
        for state, name in ((True, rules.get(key, {}).get("label", key)),
                            (False, rules.get(key, {}).get("opposite", "not " + key))):
            mask = (lab == state) & lab.notna()
            n = int(mask.sum())
            c, b, nw = cand[idx][mask], base[idx][mask], new[idx][mask]
            rows.append({
                "Regime": name, "Key": key if state else f"not_{key}", "Months": n,
                "Candidate avg monthly return": float(c.mean()) if n else np.nan,
                "Candidate volatility (ann.)": float(c.std(ddof=0) * np.sqrt(12)) if n > 2 else np.nan,
                "Contribution to portfolio (monthly)": float(weight * c.mean()) if n else np.nan,
                "Portfolio avg monthly return, without": float(b.mean()) if n else np.nan,
                "Portfolio avg monthly return, with": float(nw.mean()) if n else np.nan,
                "Change in portfolio avg monthly return": float((nw - b).mean()) if n else np.nan,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Stress windows (daily)
# ----------------------------------------------------------------------
def _window(px: pd.Series, w: StressWindow) -> Optional[pd.Series]:
    """Prices from the start close to the end close, or None if the series
    does not cover the whole window."""
    px = px.dropna()
    s, e = pd.Timestamp(w.start), pd.Timestamp(w.end)
    if px.empty or px.index[0] > s or px.index[-1] < e:
        return None
    before = px[px.index <= s]
    inside = px[(px.index > s) & (px.index <= e)]
    if before.empty or inside.empty:
        return None
    return pd.concat([before.iloc[[-1]], inside])


def _mdd(path: pd.Series) -> float:
    return float((path / path.cummax() - 1).min())


def _window_monthly(px: pd.Series, w: StressWindow) -> Optional[pd.Series]:
    """Month-end values from the last month end on or before the start to
    the first month end on or after the end (used when a series has no daily
    prices). None if the series does not cover both."""
    px = px.dropna()
    if px.empty:
        return None
    me = px.groupby(px.index.to_period("M")).last()
    me.index = me.index.to_timestamp("M")
    s, e = pd.Timestamp(w.start), pd.Timestamp(w.end)
    first = me[me.index <= s]
    last = me[me.index >= e]
    if first.empty or last.empty:
        return None
    return me.loc[first.index[-1]:last.index[0]]


def stress_table(cand: str, daily: Dict[str, pd.Series], base_w: Dict[str, float],
                 new_w: Dict[str, float], windows: List[StressWindow],
                 monthly_only: Iterable[str] = (), min_monthly_days: int = 45) -> pd.DataFrame:
    """One row per window: candidate return and max drawdown, portfolio
    with and without, and the candidate's daily correlation with the
    portfolio. Windows the candidate (or a holding) does not cover are
    'no data'."""
    rows = []
    hold = sorted(set(base_w) | set(new_w) - {cand})
    for w in windows:
        row = {"Window": w.name, "Key": w.key, "Start": w.start, "End": w.end}
        monthly = bool(set(monthly_only) & set(hold + [cand]))
        row["Basis"] = "month-end" if monthly else "daily"
        if monthly and (pd.Timestamp(w.end) - pd.Timestamp(w.start)).days < min_monthly_days:
            row["Status"] = "no data (monthly series; window shorter than two months)"
            rows.append(row)
            continue
        cut_fn = _window_monthly if monthly else _window
        cut = {t: cut_fn(daily[t], w) if t in daily else None for t in hold + [cand]}
        if any(cut[t] is None for t in hold + [cand]):
            missing = [t for t in hold + [cand] if cut[t] is None]
            row.update({"Status": "no data (" + ", ".join(missing) + ")"})
            rows.append(row)
            continue
        cal = pd.concat({t: cut[t] for t in hold + [cand]}, axis=1).sort_index().ffill()
        cal = cal.dropna()
        pb = buy_and_hold_path(cal, base_w)
        pn = buy_and_hold_path(cal, new_w)
        cp = cal[cand] / cal[cand].iloc[0]
        rc, rb = cp.pct_change().dropna(), pb.pct_change().dropna()
        row.update({
            "Status": "ok", "Sessions": int(len(cal) - 1) if not monthly else np.nan,
            "Candidate return": float(cp.iloc[-1] - 1),
            "Candidate max drawdown": _mdd(cp),
            "Portfolio return, without": float(pb.iloc[-1] - 1),
            "Portfolio return, with": float(pn.iloc[-1] - 1),
            "Change in portfolio return": float(pn.iloc[-1] - pb.iloc[-1]),
            "Portfolio max drawdown, without": _mdd(pb),
            "Portfolio max drawdown, with": _mdd(pn),
            "Candidate correlation to portfolio (daily)": float(rc.corr(rb)) if len(rc) > 3 else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)
