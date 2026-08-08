"""Strategy foundation.

Single, deliberately narrow contract:

    generate(prices: DataFrame, params: dict) -> DataFrame of target weights

- `prices`: adjusted prices, one column per instrument.
- Output: same index and columns, target weights desired *at the close of
  each day*. A row can sum to less than 1 (the rest sits in cash) but never
  more than max_leverage.
- No access to the future: any value at t depends only on t and before. The
  execution lag is applied by the engine, not here.

Each strategy is a pure function registered in REGISTRY along with its
parameter descriptions, which lets the interface build its controls
automatically.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

REGISTRY: Dict[str, "Strategy"] = {}


@dataclass
class Param:
    key: str
    label: str
    kind: str                      # int | float | choice | bool | series
    default: Any
    min: float = 0
    max: float = 100
    step: float = 1
    choices: List[Any] = field(default_factory=list)
    help: str = ""


@dataclass
class Strategy:
    key: str
    label: str
    description: str
    params: List[Param]
    fn: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]

    def defaults(self) -> Dict[str, Any]:
        return {p.key: p.default for p in self.params}

    @property
    def needs_exog(self) -> bool:
        """A strategy that declares a third argument consumes exogenous
        series. Others keep the two-argument signature."""
        return len(inspect.signature(self.fn).parameters) >= 3

    def generate(self, prices: pd.DataFrame,
                 params: Dict[str, Any] | None = None,
                 exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        p = self.defaults()
        p.update(params or {})
        if self.needs_exog:
            e = exog if exog is not None else pd.DataFrame(index=prices.index)
            w = self.fn(prices, p, e.reindex(prices.index))
        else:
            w = self.fn(prices, p)
        return sanitize_weights(w, prices)


def register(key: str, label: str, description: str, params: List[Param]):
    def deco(fn):
        REGISTRY[key] = Strategy(key, label, description, params, fn)
        return fn
    return deco


def sanitize_weights(w: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Safety net: alignment, NaN, outliers."""
    w = w.reindex(index=prices.index, columns=prices.columns)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # No position in an asset with no price that day
    return w.where(prices.notna(), 0.0)


# ----------------------------------------------------------------------
# Shared indicators
# ----------------------------------------------------------------------
def sma(px: pd.DataFrame, n: int) -> pd.DataFrame:
    return px.rolling(n, min_periods=n).mean()


def ema(px: pd.DataFrame, n: int) -> pd.DataFrame:
    return px.ewm(span=n, adjust=False, min_periods=n).mean()


def total_return(px: pd.DataFrame, n: int) -> pd.DataFrame:
    return px / px.shift(n) - 1.0


def realized_vol(px: pd.DataFrame, n: int, ppy: int = 252) -> pd.DataFrame:
    return px.pct_change().rolling(n, min_periods=max(5, n // 2)).std() * np.sqrt(ppy)


def downside_vol(px: pd.DataFrame, n: int, ppy: int = 252,
                 mar_pa: float = 0.0) -> pd.DataFrame:
    """Downside deviation: root mean square of returns below the minimum
    acceptable return (0 by default).

    Deliberately NOT the standard deviation of the subset of negative days.
    Taking the std of `r.where(r < 0)` drops every positive day as NaN, which
    leaves only ~half the window populated; a `min_periods` sized for the full
    window then rejects most rows, and the measure silently disappears exactly
    for the calmest assets, which have the fewest negative days. Squaring the
    clipped series keeps every observation and matches the textbook
    definition used by the Sortino ratio.
    """
    d = (px.pct_change() - mar_pa / ppy).clip(upper=0.0)
    msd = (d ** 2).rolling(n, min_periods=max(5, n // 2)).mean()
    return np.sqrt(msd) * np.sqrt(ppy)


def rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def efficiency_ratio(px: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Kaufman: |net displacement| / sum of absolute displacements. 0 to 1."""
    direction = (px - px.shift(n)).abs()
    volatility = px.diff().abs().rolling(n, min_periods=n).sum()
    return (direction / volatility.replace(0, np.nan)).clip(0, 1)


def zscore(df: pd.DataFrame, n: int) -> pd.DataFrame:
    mu = df.rolling(n, min_periods=n).mean()
    sd = df.rolling(n, min_periods=n).std(ddof=1)
    return (df - mu) / sd.replace(0, np.nan)


# ----------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------
def size_equal(mask: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    n = mask.sum(axis=1).replace(0, np.nan)
    return mask.div(n, axis=0).fillna(0.0) * gross


def size_inverse_vol(mask: pd.DataFrame, px: pd.DataFrame, n: int = 60,
                     gross: float = 1.0, downside: bool = False,
                     slots=None) -> pd.DataFrame:
    """Weights inversely proportional to volatility, among the selected names.

    `slots` controls what happens to unallocated capital. Left at None, the
    held names are renormalized to `gross`, so the portfolio is always fully
    invested. Given a number of slots (or a per-day Series), the weights sum
    to (names held / slots) x gross instead, and the balance stays in cash.
    That second form is what preserves the defensive behaviour of a strategy
    whose filter rejects most of the universe: without it, rejecting five of
    six assets would simply concentrate the whole portfolio into the sixth.

    A name whose volatility cannot be measured yet is dropped from the
    weighting rather than silently inheriting the others' capital.
    """
    v = downside_vol(px, n) if downside else realized_vol(px, n)
    usable = mask.astype(bool) & v.notna() & (v > 0)
    inv = (1.0 / v.where(usable)).where(usable)
    tot = inv.sum(axis=1).replace(0, np.nan)
    w = inv.div(tot, axis=0).fillna(0.0) * gross

    if slots is not None:
        held = usable.sum(axis=1)
        denom = (slots if isinstance(slots, pd.Series)
                 else pd.Series(float(slots), index=w.index))
        scale = (held / denom.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        w = w.mul(scale, axis=0)
    return w


def apply_vol_target(w: pd.DataFrame, px: pd.DataFrame, target: float,
                     lookback: int = 60, cap: float = 1.0) -> pd.DataFrame:
    """Scales the portfolio to target an ex-ante volatility."""
    r = px.pct_change()
    port_r = (w.shift(1).fillna(0) * r).sum(axis=1)
    rv = port_r.rolling(lookback, min_periods=max(10, lookback // 3)).std() * np.sqrt(252)
    scale = (target / rv.replace(0, np.nan)).clip(upper=cap).fillna(0.0)
    return w.mul(scale, axis=0)
