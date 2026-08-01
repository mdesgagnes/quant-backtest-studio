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


def downside_vol(px: pd.DataFrame, n: int, ppy: int = 252) -> pd.DataFrame:
    r = px.pct_change()
    neg = r.where(r < 0)
    return neg.rolling(n, min_periods=max(5, n // 2)).std() * np.sqrt(ppy)


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
                     gross: float = 1.0, downside: bool = False) -> pd.DataFrame:
    v = downside_vol(px, n) if downside else realized_vol(px, n)
    inv = (1.0 / v.replace(0, np.nan)).where(mask.astype(bool))
    tot = inv.sum(axis=1).replace(0, np.nan)
    return inv.div(tot, axis=0).fillna(0.0) * gross


def apply_vol_target(w: pd.DataFrame, px: pd.DataFrame, target: float,
                     lookback: int = 60, cap: float = 1.0) -> pd.DataFrame:
    """Scales the portfolio to target an ex-ante volatility."""
    r = px.pct_change()
    port_r = (w.shift(1).fillna(0) * r).sum(axis=1)
    rv = port_r.rolling(lookback, min_periods=max(10, lookback // 3)).std() * np.sqrt(252)
    scale = (target / rv.replace(0, np.nan)).clip(upper=cap).fillna(0.0)
    return w.mul(scale, axis=0)
