"""Trading-cost inputs estimated from market data.

Market impact follows the square-root law, the best-established empirical
regularity in the price-impact literature (Almgren et al. 2005; Toth et al.
2011; Bouchaud et al. 2018): the cost of an order is proportional to the
instrument's daily volatility times the square root of the order's share of
daily volume,

    impact = Y * sigma_daily * sqrt(Q / ADV),

with Y of order one across markets and periods. Scaling by volatility is
what makes the same order cost more in a turbulent week than a calm one.
The formula itself lives in the engine, next to the other cost terms; the
volatility it needs is estimated here, from prices known before the trade.

The spread stays a flat rate, deliberately. Estimating it from daily OHLC
prices (Roll 1984; Corwin & Schultz 2012; Abdi & Ranaldo 2017; and the
most efficient of them, EDGE, Ardia, Guidotti & Kroencke, JFE 2024) was
tested on Yahoo Finance data for the ETFs this app is used with and
rejected: it put SPY's spread near 27 bps against a real quoted spread
near 0.2 bps, and could not identify a spread at all for liquid Canadian
ETFs. For instruments this liquid the daily-price footprint of the spread
is buried in noise; a flat half-spread with per-instrument multipliers,
set from quoted spreads, is more accurate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_volatility(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Trailing standard deviation of daily log returns, lagged one session
    so a trade is never priced on the volatility of its own day."""
    r = np.log(close.where(close > 0)).diff()
    return r.rolling(window, min_periods=max(5, window // 4)).std().shift(1)
