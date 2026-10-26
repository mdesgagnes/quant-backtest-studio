"""Factor-zoo strategy: hold the names a chosen library factor scores highest.

Bridges `qbt.lab.factors` into the ordinary strategy registry, so any factor
that survives a screen in the Research workspace can be run, swept, stress
tested and exported by every existing tool in the Backtest workspace without
a line of new code. The factor sees prices up to and including each day; the
engine applies its usual execution lag.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .base import Param, register
from ..lab import factors as F


@register(
    key="zoo_factor",
    label="Factor Zoo (single factor, top N)",
    description="Holds the N instruments with the highest score on one factor "
                "from the research library (momentum, reversal, volatility, "
                "trend, beta, seasonality...). Factors whose literature sign is "
                "negative are flipped, so a higher score is always better. "
                "Screen factors in the Research workspace first: choosing one "
                "here after looking at many is a trial the validation suite "
                "will count.",
    params=[
        Param("factor", "Factor", "choice", "mom_12_1", choices=list(F.FACTORS)),
        Param("top_n", "Number of Positions", "int", 3, 1, 30, 1),
        Param("weighting", "Weighting", "choice", "equal", choices=["equal", "inverse_vol"]),
    ],
)
def _zoo_factor(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    scores = F.compute(str(p["factor"]), px)
    return F.factor_to_weights(scores, int(p["top_n"]), 1.0, None, str(p["weighting"]), px)
