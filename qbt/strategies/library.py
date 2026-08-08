"""Built-in strategy library.

To add one: copy a block, change the key, decorate with @register. The
interface picks it up automatically at startup.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import (
    Param, register, sma, ema, total_return, realized_vol, rsi,
    efficiency_ratio, size_equal, size_inverse_vol, apply_vol_target,
)


# ----------------------------------------------------------------------
@register(
    key="buy_hold",
    label="Buy & Hold",
    description="Passive benchmark. Equal weights across the whole universe, "
                "rebalanced at the chosen frequency. Serves as a control for "
                "any other strategy.",
    params=[
        Param("gross", "Gross Exposure", "float", 1.0, 0.1, 1.0, 0.05,
              help="1.0 = fully invested."),
    ],
)
def _buy_hold(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    mask = px.notna().astype(float)
    return size_equal(mask, p["gross"])


# ----------------------------------------------------------------------
@register(
    key="sma_trend",
    label="Trend Filter (Moving Average)",
    description="Each asset is held as long as its price stays above its "
                "moving average. Otherwise that portion moves to cash.",
    params=[
        Param("window", "Moving Average Window", "int", 200, 20, 400, 5),
        Param("ma_type", "Moving Average Type", "choice", "SMA", choices=["SMA", "EMA"]),
        Param("buffer", "Buffer Zone (%)", "float", 0.0, 0.0, 10.0, 0.5,
              help="Margin above/below the average before switching. "
                   "Reduces the number of round-trips."),
        Param("sizing", "Sizing", "choice", "Equal Weight",
              choices=["Equal Weight", "Inverse Volatility"]),
        Param("vol_window", "Volatility Window", "int", 60, 20, 250, 5),
    ],
)
def _sma_trend(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    ma = ema(px, p["window"]) if p["ma_type"] == "EMA" else sma(px, p["window"])
    b = p["buffer"] / 100.0
    raw = px > ma * (1 + b)
    # Hysteresis: only exit once back below the lower bound
    exit_lvl = px < ma * (1 - b)
    state = raw.astype(float)
    if b > 0:
        state = state.where(raw | exit_lvl).ffill().fillna(0.0)
    mask = state.fillna(0.0)
    n_assets = px.notna().sum(axis=1).replace(0, np.nan)
    if p["sizing"] == "Inverse Volatility":
        # Same denominator as the equal-weight branch: an asset failing the
        # trend filter leaves its share in cash instead of being redistributed.
        return size_inverse_vol(mask, px, p["vol_window"], slots=n_assets)
    return mask.div(n_assets, axis=0).fillna(0.0)


# ----------------------------------------------------------------------
@register(
    key="xs_momentum",
    label="Cross-Sectional Momentum",
    description="Ranks the universe by past return and holds the top N. "
                "No absolute filter: always invested.",
    params=[
        Param("lookback", "Momentum Window (days)", "int", 126, 20, 504, 5),
        Param("skip", "Skip Days (reversal effect)", "int", 21, 0, 63, 1,
              help="Ignores the most recent month, standard practice in "
                   "equity momentum."),
        Param("top_n", "Number of Positions", "int", 3, 1, 20, 1),
        Param("sizing", "Sizing", "choice", "Equal Weight",
              choices=["Equal Weight", "Inverse Volatility", "Inverse Downside Volatility"]),
        Param("vol_window", "Volatility Window", "int", 60, 20, 250, 5),
    ],
)
def _xs_momentum(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    lb, sk = int(p["lookback"]), int(p["skip"])
    base = px.shift(sk)
    mom = base / base.shift(lb) - 1.0
    ranks = mom.rank(axis=1, ascending=False, na_option="keep", method="first")
    mask = (ranks <= p["top_n"]).astype(float).where(mom.notna(), 0.0)
    # Cross-sectional ranking always fills top_n slots, so renormalizing to
    # fully invested is the intended behaviour here.
    if p["sizing"] == "Inverse Volatility":
        return size_inverse_vol(mask, px, p["vol_window"], slots=p["top_n"])
    if p["sizing"] == "Inverse Downside Volatility":
        return size_inverse_vol(mask, px, p["vol_window"], downside=True,
                                slots=p["top_n"])
    return size_equal(mask)


# ----------------------------------------------------------------------
@register(
    key="dual_momentum",
    label="Dual Momentum (Relative + Absolute)",
    description="Cross-sectional momentum combined with an absolute filter: "
                "a position is only taken if its past return exceeds the "
                "threshold. The rest stays in cash.",
    params=[
        Param("lookback", "Momentum Window (days)", "int", 126, 20, 504, 5),
        Param("skip", "Skip Days", "int", 21, 0, 63, 1),
        Param("top_n", "Number of Positions", "int", 3, 1, 20, 1),
        Param("abs_threshold", "Absolute Threshold (%)", "float", 0.0, -20.0, 20.0, 0.5),
        Param("trend_window", "Trend Filter (0 = none)", "int", 200, 0, 400, 10),
        Param("sizing", "Sizing", "choice", "Equal Weight",
              choices=["Equal Weight", "Inverse Volatility", "Inverse Downside Volatility"]),
        Param("vol_window", "Volatility Window", "int", 60, 20, 250, 5),
    ],
)
def _dual_momentum(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    lb, sk = int(p["lookback"]), int(p["skip"])
    base = px.shift(sk)
    mom = base / base.shift(lb) - 1.0

    eligible = mom > p["abs_threshold"] / 100.0
    if int(p["trend_window"]) > 0:
        eligible &= px > sma(px, int(p["trend_window"]))

    ranked = mom.where(eligible)
    ranks = ranked.rank(axis=1, ascending=False, na_option="keep", method="first")
    mask = (ranks <= p["top_n"]).astype(float).where(ranked.notna(), 0.0)

    # Unallocated capital stays in cash: divide by top_n, not by the number of
    # names actually held. That is what makes the strategy defensive.
    if p["sizing"] == "Equal Weight":
        return mask / float(p["top_n"])
    return size_inverse_vol(mask, px, p["vol_window"],
                            downside=p["sizing"].startswith("Inverse Downside"),
                            slots=p["top_n"])


# ----------------------------------------------------------------------
@register(
    key="trend_quality",
    label="Trend Quality (Composite)",
    description="Composite score: momentum, Kaufman efficiency ratio, and "
                "distance from the moving average, each normalized then "
                "averaged. Favors clean trends over the fastest ones.",
    params=[
        Param("mom_window", "Momentum Window", "int", 126, 20, 504, 5),
        Param("er_window", "Efficiency Ratio Window", "int", 20, 5, 120, 1),
        Param("ma_window", "Reference Moving Average", "int", 100, 20, 300, 5),
        Param("top_n", "Number of Positions", "int", 3, 1, 20, 1),
        Param("min_score", "Minimum Score (0 to 1)", "float", 0.5, 0.0, 1.0, 0.05),
        Param("regime_window", "Regime Filter (0 = none)", "int", 200, 0, 400, 10,
              help="Applies a long-term trend filter to each asset."),
    ],
)
def _trend_quality(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    mom = total_return(px, int(p["mom_window"]))
    er = efficiency_ratio(px, int(p["er_window"]))
    ma = sma(px, int(p["ma_window"]))
    dist = (px / ma - 1.0)

    def _pct_rank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True, na_option="keep")

    score = (_pct_rank(mom) + _pct_rank(er) + _pct_rank(dist)) / 3.0
    eligible = score >= p["min_score"]
    if int(p["regime_window"]) > 0:
        eligible &= px > sma(px, int(p["regime_window"]))

    ranked = score.where(eligible)
    ranks = ranked.rank(axis=1, ascending=False, na_option="keep", method="first")
    mask = (ranks <= p["top_n"]).astype(float).where(ranked.notna(), 0.0)
    return mask / float(p["top_n"])


# ----------------------------------------------------------------------
@register(
    key="risk_parity",
    label="Risk Parity (Inverse Volatility)",
    description="Always invested, each asset weighted by the inverse of its "
                "realized volatility. Approximately equal risk contribution.",
    params=[
        Param("vol_window", "Volatility Window", "int", 60, 20, 250, 5),
        Param("downside", "Downside Volatility Only", "bool", False),
        Param("max_weight", "Max Weight per Asset", "float", 0.4, 0.05, 1.0, 0.05),
    ],
)
def _risk_parity(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    mask = px.notna().astype(float)
    w = size_inverse_vol(mask, px, int(p["vol_window"]), downside=bool(p["downside"]))
    w = w.clip(upper=p["max_weight"])
    tot = w.sum(axis=1).replace(0, np.nan)
    return w.div(tot, axis=0).fillna(0.0)


# ----------------------------------------------------------------------
@register(
    key="vol_target",
    label="Volatility Target",
    description="Equal-weight portfolio whose exposure is adjusted to target "
                "a constant annualized volatility. The balance goes to cash.",
    params=[
        Param("target_vol", "Target Volatility (%)", "float", 10.0, 2.0, 30.0, 0.5),
        Param("vol_window", "Estimation Window", "int", 60, 20, 250, 5),
        Param("max_exposure", "Max Exposure", "float", 1.0, 0.2, 2.0, 0.1),
    ],
)
def _vol_target(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    base = size_equal(px.notna().astype(float))
    return apply_vol_target(base, px, p["target_vol"] / 100.0,
                            int(p["vol_window"]), p["max_exposure"])


# ----------------------------------------------------------------------
@register(
    key="rsi_reversion",
    label="Mean Reversion (RSI)",
    description="Buys oversold assets according to RSI and holds until the "
                "zone is exited. Equal position among active names.",
    params=[
        Param("rsi_window", "RSI Window", "int", 14, 2, 50, 1),
        Param("entry", "Entry Threshold", "float", 30.0, 5.0, 50.0, 1.0),
        Param("exit", "Exit Threshold", "float", 55.0, 40.0, 90.0, 1.0),
        Param("trend_filter", "Trend Filter (0 = none)", "int", 200, 0, 400, 10,
              help="Only buys the dip if the asset stays above its long-term "
                   "average. Avoids buying into a bear market."),
    ],
)
def _rsi_reversion(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    r = rsi(px, int(p["rsi_window"]))
    enter = r < p["entry"]
    leave = r > p["exit"]
    state = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    state = state.mask(enter, 1.0).mask(leave, 0.0).ffill().fillna(0.0)
    if int(p["trend_filter"]) > 0:
        state = state.where(px > sma(px, int(p["trend_filter"])), 0.0)
    n = px.notna().sum(axis=1).replace(0, np.nan)
    return state.div(n, axis=0).fillna(0.0)


# ----------------------------------------------------------------------
@register(
    key="fixed_weights",
    label="Fixed Weights",
    description="Static allocation set manually, rebalanced at the chosen "
                "frequency. Useful for comparing against a target policy.",
    params=[
        Param("weights", "Weights (e.g. XIC.TO:0.6, ZAG.TO:0.4)", "choice", "",
              choices=[], help="Leave blank to equal-weight."),
    ],
)
def _fixed_weights(px: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    spec = str(p.get("weights") or "").strip()
    if not spec:
        return size_equal(px.notna().astype(float))
    target = {}
    for part in spec.replace(";", ",").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                target[k.strip().upper()] = float(v)
            except ValueError:
                continue
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for k, v in target.items():
        if k in w.columns:
            w[k] = v
    return w.where(px.notna(), 0.0)


# ======================================================================
# Strategies fed by exogenous series
# ----------------------------------------------------------------------
# These functions declare a third argument `ex`: the engine injects the
# imported series into it, already shifted by their publication lag and
# aligned to the price calendar.
# ======================================================================

def _pct_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, na_option="keep")


def _match_columns(ex: pd.DataFrame, cols) -> pd.DataFrame:
    """Keeps the exogenous columns that match an instrument."""
    up = {str(c).upper(): c for c in ex.columns}
    keep = {t: up[str(t).upper()] for t in cols if str(t).upper() in up}
    if not keep:
        return pd.DataFrame(index=ex.index, columns=list(cols), dtype=float)
    out = ex[list(keep.values())].copy()
    out.columns = list(keep.keys())
    return out.reindex(columns=list(cols))


@register(
    key="macro_gate",
    label="Macro Gate",
    description="An imported economic series drives exposure: fully invested "
                "in a favorable regime, reduced exposure or cash otherwise. "
                "Security selection stays mechanical.",
    params=[
        Param("series", "Reference Series", "series", "",
              help="Column from the exogenous data file (rate, ISM, credit "
                   "spread, surprise index...)."),
        Param("rule", "Regime Rule", "choice", "Above its moving average",
              choices=["Above its moving average", "Positive change",
                       "Z-score above threshold", "Z-score below threshold"]),
        Param("window", "Reference Window (sessions)", "int", 250, 20, 1000, 10),
        Param("threshold", "Threshold (z-score)", "float", 0.0, -3.0, 3.0, 0.25),
        Param("risk_off", "Exposure in unfavorable regime", "float", 0.0, 0.0, 1.0, 0.1,
              help="0 = fully in cash. 0.5 = half exposure."),
        Param("sizing", "Sizing", "choice", "Equal Weight",
              choices=["Equal Weight", "Inverse Volatility"]),
        Param("vol_window", "Volatility Window", "int", 60, 20, 250, 5),
    ],
)
def _macro_gate(px: pd.DataFrame, p: Dict[str, Any], ex: pd.DataFrame) -> pd.DataFrame:
    mask = px.notna().astype(float)
    base = (size_inverse_vol(mask, px, int(p["vol_window"]),
                             slots=mask.sum(axis=1))
            if p["sizing"] == "Inverse Volatility" else size_equal(mask))

    col = p.get("series")
    if not col or col not in ex.columns:
        return base                      # no valid series: stay invested

    s = pd.to_numeric(ex[col], errors="coerce")
    n = int(p["window"])
    rule = p["rule"]
    if rule == "Above its moving average":
        on = s > s.rolling(n, min_periods=max(5, n // 4)).mean()
    elif rule == "Positive change":
        on = s.diff(n) > 0
    else:
        mu = s.rolling(n, min_periods=max(5, n // 4)).mean()
        sd = s.rolling(n, min_periods=max(5, n // 4)).std(ddof=1)
        z = (s - mu) / sd.replace(0, np.nan)
        on = z > p["threshold"] if rule == "Z-score above threshold" else z < p["threshold"]

    expo = on.map({True: 1.0, False: float(p["risk_off"])}).astype(float)
    expo = expo.reindex(px.index).ffill().fillna(float(p["risk_off"]))
    return base.mul(expo, axis=0)


@register(
    key="factor_rank",
    label="Imported Factor Ranking",
    description="Ranks the universe by an imported fundamental or "
                "quantitative factor (one column per symbol) and holds the "
                "top N. Can be blended with price momentum.",
    params=[
        Param("direction", "Factor Direction", "choice", "High value = favorable",
              choices=["High value = favorable", "Low value = favorable"]),
        Param("top_n", "Number of Positions", "int", 3, 1, 20, 1),
        Param("blend_momentum", "Blend with Price Momentum", "float", 0.0, 0.0, 1.0, 0.1,
              help="0 = factor only. 1 = momentum only. Both rankings are "
                   "combined as percentile ranks."),
        Param("mom_window", "Momentum Window", "int", 126, 20, 504, 5),
        Param("trend_filter", "Trend Filter (0 = none)", "int", 0, 0, 400, 10),
        Param("sizing", "Sizing", "choice", "Equal Weight",
              choices=["Equal Weight", "Inverse Volatility"]),
        Param("vol_window", "Volatility Window", "int", 60, 20, 250, 5),
        Param("hold_cash", "Hold unfilled positions in cash", "bool", True,
              help="If the factor is missing for some names, the "
                   "corresponding capital stays in cash rather than being "
                   "redistributed."),
    ],
)
def _factor_rank(px: pd.DataFrame, p: Dict[str, Any], ex: pd.DataFrame) -> pd.DataFrame:
    f = _match_columns(ex, px.columns).astype(float)
    if f.notna().sum().sum() == 0:
        return pd.DataFrame(0.0, index=px.index, columns=px.columns)

    if p["direction"] == "Low value = favorable":
        f = -f
    score = _pct_rank(f)

    b = float(p["blend_momentum"])
    if b > 0:
        mom = _pct_rank(total_return(px, int(p["mom_window"])))
        score = (1 - b) * score.fillna(mom) + b * mom.fillna(score)

    if int(p["trend_filter"]) > 0:
        score = score.where(px > sma(px, int(p["trend_filter"])))

    ranks = score.rank(axis=1, ascending=False, na_option="keep", method="first")
    mask = (ranks <= p["top_n"]).astype(float).where(score.notna(), 0.0)

    if p["sizing"] == "Inverse Volatility":
        return size_inverse_vol(mask, px, int(p["vol_window"]),
                                slots=p["top_n"] if p["hold_cash"] else None)
    if p["hold_cash"]:
        return mask / float(p["top_n"])
    return size_equal(mask)


@register(
    key="exog_signal",
    label="Imported Signals",
    description="The exogenous file directly contains a signal per symbol "
                "(for example -1, 0, 1 or a continuous score). The "
                "application handles sizing, frictions, and statistics.",
    params=[
        Param("mode", "Signal Interpretation", "choice", "Binary (> threshold)",
              choices=["Binary (> threshold)", "Proportional to score",
                       "Cross-sectional rank"]),
        Param("threshold", "Activation Threshold", "float", 0.0, -5.0, 5.0, 0.1),
        Param("top_n", "Positions held (rank mode)", "int", 3, 1, 20, 1),
        Param("gross", "Max Gross Exposure", "float", 1.0, 0.1, 1.0, 0.05),
        Param("allow_short", "Allow negative signals", "bool", False),
    ],
)
def _exog_signal(px: pd.DataFrame, p: Dict[str, Any], ex: pd.DataFrame) -> pd.DataFrame:
    s = _match_columns(ex, px.columns).astype(float)
    if s.notna().sum().sum() == 0:
        return pd.DataFrame(0.0, index=px.index, columns=px.columns)

    mode = p["mode"]
    if mode == "Cross-sectional rank":
        r = s.rank(axis=1, ascending=False, na_option="keep", method="first")
        raw = (r <= p["top_n"]).astype(float).where(s.notna(), 0.0)
    elif mode == "Proportional to score":
        raw = s.where(s.notna(), 0.0)
        if not p["allow_short"]:
            raw = raw.clip(lower=0.0)
    else:
        raw = (s > p["threshold"]).astype(float).where(s.notna(), 0.0)
        if p["allow_short"]:
            raw = raw - (s < -abs(p["threshold"])).astype(float).where(s.notna(), 0.0)

    tot = raw.abs().sum(axis=1).replace(0, np.nan)
    return raw.div(tot, axis=0).fillna(0.0) * float(p["gross"])
