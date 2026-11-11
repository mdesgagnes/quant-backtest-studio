"""Instrument groups: the numbers behind the Markets "Groups" tab.

After the Groups page on Sparkline Analytics, which lays out a fund taxonomy
with one row per peer group (how many funds, the median return over each
horizon, a trend line, the largest members). Here a group is any named list
of instruments: a built-in preset, an asset class pulled out of the presets,
or one the user types in. Every group is summarised the same way, so groups
can be compared with each other and ranked.

Price-only and pure pandas (no Streamlit), like qbt/snapshot.py.

Two choices worth knowing about:

- The group's trend line is an equal-weight, daily-rebalanced index of the
  members that have a price that day. It describes the group, not a
  portfolio anyone could hold; it is there so groups can be drawn against
  each other.
- A statistic is computed only over members that have history for that
  horizon, and the member count shown is the number with any data. A group
  with fewer than SMALL_SAMPLE members is flagged, the way Sparkline daggers
  small groups, because a median of three is not a robust figure.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from . import monitor as MON
from . import presets as PRESETS
from . import snapshot as S

HORIZON_COLS = ["1M", "3M", "YTD", "1Y", "3Y", "5Y"]   # 3Y and 5Y annualized
SMALL_SAMPLE = 5

STATS: Dict[str, Callable[[pd.Series], float]] = {
    "Median": lambda v: float(v.median()),
    "Mean": lambda v: float(v.mean()),
    "25th percentile": lambda v: float(v.quantile(0.25)),
    "75th percentile": lambda v: float(v.quantile(0.75)),
    "Min": lambda v: float(v.min()),
    "Max": lambda v: float(v.max()),
}


# ----------------------------------------------------------------------
# Defining groups
# ----------------------------------------------------------------------
def _clean(tickers) -> List[str]:
    seen, out = set(), []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def parse_groups(text: str) -> Dict[str, List[str]]:
    """'Name: A, B, C' per line -> {name: [tickers]}. Lines starting with #
    and blank lines are ignored; a line with no members is dropped; a
    repeated name is merged into the first."""
    out: Dict[str, List[str]] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, _, rest = line.partition(":")
        name = name.strip()
        members = _clean(rest.replace(";", ",").replace("\t", ",").replace(" ", ",").split(","))
        if name and members:
            out[name] = _clean(out.get(name, []) + members)
    return out


def groups_from_presets(names: List[str]) -> Dict[str, List[str]]:
    """Each chosen preset becomes one group."""
    out = {}
    for n in names:
        tk = _clean(PRESETS.get(n).get("tickers", []) or [])
        if tk:
            out[n] = tk
    return out


def groups_by_class(names: List[str]) -> Dict[str, List[str]]:
    """Regroup the chosen presets' instruments by their asset-class tag.

    The tag is whatever the preset says; an instrument two presets tag
    differently lands in the class of the first preset listed, so it is
    counted once. Untagged instruments are not guessed at and are left out.
    """
    placed: Dict[str, str] = {}
    for n in names:
        for t, cls in (PRESETS.get(n).get("classes", {}) or {}).items():
            placed.setdefault(str(t).upper(), str(cls))
    out: Dict[str, List[str]] = {}
    for t, cls in placed.items():
        out.setdefault(cls, []).append(t)
    return out


def union_tickers(groups: Dict[str, List[str]]) -> List[str]:
    return _clean(t for members in groups.values() for t in members)


# ----------------------------------------------------------------------
# Summarising them
# ----------------------------------------------------------------------
def group_index(prices: pd.DataFrame, members: List[str]) -> pd.Series:
    """Equal-weight, daily-rebalanced index (base 100) of the members that
    have a price that day. Empty when no member has two prices."""
    cols = [m for m in members if m in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    r = prices[cols].pct_change(fill_method=None)
    r = r.dropna(how="all")
    if r.empty:
        return pd.Series(dtype=float)
    eq = r.mean(axis=1, skipna=True)
    idx = (1.0 + eq).cumprod() * 100.0
    return idx


def group_table(prices: pd.DataFrame, groups: Dict[str, List[str]],
                stat: str = "Median", trend_window: str = "1Y",
                rf_pa: float = 0.0) -> pd.DataFrame:
    """One row per group. Returns are decimal; the app formats them.

    Columns: Members (with data), Listed (asked for), Small (fewer than
    SMALL_SAMPLE members), the chosen statistic of each horizon return,
    median volatility / Sharpe / max drawdown over the whole loaded period,
    the 1Y spread (75th minus 25th percentile), best and worst member over
    1Y, and a trend of the group's equal-weight index.
    """
    if stat not in STATS:
        raise ValueError(f"unknown statistic {stat!r}")
    agg = STATS[stat]
    avail = [c for c in prices.columns if prices[c].dropna().size >= 30]
    if not avail:
        return pd.DataFrame()
    grid = MON.performance_grid(prices[avail]).set_index("Instrument")
    risk = {}
    for c in avail:
        s = prices[c].dropna()
        r = s.pct_change().dropna()
        risk[c] = (M.annual_vol(r), M.sharpe(r, rf_pa), M.max_drawdown(s))

    rows = []
    for name, members in groups.items():
        have = [m for m in members if m in avail]
        if not have:
            continue
        row: Dict[str, object] = {"Group": name, "Members": len(have),
                                  "Listed": len(members),
                                  "Small": len(have) < SMALL_SAMPLE}
        for h in HORIZON_COLS:
            v = grid.loc[have, h].dropna() if h in grid.columns else pd.Series(dtype=float)
            row[h] = agg(v) if len(v) else np.nan
        vol = pd.Series([risk[m][0] for m in have]).dropna()
        shp = pd.Series([risk[m][1] for m in have]).dropna()
        mdd = pd.Series([risk[m][2] for m in have]).dropna()
        row["Volatility"] = float(vol.median()) if len(vol) else np.nan
        row["Sharpe"] = float(shp.median()) if len(shp) else np.nan
        row["Max DD"] = float(mdd.median()) if len(mdd) else np.nan
        y1 = grid.loc[have, "1Y"].dropna()
        row["1Y spread"] = (float(y1.quantile(0.75) - y1.quantile(0.25))
                            if len(y1) >= 2 else np.nan)
        row["Best (1Y)"] = f"{y1.idxmax()}" if len(y1) else ""
        row["Worst (1Y)"] = f"{y1.idxmin()}" if len(y1) else ""
        row["Largest"] = ", ".join(have[:4]) + (" ..." if len(have) > 4 else "")
        idx = group_index(prices, have)
        if len(idx) > 2:
            win = S.slice_window(idx.to_frame("g"), trend_window)
            row["Trend"] = S.trend_points(win["g"]) if "g" in win.columns else []
        else:
            row["Trend"] = []
        rows.append(row)
    return pd.DataFrame(rows)


def group_curves(prices: pd.DataFrame, groups: Dict[str, List[str]],
                 names: List[str], window: str = "1Y",
                 mode: str = "Cumulative") -> pd.DataFrame:
    """Chart series for several groups over a common window, in percent."""
    cols = {}
    for n in names:
        idx = group_index(prices, groups.get(n, []))
        if len(idx) > 2:
            cols[n] = idx
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    win = S.slice_window(df, window)
    shared = win.dropna()
    if len(shared) < 3:
        return pd.DataFrame()
    return S.rebased_curves(shared, mode)
