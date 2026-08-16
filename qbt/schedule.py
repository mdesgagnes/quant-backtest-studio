"""Rebalance scheduling.

The date a strategy trades on is a free parameter that usually goes
unexamined. Every backtest in this app previously rebalanced on the last
business day of the period, and a result that only works on that one day of
the month is not a strategy -- it is a coincidence with good timing.

This module makes the trading day explicit, so it can be varied:

- monthly on the last business day, the first, the 15th, the third Friday;
- quarterly anchored to January, February or March;
- annually every July rather than every December.

The default reproduces the old behaviour exactly, so existing
configurations keep their dates.

Every rule resolves to *actual trading days present in the price index*. A
15th that lands on a Saturday snaps to a neighbouring session rather than
being dropped, and a rule that would fall outside the data is simply
absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

DAY_RULES = {
    "last": "Last trading day of the period",
    "first": "First trading day of the period",
    "day": "A chosen day of the month",
    "nth_weekday": "The nth weekday of the period",
    "last_weekday": "The last chosen weekday of the period",
}


@dataclass
class RebalanceSpec:
    """How a rebalance calendar is built."""
    frequency: str = "M"            # D | W | M | Q | A
    day_rule: str = "last"          # see DAY_RULES
    day_of_month: int = 15          # for day_rule="day"
    weekday: int = 4                # 0=Monday .. 4=Friday
    nth: int = 1                    # 1..4, for nth_weekday
    anchor_month: int = 12          # A: which month; Q: 1-3 within the quarter
    snap: str = "next"              # next | previous, when the target is closed

    def label(self) -> str:
        f = {"D": "Daily", "W": "Weekly", "M": "Monthly",
             "Q": "Quarterly", "A": "Annual"}.get(self.frequency, self.frequency)
        if self.frequency == "D":
            return "Daily"
        if self.day_rule == "last":
            day = "last trading day"
        elif self.day_rule == "first":
            day = "first trading day"
        elif self.day_rule == "day":
            day = f"day {self.day_of_month}"
        elif self.day_rule == "nth_weekday":
            day = f"{['1st','2nd','3rd','4th'][max(0, min(3, self.nth-1))]} " \
                  f"{WEEKDAYS[self.weekday]}"
        else:
            day = f"last {WEEKDAYS[self.weekday]}"
        if self.frequency == "W":
            return f"Weekly, {WEEKDAYS[self.weekday]}"
        if self.frequency == "A":
            return f"Annual, {MONTHS[self.anchor_month-1]}, {day}"
        if self.frequency == "Q":
            months = _quarter_months(self.anchor_month)
            return (f"Quarterly, {'/'.join(MONTHS[m-1][:3] for m in months)}, "
                    f"{day}")
        return f"{f}, {day}"


def _quarter_months(anchor: int) -> List[int]:
    """Which months a quarterly rebalance lands in.

    `anchor_month` means the same thing at both frequencies: a month that is
    always included. Quarterly then takes that month and every third one
    after it, so December gives Mar/Jun/Sep/Dec -- the calendar quarters,
    and the default. January gives Jan/Apr/Jul/Oct, February gives
    Feb/May/Aug/Nov: the same cadence started a month later, which is the
    offset worth testing when asking whether a quarterly result depends on
    the calendar rather than the strategy.
    """
    m = ((int(anchor) - 1) % 12) + 1
    return sorted({((m - 1 + k * 3) % 12) + 1 for k in range(4)})


def _snap(index: pd.DatetimeIndex, targets: pd.DatetimeIndex,
          how: str = "next") -> pd.DatetimeIndex:
    """Maps target dates onto real trading days."""
    if len(targets) == 0:
        return pd.DatetimeIndex([])
    method = "bfill" if how == "next" else "ffill"
    pos = index.get_indexer(targets, method=method)
    pos = pos[pos >= 0]
    return pd.DatetimeIndex(sorted(set(index[pos])))


def _period_key(index: pd.DatetimeIndex, spec: RebalanceSpec) -> Optional[pd.Series]:
    """A label per session identifying which period it belongs to."""
    if spec.frequency == "M":
        return pd.Series(index.to_period("M"), index=index)
    if spec.frequency == "Q":
        months = _quarter_months(spec.anchor_month)
        keep = pd.Series(index.month, index=index).isin(months)
        return pd.Series(index.to_period("M"), index=index).where(keep)
    if spec.frequency == "A":
        keep = pd.Series(index.month, index=index) == int(spec.anchor_month)
        return pd.Series(index.to_period("M"), index=index).where(keep)
    if spec.frequency == "W":
        return pd.Series(index.to_period("W"), index=index)
    return None


_RESAMPLE = {"D": None, "W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}


def _canonical(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """The original period-end calendar, kept verbatim.

    Any spec that asks for the historical behaviour is routed here rather
    than rebuilt, so stored configurations cannot drift by a day. It also
    keeps one detail the general rules would lose: the final, incomplete
    period contributes its last available session.
    """
    if frequency == "D" or _RESAMPLE.get(frequency) is None:
        return pd.DatetimeIndex(index)
    ser = pd.Series(index, index=index)
    dates = ser.resample(_RESAMPLE[frequency]).last().dropna()
    return pd.DatetimeIndex(sorted(set(dates.values))).intersection(index)


def is_default(spec: "RebalanceSpec") -> bool:
    """Whether this spec asks for the historical period-end calendar."""
    if spec.day_rule != "last":
        return False
    if spec.frequency in ("D", "W", "M"):
        return True
    if spec.frequency == "Q":
        return _quarter_months(spec.anchor_month) == [3, 6, 9, 12]
    if spec.frequency == "A":
        return int(spec.anchor_month) == 12
    return False


def build_calendar(index: pd.DatetimeIndex,
                   spec: Optional[RebalanceSpec] = None) -> pd.DatetimeIndex:
    """Resolves a spec into trading days present in `index`."""
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return pd.DatetimeIndex([])
    spec = spec or RebalanceSpec()

    if spec.frequency == "D":
        return index

    if is_default(spec):
        return _canonical(index, spec.frequency)

    # Weekly with an explicit weekday is its own case: every occurrence of
    # that weekday, not one per calendar week bucket.
    if spec.frequency == "W":
        if spec.day_rule in ("last", "first"):
            keys = _period_key(index, spec)
            grouped = pd.Series(index, index=index).groupby(keys)
            picked = grouped.last() if spec.day_rule == "last" else grouped.first()
            return pd.DatetimeIndex(sorted(set(picked.dropna())))
        want = pd.Series(index.weekday, index=index) == int(spec.weekday)
        return pd.DatetimeIndex(sorted(set(index[want.to_numpy()])))

    keys = _period_key(index, spec)
    if keys is None:
        return index
    valid = keys.dropna()
    if valid.empty:
        return pd.DatetimeIndex([])

    if spec.day_rule in ("last", "first"):
        grouped = pd.Series(valid.index, index=valid.index).groupby(valid)
        picked = grouped.last() if spec.day_rule == "last" else grouped.first()
        return pd.DatetimeIndex(sorted(set(pd.to_datetime(picked.dropna()))))

    # Targets are built from the calendar, then snapped onto trading days.
    #
    # Counting occurrences inside the price index instead would be wrong
    # twice over: a truncated first month would make the "third Friday" the
    # third Friday *present in the data*, and a market holiday on a Friday
    # would shift every later count in that month.
    periods = pd.unique(valid)
    targets: List[pd.Timestamp] = []
    bounds: List[tuple] = []

    for period in periods:
        ts = period.to_timestamp()
        month_start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)

        if spec.day_rule == "day":
            day = min(max(1, int(spec.day_of_month)), month_end.day)
            targets.append(pd.Timestamp(year=ts.year, month=ts.month, day=day))
        else:
            days = pd.date_range(month_start, month_end, freq="D")
            hits = days[days.weekday == int(spec.weekday)]
            if len(hits) == 0:
                continue
            if spec.day_rule == "last_weekday":
                targets.append(hits[-1])
            else:
                n = max(1, int(spec.nth))
                if len(hits) < n:
                    continue
                targets.append(hits[n - 1])
        bounds.append((month_start, month_end))

    if not targets:
        return pd.DatetimeIndex([])

    # Snap inside the month. Falling forward off the end of a month would
    # put a "day 31" rebalance on the 1st of the month after, which is not
    # what anyone means by it.
    out = []
    for target, (lo, hi) in zip(targets, bounds):
        fwd = index[index >= target]
        back = index[index <= target]
        pick = None
        if spec.snap == "previous":
            if len(back) and back[-1] >= lo:
                pick = back[-1]
            elif len(fwd) and fwd[0] <= hi:
                pick = fwd[0]
        else:
            if len(fwd) and fwd[0] <= hi:
                pick = fwd[0]
            elif len(back) and back[-1] >= lo:
                pick = back[-1]
        if pick is not None:
            out.append(pick)
    return pd.DatetimeIndex(sorted(set(out)))


# ----------------------------------------------------------------------
def day_variants(spec: RebalanceSpec, max_variants: int = 24) -> List[RebalanceSpec]:
    """A family of specs differing only in the trading day.

    This is the point of the whole module: running the same strategy across
    every plausible trading day says whether the result is a strategy or a
    date. A wide spread of outcomes across days is the same warning sign as
    a wide spread across parameter values.
    """
    out: List[RebalanceSpec] = []
    if spec.frequency == "D":
        return [spec]

    if spec.frequency == "W":
        for wd in range(5):
            s = RebalanceSpec(**{**spec.__dict__})
            s.day_rule, s.weekday = "weekday", wd
            s.day_rule = "nth_weekday"
            s.nth = 1
            out.append(s)
        return out[:max_variants]

    for rule, extra in [("first", {}), ("last", {})]:
        s = RebalanceSpec(**{**spec.__dict__}); s.day_rule = rule
        out.append(s)
    for d in (5, 10, 15, 20, 25):
        s = RebalanceSpec(**{**spec.__dict__}); s.day_rule = "day"; s.day_of_month = d
        out.append(s)
    for n in (1, 2, 3):
        for wd in (0, 2, 4):
            s = RebalanceSpec(**{**spec.__dict__})
            s.day_rule, s.nth, s.weekday = "nth_weekday", n, wd
            out.append(s)
    return out[:max_variants]


def month_variants(spec: RebalanceSpec) -> List[RebalanceSpec]:
    """A family differing only in the anchor month (annual) or anchor
    offset (quarterly)."""
    out: List[RebalanceSpec] = []
    if spec.frequency == "A":
        for m in range(1, 13):
            s = RebalanceSpec(**{**spec.__dict__}); s.anchor_month = m
            out.append(s)
    elif spec.frequency == "Q":
        for a in (1, 2, 3):
            s = RebalanceSpec(**{**spec.__dict__}); s.anchor_month = a
            out.append(s)
    return out
