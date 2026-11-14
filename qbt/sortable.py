"""Tables that sort by value while showing exactly the text they showed before.

Most tables reach the screen with their figures already formatted
("10.59%", "-55.19%", "$1,234", "0.44"). A table column of text sorts as
text: "9.75%" comes after "10.59%" and the minus signs scramble the
negatives. `sortable` hands the table numbers instead, and keeps every
cell's original text as its display, so the table looks the same and sorts
by value.

A Styler (colours by sign, say) keeps its styling: the styles are worked
out first on the original text, exactly as before, then carried over cell
by cell onto the numeric table.

A column is converted only if every cell reads as a figure and each number
has one spelling; anything else is left as it was.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

FIGURE = re.compile(
    r"^\s*(n/a|[+\-−]?\$?[\d,]*\.?\d+(e[+\-]?\d+)?\s*(%|x|bps|pp)?)\s*$",
    re.IGNORECASE)
_SUFFIX = re.compile(r"\s*(%|x|bps|pp)\s*$", re.IGNORECASE)
_BLANK = ("", "n/a", "nan", "none", "—")
MAX_CELLS = 200_000           # beyond this the table is shown as it was


def to_number(text: Any) -> float:
    """The number a formatted figure stands for (NaN for a blank or n/a).
    Raises ValueError when the text is not a figure."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return float("nan")
    t = str(text).strip()
    if t.lower() in _BLANK:
        return float("nan")
    if not FIGURE.match(t):
        raise ValueError(t)
    t = _SUFFIX.sub("", t).replace("−", "-").replace(",", "").replace("$", "")
    return float(t)


Plan = Dict[Any, Tuple[pd.Series, Dict[float, str], str]]


def plan(frame: pd.DataFrame) -> Plan:
    """For each text column that holds figures: (numbers, number -> text,
    text shown for a blank)."""
    out: Plan = {}
    if frame.columns.has_duplicates or frame.index.has_duplicates \
            or frame.size == 0 or frame.size > MAX_CELLS:
        return out
    for c in frame.columns:
        col = frame[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            continue
        texts = [v for v in col.tolist() if not (isinstance(v, float) and math.isnan(v))
                 and v is not None and str(v).strip().lower() not in _BLANK]
        if not texts or not all(isinstance(v, str) for v in texts):
            continue
        try:
            nums = [to_number(v) for v in col.tolist()]
        except ValueError:
            continue
        show: Dict[float, str] = {}
        blanks = set()
        ok = True
        for v, n in zip(col.tolist(), nums):
            if math.isnan(n):
                blanks.add("" if v is None or (isinstance(v, float) and math.isnan(v))
                           else str(v))
                continue
            if show.setdefault(n, str(v)) != str(v):
                ok = False          # one number, two spellings: leave the column alone
                break
        if not ok or len(blanks) > 1:
            continue
        out[c] = (pd.Series(nums, index=col.index, dtype=float), show,
                  blanks.pop() if blanks else "")
    return out


def _numeric(frame: pd.DataFrame, p: Plan) -> pd.DataFrame:
    num = frame.copy()
    for c, (nums, _, _) in p.items():
        num[c] = nums
    return num


def _formatted(styler, p: Plan):
    for c, (_, show, na) in p.items():
        styler = styler.format(lambda v, s=show: s.get(v, ""), subset=[c], na_rep=na)
    return styler


def _plain_safe(frame: pd.DataFrame, p: Plan, index_shown: bool) -> bool:
    """A plain table becomes a Styler only if nothing else in it would look
    different: a Styler shows every cell as its own text, so a number column
    left alone ("0.44") would turn into "0.440000", and a missing value or a
    shown index would be spelled differently."""
    for c in frame.columns:
        if c in p:
            continue
        col = frame[c]
        if not pd.api.types.is_object_dtype(col) and not pd.api.types.is_string_dtype(col):
            return False
        if col.isna().any():
            return False
    if index_shown and not (pd.api.types.is_object_dtype(frame.index)
                            or pd.api.types.is_string_dtype(frame.index)):
        return False
    return True


def sortable(data: Any, index_shown: bool = True) -> Any:
    """`data` (a DataFrame or a Styler) as a Styler that sorts by value and
    shows the same text, or `data` unchanged when nothing needs it, or when
    converting it would change how any other cell looks."""
    try:
        from pandas.io.formats.style import Styler
    except ImportError:              # pragma: no cover
        return data
    if isinstance(data, Styler):
        frame = data.data
        p = plan(frame)
        if not p:
            return data
        data._compute()             # styles worked out on the original text
        css = np.full(frame.shape, "", dtype=object)
        for (i, j), props in data.ctx.items():
            css[i, j] = "; ".join(f"{a}: {v}" for a, v in props)
        new = _numeric(frame, p).style
        if (css != "").any():
            new = new.apply(lambda d, c=css: pd.DataFrame(c, index=d.index, columns=d.columns),
                            axis=None)
        moved = {frame.columns.get_loc(c) for c in p}
        new._display_funcs.update({k: f for k, f in data._display_funcs.items()
                                   if k[1] not in moved})
        new._display_funcs_index.update(data._display_funcs_index)
        new._display_funcs_columns.update(data._display_funcs_columns)
        new.hidden_columns = list(data.hidden_columns)
        new.hide_index_ = data.hide_index_
        new.table_styles = data.table_styles
        return _formatted(new, p)
    if isinstance(data, pd.DataFrame):
        if isinstance(data.columns, pd.MultiIndex):
            return data
        p = plan(data)
        if not p or not _plain_safe(data, p, index_shown):
            return data
        return _formatted(_numeric(data, p).style, p)
    return data
