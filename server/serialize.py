"""Screen blocks to JSON for the browser.

The browser draws what it is given. Cell text, sign tones and column
alignment are decided here, with the rules app.py applies when it styles a
table (`_sign_color`, `_is_figures`), so no figure is formatted client-side.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_FIGURE = re.compile(
    r"^\s*(n/a|[+\-−]?\$?[\d,]*\.?\d+(e[+\-]?\d+)?\s*(%|x|bps|pp)?)\s*$",
    re.IGNORECASE)


def sign_tone(v: Any) -> str:
    """app.py `_sign_color`, as a tone name: pos | neg | dim | ''."""
    if v is None:
        return "dim"
    if isinstance(v, str):
        t = v.strip()
        if t in ("", "—", "n/a", "nan", "None"):
            return "dim"
        if t.startswith("-"):
            return "neg"
        try:
            if t.startswith("+") or (t[:1].isdigit() and float(
                    t.replace("%", "").replace(",", "") or 0) > 0):
                return "pos"
        except ValueError:
            return ""
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:
        return "dim"
    return "pos" if f > 0 else ("neg" if f < 0 else "dim")


def is_figures(col: pd.Series) -> bool:
    """app.py `_is_figures`: should this column be right-aligned?"""
    if pd.api.types.is_bool_dtype(col):
        return False
    if pd.api.types.is_numeric_dtype(col):
        return True
    vals = [str(v) for v in col.dropna().tolist()[:300]]
    vals = [v for v in vals if v.strip() not in ("", "n/a")]
    return bool(vals) and all(_FIGURE.match(v) for v in vals)


def cell_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}" if abs(int(v)) >= 10000 else str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if math.isnan(v) else f"{float(v):,.4g}"
    return str(v)


def table_json(block: Dict[str, Any]) -> Dict[str, Any]:
    df = block["frame"]
    if df is None or df.empty:
        return {"kind": "table", "key": block.get("key", ""), "columns": [], "rows": []}
    if not block.get("hide_index", True):
        df = df.reset_index()
    cols = []
    for c in df.columns:
        cols.append({"name": str(c),
                     "align": "right" if is_figures(df[c]) else "left",
                     "signed": c in block["signed"],
                     "strong": c in block["emphasise"]})
    rows = []
    for rec in df.itertuples(index=False, name=None):
        row = []
        for meta, v in zip(cols, rec):
            cell: List[Any] = [cell_text(v)]
            if meta["signed"]:
                cell.append(sign_tone(v))
            row.append(cell)
        rows.append(row)
    return {"kind": "table", "key": block.get("key", ""), "columns": cols, "rows": rows}


def numbers(values: Any) -> List[Any]:
    arr = np.asarray(values, dtype="float64")
    return [None if not math.isfinite(x) else float(x) for x in arr.tolist()]


def dates(index: Any) -> List[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(index)]


def chart_json(block: Dict[str, Any]) -> Dict[str, Any]:
    data = block["data"]
    out: Dict[str, Any] = {k: v for k, v in block.items()
                           if k not in ("data", "legacy")}
    kind = block["chart"]
    if "series" in data:
        out["x"] = dates(data["x"])
        out["series"] = [{"name": s["name"], "y": numbers(s["y"])} for s in data["series"]]
    elif kind == "monthly_heatmap":
        out["years"] = list(data["years"])
        out["z"] = [numbers(row) for row in np.asarray(data["z"], dtype="float64")]
    elif kind == "distribution":
        vals = np.asarray(data["values"], dtype="float64")
        vals = vals[np.isfinite(vals)]
        counts, edges = np.histogram(vals, bins=min(int(data["bins"]), max(10, len(vals) // 20)))
        out["counts"] = [int(c) for c in counts]
        out["edges"] = numbers(edges)
        out["var95"] = data["var95"]
        out["n"] = int(len(vals))
    elif kind == "bars":
        out["labels"] = [str(x) for x in data["labels"]]
        out["values"] = numbers(data["values"])
    return out


def block_json(block: Dict[str, Any]) -> Dict[str, Any]:
    kind = block["kind"]
    if kind == "table":
        return table_json(block)
    if kind == "chart":
        return chart_json(block)
    if kind == "panel":
        return {"kind": "panel", "label": block["label"], "open": block["open"],
                "blocks": [block_json(b) for b in block["blocks"]]}
    return dict(block)


def screen_json(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [block_json(b) for b in blocks]
