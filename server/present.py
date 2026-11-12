"""Recorded nodes to JSON for the browser.

The classic code describes its output in Streamlit's terms and in its own
small HTML vocabulary (dials, notes, flags, run bars). This module turns
that into blocks the new front end draws: a KPI is a label, a value, a
sub-line and a tone; a table is cells of text with a tone and an alignment;
a chart is the series it plots. Every figure is already formatted by the
classic code; nothing is recomputed.
"""
from __future__ import annotations

import base64
import datetime as dt
import html as _html
import json
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qbt.brand import BRAND

from .serialize import cell_text, is_figures

MAX_ROWS = 3000

# The classic palette, as roles in the new one (docs/DESIGN.md).
_COLOUR = {
    # Red inside content is text or a chart mark, where the brand red is not
    # used: those become gold.
    BRAND["red"].lower(): "var(--gold)", BRAND["red_dark"].lower(): "var(--gold)",
    BRAND["red_bright"].lower(): "var(--gold-strong)",
    BRAND["gain"].lower(): "var(--gain)", BRAND["loss"].lower(): "var(--loss)",
    BRAND["muted"].lower(): "var(--text-2)", BRAND["faint"].lower(): "var(--text-3)",
    BRAND["rule"].lower(): "var(--line)", BRAND["rule_soft"].lower(): "var(--line-soft)",
    BRAND["panel"].lower(): "var(--surface)", BRAND["panel_2"].lower(): "var(--raised)",
    BRAND["text"].lower(): "var(--text)", BRAND["bg"].lower(): "var(--bg)",
    "#d9a441": "var(--warn)", "#8dba6b": "var(--gain-soft)", "#111": "var(--on-accent)",
}
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#111\b")
_TAG = re.compile(r"<[^>]+>")
_DIAL = re.compile(r'<div class="dial ?(\w*)"><span class="k">(.*?)</span>'
                   r'<span class="v">(.*?)</span>(?:<div class=d>(.*?)</div>)?</div>', re.S)


def clean_html(markup: str) -> str:
    """Markup written by the app's own code, made safe to insert: scripts
    and event handlers removed, the old palette mapped to the new roles."""
    s = re.sub(r"<\s*(script|style|iframe|object|embed)\b.*?</\s*\1\s*>", "", markup, flags=re.S | re.I)
    s = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", s, flags=re.I)
    s = re.sub(r"javascript:", "", s, flags=re.I)
    return _HEX.sub(lambda m: _COLOUR.get(m.group(0).lower(), m.group(0)), s)


def text_of(markup: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(_TAG.sub(" ", markup))).strip()


def md_html(body: str) -> str:
    """The little Markdown the classic code uses: bold, italics, code,
    links are not used; lines starting with '- ' are a list."""
    out, items = [], []

    def inline(t: str) -> str:
        t = _html.escape(t, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        return re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()
    for line in body.split("\n"):
        if line.strip().startswith("- "):
            items.append(inline(line.strip()[2:]))
        else:
            flush()
            if line.strip():
                out.append(f"<p>{inline(line)}</p>")
    flush()
    return "".join(out)


def _inner(markup: str) -> str:
    """Content of the outermost element."""
    m = re.match(r"^<(\w+)[^>]*>(.*)</\1>\s*$", markup.strip(), flags=re.S)
    return m.group(2) if m else markup


def markdown_block(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    body = node["body"].strip()
    if not body or body.startswith("<style") or text_of(body) == "":
        return None
    if not node.get("html") and "<" not in body:
        return {"kind": "html", "html": md_html(body)}
    dials = _DIAL.findall(body)
    if dials:
        return {"kind": "kpis", "items": [
            {"label": text_of(k), "value": text_of(v), "sub": text_of(d), "tone": tone}
            for tone, k, v, d in dials]}
    m = re.match(r'<(div|details|span) class="([\w-]+)[" ]', body)
    cls = m.group(2) if m else ""
    if cls == "masthead":
        h1 = re.search(r"<h1>(.*?)</h1>", body, flags=re.S)
        sub = re.search(r'<div class="sub">(.*?)</div>', body, flags=re.S)
        return {"kind": "title", "text": text_of(h1.group(1)) if h1 else "",
                "sub": text_of(sub.group(1)) if sub else ""}
    if cls == "runbar":
        lead = re.search(r'<span class="lead">(.*?)</span>(?=<span|</div>)', body, flags=re.S)
        items = re.findall(r'<span class="item">(.*?)</span>(?=<span|</div>)', body, flags=re.S)
        return {"kind": "context", "lead": text_of(lead.group(1)) if lead else "",
                "items": [clean_html(i) for i in items]}
    if cls in ("eyebrow", "side-label"):
        return {"kind": "heading", "text": text_of(body)}
    if cls == "flag":
        return {"kind": "flag", "html": clean_html(_inner(body))}
    if cls == "note":
        inner = _inner(body)
        s = re.match(r"<summary>(.*?)<span class=\"more\">.*?</span></summary>(.*)$", inner, flags=re.S)
        if s:
            inner = f"{s.group(1)} {s.group(2)}"
        return {"kind": "note", "html": clean_html(inner)}
    return {"kind": "html", "html": clean_html(body), "cls": cls}


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def _tone_of_css(rules: List[tuple]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for attr, val in rules:
        v = str(val).lower().replace(" ", "")
        if attr == "color":
            if v == BRAND["gain"].lower():
                out["tone"] = "pos"
            elif v == BRAND["loss"].lower():
                out["tone"] = "neg"
            elif v == BRAND["faint"].lower():
                out["tone"] = "dim"
        elif attr == "background-color":
            m = re.match(r"rgba\((\d+),(\d+),(\d+),([\d.]+)\)", v)
            if m:
                out["heat"] = ("pos" if m.group(1) == "79" else "neg", round(float(m.group(4)) / 0.42 * 50))
        elif attr == "font-weight":
            out["strong"] = True
    return out


def _printf(fmt: str, v: Any) -> str:
    try:
        return fmt % v
    except Exception:
        return cell_text(v)


def table_block(data: Any, hide_index: bool, column_config: Dict[str, Any],
                height: Optional[int], editor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    styler = data if hasattr(data, "ctx") and hasattr(data, "data") else None
    df = styler.data if styler is not None else data
    if df is None:
        return {"kind": "table", "columns": [], "rows": []}
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    css, fmt = {}, {}
    if styler is not None:
        try:
            styler._compute()
            css = dict(styler.ctx)
            fmt = {k: f for k, f in styler._display_funcs.items()}
        except Exception:
            css, fmt = {}, {}
    n_index = 0
    show = df
    if not hide_index and editor is None:
        n_index = df.index.nlevels
        show = df.reset_index()
        show.columns = [str(c) if not isinstance(c, tuple) else " / ".join(map(str, c))
                        for c in show.columns]
    total = len(show)
    clipped = total > MAX_ROWS
    if clipped:
        show = show.iloc[:MAX_ROWS]
    cols = []
    for j, c in enumerate(show.columns):
        name = " / ".join(map(str, c)) if isinstance(c, tuple) else str(c)
        cfg = dict(column_config.get(c) or column_config.get(name) or {})
        kind = cfg.get("type", "")
        col: Dict[str, Any] = {"name": cfg.get("label") or name, "key": name,
                               "align": cfg.get("alignment") or ("right" if is_figures(show[c]) else "left")}
        if kind == "ProgressColumn":
            col["bar"] = [cfg.get("min_value", 0), cfg.get("max_value", 100)]
            col["align"] = "right"
        elif kind == "LineChartColumn":
            col["spark"] = True
        if cfg.get("help"):
            col["help"] = cfg["help"]
        if editor is not None:
            col["edit"] = None if name in editor["disabled"] else (
                {"type": "select", "options": [str(o) for o in cfg.get("options", [])]}
                if kind == "SelectboxColumn" else
                {"type": "number", "min": cfg.get("min_value"), "max": cfg.get("max_value"),
                 "step": cfg.get("step")}
                if (kind == "NumberColumn" or pd.api.types.is_numeric_dtype(show[c])) else
                {"type": "text"})
        col["_fmt"] = cfg.get("format")
        cols.append(col)
    rows = []
    values = show.to_numpy(dtype=object)
    for i in range(len(show)):
        row = []
        for j, col in enumerate(cols):
            v = values[i, j]
            if col.get("spark"):
                arr = [x for x in (list(v) if v is not None and not isinstance(v, float) else [])
                       if x is not None and x == x]
                row.append({"s": [round(float(x), 4) for x in arr]})
                continue
            cell: Dict[str, Any] = {}
            missing = v is None or (isinstance(v, float) and math.isnan(v))
            key = (i, j - n_index)
            if editor is not None:
                cell["v"] = None if missing else (v.item() if hasattr(v, "item") else v)
            if missing:
                cell["t"] = ""
            elif col["_fmt"] and isinstance(v, (int, float, np.number)):
                cell["t"] = _printf(col["_fmt"], v)
            elif j >= n_index and key in fmt:
                try:
                    cell["t"] = str(fmt[key](v))
                except Exception:
                    cell["t"] = cell_text(v)
            else:
                cell["t"] = cell_text(v)
            if col.get("bar") and not missing:
                cell["b"] = float(v)
            if j >= n_index and key in css:
                cell.update(_tone_of_css(css[key]))
            row.append(cell)
        rows.append(row)
    for col in cols:
        col.pop("_fmt", None)
    out = {"kind": "table", "columns": cols, "rows": rows, "index_cols": n_index,
           "scroll": bool(height) or len(rows) > 40}
    if clipped:
        out["clipped"] = [MAX_ROWS, total]
    return out


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def _arr(v: Any) -> List[Any]:
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
        if "shape" in v:
            a = a.reshape([int(x) for x in str(v["shape"]).split(",")])
        return a.tolist()
    if isinstance(v, (list, tuple)):
        return [(_arr(x) if isinstance(x, (dict, list, tuple)) else x) for x in v]
    return [] if v is None else [v]


def _nums(v: List[Any]) -> List[Any]:
    out = []
    for x in v:
        if isinstance(x, list):
            out.append(_nums(x))
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
            out.append(round(float(x), 6))
        else:
            out.append(None)
    return out


def _xs(v: List[Any]) -> List[Any]:
    out = []
    for x in v:
        if isinstance(x, str) and re.match(r"^\d{4}-\d{2}-\d{2}", x):
            out.append(x[:10])
        elif isinstance(x, float) and math.isfinite(x):
            out.append(round(x, 6))
        elif isinstance(x, (int, str)):
            out.append(x)
        else:
            out.append(None)
    return out


def _axis_kind(xs: List[Any]) -> str:
    vals = [x for x in xs if x is not None]
    if vals and all(isinstance(x, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", x) for x in vals[:50]):
        return "date"
    if vals and all(isinstance(x, (int, float)) for x in vals[:50]):
        return "value"
    return "category"


def chart_block(fig: Any) -> Dict[str, Any]:
    spec = json.loads(fig.to_json())
    layout = spec.get("layout", {})
    title = layout.get("title", {})
    yaxis, xaxis = layout.get("yaxis", {}) or {}, layout.get("xaxis", {}) or {}
    out: Dict[str, Any] = {
        "kind": "chart", "chart": "figure",
        "title": (title.get("text", "") if isinstance(title, dict) else str(title)) or "",
        "height": int(layout.get("height") or 320),
        "ylog": yaxis.get("type") == "log",
        "ysuffix": yaxis.get("ticksuffix", "") or "", "xsuffix": xaxis.get("ticksuffix", "") or "",
        "yformat": yaxis.get("tickformat", "") or "",
        "xtitle": ((xaxis.get("title") or {}).get("text", "") if isinstance(xaxis.get("title"), dict) else "") or "",
        "ytitle": ((yaxis.get("title") or {}).get("text", "") if isinstance(yaxis.get("title"), dict) else "") or "",
        "barmode": layout.get("barmode", ""), "yrange": yaxis.get("range"),
        "traces": [], "hlines": [], "vlines": [], "bands": [],
    }
    band_names: Dict[str, str] = {}
    for tr in spec.get("data", []):
        kind = tr.get("type", "scatter")
        name = str(tr.get("name", "") or "")
        if kind == "scatter":
            xs, ys = _xs(_arr(tr.get("x"))), _nums(_arr(tr.get("y")))
            if xs == [None]:                                 # legend entry for a shaded band
                band_names[str((tr.get("marker") or {}).get("color", "")).lower()] = name
                continue
            mode = tr.get("mode", "lines")
            line = tr.get("line") or {}
            t: Dict[str, Any] = {"t": "points" if "markers" in mode and "lines" not in mode else "line",
                                 "name": name, "x": xs, "y": ys}
            if "text" in mode and tr.get("text") is not None:
                t["labels"] = [str(x) for x in _arr(tr.get("text"))]
            if tr.get("stackgroup"):
                t["stack"] = True
            if tr.get("fill") == "tozeroy":
                t["area"] = True
            if tr.get("fill") == "tonexty" and not tr.get("stackgroup"):
                t["band_to_prev"] = True
            if line.get("width") == 0:
                t["hidden_line"] = True
            if line.get("dash") in ("dot", "dash"):
                t["dash"] = True
            if tr.get("showlegend") is False:
                t["no_legend"] = True
            if str(line.get("color", "")).lower() == BRAND["loss"].lower():
                t["tone"] = "neg"
            out["traces"].append(t)
        elif kind == "bar":
            colors = (tr.get("marker") or {}).get("color")
            t = {"t": "bar", "name": name, "x": _xs(_arr(tr.get("x"))), "y": _nums(_arr(tr.get("y")))}
            if isinstance(colors, list):
                t["signed"] = True
            elif str(colors or "").lower() == BRAND["faint"].lower():
                t["quiet"] = True
            if tr.get("opacity") is not None:
                t["opacity"] = tr["opacity"]
            out["traces"].append(t)
        elif kind == "histogram":
            vals = np.asarray([x for x in _arr(tr.get("x")) if isinstance(x, (int, float)) and math.isfinite(x)],
                              dtype="float64")
            if len(vals):
                bins = int(tr.get("nbinsx") or 40)
                counts, edges = np.histogram(vals, bins=max(6, min(bins, max(6, len(vals) // 12))))
                out["traces"].append({"t": "hist", "name": name, "counts": counts.tolist(),
                                      "edges": [round(float(e), 6) for e in edges], "n": int(len(vals))})
        elif kind == "heatmap":
            z = _nums(_arr(tr.get("z")))
            text = tr.get("text")
            cells = _arr(text) if text is not None else None
            out["traces"].append({
                "t": "heat", "x": [str(x) for x in _arr(tr.get("x"))], "y": [str(y) for y in _arr(tr.get("y"))],
                "z": z, "text": cells if (cells and tr.get("texttemplate") == "%{text}") else None,
                "decimals": 1 if tr.get("texttemplate") == "%{z:.1f}" else 2,
                "scale": ("corr" if (tr.get("zmin") == -1 and tr.get("zmax") == 1) else
                          "sign" if tr.get("zmid") == 0 else "rank"),
                "mid": tr.get("zmid")})
        elif kind == "surface":
            out["traces"].append({"t": "surface", "x": _nums(_arr(tr.get("x"))), "y": _nums(_arr(tr.get("y"))),
                                  "z": _nums(_arr(tr.get("z")))})
            scene = layout.get("scene", {}) or {}
            out["xtitle"] = ((scene.get("xaxis") or {}).get("title") or {}).get("text", "")
            out["ytitle"] = ((scene.get("yaxis") or {}).get("title") or {}).get("text", "")
            out["ztitle"] = ((scene.get("zaxis") or {}).get("title") or {}).get("text", "")
    notes = {(_a.get("x"), _a.get("y")): _a.get("text", "") for _a in layout.get("annotations", []) or []}
    for sh in layout.get("shapes", []) or []:
        if sh.get("type") == "rect":
            colour = str(sh.get("fillcolor", "")).lower()
            out["bands"].append({"x0": str(sh.get("x0"))[:10], "x1": str(sh.get("x1"))[:10],
                                 "label": band_names.get(colour, ""), "c": colour})
        elif sh.get("type") == "line":
            if sh.get("yref", "y") == "y" and sh.get("y0") == sh.get("y1") and sh.get("xref") != "x":
                out["hlines"].append(sh.get("y0"))
            elif sh.get("x0") == sh.get("x1"):
                x = sh.get("x0")
                label = next((t for (ax, _ay), t in notes.items() if ax == x), "")
                out["vlines"].append({"x": x if not isinstance(x, str) else x[:10], "label": label,
                                      "tone": "neg" if str((sh.get("line") or {}).get("color", "")).lower()
                                      == BRAND["loss"].lower() else ""})
    if out["bands"]:                                        # one palette slot per regime, in legend order
        order = list(band_names.values())
        for b in out["bands"]:
            b["i"] = order.index(b["label"]) if b["label"] in order else 0
            b.pop("c", None)
        out["band_labels"] = order
    xs_all = [x for t in out["traces"] if "x" in t and t["t"] in ("line", "points", "bar") for x in t["x"]]
    out["xtype"] = _axis_kind(xs_all) if xs_all else "category"
    if xaxis.get("type") == "date":
        out["xtype"] = "date"
    if any(t["t"] == "bar" for t in out["traces"]) and out["xtype"] != "date":
        out["xtype"] = "category"
    out["yreversed"] = yaxis.get("autorange") == "reversed"
    return out


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
def _jsonable(v: Any) -> Any:
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()[:10]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def control_block(node: Dict[str, Any]) -> Dict[str, Any]:
    kind = node["kind"]
    out: Dict[str, Any] = {"kind": "control", "control": kind, "id": node["id"],
                           "label": node["label"], "help": node.get("help") or ""}
    for k in ("min", "max", "step", "placeholder", "horizontal", "height", "secret", "style",
              "clearable", "max_selections", "primary", "disabled", "is_int"):
        if node.get(k) is not None:
            out[k] = _jsonable(node[k])
    if node.get("label_visibility") in ("collapsed", "hidden"):
        out["hide_label"] = True
    opts = node.get("options")
    if opts is not None:
        out["options"] = node["option_labels"]
        value = node["value"]
        if kind == "multiselect":
            out["value"] = [i for i, o in enumerate(opts) if any(o is v or o == v for v in value)]
        else:
            out["value"] = next((i for i, o in enumerate(opts) if o is value or o == value), None)
    else:
        out["value"] = _jsonable(node["value"])
    return out


def node_blocks(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    kind = node["kind"]
    kids = lambda: [b for c in node.get("children", []) for b in node_blocks(c)]   # noqa: E731
    if kind == "markdown":
        b = markdown_block(node)
        return [b] if b else []
    if kind == "caption":
        return [{"kind": "caption", "html": md_html(node["body"])}]
    if kind == "code":
        return [{"kind": "code", "text": node["body"]}]
    if kind in ("error", "warning", "info", "success"):
        return [{"kind": "alert", "level": kind, "html": md_html(node["body"])}]
    if kind == "dataframe":
        return [table_block(node["data"], node["hide_index"], node["column_config"], node["height"])]
    if kind == "data_editor":
        b = table_block(node["frame"], True, node["column_config"], node["height"],
                        editor={"disabled": node["disabled"]})
        b.update({"kind": "editor", "id": node["id"], "dynamic": node["dynamic"]})
        return [b]
    if kind == "plotly_chart":
        return [chart_block(node["figure"])]
    if kind == "image":
        data = node["data"]
        if isinstance(data, (bytes, bytearray)):
            return [{"kind": "image", "src": "data:image/png;base64," + base64.b64encode(data).decode("ascii")}]
        return []
    if kind == "iframe":
        return [{"kind": "frame", "html": node["html"], "height": node["height"]}]
    if kind == "download":
        return [{"kind": "download", "id": node["id"], "label": node["label"],
                 "file_name": node.get("file_name") or "download"}]
    if kind == "file_uploader":
        return [{"kind": "upload", "id": node["id"], "label": node["label"], "types": node["types"],
                 "multiple": node["multiple"], "names": node["names"], "help": node.get("help") or ""}]
    if kind in ("selectbox", "radio", "multiselect", "number_input", "slider", "text_input",
                "text_area", "date_input", "checkbox", "toggle", "button", "button_group",
                "select_slider", "chat_input"):
        return [control_block(node)]
    if kind == "row":
        cols = [[b for c in col.get("children", []) for b in node_blocks(c)] for col in node["children"]]
        if not any(cols):
            return []
        return [{"kind": "row", "weights": node["weights"], "cols": cols}]
    if kind == "expander":
        return [{"kind": "panel", "label": node["label"], "open": node["expanded"], "blocks": kids()}]
    if kind == "form":
        return [{"kind": "form", "blocks": kids()}]
    if kind == "chat_message":
        return [{"kind": "chat", "role": node["label"], "blocks": kids()}]
    if kind == "tabs":
        labels = node["labels"]
        active = node.get("active") or labels[0]
        shown = next((c for c in node["children"] if c["label"] == active), None)
        return [{"kind": "tabs", "labels": labels, "active": active,
                 "blocks": [b for c in (shown or {}).get("children", []) for b in node_blocks(c)]
                 if shown is not None and shown.get("open") else []}]
    return kids()                                           # root, container, column, tab


def page(rec: Any) -> Dict[str, Any]:
    return {"sidebar": node_blocks(rec.sidebar), "main": node_blocks(rec.main)}
