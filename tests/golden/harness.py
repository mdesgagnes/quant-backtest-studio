"""Golden-output harness: drives the real Streamlit app headlessly.

The reference behaviour of this project is what `app.py` does today. Rather
than re-implement its run pipeline in a test (and risk testing the copy), the
harness runs the actual script through Streamlit's `AppTest`, with three
test-only substitutions:

* file uploaders return fixture files instead of a browser upload;
* download buttons record what they would have served, so every export is
  captured exactly as the app produced it;
* network fetches (Yahoo Finance, Bank of Canada) are refused, so a golden
  run can never depend on today's market data.

Nothing in `app.py` or `qbt/` is modified. A run is reduced to a *record*: a
dict of plain text and numbers (metrics, every screen's content, every
export's content). `tests/golden/expected/<scenario>/` holds the recorded
reference; `compare()` reports every difference between two records.

The same record format is the contract for the redesigned app: a new front
end passes when the record built from its own API equals the reference.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import zipfile
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIXTURES = os.path.join(HERE, "fixtures")
EXPECTED = os.path.join(HERE, "expected")
ACTUAL = os.path.join(HERE, "_actual")          # written on mismatch; git-ignored

FLOAT_FMT = "%.9g"          # digits kept when a table is reduced to text
RTOL, ATOL = 1e-9, 1e-12    # tolerance for raw metric values
FULL_ROWS = 60              # tables up to this many rows are stored whole


def fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as fh:
        return fh.read()


# ----------------------------------------------------------------------
# Text normalisation: the only things allowed to differ between two runs
# are the wall-clock stamps the app prints.
# ----------------------------------------------------------------------
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def scrub(text: str) -> str:
    today = date.today().isoformat()
    text = re.sub(re.escape(today) + r"[ T]\d{2}:\d{2}(:\d{2})?", "<NOW>", str(text))
    text = text.replace(today, "<TODAY>")
    return _UUID.sub("<UUID>", text)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def frame_text(df: pd.DataFrame, index: bool = True) -> str:
    """A table as canonical text: fixed float precision, stable line ends."""
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: scrub(v) if isinstance(v, str) else v)
    return d.to_csv(index=index, float_format=FLOAT_FMT, lineterminator="\n",
                    na_rep="")


def table_record(df: Any, index: bool = True) -> Dict[str, Any]:
    if df is None:
        return {"empty": True}
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    text = frame_text(df, index)
    rec: Dict[str, Any] = {"columns": [str(c) for c in df.columns],
                           "rows": int(len(df)), "digest": digest(text)}
    lines = text.splitlines()
    if len(df) <= FULL_ROWS:
        rec["text"] = lines
    else:
        rec["head"], rec["tail"] = lines[:4], lines[-3:]
    return rec


# ----------------------------------------------------------------------
# Charts: a Plotly figure is reduced to what it draws, not how it is styled,
# so a restyle passes and a change in the plotted numbers does not.
# ----------------------------------------------------------------------
def _array(v: Any) -> Optional[np.ndarray]:
    if isinstance(v, dict) and "bdata" in v:
        arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
        return arr.astype("float64")
    if isinstance(v, list):
        flat: List[Any] = []
        stack = [v]
        while stack:
            cur = stack.pop()
            for item in cur:
                if isinstance(item, list):
                    stack.append(item)
                elif isinstance(item, dict) and "bdata" in item:
                    flat.extend(_array(item).tolist())
                else:
                    flat.append(item)
        nums = [float(x) for x in flat if isinstance(x, (int, float)) and x is not None]
        if len(nums) >= max(1, len(flat) // 2):
            return np.asarray(nums, dtype="float64")
    return None


def _sig(arr: Optional[np.ndarray]) -> Optional[str]:
    if arr is None or arr.size == 0:
        return None
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return "nan"
    return f"n={arr.size} sum={FLOAT_FMT % a.sum()} abs={FLOAT_FMT % np.abs(a).sum()}"


def chart_record(spec_json: str) -> Dict[str, Any]:
    spec = json.loads(spec_json)
    layout = spec.get("layout", {})
    title = layout.get("title", {})
    traces = []
    for tr in spec.get("data", []):
        item = {"type": tr.get("type", "scatter"), "name": scrub(tr.get("name", ""))}
        for axis in ("x", "y", "z"):
            if axis in tr:
                raw = tr[axis]
                sig = _sig(_array(raw))
                if sig is None and isinstance(raw, list):
                    sig = f"n={len(raw)} labels={digest(scrub(json.dumps(raw, default=str)))}"
                item[axis] = sig
        traces.append(item)
    return {"title": scrub(title.get("text", "") if isinstance(title, dict) else str(title)),
            "yaxis": (layout.get("yaxis", {}) or {}).get("type", "linear"),
            "traces": traces}


# ----------------------------------------------------------------------
# The app under test
# ----------------------------------------------------------------------
class _Upload:
    def __init__(self, name: str, data: bytes):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


class LegacyApp:
    """`app.py` running under AppTest. Use as a context manager."""

    def __init__(self, timeout: int = 600, offline: bool = True):
        self.timeout = timeout
        self.offline = offline          # False only for performance measurement
        self.uploads: Dict[str, List[_Upload]] = {}
        self.downloads: Dict[str, Dict[str, Any]] = {}
        self.timings: List[Tuple[str, float]] = []
        self._undo: List[Callable[[], None]] = []
        self.at = None

    # -- patches --------------------------------------------------------
    def _patch(self, obj: Any, name: str, new: Any) -> None:
        old = getattr(obj, name)
        setattr(obj, name, new)
        self._undo.append(lambda: setattr(obj, name, old))

    def __enter__(self) -> "LegacyApp":
        import streamlit as st
        from streamlit.delta_generator import DeltaGenerator
        from streamlit.testing.v1 import AppTest

        app = self

        def file_uploader(_dg, label, type=None, accept_multiple_files=False,
                          key=None, **_kw):
            files = app.uploads.get(key) or app.uploads.get(label) or []
            if accept_multiple_files:
                return list(files)
            return files[0] if files else None

        def download_button(_dg, label, data, file_name=None, mime=None,
                            key=None, **_kw):
            app.downloads[str(key or label)] = {
                "label": label, "data": data, "file_name": file_name, "mime": mime}
            return False

        self._patch(DeltaGenerator, "file_uploader", file_uploader)
        self._patch(DeltaGenerator, "download_button", download_button)
        self._patch(st, "file_uploader", lambda *a, **k: file_uploader(None, *a, **k))
        self._patch(st, "download_button", lambda *a, **k: download_button(None, *a, **k))

        def _offline(*_a, **_k):
            raise RuntimeError("network access is disabled in golden tests")

        if self.offline:
            from qbt import boc, data, regimes
            self._patch(regimes, "load_market", _offline)
            self._patch(data, "load_yfinance", _offline)
            self._patch(data, "load_market_data", _offline)
            self._patch(boc, "fetch_series", _offline)

        st.cache_data.clear()
        cwd = os.getcwd()
        os.chdir(ROOT)
        self._undo.append(lambda: os.chdir(cwd))
        self.at = AppTest.from_file(os.path.join(ROOT, "app.py"),
                                    default_timeout=self.timeout)
        self.run("open")
        return self

    def __exit__(self, *exc) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    # -- driving --------------------------------------------------------
    def run(self, what: str = "rerun") -> "LegacyApp":
        import time
        self.downloads.clear()
        t0 = time.perf_counter()
        self.at.run()
        self.timings.append((what, time.perf_counter() - t0))
        if self.at.exception:
            raise AssertionError(f"app raised during '{what}': "
                                 + "; ".join(str(e.value) for e in self.at.exception))
        return self

    def upload(self, key: str, *names: str) -> "LegacyApp":
        self.uploads[key] = [_Upload(n, fixture(n)) for n in names]
        return self

    def widget(self, kind: str, label: Optional[str] = None, key: Optional[str] = None,
               sidebar: bool = True, nth: int = 0):
        scope = self.at.sidebar if sidebar else self.at.main
        found = [w for w in getattr(scope, kind)
                 if (key is None or w.key == key) and (label is None or w.label == label)]
        if len(found) <= nth:
            have = [(w.label, w.key) for w in getattr(scope, kind)]
            raise LookupError(f"no {kind} label={label!r} key={key!r}; have {have}")
        return found[nth]

    def set(self, kind: str, value: Any, label: Optional[str] = None,
            key: Optional[str] = None, sidebar: bool = True, nth: int = 0) -> "LegacyApp":
        self.widget(kind, label, key, sidebar, nth).set_value(value)
        return self

    def click(self, label: str, sidebar: bool = True, what: Optional[str] = None) -> "LegacyApp":
        self.widget("button", label, sidebar=sidebar).click()
        return self.run(what or f"click {label}")

    def download(self, key: str) -> bytes:
        d = self.downloads[key]["data"]
        d = d() if callable(d) else d
        return d.encode("utf-8") if isinstance(d, str) else bytes(d)


# ----------------------------------------------------------------------
# Screen transcript: everything the page shows, in order.
# ----------------------------------------------------------------------
_WIDGETS = {"button", "checkbox", "date_input", "multiselect", "number_input",
            "radio", "select_slider", "selectbox", "slider", "text_area",
            "text_input", "toggle", "button_group", "time_input", "color_picker"}
_TEXT = {"markdown", "caption", "code", "text", "title", "header", "subheader",
         "error", "warning", "info", "success", "latex", "json"}


_TAG = re.compile(r"<[^>]+>")
_DIAL = re.compile(r'<div class="dial ?(\w*)"><span class="k">(.*?)</span>'
                   r'<span class="v">(.*?)</span>(?:<div class=d>(.*?)</div>)?</div>', re.S)


def plain(markup: str) -> str:
    """Markup reduced to the words a reader sees."""
    import html as _html
    text = re.sub(r'<span class="more">.*?</span>', " ", str(markup), flags=re.S)
    text = _html.unescape(_TAG.sub(" ", text)).replace("\xa0", " ")
    return scrub(re.sub(r"\s+", " ", text).strip())


def _markdown(body: str) -> Optional[Dict[str, Any]]:
    """The app writes its own HTML for KPIs, notes, flags and headings. They
    are recorded by role and content, so a different front end that shows
    the same thing produces the same record."""
    t = str(body).strip()
    if t.startswith("<style") or plain(t) == "":
        return None
    dials = _DIAL.findall(t)
    if dials:
        items = [{"label": plain(k), "value": plain(v), "sub": plain(d), "tone": tone}
                 for tone, k, v, d in dials]
        return {"kind": "kpi", **items[0]} if len(items) == 1 else {"kind": "kpis",
                                                                    "items": items}
    for cls, role in (("runbar", "context"), ("masthead", "title"),
                      ("eyebrow", "heading"), ("side-label", "heading"),
                      ("flag", "flag"), ("note", "note"), ("hero", "hero"),
                      ("rulecard", "rule"), ("joiner", "rule")):
        if re.match(rf'<(div|details|span) class="{cls}[" ]', t):
            if role == "context":
                return {"kind": role, "items": [plain(s) for s in
                                                re.findall(r"<span class=\"(?:lead|item)\">(.*?)</span>(?=<span|</div>)", t, flags=re.S)]}
            return {"kind": role, "text": plain(t)}
    return {"kind": "text", "text": plain(t)}


def _element(el: Any) -> Optional[Dict[str, Any]]:
    kind = el.type
    if kind == "markdown":
        return _markdown(el.value)
    if kind in _TEXT:
        return {"kind": kind, "text": plain(el.value)}
    if kind in _WIDGETS:
        rec: Dict[str, Any] = {"kind": kind, "label": scrub(getattr(el, "label", ""))}
        try:
            value = el.value
        except Exception:
            value = None
        if kind != "button":
            rec["value"] = scrub(json.dumps(value, default=str))
        opts = getattr(el, "options", None)
        if opts is not None:
            opts = [str(o) for o in opts]
            rec["options"] = opts if len(opts) <= 40 else [f"{len(opts)} options",
                                                           digest("|".join(opts))]
        return rec
    if kind in ("arrow_data_frame", "dataframe", "arrow_table", "table"):
        try:
            return {"kind": "table", **table_record(el.value, index=True)}
        except Exception as exc:                       # pragma: no cover
            return {"kind": "table", "error": str(exc)}
    if kind == "plotly_chart":
        return {"kind": "chart", **chart_record(el.proto.spec)}
    if kind in ("imgs", "image"):
        return {"kind": "image"}
    if kind in ("empty", "space", "divider", "spinner"):
        return None
    return {"kind": kind}


def _walk(node: Any, out: List[Dict[str, Any]], depth: int = 0) -> None:
    children = getattr(node, "children", None)
    if children is None:
        rec = _element(node)
        if rec is not None:
            out.append(rec)
        return
    kind = getattr(node, "type", "")
    if kind in ("tab", "expander", "chat_message", "form", "status"):
        out.append({"kind": f"<{kind}>", "label": scrub(getattr(node, "label", "") or "")})
    for key in sorted(children):
        _walk(children[key], out, depth + 1)
    if kind in ("tab", "expander", "chat_message", "form", "status"):
        out.append({"kind": f"</{kind}>"})


def transcript(block: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    _walk(block, out)
    return out


def split_tabs(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Top-level tabs become named screens; anything outside them is 'page'."""
    screens: Dict[str, List[Dict[str, Any]]] = {"page": []}
    current, depth = "page", 0
    for it in items:
        if it["kind"] == "<tab>":
            depth += 1
            if depth == 1:
                current = f"tab: {it['label']}"
                screens[current] = []
                continue
        elif it["kind"] == "</tab>":
            depth -= 1
            if depth == 0:
                current = "page"
                continue
        screens[current].append(it)
    return screens


# ----------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------
def _workbook_record(data: bytes) -> Dict[str, Any]:
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(data), data_only=False)
    sheets: Dict[str, Any] = {}
    for ws in wb.worksheets:
        lines = []
        for row in ws.iter_rows(values_only=True):
            cells = []
            for v in row:
                if v is None:
                    cells.append("")
                elif isinstance(v, bool):
                    cells.append(str(v))
                elif isinstance(v, (int, float)):
                    cells.append(FLOAT_FMT % v)
                else:
                    cells.append(scrub(str(v)))
            while cells and cells[-1] == "":
                cells.pop()
            lines.append("\t".join(cells))
        while lines and lines[-1] == "":
            lines.pop()
        rec: Dict[str, Any] = {"rows": len(lines), "cols": ws.max_column,
                               "digest": digest("\n".join(lines)),
                               "charts": len(getattr(ws, "_charts", [])),
                               "images": len(getattr(ws, "_images", []))}
        if len(lines) <= 45:
            rec["text"] = lines
        else:
            rec["head"] = lines[:6]
        sheets[ws.title] = rec
    return {"sheet_order": [ws.title for ws in wb.worksheets], "sheets": sheets}


def _html_record(data: bytes) -> Dict[str, Any]:
    text = scrub(data.decode("utf-8"))
    # Tables and visible text, and the figures by what they plot.
    figures = []
    for m in re.finditer(r"Plotly\.newPlot\(\s*\"[^\"]+\",\s*(\[.*?\]),\s*(\{.*?\}),\s*\{",
                         text, flags=re.S):
        try:
            figures.append(chart_record(json.dumps(
                {"data": json.loads(m.group(1)), "layout": json.loads(m.group(2))})))
        except Exception:
            figures.append({"unparsed": digest(m.group(1))})
    body = re.sub(r"<script\b.*?</script>", "", text, flags=re.S)
    body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S)
    body = re.sub(r"<img[^>]*>", "<img>", body)
    words = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    return {"visible_text_digest": digest(words), "visible_text_chars": len(words),
            "visible_text_head": words[:400], "figures": figures,
            "images": len(re.findall(r"<img>", body))}


def export_record(app: LegacyApp, key: str) -> Dict[str, Any]:
    meta = app.downloads[key]
    data = app.download(key)
    name = meta["file_name"] or ""
    rec: Dict[str, Any] = {"label": scrub(meta["label"]), "file_name": scrub(name),
                           "mime": meta["mime"]}
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(data))
            rec.update(table_record(df, index=False))
        elif ext == ".xlsx":
            rec.update(_workbook_record(data))
        elif ext in (".html", ".htm"):
            rec.update(_html_record(data))
        elif ext in (".yaml", ".yml", ".json", ".txt"):
            text = scrub(data.decode("utf-8")).replace("\r\n", "\n")
            rec.update({"digest": digest(text), "text": text.splitlines()})
        elif ext == ".pdf":
            rec.update({"is_pdf": data[:5] == b"%PDF-",
                        "pages": len(re.findall(rb"/Type\s*/Page[^s]", data))})
        elif ext == ".pptx":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                rec["slides"] = len([n for n in z.namelist()
                                     if re.match(r"ppt/slides/slide\d+\.xml$", n)])
        else:
            rec["bytes"] = len(data)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


# ----------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------
def _num(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return None if f != f else f


def backtest_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    """Raw statistics of the run on screen, computed as app.py computes them."""
    from qbt import metrics as M
    from qbt.engine import _cash_returns
    res, bench, cfg = run["result"], run["bench"], run["cfg"]
    ppy = cfg.engine.periods_per_year
    rf = _cash_returns(res.equity.index, run.get("cash"), cfg.costs.cash_rate_pa, ppy)
    stats = M.summary(res.returns, res.equity,
                      bench.returns if bench is not None else None,
                      res.turnover, res.exposure, rf, ppy)
    out: Dict[str, Any] = {
        "label": res.label,
        "start": str(res.equity.index[0].date()), "end": str(res.equity.index[-1].date()),
        "sessions": int(len(res.equity)),
        "final_value": float(res.equity.iloc[-1]),
        "strategy": {k: _num(v) for k, v in stats.items()},
        "n_trades": int(len(res.trades)),
        "n_rebalances": int(len(res.rebalance_dates)),
        "total_costs": float(res.costs.sum()),
        "total_turnover": float(res.turnover.sum()),
    }
    if bench is not None:
        b = M.summary(bench.returns, bench.equity, None, None, None, rf, ppy)
        out["benchmark_label"] = bench.label
        out["benchmark"] = {k: _num(v) for k, v in b.items()}
    for name, r in (run.get("compare") or {}).items():
        out.setdefault("compare", {})[name] = {
            "final_value": float(r.equity.iloc[-1]), "sessions": int(len(r.equity))}
    return out


def engine_record(run: Dict[str, Any]) -> Dict[str, Any]:
    """The simulation's own outputs: value, holdings, trades."""
    res = run["result"]
    month_end = res.equity.resample("ME").last()
    month_end.index = month_end.index.strftime("%Y-%m")
    last = res.weights.iloc[-1]
    holdings = pd.DataFrame({"Instrument": list(last.index) + ["Cash"],
                             "Weight": list(last.values) + [float(res.cash_weight.iloc[-1])]})
    rec = {
        "equity_month_end": table_record(month_end.to_frame("value")),
        "equity_daily": table_record(res.equity.to_frame("value")),
        "returns_daily": table_record(res.returns.to_frame("return")),
        "weights_daily": table_record(res.weights),
        "target_weights_daily": table_record(res.target_weights),
        "holdings_last": table_record(holdings, index=False),
        "trades": table_record(res.trades, index=False),
        "rebalance_dates": digest(",".join(str(d.date()) for d in res.rebalance_dates)),
    }
    if res.shares is not None:
        rec["shares_daily"] = table_record(res.shares)
    if res.cash_flows is not None:
        rec["cash_flows"] = table_record(res.cash_flows, index=False)
    if run.get("bench") is not None:
        rec["benchmark_equity_daily"] = table_record(run["bench"].equity.to_frame("value"))
    return rec


def record(app: LegacyApp, exports: bool = True) -> Dict[str, Any]:
    """Everything a scenario locks, as plain data."""
    at = app.at
    rec: Dict[str, Any] = {}
    state = at.session_state
    run = state["run"] if "run" in state else None
    if run is not None:
        rec["metrics"] = backtest_metrics(run)
        rec["engine"] = engine_record(run)
    rec["controls"] = transcript(at.sidebar)
    rec["screens"] = split_tabs(transcript(at.main))
    if exports:
        rec["exports"] = {key: export_record(app, key) for key in sorted(app.downloads)}
    return rec


# ----------------------------------------------------------------------
# Storage and comparison
# ----------------------------------------------------------------------
PARTS = ("metrics", "engine", "controls", "screens", "exports")


def save(rec: Dict[str, Any], folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for part in PARTS:
        path = os.path.join(folder, f"{part}.json")
        if part in rec:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(rec[part], fh, indent=1, ensure_ascii=False, sort_keys=False)
                fh.write("\n")
        elif os.path.exists(path):
            os.remove(path)


def load(folder: str) -> Dict[str, Any]:
    rec = {}
    for part in PARTS:
        path = os.path.join(folder, f"{part}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                rec[part] = json.load(fh)
    return rec


def _diff(a: Any, b: Any, path: str, out: List[str], numeric: bool) -> None:
    if len(out) >= 60:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                out.append(f"{path}/{k}: missing")
        for k in b:
            if k not in a:
                out.append(f"{path}/{k}: unexpected")
        for k in a:
            if k in b:
                _diff(a[k], b[k], f"{path}/{k}", out, numeric)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items expected, {len(b)} found")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, numeric)
    elif (numeric and isinstance(a, (int, float)) and isinstance(b, (int, float))
          and not isinstance(a, bool) and not isinstance(b, bool)):
        if not np.isclose(a, b, rtol=RTOL, atol=ATOL):
            out.append(f"{path}: expected {a!r}, found {b!r}")
    elif a != b:
        out.append(f"{path}: expected {str(a)[:140]!r}, found {str(b)[:140]!r}")


def compare(expected: Dict[str, Any], actual: Dict[str, Any],
            parts: Tuple[str, ...] = PARTS) -> List[str]:
    """Every difference between two records, as readable lines."""
    out: List[str] = []
    for part in parts:
        if part not in expected:
            continue
        if part not in actual:
            out.append(f"{part}: not produced")
            continue
        # JSON round-trip so tuples, numpy scalars and dates compare as stored.
        act = json.loads(json.dumps(actual[part], default=str))
        _diff(expected[part], act, part, out, numeric=(part == "metrics"))
    return out
