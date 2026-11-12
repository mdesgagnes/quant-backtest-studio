"""Drives the new server's rendering the way harness.LegacyApp drives
Streamlit, so `scenarios.py` runs unchanged against both, and reduces what
it renders to the same record format."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import harness as H                                   # noqa: E402
from server import classic, shim                      # noqa: E402

_CONTAINERS = {"expander": "expander", "tab": "tab", "chat_message": "chat_message",
               "form": "form"}
_WIDGETS = {"selectbox", "radio", "multiselect", "number_input", "slider", "text_input",
            "text_area", "date_input", "checkbox", "toggle", "button", "button_group",
            "select_slider"}


def _frame(data: Any):
    return getattr(data, "data", data)


def node_records(node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    kind = node["kind"]
    if kind == "markdown":
        rec = H._markdown(node["body"])
        if rec is not None:
            out.append(rec)
    elif kind in ("caption", "code", "error", "warning", "info", "success"):
        out.append({"kind": kind, "text": H.plain(node["body"])})
    elif kind in _WIDGETS:
        rec: Dict[str, Any] = {"kind": kind, "label": H.scrub(node["label"])}
        if kind != "button":
            rec["value"] = H.scrub(json.dumps(node["value"], default=str))
        if node.get("option_labels") is not None:
            opts = [str(o) for o in node["option_labels"]]
            rec["options"] = opts if len(opts) <= 40 else [f"{len(opts)} options",
                                                           H.digest("|".join(opts))]
        elif kind == "slider":
            rec["options"] = []
        out.append(rec)
    elif kind in ("dataframe", "data_editor"):
        out.append({"kind": "table", **H.table_record(
            _frame(node["data"] if kind == "dataframe" else node["frame"]), index=True)})
    elif kind == "plotly_chart":
        out.append({"kind": "chart", **H.chart_record(node["figure"].to_json())})
    elif kind == "image":
        out.append({"kind": "image"})
    elif kind == "iframe":
        out.append({"kind": "iframe"})
    elif kind in _CONTAINERS:
        out.append({"kind": f"<{kind}>", "label": H.scrub(node.get("label", "") or "")})
        for child in node.get("children", []):
            node_records(child, out)
        out.append({"kind": f"</{kind}>"})
    else:                                   # root, row, column, container, tabs, download, upload
        for child in node.get("children", []):
            node_records(child, out)


def transcript(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    node_records(root, out)
    return out


class ShimApp:
    """The new server, driven in-process. Same surface as harness.LegacyApp."""

    def __init__(self, offline: bool = True):
        self.session = shim.Session()
        self.timings: List[Tuple[str, float]] = []
        self.rec: Optional[shim.Recorder] = None
        self.offline = offline
        self._undo: List[Any] = []

    def __enter__(self) -> "ShimApp":
        if self.offline:
            def _offline(*_a, **_k):
                raise RuntimeError("network access is disabled in golden tests")
            from qbt import boc, data, regimes
            for mod, name in ((regimes, "load_market"), (data, "load_yfinance"),
                              (data, "load_market_data"), (boc, "fetch_series")):
                old = getattr(mod, name)
                setattr(mod, name, _offline)
                self._undo.append((mod, name, old))
        shim.cache_data.clear()
        cwd = os.getcwd()
        os.chdir(H.ROOT)
        self._undo.append((os, "chdir", cwd))
        self.run("open")
        return self

    def __exit__(self, *exc) -> None:
        for mod, name, old in reversed(self._undo):
            if name == "chdir":
                os.chdir(old)
            else:
                setattr(mod, name, old)

    @property
    def downloads(self) -> Dict[str, Dict[str, Any]]:
        return self.session.downloads

    def run(self, what: str = "rerun") -> "ShimApp":
        self.session.downloads.clear()
        t0 = time.perf_counter()
        self.rec = classic.render(self.session, all_tabs=True)
        self.timings.append((what, time.perf_counter() - t0))
        return self

    def upload(self, key: str, *names: str) -> "ShimApp":
        self.session.uploads[key] = [shim.Upload(n, H.fixture(n)) for n in names]
        return self

    def _find(self, kind: str, label: Optional[str], key: Optional[str], sidebar: bool, nth: int):
        found: List[Dict[str, Any]] = []

        def walk(n):
            if n["kind"] == kind and (label is None or n.get("label") == label):
                reg = self.rec.registry.get(n.get("id"), {})
                if key is None or reg.get("key") == key:
                    found.append(n)
            for c in n.get("children", []):
                walk(c)
        walk(self.rec.sidebar if sidebar else self.rec.main)
        if len(found) <= nth:
            raise LookupError(f"no {kind} label={label!r} key={key!r}")
        return found[nth]

    def set(self, kind: str, value: Any, label: Optional[str] = None, key: Optional[str] = None,
            sidebar: bool = True, nth: int = 0) -> "ShimApp":
        node = self._find(kind, label, key, sidebar, nth)
        reg = self.rec.registry[node["id"]]
        if reg["key"] is not None:
            self.session.state[reg["key"]] = value
        else:
            self.session.auto[node["id"]] = value
        return self

    def click(self, label: str, sidebar: bool = True, what: Optional[str] = None) -> "ShimApp":
        node = self._find("button", label, None, sidebar, 0)
        self.session.clicked.add(node["id"])
        return self.run(what or f"click {label}")

    def download(self, key: str) -> bytes:
        d = self.session.downloads[key]["data"]
        d = d() if callable(d) else d
        return d.encode("utf-8") if isinstance(d, str) else bytes(d)


def record(app: ShimApp) -> Dict[str, Any]:
    rec: Dict[str, Any] = {}
    run = app.session.state.get("run")
    if run is not None:
        rec["metrics"] = H.backtest_metrics(run)
        rec["engine"] = H.engine_record(run)
    rec["controls"] = transcript(app.rec.sidebar)
    rec["screens"] = H.split_tabs(transcript(app.rec.main))
    rec["exports"] = {key: H.export_record(app, key) for key in sorted(app.downloads)}
    return rec
