"""HTTP API of the redesigned app, and host of its built front end.

    uvicorn server.app:app --port 8000

Endpoints (all under /api):

    GET  /meta                         models, presets, option lists, defaults
    POST /uploads/{kind}               kind: prices | exog | weights
    POST /runs                         {spec, files} -> run summary
    GET  /runs/{id}                    run summary (404 once evicted)
    GET  /runs/{id}/screens/{screen}   a screen's blocks

Everything a screen shows is computed in `server/views`; this file only
routes. The Streamlit app (`app.py`) is untouched and still runs on its own.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, fields
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qbt import presets as PRESETS
from qbt import price_import as PI
from qbt.config import REBALANCE_RULES
from qbt.strategies import REGISTRY

from . import store
from .pipeline import (BENCH_MODES, BENCH_STRAT, BLEND, RunError, RunSpec,
                       read_exog, read_weights)
from .serialize import screen_json
from .views import results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "web", "dist")
LEGACY_URL = os.environ.get("QBT_LEGACY_URL", "http://localhost:8501")

# Screens rebuilt so far: name -> builder. Add one line per screen.
SCREENS = {
    "results": results.build,
}
# Every Backtest tab, in the app's order; the front end shows the ones not
# in SCREENS as "open in the classic app".
BACKTEST_TABS = ["Results", "Compare", "Signals", "Positions", "Attribution",
                 "Tax", "Robustness", "Data", "Builder", "Export"]

app = FastAPI(title="Quant Backtest Studio", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=800)


def _models() -> List[Dict[str, Any]]:
    out = []
    for key, s in sorted(REGISTRY.items(), key=lambda kv: kv[1].label):
        out.append({
            "key": key, "label": s.label, "description": s.description,
            "needs_exog": bool(s.needs_exog), "needs_cash": bool(s.needs_cash),
            "params": [{"key": p.key, "label": p.label, "kind": p.kind,
                        "min": p.min, "max": p.max, "step": p.step,
                        "default": p.default, "choices": list(p.choices or []),
                        "help": p.help or ""} for p in s.params]})
    return out


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    return {
        "models": _models(),
        "presets": [{"name": n, "tickers": PRESETS.get(n)["tickers"],
                     "benchmark": PRESETS.get(n).get("benchmark"),
                     "cash": PRESETS.get(n).get("cash"),
                     "note": PRESETS.get(n).get("note", "")} for n in PRESETS.names()],
        "rebalance": REBALANCE_RULES,
        "bench_modes": BENCH_MODES,
        "bench_special": {"blend": BLEND, "strategy": BENCH_STRAT},
        "file_conventions": PI.CONVENTIONS,
        "defaults": asdict(RunSpec()),
        "tabs": [{"name": t, "screen": t.lower() if t.lower() in SCREENS else None}
                 for t in BACKTEST_TABS],
        "legacy_url": LEGACY_URL,
    }


@app.post("/api/uploads/{kind}")
async def upload(kind: str, files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if kind not in ("prices", "exog", "weights"):
        raise HTTPException(404, "Unknown upload kind.")
    held = []
    for f in files:
        content = await f.read()
        held.append((store.put_upload(f.filename or "file", content), f.filename or "file", content))
    ids = [h[0] for h in held]
    pairs = [(h[1], h[2]) for h in held]
    out: Dict[str, Any] = {"ids": ids, "names": [h[1] for h in held]}
    try:
        if kind == "prices":
            imp = PI.read_files(pairs)
            out.update({
                "instruments": imp.instruments,
                "fields": [PI.FIELD_LABELS[f] for f in PI.FIELDS if imp.has(f)],
                "has_open": imp.has("open"), "has_volume": imp.has("volume"),
                "sources": len(imp.sources),
                "warnings": list(imp.warnings[:10]),
                "convention": (PI.resolve_convention(imp, "auto") if imp.instruments else None),
                "error": None if imp.instruments else
                "The file could not be read: no prices were found. " + " ".join(imp.warnings[:3]),
            })
        elif kind == "exog":
            raw, errors = read_exog(pairs)
            out.update({"columns": [str(c) for c in raw.columns] if raw is not None else [],
                        "start": str(raw.index.min().date()) if raw is not None else None,
                        "end": str(raw.index.max().date()) if raw is not None else None,
                        "error": "; ".join(errors) or None})
        else:
            w = read_weights(pairs[0], None)
            out.update({"columns": [str(c) for c in w.columns], "rows": int(len(w)),
                        "error": None})
    except Exception as exc:
        out["error"] = f"Could not read the file: {exc}"
    return out


def _summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    view, run = entry["view"], entry["run"]
    return {"id": entry["id"], "cached": entry["cached"],
            "seconds": round(entry["seconds"], 3), "stamp": run["stamp"],
            "label": view.cfg.label, "context": view.context(),
            "kpis": results.kpis(view), "notices": run.get("notices", []),
            "screens": list(SCREENS)}


@app.post("/api/runs")
def create_run(body: Dict[str, Any]) -> Dict[str, Any]:
    known = {f.name for f in fields(RunSpec)}
    given = body.get("spec") or {}
    unknown = sorted(set(given) - known)
    if unknown:
        raise HTTPException(422, f"Unknown controls: {', '.join(unknown)}")
    spec = RunSpec(**given)
    t0 = time.perf_counter()
    try:
        entry = store.run(spec, body.get("files") or {})
    except RunError as exc:
        raise HTTPException(422, str(exc))
    except KeyError as exc:
        raise HTTPException(410, str(exc).strip("'\""))
    except Exception as exc:
        raise HTTPException(422, f"The backtest stopped: {exc}")
    out = _summary(entry)
    out["request_seconds"] = round(time.perf_counter() - t0, 3)
    return out


@app.get("/api/runs/{rid}")
def read_run(rid: str) -> Dict[str, Any]:
    entry = store.get_run(rid)
    if entry is None:
        raise HTTPException(404, "This run is no longer held. Run it again.")
    return _summary(dict(entry, cached=True))


@app.get("/api/runs/{rid}/screens/{name}")
def read_screen(rid: str, name: str) -> JSONResponse:
    if name not in SCREENS:
        raise HTTPException(404, "This screen has not been rebuilt yet.")
    blocks = store.screen(rid, name, SCREENS[name], screen_json)
    if blocks is None:
        raise HTTPException(404, "This run is no longer held. Run it again.")
    return JSONResponse({"blocks": blocks})


# ----------------------------------------------------------------------
# The app itself: the classic screen logic, rendered for the new front end
# ----------------------------------------------------------------------
import datetime as _dt
import secrets as _secrets
from collections import OrderedDict as _OrderedDict

from fastapi import Request, Response

from . import classic, present, shim
from .views.common import RunView

_sessions: "_OrderedDict[str, shim.Session]" = _OrderedDict()
MAX_SESSIONS = 12
COOKIE = "qbt_sid"
# Backtest > Results is drawn by its own screen (server/views/results.py),
# so the classic tab body is not run for it.
DEDICATED = {"10:Results"}


def _session(request: Request, response: Response) -> shim.Session:
    sid = request.cookies.get(COOKIE)
    if not sid or sid not in _sessions:
        sid = _secrets.token_urlsafe(18)
        _sessions[sid] = shim.Session()
        response.set_cookie(COOKIE, sid, httponly=True, samesite="lax", max_age=7 * 24 * 3600)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    _sessions.move_to_end(sid)
    return _sessions[sid]


def _locked(session: shim.Session) -> bool:
    return bool(os.environ.get("QBT_PASSWORD")) and not session.state.get("auth_ok")


def _apply(session: shim.Session, body: Dict[str, Any]) -> None:
    """Puts what the browser changed into the session, using the widgets of
    the last render to turn positions back into values."""
    for ev in body.get("events") or []:
        wid, value = ev.get("id"), ev.get("value")
        meta = session.registry.get(wid)
        if meta is None:
            continue
        kind, opts = meta["kind"], meta.get("options")
        if kind == "data_editor":
            cols = meta.get("columns") or []
            session.auto["editor:" + wid] = [dict(zip(cols, row)) for row in (value or [])]
            continue
        if kind == "multiselect":
            value = [opts[i] for i in (value or []) if isinstance(i, int) and 0 <= i < len(opts)]
        elif opts is not None:
            value = opts[value] if isinstance(value, int) and 0 <= value < len(opts) else None
        elif kind == "date_input":
            try:
                value = _dt.date.fromisoformat(str(value)[:10])
            except ValueError:
                continue
        elif kind in ("checkbox", "toggle"):
            value = bool(value)
        if meta.get("key") is not None:
            session.state[meta["key"]] = value
        else:
            session.auto[wid] = value
    session.clicked = {c for c in (body.get("clicks") or []) if c in session.registry}
    session.submitted = {k: str(v) for k, v in (body.get("submits") or {}).items()
                         if k in session.registry}


@app.post("/api/render")
def render(body: Dict[str, Any], request: Request, response: Response) -> Dict[str, Any]:
    session = _session(request, response)
    t0 = time.perf_counter()
    with session.lock:
        _apply(session, body)
        try:
            rec = classic.render(session, tab=body.get("tab"), skip=DEDICATED)
        except Exception as exc:                       # a screen that fails must say so
            raise HTTPException(500, f"This view could not be built: {type(exc).__name__}: {exc}")
        out = present.page(rec)
        tabs = next((b for b in out["main"] if b["kind"] == "tabs"), None)
        run = session.state.get("run")
        if tabs and f"{len(tabs['labels'])}:{tabs['active']}" in DEDICATED and run is not None:
            key = ("results-screen", id(run))
            if key not in session.memo:
                session.memo[key] = screen_json(results.build(RunView(run)))
            tabs["blocks"] = [{"kind": "results", "blocks": session.memo[key]}]
        out["seconds"] = round(time.perf_counter() - t0, 3)
        out["legacy_url"] = LEGACY_URL
        return out


@app.post("/api/files/{wid:path}")
async def put_files(wid: str, request: Request, response: Response,
                    files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    session = _session(request, response)
    if _locked(session):
        raise HTTPException(401, "Sign in first.")
    held = [shim.Upload(f.filename or "file", await f.read()) for f in files]
    with session.lock:
        session.uploads[wid] = held
    return {"names": [h.name for h in held]}


@app.delete("/api/files/{wid:path}")
def drop_files(wid: str, request: Request, response: Response) -> Dict[str, Any]:
    session = _session(request, response)
    with session.lock:
        session.uploads.pop(wid, None)
    return {"names": []}


@app.get("/api/download/{wid:path}")
def download(wid: str, request: Request, response: Response) -> Response:
    session = _session(request, response)
    if _locked(session):
        raise HTTPException(401, "Sign in first.")
    with session.lock:
        item = session.downloads.get(wid)
        if item is None:
            raise HTTPException(404, "This file is no longer available. Reload the view and try again.")
        data = item["data"]
        try:
            data = data() if callable(data) else data
        except Exception as exc:
            raise HTTPException(500, f"The file could not be built: {exc}")
    if isinstance(data, str):
        data = data.encode("utf-8")
    name = (item["file_name"] or "download").replace('"', "")
    return Response(content=bytes(data), media_type=item["mime"] or "application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# The built front end, when present (`npm run build` in web/).
if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        full = os.path.realpath(os.path.join(DIST, path))
        inside = full.startswith(os.path.realpath(DIST) + os.sep)
        if path and inside and os.path.isfile(full):
            return FileResponse(full)
        return FileResponse(os.path.join(DIST, "index.html"))
