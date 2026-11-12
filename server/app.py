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
