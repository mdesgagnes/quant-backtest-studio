"""In-memory state of the service: uploads, market-data cache, finished runs.

One process, one user at a time, like the Streamlit app it replaces. Nothing
is written to disk.

- An upload is kept by the digest of its bytes, so the same file is parsed
  once however often it is sent.
- A finished run is kept under the digest of its controls and files. Asking
  for the identical run again returns it without simulating.
- Yahoo Finance downloads are reused for an hour, as before; a cached run on
  Yahoo data expires with them.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from qbt.data import load_market_data

from .pipeline import RunFiles, RunSpec, execute
from .views.common import RunView

YAHOO_TTL = 3600.0
MAX_RUNS = 8
MAX_UPLOADS = 40

_lock = threading.RLock()
_uploads: "OrderedDict[str, Tuple[str, bytes]]" = OrderedDict()
_runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_market: Dict[tuple, Tuple[float, Any]] = {}


def digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
        h.update(b"\0")
    return h.hexdigest()[:16]


def put_upload(name: str, content: bytes) -> str:
    uid = digest(name.encode("utf-8"), content)
    with _lock:
        _uploads[uid] = (name, content)
        _uploads.move_to_end(uid)
        while len(_uploads) > MAX_UPLOADS:
            _uploads.popitem(last=False)
    return uid


def get_upload(uid: str) -> Tuple[str, bytes]:
    with _lock:
        if uid not in _uploads:
            raise KeyError(f"Upload {uid} is no longer held. Choose the file again.")
        _uploads.move_to_end(uid)
        return _uploads[uid]


def fetch_market(tickers, start, end, adjusted, want_open, want_div, want_ohlc=False):
    """Yahoo Finance prices, reused for an hour for the same request."""
    key = (tuple(tickers), start, end, adjusted, want_open, want_div, want_ohlc)
    now = time.time()
    with _lock:
        hit = _market.get(key)
        if hit and now - hit[0] < YAHOO_TTL:
            return hit[1]
    data = load_market_data(list(tickers), start, end, adjusted=adjusted,
                            want_open=want_open, want_dividends=want_div,
                            want_ohlc=want_ohlc)
    with _lock:
        _market[key] = (now, data)
        for k in [k for k, v in _market.items() if now - v[0] >= YAHOO_TTL]:
            _market.pop(k, None)
    return data


def run_id(spec: RunSpec, file_ids: Dict[str, Any]) -> str:
    payload = json.dumps({"spec": asdict(spec), "files": file_ids},
                         sort_keys=True, default=str)
    return digest(payload.encode("utf-8"))


def run(spec: RunSpec, file_ids: Dict[str, Any],
        executor: Callable[..., Dict[str, Any]] = execute) -> Dict[str, Any]:
    """Runs, or returns the identical run already held.

    Returns the store entry: id, run, view, seconds, cached."""
    rid = run_id(spec, file_ids)
    now = time.time()
    with _lock:
        hit = _runs.get(rid)
        if hit and (spec.source != "yfinance" or now - hit["created"] < YAHOO_TTL):
            _runs.move_to_end(rid)
            return dict(hit, cached=True)
    files = RunFiles(
        prices=[get_upload(u) for u in file_ids.get("prices", [])],
        exog=[get_upload(u) for u in file_ids.get("exog", [])],
        weights=get_upload(file_ids["weights"]) if file_ids.get("weights") else None)
    t0 = time.perf_counter()
    result = executor(spec, files, fetch_market=fetch_market)
    entry = {"id": rid, "run": result, "view": RunView(result), "created": now,
             "seconds": time.perf_counter() - t0, "screens": {}, "cached": False}
    with _lock:
        _runs[rid] = entry
        _runs.move_to_end(rid)
        while len(_runs) > MAX_RUNS:
            _runs.popitem(last=False)
    return entry


def get_run(rid: str) -> Optional[Dict[str, Any]]:
    with _lock:
        hit = _runs.get(rid)
        if hit:
            _runs.move_to_end(rid)
        return hit


def screen(rid: str, name: str, build: Callable[[RunView], List[Dict[str, Any]]],
           to_json: Callable[[List[Dict[str, Any]]], Any]) -> Optional[Any]:
    """A screen of a held run as JSON, built once."""
    entry = get_run(rid)
    if entry is None:
        return None
    with _lock:
        if name not in entry["screens"]:
            entry["screens"][name] = to_json(build(entry["view"]))
        return entry["screens"][name]
