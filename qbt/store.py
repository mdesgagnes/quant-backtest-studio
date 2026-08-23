"""Saved strategy presets.

Stores named configurations so a setup can be recalled by name instead of
being exported to YAML and re-uploaded every time.

**Where these live, and why it matters.** Presets are written to a JSON file
on the machine running the app. Locally that is permanent. On Streamlit
Community Cloud it is not: the container filesystem is rebuilt on every
redeploy, reboot and wake-from-sleep, and everything saved here goes with
it. That is a property of the host, not a bug to be fixed in this file --
there is no writable persistent disk on the free tier.

So the store is deliberately paired with a backup: `export_all` produces one
file containing every preset, and `import_all` restores it. The interface
surfaces both prominently rather than burying them, because a save button
that silently loses work on the next deploy is worse than no save button.
Anything worth keeping should be exported and committed to the repository,
where it survives.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

STORE_DIR = os.environ.get("QBT_PRESET_DIR", ".qbt_presets")
STORE_FILE = os.path.join(STORE_DIR, "presets.json")
SCHEMA = 1
MAX_PRESETS = 200
MAX_NAME = 60


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _safe_name(name: str) -> str:
    """Trimmed, bounded, and free of characters that break a filename."""
    cleaned = "".join(c for c in str(name) if c.isprintable()).strip()
    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    return cleaned[:MAX_NAME]


def _read() -> Dict[str, Any]:
    if not os.path.exists(STORE_FILE):
        return {"schema": SCHEMA, "presets": {}}
    try:
        with open(STORE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "presets" not in data:
            return {"schema": SCHEMA, "presets": {}}
        return data
    except Exception:
        # A corrupt store must not take the application down with it.
        return {"schema": SCHEMA, "presets": {}}


def _write(data: Dict[str, Any]) -> bool:
    try:
        os.makedirs(STORE_DIR, exist_ok=True)
        tmp = STORE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, STORE_FILE)     # atomic, so a crash cannot truncate
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
def list_presets() -> List[Dict[str, Any]]:
    """Saved presets, most recent first."""
    data = _read()
    rows = []
    for name, entry in data.get("presets", {}).items():
        rows.append({
            "name": name,
            "saved": entry.get("saved", ""),
            "label": entry.get("label", ""),
            "strategy": entry.get("strategy", ""),
            "note": entry.get("note", ""),
        })
    return sorted(rows, key=lambda r: r.get("saved", ""), reverse=True)


def save(name: str, payload: Dict[str, Any], note: str = "") -> Tuple[bool, str]:
    """Stores one preset. Returns (ok, message)."""
    name = _safe_name(name)
    if not name:
        return False, "Give the preset a name."
    data = _read()
    presets = data.setdefault("presets", {})
    if name not in presets and len(presets) >= MAX_PRESETS:
        return False, f"The store holds {MAX_PRESETS} presets. Delete one first."

    existed = name in presets
    presets[name] = {
        "saved": _now(),
        "label": payload.get("label", ""),
        "strategy": payload.get("strategy_name", ""),
        "note": str(note)[:400],
        "payload": payload,
    }
    if not _write(data):
        return False, ("Could not write to disk. On a read-only host, use "
                       "Export instead.")
    return True, (f"Updated \u201c{name}\u201d." if existed
                  else f"Saved \u201c{name}\u201d.")


def load(name: str) -> Optional[Dict[str, Any]]:
    # Names are sanitised on save, so lookups must be sanitised too or a
    # preset saved as "60/40" can never be loaded or deleted again.
    name = _safe_name(name)
    entry = _read().get("presets", {}).get(name)
    return entry.get("payload") if entry else None


def delete(name: str) -> Tuple[bool, str]:
    name = _safe_name(name)
    data = _read()
    if name not in data.get("presets", {}):
        return False, "No preset by that name."
    data["presets"].pop(name)
    if not _write(data):
        return False, "Could not write to disk."
    return True, f"Deleted \u201c{name}\u201d."


def rename(old: str, new: str) -> Tuple[bool, str]:
    new = _safe_name(new)
    if not new:
        return False, "Give the preset a name."
    old = _safe_name(old)
    data = _read()
    presets = data.get("presets", {})
    if old not in presets:
        return False, "No preset by that name."
    if new in presets and new != old:
        return False, f"\u201c{new}\u201d already exists."
    presets[new] = presets.pop(old)
    presets[new]["saved"] = _now()
    if not _write(data):
        return False, "Could not write to disk."
    return True, f"Renamed to \u201c{new}\u201d."


# ----------------------------------------------------------------------
def export_all() -> str:
    """Every preset in one file. This is the durable copy."""
    data = _read()
    data["exported"] = _now()
    return json.dumps(data, indent=1, ensure_ascii=False)


def import_all(text: str, replace: bool = False) -> Tuple[bool, str]:
    """Restores from an exported file.

    Merges by default: an import cannot silently destroy presets that only
    exist on this machine. `replace` is offered explicitly for the case
    where the file is meant to be authoritative.
    """
    try:
        incoming = json.loads(text)
    except Exception as exc:
        return False, f"Not a valid preset file: {exc}"
    if not isinstance(incoming, dict) or "presets" not in incoming:
        return False, "That file does not contain presets."

    new = incoming.get("presets", {})
    if not isinstance(new, dict):
        return False, "That file does not contain presets."

    data = _read()
    if replace:
        data["presets"] = new
        msg = f"Replaced the store with {len(new)} preset(s)."
    else:
        existing = data.setdefault("presets", {})
        added = sum(1 for k in new if k not in existing)
        updated = len(new) - added
        existing.update(new)
        msg = f"Imported {len(new)} preset(s): {added} new, {updated} updated."
    if not _write(data):
        return False, "Could not write to disk."
    return True, msg


def store_location() -> str:
    return os.path.abspath(STORE_FILE)


def is_ephemeral() -> bool:
    """Whether this looks like a host that wipes the disk on redeploy.

    Streamlit Community Cloud mounts the repository under /mount/src. It is
    a heuristic, and it only drives a warning, so a false positive costs
    nothing but a sentence of caution.
    """
    return os.path.abspath(".").startswith("/mount/src")
