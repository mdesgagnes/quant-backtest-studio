"""The research ledger: hypotheses, every experiment run under them, and the
strategies promoted out of them.

Why this exists: the single biggest source of false discoveries in
systematic research is not a bug in a backtest, it is the number of backtests
nobody wrote down. A Sharpe of 1.2 found on the first try and the same Sharpe
found as the best of 80 variants are different facts. The ledger records each
experiment against the hypothesis it tests, so `n_trials` for the deflated
Sharpe ratio is a count the system holds rather than a number someone
remembers.

Storage is one SQLite file (stdlib, atomic, safe for the UI and the MCP server
to write at once). Location: `$QBT_LAB_DIR` or `.qbt_lab/` in the project
folder. Like the preset store, it lives on the machine running the app: on
a host with an ephemeral disk, export it (`export_json`) before it is lost.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

LAB_DIR_ENV = "QBT_LAB_DIR"
STATUSES = ["idea", "testing", "validated", "rejected", "promoted", "retired"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL, statement TEXT DEFAULT '', rationale TEXT DEFAULT '',
  universe TEXT DEFAULT '', status TEXT DEFAULT 'idea',
  created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS experiments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hypothesis_id INTEGER, kind TEXT, name TEXT, config_hash TEXT, data_hash TEXT,
  n_variants INTEGER DEFAULT 1, run_dir TEXT DEFAULT '',
  metrics TEXT DEFAULT '{}', verdict TEXT DEFAULT '', note TEXT DEFAULT '',
  created TEXT,
  FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(id));
CREATE TABLE IF NOT EXISTS strategies(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL, hypothesis_id INTEGER, strategy_key TEXT,
  params TEXT, engine TEXT, costs TEXT, tickers TEXT, benchmark TEXT,
  registered TEXT, baseline TEXT, status TEXT DEFAULT 'active', note TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS ix_exp_h ON experiments(hypothesis_id);
"""


def lab_dir() -> str:
    # Anchored to the project folder, not the working directory: the app can
    # be launched from anywhere, and the ledger must be the same one each time.
    d = os.environ.get(LAB_DIR_ENV) or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".qbt_lab")
    os.makedirs(d, exist_ok=True)
    return d


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _db():
    con = sqlite3.connect(os.path.join(lab_dir(), "lab.db"), timeout=15)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _rows(cur) -> List[Dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


def config_hash(obj: Any) -> str:
    """Stable hash of a JSON-able config: key order and whitespace do not matter."""
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# Hypotheses
# ----------------------------------------------------------------------
def add_hypothesis(title: str, statement: str = "", rationale: str = "",
                   universe: str = "", status: str = "idea") -> int:
    title = (title or "").strip()
    if not title:
        raise ValueError("A hypothesis needs a title.")
    if status not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}.")
    with _db() as con:
        cur = con.execute(
            "INSERT INTO hypotheses(title,statement,rationale,universe,status,created,updated)"
            " VALUES(?,?,?,?,?,?,?)",
            (title[:200], statement[:2000], rationale[:2000], universe[:400], status, _now(), _now()))
        return int(cur.lastrowid)


def update_hypothesis(hid: int, **fields) -> None:
    allowed = {"title", "statement", "rationale", "universe", "status"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if "status" in sets and sets["status"] not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}.")
    if not sets:
        return
    with _db() as con:
        con.execute(f"UPDATE hypotheses SET {', '.join(k + '=?' for k in sets)}, updated=? WHERE id=?",
                    (*sets.values(), _now(), hid))


def list_hypotheses() -> List[Dict[str, Any]]:
    with _db() as con:
        rows = _rows(con.execute(
            "SELECT h.*, COUNT(e.id) AS n_experiments, COALESCE(SUM(e.n_variants),0) AS n_trials "
            "FROM hypotheses h LEFT JOIN experiments e ON e.hypothesis_id=h.id "
            "GROUP BY h.id ORDER BY h.updated DESC"))
    return rows


def get_hypothesis(hid: int) -> Optional[Dict[str, Any]]:
    with _db() as con:
        r = con.execute("SELECT * FROM hypotheses WHERE id=?", (hid,)).fetchone()
        return dict(r) if r else None


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def log_experiment(hypothesis_id: Optional[int], kind: str, name: str, config: Any,
                   data_hash: str = "", n_variants: int = 1, run_dir: str = "",
                   metrics: Optional[Dict[str, Any]] = None, verdict: str = "",
                   note: str = "") -> int:
    """Records one experiment. `n_variants` is how many configurations were
    effectively tried (a 40-point sweep logs 40, a factor screen logs the
    number of factors)."""
    ch = config_hash(config)
    with _db() as con:
        cur = con.execute(
            "INSERT INTO experiments(hypothesis_id,kind,name,config_hash,data_hash,n_variants,"
            "run_dir,metrics,verdict,note,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (hypothesis_id, kind, name[:200], ch, data_hash, max(1, int(n_variants)), run_dir,
             json.dumps(_plain(metrics or {})), verdict, note[:1000], _now()))
        if hypothesis_id:
            con.execute("UPDATE hypotheses SET updated=?, status=CASE WHEN status='idea' "
                        "THEN 'testing' ELSE status END WHERE id=?", (_now(), hypothesis_id))
        return int(cur.lastrowid)


def list_experiments(hypothesis_id: Optional[int] = None) -> List[Dict[str, Any]]:
    with _db() as con:
        if hypothesis_id is None:
            cur = con.execute("SELECT * FROM experiments ORDER BY id DESC")
        else:
            cur = con.execute("SELECT * FROM experiments WHERE hypothesis_id=? ORDER BY id DESC",
                              (hypothesis_id,))
        rows = _rows(cur)
    for r in rows:
        try:
            r["metrics"] = json.loads(r["metrics"] or "{}")
        except ValueError:
            r["metrics"] = {}
    return rows


def trial_count(hypothesis_id: Optional[int]) -> int:
    """Distinct configurations tried under a hypothesis. Re-running the
    identical config does not inflate it; a sweep counts every point."""
    if not hypothesis_id:
        return 1
    with _db() as con:
        rows = con.execute("SELECT config_hash, MAX(n_variants) FROM experiments "
                           "WHERE hypothesis_id=? GROUP BY config_hash", (hypothesis_id,)).fetchall()
    return max(1, sum(int(r[1]) for r in rows))


def _plain(o: Any) -> Any:
    """JSON-safe copy: numpy scalars, NaN and inf become plain values or None."""
    import math
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:
            return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ----------------------------------------------------------------------
# Strategies promoted for monitoring
# ----------------------------------------------------------------------
def register_strategy(name: str, spec: Dict[str, Any], baseline: Dict[str, Any],
                      hypothesis_id: Optional[int] = None, note: str = "") -> int:
    """Freezes a strategy as it stood on the registration date. Everything
    after that date is out-of-sample for it: the decay monitor compares the
    forward record with `baseline` (the in-sample statistics at registration)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("A registered strategy needs a name.")
    with _db() as con:
        if con.execute("SELECT 1 FROM strategies WHERE name=?", (name,)).fetchone():
            raise ValueError(f"'{name}' is already registered. Retire it or pick another name.")
        cur = con.execute(
            "INSERT INTO strategies(name,hypothesis_id,strategy_key,params,engine,costs,tickers,"
            "benchmark,registered,baseline,note) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (name[:80], hypothesis_id, spec["strategy_key"], json.dumps(spec.get("params", {})),
             json.dumps(spec.get("engine", {})), json.dumps(spec.get("costs", {})),
             json.dumps(spec.get("tickers", [])), spec.get("benchmark") or "",
             spec.get("registered") or time.strftime("%Y-%m-%d"), json.dumps(_plain(baseline)), note[:500]))
        if hypothesis_id:
            con.execute("UPDATE hypotheses SET status='promoted', updated=? WHERE id=?", (_now(), hypothesis_id))
        return int(cur.lastrowid)


def list_strategies(active_only: bool = False) -> List[Dict[str, Any]]:
    with _db() as con:
        cur = con.execute("SELECT * FROM strategies" + (" WHERE status='active'" if active_only else "")
                          + " ORDER BY id DESC")
        rows = _rows(cur)
    for r in rows:
        for k in ("params", "engine", "costs", "tickers", "baseline"):
            try:
                r[k] = json.loads(r[k] or "null")
            except ValueError:
                r[k] = None
    return rows


def retire_strategy(name: str) -> None:
    with _db() as con:
        con.execute("UPDATE strategies SET status='retired' WHERE name=?", (name,))


# ----------------------------------------------------------------------
def export_json() -> str:
    return json.dumps({"hypotheses": list_hypotheses(), "experiments": list_experiments(),
                       "strategies": list_strategies()}, indent=1, default=str)
