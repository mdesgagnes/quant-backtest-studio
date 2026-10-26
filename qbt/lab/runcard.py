"""Run cards: a result that can be checked, reproduced and audited.

A run card is a directory:

    run_card.json   canonical record (config, data hash, metrics, warnings, manifest)
    run_card.md     the same, readable
    config.json     exactly what produced the run
    artifacts/      equity.csv, returns.csv, weights.csv, trades.csv, metrics.csv,
                    prices.csv.gz (the frozen price snapshot), validation.json

Three guarantees, each with a function that tests it:

- `verify_run_card`     every artifact still matches its SHA-256, and every
                        metric quoted in the card matches its cell in metrics.csv.
- `reproduce_run_card`  re-running the saved config on the saved snapshot
                        gives the same equity curve.
- warnings              the card states what the backtest cannot tell you
                        (hindsight universe, short sample, untested regimes...).
                        A card without warnings would be a claim, not a record.

The price snapshot is what makes reproduction possible offline: Yahoo revises
adjusted history, so "the same tickers and dates" is not "the same data".
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import LAB_VERSION
from .ledger import config_hash
from .run import Dataset, run_benchmark, run_strategy
from .. import metrics as M
from ..config import CostConfig, EngineConfig
from ..engine import BacktestResult

SCHEMA = "1.0"
_BIG = {"equity.csv", "returns.csv", "weights.csv", "trades.csv", "metrics.csv", "prices.csv.gz",
        "validation.json"}


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=3, cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    if isinstance(o, (pd.DataFrame, pd.Series, np.ndarray)):
        return None                              # bulk arrays belong in artifacts, not the card
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:
            return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def build_warnings(ds: Dataset, res: BacktestResult, engine: EngineConfig, costs: CostConfig,
                   metrics: Dict[str, Any], validation: Optional[Dict[str, Any]],
                   n_trials: int) -> List[str]:
    w: List[str] = []
    years = len(res.returns) / engine.periods_per_year
    if engine.execution_lag < 1:
        w.append("Execution lag is 0: signals trade at the price that produced them (look-ahead bias).")
    if years < 3:
        w.append(f"Only {years:.1f} years of returns: Sharpe and drawdown estimates are very uncertain.")
    first = res.returns.index[0]
    if first > pd.Timestamp("2008-12-31"):
        w.append("The record starts after 2008: it has not been tested by a global financial crisis.")
    if first > pd.Timestamp("2020-03-31"):
        w.append("The record starts after March 2020: it has not seen a liquidity-shock crash.")
    if ds.source == "yfinance":
        w.append("The universe was picked today from instruments that exist today: survivorship and "
                 "hindsight bias are likely, and delisted names are absent.")
    if ds.prices.shape[1] < 5:
        w.append(f"Small universe ({ds.prices.shape[1]} instruments): cross-sectional signals have little to rank.")
    if costs.commission_bps + costs.slippage_bps <= 0:
        w.append("Trading frictions are set to zero.")
    to = metrics.get("Annual Turnover")
    if isinstance(to, (int, float)) and to > 5:
        w.append(f"Annual turnover is {to:.0%} of the portfolio: results are sensitive to cost assumptions.")
    if n_trials > 1 and not (validation and any(c.key == "dsr" and c.status != "na" for c in validation["checks"])):
        w.append(f"{n_trials} configurations were tried under this hypothesis and the Sharpe has not been deflated for it.")
    for q in ds.quality_warnings[:5]:
        w.append(f"Data: {q}")
    if validation:
        g = validation["grade"]
        if g["label"] in ("Fragile", "Unsupported"):
            w.append(f"Validation grade is {g['label']} ({g['fails']} failed check(s)).")
    return w


def write_run_card(run_dir: str, ds: Dataset, res: BacktestResult, strategy_key: str,
                   params: Dict[str, Any], engine: EngineConfig, costs: CostConfig,
                   label: str = "", validation: Optional[Dict[str, Any]] = None,
                   n_trials: int = 1, hypothesis_id: Optional[int] = None,
                   notes: str = "") -> Dict[str, Any]:
    art = os.path.join(run_dir, "artifacts")
    os.makedirs(art, exist_ok=True)
    ppy = engine.periods_per_year

    bench = run_benchmark(ds, engine, start=res.returns.index[0])
    metrics = M.summary(res.returns, res.equity, bench.returns if bench is not None else None,
                        res.turnover, res.exposure, 0.0, ppy)
    metrics = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in metrics.items()}

    res.equity.rename("equity").to_csv(os.path.join(art, "equity.csv"), header=True)
    res.returns.rename("return").to_csv(os.path.join(art, "returns.csv"), header=True, float_format="%.12g")
    res.weights.to_csv(os.path.join(art, "weights.csv"), float_format="%.8g")
    res.trades.to_csv(os.path.join(art, "trades.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(art, "metrics.csv"), index=False)
    ds.to_frame().to_csv(os.path.join(art, "prices.csv.gz"), compression="gzip", float_format="%.10g")
    if validation is not None:
        v = {"grade": validation["grade"], "n_trials": validation["n_trials"],
             "checks": [_safe(asdict(c)) for c in validation["checks"]]}
        with open(os.path.join(art, "validation.json"), "w", encoding="utf-8") as fh:
            json.dump(_safe(v), fh, indent=1, allow_nan=False)

    config = {"strategy_key": strategy_key, "params": _safe(params),
              "engine": _safe(asdict(engine)), "costs": _safe(asdict(costs)),
              "tickers": ds.tickers, "benchmark": ds.benchmark.name if ds.benchmark is not None else None,
              "label": label or strategy_key}
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=1, allow_nan=False)

    manifest = []
    for root, _, files in os.walk(run_dir):
        for fn in sorted(files):
            if fn in ("run_card.json", "run_card.md"):
                continue
            p = os.path.join(root, fn)
            manifest.append({"path": os.path.relpath(p, run_dir).replace("\\", "/"),
                             "size_bytes": os.path.getsize(p), "sha256": _sha(p)})
    manifest.sort(key=lambda m: m["path"])
    mrow = next((m for m in manifest if m["path"] == "artifacts/metrics.csv"), None)
    citations = [{"metric": k, "artifact": "artifacts/metrics.csv", "column": k, "row": 1,
                  "sha256": mrow["sha256"]} for k, v in metrics.items() if v is not None] if mrow else []

    card = {
        "schema_version": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label or strategy_key, "hypothesis_id": hypothesis_id,
        "strategy": {"key": strategy_key, "params": _safe(params)},
        "backtest": {"tickers": ds.tickers, "start": str(res.returns.index[0].date()),
                     "end": str(res.returns.index[-1].date()), "sessions": int(len(res.returns)),
                     "initial_capital": engine.initial_capital, "rebalance": engine.rebalance,
                     "execution_lag": engine.execution_lag},
        "reproducibility": {"config_hash": config_hash(config), "data_hash": ds.content_hash(),
                            "data_source": ds.source, "data_note": ds.note,
                            "qbt_lab": LAB_VERSION, "python": platform.python_version(),
                            "pandas": pd.__version__, "numpy": np.__version__,
                            "git_commit": _git_commit()},
        "metrics": _safe(metrics),
        "validation": ({"grade": _safe(validation["grade"]), "n_trials": validation["n_trials"],
                        "checks": [{"check": c.title, "status": c.status, "evidence": c.detail}
                                   for c in validation["checks"]]} if validation else None),
        "warnings": build_warnings(ds, res, engine, costs, metrics, validation, n_trials),
        "notes": notes,
        "artifacts": manifest, "metric_citations": citations,
    }
    with open(os.path.join(run_dir, "run_card.json"), "w", encoding="utf-8") as fh:
        json.dump(_safe(card), fh, indent=1, allow_nan=False)
    with open(os.path.join(run_dir, "run_card.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(card))
    return card


def render_markdown(card: Dict[str, Any]) -> str:
    L = [f"# Run card: {card['label']}", "",
         f"Generated {card['generated_at']} | schema {card['schema_version']}", ""]
    b = card["backtest"]
    L += ["## Backtest", f"- Strategy: `{card['strategy']['key']}` {json.dumps(card['strategy']['params'])}",
          f"- Universe: {', '.join(b['tickers'])}",
          f"- Period: {b['start']} to {b['end']} ({b['sessions']} sessions)",
          f"- Capital {b['initial_capital']:,.0f}, rebalance `{b['rebalance']}`, execution lag {b['execution_lag']}", ""]
    r = card["reproducibility"]
    L += ["## Reproducibility", f"- config hash `{r['config_hash']}`", f"- data hash `{r['data_hash'][:16]}...` "
          f"({r['data_source']}; {r['data_note']})",
          f"- python {r['python']}, pandas {r['pandas']}, numpy {r['numpy']}, lab {r['qbt_lab']}"
          + (f", commit {r['git_commit']}" if r['git_commit'] else ""), ""]
    L += ["## Metrics", "", "| Metric | Value |", "|---|---|"]
    for k, v in card["metrics"].items():
        L.append(f"| {k} | {M.format_metric(k, v) if v is not None else 'n/a'} |")
    L.append("")
    if card.get("validation"):
        v = card["validation"]
        g = v["grade"]
        L += [f"## Validation: {g['label']}", f"{g['passes']} passed, {g['warnings']} warnings, {g['fails']} failed "
              f"(trials counted: {v['n_trials']})", "", "| Check | Status | Evidence |", "|---|---|---|"]
        for c in v["checks"]:
            L.append(f"| {c['check']} | {c['status'].upper()} | {c['evidence']} |")
        L.append("")
    L += ["## Warnings"] + ([f"- {w}" for w in card["warnings"]] or ["- none raised"]) + [""]
    if card.get("notes"):
        L += ["## Notes", card["notes"], ""]
    L += ["## Artifacts", "", "| File | Size | SHA-256 |", "|---|---|---|"]
    for a in card["artifacts"]:
        L.append(f"| {a['path']} | {a['size_bytes']:,} | `{a['sha256'][:16]}...` |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------
def verify_run_card(run_dir: str) -> Dict[str, Any]:
    """Re-hashes every artifact and re-checks every quoted metric."""
    with open(os.path.join(run_dir, "run_card.json"), encoding="utf-8") as fh:
        card = json.load(fh)
    problems: List[str] = []
    for a in card["artifacts"]:
        p = os.path.join(run_dir, a["path"])
        if not os.path.isfile(p):
            problems.append(f"missing: {a['path']}")
        elif _sha(p) != a["sha256"]:
            problems.append(f"changed: {a['path']}")
    mp = os.path.join(run_dir, "artifacts", "metrics.csv")
    if os.path.isfile(mp):
        row = pd.read_csv(mp).iloc[0]
        for k, v in card["metrics"].items():
            if v is None:
                continue
            if k not in row or not np.isclose(float(row[k]), float(v), rtol=1e-9, atol=1e-12):
                problems.append(f"metric differs from metrics.csv: {k}")
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            if config_hash(json.load(fh)) != card["reproducibility"]["config_hash"]:
                problems.append("config.json does not match the recorded config hash")
    return {"ok": not problems, "problems": problems, "checked": len(card["artifacts"])}


def reproduce_run_card(run_dir: str, tol: float = 1e-8) -> Dict[str, Any]:
    """Re-run the saved config on the saved snapshot and compare equity curves."""
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    snap = pd.read_csv(os.path.join(run_dir, "artifacts", "prices.csv.gz"), index_col=0, parse_dates=True)
    ds = Dataset.from_frame(snap)
    engine = EngineConfig(**cfg["engine"])
    costs = CostConfig(**cfg["costs"])
    res = run_strategy(ds, cfg["strategy_key"], cfg["params"], engine, costs, cfg.get("label"))
    saved = pd.read_csv(os.path.join(run_dir, "artifacts", "equity.csv"), index_col=0, parse_dates=True)["equity"]
    new = res.equity.reindex(saved.index)
    diff = float(((new - saved).abs() / saved.abs().clip(lower=1e-12)).max())
    return {"ok": bool(np.isfinite(diff) and diff <= tol and len(new.dropna()) == len(saved)),
            "max_relative_difference": diff, "sessions": int(len(saved))}
