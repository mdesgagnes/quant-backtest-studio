"""The lab's tool layer: every capability as a JSON-in, JSON-out function.

One definition, three front ends. The MCP server (`mcp_server.py`) exposes
these to Claude, the optional in-app assistant (`assistant.py`) calls them in
a tool-use loop, and the command line (`__main__.py`) calls them directly. The
Streamlit workspace uses the underlying modules but shares this session state
conventions (dataset ids, run ids).

Each tool returns a plain dict. Failures return `{"error": "..."}` rather than
raising, so a caller driving the lab through language gets a message it can
act on instead of a stack trace. Nothing here trades: the lab is research
only.

Session state is process-local. Datasets are also snapshotted to
`<lab dir>/data/<id>.csv.gz`, so a new process can reload one by id.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
import traceback
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import factors as F
from . import ledger as L
from . import stats as S
from .run import Dataset, load_dataset, run_benchmark, run_strategy, synthetic_dataset
from .. import metrics as M
from ..config import CostConfig, EngineConfig
from ..strategies import REGISTRY, get as get_strategy

DATASETS: Dict[str, Dataset] = {}
RUNS: Dict[str, Dict[str, Any]] = {}
TOOLS: Dict[str, Dict[str, Any]] = {}


# ----------------------------------------------------------------------
def tool(name: str, description: str, properties: Dict[str, Any], required: Optional[List[str]] = None):
    def deco(fn: Callable[..., Dict[str, Any]]):
        def safe(**kwargs):
            try:
                return plain(fn(**kwargs))
            except Exception as exc:                     # a tool must never crash its host
                return {"error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc(limit=3) if os.environ.get("QBT_LAB_DEBUG") else None}
        safe.__name__ = fn.__name__
        TOOLS[name] = {"name": name, "description": description, "fn": safe,
                       "inputSchema": {"type": "object", "properties": properties,
                                       "required": required or [], "additionalProperties": False}}
        return safe
    return deco


def plain(o: Any, depth: int = 0) -> Any:
    """JSON-safe: DataFrames become records, Series dicts, NaN/inf None."""
    import math
    if depth > 8:
        return str(o)
    if isinstance(o, pd.DataFrame):
        return plain(json.loads(o.reset_index().to_json(orient="records", date_format="iso", double_precision=8))
                     if len(o) else [], depth + 1)
    if isinstance(o, pd.Series):
        return {str(k): plain(v, depth + 1) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [plain(v, depth + 1) for v in o.tolist()[:2000]]
    if isinstance(o, dict):
        return {str(k): plain(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v, depth + 1) for v in o]
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return plain(o.item(), depth + 1)
        except Exception:
            return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if hasattr(o, "__dataclass_fields__"):
        return plain(asdict(o), depth + 1)
    return o


def _ds(dataset_id: str) -> Dataset:
    if dataset_id in DATASETS:
        return DATASETS[dataset_id]
    p = os.path.join(L.lab_dir(), "data", f"{dataset_id}.csv.gz")
    if os.path.isfile(p):
        ds = Dataset.from_frame(pd.read_csv(p, index_col=0, parse_dates=True), source="snapshot")
        DATASETS[dataset_id] = ds
        return ds
    raise KeyError(f"Unknown dataset '{dataset_id}'. Call load_data first.")


def _register_dataset(ds: Dataset) -> str:
    did = ds.content_hash()[:10]
    DATASETS[did] = ds
    d = os.path.join(L.lab_dir(), "data")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{did}.csv.gz")
    if not os.path.isfile(p):
        ds.to_frame().to_csv(p, compression="gzip", float_format="%.10g")
    return did


def _engine(a: Dict[str, Any]) -> EngineConfig:
    e = EngineConfig()
    for k in ("rebalance", "execution_lag", "initial_capital", "max_leverage"):
        if a.get(k) is not None:
            setattr(e, k, type(getattr(e, k))(a[k]))
    return e


def _costs(a: Dict[str, Any]) -> CostConfig:
    c = CostConfig()
    for k in ("commission_bps", "slippage_bps"):
        if a.get(k) is not None:
            setattr(c, k, float(a[k]))
    return c


def _run(run_id: str) -> Dict[str, Any]:
    if run_id not in RUNS:
        raise KeyError(f"Unknown run '{run_id}'. Call run_backtest first.")
    return RUNS[run_id]


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
@tool("list_strategies", "List the backtest strategies available, with their parameters and defaults.", {})
def list_strategies() -> Dict[str, Any]:
    out = []
    for k, s in sorted(REGISTRY.items()):
        out.append({"key": k, "label": s.label, "description": s.description[:240],
                    "params": [{"key": p.key, "kind": p.kind, "default": p.default,
                                "choices": p.choices[:60] if p.choices else None} for p in s.params],
                    "needs_exog": s.needs_exog})
    return {"strategies": out}


@tool("list_factors", "List the research factor library (key, label, category, expected sign).", {})
def list_factors() -> Dict[str, Any]:
    return {"factors": F.list_factors()}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@tool("load_data",
      "Load a price universe (total-return closes from Yahoo Finance, or a reproducible synthetic set) and "
      "return a dataset_id used by every other tool. Use .TO for TSX symbols.",
      {"tickers": {"type": "array", "items": {"type": "string"}},
       "start": {"type": "string", "description": "YYYY-MM-DD"}, "end": {"type": "string"},
       "benchmark": {"type": "string"}, "cash_proxy": {"type": "string"},
       "synthetic": {"type": "boolean", "description": "Use fake data (offline demos/tests)."},
       "edge": {"type": "number", "description": "Synthetic only: planted momentum strength."}},
      [])
def load_data(tickers: Optional[List[str]] = None, start: str = "2012-01-01", end: Optional[str] = None,
              benchmark: Optional[str] = None, cash_proxy: Optional[str] = None,
              synthetic: bool = False, edge: float = 0.0) -> Dict[str, Any]:
    if synthetic or not tickers:
        ds = synthetic_dataset(edge=edge)
    else:
        ds = load_dataset(tickers, start, end, benchmark, cash_proxy)
    did = _register_dataset(ds)
    cover = ds.prices.notna().sum()
    return {"dataset_id": did, "tickers": ds.tickers, "start": ds.start, "end": ds.end,
            "sessions": int(len(ds.prices)), "source": ds.source, "note": ds.note,
            "history_sessions": cover.to_dict(), "benchmark": ds.benchmark is not None,
            "warnings": ds.quality_warnings}


# ----------------------------------------------------------------------
# Factor research
# ----------------------------------------------------------------------
@tool("factor_screen",
      "Screen the factor library on a dataset: rank IC, ICIR, Newey-West t-stat, long-short spread, and "
      "Bonferroni / Benjamini-Hochberg adjusted p-values. Counts every factor as a trial in the ledger.",
      {"dataset_id": {"type": "string"}, "horizon": {"type": "integer", "description": "forward days (default 21)"},
       "factors": {"type": "array", "items": {"type": "string"}}, "top": {"type": "integer"},
       "hypothesis_id": {"type": "integer"}}, ["dataset_id"])
def factor_screen(dataset_id: str, horizon: int = 21, factors: Optional[List[str]] = None,
                  top: int = 15, hypothesis_id: Optional[int] = None) -> Dict[str, Any]:
    ds = _ds(dataset_id)
    df = F.screen(ds.prices, factors, horizon)
    survivors = int((df["p_bh"] < 0.10).sum()) if "p_bh" in df else 0
    if hypothesis_id:
        L.log_experiment(hypothesis_id, "factor_screen", f"screen h={horizon}", {"factors": factors or "all", "h": horizon},
                         ds.content_hash(), n_variants=len(df),
                         metrics={"survivors_bh10": survivors, "best": df.index[0] if len(df) else None})
    cols = ["Factor", "Category", "Mean IC", "ICIR", "t-stat", "p", "p_bonferroni", "p_bh", "Spread Sharpe", "n"]
    return {"horizon": horizon, "n_factors": len(df), "survivors_after_fdr_10pct": survivors,
            "note": "Rank by |t|. Factors with p_bh >= 0.10 are indistinguishable from what a screen this wide "
                    "finds by chance.", "top": df[[c for c in cols if c in df]].head(top)}


@tool("factor_report", "Detailed research on one factor: IC statistics, decay across horizons, quantile returns.",
      {"dataset_id": {"type": "string"}, "factor": {"type": "string"}, "horizon": {"type": "integer"},
       "n_quantiles": {"type": "integer"}}, ["dataset_id", "factor"])
def factor_report(dataset_id: str, factor: str, horizon: int = 21, n_quantiles: int = 0) -> Dict[str, Any]:
    ds = _ds(dataset_id)
    if factor not in F.FACTORS:
        raise KeyError(f"Unknown factor '{factor}'. Call list_factors.")
    sc = F.compute(factor, ds.prices)
    ic = F.information_coefficient(sc, ds.prices, horizon)
    nq = n_quantiles or (3 if ds.prices.shape[1] < 15 else 5)
    qr = F.quantile_returns(sc, ds.prices, nq, horizon)
    return {"factor": factor, "label": F.FACTORS[factor].label, "horizon": horizon,
            "ic": {k: v for k, v in ic.items() if k not in ("ic", "ic_sampled")},
            "decay": F.ic_decay(sc, ds.prices),
            "quantile_mean_forward_return": qr.get("mean_by_q"),
            "spread": {k: qr.get(k) for k in ("spread_mean", "spread_sharpe", "spread_hit", "monotonicity",
                                              "top_turnover", "n_dates")}}


# ----------------------------------------------------------------------
# Backtest + validation
# ----------------------------------------------------------------------
_RUN_PROPS = {"dataset_id": {"type": "string"}, "strategy": {"type": "string"},
              "params": {"type": "object"}, "rebalance": {"type": "string", "description": "D/W/M/Q/S/A"},
              "commission_bps": {"type": "number"}, "slippage_bps": {"type": "number"},
              "initial_capital": {"type": "number"}, "execution_lag": {"type": "integer"},
              "hypothesis_id": {"type": "integer"}, "label": {"type": "string"}}


@tool("run_backtest",
      "Run a strategy through the QBS engine (frictions, execution lag, rebalance calendar) and return the "
      "headline metrics and a run_id. Logs the run as an experiment when a hypothesis_id is given.",
      _RUN_PROPS, ["dataset_id", "strategy"])
def run_backtest(dataset_id: str, strategy: str, params: Optional[Dict[str, Any]] = None,
                 hypothesis_id: Optional[int] = None, label: Optional[str] = None, **kw) -> Dict[str, Any]:
    ds = _ds(dataset_id)
    get_strategy(strategy)
    eng, cst = _engine(kw), _costs(kw)
    res = run_strategy(ds, strategy, params or {}, eng, cst, label)
    bench = run_benchmark(ds, eng, start=res.returns.index[0])
    met = M.summary(res.returns, res.equity, bench.returns if bench is not None else None,
                    res.turnover, res.exposure, 0.0, eng.periods_per_year)
    rid = f"r{len(RUNS) + 1}"
    RUNS[rid] = {"ds_id": dataset_id, "ds": ds, "res": res, "bench": bench, "strategy": strategy,
                 "params": params or {}, "engine": eng, "costs": cst, "label": label or strategy,
                 "hypothesis_id": hypothesis_id, "metrics": met, "validation": None}
    if hypothesis_id:
        L.log_experiment(hypothesis_id, "backtest", label or strategy,
                         {"s": strategy, "p": params or {}, "e": asdict(eng), "c": asdict(cst)},
                         ds.content_hash(), metrics={k: met.get(k) for k in ("CAGR", "Sharpe", "Max Drawdown")})
    keep = ["Total Return", "CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown",
            "Annual Turnover", "Beta", "Alpha (ann.)", "Information Ratio"]
    return {"run_id": rid, "period": [str(res.returns.index[0].date()), str(res.returns.index[-1].date())],
            "metrics": {k: met.get(k) for k in keep if k in met}}


@tool("validate_run",
      "Run the statistical validation battery on a finished backtest: probabilistic and deflated Sharpe, "
      "stationary bootstrap, sign-flip and random-timing tests, walk-forward consistency, cost stress, and "
      "(given a parameter grid) probability of backtest overfitting and combinatorial purged CV. "
      "Returns a pass/warn/fail table and a grade.",
      {"run_id": {"type": "string"},
       "grid": {"type": "object", "description": "param name -> list of values, e.g. {\"lookback\": [60,126,252]}"},
       "n_trials": {"type": "integer", "description": "configurations tried (default: from the ledger)"}},
      ["run_id"])
def validate_run_tool(run_id: str, grid: Optional[Dict[str, List[Any]]] = None,
                      n_trials: Optional[int] = None) -> Dict[str, Any]:
    from .validate import validate_run, checks_table
    r = _run(run_id)
    nt = n_trials or L.trial_count(r["hypothesis_id"])
    v = validate_run(r["ds"], r["res"], r["strategy"], r["params"], r["engine"], r["costs"], nt, grid)
    r["validation"] = v
    r["n_trials"] = v["n_trials"]
    if r["hypothesis_id"]:
        L.log_experiment(r["hypothesis_id"], "validation", r["label"],
                         {"s": r["strategy"], "p": r["params"], "grid": grid},
                         r["ds"].content_hash(), n_variants=(int(v["sweep_returns"].shape[1]) if v.get("sweep_returns") is not None else 1),
                         metrics={"grade": v["grade"]["label"], "score": v["grade"]["score"]},
                         verdict=v["grade"]["label"])
    return {"grade": v["grade"], "n_trials_counted": v["n_trials"],
            "checks": [{"key": c.key, "check": c.title, "status": c.status, "value": c.value, "evidence": c.detail,
                        "threshold": c.threshold} for c in v["checks"]]}


@tool("save_run_card",
      "Write the run card (JSON + Markdown + artifacts + frozen price snapshot) and the HTML dossier for a run. "
      "Returns the folder, the warnings the card raised, and the dossier path.",
      {"run_id": {"type": "string"}, "out_dir": {"type": "string"}, "notes": {"type": "string"}}, ["run_id"])
def save_run_card(run_id: str, out_dir: Optional[str] = None, notes: str = "") -> Dict[str, Any]:
    from .dossier import build_dossier
    from .runcard import write_run_card
    r = _run(run_id)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in r["label"])[:40]
    run_dir = out_dir or os.path.join(L.lab_dir(), "runs", f"{stamp}_{safe}")
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise ValueError(f"{run_dir} already has files in it; a run card never overwrites another. Pick a new folder.")
    card = write_run_card(run_dir, r["ds"], r["res"], r["strategy"], r["params"], r["engine"], r["costs"],
                          r["label"], r.get("validation"), r.get("n_trials", 1), r["hypothesis_id"], notes)
    page = build_dossier(card, r["res"], r["bench"])
    path = os.path.join(run_dir, "dossier.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return {"run_dir": os.path.abspath(run_dir), "dossier": os.path.abspath(path), "warnings": card["warnings"],
            "config_hash": card["reproducibility"]["config_hash"], "data_hash": card["reproducibility"]["data_hash"]}


@tool("verify_run_card", "Check a saved run card: artifact hashes, quoted metrics against metrics.csv, config hash.",
      {"run_dir": {"type": "string"}}, ["run_dir"])
def verify_run_card_tool(run_dir: str) -> Dict[str, Any]:
    from .runcard import verify_run_card
    return verify_run_card(run_dir)


@tool("reproduce_run_card", "Re-run a saved run card on its frozen price snapshot and compare equity curves.",
      {"run_dir": {"type": "string"}}, ["run_dir"])
def reproduce_run_card_tool(run_dir: str) -> Dict[str, Any]:
    from .runcard import reproduce_run_card
    return reproduce_run_card(run_dir)


# ----------------------------------------------------------------------
# Risk, events
# ----------------------------------------------------------------------
@tool("risk_xray",
      "Risk x-ray of a portfolio: concentration, volatility, drawdown, VaR/ES, risk contribution per holding, "
      "effective number of bets, diversification ratio. Give explicit weights, or a run_id (average or latest "
      "held weights).",
      {"dataset_id": {"type": "string"}, "weights": {"type": "object"}, "run_id": {"type": "string"},
       "mode": {"type": "string", "enum": ["average", "current"]}, "window": {"type": "integer"}}, [])
def risk_xray_tool(dataset_id: Optional[str] = None, weights: Optional[Dict[str, float]] = None,
                   run_id: Optional[str] = None, mode: str = "current", window: Optional[int] = None) -> Dict[str, Any]:
    from . import xray
    if run_id:
        r = _run(run_id)
        ds, w = r["ds"], (xray.current_weights(r["res"].weights) if mode == "current"
                          else xray.average_weights(r["res"].weights))
    else:
        ds, w = _ds(dataset_id), weights
    out = xray.risk_xray(ds.prices, w or {}, window=window)
    out.pop("portfolio_returns", None)
    out.pop("corr", None)
    return out


@tool("event_study",
      "Market-model event study: average abnormal return and CAR around dated events. events = list of "
      "[ticker, YYYY-MM-DD]. The market proxy is the dataset's benchmark.",
      {"dataset_id": {"type": "string"}, "events": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
       "pre": {"type": "integer"}, "post": {"type": "integer"}}, ["dataset_id", "events"])
def event_study_tool(dataset_id: str, events: List[List[str]], pre: int = 5, post: int = 20) -> Dict[str, Any]:
    from .eventstudy import event_study
    ds = _ds(dataset_id)
    if ds.benchmark is None:
        raise ValueError("The dataset has no benchmark; reload with benchmark set to a market index or ETF.")
    out = event_study(ds.prices, ds.benchmark, [(e[0], e[1]) for e in events], pre, post)
    return {k: out.get(k) for k in ("n_events", "mean_car", "median_car", "t_car", "p_car", "share_positive",
                                   "average", "table", "skipped")}


# ----------------------------------------------------------------------
# Ledger & monitoring
# ----------------------------------------------------------------------
@tool("add_hypothesis", "Record a research hypothesis (what you expect and why) before testing it.",
      {"title": {"type": "string"}, "statement": {"type": "string"}, "rationale": {"type": "string"},
       "universe": {"type": "string"}}, ["title"])
def add_hypothesis(title: str, statement: str = "", rationale: str = "", universe: str = "") -> Dict[str, Any]:
    return {"hypothesis_id": L.add_hypothesis(title, statement, rationale, universe)}


@tool("list_hypotheses", "List hypotheses with the number of experiments and trials logged under each.", {})
def list_hypotheses() -> Dict[str, Any]:
    return {"hypotheses": L.list_hypotheses()}


@tool("update_hypothesis", "Change a hypothesis's status (idea/testing/validated/rejected/promoted/retired) or text.",
      {"hypothesis_id": {"type": "integer"}, "status": {"type": "string"}, "statement": {"type": "string"},
       "rationale": {"type": "string"}}, ["hypothesis_id"])
def update_hypothesis(hypothesis_id: int, **fields) -> Dict[str, Any]:
    L.update_hypothesis(hypothesis_id, **fields)
    return {"ok": True, "hypothesis": L.get_hypothesis(hypothesis_id)}


@tool("list_experiments", "List logged experiments, optionally for one hypothesis.",
      {"hypothesis_id": {"type": "integer"}}, [])
def list_experiments(hypothesis_id: Optional[int] = None) -> Dict[str, Any]:
    return {"experiments": L.list_experiments(hypothesis_id),
            "trials": L.trial_count(hypothesis_id) if hypothesis_id else None}


@tool("register_strategy",
      "Freeze a run as a monitored strategy. Data after today is out-of-sample for it; decay_scan compares the "
      "forward record with the in-sample baseline.",
      {"run_id": {"type": "string"}, "name": {"type": "string"}, "note": {"type": "string"}}, ["run_id", "name"])
def register_strategy(run_id: str, name: str, note: str = "") -> Dict[str, Any]:
    r = _run(run_id)
    res, ppy = r["res"], r["engine"].periods_per_year
    base = {"sharpe": S.annualised_sharpe(res.returns, ppy),
            "vol": float(res.returns.std() * np.sqrt(ppy)), "cagr": M.cagr(res.equity, ppy),
            "max_drawdown": M.max_drawdown(res.equity), "start": str(res.returns.index[0].date()),
            "end": str(res.returns.index[-1].date()), "data_source": r["ds"].source}
    spec = {"strategy_key": r["strategy"], "params": r["params"], "engine": asdict(r["engine"]),
            "costs": asdict(r["costs"]), "tickers": r["ds"].tickers,
            "benchmark": r["ds"].benchmark.name if r["ds"].benchmark is not None else "",
            "registered": str(res.returns.index[-1].date())}
    sid = L.register_strategy(name, spec, base, r["hypothesis_id"], note)
    return {"strategy_id": sid, "registered": spec["registered"], "baseline": base}


@tool("decay_scan",
      "Re-run every active registered strategy on fresh data and report fresh/healthy/aging/stale against "
      "its registration baseline. Downloads current prices.",
      {"name": {"type": "string"}}, [])
def decay_scan(name: Optional[str] = None) -> Dict[str, Any]:
    from .decay import assess
    out = []
    for s in L.list_strategies(active_only=True):
        if name and s["name"] != name:
            continue
        base = s["baseline"] or {}
        try:
            if base.get("data_source") == "synthetic":
                raise ValueError("synthetic data cannot be refreshed")
            ds = load_dataset(s["tickers"], base.get("start", "2010-01-01"), None, s["benchmark"] or None)
            eng, cst = EngineConfig(**s["engine"]), CostConfig(**s["costs"])
            res = run_strategy(ds, s["strategy_key"], s["params"], eng, cst, s["name"])
            a = assess(res.returns, s["registered"], base, eng.periods_per_year)
            a.pop("cusum_path", None)
            out.append({"name": s["name"], **a})
        except Exception as exc:
            out.append({"name": s["name"], "status": "error", "detail": str(exc)})
    return {"strategies": out}
