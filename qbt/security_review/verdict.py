"""Scores, verdicts, confidence and the written rationale.

Rules (thresholds live in config/objectives.yaml):

* score_i  = clip(direction * (value - center) / scale, -1, +1)
* composite = sum(w_i * score_i) / sum(w_i) over metrics that have a value
* Add if composite >= add, Consider if >= consider, else Do not add
* a failed guardrail, or a rolling-window share below the minimum, caps the
  verdict at Consider
* confidence starts High and is lowered by short history, weak rolling
  consistency, a verdict that flips with the add weight, or missing metrics
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MetricSpec, Objective, ObjectivesConfig

VERDICTS = ("Add", "Consider", "Do not add")
LEVELS = ("High", "Moderate", "Low")


def metric_score(spec: MetricSpec, value: float) -> float:
    if value is None or not np.isfinite(value):
        return np.nan
    return float(np.clip(spec.direction * (value - spec.center) / spec.scale, -1.0, 1.0))


def score(obj: Objective, metrics: Dict[str, float], cat: Dict[str, MetricSpec]) -> Dict[str, Any]:
    rows = []
    num = den = missing = 0.0
    tot = sum(obj.weights.values())
    for k, w in obj.weights.items():
        v = metrics.get(k, np.nan)
        s = metric_score(cat[k], v)
        rows.append({"Metric": k, "Label": cat[k].label, "Value": v, "Score": s,
                     "Weight": w / tot, "Contribution": (w / tot) * s if s == s else np.nan})
        if s == s:
            num += w * s
            den += w
        else:
            missing += w
    comp = num / den if den > 0 else np.nan
    return {"composite": comp, "components": pd.DataFrame(rows),
            "missing_share": missing / tot if tot else 1.0}


def raw_verdict(obj: Objective, comp: float) -> str:
    if comp != comp:
        return "Do not add"
    if comp >= obj.add:
        return "Add"
    if comp >= obj.consider:
        return "Consider"
    return "Do not add"


def guardrail_failures(obj: Objective, metrics: Dict[str, float]) -> List[str]:
    out = []
    for g in obj.guardrails:
        v = metrics.get(g.get("metric"), np.nan)
        if v == v and "min" in g and v < float(g["min"]):
            out.append(str(g.get("text", g["metric"])))
        if v == v and "max" in g and v > float(g["max"]):
            out.append(str(g.get("text", g["metric"])))
    return out


def _down(level: str) -> str:
    return LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]


def confidence(months: int, rolling_share: float, min_share: float,
               weight_verdicts: Dict[float, str], missing_share: float) -> Dict[str, Any]:
    level, why = "High", []
    if months < 36:
        level, why = "Low", [f"only {months} months of overlapping history (< 36)"]
    elif months < 60:
        level, why = "Moderate", [f"{months} months of overlapping history (< 60)"]
    if rolling_share == rolling_share and rolling_share < min_share:
        level = _down(level)
        why.append(f"improves the objective in only {rolling_share:.0%} of rolling 36-month windows")
    vs = set(weight_verdicts.values())
    if "Add" in vs and "Do not add" in vs:
        level = _down(level)
        why.append("verdict changes from Add to Do not add across add weights "
                   + ", ".join(f"{w:.0%}" for w in weight_verdicts))
    if missing_share > 0.5:
        level = "Low"
        why.append(f"{missing_share:.0%} of the objective's weight has no data")
    elif missing_share > 0:
        why.append(f"{missing_share:.0%} of the objective's weight has no data and was left out")
    return {"level": level, "reasons": why}


def decide(obj: Objective, metrics: Dict[str, float], cat: Dict[str, MetricSpec],
           months: int, rolling_share: float, min_share: float,
           weight_verdicts: Optional[Dict[float, str]] = None) -> Dict[str, Any]:
    sc = score(obj, metrics, cat)
    v0 = raw_verdict(obj, sc["composite"])
    v = v0
    caps = []
    fails = guardrail_failures(obj, metrics)
    if v == "Add" and fails:
        v = "Consider"
        caps.append("capped at Consider: " + "; ".join(fails))
    if v == "Add" and rolling_share == rolling_share and rolling_share < min_share:
        v = "Consider"
        caps.append(f"capped at Consider: improves the objective in only {rolling_share:.0%} "
                    f"of rolling windows (minimum {min_share:.0%})")
    conf = confidence(months, rolling_share, min_share, weight_verdicts or {}, sc["missing_share"])
    return {"verdict": v, "raw_verdict": v0, "composite": sc["composite"],
            "components": sc["components"], "missing_share": sc["missing_share"],
            "guardrails_failed": fails, "caps": caps, "confidence": conf["level"],
            "confidence_reasons": conf["reasons"]}


# ----------------------------------------------------------------------
# Written rationale
# ----------------------------------------------------------------------
def fmt_metric(key: str, v: float) -> str:
    if v is None or v != v:
        return "n/a"
    pts = {"d_vol", "d_cagr", "d_mdd", "d_cvar", "d_worst_12m", "d_tail_return",
           "d_up_month_return", "d_portfolio_yield", "d_underwater_share",
           "d_prob_dd20", "d_prob_dd30", "d_depletion_prob"}
    if key in pts or key.startswith("d_regime_") or key.startswith("d_scen_"):
        return f"{v * 100:+.2f} pts"
    if key == "cand_tail_return" or key.startswith("cand_regime_"):
        return f"{v * 100:+.2f}% a month"
    if key == "dist_yield":
        return f"{v * 100:.2f}%"
    if key == "d_max_underwater":
        return f"{v:+.0f} months"
    if key.startswith("d_"):
        return f"{v:+.2f}"
    return f"{v:.2f}"


# Metrics that describe what an investor gives up; the most negative one
# that is not part of the objective is the "main trade-off".
TRADEOFF_METRICS = ["d_cagr", "up_capture", "d_sharpe", "d_mdd", "d_vol", "d_cvar",
                    "d_portfolio_yield"]


def rationale(obj: Objective, dec: Dict[str, Any], metrics: Dict[str, float],
              cat: Dict[str, MetricSpec], flags: List[str], candidate: str) -> Dict[str, Any]:
    comp = dec["components"].dropna(subset=["Score"]).copy()
    comp["abs"] = comp["Contribution"].abs()
    facts = []
    for _, r in comp.sort_values("abs", ascending=False).head(4).iterrows():
        tone = "helps" if r["Score"] > 0 else "hurts" if r["Score"] < 0 else "neutral"
        facts.append(f"{r['Label']}: {fmt_metric(r['Metric'], r['Value'])} ({tone}, score {r['Score']:+.2f})")
    good = [k for k in obj.weights if metric_score(cat[k], metrics.get(k, np.nan)) > 0.2]
    worst, worst_s = None, 0.0
    for k in TRADEOFF_METRICS:
        if k in obj.weights or k not in cat:
            continue
        s = metric_score(cat[k], metrics.get(k, np.nan))
        if s == s and s < worst_s:
            worst, worst_s = k, s
    if worst and worst_s <= -0.2:
        gain = cat[good[0]].label.lower() if good else "the objective"
        trade = (f"Improves {gain} but costs on {cat[worst].label.lower()} "
                 f"({fmt_metric(worst, metrics.get(worst))}).")
    elif dec["guardrails_failed"]:
        trade = "Trade-off: " + "; ".join(dec["guardrails_failed"]) + "."
    else:
        trade = "No material trade-off on return, risk-adjusted return or upside capture."
    caveats = list(dec["caps"]) + list(dec["confidence_reasons"]) + list(flags)
    return {"facts": facts, "tradeoff": trade, "caveats": caveats}
