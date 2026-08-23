"""Rule builder.

Builds a strategy from a list of criteria instead of a typed expression.
Each rule is an indicator, a comparison and a value; rules combine with AND
or OR, and the whole thing compiles to the same sandboxed formula the
Builder already evaluates. Nothing new is executed -- the compiler emits
text, and that text goes through `qbt.formula` exactly as a hand-typed
expression does, whitelist and all.

Two kinds of rule, because a strategy needs both:

- **Filter** rules answer yes or no and decide what is eligible.
  `price > sma(price, 200)`
- **Score** rules produce a number and decide the ranking among what is
  eligible, each with a weight. `pctrank(mom(price, 126))`

On drag and drop: Streamlit has no native drag-and-drop, and adding one
means shipping a compiled React component, a build step and a maintenance
burden on a tool whose whole point is that it is a single Python package.
Reordering here is done with move controls instead. For scoring rules the
order is presentational anyway -- weights decide the outcome, not position
-- so what would be gained is mostly the gesture.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# What a rule can be built from. Each entry is (label, template, needs_n).
# The template is formula text with {x} for the series and {n} for a window.
# ----------------------------------------------------------------------
INDICATORS: Dict[str, Dict[str, Any]] = {
    "Price": {"expr": "price", "n": False,
              "help": "The adjusted price itself."},
    "Moving average": {"expr": "sma(price, {n})", "n": True, "default_n": 200,
                       "help": "Simple moving average over n sessions."},
    "Exponential average": {"expr": "ema(price, {n})", "n": True, "default_n": 100,
                            "help": "Exponentially weighted average."},
    "Momentum (total return)": {"expr": "mom(price, {n})", "n": True, "default_n": 126,
                                "help": "Return over the past n sessions."},
    "Volatility": {"expr": "vol(price, {n})", "n": True, "default_n": 60,
                   "help": "Annualized realized volatility."},
    "Downside volatility": {"expr": "dvol(price, {n})", "n": True, "default_n": 60,
                            "help": "Volatility of losses only."},
    "RSI": {"expr": "rsi(price, {n})", "n": True, "default_n": 14,
            "help": "Relative strength index, 0 to 100."},
    "Trend efficiency": {"expr": "er(price, {n})", "n": True, "default_n": 20,
                         "help": "How direct the path was, 0 to 1."},
    "Z-score of price": {"expr": "zscore(price, {n})", "n": True, "default_n": 252,
                         "help": "Standard deviations from the mean."},
    "Distance from average": {"expr": "(price / sma(price, {n}) - 1)", "n": True,
                              "default_n": 200,
                              "help": "Percentage above or below the average."},
    "Rolling high": {"expr": "mmax(shift(price, 1), {n})", "n": True, "default_n": 60,
                     "help": "Highest prior close over n sessions."},
    "Rolling low": {"expr": "mmin(shift(price, 1), {n})", "n": True, "default_n": 60,
                    "help": "Lowest prior close over n sessions."},
}

COMPARISONS = {
    "is above": ">",
    "is below": "<",
    "is at least": ">=",
    "is at most": "<=",
}

# What the left side is compared against.
TARGETS = {
    "a number": "value",
    "another indicator": "indicator",
}

SCORE_TRANSFORMS = {
    "Rank across universe (high is good)": "pctrank({expr})",
    "Rank across universe (low is good)": "pctrank(-({expr}))",
    "Raw value": "({expr})",
    "Negative of value": "(-({expr}))",
}


@dataclass
class Rule:
    kind: str = "filter"                 # filter | score
    indicator: str = "Moving average"
    window: int = 200
    comparison: str = "is above"
    target: str = "a number"
    value: float = 0.0
    other_indicator: str = "Price"
    other_window: int = 200
    transform: str = "Rank across universe (high is good)"
    weight: float = 1.0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _indicator_expr(name: str, window: int) -> str:
    spec = INDICATORS.get(name)
    if not spec:
        return "price"
    expr = spec["expr"]
    if spec.get("n"):
        expr = expr.replace("{n}", str(int(window)))
    return expr


def rule_expression(rule: Rule) -> str:
    """One rule as formula text."""
    left = _indicator_expr(rule.indicator, rule.window)
    if rule.kind == "score":
        template = SCORE_TRANSFORMS.get(rule.transform, "({expr})")
        return template.replace("{expr}", left)

    op = COMPARISONS.get(rule.comparison, ">")
    if TARGETS.get(rule.target) == "indicator":
        right = _indicator_expr(rule.other_indicator, rule.other_window)
    else:
        right = repr(float(rule.value))
    return f"({left} {op} {right})"


def compile_rules(rules: List[Rule], joiner: str = "AND") -> Dict[str, str]:
    """Turns a rule list into a filter expression and a score expression.

    Returns {"filter": ..., "score": ...}; either may be empty, which the
    caller treats as "no filter" or "no score".
    """
    active = [r for r in rules if r.enabled]
    filters = [rule_expression(r) for r in active if r.kind == "filter"]
    scores = [(r, rule_expression(r)) for r in active if r.kind == "score"]

    join = " and " if str(joiner).upper() == "AND" else " or "
    filter_expr = join.join(filters) if filters else ""

    if not scores:
        score_expr = ""
    elif len(scores) == 1 and abs(scores[0][0].weight - 1.0) < 1e-9:
        score_expr = scores[0][1]
    else:
        total = sum(abs(r.weight) for r, _ in scores) or 1.0
        parts = [f"{abs(r.weight) / total:.4f} * {expr}" for r, expr in scores]
        score_expr = " + ".join(parts)

    return {"filter": filter_expr, "score": score_expr}


def describe_rule(rule: Rule) -> str:
    """A plain-language line for the rule card."""
    spec = INDICATORS.get(rule.indicator, {})
    left = rule.indicator + (f" ({rule.window})" if spec.get("n") else "")
    if rule.kind == "score":
        return f"{rule.transform} \u2014 {left} \u00b7 weight {rule.weight:g}"
    if TARGETS.get(rule.target) == "indicator":
        other = INDICATORS.get(rule.other_indicator, {})
        right = rule.other_indicator + (f" ({rule.other_window})"
                                        if other.get("n") else "")
    else:
        right = f"{rule.value:g}"
    return f"{left} {rule.comparison} {right}"


def move(rules: List[Rule], index: int, delta: int) -> List[Rule]:
    """Moves one rule up or down, leaving the list valid at the ends."""
    target = index + delta
    if index < 0 or index >= len(rules) or target < 0 or target >= len(rules):
        return rules
    out = list(rules)
    out[index], out[target] = out[target], out[index]
    return out


def from_dicts(raw: List[Dict[str, Any]]) -> List[Rule]:
    """Rebuilds rules from a saved preset, ignoring unknown keys."""
    fields = set(Rule().to_dict())
    out = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append(Rule(**{k: v for k, v in item.items() if k in fields}))
    return out
