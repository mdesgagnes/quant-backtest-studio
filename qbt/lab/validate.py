"""Validation suite: one call, one verdict, every number visible.

`validate_run` takes a finished backtest and runs the statistical battery from
`stats.py` plus the engine-based robustness checks that already exist in
`qbt.robustness` (cost stress, in/out of sample). Each check returns a
status -- pass, warn, fail or n/a -- with its evidence and the threshold it
was held to, and the suite adds them up into a grade.

The grade is a summary of how many independent lines of evidence agree. It is
not a probability of making money. A strategy can pass everything and lose;
one that fails several is not a good candidate for capital however attractive
its backtest looks. Thresholds are conventions (listed in `THRESHOLDS`), not
laws, and the report prints them so a reader can disagree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import stats as S
from .run import Dataset, sweep_returns
from .. import metrics as M
from ..config import CostConfig, EngineConfig
from ..engine import BacktestResult
from ..strategies import get as get_strategy

THRESHOLDS = {
    "psr": (0.95, 0.80),                 # (pass at or above, warn at or above)
    "dsr": (0.95, 0.75),
    "bootstrap_ci_low": (0.0, -0.25),    # lower bound of the Sharpe CI
    "sign_flip_p": (0.05, 0.10),         # pass at or below
    "timing_p": (0.05, 0.15),
    "windows_positive": (0.80, 0.60),    # share of walk-forward windows with Sharpe > 0
    "pbo": (0.20, 0.50),                 # pass at or below
    "cpcv_prob_loss": (0.20, 0.40),
    "cost_x2_sharpe_keep": (0.70, 0.40), # share of Sharpe left at 2x costs
    "oos_sharpe_keep": (0.60, 0.30),     # OOS Sharpe / IS Sharpe
    "min_years": 3.0,
}


@dataclass
class Check:
    key: str
    title: str
    status: str                 # pass | warn | fail | na
    value: Optional[float]
    detail: str
    threshold: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _grade(value: float, thresholds, higher_is_better: bool) -> str:
    good, ok = thresholds
    if value is None or not np.isfinite(value):
        return "na"
    if higher_is_better:
        return "pass" if value >= good else "warn" if value >= ok else "fail"
    return "pass" if value <= good else "warn" if value <= ok else "fail"


def validate_run(ds: Dataset, res: BacktestResult, strategy_key: str,
                 params: Dict[str, Any], engine: EngineConfig, costs: CostConfig,
                 n_trials: int = 1, grid: Optional[Dict[str, List[Any]]] = None,
                 n_boot: int = 1000, n_timing: int = 1000, seed: int = 42,
                 run_cost_stress: bool = True, run_oos: bool = True) -> Dict[str, Any]:
    """Runs the battery. `n_trials` should come from the ledger (all configs
    tried under the hypothesis); `grid` (optional) adds the overfitting checks
    that need many configurations."""
    ppy = engine.periods_per_year
    r = res.returns.dropna()
    checks: List[Check] = []
    years = len(r) / ppy

    # --- sample length ------------------------------------------------
    mtrl = S.min_track_record_length(r, 0.0, 0.95, ppy)
    checks.append(Check(
        "length", "Track record length",
        "pass" if years >= THRESHOLDS["min_years"] and mtrl["enough"]
        else "warn" if years >= 1.5 else "fail",
        years, f"{years:.1f} years of returns; needs {mtrl['min_years']:.1f} to call the Sharpe "
        f"positive at 95%." if np.isfinite(mtrl["min_years"]) else
        f"{years:.1f} years; observed Sharpe is not above zero, so no record length would suffice.",
        f">= {THRESHOLDS['min_years']:.0f} years and >= MinTRL", dict(mtrl)))

    # --- PSR / DSR ------------------------------------------------------
    psr = S.probabilistic_sharpe(r, 0.0, ppy)
    checks.append(Check("psr", "Probabilistic Sharpe (vs 0)",
                        _grade(psr["psr"], THRESHOLDS["psr"], True), psr["psr"],
                        f"P(true Sharpe > 0) = {psr['psr']:.1%} given skew {psr['skew']:.2f}, "
                        f"kurtosis {psr['kurtosis']:.1f}, {psr['n_obs']} obs.",
                        f">= {THRESHOLDS['psr'][0]:.0%} pass, >= {THRESHOLDS['psr'][1]:.0%} warn", psr))

    trial_sharpes = None
    perf = None
    if grid:
        perf = sweep_returns(ds, strategy_key, params, grid, engine, costs)
        if not perf.empty:
            trial_sharpes = [S.annualised_sharpe(perf[c], ppy) for c in perf.columns]
            n_trials = max(n_trials, perf.shape[1])
    dsr = S.deflated_sharpe(r, n_trials, trial_sharpes, ppy)
    checks.append(Check(
        "dsr", f"Deflated Sharpe ({n_trials} trial{'s' if n_trials != 1 else ''})",
        "na" if n_trials < 2 else _grade(dsr["dsr"], THRESHOLDS["dsr"], True),
        dsr["dsr"],
        "Only one configuration was tried, so there is nothing to deflate." if n_trials < 2 else
        f"After {n_trials} trials luck alone is expected to reach a Sharpe of "
        f"{dsr['expected_max_sharpe_pa']:.2f}; this record is {dsr['dsr']:.1%} likely to beat that.",
        f">= {THRESHOLDS['dsr'][0]:.0%} pass, >= {THRESHOLDS['dsr'][1]:.0%} warn", dsr))

    # --- resampling -----------------------------------------------------
    bs = S.bootstrap_sharpe(r, n_boot, 0.95, None, ppy, seed)
    if "error" in bs:
        checks.append(Check("bootstrap", "Bootstrap Sharpe interval", "na", None, bs["error"]))
    else:
        checks.append(Check(
            "bootstrap", "Bootstrap Sharpe interval (95%)",
            _grade(bs["ci_low"], THRESHOLDS["bootstrap_ci_low"], True), bs["ci_low"],
            f"Sharpe {bs['observed']:.2f}, interval [{bs['ci_low']:.2f}, {bs['ci_high']:.2f}] "
            f"(stationary bootstrap, mean block {bs['mean_block']:.0f} days).",
            "lower bound > 0 pass, > -0.25 warn", {k: v for k, v in bs.items() if k != "samples"}))

    sf = S.sign_flip_test(r, 2000, ppy, seed)
    checks.append(Check(
        "sign_flip", "Sign-flip test (mean return = 0)",
        _grade(sf.get("p_value"), THRESHOLDS["sign_flip_p"], False), sf.get("p_value"),
        f"p = {sf['p_value']:.3f}: share of random sign patterns that did as well." if "error" not in sf
        else sf["error"],
        f"p <= {THRESHOLDS['sign_flip_p'][0]} pass, <= {THRESHOLDS['sign_flip_p'][1]} warn", sf))

    tt = S.random_timing_test(res.weights, ds.prices, n_timing, 21, ppy, seed)
    if "error" in tt:
        checks.append(Check("timing", "Random-timing test", "na", None, tt["error"]))
    else:
        checks.append(Check(
            "timing", "Random-timing test",
            _grade(tt["p_value"], THRESHOLDS["timing_p"], False), tt["p_value"],
            f"p = {tt['p_value']:.3f}: same positions rotated in time scored a gross Sharpe of "
            f"{tt['null_mean']:.2f} on average vs {tt['observed_gross']:.2f} actual. "
            f"A high p means the result is mostly exposure (beta), not timing.",
            f"p <= {THRESHOLDS['timing_p'][0]} pass, <= {THRESHOLDS['timing_p'][1]} warn",
            {k: v for k, v in tt.items() if k != "samples"}))

    # --- stability over time ---------------------------------------------
    wf = S.walk_forward_windows(r, 5, ppy)
    if len(wf) >= 3:
        share = float((wf["Sharpe"] > 0).mean())
        checks.append(Check(
            "walk_forward", "Consistency across periods",
            _grade(share, THRESHOLDS["windows_positive"], True), share,
            f"{int((wf['Sharpe'] > 0).sum())} of {len(wf)} equal windows had a positive Sharpe "
            f"(range {wf['Sharpe'].min():.2f} to {wf['Sharpe'].max():.2f}).",
            f">= {THRESHOLDS['windows_positive'][0]:.0%} pass, >= {THRESHOLDS['windows_positive'][1]:.0%} warn",
            {"windows": wf.to_dict("records")}))
    else:
        checks.append(Check("walk_forward", "Consistency across periods", "na", None, "Too short."))

    if run_oos and len(r) > 6 * ppy / 2:
        half = len(r) // 2
        is_sr, oos_sr = S.annualised_sharpe(r.iloc[:half], ppy), S.annualised_sharpe(r.iloc[half:], ppy)
        keep = oos_sr / is_sr if is_sr and is_sr > 0 else float("nan")
        checks.append(Check(
            "oos", "Second half vs first half", _grade(keep, THRESHOLDS["oos_sharpe_keep"], True), keep,
            f"Sharpe {is_sr:.2f} in the first half, {oos_sr:.2f} in the second "
            f"({keep:.0%} retained)." if np.isfinite(keep) else
            f"First-half Sharpe {is_sr:.2f}: nothing to retain.",
            f">= {THRESHOLDS['oos_sharpe_keep'][0]:.0%} pass, >= {THRESHOLDS['oos_sharpe_keep'][1]:.0%} warn",
            {"first_half_sharpe": is_sr, "second_half_sharpe": oos_sr}))

    # --- frictions --------------------------------------------------------
    if run_cost_stress:
        from .run import run_strategy
        from dataclasses import replace
        base_sr = S.annualised_sharpe(r, ppy)
        stressed = replace(costs, commission_bps=costs.commission_bps * 2,
                           slippage_bps=costs.slippage_bps * 2)
        try:
            r2 = run_strategy(ds, strategy_key, params, engine, stressed).returns
            sr2 = S.annualised_sharpe(r2, ppy)
            keep = sr2 / base_sr if base_sr and base_sr > 0 else float("nan")
            checks.append(Check(
                "costs", "Sharpe at double the costs", _grade(keep, THRESHOLDS["cost_x2_sharpe_keep"], True), keep,
                f"Sharpe {base_sr:.2f} -> {sr2:.2f} with commission and slippage doubled "
                f"({keep:.0%} retained)." if np.isfinite(keep) else f"Base Sharpe {base_sr:.2f}.",
                f">= {THRESHOLDS['cost_x2_sharpe_keep'][0]:.0%} pass, >= {THRESHOLDS['cost_x2_sharpe_keep'][1]:.0%} warn",
                {"base": base_sr, "stressed": sr2}))
        except Exception as exc:
            checks.append(Check("costs", "Sharpe at double the costs", "na", None, f"Could not run: {exc}"))

    # --- selection overfitting (needs a grid) ----------------------------
    if perf is not None and perf.shape[1] >= 3:
        pb = S.pbo(perf, 16, ppy)
        if "error" not in pb:
            checks.append(Check(
                "pbo", "Probability of backtest overfitting", _grade(pb["pbo"], THRESHOLDS["pbo"], False), pb["pbo"],
                f"Across {pb['n_configs']} configurations, the best in-sample pick ranked below the median "
                f"out-of-sample in {pb['pbo']:.0%} of {pb['n_combinations']} splits "
                f"(in-sample Sharpe {pb['mean_is_sharpe']:.2f} -> out-of-sample {pb['mean_oos_sharpe']:.2f}).",
                f"<= {THRESHOLDS['pbo'][0]:.0%} pass, <= {THRESHOLDS['pbo'][1]:.0%} warn",
                {k: v for k, v in pb.items() if k not in ("logits", "is_sharpes", "oos_sharpes")}))
        cp = S.cpcv_selection(perf, 10, 2, purge=max(5, int(params.get("lookback", 0) // 20)), ppy=ppy)
        if "error" not in cp:
            checks.append(Check(
                "cpcv", "Combinatorial purged CV", _grade(cp["prob_oos_loss"], THRESHOLDS["cpcv_prob_loss"], False),
                cp["prob_oos_loss"],
                f"Picking the best-in-training configuration delivered a median out-of-sample Sharpe of "
                f"{cp['median_oos_sharpe']:.2f} (10th pct {cp['p10_oos_sharpe']:.2f}); lost money in "
                f"{cp['prob_oos_loss']:.0%} of {cp['n_splits']} splits.",
                f"loss share <= {THRESHOLDS['cpcv_prob_loss'][0]:.0%} pass, <= {THRESHOLDS['cpcv_prob_loss'][1]:.0%} warn",
                {k: v for k, v in cp.items() if k != "oos_sharpes"}))

    return {"checks": checks, "grade": grade_of(checks), "n_trials": n_trials,
            "sweep_returns": perf, "sharpe": S.annualised_sharpe(r, ppy)}


def grade_of(checks: List[Check]) -> Dict[str, Any]:
    """Weighted tally. The statistical-significance checks and the overfitting
    checks carry more weight than the stability checks; one hard fail on a
    heavy check caps the grade."""
    weights = {"psr": 2, "dsr": 3, "bootstrap": 2, "sign_flip": 1, "timing": 2, "walk_forward": 2,
               "oos": 1, "costs": 2, "pbo": 3, "cpcv": 2, "length": 1}
    score = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    num = den = 0.0
    heavy_fail = False
    for c in checks:
        if c.status not in score:
            continue
        w = weights.get(c.key, 1)
        num += w * score[c.status]
        den += w
        if c.status == "fail" and c.key in ("dsr", "pbo", "psr", "bootstrap"):
            heavy_fail = True
    pct = num / den if den else float("nan")
    n_fail = sum(c.status == "fail" for c in checks)
    n_warn = sum(c.status == "warn" for c in checks)
    n_pass = sum(c.status == "pass" for c in checks)
    if not np.isfinite(pct):
        label = "Unrated"
    elif pct >= 0.85 and not heavy_fail:
        label = "Robust"
    elif pct >= 0.65 and not heavy_fail:
        label = "Promising"
    elif pct >= 0.40:
        label = "Fragile"
    else:
        label = "Unsupported"
    return {"label": label, "score": pct, "passes": n_pass, "warnings": n_warn,
            "fails": n_fail, "heavy_fail": heavy_fail}


def checks_table(checks: List[Check]) -> pd.DataFrame:
    return pd.DataFrame([{"Check": c.title, "Status": c.status.upper(), "Evidence": c.detail,
                          "Threshold": c.threshold} for c in checks])
