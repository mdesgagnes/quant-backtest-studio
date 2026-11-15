"""Tables of a review, in the order they are shown.

`sections(review, sweep)` returns every table the page shows and the
workbook holds, as (sheet, title, frame, note). The Streamlit page renders
exactly this list and the Excel export writes exactly this list, so a
metric cannot appear in one and be missing from the other.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AllConfig
from .engine import Review

Section = Tuple[str, str, pd.DataFrame, str]

SHEETS = ["Verdict", "Ranking", "Fund Review", "Sleeve Comparison", "Diversification",
          "Portfolio Impact", "Scenarios", "Regimes", "Factor Exposures", "Robustness",
          "Profile Sweep", "Methodology", "Series"]

PCT = '0.00%;-0.00%;0.00%'
PTS = '+0.00%;-0.00%;0.00%'
RATIO = '0.00'
INT = '0'
MONEY = '#,##0'

_PCT_ROWS = {"CAGR", "Volatility", "Max drawdown", "CVaR 95% (monthly)", "Worst month",
             "Worst 12 months", "Risk-free (annual)", "Distribution yield (trailing 12m)",
             "Expense ratio"}
_MONEY_ROWS = {"Assets (AUM)", "Average daily value traded"}
_INT_ROWS = {"Months", "Longest underwater (months)", "Recovery from max drawdown (months)"}


def _cols(df: pd.DataFrame, pct=(), pts=(), ratio=(), integer=()) -> pd.DataFrame:
    fmt = {}
    for names, f in ((pct, PCT), (pts, PTS), (ratio, RATIO), (integer, INT)):
        for n in names:
            if n in df.columns:
                fmt[n] = f
    df.attrs["col_fmt"] = fmt
    return df


def _rows(df: pd.DataFrame) -> pd.DataFrame:
    """Row-oriented table (measures as rows, in the index): the number
    format follows the row. The index becomes the first column."""
    fmt = [PCT if i in _PCT_ROWS else INT if i in _INT_ROWS else MONEY if i in _MONEY_ROWS else
           (RATIO if i not in ("Start", "End") else None) for i in df.index]
    out = df.reset_index()
    out.attrs["row_fmt"] = fmt
    return out


def _kv(d: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({"Item": list(d), "Value": [str(v) for v in d.values()]})


# Catalogue metrics shown as percentage-point changes / percentages.
_PTS_METRICS = {"d_vol", "d_cagr", "d_mdd", "d_cvar", "d_worst_12m", "d_tail_return",
                "d_up_month_return", "d_portfolio_yield", "d_underwater_share",
                "d_prob_dd20", "d_prob_dd30", "d_depletion_prob"}
_PCT_METRICS = {"cand_tail_return", "dist_yield"}


def _metric_fmt(key: str) -> str:
    if key in _PTS_METRICS or key.startswith(("d_regime_", "d_scen_")):
        return PTS
    if key in _PCT_METRICS or key.startswith("cand_regime_"):
        return PCT
    return RATIO


def verdict_sections(rv: Review, cfg: AllConfig) -> List[Section]:
    req = rv.request
    mode = (f"Add {rv.weight:.0%}, funded pro rata from every holding" if req.mode == "add"
            else f"Replace {req.replace_holding} ({rv.profile.weights.get(req.replace_holding, 0):.0%})")
    info = {"Profile": rv.profile.describe(), "Objective": rv.objective_label,
            "What it asks": rv.objective_description, "Mode": mode,
            "Rebalancing": req.rebalance or rv.profile.rebalance,
            "Analysis period": rv.period,
            "Candidates analysed": ", ".join(rv.results) or "none",
            "Skipped": "; ".join(f"{k}: {v}" for k, v in rv.skipped.items()) or "none"}
    out: List[Section] = [("Verdict", "Request", _kv(info), "")]
    rows = []
    for c, r in rv.results.items():
        d = r.decision
        rows.append({"Candidate": c, "Verdict": d["verdict"], "Confidence": d["confidence"],
                     "Score": d["composite"], "Benchmark": r.benchmark_label, "Months": r.months,
                     "Deciding facts": " | ".join(r.rationale["facts"]),
                     "Main trade-off": r.rationale["tradeoff"],
                     "Caveats": " | ".join(r.rationale["caveats"]) or "none"})
    out.append(("Verdict", "Verdict and rationale", _cols(pd.DataFrame(rows), ratio=["Score"],
                                                          integer=["Months"]), ""))
    if rv.results:
        tab = {}
        for c, r in rv.results.items():
            o = r.objectives.set_index("Objective")
            tab[c] = [f"{v} ({s:+.2f}, {cf})" if s == s else v
                      for v, s, cf in zip(o["Verdict"], o["Score"], o["Confidence"])]
        df = pd.DataFrame(tab, index=next(iter(rv.results.values())).objectives["Objective"])
        df.index.name = "Objective"
        out.append(("Verdict", "Verdict for every objective (verdict, score, confidence)",
                    df.reset_index(), "Same candidates and profile; only the objective changes."))
    eq = pd.DataFrame(rv.equivalents)
    if len(eq):
        eq["Conclusion"] = "Functionally equivalent: the choice comes down to cost or liquidity."
        out.append(("Verdict", "Functionally equivalent pairs",
                    _cols(eq, pct=["Tracking error between them", "Expense ratio A",
                                   "Expense ratio B"], integer=["Months"]),
                    "Expense ratio and trading volume from yfinance."))
    return out


def ranking_sections(rv: Review) -> List[Section]:
    out = [("Ranking", f"Ranking for: {rv.objective_label}",
            _cols(rv.ranking.copy(), ratio=["Score", "Unclipped score"], integer=["Rank", "Months"],
                  pct=["Expense ratio"]),
            "Composite score in [-1, +1]; ties broken by the unclipped score.")]
    if len(rv.components):
        comp = rv.components.copy()
        comp["Value shown"] = [_fmt_value(k, v) for k, v in zip(comp["Metric"], comp["Value"])]
        comp = comp[["Candidate", "Metric", "Label", "Value shown", "Value", "Weight", "Score",
                     "Contribution"]]
        out.append(("Ranking", "Score breakdown by component",
                    _cols(comp, pct=["Weight"], ratio=["Score", "Contribution", "Value"]), ""))
    return out


def _fmt_value(k: str, v: float) -> str:
    from .verdict import fmt_metric
    return fmt_metric(k, v)


def fund_review_sections(rv: Review) -> List[Section]:
    if rv.fund_review.empty:
        return []
    df = rv.fund_review.copy()
    df.index.name = "Measure"
    return [("Fund Review", "Standalone profile, all tickers side by side", _rows(df),
             "Each candidate over its analysis period, in CAD.")]


def sleeve_sections(rv: Review) -> List[Section]:
    rows = []
    for c, r in rv.results.items():
        row = {"Candidate": c, "Benchmark": r.benchmark_label}
        row.update(r.relative)
        rows.append(row)
    if not rows:
        return []
    df = _cols(pd.DataFrame(rows), pct=["Active return", "Tracking error", "Relative drawdown",
                                        "Hit rate in benchmark down months"],
               ratio=["Information ratio", "Beta", "Correlation", "Upside capture",
                      "Downside capture", "Upside beta", "Downside beta"])
    return [("Sleeve Comparison", "Within-sleeve statistics vs the sleeve benchmark", df,
             "Monthly, CAD. Capture ratios: geometric mean in benchmark up (down) months over the "
             "benchmark's.")]


def diversification_sections(rv: Review) -> List[Section]:
    out = []
    for title, m, note in (
            ("Correlation of monthly returns", rv.corr, "Includes the profile without the candidate."),
            ("Excess-return correlation (each vs its sleeve benchmark)", rv.excess_corr,
             "Separates same-sleeve funds by their active bets."),
            ("Crisis correlation (portfolio's worst 10% months)", rv.crisis_corr,
             "Correlation measured only in the profile's worst decile of months.")):
        if m is not None and len(m):
            df = m.copy()
            df.index.name = "Ticker"
            df = df.reset_index()
            out.append(("Diversification", title, _cols(df, ratio=list(m.columns)), note))
    return out


def impact_sections(rv: Review, cfg: AllConfig) -> List[Section]:
    out = []
    cat = cfg.objectives.metrics
    rows = []
    for c, r in rv.results.items():
        for k, v in r.metrics.items():
            rows.append({"Candidate": c, "Metric": k,
                         "Label": cat[k].label if k in cat else k.replace("_", " "),
                         "Value": v, "Shown as": _fmt_value(k, v)})
    if rows:
        out.append(("Portfolio Impact", "All metrics (portfolio with minus without, and candidate measures)",
                    _cols(pd.DataFrame(rows), ratio=["Value"]), "d_ = with minus without."))
    stats = []
    for c, r in rv.results.items():
        for name in ("CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar",
                     "Ulcer index", "CVaR 95% (monthly)", "Worst month", "Worst 12 months",
                     "Longest underwater (months)"):
            a, b = r.base_stats.get(name, np.nan), r.new_stats.get(name, np.nan)
            stats.append({"Candidate": c, "Measure": name, "Without": a, "With": b,
                          "Change": b - a if a == a and b == b else np.nan})
    if stats:
        df = pd.DataFrame(stats)
        df.attrs["row_fmt"] = [PCT if m in _PCT_ROWS else INT if m in _INT_ROWS else RATIO
                               for m in df["Measure"]]
        df.attrs["fmt_cols"] = [3, 4, 5]
        out.append(("Portfolio Impact", "Profile without and with the candidate", df, ""))
    sims = []
    for c, r in rv.results.items():
        if r.simulation:
            t = r.simulation["table"].copy()
            t.insert(0, "Candidate", c)
            sims.append(t)
    if sims:
        df = pd.concat(sims, ignore_index=True)
        df.attrs["row_fmt"] = [RATIO if "wealth" in m.lower() else PCT for m in df["Measure"]]
        df.attrs["fmt_cols"] = [3, 4, 5]
        s = next(r.simulation["settings"] for r in rv.results.values() if r.simulation)
        out.append(("Portfolio Impact", "Risk of ruin and decumulation (block bootstrap)", df,
                    f"{s['paths']} paths, {s['block']}-month blocks, seed {s['seed']}, resampled "
                    f"from {s['history_months']} months of history; same paths with and without."))
    wrows = []
    for c, r in rv.results.items():
        for t, w in sorted(r.weights_new.items(), key=lambda x: -x[1]):
            wrows.append({"Candidate": c, "Holding": t, "Weight without": rv.profile.weights.get(t, 0.0),
                          "Weight with": w})
    if wrows:
        out.append(("Portfolio Impact", "Target weights without and with the candidate",
                    _cols(pd.DataFrame(wrows), pct=["Weight without", "Weight with"]), ""))
    return out


def _stack(rv: Review, attr: str) -> pd.DataFrame:
    parts = []
    for c, r in rv.results.items():
        t = getattr(r, attr)
        if t is not None and len(t):
            t = t.copy()
            t.insert(0, "Candidate", c)
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def scenario_sections(rv: Review) -> List[Section]:
    df = _stack(rv, "stress")
    if df.empty:
        return []
    pct = [c for c in df.columns if "return" in c or "drawdown" in c]
    return [("Scenarios", "Historical stress windows",
             _cols(df, pct=pct, ratio=["Candidate correlation to portfolio (daily)"],
                   integer=["Sessions"]),
             "Close of start date to close of end date, daily, CAD; portfolios bought at target "
             "weights on the start date and left to drift. 'no data' = history does not cover the "
             "window (never filled).")]


def regime_sections(rv: Review) -> List[Section]:
    df = _stack(rv, "regimes")
    if df.empty:
        return []
    pct = [c for c in df.columns if "return" in c or "volatility" in c or "Contribution" in c]
    return [("Regimes", "Regimes (monthly)", _cols(df, pct=pct, integer=["Months"]),
             "Rules in config/scenarios.yaml. Contribution = weight x candidate average return.")]


def factor_sections(rv: Review, cfg: AllConfig) -> List[Section]:
    rows = []
    for c, r in rv.results.items():
        f = r.factor
        row = {"Candidate": c, "Months": f.get("n"), "R2": f.get("r2")}
        if f.get("ok"):
            for k, v in f["coef"].items():
                row[f"{k} loading" if k != "alpha" else "Alpha (annual)"] = v
            for k, v in f["t"].items():
                row[f"{k} t-stat"] = v
        else:
            row["Note"] = "too few overlapping months for the regression"
        row["Mega-cap tilt (active, vs QQQ-RSP)"] = r.metrics.get("megacap_tilt")
        rows.append(row)
    if not rows:
        return []
    df = pd.DataFrame(rows)
    ratio = [c for c in df.columns if c not in ("Candidate", "Months", "Note", "Alpha (annual)")]
    labels = {k: v.get("label", k) for k, v in (cfg.factors.get("factors") or {}).items()}
    src = next((n for n in rv.notes if n.startswith("Factor model:")), "")
    note = ("Explanatory only. Excess return on the sleeve benchmark's excess return and US "
            "long-short style factors. " + src + " "
            + ", ".join(f"{k}: {v}" for k, v in labels.items()) + ".")
    return [("Factor Exposures", "Factor / style regression", _cols(df, pct=["Alpha (annual)"],
                                                                   ratio=ratio, integer=["Months"]), note)]


def robustness_sections(rv: Review, cfg: AllConfig) -> List[Section]:
    out = []
    rows = []
    obj = cfg.objectives.objectives[rv.objective_key]
    for c, r in rv.results.items():
        for key, res in r.rolling.items():
            rows.append({"Candidate": c, "Rolling metric": key,
                         "Used by this objective": "yes" if key == obj.rolling_metric else "",
                         "Windows": res["windows"], "Share of windows improved": res["share"]})
    if rows:
        out.append(("Robustness", f"Rolling {cfg.objectives.rolling_window}-month windows",
                    _cols(pd.DataFrame(rows), pct=["Share of windows improved"], integer=["Windows"]),
                    "Step one month. 'Improved' = on the good side of the metric's centre."))
    w = _stack(rv, "weight_grid")
    if len(w):
        out.append(("Robustness", "Add-weight sensitivity",
                    _cols(w, pct=["Add weight"], pts=["Change in volatility", "Change in CAGR",
                                                      "Change in max drawdown"],
                          ratio=["Score", "Change in Sharpe"]), ""))
    cons = [{"Candidate": c, **r.consistency} for c, r in rv.results.items()]
    if cons:
        df = pd.DataFrame(cons)
        out.append(("Robustness", "Consistency across scenarios and regimes",
                    _cols(df, pct=[c for c in df.columns if "where" in c],
                          integer=["Stress windows covered"]), ""))
    fl = [{"Candidate": c, "Flag": f} for c, r in rv.results.items() for f in r.flags]
    fl += [{"Candidate": c, "Flag": f"Skipped: {why}"} for c, why in rv.skipped.items()]
    if fl:
        out.append(("Robustness", "Data quality and confidence flags", pd.DataFrame(fl), ""))
    return out


def sweep_sections(sweep: Optional[pd.DataFrame]) -> List[Section]:
    if sweep is None or sweep.empty:
        return []
    return [("Profile Sweep", "One candidate across every preset profile",
             _cols(sweep.drop(columns=["Key"], errors="ignore"),
                   pts=["Change in volatility", "Change in CAGR", "Change in max drawdown"],
                   ratio=["Score", "Change in Sharpe", "Correlation to portfolio"],
                   integer=["Months"]), "")]


def methodology_sections(rv: Review, cfg: AllConfig) -> List[Section]:
    out = [("Methodology", "Data, period and assumptions",
            pd.DataFrame({"Note": rv.notes + [f"Analysis period: {rv.period}."]}), "")]
    out.append(("Methodology", "Profile holdings and proxy splices",
                _cols(rv.holdings.copy(), pct=["Weight"], integer=["Spliced months"]),
                "Before a holding's own first monthly return, its proxy's CAD return is used."))
    obj = cfg.objectives.objectives[rv.objective_key]
    cat = cfg.objectives.metrics
    rows = [{"Metric": k, "Label": cat[k].label, "Weight": w / sum(obj.weights.values()),
             "Better": cat[k].better, "Centre": cat[k].center, "Scale (full credit)": cat[k].scale}
            for k, w in obj.weights.items()]
    out.append(("Methodology", f"Objective definition: {obj.label}",
                _cols(pd.DataFrame(rows), pct=["Weight"]),
                f"Add if score >= {obj.add:+.2f}; Consider if >= {obj.consider:+.2f}; else Do not add. "
                "score_i = clip(direction x (value - centre) / scale, -1, 1). Guardrails: "
                + ("; ".join(f"{g['metric']} >= {g.get('min')}" for g in obj.guardrails) or "none")
                + f". Rolling metric: {obj.rolling_metric}; Add is capped at Consider below "
                  f"{cfg.objectives.min_rolling_share:.0%} of windows."))
    defs = [
        ("CAGR", "Annualised geometric return, (prod(1+r))^(12/n) - 1."),
        ("Volatility", "Population standard deviation of monthly returns x sqrt(12)."),
        ("Sharpe / Sortino", "Excess of annualised return over the CAD T-bill, over volatility / downside deviation."),
        ("Max drawdown", "Worst fall from a previous peak of the month-end value path."),
        ("Calmar", "CAGR / |max drawdown|."),
        ("Ulcer index", "Root mean square of drawdowns, in percent."),
        ("CVaR 95%", "Average of the worst 5% of monthly returns."),
        ("Tracking error", "Population SD of geometric excess returns x sqrt(12)."),
        ("Capture", "Geometric mean return in benchmark up (down) months over the benchmark's."),
        ("Diversification ratio", "Weighted average holding volatility over portfolio volatility."),
        ("Worst 10% months", "The profile's (without candidate) worst decile of months on the analysis period."),
        ("Confidence", "High; Moderate below 60 months; Low below 36; one level down for weak rolling "
                       "consistency or a verdict that flips with the add weight."),
    ]
    out.append(("Methodology", "Definitions", pd.DataFrame(defs, columns=["Term", "Definition"]),
                "Full detail: docs/METHODOLOGY.md."))
    return out


def series_sections(rv: Review) -> List[Section]:
    parts = []
    for c, r in rv.results.items():
        s = r.series.copy()
        s.columns = [f"{c}: {x}" if x != c else f"{c}: candidate" for x in s.columns]
        parts.append(s)
    if not parts:
        return []
    df = pd.concat(parts, axis=1).sort_index()
    df.index = df.index.strftime("%Y-%m")
    df.index.name = "Month"
    return [("Series", "Monthly returns used (CAD)",
             _cols(df.reset_index(), pct=list(df.columns)), "")]


def sections(rv: Review, cfg: AllConfig, sweep: Optional[pd.DataFrame] = None) -> List[Section]:
    return (verdict_sections(rv, cfg) + ranking_sections(rv) + fund_review_sections(rv)
            + sleeve_sections(rv) + diversification_sections(rv) + impact_sections(rv, cfg)
            + scenario_sections(rv) + regime_sections(rv) + factor_sections(rv, cfg)
            + robustness_sections(rv, cfg) + sweep_sections(sweep)
            + methodology_sections(rv, cfg) + series_sections(rv))
