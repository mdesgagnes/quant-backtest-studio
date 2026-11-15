"""The Security review decision engine: one pure entry point, `run`.

    review = run(Request(profile="balanced", objective="lower_volatility",
                         candidates=["USMV", "SPLV", "QUAL"]), cfg, bundle)

Steps for each candidate, on the analysis period (see `alignment`):
standalone profile -> within-sleeve statistics vs the sleeve benchmark ->
factor exposures -> portfolio impact (Add at a weight funded pro rata, or
Replace a holding) -> risk of ruin and decumulation -> stress windows and
regimes -> robustness (rolling windows, add-weight grid) -> verdict for
every objective, with the requested objective first.

Nothing here draws or reads the interface; the Streamlit page, the Excel
export and the scenario-matrix script all call `run`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import impact as I
from . import scenarios as SC
from . import stats as S
from . import verdict as V
from .bundle import Bundle
from .config import AllConfig, ConfigError, Profile
from .portfolio import add_weights, fund_weights, portfolio_returns, replace_weights

CURRENCY_NOTE = ("All returns in CAD. US-listed prices are converted at each day's USD/CAD "
                 "close (Yahoo CAD=X; Fed H.10 history before 2003-09-17); CAD-listed "
                 "securities are untouched (a CAD-listed fund holding foreign assets keeps "
                 "whatever hedging it has).")
SOURCE_NOTE = ("Total returns from Yahoo Finance 'Adj Close' (distributions and splits "
               "reinvested), month-end to month-end.")


@dataclass
class Request:
    profile: Union[str, Profile]
    objective: Optional[str]                 # None: impact only, no verdict shown
    candidates: List[str]
    # Benchmark for the relative statistics (capture, tracking error, factor
    # regression...): None = the portfolio itself (default, so every figure
    # is measured against what the security is added to); "sleeve" = each
    # candidate's sleeve benchmark from config/universe.yaml; a ticker = that
    # ticker for every candidate.
    benchmark: Optional[str] = None
    mode: str = "add"                        # add | replace
    weight: Optional[float] = None           # add weight; default from objectives.yaml
    replace_holding: Optional[str] = None
    rebalance: Optional[str] = None
    alignment: str = "common"                # common | each
    robustness: bool = True
    rolling_step: int = 1
    # Add mode: where the add weight comes from. None = every holding, pro
    # rata. Otherwise a list of holdings and / or groups ("bonds",
    # "equities"), reduced pro rata to their weights; the rest is untouched.
    fund_from: Optional[List[str]] = None


@dataclass
class CandidateResult:
    ticker: str
    benchmark: Optional[str]
    benchmark_label: str
    months: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    weights_new: Dict[str, float]
    metrics: Dict[str, float]
    decision: Dict[str, Any]
    rationale: Dict[str, Any]
    objectives: pd.DataFrame                 # verdict for every objective
    standalone: Dict[str, float]
    relative: Dict[str, float]
    factor: Dict[str, Any]
    regimes: pd.DataFrame
    stress: pd.DataFrame
    simulation: Optional[Dict[str, Any]]
    rolling: Dict[str, Any]
    weight_grid: pd.DataFrame
    consistency: Dict[str, float]
    flags: List[str]
    base_stats: Dict[str, float]
    new_stats: Dict[str, float]
    series: pd.DataFrame                     # candidate, benchmark, base, new (monthly)
    costs: Dict[str, float] = field(default_factory=dict)   # expense ratio, AUM, volume
    vs_base: Dict[str, float] = field(default_factory=dict)  # beta, capture... against the base
    # Weighted MER and distribution yield of the portfolio, without and with
    # the security: {"MER without", "MER with", "Yield without", "Yield with",
    # "MER missing", "Yield missing"}.
    fund_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Review:
    request: Request
    profile: Profile
    objective_key: str
    objective_label: str
    objective_description: str
    weight: float
    holdings: pd.DataFrame
    splices: pd.DataFrame
    results: Dict[str, CandidateResult]
    skipped: Dict[str, str]
    ranking: pd.DataFrame
    components: pd.DataFrame
    corr: pd.DataFrame
    excess_corr: pd.DataFrame
    crisis_corr: pd.DataFrame
    equivalents: List[Dict[str, Any]]
    fund_review: pd.DataFrame
    notes: List[str]
    period: str
    rf_label: str
    cpi_label: str
    objective_chosen: bool = True            # False: the verdict is computed but not the point
    rf: Optional[pd.Series] = None           # monthly risk-free returns used (CAD T-bill)
    funded_from: Optional[List[str]] = None  # holdings reduced to pay for the addition


# ----------------------------------------------------------------------
GROUPS = {"bonds": "bond", "equities": "equity"}
GROUP_LABEL = {"bonds": "bonds", "equities": "equities"}


def asset_class(cfg: AllConfig, t: str) -> Optional[str]:
    """'equity', 'bond' or None: the building block's asset class, else the
    universe sleeve's (a sleeve whose key names bonds or equity)."""
    blk = cfg.profiles.block_for(t)
    if blk is not None:
        return blk.asset_class
    sl = next((k for k, s in cfg.universe.sleeves.items() if t in s.tickers), "")
    return "bond" if "bond" in sl else "equity" if "equity" in sl else None


def funding_sources(cfg: AllConfig, prof: Profile, fund_from: Optional[List[str]]) -> Optional[List[str]]:
    """The holdings of `prof` that pay for the addition (None = all, pro rata)."""
    if not fund_from:
        return None
    out = []
    for f in fund_from:
        g = GROUPS.get(str(f).lower())
        if g:
            out += [t for t in prof.weights if asset_class(cfg, t) == g]
        elif str(f).upper() in prof.weights:
            out.append(str(f).upper())
    return list(dict.fromkeys(out))


def funding_label(fund_from: Optional[List[str]], sources: Optional[List[str]] = None) -> str:
    if not fund_from:
        return "funded pro rata from every holding"
    names = [GROUP_LABEL.get(str(f).lower(), str(f).upper()) for f in fund_from]
    held = f" ({', '.join(sources)})" if sources and any(str(f).lower() in GROUPS for f in fund_from) else ""
    return "taken from " + " and ".join(names) + held


def resolve_profile(cfg: AllConfig, p: Union[str, Profile]) -> Profile:
    if isinstance(p, Profile):
        return p
    if p not in cfg.profiles.profiles:
        raise ConfigError(f"Unknown profile '{p}'.")
    return cfg.profiles.profiles[p]


def holding_returns(cfg: AllConfig, b: Bundle, weights: Dict[str, float]):
    cols, rows = {}, []
    for t in weights:
        block = cfg.profiles.block_for(t)
        proxy = block.proxy if block else None
        if not b.has(t) and not (proxy and b.has(proxy)):
            raise ConfigError(f"No data for holding {t}.")
        s, flag, used = b.block_monthly(t, proxy)
        cols[t] = s
        sp = flag[flag]
        rows.append({"Holding": t, "Block": block.label if block else "(custom)",
                     "Proxy": used or "", "Spliced from": sp.index[0].strftime("%Y-%m") if len(sp) else "",
                     "Spliced to": sp.index[-1].strftime("%Y-%m") if len(sp) else "",
                     "Spliced months": int(len(sp)),
                     "Own history from": b.monthly[t].dropna().index[0].strftime("%Y-%m")
                     if b.has(t) else "none"})
    return pd.DataFrame(cols), pd.DataFrame(rows)


def profile_series(cfg: AllConfig, b: Bundle, key: str, rebalance: Optional[str] = None) -> pd.Series:
    p = cfg.profiles.profiles[key]
    R, _ = holding_returns(cfg, b, p.weights)
    return portfolio_returns(R, p.weights, rebalance or p.rebalance)


def regime_labels(cfg: AllConfig, b: Bundle) -> Dict[str, pd.Series]:
    rules = cfg.scenarios["regimes"]
    inputs: Dict[str, Optional[pd.Series]] = {}
    for k, r in rules.items():
        if r.get("series"):
            inputs[k] = b.raw_px.get(str(r["series"]).upper())
        elif r.get("rule") == "cpi_yoy":
            inputs[k] = b.cpi
    eq = [k for k, p in cfg.profiles.profiles.items() if p.equity == 1.0]
    bd = [k for k, p in cfg.profiles.profiles.items() if p.bonds == 1.0]
    if eq and bd:
        inputs["equity"] = profile_series(cfg, b, eq[0], "monthly")
        inputs["bonds"] = profile_series(cfg, b, bd[0], "monthly")
    return SC.build_regimes(rules, inputs)


def _ctx(cfg: AllConfig, b: Bundle, regimes: Dict[str, pd.Series]) -> I.Context:
    f = cfg.factors
    mc = f.get("megacap_spread") or {}
    spread = None
    if mc and b.has(mc["long"]) and b.has(mc["short"]):
        spread = (b.usd_monthly[mc["long"]] - b.usd_monthly[mc["short"]]).dropna()
    home = b.monthly[f["home_equity"]] if b.has(f.get("home_equity", "")) else None
    usdm = b.usd_monthly[f["usd_market"]] if f.get("usd_market") in b.usd_monthly else None
    mkt = b.monthly[f["usd_market"]] if b.has(f.get("usd_market", "")) else None
    return I.Context(b.rf, regimes, home, usdm, b.usdcad_m if len(b.usdcad_m) else None,
                     mkt, spread, b.yields, cfg.objectives.simulation)


def factor_sources(cfg: AllConfig, b: Bundle) -> Dict[str, str]:
    """Where each style factor comes from: a Ken French column, or an ETF spread."""
    kfmap = cfg.factors.get("ken_french") or {}
    use_kf = b.kf is not None and str(cfg.factors.get("source", "auto")) != "etf_proxies"
    out = {}
    for k, spec in (cfg.factors.get("factors") or {}).items():
        if use_kf and k in kfmap and kfmap[k] in b.kf.columns:
            out[k] = f"Ken French {kfmap[k]}"
        else:
            out[k] = f"{spec['long']} - {spec['short']}"
    return out


def factor_frame(cfg: AllConfig, b: Bundle) -> pd.DataFrame:
    """Monthly style factors: Ken French columns where mapped and reachable,
    ETF long-short spreads otherwise (always for low volatility)."""
    kfmap = cfg.factors.get("ken_french") or {}
    cols = {}
    for k, src in factor_sources(cfg, b).items():
        if src.startswith("Ken French"):
            cols[k] = b.kf[kfmap[k]]
        else:
            spec = cfg.factors["factors"][k]
            if spec["long"] in b.usd_monthly and spec["short"] in b.usd_monthly:
                cols[k] = b.usd_monthly[spec["long"]] - b.usd_monthly[spec["short"]]
    return pd.DataFrame(cols)


def _decide_all(cfg: AllConfig, metrics: Dict[str, float], months: int,
                rolling: Dict[str, Dict[str, Any]], wverdicts: Dict[str, Dict[float, str]],
                coverage: float = 1.0):
    rows, decs = [], {}
    oc = cfg.objectives
    for k, o in oc.objectives.items():
        rs = rolling.get(o.rolling_metric or "", {}).get("share", np.nan)
        d = V.decide(o, metrics, oc.metrics, months, rs, oc.min_rolling_share, wverdicts.get(k),
                     coverage, oc.min_stress_coverage)
        decs[k] = d
        rows.append({"Objective": o.label, "Key": k, "Verdict": d["verdict"],
                     "Confidence": d["confidence"], "Score": d["composite"],
                     "Rolling share": rs})
    return pd.DataFrame(rows), decs


def _period(series: List[pd.Series]) -> pd.DatetimeIndex:
    idx = None
    for s in series:
        i = s.dropna().index
        idx = i if idx is None else idx.intersection(i)
    return idx if idx is not None else pd.DatetimeIndex([])


def run(req: Request, cfg: AllConfig, b: Bundle,
        regimes: Optional[Dict[str, pd.Series]] = None) -> Review:
    oc = cfg.objectives
    if req.objective is not None and req.objective not in oc.objectives:
        raise ConfigError(f"Unknown objective '{req.objective}'.")
    # With no objective, every objective is still scored (Verdict sheet) and the
    # first one stands in for the internal ranking; the page does not show it.
    okey = req.objective or next(iter(oc.objectives))
    obj = oc.objectives[okey]
    prof = resolve_profile(cfg, req.profile)
    reb = req.rebalance or prof.rebalance
    weight = float(req.weight if req.weight is not None else oc.add_weight)
    if req.mode not in ("add", "replace"):
        raise ConfigError("Mode must be add or replace.")
    if req.mode == "replace" and req.replace_holding not in prof.weights:
        raise ConfigError(f"Replace mode needs a holding of {prof.label}: "
                          f"{', '.join(prof.weights)}.")
    sources = funding_sources(cfg, prof, req.fund_from) if req.mode == "add" else None
    if req.mode == "add" and req.fund_from:
        if not sources:
            raise ConfigError(f"{prof.describe()} holds none of: {', '.join(req.fund_from)}.")
        avail = sum(prof.weights[t] for t in sources)
        if weight > avail + 1e-12:
            raise ConfigError(f"{prof.describe()} holds only {avail:.0%} in "
                              f"{', '.join(sources)}: it cannot fund {weight:.0%}.")

    def with_weights(c: str, w: float) -> Dict[str, float]:
        if req.mode == "replace":
            return replace_weights(prof.weights, req.replace_holding, c)
        return fund_weights(prof.weights, c, w, sources) if sources else add_weights(prof.weights, c, w)
    regimes = regimes if regimes is not None else regime_labels(cfg, b)
    ctx = _ctx(cfg, b, regimes)
    F = factor_frame(cfg, b)
    H, holdings = holding_returns(cfg, b, prof.weights)
    holdings.insert(1, "Weight", [prof.weights[t] for t in holdings["Holding"]])
    notes = [SOURCE_NOTE, CURRENCY_NOTE, f"Risk-free: {b.rf_label}.",
             f"Inflation regime source: {b.cpi_label}.", f"Rebalancing: {reb}.",
             f"Factor model: {b.kf_label or 'ETF proxies'}; "
             + "; ".join(f"{k} = {v}" for k, v in factor_sources(cfg, b).items()) + ".",
             "Expense ratios, assets and trading volume: yfinance (fund_operations, info), "
             "cached 30 days; shown and used to break ties between equivalent funds, not scored "
             "(adjusted prices are already net of fees)."] + list(b.notes)

    skipped: Dict[str, str] = {}
    cands = []
    for c in dict.fromkeys(t.strip().upper() for t in req.candidates if t.strip()):
        if not b.has(c):
            skipped[c] = "no price data (download failed or unknown ticker)"
        else:
            cands.append(c)

    bmode = (req.benchmark or "portfolio").strip()
    bench_of: Dict[str, Optional[str]] = {}
    for c in cands:
        if bmode.lower() == "portfolio":
            bt = None
        elif bmode.lower() == "sleeve":
            bt = (cfg.universe.benchmark_for(c) or "").upper() or None
        else:
            bt = bmode.upper()
        if bt and not b.has(bt):
            notes.append(f"Benchmark {bt} for {c} has no data: the portfolio is used instead.")
            bt = None
        bench_of[c] = bt
    notes.append("Benchmark for relative statistics: "
                 + ("the portfolio the security is added to." if bmode.lower() == "portfolio" else
                    "each security's sleeve benchmark (config/universe.yaml)." if bmode.lower() == "sleeve"
                    else f"{bmode.upper()}."))

    def bench_series(c):
        return b.monthly[bench_of[c]] if bench_of[c] else None

    hold_ok = H.dropna().index
    common = _period([b.monthly[c] for c in cands] +
                     [bench_series(c) for c in cands if bench_of[c]]
                     + [pd.Series(1.0, index=hold_ok)]) if cands else pd.DatetimeIndex([])

    # Daily price levels for the stress windows (holdings spliced, candidates raw).
    daily = {t: b.block_daily(t, (cfg.profiles.block_for(t).proxy if cfg.profiles.block_for(t) else None))
             for t in prof.weights}
    windows = cfg.scenarios["windows"]

    results: Dict[str, CandidateResult] = {}
    base_by_c: Dict[str, pd.Series] = {}
    for c in cands:
        if req.alignment == "common":
            P = common
        else:
            parts = [b.monthly[c], pd.Series(1.0, index=hold_ok)]
            if bench_of[c]:
                parts.append(bench_series(c))
            P = _period(parts)
        if len(P) < 24:
            skipped[c] = f"only {len(P)} overlapping months (24 needed)"
            continue
        cand = b.monthly[c].reindex(P)
        R = H.reindex(P).copy()
        R[c] = cand
        new_w = with_weights(c, weight)
        wt = weight if req.mode == "add" else prof.weights[req.replace_holding]
        base = portfolio_returns(R, prof.weights, reb)
        new = portfolio_returns(R, new_w, reb)
        bench = bench_series(c).reindex(P) if bench_of[c] else base
        base_by_c[c] = base
        imp = I.impact_metrics(cand, base, new, R, prof.weights, new_w, bench, ctx, c)
        metrics = imp["metrics"]
        st = SC.stress_table(c, {**daily, c: b.daily_px[c]}, prof.weights, new_w, windows,
                             b.streams)
        for _, r in st.iterrows():
            metrics[f"d_scen_{r['Key']}"] = r.get("Change in portfolio return", np.nan) \
                if r.get("Status") == "ok" else np.nan
        reg = SC.regime_table(cand, base, new, wt, regimes, cfg.scenarios["regimes"])
        fe = S.factor_exposure(cand, bench, F, b.rf, int(cfg.factors.get("min_months", 36)))
        if fe.get("ok") and "size" in fe["coef"]:
            metrics["size_loading"] = float(fe["coef"]["size"])
        else:
            metrics["size_loading"] = np.nan

        rolling: Dict[str, Dict[str, Any]] = {}
        wgrid_rows, wverdicts = [], {}
        if req.robustness:
            for key in dict.fromkeys(o.rolling_metric for o in oc.objectives.values() if o.rolling_metric):
                spec = oc.metrics.get(key)
                if spec is None:
                    continue
                rolling[key] = I.rolling_share(key, spec.better, spec.center, cand, base, new,
                                               ctx, oc.rolling_window, req.rolling_step)
            if req.mode == "add":
                cap = sum(prof.weights[t] for t in sources) if sources else 1.0
                for w in [x for x in oc.weight_grid if x <= cap + 1e-12]:
                    nw = with_weights(c, w)
                    nn = portfolio_returns(R, nw, reb)
                    mw = I.impact_metrics(cand, base, nn, R, prof.weights, nw, bench, ctx, c)["metrics"]
                    for k2 in ("size_loading",):
                        mw[k2] = metrics.get(k2, np.nan)
                    stw = SC.stress_table(c, {**daily, c: b.daily_px[c]}, prof.weights, nw,
                                          windows, b.streams)
                    for _, r in stw.iterrows():
                        mw[f"d_scen_{r['Key']}"] = r.get("Change in portfolio return", np.nan) \
                            if r.get("Status") == "ok" else np.nan
                    row = {"Add weight": w}
                    for k, o in oc.objectives.items():
                        sc = V.score(o, mw, oc.metrics)["composite"]
                        wverdicts.setdefault(k, {})[w] = V.raw_verdict(o, sc)
                        if k == okey:
                            row.update({"Score": sc, "Verdict (before caps)": V.raw_verdict(o, sc)})
                    row.update({"Change in volatility": mw["d_vol"], "Excess return (geometric)": mw["d_cagr"],
                                "Change in max drawdown": mw["d_mdd"], "Change in Sharpe": mw["d_sharpe"]})
                    wgrid_rows.append(row)

        covered = float((st["Status"] == "ok").mean()) if len(st) and "Status" in st else np.nan
        objs, decs = _decide_all(cfg, metrics, len(P), rolling, wverdicts, covered)
        dec = decs[okey]
        scen_ok = st[st["Status"] == "ok"] if "Status" in st else st.iloc[0:0]
        consistency = {
            "Stress windows covered": int(len(scen_ok)),
            "Stress windows where the portfolio did better with it":
                float((scen_ok["Change in portfolio return"] > 0).mean()) if len(scen_ok) else np.nan,
            "Regimes where the portfolio did better with it":
                float((reg.loc[reg["Key"].isin(list(regimes)), "Change in portfolio ann. return"]
                       .dropna() > 0).mean()) if len(reg) else np.nan,
        }
        flags = _flags(cfg, b, c, P, holdings, bench_of[c], st, bmode.lower() == "sleeve")
        fw, fn = weighted_fund_data(b, prof.weights), weighted_fund_data(b, new_w)
        fdata = {"MER without": fw["MER"], "MER with": fn["MER"],
                 "Yield without": fw["Yield"], "Yield with": fn["Yield"],
                 "MER missing": fn["MER missing"], "Yield missing": fn["Yield missing"]}
        if fn["MER missing"] and len(fn["MER missing"]) < len(new_w):
            flags.append("No expense ratio for " + ", ".join(fn["MER missing"])
                         + ": the portfolio's weighted MER is not shown.")
        rat = V.rationale(obj, dec, metrics, oc.metrics, flags, c)
        series = pd.DataFrame({c: cand, "Benchmark": bench, "Portfolio without": base,
                               "Portfolio with": new})
        results[c] = CandidateResult(
            c, bench_of[c], bench_of[c] or prof.describe(), len(P), P[0], P[-1], new_w,
            metrics, dec, rat, objs, S.standalone(cand, b.rf), imp["relative"], fe, reg, st,
            imp["sim"], rolling, pd.DataFrame(wgrid_rows), consistency, flags,
            imp["base_stats"], imp["new_stats"], series, b.info(c),
            S.relative(cand, base, b.rf), fdata)

    ranking, components = _ranking(results, okey)
    corr, excorr, crisis, equiv = _matrices(results, b, bench_of, base_by_c)
    for e in equiv:
        e.update(_cost_compare(b, e["A"], e["B"]))
    fund = _fund_review(results, b, prof.describe())
    period = (f"{common[0]:%Y-%m} to {common[-1]:%Y-%m} ({len(common)} months, common to all)"
              if req.alignment == "common" and len(common) else "each candidate's own overlap")
    if b.streams:
        notes.append("Uploaded return streams: " + "; ".join(f"{k} ({v})" for k, v in b.streams.items())
                     + ". Month-end points are used in stress windows that involve them.")
    return Review(req, prof, okey, obj.label if req.objective else "No objective (impact only)",
                  obj.description if req.objective else "", weight, holdings,
                  holdings[holdings["Spliced months"] > 0].copy(), results, skipped, ranking,
                  components, corr, excorr, crisis, equiv, fund, notes, period,
                  b.rf_label, b.cpi_label, req.objective is not None, b.rf, sources)


def run_many(req: Request, bases: List[Union[str, Profile]], cfg: AllConfig, b: Bundle,
             regimes: Optional[Dict[str, pd.Series]] = None) -> List[Review]:
    """The same candidates added to several bases (benchmarks, blends,
    uploaded portfolios or profiles), one review per base."""
    from dataclasses import replace as _r
    regimes = regimes if regimes is not None else regime_labels(cfg, b)
    return [run(_r(req, profile=p), cfg, b, regimes) for p in bases]


def _flags(cfg: AllConfig, b: Bundle, c: str, P: pd.DatetimeIndex, holdings: pd.DataFrame,
           bench: Optional[str], st: pd.DataFrame, sleeve_mode: bool = False) -> List[str]:
    out = []
    n = len(P)
    if n < 36:
        out.append(f"Short overlap: {n} months (< 36): low confidence.")
    elif n < 60:
        out.append(f"Short overlap: {n} months (< 60): moderate confidence.")
    for _, h in holdings.iterrows():
        if h["Spliced months"] and h["Spliced to"] and h["Spliced to"] >= P[0].strftime("%Y-%m"):
            out.append(f"Profile holding {h['Holding']} uses proxy {h['Proxy']} until {h['Spliced to']} "
                       f"inside the analysis period.")
    if bench is None and sleeve_mode:
        out.append("No sleeve benchmark: relative statistics are measured against the portfolio.")
    out.extend(b.flags.get(c, []))
    srcs = factor_sources(cfg, b)
    legs = [k for k, sp in (cfg.factors.get("factors") or {}).items()
            if c in (sp.get("long"), sp.get("short")) and not srcs.get(k, "").startswith("Ken French")]
    if legs:
        out.append(f"{c} is itself a leg of the {', '.join(legs)} factor proxy: those loadings "
                   f"are mechanical, not evidence of a tilt.")
    inf = b.info(c)
    adv, floor = inf.get("avg_dollar_volume"), float(cfg.factors.get("min_avg_dollar_volume", 1e6))
    if adv is not None and adv < floor:
        out.append(f"Thin trading: average daily value traded about {adv:,.0f} "
                   f"(below {floor:,.0f}); check spreads before trading.")
    for fl in cfg.universe.income_flags(c):
        out.append(f"Distributions include {fl.replace('_', ' ')}: yield overstates income earned.")
    nd = st[st.get("Status", pd.Series(dtype=str)).astype(str).str.startswith("no data")] if len(st) else st
    if len(nd):
        out.append(f"{len(nd)} of {len(st)} stress windows have no data for this comparison.")
    y = b.yields.get(c, {})
    tr = S.standalone(b.monthly[c].dropna().iloc[-60:], b.rf)
    if y.get("dist_yield", 0) == y.get("dist_yield", 0) and tr.get("CAGR") == tr.get("CAGR") \
            and y.get("dist_yield", 0) > 0.05 and y["dist_yield"] > tr["CAGR"] + 0.02:
        out.append(f"Distribution yield {y['dist_yield']:.1%} exceeds the 5-year total return "
                   f"{tr['CAGR']:.1%} by more than 2 points: payouts may include return of capital.")
    return out


def _ranking(results: Dict[str, CandidateResult], objective: str):
    rows, comps = [], []
    for c, r in results.items():
        d = r.decision
        rows.append({"Candidate": c, "Score": d["composite"], "Unclipped score": d["unclipped"],
                     "Verdict": d["verdict"],
                     "Expense ratio": r.costs.get("expense_ratio", np.nan),
                     "Confidence": d["confidence"], "Months": r.months,
                     "Benchmark": r.benchmark_label})
        cc = d["components"].copy()
        cc.insert(0, "Candidate", c)
        comps.append(cc)
    rk = pd.DataFrame(rows)
    if len(rk):
        # Ties at a capped score (every component at +1 or -1) are broken by
        # the unclipped score: same weights, no clip.
        rk = rk.sort_values(["Score", "Unclipped score"], ascending=False,
                            na_position="last").reset_index(drop=True)
        rk.insert(0, "Rank", range(1, len(rk) + 1))
    return rk, (pd.concat(comps, ignore_index=True) if comps else pd.DataFrame())


def _matrices(results, b: Bundle, bench_of, base_by_c):
    if not results:
        e = pd.DataFrame()
        return e, e, e, []
    cols = {c: r.series[c] for c, r in results.items()}
    first = next(iter(results))
    cols["Portfolio"] = base_by_c[first]
    df = pd.DataFrame(cols)
    corr = df.corr()
    bench = {c: (b.monthly[bench_of[c]] if bench_of[c] else base_by_c[c]) for c in results}
    rets = pd.DataFrame({c: r.series[c] for c, r in results.items()})
    excorr = S.excess_correlation(rets, bench)
    pb = base_by_c[first]
    tail = pb <= pb.quantile(0.10)
    crisis = df[tail.reindex(df.index).fillna(False)].corr() if tail.sum() >= 6 else pd.DataFrame()
    eq = S.equivalents(rets, bench)
    return corr, excorr, crisis, eq


def _cost_compare(b: Bundle, a: str, c: str) -> Dict[str, Any]:
    """Who is cheaper and who trades more, for a functionally equivalent pair."""
    ia, ic = b.info(a), b.info(c)
    ea, ec = ia.get("expense_ratio"), ic.get("expense_ratio")
    va, vc = ia.get("avg_dollar_volume"), ic.get("avg_dollar_volume")
    out: Dict[str, Any] = {"Expense ratio A": ea, "Expense ratio B": ec}
    bits = []
    if ea is not None and ec is not None:
        if abs(ea - ec) < 5e-6:
            bits.append(f"same expense ratio ({ea:.2%})")
        else:
            lo = a if ea < ec else c
            bits.append(f"{lo} is cheaper ({min(ea, ec):.2%} vs {max(ea, ec):.2%} a year)")
    if va and vc:
        hi = a if va > vc else c
        bits.append(f"{hi} trades more ({max(va, vc):,.0f} vs {min(va, vc):,.0f} a day)")
    out["Cost and liquidity"] = "; ".join(bits) if bits else "cost and volume unavailable"
    return out


def _cost_rows(inf: Dict[str, float]) -> Dict[str, float]:
    return {"Expense ratio": inf.get("expense_ratio", np.nan),
            "Assets (AUM)": inf.get("aum", np.nan),
            "Average daily value traded": inf.get("avg_dollar_volume", np.nan)}


def weighted_fund_data(b: Bundle, weights: Dict[str, float]) -> Dict[str, Any]:
    """MER (yfinance expense ratio) and trailing 12-month distribution yield
    of a portfolio, weighted by target weights. A figure is n/a when any
    holding lacks it (the holdings missing it are listed), so a partial
    average is never shown as the portfolio's."""
    out: Dict[str, Any] = {}
    for key, get in (("MER", lambda t: b.info(t).get("expense_ratio")),
                     ("Yield", lambda t: b.yields.get(t, {}).get("dist_yield"))):
        vals = {t: get(t) for t in weights}
        missing = [t for t, v in vals.items() if v is None or v != v]
        out[key] = np.nan if missing else float(sum(weights[t] * vals[t] for t in weights))
        out[f"{key} missing"] = missing
    return out


def _fund_review(results, b: Bundle, base_label: str = "Portfolio") -> pd.DataFrame:
    """Each security, then the portfolio over each security's analysis
    period, then any explicit benchmark: all from the same standalone
    statistics (qbt.metrics.summary)."""
    cols, port, bench = {}, {}, {}
    for c, r in results.items():
        st = dict(r.standalone)
        y = b.yields.get(c, {})
        st["Distribution yield (trailing 12m)"] = y.get("dist_yield", np.nan)
        st["Distribution variability (CV)"] = y.get("yield_stability", np.nan)
        st.update(_cost_rows(b.info(c)))
        cols[c] = st
        pk = f"{base_label}, without ({r.start:%Y-%m} to {r.end:%Y-%m})"
        port.setdefault(pk, r.base_stats)
        if r.benchmark and f"{r.benchmark} (benchmark)" not in bench and r.benchmark not in results:
            bs = S.standalone(r.series["Benchmark"], b.rf)
            yb = b.yields.get(r.benchmark, {})
            bs["Distribution yield (trailing 12m)"] = yb.get("dist_yield", np.nan)
            bs["Distribution variability (CV)"] = yb.get("yield_stability", np.nan)
            bs.update(_cost_rows(b.info(r.benchmark)))
            bench[f"{r.benchmark} (benchmark)"] = bs
    return pd.DataFrame({**cols, **port, **bench})


# ----------------------------------------------------------------------
def sweep(cfg: AllConfig, b: Bundle, candidate: str, objective: str, mode: str = "add",
          weight: Optional[float] = None, replace_holding: Optional[str] = None,
          regimes: Optional[Dict[str, pd.Series]] = None, robustness: bool = True,
          fund_from: Optional[List[str]] = None) -> pd.DataFrame:
    """One candidate across every preset profile."""
    regimes = regimes if regimes is not None else regime_labels(cfg, b)
    rows = []
    for k, p in cfg.profiles.profiles.items():
        if p.custom:
            continue
        row = {"Profile": p.label, "Key": k}
        if mode == "replace" and replace_holding not in p.weights:
            row.update({"Verdict": "n/a", "Note": f"{replace_holding} is not held"})
            rows.append(row)
            continue
        try:
            rv = run(Request(k, objective, [candidate], mode=mode, weight=weight,
                             replace_holding=replace_holding, robustness=robustness,
                             fund_from=fund_from), cfg, b, regimes)
        except ConfigError as exc:                # e.g. no bonds to reduce in 100% Equities
            row.update({"Verdict": "n/a", "Note": str(exc)})
            rows.append(row)
            continue
        if candidate not in rv.results:
            row.update({"Verdict": "n/a", "Note": rv.skipped.get(candidate, "")})
            rows.append(row)
            continue
        r = rv.results[candidate]
        m = r.metrics
        row.update({"Verdict": r.decision["verdict"], "Confidence": r.decision["confidence"],
                    "Score": r.decision["composite"], "Months": r.months,
                    "Change in volatility": m["d_vol"], "Excess return (geometric)": m["d_cagr"],
                    "Change in max drawdown": m["d_mdd"], "Change in Sharpe": m["d_sharpe"],
                    "Correlation to portfolio": m["corr_to_portfolio"],
                    "Main trade-off": r.rationale["tradeoff"], "Note": ""})
        rows.append(row)
    return pd.DataFrame(rows)
