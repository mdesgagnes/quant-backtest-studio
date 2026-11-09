"""Fund review: does a product earn its place, and what should it replace?

The question a return stream alone cannot answer is whether the manager is
adding anything the client could not buy for a few basis points. The test
used here is the honest hurdle for any low-beta or option-income product:
a **beta-matched replica** -- beta x the relevant market factor plus
(1 - beta) x cash, rebalanced monthly. Same market sensitivity, two cheap
ETFs. If the product cannot beat that after fees, its extra machinery is
not adding value for the investor.

Everything here is a pure function of a returns frame, a risk-free series
and a `FundReviewConfig`; the interface only collects the config and draws
the result, and the same `run_review` call can be scripted from a YAML file
(`configs/fund_review_example.yaml`) without the interface at all.

What a review contains:

1. Headline statistics for the product, its beta-matched and
   volatility-matched replicas, and the factor.
2. A monthly regression with Newey-West t-stats: alpha, beta, a bond
   factor, Treynor-Mazuy convexity, up/down betas, and alpha before and
   after a split date (an AUM ramp, a PM or mandate change).
3. Stress episodes with the product, its replica and the factor side by
   side; the share of each factor sell-off the product absorbed; capture on
   big factor days.
4. A payoff profile by factor-month quintile and a rolling beta.
5. A portfolio funding test: a base portfolio, plus a slice of the product
   funded from each sleeve in turn and beta-proportionally across all of
   them, run through the same engine as every backtest.
6. Fees and an illustrative tax comparison against the replica.
7. A verdict from three explicit checks (`decide`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from . import cipm as C
from . import metrics as M
from . import stress as S
from .config import EngineConfig
from .engine import blended_benchmark


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class TaxSettings:
    """Marginal rates as decimals. Defaults: Quebec top bracket, 2026."""
    income_rate: float = 0.5331          # interest, foreign dividends, fund income
    capital_gains_rate: float = 0.2665
    eligible_dividend_rate: float = 0.4011


@dataclass
class Distribution:
    """One year of distributions per unit, from the fund's MRFP. Leave
    `nav` at 0 to skip the tax comparison."""
    nav: float = 0.0
    income: float = 0.0                  # net investment income (interest, premiums taxed as income)
    dividends: float = 0.0               # Canadian eligible dividends
    capital_gains: float = 0.0
    return_of_capital: float = 0.0
    source: str = ""                     # e.g. "MRFP FY2025, Series F"


@dataclass
class FundReviewConfig:
    product: str
    factor: str
    bond: Optional[str] = None           # bond factor for the 2-factor regression
    split_date: Optional[str] = None     # alpha before / after this date
    # Portfolio funding test. Weights by column; empty disables the test.
    base_portfolio: Dict[str, float] = field(default_factory=dict)
    equity_sleeves: List[str] = field(default_factory=list)
    bond_sleeves: List[str] = field(default_factory=list)
    funding_amount: float = 0.10
    rebalance: str = "Q"
    # Costs
    mer: Optional[float] = None          # product MER, decimal
    replica_cost: float = 0.001          # blended ETF cost of the replica
    factor_dividend_yield: float = 0.012 # used for the replica's tax drag
    distribution: Distribution = field(default_factory=Distribution)
    tax: TaxSettings = field(default_factory=TaxSettings)
    # Extra stress windows: [{"name":..., "start":..., "end":...}]
    extra_periods: List[Dict[str, str]] = field(default_factory=list)
    big_day_threshold: float = 0.02
    rolling_window: int = 36             # months
    # Decision thresholds
    alpha_t_required: float = 2.0
    crash_ratio_tolerance: float = 1.25  # crash share / beta, median across sell-offs
    min_sharpe_gain: float = 0.02        # beta-neutral funding must add this much
    label: str = ""                      # display name; defaults to `product`

    @property
    def name(self) -> str:
        return self.label or self.product


def config_from_dict(d: Dict[str, Any]) -> FundReviewConfig:
    d = dict(d or {})
    if "product" not in d or "factor" not in d:
        raise ValueError("A fund review needs at least `product` and `factor`.")
    dist = Distribution(**(d.pop("distribution", None) or {}))
    tax = TaxSettings(**(d.pop("tax", None) or {}))
    known = set(FundReviewConfig.__dataclass_fields__)
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError("Unknown fund review setting(s): " + ", ".join(unknown))
    return FundReviewConfig(distribution=dist, tax=tax, **d)


def config_from_yaml(text: str) -> FundReviewConfig:
    data = yaml.safe_load(text) or {}
    return config_from_dict(data.get("fund_review", data))


def config_to_yaml(cfg: FundReviewConfig) -> str:
    return yaml.safe_dump({"fund_review": asdict(cfg)}, sort_keys=False, allow_unicode=True)


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def to_monthly(r: pd.Series, ppy: int) -> pd.Series:
    """Compound to calendar months. A final month that has not finished is
    dropped, so the regressions never mix a partial month with full ones."""
    r = r.dropna()
    if ppy <= 12 or r.empty:
        return r
    m = (1 + r).resample("ME").prod() - 1
    last = r.index[-1]
    if last < (last + pd.offsets.BMonthEnd(0)).normalize():
        m = m.iloc[:-1]
    return m


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 3) -> Dict[str, np.ndarray]:
    """OLS with an intercept and Newey-West (Bartlett) standard errors."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    Xe = X * e[:, None]
    S0 = Xe.T @ Xe
    for L in range(1, min(lags, len(y) - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Xe[L:].T @ Xe[:-L]
        S0 += w * (G + G.T)
    se = np.sqrt(np.maximum(np.diag(xtx_inv @ S0 @ xtx_inv), 0))
    ss = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - float(e @ e) / ss if ss > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    return {"coef": beta, "t": t, "r2": r2, "n": len(y)}


def _ann(alpha_per_month: float) -> float:
    return (1.0 + alpha_per_month) ** 12 - 1.0


def regression(prod_m: pd.Series, fac_m: pd.Series, rf_m: pd.Series,
               bond_m: Optional[pd.Series] = None, lags: int = 3) -> Dict[str, float]:
    """Monthly excess-return regressions of the product on the factor."""
    cols = {"p": prod_m, "f": fac_m, "rf": rf_m}
    if bond_m is not None:
        cols["b"] = bond_m
    d = pd.DataFrame(cols).dropna()
    if len(d) < 24:
        raise ValueError("At least 24 monthly observations are needed for the regression.")
    y = (d.p - d.rf).to_numpy()
    x = (d.f - d.rf).to_numpy()
    one = newey_west_ols(y, x, lags)
    out = {"months": float(len(d)), "alpha": _ann(one["coef"][0]), "alpha_t": float(one["t"][0]),
           "beta": float(one["coef"][1]), "beta_t": float(one["t"][1]), "r2": float(one["r2"]),
           "corr_factor": float(d.p.corr(d.f))}
    tm = newey_west_ols(y, np.column_stack([x, x ** 2]), lags)
    out["tm_gamma"], out["tm_gamma_t"] = float(tm["coef"][2]), float(tm["t"][2])
    pw = newey_west_ols(y, np.column_stack([np.maximum(x, 0), np.minimum(x, 0)]), lags)
    out["beta_up"], out["beta_down"] = float(pw["coef"][1]), float(pw["coef"][2])
    if "b" in d:
        xb = (d.b - d.rf).to_numpy()
        two = newey_west_ols(y, np.column_stack([x, xb]), lags)
        out.update({"alpha_2f": _ann(two["coef"][0]), "alpha_2f_t": float(two["t"][0]),
                    "beta_bond": float(two["coef"][2]), "beta_bond_t": float(two["t"][2]),
                    "corr_bond": float(d.p.corr(d.b))})
    cap = C.capture_ratios(d.p, d.f, 12)
    out["up_capture"], out["down_capture"] = cap["up_capture"], cap["down_capture"]
    return out


def split_regression(prod_m, fac_m, rf_m, split: str, lags: int = 3) -> Dict[str, float]:
    """Alpha and beta before and after `split`; NaN for a side with < 12 months."""
    d = pd.DataFrame({"p": prod_m, "f": fac_m, "rf": rf_m}).dropna()
    t0 = pd.Timestamp(split)
    out: Dict[str, float] = {}
    for tag, part in (("before", d[d.index < t0]), ("after", d[d.index >= t0])):
        if len(part) < 12:
            out.update({f"alpha_{tag}": np.nan, f"alpha_{tag}_t": np.nan, f"beta_{tag}": np.nan,
                        f"months_{tag}": float(len(part))})
            continue
        r = newey_west_ols((part.p - part.rf).to_numpy(), (part.f - part.rf).to_numpy(), lags)
        out.update({f"alpha_{tag}": _ann(r["coef"][0]), f"alpha_{tag}_t": float(r["t"][0]),
                    f"beta_{tag}": float(r["coef"][1]), f"months_{tag}": float(len(part))})
    return out


def replica_returns(factor: pd.Series, cash: pd.Series, weight: float, ppy: int) -> pd.Series:
    """`weight` in the factor and the rest in cash, rebalanced at each month
    start (each period, for monthly or coarser data). Returned at the
    input's own frequency."""
    d = pd.DataFrame({"f": factor, "c": cash}).dropna()
    if ppy <= 12:
        return weight * d.f + (1 - weight) * d.c
    parts = []
    for _, g in d.groupby(d.index.to_period("M")):
        val = weight * (1 + g.f).cumprod() + (1 - weight) * (1 + g.c).cumprod()
        parts.append(val / val.shift(1).fillna(1.0) - 1.0)
    return pd.concat(parts)


def _stats(r: pd.Series, rf: pd.Series, ppy: int) -> Dict[str, float]:
    r = r.dropna()
    eq = M.to_equity(r)
    m = to_monthly(r, ppy)
    cg = M.cagr(eq, ppy)
    rfa = M.ann_return(rf.reindex(r.index).fillna(0.0), ppy)
    vol = float(m.std(ddof=0) * np.sqrt(12)) if len(m) > 2 else np.nan
    return {"CAGR": cg, "Volatility": vol,
            "Sharpe": (cg - rfa) / vol if vol and vol == vol else np.nan,
            "Max Drawdown": M.max_drawdown(eq), "Worst month": float(m.min()) if len(m) else np.nan,
            "Growth of 100": float(100 * eq.iloc[-1] / eq.iloc[0]) if len(eq) else np.nan}


def episode_return(r: pd.Series, start, end) -> float:
    x = r.loc[str(start):str(end)].dropna()
    return float((1 + x).prod() - 1) if len(x) else np.nan


def stress_periods(start: pd.Timestamp, end: pd.Timestamp,
                   extra: List[Dict[str, str]]) -> List[S.StressPeriod]:
    """Default episodes fully inside the data, then the config's own."""
    out = [p for p in S.DEFAULT_PERIODS
           if pd.Timestamp(p.start) >= start and pd.Timestamp(p.end) <= end]
    for e in extra or []:
        out.append(S.StressPeriod(e["name"], str(e["start"]), str(e["end"]),
                                  e.get("category", "Custom")))
    return out


def current_drawdown_period(r: pd.Series, min_depth: float = 0.03) -> Optional[S.StressPeriod]:
    """The drawdown the product is in at the last date, if deeper than
    `min_depth`: from its last peak to the end of the data."""
    eq = M.to_equity(r.dropna())
    dd = eq / eq.cummax() - 1
    if dd.iloc[-1] > -min_depth:
        return None
    peak = eq.loc[:dd.index[-1]].idxmax()
    nxt = eq.index[eq.index.get_indexer([peak])[0] + 1]
    return S.StressPeriod("Current drawdown", str(nxt.date()), str(eq.index[-1].date()), "Ongoing")


def big_day_capture(prod: pd.Series, fac: pd.Series, threshold: float) -> Dict[str, float]:
    d = pd.DataFrame({"p": prod, "f": fac}).dropna()
    dn, up = d[d.f <= -threshold], d[d.f >= threshold]
    return {"down_days": float(len(dn)), "up_days": float(len(up)),
            "down_share": float(dn.p.mean() / dn.f.mean()) if len(dn) else np.nan,
            "up_share": float(up.p.mean() / up.f.mean()) if len(up) else np.nan}


def quintile_profile(prod_m: pd.Series, fac_m: pd.Series, extra: Dict[str, pd.Series]) -> pd.DataFrame:
    """Average monthly return by factor-month quintile."""
    d = pd.DataFrame({"Product": prod_m, "Factor": fac_m, **extra}).dropna()
    q = pd.qcut(d.Factor, 5, labels=["Worst 20%", "Q2", "Q3", "Q4", "Best 20%"])
    return d.groupby(q, observed=True).mean()


def rolling_beta(prod_m, fac_m, rf_m, window: int = 36) -> pd.DataFrame:
    d = pd.DataFrame({"p": prod_m, "f": fac_m, "rf": rf_m}).dropna()
    y, x = d.p - d.rf, d.f - d.rf
    return pd.DataFrame({"Beta": y.rolling(window).cov(x) / x.rolling(window).var(),
                         "Correlation": d.p.rolling(window).corr(d.f)}).dropna()


def funding_weights(base: Dict[str, float], product: str, amount: float, source: str,
                    beta: float, equity: List[str], bonds: List[str]) -> Dict[str, float]:
    """The base portfolio plus `amount` of the product, paid for by `source`:
    one sleeve's name, or "beta-proportional" (beta x amount from the equity
    sleeves, the rest from the bond sleeves, each pro rata to its weight)."""
    w = dict(base)
    w[product] = w.get(product, 0.0) + amount
    if source == "beta-proportional":
        for group, share in ((equity, beta), (bonds, 1.0 - beta)):
            tot = sum(base.get(k, 0.0) for k in group)
            if tot <= 0:
                raise ValueError("Beta-proportional funding needs weights in both the "
                                 "equity and the bond sleeves.")
            for k in group:
                w[k] -= amount * share * base[k] / tot
    else:
        if base.get(source, 0.0) < amount - 1e-12:
            raise ValueError(f"{source} holds less than the {amount:.0%} being funded.")
        w[source] -= amount
    return w


def funding_test(returns: pd.DataFrame, rf: pd.Series, cfg: FundReviewConfig,
                 beta: float, ppy: int) -> Optional[pd.DataFrame]:
    """The base portfolio and each funding variant, through the engine."""
    base = {k: float(v) for k, v in (cfg.base_portfolio or {}).items() if v}
    if not base:
        return None
    need = sorted(set(base) | {cfg.product})
    missing = [c for c in need if c not in returns]
    if missing:
        raise ValueError("Funding test: column(s) not in the data: " + ", ".join(missing))
    start = max(returns[c].first_valid_index() for c in need)
    r = returns.loc[start:, need].fillna(0.0)
    px = (1 + r).cumprod()
    px = px / px.iloc[0]
    eng = EngineConfig(rebalance=cfg.rebalance, trim_warmup=False, periods_per_year=ppy)
    variants: Dict[str, Dict[str, float]] = {"Base portfolio": base}
    for sleeve in base:
        if base[sleeve] >= cfg.funding_amount:
            variants[f"+{cfg.funding_amount:.0%} from {sleeve}"] = funding_weights(
                base, cfg.product, cfg.funding_amount, sleeve, beta, [], [])
    if cfg.equity_sleeves and cfg.bond_sleeves:
        variants[f"+{cfg.funding_amount:.0%} beta-proportional"] = funding_weights(
            base, cfg.product, cfg.funding_amount, "beta-proportional", beta,
            cfg.equity_sleeves, cfg.bond_sleeves)
    rows = {}
    rf_w = rf.reindex(r.index).fillna(0.0)
    for name, w in variants.items():
        res = blended_benchmark(px, w, eng, label=name, rebalance=cfg.rebalance)
        rr = res.returns.dropna()
        s = M.summary(rr, None, None, None, None, rf_w, ppy)
        vc = M.var_cvar(rr)
        rows[name] = {"CAGR": s.get("CAGR"), "Volatility": s.get("Volatility"),
                      "Sharpe": s.get("Sharpe"), "Max Drawdown": s.get("Max Drawdown"),
                      "CVaR 95%": vc.get("cvar", np.nan) if isinstance(vc, dict) else np.nan,
                      "Worst month": float(to_monthly(rr, ppy).min())}
    out = pd.DataFrame(rows).T
    out.attrs["start"] = str(start.date())
    out.attrs["end"] = str(r.index[-1].date())
    return out


def tax_comparison(cfg: FundReviewConfig, beta: float, cash_yield: float) -> Optional[Dict[str, float]]:
    """Illustrative annual tax as a share of NAV, product vs replica, in a
    non-registered account. Timing, not a permanent loss, for the part that
    is realized gains -- the report says so."""
    d, t = cfg.distribution, cfg.tax
    if not d.nav:
        return None
    inc, div, cg, roc = (x / d.nav for x in (d.income, d.dividends, d.capital_gains, d.return_of_capital))
    product = inc * t.income_rate + div * t.eligible_dividend_rate + cg * t.capital_gains_rate
    replica = ((1 - beta) * cash_yield * t.income_rate
               + beta * cfg.factor_dividend_yield * t.income_rate)
    return {"income_pct": inc, "dividends_pct": div, "capital_gains_pct": cg, "roc_pct": roc,
            "distribution_pct": inc + div + cg + roc, "product_tax": product,
            "replica_tax": replica, "difference": product - replica, "cash_yield": cash_yield}


# ----------------------------------------------------------------------
# The review
# ----------------------------------------------------------------------
@dataclass
class FundReview:
    config: FundReviewConfig
    start: pd.Timestamp
    end: pd.Timestamp
    ppy: int
    beta: float
    vol_weight: float
    headline: pd.DataFrame
    regression: Dict[str, float]
    split: Optional[Dict[str, float]]
    calendar: pd.DataFrame
    trailing: pd.DataFrame
    equity: pd.DataFrame
    relative: pd.Series
    stress: pd.DataFrame
    crash: pd.DataFrame
    big_days: Optional[Dict[str, float]]
    quintiles: pd.DataFrame
    rolling: pd.DataFrame
    funding: Optional[pd.DataFrame]
    costs: Dict[str, Any]
    verdict: Dict[str, Any]
    notes: List[str] = field(default_factory=list)


def run_review(returns: pd.DataFrame, rf: M.RiskFree, cfg: FundReviewConfig,
               ppy: int) -> FundReview:
    if cfg.product not in returns or cfg.factor not in returns:
        raise ValueError("Product and factor must both be columns of the data.")
    notes: List[str] = []
    prod = returns[cfg.product].dropna()
    start, end = prod.index[0], prod.index[-1]
    fac_full = returns[cfg.factor]
    if fac_full.first_valid_index() is None or fac_full.first_valid_index() > start:
        raise ValueError(f"{cfg.factor} must cover the product's whole history "
                         f"(from {start.date()}).")
    d = returns.loc[start:end]
    rf_s = M.rf_per_period(d.index, rf, ppy)
    if float(rf_s.abs().sum()) == 0.0:
        notes.append("Risk-free rate is 0%: alpha includes the cash yield. "
                     "Use a T-bill series for a fair test.")
    prod, fac = d[cfg.product], d[cfg.factor].fillna(0.0)
    prod_m, fac_m, rf_m = to_monthly(prod, ppy), to_monthly(fac, ppy), to_monthly(rf_s, ppy)
    bond_m = to_monthly(d[cfg.bond], ppy) if cfg.bond and cfg.bond in d else None

    reg = regression(prod_m, fac_m, rf_m, bond_m)
    beta = reg["beta"]
    split = split_regression(prod_m, fac_m, rf_m, cfg.split_date) if cfg.split_date else None

    rep_b = replica_returns(fac, rf_s, beta, ppy)
    vol_w = float(prod_m.std() / fac_m.reindex(prod_m.index).std())
    rep_v = replica_returns(fac, rf_s, vol_w, ppy)
    nm = cfg.name
    lab_b = f"Beta replica ({beta:.2f} factor / {1 - beta:.2f} cash)"
    lab_v = f"Vol-matched replica ({vol_w:.2f} factor)"
    series = {nm: prod, lab_b: rep_b, lab_v: rep_v, cfg.factor: fac}
    headline = pd.DataFrame({k: _stats(v, rf_s, ppy) for k, v in series.items()}).T

    years = sorted(set(prod.index.year))
    cal = pd.DataFrame({k: [episode_return(v, f"{y}-01-01", f"{y}-12-31") for y in years]
                        for k, v in ((nm, prod), ("Beta replica", rep_b), (cfg.factor, fac))},
                       index=years)
    cal["Partial"] = [prod.index[0].year == y and prod.index[0].month > 1 or
                      (prod.index[-1].year == y and prod.index[-1].month < 12) for y in years]

    eq_prod = M.to_equity(prod)
    def _trail(eq: pd.Series) -> pd.Series:
        t = M.trailing_returns(eq, ppy)
        return t.set_index("Period")["Return"] if not t.empty else pd.Series(dtype=float)
    trailing = pd.DataFrame({nm: _trail(eq_prod), "Beta replica": _trail(M.to_equity(rep_b)),
                             cfg.factor: _trail(M.to_equity(fac))})

    equity = pd.DataFrame({nm: eq_prod, "Beta replica": M.to_equity(rep_b),
                           cfg.factor: M.to_equity(fac)}) * 100
    relative = (equity[nm] / equity["Beta replica"] - 1).dropna()

    periods = stress_periods(start, end, cfg.extra_periods)
    cur = current_drawdown_period(prod)
    if cur is not None and not any(p.name == cur.name for p in periods):
        periods.append(cur)
    st_rows = []
    for p in periods:
        st_rows.append({"Episode": p.name, "Start": p.start, "End": p.end,
                        nm: episode_return(prod, p.start, p.end),
                        "Beta replica": episode_return(rep_b, p.start, p.end),
                        cfg.factor: episode_return(fac, p.start, p.end)})
    stress_tab = pd.DataFrame(st_rows)
    crash = pd.DataFrame()
    if not stress_tab.empty:
        sell = stress_tab[stress_tab[cfg.factor] <= -0.05].copy()
        if not sell.empty:
            sell["Share absorbed"] = sell[nm] / sell[cfg.factor]
            sell["Share / beta"] = sell["Share absorbed"] / beta
            crash = sell[["Episode", nm, cfg.factor, "Share absorbed", "Share / beta"]]

    big = big_day_capture(prod, fac, cfg.big_day_threshold) if ppy >= 200 else None
    quint = quintile_profile(prod_m, fac_m, {"Beta replica": to_monthly(rep_b, ppy)})
    quint = quint.rename(columns={"Product": nm, "Factor": cfg.factor})
    roll = rolling_beta(prod_m, fac_m, rf_m, cfg.rolling_window)

    try:
        funding = funding_test(returns, rf_s.reindex(returns.index).fillna(0.0)
                               if isinstance(rf_s, pd.Series) else rf_s, cfg, beta, ppy)
    except ValueError as exc:
        funding = None
        notes.append(str(exc))

    rf_12 = rf_s.loc[rf_s.index >= end - pd.DateOffset(years=1)]
    cash_yield = float((1 + rf_12).prod() - 1) if len(rf_12) else 0.0
    costs: Dict[str, Any] = {"mer": cfg.mer, "replica_cost": cfg.replica_cost,
                             "fee_gap": (cfg.mer - cfg.replica_cost) if cfg.mer is not None else None,
                             "tax": tax_comparison(cfg, beta, cash_yield)}

    review = FundReview(cfg, start, end, ppy, beta, vol_w, headline, reg, split, cal,
                        trailing, equity, relative, stress_tab, crash, big, quint, roll,
                        funding, costs, {}, notes)
    review.verdict = decide(review)
    return review


def decide(rv: FundReview) -> Dict[str, Any]:
    """Three checks, each pass / fail / not run, and the verdict they imply."""
    cfg, reg = rv.config, rv.regression
    checks: List[Dict[str, str]] = []

    a, t = reg["alpha"], reg["alpha_t"]
    recent = rv.split.get("alpha_after") if rv.split else None
    if t >= cfg.alpha_t_required:
        s1 = "pass"
    elif a > 0 and (recent is None or (recent == recent and recent > 0)):
        s1 = "borderline"
    else:
        s1 = "fail"
    txt = f"Alpha vs. the beta replica {a:+.1%}/yr (t = {t:.2f}; {cfg.alpha_t_required:.1f} required)"
    if recent is not None and recent == recent:
        txt += f"; {recent:+.1%}/yr since {cfg.split_date}"
    checks.append({"check": "Alpha after fees", "status": s1, "detail": txt})

    if rv.crash is not None and not rv.crash.empty:
        med = float(rv.crash["Share / beta"].median())
        s2 = "pass" if med <= cfg.crash_ratio_tolerance else "fail"
        checks.append({"check": "Crash behaviour", "status": s2,
                       "detail": f"Median sell-off share is {med:.2f}x its beta "
                                 f"({len(rv.crash)} sell-offs; tolerance {cfg.crash_ratio_tolerance:.2f}x)"})
    else:
        checks.append({"check": "Crash behaviour", "status": "not run",
                       "detail": "No factor sell-off of 5% or more in the history."})

    f = rv.funding
    key = next((k for k in (f.index if f is not None else []) if "beta-proportional" in k), None)
    if f is not None and key is not None:
        gain = float(f.loc[key, "Sharpe"] - f.loc["Base portfolio", "Sharpe"])
        s3 = "pass" if gain >= cfg.min_sharpe_gain else "fail"
        checks.append({"check": "Portfolio improvement", "status": s3,
                       "detail": f"Beta-neutral funding changes Sharpe by {gain:+.2f} "
                                 f"({cfg.min_sharpe_gain:+.2f} required)"})
    else:
        checks.append({"check": "Portfolio improvement", "status": "not run",
                       "detail": "Set a base portfolio with equity and bond sleeves to run it."})

    st = [c["status"] for c in checks]
    if all(s in ("pass", "not run") for s in st) and st[0] == "pass":
        verdict, short = "Add", "Passes every check that could be run."
    elif st[0] == "fail":
        verdict = "Do not add: use the replica"
        short = (f"No alpha over {rv.beta:.2f} x {cfg.factor} + {1 - rv.beta:.2f} x cash; "
                 f"that mix delivers the same exposure for about {cfg.replica_cost:.2%}.")
    else:
        verdict, short = "Hold / monitor", "Mixed evidence: keep existing positions capped and re-test."
    return {"verdict": verdict, "summary": short, "checks": checks}
