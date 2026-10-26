"""Statistical validation: is a backtest result evidence of skill, or of luck?

Nothing here re-runs the engine. Each function takes the output of one (a
return series, a trade list, a matrix of returns from many configurations)
and asks how likely that result is by chance.

What is covered, and the question each answers:

- `probabilistic_sharpe`      -- given skew, kurtosis and sample length, how
                                 sure are we that the true Sharpe beats a target?
- `deflated_sharpe`           -- the same, after charging for every
                                 configuration that was tried to find this one.
- `min_track_record_length`   -- how long a record is needed to say so?
- `bootstrap_sharpe`          -- a confidence interval on the Sharpe that
                                 respects serial dependence (stationary bootstrap).
- `sign_flip_test`            -- is the mean return distinguishable from zero?
- `random_timing_test`        -- does the exposure timing add anything over the
                                 same exposure applied at random times?
- `trade_order_test`          -- how much of the drawdown is the luck of the
                                 order the trades arrived in?
- `walk_forward_windows`      -- does the edge show up in every period?
- `pbo`                       -- probability of backtest overfitting (CSCV).
- `cpcv_selection`            -- combinatorial purged CV of a "pick the best
                                 parameter set" procedure.

References: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014);
Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest
Overfitting" (2017); Politis & Romano, "The Stationary Bootstrap" (1994);
Lopez de Prado, "Advances in Financial Machine Learning" (2018), ch. 7, 12.

No scipy dependency: the normal CDF and its inverse are implemented here.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..robustness import _norm_ppf

EULER = 0.5772156649015329


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    return float(_norm_ppf(p))


def _clean(returns) -> np.ndarray:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    return r.to_numpy(dtype=float)


def sharpe_per_period(r: np.ndarray) -> float:
    """Mean over standard deviation of one period's return (not annualised)."""
    if len(r) < 3:
        return float("nan")
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(r.mean() / sd)


def annualised_sharpe(r, ppy: int = 252) -> float:
    s = sharpe_per_period(_clean(r))
    return s * math.sqrt(ppy) if np.isfinite(s) else float("nan")


def _moments(r: np.ndarray) -> Tuple[float, float]:
    """Skewness and (non-excess) kurtosis. Normal returns give 0 and 3."""
    n = len(r)
    if n < 4:
        return 0.0, 3.0
    m = r.mean()
    sd = r.std(ddof=0)
    if sd < 1e-12:
        return 0.0, 3.0
    z = (r - m) / sd
    return float((z ** 3).mean()), float((z ** 4).mean())


def _sr_std_error(sr: float, skew: float, kurt: float, n: int) -> float:
    """Standard error of an estimated per-period Sharpe, non-normal returns
    (Mertens 2002). A strategy with negative skew and fat tails is measured
    with less precision than the textbook sqrt(1/n) suggests."""
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / max(n - 1, 1)
    return math.sqrt(max(var, 1e-18))


# ----------------------------------------------------------------------
# Sharpe significance
# ----------------------------------------------------------------------
def probabilistic_sharpe(returns, benchmark_sharpe_pa: float = 0.0,
                         ppy: int = 252) -> Dict[str, float]:
    """P(true Sharpe > benchmark), given the sample's skew and kurtosis.

    `benchmark_sharpe_pa` is annualised, like the Sharpe it is compared to.
    """
    r = _clean(returns)
    n = len(r)
    sr = sharpe_per_period(r)
    if n < 30 or not np.isfinite(sr):
        return {"psr": float("nan"), "sharpe_pa": float("nan"), "n_obs": n,
                "skew": float("nan"), "kurtosis": float("nan")}
    skew, kurt = _moments(r)
    target = benchmark_sharpe_pa / math.sqrt(ppy)
    se = _sr_std_error(sr, skew, kurt, n)
    return {"psr": norm_cdf((sr - target) / se), "sharpe_pa": sr * math.sqrt(ppy),
            "n_obs": n, "skew": skew, "kurtosis": kurt}


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected best per-period Sharpe among `n_trials` unskilled
    configurations whose Sharpes have variance `sr_variance`."""
    if n_trials < 2 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * (
        (1 - EULER) * norm_ppf(1 - 1.0 / n_trials)
        + EULER * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(returns, n_trials: int, trial_sharpes_pa: Optional[Sequence[float]] = None,
                    ppy: int = 252) -> Dict[str, float]:
    """PSR measured against the best Sharpe luck alone would have produced
    after `n_trials` attempts.

    `trial_sharpes_pa` (annualised Sharpe of every configuration tried) gives
    the cross-trial dispersion directly. Without it the dispersion is
    approximated by the sampling variance of one Sharpe estimate, which is
    the right order of magnitude for unrelated trials and conservative for
    correlated ones. With one trial there is nothing to deflate, so the
    result equals the PSR against zero.
    """
    r = _clean(returns)
    n = len(r)
    sr = sharpe_per_period(r)
    if n < 30 or not np.isfinite(sr):
        return {"dsr": float("nan"), "expected_max_sharpe_pa": float("nan"),
                "sharpe_pa": float("nan"), "n_trials": n_trials}
    skew, kurt = _moments(r)
    if trial_sharpes_pa is not None and len(trial_sharpes_pa) >= 2:
        ts = np.asarray(trial_sharpes_pa, dtype=float)
        ts = ts[np.isfinite(ts)] / math.sqrt(ppy)
        var = float(ts.var(ddof=1)) if len(ts) >= 2 else float("nan")
        n_trials = max(int(n_trials), len(ts))
    else:
        var = _sr_std_error(sr, skew, kurt, n) ** 2
    sr_star = expected_max_sharpe(n_trials, var)
    se = _sr_std_error(sr, skew, kurt, n)
    return {"dsr": norm_cdf((sr - sr_star) / se),
            "expected_max_sharpe_pa": sr_star * math.sqrt(ppy),
            "sharpe_pa": sr * math.sqrt(ppy), "n_trials": int(n_trials)}


def min_track_record_length(returns, target_sharpe_pa: float = 0.0,
                            confidence: float = 0.95, ppy: int = 252) -> Dict[str, float]:
    """Observations (and years) needed before the observed Sharpe beats
    `target_sharpe_pa` at the given confidence. NaN when the observed Sharpe
    does not exceed the target: no record length would be enough."""
    r = _clean(returns)
    sr = sharpe_per_period(r)
    if len(r) < 30 or not np.isfinite(sr):
        return {"min_obs": float("nan"), "min_years": float("nan"),
                "have_years": len(r) / ppy, "enough": False}
    skew, kurt = _moments(r)
    target = target_sharpe_pa / math.sqrt(ppy)
    if sr <= target:
        return {"min_obs": float("nan"), "min_years": float("nan"),
                "have_years": len(r) / ppy, "enough": False}
    factor = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    need = 1.0 + factor * (norm_ppf(confidence) / (sr - target)) ** 2
    return {"min_obs": float(need), "min_years": float(need / ppy),
            "have_years": len(r) / ppy, "enough": bool(len(r) >= need)}


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------
def _stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap index path of length n (Politis & Romano):
    block lengths are geometric with the given mean, blocks wrap around."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    jumps = rng.random(n) < p
    starts = rng.integers(0, n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe(returns, n_boot: int = 1000, confidence: float = 0.95,
                     mean_block: Optional[float] = None, ppy: int = 252,
                     seed: int = 42) -> Dict[str, float]:
    """Confidence interval on the annualised Sharpe by stationary bootstrap.

    Plain i.i.d. resampling understates the interval whenever returns are
    serially dependent, which strategy returns usually are (positions persist
    for weeks). `mean_block` defaults to n**(1/3), the usual rule of thumb;
    pass 1 for the i.i.d. bootstrap.
    """
    r = _clean(returns)
    n = len(r)
    if n < 30:
        return {"error": "Fewer than 30 observations.", "observed": float("nan")}
    if mean_block is None:
        mean_block = max(2.0, n ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        s = r[_stationary_indices(n, mean_block, rng)]
        sd = s.std(ddof=1)
        sharpes[b] = s.mean() / sd * math.sqrt(ppy) if sd > 1e-12 else np.nan
    sharpes = sharpes[np.isfinite(sharpes)]
    a = (1.0 - confidence) / 2.0
    return {"observed": sharpe_per_period(r) * math.sqrt(ppy),
            "ci_low": float(np.quantile(sharpes, a)),
            "ci_high": float(np.quantile(sharpes, 1 - a)),
            "median": float(np.median(sharpes)),
            "prob_positive": float((sharpes > 0).mean()),
            "confidence": confidence, "n_boot": int(len(sharpes)),
            "mean_block": float(mean_block), "samples": sharpes}


def sign_flip_test(returns, n_sims: int = 2000, ppy: int = 252,
                   seed: int = 42) -> Dict[str, float]:
    """Randomisation test of 'the mean return is zero'.

    Under that null, a return is as likely to be +x as -x. Flipping every
    sign at random and recomputing the Sharpe gives its distribution. The
    p-value is one-sided: the share of flips at least as good as the record.
    """
    r = _clean(returns)
    if len(r) < 30:
        return {"error": "Fewer than 30 observations.", "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_sims, len(r)))
    flipped = signs * r
    sd = flipped.std(axis=1, ddof=1)
    sims = np.where(sd > 1e-12, flipped.mean(axis=1) / np.where(sd > 1e-12, sd, 1), 0.0) * math.sqrt(ppy)
    obs = annualised_sharpe(r, ppy)
    # +1 on both sides: a Monte Carlo p-value is never exactly zero
    p = (1 + int((sims >= obs).sum())) / (n_sims + 1)
    return {"observed": obs, "p_value": float(p), "null_mean": float(sims.mean()),
            "null_p95": float(np.quantile(sims, 0.95)), "n_sims": n_sims}


def random_timing_test(weights: pd.DataFrame, prices: pd.DataFrame,
                       n_sims: int = 1000, min_shift: int = 21, ppy: int = 252,
                       seed: int = 42, observed: Optional[pd.Series] = None) -> Dict[str, float]:
    """Does the strategy's timing add anything beyond the exposure it takes?

    The null keeps every property of the position book -- how much is held,
    how long, how concentrated, how often it changes -- and destroys only the
    link to *when*: the whole weight panel is rotated in time by a random
    offset against the (unrotated) returns. A buy-and-hold-like strategy
    scores about as well shifted as not, so its p-value is high even when
    its Sharpe is high: it is beta, not timing. A trend or momentum rule with
    a real edge beats most rotations.

    `weights` are the end-of-day held weights the engine reports
    (`BacktestResult.weights`); they earn the next day's return. Computed on
    gross returns, so frictions do not blur the comparison.
    """
    rets = prices.pct_change().reindex(weights.index)[weights.columns].fillna(0.0).to_numpy()
    w = weights.fillna(0.0).to_numpy()
    n = len(w)
    if n < max(3 * min_shift, 60):
        return {"error": "Too few observations for a timing test.", "p_value": float("nan")}

    def sharpe_of(ws: np.ndarray) -> float:
        port = (ws[:-1] * rets[1:]).sum(axis=1)
        sd = port.std(ddof=1)
        return float(port.mean() / sd * math.sqrt(ppy)) if sd > 1e-12 else float("nan")

    obs = sharpe_of(w) if observed is None else annualised_sharpe(observed, ppy)
    own = sharpe_of(w)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, n - min_shift, size=n_sims)
    sims = np.array([sharpe_of(np.roll(w, int(s), axis=0)) for s in shifts])
    sims = sims[np.isfinite(sims)]
    # Compare like with like: the gross Sharpe of the unrotated book.
    p = (1 + int((sims >= own).sum())) / (len(sims) + 1)
    return {"observed_gross": own, "p_value": float(p),
            "null_mean": float(sims.mean()), "null_p5": float(np.quantile(sims, 0.05)),
            "null_p95": float(np.quantile(sims, 0.95)), "n_sims": int(len(sims)),
            "samples": sims}


def trade_order_test(pnls: Sequence[float], initial_capital: float = 1.0,
                     n_sims: int = 1000, seed: int = 42) -> Dict[str, float]:
    """How lucky was the order of the trades?

    Shuffles the sequence of per-trade P&L. The total is unchanged by
    construction, so only path-dependent figures move: this answers
    'was the drawdown mild because of the order the wins and losses came in?',
    NOT 'is the strategy profitable?'. `dd_percentile` near 100 means the
    realised drawdown was better than nearly every reordering.
    """
    p = np.asarray([x for x in pnls if np.isfinite(x)], dtype=float)
    if len(p) < 10:
        return {"error": "Fewer than 10 trades.", "dd_percentile": float("nan")}

    def mdd(seq: np.ndarray) -> float:
        eq = initial_capital + np.concatenate([[0.0], np.cumsum(seq)])
        peak = np.maximum.accumulate(eq)
        return float(((eq - peak) / np.where(peak > 0, peak, 1.0)).min())

    rng = np.random.default_rng(seed)
    actual = mdd(p)
    sims = np.array([mdd(rng.permutation(p)) for _ in range(n_sims)])
    return {"actual_max_dd": actual,
            "dd_percentile": float((sims <= actual).mean() * 100.0),
            "sim_median_dd": float(np.median(sims)),
            "sim_p5_dd": float(np.quantile(sims, 0.05)),
            "n_trades": int(len(p)), "n_sims": n_sims}


def walk_forward_windows(returns: pd.Series, n_windows: int = 5,
                         ppy: int = 252) -> pd.DataFrame:
    """Contiguous, equal windows; one row of statistics each."""
    r = returns.dropna()
    rows = []
    for i, seg in enumerate(np.array_split(np.arange(len(r)), max(2, n_windows)), 1):
        if len(seg) < 10:
            continue
        w = r.iloc[seg]
        eq = (1 + w).cumprod()
        sd = w.std(ddof=1)
        rows.append({
            "Window": i, "Start": w.index[0].date(), "End": w.index[-1].date(),
            "Return": float(eq.iloc[-1] - 1.0),
            "Sharpe": float(w.mean() / sd * math.sqrt(ppy)) if sd > 1e-12 else float("nan"),
            "Max Drawdown": float((eq / eq.cummax() - 1).min()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Overfitting of a selection procedure
# ----------------------------------------------------------------------
def _block_edges(n: int, s: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, n, s + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(s)]


def _sharpe_from_sums(sm: np.ndarray, sq: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    mean = sm / cnt
    var = (sq - cnt * mean ** 2) / np.maximum(cnt - 1, 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.where(sd > 1e-12, mean / np.where(sd > 1e-12, sd, 1.0), 0.0)


def pbo(perf: pd.DataFrame, n_splits: int = 16, ppy: int = 252) -> Dict[str, object]:
    """Probability of Backtest Overfitting by combinatorially symmetric CV.

    `perf` holds the return stream of every configuration tried (rows = time,
    columns = configurations, all on the same dates). The history is cut into
    `n_splits` blocks; for every way of choosing half the blocks as 'in-sample',
    the configuration that looks best in-sample is found and its rank among all
    configurations is read off out-of-sample. PBO is the share of splits where
    the in-sample winner lands *below the median* out of sample. Near 0: the
    selection procedure picks things that keep working. Near 0.5 or above:
    picking the best of these is no better than picking at random.
    """
    m = perf.dropna(how="any")
    if m.shape[1] < 3:
        return {"error": "Need at least 3 configurations.", "pbo": float("nan")}
    s = int(n_splits) - int(n_splits) % 2
    if len(m) < s * 5:
        s = max(4, (len(m) // 5) - (len(m) // 5) % 2)
    if s < 4:
        return {"error": "History too short.", "pbo": float("nan")}
    x = m.to_numpy()
    edges = _block_edges(len(x), s)
    sums = np.stack([x[a:b].sum(axis=0) for a, b in edges])
    sqs = np.stack([(x[a:b] ** 2).sum(axis=0) for a, b in edges])
    cnts = np.array([b - a for a, b in edges], dtype=float)[:, None]
    n_cfg = x.shape[1]

    logits, oos_sr, is_sr, degr = [], [], [], []
    for comb in itertools.combinations(range(s), s // 2):
        ins = np.array(comb)
        oos = np.setdiff1d(np.arange(s), ins)
        sr_is = _sharpe_from_sums(sums[ins].sum(0), sqs[ins].sum(0), cnts[ins].sum(0))
        sr_oos = _sharpe_from_sums(sums[oos].sum(0), sqs[oos].sum(0), cnts[oos].sum(0))
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1)
        w = (rank + 1) / (n_cfg + 1)
        logits.append(math.log(w / (1 - w)))
        is_sr.append(sr_is[best] * math.sqrt(ppy))
        oos_sr.append(sr_oos[best] * math.sqrt(ppy))
    logits = np.array(logits)
    is_sr, oos_sr = np.array(is_sr), np.array(oos_sr)
    slope = float(np.polyfit(is_sr, oos_sr, 1)[0]) if np.std(is_sr) > 1e-9 else float("nan")
    return {"pbo": float((logits <= 0).mean()), "n_splits": s,
            "n_combinations": len(logits), "n_configs": n_cfg,
            "mean_is_sharpe": float(is_sr.mean()), "mean_oos_sharpe": float(oos_sr.mean()),
            "prob_oos_loss": float((oos_sr < 0).mean()),
            "is_oos_slope": slope, "logits": logits, "is_sharpes": is_sr,
            "oos_sharpes": oos_sr}


def purged_kfold(n: int, k: int = 5, purge: int = 5, embargo: int = 0
                 ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K contiguous test folds; training drops `purge` bars before each test
    fold (their features overlap its labels) and `embargo` bars after it
    (serial correlation leaks the test period forward)."""
    folds = _block_edges(n, k)
    out = []
    for a, b in folds:
        test = np.arange(a, b)
        keep = np.ones(n, dtype=bool)
        keep[max(0, a - purge):min(n, b + embargo)] = False
        out.append((np.flatnonzero(keep), test))
    return out


def cpcv_splits(n: int, n_groups: int = 10, k_test: int = 2, purge: int = 5,
                embargo: int = 0) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
    """Combinatorial purged cross-validation splits.

    The sample is cut into `n_groups`; every choice of `k_test` groups is a
    test set (C(n_groups, k_test) splits). Because each group is tested in
    many splits, the out-of-sample record can be reassembled into several
    complete paths instead of one. Training indices adjacent to a test group
    are purged/embargoed as in `purged_kfold`.
    """
    groups = _block_edges(n, n_groups)
    out = []
    for comb in itertools.combinations(range(n_groups), k_test):
        test_mask = np.zeros(n, dtype=bool)
        keep = np.ones(n, dtype=bool)
        for g in comb:
            a, b = groups[g]
            test_mask[a:b] = True
            keep[max(0, a - purge):min(n, b + embargo)] = False
        keep &= ~test_mask
        out.append((np.flatnonzero(keep), np.flatnonzero(test_mask), comb))
    return out


def cpcv_selection(perf: pd.DataFrame, n_groups: int = 10, k_test: int = 2,
                   purge: int = 5, embargo: int = 0, ppy: int = 252) -> Dict[str, object]:
    """Out-of-sample behaviour of 'choose the best configuration in training'.

    For every CPCV split the configuration with the best training Sharpe is
    picked and scored on the held-out groups. The spread of those held-out
    Sharpes -- not the single best backtest -- is the honest range of what
    the selection procedure delivers.
    """
    m = perf.dropna(how="any")
    if m.shape[1] < 2 or len(m) < n_groups * 5:
        return {"error": "Need at least 2 configurations and enough history.",
                "median_oos_sharpe": float("nan")}
    x = m.to_numpy()

    def sharpes(rows: np.ndarray) -> np.ndarray:
        seg = x[rows]
        sd = seg.std(axis=0, ddof=1)
        return np.where(sd > 1e-12, seg.mean(axis=0) / np.where(sd > 1e-12, sd, 1.0), 0.0)

    oos, train, picked = [], [], []
    for tr, te, _ in cpcv_splits(len(x), n_groups, k_test, purge, embargo):
        s_tr = sharpes(tr)
        best = int(np.argmax(s_tr))
        oos.append(sharpes(te)[best] * math.sqrt(ppy))
        train.append(s_tr[best] * math.sqrt(ppy))
        picked.append(best)
    oos, train = np.array(oos), np.array(train)
    n_paths = int(math.comb(n_groups - 1, k_test - 1))
    return {"n_splits": len(oos), "n_paths": n_paths,
            "median_oos_sharpe": float(np.median(oos)),
            "p10_oos_sharpe": float(np.quantile(oos, 0.10)),
            "p90_oos_sharpe": float(np.quantile(oos, 0.90)),
            "mean_train_sharpe": float(train.mean()),
            "prob_oos_loss": float((oos < 0).mean()),
            "degradation": float(train.mean() - oos.mean()),
            "picked_share": pd.Series(picked).map(lambda i: m.columns[i]).value_counts(normalize=True).to_dict(),
            "oos_sharpes": oos}
