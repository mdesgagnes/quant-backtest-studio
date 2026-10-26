"""Factor library and factor research.

A *factor* here is a function from a price panel to a score panel (same
index and columns): a higher score means the model expects a higher forward
return, unless the factor's `sign` says otherwise (then the score is flipped
so that higher is always better). Every factor at date t uses prices up to
and including t, nothing later.

Research functions answer, for one factor:

- `information_coefficient` -- does the cross-sectional rank of the score
  predict the rank of the forward return? (IC, ICIR, t-stat, hit rate)
- `quantile_returns`        -- do the top-scored names beat the bottom ones,
  and monotonically?
- `ic_decay`                -- how fast does the signal go stale?
- `screen`                  -- all of the above for every factor, with the
  multiple-testing correction a screen of 40 factors needs.
- `factor_to_weights`       -- turn a score into target weights for the engine.

The price-only library is deliberately modest (about 45 factors from the
published literature). Price-based factors work for any instrument the app
can load; the fundamental layer is covered by the existing `factor_rank`
strategy, which accepts any imported series.

Forward returns are measured from the close of day t+lag to the close of
t+lag+h. The default lag of 1 mirrors the engine's one-day execution lag:
the score is known at the close, the trade happens the next session.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Factor definitions
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    category: str
    expected: str          # the literature's expected sign: "+", "-" or "?"
    fn: Callable[[pd.DataFrame], pd.DataFrame]
    note: str = ""


FACTORS: Dict[str, Factor] = {}


def _reg(key, label, category, expected, note=""):
    def deco(fn):
        FACTORS[key] = Factor(key, label, category, expected, fn, note)
        return fn
    return deco


def _ret(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change(fill_method=None)


def _mom(n: int, skip: int = 0):
    def f(px):
        return px.shift(skip) / px.shift(n + skip) - 1.0
    return f


for _n, _lab in [(21, "1-month"), (63, "3-month"), (126, "6-month"), (252, "12-month")]:
    FACTORS[f"mom_{_n}"] = Factor(f"mom_{_n}", f"Momentum {_lab}", "Momentum", "+",
                                  _mom(_n), "Total return over the window.")
FACTORS["mom_12_1"] = Factor("mom_12_1", "Momentum 12-1", "Momentum", "+",
                             _mom(231, 21),
                             "12-month return skipping the latest month "
                             "(Jegadeesh & Titman 1993).")
FACTORS["mom_6_1"] = Factor("mom_6_1", "Momentum 6-1", "Momentum", "+",
                            _mom(105, 21), "6-month return skipping the latest month.")


@_reg("mom_accel", "Momentum acceleration", "Momentum", "+",
      "3-month return minus the preceding 3-month return.")
def _mom_accel(px):
    return (px / px.shift(63) - 1) - (px.shift(63) / px.shift(126) - 1)


@_reg("high_52w", "Closeness to 52-week high", "Momentum", "+",
      "Price over its 252-day high (George & Hwang 2004).")
def _high52(px):
    return px / px.rolling(252, min_periods=126).max()


@_reg("sharpe_126", "Risk-adjusted momentum 6m", "Momentum", "+",
      "Mean over standard deviation of daily returns, 126 days.")
def _sharpe126(px):
    r = _ret(px)
    return r.rolling(126, min_periods=63).mean() / r.rolling(126, min_periods=63).std()


@_reg("sharpe_252", "Risk-adjusted momentum 12m", "Momentum", "+",
      "Mean over standard deviation of daily returns, 252 days.")
def _sharpe252(px):
    r = _ret(px)
    return r.rolling(252, min_periods=126).mean() / r.rolling(252, min_periods=126).std()


@_reg("mom_consistency", "Momentum consistency", "Momentum", "+",
      "Share of positive months in the last 12 (a 'frog in the pan' proxy).")
def _consistency(px):
    m = px.resample("ME").last().pct_change(fill_method=None)
    pos = (m > 0).astype(float).where(m.notna()).rolling(12, min_periods=8).mean()
    return pos.reindex(px.index, method="ffill")


for _n, _lab in [(5, "1-week"), (10, "2-week")]:
    FACTORS[f"rev_{_n}"] = Factor(f"rev_{_n}", f"Short-term reversal {_lab}", "Reversal",
                                  "-", _mom(_n), "Recent winners tend to give back; sign flipped.")
FACTORS["rev_21"] = Factor("rev_21", "Short-term reversal 1-month", "Reversal", "-",
                           _mom(21), "One-month reversal (Jegadeesh 1990).")


@_reg("zscore_20", "Price z-score 20d", "Reversal", "-",
      "Distance from the 20-day mean in standard deviations.")
def _z20(px):
    return (px - px.rolling(20).mean()) / px.rolling(20).std()


@_reg("rsi_14", "RSI 14", "Reversal", "-", "Wilder's RSI; high = overbought.")
def _rsi14(px):
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


@_reg("rev_long", "Long-term reversal 3-5y", "Reversal", "-",
      "Return from t-1260 to t-252 (De Bondt & Thaler 1985).")
def _revlong(px):
    return px.shift(252) / px.shift(1260) - 1


for _n in (21, 63, 126):
    FACTORS[f"vol_{_n}"] = Factor(
        f"vol_{_n}", f"Realised volatility {_n}d", "Volatility", "-",
        (lambda n: lambda px: _ret(px).rolling(n, min_periods=max(10, n // 2)).std() * math.sqrt(TRADING_DAYS))(_n),
        "Low-volatility anomaly (Ang et al. 2006); sign flipped.")


@_reg("downvol_126", "Downside volatility 126d", "Volatility", "-",
      "Root mean square of negative returns.")
def _dvol(px):
    d = _ret(px).clip(upper=0)
    return np.sqrt((d ** 2).rolling(126, min_periods=63).mean()) * math.sqrt(TRADING_DAYS)


@_reg("max_ret_21", "Lottery: max daily return 1m", "Volatility", "-",
      "Largest one-day gain in the last month (Bali et al. 2011).")
def _maxret(px):
    return _ret(px).rolling(21, min_periods=15).max()


@_reg("skew_126", "Return skewness 126d", "Volatility", "-",
      "Negative skewness commands a premium; high skew is overpriced.")
def _skew(px):
    return _ret(px).rolling(126, min_periods=63).skew()


@_reg("kurt_126", "Excess kurtosis 126d", "Volatility", "-", "Tail heaviness.")
def _kurt(px):
    return _ret(px).rolling(126, min_periods=63).kurt()


@_reg("vol_of_vol", "Volatility of volatility", "Volatility", "-",
      "Standard deviation of 21-day volatility over 126 days.")
def _vov(px):
    v = _ret(px).rolling(21, min_periods=15).std()
    return v.rolling(126, min_periods=63).std() / v.rolling(126, min_periods=63).mean()


@_reg("vol_ratio", "Volatility ratio 21/126", "Volatility", "-",
      "Short-term versus long-term volatility.")
def _volratio(px):
    r = _ret(px)
    return r.rolling(21, min_periods=15).std() / r.rolling(126, min_periods=63).std()


@_reg("drawdown", "Current drawdown", "Risk", "+",
      "Price over its running 252-day high minus one (closer to zero = better).")
def _dd(px):
    return px / px.rolling(252, min_periods=60).max() - 1


@_reg("max_dd_252", "Worst drawdown 12m", "Risk", "+",
      "Deepest peak-to-trough decline in the last 252 days (shallower = better).")
def _maxdd(px):
    peak = px.rolling(252, min_periods=60).max()
    return (px / peak - 1).rolling(252, min_periods=60).min()


@_reg("ulcer_126", "Ulcer index 126d", "Risk", "-",
      "Root mean square of drawdowns over 126 days.")
def _ulcer(px):
    dd = px / px.rolling(126, min_periods=60).max() - 1
    return np.sqrt((dd ** 2).rolling(126, min_periods=60).mean())


@_reg("beta_252", "Market beta 12m", "Beta", "-",
      "Beta against the cross-sectional average return; betting-against-beta "
      "(Frazzini & Pedersen 2014), sign flipped.")
def _beta(px):
    r = _ret(px)
    m = r.mean(axis=1)
    cov = r.rolling(252, min_periods=126).cov(m)
    return cov.div(m.rolling(252, min_periods=126).var(), axis=0)


@_reg("corr_mkt", "Correlation to the universe", "Beta", "-",
      "126-day correlation with the cross-sectional average return.")
def _corr(px):
    r = _ret(px)
    m = r.mean(axis=1)
    return r.rolling(126, min_periods=63).corr(m)


@_reg("idio_vol", "Idiosyncratic volatility", "Beta", "-",
      "Volatility of the residual to the universe average (Ang et al. 2006).")
def _idio(px):
    r = _ret(px)
    m = r.mean(axis=1)
    beta = r.rolling(126, min_periods=63).cov(m).div(m.rolling(126, min_periods=63).var(), axis=0)
    resid = r.sub(beta.mul(m, axis=0))
    return resid.rolling(126, min_periods=63).std() * math.sqrt(TRADING_DAYS)


@_reg("price_sma_50", "Price over 50-day average", "Trend", "+")
def _ps50(px):
    return px / px.rolling(50, min_periods=50).mean() - 1


@_reg("price_sma_200", "Price over 200-day average", "Trend", "+")
def _ps200(px):
    return px / px.rolling(200, min_periods=200).mean() - 1


@_reg("sma_cross", "50/200 average spread", "Trend", "+",
      "Gap between the 50- and 200-day averages.")
def _cross(px):
    return px.rolling(50, min_periods=50).mean() / px.rolling(200, min_periods=200).mean() - 1


@_reg("macd", "MACD histogram (normalised)", "Trend", "+",
      "MACD line minus signal, divided by price.")
def _macd(px):
    line = px.ewm(span=12, adjust=False, min_periods=26).mean() - px.ewm(span=26, adjust=False, min_periods=26).mean()
    return (line - line.ewm(span=9, adjust=False, min_periods=9).mean()) / px


@_reg("donchian_pos", "Donchian channel position", "Trend", "+",
      "Where price sits in its 100-day high-low range (0 to 1).")
def _donch(px):
    hi, lo = px.rolling(100, min_periods=60).max(), px.rolling(100, min_periods=60).min()
    return (px - lo) / (hi - lo).replace(0, np.nan)


@_reg("efficiency_60", "Efficiency ratio 60d", "Trend", "+",
      "Net move over total path (Kaufman); high = clean trend.")
def _er(px):
    net = (px - px.shift(60)).abs()
    path = px.diff().abs().rolling(60, min_periods=40).sum()
    return net / path.replace(0, np.nan)


@_reg("trend_r2", "Trend quality (R-squared)", "Trend", "+",
      "R-squared of log price on time over 126 days, signed by slope.")
def _r2(px):
    lp = np.log(px)
    n = 126
    t = np.arange(n, dtype=float)
    tc = t - t.mean()

    def f(col):
        v = col.to_numpy()
        out = np.full(len(v), np.nan)
        for i in range(n - 1, len(v)):
            y = v[i - n + 1:i + 1]
            if np.isnan(y).any():
                continue
            yc = y - y.mean()
            sxx = (tc ** 2).sum()
            sxy = (tc * yc).sum()
            syy = (yc ** 2).sum()
            if syy > 0:
                out[i] = np.sign(sxy) * (sxy ** 2) / (sxx * syy)
        return pd.Series(out, index=col.index)

    return lp.apply(f)


@_reg("seasonal_12m", "Same-month last year", "Seasonality", "+",
      "Return of the same calendar month a year ago (Heston & Sadka 2008).")
def _seas(px):
    m = px.resample("ME").last().pct_change(fill_method=None).shift(11)
    return m.reindex(px.index, method="ffill")


@_reg("seasonal_avg", "Same-month 5y average", "Seasonality", "+",
      "Average return of this calendar month over the previous five years.")
def _seasavg(px):
    m = px.resample("ME").last().pct_change(fill_method=None)
    parts = [m.shift(12 * k - 1) for k in range(1, 6)]
    avg = sum(p.fillna(0) for p in parts) / sum(p.notna().astype(float) for p in parts).replace(0, np.nan)
    return avg.reindex(px.index, method="ffill")


@_reg("up_share_63", "Up-day share 63d", "Behavioural", "?",
      "Share of positive days over 63 sessions.")
def _upshare(px):
    return (_ret(px) > 0).astype(float).rolling(63, min_periods=40).mean()


@_reg("autocorr_126", "Return autocorrelation 126d", "Behavioural", "?",
      "Lag-1 autocorrelation of daily returns: positive = trending, negative = mean-reverting.")
def _autocorr(px):
    r = _ret(px)
    return pd.DataFrame({c: r[c].rolling(126, min_periods=63).corr(r[c].shift(1))
                         for c in r.columns})


def list_factors() -> pd.DataFrame:
    return pd.DataFrame([{"key": f.key, "label": f.label, "category": f.category,
                          "expected": f.expected, "note": f.note}
                         for f in FACTORS.values()])


def compute(key: str, prices: pd.DataFrame, oriented: bool = True) -> pd.DataFrame:
    """Score panel for one factor. `oriented=True` flips factors whose literature
    sign is negative, so that a higher score always means 'expected better'."""
    f = FACTORS[key]
    out = f.fn(prices).replace([np.inf, -np.inf], np.nan)
    out = out.reindex(index=prices.index, columns=prices.columns)
    return -out if (oriented and f.expected == "-") else out


# ----------------------------------------------------------------------
# Research
# ----------------------------------------------------------------------
def forward_returns(prices: pd.DataFrame, horizon: int = 21, lag: int = 1) -> pd.DataFrame:
    """Return from the close of t+lag to the close of t+lag+horizon."""
    return prices.shift(-(lag + horizon)) / prices.shift(-lag) - 1.0


def _rank_corr_rows(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Spearman correlation of each row of a with the same row of b."""
    ra = a.rank(axis=1)
    rb = b.rank(axis=1).where(ra.notna())
    ra = ra.where(rb.notna())
    ma, mb = ra.mean(axis=1), rb.mean(axis=1)
    da, db = ra.sub(ma, axis=0), rb.sub(mb, axis=0)
    cov = (da * db).sum(axis=1)
    den = np.sqrt((da ** 2).sum(axis=1) * (db ** 2).sum(axis=1))
    n = ra.notna().sum(axis=1)
    return (cov / den.replace(0, np.nan)).where(n >= 3)


def _newey_west_t(x: np.ndarray, lags: int) -> float:
    """t-statistic of the mean with a Newey-West variance. Forward returns over
    h days overlap, so successive ICs are correlated; an ordinary t-stat on
    them overstates significance by roughly sqrt(h)."""
    n = len(x)
    if n < 10:
        return float("nan")
    xm = x - x.mean()
    gamma0 = float((xm ** 2).sum() / n)
    s = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float((xm[k:] * xm[:-k]).sum() / n)
    s = max(s, 1e-18)
    return float(x.mean() / math.sqrt(s / n))


def _norm_two_sided_p(t: float) -> float:
    if not np.isfinite(t):
        return float("nan")
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def information_coefficient(scores: pd.DataFrame, prices: pd.DataFrame,
                            horizon: int = 21, lag: int = 1,
                            step: Optional[int] = None) -> Dict[str, object]:
    """Rank IC series and its statistics.

    Sampled every `step` sessions (default: the horizon) so the forward
    windows do not overlap; the t-statistic is Newey-West adjusted on top of
    that, which is conservative.
    """
    fwd = forward_returns(prices, horizon, lag)
    ic = _rank_corr_rows(scores, fwd).dropna()
    step = step or max(1, horizon)
    ic_s = ic.iloc[::step]
    x = ic_s.to_numpy()
    if len(x) < 8:
        return {"ic": ic, "n": len(x), "mean_ic": float("nan"), "icir": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"), "hit_rate": float("nan")}
    t = _newey_west_t(x, lags=max(1, horizon // step))
    return {"ic": ic, "ic_sampled": ic_s, "n": len(x), "mean_ic": float(x.mean()),
            "std_ic": float(x.std(ddof=1)),
            "icir": float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("nan"),
            "t_stat": t, "p_value": _norm_two_sided_p(t),
            "hit_rate": float((x > 0).mean())}


def quantile_returns(scores: pd.DataFrame, prices: pd.DataFrame, n_q: int = 5,
                     horizon: int = 21, lag: int = 1) -> Dict[str, object]:
    """Average forward return by score bucket, sampled at non-overlapping dates.

    With a small universe use few buckets (3 for under 15 names): a quintile
    of 8 ETFs is 1.6 securities and says nothing. Bucket 1 is the lowest
    score, bucket `n_q` the highest.
    """
    fwd = forward_returns(prices, horizon, lag)
    sc = scores.iloc[::max(1, horizon)]
    fw = fwd.reindex(sc.index)
    ok = sc.notna() & fw.notna()
    n_ok = ok.sum(axis=1)
    keep = n_ok >= max(n_q, 4)
    sc, fw, ok, n_ok = sc[keep].where(ok[keep]), fw[keep].where(ok[keep]), ok[keep], n_ok[keep]
    # Rank-based buckets (ties broken by order) are robust to repeated scores.
    bucket = np.ceil(sc.rank(axis=1, method="first").div(n_ok, axis=0) * n_q).clip(1, n_q)
    rows = [fw.where(bucket == q).mean(axis=1).rename(q) for q in range(1, n_q + 1)]
    top = (bucket == n_q).to_numpy()
    turnover = []
    for i in range(1, len(top)):
        if top[i].any():
            turnover.append(1 - (top[i] & top[i - 1]).sum() / top[i].sum())
    if sc.empty:
        return {"table": pd.DataFrame(), "spread": pd.Series(dtype=float)}
    tab = pd.concat(rows, axis=1).dropna(how="all")
    tab.columns = [int(c) for c in tab.columns]
    spread = (tab[n_q] - tab[1]).dropna()
    per_year = TRADING_DAYS / max(1, horizon)
    mean_by_q = tab.mean()
    # Spearman by hand (Pearson on ranks): pandas' method="spearman" needs scipy,
    # which the deployed app does not install.
    mono = float(pd.Series(mean_by_q.values).rank().corr(pd.Series(range(len(mean_by_q))).rank())) \
        if len(mean_by_q) > 2 else float("nan")
    sd = spread.std(ddof=1)
    return {"table": tab, "mean_by_q": mean_by_q, "spread": spread,
            "spread_mean": float(spread.mean()),
            "spread_sharpe": float(spread.mean() / sd * math.sqrt(per_year)) if sd > 0 else float("nan"),
            "spread_hit": float((spread > 0).mean()),
            "monotonicity": mono,
            "top_turnover": float(np.mean(turnover)) if turnover else float("nan"),
            "n_dates": len(tab), "n_q": n_q}


def ic_decay(scores: pd.DataFrame, prices: pd.DataFrame,
             horizons=(1, 5, 10, 21, 42, 63), lag: int = 1) -> pd.DataFrame:
    rows = []
    for h in horizons:
        r = information_coefficient(scores, prices, h, lag)
        rows.append({"Horizon (days)": h, "Mean IC": r["mean_ic"], "ICIR": r["icir"],
                     "t-stat": r["t_stat"], "n": r["n"]})
    return pd.DataFrame(rows)


def multiple_testing(p: pd.Series) -> pd.DataFrame:
    """Bonferroni and Benjamini-Hochberg adjusted p-values.

    Screening 45 factors at 5% will 'find' about two by chance. The adjusted
    columns say how many survive having looked."""
    p = p.dropna()
    m = len(p)
    if m == 0:
        return pd.DataFrame(columns=["p", "p_bonferroni", "p_bh"])
    order = p.sort_values().index
    ps = p.loc[order].to_numpy()
    bh = np.minimum.accumulate((ps * m / (np.arange(m) + 1))[::-1])[::-1]
    return pd.DataFrame({"p": ps, "p_bonferroni": np.minimum(ps * m, 1.0),
                         "p_bh": np.minimum(bh, 1.0)}, index=order)


def screen(prices: pd.DataFrame, keys: Optional[List[str]] = None, horizon: int = 21,
           lag: int = 1, n_q: Optional[int] = None) -> pd.DataFrame:
    """Every factor, ranked by |t-stat|, with multiple-testing adjustment."""
    keys = keys or list(FACTORS)
    n_q = n_q or (3 if prices.shape[1] < 15 else 5)
    rows = []
    for k in keys:
        try:
            sc = compute(k, prices)
            ic = information_coefficient(sc, prices, horizon, lag)
            qr = quantile_returns(sc, prices, n_q, horizon, lag)
            rows.append({"key": k, "Factor": FACTORS[k].label, "Category": FACTORS[k].category,
                         "Mean IC": ic["mean_ic"], "ICIR": ic["icir"], "t-stat": ic["t_stat"],
                         "p": ic["p_value"], "IC hit rate": ic["hit_rate"],
                         "Spread Sharpe": qr.get("spread_sharpe", np.nan),
                         "Monotonicity": qr.get("monotonicity", np.nan),
                         "Top-bucket turnover": qr.get("top_turnover", np.nan),
                         "n": ic["n"]})
        except Exception as exc:                 # one bad factor must not sink the screen
            rows.append({"key": k, "Factor": FACTORS[k].label, "Category": FACTORS[k].category,
                         "error": str(exc)[:80]})
    df = pd.DataFrame(rows).set_index("key")
    if "p" in df:
        adj = multiple_testing(df["p"])
        df = df.join(adj[["p_bonferroni", "p_bh"]])
        df["abs_t"] = df["t-stat"].abs()
        df = df.sort_values("abs_t", ascending=False).drop(columns="abs_t")
    return df


def ic_correlation(scores_by_key: Dict[str, pd.DataFrame], prices: pd.DataFrame,
                   horizon: int = 21, lag: int = 1) -> pd.DataFrame:
    """Correlation between factors' IC series: do two 'different' factors
    win and lose on the same dates?"""
    series = {k: information_coefficient(s, prices, horizon, lag)["ic"] for k, s in scores_by_key.items()}
    return pd.DataFrame(series).corr()


def factor_to_weights(scores: pd.DataFrame, top_n: int = 3, gross: float = 1.0,
                      min_score: Optional[float] = None, weighting: str = "equal",
                      prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Hold the `top_n` highest-scored names. Dates with fewer than `top_n`
    valid scores hold what exists; dates with none hold cash. `min_score`
    (in score units) drops names below an absolute floor, so the portfolio
    can go to cash when nothing qualifies."""
    ok = scores.notna()
    if min_score is not None:
        ok &= scores >= min_score
    rk = scores.where(ok).rank(axis=1, ascending=False, method="first")
    mask = (rk <= top_n) & ok
    if weighting == "inverse_vol" and prices is not None:
        v = _ret(prices).rolling(60, min_periods=30).std()
        inv = (1.0 / v.where(mask & (v > 0)))
        w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        w = mask.astype(float).div(mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w * gross
