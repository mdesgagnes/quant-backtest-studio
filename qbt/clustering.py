"""Diversification: which return streams are distinct bets, and which repeat
an exposure already held.

Pure functions over a DataFrame of periodic returns (one column per stream,
missing values allowed, different start dates allowed). Nothing here reads
the interface or the network; every parameter comes from a
`ClusteringConfig`, which loads from the `clustering:` block of a YAML file
(`configs/clustering_default.yaml`).

Method, kept deliberately simple:

1. Align. Returns are compounded to a common frequency (weekly by default:
   Canadian and US closes are not synchronous, and daily correlations
   between them are biased toward zero), then cut to the window every
   stream shares. Streams that are too short or never move are dropped,
   with the reason.
2. Correlate (Pearson, or Spearman) and turn correlation into a distance,
   d = sqrt(0.5 * (1 - rho)): 0 for identical streams, 1 for opposite ones.
3. Cluster hierarchically. With average linkage (the default) the cut line
   is itself a correlation: streams whose average correlation is at or
   above the threshold are one bet. Ward is available, and suits a cut at a
   fixed number of clusters.
4. Summarise: effective number of independent bets from the eigenvalues of
   the correlation matrix, average correlation within and between
   clusters, every pair above the threshold, and the same matrix measured
   only in the benchmark's worst periods.

No statistic uses data outside the window it is given.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

FREQUENCIES = {"daily": ("Daily", None, 252), "weekly": ("Weekly", "W-FRI", 52),
               "monthly": ("Monthly", "ME", 12)}
_RANK = {"daily": 0, "business-daily": 0, "weekly": 1, "monthly": 2,
         "quarterly": 3, "semi-annual": 4, "annual": 5}
_PPY = {"daily": 252, "business-daily": 252, "weekly": 52, "monthly": 12,
        "quarterly": 4, "semi-annual": 2, "annual": 1}
MODULE = "Diversification"


class ClusteringError(ValueError):
    """Raised with a message meant for the user."""


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class StressSettings:
    benchmark: Optional[str] = None   # column name; None = caller's choice, else equal weight
    percentile: float = 0.20          # worst share of benchmark periods
    min_observations: int = 26        # below this the down-market matrix is flagged


@dataclass
class ClusteringConfig:
    frequency: str = "weekly"         # daily | weekly | monthly
    min_observations: int = 52        # per stream, and for the shared window
    correlation: str = "pearson"      # pearson | spearman
    linkage: str = "average"          # average | ward
    cut: str = "threshold"            # threshold | clusters
    threshold: float = 0.80           # correlation at or above which streams are one bet
    n_clusters: int = 4               # used when cut = clusters
    effective_bets: str = "entropy"   # entropy | participation
    rolling_window: int = 52          # periods, for the pair-stability chart
    stress: StressSettings = field(default_factory=StressSettings)

    def validate(self) -> "ClusteringConfig":
        checks = [
            (self.frequency in FREQUENCIES, "frequency must be daily, weekly or monthly"),
            (self.correlation in ("pearson", "spearman"), "correlation must be pearson or spearman"),
            (self.linkage in ("average", "ward"), "linkage must be average or ward"),
            (self.cut in ("threshold", "clusters"), "cut must be threshold or clusters"),
            (self.effective_bets in ("entropy", "participation"),
             "effective_bets must be entropy or participation"),
            (-1.0 < float(self.threshold) < 1.0, "threshold must be between -1 and 1"),
            (int(self.n_clusters) >= 1, "n_clusters must be at least 1"),
            (int(self.min_observations) >= 3, "min_observations must be at least 3"),
            (int(self.rolling_window) >= 3, "rolling_window must be at least 3"),
            (0.0 < float(self.stress.percentile) < 1.0, "stress percentile must be between 0 and 1"),
            (int(self.stress.min_observations) >= 3, "stress min_observations must be at least 3"),
        ]
        bad = [msg for ok, msg in checks if not ok]
        if bad:
            raise ClusteringError("Clustering settings: " + "; ".join(bad) + ".")
        return self


def config_from_dict(data: Optional[Dict[str, Any]]) -> ClusteringConfig:
    """A config from a mapping; unknown keys are ignored, missing keys take defaults."""
    data = dict(data or {})
    if "clustering" in data and isinstance(data["clustering"], dict):
        data = dict(data["clustering"])
    stress = data.pop("stress", None) or {}
    known = set(ClusteringConfig.__dataclass_fields__) - {"stress"}
    cfg = ClusteringConfig(**{k: v for k, v in data.items() if k in known})
    s_known = set(StressSettings.__dataclass_fields__)
    cfg.stress = StressSettings(**{k: v for k, v in stress.items() if k in s_known})
    return cfg.validate()


def config_from_yaml(text: str) -> ClusteringConfig:
    """Reads the `clustering:` block of a YAML document."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ClusteringError(f"The YAML could not be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ClusteringError("The YAML must contain a clustering: block.")
    return config_from_dict(data)


def config_to_yaml(cfg: ClusteringConfig) -> str:
    return yaml.safe_dump({"clustering": asdict(cfg)}, sort_keys=False, allow_unicode=True)


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------
@dataclass
class Alignment:
    returns: pd.DataFrame        # complete rows only, at `frequency`
    frequency: str               # "weekly", ...
    ppy: int
    dropped: pd.DataFrame        # Stream, Reason
    truncated: pd.DataFrame      # Stream, Own history from, Used from, Own history to, Used to
    gap_rows: int                # rows inside the window removed for a missing value
    notes: List[str] = field(default_factory=list)


def _native_frequency(rets: pd.DataFrame) -> str:
    from .returns_input import detect_frequency
    idx = rets.dropna(how="all").index
    label, _ = detect_frequency(idx)
    return "daily" if label in ("undetermined", "business-daily") else label


def _to_frequency(s: pd.Series, rule: Optional[str]) -> pd.Series:
    """Compounds one stream on its own dates, so a holiday only one exchange
    observes does not cost the other stream its week."""
    s = s.dropna()
    if rule is None or s.empty:
        return s
    g = (1.0 + s).resample(rule)
    out = g.prod() - 1.0
    return out[g.count() > 0]


def align(rets: pd.DataFrame, cfg: ClusteringConfig,
          start: Any = None, end: Any = None) -> Alignment:
    """Returns at the configured frequency over the window every kept stream
    shares, with a report of what was dropped or shortened and why."""
    cfg.validate()
    df = rets.copy()
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    df.columns = [str(c) for c in df.columns]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    df = df.apply(pd.to_numeric, errors="coerce")
    notes: List[str] = []
    dropped: List[Tuple[str, str]] = []

    native = _native_frequency(df) if len(df.dropna(how="all")) >= 3 else "daily"
    target = cfg.frequency
    if _RANK.get(native, 0) > _RANK[target]:
        notes.append(f"The data is {native}, so it is used as is rather than "
                     f"{target}.")
        target, rule = native, None
    elif _RANK.get(native, 0) == _RANK[target]:
        rule = None
    else:
        rule = FREQUENCIES[target][1]
    ppy = _PPY.get(target, 52)
    word = target

    conv: Dict[str, pd.Series] = {}
    for c in df.columns:
        s = _to_frequency(df[c], rule)
        if s.empty:
            dropped.append((c, "no returns in the selected dates"))
            continue
        conv[c] = s
    data = pd.DataFrame(conv).sort_index()

    m = int(cfg.min_observations)
    for c in list(data.columns):
        n = int(data[c].notna().sum())
        if n < m:
            dropped.append((c, f"only {n} {word} returns, below the minimum of {m}"))
            data = data.drop(columns=c)
        elif float(data[c].dropna().std(ddof=0)) < 1e-12:
            dropped.append((c, "its return never changes (zero variance)"))
            data = data.drop(columns=c)

    # Shared window: while too few complete rows remain, drop the stream
    # whose removal frees the most shared history (a late start, an early
    # end or a long gap).
    while data.shape[1] > 2:
        complete_n = len(data.dropna())
        if complete_n >= m:
            break
        gain = {c: len(data.drop(columns=c).dropna()) for c in data.columns}
        worst = max(data.columns, key=lambda c: (gain[c], -data[c].notna().sum()))
        s = data[worst]
        dropped.append((worst, f"covers {s.first_valid_index().date()} to "
                               f"{s.last_valid_index().date()}: keeping it would leave "
                               f"only {complete_n} shared {word} returns"))
        data = data.drop(columns=worst)

    if data.shape[1] < 2:
        raise ClusteringError("At least two streams with enough history are needed. "
                              + _dropped_text(dropped))
    firsts = {c: data[c].first_valid_index() for c in data.columns}
    lasts = {c: data[c].last_valid_index() for c in data.columns}
    lo, hi = max(firsts.values()), min(lasts.values())
    window = data.loc[lo:hi] if lo <= hi else data.iloc[0:0]
    complete = window.dropna()
    if len(complete) < m:
        raise ClusteringError(
            f"Only {len(complete)} {word} returns are shared by "
            f"{', '.join(data.columns)}; the minimum is {m}. " + _dropped_text(dropped))
    # A stream can be constant over the shared window even if not overall.
    flat = [c for c in complete.columns if float(complete[c].std(ddof=0)) < 1e-12]
    for c in flat:
        dropped.append((c, "its return never changes in the shared window (zero variance)"))
    complete = complete.drop(columns=flat)
    if complete.shape[1] < 2:
        raise ClusteringError("At least two streams that move are needed. "
                              + _dropped_text(dropped))

    gap_rows = int(len(window) - len(complete))
    if gap_rows:
        notes.append(f"{gap_rows} {word} period{'s' if gap_rows != 1 else ''} inside "
                     f"the window had a missing value and {'were' if gap_rows != 1 else 'was'} "
                     f"left out for every stream.")
    trunc = []
    for c in complete.columns:
        if firsts[c] < complete.index[0] or lasts[c] > complete.index[-1]:
            trunc.append({"Stream": c, "Own history from": firsts[c].date(),
                          "Used from": complete.index[0].date(),
                          "Own history to": lasts[c].date(),
                          "Used to": complete.index[-1].date()})
    return Alignment(
        returns=complete, frequency=word, ppy=ppy,
        dropped=pd.DataFrame(dropped, columns=["Stream", "Reason"]),
        truncated=pd.DataFrame(trunc, columns=["Stream", "Own history from", "Used from",
                                               "Own history to", "Used to"]),
        gap_rows=gap_rows, notes=notes)


def _dropped_text(dropped: List[Tuple[str, str]]) -> str:
    if not dropped:
        return ""
    return "Left out: " + "; ".join(f"{s} ({r})" for s, r in dropped) + "."


# ----------------------------------------------------------------------
# Correlation, distance, clusters
# ----------------------------------------------------------------------
def correlation(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix of complete rows; the diagonal is exactly 1."""
    c = returns.corr(method=method)
    v = c.to_numpy(copy=True)
    np.fill_diagonal(v, 1.0)
    return pd.DataFrame(np.clip(v, -1.0, 1.0), index=c.index, columns=c.columns)


def distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Correlation distance d = sqrt(0.5 * (1 - rho))."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def rho_to_distance(rho: float) -> float:
    return math.sqrt(max(0.0, 0.5 * (1.0 - rho)))


def distance_to_rho(d: float) -> float:
    return 1.0 - 2.0 * d * d


@dataclass
class ClusterResult:
    labels: pd.Series            # stream -> cluster number (1 = leftmost on the dendrogram)
    linkage: np.ndarray          # scipy linkage matrix
    order: List[str]             # leaf order, left to right
    cut_height: float            # on the distance axis
    silhouette: float            # guide only; NaN when undefined
    n_clusters: int


def cluster(corr: pd.DataFrame, cfg: ClusteringConfig) -> ClusterResult:
    """Hierarchical clustering on correlation distance, cut by threshold or count."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    names = [str(c) for c in corr.columns]
    n = len(names)
    if n < 2:
        raise ClusteringError("At least two streams are needed to form clusters.")
    dist = distance(corr)
    dvec = squareform(dist.to_numpy(), checks=False)
    Z = linkage(dvec, method=cfg.linkage)
    heights = Z[:, 2]
    if cfg.cut == "threshold":
        h = rho_to_distance(float(cfg.threshold))
        raw = fcluster(Z, t=h, criterion="distance")
    else:
        k = int(min(max(1, cfg.n_clusters), n))
        raw = fcluster(Z, t=k, criterion="maxclust")
        if k >= n:
            h = heights[0] / 2.0
        elif k <= 1:
            h = heights[-1] * 1.02 + 1e-9
        else:
            h = float((heights[n - k - 1] + heights[n - k]) / 2.0)
    order = [names[i] for i in leaves_list(Z)]
    pos = {nm: i for i, nm in enumerate(names)}
    renum: Dict[int, int] = {}
    for nm in order:
        renum.setdefault(int(raw[pos[nm]]), len(renum) + 1)
    labels = pd.Series({nm: renum[int(raw[pos[nm]])] for nm in names}, name="Cluster")
    return ClusterResult(labels=labels, linkage=Z, order=order, cut_height=float(h),
                         silhouette=silhouette(dist, labels),
                         n_clusters=int(labels.nunique()))


def silhouette(dist: pd.DataFrame, labels: pd.Series) -> float:
    """Mean silhouette width on the correlation distance. A guide only:
    near 1, clusters are well separated; near 0, the cut is arbitrary."""
    k = labels.nunique()
    n = len(labels)
    if k < 2 or k >= n:
        return float("nan")
    D = dist.loc[labels.index, labels.index].to_numpy()
    lab = labels.to_numpy()
    s = np.zeros(n)
    for i in range(n):
        same = (lab == lab[i])
        if same.sum() <= 1:
            continue
        a = D[i, same & (np.arange(n) != i)].mean()
        b = min(D[i, lab == o].mean() for o in set(lab) if o != lab[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


# ----------------------------------------------------------------------
# Diversification measures
# ----------------------------------------------------------------------
def effective_bets(corr: pd.DataFrame, method: str = "entropy") -> float:
    """Effective number of independent bets from the eigenvalues of the
    correlation matrix: N for uncorrelated streams, 1 for identical ones.

    entropy: exp of the Shannon entropy of the eigenvalue shares.
    participation: (sum of eigenvalues)^2 / sum of squared eigenvalues.
    """
    v = corr.to_numpy(dtype=float)
    if v.size == 0:
        return float("nan")
    lam = np.clip(np.linalg.eigvalsh((v + v.T) / 2.0), 0.0, None)
    tot = lam.sum()
    if tot <= 0:
        return float("nan")
    if method == "participation":
        return float(tot ** 2 / (lam ** 2).sum())
    p = lam[lam > 0] / tot
    return float(np.exp(-(p * np.log(p)).sum()))


def cluster_correlation(corr: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
    """Average pairwise correlation within clusters and between clusters."""
    names = list(labels.index)
    v = corr.loc[names, names].to_numpy()
    lab = labels.to_numpy()
    iu = np.triu_indices(len(names), k=1)
    same = lab[iu[0]] == lab[iu[1]]
    vals = v[iu]
    return {"within": float(vals[same].mean()) if same.any() else float("nan"),
            "between": float(vals[~same].mean()) if (~same).any() else float("nan")}


def nearest(corr: pd.DataFrame) -> pd.DataFrame:
    """For each stream, the other stream it is most correlated with."""
    v = corr.to_numpy(copy=True).astype(float)
    np.fill_diagonal(v, -np.inf)
    j = v.argmax(axis=1)
    cols = list(corr.columns)
    return pd.DataFrame({"Closest to": [cols[k] for k in j],
                         "Correlation": [v[i, k] for i, k in enumerate(j)]},
                        index=corr.index)


def cluster_table(returns: pd.DataFrame, corr: pd.DataFrame, res: ClusterResult,
                  ppy: int) -> pd.DataFrame:
    """Stream, cluster, nearest neighbour and its correlation, annualized
    volatility and max drawdown, all over the aligned window."""
    near = nearest(corr)
    rows = []
    for nm in res.order:
        r = returns[nm]
        eq = (1.0 + r).cumprod()
        eq = pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)])
        rows.append({"Stream": nm, "Cluster": int(res.labels[nm]),
                     "Cluster size": int((res.labels == res.labels[nm]).sum()),
                     "Closest to": near.at[nm, "Closest to"],
                     "Correlation": float(near.at[nm, "Correlation"]),
                     "Volatility": float(r.std(ddof=0) * math.sqrt(ppy)),
                     "Max drawdown": float((eq / eq.cummax() - 1.0).min())})
    return pd.DataFrame(rows)


def redundancy_table(corr: pd.DataFrame, threshold: float,
                     down: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every pair at or above the threshold, most correlated first."""
    names = list(corr.columns)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = float(corr.iat[i, j])
            if rho >= threshold:
                row = {"Stream": names[i], "Same bet as": names[j], "Correlation": rho}
                if down is not None:
                    row["Down-market correlation"] = float(down.loc[names[i], names[j]])
                rows.append(row)
    cols = ["Stream", "Same bet as", "Correlation"] + (
        ["Down-market correlation"] if down is not None else [])
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("Correlation", ascending=False, ignore_index=True)


def down_market_mask(bench: pd.Series, pct: float) -> pd.Series:
    """True on the benchmark's worst `pct` share of periods (at least one)."""
    b = bench.dropna()
    k = max(1, int(math.ceil(pct * len(b))))
    worst = b.sort_values(kind="mergesort").index[:k]
    return pd.Series(bench.index.isin(worst), index=bench.index)


def stress_correlation(returns: pd.DataFrame, bench: pd.Series, pct: float,
                       method: str = "pearson") -> Tuple[pd.DataFrame, int]:
    """Correlation measured only in the benchmark's worst periods, with the
    number of periods used. `bench` must be on the same dates as `returns`."""
    b = bench.reindex(returns.index)
    mask = down_market_mask(b, pct)
    sub = returns[mask.to_numpy()]
    if len(sub) < 3:
        nan = pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
        return nan, int(len(sub))
    c = sub.corr(method=method)
    v = c.to_numpy(copy=True)
    np.fill_diagonal(v, 1.0)
    return pd.DataFrame(v, index=c.index, columns=c.columns), int(len(sub))


def rolling_correlation(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling correlation of two aligned return series."""
    df = pd.concat([a, b], axis=1).dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).dropna()


# ----------------------------------------------------------------------
# Everything at once
# ----------------------------------------------------------------------
@dataclass
class Diversification:
    cfg: ClusteringConfig
    alignment: Alignment
    corr: pd.DataFrame
    clusters: ClusterResult
    effective_bets: float
    within: float
    between: float
    table: pd.DataFrame
    redundant: pd.DataFrame
    down_corr: Optional[pd.DataFrame]
    down_n: int
    down_label: str

    @property
    def n_streams(self) -> int:
        return self.corr.shape[0]

    @property
    def window_text(self) -> str:
        r = self.alignment.returns
        return (f"{len(r):,} {self.alignment.frequency} returns, "
                f"{r.index[0].date()} to {r.index[-1].date()}")


def _bench_for(rets: pd.DataFrame, aligned: Alignment, cfg: ClusteringConfig,
               bench: Optional[pd.Series], bench_label: Optional[str]
               ) -> Tuple[pd.Series, str]:
    """The down-market reference on the aligned dates."""
    idx = aligned.returns.index
    name = cfg.stress.benchmark
    if name and name in aligned.returns.columns:
        return aligned.returns[name], str(name)
    if name and name in rets.columns:
        bench, bench_label = rets[name], str(name)
    if bench is not None:
        rule = None if aligned.frequency not in ("weekly", "monthly") else \
            FREQUENCIES[aligned.frequency][1]
        native = _native_frequency(bench.to_frame())
        if _RANK.get(native, 0) >= _RANK.get(aligned.frequency, 0):
            rule = None
        b = _to_frequency(pd.Series(bench, dtype=float), rule).reindex(idx)
        if b.notna().sum() >= 3:
            return b, str(bench_label or "benchmark")
    return aligned.returns.mean(axis=1), "equal-weight average of the streams"


def analyse(rets: pd.DataFrame, cfg: Optional[ClusteringConfig] = None,
            bench: Optional[pd.Series] = None, bench_label: Optional[str] = None,
            start: Any = None, end: Any = None) -> Diversification:
    """Align, correlate, cluster and summarise a set of return streams.

    `bench` (optional) defines the down markets when the config names no
    benchmark; without either, the equal-weight average of the streams does.
    """
    cfg = (cfg or ClusteringConfig()).validate()
    rets = rets.copy()
    rets.columns = [str(c) for c in rets.columns]
    al = align(rets, cfg, start, end)
    corr = correlation(al.returns, cfg.correlation)
    res = cluster(corr, cfg)
    if bench is not None:
        bench = bench.copy()
        bench.index = pd.DatetimeIndex(bench.index)
        if start is not None:
            bench = bench[bench.index >= pd.Timestamp(start)]
        if end is not None:
            bench = bench[bench.index <= pd.Timestamp(end)]
    b, blabel = _bench_for(rets, al, cfg, bench, bench_label)
    b = b.reindex(al.returns.index)
    keep = b.notna().to_numpy()
    down, dn = stress_correlation(al.returns[keep], b[keep], cfg.stress.percentile,
                                  cfg.correlation)
    cc = cluster_correlation(corr, res.labels)
    return Diversification(
        cfg=cfg, alignment=al, corr=corr, clusters=res,
        effective_bets=effective_bets(corr, cfg.effective_bets),
        within=cc["within"], between=cc["between"],
        table=cluster_table(al.returns, corr, res, al.ppy),
        redundant=redundancy_table(corr, float(cfg.threshold), down),
        down_corr=down, down_n=dn, down_label=blabel)


# ----------------------------------------------------------------------
# Candidate test
# ----------------------------------------------------------------------
@dataclass
class CandidateVerdict:
    candidate: str
    verdict: str                 # "duplicates" | "overlaps" | "new" | "untestable"
    nearest: Optional[str]
    max_correlation: float
    joins: List[str]             # holdings in the candidate's cluster
    bets_before: float
    bets_after: float
    sentence: str
    bets_if_independent: float = float("nan")
    window: str = ""
    dropped: Optional[pd.DataFrame] = None


def candidate_test(rets: pd.DataFrame, holdings: List[str], candidate: str,
                   cfg: Optional[ClusteringConfig] = None,
                   start: Any = None, end: Any = None) -> CandidateVerdict:
    """Does `candidate` add a new bet to `holdings`, or repeat one?"""
    cfg = (cfg or ClusteringConfig()).validate()
    holdings = [h for h in dict.fromkeys(map(str, holdings)) if h != str(candidate)]
    if not holdings:
        raise ClusteringError("Pick at least one current holding to compare with.")
    cols = holdings + [str(candidate)]
    df = rets.copy()
    df.columns = [str(c) for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ClusteringError("Not loaded: " + ", ".join(missing) + ".")
    try:
        al = align(df[cols], cfg, start, end)
    except ClusteringError as exc:
        return CandidateVerdict(str(candidate), "untestable", None, float("nan"), [],
                                float("nan"), float("nan"),
                                f"{candidate} cannot be tested: {exc}")
    if str(candidate) not in al.returns.columns:
        why = al.dropped.loc[al.dropped["Stream"] == str(candidate), "Reason"]
        reason = why.iloc[0] if len(why) else "not enough history"
        return CandidateVerdict(str(candidate), "untestable", None, float("nan"), [],
                                float("nan"), float("nan"),
                                f"{candidate} cannot be tested: {reason}.",
                                dropped=al.dropped)
    kept = [h for h in holdings if h in al.returns.columns]
    corr = correlation(al.returns, cfg.correlation)
    res = cluster(corr, cfg)
    row = corr.loc[str(candidate), kept]
    near = str(row.idxmax())
    rho = float(row.max())
    lab = res.labels[str(candidate)]
    joins = [h for h in kept if res.labels[h] == lab]
    before = 1.0 if len(kept) == 1 else effective_bets(corr.loc[kept, kept], cfg.effective_bets)
    after = effective_bets(corr, cfg.effective_bets)
    # The scale for that change: the same set plus a stream uncorrelated with all.
    indep = corr.copy()
    indep.loc[str(candidate), kept] = 0.0
    indep.loc[kept, str(candidate)] = 0.0
    ceiling = effective_bets(indep, cfg.effective_bets)
    thr = float(cfg.threshold)
    change = (f"Effective bets go from {before:.1f} to {after:.1f}; a fully "
              f"independent stream would take them to {ceiling:.1f}.")
    if joins:
        verdict = "duplicates"
        group = ", ".join(joins[:3]) + (f" and {len(joins) - 3} more" if len(joins) > 3 else "")
        sentence = (f"{candidate} duplicates an existing bet: {rho:.2f} correlated with "
                    f"{near}, same cluster as {group}. {change}")
    elif rho >= thr:
        verdict = "overlaps"
        sentence = (f"{candidate} partly repeats {near}: {rho:.2f} correlated, above the "
                    f"{thr:.2f} line, though not in the same cluster. {change}")
    else:
        verdict = "new"
        sentence = (f"{candidate} adds a new bet: its highest correlation is {rho:.2f}, "
                    f"with {near}. {change}")
    win = (f"{len(al.returns):,} {al.frequency} returns, "
           f"{al.returns.index[0].date()} to {al.returns.index[-1].date()}")
    return CandidateVerdict(str(candidate), verdict, near, rho, joins, before, after,
                            sentence, ceiling, win, al.dropped)


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
def summary_frame(d: Diversification) -> pd.DataFrame:
    cfg = d.cfg
    cut = (f"correlation {cfg.threshold:.2f}" if cfg.cut == "threshold"
           else f"{cfg.n_clusters} clusters")
    rows = [
        ("Streams analysed", d.n_streams),
        ("Streams left out", len(d.alignment.dropped)),
        ("Window", d.window_text),
        ("Effective number of bets", round(d.effective_bets, 2)),
        ("Effective bets method", cfg.effective_bets),
        ("Clusters", d.clusters.n_clusters),
        ("Cut", cut),
        ("Linkage", cfg.linkage),
        ("Correlation", cfg.correlation),
        ("Average correlation within clusters", d.within),
        ("Average correlation between clusters", d.between),
        ("Silhouette (guide only)", d.clusters.silhouette),
        ("Down markets defined by", d.down_label),
        ("Down-market periods",
         f"{d.down_n} (worst {cfg.stress.percentile:.0%} of periods)"),
    ]
    return pd.DataFrame(rows, columns=["Measure", "Value"])


def export_sections(d: Diversification, module: str = MODULE) -> List[Tuple[Any, ...]]:
    """Workbook sections (module, title, frame, write index) for `excel_export`."""
    order = d.clusters.order
    secs: List[Tuple[Any, ...]] = [
        (module, "Diversification Summary", summary_frame(d), False),
        (module, "Cluster Assignment", d.table, False),
        (module, "Redundant Pairs", d.redundant, False),
        (module, "Correlation, Cluster Order", d.corr.loc[order, order], True),
    ]
    if d.down_corr is not None:
        secs.append((module, "Down-Market Correlation, Cluster Order",
                     d.down_corr.loc[order, order], True))
    if not d.alignment.dropped.empty:
        secs.append((module, "Streams Left Out", d.alignment.dropped, False))
    return secs
