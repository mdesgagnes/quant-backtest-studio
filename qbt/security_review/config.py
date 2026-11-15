"""YAML configuration for the Security review: profiles, universe, objectives,
scenarios, factors and feature flags.

Everything the decision engine depends on (profiles, building blocks,
candidate universe, objective weights and thresholds, stress windows,
regime rules, factor proxies) is read from `config/*.yaml`. Nothing here
touches the network. Each loader validates what it reads and raises
`ConfigError` with a message meant for the user.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(ROOT, "config")
CUSTOM_PROFILES = "profiles_custom.yaml"
REBALANCE_OPTIONS = ("monthly", "quarterly", "annual", "none")


class ConfigError(ValueError):
    """A configuration file is missing or inconsistent."""


def _read(name: str, folder: Optional[str] = None) -> Dict[str, Any]:
    path = os.path.join(folder or CONFIG_DIR, name)
    if not os.path.exists(path):
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must hold a mapping at the top level.")
    return data


# ----------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------
@dataclass
class Block:
    key: str
    label: str
    ticker: str
    asset_class: str
    proxy: Optional[str] = None


@dataclass
class Profile:
    key: str
    label: str
    weights: Dict[str, float]          # ticker -> weight, sums to 1
    rebalance: str = "quarterly"
    equity: Optional[float] = None
    bonds: Optional[float] = None
    custom: bool = False

    def describe(self) -> str:
        if self.equity is not None:
            return f"{self.label} ({self.equity:.0%} equity / {self.bonds:.0%} bonds)"
        return self.label


@dataclass
class ProfilesConfig:
    base_currency: str
    rebalance_default: str
    blocks: Dict[str, Block]
    equity_mix: Dict[str, float]
    bond_mix: Dict[str, float]
    profiles: Dict[str, Profile]

    def proxy_for(self, ticker: str) -> Optional[str]:
        for b in self.blocks.values():
            if b.ticker == ticker:
                return b.proxy
        return None

    def block_for(self, ticker: str) -> Optional[Block]:
        return next((b for b in self.blocks.values() if b.ticker == ticker), None)


def _check_sum(d: Dict[str, float], what: str) -> None:
    s = sum(float(v) for v in d.values())
    if abs(s - 1.0) > 1e-6:
        raise ConfigError(f"{what} must sum to 1 (got {s:.4f}).")


def profile_weights(equity: float, bonds: float, blocks: Dict[str, Block],
                    equity_mix: Dict[str, float], bond_mix: Dict[str, float]
                    ) -> Dict[str, float]:
    """Ticker weights for an equity / bond split."""
    out: Dict[str, float] = {}
    for mix, share in ((equity_mix, equity), (bond_mix, bonds)):
        for k, w in mix.items():
            if share * w > 0:
                t = blocks[k].ticker
                out[t] = out.get(t, 0.0) + float(share) * float(w)
    return out


def load_profiles(folder: Optional[str] = None) -> ProfilesConfig:
    d = _read("profiles.yaml", folder)
    blocks = {k: Block(k, v.get("label", k), str(v["ticker"]).upper(),
                       v.get("asset_class", ""), (str(v["proxy"]).upper()
                                                   if v.get("proxy") else None))
              for k, v in (d.get("building_blocks") or {}).items()}
    eq, bd = d.get("equity_mix") or {}, d.get("bond_mix") or {}
    for mix, name in ((eq, "equity_mix"), (bd, "bond_mix")):
        unknown = [k for k in mix if k not in blocks]
        if unknown:
            raise ConfigError(f"{name} names unknown building blocks: {unknown}")
        _check_sum(mix, name)
    reb = (d.get("rebalancing") or {}).get("default", "quarterly")
    if reb not in REBALANCE_OPTIONS:
        raise ConfigError(f"rebalancing.default must be one of {REBALANCE_OPTIONS}.")
    profiles: Dict[str, Profile] = {}
    for k, v in (d.get("profiles") or {}).items():
        e, b = float(v["equity"]), float(v["bonds"])
        if abs(e + b - 1) > 1e-6:
            raise ConfigError(f"Profile {k}: equity + bonds must be 1.")
        profiles[k] = Profile(k, v.get("label", k), profile_weights(e, b, blocks, eq, bd),
                              v.get("rebalance", reb), e, b)
    cfg = ProfilesConfig(d.get("base_currency", "CAD"), reb, blocks, eq, bd, profiles)
    for k, p in load_custom_profiles(folder).items():
        cfg.profiles.setdefault(k, p)
    return cfg


def load_custom_profiles(folder: Optional[str] = None) -> Dict[str, Profile]:
    path = os.path.join(folder or CONFIG_DIR, CUSTOM_PROFILES)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    out = {}
    for k, v in (d.get("profiles") or {}).items():
        w = {str(t).upper(): float(x) for t, x in (v.get("weights") or {}).items()}
        if w:
            out[k] = Profile(k, v.get("label", k), w, v.get("rebalance", "quarterly"),
                             custom=True)
    return out


def custom_profile(label: str, weights: Dict[str, float],
                   rebalance: str = "quarterly") -> Profile:
    """A user-defined profile; weights are normalised to sum to 1."""
    w = {str(t).strip().upper(): float(x) for t, x in weights.items()
         if str(t).strip() and float(x) != 0}
    tot = sum(w.values())
    if not w or tot <= 0:
        raise ConfigError("A custom profile needs at least one positive weight.")
    if rebalance not in REBALANCE_OPTIONS:
        raise ConfigError(f"Rebalancing must be one of {REBALANCE_OPTIONS}.")
    key = "custom_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    return Profile(key, label, {t: x / tot for t, x in w.items()}, rebalance, custom=True)


def save_custom_profile(p: Profile, folder: Optional[str] = None) -> str:
    """Adds (or replaces) a custom profile in config/profiles_custom.yaml."""
    path = os.path.join(folder or CONFIG_DIR, CUSTOM_PROFILES)
    d: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
    d.setdefault("profiles", {})[p.key] = {
        "label": p.label, "rebalance": p.rebalance,
        "weights": {t: round(float(w), 6) for t, w in p.weights.items()}}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Custom profiles saved from the Security review page.\n")
        yaml.safe_dump(d, fh, sort_keys=False)
    return path


# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------
@dataclass
class Sleeve:
    key: str
    label: str
    benchmark: Optional[str]
    tickers: List[str]


@dataclass
class UniverseConfig:
    sleeves: Dict[str, Sleeve]
    notes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def sleeve_of(self, ticker: str) -> Optional[Sleeve]:
        return next((s for s in self.sleeves.values() if ticker in s.tickers), None)

    def benchmark_for(self, ticker: str) -> Optional[str]:
        s = self.sleeve_of(ticker)
        return s.benchmark if s else None

    def all_tickers(self) -> List[str]:
        return list(dict.fromkeys(t for s in self.sleeves.values() for t in s.tickers))

    def income_flags(self, ticker: str) -> List[str]:
        return list((self.notes.get(ticker) or {}).get("income_flags") or [])

    def note(self, ticker: str) -> str:
        return str((self.notes.get(ticker) or {}).get("note") or "")


def load_universe(folder: Optional[str] = None) -> UniverseConfig:
    d = _read("universe.yaml", folder)
    sleeves = {k: Sleeve(k, v.get("label", k),
                         str(v["benchmark"]).upper() if v.get("benchmark") else None,
                         [str(t).upper() for t in v.get("tickers") or []])
               for k, v in (d.get("sleeves") or {}).items()}
    notes = {str(k).upper(): v or {} for k, v in (d.get("ticker_notes") or {}).items()}
    return UniverseConfig(sleeves, notes)


# ----------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------
@dataclass
class MetricSpec:
    key: str
    better: str
    center: float
    scale: float
    label: str

    @property
    def direction(self) -> float:
        return 1.0 if self.better == "higher" else -1.0


@dataclass
class Objective:
    key: str
    label: str
    description: str
    weights: Dict[str, float]
    rolling_metric: Optional[str]
    guardrails: List[Dict[str, Any]]
    add: float
    consider: float
    candidate_groups: List[str]


@dataclass
class ObjectivesConfig:
    metrics: Dict[str, MetricSpec]
    objectives: Dict[str, Objective]
    add_weight: float
    min_rolling_share: float
    rolling_window: int
    weight_grid: List[float]
    simulation: Dict[str, Any]
    min_stress_coverage: float = 0.5


def load_objectives(folder: Optional[str] = None) -> ObjectivesConfig:
    d = _read("objectives.yaml", folder)
    dflt = d.get("defaults") or {}
    metrics = {}
    for k, v in (d.get("metrics") or {}).items():
        if v.get("better") not in ("higher", "lower"):
            raise ConfigError(f"Metric {k}: better must be higher or lower.")
        if not float(v.get("scale", 0)) > 0:
            raise ConfigError(f"Metric {k}: scale must be positive.")
        metrics[k] = MetricSpec(k, v["better"], float(v.get("center", 0.0)),
                                float(v["scale"]), v.get("label", k))
    ver = dflt.get("verdict") or {}
    rob = dflt.get("robustness") or {}
    objectives = {}
    for k, v in (d.get("objectives") or {}).items():
        w = {m: float(x) for m, x in (v.get("weights") or {}).items()}
        unknown = [m for m in w if m not in metrics]
        if unknown:
            raise ConfigError(f"Objective {k} uses unknown metrics: {unknown}")
        if not w or sum(w.values()) <= 0:
            raise ConfigError(f"Objective {k} needs positive metric weights.")
        vv = v.get("verdict") or {}
        rm = v.get("rolling_metric")
        objectives[k] = Objective(
            k, v.get("label", k), v.get("description", ""), w, rm,
            list(v.get("guardrails", dflt.get("guardrails") or [])),
            float(vv.get("add", ver.get("add", 0.30))),
            float(vv.get("consider", ver.get("consider", 0.05))),
            list(v.get("candidate_groups") or []))
    return ObjectivesConfig(metrics, objectives, float(dflt.get("add_weight", 0.10)),
                            float(rob.get("min_rolling_share", 0.6)),
                            int(rob.get("rolling_window_months", 36)),
                            [float(x) for x in rob.get("weight_grid", [0.05, 0.10, 0.20])],
                            d.get("simulation") or {},
                            float(rob.get("min_stress_coverage", 0.5)))


# ----------------------------------------------------------------------
# Scenarios, factors, features
# ----------------------------------------------------------------------
@dataclass
class StressWindow:
    key: str
    name: str
    start: str
    end: str


def load_scenarios(folder: Optional[str] = None) -> Dict[str, Any]:
    d = _read("scenarios.yaml", folder)
    wins = [StressWindow(str(w["key"]), str(w["name"]), str(w["start"]), str(w["end"]))
            for w in d.get("stress_windows") or []]
    for w in wins:
        if w.start >= w.end:
            raise ConfigError(f"Stress window {w.key}: start must be before end.")
    return {"windows": wins, "regimes": d.get("regimes") or {}}


def load_factors(folder: Optional[str] = None) -> Dict[str, Any]:
    return _read("factors.yaml", folder)


def load_features(folder: Optional[str] = None) -> Dict[str, bool]:
    """Workspace flags from config/features.yaml. The environment variable
    QBT_ENABLE_WORKSPACES (comma-separated keys, e.g. "markets,research")
    turns hidden workspaces back on without editing the file (used by tests)."""
    try:
        d = _read("features.yaml", folder)
    except ConfigError:
        d = {}
    out = {k: bool((v or {}).get("enabled", True))
           for k, v in (d.get("workspaces") or {}).items()}
    for k in os.environ.get("QBT_ENABLE_WORKSPACES", "").split(","):
        if k.strip():
            out[k.strip().lower()] = True
    return out


@dataclass
class AllConfig:
    profiles: ProfilesConfig
    universe: UniverseConfig
    objectives: ObjectivesConfig
    scenarios: Dict[str, Any]
    factors: Dict[str, Any]


def load_all(folder: Optional[str] = None) -> AllConfig:
    return AllConfig(load_profiles(folder), load_universe(folder), load_objectives(folder),
                     load_scenarios(folder), load_factors(folder))
