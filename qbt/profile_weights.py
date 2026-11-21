"""The Security review's investor profiles, as weights the Backtest can use.

The Backtest takes weights as text ("XIC.TO:18, VFV.TO:27, ..."), in the
Fixed Weights model and in a "Blend of several" benchmark. This module turns
the profiles of `config/profiles.yaml` (and the saved custom ones in
`config/profiles_custom.yaml`) into that text, so a profile is picked from a
list instead of typed each time. The profiles stay defined in one place.
"""
from __future__ import annotations

from typing import Dict, List

from .security_review.config import ConfigError, Profile, load_profiles


def profiles() -> Dict[str, Profile]:
    """Label -> profile: the presets first, then saved custom profiles."""
    try:
        pc = load_profiles()
    except (ConfigError, OSError):
        return {}
    return {p.describe(): p for p in pc.profiles.values()}


def weights_spec(p: Profile) -> str:
    """'XIC.TO:0.18, VFV.TO:0.27, ...': fractions summing to 1, the format
    both the Fixed Weights model (which does not rescale) and the benchmark
    blend read."""
    def frac(w: float) -> str:
        return f"{w:.4f}".rstrip("0").rstrip(".")
    return ", ".join(f"{t}:{frac(w)}" for t, w in p.weights.items() if w > 0)


def weights_words(p: Profile) -> str:
    """'XIC.TO 18%, VFV.TO 27%, ...' for display."""
    return ", ".join(f"{t} {w * 100:.4g}%" for t, w in p.weights.items() if w > 0)


def merge_tickers(box: str, tickers: List[str]) -> str:
    """The symbols box with any missing tickers appended, one per line."""
    have = [t.strip().upper() for t in box.replace(",", "\n").replace(";", "\n").split("\n")
            if t.strip()]
    return "\n".join(have + [t for t in tickers if t.upper() not in have])
