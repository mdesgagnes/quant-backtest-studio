"""Strategy registry.

`from qbt.strategies import REGISTRY` is enough: importing `library`
registers every built-in strategy.
"""
from .base import (  # noqa: F401
    REGISTRY, Param, Strategy, register, sanitize_weights,
    sma, ema, total_return, realized_vol, downside_vol, rsi,
    efficiency_ratio, zscore, size_equal, size_inverse_vol, apply_vol_target,
)
from . import library  # noqa: F401  (side effect: registration)


def get(key: str) -> Strategy:
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown strategy: {key}. Available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[key]


def list_strategies():
    return [(k, s.label) for k, s in sorted(REGISTRY.items(), key=lambda x: x[1].label)]
