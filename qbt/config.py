"""Configuration: dataclasses + YAML loading/serialization.

All engine behavior is driven by these objects. No parameter is hard-coded
anywhere else in the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

REBALANCE_RULES = {
    "D": "Daily",
    "W": "Weekly (Friday)",
    "M": "Monthly (month-end)",
    "Q": "Quarterly (quarter-end)",
    "A": "Annual",
}


@dataclass
class DataConfig:
    """Where prices come from."""
    source: str = "yfinance"              # yfinance | upload
    tickers: List[str] = field(default_factory=lambda: ["XIC.TO", "ZEB.TO", "XEI.TO"])
    start: str = "2010-01-01"
    end: Optional[str] = None
    benchmark: Optional[str] = "XIC.TO"
    cash_proxy: Optional[str] = None       # e.g. PSA.TO; otherwise a fixed rate
    price_field: str = "Close"             # adjusted close by default
    fill_limit: int = 5                    # forward-fill days tolerated
    min_history: int = 60                  # days required before the first signal
    adjusted: bool = True                  # True = total-return prices
    use_dividends: bool = False            # credit dividends as cash (needs adjusted=False)


@dataclass
class CostConfig:
    """Frictions. Defaults calibrated for Canadian ETFs (a conservative assumption)."""
    commission_bps: float = 5.0
    slippage_bps: float = 25.0
    cash_rate_pa: float = 0.0              # used when no cash_proxy is provided
    borrow_rate_pa: float = 0.0            # cost of leverage beyond 100%


@dataclass
class EngineConfig:
    initial_capital: float = 100_000.0
    rebalance: str = "M"
    execution_lag: int = 1                 # signal at t -> execution at t+lag
    max_leverage: float = 1.0
    min_trade_weight: float = 0.005        # ignore micro-adjustments
    periods_per_year: int = 252
    execute_at_open: bool = False          # trade at the open, mark at the close
    trim_warmup: bool = True               # drop the leading uninvested stretch


@dataclass
class StrategyConfig:
    mode: str = "builtin"                  # builtin | external_weights
    name: str = "dual_momentum"
    params: Dict[str, Any] = field(default_factory=dict)
    weights_normalize: str = "None"        # None | "Scale to 100%"
    weights_calendar: str = "File dates"   # or "Engine calendar"
    weights_source: str = ""               # name of the imported file, for traceability


@dataclass
class ExogConfig:
    """Imported series that feed signals (macro, fundamental, etc.)."""
    enabled: bool = False
    publication_lag_days: int = 1          # delay between reference date and release
    ffill_limit: Optional[int] = None      # None = unlimited carry-forward (quarterly data)
    columns: List[str] = field(default_factory=list)
    note: str = ""                         # file provenance, for traceability


@dataclass
class RunConfig:
    label: str = "Backtest"
    data: DataConfig = field(default_factory=DataConfig)
    exog: ExogConfig = field(default_factory=ExogConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    # ------------------------------------------------------------------
    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        d = dict(d or {})
        return cls(
            label=d.get("label", "Backtest"),
            data=DataConfig(**(d.get("data") or {})),
            exog=ExogConfig(**(d.get("exog") or {})),
            strategy=StrategyConfig(**(d.get("strategy") or {})),
            engine=EngineConfig(**(d.get("engine") or {})),
            costs=CostConfig(**(d.get("costs") or {})),
        )

    @classmethod
    def from_yaml(cls, text: str) -> "RunConfig":
        return cls.from_dict(yaml.safe_load(text))

    def validate(self) -> List[str]:
        """Returns the list of blocking issues (empty = OK)."""
        errs: List[str] = []
        if self.engine.rebalance not in REBALANCE_RULES:
            errs.append(f"Unknown rebalance frequency: {self.engine.rebalance}")
        if self.engine.execution_lag < 0:
            errs.append("execution_lag must be >= 0 (0 = execution at the signal day's price).")
        if self.engine.execution_lag == 0:
            errs.append("WARNING: execution_lag=0 introduces look-ahead bias.")
        if self.engine.initial_capital <= 0:
            errs.append("Initial capital must be positive.")
        if self.engine.max_leverage < 0.1:
            errs.append("max_leverage is too low.")
        if self.data.source == "yfinance" and not self.data.tickers:
            errs.append("No ticker provided.")
        if self.strategy.mode not in ("builtin", "external_weights"):
            errs.append(f"Unknown strategy mode: {self.strategy.mode}")
        if self.exog.enabled and self.exog.publication_lag_days < 0:
            errs.append("Publication lag cannot be negative.")
        if self.data.use_dividends and self.data.adjusted:
            errs.append(
                "Dividends cannot be credited on top of adjusted prices: they "
                "are already inside the price series. Switch to price-return "
                "prices, or turn dividends off.")
        return errs
