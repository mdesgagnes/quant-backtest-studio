"""Sleeve allocation.

Two things people ask for turn out to be one mechanism:

- *"60% equities, 30% bonds, 10% elsewhere, and pick the best names inside
  each"* -- three sleeves, one per asset class, each running the same model
  over its own members.
- *"Half permanently in XYZ, half run by a strategy"* -- two sleeves, one
  holding fixed weights, one running a model.

A sleeve is a budget, a set of members, and a rule for splitting that budget
among them. This module resolves sleeves into a single target-weight frame
and hands it to the engine, which is untouched: drift, execution lag,
frictions, dividends and warm-up all behave exactly as they do for a single
strategy, because the engine never learns that sleeves existed.

Two decisions worth stating, because they are what make the result behave
the way a desk would expect:

**Cash inside a sleeve stays in that sleeve.** If the bond sleeve's model
rejects every bond, that 30% sits in cash. It is not handed to the equity
sleeve. Spilling it over would quietly convert a defensive signal into
extra equity risk at precisely the wrong moment, and would mean the 30%
bond budget was never really a budget.

**Budgets are ceilings, not floors.** A sleeve holds *at most* its budget.
A model that goes half to cash leaves the portfolio under-invested rather
than levering the rest, which is what "no more than 60% in equities" has to
mean if it is to mean anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Sleeve:
    """One budgeted piece of the portfolio."""
    name: str
    budget: float                       # fraction of the portfolio, 0-1
    members: List[str]                  # instruments this sleeve may hold
    mode: str = "strategy"              # strategy | fixed
    strategy_key: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    fixed: Dict[str, float] = field(default_factory=dict)   # ticker -> weight


@dataclass
class SleeveReport:
    rows: pd.DataFrame                  # per-sleeve summary
    warnings: List[str] = field(default_factory=list)
    total_budget: float = 0.0
    exposure: Optional[pd.Series] = None      # realized total exposure
    by_sleeve: Optional[pd.DataFrame] = None  # realized weight per sleeve


# ----------------------------------------------------------------------
def _fixed_frame(sleeve: Sleeve, index: pd.DatetimeIndex,
                 columns: List[str]) -> pd.DataFrame:
    """Static weights, normalized to the sleeve budget."""
    w = pd.DataFrame(0.0, index=index, columns=columns)
    raw = {k: float(v) for k, v in sleeve.fixed.items()
           if k in columns and float(v) != 0.0}
    total = sum(abs(v) for v in raw.values())
    if total <= 0:
        return w
    for k, v in raw.items():
        w[k] = v / total * sleeve.budget
    return w


def _strategy_frame(sleeve: Sleeve, prices: pd.DataFrame,
                    registry: Dict[str, Any],
                    exog: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Runs the sleeve's model over its own members only.

    The model sees a universe of just this sleeve's instruments, so a
    ranking model picks the best bonds among bonds rather than discovering
    that equities out-ranked every bond.
    """
    out = pd.DataFrame(0.0, index=prices.index, columns=list(prices.columns))
    members = [m for m in sleeve.members if m in prices.columns]
    if not members or sleeve.strategy_key not in registry:
        return out

    strat = registry[sleeve.strategy_key]
    sub = prices[members]
    sub_exog = exog if (exog is not None and not exog.empty) else None
    w = strat.generate(sub, sleeve.params, sub_exog)

    # The model spends a budget of 1 inside its own universe; scale that to
    # the sleeve's share. Whatever it left in cash stays in cash.
    gross = w.abs().sum(axis=1)
    over = gross > 1.0 + 1e-9
    if over.any():
        w.loc[over] = w.loc[over].div(gross[over], axis=0)
    out.loc[:, members] = (w * sleeve.budget).to_numpy()
    return out


def resolve(prices: pd.DataFrame, sleeves: List[Sleeve],
            registry: Dict[str, Any],
            exog: Optional[pd.DataFrame] = None,
            max_leverage: float = 1.0) -> Tuple[pd.DataFrame, SleeveReport]:
    """Turns sleeves into one target-weight frame, plus a report.

    The output is an ordinary weight frame; nothing downstream can tell it
    came from sleeves.
    """
    columns = list(prices.columns)
    index = prices.index
    warnings: List[str] = []

    live = [s for s in sleeves if s.budget > 1e-9]
    if not live:
        return pd.DataFrame(0.0, index=index, columns=columns), SleeveReport(
            rows=pd.DataFrame(), warnings=["No sleeve carries a budget."])

    total_budget = float(sum(s.budget for s in live))
    if total_budget > max_leverage + 1e-9:
        scale = max_leverage / total_budget
        for s in live:
            s.budget *= scale
        warnings.append(
            f"Budgets summed to {total_budget*100:.0f}%, above the "
            f"{max_leverage*100:.0f}% allowed. Every sleeve was scaled back "
            f"proportionally.")
        total_budget = float(sum(s.budget for s in live))
    elif total_budget < 0.999:
        warnings.append(
            f"Budgets sum to {total_budget*100:.1f}%. The remaining "
            f"{(1-total_budget)*100:.1f}% stays in cash by construction.")

    frames, rows = {}, []
    for s in live:
        members = [m for m in s.members if m in columns]
        if not members:
            warnings.append(
                f"Sleeve \u201c{s.name}\u201d has no instrument in the loaded "
                f"universe and contributes nothing.")
            continue

        if s.mode == "fixed":
            f = _fixed_frame(s, index, columns)
            # Show the shares as they end up inside the sleeve, not the raw
            # numbers typed in: "60, 40" and "0.6, 0.4" mean the same thing.
            tot = sum(abs(v) for v in s.fixed.values()) or 1.0
            detail = ", ".join(f"{k} {abs(v)/tot*100:.0f}%" for k, v in
                               list(s.fixed.items())[:4]) or "\u2014"
            if len(s.fixed) > 4:
                detail += f", +{len(s.fixed)-4} more"
        else:
            f = _strategy_frame(s, prices, registry, exog)
            detail = registry[s.strategy_key].label if s.strategy_key in registry \
                else "\u2014"
            if s.strategy_key not in registry:
                warnings.append(
                    f"Sleeve \u201c{s.name}\u201d names an unknown model and "
                    f"contributes nothing.")

        frames[s.name] = f
        realized = f.abs().sum(axis=1)
        rows.append({
            "Sleeve": s.name,
            "Budget": s.budget,
            "Source": "Fixed weights" if s.mode == "fixed" else "Model",
            "Detail": detail,
            "Instruments": len(members),
            "Average weight": float(realized.mean()),
            "Cash within sleeve": float((s.budget - realized).clip(lower=0).mean()),
        })

    if not frames:
        return pd.DataFrame(0.0, index=index, columns=columns), SleeveReport(
            rows=pd.DataFrame(rows), warnings=warnings, total_budget=total_budget)

    combined = pd.DataFrame(0.0, index=index, columns=columns)
    for f in frames.values():
        combined = combined.add(f, fill_value=0.0)

    # An instrument can legitimately sit in two sleeves; the sum could then
    # breach the overall ceiling even though each sleeve respected its own.
    gross = combined.abs().sum(axis=1)
    over = gross > max_leverage + 1e-9
    if over.any():
        combined.loc[over] = combined.loc[over].div(gross[over], axis=0) * max_leverage
        warnings.append(
            f"On {int(over.sum())} day(s) the sleeves together exceeded the "
            f"leverage ceiling and were scaled back. This happens when an "
            f"instrument belongs to more than one sleeve.")

    by_sleeve = pd.DataFrame({k: v.abs().sum(axis=1) for k, v in frames.items()})

    return combined, SleeveReport(
        rows=pd.DataFrame(rows), warnings=warnings, total_budget=total_budget,
        exposure=combined.abs().sum(axis=1), by_sleeve=by_sleeve)


# ----------------------------------------------------------------------
# Asset classes
# ----------------------------------------------------------------------
DEFAULT_CLASSES = ["Equities", "Fixed income", "Real assets", "Cash equivalent"]


def sleeves_from_classes(classes: Dict[str, str],
                         budgets: Dict[str, float],
                         strategy_key: str,
                         params: Dict[str, Any]) -> List[Sleeve]:
    """One sleeve per asset class, all running the same model.

    `classes` maps instrument -> class name; `budgets` maps class name ->
    fraction of the portfolio.
    """
    grouped: Dict[str, List[str]] = {}
    for ticker, cls in classes.items():
        if cls and str(cls).strip():
            grouped.setdefault(str(cls).strip(), []).append(ticker)

    out = []
    for cls, members in grouped.items():
        b = float(budgets.get(cls, 0.0))
        if b <= 1e-9:
            continue
        out.append(Sleeve(name=cls, budget=b, members=sorted(members),
                          mode="strategy", strategy_key=strategy_key,
                          params=dict(params)))
    return out


def parse_fixed(spec: str, universe: List[str]) -> Dict[str, float]:
    """Reads 'XIC.TO:60, ZAG.TO:40' or 'XIC.TO 0.6; ZAG.TO 0.4'.

    Percentages and fractions both work; the result is normalized by the
    caller against the sleeve budget, so only the ratios matter here.
    """
    out: Dict[str, float] = {}
    if not spec:
        return out
    up = {u.upper(): u for u in universe}
    for part in str(spec).replace(";", ",").replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        token = part.replace(":", " ").replace("=", " ").split()
        if len(token) < 2:
            continue
        key = token[0].strip().upper()
        try:
            val = float(token[1].replace("%", ""))
        except ValueError:
            continue
        if key in up and val != 0:
            out[up[key]] = out.get(up[key], 0.0) + val
    return out
