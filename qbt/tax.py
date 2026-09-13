"""Tax friendliness, for a Canadian non-registered account.

Answers a different question from anything else in this app: not "how did
the strategy perform," but "how much of that performance would actually
reach the investor after tax, and how much of the tax bill was the
strategy's own trading behaviour responsible for." A low-turnover strategy
defers gains it hasn't sold; a high-turnover one crystallizes them every
year even if the two have identical pre-tax returns. That difference is
invisible anywhere else in this app and is often the deciding factor
between two strategies with a similar Sharpe ratio.

This is pure post-hoc analysis of a completed backtest, exactly like
`qbt/stress.py` and `qbt/monitor.py`. It reads `BacktestResult.trades` and
`BacktestResult.shares`, plus the per-share dividend series already used
to run the backtest, and computes a separate, additional set of numbers.
It does not touch `qbt/engine.py` and cannot change a single figure the
engine reports.

**Scope, stated plainly rather than buried in a footnote.** This models a
**non-registered (taxable) account** for an **individual Canadian
resident**, where the tax year is the calendar year. Inside a TFSA or an
RRSP none of this applies -- there is no annual tax on either capital
gains or dividends in either registered account, which is a large part of
why the account type matters more than the strategy in many cases.

**What is modelled, and at what level of precision:**

- **Capital gains use the Adjusted Cost Base (ACB) method** -- the
  average cost of everything currently held, recomputed after every buy
  and sell -- which is what Canadian tax law requires. This is not FIFO
  and not specific-lot identification, both of which are allowed in some
  other countries but not here.
- **The capital gains inclusion rate defaults to 50%.** The 2024 federal
  budget proposed raising it to 66.67% on individual gains above
  $250,000/year; that increase was deferred and then cancelled outright in
  March 2025. As of this writing the rate remains 50% for everyone, with
  no threshold -- but tax law can change again, so it is a parameter, not
  a constant.
- **Eligible vs. non-eligible/foreign dividend income is inferred from the
  ticker**, on the simple and visible heuristic that a ``.TO``-suffixed
  symbol is Canadian-listed and everything else is not. This is a
  starting point, overridable per ticker, not a determination of what the
  issuing corporation actually designates -- a Canadian-listed ETF holding
  foreign equities may itself pass through foreign, non-eligible income,
  which this cannot see from price and distribution data alone.
- **Eligible dividends get the federal gross-up and dividend tax credit**
  (38% gross-up, 15.0198% federal credit on the grossed-up amount, both
  fixed nationally). **Provincial dividend tax credits are not modelled**
  -- they vary by province and are usually large enough to matter, so the
  computed eligible-dividend rate is a conservative, federal-only
  estimate unless a full effective rate is supplied directly.
- **Foreign/non-eligible dividends are taxed at the plain marginal rate**,
  with no gross-up or credit, which is accurate. US withholding tax on
  US-listed dividends is assumed fully recoverable through the foreign
  tax credit by default (true for most taxpayers in a taxable account,
  since it works against Canadian tax otherwise owed on the same income);
  turning that assumption off adds it back as a flat, uncredited cost.
- **Not modelled at all:** the supersuperficial loss rule, the lifetime
  capital gains exemption, capital gains from anywhere outside this
  backtest, provincial surtaxes and clawback thresholds (OAS, etc.), and
  any account other than non-registered. Loss carryforward is modelled
  only *forward* within the backtest's own date range -- Canada also
  allows carrying a loss back three years, which needs tax history this
  module cannot see.

Tax is assumed **paid from outside the portfolio** -- from other income
or cash -- rather than funded by selling more of the position, which is
both the simpler and the more common real assumption for anyone with
other income. The after-tax equity curve is a bookkeeping construction on
top of the untouched equity curve, not a different simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Federal dividend mechanics. Stable, legislated nationally; provincial
# credits are the part that varies and is not modelled here.
# ----------------------------------------------------------------------
ELIGIBLE_GROSS_UP = 0.38
ELIGIBLE_FEDERAL_CREDIT = 0.150198        # of the grossed-up amount
NON_ELIGIBLE_GROSS_UP = 0.15
NON_ELIGIBLE_FEDERAL_CREDIT = 0.090301
US_WITHHOLDING_RATE = 0.15


@dataclass
class TaxSettings:
    marginal_rate: float = 0.40
    inclusion_rate: float = 0.50
    eligible_dividend_rate: Optional[float] = None   # None -> derive federally
    foreign_dividend_rate: Optional[float] = None    # None -> = marginal_rate
    us_withholding_recoverable: bool = True
    canadian_suffixes: Tuple[str, ...] = (".TO", ".V", ".NE", ".CN")
    ticker_overrides: Dict[str, str] = field(default_factory=dict)  # ticker -> "eligible"|"foreign"

    def eligible_rate(self) -> float:
        if self.eligible_dividend_rate is not None:
            return float(self.eligible_dividend_rate)
        grossed = 1.0 + ELIGIBLE_GROSS_UP
        return max(0.0, self.marginal_rate * grossed - ELIGIBLE_FEDERAL_CREDIT * grossed)

    def foreign_rate(self) -> float:
        base = (float(self.foreign_dividend_rate) if self.foreign_dividend_rate is not None
               else self.marginal_rate)
        if not self.us_withholding_recoverable:
            base += US_WITHHOLDING_RATE * (1.0 - self.marginal_rate)
        return base

    def classify(self, ticker: str) -> str:
        override = self.ticker_overrides.get(str(ticker).strip().upper())
        if override in ("eligible", "foreign"):
            return override
        return ("eligible" if str(ticker).strip().upper().endswith(self.canadian_suffixes)
               else "foreign")


@dataclass
class TaxReport:
    by_year: pd.DataFrame
    realized_trades: pd.DataFrame
    total_tax: float
    total_pretax_gain: float
    tax_drag_pa: float               # annualized, pretax CAGR minus after-tax CAGR
    after_tax_equity: pd.Series
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
def realized_gains(trades: pd.DataFrame) -> pd.DataFrame:
    """Every sale, at the Adjusted Cost Base in effect at the moment it
    happened -- the average cost of the position, recomputed after each
    trade, which is the method Canadian tax law requires rather than
    FIFO or a specific lot.

    Processes trades in chronological order per instrument so that each
    sale sees only the cost base built up from trades that actually came
    before it.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["Date", "Instrument", "Shares Sold",
                                     "Proceeds", "ACB of Shares Sold",
                                     "Realized Gain"])
    t = trades.sort_values("Date")
    rows = []
    acb_total: Dict[str, float] = {}
    shares_held: Dict[str, float] = {}

    for _, r in t.iterrows():
        inst = r["Instrument"]
        change = float(r["Change"])
        price = float(r["Price"])
        acb_total.setdefault(inst, 0.0)
        shares_held.setdefault(inst, 0.0)

        if change > 1e-12:
            acb_total[inst] += change * price
            shares_held[inst] += change
        elif change < -1e-12:
            sold = -change
            held = shares_held[inst]
            if held > 1e-12:
                acb_per_share = acb_total[inst] / held
                sold_eff = min(sold, held)   # never sell more than the model held
                cost = sold_eff * acb_per_share
                proceeds = sold_eff * price
                rows.append({
                    "Date": r["Date"], "Instrument": inst,
                    "Shares Sold": sold_eff, "Proceeds": proceeds,
                    "ACB of Shares Sold": cost,
                    "Realized Gain": proceeds - cost,
                })
                acb_total[inst] -= cost
                shares_held[inst] -= sold_eff
            # A sale with no recorded prior holding (data starting mid
            # position, or a rounding residual) has no cost base to compare
            # against and is skipped rather than assigned an invented one.

    return pd.DataFrame(rows)


def dividend_income_by_instrument(shares: pd.DataFrame,
                                  dividends: pd.DataFrame) -> pd.DataFrame:
    """Dollar dividends received per instrument per day.

    Paid on the units held going *into* the ex-dividend day -- the same
    convention the engine itself credits dividends under -- so this
    reconciles with the engine's own aggregate dividend figure rather than
    quietly using a different one.
    """
    div = dividends.reindex(columns=shares.columns).reindex(shares.index).fillna(0.0)
    held_into_day = shares.shift(1).fillna(0.0)
    return held_into_day * div


def evaluate(trades: pd.DataFrame, shares: pd.DataFrame, dividends: pd.DataFrame,
            equity: pd.Series, settings: Optional[TaxSettings] = None
            ) -> TaxReport:
    """The full annual tax picture for one completed backtest."""
    settings = settings or TaxSettings()
    warnings: List[str] = []

    gains = realized_gains(trades)
    if not gains.empty:
        gains["Year"] = pd.DatetimeIndex(gains["Date"]).year

    div_dollars = dividend_income_by_instrument(shares, dividends)
    has_dividends = float(div_dollars.to_numpy().sum()) > 0
    if not has_dividends:
        warnings.append(
            "No dividend cash flow found. If the run used total-return "
            "(adjusted) prices, dividends are folded invisibly into the "
            "price series and cannot be separated out here -- switch the "
            "Price convention to \u201cPrice return + cash dividends\u201d "
            "for this analysis to see them.")

    classes = {c: settings.classify(c) for c in div_dollars.columns}
    eligible_cols = [c for c, k in classes.items() if k == "eligible"]
    foreign_cols = [c for c, k in classes.items() if k == "foreign"]
    div_by_year = pd.DataFrame({
        "Eligible Dividends": div_dollars[eligible_cols].sum(axis=1) if eligible_cols
                             else pd.Series(0.0, index=div_dollars.index),
        "Foreign Dividends": div_dollars[foreign_cols].sum(axis=1) if foreign_cols
                            else pd.Series(0.0, index=div_dollars.index),
    })
    div_by_year["Year"] = div_by_year.index.year
    div_annual = div_by_year.groupby("Year")[["Eligible Dividends", "Foreign Dividends"]].sum()

    years = sorted(set(equity.index.year) | set(div_annual.index) |
                   (set(gains["Year"]) if not gains.empty else set()))
    if not years:
        empty = pd.DataFrame()
        return TaxReport(empty, gains, 0.0, 0.0, 0.0, equity.copy(), warnings)

    gains_annual = (gains.groupby("Year")["Realized Gain"].sum()
                    if not gains.empty else pd.Series(dtype=float))

    eligible_rate = settings.eligible_rate()
    foreign_rate = settings.foreign_rate()

    rows = []
    carryforward = 0.0
    for y in years:
        net_gain = float(gains_annual.get(y, 0.0)) - carryforward
        if net_gain >= 0:
            taxable_gain = net_gain
            carryforward = 0.0
        else:
            taxable_gain = 0.0
            carryforward = -net_gain

        elig = float(div_annual["Eligible Dividends"].get(y, 0.0)) if y in div_annual.index else 0.0
        fore = float(div_annual["Foreign Dividends"].get(y, 0.0)) if y in div_annual.index else 0.0

        cap_gains_tax = taxable_gain * settings.inclusion_rate * settings.marginal_rate
        elig_tax = elig * eligible_rate
        fore_tax = fore * foreign_rate
        total = cap_gains_tax + elig_tax + fore_tax

        rows.append({
            "Year": y,
            "Realized Gain/Loss": float(gains_annual.get(y, 0.0)),
            "Taxable Capital Gain": taxable_gain,
            "Capital Gains Tax": cap_gains_tax,
            "Eligible Dividends": elig,
            "Eligible Dividend Tax": elig_tax,
            "Foreign Dividends": fore,
            "Foreign Dividend Tax": fore_tax,
            "Total Tax": total,
            "Loss Carryforward (end of year)": carryforward,
        })

    by_year = pd.DataFrame(rows)

    total_tax = float(by_year["Total Tax"].sum())
    total_pretax_gain = float(gains["Realized Gain"].sum()) if not gains.empty else 0.0

    # After-tax equity: grow at the strategy's own realized return each
    # calendar year, then subtract that year's tax bill in one lump sum at
    # year end -- consistent with an annual filing, and with tax assumed
    # paid from money outside the portfolio rather than funded by selling
    # more of it.
    tax_by_year = by_year.set_index("Year")["Total Tax"]
    yearly_last = equity.resample("YE").last()
    yearly_first = equity.resample("YE").first()
    prev_level = float(equity.iloc[0]) if len(equity) else 0.0
    after_tax_points = [(equity.index[0], prev_level)] if len(equity) else []
    for y in sorted(yearly_last.index.year):
        y_end = yearly_last.loc[yearly_last.index.year == y]
        y_start = yearly_first.loc[yearly_first.index.year == y]
        if y_end.empty or y_start.empty or float(y_start.iloc[0]) == 0:
            continue
        growth = float(y_end.iloc[0]) / float(y_start.iloc[0])
        prev_level = prev_level * growth - float(tax_by_year.get(y, 0.0))
        after_tax_points.append((y_end.index[0], prev_level))
    after_tax_equity = (pd.Series({d: v for d, v in after_tax_points}).sort_index()
                        if after_tax_points else equity.copy())

    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    pretax_cagr = (float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1 / yrs) - 1.0
    at_last = float(after_tax_equity.iloc[-1]) if len(after_tax_equity) else float(equity.iloc[-1])
    at_first = float(after_tax_equity.iloc[0]) if len(after_tax_equity) else float(equity.iloc[0])
    after_tax_cagr = (max(at_last, 1e-9) / max(at_first, 1e-9)) ** (1 / yrs) - 1.0
    tax_drag_pa = pretax_cagr - after_tax_cagr

    return TaxReport(by_year=by_year, realized_trades=gains, total_tax=total_tax,
                     total_pretax_gain=total_pretax_gain, tax_drag_pa=tax_drag_pa,
                     after_tax_equity=after_tax_equity, warnings=warnings)
