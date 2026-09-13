"""Stress test periods.

Evaluates a strategy's *own realized returns* against specific, named
historical windows -- the 2008 crisis, the COVID crash, and so on -- rather
than a statistical resample of them. A Monte Carlo simulation asks how the
strategy performs across many synthetic paths; this asks the narrower and
more concrete question a manager or an allocator actually asks: what did it
do in October 1987, in the autumn of 2008, in March 2020?

This is post-hoc analysis on a return stream the engine already produced.
It reads `BacktestResult.returns` (or any return series, including an
imported track record) and reports what happened inside each window. It
does not simulate anything, does not touch `qbt/engine.py`, and cannot
change a single number the engine reports -- exactly the same relationship
`qbt/robustness.py` and `qbt/monitor.py` already have to the engine.

**On the period list.** These dates are drawn from cross-referenced public
sources -- S&P/Yardeni-style bear-market chronologies, NBER-adjacent
event studies, and the trend-following/"crisis alpha" literature (Hamill,
Rattray & Van Hemert; Kaminski) that treats a specific, named set of
episodes as the standard test bed for tail-risk behavior. Exact peak and
trough dates vary by a session or two between sources depending on
methodology (closing vs. intraday, which index, US vs. global trading
hours); where sources disagreed by more than a few days the window here is
widened slightly rather than picking one side arbitrarily. These are
starting points, not settled fact, and the list is a plain Python
structure specifically so a period can be added, removed, or redated
without touching anything else in the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class StressPeriod:
    name: str
    start: str                     # "YYYY-MM-DD"
    end: str                       # "YYYY-MM-DD"
    category: str                  # Crash | Bear market | Liquidity event | Rate shock
    note: str = ""


# ----------------------------------------------------------------------
# Sixteen episodes, 1987 to 2024. Categories follow the shape of the
# stress, not just its size: a single-session crash tests a strategy's
# execution and gap risk in a way a slow-motion bear market never does,
# and the two should not be judged by the same yardstick.
# ----------------------------------------------------------------------
DEFAULT_PERIODS: List[StressPeriod] = [
    StressPeriod(
        "Black Monday (1987)", "1987-08-25", "1987-12-04", "Crash",
        "S&P 500 fell roughly 34% peak to trough; October 19 alone was a "
        "20%+ single-session decline, the sharpest one-day drop on record. "
        "Tests reaction to a shock with essentially no preceding trend to "
        "trade."),
    StressPeriod(
        "Gulf War / S&L crisis (1990)", "1990-07-16", "1990-10-11", "Bear market",
        "A ~20% decline tied to the Iraqi invasion of Kuwait, an oil-price "
        "spike, and the domestic savings-and-loan crisis; one of the "
        "fastest-resolving bears on record."),
    StressPeriod(
        "Asian financial crisis (1997)", "1997-10-01", "1998-01-30", "Liquidity event",
        "Contagion from the Thai baht's July float; the US equity impact "
        "concentrated around the October 27, 1997 single-day decline of "
        "roughly 7%, with a fast recovery."),
    StressPeriod(
        "Russian default / LTCM (1998)", "1998-08-01", "1998-10-08", "Liquidity event",
        "Russia's August default and the near-collapse of Long-Term Capital "
        "Management drove a sharp, liquidity-driven equity and credit "
        "selloff, reversed within the same quarter."),
    StressPeriod(
        "Dot-com crash (2000-2002)", "2000-03-24", "2002-10-09", "Bear market",
        "The slowest-moving episode on this list: roughly 30 months from "
        "peak to trough and a ~49% S&P 500 decline. Tests behavior over a "
        "long, grinding decline rather than a shock."),
    StressPeriod(
        "September 11, 2001", "2001-09-10", "2001-09-21", "Crash",
        "Markets closed September 11-14; on reopening the S&P 500 fell "
        "through September 21 for a cumulative decline of about 12% in "
        "four trading sessions, nested inside the broader dot-com bear."),
    StressPeriod(
        "Global Financial Crisis (2007-2009)", "2007-10-09", "2009-03-09", "Bear market",
        "The deepest post-war US bear market: roughly 57% peak to trough "
        "over 17 months, triggered by the subprime mortgage collapse and "
        "the failure of Lehman Brothers."),
    StressPeriod(
        "Flash Crash (May 2010)", "2010-05-03", "2010-05-07", "Liquidity event",
        "On May 6 the Dow fell nearly 9% intraday and mostly recovered "
        "within the same session -- an acute liquidity and market-structure "
        "event rather than a fundamentals-driven decline."),
    StressPeriod(
        "US downgrade / European debt crisis (2011)", "2011-07-22", "2011-10-03", "Bear market",
        "S&P's first-ever downgrade of US sovereign debt (August 5) "
        "compounded fears over Greek, Italian and Spanish debt; the S&P 500 "
        "fell close to 19% peak to trough, just under the conventional bear "
        "threshold on some closing-price measures."),
    StressPeriod(
        "China slowdown / oil crash (2015-2016)", "2015-08-18", "2016-02-11", "Bear market",
        "A Chinese growth scare and the August 24, 2015 intraday break, "
        "compounded by crude oil falling below $30, produced a shallow but "
        "sustained ~15% decline into early 2016."),
    StressPeriod(
        "Volmageddon (February 2018)", "2018-02-02", "2018-02-09", "Liquidity event",
        "A rapid, technically-driven unwind of short-volatility positioning; "
        "the VIX nearly tripled intraday on February 5, a stress episode "
        "concentrated in derivatives and volatility-linked products more "
        "than the cash equity market."),
    StressPeriod(
        "Q4 2018 selloff", "2018-09-20", "2018-12-24", "Bear market",
        "A Fed-tightening and trade-war-driven decline of almost 20%, "
        "concentrated in the final quarter and bottoming on Christmas Eve."),
    StressPeriod(
        "COVID-19 crash (2020)", "2020-02-19", "2020-03-23", "Crash",
        "The fastest 30%+ drawdown in S&P 500 history: about 34% in 33 "
        "calendar days, followed by the fastest recovery of any bear "
        "market on record."),
    StressPeriod(
        "2022 inflation / rate-hike bear market", "2022-01-03", "2022-10-12", "Bear market",
        "The fastest Fed hiking cycle since Volcker drove a ~25% decline "
        "over nine months; unusually, government bonds fell alongside "
        "equities rather than cushioning the drawdown."),
    StressPeriod(
        "SVB / regional banking crisis (2023)", "2023-03-08", "2023-03-24", "Liquidity event",
        "Silicon Valley Bank's failure (March 10) and Signature Bank's "
        "(March 12), followed by Credit Suisse's forced sale (March 19), "
        "produced an acute, bank-concentrated stress episode."),
    StressPeriod(
        "Yen carry-trade unwind (August 2024)", "2024-07-31", "2024-08-09", "Liquidity event",
        "A Bank of Japan rate rise triggered a rapid unwind of yen-funded "
        "carry positioning; the VIX spiked from the mid-teens to 65 "
        "intraday on August 5 and normalized within about a week."),
]

CATEGORIES = sorted({p.category for p in DEFAULT_PERIODS})


# ----------------------------------------------------------------------
def evaluate_periods(returns: pd.Series, periods: Optional[List[StressPeriod]] = None,
                     benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per period: what the return stream actually did inside it.

    A period outside the data's date range, or only partly inside it, is
    reported rather than silently dropped or silently clipped -- coverage
    is itself part of the answer, since a strategy that "survived" 2008
    only because its backtest starts in 2009 has not been tested by 2008
    at all.
    """
    periods = periods if periods is not None else DEFAULT_PERIODS
    returns = returns.dropna()
    if returns.empty:
        return pd.DataFrame()

    data_start, data_end = returns.index.min(), returns.index.max()
    rows = []
    for p in periods:
        p_start, p_end = pd.Timestamp(p.start), pd.Timestamp(p.end)
        window = returns.loc[(returns.index >= p_start) & (returns.index <= p_end)]

        if window.empty:
            rows.append({
                "Period": p.name, "Category": p.category,
                "Start": p_start, "End": p_end, "Coverage": "No data",
                "Sessions": 0, "Return": np.nan, "Max Drawdown": np.nan,
                "Best Day": np.nan, "Worst Day": np.nan,
                "Excess vs Benchmark": np.nan,
            })
            continue

        covers_full = (p_start >= data_start) and (p_end <= data_end)
        coverage = "Full" if covers_full else "Partial"

        # Compounded directly from the returns, not from the ratio of two
        # points on an equity curve. `to_equity`'s first element already has
        # the first period's return folded in -- comparing eq[-1] to eq[0]
        # silently drops that first session from the period return, and for
        # a one-session window (a single monthly observation, say) it
        # returns exactly zero regardless of what actually happened.
        period_return = float((1.0 + window).prod() - 1.0)
        eq = M.to_equity(window, 100.0)
        dd = M.max_drawdown(eq)

        excess = np.nan
        if benchmark is not None:
            bwin = benchmark.dropna().loc[(benchmark.index >= p_start) &
                                          (benchmark.index <= p_end)]
            common = window.index.intersection(bwin.index)
            if len(common) >= 1:
                s_ret = float((1.0 + window.loc[common]).prod() - 1.0)
                b_ret = float((1.0 + bwin.loc[common]).prod() - 1.0)
                excess = s_ret - b_ret

        rows.append({
            "Period": p.name, "Category": p.category,
            "Start": p_start, "End": p_end, "Coverage": coverage,
            "Sessions": int(len(window)), "Return": period_return,
            "Max Drawdown": dd, "Best Day": float(window.max()),
            "Worst Day": float(window.min()), "Excess vs Benchmark": excess,
        })

    return pd.DataFrame(rows)


def summary_stats(evaluated: pd.DataFrame) -> Dict[str, object]:
    """A few headline numbers across whichever periods had data."""
    covered = evaluated[evaluated["Coverage"].isin(["Full", "Partial"])]
    if covered.empty:
        return {"n_covered": 0, "n_total": len(evaluated)}
    ret = covered["Return"].dropna()
    return {
        "n_covered": int(len(covered)),
        "n_total": int(len(evaluated)),
        "n_positive": int((ret > 0).sum()),
        "median_return": float(ret.median()) if len(ret) else np.nan,
        "worst_period": (covered.loc[ret.idxmin(), "Period"]
                        if len(ret) else None),
        "worst_return": float(ret.min()) if len(ret) else np.nan,
        "best_period": (covered.loc[ret.idxmax(), "Period"]
                       if len(ret) else None),
        "best_return": float(ret.max()) if len(ret) else np.nan,
        "worst_drawdown": (float(covered["Max Drawdown"].min())
                          if covered["Max Drawdown"].notna().any() else np.nan),
    }


def from_text(text: str) -> List[StressPeriod]:
    """Parses user-entered custom periods: one per line,
    'Name, YYYY-MM-DD, YYYY-MM-DD[, Category]'. Malformed lines are
    skipped rather than raising, since this feeds a text box a person is
    actively typing into.
    """
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        name, start, end = parts[0], parts[1], parts[2]
        category = parts[3] if len(parts) > 3 else "Custom"
        try:
            pd.Timestamp(start); pd.Timestamp(end)
        except Exception:
            continue
        out.append(StressPeriod(name, start, end, category))
    return out
