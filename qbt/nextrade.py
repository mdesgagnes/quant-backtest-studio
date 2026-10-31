"""Next rebalance: the trades the latest signal calls for.

A backtest ends on the last price in the data, but a portfolio does not: the
signal produced by that last close has not been traded yet. With a one-session
execution lag, the month-end signal of Friday 30 October is acted on at
Monday's open, which is after the data stops. The simulation never shows that
trade, and the "target weight" it reports on the last date is the PREVIOUS
signal, shifted by the lag.

This module answers "what would I have to do on the next trading day?" by
replaying the engine itself over a few extra sessions:

- the strategy's own weights on the last date are carried forward as the new
  signal (nothing is recomputed, so the signal is exactly the one the model
  produced from the latest close);
- the extra sessions are priced flat at the last close, which is the only
  price known today, and a fill at the open uses that close as its estimate;
- the engine then trades as it always does: execution lag, minimum trade
  size, whole units, commission, slippage, impact, currencies.

Using the engine rather than a second sizing formula means the trade list
follows the same rules as every trade in the backtest. As a safeguard the
replay's history is compared with the original run, and the answer says so
if they disagree.

Pure functions: data in, a result object out. Nothing here fetches or renders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import CostConfig, EngineConfig
from .engine import BacktestResult, run_backtest, spec_from_engine
from .schedule import build_calendar

# Sessions of future calendar searched for the next scheduled signal date:
# enough for an annual schedule from any starting point.
HORIZON = 430


def _day(d: Optional[pd.Timestamp]) -> str:
    return "n/a" if d is None else f"{d:%A} {d:%b} {d.day}"


@dataclass
class NextRebalance:
    as_of: pd.Timestamp                     # last price in the data
    exec_date: pd.Timestamp                 # first session the latest signal can trade
    at_open: bool
    lag: int
    scheduled_signal: Optional[pd.Timestamp]   # next signal date on the rebalance calendar
    scheduled_exec: Optional[pd.Timestamp]     # ... and the session it trades
    is_scheduled: bool                      # the latest bar IS a scheduled signal date
    portfolio_value: float                  # model portfolio at the last close
    table: pd.DataFrame                     # one row per instrument, plus cash
    cost: float                             # estimated frictions, in currency
    n_buys: int
    n_sells: int
    frequency: str
    notes: List[str] = field(default_factory=list)
    replay_diff: float = 0.0                # worst daily-return gap vs the original run

    @property
    def has_trades(self) -> bool:
        return (self.n_buys + self.n_sells) > 0

    def headline(self) -> str:
        """One sentence for the top of a page."""
        trade = (f"{self.n_buys} buy{'s' if self.n_buys != 1 else ''} and "
                 f"{self.n_sells} sell{'s' if self.n_sells != 1 else ''}"
                 if self.has_trades else "no trades")
        when = f"{_day(self.exec_date)} {'at the open' if self.at_open else 'at the close'}"
        if self.is_scheduled:
            return (f"The {self.as_of:%b} {self.as_of.day} close is a rebalance "
                    f"signal: {trade} to do on {when}.")
        return (f"Next scheduled rebalance: signal {_day(self.scheduled_signal)}, "
                f"trades {_day(self.scheduled_exec)}. If the latest signal "
                f"(close of {self.as_of:%b} {self.as_of.day}) were traded now "
                f"it would mean {trade} on {when}.")


def _extend(frame: Optional[pd.DataFrame], dates: pd.DatetimeIndex,
            row: Optional[pd.Series]) -> Optional[pd.DataFrame]:
    """`frame` with `dates` appended, each carrying `row`. Only the tail is
    filled: gaps inside the history stay gaps."""
    if frame is None:
        return None
    r = frame.iloc[-1] if row is None else row.reindex(frame.columns)
    tail = pd.DataFrame(np.tile(r.to_numpy(dtype=float), (len(dates), 1)),
                        index=dates, columns=frame.columns)
    return pd.concat([frame, tail])


def _scheduled(idx: pd.DatetimeIndex, engine: EngineConfig,
               rebalance_dates: Optional[pd.DatetimeIndex], lag: int):
    """(signal date, execution date, is the last bar the signal date).

    The calendar is built over the real index plus the weekdays that follow,
    because the engine's own calendar treats the last, unfinished period as
    ending on its last available session. Market holidays after the last
    price are unknown, so a closed Monday would push the real date back a day.
    """
    last = idx[-1]
    if rebalance_dates is not None:
        given = pd.DatetimeIndex(rebalance_dates)
        return None, None, bool(last in given)
    future = pd.bdate_range(last + pd.Timedelta(days=1), periods=HORIZON)
    long_idx = idx.append(future)
    cal = build_calendar(long_idx, spec_from_engine(engine))
    ahead = cal[cal >= last]
    if len(ahead) == 0:
        return None, None, False
    sig = ahead[0]
    pos = long_idx.get_loc(sig) + max(0, lag)
    ex = long_idx[pos] if pos < len(long_idx) else None
    return sig, ex, bool(sig == last)


def next_rebalance(prices: pd.DataFrame,
                   weights: pd.DataFrame,
                   engine: EngineConfig,
                   costs: CostConfig,
                   cash_prices: Optional[pd.Series] = None,
                   rebalance_dates: Optional[pd.DatetimeIndex] = None,
                   open_prices: Optional[pd.DataFrame] = None,
                   dividends: Optional[pd.DataFrame] = None,
                   volume: Optional[pd.DataFrame] = None,
                   fx=None,
                   reference: Optional[BacktestResult] = None) -> NextRebalance:
    """The trades the latest signal would make at the first session it can.

    `prices`, `weights` and the optional frames are the ones the backtest was
    run with (untrimmed). `reference` is its result, used only to check that
    the replay reproduces the same history.
    """
    prices = prices.sort_index()
    idx, assets = prices.index, list(prices.columns)
    n, last = len(idx), idx[-1]
    lag = max(0, int(engine.execution_lag))

    # Position of the signal and of the trade. With a lag, the signal is the
    # last bar's; with none, the trade and the signal share a session, which
    # can only be the next one.
    sig_pos = n - 1 if lag >= 1 else n
    exec_pos = sig_pos + lag
    n_add = exec_pos - (n - 1) + 1          # one spare session after the trade
    new = pd.bdate_range(last + pd.Timedelta(days=1), periods=n_add)
    ext_idx = idx.append(new)

    w = weights.reindex(index=idx, columns=assets).fillna(0.0)
    sig_w = w.iloc[-1]
    gross = float(sig_w.abs().sum())
    target = sig_w * (float(engine.max_leverage) / gross
                      if gross > float(engine.max_leverage) else 1.0)

    last_close = prices.iloc[-1]
    p_ext = _extend(prices, new, None)
    w_ext = _extend(w, new, sig_w)
    o_ext = (_extend(open_prices.reindex(index=idx, columns=assets), new, last_close)
             if open_prices is not None else None)
    d_ext = (_extend(dividends.reindex(index=idx, columns=assets).fillna(0.0), new,
                     pd.Series(0.0, index=assets))
             if dividends is not None else None)
    v_ext = (_extend(volume.reindex(index=idx, columns=assets), new, None)
             if volume is not None else None)

    if rebalance_dates is not None:
        signals = pd.DatetimeIndex(rebalance_dates)
    else:
        signals = build_calendar(idx, spec_from_engine(engine))
    signals = signals.union(pd.DatetimeIndex([ext_idx[sig_pos]]))

    ext = run_backtest(p_ext, w_ext, engine, costs, cash_prices, "Next rebalance",
                       signals, open_prices=o_ext, dividends=d_ext, volume=v_ext,
                       fx=fx)

    # --- Does the replay reproduce the original history? ----------------
    diff = 0.0
    if reference is not None:
        common = reference.returns.index[:-1]       # the final bar may carry a fee
        common = common[common.isin(ext.returns.index)][1:]
        if len(common):
            diff = float((ext.returns.reindex(common)
                          - reference.returns.reindex(common)).abs().max())

    # --- What traded --------------------------------------------------
    value = float(ext.equity.iloc[n - 1])
    t = ext.trades
    if len(t):
        t = t[(pd.to_datetime(t["Date"]) > last) & (t["Reason"] == "Rebalance")]
    chg = t.groupby("Instrument")["Change"].sum() if len(t) else pd.Series(dtype=float)
    fill = t.groupby("Instrument")["Price"].last() if len(t) else pd.Series(dtype=float)
    cost_ccy = float((t["Notional"] * t["Effective Cost (bps)"] / 1e4).sum()) if len(t) else 0.0

    units_now = ext.shares.iloc[n - 1] if ext.shares is not None else pd.Series(0.0, index=assets)
    w_now = ext.weights.iloc[n - 1]
    w_after = ext.weights.iloc[exec_pos]
    rows = []
    for a in assets:
        ch = float(chg.get(a, 0.0))
        if not (abs(w_now[a]) > 1e-9 or abs(target[a]) > 1e-9 or abs(w_after[a]) > 1e-9
                or ch != 0.0):
            continue
        px = float(fill.get(a, last_close[a]))
        if ch > 0:
            action = "Buy (new)" if abs(units_now[a]) < 1e-9 else "Buy"
        elif ch < 0:
            action = "Sell (exit)" if abs(units_now[a] + ch) < 1e-9 else "Sell"
        else:
            action = "Hold"
        rows.append({
            "Instrument": a, "Action": action,
            "Weight now": float(w_now[a]), "Target weight": float(target[a]),
            "Weight after": float(w_after[a]),
            "Change": float(w_after[a] - w_now[a]),
            "Units now": float(units_now[a]), "Units to trade": ch,
            "Price": px, "Amount": ch * px,
        })
    cash_now, cash_after = float(ext.cash_weight.iloc[n - 1]), float(ext.cash_weight.iloc[exec_pos])
    rows.append({
        "Instrument": "Cash", "Action": "",
        "Weight now": cash_now, "Target weight": float(max(0.0, 1.0 - target.sum())),
        "Weight after": cash_after, "Change": cash_after - cash_now,
        "Units now": np.nan, "Units to trade": np.nan, "Price": np.nan,
        "Amount": cash_after * value - cash_now * value,
    })
    table = pd.DataFrame(rows)

    sched_sig, sched_exec, is_sched = _scheduled(idx, engine, rebalance_dates, lag)
    nb = int((table["Action"].str.startswith("Buy")).sum())
    ns = int((table["Action"].str.startswith("Sell")).sum())

    notes: List[str] = []
    if lag == 0:
        notes.append("The execution lag is 0, so this model trades at the same "
                     "close that produces its signal. The list is for the next "
                     "session's close and assumes the signal does not change.")
    if rebalance_dates is not None and not is_sched:
        notes.append("Imported weights carry their own dates: the next one is "
                     "not known from the file, so only the latest row is traded.")
    if diff > 1e-7:
        notes.append(f"The replay differs from the original run by up to "
                     f"{diff:.2e} in daily return, so treat the sizes as "
                     f"approximate.")
    return NextRebalance(
        as_of=last, exec_date=ext_idx[exec_pos],
        at_open=open_prices is not None, lag=lag,
        scheduled_signal=sched_sig, scheduled_exec=sched_exec, is_scheduled=is_sched,
        portfolio_value=value, table=table, cost=cost_ccy,
        n_buys=nb, n_sells=ns, frequency=engine.rebalance, notes=notes,
        replay_diff=diff)
