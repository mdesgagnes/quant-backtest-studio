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
    # What fit_to_holdings needs to size trades for an account that is not the
    # model portfolio: the signal's targets, the last prices, per-instrument
    # trading cost, and the engine's trade-size rules.
    target: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cost_rate: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    min_trade_weight: float = 0.0
    whole_shares: bool = False

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
    rate = pd.Series((costs.commission_bps + costs.slippage_bps) / 1e4, index=assets)
    if fx is not None:                       # converting money has its own spread
        for a in assets:
            if fx.ccy.get(a, fx.base) != fx.base:
                rate[a] += costs.fx_spread_bps / 1e4
    return NextRebalance(
        as_of=last, exec_date=ext_idx[exec_pos],
        at_open=open_prices is not None, lag=lag,
        scheduled_signal=sched_sig, scheduled_exec=sched_exec, is_scheduled=is_sched,
        portfolio_value=value, table=table, cost=cost_ccy,
        n_buys=nb, n_sells=ns, frequency=engine.rebalance, notes=notes,
        replay_diff=diff, target=target, prices=last_close, cost_rate=rate,
        min_trade_weight=float(engine.min_trade_weight),
        whole_shares=bool(engine.whole_shares))


# ----------------------------------------------------------------------
# The user's own account
# ----------------------------------------------------------------------
@dataclass
class HoldingsFit:
    table: pd.DataFrame                     # one row per instrument, plus cash
    value_before: float                     # holdings at last close + cash
    value_after: float                      # the same, less trading costs
    cash_before: float
    cash_after: float
    cost: float
    n_buys: int
    n_sells: int
    distance_before: float                  # share of the portfolio away from target
    distance_after: float
    notes: List[str] = field(default_factory=list)

    @property
    def has_trades(self) -> bool:
        return (self.n_buys + self.n_sells) > 0


def _distance(values: np.ndarray, cash: float, T: np.ndarray, total: float) -> float:
    """Half the sum of absolute weight gaps, cash included: the share of the
    portfolio that would have to move to reach the target exactly."""
    if total <= 0:
        return 0.0
    gaps = np.append(values - T, cash - (total - T.sum()))
    return float(np.abs(gaps).sum() / (2 * total))


def _solve_units(T, p, q0, cash0, floor, r, frozen, whole, mtw_d, max_iter=20000):
    """Final holdings (units) that minimise the sum of |value - target| over
    instruments, subject to cash after costs staying at or above `floor`,
    no flips from long to short, and the minimum trade size.

    Fractional units: the exact target. Whole units: start from the rounded
    target, repair any cash shortfall, then improve with one-unit moves and
    swaps (buy one of i, sell one of j) until none helps.
    """
    ideal = T / p
    h = np.where(frozen, q0, ideal)
    if not whole:
        return h
    h = np.where(frozen, q0, np.rint(ideal))
    long_ok = T >= 0

    def cash_of(x):
        n = x - q0
        return cash0 - float(n @ p) - float((r * np.abs(n) * p).sum())

    def allowed(x):
        n = np.abs(x - q0)
        sign_ok = np.where(long_ok, x >= -1e-12, x <= 1e-12)
        size_ok = (n < 1e-9) | (n * p >= mtw_d) | (np.abs(x) < 1e-9)
        return sign_ok & size_ok & ~frozen

    # A rounded trade that is smaller than the minimum is simply not made.
    bad = ~((np.abs(h - q0) < 1e-9) | (np.abs(h - q0) * p >= mtw_d) | (np.abs(h) < 1e-9))
    h = np.where(bad & ~frozen, q0, h)

    def f_of(x):
        return np.abs(x * p - T)

    # Repair: too little cash, so give up the unit that costs the least accuracy.
    for _ in range(max_iter):
        if cash_of(h) >= floor - 1e-9:
            break
        dn = h - 1.0
        ok = allowed(dn)
        if not ok.any():
            break
        cost = np.where(ok, f_of(dn) - f_of(h), np.inf)
        h[int(np.argmin(cost))] -= 1.0

    tol = 1e-9 * max(1.0, float(np.abs(T).sum()))
    for _ in range(max_iter):
        cash = cash_of(h)
        n = h - q0
        up, dn = h + 1.0, h - 1.0
        f0 = f_of(h)
        d_up, d_dn = f_of(up) - f0, f_of(dn) - f0
        ok_up, ok_dn = allowed(up), allowed(dn)
        c_up = -p - r * p * (np.abs(n + 1) - np.abs(n))
        c_dn = p - r * p * (np.abs(n - 1) - np.abs(n))
        # singles
        s_up = np.where(ok_up & (cash + c_up >= floor - 1e-9), d_up, np.inf)
        s_dn = np.where(ok_dn & (cash + c_dn >= floor - 1e-9), d_dn, np.inf)
        # swaps: i up, j down
        pair = d_up[:, None] + d_dn[None, :]
        feas = ok_up[:, None] & ok_dn[None, :] & (cash + c_up[:, None] + c_dn[None, :] >= floor - 1e-9)
        np.fill_diagonal(feas, False)
        pair = np.where(feas, pair, np.inf)
        best = min(s_up.min(), s_dn.min(), pair.min())
        if not best < -tol:
            break
        if best == s_up.min():
            h[int(np.argmin(s_up))] += 1.0
        elif best == s_dn.min():
            h[int(np.argmin(s_dn))] -= 1.0
        else:
            i, j = np.unravel_index(int(np.argmin(pair)), pair.shape)
            h[i] += 1.0
            h[j] -= 1.0
    return h


def fit_to_holdings(target: pd.Series, prices: pd.Series, holdings: pd.Series,
                    cash: float, cost_rate: Optional[pd.Series] = None,
                    whole_units: bool = True,
                    min_trade_weight: float = 0.0) -> HoldingsFit:
    """The trades that bring an actual account as close as possible to a
    target allocation.

    `target` is a weight per instrument (cash is the remainder), `prices` the
    price to trade at, `holdings` the units held now, `cash` the cash held now,
    all in one currency. The result minimises the total gap between each
    position's value and its target value after trading costs, in whole units
    when asked: a target of 33.3% of 10,000 in a 410 stock lands on 8 shares,
    and the leftover cash goes to whichever other position is furthest below
    its target. Cash never goes negative unless the target itself is levered.
    Trades under `min_trade_weight` of the account are not made, as in the
    backtest.
    """
    names = [a for a in prices.index if np.isfinite(prices[a]) and prices[a] > 0]
    p = prices.reindex(names).to_numpy(float)
    t = target.reindex(names).fillna(0.0).to_numpy(float)
    q0 = holdings.reindex(names).fillna(0.0).to_numpy(float)
    r = (np.zeros(len(names)) if cost_rate is None
         else cost_rate.reindex(names).fillna(0.0).to_numpy(float))
    cash = float(cash)
    v0 = float(q0 @ p) + cash
    if v0 <= 0:
        raise ValueError("The account is worth nothing: enter units held or cash.")

    cost = 0.0
    h = q0.copy()
    for _ in range(5):                           # costs shrink the pot the target applies to
        v = v0 - cost
        T = t * v
        floor = min(0.0, (1.0 - float(t.sum())) * v)
        d = T / p - q0
        mtw_d = float(min_trade_weight) * v0
        exit_ = (np.abs(T) < 1e-9) & (np.abs(q0) > 0)
        frozen = (np.abs(d * p) < mtw_d) & ~exit_
        h = _solve_units(T, p, q0, cash, floor, r, frozen, whole_units, mtw_d)
        new_cost = float((r * np.abs(h - q0) * p).sum())
        if abs(new_cost - cost) < 1e-6:
            cost = new_cost
            break
        cost = new_cost

    n = h - q0
    cash_after = cash - float(n @ p) - cost
    v_after = v0 - cost
    w_now, w_after = q0 * p / v0, h * p / v_after
    rows = []
    for k, a in enumerate(names):
        if not (abs(q0[k]) > 1e-12 or abs(t[k]) > 1e-12 or abs(n[k]) > 1e-12):
            continue
        if n[k] > 1e-9:
            act = "Buy (new)" if abs(q0[k]) < 1e-9 else "Buy"
        elif n[k] < -1e-9:
            act = "Sell (exit)" if abs(h[k]) < 1e-9 else "Sell"
        else:
            act = "Hold"
        rows.append({
            "Instrument": a, "Action": act, "Units held": q0[k], "Price": p[k],
            "Weight now": w_now[k], "Target weight": t[k],
            "Units to trade": n[k], "Amount": n[k] * p[k],
            "Units after": h[k], "Weight after": w_after[k],
            "Gap to target": w_after[k] - t[k],
        })
    rows.append({
        "Instrument": "Cash", "Action": "", "Units held": np.nan, "Price": np.nan,
        "Weight now": cash / v0, "Target weight": 1.0 - float(t.sum()),
        "Units to trade": np.nan, "Amount": cash_after - cash,
        "Units after": np.nan, "Weight after": cash_after / v_after,
        "Gap to target": cash_after / v_after - (1.0 - float(t.sum())),
    })
    table = pd.DataFrame(rows)
    tv = t * v_after
    notes: List[str] = []
    if cash_after < -1e-6 and float(t.sum()) <= 1.0 + 1e-9:
        notes.append("The cash does not cover every trade, so the list is short "
                     "of the target.")
    return HoldingsFit(
        table=table, value_before=v0, value_after=v_after, cash_before=cash,
        cash_after=cash_after, cost=cost,
        n_buys=int((n > 1e-9).sum()), n_sells=int((n < -1e-9).sum()),
        distance_before=_distance(q0 * p, cash, t * v0, v0),
        distance_after=_distance(h * p, cash_after, tv, v_after), notes=notes)
