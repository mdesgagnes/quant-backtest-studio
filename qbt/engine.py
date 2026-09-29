"""Backtest engine.

Simulation model, explicit and verifiable:

1. The strategy produces target weights for each day, from information
   available at that day's close.
2. The engine shifts these weights by `execution_lag` business days. With the
   default of 1, a Monday-evening signal executes on Tuesday. No look-ahead
   bias is possible.
3. Between rebalances, weights **drift** with asset returns: an implicit
   daily rebalance is never assumed, the classic mistake that inflates
   backtested results.
4. On rebalance dates, turnover is charged. Commission is a flat rate.
   Slippage is a flat half-spread (with optional per-instrument
   multipliers) plus, optionally, a market-impact term that scales with trade size relative
   to the instrument's own trailing average daily volume: either a fixed
   cost at 10% of volume, or the square-root law scaled by the
   instrument's own recent volatility. The same order costs more in a
   thinner name, more again the larger it is, and more in a turbulent
   week than a calm one.
   Manual per-instrument multipliers on the flat rate are also supported,
   for instruments with no volume data or known liquidity quirks. The
   monthly management fee, when it must be raised by selling holdings
   rather than from idle cash, pays these same costs by default -- an
   involuntary liquidation is not free just because it was involuntary --
   and sells whole units only when the book trades whole units.
5. The uninvested portion earns the cash rate, or tracks the return of a
   cash proxy asset (e.g. PSA.TO) if supplied.
6. Optionally, trades execute at the **open** and the portfolio is marked to
   market at the **close**. The day then has two legs: the overnight move
   from the prior close to the open is earned on the old weights, and the
   intraday move from open to close on the new ones.
7. Optionally, **cash dividends** are credited on their ex-date and held in
   the cash bucket until the next rebalance, rather than being assumed
   instantly reinvested -- or, with cash reinvestment on, until the next
   scheduled or threshold-triggered reinvestment, which buys toward the
   last rebalance's target weights and keeps a cash buffer. This requires price-return (non dividend-adjusted)
   prices, otherwise the dividend would be counted twice.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig, CostConfig
from .schedule import RebalanceSpec, build_calendar

_RESAMPLE = {"D": None, "W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE"}


# Non-instrument lines of the contribution ledger.
CASH_LINE = "Cash & financing"
COST_LINE = "Trading costs"
FEE_LINE = "Management fee"
CASH_FLOW_COLUMNS = ["Date", "Type", "Instrument", "Amount", "Units", "Per unit",
                     "Sessions accrued", "Average value", "Annual rate",
                     "Paid from cash", "Raised by selling", "Trading costs",
                     "Cash before", "Cash after"]


@dataclass
class BacktestResult:
    equity: pd.Series             # portfolio value
    returns: pd.Series            # net daily returns
    gross_returns: pd.Series      # before frictions
    weights: pd.DataFrame         # effective weights held (post-drift)
    target_weights: pd.DataFrame  # weights targeted at execution
    turnover: pd.Series           # one-way turnover per day
    costs: pd.Series              # daily cost, as a fraction of value
    exposure: pd.Series           # sum of risky weights
    cash_weight: pd.Series
    rebalance_dates: pd.DatetimeIndex
    trades: pd.DataFrame
    label: str = "Backtest"
    dividend_income: Optional[pd.Series] = None   # daily, as a fraction of value
    warmup_start: Optional[pd.Timestamp] = None   # first day actually invested
    shares: Optional[pd.DataFrame] = None         # units held, the real state
    fees: Optional[pd.Series] = None              # management fee, as a fraction
    # Daily contribution to the net return, as a fraction of the previous
    # day's value: one column per instrument (price move plus dividends),
    # then cash and financing, trading costs and the management fee. Each
    # row sums to that day's net return exactly.
    contributions: Optional[pd.DataFrame] = None
    # Every movement of cash that is not a trade, in currency: each
    # management-fee deduction (with what it was accrued on and how it was
    # paid), each dividend, and each cash reinvestment.
    cash_flows: Optional[pd.DataFrame] = None

    @property
    def nav(self) -> pd.Series:
        return self.equity


# ----------------------------------------------------------------------
def rebalance_calendar(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """Last available business day of each period.

    Kept for callers that only care about the frequency. The engine itself
    goes through `spec_from_engine`, which also honours the trading-day
    rule.
    """
    if rule == "D" or _RESAMPLE.get(rule) is None:
        return pd.DatetimeIndex(index)
    s = pd.Series(index, index=index)
    dates = s.resample(_RESAMPLE[rule]).last().dropna()
    return pd.DatetimeIndex(sorted(set(dates.values))).intersection(index)


def spec_from_engine(engine: EngineConfig) -> RebalanceSpec:
    """The trading-day rule carried by an EngineConfig."""
    return RebalanceSpec(
        frequency=engine.rebalance,
        day_rule=getattr(engine, "day_rule", "last"),
        day_of_month=int(getattr(engine, "day_of_month", 15)),
        weekday=int(getattr(engine, "weekday", 4)),
        nth=int(getattr(engine, "nth", 1)),
        anchor_month=int(getattr(engine, "anchor_month", 12)),
    )


def _cash_returns(index: pd.DatetimeIndex, cash_prices: Optional[pd.Series],
                  rate_pa: float, ppy: int) -> pd.Series:
    if cash_prices is not None and len(cash_prices.dropna()) > 1:
        return cash_prices.reindex(index).ffill().pct_change().fillna(0.0)
    # The daily rate that compounds to the stated annual rate. Dividing by
    # the number of periods instead (3% / 252 per day) compounds to 3.045%
    # a year, so the cash leg would quietly earn more than the rate typed.
    daily = (1.0 + float(rate_pa)) ** (1.0 / max(1, int(ppy))) - 1.0 if rate_pa > -1 else 0.0
    return pd.Series(daily, index=index)


# ----------------------------------------------------------------------
def _whole_unit_sale(frac_sold: np.ndarray, shares: np.ndarray,
                     mark: np.ndarray, rate: np.ndarray,
                     need: float) -> np.ndarray:
    """Whole units to sell so the proceeds, net of costs, cover `need`.

    Starts from the pro-rata fractional sale rounded down, then adds one
    unit at a time: the cheapest unit that closes the remaining gap on its
    own, or, while no single unit does, a unit of whichever position is
    still largest, so the book stays close to its proportions. The small
    excess this raises stays in cash. Long positions only; a book too small
    to cover the fee sells what it has.
    """
    unit_net = np.where(mark > 0, mark * (1.0 - np.clip(rate, 0.0, 1.0)), 0.0)
    long_ = (shares >= 1.0) & (unit_net > 0)
    sold = np.where(long_, np.clip(np.floor(frac_sold + 1e-9), 0.0, np.floor(shares)), 0.0)
    got = float((sold * unit_net).sum())
    while got < need - 1e-9:
        room = long_ & (shares - sold >= 1.0)
        if not room.any():
            break
        gap = need - got
        closes = room & (unit_net >= gap)
        if closes.any():
            j = int(np.where(closes, unit_net, np.inf).argmin())
        else:
            j = int(np.where(room, (shares - sold) * mark, -np.inf).argmax())
        sold[j] += 1.0
        got += float(unit_net[j])
    return sold


def run_backtest(prices: pd.DataFrame,
                 target_weights: pd.DataFrame,
                 engine: EngineConfig,
                 costs: CostConfig,
                 cash_prices: Optional[pd.Series] = None,
                 label: str = "Backtest",
                 rebalance_dates: Optional[pd.DatetimeIndex] = None,
                 open_prices: Optional[pd.DataFrame] = None,
                 dividends: Optional[pd.DataFrame] = None,
                 volume: Optional[pd.DataFrame] = None) -> BacktestResult:
    """Simulates the portfolio day by day. Returns all diagnostics.

    `rebalance_dates` overrides the periodic calendar when dates are imposed
    externally (e.g. an imported target-weights file). These dates are
    interpreted as *signal* dates and shifted by `execution_lag`, exactly
    like a built-in strategy's signals.

    `open_prices`, when supplied, switches execution to the open: the
    overnight leg is earned on the pre-trade weights and the intraday leg on
    the post-trade weights. Without it, the whole day is earned on the
    pre-trade weights and the trade happens at the close.

    `dividends` holds per-share cash amounts on their ex-date, in the same
    units as `prices`. They are credited to the cash bucket, where they sit
    until the next rebalance reinvests them. Supply these only alongside
    price-return prices: with dividend-adjusted prices the payment is
    already inside the price series and would be counted twice.

    `volume` (shares per session) drives the optional market-impact cost
    model. A trailing 20-session average, shifted so the trade day's own
    volume is never used, stands in for the average daily volume a real
    order would be sized against -- using the same day's volume would let
    the simulation "know" a print it could not have known when the order
    was placed. Instruments with no volume data supplied fall back to the
    flat slippage rate, since impact cannot be estimated without it.
    """
    prices = prices.sort_index()
    idx = prices.index
    assets = list(prices.columns)

    w_target = target_weights.reindex(index=idx, columns=assets).fillna(0.0)

    # --- Leverage control ---------------------------------------------
    gross = w_target.abs().sum(axis=1)
    over = gross > engine.max_leverage
    if over.any():
        w_target.loc[over] = w_target.loc[over].div(gross[over], axis=0) * engine.max_leverage

    # --- Execution lag: this is where the future is neutralized -------
    w_exec = w_target.shift(max(0, int(engine.execution_lag))).fillna(0.0)

    # Rebalance dates are *signal* dates and are shifted forward by the
    # execution lag, so the trade lands on the session the order could
    # actually be placed. A month-end signal with a one-day lag trades on
    # the first session of the next month.
    #
    # The shift has to happen here as well as on the weights. Applying the
    # lag only to the weights while firing the trade on the signal date
    # counts it twice: the engine would trade on month-end using the
    # weights from the day before, so a signal produced at month-end would
    # not reach the book until the *following* rebalance -- a full period
    # late for a quarterly strategy.
    lag = max(0, int(engine.execution_lag))
    if rebalance_dates is not None and len(rebalance_dates):
        sig = pd.DatetimeIndex(rebalance_dates).intersection(idx)
    else:
        sig = build_calendar(idx, spec_from_engine(engine))
    pos = idx.get_indexer(pd.DatetimeIndex(sig)) + lag
    rebal = idx[pos[(pos >= 0) & (pos < len(idx))]]
    rebal_set = set(rebal)

    # --- Prices used for marking and for trading ------------------------
    # Two distinct things. `close` marks the book at the end of each day;
    # `exec_px` is what a trade actually fills at. They differ only when
    # trading at the open, which is the whole point of the option.
    close_px = prices.to_numpy(dtype=float)
    trade_at_open = open_prices is not None
    if trade_at_open:
        op = open_prices.reindex(index=idx, columns=assets)
        # A missing or nonsensical open falls back to that day's close
        # rather than inventing a price out of a neighbouring bar.
        op = op.where(op.notna() & (op > 0), prices)
        exec_px = op.to_numpy(dtype=float)
    else:
        exec_px = close_px

    div_ps = (dividends.reindex(index=idx, columns=assets).fillna(0.0)
              .to_numpy(dtype=float) if dividends is not None
              else np.zeros((len(idx), len(assets))))

    cash_ret = _cash_returns(idx, cash_prices, costs.cash_rate_pa, engine.periods_per_year)
    Rc = cash_ret.to_numpy(dtype=float)
    ppy = max(1, int(engine.periods_per_year))
    borrow_daily = costs.borrow_rate_pa / ppy
    fee_daily = float(getattr(costs, "management_fee_pa", 0.0)) / ppy
    commission_rate = float(costs.commission_bps) / 10_000.0
    min_trade = float(engine.min_trade_weight)
    whole = bool(getattr(engine, "whole_shares", False))

    # --- Per-instrument slippage: a flat rate, a manual multiplier per
    # instrument, and an optional volume-scaled impact term added on top.
    # Commission stays a pure flat rate throughout -- it is a broker fee,
    # not a liquidity cost, so it is never scaled by size or overrides.
    overrides = dict(getattr(costs, "slippage_overrides", {}) or {})
    slip_mult = np.array([float(overrides.get(a, 1.0)) for a in assets])
    base_slip_rate = (float(costs.slippage_bps) / 10_000.0) * slip_mult

    impact_model = getattr(costs, "impact_model", "flat")
    impact_coef = float(getattr(costs, "impact_coef", 1.0))
    ADV_WINDOW = 20
    adv_dollars = None
    sigma = None
    no_volume_assets: List[str] = []
    if impact_model == "sqrt_vol":
        from .costs import daily_volatility
        sigma = daily_volatility(prices, ADV_WINDOW).to_numpy(dtype=float)
    if impact_model in ("sqrt", "sqrt_vol"):
        if volume is not None:
            vol = volume.reindex(index=idx, columns=assets)
            # Shifted by one session: the impact of today's trade is priced
            # against liquidity known BEFORE today, never against today's
            # own not-yet-complete volume.
            trail = vol.rolling(ADV_WINDOW, min_periods=max(5, ADV_WINDOW // 4)).mean().shift(1)
            adv_dollars = (trail * prices).to_numpy(dtype=float)
        no_volume_assets = [a for i, a in enumerate(assets)
                            if adv_dollars is None or not np.isfinite(adv_dollars[:, i]).any()]

    impact_bps_10 = float(getattr(costs, "impact_bps_at_10pct_adv", 0.0))
    apply_fee_frictions = bool(getattr(costs, "apply_frictions_to_fee_liquidation", True))

    def _rate_vector(i: int, notional: np.ndarray) -> np.ndarray:
        """Effective slippage rate per instrument for this trade, before
        commission. `notional` is the dollar size of THIS trade, per
        instrument -- the participation rate is trade size over trailing
        ADV, so the same order costs more in a thinner name."""
        rate = base_slip_rate.copy()
        if adv_dollars is None:
            return rate
        adv_i = adv_dollars[i]
        usable = np.isfinite(adv_i) & (adv_i > 0)
        participation = np.clip(
            np.where(usable, notional / np.where(usable, adv_i, 1.0), 0.0), 0.0, None)
        if impact_model == "sqrt" and impact_bps_10 > 0:
            impact = (impact_bps_10 / 10_000.0) * np.sqrt(participation / 0.10)
            rate = np.where(usable, rate + impact, rate)
        elif impact_model == "sqrt_vol" and sigma is not None:
            sig = sigma[i]
            ok = usable & np.isfinite(sig)
            impact = impact_coef * np.where(ok, sig, 0.0) * np.sqrt(participation)
            rate = np.where(ok, rate + impact, rate)
        return rate

    n, m = len(idx), len(assets)
    T = w_exec.to_numpy(dtype=float)

    # Fee charge dates: the last session of each calendar month. Accrual is
    # daily, the deduction is monthly, which is how a management fee is
    # actually billed.
    month_end = set()
    if fee_daily > 0:
        ser = pd.Series(idx, index=idx)
        month_end = {np.datetime64(d, "ns")
                     for d in ser.resample("ME").last().dropna()}

    # Cash reinvestment between rebalances.
    sweep_rule = str(getattr(engine, "cash_sweep", "none") or "none")
    sweep_thr = float(getattr(engine, "cash_sweep_threshold", 0.0) or 0.0)
    buffer_w = float(getattr(engine, "cash_buffer", 0.01) or 0.0)
    sweep_set = (set(rebalance_calendar(idx, sweep_rule))
                 if sweep_rule in ("W", "M", "Q") else set())
    sweep_on = bool(sweep_set) or sweep_thr > 0
    last_target: Optional[np.ndarray] = None

    nav = np.zeros(n); gross_r = np.zeros(n); net_r = np.zeros(n)
    turn = np.zeros(n); cost_arr = np.zeros(n); div_arr = np.zeros(n)
    fee_arr = np.zeros(n)
    W = np.zeros((n, m)); cash_w = np.zeros(n); SH = np.zeros((n, m))
    # Contribution ledger, in currency: instruments, then cash & financing,
    # trading costs, management fee. Built from the same marks and share
    # counts as the book itself, so it reconciles to the NAV by construction.
    PNL = np.zeros((n, m + 3))

    shares = np.zeros(m)              # units held, the actual state
    cash = float(engine.initial_capital)
    fee_accrued = 0.0
    fee_sessions = 0
    trades = []
    flows = []

    def _valid(row):
        return np.isfinite(row) & (row > 0)

    def _execute(i, date, price_row, portfolio_value):
        """Moves the book to the target at `price_row`. Returns cost paid.

        Everything here is in shares and currency. The target weight is
        turned into a target *number of units at the execution price*, and
        the trade log records that price, so what filled and at what is
        auditable rather than implied.
        """
        nonlocal shares, cash
        ok = _valid(price_row)
        if portfolio_value <= 0 or not ok.any():
            return 0.0, 0.0

        target_val = T[i] * portfolio_value
        target_sh = np.where(ok, target_val / np.where(ok, price_row, 1.0), shares)
        if whole:
            target_sh = np.trunc(target_sh)

        delta = target_sh - shares
        notional = np.abs(delta) * np.where(ok, price_row, 0.0)

        # Ignore adjustments too small to be worth their own commission --
        # but never block a full exit. Closing a position is not a micro
        # adjustment: skipping it leaves a residual the model no longer
        # wants, and under rotation those residuals accumulate until the
        # book holds more names than the strategy ever selected. A "top 3"
        # that drifts to eight positions is the symptom.
        exiting = (target_sh == 0.0) & (shares != 0.0)
        tiny = (notional < min_trade * portfolio_value) & ~exiting
        delta = np.where(tiny, 0.0, delta)
        if not np.any(delta):
            return 0.0, 0.0

        # The rate is computed once, from the size implied by the target
        # weights, and reused through the cash-availability adjustment
        # below. A trade that gets scaled down for lack of cash would, in
        # a perfectly exact model, also cost slightly less to match its
        # smaller size -- but that is a second-order refinement chasing a
        # fixed point, and a single-pass estimate is standard practice for
        # a simulation rather than a live order router.
        notional0 = np.abs(delta) * np.where(ok, price_row, 0.0)
        slip_rate = _rate_vector(i, notional0)
        rate = commission_rate + slip_rate

        # Sell first, then buy with the proceeds -- the order a desk would
        # actually use. Scaling the whole order when cash is short would
        # shrink the sales too, which is precisely what leaves a position
        # half-closed and the book holding names it meant to exit.
        sells = np.where(delta < 0, delta, 0.0)
        buys = np.where(delta > 0, delta, 0.0)

        sell_notional_i = np.abs(sells) * np.where(ok, price_row, 0.0)
        cash_avail = cash + float((sell_notional_i * (1.0 - rate)).sum())

        buy_notional_i = buys * np.where(ok, price_row, 0.0)
        buy_notional = float(buy_notional_i.sum())
        if buy_notional > 0:
            need = float((buy_notional_i * (1.0 + rate)).sum())
            if need > cash_avail + 1e-9:
                scale = max(0.0, cash_avail / need) if need > 1e-12 else 0.0
                buys = buys * min(1.0, scale)
                if whole:
                    buys = np.trunc(buys)

        delta = sells + buys
        notional = np.abs(delta) * np.where(ok, price_row, 0.0)
        if not np.any(delta):
            return 0.0, 0.0
        cost = float((notional * rate).sum())
        net_spend = float((delta * np.where(ok, price_row, 0.0)).sum())

        for j, a in enumerate(assets):
            if delta[j] != 0.0:
                trades.append({
                    "Date": date, "Instrument": a,
                    "Shares Before": shares[j], "Shares After": shares[j] + delta[j],
                    "Change": delta[j], "Price": float(price_row[j]),
                    "Notional": float(abs(delta[j]) * price_row[j]),
                    "Effective Cost (bps)": float(rate[j]) * 10_000.0,
                })
        shares = shares + delta
        cash -= net_spend + cost
        return cost, float(notional.sum()) / portfolio_value

    def _reinvest(i, date, price_row, portfolio_value):
        """Puts cash above the buffer back to work. Returns cost, turnover.

        Buys only, toward the weights the last rebalance targeted -- no new
        signal is read between rebalances -- filling the positions that
        are furthest below target first, and never lifting one above its
        target: a strategy that meant to hold cash keeps it.
        """
        nonlocal shares, cash
        ok = _valid(price_row) & (last_target > 0)
        if portfolio_value <= 0 or not ok.any():
            return 0.0, 0.0
        excess = cash - buffer_w * portfolio_value
        if excess <= max(min_trade * portfolio_value, 1e-9):
            return 0.0, 0.0
        px_ = np.where(ok, price_row, 1.0)
        gaps = np.where(ok, np.clip(last_target * portfolio_value - shares * px_, 0.0, None), 0.0)
        total_gap = float(gaps.sum())
        if total_gap <= 0:
            return 0.0, 0.0
        spend = min(excess, total_gap)
        alloc = gaps * (spend / total_gap)              # dollars, costs included
        rate = commission_rate + _rate_vector(i, alloc)
        buy = np.where(ok, alloc / (1.0 + rate) / px_, 0.0)
        if whole:
            buy = np.floor(buy)
        notional = buy * px_
        if float(notional.sum()) <= 0:
            return 0.0, 0.0
        cost = float((notional * rate).sum())
        cash_before = cash
        for j, a in enumerate(assets):
            if buy[j] > 0:
                trades.append({
                    "Date": date, "Instrument": a,
                    "Shares Before": shares[j], "Shares After": shares[j] + buy[j],
                    "Change": buy[j], "Price": float(price_row[j]),
                    "Notional": float(notional[j]),
                    "Effective Cost (bps)": float(rate[j]) * 10_000.0,
                    "Reason": "Cash reinvestment",
                })
        shares = shares + buy
        cash -= float(notional.sum()) + cost
        flows.append({"Date": date, "Type": "Cash reinvestment",
                      "Amount": -(float(notional.sum()) + cost),
                      "Trading costs": cost, "Cash before": cash_before,
                      "Cash after": cash})
        return cost, float(notional.sum()) / portfolio_value

    # ------------------------------------------------------------------
    # One day at a time. Nothing below reads a future row.
    # ------------------------------------------------------------------
    for i, date in enumerate(idx):
        prev_value = nav[i - 1] if i > 0 else float(engine.initial_capital)
        cost_i = 0.0
        div_i = 0.0
        fee_i = 0.0
        shares_open = shares.copy()        # units carried into the day
        cash_pnl = 0.0
        div_vec = np.zeros(m)

        if i > 0:
            # 1) Cash earns overnight; leverage is charged for.
            cash_pnl = cash * Rc[i]
            cash *= (1.0 + Rc[i])
            gross_prev = float(np.abs(shares * np.nan_to_num(close_px[i - 1])).sum())
            lev = max(0.0, gross_prev - prev_value)
            cash -= lev * borrow_daily
            cash_pnl -= lev * borrow_daily

            # 2) Dividends go ex at the open, on the units held into the day.
            if div_ps[i].any():
                div_vec = shares * div_ps[i]
                div_i = float(div_vec.sum())
                cash += div_i
                for j in np.flatnonzero(div_vec):
                    flows.append({"Date": date, "Type": "Dividend",
                                  "Instrument": assets[j],
                                  "Amount": float(div_vec[j]),
                                  "Units": float(shares[j]),
                                  "Per unit": float(div_ps[i][j])})

        # 3) Trade. At the open the book is valued at open prices first, so
        #    the order is sized on what it is worth when it is placed.
        can_trade = (date in rebal_set) and (i >= int(engine.execution_lag))
        if can_trade:
            last_target = T[i].copy()
        can_sweep = (sweep_on and not can_trade and i > 0
                     and last_target is not None
                     and ((date in sweep_set)
                          or (sweep_thr > 0 and cash_w[i - 1] > sweep_thr)))
        trades_at_open = trade_at_open and (can_trade or can_sweep)
        if trades_at_open:
            mark_open = np.where(_valid(exec_px[i]), exec_px[i], 0.0)
            value_at_open = float((shares * mark_open).sum()) + cash
            if can_trade:
                cost_i, tr = _execute(i, date, exec_px[i], value_at_open)
            else:
                cost_i, tr = _reinvest(i, date, exec_px[i], value_at_open)
            turn[i] = tr

        # 4) Mark to the close.
        mark = np.where(_valid(close_px[i]), close_px[i], 0.0)
        value = float((shares * mark).sum()) + cash

        # Price P&L per instrument. Trading at the open splits the day:
        # the overnight move on the units carried in, the intraday move on
        # the units held after the trade. Trading at the close earns the
        # whole day on the units carried in. A trade itself swaps cash for
        # units at the fill price, so only its cost touches the P&L.
        if i > 0:
            mark_prev = np.where(_valid(close_px[i - 1]), close_px[i - 1], 0.0)
            if trades_at_open:
                mark_open = np.where(_valid(exec_px[i]), exec_px[i], 0.0)
                price_pnl = (shares_open * (mark_open - mark_prev)
                             + shares * (mark - mark_open))
            else:
                price_pnl = shares_open * (mark - mark_prev)
            PNL[i, :m] = price_pnl + div_vec
            PNL[i, m] = cash_pnl

        if (can_trade or can_sweep) and not trade_at_open:
            if can_trade:
                cost_i, tr = _execute(i, date, close_px[i], value)
            else:
                cost_i, tr = _reinvest(i, date, close_px[i], value)
            turn[i] = tr
            value = float((shares * mark).sum()) + cash

        # 5) Management fee: accrued every day on the marked value,
        #    deducted once a month.
        #
        #    A fully invested book holds no cash, so the fee has to come out
        #    of the holdings. Capping it at whatever cash happens to be
        #    lying around would mean a fully invested strategy quietly pays
        #    almost nothing, which is the opposite of the truth.
        liq_cost = 0.0
        if fee_daily > 0 and i > 0:
            fee_accrued += value * fee_daily
            fee_sessions += 1
            if (np.datetime64(date, "ns") in month_end) or i == n - 1:
                fee_i = min(fee_accrued, max(0.0, value))
                cash_before_fee = cash
                raised = 0.0
                shortfall = fee_i - cash
                if shortfall > 0:
                    holdings_val = float((shares * mark).sum())
                    if holdings_val > 1e-12:
                        # Sell pro rata across the book, exactly as a fund
                        # liquidates units to meet its own fee. When this
                        # sale pays the same trading costs as any other
                        # trade, slightly more than the shortfall itself
                        # must be raised to net the right amount after
                        # those costs -- an involuntary redemption trade
                        # is not free just because it was involuntary.
                        # The rate is estimated once, from the cost-free
                        # sale size, then used to solve the slightly larger
                        # sale that nets the shortfall -- the same
                        # single-pass approach as an ordinary trade, not a
                        # fixed-point iteration.
                        naive_keep = max(0.0, 1.0 - shortfall / holdings_val)
                        if apply_fee_frictions:
                            naive_sold = shares * (1.0 - naive_keep)
                            naive_notional = np.abs(naive_sold * mark)
                            rate0 = commission_rate + _rate_vector(i, naive_notional)
                            sold_total = float(naive_notional.sum())
                            # The rate weighted by what is actually being
                            # SOLD, not by total holdings -- dividing by
                            # holdings_val here would silently shrink this
                            # toward zero whenever the fee is small relative
                            # to the book (the ordinary case), making the
                            # sale barely account for its own cost at all.
                            blended = (float((naive_notional * rate0).sum()) / sold_total
                                      if sold_total > 0 else 0.0)
                            keep = (0.0 if blended >= 1.0 else
                                   max(0.0, 1.0 - (shortfall / (1.0 - blended)) / holdings_val))
                        else:
                            rate0 = np.zeros(m)
                            keep = naive_keep
                        sold_shares = shares * (1.0 - keep)
                        if whole:
                            sold_shares = _whole_unit_sale(
                                sold_shares, shares, mark, rate0, shortfall)
                        proceeds_gross = float((sold_shares * mark).sum())
                        rate_final = np.zeros(m)
                        if apply_fee_frictions and proceeds_gross > 0:
                            rate_final = commission_rate + _rate_vector(i, np.abs(sold_shares * mark))
                            liq_cost = float((np.abs(sold_shares * mark) * rate_final).sum())
                        # Every sale is logged, whether or not it pays
                        # frictions: units leaving the book are a trade.
                        for j, a in enumerate(assets):
                            if sold_shares[j] != 0.0:
                                trades.append({
                                    "Date": date, "Instrument": a,
                                    "Shares Before": shares[j],
                                    "Shares After": shares[j] - sold_shares[j],
                                    "Change": -sold_shares[j],
                                    "Price": float(mark[j]),
                                    "Notional": float(abs(sold_shares[j]) * mark[j]),
                                    "Effective Cost (bps)": float(rate_final[j]) * 10_000.0,
                                    "Reason": "Fee liquidation",
                                })
                        shares = shares - sold_shares
                        cash += proceeds_gross - liq_cost
                        raised = proceeds_gross - liq_cost
                cash -= fee_i
                flows.append({
                    "Date": date, "Type": "Management fee", "Amount": -fee_i,
                    "Sessions accrued": fee_sessions,
                    "Average value": (fee_accrued / (fee_daily * fee_sessions)
                                      if fee_sessions else np.nan),
                    "Annual rate": fee_daily * ppy,
                    "Paid from cash": min(fee_i, max(0.0, cash_before_fee)),
                    "Raised by selling": raised, "Trading costs": liq_cost,
                    "Cash before": cash_before_fee, "Cash after": cash})
                fee_accrued -= fee_i
                fee_sessions = 0
                value = float((shares * mark).sum()) + cash

        nav[i] = value
        if i > 0:
            PNL[i, m + 1] = -(cost_i + liq_cost)
            PNL[i, m + 2] = -fee_i
        r_port = (value / prev_value - 1.0) if (i > 0 and prev_value > 0) else 0.0
        net_r[i] = r_port
        gross_r[i] = ((value + cost_i + liq_cost + fee_i) / prev_value - 1.0) if (i > 0 and prev_value > 0) else 0.0
        cost_arr[i] = (cost_i + liq_cost) / prev_value if prev_value > 0 else 0.0
        fee_arr[i] = fee_i / prev_value if prev_value > 0 else 0.0
        div_arr[i] = div_i / prev_value if prev_value > 0 else 0.0
        SH[i] = shares
        if value > 0:
            W[i] = shares * mark / value
            cash_w[i] = cash / value

    weights = pd.DataFrame(W, index=idx, columns=assets)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Date", "Instrument", "Shares Before", "Shares After",
                 "Change", "Price", "Notional", "Effective Cost (bps)", "Reason"])
    if "Reason" not in trades_df.columns:
        trades_df["Reason"] = "Rebalance"
    else:
        trades_df["Reason"] = trades_df["Reason"].fillna("Rebalance")

    exposure = weights.abs().sum(axis=1)
    active = exposure[exposure > 1e-9]

    prev_nav = np.concatenate([[float(engine.initial_capital)], nav[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(prev_nav[:, None] > 0, PNL / prev_nav[:, None], 0.0)
    contributions = pd.DataFrame(
        contrib, index=idx,
        columns=assets + [CASH_LINE, COST_LINE, FEE_LINE])

    return BacktestResult(
        equity=pd.Series(nav, index=idx, name=label),
        returns=pd.Series(net_r, index=idx, name=label),
        gross_returns=pd.Series(gross_r, index=idx, name=label),
        weights=weights,
        target_weights=w_exec,
        turnover=pd.Series(turn, index=idx),
        costs=pd.Series(cost_arr, index=idx),
        exposure=exposure,
        cash_weight=pd.Series(cash_w, index=idx),
        rebalance_dates=rebal,
        trades=trades_df,
        label=label,
        dividend_income=pd.Series(div_arr, index=idx),
        warmup_start=active.index[0] if len(active) else None,
        shares=pd.DataFrame(SH, index=idx, columns=assets),
        fees=pd.Series(fee_arr, index=idx),
        contributions=contributions,
        cash_flows=pd.DataFrame(flows, columns=CASH_FLOW_COLUMNS),
    )


# ----------------------------------------------------------------------
def first_active_date(res: BacktestResult,
                      threshold: float = 1e-9) -> Optional[pd.Timestamp]:
    """First day the portfolio actually holds something."""
    active = res.exposure[res.exposure > threshold]
    return active.index[0] if len(active) else None


def trim_warmup(res: BacktestResult, start: Optional[pd.Timestamp] = None,
                initial_capital: Optional[float] = None) -> BacktestResult:
    """Drops the leading stretch where the strategy had no position yet.

    Indicators need history before they can produce a first signal: a
    200-day moving average is blind for its first 200 sessions. Those
    sessions are not a flat, neutral prologue. The portfolio sits in cash
    and *earns the cash rate*, which quietly lifts the reported return,
    lengthens the measured period, and dilutes volatility and drawdown --
    all of it manufactured by the warm-up rather than by the strategy.

    Trimming starts the record on the first day capital is actually at
    risk. Only the leading run is removed: a strategy that deliberately
    moves to cash mid-period keeps that stretch, because there the cash
    position is a decision rather than an artefact.
    """
    start = start or first_active_date(res)
    if start is None or start <= res.equity.index[0]:
        return res

    idx = res.equity.index
    keep = idx[idx >= start]
    if len(keep) < 2:
        return res

    base = float(res.equity.loc[start])
    cap = float(initial_capital) if initial_capital else base
    equity = res.equity.loc[keep] / base * cap

    returns = res.returns.loc[keep].copy()
    returns.iloc[0] = 0.0          # the first retained day is the new origin
    gross = res.gross_returns.loc[keep].copy()
    gross.iloc[0] = 0.0

    trades = res.trades
    if not trades.empty and "Date" in trades.columns:
        trades = trades[trades["Date"] >= start].reset_index(drop=True)

    return BacktestResult(
        equity=equity, returns=returns, gross_returns=gross,
        weights=res.weights.loc[keep], target_weights=res.target_weights.loc[keep],
        turnover=res.turnover.loc[keep], costs=res.costs.loc[keep],
        exposure=res.exposure.loc[keep], cash_weight=res.cash_weight.loc[keep],
        rebalance_dates=pd.DatetimeIndex([d for d in res.rebalance_dates if d >= start]),
        trades=trades, label=res.label,
        dividend_income=(res.dividend_income.loc[keep]
                         if res.dividend_income is not None else None),
        warmup_start=start,
        # `shares` and `fees` were added to BacktestResult after this
        # function was first written, and reconstructing the object here
        # without them silently dropped both on every trimmed run -- the
        # common case, since most strategies need history before their
        # first signal. Neither field affects a number already computed
        # above; both are carried through exactly like every other field.
        shares=(res.shares.loc[keep] if res.shares is not None else None),
        fees=(res.fees.loc[keep] if res.fees is not None else None),
        contributions=_trim_contrib(res.contributions, keep),
        cash_flows=_cut_flows(res.cash_flows, start, None),
    )


def window(res: BacktestResult, start: Optional[pd.Timestamp] = None,
           end: Optional[pd.Timestamp] = None) -> BacktestResult:
    """The result restricted to the sessions from `start` to `end`.

    For reporting on part of a finished run, not for re-running it: the
    positions, trades and signals are exactly those of the full backtest.
    The first retained session becomes the origin (its return is zeroed,
    as in `trim_warmup`), but the value keeps its dollar level, so holdings,
    share counts and trades still reconcile with the full run.
    """
    idx = res.equity.index
    if start is not None:
        after = idx[idx >= pd.Timestamp(start)]
        start = after[0] if len(after) else None
    out = trim_warmup(res, start) if start is not None and start > idx[0] else res
    if end is None:
        return out
    keep = out.equity.index[out.equity.index <= pd.Timestamp(end)]
    if len(keep) < 2 or len(keep) == len(out.equity.index):
        return out

    def cut(x):
        return None if x is None else x.loc[keep]

    trades = out.trades
    if not trades.empty and "Date" in trades.columns:
        trades = trades[trades["Date"] <= keep[-1]].reset_index(drop=True)
    return replace(
        out, equity=cut(out.equity), returns=cut(out.returns),
        gross_returns=cut(out.gross_returns), weights=cut(out.weights),
        target_weights=cut(out.target_weights), turnover=cut(out.turnover),
        costs=cut(out.costs), exposure=cut(out.exposure),
        cash_weight=cut(out.cash_weight),
        rebalance_dates=pd.DatetimeIndex(
            [d for d in out.rebalance_dates if d <= keep[-1]]),
        trades=trades, dividend_income=cut(out.dividend_income),
        shares=cut(out.shares), fees=cut(out.fees),
        contributions=cut(out.contributions),
        cash_flows=_cut_flows(out.cash_flows, None, keep[-1]))


def _cut_flows(f: Optional[pd.DataFrame], start, end) -> Optional[pd.DataFrame]:
    if f is None or f.empty:
        return f
    d = pd.to_datetime(f["Date"])
    keep = pd.Series(True, index=f.index)
    if start is not None:
        keep &= d >= pd.Timestamp(start)
    if end is not None:
        keep &= d <= pd.Timestamp(end)
    return f[keep].reset_index(drop=True)


def _trim_contrib(c: Optional[pd.DataFrame],
                  keep: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Contributions from the new origin on. Its first row is zeroed, as
    the first retained return is: that day is the base, not a result."""
    if c is None:
        return None
    out = c.loc[keep].copy()
    out.iloc[0] = 0.0
    return out


def align_start(*results: Optional[BacktestResult],
                initial_capital: Optional[float] = None,
                signal_start: Optional[pd.Timestamp] = None):
    """Trims a set of results to their common first active day.

    The benchmark is invested from day one, so left alone it would be
    credited with the whole warm-up while the strategy sat in cash. Both
    sides have to start on the same date for the comparison to mean
    anything.

    `signal_start` overrides the portfolio's own first active day, for the
    case where part of the book is invested before the model has a signal.
    A permanently held core makes the portfolio look active from session
    one, so the total exposure never reveals the warm-up: what gets
    measured is a stretch of core-plus-cash that the strategy had no hand
    in. Passing the date the model first takes a position starts the record
    where the strategy actually begins.
    """
    live = [r for r in results if r is not None]
    if not live:
        return list(results)
    starts = [first_active_date(r) for r in live]
    starts = [s for s in starts if s is not None]
    if signal_start is not None:
        starts.append(pd.Timestamp(signal_start))
    if not starts:
        return list(results)
    common = max(starts)
    return [None if r is None else trim_warmup(r, common, initial_capital)
            for r in results]


# ----------------------------------------------------------------------
def benchmark_result(bench_prices: pd.Series, engine: EngineConfig,
                     label: str = "Benchmark",
                     dividends: Optional[pd.Series] = None,
                     reinvest_rule: Optional[str] = None) -> BacktestResult:
    """Buy-and-hold on the index, no frictions, for comparison.

    The purchase happens on the first available session. Leaving it to a
    periodic calendar would park the benchmark in cash until the first
    scheduled rebalance -- up to a full year on an annual calendar -- and
    quietly understate the bar the strategy is measured against. After that
    single purchase the weight simply drifts, which is what buy-and-hold
    means.
    """
    px = bench_prices.dropna().to_frame()
    if px.empty:
        raise ValueError("Benchmark series is empty.")
    w = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    zero_costs = CostConfig(commission_bps=0, slippage_bps=0,
                            cash_rate_pa=0, borrow_rate_pa=0)
    eng = EngineConfig(initial_capital=engine.initial_capital, rebalance="D",
                       execution_lag=0, max_leverage=1.0,
                       min_trade_weight=0.0,
                       periods_per_year=engine.periods_per_year)
    div = None
    if dividends is not None:
        div = dividends.reindex(px.index).fillna(0.0).to_frame()
        div.columns = px.columns

    # With dividends paid as cash, the benchmark needs a recurring
    # rebalance or the cash would pile up uninvested forever and the
    # comparison would drift below a true total-return index. Without
    # dividends a single purchase is enough: the weight simply drifts.
    if div is not None and float(div.to_numpy().sum()) > 0:
        eng.rebalance = reinvest_rule or engine.rebalance
        dates = rebalance_calendar(px.index, eng.rebalance)
        dates = pd.DatetimeIndex(
            sorted(set([px.index[0]]) | set(dates)))
    else:
        dates = pd.DatetimeIndex([px.index[0]])

    return run_backtest(px, w, eng, zero_costs, label=label,
                        rebalance_dates=dates, dividends=div)


def blended_benchmark(prices: pd.DataFrame,
                      weights: Dict[str, float],
                      engine: EngineConfig,
                      label: str = "Benchmark",
                      dividends: Optional[pd.DataFrame] = None,
                      rebalance: str = "A") -> BacktestResult:
    """A fixed-weight benchmark across several instruments.

    The blend is run through the same engine as everything else rather than
    being averaged: a 60/40 that is rebalanced is not the same thing as the
    weighted average of its two return series, because the drift between
    rebalances is real. Frictions are zero, since a policy benchmark is a
    measuring stick and nobody pays commissions on it.

    `rebalance` sets how often the blend is restored to its target weights.
    This matters more than it looks: an unrebalanced 60/40 drifts toward
    equities over a long window and quietly becomes a harder benchmark to
    beat in a bull market, and an easier one in a drawdown.
    """
    cols = [c for c in weights if c in prices.columns and float(weights[c]) != 0.0]
    if not cols:
        raise ValueError("None of the benchmark's instruments are available.")

    px = prices[cols].dropna(how="all")
    if px.empty:
        raise ValueError("The benchmark's instruments have no overlapping history.")
    # Start only once every component has a price, otherwise the first
    # weights would be applied to an incomplete blend.
    first = px.dropna().index
    if len(first):
        px = px.loc[first[0]:]

    total = sum(abs(float(weights[c])) for c in cols) or 1.0
    w = pd.DataFrame(
        {c: float(weights[c]) / total for c in cols},
        index=px.index)

    zero = CostConfig(commission_bps=0, slippage_bps=0,
                      cash_rate_pa=0, borrow_rate_pa=0)
    eng = EngineConfig(initial_capital=engine.initial_capital,
                       rebalance=rebalance, execution_lag=0,
                       max_leverage=1.0, min_trade_weight=0.0,
                       periods_per_year=engine.periods_per_year)

    div = None
    if dividends is not None:
        div = dividends.reindex(index=px.index, columns=cols).fillna(0.0)
        if float(div.to_numpy().sum()) <= 0:
            div = None

    dates = rebalance_calendar(px.index, rebalance)
    dates = pd.DatetimeIndex(sorted(set([px.index[0]]) | set(dates)))
    return run_backtest(px, w, eng, zero, label=label,
                        rebalance_dates=dates, dividends=div)


def align_results(results: Dict[str, "BacktestResult | pd.Series"]) -> pd.DataFrame:
    """Value curves rebased to 100 over the common period.

    Accepts either BacktestResult objects or raw value series.
    """
    eq = pd.DataFrame({
        k: (v.equity if isinstance(v, BacktestResult) else v)
        for k, v in results.items()
    }).dropna()
    if eq.empty:
        return eq
    return eq.div(eq.iloc[0]) * 100.0
