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
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig, CostConfig
from .fx import FxBook
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
    # Multi-currency runs only. `foreign_cash`: what each foreign-currency
    # cash balance is worth in the base currency, per day. `fx_costs`: the
    # spread paid on conversions, as a fraction of the previous day's value
    # (already inside `costs`; shown apart so the friction can be read on
    # its own).
    foreign_cash: Optional[pd.DataFrame] = None
    fx_costs: Optional[pd.Series] = None

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
    if rule == "S":
        return build_calendar(index, RebalanceSpec(frequency="S"))
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
                 volume: Optional[pd.DataFrame] = None,
                 fx: Optional[FxBook] = None) -> BacktestResult:
    """Simulates the portfolio day by day. Returns all diagnostics.

    `fx` switches on multi-currency accounting. `prices` and `dividends`
    (and `open_prices`) must already be in the base currency -- the book
    converts them once, up front, so signals and everything else live in the
    investor's currency -- and `fx` says which instrument is quoted in which
    currency, with the daily rates. The account then keeps one cash balance
    per foreign currency. A trade in a foreign instrument settles in that
    currency; dividends land there; the spread is paid only on the NET
    amount that crosses between currencies at a settlement (see
    `EngineConfig.fx_cash`). Without foreign instruments, or without `fx`,
    nothing below differs from a single-currency run.

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

    # --- Currencies -----------------------------------------------------
    # K foreign currencies actually present in the book. With none, every
    # branch below guarded by `K` is skipped and the arithmetic is the
    # single-currency arithmetic, operation for operation.
    fx_ccys: List[str] = fx.foreign_currencies(assets) if fx is not None else []
    K = len(fx_ccys)
    kidx = np.full(m, -1, dtype=int)            # which foreign currency each asset settles in
    for j, a in enumerate(assets):
        if K and fx.currency_of(a) in fx_ccys:
            kidx[j] = fx_ccys.index(fx.currency_of(a))
    FXR = fx.aligned(idx, fx_ccys) if K else np.ones((n, 0))   # base per foreign unit
    # The rate at which the day's trades, conversions and dividends happen.
    # A fill at the close can use the close rate. A fill at the open cannot:
    # that day's close is in the future, and the only rate known at the open
    # is the previous close. Using today's would let a fill (and the size of
    # the order) depend on information that does not exist yet. The rest of
    # the day then earns the move from that rate to the close.
    FXE = (np.vstack([FXR[:1], FXR[:-1]]) if (K and trade_at_open) else FXR)
    fx_spread = float(getattr(costs, "fx_spread_bps", 0.0)) / 10_000.0 if K else 0.0
    fx_hold = bool(K) and str(getattr(engine, "fx_cash", "net")) == "hold"
    foreign_daily = ((1.0 + float(getattr(costs, "foreign_cash_rate_pa", 0.0)))
                     ** (1.0 / ppy) - 1.0) if K else 0.0
    # The spread each foreign trade could pay, added to its rate only when
    # SIZING an order. The order is sized as if every foreign trade were
    # converted on its own; the actual settlement nets them, so what is
    # really paid never exceeds what was reserved and cash cannot go short.
    fxv = np.where(kidx >= 0, fx_spread, 0.0)
    fc = np.zeros(K)                 # foreign cash, in units of each foreign currency
    FCV = np.zeros((n, K))           # ... and its worth in base currency, per day
    FXC = np.zeros(n)
    fx_day = 0.0                     # spread paid today
    FX_DUST = 1e-9

    def _fcv(i, rates=None):
        """Foreign cash, valued in the base currency: at the close rate, or
        at `rates` (the execution-time rate, for an order placed at the open)."""
        return float((fc * (FXR if rates is None else rates)[i]).sum()) if K else 0.0

    def _fx_cols(j, i, price_base):
        """Currency columns for a trade-log row; none in a one-currency run."""
        if not K:
            return {}
        k = kidx[j]
        rate = 1.0 if k < 0 else float(FXE[i, k])
        return {"Currency": fx_ccys[k] if k >= 0 else fx.base,
                "Local Price": float(price_base) / rate, "FX Rate": rate}

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

    # --- Settling a trade across currencies -----------------------------
    # `flow` is a trade's cash effect per asset in base units, costs
    # included (positive = cash in). A foreign instrument's flow moves that
    # currency's balance; base-currency instruments move the base balance.
    # The balances are then reconciled: under "net" every foreign balance is
    # brought to zero, under "hold" only a shortfall is bought. Either way
    # the spread is paid on the amount converted -- the NET of the day's
    # buys and sells in that currency, not each trade on its own.
    kmasks = [kidx == k for k in range(K)]
    base_mask = kidx < 0
    KM = np.zeros((m, K))                       # asset -> its foreign currency
    for k in range(K):
        KM[kmasks[k], k] = 1.0
    BM = base_mask.astype(float)

    def _plan(i, flow, detail=False):
        """What settling `flow` would do, with no side effects:
        (base cash change, new foreign balances, conversions, reserve),
        where `reserve` is the base value foreign cash left over could still
        raise, so feasibility is judged on total liquidity. The conversions
        are listed only when `detail` asks for them."""
        fxi = FXE[i]
        bal = fc + (flow @ KM) / fxi
        conv = np.minimum(bal, 0.0) if fx_hold else bal      # + sold, - bought
        value = np.abs(conv) * fxi
        live = value > FX_DUST
        c_k = np.where(live, value * fx_spread, 0.0)
        base_flow = float(flow @ BM) + float(
            np.where(live, np.where(conv > 0, value, -value) - c_k, 0.0).sum())
        new_fc = np.where(live, bal - conv, bal)
        reserve = float((np.clip(new_fc, 0.0, None) * fxi).sum()) * (1.0 - fx_spread)
        convs = ([(k, float(conv[k]), float(value[k]), float(c_k[k]))
                  for k in range(K) if live[k]] if detail else None)
        return base_flow, new_fc, convs, reserve

    def _raise_from_foreign(i, date, short, fxi=None):
        """Sells foreign cash, largest first, to raise `short` of base cash,
        at the execution-time rate unless `fxi` says otherwise."""
        nonlocal cash, fx_day
        fxi = FXE[i] if fxi is None else fxi
        got = 0.0
        for k in np.argsort(-(fc * fxi)):
            if short - got <= FX_DUST:
                break
            if fc[k] <= 0:
                continue
            per_unit = fxi[k] * (1.0 - fx_spread)
            take = min(short - got, fc[k] * per_unit)
            units = take / per_unit
            c_k = units * fxi[k] * fx_spread
            fc[k] = 0.0 if units >= fc[k] * (1.0 - 1e-12) else fc[k] - units
            cash += take
            fx_day += c_k
            got += take
            flows.append({"Date": date, "Type": "FX conversion",
                          "Instrument": fx_ccys[k], "Amount": float(take),
                          "Units": float(units), "Per unit": float(fxi[k]),
                          "Trading costs": float(c_k)})
        return got

    def _settle(i, date, flow):
        """Books `flow`, converting as the policy requires. The spread paid
        is added to today's `fx_day`."""
        nonlocal cash, fx_day
        fxi = FXE[i]
        base_flow, new_fc, convs, _ = _plan(i, flow, True)
        for k, conv, value, c_k in convs:
            fx_day += c_k
            flows.append({"Date": date, "Type": "FX conversion",
                          "Instrument": fx_ccys[k],
                          "Amount": float((value if conv > 0 else -value) - c_k),
                          "Units": float(conv), "Per unit": float(fxi[k]),
                          "Trading costs": float(c_k)})
        cash += base_flow
        fc[:] = new_fc
        if fx_hold and cash < -FX_DUST:
            _raise_from_foreign(i, date, -cash)

    def _fit_buys(i, sells, buys, px_, rate):
        """The largest fraction of the buys that the account can pay for,
        spread included, found against the same settlement it will use.

        Liquidity falls steadily as more is bought, and is piecewise linear
        (it bends only where a currency's balance changes sign), so a
        bracketed secant search lands on it in a handful of steps."""
        def liquidity(lam):
            d = sells + lam * buys
            flow = -d * px_ - np.abs(d) * px_ * rate
            base_flow, _, _, reserve = _plan(i, flow)
            return cash + base_flow + reserve
        f_hi = liquidity(1.0)
        if f_hi >= -1e-9:
            return 1.0
        f_lo = liquidity(0.0)
        if f_lo < -1e-9:
            return 0.0
        lo, hi, side = 0.0, 1.0, 0
        for _ in range(80):
            lam = hi - f_hi * (hi - lo) / (f_hi - f_lo)
            f = liquidity(lam)
            if -1e-9 <= f <= 1e-7:
                return lam
            if f > 0:
                lo, f_lo = lam, f
                if side == 1:
                    f_hi *= 0.5            # Illinois step: keeps the secant from stalling
                side = 1
            else:
                hi, f_hi = lam, f
                if side == -1:
                    f_lo *= 0.5
                side = -1
        return lo

    def _idle(i, date):
        """No trade today. Under net settlement, foreign cash left from
        dividends still goes home. Returns (spread paid, turnover)."""
        if K and not fx_hold and np.any(np.abs(fc * FXE[i]) > FX_DUST):
            before = fx_day
            _settle(i, date, np.zeros(m))
            return fx_day - before, 0.0
        return 0.0, 0.0

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
            return _idle(i, date)

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
            if K:
                # Several currencies: how much can be bought depends on how
                # much of it must be converted, so the fraction is solved
                # against the settlement itself rather than a flat rate.
                buys = buys * _fit_buys(i, sells, buys, np.where(ok, price_row, 0.0), rate)
                if whole:
                    buys = np.trunc(buys)
            else:
                need = float((buy_notional_i * (1.0 + rate)).sum())
                if need > cash_avail + 1e-9:
                    scale = max(0.0, cash_avail / need) if need > 1e-12 else 0.0
                    buys = buys * min(1.0, scale)
                    if whole:
                        buys = np.trunc(buys)

        delta = sells + buys
        notional = np.abs(delta) * np.where(ok, price_row, 0.0)
        if not np.any(delta):
            return _idle(i, date)
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
                    **_fx_cols(j, i, price_row[j]),
                })
        shares = shares + delta
        if K:
            before = fx_day
            _settle(i, date, -(delta * np.where(ok, price_row, 0.0)) - notional * rate)
            cost += fx_day - before        # the spread is a cost of this trade
        else:
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
        excess = cash + _fcv(i, FXE) - buffer_w * portfolio_value
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
        # A foreign purchase may need converting first: size it as if it
        # does. What settlement really charges is never more than this.
        buy = np.where(ok, alloc / (1.0 + (rate + fxv if K else rate)) / px_, 0.0)
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
                    **_fx_cols(j, i, price_row[j]),
                })
        shares = shares + buy
        fx_paid = 0.0
        if K:
            before = fx_day
            _settle(i, date, -notional - notional * rate)
            fx_paid = fx_day - before
        else:
            cash -= float(notional.sum()) + cost
        flows.append({"Date": date, "Type": "Cash reinvestment",
                      "Amount": -(float(notional.sum()) + cost + fx_paid),
                      "Trading costs": cost + fx_paid, "Cash before": cash_before,
                      "Cash after": cash})
        return cost + fx_paid, float(notional.sum()) / portfolio_value

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
        fx_day = 0.0                       # exchange spread paid today

        if i > 0:
            # 1) Cash earns overnight; leverage is charged for.
            cash_pnl = cash * Rc[i]
            cash *= (1.0 + Rc[i])
            gross_prev = float(np.abs(shares * np.nan_to_num(close_px[i - 1])).sum())
            lev = max(0.0, gross_prev - prev_value)
            cash -= lev * borrow_daily
            cash_pnl -= lev * borrow_daily
            if K:
                # Foreign cash earns its own rate, and is worth whatever the
                # exchange rate says today: that revaluation is the currency
                # return on cash, and sits on the cash line of the ledger.
                fc_pre = fc.copy()
                fc *= (1.0 + foreign_daily)
                if trade_at_open:
                    # Interest, at the rate it accrued under. The exchange-rate
                    # move is earned after the open, on what is held then.
                    cash_pnl += float(((fc - fc_pre) * FXR[i - 1]).sum())
                else:
                    cash_pnl += (float((fc * FXR[i]).sum())
                                 - float((fc_pre * FXR[i - 1]).sum()))

            # 2) Dividends go ex at the open, on the units held into the day.
            if div_ps[i].any():
                div_vec = shares * div_ps[i]
                div_i = float(div_vec.sum())
                if K:
                    # A dividend is paid in the currency of its instrument
                    # and stays there until the account converts it.
                    cash += float(div_vec[base_mask].sum())
                    for k in range(K):
                        fc[k] += float(div_vec[kmasks[k]].sum()) / FXE[i, k]
                else:
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
            value_at_open = float((shares * mark_open).sum()) + cash + _fcv(i, FXE)
            if can_trade:
                cost_i, tr = _execute(i, date, exec_px[i], value_at_open)
            else:
                cost_i, tr = _reinvest(i, date, exec_px[i], value_at_open)
            turn[i] = tr

        if K and trade_at_open and i > 0:
            # Foreign cash held after the open's trades earns the move from
            # the rate it was converted at to the close.
            cash_pnl += float((fc * (FXR[i] - FXE[i])).sum())

        # 4) Mark to the close.
        mark = np.where(_valid(close_px[i]), close_px[i], 0.0)
        value = float((shares * mark).sum()) + cash + _fcv(i)

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
            value = float((shares * mark).sum()) + cash + _fcv(i)

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
                fx_raise_cost = 0.0
                shortfall = fee_i - cash
                if shortfall > 0 and K:
                    # Idle foreign cash pays before any holding is sold.
                    before = fx_day
                    raised += _raise_from_foreign(i, date, shortfall, FXR[i])
                    fx_raise_cost = fx_day - before
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
                        if apply_fee_frictions or K:
                            naive_sold = shares * (1.0 - naive_keep)
                            naive_notional = np.abs(naive_sold * mark)
                            # A foreign holding sold for the fee is converted
                            # on the spot, so its spread is part of the rate.
                            rate0 = ((commission_rate + _rate_vector(i, naive_notional))
                                     if apply_fee_frictions else np.zeros(m))
                            if K:
                                rate0 = rate0 + fxv
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
                        if (apply_fee_frictions or K) and proceeds_gross > 0:
                            rate_final = ((commission_rate + _rate_vector(i, np.abs(sold_shares * mark)))
                                          if apply_fee_frictions else np.zeros(m))
                            if K:
                                rate_final = rate_final + fxv
                                fx_day += float((np.abs(sold_shares * mark) * fxv).sum())
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
                                    **_fx_cols(j, i, mark[j]),
                                })
                        shares = shares - sold_shares
                        cash += proceeds_gross - liq_cost
                        raised += proceeds_gross - liq_cost
                cash -= fee_i
                liq_cost += fx_raise_cost
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
                value = float((shares * mark).sum()) + cash + _fcv(i)

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
        if K:
            FCV[i] = fc * FXR[i]
            FXC[i] = fx_day / prev_value if prev_value > 0 else 0.0
        if value > 0:
            W[i] = shares * mark / value
            cash_w[i] = (cash + _fcv(i)) / value

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
        foreign_cash=(pd.DataFrame(FCV, index=idx, columns=fx_ccys) if K else None),
        fx_costs=(pd.Series(FXC, index=idx) if K else None),
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

    When the record starts on the day the strategy first buys, that buy is
    part of the record: the commission and slippage it paid come out of the
    first day's return and the starting value, so the strategy starts with
    capital less its entry cost, as a real account would. Cutting the
    record at any other session (a report window, a later common start)
    treats that session as a plain origin and charges nothing.
    """
    first_buy = first_active_date(res)
    start = start or first_buy
    if start is None or start <= res.equity.index[0]:
        return res

    idx = res.equity.index
    keep = idx[idx >= start]
    if len(keep) < 2:
        return res

    # Fraction of the prior value spent on that first buy; zero when the
    # record is not starting on it.
    entry = (-float(res.costs.loc[start])
             if first_buy is not None and start == first_buy else 0.0)

    base = float(res.equity.loc[start])
    cap = float(initial_capital) if initial_capital else base
    equity = res.equity.loc[keep] / base * cap * (1.0 + entry)

    returns = res.returns.loc[keep].copy()
    returns.iloc[0] = entry        # the new origin: nothing earned, entry paid
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
        contributions=_trim_contrib(res.contributions, keep, entry),
        cash_flows=_cut_flows(res.cash_flows, start, None),
        foreign_cash=(res.foreign_cash.loc[keep]
                      if res.foreign_cash is not None else None),
        fx_costs=(res.fx_costs.loc[keep] if res.fx_costs is not None else None),
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
        cash_flows=_cut_flows(out.cash_flows, None, keep[-1]),
        foreign_cash=cut(out.foreign_cash), fx_costs=cut(out.fx_costs))


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


def _trim_contrib(c: Optional[pd.DataFrame], keep: pd.DatetimeIndex,
                  entry: float = 0.0) -> Optional[pd.DataFrame]:
    """Contributions from the new origin on. Its first row is zeroed, as
    the first retained return is: that day is the base, not a result. The
    one exception is an entry cost (`entry`, a negative fraction), which is
    that return and sits on the trading-cost line."""
    if c is None:
        return None
    out = c.loc[keep].copy()
    out.iloc[0] = 0.0
    if entry:
        out.iloc[0, out.columns.get_loc(COST_LINE)] = entry
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
    common = common_start(*results, signal_start=signal_start)
    if common is None:
        return list(results)
    return [None if r is None else trim_warmup(r, common, initial_capital)
            for r in results]


def common_start(*results: Optional[BacktestResult],
                 signal_start: Optional[pd.Timestamp] = None
                 ) -> Optional[pd.Timestamp]:
    """The first day every result (and `signal_start`) is active, which is
    where `align_start` cuts. None when nothing is active yet."""
    starts = [first_active_date(r) for r in results if r is not None]
    starts = [s for s in starts if s is not None]
    if signal_start is not None:
        starts.append(pd.Timestamp(signal_start))
    return max(starts) if starts else None


# ----------------------------------------------------------------------
BENCHMARK_FRICTIONS = ("commission", "slippage", "fee")


def benchmark_costs(costs: CostConfig,
                    apply: Iterable[str]) -> Optional[CostConfig]:
    """The strategy's own frictions, cut down to the ones a benchmark is
    meant to bear. None when nothing is selected, which keeps the benchmark
    a frictionless index.

    `apply` takes any of: "commission", "slippage" (the half-spread, the
    per-instrument multipliers and the market-impact model, which are one
    liquidity cost), and "fee" (the management fee, with its liquidation
    trades paying whichever of the other two are on). Cash and borrow rates
    are left at zero: a fully invested benchmark holds no idle cash and
    uses no leverage, so they would only touch dividends waiting to be
    reinvested.
    """
    on = set(apply)
    if not on & set(BENCHMARK_FRICTIONS):
        return None
    slip = "slippage" in on
    return CostConfig(
        commission_bps=costs.commission_bps if "commission" in on else 0.0,
        slippage_bps=costs.slippage_bps if slip else 0.0,
        impact_model=costs.impact_model if slip else "flat",
        impact_bps_at_10pct_adv=costs.impact_bps_at_10pct_adv,
        impact_coef=costs.impact_coef,
        slippage_overrides=dict(costs.slippage_overrides) if slip else {},
        apply_frictions_to_fee_liquidation=costs.apply_frictions_to_fee_liquidation,
        cash_rate_pa=0.0, borrow_rate_pa=0.0,
        management_fee_pa=costs.management_fee_pa if "fee" in on else 0.0,
    )


def _carry_entry_cost(res: BacktestResult, capital: float) -> BacktestResult:
    """Puts a benchmark's entry cost into its return series.

    The engine treats the first session as the base and records no return
    for it, so a purchase made that day lowers the value (capital less its
    cost) while the returns, which the statistics are computed from, stay
    blind to it. Setting that session's return to what the value did
    relative to the capital makes the two reconcile.
    """
    first = float(res.equity.iloc[0]) / float(capital) - 1.0
    returns = res.returns.copy()
    returns.iloc[0] = first
    return replace(res, returns=returns)


def benchmark_result(bench_prices: pd.Series, engine: EngineConfig,
                     label: str = "Benchmark",
                     dividends: Optional[pd.Series] = None,
                     reinvest_rule: Optional[str] = None,
                     costs: Optional[CostConfig] = None,
                     volume: "Optional[pd.Series | pd.DataFrame]" = None,
                     start: Optional[pd.Timestamp] = None) -> BacktestResult:
    """Buy-and-hold on the index, for comparison.

    The purchase happens on the first available session. Leaving it to a
    periodic calendar would park the benchmark in cash until the first
    scheduled rebalance -- up to a full year on an annual calendar -- and
    quietly understate the bar the strategy is measured against. After that
    single purchase the weight simply drifts, which is what buy-and-hold
    means.

    `costs` are the frictions the benchmark pays (see `benchmark_costs`);
    left None it is frictionless, as an index is. `volume` feeds the
    market-impact model, as in `run_backtest`.

    `start` moves the purchase to that session instead of the first one.
    A benchmark that bought long before the strategy's warm-up ended would
    have its entry cost cut away with the warm-up when the two are aligned;
    buying on the shared start date keeps it in the record.
    """
    px = bench_prices.dropna().to_frame()
    if start is not None:
        px = px.loc[px.index >= pd.Timestamp(start)]
    if px.empty:
        raise ValueError("Benchmark series is empty.")
    w = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    zero_costs = costs if costs is not None else CostConfig(
        commission_bps=0, slippage_bps=0, cash_rate_pa=0, borrow_rate_pa=0)
    vol = None
    if volume is not None:
        vol = volume.to_frame(px.columns[0]) if isinstance(volume, pd.Series) else volume
        vol = vol.reindex(index=px.index, columns=px.columns)
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

    res = run_backtest(px, w, eng, zero_costs, label=label,
                       rebalance_dates=dates, dividends=div, volume=vol)
    return (_carry_entry_cost(res, engine.initial_capital)
            if costs is not None else res)


def blended_benchmark(prices: pd.DataFrame,
                      weights: Dict[str, float],
                      engine: EngineConfig,
                      label: str = "Benchmark",
                      dividends: Optional[pd.DataFrame] = None,
                      rebalance: str = "A",
                      costs: Optional[CostConfig] = None,
                      volume: Optional[pd.DataFrame] = None,
                      start: Optional[pd.Timestamp] = None) -> BacktestResult:
    """A fixed-weight benchmark across several instruments.

    The blend is run through the same engine as everything else rather than
    being averaged: a 60/40 that is rebalanced is not the same thing as the
    weighted average of its two return series, because the drift between
    rebalances is real. Frictions are zero by default, since a policy
    benchmark is a measuring stick; `costs` (see `benchmark_costs`) charges
    it the strategy's commission, slippage and fee on its entry and on every
    rebalance back to target. `volume` feeds the market-impact model, and
    `start` moves the purchase to that session, as in `benchmark_result`.

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
    if start is not None:
        px = px.loc[px.index >= pd.Timestamp(start)]
        if px.empty:
            raise ValueError("The benchmark's instruments have no history "
                             "from the requested start.")

    total = sum(abs(float(weights[c])) for c in cols) or 1.0
    w = pd.DataFrame(
        {c: float(weights[c]) / total for c in cols},
        index=px.index)

    zero = costs if costs is not None else CostConfig(
        commission_bps=0, slippage_bps=0, cash_rate_pa=0, borrow_rate_pa=0)
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
    vol = (volume.reindex(index=px.index, columns=cols)
           if volume is not None else None)
    res = run_backtest(px, w, eng, zero, label=label,
                       rebalance_dates=dates, dividends=div, volume=vol)
    return (_carry_entry_cost(res, engine.initial_capital)
            if costs is not None else res)


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
