"""Multi-currency checks. Synthetic data only, no network.

The engine keeps one cash balance per currency and pays the exchange
spread only on what is actually converted. Each test pins one consequence
of that to a number worked out by hand or by an independent calculation.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt import fx as FX                                    # noqa: E402
from qbt.config import CostConfig, EngineConfig             # noqa: E402
from qbt.data import MarketData                             # noqa: E402
from qbt.engine import run_backtest, trim_warmup, window    # noqa: E402

ZERO = CostConfig(commission_bps=0, slippage_bps=0, fx_spread_bps=0)


def _world(n=700, seed=3, dividends=True):
    """A CAD instrument and two USD ones, with a wandering USD/CAD rate.
    Returns local-currency prices, the rate, the FxBook, base-currency
    prices and base-currency dividends."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    r = rng.normal(0.0004, 0.011, (n, 3)) + rng.normal(0, 0.005, (n, 1))
    local = pd.DataFrame(40 * np.exp(np.cumsum(r, 0)), idx, ["CAD1", "US1", "US2"])
    rate = pd.Series(1.30 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), idx)
    book = FX.FxBook(base="CAD", ccy={"CAD1": "CAD", "US1": "USD", "US2": "USD"},
                     scale={}, source={}, rates=pd.DataFrame({"USD": rate}))
    px = local.copy()
    for c in ("US1", "US2"):
        px[c] = local[c] * rate
    div = pd.DataFrame(0.0, idx, local.columns)
    if dividends:
        q = idx[::63][1:]
        for c in local.columns:
            div.loc[q, c] = local.loc[q, c] * 0.006
        for c in ("US1", "US2"):
            div[c] = div[c] * rate
    return local, rate, book, px, div


def _hold(px, w):
    """Constant weights from the first session."""
    return pd.DataFrame([w] * len(px), index=px.index, columns=px.columns)


# ----------------------------------------------------------------------
def test_conversion_is_price_times_daily_rate():
    local, rate, book, px, div = _world()
    m = MarketData(close=local, open=local * 0.99, dividends=local * 0.001,
                   volume=local * 10, adjusted=False)
    out = book.convert_market(m)
    assert np.allclose(out.close["US1"], local["US1"] * rate)
    assert np.allclose(out.open["US2"], local["US2"] * 0.99 * rate)
    assert np.allclose(out.dividends["US1"], local["US1"] * 0.001 * rate)
    assert out.close["CAD1"].equals(local["CAD1"])           # base: untouched, bit for bit
    assert out.volume.equals(m.volume)                       # shares are not money


def test_pence_are_converted_to_pounds():
    book = FX.FxBook.resolve(["VOD.L", "BP"], "CAD", overrides={"BP": "GBp"})
    assert book.ccy == {"VOD.L": "GBP", "BP": "GBP"}
    assert book.scale == {"VOD.L": 0.01, "BP": 0.01}
    idx = pd.bdate_range("2020-01-01", periods=5)
    book.rates = pd.DataFrame({"GBP": 1.70}, index=idx)
    out = book.convert_frame(pd.DataFrame({"VOD.L": 100.0, "BP": 250.0}, index=idx))
    assert np.allclose(out["VOD.L"], 100 * 0.01 * 1.70)      # 100p = 1 pound = 1.70 CAD
    assert np.allclose(out["BP"], 250 * 0.01 * 1.70)


def test_a_base_currency_quoted_in_pence_is_scaled_without_a_rate():
    book = FX.FxBook.resolve(["VOD.L"], "GBP")
    idx = pd.bdate_range("2020-01-01", periods=3)
    out = book.convert_frame(pd.DataFrame({"VOD.L": 150.0}, index=idx))
    assert np.allclose(out["VOD.L"], 1.5)
    assert book.foreign_currencies(["VOD.L"]) == []


def test_currency_decision_order_and_guess():
    det = FX.detect_currencies(["XIC.TO", "SPY", "ODD"],
                               lookup=lambda t: {"SPY": "USD"}.get(t))
    assert det["SPY"] == ("USD", "Yahoo Finance")
    assert det["XIC.TO"] == ("CAD", "guessed from ticker")
    book = FX.FxBook.resolve(["XIC.TO", "SPY", "ODD"], "CAD",
                             overrides={"ODD": "EUR"}, detected=det)
    assert book.ccy == {"XIC.TO": "CAD", "SPY": "USD", "ODD": "EUR"}
    assert book.source["ODD"] == "set by you"
    assert FX.guess_currency("VOD.L") == "GBp"
    assert FX.guess_currency("^GSPTSE") == "CAD"
    assert FX.guess_currency("BTC-CAD") == "CAD"


def test_rates_direct_inverse_and_cross_through_usd():
    idx = pd.bdate_range("2021-01-01", periods=30)

    def dl(names, start, end):
        feed = {"USDCAD=X": pd.Series(1.25, idx),
                "EURUSD=X": pd.Series(1.10, idx),
                "JPYUSD=X": pd.Series(np.nan, idx),
                "USDJPY=X": pd.Series(110.0, idx)}
        return pd.DataFrame({n: feed[n] for n in names if n in feed})

    rates, notes = FX.load_rates(["USD", "EUR", "JPY", "XXX"], "CAD", "2021-01-04", None, dl)
    assert np.allclose(rates["USD"], 1.25)
    assert np.allclose(rates["EUR"], 1.10 * 1.25)            # EUR -> USD -> CAD
    assert np.allclose(rates["JPY"], (1 / 110.0) * 1.25)     # inverse leg, then USD -> CAD
    assert "XXX" not in rates.columns
    assert any("XXX" in n for n in notes) and any("EUR" in n for n in notes)


def test_a_bad_print_in_the_rate_is_flagged():
    idx = pd.bdate_range("2021-01-01", periods=30)
    s = pd.Series(1.25, idx)
    s.iloc[15] = 1.60
    _, notes = FX.load_rates(["USD"], "CAD", None, None,
                             lambda n, a, b: pd.DataFrame({"USDCAD=X": s}))
    assert any("one-day move" in n for n in notes)


# ----------------------------------------------------------------------
def test_zero_cost_multicurrency_equals_converted_single_currency_run():
    """With no spread, no costs and no dividends, keeping per-currency cash
    must change nothing: the only difference between the two models is the
    cost of moving between currencies."""
    local, rate, book, px, _ = _world(dividends=False)
    w = _hold(px, {"CAD1": 0.4, "US1": 0.35, "US2": 0.25})
    w.iloc[:30] = 0.0
    eng = EngineConfig(min_trade_weight=0.0, rebalance="M", execution_lag=1)
    one = run_backtest(px, w, eng, ZERO)
    multi = run_backtest(px, w, eng, ZERO, fx=book)
    assert np.allclose(one.equity, multi.equity, rtol=1e-10, atol=1e-6)
    assert np.allclose(multi.fx_costs, 0.0)


def test_entry_into_a_foreign_asset_pays_the_spread_once():
    local, rate, book, px, _ = _world(dividends=False)
    s = 0.0100
    cost = CostConfig(commission_bps=0, slippage_bps=0, fx_spread_bps=s * 1e4)
    w = _hold(px, {"CAD1": 0.0, "US1": 1.0, "US2": 0.0})
    w.iloc[:10] = 0.0
    eng = EngineConfig(min_trade_weight=0.0, rebalance="A", execution_lag=0,
                       initial_capital=100_000.0, trim_warmup=False)
    res = run_backtest(px, w, eng, cost, fx=book)
    day = res.trades["Date"].iloc[0]
    # 100% in US1 needs V(1+s) = capital of Canadian dollars
    assert abs(res.equity[day] - 100_000.0 / (1 + s)) < 1e-6
    assert abs(res.fx_costs[day] - (1 - 1 / (1 + s))) < 1e-9
    assert abs(res.costs[day] - res.fx_costs[day]) < 1e-12
    # converting the USD back at the exit pays it again, on the value then
    out = _hold(px, {"CAD1": 0.0, "US1": 0.0, "US2": 0.0})
    out.iloc[:10] = 0.0
    out.iloc[10:20] = [0, 1, 0]
    res2 = run_backtest(px, out, EngineConfig(min_trade_weight=0.0, rebalance="D",
                                              execution_lag=0, trim_warmup=False), cost, fx=book)
    sells = res2.cash_flows[res2.cash_flows["Type"] == "FX conversion"]
    assert len(sells) == 2 and sells["Units"].iloc[0] < 0 < sells["Units"].iloc[1]


def test_rotating_between_two_us_assets_costs_no_conversion():
    """Selling one USD instrument to buy another settles inside USD."""
    local, rate, book, px, _ = _world(dividends=False)
    cost = CostConfig(commission_bps=0, slippage_bps=0, fx_spread_bps=200)
    w = _hold(px, {"CAD1": 0.0, "US1": 1.0, "US2": 0.0})
    w.iloc[:10] = 0.0
    w.iloc[300:] = [0.0, 0.0, 1.0]
    eng = EngineConfig(min_trade_weight=0.0, rebalance="D", execution_lag=0, trim_warmup=False)
    res = run_backtest(px, w, eng, cost, fx=book)
    rot = res.trades[res.trades["Date"] == px.index[300]]
    assert set(rot["Instrument"]) == {"US1", "US2"}
    assert res.fx_costs.iloc[300] < 1e-9                    # no conversion on the rotation day
    assert res.fx_costs.iloc[:299].sum() > 0.01             # the entry did pay
    # the same rotation, with every trade converted on its own, would not be free
    assert res.foreign_cash.abs().to_numpy().max() < 1e-3


def test_hold_keeps_proceeds_in_foreign_currency_and_reuses_them():
    local, rate, book, px, _ = _world(dividends=False)
    cost = CostConfig(commission_bps=0, slippage_bps=0, fx_spread_bps=150)
    w = _hold(px, {"CAD1": 0.0, "US1": 1.0, "US2": 0.0})
    w.iloc[:10] = 0.0
    w.iloc[200:300] = 0.0                                   # out to cash for a while
    w.iloc[300:] = [0.0, 0.0, 1.0]                          # then back into a US asset
    eng = EngineConfig(min_trade_weight=0.0, rebalance="D", execution_lag=0,
                       trim_warmup=False, fx_cash="hold")
    res = run_backtest(px, w, eng, cost, fx=book)
    # while "in cash" the money is still USD: no spread, and it floats with the rate
    mid = res.foreign_cash["USD"].iloc[250]
    assert mid > 0.9 * res.equity.iloc[250]
    assert res.fx_costs.iloc[200:300].sum() < 1e-9
    assert res.fx_costs.iloc[300] < 1e-9                    # re-entry used the USD, no conversion
    ratio = (res.foreign_cash["USD"].iloc[250] / res.foreign_cash["USD"].iloc[210]
             / (rate.iloc[250] / rate.iloc[210]))
    assert abs(ratio - 1.0) < 1e-9
    # under net settlement the same exit is converted home and pays for it
    net = run_backtest(px, w, EngineConfig(min_trade_weight=0.0, rebalance="D",
                                           execution_lag=0, trim_warmup=False,
                                           fx_cash="net"), cost, fx=book)
    assert net.fx_costs.iloc[200:300].sum() > 0.01
    assert net.foreign_cash["USD"].iloc[250] == 0.0


def test_dividends_arrive_in_the_instruments_currency():
    local, rate, book, px, div = _world()
    cost = CostConfig(commission_bps=0, slippage_bps=0, fx_spread_bps=100)
    w = _hold(px, {"CAD1": 0.0, "US1": 1.0, "US2": 0.0})
    w.iloc[:10] = 0.0
    eng = EngineConfig(min_trade_weight=0.0, rebalance="A", execution_lag=0,
                       trim_warmup=False, fx_cash="hold")
    res = run_backtest(px, w, eng, cost, dividends=div, fx=book)
    dv = res.cash_flows[res.cash_flows["Type"] == "Dividend"]
    assert len(dv) >= 5
    d0 = pd.Timestamp(dv["Date"].iloc[0])
    units = float(dv["Units"].iloc[0])
    per_unit_local = float(local.loc[d0, "US1"]) * 0.006
    # the balance holds the payment in USD, not its Canadian equivalent
    assert abs(res.foreign_cash["USD"][d0] / rate[d0] - units * per_unit_local) < 1e-6
    assert res.foreign_cash["USD"][d0] > 0


def test_foreign_cash_interest_and_revaluation_reconcile():
    local, rate, book, px, div = _world()
    cost = CostConfig(commission_bps=2, slippage_bps=3, fx_spread_bps=40,
                      foreign_cash_rate_pa=0.03, cash_rate_pa=0.02)
    for fx_cash in ("net", "hold"):
        w = _hold(px, {"CAD1": 0.3, "US1": 0.3, "US2": 0.2})
        w.iloc[:30] = 0.0
        w.iloc[250:400] = [0.2, 0.0, 0.0]
        eng = EngineConfig(min_trade_weight=0.0, rebalance="M", execution_lag=1,
                           fx_cash=fx_cash, initial_capital=250_000.0)
        res = run_backtest(px, w, eng, cost, dividends=div, fx=book)
        _assert_sound(res)


def _assert_sound(res, tol=1e-9):
    """The ledger adds up every day, base cash never goes short, and the
    exchange spread is a part of the trading cost, not an addition to it."""
    daily = res.contributions.sum(axis=1)
    assert np.abs(daily - res.returns).max() < tol
    held = (res.weights.mul(res.equity, axis=0)).sum(axis=1)
    fcash = res.foreign_cash.sum(axis=1) if res.foreign_cash is not None else 0.0
    base_cash = res.equity - held - fcash
    assert base_cash.min() > -1e-4 * res.equity.iloc[0] * 1e-3
    if res.fx_costs is not None:
        assert (res.fx_costs <= res.costs + 1e-12).all()
        assert (res.fx_costs >= -1e-15).all()
    assert np.isfinite(res.equity).all()


def test_every_engine_option_stays_sound_with_two_currencies():
    local, rate, book, px, div = _world(n=500)
    # Opening prices and dividends the way the app converts them for a run
    # that trades at the open: at the previous close's exchange rate.
    op_local = local.shift(1).bfill() * 1.001
    div_local = pd.DataFrame(0.0, local.index, local.columns)
    for c in local.columns:
        div_local.loc[local.index[::63][1:], c] = local.loc[local.index[::63][1:], c] * 0.006
    conv_open = book.convert_market(MarketData(close=local, open=op_local,
                                               dividends=div_local, adjusted=False),
                                    at_open=True)
    op = conv_open.open
    vol = pd.DataFrame(50_000.0, px.index, px.columns)
    cases = [
        dict(eng=dict(whole_shares=True, initial_capital=40_000.0), cost=dict()),
        dict(eng=dict(execute_at_open=True), cost=dict(), open=True),
        dict(eng=dict(cash_sweep="M", cash_sweep_threshold=0.02), cost=dict()),
        dict(eng=dict(rebalance="Q", min_trade_weight=0.0, whole_shares=True,
                      initial_capital=60_000.0), cost=dict(management_fee_pa=0.02)),
        dict(eng=dict(max_leverage=1.2), cost=dict(borrow_rate_pa=0.03)),
        dict(eng=dict(), cost=dict(impact_model="sqrt_vol", management_fee_pa=0.01)),
    ]
    for fx_cash in ("net", "hold"):
        for c in cases:
            w = _hold(px, {"CAD1": 0.2, "US1": 0.4, "US2": 0.38})
            w.iloc[:30] = 0.0
            w.iloc[200:260] = [0.5, 0.0, 0.0]
            eng = EngineConfig(fx_cash=fx_cash, **{"min_trade_weight": 0.005, **c["eng"]})
            cost = CostConfig(fx_spread_bps=60, foreign_cash_rate_pa=0.01, **c["cost"])
            res = run_backtest(px, w, eng, cost, open_prices=op if c.get("open") else None,
                               dividends=conv_open.dividends if c.get("open") else div,
                               volume=vol, fx=book)
            _assert_sound(res)
            assert res.fx_costs.sum() > 0


def test_the_extra_state_survives_trimming_and_windowing():
    local, rate, book, px, div = _world()
    w = _hold(px, {"CAD1": 0.3, "US1": 0.4, "US2": 0.3})
    w.iloc[:120] = 0.0
    res = run_backtest(px, w, EngineConfig(min_trade_weight=0.0, rebalance="M",
                                           fx_cash="hold"), CostConfig(), dividends=div, fx=book)
    t = trim_warmup(res)
    assert t.foreign_cash is not None and len(t.foreign_cash) == len(t.equity)
    assert t.fx_costs is not None and len(t.fx_costs) == len(t.equity)
    cut = window(t, t.equity.index[50], t.equity.index[300])
    assert len(cut.foreign_cash) == len(cut.equity) == len(cut.fx_costs)
    assert {"Currency", "Local Price", "FX Rate"} <= set(res.trades.columns)
    us = res.trades[res.trades["Instrument"] == "US1"]
    assert (us["Currency"] == "USD").all()
    assert np.allclose(us["Price"], us["Local Price"] * us["FX Rate"])


def test_a_book_with_only_base_currency_instruments_is_a_single_currency_run():
    local, rate, book, px, div = _world()
    cols = ["CAD1"]
    w = _hold(px[cols], {"CAD1": 1.0})
    w.iloc[:30] = 0.0
    eng = EngineConfig(min_trade_weight=0.0, rebalance="M")
    a = run_backtest(px[cols], w, eng, CostConfig(), dividends=div[cols])
    b = run_backtest(px[cols], w, eng, CostConfig(), dividends=div[cols], fx=book)
    pd.testing.assert_series_equal(a.equity, b.equity, check_exact=True)
    assert b.foreign_cash is None and "Currency" not in b.trades.columns


def test_robustness_reruns_see_the_same_currencies():
    from qbt import robustness as R
    local, rate, book, px, div = _world()
    w = _hold(px, {"CAD1": 0.3, "US1": 0.4, "US2": 0.3})
    w.iloc[:60] = 0.0
    eng = EngineConfig(min_trade_weight=0.0, rebalance="M", fx_cash="net")
    cost = CostConfig(fx_spread_bps=150)
    head = trim_warmup(run_backtest(px, w, eng, cost, dividends=div, fx=book),
                       None, eng.initial_capital)
    again = R._run(px, w, eng, cost, dividends=div, fx=book,
                   start=head.equity.index[0])
    assert np.allclose(head.equity, again.equity)
    plain = R._run(px, w, eng, cost, dividends=div, start=head.equity.index[0])
    assert not np.allclose(head.equity, plain.equity)       # leaving fx out is a different run
    # the cost-sensitivity grid keeps the exchange spread it is not varying
    rows = R.cost_sensitivity(px, None, {}, eng, cost, [0.0, 50.0], weights=w,
                              dividends=div, fx=book)
    assert len(rows) == 2


def test_config_round_trips_and_old_files_still_load():
    from qbt.config import RunConfig
    cfg = RunConfig.from_yaml("data:\n  tickers: [XIC.TO]\nengine:\n  rebalance: M\n")
    assert cfg.data.base_currency == "" and cfg.engine.fx_cash == "net"
    assert cfg.validate() == []
    cfg.data.base_currency = "CAD"
    cfg.data.currencies = {"VOD.L": "GBp"}
    cfg.costs.fx_spread_bps = 10.0
    again = RunConfig.from_yaml(cfg.to_yaml())
    assert again.data.base_currency == "CAD" and again.data.currencies == {"VOD.L": "GBp"}
    assert again.costs.fx_spread_bps == 10.0
    bad = RunConfig.from_dict({"data": {"base_currency": "CANADA", "currencies": {"X": "US"}},
                               "engine": {"fx_cash": "sometimes"}, "costs": {"fx_spread_bps": -1}})
    msgs = " ".join(bad.validate())
    assert "three-letter" in msgs and "fx_cash" in msgs and "spread" in msgs


def test_a_cash_proxy_can_be_left_in_its_own_currency():
    local, rate, book, px, _ = _world()
    m = MarketData(close=local, adjusted=True)
    out = book.convert_market(m, skip=["US2"])
    assert out.close["US2"].equals(local["US2"])                 # left alone
    assert np.allclose(out.close["US1"], local["US1"] * rate)    # the rest converted


def _open_run(rates_override=None, n=400):
    """A run that trades at the open, on data converted the way the app does."""
    local, rate, book, px, _ = _world(n=n, dividends=False)
    if rates_override is not None:
        book.rates = pd.DataFrame({"USD": rates_override(rate)})
    rng = np.random.default_rng(11)
    op_local = local.shift(1).bfill() * np.exp(rng.normal(0, 0.004, local.shape))
    conv = book.convert_market(MarketData(close=local, open=op_local, adjusted=True),
                               at_open=True)
    w = _hold(px, {"CAD1": 0.3, "US1": 0.4, "US2": 0.3})
    w.iloc[:30] = 0.0
    w.iloc[200:] = [0.1, 0.1, 0.8]
    eng = EngineConfig(min_trade_weight=0.0, rebalance="M", execution_lag=1,
                       execute_at_open=True, trim_warmup=False,
                       initial_capital=100_000.0)
    res = run_backtest(conv.close, w, eng, CostConfig(commission_bps=2, slippage_bps=3,
                                                      fx_spread_bps=50),
                       open_prices=conv.open, fx=book)
    return local, rate, book, conv, res


def test_opening_prices_and_dividends_use_the_previous_close_rate():
    local, rate, book, px, _ = _world()
    m = MarketData(close=local, open=local * 0.99, dividends=local * 0.001,
                   high=local * 1.01, adjusted=False)
    out = book.convert_market(m, at_open=True)
    prev = rate.shift(1)
    prev.iloc[0] = rate.iloc[0]                         # nothing earlier to use
    assert np.allclose(out.open["US1"], local["US1"] * 0.99 * prev)
    assert np.allclose(out.dividends["US2"], local["US2"] * 0.001 * prev)
    assert np.allclose(out.close["US1"], local["US1"] * rate)       # a close is a close
    assert np.allclose(out.high["US1"], local["US1"] * 1.01 * rate)
    assert out.open["CAD1"].equals(local["CAD1"] * 0.99)
    # and without trading at the open, nothing changes
    same = book.convert_market(m)
    assert np.allclose(same.open["US1"], local["US1"] * 0.99 * rate)


def test_a_fill_at_the_open_cannot_depend_on_that_days_closing_rate():
    """Change only the closing exchange rate of one session. Every fill made
    up to and including the open of that session must be identical: it
    cannot have been influenced by a rate that did not exist yet."""
    _, rate, _, _, base = _open_run()
    day = base.trades["Date"].iloc[len(base.trades) // 2]
    k = rate.index.get_loc(day)

    def shock(r):
        r = r.copy()
        r.iloc[k] *= 1.04                                # a 4% move, on that close only
        return r

    _, _, _, _, bumped = _open_run(shock)
    a = base.trades[base.trades["Date"] <= day].reset_index(drop=True)
    b = bumped.trades[bumped.trades["Date"] <= day].reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 0
    for col in ("Instrument", "Shares Before", "Shares After", "Change", "Price",
                "Notional", "FX Rate"):
        assert a[col].equals(b[col]), col
    assert (a["Date"] == day).any()                      # it did trade that day
    # ... while the close of that session, and what the book is worth, did move
    assert abs(bumped.equity.iloc[k] - base.equity.iloc[k]) > 1e-6
    assert (base.equity.iloc[:k] == bumped.equity.iloc[:k]).all()


def test_open_fills_convert_at_the_previous_close_and_the_day_earns_the_rest():
    local, rate, _, _, res = _open_run()
    us = res.trades[res.trades["Instrument"] == "US1"]
    d = us["Date"].iloc[1]
    k = rate.index.get_loc(d)
    row = us[us["Date"] == d].iloc[0]
    assert abs(row["FX Rate"] - rate.iloc[k - 1]) < 1e-12           # yesterday's close
    assert abs(row["Price"] - row["Local Price"] * rate.iloc[k - 1]) < 1e-9
    # the units still held at the close are worth today's close rate: the
    # position earned the move from the conversion rate to the close
    held = res.shares["US1"].iloc[k]
    marked = res.weights["US1"].iloc[k] * res.equity.iloc[k]
    assert abs(marked - held * local["US1"].iloc[k] * rate.iloc[k]) < 1e-6 * marked
    _assert_sound(res)
