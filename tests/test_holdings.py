"""Fitting an actual account to a target allocation. Synthetic numbers only.

The whole-unit case is checked against an exhaustive search over every
affordable combination of units, which does not share any code with the
solver.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt.nextrade import fit_to_holdings                    # noqa: E402


def _s(d):
    return pd.Series(d, dtype=float)


def _row(fit, a):
    return fit.table.set_index("Instrument").loc[a]


def test_fractional_units_land_exactly_on_target_with_no_costs():
    px = _s({"A": 50, "B": 20, "C": 410})
    tgt = _s({"A": 0.5, "B": 0.3, "C": 0.1})
    fit = fit_to_holdings(tgt, px, _s({"A": 10, "B": 100}), 3000.0, whole_units=False)
    for a in px.index:
        assert abs(_row(fit, a)["Weight after"] - tgt[a]) < 1e-9
    assert abs(_row(fit, "Cash")["Weight after"] - 0.1) < 1e-9
    assert abs(fit.value_after - (10 * 50 + 100 * 20 + 3000)) < 1e-9
    assert fit.distance_after < 1e-9 < fit.distance_before


def test_whole_units_match_an_exhaustive_search():
    px = _s({"A": 47.0, "B": 23.0, "C": 410.0})
    tgt = _s({"A": 0.40, "B": 0.35, "C": 0.20})
    held = _s({"A": 12, "B": 30})
    cash = 1500.0
    fit = fit_to_holdings(tgt, px, held, cash, whole_units=True)
    v0 = float((held * px).sum(skipna=True)) + cash
    T = tgt * v0
    best = (np.inf, None)
    for h in itertools.product(range(0, 60), range(0, 120), range(0, 8)):
        spend = sum(x * px[a] for x, a in zip(h, px.index))
        if spend > v0 + 1e-9:
            continue
        gap = sum(abs(x * px[a] - T[a]) for x, a in zip(h, px.index))
        if gap < best[0] - 1e-9:
            best = (gap, h)
    got = tuple(int(_row(fit, a)["Units after"]) for a in px.index)
    got_gap = sum(abs(x * px[a] - T[a]) for x, a in zip(got, px.index))
    assert abs(got_gap - best[0]) < 1e-6
    assert fit.cash_after >= -1e-9
    assert all(float(_row(fit, a)["Units after"]).is_integer() for a in px.index)


def test_costs_come_out_of_cash_and_cash_never_goes_negative():
    px = _s({"A": 33.0, "B": 71.0, "C": 12.0})
    tgt = _s({"A": 0.4, "B": 0.4, "C": 0.2})
    rate = _s({"A": 0.003, "B": 0.003, "C": 0.003})
    fit = fit_to_holdings(tgt, px, _s({}), 10_000.0, cost_rate=rate, whole_units=True)
    traded = float((fit.table.dropna(subset=["Units to trade"])["Amount"].abs()).sum())
    assert abs(fit.cost - 0.003 * traded) < 1e-6
    assert fit.cash_after >= -1e-9
    assert abs(fit.value_after - (10_000.0 - fit.cost)) < 1e-6
    buys = fit.table[fit.table["Action"].str.startswith("Buy")]["Amount"].sum()
    assert abs(10_000.0 - buys - fit.cost - fit.cash_after) < 1e-6


def test_a_position_the_target_no_longer_wants_is_sold_out():
    px = _s({"A": 50, "B": 20})
    fit = fit_to_holdings(_s({"A": 1.0, "B": 0.0}), px, _s({"A": 10, "B": 40}), 0.0,
                          whole_units=True, min_trade_weight=0.005)
    assert _row(fit, "B")["Action"] == "Sell (exit)"
    assert _row(fit, "B")["Units after"] == 0
    assert _row(fit, "A")["Units after"] == 10 + 16


def test_a_portfolio_already_on_target_makes_no_trades():
    px = _s({"A": 50, "B": 20})
    fit = fit_to_holdings(_s({"A": 0.5, "B": 0.5}), px, _s({"A": 10, "B": 25}), 0.0,
                          whole_units=True, min_trade_weight=0.005)
    assert not fit.has_trades
    assert fit.distance_after == fit.distance_before


def test_trades_under_the_minimum_size_are_skipped_but_big_ones_are_made():
    px = _s({"A": 10.0, "B": 10.0})
    tgt = _s({"A": 0.5, "B": 0.5})
    # 0.2 point off target: below the 0.5% minimum, so no trade.
    near = fit_to_holdings(tgt, px, _s({"A": 502, "B": 498}), 0.0,
                           whole_units=True, min_trade_weight=0.005)
    assert not near.has_trades
    far = fit_to_holdings(tgt, px, _s({"A": 600, "B": 400}), 0.0,
                          whole_units=True, min_trade_weight=0.005)
    assert far.has_trades and _row(far, "A")["Units after"] == 500


def test_an_unaffordable_target_is_approached_without_borrowing():
    px = _s({"A": 800.0, "B": 900.0})
    fit = fit_to_holdings(_s({"A": 0.5, "B": 0.5}), px, _s({}), 1000.0, whole_units=True)
    assert fit.cash_after >= -1e-9
    assert fit.distance_after < fit.distance_before
    assert fit.n_buys == 1                     # one 800 share is all that fits
