"""Builds the frozen input files the golden tests run on.

Run once, by hand, and commit the output:

    python tests/golden/make_fixtures.py

The tests never call this. They read the committed files in
`tests/golden/fixtures/`, so the inputs cannot drift with a library upgrade
or with what Yahoo Finance returns today. Regenerating the fixtures
invalidates every expected output: only do it deliberately, then re-record
(see tests/golden/README.md).

The prices are synthetic. They are shaped to exercise the engine (trends,
drawdowns, dividends, volume) and carry no information about the real
instruments whose tickers they borrow. The tickers match the repository's
example target-weights and factor files so those can be used unchanged.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")
ROOT = os.path.dirname(os.path.dirname(HERE))

TICKERS = ["XIC.TO", "ZEB.TO", "XEI.TO", "ZLB.TO"]
START, END = "2012-01-02", "2025-07-31"
SEED = 20260149


def build_prices() -> pd.DataFrame:
    """Long-format OHLCV + dividends, one row per date and instrument."""
    rs = np.random.RandomState(SEED)          # frozen stream, unlike Generator
    idx = pd.bdate_range(START, END)
    n = len(idx)

    # A slow market regime (drift and volatility move together), a common
    # factor, and an idiosyncratic part with fat tails.
    regime = np.zeros(n)
    state, i = 0, 0
    while i < n:
        length = int(rs.randint(120, 520)) if state == 0 else int(rs.randint(30, 140))
        regime[i:i + length] = state
        state, i = 1 - state, i + length
    mkt_mu = np.where(regime == 0, 0.00075, -0.00100)
    mkt_sd = np.where(regime == 0, 0.0068, 0.0165)
    market = mkt_mu + mkt_sd * rs.standard_t(5, n) / np.sqrt(5 / 3)

    spec = {            # beta, idio vol, extra drift, start price, yield
        "XIC.TO": (1.00, 0.0030, 0.00000, 21.50, 0.028),
        "ZEB.TO": (1.15, 0.0062, 0.00006, 17.80, 0.041),
        "XEI.TO": (0.90, 0.0045, -0.00003, 19.20, 0.047),
        "ZLB.TO": (0.62, 0.0038, 0.00010, 16.40, 0.024),
    }
    frames = []
    for t in TICKERS:
        beta, idio, drift, p0, yld = spec[t]
        # Each name also carries its own slow trend so rankings rotate.
        period, phase = rs.randint(380, 900), rs.uniform(0, 1)
        own = 0.00028 * np.sin(2 * np.pi * (np.arange(n) / period + phase))
        r = beta * market + drift + own + idio * rs.standard_t(4, n) / np.sqrt(2)
        close = p0 * np.exp(np.cumsum(r))
        gap = rs.normal(0, 0.0025, n)
        open_ = np.empty(n)
        open_[0] = close[0] * (1 - 0.001)
        open_[1:] = close[:-1] * np.exp(gap[1:])
        span = np.abs(rs.normal(0, 0.004, n)) + 0.0008
        high = np.maximum(open_, close) * (1 + span)
        low = np.minimum(open_, close) * (1 - span)
        volume = np.round(np.exp(rs.normal(12.6, 0.55, n)) *
                          (1 + 2.5 * np.abs(r) / 0.01)).astype("int64")
        # Quarterly cash dividend on the 3rd-last session of Mar/Jun/Sep/Dec.
        div = np.zeros(n)
        s = pd.Series(np.arange(n), index=idx)
        for (_y, m), pos in s.groupby([idx.year, idx.month]):
            if m in (3, 6, 9, 12) and len(pos) >= 3:
                k = int(pos.iloc[-3])
                div[k] = round(close[k] * yld / 4, 4)
        frames.append(pd.DataFrame({
            "Date": idx.strftime("%Y-%m-%d"), "Ticker": t,
            "Open": np.round(open_, 4), "High": np.round(high, 4),
            "Low": np.round(low, 4), "Close": np.round(close, 4),
            "Volume": volume, "Dividends": div}))
    out = pd.concat(frames).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return out


def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """A monthly return-stream file: two 'funds' and a benchmark, in percent."""
    close = prices.pivot(index="Date", columns="Ticker", values="Close")
    close.index = pd.to_datetime(close.index)
    m = close.resample("ME").last().pct_change().dropna()
    out = pd.DataFrame({
        "Date": m.index.strftime("%Y-%m-%d"),
        "Fund A": np.round((0.6 * m["ZLB.TO"] + 0.4 * m["ZEB.TO"]) * 100, 4),
        "Fund B": np.round((0.5 * m["XEI.TO"] + 0.5 * m["ZLB.TO"] - 0.0008) * 100, 4),
        "Benchmark": np.round(m["XIC.TO"] * 100, 4),
    })
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    px = build_prices()
    px.to_csv(os.path.join(OUT, "prices_long.csv"), index=False, lineterminator="\n")
    build_returns(px).to_csv(os.path.join(OUT, "returns_monthly.csv"), index=False,
                             lineterminator="\n")
    # The signal and weights files are the repository's own examples, copied
    # so that editing an example later cannot move a golden result.
    for name in ("example_target_weights.csv", "example_fundamental_factor.csv",
                 "example_economic_data.csv"):
        with open(os.path.join(ROOT, name), "rb") as src, \
                open(os.path.join(OUT, name.replace("example_", "")), "wb") as dst:
            dst.write(src.read().replace(b"\r\n", b"\n"))
    print(f"{len(px):,} price rows, {px['Date'].min()} to {px['Date'].max()}")


if __name__ == "__main__":
    main()
