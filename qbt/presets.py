"""Preset universes.

Starting points, not recommendations. Each is a plain list of symbols that
drops into the ticker box and stays fully editable afterwards.

Two warnings that apply to every list here, and to any universe assembled
today from instruments that exist today:

- **Survivorship.** These are funds that survived. Anything wound up or
  merged along the way is absent, which flatters any backtest run over a
  long window.
- **Inception dates.** A backtest can only start once every member has
  history. Several of these ETFs launched in the 2010s, so asking for a
  2005 start silently shortens the usable period or drops names. The Data
  tab reports the first usable date per instrument.

Editing this file is the intended way to add your own: append an entry to
UNIVERSES and it appears in the interface at the next reload.
"""
from __future__ import annotations

from typing import Dict, List

UNIVERSES: Dict[str, Dict[str, object]] = {
    "Canadian ETFs \u2014 broad": {
        "tickers": ["XIC.TO", "ZEB.TO", "XEI.TO", "ZLB.TO", "XDV.TO", "XCG.TO"],
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "Canadian equity: broad market, banks, dividend, low volatility, growth.",
    },
    "Canadian asset classes": {
        "tickers": ["XIC.TO", "XBB.TO", "XSB.TO", "XRE.TO", "CGL.TO", "XEF.TO", "XUU.TO"],
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "Multi-asset in CAD: equity, aggregate and short bonds, REITs, "
                "gold, developed and U.S. equity.",
    },
    "Canadian sectors": {
        "tickers": ["XFN.TO", "XEG.TO", "XMA.TO", "XIT.TO", "XST.TO", "XUT.TO", "XRE.TO"],
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "TSX sectors: financials, energy, materials, technology, "
                "staples, utilities, real estate.",
    },
    "U.S. sectors (SPDR)": {
        "tickers": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB"],
        "benchmark": "SPY",
        "cash": "BIL",
        "note": "The classic sector-rotation universe. Long, clean history.",
    },
    "Global asset classes": {
        "tickers": ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "VNQ", "DBC"],
        "benchmark": "SPY",
        "cash": "BIL",
        "note": "The standard multi-asset momentum universe: U.S., developed, "
                "emerging, intermediate and long treasuries, gold, REITs, "
                "commodities.",
    },
    "U.S. factors": {
        "tickers": ["MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPY"],
        "benchmark": "SPY",
        "cash": "BIL",
        "note": "Momentum, quality, minimum volatility, value, size. Most "
                "launched in 2013.",
    },
    "Sixty-forty building blocks": {
        "tickers": ["XIC.TO", "XUU.TO", "XEF.TO", "XBB.TO", "XSB.TO"],
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "Enough to rebuild a balanced mandate and test deviations "
                "from it.",
    },
}


def names() -> List[str]:
    return list(UNIVERSES.keys())


def get(name: str) -> Dict[str, object]:
    return UNIVERSES.get(name, {})
