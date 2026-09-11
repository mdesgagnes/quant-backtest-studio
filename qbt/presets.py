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
        "classes": {
            "XIC.TO": "Equities",
            "ZEB.TO": "Equities",
            "XEI.TO": "Equities",
            "ZLB.TO": "Equities",
            "XDV.TO": "Equities",
            "XCG.TO": "Equities",
        },
        "tickers": ["XIC.TO", "ZEB.TO", "XEI.TO", "ZLB.TO", "XDV.TO", "XCG.TO"],
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "Canadian equity: broad market, banks, dividend, low volatility, growth.",
    },
    "Canadian asset classes": {
        "classes": {
            "XIC.TO": "Equities",
            "XEF.TO": "Equities",
            "XUU.TO": "Equities",
            "XBB.TO": "Fixed income",
            "XSB.TO": "Fixed income",
            "XRE.TO": "Real assets",
            "CGL.TO": "Real assets",
        },
        "tickers": ["XIC.TO", "XBB.TO", "XSB.TO", "XRE.TO", "CGL.TO", "XEF.TO", "XUU.TO"],
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "Multi-asset in CAD: equity, aggregate and short bonds, REITs, "
                "gold, developed and U.S. equity.",
    },
    "Canadian sectors": {
        "classes": {
            "XFN.TO": "Equities",
            "XEG.TO": "Equities",
            "XMA.TO": "Equities",
            "XIT.TO": "Equities",
            "XST.TO": "Equities",
            "XUT.TO": "Equities",
            "XRE.TO": "Real assets",
        },
        "tickers": ["XFN.TO", "XEG.TO", "XMA.TO", "XIT.TO", "XST.TO", "XUT.TO", "XRE.TO"],
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "TSX sectors: financials, energy, materials, technology, "
                "staples, utilities, real estate.",
    },
    "U.S. sectors (SPDR)": {
        "classes": {
            "XLK": "Equities",
            "XLF": "Equities",
            "XLV": "Equities",
            "XLE": "Equities",
            "XLI": "Equities",
            "XLY": "Equities",
            "XLP": "Equities",
            "XLU": "Equities",
            "XLB": "Equities",
        },
        "tickers": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB"],
        "benchmark": "SPY",
        "cash": "BIL",
        "note": "The classic sector-rotation universe. Long, clean history.",
    },
    "Global asset classes": {
        "classes": {
            "SPY": "Equities",
            "EFA": "Equities",
            "EEM": "Equities",
            "IEF": "Fixed income",
            "TLT": "Fixed income",
            "GLD": "Real assets",
            "VNQ": "Real assets",
            "DBC": "Real assets",
        },
        "tickers": ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "VNQ", "DBC"],
        "benchmark": "SPY",
        "cash": "BIL",
        "note": "The standard multi-asset momentum universe: U.S., developed, "
                "emerging, intermediate and long treasuries, gold, REITs, "
                "commodities.",
    },
    "U.S. factors": {
        "classes": {
            "MTUM": "Equities",
            "QUAL": "Equities",
            "USMV": "Equities",
            "VLUE": "Equities",
            "SIZE": "Equities",
            "SPY": "Equities",
        },
        "tickers": ["MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPY"],
        "benchmark": "SPY",
        "cash": "BIL",
        "note": "Momentum, quality, minimum volatility, value, size. Most "
                "launched in 2013.",
    },
    "Canadian equity factor mix": {
        "tickers": ["XIC.TO", "HCAL.TO", "WXM.TO", "XDV.TO", "XCV.TO",
                    "XCG.TO", "XDIV.TO", "XEI.TO", "HXT.TO", "ZLB.TO"],
        "classes": {
            "XIC.TO": "Equities", "HCAL.TO": "Equities", "WXM.TO": "Equities",
            "XDV.TO": "Equities", "XCV.TO": "Equities", "XCG.TO": "Equities",
            "XDIV.TO": "Equities", "XEI.TO": "Equities", "HXT.TO": "Equities",
            "ZLB.TO": "Equities",
        },
        "benchmark": "XIC.TO",
        "cash": "PSA.TO",
        "note": "Canadian equity styles for cross-sectional rotation: broad "
                "market, banks, multifactor, dividend, value, growth, income "
                "and low volatility. Three things to know before reading a "
                "result. HCAL is leveraged, so it will win most momentum "
                "rankings in a rising market and lose them badly otherwise. "
                "HXT is a total-return swap structure that pays no "
                "distributions, so it is unaffected by the dividend setting "
                "while its peers are not. And several of these launched in "
                "the mid-2010s, so an early start date quietly drops them.",
    },
    "Global multi-asset ETF universe": {
        "tickers": ["SPY", "PDBC", "IEFA", "IEF", "IEMG", "VNQ", "QQQ", "XLE",
                    "GLD", "TLT", "EWJ", "TIP", "LQD", "VTV", "VGK", "UUP",
                    "VBR", "IWM", "MTUM", "HYG", "EMB", "BNDX", "EFV", "SCZ",
                    "XLK", "XLV", "BWX", "SHY", "REET", "DBA", "XLB", "EWC"],
        "classes": {
            "SPY": "Equities", "QQQ": "Equities", "IWM": "Equities",
            "VTV": "Equities", "VBR": "Equities", "MTUM": "Equities",
            "IEFA": "Equities", "IEMG": "Equities", "EWJ": "Equities",
            "EWC": "Equities", "VGK": "Equities", "EFV": "Equities",
            "SCZ": "Equities", "XLE": "Equities", "XLK": "Equities",
            "XLV": "Equities", "XLB": "Equities",
            "IEF": "Fixed income", "TLT": "Fixed income", "SHY": "Fixed income",
            "TIP": "Fixed income", "LQD": "Fixed income", "HYG": "Fixed income",
            "EMB": "Fixed income", "BNDX": "Fixed income", "BWX": "Fixed income",
            "VNQ": "Real assets", "REET": "Real assets", "GLD": "Real assets",
            "PDBC": "Real assets", "DBA": "Real assets",
            "UUP": "Currency",
        },
        "benchmark": "SPY",
        "cash": "SHY",
        "note": "Thirty-two ETFs spanning US and international equities, "
                "sectors, factors, sovereign and credit fixed income, "
                "commodities and gold, real estate, and the dollar -- built "
                "for cross-sectional and trend strategies with real "
                "asset-class breadth to rotate across, rather than variations "
                "on one market. SHY is offered as the cash leg; nothing here "
                "requires it.",
    },
    "HIDE target asset classes": {
        "tickers": ["SCHR", "VNQ", "BCI"],
        "classes": {
            "SCHR": "Fixed income",
            "VNQ": "Real assets",
            "BCI": "Real assets",
        },
        "benchmark": "SCHR",
        "cash": "BIL",
        "note": "The three asset classes behind Alpha Architect's HIDE: "
                "intermediate Treasuries, REITs, commodities, with BIL as "
                "the cash leg. Pair with the Trend-Gated Target Weights "
                "model at 50/25/25; its momentum test measures each class "
                "against BIL. BCI launched in 2017, so an earlier start "
                "drops it.",
    },
    "Sixty-forty building blocks": {
        "classes": {
            "XIC.TO": "Equities",
            "XUU.TO": "Equities",
            "XEF.TO": "Equities",
            "XBB.TO": "Fixed income",
            "XSB.TO": "Fixed income",
        },
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


def classes_for(name: str, tickers: List[str]) -> Dict[str, str]:
    """Asset class per instrument, for a preset or a hand-typed universe.

    Anything the preset does not name -- including every ticker in a custom
    universe -- comes back as "Unclassified" rather than being guessed at
    from the symbol. A wrong guess would silently misallocate a budget,
    which is worse than asking.
    """
    tagged = dict(UNIVERSES.get(name, {}).get("classes", {}) or {})
    return {t: tagged.get(t, "Unclassified") for t in tickers}
