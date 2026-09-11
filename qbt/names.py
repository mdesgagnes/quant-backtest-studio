"""Security names.

A ticker on its own is fine once you know the universe by heart; on a
screen of thirty tickers it stops being obvious which is which. This is a
static lookup rather than a live one on purpose: a per-ticker network call
for a name is slow, and yfinance's `.info` endpoint -- the only free source
for it -- is unreliable enough that several other bugs in this app trace
back to trusting it. A name that is wrong or silently missing half the time
would be worse than no name at all.

Coverage is every ticker used in the app's own presets, plus the common
US-listed ETFs likely to be typed into a custom watchlist. Names were
checked against issuer fact sheets and standard fund-data sources rather
than guessed from the ticker. Anything not in the table falls back to the
bare ticker -- the safe default, not a placeholder.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

NAMES: Dict[str, str] = {
    # --- US broad market / style / factor ---------------------------------
    "SPY": "SPDR S&P 500 ETF Trust",
    "IVV": "iShares Core S&P 500 ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "VTV": "Vanguard Value ETF",
    "VUG": "Vanguard Growth ETF",
    "VBR": "Vanguard Small-Cap Value ETF",
    "VB": "Vanguard Small-Cap ETF",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "QUAL": "iShares MSCI USA Quality Factor ETF",
    "USMV": "iShares MSCI USA Min Vol Factor ETF",
    "VLUE": "iShares MSCI USA Value Factor ETF",
    "SIZE": "iShares MSCI USA Size Factor ETF",
    "QVAL": "Alpha Architect U.S. Quantitative Value ETF",
    "QMOM": "Alpha Architect U.S. Quantitative Momentum ETF",
    "FRDM": "Freedom 100 Emerging Markets ETF",
    "CAOS": "Alpha Architect Tail Risk ETF",

    # --- US sectors (SPDR Select Sector) -----------------------------------
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",

    # --- International developed / emerging --------------------------------
    "EFA": "iShares MSCI EAFE ETF",
    "EFV": "iShares MSCI EAFE Value ETF",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VGK": "Vanguard FTSE Europe ETF",
    "SCZ": "iShares MSCI EAFE Small-Cap ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "EWC": "iShares MSCI Canada ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "IEMG": "iShares Core MSCI Emerging Markets ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "IVAL": "Alpha Architect International Quantitative Value ETF",
    "IMOM": "Alpha Architect International Quantitative Momentum ETF",

    # --- Real assets ---------------------------------------------------------
    "VNQ": "Vanguard Real Estate ETF",
    "REET": "iShares Global REIT ETF",
    "GLD": "SPDR Gold Shares",
    "IAU": "iShares Gold Trust",
    "DBC": "Invesco DB Commodity Index Tracking Fund",
    "PDBC": "Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF",
    "DBA": "Invesco DB Agriculture Fund",
    "BCI": "abrdn Bloomberg All Commodity Strategy K-1 Free ETF",

    # --- Fixed income --------------------------------------------------------
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "SCHR": "Schwab Intermediate-Term U.S. Treasury ETF",
    "TIP": "iShares TIPS Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
    "BNDX": "Vanguard Total International Bond ETF",
    "BWX": "SPDR Bloomberg International Treasury Bond ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",

    # --- Currency --------------------------------------------------------
    "UUP": "Invesco DB US Dollar Index Bullish Fund",

    # --- Canada (TSX) --------------------------------------------------------
    "XIC.TO": "iShares Core S&P/TSX Capped Composite Index ETF",
    "XIU.TO": "iShares S&P/TSX 60 Index ETF",
    "ZEB.TO": "BMO Equal Weight Banks Index ETF",
    "ZLB.TO": "BMO Low Volatility Canadian Equity ETF",
    "HCAL.TO": "Hamilton Enhanced Canadian Bank ETF",
    "HXT.TO": "Horizons S&P/TSX 60 Index ETF",
    "WXM.TO": "First Asset Morningstar Canada Momentum Index ETF",
    "XDV.TO": "iShares Canadian Select Dividend Index ETF",
    "XCV.TO": "iShares Canadian Value Index ETF",
    "XCG.TO": "iShares Canadian Growth Index ETF",
    "XDIV.TO": "iShares Core MSCI Canadian Quality Dividend Index ETF",
    "XEI.TO": "iShares S&P/TSX Composite High Dividend Index ETF",
    "XEF.TO": "iShares Core MSCI EAFE IMI Index ETF",
    "XEG.TO": "iShares S&P/TSX Capped Energy Index ETF",
    "XIT.TO": "iShares S&P/TSX Capped Information Technology Index ETF",
    "XFN.TO": "iShares S&P/TSX Capped Financials Index ETF",
    "XMA.TO": "iShares S&P/TSX Capped Materials Index ETF",
    "XRE.TO": "iShares S&P/TSX Capped REIT Index ETF",
    "XST.TO": "iShares S&P/TSX Capped Consumer Staples Index ETF",
    "XUT.TO": "iShares S&P/TSX Capped Utilities Index ETF",
    "XSB.TO": "iShares Core Canadian Short Term Bond Index ETF",
    "XBB.TO": "iShares Core Canadian Universe Bond Index ETF",
    "XUU.TO": "iShares Core S&P U.S. Total Market Index ETF",
    "CGL.TO": "iShares Gold Bullion ETF",
    "PSA.TO": "Purpose High Interest Savings Fund",
}


def name_of(ticker: str) -> Optional[str]:
    """The full name for a ticker, or None if it isn't in the table."""
    return NAMES.get(str(ticker).strip().upper())


def label(ticker: str, sep: str = " \u2014 ") -> str:
    """"TICKER" alone, or "TICKER — Full Name" when the name is known."""
    t = str(ticker).strip()
    n = name_of(t)
    return f"{t}{sep}{n}" if n else t


def label_map(tickers: Iterable[str], sep: str = " \u2014 ") -> Dict[str, str]:
    """{ticker: label} for a collection, e.g. for a selectbox's format_func."""
    return {t: label(t, sep) for t in tickers}
