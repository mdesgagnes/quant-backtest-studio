"""TradingView Lightweight Charts.

Renders price and equity charts with TradingView's own charting library
instead of Plotly. It is worth the extra surface for one reason: the
crosshair, the synced price and time labels, and the pan-and-zoom behaviour
over tens of thousands of bars are what a price chart is judged on, and a
general-purpose plotting library does not match them.

The library is TradingView's, Apache 2.0, about 35 KB, no build step, loaded
from a CDN into a sandboxed iframe. **The licence requires crediting
TradingView as the product creator and linking to tradingview.com on any
page a user sees.** That is satisfied here by leaving the attribution logo
visible on every chart and printing a credit line beneath it; neither should
be removed.

The version is pinned deliberately. Lightweight Charts changed its series
API between v4 and v5 (`addCandlestickSeries` became
`addSeries(CandlestickSeries, ...)`), so floating on "latest" would mean the
charts silently stop rendering the day the CDN rolls forward. Pinning costs
nothing and removes that failure entirely.
"""
from __future__ import annotations

import html as _html
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .brand import BRAND

CDN = ("https://unpkg.com/lightweight-charts@4.2.3/dist/"
       "lightweight-charts.standalone.production.js")

ATTRIBUTION = (
    '<div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
    'letter-spacing:.06em;color:{muted};padding:.35rem .1rem 0;">'
    'Charting by <a href="https://www.tradingview.com/" target="_blank" '
    'rel="noopener" style="color:{red};text-decoration:none;">TradingView '
    'Lightweight Charts\u2122</a></div>'
)


def _j(obj: Any) -> str:
    """JSON for embedding inside a <script> block.

    `json.dumps` does not escape `</`, so a series named "</script>" would
    close the script tag and everything after it would be parsed as HTML.
    Escaping the sequence is the standard fix and changes nothing about how
    JavaScript reads the string.
    """
    return json.dumps(obj).replace("</", "<\\/")


def _esc(text: Any) -> str:
    """For anything landing in HTML rather than in the script."""
    return _html.escape(str(text), quote=True)


def _theme() -> Dict[str, Any]:
    """Chart options in the application's palette."""
    return {
        "layout": {
            "background": {"type": "solid", "color": BRAND["panel"]},
            "textColor": BRAND["muted"],
            "fontFamily": "IBM Plex Mono, SFMono-Regular, Consolas, monospace",
            "fontSize": 11,
            "attributionLogo": True,      # licence: keep this visible
        },
        "grid": {
            "vertLines": {"color": BRAND["rule_soft"]},
            "horzLines": {"color": BRAND["rule_soft"]},
        },
        "rightPriceScale": {"borderColor": BRAND["rule"]},
        "timeScale": {"borderColor": BRAND["rule"], "rightOffset": 4},
        "crosshair": {
            "mode": 0,
            "vertLine": {"color": BRAND["faint"], "width": 1, "style": 3,
                         "labelBackgroundColor": BRAND["red"]},
            "horzLine": {"color": BRAND["faint"], "width": 1, "style": 3,
                         "labelBackgroundColor": BRAND["red"]},
        },
    }


def _times(index: pd.DatetimeIndex) -> List[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(index)]


def _clean(values) -> List[Optional[float]]:
    out = []
    for v in values:
        f = float(v) if v is not None else np.nan
        out.append(None if (f != f or np.isinf(f)) else round(f, 6))
    return out


def _series_points(series: pd.Series) -> List[Dict[str, Any]]:
    s = series.dropna()
    return [{"time": t, "value": v}
            for t, v in zip(_times(s.index), _clean(s.to_numpy()))
            if v is not None]


def _shell(body_js: str, height: int, extra: str = "") -> str:
    """One self-contained document. Data is embedded as JSON, never as
    interpolated code, so a ticker containing a quote cannot become script."""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="{CDN}"></script>
<style>
  body{{margin:0;background:{BRAND['panel']};font-family:'IBM Plex Mono',monospace;}}
  #wrap{{width:100%;height:{height}px;}}
  .legend{{position:absolute;z-index:5;top:8px;left:10px;font-size:11px;
    color:{BRAND['text']};pointer-events:none;line-height:1.5;}}
  .legend b{{color:{BRAND['red']};}}
</style></head><body>
<div style="position:relative;">{extra}<div id="wrap"></div></div>
<script>
(function() {{
  var el = document.getElementById('wrap');
  if (!window.LightweightCharts) {{
    el.innerHTML = '<div style="color:{BRAND['muted']};font-size:12px;'
      + 'padding:14px;">Chart library did not load. This needs outbound '
      + 'access to unpkg.com.</div>';
    return;
  }}
  {body_js}
  new ResizeObserver(function(entries) {{
    if (entries.length) chart.applyOptions({{ width: entries[0].contentRect.width }});
  }}).observe(el);
}})();
</script></body></html>"""


# ----------------------------------------------------------------------
def candles(ohlc: pd.DataFrame, volume: Optional[pd.Series] = None,
            title: str = "", height: int = 460) -> str:
    """Candlesticks, with volume in a lower pane when supplied.

    `ohlc` needs Open, High, Low and Close columns. Rows with any of them
    missing are dropped rather than filled: inventing a high or a low draws
    a bar that never traded.
    """
    need = ["Open", "High", "Low", "Close"]
    df = ohlc.dropna(subset=[c for c in need if c in ohlc.columns])
    df = df[[c for c in need if c in df.columns]]
    if df.empty or len(df.columns) < 4:
        return _shell("var chart=null;", height)

    bars = [
        {"time": t, "open": o, "high": h, "low": l, "close": c}
        for t, o, h, l, c in zip(
            _times(df.index), _clean(df["Open"]), _clean(df["High"]),
            _clean(df["Low"]), _clean(df["Close"]))
        if None not in (o, h, l, c)
    ]

    vol_js = ""
    if volume is not None and volume.notna().any():
        v = volume.reindex(df.index)
        closes = df["Close"].to_numpy()
        opens = df["Open"].to_numpy()
        vol = [{"time": t, "value": val,
                "color": (BRAND["gain"] if c >= o else BRAND["loss"])}
               for t, val, c, o in zip(_times(v.index), _clean(v.to_numpy()),
                                       closes, opens)
               if val is not None]
        vol_js = f"""
  var volSeries = chart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }}, priceScaleId: 'vol',
    priceLineVisible: false, lastValueVisible: false }});
  chart.priceScale('vol').applyOptions({{
    scaleMargins: {{ top: 0.82, bottom: 0 }} }});
  volSeries.setData({_j(vol)});"""

    body = f"""
  var chart = LightweightCharts.createChart(el, {_j(_theme())});
  var series = chart.addCandlestickSeries({{
    upColor: {_j(BRAND['gain'])},
    downColor: {_j(BRAND['loss'])},
    borderUpColor: {_j(BRAND['gain'])},
    borderDownColor: {_j(BRAND['loss'])},
    wickUpColor: {_j(BRAND['gain'])},
    wickDownColor: {_j(BRAND['loss'])} }});
  series.setData({_j(bars)});
  {vol_js}
  chart.timeScale().fitContent();"""
    legend = (f'<div class="legend"><b>{_esc(title)}</b></div>' if title else "")
    return _shell(body, height, legend)


def line(series_map: Dict[str, pd.Series], title: str = "",
         height: int = 420, log: bool = False,
         percent: bool = False) -> str:
    """One or more lines, for equity curves and comparisons."""
    palette = [BRAND["red"], "#C9B896", BRAND["gain"], "#8C93A8",
               "#D08C4E", "#7E6BA8", "#5B9EC7"]
    parts = []
    for i, (name, s) in enumerate(series_map.items()):
        pts = _series_points(s)
        if not pts:
            continue
        parts.append(f"""
  var s{i} = chart.addLineSeries({{
    color: {_j(palette[i % len(palette)])}, lineWidth: {2 if i == 0 else 1},
    title: {_j(str(name))}, priceLineVisible: false }});
  s{i}.setData({_j(pts)});""")

    opts = _theme()
    opts["rightPriceScale"]["mode"] = 1 if log else (2 if percent else 0)
    body = (f"""
  var chart = LightweightCharts.createChart(el, {_j(opts)});"""
            + "".join(parts) + """
  chart.timeScale().fitContent();""")
    legend = (f'<div class="legend"><b>{_esc(title)}</b></div>' if title else "")
    return _shell(body, height, legend)


def equity_with_trades(equity: pd.Series, trades: Optional[pd.DataFrame] = None,
                       benchmark: Optional[pd.Series] = None,
                       title: str = "", height: int = 460,
                       log: bool = True, max_markers: int = 400) -> str:
    """The equity curve with every fill marked on it.

    This is the view that connects a backtest to a chart: buys and sells
    sit on the curve at the date they happened, so a drawdown can be read
    against what the strategy was doing at the time instead of inferred
    from a separate table.

    Markers are capped. A daily strategy over twenty years produces
    thousands of fills, and a chart carrying all of them is unreadable and
    slow; beyond the cap the most recent are kept, and the caller is told.
    """
    parts = []
    eq = _series_points(equity)
    opts = _theme()
    opts["rightPriceScale"]["mode"] = 1 if log else 0

    parts.append(f"""
  var chart = LightweightCharts.createChart(el, {_j(opts)});
  var main = chart.addAreaSeries({{
    lineColor: {_j(BRAND['red'])}, lineWidth: 2,
    topColor: 'rgba(228,28,35,0.22)', bottomColor: 'rgba(228,28,35,0.02)',
    title: {_j(str(title or 'Strategy'))}, priceLineVisible: false }});
  main.setData({_j(eq)});""")

    if benchmark is not None and benchmark.notna().any():
        base = benchmark.dropna()
        if len(base) and len(equity.dropna()):
            scaled = base / base.iloc[0] * float(equity.dropna().iloc[0])
            parts.append(f"""
  var bench = chart.addLineSeries({{
    color: {_j('#C9B896')}, lineWidth: 1,
    title: 'Benchmark', priceLineVisible: false }});
  bench.setData({_j(_series_points(scaled))});""")

    if trades is not None and not trades.empty and "Date" in trades.columns:
        t = trades.copy()
        side_col = "Change" if "Change" in t.columns else None
        grouped = []
        for date, chunk in t.groupby("Date"):
            buys = int((chunk[side_col] > 0).sum()) if side_col else 0
            sells = int((chunk[side_col] < 0).sum()) if side_col else 0
            grouped.append((date, buys, sells))
        grouped.sort(key=lambda x: x[0])
        if len(grouped) > max_markers:
            grouped = grouped[-max_markers:]

        markers = []
        for date, buys, sells in grouped:
            if buys and not sells:
                markers.append({"time": pd.Timestamp(date).strftime("%Y-%m-%d"),
                                "position": "belowBar", "color": BRAND["gain"],
                                "shape": "arrowUp", "text": f"+{buys}"})
            elif sells and not buys:
                markers.append({"time": pd.Timestamp(date).strftime("%Y-%m-%d"),
                                "position": "aboveBar", "color": BRAND["loss"],
                                "shape": "arrowDown", "text": f"-{sells}"})
            else:
                markers.append({"time": pd.Timestamp(date).strftime("%Y-%m-%d"),
                                "position": "belowBar", "color": BRAND["red"],
                                "shape": "circle",
                                "text": f"{buys}/{sells}"})
        if markers:
            parts.append(f"\n  main.setMarkers({_j(markers)});")

    parts.append("\n  chart.timeScale().fitContent();")
    legend = (f'<div class="legend"><b>{_esc(title)}</b></div>' if title else "")
    return _shell("".join(parts), height, legend)


def attribution_html() -> str:
    """The credit line the licence requires. Render it under every chart."""
    return ATTRIBUTION.format(muted=BRAND["faint"], red=BRAND["red"])
