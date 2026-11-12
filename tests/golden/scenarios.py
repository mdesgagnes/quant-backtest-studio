"""The reference runs. Each function drives the app the way a person would.

All five use the frozen files in `tests/golden/fixtures/`, never the network.
Between them they cover: uploaded prices (long format with open, high, low,
volume and dividends), a built-in model, an imported target-weights file, a
cross-sectional signal file, a macro signal file, a blended benchmark, a
comparison strategy, execution at the open, the volume-aware impact model, a
management fee, Monte Carlo, and the return-stream workspace.

Yahoo Finance downloads are not covered: they change daily, so they cannot
be golden. Everything downstream of the download is.
"""
from __future__ import annotations

from typing import Callable, Dict

from harness import LegacyApp


def _file_source(app: LegacyApp) -> None:
    app.upload("price_files", "prices_long.csv")
    app.set("radio", "File (CSV / Excel)", label="Source").run("choose file source")


def momentum_file(app: LegacyApp) -> None:
    """Built-in Dual Momentum on uploaded prices, defaults throughout,
    benchmark XIC.TO, then a Monte Carlo run from the Robustness tab."""
    _file_source(app)
    app.click("Run backtest", what="run backtest")
    app.click("Run Monte Carlo simulation", sidebar=False, what="monte carlo")


def imported_weights(app: LegacyApp) -> None:
    """A target-weights file from an existing strategy, on file dates."""
    _file_source(app)
    app.upload("wup", "target_weights.csv")
    app.set("radio", "Imported target weights", label="Signal origin").run("choose weights")
    app.click("Run backtest", what="run backtest")


def factor_signal(app: LegacyApp) -> None:
    """A cross-sectional signal file (one column per symbol) ranked by the
    Imported Factor Ranking model, with a 45-day publication lag, execution
    at the open, volume-scaled impact, a cash rate and a management fee."""
    app.upload("exogup", "fundamental_factor.csv")
    _file_source(app)
    app.set("selectbox", "factor_rank", label="Model").run("choose model")
    app.set("slider", 45, label="Publication lag (calendar days)")
    app.set("selectbox", "Low value = favorable", key="p_factor_rank_direction")
    app.set("number_input", 2, key="p_factor_rank_top_n")
    app.set("selectbox", "Open (marked at the close)", label="Execution price")
    app.set("selectbox", "Square-root law, scaled by volatility (recommended)",
            label="Market impact")
    app.set("number_input", 2.0, key="cashrate_pct")
    app.set("number_input", 0.5, key="mgmtfee_pct")
    app.click("Run backtest", what="run backtest")


def macro_blend(app: LegacyApp) -> None:
    """A macro signal file driving the Macro Gate model, quarterly, against
    a 60/40 blended benchmark, with one comparison strategy."""
    app.upload("exogup", "economic_data.csv")
    _file_source(app)
    app.set("selectbox", "macro_gate", label="Model")
    app.set("selectbox", "Blend of several", label="Comparison benchmark")
    app.run("choose model and blend")
    app.set("text_area", "XIC.TO:60, ZLB.TO:40", key="benchspec")
    app.set("slider", 30, label="Publication lag (calendar days)")
    app.set("selectbox", "ISM_MANUFACTURING", key="p_macro_gate_series")
    app.set("selectbox", "Q", label="Rebalance")
    app.set("number_input", 1, key="ncmp")
    app.run("set blend and comparison")
    app.click("Run backtest", what="run backtest")


def return_stream(app: LegacyApp) -> None:
    """The Return stream workspace on a monthly file, percent scale, against
    a benchmark column, 2% fixed risk-free rate, then Monte Carlo."""
    app.upload("rsup", "returns_monthly.csv")
    app.set("radio", "Return stream", label="Source").run("choose return stream")
    app.set("selectbox", "Fixed annual rate", key="rs_rf").run("risk-free")
    app.set("selectbox", "Benchmark", key="rsbench", sidebar=False).run("benchmark")
    app.click("Run Monte Carlo simulation", sidebar=False, what="monte carlo")


SCENARIOS: Dict[str, Callable[[LegacyApp], None]] = {
    "momentum_file": momentum_file,
    "imported_weights": imported_weights,
    "factor_signal": factor_signal,
    "macro_blend": macro_blend,
    "return_stream": return_stream,
}
