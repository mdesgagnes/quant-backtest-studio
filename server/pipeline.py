"""The backtest run, with no interface attached.

This is the block of `app.py` between "Configuration assembly" and
`st.session_state["run"] = {...}` (app.py:3351-3771 at commit 8c170f5),
lifted into one function. It is a faithful port, kept in the same order with
the same comments, so the two can be read side by side. It calls the engine;
it does not change it.

`RunSpec` is the control panel as data: one field per control. `execute`
returns the same dictionary the Streamlit app keeps in its session, which is
what every screen and export is built from.

Proven against the recorded reference runs by tests/golden/test_golden_service.py.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qbt import allocation as ALLOC
from qbt import compare as CMP
from qbt import fx as FX
from qbt import presets as PRESETS
from qbt import price_import as PI
from qbt.config import (CostConfig, DataConfig, EngineConfig, ExogConfig,
                        RunConfig, StrategyConfig)
from qbt.data import MarketData, clean_prices, load_market_data, load_yfinance
from qbt.engine import (align_start, benchmark_costs, benchmark_result,
                        blended_benchmark, common_start, first_active_date,
                        run_backtest)
from qbt.exog import exog_report, load_exog, prepare_exog
from qbt.external import load_target_weights, prepare_target_weights
from qbt.strategies import REGISTRY

BLEND = "Blend of several"
BENCH_STRAT = "Another strategy"
BENCH_MODES = ["Total return (adjusted close)",
               "Price return + dividends reinvested at rebalance",
               "Price return only"]

Upload = Tuple[str, bytes]          # (file name, content)


class RunError(Exception):
    """A run that cannot proceed, with the message the user should read."""


@dataclass
class RunSpec:
    """Every control of the Backtest panel. Defaults are the app's defaults."""
    # Data
    source: str = "yfinance"                       # yfinance | upload
    tickers: List[str] = field(default_factory=lambda: ["XIC.TO", "ZEB.TO", "XEI.TO"])
    start: Optional[str] = "2010-01-01"
    end: Optional[str] = None
    universe: Optional[List[str]] = None           # None = every loaded symbol
    preset: str = "Custom"
    benchmark: Optional[str] = "XIC.TO"            # ticker | BLEND | BENCH_STRAT | None
    bench_blend: str = "XIC.TO:60, XBB.TO:40"
    bench_blend_rule: str = "A"
    bench_mode: str = BENCH_MODES[0]
    bench_frictions: List[str] = field(default_factory=list)   # commission | slippage | fee
    adjusted: bool = False                         # Yahoo: total-return prices
    file_convention: str = "auto"                  # upload: auto | total | price
    cash_proxy: Optional[str] = None
    base_currency: str = ""
    currency_overrides: Dict[str, str] = field(default_factory=dict)
    # Signal
    exog_lag: int = 1
    mode: str = "builtin"                          # builtin | external_weights
    weights_normalize: str = "None"
    weights_calendar: str = "File dates"
    weights_sheet: Optional[str] = None
    strategy: str = "dual_momentum"
    params: Dict[str, Any] = field(default_factory=dict)
    construction: str = "Single strategy"
    class_map: Dict[str, str] = field(default_factory=dict)
    class_budgets: Dict[str, float] = field(default_factory=dict)
    core_spec: str = ""
    core_budget: float = 0.5
    core_exclusive: bool = True
    blend_sleeves: List[Dict[str, Any]] = field(default_factory=list)
    # Execution
    rebalance: str = "M"
    day_rule: str = "last"
    day_of_month: int = 15
    weekday: int = 4
    nth: int = 1
    anchor_month: int = 12
    execution_lag: int = 1
    execute_at_open: bool = False
    trim_warmup: bool = True
    initial_capital: float = 100_000.0
    max_leverage: float = 1.0
    whole_shares: bool = False
    cash_sweep: str = "none"
    cash_buffer: float = 0.01
    cash_sweep_threshold: float = 0.0
    # Comparison strategies: [{"key", "params", "label"}]
    compare: List[Dict[str, Any]] = field(default_factory=list)
    compare_bench_index: int = 0
    # Frictions
    commission_bps: float = 5.0
    slippage_bps: float = 25.0
    impact_model: str = "flat"
    impact_bps_at_10pct_adv: float = 25.0
    impact_coef: float = 1.0
    slippage_overrides: Dict[str, float] = field(default_factory=dict)
    cash_rate_pa: float = 0.0
    management_fee_pa: float = 0.0
    fx_spread_bps: float = 25.0
    fx_cash: str = "net"
    foreign_cash_rate_pa: float = 0.0
    fee_pays_frictions: bool = True


@dataclass
class RunFiles:
    prices: List[Upload] = field(default_factory=list)
    exog: List[Upload] = field(default_factory=list)
    weights: Optional[Upload] = None


# ----------------------------------------------------------------------
# Readers (the app's cached parse_* functions)
# ----------------------------------------------------------------------
def read_exog(files: List[Upload]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Several exogenous files as one frame; a repeated column name is
    suffixed with its file name. Returns (frame, read errors)."""
    frames, errors = [], []
    for name, content in files:
        try:
            buf = io.BytesIO(content)
            buf.name = name
            fr = load_exog(buf, None)
            fr.columns = [f"{c}" if c not in [x for fr2 in frames for x in fr2.columns]
                          else f"{c} ({name})" for c in fr.columns]
            frames.append(fr)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not frames:
        return None, errors
    raw = pd.concat(frames, axis=1).sort_index()
    return raw.loc[:, ~raw.columns.duplicated(keep="first")], errors


def read_weights(upload: Upload, sheet: Optional[str]) -> pd.DataFrame:
    buf = io.BytesIO(upload[1])
    buf.name = upload[0]
    return load_target_weights(buf, sheet)


def _defaults(strategy_key: str, given: Dict[str, Any]) -> Dict[str, Any]:
    """A model's parameters: its defaults, overridden by what was set."""
    out: Dict[str, Any] = {}
    for p in REGISTRY[strategy_key].params:           # what each widget returns
        v = given.get(p.key, p.default)
        if p.kind == "int":
            out[p.key] = int(v)
        elif p.kind == "float":
            out[p.key] = float(v)
        elif p.kind == "bool":
            out[p.key] = bool(v)
        elif p.kind == "series":
            out[p.key] = given.get(p.key, "") or ""
        elif p.kind == "choice" and p.choices:
            out[p.key] = v if v in list(p.choices) else list(p.choices)[0]
        else:
            out[p.key] = str(v or "")
    return out


# ----------------------------------------------------------------------
def build_config(spec: RunSpec, files: RunFiles,
                 price_import: Optional["PI.ImportResult"],
                 exog_raw: Optional[pd.DataFrame],
                 fx_preview: Optional["FX.FxBook"],
                 sel_universe: List[str], benchmark: Optional[str],
                 params: Dict[str, Any]) -> RunConfig:
    """The RunConfig the app assembles from its controls (app.py:3351)."""
    mode = spec.mode
    strategy = REGISTRY[spec.strategy] if mode == "builtin" else None
    suffix = {"By asset class": " by class", "Core + strategy": " + core",
              "Blend of strategies": " blend"}.get(spec.construction, "")
    blend_label = (" + ".join(b["label"] for b in spec.blend_sleeves)
                   if (mode == "builtin" and spec.construction == "Blend of strategies"
                       and spec.blend_sleeves) else None)
    w_source = files.weights[0] if (mode == "external_weights" and files.weights) else ""
    run_label = (blend_label if blend_label else
                 ((strategy.label + suffix) if mode == "builtin"
                  else (f"Imported weights: {w_source}" if w_source
                        else "Imported weights")))
    has_open = spec.source == "yfinance" or (price_import is not None
                                             and price_import.has("open"))
    if spec.source == "yfinance":
        adjusted = bool(spec.adjusted)
    else:
        conv = (PI.resolve_convention(price_import, spec.file_convention)
                if price_import is not None else "total")
        adjusted = conv == "total"
    exog_columns = list(exog_raw.columns) if exog_raw is not None else []
    return RunConfig(
        label=run_label,
        data=DataConfig(
            source="yfinance" if spec.source == "yfinance" else "upload",
            tickers=sel_universe,
            start=str(spec.start) if (spec.source == "yfinance" and spec.start) else "1990-01-01",
            end=str(spec.end) if (spec.source == "yfinance" and spec.end) else None,
            benchmark=(None if benchmark == BLEND else benchmark),
            cash_proxy=spec.cash_proxy,
            adjusted=adjusted, use_dividends=not adjusted,
            base_currency=spec.base_currency,
            currencies=(dict(fx_preview.raw) if (spec.base_currency and fx_preview is not None)
                        else {}),
        ),
        exog=ExogConfig(
            enabled=exog_raw is not None,
            publication_lag_days=int(spec.exog_lag),
            columns=exog_columns,
            note=", ".join(n for n, _ in files.exog) if files.exog else "",
        ),
        strategy=StrategyConfig(mode=mode,
                                name=spec.strategy if mode == "builtin" else "external_weights",
                                params=params,
                                weights_normalize=spec.weights_normalize,
                                weights_calendar=spec.weights_calendar,
                                weights_source=w_source),
        engine=EngineConfig(initial_capital=float(spec.initial_capital),
                            rebalance=spec.rebalance,
                            execution_lag=int(spec.execution_lag),
                            max_leverage=float(spec.max_leverage),
                            execute_at_open=bool(spec.execute_at_open and has_open),
                            trim_warmup=bool(spec.trim_warmup),
                            whole_shares=bool(spec.whole_shares),
                            cash_sweep=spec.cash_sweep,
                            cash_sweep_threshold=float(spec.cash_sweep_threshold),
                            cash_buffer=float(spec.cash_buffer),
                            day_rule=spec.day_rule, day_of_month=int(spec.day_of_month),
                            weekday=int(spec.weekday), nth=int(spec.nth),
                            anchor_month=int(spec.anchor_month),
                            fx_cash=spec.fx_cash),
        costs=CostConfig(commission_bps=spec.commission_bps,
                         slippage_bps=spec.slippage_bps,
                         fx_spread_bps=float(spec.fx_spread_bps),
                         foreign_cash_rate_pa=float(spec.foreign_cash_rate_pa),
                         impact_model=spec.impact_model,
                         impact_bps_at_10pct_adv=spec.impact_bps_at_10pct_adv,
                         impact_coef=spec.impact_coef,
                         slippage_overrides=dict(spec.slippage_overrides),
                         cash_rate_pa=spec.cash_rate_pa,
                         management_fee_pa=spec.management_fee_pa,
                         apply_frictions_to_fee_liquidation=spec.fee_pays_frictions),
    )


def execute(spec: RunSpec, files: RunFiles,
            fetch_market: Callable[..., MarketData] = None,
            fetch_vix: Callable[..., Optional[pd.Series]] = None) -> Dict[str, Any]:
    """Runs the backtest the controls describe. Raises RunError with the
    app's own message when it cannot."""
    if fetch_market is None:
        def fetch_market(tickers, start, end, adjusted, want_open, want_div, want_ohlc=False):
            return load_market_data(list(tickers), start, end, adjusted=adjusted,
                                    want_open=want_open, want_dividends=want_div,
                                    want_ohlc=want_ohlc)
    mode = spec.mode
    source_yahoo = spec.source == "yfinance"

    # --- what the panel knows before Run is pressed -------------------
    price_import = None
    if source_yahoo:
        tickers = [t.strip().upper() for t in spec.tickers if t and t.strip()]
        start, end = spec.start, spec.end or str(pd.Timestamp.today().date())
        spec = _replace(spec, end=end)
        adjusted = bool(spec.adjusted)
        file_convention = None
    else:
        if not files.prices:
            raise RunError("No file loaded.")
        price_import = PI.read_files(list(files.prices))
        tickers = price_import.instruments
        if not tickers:
            raise RunError("The file could not be read: no prices were found. "
                           + " ".join(price_import.warnings[:3]))
        start, end = None, None
        file_convention = PI.resolve_convention(price_import, spec.file_convention)
        adjusted = file_convention == "total"
    use_divs = not adjusted

    univ_options = tickers or []
    sel_universe = ([t for t in spec.universe if t in univ_options]
                    if spec.universe is not None else list(univ_options))

    benchmark = spec.benchmark
    bench_blend: Dict[str, float] = {}
    bench_blend_rule = "A"
    if benchmark == BLEND:
        bench_spec = spec.bench_blend
        bench_blend = ALLOC.parse_fixed(
            bench_spec, univ_options + [t.strip().upper() for t in
                                        bench_spec.replace(":", " ").replace(",", " ")
                                        .replace(";", " ").split()
                                        if any(ch.isalpha() for ch in t)])
        bench_blend_rule = spec.bench_blend_rule
        benchmark = BLEND if bench_blend else None
    elif benchmark not in (BENCH_STRAT,) and benchmark not in univ_options:
        benchmark = None
    bench_is_strategy = benchmark == BENCH_STRAT
    if bench_is_strategy:
        benchmark = None
    bench_mode = spec.bench_mode if benchmark else BENCH_MODES[0]
    bench_fric = list(spec.bench_frictions) if benchmark else []
    cash_proxy = spec.cash_proxy

    fx_universe = list(dict.fromkeys(
        [str(t).strip().upper() for t in (
            list(sel_universe)
            + ([benchmark] if isinstance(benchmark, str) and benchmark != BLEND else [])
            + list(bench_blend.keys())
            + ([cash_proxy] if cash_proxy else []))]))
    fx_preview = None
    if spec.base_currency and fx_universe:
        det = FX.detect_currencies(tuple(fx_universe),
                                   None if source_yahoo else (lambda t: None))
        fx_preview = FX.FxBook.resolve(fx_universe, spec.base_currency,
                                       dict(spec.currency_overrides), det)

    exog_raw, _exog_errors = read_exog(files.exog) if files.exog else (None, [])
    exog_columns = list(exog_raw.columns) if exog_raw is not None else []

    weights_raw = None
    strategy = None
    strat_key = "external_weights"
    params: Dict[str, Any] = {}
    if mode == "external_weights":
        if files.weights is not None:
            weights_raw = read_weights(files.weights, spec.weights_sheet)
    else:
        strat_key = spec.strategy
        strategy = REGISTRY[strat_key]
        if strat_key == "vix_adaptive_momentum" and "VIX (auto)" not in exog_columns:
            vix = None
            try:
                if fetch_vix is not None:
                    vix = fetch_vix(str(start), str(end))
                else:
                    df = load_yfinance(["^VIX"], str(start), str(end), "Close")
                    vix = df.iloc[:, 0].dropna() if not df.empty else None
            except Exception:
                vix = None
            if vix is not None and not vix.empty:
                frame = vix.to_frame("VIX (auto)")
                exog_raw = (pd.concat([exog_raw, frame], axis=1)
                            if exog_raw is not None else frame)
                exog_columns = list(exog_raw.columns)
        params = _defaults(strat_key, spec.params)
        for p in strategy.params:                    # a series picker shows the first column
            if p.kind == "series" and not params.get(p.key) and exog_columns:
                params[p.key] = exog_columns[0]

    blend_sleeves = [dict(b, params=_defaults(b["key"], b.get("params", {})),
                          label=b.get("label") or REGISTRY[b["key"]].label)
                     for b in spec.blend_sleeves]
    cmp_specs = [dict(c, params=_defaults(c["key"], c.get("params", {})),
                      label=c.get("label") or REGISTRY[c["key"]].label)
                 for c in spec.compare]
    spec_for_cfg = spec if not blend_sleeves else _replace(spec, blend_sleeves=blend_sleeves)

    cfg = build_config(spec_for_cfg, files, price_import, exog_raw, fx_preview,
                       sel_universe, benchmark, params)
    run_label = cfg.label
    construction = spec.construction if mode == "builtin" else "Single strategy"
    problems = cfg.validate()
    blocking = [p for p in problems if not p.startswith("WARNING")]
    if blocking:
        raise RunError(" ".join(blocking))
    notices: List[str] = [p for p in problems if p not in blocking]

    impact_model = spec.impact_model
    exec_at_open = spec.execute_at_open

    # --- Execution (app.py:3429) --------------------------------------
    def build_market() -> MarketData:
        """Prices, plus opens and dividends when the settings call for them."""
        if source_yahoo:
            needed = list(dict.fromkeys(
                sel_universe
                + [t for t in (benchmark, cash_proxy) if t and t != BLEND]
                + list(bench_blend.keys())))
            need_div = bool(use_divs) or (
                benchmark is not None
                and bench_mode.startswith("Price return + dividends"))
            need_ohlc = (impact_model != "flat")
            return fetch_market(tuple(needed), str(start), str(end),
                                bool(adjusted), bool(exec_at_open) or need_ohlc,
                                need_div, need_ohlc)
        return PI.to_market(price_import, file_convention or "auto")

    market = build_market()
    fx_book = None
    if cfg.data.base_currency:
        fx_book = FX.FxBook.build(
            list(fx_universe), cfg.data.base_currency,
            dict(sorted((k, v) for k, v in cfg.data.currencies.items())),
            str(cfg.data.start), cfg.data.end,
            detect=None if source_yahoo else (lambda t: None))
        # A cash proxy outside the universe is a RATE, not a holding.
        market = fx_book.convert_market(
            market, skip=([cash_proxy] if cash_proxy and
                          cash_proxy not in sel_universe else []),
            at_open=bool(cfg.engine.execute_at_open and market.open is not None))
    prices, quality = clean_prices(market.close, cfg.data)
    for msg in market.notes:
        quality.warnings.append(msg)
    if fx_book is not None and cash_proxy:
        pc_ccy = fx_book.currency_of(cash_proxy)
        if pc_ccy != fx_book.base:
            quality.warnings.append(
                f"The cash proxy {cash_proxy} is quoted in {pc_ccy}, not "
                f"{fx_book.base}: its own-currency return is used as the "
                f"{fx_book.base} cash rate. Use a {fx_book.base} cash ETF "
                f"or a fixed rate for a cash rate that belongs to the "
                f"base currency.")
    if fx_book is not None and not fx_book.is_active([c for c in sel_universe]):
        quality.warnings.append(
            f"Every instrument is quoted in {fx_book.base}: nothing was converted.")

    cash_px = prices[cash_proxy] if (cash_proxy and cash_proxy in prices.columns) else None
    drop = [c for c in [cash_proxy] if c and c in prices.columns and c not in sel_universe]
    universe = prices.drop(columns=drop)
    universe = universe[[c for c in sel_universe if c in universe.columns]]

    if universe.empty or universe.shape[1] == 0:
        raise RunError("No usable instrument in the selected universe.")

    # Exogenous series: publication lag, then alignment
    exog_aligned, ex_rep = None, None
    exog_lag = int(spec.exog_lag)
    if exog_raw is not None and not exog_raw.empty:
        exog_aligned = prepare_exog(exog_raw, universe.index, exog_lag)
        ex_rep = exog_report(exog_raw, exog_aligned, exog_lag, universe.index)

    # The cash series strategies compare against.
    if cash_px is not None:
        cash_series = cash_px
    elif float(cfg.costs.cash_rate_pa) != 0.0:
        cash_series = pd.Series(
            100.0 * (1.0 + float(cfg.costs.cash_rate_pa) / 252.0)
            ** np.arange(len(universe.index)), index=universe.index)
    else:
        cash_series = None

    w_report, rebal_dates, sleeve_report = None, None, None
    max_lev = float(spec.max_leverage)
    class_map, class_budgets = dict(spec.class_map), dict(spec.class_budgets)
    if mode == "external_weights":
        if weights_raw is None or weights_raw.empty:
            raise RunError("No target weights file was imported.")
        weights, rebal_dates, w_report = prepare_target_weights(
            weights_raw, universe.index, list(universe.columns),
            spec.weights_normalize, max_lev)
        if spec.weights_calendar == "Engine calendar":
            rebal_dates = None
    else:
        # Sleeves resolve to an ordinary weight frame.
        if construction == "By asset class" and class_budgets:
            sleeves = ALLOC.sleeves_from_classes(
                {t: c for t, c in class_map.items() if t in universe.columns},
                class_budgets, strat_key, params)
            weights, sleeve_report = ALLOC.resolve(
                universe, sleeves, REGISTRY, exog_aligned,
                float(cfg.engine.max_leverage), cash=cash_series)
        elif construction == "Blend of strategies" and blend_sleeves:
            sleeves = [
                ALLOC.Sleeve(
                    f"{b['label']}", float(b["budget"]),
                    list(universe.columns), mode="strategy",
                    strategy_key=b["key"], params=dict(b["params"]))
                for b in blend_sleeves if b["budget"] > 1e-9
            ]
            weights, sleeve_report = ALLOC.resolve(
                universe, sleeves, REGISTRY, exog_aligned,
                float(cfg.engine.max_leverage), cash=cash_series)
        elif construction == "Core + strategy":
            core = ALLOC.parse_fixed(spec.core_spec, list(universe.columns))
            sleeves = []
            if core and spec.core_budget > 0:
                sleeves.append(ALLOC.Sleeve(
                    "Core", float(spec.core_budget), list(core),
                    mode="fixed", fixed=core))
            sat = 1.0 - float(spec.core_budget)
            if sat > 1e-9:
                pool = [c for c in universe.columns
                        if not (spec.core_exclusive and c in core)]
                if not pool:
                    notices.append(
                        "The core covers the whole universe, leaving "
                        "the strategy nothing to pick from. Its share "
                        "stays in cash.")
                else:
                    sleeves.append(ALLOC.Sleeve(
                        "Strategy", sat, pool,
                        mode="strategy", strategy_key=strat_key,
                        params=dict(params)))
            weights, sleeve_report = ALLOC.resolve(
                universe, sleeves, REGISTRY, exog_aligned,
                float(cfg.engine.max_leverage), cash=cash_series)
        else:
            weights = strategy.generate(universe, params, exog_aligned, cash_series)

    cols = list(universe.columns)
    open_px = None
    if market.open is not None and cfg.engine.execute_at_open:
        open_px = market.open.reindex(index=universe.index, columns=cols)
    div_px = None
    if market.dividends is not None and cfg.data.use_dividends:
        div_px = market.dividends.reindex(index=universe.index, columns=cols).fillna(0.0)
    vol_px = None
    if market.volume is not None and cfg.costs.impact_model != "flat":
        vol_px = market.volume.reindex(index=universe.index, columns=cols)

    result = run_backtest(universe, weights, cfg.engine, cfg.costs,
                          cash_px, run_label, rebal_dates,
                          open_prices=open_px, dividends=div_px,
                          volume=vol_px, fx=fx_book)
    # Comparison strategies: the same inputs, calendar, execution and
    # frictions as the main run, each with its own model.
    cmp_results: Dict[str, Any] = {}
    cmp_weights: Dict[str, pd.DataFrame] = {}
    cmp_names = CMP.unique_labels([run_label], [s["label"] for s in cmp_specs])
    for cspec, name in zip(cmp_specs, cmp_names):
        try:
            w_c = REGISTRY[cspec["key"]].generate(
                universe, cspec["params"], exog_aligned, cash_series)
            cmp_weights[name] = w_c
            cmp_results[name] = run_backtest(
                universe, w_c, cfg.engine, cfg.costs, cash_px, name,
                None, open_prices=open_px, dividends=div_px,
                volume=vol_px, fx=fx_book)
        except Exception as exc:
            quality.warnings.append(f"Comparison strategy {name} failed: {exc}")
    cmp_bench_idx = int(spec.compare_bench_index)
    bench_strat_name = (cmp_names[cmp_bench_idx] if bench_is_strategy
                        and cmp_bench_idx < len(cmp_names)
                        and cmp_names[cmp_bench_idx] in cmp_results
                        else None)

    # Where the model itself first takes a position.
    sig_start = (sleeve_report.signal_start if sleeve_report is not None else None)
    # With several strategies the record starts on the latest date any of
    # them first holds a position; every strategy buys, and pays, on it.
    shared_start = None
    if cfg.engine.trim_warmup:
        shared_start = common_start(result, *cmp_results.values(), signal_start=sig_start)
    if shared_start is not None and cmp_results:
        def _enter_on_shared_start(r, w, dates=None):
            first = first_active_date(r)
            if first is None or first >= shared_start:
                return r
            return run_backtest(
                universe,
                CMP.masked_until(w, shared_start, cfg.engine.execution_lag),
                cfg.engine, cfg.costs, cash_px, r.label, dates,
                open_prices=open_px, dividends=div_px, volume=vol_px,
                fx=fx_book)
        result = _enter_on_shared_start(result, weights, rebal_dates)
        cmp_results = {k: _enter_on_shared_start(v, cmp_weights[k])
                       for k, v in cmp_results.items()}

    # A benchmark that pays frictions buys on the shared start date.
    bench_costs = benchmark_costs(cfg.costs, bench_fric)
    bench_vol, bench_start = None, None
    if bench_costs is not None:
        if market.volume is not None and cfg.costs.impact_model != "flat":
            bench_vol = market.volume.reindex(index=prices.index)
        bench_start = shared_start
    bench = None
    bdiv = None
    if benchmark == BLEND and bench_blend:
        have = {k: v for k, v in bench_blend.items() if k in prices.columns}
        missing = [k for k in bench_blend if k not in prices.columns]
        if missing:
            quality.warnings.append(
                "Benchmark component(s) unavailable and dropped: " + ", ".join(missing))
        if have:
            bsrc, bdiv = prices, None
            if bench_mode.startswith("Total return") and market.adj_close is not None:
                bsrc = market.adj_close.reindex(prices.index)
            elif bench_mode.startswith("Price return + dividends"):
                if market.dividends is not None:
                    bdiv = market.dividends.reindex(prices.index)
                else:
                    quality.warnings.append(
                        "No dividend data for the benchmark blend: it "
                        "is shown on price return only, which understates it.")
            tot = sum(abs(v) for v in have.values()) or 1.0
            blabel = " / ".join(f"{k} {abs(v)/tot*100:.0f}%" for k, v in have.items())
            try:
                bench = blended_benchmark(
                    bsrc, have, cfg.engine, blabel,
                    dividends=bdiv, rebalance=bench_blend_rule,
                    costs=bench_costs, volume=bench_vol, start=bench_start)
            except Exception as exc:
                quality.warnings.append(f"Benchmark blend failed: {exc}")
    elif benchmark and benchmark in prices.columns:
        bseries, bdiv = prices[benchmark], None
        if bench_mode.startswith("Total return"):
            if market.adj_close is not None and benchmark in market.adj_close.columns:
                bseries = market.adj_close[benchmark].reindex(prices.index)
            elif not market.adjusted:
                quality.warnings.append(
                    f"No adjusted close for {benchmark}: the benchmark "
                    f"is measured on price return, which understates "
                    f"it. Supply an Adj Close column, or choose "
                    f"“Price return + dividends”.")
        elif bench_mode.startswith("Price return + dividends"):
            if market.dividends is not None and benchmark in market.dividends.columns:
                bdiv = market.dividends[benchmark].reindex(prices.index)
            else:
                quality.warnings.append(
                    f"No dividend data available for {benchmark}: the "
                    f"benchmark is shown on price return only, which understates it.")
        bench = benchmark_result(bseries.dropna(), cfg.engine, benchmark,
                                 dividends=bdiv, reinvest_rule=cfg.engine.rebalance,
                                 costs=bench_costs, volume=bench_vol, start=bench_start)

    if bench_strat_name is not None:
        bench = cmp_results[bench_strat_name]
        bdiv = div_px          # for the benchmark's tax calculation
    elif bench_is_strategy:
        quality.warnings.append(
            "No strategy was available to use as the benchmark: the run has none.")

    # Warm-up: both series must start on the same day.
    raw_start = result.equity.index[0]
    if cfg.engine.trim_warmup:
        aligned = align_start(result, bench, *cmp_results.values(),
                              initial_capital=cfg.engine.initial_capital,
                              signal_start=sig_start)
        result, bench = aligned[0], aligned[1]
        cmp_results = dict(zip(cmp_results, aligned[2:]))
        if bench_strat_name is not None:
            bench = cmp_results[bench_strat_name]

    return {
        "result": result, "bench": bench, "bench_dividends": bdiv,
        "prices": universe,
        "cash": cash_px, "quality": quality, "cfg": cfg,
        "params": dict(params), "strategy_key": strat_key, "mode": mode,
        "exog": exog_aligned, "exog_raw": exog_raw, "exog_report": ex_rep,
        "weights_report": w_report, "rebalance_dates": rebal_dates,
        "weights": weights,
        "sleeves": sleeve_report,
        "cash_series": cash_series,
        "volume": vol_px,
        "dividends": div_px,
        "open": open_px,
        "fx": fx_book,
        "construction": construction,
        "classes": (dict(class_map) if class_map else PRESETS.classes_for(
            spec.preset if spec.preset != "Custom" else "", list(universe.columns))),
        "market": market,
        "bench_mode": bench_mode,
        "bench_frictions": (list(bench_fric) if bench is not None
                            and bench_strat_name is None else []),
        "compare": cmp_results,
        "bench_strategy": bench_strat_name,
        "raw_start": raw_start,
        "trimmed": bool(cfg.engine.trim_warmup and result.equity.index[0] > raw_start),
        "stamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notices": notices,
    }


def _replace(spec: RunSpec, **changes) -> RunSpec:
    from dataclasses import replace
    return replace(spec, **changes)
