# Decisions

Every judgment call made on `feature/security-decision-engine`, why, and how to change it.

## Architecture

### Module placement
**Decision.** The decision engine lives in a new package, `qbt/security_review/`. The return-stream building blocks (`qbt/returns_input.py`, `qbt/fund_review.py`, `qbt/clustering.py`) stay where they are and are re-exported from `qbt.security_review` (`from qbt.security_review import fund_review`).
**Why.** The audit showed these modules are already UI-agnostic; only their page lives inside the Backtest workspace in `app.py`. Moving the files would break `app.py`, the tests and any script you have, for no functional gain. The new package gives one entry point for the new work and is ready for the planned non-Streamlit UI (the page, `qbt/security_review/ui.py`, only renders `report.sections`).
**Change it.** Move the three files into the package and leave one-line shims at the old paths (`from .security_review.fund_review import *`).

### Config folder
**Decision.** New YAML files are in `config/` (as the brief names them: `config/profiles.yaml`, ...), next to the existing `configs/` folder, which holds backtest and clustering examples.
**Why.** The brief named the path explicitly. **Change it.** Set `CONFIG_DIR` in `qbt/security_review/config.py`.

### One table list for page and workbook
**Decision.** `report.sections(review)` returns every table; the page and the Excel export both render that list.
**Why.** It guarantees "every metric in the app is in the export, and vice versa" by construction, and a test checks the sheets and every metric key.

### The page starts from the security
**Decision.** Inputs in the order: securities, what to add them to (benchmark, blend, uploaded portfolio or profile), how much, optional objective. The result opens with one sentence per security and portfolio and one summary table; every other table sits behind Details and in the workbook. Sentences only for up to 4 results.
**Why.** The question is "what does this security do to my portfolio", against whatever portfolio the PM has; the profiles are one possible portfolio, not the starting point. The scored list of every metric was the noise. **Change it.** `report.summary_sections` (the summary) and `ui.render` (the order).

### The portfolio is the benchmark
**Decision.** Relative statistics, factor regression, excess correlation and Fund Review compare each security with the portfolio it is added to, unless the user picks the sleeve benchmark or a ticker.
**Why.** The question is what the security does to that portfolio; comparing USMV with SPY while the portfolio is XIC.TO answered a different question. The calibration script keeps the sleeve benchmark explicitly, since its checks are about each fund against its sleeve. **Change it.** `Request.benchmark` (None, "sleeve" or a ticker).

### One measurement code path with the Backtest
**Decision.** `stats.standalone` calls `qbt.metrics.summary`; the full Excel report is built by `excel_export.workbook_from_returns`, the Backtest's own return-stream export, with the portfolio with the security as the series and the portfolio without it as the benchmark.
**Why.** Two implementations of the same statistic drift apart (Calmar without the risk-free, CVaR with the opposite sign, arithmetic regime averages); one code path cannot. A test checks the figures are identical.

### Objective optional
**Decision.** With no objective, no verdict or score is shown; the engine still scores every objective (the first stands in internally for the ranking and robustness tables, which are hidden).
**Why.** A verdict needs a stated goal; without one, facts are more honest than a default score.

### Uploaded streams in stress windows
**Decision.** Month-end points, and windows shorter than 45 days skipped, when an uploaded monthly series is involved.
**Why.** A monthly series has no daily path; a two-week crash measured month end to month end mostly measures the rebound. **Change it.** `min_monthly_days` in `scenarios.stress_table`.

### Security review is the first workspace
**Decision.** The sidebar shows Security review first, then Backtest.
**Why.** It is the purpose of this branch and what you will test first. **Change it.** Reorder `_ALL_WORKSPACES` in `app.py`.

## Data

| Decision | Why | Change it |
|---|---|---|
| Total return = Yahoo `Adj Close` (`auto_adjust=False`), `Close` and `Dividends` kept for the yield | Adj Close is the dividend-reinvested series; Close is needed for a distribution yield on the price actually paid | `data.yahoo_downloader` |
| Monthly frequency for every statistic except stress windows | Toronto and New York close together, but monthly is the convention for comparing funds, removes daily noise and matches the FX and CPI data. Stress windows of a few days need daily data | n/a (design) |
| In-progress month dropped; an ETF's first partial month dropped | A partial month is not comparable | `data.month_end_prices` |
| A month with no price stays NaN, and so does the next month's return | Otherwise the next return would silently span two months (bug found and fixed while writing tests) | `data.month_end_prices` |
| Price levels carried over an exchange holiday inside stress windows; returns never filled | The market was shut, so its return really was 0; a missing month is never filled | `scenarios.stress_table` |
| USD to CAD at the daily close of Yahoo `CAD=X`, before 2003-09-17 the stored Fed H.10 series (`data/fx`), carried forward at most 5 sessions | Reuses `qbt/fx.py` exactly as the backtester does | `data.to_cad` |
| A CAD-listed fund is never converted (XEF.TO keeps its own hedging policy) | It is quoted in CAD | `data.currency_of` |
| Only CAD and USD quotes supported | The universe has nothing else; anything else is refused with a message, not guessed | `data.to_cad` |
| Disk cache in `data/cache/` (CSV per ticker), refreshed after 3 days; on a failed download a stale copy is used and flagged | Repeat runs (matrix, demos) in seconds; works offline | `data.MAX_AGE_DAYS` |
| Risk-free = Canadian 3-month T-bill (Bank of Canada V122531), each month earns the yield known at its start | CAD investor; the US T-bill would flatter low-beta products (see `qbt/boc.py`) | `data.cad_tbill_monthly` |
| Failed tickers are listed and skipped | Brief: flag and skip, never fill | `data.fetch_prices` |

All 44 universe tickers downloaded on 2026-11-15; none failed.

## Profiles and splicing

- **US equity block = VFV.TO** (Vanguard S&P 500, unhedged, CAD-quoted). XUS.TO would do as well; VFV.TO has the larger AUM. Change `building_blocks.us_equity.ticker`.
- **Proxies.** XIC.TO -> EWC (MSCI Canada, USD since 1996, converted to CAD); VFV.TO -> SPY; XEF.TO -> EFA; XEC.TO -> EEM; XBB.TO none (own history from 2000). EWC was preferred to `^GSPTSE` because the index is price-only (it would understate return by the dividend yield). `^GSPC` is price-only for the same reason; SPY is a total-return series back to 1993.
- **Splice rule.** The proxy's CAD return is used for every month before the block's first own monthly return; nothing is rescaled. Spliced months are listed per holding (Methodology sheet and "Profile holdings and proxy splices"), and every candidate whose analysis period overlaps a spliced stretch gets a flag.
- **Rebalancing** default quarterly, at the end of March, June, September and December; drift between dates.
- **Custom profiles** are saved to `config/profiles_custom.yaml` (created on first save) and appear in the list after the presets.

## Engine

| Decision | Why | Change it |
|---|---|---|
| Add mode funds the candidate pro rata from every holding | Brief default; keeps the profile's internal mix | `portfolio.add_weights` |
| Replace mode gives the replaced holding's whole weight to the candidate | The clean within-sleeve test (e.g. USMV for VFV.TO) | `portfolio.replace_weights` |
| Default analysis period = months common to every candidate, the profile and the benchmarks; option "each candidate's own history" | Fair ranking needs the same months; a 2023 launch would otherwise truncate everyone, hence the option. The scenario matrix uses "each" | Request `alignment` |
| Sleeve benchmark from `universe.yaml`; real assets and alternatives have none, so the profile itself is the benchmark | No natural benchmark for gold or managed futures; flagged | `universe.yaml` |
| Benchmarks are never spliced | A benchmark with a proxy would make an ETF and its proxy identical before inception | `engine.run` |
| Worst 10% months = the profile's (without candidate) worst decile on the analysis period | "Behaviour in the existing portfolio's worst periods" | `impact._tail_mask` |
| Factor model: Ken French (SMB, HML, Mom, RMW) when reachable, ETF long-short spreads otherwise; low volatility always USMV - SPY; market = sleeve benchmark | Ken French has no low-volatility factor. The library is read directly from its CSV zips with the standard library rather than through `pandas-datareader`: same files, same server, no new dependency (that package is not installed here and is lightly maintained). On this network Dartmouth resets the connection, so the ETF proxies are what runs here; the source used is stated in the Methodology sheet. US style factors are used for every sleeve: a known limitation for Canadian and international funds | `config/factors.yaml: source` |
| A candidate that is itself a leg of an ETF-proxy factor (SPY, USMV, QUAL, MTUM, IWM...) is flagged: its loadings are mechanical. Ken French factors are not ETF legs, so they raise no such flag | Avoids reading USMV's 1.0 low-vol loading as evidence | `engine._flags` |
| Fund costs and liquidity from yfinance: expense ratio (`funds_data.fund_operations`, else `info.netExpenseRatio`), AUM, average daily value traded; cached 30 days | Shown in Fund Review and Ranking; used to say which of two functionally equivalent funds is cheaper and more liquid; a fund trading under $1M a day is flagged. **Not scored**: Yahoo's adjusted prices are already net of fees, so a cost metric would count the fee twice | `config/factors.yaml: min_avg_dollar_volume` |
| Mega-cap tilt = slope of active return (vs sleeve benchmark) on QQQ minus RSP | Simple, transparent proxy for concentration in mega-cap growth | `factors.yaml: megacap_spread` |
| Currency sensitivity = slope on the monthly USD/CAD change, controlling for SPY in USD | Univariate beta would mix equity and currency, since CAD falls when equities fall | `impact.impact_metrics` |
| Distribution yield = trailing 12-month distributions / last close; stability = coefficient of variation of the last 5 calendar-year yields | Simple and available for every ticker. Yahoo does not split return of capital, so ROC is flagged (a) from `universe.yaml` (option writing, leverage) and (b) when the yield exceeds the 5-year total return by over 2 points | `data.distribution_yield`, `engine._flags` |
| Inflation: FRED CPIAUCSL first, then Bank of Canada CPI (V41690973, from 1995), then no inflation regime | FRED was unreachable here, so the **Canadian** CPI is what the app uses on this network. For a CAD investor this is arguably the better series. Months before 1996 have no inflation label (not filled) | `scenarios.yaml: high_inflation.sources` |
| 10-year yield regime on the US 10-year (^TNX) | Available daily on Yahoo to 1962; Canadian yields move with it. Change `series` to another Yahoo symbol | `scenarios.yaml` |
| Equity trend regime lags one month; volatility regime uses an expanding median of past months only | No look-ahead | `scenarios.yaml` |
| Stock-bond-down months = the 100% Equities profile and the 100% Bonds profile both negative, in CAD | The investor's own sleeves, not US indices | `scenarios.yaml` |
| Stress window returns: close of the start date to close of the end date, buy-and-hold at target weights | Short windows: rebalancing inside them would be fiction | `scenarios.stress_table` |
| Risk of ruin and decumulation: circular block bootstrap, 12-month blocks, 2000 paths, seed 42; the same random paths with and without the candidate | Common random numbers: the difference is the candidate, not simulation noise | `objectives.yaml: simulation` |
| Decumulation: 4% of initial wealth a year, paid monthly, raised 2% a year, 30 years, nominal | Classic 4% rule with inflation indexation; nominal ending wealth | `objectives.yaml: simulation.decumulation` |

## Verdicts

- **Scores are linear and clipped** to [-1, +1] with a centre and a "full credit" scale per metric (`objectives.yaml: metrics`). No optimiser, no model: every number can be traced in the "Score breakdown by component" table.
- **Scales were set before looking at results** from what a portfolio manager would call a material change at a 10% weight (e.g. 1 point of volatility, 2 points of drawdown, 0.10 Sharpe, 5 points of ruin probability). None were changed after running the calibration or the matrix.
- **Thresholds:** Add >= +0.30, Consider >= +0.05, else Do not add, for every objective (overridable per objective with a `verdict:` block).
- **Guardrail:** by default an Add that costs more than 1.0% a year of CAGR is capped at Consider (Upside participation also caps a drawdown deepened by more than 3 points; Income caps a CAGR loss above 0.5%).
- **Robustness cap:** an Add whose objective improves in fewer than 60% of rolling 36-month windows becomes Consider.
- **Rolling metric per objective** is a cheap window statistic close to the objective (e.g. `d_vol` for lower volatility, the regime change for the regime objectives). For risk of ruin it is `d_mdd`; for sequence risk `d_sortino`: a window cannot run a 30-year simulation.
- **Confidence:** High; Moderate below 60 months of overlap; Low below 36; one level down when the rolling share is below 60%, when the verdict flips between Add and Do not add across the 5/10/20% add weights, or when the candidate's history covers fewer than half of the 9 stress windows; Low when over half of the objective's weight has no data.
- **Stress-coverage rule (added after the first demo run).** Young alternatives scored Add with High confidence on histories that miss most of the stress windows (KMLM from 2020-12 covers 4 of 9; RSST/RSSB 2 of 9). A verdict resting on one stock-bond crisis should not read "High". The rule lowers confidence only; it never changes a verdict or a score, and no calibration check depends on it. DBMF (from 2019-05) covers 5 of 9 windows, including COVID and 2022, so it keeps High; the threshold was not moved to catch it. Change `robustness.min_stress_coverage` in `objectives.yaml`.
- **Ties** at a saturated score (+1.00) are ranked by the unclipped score (same weights, no clip), shown next to it.
- **Functionally equivalent:** tracking error between the two < 1% a year, and either their excess returns over the sleeve benchmark correlate above 0.95, or (when one of them barely departs from the benchmark, so excess correlation is noise) their raw correlation exceeds 0.95. The brief's rule taken literally would never fire for two index trackers, whose excess returns are pure noise.
- **Missing metrics** (e.g. no 2022 data for a 2023 launch) are left out and the weights renormalised; the share left out is reported and lowers confidence.

## Research and Markets

| Workspace | Purpose | Overlap with LSEG Workspace | Maintenance cost | Recommendation |
|---|---|---|---|---|
| Markets (`qbt/monitor.py`, `snapshot*`, `groups*`, `tvchart.py`, `sortable.py`; `app.py` ~800-1250) | Watchlist: returns over ten horizons, risk, relative strength, rolling correlation, seasonality, TradingView chart | High: quotes, performance tables, charts, correlation and screening are core LSEG Workspace functions with better data | Medium: depends on Yahoo endpoints and an embedded TradingView widget | **Hide** (`enabled: false`) |
| Research (`qbt/lab/*`, ~3,000 lines, CLI and MCP server) | Factor library, multiple-testing corrections, deflated Sharpe, PBO, purged CV, run cards, decay monitor | Low: LSEG does not do backtest-overfitting statistics | High: large surface, specialised, not used for ETF selection decisions | **Hide** from the sidebar; the CLI `python -m qbt.lab` and its tests still work |

Both are hidden with `config/features.yaml`; no code was deleted. Set `enabled: true` to bring either back. The Backtest workspace, including Return stream, Fund review and Diversification, is unchanged.

## Not done, or done differently

- **Ken French factors**: implemented (direct CSV, tested on a sample file) but unreachable from this network, so not verified live; ETF proxies run here.
- **FRED**: unreachable here; the loader is implemented and tried first, so on a network that reaches FRED, US CPI is used automatically (cached in `data/cache/macro`).
- **Fund costs (MER) and liquidity** are shown and compared but not scored (see Engine above).
