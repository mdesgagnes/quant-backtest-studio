# Changelog

## Unreleased: `feature/symbol-impact-view` (2026-11-15), fourth round

### Added
- **Funded from.** In Add mode the weight can come from every holding (pro rata, as before), from the **bonds**, from the **equities**, or from **chosen holdings** only, each reduced pro rata within that group while the rest of the portfolio is untouched. The sidebar shows the change (XBB.TO 40% -> 30%). Taking a holding's whole weight is the same as Replace. Works for profiles, blends and several portfolios at once (bonds / equities are resolved per portfolio), in the profile sweep (n/a where the group is not held) and on the command line (`fund_from: [bonds]`). Demo 8: DBMF and KMLM taken from bonds in Balanced. 2 tests.


### Changed
- **Faster loading.** A run loads prices only for the tickers it needs (about 20 for two ETFs) instead of the whole 44-ticker universe plus those. The T-bill, CPI and Ken French series are loaded once per server (`bundle.load_macro`), not again with every new set of tickers. A data host that does not answer (FRED and the Ken French library on some networks) is remembered for 24 hours (`data/cache/macro/_unreachable.json`) instead of costing a timeout on every load; the timeout is 10 seconds instead of 20. Measured here with prices on disk: price loading for a two-ETF run went from about 20 seconds to 0.5 to 1.2 seconds. The command line loads the same way.


### Changed
- **The benchmark is the portfolio by default, everywhere.** Relative statistics (excess return, tracking error, information ratio, capture, beta), the factor regression, the excess-return correlation and the Fund Review now compare each security with the portfolio it is added to. Before, they used each security's sleeve benchmark (USMV against SPY while the portfolio was XIC.TO). Advanced keeps the sleeve benchmark, or any ticker, as an option. Fund Review lists the portfolio over each security's period instead of a sleeve benchmark.
- **One measurement standard with the Backtest (CIPM).** Standalone figures now come from `qbt.metrics.summary`, the function behind the Backtest's statistics, so the two modules cannot disagree. Consequences: Calmar nets off the T-bill (Bacon); VaR and CVaR are positive losses (the CVaR objective metric now reads "lower is better"); return added is the geometric excess of with over without, (1 + with) / (1 + without) - 1, in the summary, scores, add-weight grid, profile sweep and stress windows; regime returns link the regime's months geometrically and are annualised (n/a under 12 months) instead of arithmetic monthly averages; excess-return correlation and equivalence tests use geometric excess; beta and up / down beta are withheld below R2 0.70 and R2 is shown.
- **Tables on the page are formatted like the Backtest's**: right-aligned figures that sort by value, every change coloured teal when it helps the investor and rust when it hurts (a fall in volatility is teal, a fall in return rust), security names in bold. "Sleeve Comparison" is now "Relative to Benchmark".
- **Excel: the Backtest's full report.** "Full report (Excel)" builds, on click, the same workbook as the Backtest's return-stream export (`excel_export.workbook_from_returns`): two-page executive summary for the CIO, About sheet indexing every table with links, statistics, CIPM appraisal, trailing periods, calendar years, drawdowns, monthly returns and the in-depth performance analysis -- for the portfolio with the security against the portfolio without it -- followed by every Security review table. With several results, "Report on" picks the one the summary describes. The command line writes the same file.
- Calibration and scenario-matrix reports regenerated under the new definitions.


### Changed
- **The Security review page starts from the security, not the profile.** Inputs, in order: the securities (tickers, or uploaded return streams); what to add them to (a benchmark ticker or several, a blend of tickers, an uploaded portfolio return stream, or one or more investor profiles); how much (Add at a weight, or Replace a holding); and an optional objective.
- **The result is a summary first**: one sentence per security and portfolio (change in return, volatility, max drawdown and Sharpe, correlation, stress windows helped), one sortable "What adding it changes" table, a before-and-after table when there is a single result, and growth and drawdown charts. Every earlier table is still there, behind **Details** and in the workbook. Sentences are shown only for up to 4 results; beyond that, the table alone.
- With no objective, no verdict or score is shown; the workbook's Verdict sheet still scores every objective. With an objective, each sentence ends with the verdict, its confidence and the trade-off.
- The workbook gains a **Summary** sheet first. With several portfolios, each detail table's title names its portfolio.
- Command line: `benchmarks:`, `blend:`, `profiles:`, `portfolio_returns:`, `candidate_returns:` and `returns_currency:` request keys; `objective:` is optional. Two new demos (6: three securities added to an uploaded portfolio; 7: one security against three benchmarks).

### Added
- `bundle.with_streams`: uploaded monthly return streams (CAD, or USD converted month by month) used as if they were tickers, on a copy of the price bundle.
- Stress windows fall back to month-end points when a series has no daily prices; windows shorter than two months are skipped for it (`Basis` column: daily or month-end).
- `engine.run_many` (the same candidates on several portfolios) and `CandidateResult.vs_base` (beta, capture and correlation against the portfolio itself).
- 4 tests (uploaded streams and USD conversion, month-end stress fallback, every kind of portfolio with no objective, the verdict in the summary).

## Unreleased: `feature/security-decision-engine` (2026-11-15)

### Added
- **Security review workspace** (`qbt/security_review/`): pick a preset investor profile (or a custom one), an objective and one or more candidates; get a verdict (Add / Consider / Do not add) with confidence, a ranking with the score breakdown, and the evidence behind it.
- **Preset profiles** (`config/profiles.yaml`): 100% Equities, Max Growth, Growth, Balanced, Conservative, Income, 100% Bonds, built from shared building blocks (XIC.TO, VFV.TO, XEF.TO, XEC.TO, XBB.TO) with proxy splicing (EWC, SPY, EFA, EEM converted to CAD) before each ETF's inception, flagged month by month. Rebalancing monthly / quarterly (default) / annual / none. Custom profiles typed or uploaded, and saved to `config/profiles_custom.yaml`.
- **Candidate universe** (`config/universe.yaml`): 44 ETFs in 7 sleeves, each with a default sleeve benchmark.
- **15 objectives** (`config/objectives.yaml`) with metric weights, scales, thresholds, guardrails and plain-language descriptions.
- **Scenarios** (`config/scenarios.yaml`): 9 historical stress windows (daily) and 5 monthly regimes with YAML rules.
- **Analysis engine**: standalone profile, within-sleeve statistics (active return, tracking error, IR, capture, up/down beta, hit rate, relative drawdown), excess-return and crisis correlation, factor/style regression on ETF spreads (`config/factors.yaml`), portfolio impact in Add or Replace mode, block-bootstrap risk of ruin and decumulation, rolling-window and add-weight robustness, functionally-equivalent detection, profile sweep.
- **Excel export** in the backtest workbook's style: Verdict, Ranking, Fund Review, Sleeve Comparison, Diversification, Portfolio Impact, Scenarios, Regimes, Factor Exposures, Robustness, Profile Sweep, Methodology, Series. The page and the workbook render the same table list.
- **Command line**: `python -m qbt.security_review <request.yaml>`; five demo requests in `examples/`.
- **Scripts**: `scripts/calibration.py` (writes `docs/CALIBRATION_REPORT.md`), `scripts/scenario_matrix.py` (writes `docs/SCENARIO_MATRIX.md` and `examples/scenario_matrix.xlsx`).
- **Disk cache** for downloads (`data/cache/`, git-ignored).
- **Feature flags** (`config/features.yaml`) for the sidebar workspaces.
- **Fund costs and liquidity** from yfinance (expense ratio, AUM, average daily value traded), shown in Fund Review and Ranking, used to say which equivalent fund is cheaper and more liquid, and flagged when trading is thin.
- **Ken French factors** (SMB, HML, Mom, RMW) read directly from the library's CSV zips when reachable, with the ETF spreads as fallback (`config/factors.yaml: source`).
- 45 tests in `tests/test_security_review.py`, and one UI test for the hidden workspaces.
- Docs: `docs/TESTING_GUIDE.md`, `docs/METHODOLOGY.md`, `docs/DECISIONS.md`, `docs/CALIBRATION_REPORT.md`, `docs/SCENARIO_MATRIX.md`, `docs/ARCHITECTURE_AUDIT.md`.

### Changed
- The sidebar lists Security review first, then Backtest.
- **Markets and Research are hidden** (`enabled: false` in `config/features.yaml`); no code was removed. See `docs/DECISIONS.md`.
- `.gitignore`: ignores `data/cache/`; keeps `examples/*.xlsx`.

### Fixed
- No existing calculation was found to be wrong. One bug in the new code was caught by its tests before release: a month with no price would have let the next monthly return span two months (`data.month_end_prices` now keeps the gap).
