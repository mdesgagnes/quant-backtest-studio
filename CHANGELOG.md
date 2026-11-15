# Changelog

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
