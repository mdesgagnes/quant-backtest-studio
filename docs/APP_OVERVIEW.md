# Quant Backtest Studio: where the app stands

Written 2026-11-13 as a starting point for a new session. It describes the
**Streamlit app on `main`**, which is the app in use. Read section 1 first:
it says which branch is live and what the parked redesign branch is.

## 1. Branches, and what is live

| Branch | Commit | What it is |
|---|---|---|
| `main` | `8c170f5` "Markets: Snapshot, Screener and Groups tabs" | **The live app.** Streamlit. Unchanged since before the redesign work |
| `ui/terminal-redesign` | `e82f234` | A second interface (FastAPI + React) over the same engine. Complete, pushed, **parked**. The owner tried it on Render and chose to stay on Streamlit |

Facts checked against GitHub on 2026-11-13:

- `origin/main` is still `8c170f5`. Nothing from the redesign was merged or
  pushed to `main`, so the Streamlit deployment is exactly as it was.
- The redesign branch **adds** files (`server/`, `web/`, `tests/golden/`,
  `docs/`, `deploy/studio/`, `tools/`, `CLAUDE.md`, `pytest.ini`,
  `.gitattributes`, `.dockerignore`, `.claude/`) and changes one existing
  file, `.gitignore` (two added lines). It does not modify `app.py`,
  anything under `qbt/`, `requirements.txt`, `.streamlit/`, `README.md`,
  `Dockerfile` or `render.yaml`.
- Repository: `https://github.com/mdesgagnes/quant-backtest-studio`.

**To work on the Streamlit app, work from `main`.** Do not merge the
redesign branch into `main`: its `CLAUDE.md` contains rules written for the
redesign ("never edit `app.py`", "work on the redesign branch only") that
are wrong for Streamlit work. Section 9 lists the parts of that branch
worth bringing over on their own.

Deployment: the repo is set up for Streamlit Community Cloud, which
redeploys on every push to `main`. `Dockerfile`, `render.yaml` and
`deploy/huggingface_spaces.md` are alternatives. The repo does not record
the live URL.

## 2. What the app is

A web app for backtesting systematic strategies, used daily by a portfolio
manager. Data comes from Yahoo Finance or uploaded CSV / Excel files. It
accepts signal files (economic and fundamental data) and target-weight files
from existing strategies, and exports tearsheets, holdings, equity curves
and a full Excel report.

- `qbt/` is the engine and everything computational: pure functions, usable
  with no interface (`run_from_config`, `python -m qbt.lab`, an MCP server).
- `app.py` is the Streamlit interface: 6,247 lines, one script.
- `README.md` (1,781 lines) is the user and methodology manual. It is
  accurate and detailed; read the relevant section before changing a
  feature.

Python 3.12, conda environment `quant`
(`C:\Users\marcd\miniconda3\envs\quant\python.exe`). Versions in that
environment: Streamlit 1.64, pandas 3.0, numpy 2.5, Plotly 7.1, yfinance 1.7.

```bash
streamlit run app.py          # http://localhost:8501
pytest -q                     # 235 tests on main, about 2.5 minutes
```

## 3. Workspaces and tabs

The sidebar's first control switches workspace. Each has its own sidebar.

| Workspace | Tabs | Code |
|---|---|---|
| **Backtest** | Results, Compare, Signals, Positions, Attribution, Tax, Robustness, Data, Builder, Export | `app.py:1175-1247` (presets, config import), `2379-3408` (sidebar), `3410-3776` (run), `3811-6246` (tabs) |
| **Return stream** (Backtest with Source = "Return stream") | Results, Stress tests, Market regimes, Robustness, Export, Fund review | `app.py:1252-2376` |
| **Markets** | Snapshot, Screener, Groups, Chart, Performance, Risk, Relationships, Seasonality, Data | `app.py:791-1172`, `qbt/snapshot_ui.py`, `qbt/groups_ui.py` |
| **Research** | Start here, Ledger, Factor Lab, Strategy Lab, Risk X-Ray, Event Study, Monitor, Assistant | `qbt/lab/ui.py` |

Backtest tab line ranges in `app.py`: Results 3861, Compare 3996, Signals
4069, Positions 4180, Attribution 4605, Tax 4836, Robustness 5056, Data
5429, Builder 5497, Export 5667.

A complete feature list, 259 numbered items with source lines, is in
`docs/ui-parity-checklist.md` on the redesign branch (section 9).

### Backtest sidebar, in order

Presets (save, load, delete, export, restore) and YAML configuration import;
data source (Yahoo Finance, file, return stream); preset universe, symbols,
dates, or price-file upload; investable universe; comparison benchmark
(ticker, blend of several, or another strategy), its convention and
frictions; price convention; cash remuneration; base currency and overrides;
exogenous series upload and publication lag; signal origin (built-in model
or imported target weights); model and parameters; portfolio construction
(single, by asset class, core + strategy, blend of strategies); Execution
(rebalance, trading-day rule, lag, price, warm-up, capital, leverage, whole
units, cash sweep); Compare strategies; Frictions (commission, slippage,
market impact, overrides, cash rate, management fee, currency conversion).
Then **Run backtest**.

### Models (21, `qbt/strategies/`)

Accelerating Dual Momentum, Buy & Hold, Cross-Sectional Momentum, Custom
Formula, Dual Momentum (Relative + Absolute), Factor Zoo, Fixed Weights,
Imported Factor Ranking, Imported Signals, Macro Gate, Mean Reversion (RSI),
MomentumOG, Quantitative Momentum (Gray & Vogel), Risk Parity, Time-Series
Momentum, Trend Filter (Moving Average), Trend-Gated Target Weights
(HIDE-style), Trend Quality, Turtle Breakout, VIX-Adaptive Momentum,
Volatility Target. Each declares typed parameters (`qbt/strategies/base.py`)
that the sidebar renders generically. Adding one is README section 6.

## 4. How a run works (`app.py:3410-3776`)

1. Assemble a `RunConfig` from the sidebar (`qbt/config.py`: `DataConfig`,
   `ExogConfig`, `StrategyConfig`, `EngineConfig`, `CostConfig`) and
   validate it. Blocking problems stop the run; warnings do not.
2. Load prices: `load_market_data` (Yahoo, cached one hour) or
   `price_import.read_files` + `to_market` for uploads. Both produce the
   same `MarketData` object (close, and optionally open, high, low, volume,
   dividends, adjusted close).
3. If a base currency is set, convert through an `FxBook` (`qbt/fx.py`).
4. `clean_prices`, then align exogenous series with the publication lag
   (`qbt/exog.py`).
5. Generate target weights: the model's `generate`, or `allocation.resolve`
   for sleeves, or `prepare_target_weights` for an imported file.
6. `run_backtest` (`qbt/engine.py`): simulates share counts and cash one
   session at a time, with execution lag, drift, frictions, dividends,
   fees, cash sweep and per-currency cash.
7. Comparison strategies run the same way; with warm-up trimming all start
   on the latest date any first holds a position.
8. Benchmark: `benchmark_result` or `blended_benchmark`, optionally charged
   frictions; then `align_start`.
9. Everything is stored in `st.session_state["run"]`: `result`, `bench`,
   `prices`, `cfg`, `params`, `weights`, `quality`, `exog`, `sleeves`,
   `dividends`, `open`, `volume`, `fx`, `compare`, `classes`, `market`,
   `stamp` and more. Every tab reads from it.

`BacktestResult` (`qbt/engine.py:66`) holds equity, returns, weights, target
weights, turnover, costs, exposure, cash weight, rebalance dates, trades,
and optionally dividend income, shares, fees, contributions, cash flows,
foreign cash and FX costs.

## 5. The `qbt` package

| Module | Role |
|---|---|
| `engine.py` | The simulator, benchmarks, warm-up trimming, alignment, windows |
| `config.py` | Run configuration dataclasses, YAML, validation |
| `data.py` | Yahoo downloads, file loading, price cleaning, quality report |
| `price_import.py` | Reads uploaded price files in any supported layout |
| `exog.py`, `external.py` | Signal files and imported target weights |
| `strategies/` | Model registry (`REGISTRY`), base classes, the models |
| `allocation.py` | Sleeves: asset-class budgets, core + strategy, blends |
| `schedule.py` | Rebalance calendars and trading-day rules |
| `costs.py`, `fx.py`, `boc.py` | Frictions; currencies and rates; Bank of Canada yields |
| `metrics.py`, `cipm.py` | Statistics, period tables, drawdowns; CIPM appraisal |
| `attribution.py` | Contribution and Brinson-Fachler attribution, four linking methods |
| `robustness.py`, `stress.py`, `regimes.py` | Walk-forward, sweeps, Monte Carlo, cost sensitivity; stress periods; market regimes |
| `tax.py` | Canadian non-registered tax analysis (ACB) |
| `nextrade.py` | Next rebalance trades and fitting an actual account to target |
| `compare.py`, `monitor.py`, `snapshot.py`, `groups.py` | Strategy comparison; Markets analytics |
| `formula.py`, `rules.py` | Expression language and the rule Builder |
| `returns_input.py` | Forgiving reader for return-stream files |
| `fund_review.py`, `fund_review_export.py` | Beta-matched replica test; PDF, PowerPoint, Excel |
| `charts.py`, `brand.py`, `tvchart.py` | Plotly figures; palette (National Bank red, black, sand); TradingView chart |
| `report.py`, `full_report.py`, `report_charts.py` | Tearsheet; detailed report (HTML and PDF) |
| `excel_export.py`, `exec_summary.py`, `analysis_sheet.py` | The Excel workbook and its CIO summary |
| `store.py`, `presets.py`, `names.py` | Saved presets (JSON on disk); preset universes; ticker names |
| `lab/` | Research workspace: factors, statistics, validation, ledger (SQLite), run cards, event study, x-ray, decay monitor, CLI, MCP server, assistant |
| `snapshot_ui.py`, `groups_ui.py`, `lab/ui.py` | Streamlit views kept out of `app.py` |

## 6. How the interface is built

- **Styling**: one CSS block at `app.py:72-360`, palette from `qbt/brand.py`
  (the single place colour is defined; charts and reports read it too).
- **Helpers** (`app.py:402-763`): `dial` (KPI card), `eyebrow` (section
  heading), `note` (folds after the first sentence), `signed` and `heat`
  (table colouring), a `st.dataframe` wrapper that right-aligns figures,
  and the `@st.cache_data` loaders.
- **Caching**: Yahoo data one hour, regime data and Bank of Canada yields
  six hours, uploaded files by content. The next-rebalance replay is cached
  per run in the session. Nothing else is cached.
- **Session keys worth knowing**: `run`, `loaded_cfg`, `loaded_rules`,
  `sweep`, `daysweep`, `mc`, `report_html`, `report_pdf`, `rule_list`,
  `_my_units`, `_my_cash`, `mk_data`, `gr_data`, `fr_out`, `lab`,
  `auth_ok`.
- **Password**: optional, from `st.secrets["password"]`.
- **Exports**: about 45 download buttons. The Excel workbook is built only
  when clicked (`_deferred_book`). Full list in README section 4.
- **The Export tab reads other tabs' variables through `globals()`**
  (`wf`, `ios`, `cs`, `_score`, `ev`, `_periods`, `tax_rep`, `battr_cfg`).
  The tabs must therefore run in order, every time. Moving a tab's code
  into a function or a fragment breaks this unless those values are passed
  explicitly.

## 7. Known issues, measured

All measured on 2026-11-12 on the development machine. Full figures are in
`docs/perf-baseline.md` on the redesign branch.

### Every interaction re-runs everything

Streamlit re-executes the whole script on each interaction, so all ten
Backtest tabs are rebuilt each time. On a 4-instrument, 13-year run:

| Action | Server time |
|---|---|
| Run backtest and render | 28 to 29 s |
| Any control with a result on screen (even Log scale) | 15 to 26 s |
| The simulation alone, without the interface | 1.4 to 1.9 s |

Where one rerun goes:

| Work | Share | Fix available without changing results |
|---|---|---|
| `R.cost_sensitivity` (`app.py:5264`) | 26 to 31% | Cache per run: it depends only on the run |
| `REPORT.render_tearsheet` (`app.py:5677`), rebuilt for a download button | 24 to 27% | Pass a callable, as the Excel button already does |
| Building 26 Plotly figures | 8 to 13% | About 0.2 s each is Plotly applying its theme |
| `ANALYSIS.build_analysis` (`app.py:6195`), for the Excel report | 4 to 6% | Move inside the deferred workbook builder |
| `R.walk_forward`, `R.in_out_sample` (`app.py:5076`, `5094`) | 7 to 9% | Cache per run and fold count |
| `TAX.evaluate`, attribution, stress | smaller | Cache per run and settings |

At least 60% of every click is work that depends only on the run. The
first two rows alone are about half. This is the highest-value change
available to the Streamlit app, and it was offered but never applied.

### Loading a preset or YAML configuration is partial

Verified. Applying a file to a session changes the start date, rebalance,
execution lag and commission, but **not** the symbols, investable universe,
model parameters, slippage, cash rate or management fee, and it can reset
the benchmark to None. Cause: widgets with a `key` (`tickers_box`,
`p_<model>_<param>`, `slip_bps`, `cashrate_pct`, `mgmtfee_pct`,
`base_currency_pick` and others) keep their session value, so the loaded
configuration only reaches widgets that have none. Fix: when a
configuration is loaded, delete those keys from `st.session_state` before
the widgets are created, then rerun.

### Smaller

- Stale wording: the Robustness tab says "These four tests" above seven
  sections, and its stress note says "sixteen ... episodes ... through the
  2024 yen carry-trade unwind" while there are 24 periods through 2025. The
  README says 19 models; there are 21.
- Many `use_container_width=` calls; Streamlit says the argument is being
  removed in favour of `width=`. It still works on 1.64 but logs a warning
  per call.
- The Research Assistant tab prints an MCP command with a machine-specific
  Python path.
- On a phone the six KPI cards stack one per row and fill more than a
  screen before the first chart.
- Saved presets and the Research ledger are files on the host's disk, lost
  on every redeploy of a free host. The app warns about presets only on
  Streamlit Community Cloud.

## 8. Tests

`tests/` on `main`: 235 tests in 13 files, all passing. They cover the
engine's arithmetic (`test_math.py`), CIPM measures, currencies, holdings
fitting, next-rebalance trades, fund review, snapshot and groups analytics,
and the Research lab. A few of them run Streamlit views through `AppTest`.

There are **no tests of `app.py`'s own behaviour on `main`**. The redesign
branch has them (next section).

## 9. What the redesign branch has that is useful to `main`

The interface on that branch is parked, but three things on it are about
the Streamlit app and worth bringing to `main` on their own (they add
files and touch nothing existing):

| Item | Path on `ui/terminal-redesign` | Why |
|---|---|---|
| Golden-output tests of the Streamlit app | `tests/golden/harness.py`, `scenarios.py`, `record.py`, `test_golden_legacy.py`, `fixtures/`, `expected/`, `README.md`, plus `pytest.ini` and `.gitattributes` | Five reference runs driven through the real `app.py`, locking every statistic, screen, control and export. They make changes to `app.py` (such as the performance fixes in section 7) safe to verify. Commit `e082769` |
| Feature inventory | `docs/ui-parity-checklist.md` | 259 features with source lines |
| Performance baseline and the measuring script | `docs/perf-baseline.md`, `tools/perf/measure_legacy.py` | The "before" numbers, reproducible. Commit `4a9f9d5` |

Bringing them over: `git checkout main`, then
`git checkout ui/terminal-redesign -- tests/golden/harness.py tests/golden/scenarios.py tests/golden/record.py tests/golden/test_golden_legacy.py tests/golden/make_fixtures.py tests/golden/README.md tests/golden/fixtures tests/golden/expected pytest.ini .gitattributes docs/ui-parity-checklist.md docs/perf-baseline.md tools/perf`.
Leave out `test_golden_app.py`, `test_golden_service.py` and
`shim_driver.py`: they test the parked interface. The checklist's last
section ("Status in the new app") and the golden README's mention of the
new app refer to the parked interface and can be trimmed.

Not for `main`: `server/`, `web/`, `deploy/studio/`, `CLAUDE.md`,
`docs/DESIGN.md`, `docs/HANDOFF.md`, `docs/DEPLOY.md`.

Why the redesign was parked, in the owner's words: the small visual
adjustments were liked, but the free host stops after 15 minutes idle and
felt slower than Streamlit. If it is ever revisited, `docs/HANDOFF.md` on
that branch is the entry point.

## 10. Conventions

- All logic goes in `qbt/` as pure functions; `app.py` only draws. New
  analytics get a unit test in `tests/`.
- Colour is defined only in `qbt/brand.py`. The owner's palette is National
  Bank of Canada red, black and gold; do not propose another accent.
- Text shown to users: English, sentence case, plain words, no em dashes.
  Notes explain what a figure means and what to do about it.
- Engine results are considered correct. A change that moves a figure needs
  the owner's explicit agreement.
- Pushing to `main` redeploys the live app. Work on a branch and merge when
  the owner says so.
- New dependencies must be free and maintained, and added to
  `requirements.txt` (the free host has about 1 GB of memory).

## 11. Suggested first steps for the next session

1. `git checkout main && git pull`, confirm a clean tree at `8c170f5` or
   later.
2. Decide with the owner whether to bring the golden tests to `main`
   (section 9). They are the safety net for everything else.
3. If speed matters, apply the first two fixes in section 7 on a branch,
   run the golden tests, and measure with `tools/perf/measure_legacy.py`.
4. Then the additions the owner asks for.
