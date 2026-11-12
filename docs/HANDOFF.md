# Handoff: the interface redesign

Written at the end of the session that built the foundation and the
reference screen. A new session can continue from this file alone, with
`CLAUDE.md` (rules), `docs/DESIGN.md` (design system) and
`docs/ui-parity-checklist.md` (the contract).

## 1. Where things stand

Branch `ui/terminal-redesign`. Nothing is pushed or merged.

| Done | Where |
|---|---|
| Parity contract: 259 features, 5 open questions | `docs/ui-parity-checklist.md` |
| Golden references: 5 runs of the Streamlit app | `tests/golden/expected/` |
| Performance baseline of the Streamlit app | `docs/perf-baseline.md` |
| The run, without an interface | `server/pipeline.py` |
| Result cache, upload store, Yahoo cache | `server/store.py` |
| HTTP API | `server/app.py` |
| Design system | `docs/DESIGN.md`, `web/src/styles/tokens.css` |
| App shell: top bar, control panel, run header with KPI strip, tabs | `web/src/App.tsx` |
| Reference screen: Backtest > Results, desktop and phone | `server/views/results.py`, `web/src/screens/Results.tsx` |

Not done: the other 9 Backtest tabs, about half of the control panel, the
Return stream, Markets and Research workspaces, the password gate, hosting.
Each is a ticket in section 6.

The classic app is untouched: `streamlit run app.py` works exactly as
before and remains the app to use until the owner says otherwise.

## 2. Architecture, and why

```
browser (web/, React + TypeScript)        draws blocks; formats nothing
        |  JSON over /api
server/app.py (FastAPI)                   routes only
server/store.py                           uploads, run cache, Yahoo cache
server/views/<screen>.py                  builds each screen as a list of blocks
server/pipeline.py                        the run, lifted from app.py
qbt/                                      the engine, unchanged
```

**Why not keep Streamlit.** Every interaction re-executes the whole script:
15 to 26 s per click with a result on screen, of which under 2 s is ever the
simulation (`docs/perf-baseline.md`). Caching could remove most of that but
not the server round trip on every toggle, the 5 s load, or the limits on
layout and phone use.

**Why the screens are built in Python.** The risk in a rewrite is that
figures drift. So the server builds each screen as a list of *blocks* (KPI,
note, table, chart data), using the same expressions `app.py` uses, and the
browser only draws them. That makes parity testable in Python: the blocks
are compared with what the Streamlit tab showed when the goldens were
recorded. It also means most of a new screen is a copy of `app.py` code into
a view file.

**Why React and TypeScript for the drawing.** Interactions that only change
the view (log scale, range, tabs, folding, sorting) must happen in the
browser to be instant, and dense tables, editable grids and linked charts
are what this stack is good at. The front end is deliberately thin: about a
dozen block components.

**State.** One process, one user at a time, in memory, like the app it
replaces. A run is stored under the digest of its controls and files, so
the identical run returns from cache; a run on Yahoo data expires with the
hour-long price cache. The browser keeps the controls, upload ids and last
run id in `localStorage` and restores the run on reload if the server still
holds it.

### Dependencies

| Package | Why |
|---|---|
| fastapi, uvicorn, python-multipart | HTTP API, server, file uploads (`server/requirements.txt`) |
| react, react-dom | Component model |
| typescript, vite, @vitejs/plugin-react | Type-checking and build |
| @tanstack/react-query | Client cache: a screen is fetched once per run |
| echarts | One chart library, loaded on demand (Apache-2.0) |
| @fontsource/ibm-plex-sans | Self-hosted typeface, no third-party request |

Planned, not yet installed: `@tanstack/react-table` and
`@tanstack/react-virtual` for long tables (ticket B3); `echarts-gl` for the
parameter surface only (ticket B6); `lightweight-charts` for Markets candles
(ticket M2, already used by the classic app, attribution required).

## 3. How to run

Python is the conda environment `quant`:
`C:\Users\marcd\miniconda3\envs\quant\python.exe`.

```bash
# New app (serves the built front end at http://localhost:8000)
cd web && npm install && npm run build && cd ..
python -m uvicorn server.app:app --port 8000

# Front-end development with hot reload, http://localhost:5173
python -m uvicorn server.app:app --port 8000      # terminal 1
cd web && npm run dev                             # terminal 2

# Classic app, http://localhost:8501
streamlit run app.py
```

`QBT_LEGACY_URL` sets where "Open the classic app" points (default
`http://localhost:8501`). `.claude/launch.json` has both servers for the
preview pane.

Tests:

```bash
pytest tests/golden/test_golden_service.py tests/test_server_api.py -q   # 20 tests, about 15 s
pytest tests/golden -q                                                   # adds the Streamlit replays, about 4 min
pytest -q                                                                # everything, about 7 min
cd web && npm run build                                                  # type-check + build
```

## 4. The pattern for a screen

Do these in order. Results is the worked example for every step.

1. **Read the tab in `app.py`.** Line ranges: Compare 3996, Signals 4069,
   Positions 4180, Attribution 4605, Tax 4836, Robustness 5056, Data 5429,
   Builder 5497, Export 5667.
2. **Write `server/views/<tab>.py`** with `build(v: RunView) -> List[Block]`.
   Copy the tab's code; replace each Streamlit call with a block
   constructor from `server/views/common.py`:

   | Streamlit | Block |
   |---|---|
   | `note(...)` | `note(html)` |
   | `st.markdown('<div class="flag">...')` | `flag(html)` |
   | `eyebrow(...)` | `heading(text)` |
   | `dial(label, value, sub, tone)` | `kpi(label, value, sub, tone)` |
   | `st.dataframe(signed(df, cols, emphasise=...))` | `table(df, signed=cols, emphasise=...)` |
   | `st.plotly_chart(C.fn(args))` | `chart(kind, title, data, legacy=("fn", args, kwargs))` |
   | `with st.expander(label):` | `panel(label, [blocks])` |
   | `st.caption(...)` | `caption(text)` |

   Keep the DataFrames exactly as `app.py` displays them. `legacy` names the
   `qbt.charts` function and arguments the tab used; the tests rebuild that
   figure to prove the chart data is the same.
3. **Register it**: one line in `SCREENS` in `server/app.py`, one line in
   `SCREENS` in `tests/golden/test_golden_service.py` with the recorded tab
   name (for example `"positions": ("tab: Positions", positions.build)`).
4. **Run** `pytest tests/golden/test_golden_service.py -q`. It compares the
   screen with the recorded Streamlit tab on all four backtest scenarios and
   lists every difference by position. Fix the view until it passes.
5. **Widgets inside a tab** (a slider, a selector) are recorded in the
   golden transcript as `{"kind": "slider", ...}` entries. Extend
   `blocks_record` in the test for each new kind, and give the block a
   `control` description the front end renders. If the control changes
   server results (folds, stress category), the screen takes it as a query
   parameter and `store.screen` caches per parameter set. If it only changes
   the view, handle it in the browser.
6. **Front end**: the tab appears automatically and is drawn by `BlockList`.
   Add a layout in `web/src/screens/` and to `LAYOUTS` in `App.tsx` only if
   the single column wastes space. Add a chart component only for a chart
   kind not yet in `web/src/blocks/Blocks.tsx` (`Chart`).
7. **Downloads**: add a `download` block (label, file name, URL) and a route
   `GET /api/runs/{id}/exports/{name}` that returns the bytes. Compare with
   the recorded export using `harness.export_record` logic; see ticket B2.
8. **Check in the browser** at 1440 and 390 px against `docs/DESIGN.md`,
   then run the whole suite and commit.

### Components to reuse

| Need | Use | File |
|---|---|---|
| Any list of blocks | `BlockList` | `web/src/blocks/Blocks.tsx` |
| KPI row | `KpiStrip` | same |
| Table | `DataTable` | same |
| Note or flag (folds when long) | `Note` | same |
| Collapsible group | `Panel` | same |
| Value and drawdown on one axis | `TimePanel` | `web/src/charts/TimePanel.tsx` |
| One line with a reference level | `LineChart` | `web/src/charts/Simple.tsx` |
| Signed bars | `Bars` | same |
| Histogram | `Distribution` | same |
| Month-by-year grid | `MonthlyGrid` | same |
| New chart | `useChart` + `ChartFrame` | `web/src/charts/core.tsx` |
| Form fields | `Select`, `NumberBox`, `TextBox`, `TextArea`, `Check`, `Segmented`, `FileDrop`, `Group` | `web/src/controls/fields.tsx` |
| Session state (controls, files, run) | `useStudio()` | `web/src/state.tsx` |
| Server calls and types | `api`, block types | `web/src/api.ts` |

## 5. Decisions the owner has not made

From the end of `docs/ui-parity-checklist.md`. Do not settle these by
default; ask.

- **U1**: loading a preset or YAML is partial in the classic app. Fix or
  reproduce? (Blocks ticket C4.)
- **U2**: stale wording in Robustness and the README. Keep or correct?
- **U3**: compute walk-forward, in/out-of-sample and cost sensitivity once
  per run instead of on every click? (Assumed yes in ticket B6; confirm.)
- **U4**: the Assistant tab's machine-specific Python path.
- **U5**: decided. The owner confirmed brass as the accent on 2026-11-12.
  `qbt/brand.py` and `qbt/charts.py` stay as they are for the classic app
  and the exported reports.
- **Hosting**: where the new app will run, whether it must be free, and
  whether firm data may sit on a third-party host. Streamlit Community Cloud
  cannot run it. (Blocks ticket D2.)
- **Quick win in the classic app** (defer the tearsheet, compute robustness
  once): offered, not approved, not done.

## 6. Tickets, in order

Each ticket ends with: golden and API tests pass, `npm run build` passes,
the screen checked at 1440 and 390 px, the whole suite passes, one commit.
"Items" are numbers in `docs/ui-parity-checklist.md`.

### Backtest

| # | Ticket | Scope | Items | Done when |
|---|---|---|---|---|
| B1 | Positions tab | `server/views/positions.py` from `app.py:4180-4602`. New: an editable table block for "Trades for your account" (units held, paste from a spreadsheet), the account-size box, in-tab number inputs. Trade log needs the long-table component (B3 can be folded in) | 110-125 | Recorded "tab: Positions" matches on 4 scenarios; `fit_to_holdings` reachable through a POST and checked against `tests/test_holdings.py` values |
| B2 | Export tab and all downloads | `app.py:5667-6246`. Routes for every file in items 245; `_workbook_modules` and `_cio_summary` lifted into `server/exports.py`. The workbook reads other tabs' settings: pass them as parameters with the classic defaults | 162-171, 245, 249 | Each export matches `exports.json` for the 4 scenarios (CSV digests, every Excel sheet, tearsheet text, YAML) |
| B3 | Long tables | TanStack Table + Virtual: pinned header, own scroll, column sort in the browser | 7, 125 | Trade log of 486 rows scrolls smoothly on a phone; sort under 100 ms |
| B4 | Signals tab | `app.py:4069-4176`. Multiselect of instruments is view-only: send all series, filter in the browser. New chart kinds: multi-line, stacked area | 102-109 | Recorded tab matches; changing the selection makes no request |
| B5 | Compare tab | `app.py:3996-4066`. New chart kind: correlation matrix (HTML table like `MonthlyGrid`) | 95-101 | Recorded tab matches (`macro_blend` has content, others the empty state) |
| B6 | Robustness tab | `app.py:5056-5426`. Folds, stress category and custom periods as screen parameters; sweep, day sweep and Monte Carlo as on-demand jobs with progress; event-study panel from `qbt/lab/ui.py:903`. Compute once per run (U3) | 142-148 | Recorded tab matches including Monte Carlo in `momentum_file`; the tab adds nothing to the time of a run until opened |
| B7 | Attribution tab | `app.py:4605-4832`. Two editable tables (asset classes, Brinson benchmark) | 126-134 | Recorded tab matches |
| B8 | Tax tab | `app.py:4836-5053`. Settings as screen parameters | 135-141 | Recorded tab matches |
| B9 | Data tab | `app.py:5429-5494` | 149-154 | Recorded tab matches |
| B10 | Builder tab | `app.py:5497-5663`. Rule list is browser state; compile and preview through a POST | 155-161 | Recorded tab matches; "Send to the Custom Formula model" fills the panel |

### Control panel

| # | Ticket | Scope | Items | Done when |
|---|---|---|---|---|
| C1 | Benchmark: blend, strategy, frictions | Blend weights box and rebalance, "Another strategy", friction multiselect, cash proxy select. `RunSpec` already has the fields | 29-33, 36 | A blended-benchmark run from the UI equals `macro_blend` |
| C2 | Execution detail | Trading-day rule, anchor months, calendar preview, max leverage, whole units, cash sweep, buffer, threshold | 58-60, 63-64 | Every `EngineConfig` field reachable; preview text equals `RebalanceSpec.label()` |
| C3 | Compare strategies, sleeves | Comparison models with parameters; by asset class, core + strategy, blend of strategies | 53-56, 65 | `controls.json` of `macro_blend` reproduced by a control inventory test |
| C4 | Presets and configuration files | Save, load, delete, export, restore; YAML import. Blocked on U1 | 12-19, 165-166 | Round trip: save, reload page, load, run gives the same run id |
| C5 | Currency, impact detail, overrides | Base currency, overrides, detected-currency table, impact parameters, slippage multipliers, FX costs, fee-liquidation toggle | 37-41, 67-68, 70-71 | Covered by an API test with currency detection stubbed |
| C6 | Model extras | VIX auto-fetch note, cash-hurdle note, upload "What was read" table, templates, weights sheet picker | 25-26, 45-46, 49-51 | Each note appears under the same condition as in `app.py` |

### Other workspaces

| # | Ticket | Scope | Items | Done when |
|---|---|---|---|---|
| R1 | Return stream: pipeline and Results | `app.py:1252-1712` into `server/returns.py` and a view; its own controls | 172-188 | Recorded `return_stream` "page" and "tab: Results" match |
| R2 | Return stream: Stress, Regimes, Robustness, Export | `app.py:1717-2166` | 189-192, 246 | Recorded tabs and 6 exports match |
| R3 | Fund review | `app.py:2171-2375`; images served as files | 193-202 | Verdict and tables equal `tests/test_fund_review.py` expectations |
| M1 | Markets: load, Snapshot, Screener, Groups | `qbt/snapshot_ui.py`, `qbt/groups_ui.py`. No goldens exist: record one from a frozen price file first, by extending the harness to feed `fetch_market` | 203-217 | New golden scenario recorded from the classic app, then matched |
| M2 | Markets: Chart, Performance, Risk, Relationships, Seasonality, Data | `app.py:911-1172`. Candles with Lightweight Charts, attribution kept | 218-224, 247, 259 | As M1 |
| L1 | Research: universe, Start here, Ledger | `qbt/lab/ui.py` | 225-228, 238 | Manual check against the classic app; `tests/test_lab_*.py` unchanged |
| L2 | Research: Factor Lab, Strategy Lab | | 229-233 | As L1 |
| L3 | Research: Risk X-Ray, Event Study, Monitor, Assistant | Blocked on U4 for the Assistant | 234-237 | As L1 |

### Deployment

| # | Ticket | Scope | Items | Done when |
|---|---|---|---|---|
| D1 | Password gate | Login page and a signed cookie when `QBT_PASSWORD` is set; every `/api` route refuses without it. **Required before the new app is reachable from outside the machine** | 2 | API tests for allowed and refused requests |
| D2 | Hosting | Multi-stage Dockerfile (Node build, Python runtime). Blocked on the hosting decision | | The image runs the app and the tests pass inside it |
| D3 | Retire the classic app | Only when every item in the checklist is ticked and the owner agrees | all | Owner's decision |

Suggested order: B1, B2, B3, C1, C2, B4, B5, B6, C3, B7, B8, B9, B10, C4,
C5, C6, D1, R1, R2, R3, M1, M2, L1, L2, L3, D2, D3.

## 7. Pitfalls

- **Tabs feed each other.** In `app.py` the Export tab reads variables left
  by other tabs (`wf`, `ios`, `cs`, `_score`, `ev`, `_periods`, `tax_rep`,
  `battr_cfg`) through `globals()`. In the new app each of those is a
  parameter of the export, defaulting to what the tab shows by default. The
  recorded exports were made with default settings on every tab.
- **Parameter defaults are widget defaults.** `server/pipeline._defaults`
  reproduces what each Streamlit widget returns when untouched: a choice
  whose default is not in its list falls back to the first choice; a series
  picker shows the first uploaded column. Wrong defaults change results
  silently; the golden tests catch it.
- **A comparison strategy's default model** is the first model, by label,
  that is not the main one and can run without missing data.
- **File source uses start 1990-01-01 and no end** in the config, whatever
  the file covers. The YAML export depends on it.
- **Table records include the index.** The goldens were recorded from the
  DataFrame Streamlit displayed, index and all. Keep each frame as `app.py`
  builds it and hide the index in the browser.
- **Notes are compared as plain text.** Markup in a note is free; the words
  are not.
- **Digests are tied to this machine and these library versions.** After a
  pandas or numpy upgrade a digest may differ in the ninth digit.
  `metrics.json` compares numerically and tells you whether a difference is
  real. Do not re-record without the owner.
- **Chart size.** The Results screen sends every daily point (about 450 KB
  of JSON before compression for 13 years and four series). Fine so far; if
  a screen with dozens of instruments gets slow, round values in
  `serialize.numbers` before reaching for anything cleverer.
- **Log scale** is drawn as log10 on a linear axis (`TimePanel`). Do not
  switch to the library's log axis: its ticks fall on powers of ten only.
- **The chart library is loaded on demand** from `web/src/charts/bundle.ts`.
  Import new chart types there by name; importing `echarts` whole adds about
  a megabyte.
- **`localStorage` holds upload ids**, and the server forgets uploads when
  it restarts. The run then fails with "Upload ... is no longer held. Choose
  the file again." That message is intended.
- **The preview pane throttles timers when hidden.** Measure interactions
  with the Event Timing API and real clicks, not with timers.
- **No authentication yet.** Do not expose port 8000 beyond the machine
  until D1 is done.
- **`tests/golden/_actual/`** appears when a golden test fails; it is
  git-ignored. Diff it against `expected/`.

## 8. Measured so far

Golden fixture (4 instruments, 2012 to 2025), same machine as the baseline.

| | Streamlit | New app |
|---|---|---|
| Press Run to result on screen | 28.5 to 29.3 s | 0.64 to 0.78 s for the run, then 0.49 s for the Results screen |
| Identical run again | 28.3 to 29.1 s | 4 ms on the server (cache) |
| Toggle Log scale | 24.3 to 25.7 s | 40 to 64 ms, input to next paint, no request |
| Switch tab and back | 0 (but every tab is always computed) | 4 to 17 ms |
| First load | 5.1 s, 131 requests, 3.5 MB | 9 requests, 519 KB; last one finished at 0.70 s |
| Reload with a warm cache | 5.7 s to usable | last request finished at 0.18 s |

The new app's first paint time could not be read in the preview pane, so
the load rows give when the last request finished. Initial JavaScript is
91 KB compressed; the chart library is a further 192 KB loaded on demand.
