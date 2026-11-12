# Handoff: the interface redesign

Read with `CLAUDE.md` (rules), `docs/DESIGN.md` (design system),
`docs/ui-parity-checklist.md` (the contract) and `docs/DEPLOY.md` (hosting).
A new session can continue from these files alone.

## 1. Where things stand

Branch `ui/terminal-redesign`. Nothing is pushed or merged. The classic
Streamlit app is untouched and still runs (`streamlit run app.py`).

The new app is complete in function: all four workspaces, every tab, every
control, every upload and download.

| Workspace | Tabs | State |
|---|---|---|
| Backtest | Results, Compare, Signals, Positions, Attribution, Tax, Robustness, Data, Builder, Export | Working. Results has a dedicated screen; the rest are drawn generically |
| Return stream | Results, Stress tests, Market regimes, Robustness, Export, Fund review | Working |
| Markets | Snapshot, Screener, Groups, Chart, Performance, Risk, Relationships, Seasonality, Data | Working |
| Research | Start here, Ledger, Factor Lab, Strategy Lab, Risk X-Ray, Event Study, Monitor, Assistant | Working |

What is left is in section 6: known gaps, things not verified, and polish.

## 2. Architecture, and why it changed

```
browser (web/)                 draws blocks; formats nothing
   |  POST /api/render  {tab, events, clicks}  ->  {sidebar, main}
server/app.py                  routes, one session per browser cookie
server/present.py              recorded nodes -> blocks (KPI, table, chart, control ...)
server/classic.py              runs app.py's screen logic for the request
server/shim.py                 a recording stand-in for the Streamlit API
app.py, qbt/*_ui.py            the classic screen logic, unchanged on disk
qbt/                           the engine, unchanged
```

The first plan was to port each screen by hand into a view file (the
Results screen was done that way and is kept). Porting the other thirty
screens the same way meant rewriting about five thousand lines of screen
logic, each line a chance for a figure to drift.

Instead, the server runs the classic app's own screen logic with `st` bound
to a recorder. `st.dataframe(...)` records a table, `st.selectbox(...)`
records a control and returns its current value, and so on. The recorded
nodes become blocks, and the new front end draws any page from blocks.

- **Parity is by construction.** It is the same code producing the same
  DataFrames and figures. `tests/golden/test_golden_app.py` replays the five
  recorded scenarios through the new server and requires every control,
  every screen, every export and the simulation's output to equal the
  Streamlit references. All five pass.
- **`app.py` is not edited.** `server/classic.py` reads it, adjusts five
  things in memory (documented at the top of that file) and compiles it. If
  `app.py` changes in a way the adjustments do not expect, the server
  refuses to start rather than run a half-adjusted script.
- **The cost.** A control inside a screen is a round trip to the server, not
  a browser-only change. On the golden fixture that is 0.2 to 0.5 s. The
  dedicated Results screen and returning to a tab already opened are
  instant.

### What makes it fast

- **Tabs not on screen are not computed.** `with tabs[i]:` becomes
  `for _ in tabs[i].body():`, which skips the body. The Export tabs need
  values other tabs compute; `TAB_NEEDS` in `server/classic.py` lists them,
  and those tabs run muted (no charts are built for them).
- **Per-run results are computed once.** Walk-forward, in/out-of-sample,
  cost sensitivity, tax, attribution, stress periods, regime tables and the
  Excel analysis sheet are remembered, keyed by the content of their
  arguments, so a changed setting recomputes. See `_wrap_heavy`.
- **The tearsheet is rendered when downloaded**, not on every page.
- **Plotly themes are emptied in this process.** Applying a built-in theme
  costs about 0.2 s per figure, and only the figure's data is read.
- **The browser keeps tabs it has drawn** until something changes
  (`web/src/live.tsx`).

### State

One process, sessions in memory keyed by a cookie (`qbt_sid`), up to 12.
A session holds `st.session_state`, widget values, uploads, registered
downloads and the per-run cache. Nothing is written to disk except what the
classic code already wrote (saved presets, the Research ledger). A restart
clears every session.

### Behaviour that differs from the classic app, on purpose

1. **Loading a preset or YAML replaces every control** (checklist U1). In
   Streamlit only controls without a value took the file. See
   `_apply_loaded_config` and `tests/test_server_app.py`.
2. **Robustness tables are computed once per run** rather than on every
   interaction (U3). Same figures.
3. **The palette is National Bank black, red and gold** (U5, set by the owner
   on 2026-11-12). Red is used for controls and navigation, gold for accent
   text and the subject series; a loss is coral, never the brand red.

Wording is verbatim, including the stale counts noted in U2.

## 3. How to run

Python is the conda environment `quant`:
`C:\Users\marcd\miniconda3\envs\quant\python.exe`.

```bash
cd web && npm install && npm run build && cd ..
python -m uvicorn server.app:app --port 8000        # http://localhost:8000

# front-end development with hot reload (http://localhost:5173)
python -m uvicorn server.app:app --port 8000        # terminal 1
cd web && npm run dev                               # terminal 2

streamlit run app.py                                # classic app, :8501
```

`QBT_PASSWORD` turns on the password screen. `.claude/launch.json` has both
servers for the preview pane.

Tests:

```bash
pytest tests/test_server_app.py tests/test_server_api.py tests/golden/test_golden_service.py -q   # about 40 s
pytest tests/golden/test_golden_app.py -q        # the new app against the references, about 2.5 min
pytest -q                                        # everything
cd web && npm run build                          # type-check and build
```

## 4. Where to change things

| To change | Edit |
|---|---|
| How a kind of block looks | `web/src/blocks/Render.tsx`, `web/src/styles/app.css` |
| How a control looks or behaves | `web/src/blocks/Controls.tsx` |
| How charts are drawn | `web/src/charts/Figure.tsx` (generic), `TimePanel.tsx` (Results) |
| Colours, type, spacing | `web/src/styles/tokens.css` only |
| What a recorded node becomes | `server/present.py` |
| A Streamlit call the classic code uses that the recorder lacks | `server/shim.py` (add the method on `DG`) |
| Which tabs a tab depends on, what is cached | `server/classic.py` |
| A screen that deserves its own layout | Follow Results: `server/views/results.py` builds blocks, `web/src/screens/Results.tsx` arranges them, `DEDICATED` in `server/app.py` skips the classic tab |
| Screen logic itself (what a tab computes or says) | That is `app.py`. Do not edit it without the owner; see `CLAUDE.md` |

Block kinds sent to the browser: `title`, `context`, `heading`, `kpis`,
`note`, `flag`, `html`, `caption`, `code`, `alert`, `table`, `editor`,
`chart`, `image`, `frame`, `download`, `upload`, `control`, `row`, `panel`,
`form`, `chat`, `tabs`, `results`.

`server/pipeline.py`, `server/store.py` and the `/api/runs` routes are a
programmatic API for running a backtest from a `RunSpec` without a browser.
The interface does not use them; they are tested and kept.

## 5. Measured

Golden fixture (4 instruments, 2012 to 2025), same machine as the baseline.

| | Streamlit | New app |
|---|---|---|
| Press Run to result on screen | 28.5 to 29.3 s | 2.5 to 2.7 s (server), Results screen included |
| Any control with a result on screen | 21 to 27 s | 0.2 to 0.5 s (server) |
| Toggle Log scale on Results | 24.3 to 25.7 s | 40 to 64 ms, no request |
| Return to a tab already opened | 0, but every tab is always computed | 4 to 17 ms |
| Open a tab for the first time | n/a | 0.1 to 0.6 s; Robustness 7 s, Export 7 s (then cached) |
| Build the Excel workbook | 14 s | 10 s (same code) |
| First load | 5.1 s, 131 requests, 3.5 MB | 82 KB of script compressed, charts 195 KB on demand |
| Server memory after a run and every tab | not measured | 229 MB |

In the browser, Markets tabs opened in 0.2 to 1.1 s on live Yahoo data.

## 6. What is left

### Known gaps against the checklist

| Item | Gap |
|---|---|
| 243 | The parameter surface is shown as a shaded grid, not a rotatable 3D surface. Needs `echarts-gl` (check it works with ECharts 6 first) |
| 240 | Charts zoom with Ctrl + wheel but have no reset button; reload the tab to reset |
| 7 | Long tables scroll with a pinned header and sort by clicking a column, but are not virtualised. Capped at 3,000 rows on screen; downloads have everything |
| Design | "Settings changed since this run" is no longer shown |

### Not verified

- Seen in the browser: Results, Positions, Attribution, Robustness (1440),
  Positions (390), Markets Snapshot, the Research start. Every other tab was
  checked by a script for errors and overflow, not looked at.
- The TradingView chart (Markets > Chart) arrives as a sandboxed frame; it
  was not seen drawing, since it needs the CDN.
- A real upload from the browser's file picker (uploads were sent through
  the API), and a real iPhone (390 px was emulated).
- The Docker image: Docker is not installed on this machine.
- Hosting: not deployed. See `docs/DEPLOY.md`.

### Open decisions for the owner

- **U2**: correct the stale wording ("four tests", "sixteen episodes")?
- **U4**: the Assistant tab prints a machine-specific Python path.
- Whether to retire `app.py` as a runnable Streamlit app. Its code must stay
  either way: it is now the screen logic of the new app.

### Polish, in rough order of value

1. Dedicated layouts for the most-used tabs after Results (Positions,
   Export), so their view-only controls stop being round trips.
2. Look at every remaining tab at 1440 and 390 px against `DESIGN.md`.
3. `echarts-gl` for the parameter surface.
4. Virtualised long tables (TanStack Table and Virtual).
5. A reset control on zoomable charts.
6. A light theme (tokens are roles; add a second set).

## 7. Pitfalls

- **`app.py` is load-bearing.** Deleting or reformatting it breaks the new
  app. `server/classic.py` checks for the exact lines it adjusts and for 25
  tab blocks.
- **The recorder must behave like Streamlit where the code depends on it**:
  keyed widgets live in `session_state`; a keyless widget is identified by
  kind, label and default; a button is true for one request; `st.stop()` and
  `st.rerun()` work. If a screen misbehaves, look for a Streamlit behaviour
  the recorder does not reproduce before suspecting the screen.
- **The browser can send a change from a tab the server did not draw last**
  (it keeps tabs). The server therefore remembers the controls of every tab
  drawn in the session, not only the last one.
- **Cached results must not be modified in place.** Everything in
  `_wrap_heavy` is used read-only by `app.py` today. If that changes, the
  second render of a tab will differ from the first; the golden tests render
  several times and would catch it.
- **Muted tabs build no charts** (`_unless_muted` returns `None`). Code that
  uses a figure after building it would fail when muted; none does today.
- **One worker only.** Sessions and caches are in process memory.
- **Markup from the classic code is inserted as HTML** after
  `present.clean_html` (scripts and handlers removed, old palette mapped to
  tokens). It includes file names and tickers the user supplied.
- **Golden digests are tied to this machine and these library versions.**
  `metrics.json` compares numerically and tells you whether a difference is
  real. Never re-record to make a test pass.
- **The preview pane throttles timers and sometimes cannot take a
  screenshot** when it is hidden. Measure with real clicks and the Event
  Timing API.
