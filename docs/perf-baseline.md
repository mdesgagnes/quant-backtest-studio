# Performance baseline: the Streamlit app before the redesign

Measured 2026-11-12 on the development machine (Windows 11, Python 3.12.14,
Streamlit 1.64.0, pandas 3.0.6), at commit `8c170f5`. These are the "before"
figures for the redesign. Reproduce the script times with:

```bash
python tools/perf/measure_legacy.py --yahoo
```

## Why every interaction is slow

Streamlit re-executes `app.py` from the top on every interaction. With a
result on screen, that means all ten Backtest tabs are rebuilt each time,
including work no interaction needs: the cost-sensitivity sweep re-simulates
the strategy at every cost level, and the tearsheet HTML is re-rendered so
its download button has something to serve.

## Script time (server side)

The time below is what the server spends before the browser can finish
drawing. A user waits at least this long. Golden fixture: 4 instruments,
2012 to 2025, uploaded file, Dual Momentum, 3,326 sessions.

| Action | Sample 1 | Sample 2 | Recomputes |
|---|---|---|---|
| Open the app, nothing run | 6.2 s | 6.5 s | Imports, control panel |
| Change a control before any run | 2.3 s | 2.6 s | Control panel |
| Run backtest and render every tab | 28.5 s | 29.3 s | Everything |
| Any rerun with a result on screen, nothing changed | 21.3 s | 26.4 s | Every tab |
| Toggle Log scale on the equity chart | 25.7 s | 24.3 s | Every tab |
| Move the folds slider (Robustness) | 26.3 s | 25.2 s | Every tab |
| Change the attribution period | 26.7 s | 25.6 s | Every tab |
| Change a control in the panel (result unchanged) | 26.0 s | 25.4 s | Every tab |
| Run the identical backtest again | 28.3 s | 29.1 s | Everything; no result cache |
| Build the Excel workbook (on click) | 14.2 s | 14.1 s | Workbook only |

An earlier unprofiled run on a quieter machine gave 15 s for the same rerun
and 17 s for the run, so the realistic range for one interaction is 15 to
26 s. Switching between tabs costs nothing: Streamlit already holds every
tab in the browser, which is also why every tab is computed every time.

Live Yahoo Finance data (one sample):

| Action | Time |
|---|---|
| Default universe (3 ETFs from 2010, 4,023 sessions): download, run, render | 37.3 s |
| Same, rerun with nothing changed | 29.7 s |
| Same, identical run again (prices cached for an hour) | 25.0 s |
| 32-ETF preset (4,032 sessions): download, run, render | 31.6 s |
| Same, rerun with nothing changed | 20.3 s |

## The simulation alone

The same golden run, calling the engine directly with no interface (three
repetitions, measured while other tests were running, so if anything high):

| Step | Time |
|---|---|
| Parse the uploaded price file | 0.25 to 0.43 s |
| Build the market object and clean prices | 0.01 to 0.03 s |
| Generate signals (Dual Momentum) | 0.01 s |
| Simulate | 0.42 to 0.66 s |
| Benchmark and alignment | 0.63 to 0.69 s |
| Headline statistics | 0.03 to 0.12 s |
| **Total** | **1.4 to 1.9 s** |

So of the 28 to 29 s between pressing Run and the page settling, under 2 s
is the backtest. The rest is the page building all ten tabs.

## Where one rerun goes

Shares of a rerun with nothing changed (two samples):

| Work | Share | Needed for the interaction? |
|---|---|---|
| Cost-sensitivity sweep (`robustness.cost_sensitivity`) | 26 to 31% | No: depends only on the run |
| Tearsheet HTML (`report.render_tearsheet`) | 24 to 27% | No: only when downloaded |
| Building 26 Plotly figures | 8 to 13% | Only the chart that changed |
| Performance-analysis sheet for Excel (`analysis_sheet.build_analysis`) | 4 to 6% | No: only when downloaded |
| Walk-forward folds and in/out-of-sample | 7 to 9% | Only when the folds slider moves |
| Brinson attribution | 3 to 4% | Only when its settings change |
| Everything else (statistics, tax, stress, Streamlit itself) | 16 to 22% | Mostly no |

At least 60% of every interaction is work that depends only on the run and
could be done once.

## Browser side

Measured in the embedded browser against `localhost`, server already
started:

| Measure | Value |
|---|---|
| First load, largest contentful paint | 5.1 s |
| Reload with a warm browser cache, until the control panel is usable | 5.7 s |
| JavaScript requested on first load | 124 files, 2.96 MB uncompressed |
| All resources on first load | 131 requests, 3.5 MB |

Interaction times as seen in the browser were not measured reliably (the
embedded pane throttles timers when hidden), so none are reported. They are
the script times above plus transport and rendering.

## Layout observations (not timings)

- At 1440 px the control panel takes about 20% of the width and its 50-odd
  controls scroll independently of the results.
- At 375 px the six KPIs stack one per row and fill more than a full screen
  before the first chart; the control panel becomes an overlay.
