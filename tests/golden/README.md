# Golden-output tests

These tests prove the app still produces the same results. They exist for
the interface redesign: the engine is not being changed, and these are how
that is checked rather than assumed.

```bash
pytest tests/golden -q        # about 4 minutes
```

## What is locked

Five reference runs (`scenarios.py`), all on frozen files in `fixtures/`:

| Scenario | What it exercises |
|---|---|
| `momentum_file` | Uploaded prices (long format, OHLCV + dividends), built-in Dual Momentum, single-ticker benchmark, Monte Carlo |
| `imported_weights` | A target-weights file from an existing strategy, traded on its own dates |
| `factor_signal` | A cross-sectional signal file with a 45-day publication lag, execution at the open, volume-scaled impact, cash rate, management fee |
| `macro_blend` | A macro signal file, quarterly rebalance, a 60/40 blended benchmark, one comparison strategy |
| `return_stream` | The Return stream workspace on a monthly file with a benchmark column |

For each run, `expected/<scenario>/` holds:

- `metrics.json`: every raw statistic for the strategy and the benchmark, compared to 1e-9 relative.
- `engine.json`: the simulation's own output (daily value, returns, weights, target weights, units, trades, cash flows, last holdings), each as a row count and a digest, with the month-end value series in full.
- `controls.json`: every control in the panel, with its label, options and value.
- `screens.json`: everything each tab shows, in order: KPIs, notes, flags, headings, tables (in full up to 60 rows, otherwise a digest with head and tail) and, for each chart, its title, axis type and what each series plots.
- `exports.json`: every download the page offers, by content. CSV files are parsed and digested, the Excel workbook is read cell by cell on every sheet, the tearsheet by its visible text, the YAML in full.

Timestamps are the only thing replaced before comparing (`<TODAY>`, `<NOW>`).

## How it works

`harness.py` runs the real `app.py` through Streamlit's `AppTest`. Uploads
are served from `fixtures/`, download buttons record what they would have
sent, and network fetches are refused. Nothing in `app.py` or `qbt/` is
modified or mocked beyond that, so what is recorded is what the app does.

Records are deliberately free of Streamlit specifics: a KPI is
`{"kind": "kpi", "label", "value", "sub", "tone"}`, not HTML. The redesigned
app is tested by building the same record from its own output and comparing
it with the same `expected/` files (`harness.compare`).

## When a test fails

The failure lists each difference by path, and the full actual record is
written to `tests/golden/_actual/<scenario>/` (git-ignored). Diff it against
`expected/<scenario>/`.

A difference means the result changed. During the redesign that is a bug in
the new code, not a reason to re-record.

## Re-recording

```bash
python tests/golden/record.py [scenario ...]
```

Only when a change in results is intended, understood, and approved by the
owner. Review the diff of `expected/` before committing it.

## Limits

- Yahoo Finance downloads are not golden: they change daily. Everything
  after the download is covered through the uploaded-file path, which builds
  the same data object.
- Digests are computed from values printed to 9 significant digits. They are
  stable on one machine and library set. On a different machine or after a
  pandas or numpy upgrade, a last-digit difference would show as a digest
  mismatch; `metrics.json` (numeric tolerance) tells you whether it is real.
- Markets and Research are not covered here (they need live prices or write
  to a local database). Their numbers are covered by the unit tests.
- `make_fixtures.py` documents how the input files were made. Do not re-run
  it: new inputs invalidate every reference.
