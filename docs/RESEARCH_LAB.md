# Research Lab

The research layer of Quant Backtest Studio (`qbt/lab/`, Research workspace in
the app). It was built after studying
[HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) (MIT), an
LLM-driven trading agent, and keeping what matters for research while leaving
out what does not: live brokers, paper trading, 90 prose skills, a multi-agent
"swarm", Electron/React front ends. The engine, costs model, multi-currency
accounting and robustness suite stay exactly as they were; the lab sits on
top.

Research only. Nothing in `qbt/lab` places or could place an order.

## What was taken from Vibe-Trading, and what it became

| Vibe-Trading idea | Here | Why it differs |
|---|---|---|
| Monte Carlo / Bootstrap / Walk-Forward validation | `lab/stats.py`, `lab/validate.py` | Their Monte Carlo shuffles trade P&L order, which cannot change the total, so it says little about skill. Replaced with a **random-timing test** (same positions, rotated in time) and a **sign-flip test**; trade-order shuffling is kept but only claims what it can (drawdown luck). Bootstrap is **stationary** (block), because strategy returns are autocorrelated. Added what they lack: probabilistic and **deflated Sharpe**, minimum track-record length, **PBO** (CSCV) and **combinatorial purged CV**. |
| Run cards (JSON + Markdown, hashes, warnings, metric-to-CSV citations) | `lab/runcard.py` | Same idea, plus a **frozen price snapshot** so `reproduce_run_card` can re-run offline and match the equity curve, and `verify_run_card` detects tampering with any artifact or quoted metric. |
| Alpha Zoo (478 factors), IC statistics, factor research tab | `lab/factors.py` | 43 price-only, literature-sourced factors, each tested for **causality** (value at t unchanged when the future is removed). IC, ICIR, Newey-West t-stat, quantile spread, decay, and the part a screen needs: **Benjamini-Hochberg / Bonferroni** adjusted p-values. Registers a `zoo_factor` strategy so any factor runs in the Backtest workspace too. |
| Risk x-ray | `lab/xray.py` | Adds per-holding **risk contribution**, diversification ratio and effective bets, Cornish-Fisher VaR, long/short support. |
| Event study (CAR) | `lab/eventstudy.py` | Market-model, estimation window cut off before the event window. |
| Strategy store + decay monitoring (`sdm_*`) | `lab/ledger.py`, `lab/decay.py` | Registering freezes a strategy; everything after is out-of-sample for it. z-score against the baseline Sharpe, rolling Sharpe, and a CUSUM calibrated for daily data. |
| Research Autopilot (hypothesis -> backtest) | the **hypothesis ledger** | Not an LLM loop: a record. Every experiment is logged against a hypothesis and `n_trials` for the deflated Sharpe is a count the system holds. |
| MCP server (74 tools), LLM agent | `lab/mcp_server.py`, `lab/assistant.py` | A small stdlib-only MCP server (17 tools) so Claude Code/Desktop can drive the lab at no extra cost. Optional in-app assistant (needs `anthropic` + `ANTHROPIC_API_KEY`), restricted to the lab's tools. |
| Multi-market engines, 20 data loaders, brokers, paper trading, swarm presets | **not taken** | Out of scope for research, or already covered (QBS has its own engine and FX accounting). |

## The workflow

1. **Ledger** - write the hypothesis (what, and why it should work) before
   testing. Select it as the active hypothesis in the sidebar.
2. **Factor Lab** - screen the library. Trust the adjusted p-values, not the
   raw ones: of 43 factors, a few look good by chance every time.
3. **Strategy Lab** - run a strategy (any registered one, including
   `zoo_factor`). Then **validate**; add a parameter grid to get the
   overfitting checks (PBO, CPCV).
4. **Save run card + dossier** - a folder you can hand to someone: hashes,
   warnings, a one-page HTML dossier.
5. **Register for monitoring** - freeze it; later, **Monitor -> Scan**
   re-runs it on fresh prices and reports fresh / healthy / aging / stale.

### How to read the validation grade

Each check is pass / warn / fail with its evidence and threshold printed
(thresholds in `validate.THRESHOLDS`, conventions rather than laws). The grade
(Robust / Promising / Fragile / Unsupported) weighs how many independent
lines of evidence agree and is capped by a failure of the deflated Sharpe,
PBO, PSR or bootstrap. It is **not** a probability of profit.

Two caveats worth knowing:

- **No hypothesis selected = no deflation.** The deflated Sharpe needs the
  number of configurations you tried; without a hypothesis the lab cannot know
  it and shows that check as N/A. That is the main way a grade can flatter a
  result.
- **The random-timing test measures timing, not exposure.** A long-only
  always-invested strategy on a rising universe fails it, correctly: most of
  its return is beta. A trend rule that moves in and out should beat it.

## Command line

```bash
python -m qbt.lab demo                       # offline end-to-end run on synthetic data
python -m qbt.lab tools                      # list every tool
python -m qbt.lab research SPY EFA EEM TLT GLD --benchmark SPY --start 2012-01-01 \
       --strategy zoo_factor --params '{"factor":"mom_12_1","top_n":3}' \
       --grid '{"top_n":[2,3,4]}'
python -m qbt.lab call factor_report '{"dataset_id":"...","factor":"mom_12_1"}'
```

Interpreter: `C:\Users\marcd\miniconda3\envs\quant\python.exe` (the `quant` env).

## Let Claude drive it (MCP)

```bash
claude mcp add qbt-lab -- C:/Users/marcd/miniconda3/envs/quant/python.exe -m qbt.lab mcp
```

Run that from the repository folder (or set `cwd`). Claude Desktop: add to
`mcpServers` with `"args": ["-m", "qbt.lab", "mcp"]` and the folder as `cwd`.
Then ask in plain language: *"Load SPY, EFA, EEM, TLT, GLD since 2012, screen
the factors at a 21-day horizon, backtest the best survivor with a grid on
top_n, validate it and save the run card."* The server's instructions tell the
model to log trials, report the grade with any number, and never present a
backtest as a forecast.

## Where things live

- `qbt/lab/stats.py` statistics, `validate.py` battery and grade, `factors.py`,
  `xray.py`, `eventstudy.py`, `ledger.py` (SQLite), `decay.py`, `runcard.py`,
  `dossier.py`, `run.py` (datasets, runs, sweeps), `tools.py` (the JSON tool
  layer all front ends share), `mcp_server.py`, `assistant.py`, `ui.py`.
- Data written at runtime goes to `.qbt_lab/` (override with `QBT_LAB_DIR`):
  `lab.db` (ledger), `data/` (price snapshots), `runs/` (run cards). Like the
  preset store, it lives on the machine running the app; export the ledger
  from the Ledger tab before an ephemeral host wipes it.
- Tests: `tests/test_lab_*.py` (statistics against closed forms and planted
  effects, causality of every factor, tamper detection, MCP protocol, the
  Streamlit flow). `python -m pytest tests`.

## Limits, stated plainly

- Factors are **price-only**. For fundamentals use the existing `factor_rank`
  strategy with an imported series.
- Data is Yahoo total-return closes (dividend-adjusted). The Backtest
  workspace's price-return + cash-dividend mode and multi-currency accounting
  are not wired into the lab runner; the engine supports them, the lab does
  not yet expose them.
- Universes built from today's tickers carry survivorship bias; every run card
  says so.
- A small universe (under ~10 names) gives noisy cross-sectional statistics;
  the Factor Lab warns, and uses terciles rather than quintiles.
- Decay scans download fresh prices, so they need network access.
- Datasets and runs are held in process memory and shared by everyone connected to the same app
  process: it is a single-user tool, not a multi-tenant service.
