# Architecture audit (phase 1)

Branch: `feature/security-decision-engine`, cut from `feature/return-stream-clustering` at `05187ce`.
Date: 2026-11-15.

## Baseline

| Check | Result |
|---|---|
| `python -m pytest -q` (conda env `quant`, Python 3.12.14) | **285 passed**, 18 warnings, 50 s |
| Network: Yahoo Finance (prices, CAD=X, ^TNX, ^VIX) | reachable |
| Network: Bank of Canada Valet | reachable |
| Network: FRED (`fred.stlouisfed.org`) | **unreachable** (connection closed / timeout) |
| Network: Ken French data library (Dartmouth) | **unreachable** (connection reset) |

## Where things live

| Feature | Logic (pure functions) | UI | Notes |
|---|---|---|---|
| Return stream (load, clean, frequency) | `qbt/returns_input.py` | `app.py` lines ~1255-2510 (`source == "Return stream"`) inside the **Backtest** workspace | Upload a returns file or download Yahoo tickers; total return from Yahoo adjusted close. |
| Fund review | `qbt/fund_review.py` (beta-matched replica, Newey-West regression, funding test, `decide`) | `app.py` Return stream tab "Fund review" | Export: `qbt/fund_review_export.py` (PDF, PPTX, Excel). |
| Diversification | `qbt/clustering.py` (alignment, correlation, clustering, effective bets, stress correlation, `candidate_test`) | `qbt/clustering_ui.py` | Config `configs/clustering_default.yaml`. |
| Quant stats | `qbt/metrics.py` (CIPM conventions: population moments, geometric annualisation), `qbt/cipm.py` (capture, bull/bear beta, appraisal) | everywhere | Reused by the new engine; nothing duplicated. |
| Stress windows | `qbt/stress.py` (24 episodes) | Return stream / backtest tabs | |
| Regimes | `qbt/regimes.py` (S&P trend, VIX, rates, curve) | Return stream / backtest tabs | |
| FX | `qbt/fx.py` (`load_rates`, `extend_with_history` with `data/fx/USDCAD_history.csv` back to 1971) | | Reused for CAD conversion. |
| Excel export | `qbt/excel_export.py` (`_Book`, `_style_block`, `_widths`: red title, black header, % formats), `qbt/exec_summary.py`, `qbt/analysis_sheet.py` | Export tab | The new export reuses the styling helpers. |
| Markets workspace | `qbt/monitor.py`, `qbt/snapshot*.py`, `qbt/groups*.py`, `qbt/tvchart.py`, `qbt/sortable.py` | `app.py` lines ~805-1250 | Watchlist analytics. |
| Research workspace | `qbt/lab/*` (factor library, validation suite, run cards, MCP server) | `qbt/lab/ui.py` | Also a CLI (`python -m qbt.lab`). |
| Data | `qbt/data.py` (`load_yfinance`: `auto_adjust=True`), `qbt/price_import.py` | | No disk cache: Streamlit's in-memory `st.cache_data` only. |

## Dependencies (simplified)

```
app.py ─┬─ returns_input ── metrics
        ├─ fund_review ──── cipm, metrics, stress, engine
        ├─ clustering(_ui)
        ├─ excel_export ─── metrics, config, engine, exec_summary, analysis_sheet
        ├─ monitor / snapshot / groups (Markets)
        └─ lab.ui (Research)
```

## Observations that shaped the design

1. All return-stream logic is already in UI-agnostic modules; only the page is inside `app.py`. See `docs/DECISIONS.md` (Module placement).
2. There is no on-disk download cache. The scenario matrix needs one; added in `qbt/security_review/data.py`.
3. Existing regime and stress modules are daily, US-centric and hard-coded in Python. The task requires YAML-driven rules; the new engine reads `config/scenarios.yaml` and leaves the old modules untouched for the Backtest workspace.
4. `.gitignore` ignores `*.xlsx`; an exception was added for `examples/*.xlsx`.
