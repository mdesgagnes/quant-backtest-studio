# Testing guide: Security review

Branch `feature/security-decision-engine`. Nothing is merged to `main`.

## 1. Launch

From the repository folder, in the `quant` conda environment (no new dependency was added):

```bash
git checkout feature/security-decision-engine
```

```bash
conda activate quant
```

```bash
python -m pytest -q
```

Expected: **331 passed** (285 existing + 46 new). Takes about a minute.

```bash
streamlit run app.py
```

The first Security review run downloads prices for the 44-ticker universe and the profile blocks (about 30 seconds) into `data/cache/`; later runs read the cache (refreshed after 3 days).

## 2. Ready-to-run demos (no interface)

Each prints the ranking and rationale and writes the Excel workbook to `examples/output/`.

```bash
python -m qbt.security_review examples/demo_1_balanced_lower_volatility.yaml
```

```bash
python -m qbt.security_review examples/demo_2_balanced_stock_bond_hedge.yaml
```

```bash
python -m qbt.security_review examples/demo_3_income_decumulation.yaml
```

```bash
python -m qbt.security_review examples/demo_4_growth_replace_drawdown.yaml
```

```bash
python -m qbt.security_review examples/demo_5_sweep_dbmf.yaml
```

| Demo | Request | What you should see |
|---|---|---|
| 1 | Balanced, lower volatility, 11 US equity ETFs, Add 10% | Common period 2013-08 onward (QUAL's start). SPLV, USMV (and SCHD) on top with Consider, driven by downside capture around 0.6-0.7; SPY, RSP, QUAL, MTUM, VUG, QQQ, IWM Do not add, in that order. Trade-off line for SPLV/USMV: upside capture given up. |
| 2 | Balanced, stock-bond hedge, alternatives + bonds, each own history | KMLM, DBMF, BTAL Add (KMLM only Moderate confidence: its history covers 4 of the 9 stress windows); TAIL Consider; SHY Add, AGG/TIP/IEF Consider; TLT Do not add; RSST/RSSB Do not add with **Low** confidence (about 3 years, no 2022). |
| 3 | Income, sequence risk (4%/yr, 30y), Canadian bonds + income equity | XEI.TO Add, ZWC.TO Consider with an option-writing caveat; bond ETFs Do not add (more bonds in a 70% bond profile raise the depletion probability). XBB.TO and ZAG.TO flagged functionally equivalent. In the workbook's Verdict sheet the pair reads "ZAG.TO is cheaper (0.09% vs 0.10% a year); ZAG.TO trades more". |
| 4 | Growth, reduce drawdown, USMV replacing VFV.TO (Replace mode) | Add: max drawdown about 1.7 points shallower, trade-off about 0.8 point of CAGR; caveats list the SPY/EFA/EEM splices inside the period. |
| 5 | Profile sweep of DBMF, stock-bond hedge | One row per profile: Add in each, High confidence. Volatility falls by 1.0 to 1.6 points and max drawdown improves by 2.4 to 4.7 points depending on the profile. DBMF's history (from 2019) misses the GFC, euro crisis, 2015-16 and Q4 2018 (stated in the caveats). In the app, switch the objective to Faster recovery to see a sweep where the verdict changes by profile. |

`examples/custom_profile_example.csv` is a ready file for the Custom profile upload (check 12).

## 3. Checklist (in the app)

Open the **Security review** workspace (first in the sidebar). Defaults: Balanced, Lower volatility, USMV / SPLV / QUAL / SPY, Add 10%.

| # | Do this | Expected result |
|---|---|---|
| 1 | Look at the sidebar workspace selector | Security review and Backtest only. Markets and Research are hidden. |
| 2 | Open the Profile list | 7 presets first (100% Equities ... 100% Bonds, each with its split), Custom last. |
| 3 | Press **Run review** with the defaults | Header "Balanced (60% equity / 40% bonds) · Lower volatility · 2013-08 to ... (common to all)". One card per candidate, ranked: SPLV and USMV Consider, QUAL and SPY Do not add. Each card: score, confidence, months, benchmark SPY, 3 deciding facts, a trade-off line. |
| 4 | Open the **Verdict** tab | Request table, Verdict and rationale (facts, trade-off, caveats), and "Verdict for every objective": the same candidates get different verdicts under different objectives. |
| 5 | Open **Ranking** | Ranking table and "Score breakdown by component" (value, score, weight, contribution per metric). |
| 6 | Open **Sleeve Comparison** | Active return, tracking error, IR, beta, capture, up/down beta, hit rate, relative drawdown vs SPY. USMV/SPLV downside capture well below 1; QUAL close to 1. |
| 7 | Open **Diversification** | Three matrices: correlation (all > 0.85 for these four), excess-return correlation (much lower and different from pair to pair) and crisis correlation. |
| 8 | Open **Portfolio Impact** | All metrics table, profile with/without table, risk-of-ruin and decumulation table (2000 paths, seed 42), target weights with/without. |
| 9 | Open **Scenarios** and **Regimes** | 9 stress windows per candidate, on daily data; SPY covers all 9, while the GFC and Euro crisis are "no data" for USMV, SPLV and QUAL (launched later), never filled. Regimes: 10 rows per candidate. |
| 10 | Open **Robustness** | Rolling 36-month shares, add-weight sensitivity at 5/10/20%, consistency, data flags. |
| 11 | Change the objective to **Reduce concentration** and run | Ranking changes. Pick RSP and QQQ as candidates: RSP gets a negative mega-cap tilt, QQQ a positive one. |
| 12 | Profile: **Custom**, Upload a file: `examples/custom_profile_example.csv`; name it, **Save as a profile**; run | Weights shown under the uploader (XIC.TO 25%, VFV.TO 35%, XEF.TO 10%, XBB.TO 25%, GLD 5%). After saving, the profile appears in the list (after the presets) on the next page load. |
| 13 | Profile: Growth. Mode **Replace**, holding VFV.TO, candidate USMV only, objective Reduce drawdown | Add (demo 4). Portfolio Impact target weights: USMV 31.5%, no VFV.TO. |
| 14 | Fill with a sleeve: **Canadian bonds**; objective Income generation; profile Income | XBB.TO / ZAG.TO pair listed as functionally equivalent (Verdict tab), with the cheaper and more liquid fund named. Fund Review shows expense ratio, AUM and daily value traded for every ticker. |
| 15 | Fill with a sleeve: **Alternatives / diversifiers**; objective Stock-bond correlation hedge; Advanced: **Each candidate's own history** | DBMF/KMLM/BTAL at the top; RSST/RSSB Low confidence. In the common-period mode instead, the period starts with the youngest fund (a few years): note how confidence drops. |
| 16 | Advanced: tick **Also run a profile sweep**, run with DBMF first | A **Profile Sweep** tab with 7 rows. |
| 17 | Same candidate and objective, switch the profile from 100% Equities to 100% Bonds | Changes in volatility and drawdown differ by profile; the verdict can change (e.g. upside participation). |
| 18 | Open **Factor Exposures** | Loadings and t-stats for market, size, value, momentum, quality, low vol; R2; mega-cap tilt. The note under the table names the factor source: Ken French where reachable, otherwise the ETF spreads (with the caveat on factor legs). |
| 19 | Open **Charts** | Growth of 1 and drawdown with each candidate, stress-window bars, rolling metric; the monthly series table under them. Red / sand palette. |
| 20 | Press **Download the full comparison (Excel)** and open it | Sheets: Verdict, Ranking, Fund Review, Sleeve Comparison, Diversification, Portfolio Impact, Scenarios, Regimes, Factor Exposures, Robustness, (Profile Sweep), Methodology, Series. Red titles, black headers, percentages as numbers. Every tab of the page matches a sheet. |
| 21 | Methodology sheet | Source field, currency method, T-bill source, CPI source, rebalancing, holdings with proxy splice dates, objective definition and thresholds. |
| 22 | Switch to **Backtest**, Yahoo Finance source, default settings, run | The backtest works as before (results, tabs, exports). |
| 23 | Backtest > Source **Return stream** > Yahoo, a few tickers | Return stream, Fund review and Diversification work as before. |
| 24 | Set `markets: enabled: true` in `config/features.yaml`, reload | Markets is back in the sidebar (Research the same way). Set it back to `false`. |
| 25 | Run the calibration and matrix scripts (optional, about 1 and 5 minutes) | `python scripts/calibration.py` rewrites docs/CALIBRATION_REPORT.md; `python scripts/scenario_matrix.py` rewrites docs/SCENARIO_MATRIX.md and examples/scenario_matrix.xlsx. |

## 4. Where to change things

| To change | Edit |
|---|---|
| Profiles, building blocks, proxies, rebalancing default | `config/profiles.yaml` |
| Candidate universe and sleeve benchmarks | `config/universe.yaml` |
| Objective weights, scales, thresholds, guardrails, simulation settings | `config/objectives.yaml` |
| Stress windows and regime rules | `config/scenarios.yaml` |
| Factor proxies | `config/factors.yaml` |
| Which workspaces show | `config/features.yaml` |
