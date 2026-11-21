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

Expected: **347 passed** (285 existing + 62 new). Takes about a minute.

```bash
streamlit run app.py
```

A Security review run downloads prices only for the tickers it needs (the securities, the portfolio's holdings, and the profile-block, factor, regime and FX series: about 20 for two ETFs) into `data/cache/`, the first time in about 10 to 15 seconds; later runs read the cache (refreshed after 3 days).

## 2. Ready-to-run demos (no interface)

Each prints one sentence per security and portfolio, the summary table, and writes the Excel workbook to `examples/output/`.

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

```bash
python -m qbt.security_review examples/demo_6_symbol_vs_portfolio.yaml
```

```bash
python -m qbt.security_review examples/demo_7_symbol_vs_benchmarks.yaml
```

```bash
python -m qbt.security_review examples/demo_8_dbmf_from_bonds.yaml
```

```bash
python -m qbt.security_review examples/demo_9_gold_split_funding.yaml
```

| Demo | Request | What you should see |
|---|---|---|
| 1 | Balanced, lower volatility, 11 US equity ETFs, Add 10% | Common period 2013-08 onward (QUAL's start). Every fund is Do not add: measured against the Balanced portfolio itself, a 100% equity fund raises its volatility (downside capture of the portfolio above 1). USMV and SPLV come closest (volatility up only 0.07 and 0.02 points, drawdown about 0.3 to 0.6 points shallower); IWM, QQQ and VUG raise it most. Choose Advanced > Each security's sleeve benchmark to compare them with SPY instead. |
| 2 | Balanced, stock-bond hedge, alternatives + bonds, each own history | KMLM, DBMF, BTAL Add (KMLM only Moderate confidence: its history covers 4 of the 9 stress windows); TAIL Consider; SHY Add, AGG/TIP/IEF Consider; TLT Do not add; RSST/RSSB Do not add with **Low** confidence (about 3 years, no 2022). |
| 3 | Income, sequence risk (4%/yr, 30y), Canadian bonds + income equity | XEI.TO Add, ZWC.TO Consider with an option-writing caveat; bond ETFs Do not add (more bonds in a 70% bond profile raise the depletion probability). XBB.TO and ZAG.TO flagged functionally equivalent. In the workbook's Verdict sheet the pair reads "ZAG.TO is cheaper (0.09% vs 0.10% a year); ZAG.TO trades more". |
| 4 | Growth, reduce drawdown, USMV replacing VFV.TO (Replace mode) | Add: max drawdown about 1.7 points shallower, trade-off about 0.8 point of CAGR; caveats list the SPY/EFA/EEM splices inside the period. |
| 6 | USMV, GLD, DBMF added at 10% to the uploaded `examples/sample_portfolio_returns.csv`, no objective | Three sentences, no verdicts. GLD and DBMF lower volatility and drawdown the most (correlation about 0.1 and -0.35); USMV barely changes volatility (correlation about 0.8). Only 1 or 2 stress windows: the uploaded stream starts in 2016 and is monthly, so the short windows are skipped. |
| 7 | ZLB.TO added to XIC.TO, VFV.TO and XBB.TO separately, objective Lower volatility | One row per benchmark. Add for XIC.TO and VFV.TO; for XBB.TO volatility rises slightly and the verdict drops to Consider. |
| 8 | DBMF and KMLM, 10% taken from bonds in Balanced, stock-bond hedge | Weights XBB.TO 30%, equities unchanged. Against pro rata funding (demo 2) the excess return is about +1.0% a year instead of +0.3% for DBMF, because the money comes out of bonds rather than partly out of equities; volatility falls less (0.8 instead of 1.3 points). |
| 9 | 6% GLD paid 50/50 by bonds and equities, in Balanced, Growth and Conservative | Each sentence reads "taken 3.0% from bonds (XBB.TO) and 3.0% from equities"; Balanced becomes 57% equities / 37% bonds / 6% GLD, Growth 67 / 27 / 6, Conservative 37 / 57 / 6. |
| 5 | Profile sweep of DBMF, stock-bond hedge | One row per profile: Add in each, High confidence. Volatility falls by 1.0 to 1.6 points and max drawdown improves by 2.4 to 4.7 points depending on the profile. DBMF's history (from 2019) misses the GFC, euro crisis, 2015-16 and Q4 2018 (stated in the caveats). In the app, switch the objective to Faster recovery to see a sweep where the verdict changes by profile. |

`examples/custom_profile_example.csv` is a ready file for the Custom profile upload (check 12).

## 3. Checklist (in the app)

Open the **Security review** workspace (first in the sidebar). Defaults: USMV, added at 10% to the benchmark XIC.TO, no objective.

| # | Do this | Expected result |
|---|---|---|
| 1 | Look at the sidebar workspace selector | Security review and Backtest only. Markets and Research are hidden. |
| 2 | Look at the inputs | 1. Securities to analyse; 2. Add it to (Benchmark, Blend, Uploaded portfolio, Investor profile); 3. How much; 4. Objective (optional); Advanced. |
| 3 | Type `SPLV, QUAL` next to USMV and press **Run** | Header "Add 10% · 2013-08 to ... (common to all)". Three sentences, then "What adding it changes" with one row per security, then growth and drawdown charts. No verdict, no score. Click a column header: rows sort by value. |
| 4 | Benchmark: add VFV.TO next to XIC.TO; run | Six rows and a Portfolio column; no sentences above 4 results. The chart offers a picker. |
| 5 | Portfolio: **Blend**; keep the default weights; run | Results against "My blend". **Replace a holding** is now offered (one portfolio with several holdings): replace VFV.TO with USMV. |
| 6 | Portfolio: **Uploaded portfolio**; upload `examples/sample_portfolio_returns.csv`; run | Results against "MODEL PORTFOLIO" from 2016-11. Details > Scenarios: Basis "month-end"; windows shorter than two months are "no data (monthly series...)". |
| 7 | Securities: upload `examples/sample_portfolio_returns.csv` as a return stream instead | It is analysed as a security. |
| 8 | Objective: **Lower volatility**; run | Each sentence ends with the verdict, confidence and trade-off; Verdict and Confidence columns appear; Details has Verdict and Ranking tabs. |
| 9 | Portfolio: **Investor profile**; pick Balanced and Growth | One block of rows per profile. Details: a picker for the portfolio. |
| 10 | Open **Details** | Scenarios, Regimes, Portfolio Impact, Diversification, Factor Exposures, Fund Review, Relative to Benchmark, Robustness, Objectives (or Verdict and Ranking with an objective), Charts, Methodology. |
| 11 | Securities: Pick a whole sleeve: **Alternatives / diversifiers**; run | One table, no sentences; sort by Change in volatility. |
| 12 | Blend: Upload a file: `examples/custom_profile_example.csv`; name it; **Save as a profile** | Weights shown (XIC.TO 25%, VFV.TO 35%, XEF.TO 10%, XBB.TO 25%, GLD 5%). After saving, the profile appears under Investor profile on the next page load. |
| 13 | Profile: Growth only. **Replace a holding** VFV.TO, security USMV, objective Reduce drawdown | Add (demo 4). Details > Portfolio Impact target weights: USMV 31.5%, no VFV.TO. |
| 14 | Pick a whole sleeve: **Canadian bonds**; objective Income generation; profile Income | Details > Verdict lists the XBB.TO / ZAG.TO pair as functionally equivalent, naming the cheaper and more liquid fund. Fund Review shows expense ratio, AUM and daily value traded. |
| 15 | Advanced: **Each security's own history** | Months differ by row; a long-history security keeps its whole record. |
| 16 | Advanced: tick **Also run a profile sweep**, run with DBMF first | Details gains a **Profile Sweep** tab with 7 rows. |
| 17b | Details > **Fund Review** with USMV added to XIC.TO | Columns: USMV, then "XIC.TO, without (period)"; no SPY. Advanced > Benchmark: **Each security's sleeve benchmark** brings back "SPY (benchmark)". |
| 17c | Look at "What adding it changes" | Teal for changes that help (lower volatility, higher return), rust for those that hurt; the excess return is geometric. |
| 17d | Investor profile Balanced, security DBMF, **Funded from: Bonds** | The sidebar shows "XBB.TO 40% → 30%"; the sentence reads "taken from bonds (XBB.TO)". **Chosen holdings** with XBB.TO at 15%: "XBB.TO 40% → 25%". Asking for more than the holdings hold shows a warning. |
| 17e | Same run (Balanced, DBMF from bonds) | "Costs and income of the portfolio": MER about 0.11% without and 0.19% with (DBMF's 0.85% on 10%), yield about 2.3% to 2.5%; the sentence ends with the same figures. With an uploaded portfolio both read n/a. |
| 17f | Balanced, GLD, weight 6%, **Funded from: Split between bonds and equities**, share paid by bonds 50 | Sidebar: "Bonds pay 3.0% of the portfolio, equities 3.0%" and every holding's change (XBB.TO 40.0% → 37.0%, VFV.TO 27.0% → 25.7%...). The request strip reads "Paid for by: 3.0% from bonds (XBB.TO) and 3.0% from equities". Set bonds to 100 with a 45% weight: a warning that bonds hold only 40%. |
| 17g | Look at the page after a run, then change any sidebar input | Request strip, sections numbered 1 to 7 each with a one-line purpose, Details tabs each starting with what they answer. After the change, a red note says the results no longer match the sidebar until **Run** is pressed. |
| 17 | Advanced: Uploaded returns are in **USD** | The uploaded stream is converted to CAD month by month (USD/CAD). |
| 18 | Open **Factor Exposures** | Loadings and t-stats for market, size, value, momentum, quality, low vol; R2; mega-cap tilt. The note under the table names the factor source: Ken French where reachable, otherwise the ETF spreads (with the caveat on factor legs). |
| 19 | Open **Details > Charts** | Stress-window bars and (with an objective) the rolling metric; the monthly series table under them. Red / sand palette. |
| 20 | Press **Full report (Excel)** and open it | Opens on the Executive Summary (the portfolio with the security against the portfolio without it), then About (every table, with links), Performance (statistics, CIPM appraisal, trailing periods, calendar years, drawdowns, monthly returns), Performance Analysis, then the review sheets: Summary, Verdict, Ranking, Fund Review, Relative to Benchmark, Diversification, Portfolio Impact, Scenarios, Regimes, Factor Exposures, Robustness, (Profile Sweep), Methodology, Series. Red titles, black headers, percentages as numbers. The Summary sheet holds the sentences and the summary table; every Details tab matches a sheet. |
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
