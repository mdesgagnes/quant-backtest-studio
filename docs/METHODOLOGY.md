# Security review: methodology

How the Security review reaches a verdict. Every number below is set in `config/*.yaml`; the code is in `qbt/security_review/`. Judgment calls and how to change them are in [DECISIONS.md](DECISIONS.md).

## 0. How a review flows

The page, the command line and this document follow the same six steps (`qbt/security_review/engine.py` lists the function behind each):

1. **Securities**: tickers or uploaded return streams.
2. **Portfolio** ("without"): a benchmark ticker, a blend, an uploaded portfolio, or a profile. It is also the benchmark for every relative statistic unless another is chosen.
3. **Two portfolios**: "with" adds the security at the add weight, paid for by every holding pro rata, by the bonds or the equities, by a split between them, or by chosen holdings; or replaces one holding.
4. **Same months, same measures**: both portfolios over the months they share, measured with the Backtest's statistics (section 4); the difference is the answer.
5. **Checks**: stress windows, regimes, rolling windows, other add weights, data caveats (sections 6 to 9).
6. **Verdict** (optional): with an objective, the measures are scored (section 5).

## 1. Data

- **Source field.** Yahoo Finance `Adj Close` (download with `auto_adjust=False`): split- and distribution-adjusted, i.e. total return with distributions reinvested. `Close` and `Dividends` are kept only for the distribution yield.
- **Currency.** Base currency CAD. A US-listed price is multiplied by that day's USD/CAD close (Yahoo `CAD=X`, CAD per USD; before 2003-09-17 the Fed H.10 series stored in `data/fx/USDCAD_history.csv`). The rate may be carried forward up to 5 sessions to meet a price date, never backward. CAD-listed securities are not converted: a CAD-listed fund holding foreign assets keeps its own hedging policy (XEF.TO is unhedged, so its CAD return already includes the currency).
- **Monthly returns.** Month-end adjusted price to month-end adjusted price. The month in progress is dropped. A security's first, partial month is dropped. A month with no price is a gap (NaN), and the following month's return is NaN too: a return never spans two months and is never filled.
- **Daily data** is used only inside stress windows (section 6).
- **Uploaded return streams** (a portfolio, or a security) are read by `qbt/returns_input.py` (any date column, percent or decimal, CSV or Excel). Daily or weekly returns are compounded to calendar months. A USD stream is converted month by month: (1 + r) x (1 + change in CAD per USD) - 1. A stream has no daily prices and no fund data (costs, yield).
- **Risk-free.** Canadian 3-month T-bill, monthly average (Bank of Canada Valet `V122531`); each month earns the yield known at its start. If unreachable: 0, and the Methodology sheet says so.
- **Cache.** `data/cache/prices/<TICKER>.csv`, refreshed when older than 3 days; macro series in `data/cache/macro/` (7 days). A failed download falls back to a stale copy, with a note.

## 2. Profiles and building blocks

| Block | Ticker | Proxy before inception | Own monthly history from |
|---|---|---|---|
| Canadian equity | XIC.TO | EWC (USD -> CAD) | 2001-03 |
| US equity | VFV.TO | SPY (USD -> CAD) | 2012-12 |
| International developed | XEF.TO | EFA (USD -> CAD) | 2013-05 |
| Emerging markets | XEC.TO | EEM (USD -> CAD) | 2013-05 |
| Canadian aggregate bonds | XBB.TO | none | 2000-12 |

Equity portion: Canadian 30%, US 45%, international 20%, EM 5%. Bond portion: XBB.TO 100%.

| Profile | Equity / bonds | XIC.TO | VFV.TO | XEF.TO | XEC.TO | XBB.TO |
|---|---|---|---|---|---|---|
| 100% Equities | 100/0 | 30.0% | 45.0% | 20.0% | 5.0% | 0% |
| Max Growth | 80/20 | 24.0% | 36.0% | 16.0% | 4.0% | 20% |
| Growth | 70/30 | 21.0% | 31.5% | 14.0% | 3.5% | 30% |
| Balanced | 60/40 | 18.0% | 27.0% | 12.0% | 3.0% | 40% |
| Conservative | 40/60 | 12.0% | 18.0% | 8.0% | 2.0% | 60% |
| Income | 30/70 | 9.0% | 13.5% | 6.0% | 1.5% | 70% |
| 100% Bonds | 0/100 | 0% | 0% | 0% | 0% | 100% |

**Rebalancing.** Monthly, quarterly (default: end of March, June, September, December), annual (end of December) or none. Between dates, weights drift with each holding's return.

**Splicing.** Before a block's first own monthly return, the proxy's CAD monthly return is used as is (no rescaling). The same rule applies to daily returns inside stress windows. Every spliced month is listed in "Profile holdings and proxy splices" (Methodology sheet); a candidate whose analysis period overlaps a spliced stretch gets a flag naming the holding, proxy and end month. With all blocks spliced, every profile runs from 2003-05 (EEM's first month) to the last complete month.

**Custom profiles.** Any tickers and weights (typed or from a CSV/Excel file with two columns, ticker and weight); weights are normalised to 1. A custom holding that matches a building-block ticker uses that block's proxy; any other is used on its own history only. Saved custom profiles go to `config/profiles_custom.yaml`.

## 3. Analysis period and the two portfolios

**The portfolio** ("without") is any one of: a benchmark ticker (100% that ticker), a blend of tickers, an uploaded portfolio return stream (one holding: the stream itself), or a preset or saved investor profile. Several can be given; each is a separate review with the same securities and settings.

- **Analysis period** (default "common"): the months in which every candidate, every sleeve benchmark and every profile holding have a return. Option "each": each candidate on its own overlap with the profile and its benchmark (used by the scenario matrix).
- **Without** = the profile. **With** = Add mode: the candidate at weight w (default 10% of the whole portfolio), funded either from every holding (each scaled by 1 - w) or, with *Funded from*, only from the bonds, the equities or chosen holdings: those are reduced pro rata to their weights by w in total (XBB.TO 40% to 30% for 10% taken from bonds in Balanced) and every other holding keeps its weight. Bonds and equities are the building blocks' asset classes, and for other tickers the universe sleeve's. w cannot exceed what the chosen holdings hold; taking all of one holding is the same as Replace. **Split**: each side pays a set share of w, whatever the portfolio's own split (6% gold paid 50/50 takes 3 points from bonds and 3 from equities, each side pro rata within itself); each side must hold its part, so the largest w a split can pay is min(side's weight / side's share). Replace mode: the named holding's whole weight goes to the candidate. Both portfolios are simulated with the same rebalancing rule from the same first month.
- **Simulation.** Monthly returns at target weights, drifting between rebalance dates, rebalanced at month end on the schedule; no trading costs, cash, execution lag or taxes. This is not the Backtest engine (`qbt/engine.py`, daily, with costs, cash, execution lag and trades); see DECISIONS.md.

## 4. Metrics

**Costs and income of the portfolio.** Weighted MER = sum of target weight x each fund's expense ratio (yfinance `fund_operations` "Annual Report Expense Ratio", else `netExpenseRatio`; cached 30 days). Weighted yield = sum of target weight x each fund's trailing 12-month distributions over its last price. Shown without and with the security; n/a when any holding lacks the figure. The MER is already inside the adjusted prices, so returns are net of it; the figure is shown, not deducted again.

**Standard.** Every standalone figure is taken from `qbt.metrics.summary`, the function behind the Backtest's statistics table, so the two modules report identical numbers for the same series: population moments, geometric annualisation, nothing annualised under 12 months, Sharpe / Sortino / Calmar net of the same CAD T-bill series (Calmar as in Bacon), VaR and CVaR as positive losses. Any return difference is a geometric excess, (1 + r) / (1 + b) - 1. Beta (and up / down beta) is withheld below R² 0.70.

**Benchmark.** Unless another is chosen (Advanced), the benchmark for every relative statistic is the portfolio the security is added to ("without"). The sleeve benchmark of `config/universe.yaml`, or any ticker, can be chosen instead.

### 4a. Standalone (each candidate, its benchmark and the profile)
CIPM conventions from `qbt/metrics.py`: population moments, geometric annualisation.

| Metric | Definition |
|---|---|
| CAGR | (prod(1 + r))^(12/n) - 1 |
| Volatility | population SD of monthly returns x sqrt(12) |
| Sharpe | (CAGR - annualised T-bill) / volatility |
| Sortino | (CAGR - annualised T-bill) / (downside deviation x sqrt(12)); downside deviation = RMS of shortfalls below the T-bill over all months |
| Max drawdown | worst fall from a previous peak of the month-end value path (starting value 1 included) |
| Calmar | CAGR / abs(max drawdown) |
| Ulcer index | RMS of the drawdown path, in percent |
| Longest underwater | longest run of consecutive months below the previous peak |
| Recovery from max drawdown | months from the trough to the first month back at the old peak (blank if not recovered) |
| Skew, excess kurtosis | population moments |
| CVaR 95% | mean of monthly returns at or below the 5th percentile |
| Worst month, worst 12 months | minimum monthly return; minimum compounded rolling 12-month return |
| Distribution yield | trailing 12-month distributions / last close (own currency) |
| Distribution variability | coefficient of variation of the last 5 full calendar years' yields (distributions / average close) |
| Expense ratio, AUM, average daily value traded | yfinance (`funds_data.fund_operations`, else `info.netExpenseRatio`; `info.totalAssets`; `averageVolume` x last price), cached 30 days. Shown, not scored: adjusted prices are already net of fees |

### 4b. Within the sleeve (candidate vs its sleeve benchmark)
Sleeve benchmarks: US equity SPY, Canadian equity XIC.TO, international/EM XEF.TO, Canadian bonds XBB.TO, global/US bonds AGG. Real assets and alternatives have none: the profile is used and flagged.

| Metric | Definition |
|---|---|
| Active return | (1 + CAGR_c) / (1 + CAGR_b) - 1 |
| Tracking error | population SD of (1 + r_c)/(1 + r_b) - 1, x sqrt(12) |
| Information ratio | active return / tracking error |
| Beta, correlation | OLS of excess returns over the T-bill |
| Upside / downside capture | geometric mean return in benchmark up (down) months over the benchmark's (`qbt/cipm.py`) |
| Upside / downside beta | beta estimated separately in benchmark up and down months |
| Hit rate in down months | share of benchmark down months in which the candidate did better |
| Relative drawdown | max drawdown of cumulative (1 + r_c) / cumulative (1 + r_b) |
| Excess-return correlation | correlation between candidates of their returns minus their own sleeve benchmark |

### 4c. Factor and style exposures (explanatory only)
OLS of the candidate's excess return on: market = sleeve benchmark minus T-bill, and five monthly US long-short style factors. With `source: auto` (default) size, value, momentum and quality come from the Ken French data library (SMB, HML, Mom, RMW; read from its CSV zips, cached 7 days) when it can be reached, otherwise from ETF spreads (USD, total return): size IWM-SPY, value IWD-IWF, momentum MTUM-SPY, quality QUAL-SPY. Low volatility is always USMV-SPY (Ken French has none). The source used is printed in the Methodology sheet; on the network this was built on, Ken French was unreachable and the ETF spreads were used. Classical t-stats and R2; at least 36 months, otherwise not run. Alpha shown annualised. A candidate that is itself a leg of an ETF spread is flagged (its loading is mechanical). **Mega-cap tilt**: slope of the candidate's active return on QQQ minus RSP. Limitation: the style spreads are US; for Canadian and international funds they are a rough guide.

### 4d. Portfolio impact (with minus without, on the analysis period)
`d_vol`, `d_sharpe`, `d_sortino`, `d_calmar`, `d_cagr`, `d_mdd`, `d_ulcer`, `d_cvar`, `d_worst_12m` (differences of 4a). Plus:

| Metric | Definition |
|---|---|
| `d_div_ratio` | change in diversification ratio = sum(w_i sigma_i) / sigma_p at target weights |
| `corr_to_portfolio` | correlation of the candidate with the profile (without) |
| `worst10_corr` | same, only in the profile's worst 10% months |
| `cand_tail_return` | candidate's average return in those months |
| `d_tail_return` | average (with - without) in those months |
| `d_up_month_return` | average (with - without) in the profile's up months |
| `d_max_underwater`, `d_underwater_share` | change in longest underwater spell (months) and share of months underwater |
| `d_portfolio_yield` | change in the weighted distribution yield |
| `corr_to_home_equity` | correlation with XIC.TO |
| `usd_beta` | slope on the monthly USD/CAD change, controlling for SPY in USD |
| `d_megacap_beta` | change in the profile's slope on QQQ-RSP (controlling for SPY in CAD) |
| `size_loading` | the size coefficient of 4c |
| `cand_regime_<k>`, `d_regime_<k>` | candidate average monthly return, and average (with - without), in regime k (section 7) |
| `d_scen_<k>` | change in the profile's return over stress window k (section 6) |

### 4e. Risk of ruin and decumulation
Circular block bootstrap of the analysis period's monthly returns of both portfolios, 12-month blocks, 2000 paths, seed 42. The same resampled months are applied to the portfolio with and without the candidate (common random numbers), so the difference is the candidate, not noise.
- **Risk of ruin:** probability that the 10-year path's max drawdown reaches 20% (`d_prob_dd20`) and 30% (`d_prob_dd30`).
- **Decumulation:** wealth 1, withdrawals of 4%/12 of initial wealth each month, raised 2% a year, for 30 years; a path at 0 stays at 0. Depletion probability (`d_depletion_prob`) and 10th / 50th / 90th percentile ending wealth (`d_p10_wealth`, `d_median_wealth`), nominal.
Limitation: the bootstrap only reshuffles the analysis period; a short or unusually good history gives an optimistic distribution.

## 5. Objectives and verdict rules

The objective is optional. Without one, the page and the summary show only the measured changes; every objective is still scored in the workbook's Verdict sheet.

Each objective in `config/objectives.yaml` is a weighted list of metrics. For each metric: `score = clip(direction x (value - centre) / scale, -1, +1)` (direction +1 when higher is better). Composite = weighted average of the scores that have a value (missing ones are dropped and the weights renormalised; the share dropped is reported).

| Objective | Metrics (weight) | Rolling metric |
|---|---|---|
| Lower volatility | d_vol 50%, down_capture 20%, d_cvar 30% | d_vol |
| Reduce drawdown | d_mdd 50%, d_ulcer 25%, d_worst_12m 25% | d_mdd |
| Reduce risk of ruin | d_prob_dd20 50%, d_prob_dd30 50% | d_mdd |
| Tail protection | d_tail_return 40%, cand_tail_return 30%, worst10_corr 30% | d_tail_return |
| Upside participation | up_capture 40%, d_up_month_return 40%, d_cagr 20% | d_cagr |
| Diversify return sources | corr_to_portfolio 35%, worst10_corr 35%, d_div_ratio 30% | corr_to_portfolio |
| Improve risk-adjusted return | d_sharpe 40%, d_sortino 30%, d_calmar 30% | d_sharpe |
| Faster recovery | d_max_underwater 40%, d_underwater_share 30%, d_ulcer 30% | d_ulcer |
| Inflation protection | d_regime_high_inflation 50%, cand_regime_high_inflation 50% | d_regime_high_inflation |
| Rising-rate protection | d_regime_rising_yields 50%, cand_regime_rising_yields 50% | d_regime_rising_yields |
| Stock-bond correlation hedge | d_regime_stock_bond_down 35%, cand_regime_stock_bond_down 35%, d_scen_rate_shock_2022 30% | d_regime_stock_bond_down |
| Income generation | dist_yield 40%, yield_stability 20%, d_portfolio_yield 40% | d_cagr |
| Currency / home-bias diversification | corr_to_home_equity 40%, usd_beta 30%, corr_to_portfolio 30% | corr_to_portfolio |
| Reduce concentration | megacap_tilt 40%, d_megacap_beta 40%, size_loading 20% | d_megacap_beta |
| Sequence risk in decumulation | d_depletion_prob 50%, d_p10_wealth 30%, d_median_wealth 20% | d_sortino |

**Verdict.** Add if composite >= +0.30; Consider if >= +0.05; else Do not add. Then two caps (Add -> Consider): a failed guardrail (default: CAGR given up > 1.0% a year; Upside participation: drawdown deepened > 3 points; Income: CAGR given up > 0.5%) or a rolling-window share below 60%.

**Confidence.** High by default. Moderate if the analysis period has fewer than 60 months, Low if fewer than 36. One level down for each of: rolling share below 60%; verdict changes between Add and Do not add across add weights 5%, 10%, 20%; history covering fewer than half of the stress windows. Low if more than half of the objective's weight has no data.

**Ranking.** Candidates sorted by composite; ties at a saturated score broken by the unclipped composite. The "Score breakdown by component" table shows value, score, weight and contribution for every candidate and metric.

**Rationale.** Deciding facts = the 3-4 components with the largest absolute contribution. Main trade-off = the worst-scoring metric outside the objective among CAGR, upside capture, Sharpe, max drawdown, volatility, CVaR and portfolio yield (score <= -0.2); otherwise a failed guardrail; otherwise "no material trade-off". Caveats = caps, confidence reasons and data flags.

**Functionally equivalent.** Two candidates with tracking error between them below 1% a year and either excess-return correlation above 0.95, or, when one of them barely departs from the benchmark (own tracking error below 1%), raw correlation above 0.95. The app says the choice comes down to cost or liquidity and states which of the two is cheaper and which trades more.

**Profile sweep.** The same candidate, objective and mode run on each of the 7 preset profiles.

## 6. Stress windows

From `config/scenarios.yaml`: GFC 2007-10-09 to 2009-03-09; Euro crisis 2011-04-29 to 2011-10-03; 2015-16 oil/China 2015-05-21 to 2016-02-11; Q4 2018 2018-09-20 to 2018-12-24; COVID 2020-02-19 to 2020-03-23; 2022 rate shock 2022-01-03 to 2022-10-14; March 2023 regional banks 2023-03-08 to 2023-03-24; August 2024 yen carry unwind 2024-07-16 to 2024-08-05; April 2025 tariffs 2025-04-02 to 2025-04-08.

Daily, CAD. Return from the close of the start date to the close of the end date. Both portfolios are bought at target weights on the start date and left to drift (rebalancing inside a few weeks would be fiction). Reported: candidate return and max drawdown, portfolio return with and without, the change, portfolio max drawdown with and without, and the candidate's daily correlation with the portfolio. A window that the candidate (or a holding, after splicing) does not fully cover is **no data**, never filled.

**Uploaded streams.** When the security or a holding is an uploaded monthly stream, the whole comparison for that window uses month-end values: from the last month end on or before the start date to the first month end on or after the end date (`Basis` = month-end). Windows shorter than 45 days (COVID, March 2023, August 2024, April 2025) are skipped for it, since a month-end figure would mostly measure the recovery.

## 7. Regimes (monthly)

| Regime | Rule | Data |
|---|---|---|
| Rising 10-year yields | month-end yield minus previous month-end > 0 | ^TNX (US 10-year, Yahoo) |
| High inflation | CPI year-on-year > 3% | FRED CPIAUCSL if reachable, else Bank of Canada CPI V41690973 (from 1995). On this network: Bank of Canada |
| Equity bear | SPY below its 10-month average, read at the previous month-end | SPY adjusted, USD |
| High volatility | monthly average VIX above the median of all earlier months (from 60 months of history) | ^VIX |
| Stocks and bonds both down | 100% Equities profile and 100% Bonds profile both negative in the month | profiles, CAD |

For each regime and its opposite: months, candidate average monthly return and annualised volatility, contribution to the portfolio (weight x average), portfolio average with and without, and the change. A month with no input has no label.

## 8. Robustness

- **Rolling windows:** for each objective's rolling metric, the share of 36-month windows (step 1 month) in which the metric is on the good side of its centre.
- **Add-weight sensitivity:** composite and verdict (before caps) at 5%, 10% and 20%, with the changes in volatility, CAGR, max drawdown and Sharpe.
- **Consistency:** share of covered stress windows, and of regimes, in which the portfolio did better with the candidate.

## 9. Data quality flags

Short overlap (< 36 months: Low; < 60: Moderate); a profile holding spliced inside the analysis period; daily moves above 25%; gaps longer than 10 calendar days; prices unchanged for 10+ sessions; failed downloads; stress windows without data; no sleeve benchmark; factor-leg candidates; distributions from option writing or leverage (from `universe.yaml`); a yield above 5% that exceeds the 5-year total return by more than 2 points (possible return of capital); average daily value traded below $1M (thin trading); currency method (always stated).
