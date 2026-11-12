# UI parity checklist

The contract for the interface redesign. Every numbered item is something a
user can see or do in the current Streamlit app (`app.py`, `qbt/snapshot_ui.py`,
`qbt/groups_ui.py`, `qbt/lab/ui.py`) at commit `8c170f5`. The redesigned app
must do each one, with the same figures.

How to read it:

- **Source** gives where the behaviour lives today, as `app.py:line`.
- **Golden** names the recorded scenario that locks the item
  (`tests/golden/expected/<scenario>/`). "manual" means no automated lock
  exists: check it by hand against the legacy app.
- Items marked **UNCLEAR** need a decision from the owner before they are
  rebuilt. They are collected again in the last section.
- Wording in notes, flags and help text is part of the contract unless an
  item says otherwise. Layout, colour and typography are not.

Golden scenarios: `momentum_file` (MF), `imported_weights` (IW),
`factor_signal` (FS), `macro_blend` (MB), `return_stream` (RS).

Count: 259 items in 28 groups, plus 5 open questions.

---

## A. Shell and access

1. **Three workspaces**: Backtest, Markets, Research, switched from the top of the control panel. Each has its own controls and page; switching keeps the other workspaces' loaded state for the session. Source: `app.py:774`. Golden: MF (control inventory).
2. **Optional password gate**: when the host defines a `password` secret, a login screen ("Restricted access", password field, Open) blocks everything until the right value is entered; a wrong value shows "Incorrect password."; with no secret the app opens directly. Source: `app.py:367`. Golden: manual.
3. **Session state**: the last run, loaded markets, Monte Carlo and sweep results, the account holdings typed in, and Research state all survive any interaction until the browser session ends or a new run replaces them. Source: `st.session_state` throughout. Golden: manual.
4. **Page header** per workspace: "Quant Backtest Studio / Backtest", "Quant Backtest Studio / Return streams", "Markets / Market monitor", "Research Lab / Find an idea, test it honestly, decide. Research only: nothing here trades." Golden: MF, RS.
5. **Notes fold**: an explanatory note longer than 170 characters shows its first sentence and a "More" control that reveals the rest; shorter notes show whole. Source: `app.py:540`. Golden: MF (text).
6. **Flags**: warnings about data or settings appear as distinct warning blocks next to the thing they concern, not as transient toasts. Golden: all.
7. **Tables**: figure columns are right-aligned; a table of 40 rows or fewer is shown whole without its own scroll box; longer tables scroll. Source: `app.py:576`. Golden: manual.
8. **Sign colouring in tables**: named columns are coloured by sign (gain, loss, muted for zero/blank) and some are emphasised in bold (for example Excess). Source: `app.py:469`. Golden: manual (values are locked, colour is not).
9. **Heat-shaded grids**: value grids (return by horizon, seasonality, regime overview, contribution by period) shade each cell by sign and by size relative to the 95th percentile of absolute values, capped. Source: `app.py:430`. Golden: manual.
10. **Mobile**: usable at phone width; tab strip scrolls horizontally, KPI figures shrink. Source: `app.py:354`. Golden: manual.
11. **Start screen (Backtest, before any run)**: a prompt to pick a source and run, three "Where to start" routes, and the list of all 21 models, each expandable to its description. Source: `app.py:3779`. Golden: manual.

## B. Backtest controls: presets and configuration

12. **Saved presets list**: pick a saved preset; shows its strategy, save time and note. Source: `app.py:1175`. Golden: manual.
13. **Load preset**: loads the stored configuration and Builder rules into the controls. **UNCLEAR (U1)**: today only controls without a stored value pick up the loaded configuration. Source: `app.py:1187`.
14. **Delete preset**: removes it and confirms.
15. **No presets yet** message pointing to the Export tab.
16. **Ephemeral-host warning** when the app runs on Streamlit Community Cloud (path under `/mount/src`): saved presets do not survive a redeploy. Source: `qbt/store.py:192`.
17. **Export all presets** as `qbt_presets.json`. Golden: MF (export `preset_export`).
18. **Restore presets from file** (JSON), merging into the store, with a result message.
19. **Import a configuration file** (YAML) and apply it; errors are reported. Same caveat as item 13.

## C. Backtest controls: data

20. **Source**: Yahoo Finance, File (CSV / Excel), or Return stream. Return stream replaces the whole page (group V onward). Golden: all.
21. **Preset universe** (Yahoo): Custom or one of 11 presets; picking one fills the symbols box, sets the matching benchmark and cash proxy, and shows the preset's note. Source: `app.py:2384`.
22. **Symbols box**: one per line or comma/semicolon separated, upper-cased; help explains `.TO` / `.V` suffixes.
23. **Start and End dates** (Yahoo), 1970 to today; End defaults to today.
24. **Price files upload** (File source): several CSV / XLSX / XLS / XLSM / TXT files; long, wide, one-instrument-per-file and Yahoo's own CSV are all read; every sheet of a workbook is read. Golden: MF.
25. **Upload summary**: "N instrument(s) from M source(s): fields found", and a "What was read" panel with the per-source table and up to 10 warnings. Golden: MF.
26. **No file yet**: format hint and a **Template (CSV)** download (`price_template.csv`).
27. **Unreadable upload**: "The file could not be read: no prices were found..." on the main page.
28. **Investable universe**: multiselect of the loaded symbols, all selected by default. Golden: MF.
29. **Comparison benchmark**: None, any loaded symbol, "Blend of several", or "Another strategy". Golden: MF, MB.
30. **Blended benchmark**: weights box (`XIC.TO:60, XBB.TO:40`), rebalance frequency (annual by default), parsed weights echoed as percentages, a flag when nothing parses; components outside the universe are downloaded too. Golden: MB.
31. **Strategy as benchmark**: note pointing to Compare strategies; the benchmark convention and friction choices disappear. Golden: manual.
32. **Benchmark convention**: total return, price return + dividends reinvested at rebalance, or price return only. Golden: MF (default).
33. **Benchmark frictions**: any of commission, slippage and market impact, management fee; off by default. Golden: manual.
34. **Price convention** (Yahoo): total return, or price return + cash dividends (default).
35. **File price convention**: detect, total return or price return, with the resolved reading echoed ("Read as ..."). Golden: MF.
36. **Cash remuneration**: fixed rate or any loaded symbol; a preset's cash proxy is offered even when outside the universe.

## D. Backtest controls: currency

37. **Base currency**: Off, CAD, USD, EUR, GBP, CHF, JPY, AUD (plus a loaded config's own).
38. **Per-instrument currency overrides** box (`VOD.L: GBp`).
39. **Detected-currency table** for the universe, benchmark and cash proxy, with how each was decided.
40. **Flag when currencies are guessed** from the ticker suffix; note when nothing needs converting; flag when resolution fails.
41. **Mixed-currency warning** when the base currency is off and the symbols look to be quoted in different currencies.

## E. Backtest controls: signal data and model

42. **Exogenous series upload**: several CSV / Excel files; duplicate column names are suffixed with the file name. Golden: FS, MB.
43. **Publication lag** slider, 0 to 120 calendar days, shown once a file is loaded, with a "N series loaded, from ... to ..." note. Golden: FS, MB.
44. **Signal origin**: built-in model or imported target weights. Golden: IW.
45. **Target weights upload** with sheet picker for multi-sheet workbooks; read errors are reported. Golden: IW.
46. **Weights template download** sized to the current universe (`target_weights_template.csv`).
47. **Row handling**: None or Scale to 100%. **Rebalance calendar**: File dates or Engine calendar. Golden: IW (defaults).
48. **Model picker**: 21 models sorted by label, with the model's description shown underneath. Golden: MF, FS, MB.
49. **VIX auto-fetch** for the VIX-Adaptive Momentum model, with a note on success and a flag on failure; an uploaded series still takes priority.
50. **Flag when a model needs exogenous series** and none is loaded.
51. **Cash hurdle note** for models that test against cash: says which cash series is used, or flags that none is set.
52. **Model parameters**: one control per parameter, typed (integer, decimal, checkbox, choice, series picker, formula box, text), with min, max, step, default and help from the model definition. Golden: MF, FS, MB.
53. **Portfolio construction**: single strategy, by asset class, core + strategy, blend of strategies.
54. **By asset class**: assign each instrument a class (preset tags pre-filled, otherwise Unclassified), set a budget per class, with a total note or under-budget flag.
55. **Core + strategy**: core share slider, core holdings box, "keep core holdings out of the strategy universe", and the parsed core echoed with the strategy's share.
56. **Blend of strategies**: 2 to 4 sleeves, each with a model, a budget and its own parameters; total note or under-budget flag.

## F. Backtest controls: execution, comparison, frictions

57. **Rebalance frequency**: daily, weekly, monthly, quarterly, semi-annual, annual. Golden: MB (quarterly).
58. **Trading day rule**: last, first, a day of the month (slider), Nth weekday, last weekday, with weekday and ordinal pickers.
59. **Anchor months**: month for annual; quarter cycle for quarterly; month pair for semi-annual.
60. **Calendar preview**: a plain-language description of the resulting rebalance rule. Golden: MF.
61. **Execution lag** 0 to 5 days. Lag 0 raises the look-ahead warning on the page.
62. **Execution price**: close, or open marked at the close; a flag when an uploaded file has no opens. Golden: FS (open).
63. **Trim the warm-up period** (on by default), **Initial capital**, **Max leverage**, **Trade whole units only**.
64. **Reinvest cash between rebalances** (off, weekly, monthly, quarterly), **Cash buffer %**, **Or when cash exceeds %**.
65. **Compare strategies**: 0 to 3 extra models, each with its own parameters; suggested defaults skip the main model and models with no data; a picker for which one is the benchmark when "Another strategy" is chosen. Golden: MB.
66. **Commission** and **Slippage** in bps. Golden: MF (defaults).
67. **Market impact**: none, square-root scaled by volatility (coefficient Y), or square-root with a fixed cost at 10% of volume (bps); a flag when no volume data exists. Golden: FS.
68. **Per-instrument slippage multipliers** box.
69. **Cash rate** and **Management fee** (annual %). Golden: FS.
70. **Currency conversion** (only with a base currency): exchange spread, foreign-cash policy (convert at each trade / hold until needed), interest on foreign cash.
71. **Fee liquidation pays the same commission and slippage** (on by default).
72. **Run backtest** button. Nothing is simulated until it is pressed. Golden: all backtest scenarios.

## G. Backtest: running

73. **Validation**: blocking problems appear as errors and stop the run; warnings appear but do not. Source: `qbt/config.py:159`, `app.py:3421`.
74. **Progress messages**: "Loading prices...", "Loading exchange rates...", "Generating signals and running simulation...".
75. **Failure message**: "The backtest stopped: ..." and the previous result stays on screen.
76. **A new run clears** the parameter surface, trading-day sweep, Monte Carlo and generated detailed report of the previous one. Source: `app.py:3772`.
77. **Controls do not change the result on screen** until Run is pressed again; exports describe the run on screen, not the controls.
78. **Warm-up trimming and shared start**: with several strategies, all start on the latest date any first holds a position; each pays its entry cost on that date. Golden: MB.
79. **Context strip** above the tabs: run label, period, instrument count, rebalance frequency, benchmark, CAGR, and "warm-up trimmed" when applicable. Golden: MF.
80. **Footer line** on the Export tab: run time, execution lag, rebalance, bps of costs per trade.

## H. Backtest tab: Results (the reference screen)

81. **Warm-up note** when the record was trimmed, naming both dates and why. Golden: MF.
82. **Next-rebalance one-liner** pointing to Positions. Golden: MF.
83. **KPI strip**: CAGR, Volatility, Sharpe, Max Drawdown, Calmar, Sortino, each with the benchmark's figure beneath, toned positive or negative (Max Drawdown always negative, Volatility neutral). Golden: all.
84. **Log scale toggle** (on by default) for the equity chart only. Golden: MF.
85. **Equity chart**: strategy and benchmark rebased to 100 at their common start. Golden: all.
86. **Drawdown chart**: both series, the strategy filled. Golden: all.
87. **Monthly returns heatmap** (omitted for data coarser than monthly). Golden: all.
88. **Return distribution** histogram with the 5% VaR marked. Golden: all.
89. **Rolling Sharpe**, window `min(252, max(63, n/6))` sessions, zero line. Golden: all.
90. **Full statistics** table: every metric for the strategy and the benchmark. Golden: all.
91. **CIPM appraisal** panel (collapsed): the grouped measures table, the methodology note and the methodology caption. Golden: all.
92. **Trailing periods** table (1M to 20Y and since inception, with Annualized, From, To, benchmark and Excess); windows the history does not cover are omitted. Golden: all.
93. **Calendar years** table with Partial flag, benchmark and Excess, a bar chart of the strategy's yearly return, and the partial-year note. Golden: all.
94. **Main drawdown episodes** table (top 6). Golden: all.

## I. Backtest tab: Compare

95. **Empty state** explaining how to add strategies. Golden: MF.
96. **Summary note** with the shared window and which strategy is the benchmark. Golden: MB.
97. **Log scale toggle**, **equity chart** and **drawdown chart** for every strategy plus the benchmark. Golden: MB.
98. **Full statistics** across strategies. Golden: MB.
99. **Head-to-head** table (excess CAGR, tracking error, information ratio, beta, correlation, months ahead) with its note. Golden: MB.
100. **Calendar years** across strategies. Golden: MB.
101. **Return correlation** matrix. Golden: MB.

## J. Backtest tab: Signals

102. **No-model note** for imported weights. Golden: IW.
103. **No-score note** for models that allocate by rule.
104. **Current standing** table: score, rank, target weight, held weight, status (Held / Ordered / Not held), sorted by score, with the explanatory note and count of names out. Golden: MF.
105. **Score through time** for chosen instruments (top five by default). Golden: MF.
106. **Rank through time** with its note. Golden: MF.
107. **Score distribution now** bar chart. Golden: MF.
108. **Downloads**: score history, current standing (CSV). Golden: MF.
109. **Target weights through time** area chart with its note. Golden: MF.

## K. Backtest tab: Positions

110. **Weights file check** (imported weights): four KPIs, warnings, and the first rows retained. Golden: IW.
111. **Sleeves** table, warnings, note and weight-by-sleeve chart (for sleeve constructions).
112. **Composition over time** area chart including cash. Golden: all.
113. **Turnover, average exposure, cumulative friction cost** KPIs. Golden: all.
114. **Management fee** block: three KPIs, note, and every deduction in dollars. Golden: FS.
115. **Dividends** block: three KPIs, note, cumulative chart. Golden: MF.
116. **Average weight by instrument** bar chart. Golden: all.
117. **Next rebalance**: headline, stale-data flag when the last price is more than one business day old, four KPIs (trades, trade date, signal date, estimated cost). Golden: MF.
118. **Account size box**: rescales units and amounts; weights unchanged.
119. **Next trades table** and **CSV download**, with the notes on sizing and dates. Golden: MF.
120. **Trades for your account**: editable units-held table (pasteable), cash, whole-units toggle; prompt when empty; otherwise four KPIs, the trade table, CSV download and notes. Holdings persist for the session. Golden: manual.
121. **Current position** table (weight and dollar value, with cash) and **CSV download**. Golden: all.
122. **Cash reinvestment** KPIs when reinvestments occurred.
123. **Cash-flow ledger** panel and **CSV download**. Golden: MF.
124. **Currency block** (multi-currency runs): four KPIs, exposure-by-currency table, note.
125. **Trade log**: last 400 trades formatted, note, fee-liquidation flag, **full CSV download**; "No trades were generated." when empty. Golden: all.

## L. Backtest tab: Attribution

126. **Group by** security or asset class; **period table by** year, quarter or month. Golden: MF (defaults).
127. **Asset-class editor** per instrument, opened automatically when some are unclassified; changes the grouping only.
128. **Pooling** into "Other holdings" beyond 12 securities, with a note.
129. **Three KPIs**: total return with the reconciliation figure, largest contributor, largest detractor. Golden: all.
130. **Cumulative contribution** chart with the Portfolio line. Golden: all.
131. **Contribution** and **share of risk** bar charts and the summary table with its note. Golden: all.
132. **Contribution by period** heat table with its note. Golden: all.
133. **Downloads**: daily contributions, contribution by period. Golden: MF.
134. **Brinson-Fachler**: editable benchmark weights and return proxies per category, attribution period (month/quarter), linking method (Frongello, Carino, Menchero, GRAP), three KPIs, effects table, reconciliation error alert, methodology caption, **period CSV download**. The settings also feed the Excel report. Golden: MF (defaults).

## M. Backtest tab: Tax

135. **Intro note**, and "No dividend or share data available" when the run lacks them. Golden: all.
136. **Tax settings**: marginal rate, capital gains inclusion rate, US withholding recoverable, eligible and foreign dividend rate overrides, per-ticker classification overrides.
137. **Four KPIs**: total tax, Tax Cost Ratio with its label, realized gains, years with tax owed. Golden: MF.
138. **Tax Cost Ratio chart** for the strategy and the benchmark, with notes. Golden: MF.
139. **Pretax vs after-tax** chart in dollars, with note. Golden: MF.
140. **Tax by source and year** chart with note; **By tax year** table. Golden: MF.
141. **Every realized sale** panel. **Downloads**: annual summary, realized sales. Golden: MF.

## N. Backtest tab: Robustness

142. **Stability over time**: folds slider (3 to 10), fold table, Sharpe-by-fold chart, high-dispersion flag, and the in/out-of-sample table. Golden: all.
143. **Parameter stability**: first and second parameter (including rebalance frequency), metric, **Compute parameter surface** button; result as 3D surface or heatmap (heatmap only when an axis is a frequency), or a line for one parameter; spread note and the deflated-Sharpe flag. Not offered for imported weights (a note says why). Golden: manual.
144. **Trading-day sensitivity**: **Sweep trading days** button, three KPIs, chart, table, large-gap flag; a note for daily rebalancing. Golden: manual.
145. **Cost sensitivity** chart and table with its note. Golden: all.
146. **Stress test periods**: category filter, custom periods box, KPIs, coverage flags, bar chart (with benchmark when set), table, **CSV download**. Golden: all.
147. **Around dated events**: event source (stress-period starts, benchmark's worst days, rebalance dates, own dates), days before and after, KPIs, verdict callout, CAR chart, per-event table. Golden: MF.
148. **Sampling uncertainty**: simulations and block-size sliders, **Run Monte Carlo** button, fan chart, three KPIs, percentile table. Golden: MF.

## O. Backtest tab: Data

149. **Sessions, start, end** KPIs. Golden: all.
150. **Per-instrument diagnostic** table and up to 20 flags. Golden: all.
151. **Exogenous series** block: three KPIs, per-series table, warnings, which columns were read as factor or macro, a chart of chosen series after the lag, **aligned-series CSV download**. Golden: FS, MB.
152. **Correlation of daily returns** matrix. Golden: all.
153. **Normalized prices** chart (log). Golden: all.
154. **Prices used** CSV download. Golden: MF.

## P. Backtest tab: Builder

155. **Rule list** seeded with a default filter and score (or a loaded preset's rules). Golden: MF.
156. **Add filter rule / Add score rule**; **combine filters with** AND or OR.
157. **Per rule**: indicator (12), window where relevant, comparison and target (a number or another indicator with its own window) for filters, treatment and weight for scores, enable toggle, move up, move down, remove; a plain-language description of each rule.
158. **Compiled expressions** for filter and score. Golden: MF.
159. **Preview** against the loaded universe: four KPIs, score chart for chosen instruments, share-passing chart for the filter, low-coverage flag, formula errors as flags. Golden: MF.
160. **Send to the Custom Formula model** copies both expressions into that model's parameters.
161. **Indicator reference** panel.

## Q. Backtest tab: Export

162. **Tearsheet (HTML)** download. Golden: all.
163. **Detailed report**: **Generate** button, then **HTML** and **PDF** downloads; includes whichever of sweep, day sweep and Monte Carlo have been run. Golden: manual.
164. **Holdings**: current (CSV) and history (CSV). Golden: all.
165. **Save as a preset**: name, note, result message, ephemeral-host flag.
166. **Configuration** shown as YAML and downloadable. Golden: all.
167. **Excel module picker** (11 modules) and **Comments for the CIO** box (when the executive summary is included).
168. **Excel window**: from / to dates inside the run; an invalid window falls back to the full period with a flag; a windowed export re-measures statistics over the window and marks robustness tests as full-run.
169. **Daily series (CSV)**. Golden: all.
170. **Full report (Excel)**, built only when clicked, reflecting the settings chosen on the other tabs (fold count, stress category and custom periods, tax settings, attribution benchmark, Monte Carlo and sweeps when run). Golden: all (17 sheets, every cell).
171. **Excel failure** shows a flag instead of breaking the page; a failed build still downloads a one-sheet workbook with the error.

## R. Return stream: controls and header

172. **Returns from** a file or Yahoo Finance. Golden: RS (file).
173. **Returns file upload** with sheet picker, **value scale** (detect, decimals, percentages) and **date order** (detect, day first, month first). Golden: RS.
174. **Yahoo tickers** (up to 10 analysed) and **return frequency** (daily, weekly, monthly).
175. **Base value ($)**.
176. **Risk-free rate**: US 3-month T-bill, Canada 3-month T-bill (Bank of Canada), fixed annual rate (with its rate box), none, or any column of the file; flags when a download fails or history is short. Golden: RS (fixed).
177. **Empty state** with format guidance and a **Monthly CSV template** download.
178. **Series analysed** (up to 10) and **Compare against**. Golden: RS.
179. **From / To** dates and **Common period only** toggle; invalid ranges are refused.
180. **Four header KPIs** (frequency, observations, period, scale or source), the date-reading note, and every reader warning. Golden: RS.
181. **Context strip** and, for several series, **Detailed charts for**. Golden: RS.

## S. Return stream tabs

182. **Results, one series**: six KPIs with benchmark sub-values. Golden: RS.
183. **Results, several series**: side-by-side table.
184. **Log scale toggle**, value chart and drawdown chart. Golden: RS.
185. **Correlation** with a frequency choice limited to the data's own or coarser, and notes on precision. Golden: RS.
186. **Trailing periods** and **Calendar years** tables for every series, with a bar chart. Golden: RS.
187. **Detail**: monthly heatmap, distribution, rolling Sharpe. Golden: RS.
188. **Full statistics**, **drawdown episodes**, **period returns** tables. Golden: RS.
189. **Stress tests**: category, custom periods, per-series summary, chart, table, partial-coverage flag, full detail panel, **CSV download**, and the event-study panel. Golden: RS.
190. **Market regimes**: Overview heat table or one dimension (timeline chart, bar chart, table, notes); a flag when market data cannot be downloaded. Golden: RS (offline flag only).
191. **Robustness**: folds slider and table, Sharpe-by-fold chart, Monte Carlo sliders, button, fan chart, KPIs, table. Golden: RS.
192. **Export**: tearsheet (HTML), series (CSV), statistics (CSV), module picker, CIO comments, full report (Excel). Golden: RS.

## T. Return stream: Fund review

193. **Intro** and the 0% risk-free flag.
194. **Load a review config** (YAML), with flags for columns missing from the file.
195. **Product, market factor, bond factor, display name, split date, MER**.
196. **Portfolio funding test**: editable sleeve table (sleeve, weight, group), slice added, rebalance.
197. **Distributions and tax**: NAV, income, dividends, capital gains, return of capital, source, three tax rates.
198. **Extra stress windows and thresholds**: windows box, alpha t-stat, crash tolerance, minimum Sharpe gain, replica cost.
199. **Run fund review** button with failure message.
200. **Verdict strip**, checks table, key points, notes.
201. **Downloads**: report (PDF), deck (PowerPoint), tables (Excel), config (YAML).
202. **Review body**: headline table, growth and relative charts, regression table, stress table, calendar, drawdown, payoff and rolling charts, funding table and chart.

Golden for 193 to 202: RS locks the controls only; the review itself is covered by `tests/test_fund_review.py`. Otherwise manual.

## U. Markets: controls and shared

203. **Watchlist preset** and **symbols box**. Default watchlist: XIC.TO, XBB.TO, SPY, EFA, EEM, GLD, TLT, VNQ.
204. **From / To** dates (default: five calendar years back), **Reference** instrument, **Rolling window** (126 sessions).
205. **Load markets** button; nothing loads until pressed; failures are reported.
206. **Empty state** describing the four areas.
207. **Context strip**: instrument count, period, reference, window, load time.
208. **Names**: tables, selectors and legends show "TICKER — full name"; dense charts keep bare tickers. Source: `qbt/names.py`.

## V. Markets tabs

209. **Snapshot**: instrument picker; window pills (YTD, 3M, 6M, 1Y, 2Y, 3Y, 5Y, Max) or custom dates; hero card (last, 1 day, window return, from 52-week high, history start, percentile rank, sparkline); insufficient-history flag.
210. **Snapshot performance chart**: mode (cumulative, drawdown, rolling), scale (linear, log), compare-with picker, on the shared period.
211. **Snapshot summary table** with percentile bars and the captions on percentiles.
212. **Snapshot calendar returns**: table, heatmap (partial years starred) or monthly grid.
213. **Screener**: trend-window pills, text filter, column picker, sort column and direction, minimum 1Y return, maximum volatility; "Showing N of M" strip; median-of-shown strip; table with sparkline and percentile bars; captions.
214. **Screener downloads**: CSV and Excel. **Distribution** panel with metric picker and median line.
215. **Groups**: source pills (presets, asset classes, my groups) with their pickers or text box; **Load groups**; changed-definition flag; missing-instrument flag.
216. **Groups table**: statistic, sort, trend window, small-sample dagger, captions, **CSV download**.
217. **Group performance** chart (groups, mode, window) and **Inside a group** (a full screener for one group's members).
218. **Chart**: instrument, candles or line, volume toggle, with fallback to a line and a flag when OHLC is missing; TradingView Lightweight Charts with the attribution logo and credit line (a licence condition).
219. **Chart compare**: several instruments rebased on their common period, with notes and a no-overlap flag.
220. **Performance**: return-by-horizon heat grid (1D to ITD), rebased chart, risk-against-return scatter.
221. **Risk**: risk profile table, distance-from-high chart and table, underwater chart for chosen instruments.
222. **Relationships**: relative strength chart, correlation matrix with frequency choice and notes, most and least correlated pairs, rolling correlation, rolling beta.
223. **Seasonality**: average return by month heat table, years-of-data table, and the cautionary flag.
224. **Data**: coverage table, up to 20 flags, downloads (prices, performance grid, risk grid).

Golden for 203 to 224: manual (needs live prices); the numbers are covered by `tests/test_snapshot.py` and `tests/test_groups.py`.

## W. Research: controls

225. **Universe**: preset, symbols, dates, benchmark, synthetic demo data toggle, **Load universe**, with errors and up to three warnings.
226. **Active hypothesis** picker with the trial count.

## X. Research tabs

227. **Start here**: purpose, workflow checklist with live progress ticks, decision table, glossary by area, limits.
228. **Ledger**: new-hypothesis form, hypotheses table, status setter, experiments under a hypothesis, **Export ledger (JSON)**.
229. **Factor Lab**: look-ahead, factor families, **Run screen**; five KPI cards; verdict callout; table with "show every statistic"; small-universe warning.
230. **Factor detail**: KPI cards, IC over time, IC by horizon, return by bucket, top-minus-bottom curve, **Backtest this factor in the Strategy Lab**.
231. **Strategy Lab, run**: strategy, parameters, execution and costs, **Run backtest**, KPI cards, equity and drawdown charts.
232. **Strategy Lab, validate**: optional parameter grid, trial count or no-hypothesis warning, **Run validation**, grade chip, plain summary, every check with question, evidence, advice and threshold, three diagnostic charts.
233. **Strategy Lab, keep**: **Save run card + dossier**, dossier (HTML) and run card (ZIP) downloads, warnings, **Register for decay monitoring**.
234. **Risk X-Ray**: weights source, trailing window, custom weights box, ten KPI cards, callout, money-versus-risk chart, table, correlation matrix.
235. **Event Study**: events box, days before and after, **Run event study**, results as in item 147; a warning when the dataset has no benchmark.
236. **Monitor**: registered strategies table, **Scan for decay**, per-strategy status chips and KPI cards, **Retire a strategy**.
237. **Assistant**: chat with tool-call trace when an API key is configured; otherwise instructions for the MCP server. **UNCLEAR (U4)**.
238. **Per-tab guide**: a glossary panel at the bottom of each Research tab.

Golden for 225 to 238: manual; the numbers are covered by `tests/test_lab_*.py`.

## Y. Charts (shared behaviour)

239. Hovering a time-series chart shows every series' value at that date (unified hover) with a vertical guide on equity charts.
240. Legends toggle series on and off; charts can be zoomed and reset.
241. The first series of a chart is the subject and is drawn heavier; the benchmark and others are context.
242. Bars are coloured by sign; grouped bars use one colour per series.
243. The 3D parameter surface can be rotated.
244. Every chart has a title that states what is plotted and, where relevant, the anchoring date.

## Z. Exports (complete list)

245. Backtest: `qbt_presets.json`, `price_template.csv`, `target_weights_template.csv`, `scores.csv`, `signal_now.csv`, `next_trades.csv`, `my_trades.csv`, `current_holdings.csv` (two buttons), `cash_flows.csv`, `trades.csv`, `daily_contributions.csv`, `contribution_by_<period>.csv`, `benchmark_attribution_periods.csv`, `tax_by_year.csv`, `tax_realized_sales.csv`, `stress_periods.csv`, `exogenous_series_aligned.csv`, `prices.csv`, `tearsheet_<model>.html`, `detailed_report_<model>.html`, `detailed_report_<model>.pdf`, `holdings_history.csv`, `config_<model>.yaml`, `series.csv`, `backtest_<model>[_from_to].xlsx`.
246. Return stream: `returns_template.csv`, `stress_periods.csv`, `tearsheet_<series>.html`, `return_streams.csv`, `statistics.csv`, `return_stream_<series>.xlsx`, `fund_review_<name>.pdf / .pptx / .xlsx / .yaml`.
247. Markets: `watchlist_prices.csv`, `performance.csv`, `risk.csv`, `watchlist_screen.csv`, `watchlist_screen.xlsx`, `groups.csv`.
248. Research: `qbt_lab_ledger.json`, `dossier.html`, `run_card.zip`.
249. File names, column names, column order and values are identical to today's. Timestamps inside reports are the only content allowed to differ.

## AA. Performance behaviour to preserve

250. Yahoo Finance downloads are reused for one hour for the same symbols, dates and fields.
251. Regime market data and Bank of Canada yields are reused for six hours.
252. Uploaded files are parsed once per distinct content.
253. The next-rebalance replay is computed once per run.
254. The Excel workbook is built only when its download is requested.
255. Parameter surface, trading-day sweep, Monte Carlo and the detailed report run only on request and are kept until the next run.

## AB. Engine and library behaviour (out of scope, listed so nobody touches it)

256. `qbt` runs without any interface (`run_from_config`, `python -m qbt.lab`, the MCP server). The redesign must not break these.
257. The existing 235 unit tests keep passing unchanged.
258. Exported documents keep their own design: the tearsheet stays light for print, the detailed report stays dark.
259. TradingView attribution stays on any page that uses Lightweight Charts.

---

## Open questions (UNCLEAR)

- **U1. Loading a preset or YAML configuration is partial today.** Verified by applying a configuration file to a fresh session (tickers SPY and TLT, start 2015, benchmark SPY, lookback 63, top 1, quarterly, lag 2, commission 1, slippage 3, cash rate 3%, fee 1%). Applied: start date, rebalance, execution lag, commission. Not applied: symbols, investable universe, model parameters, slippage, cash rate, management fee. The benchmark was reset to None, because SPY was not among the symbols still in the box. Controls that already hold a value in the session keep it; only the rest take the file. This is an accident of how Streamlit widgets keep state, not a decision. Should the redesign load a configuration completely (recommended), or reproduce today's behaviour?
- **U2. Stale wording.** The Robustness tab says "These four tests" above seven sections, and its stress-test note says "sixteen ... episodes, 1987 through the 2024 yen carry-trade unwind" while the list holds 24 periods through the 2025 tariff tantrum. The README says 19 models; there are 21. Keep verbatim, or correct while rebuilding?
- **U3. Re-running on every interaction.** Today the walk-forward folds, in/out-of-sample split and cost sensitivity are re-simulated whenever anything on the page changes, and their latest values feed the Excel report. The redesign would compute them once per run (and again only when the folds slider changes). The numbers are the same; only the timing differs. Confirm this is acceptable.
- **U4. Assistant tab.** It prints an MCP command with a machine-specific Python path (`C:/Users/marcd/miniconda3/envs/quant/python.exe`). Keep as is, or derive the path from the running interpreter?
- **U5. Brand colour.** Decided 2026-11-12: the new app uses brass as its single accent (see `docs/DESIGN.md`). The classic app and the exported reports keep the palette in `qbt/brand.py`, which is not edited.
