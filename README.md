# Quant Backtest Studio

Web app for backtesting systematic strategies. Yahoo Finance or CSV/Excel
file data, a verifiable simulation engine, a full robustness suite,
exogenous-data-driven signals, imported target weights, and one-click exports
(tearsheet, holdings, workbook).

The `qbt/` package is usable standalone, with no interface. The Streamlit
interface (`app.py`) is just a front end: all logic lives in pure functions.

---

## 0. Two workspaces

The sidebar switches between them.

**Backtest** simulates a strategy: data, signal, execution, frictions,
robustness, exports. Everything from section 2 onward describes it.

**Markets** describes a watchlist rather than simulating one. Return across
ten horizons, risk and drawdown, relative strength against a reference,
rolling correlation and beta, the most and least correlated pairs, and
seasonality. It shares the same data layer, so anything the backtester can
load, the monitor can analyse.

Everything in Markets is derived from prices, and that is deliberate. Free
fundamental endpoints are inconsistent between tickers, change shape without
warning, and come back empty for anything outside large-cap US equity. A
screen showing a blank P/E for half a watchlist is worse than one that never
promised it. Price-derived analytics either work for every instrument or
fail visibly for all of them.

### Rebasing, and why it is done on the common period

Any chart comparing two instruments rebases them to 100. The only honest
place to anchor is the first date **both** have data. Anchoring each series
to its own first observation puts them level on different dates, so an
instrument that listed in 2015 appears to start alongside one that had been
running since 2005, and the gap between the lines measures nothing.

Both comparison charts use the common period and say which date that is.
Where a selection shares no overlap at all, they say so rather than drawing
something meaningless.

### Charting

Price and equity charts are drawn with **TradingView Lightweight Charts**,
which is TradingView's own library, Apache 2.0 licensed, about 35 KB, loaded
from a CDN into a sandboxed iframe. It is used because the crosshair, the
synced price and time labels, and the pan-and-zoom behaviour over tens of
thousands of bars are what a price chart is judged on, and a general-purpose
plotting library does not match them.

**Attribution is a licence condition.** Crediting TradingView as the product
creator and linking to tradingview.com on any user-facing page is required.
That is satisfied by the attribution logo in the chart corner and the credit
line printed beneath every chart. Neither should be removed.

The version is pinned to 4.2.3 on purpose. The series API changed between
v4 and v5, so tracking "latest" would mean the charts silently stop
rendering the day the CDN rolls forward.

Where it appears: **Markets, Chart tab only** -- candlesticks with a volume
pane, or a line, for one instrument, plus a rebased comparison of several.

It is deliberately **not** used in the backtester. It was, briefly, drawing
the equity curve with trades marked on it, and testing that view against the
Plotly reference found the benchmark anchored to its own first date rather
than to the strategy's. With a benchmark whose history starts earlier, every
relative reading off the chart was wrong -- 0.795 where the true figure was
0.698. The bug is fixed, but the backtester is where the numbers have to be
beyond question, so it keeps the chart that has been checked against the
engine since the beginning.

Nothing is scraped. TradingView's terms prohibit it, and the unofficial
libraries that pull their data work by impersonating a browser session,
which breaks without warning and risks the address being blocked. Prices
come from the same free source as the rest of the application.

---

## 1. Local setup

```bash
cd quant-backtest-studio
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 2. Deployment -- access from anywhere

The project is ready to publish: dependencies verified in a clean
environment, optional password protection, `Dockerfile` and `render.yaml`
included.

### Streamlit Community Cloud (free, the shortest path)

**Step 1 -- publish to GitHub.** A GitHub account and Git installed are enough.

```powershell
# Windows
.\deploy\deploy.ps1 -User <your-github-username>
```

```bash
# macOS / Linux
./deploy/deploy.sh <your-github-username>
```

The script initializes the repo, commits, sets the remote, and pushes.
Create the empty repo on github.com first (private is fine), under the same
name.

**Step 2 -- deploy.** On `share.streamlit.io`, sign in with GitHub, then
**New app** -> repo, branch `main`, file `app.py` -> **Deploy**. The first
boot takes two to three minutes.

The resulting URL looks like `https://<repo>.streamlit.app`. On iPhone,
open it in Safari, then "Add to Home Screen" for an icon and a full-screen
view.

**Step 3 -- restrict access.** The URL is public by default, even for a
private repo. Two protections, stackable:

- **Settings -> Sharing**: restrict to specific email addresses.
- **Settings -> Secrets**: paste `password = "..."`. The app then shows a
  login screen. Without that secret, it opens normally -- the same behavior
  as locally.

Never commit `.streamlit/secrets.toml`: it is already in `.gitignore`, and
`secrets.toml.example` is the template.

Free-tier limits: 1 GB of memory, sleeps after a few days of inactivity,
auto-restarts on the next visit. Plenty for a universe of a few dozen
instruments.

### Other hosts

| Host | File provided | Notes |
|---|---|---|
| Hugging Face Spaces | `deploy/huggingface_spaces.md` | Free, stable, private Spaces |
| Render | `render.yaml` | Free tier with sleep |
| Railway / Fly.io | `Dockerfile` | Paid, no sleep |
| Local network | -- | `streamlit run app.py --server.address 0.0.0.0` |

### Updating after deployment

Streamlit Cloud and Hugging Face redeploy on every push:

```bash
git add -A && git commit -m "Updates" && git push
```

---

## 3. Format of imported files

**Wide format** (most common) -- one date column, then one price column per
instrument:

```csv
Date,XIC.TO,ZEB.TO,XEI.TO
2020-01-02,30.15,35.02,22.41
2020-01-03,30.08,34.95,22.38
```

**Long format** -- auto-detected if the columns `date`, `ticker` (or
`symbol`, `instrument`), and `close` (or `price`, `nav`) are present.

Prices must be **adjusted** for dividends and splits. Yahoo data is adjusted
automatically (`auto_adjust=True`).

`example_prices.csv` is a template.

---

## 3 bis. Exogenous series

Anything that is not a price but can drive a signal: economic data,
fundamental ratios, earnings revisions, in-house scores, signals computed
elsewhere. Multiple files accepted.

**Macro series** -- one column per indicator. Applies to the whole portfolio
and acts as a regime filter (`example_economic_data.csv`):

```csv
Date,ISM_MANUFACTURING,BBB_CREDIT_SPREAD,POLICY_RATE
2010-01-31,51.1,1.98,2.06
```

**Cross-sectional factor** -- one column per symbol, named exactly like the
instrument. Used to rank the universe (`example_fundamental_factor.csv`):

```csv
Date,XIC.TO,ZEB.TO,XEI.TO,ZLB.TO
2010-03-31,14.66,13.70,14.12,14.80
```

Each column's role is inferred automatically: a header matching a universe
symbol becomes a factor; the rest become macro series. The "Data" tab shows
the split that was applied.

### The publication lag

This is the setting that decides whether the test is credible. A data point
dated January 31 is not known on January 31. The slider shifts the series'
index before any alignment, which makes look-ahead structurally impossible
rather than dependent on the user's vigilance.

| Data type | Reasonable lag |
|---|---|
| Monthly economic indicator (ISM, jobs, CPI) | 20 to 45 days |
| Quarterly company financials | 45 to 90 days |
| Credit spreads, rates, market data | 0 to 1 day |
| Ratio computed in-house from known prices | 0 to 1 day |
| Data revised after publication | the revision's delay, not the first release's |

The app flags a monthly or quarterly frequency paired with a lag under 15
days.

### Models that consume these series

| Model | Expects | Does |
|---|---|---|
| Macro Gate | a macro series | drives exposure by regime, otherwise sits in cash |
| Imported Factor Ranking | one column per symbol | holds the top N ranked names, optional blend with price momentum |
| Imported Signals | one column per symbol | reads a signal directly (-1/0/1, continuous score, or rank) and handles sizing and frictions |

A model with no valid series does not produce a phantom signal: the macro
gate stays invested, the other two stay in cash. The app flags this before
running.

---

## 3 ter. Imported target weights

To test an allocation produced elsewhere -- a spreadsheet, an allocation
committee, another engine's output -- in the same simulator. Choose
"Imported target weights" under **Signal**.

```csv
Date,XIC.TO,ZEB.TO,XEI.TO,ZLB.TO
2015-01-31,0.3213,0.1470,0.3572,0.1745
2015-02-28,0.2980,0.1820,0.3400,0.1800
```

Long format (`date, ticker, weight`) is also recognized, and values can be
fractions or percentages: the scale is detected and reported. The
**CSV template** button produces a file sized to the current universe.

What the file supplies: the weights and the dates. What the engine adds:
the execution lag, drift between dates, turnover costs, cash remuneration,
statistics, and the applicable robustness tests.

Automatic checks, shown in the "Positions" tab:

- dates snapped to the last available trading day (a calendar month-end
  falling on a Sunday rolls to Friday);
- columns outside the universe ignored and listed;
- universe instruments missing from the file treated as zero and listed;
- rows under 100% left as-is, the remainder in cash;
- rows over budget flagged, with an option to rescale;
- negative weights detected and flagged.

Two possible calendars:

- **File dates** -- only trades on the supplied dates. Between two rows,
  weights drift with the markets. This is faithful to a mandate rebalanced
  on those specific dates.
- **Engine calendar** -- the portfolio is additionally reset to the last
  known weights at every engine checkpoint. Higher turnover, drift
  corrected.

The parameter-stability test does not apply here: there is no parameter to
vary. The other three tests work normally.

---

## 4. Exporting results

Every export reproduces the backtest currently on screen. Everything lives
under the **Export** tab, plus a few convenience buttons in "Positions".

| Export | Format | Contents |
|---|---|---|
| Tearsheet report | HTML | One printable page: KPIs, equity curve, drawdown, monthly returns, return distribution, full stats table, top drawdown episodes, current holdings, engine assumptions |
| Current holdings | CSV | Instrument, weight, dollar value as of the last date |
| Holdings history | CSV | Full weight time series (same shape used internally for drift) |
| Trade log | CSV | Every trade, with weight before/after and the change |
| Daily series | CSV | Equity, return, exposure, cash, turnover, cost, benchmark |
| Full workbook | XLSX | Everything below, in one file |
| Trailing periods | on screen + tearsheet | 1M, 3M, 6M, YTD, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, since inception, against the benchmark |
| Calendar years | on screen + tearsheet | Year-by-year return and excess, partial years flagged |
| Configuration | YAML | Exact reproduction of the run (imported files are not included; their name, settings, and lag are) |

**The workbook is complete by design.** Anything visible in the report is
in it, so nothing has to be re-derived by hand. For a backtest that is
fifteen sheets: Notes (every setting the figures depend on), Statistics,
Trailing Periods, Calendar Years, Drawdowns, Monthly Returns, the full
daily Series, Current Holdings, Holdings History, Target Weights, Trades,
Rebalance Dates, Parameters, Prices, Exogenous Series and Data Quality. For
an imported return stream it is eight, covering everything the return
series supports; sheets that would need position data are absent because
none exists. Percentages are written as real numbers, not formatted
strings, so they can be charted or recomputed directly.

The tearsheet is the fastest way to share a result: it is a single
self-contained HTML file (Plotly loaded from a CDN), styled for print. Open
it in a browser and use Print -> Save as PDF for a clean PDF, or share the
HTML file directly.

---

## 3 quater. Preset universes and the benchmark

**Presets.** The Yahoo Finance source offers ready-made universes (Canadian
ETFs, Canadian asset classes, TSX sectors, U.S. SPDR sectors, global asset
classes, U.S. factors, sixty-forty blocks). Picking one fills the symbol
box and sets a matching benchmark and cash proxy; everything stays
editable. Add your own by appending an entry to `UNIVERSES` in
`qbt/presets.py`.

Two caveats apply to every preset, and to any universe assembled today out
of instruments that exist today: these are the funds that *survived*, and
several launched in the 2010s, so an early start date silently shortens the
usable history. The Data tab reports the first usable date per instrument.

**A blended benchmark.** Choosing "blend of several" as the comparison
benchmark lets the bar be a fixed-weight portfolio rather than one fund:
`XIC.TO:60, XBB.TO:40` for a balanced mandate, say. Components outside the
investable universe are downloaded alongside it, so the benchmark is not
restricted to what the strategy can trade.

The blend runs through the same engine as everything else rather than being
averaged, because a rebalanced 60/40 is not the weighted average of its two
return series -- the drift between rebalances is real. Its rebalance
frequency is set separately and is worth thinking about: an unrebalanced
60/40 drifts toward equities over a long window, which quietly makes it a
harder bar in a bull market and an easier one in a drawdown. Annual is the
default, matching most policy benchmarks. Frictions are zero, since nobody
pays commission on a measuring stick.

**Benchmark convention.** Set independently of the strategy, because the
two questions are unrelated: how you model your own portfolio's dividends
is not how the index you are judged against handles them.

| Option | Dividends |
|---|---|
| Total return (adjusted close) | Reinvested continuously, inside the price series |
| Price return + dividends reinvested at rebalance | Credited as cash, put back to work on the rebalance calendar |
| Price return only | Excluded |

The default is total return, and it is almost always the right one: a
price-return benchmark is not the index anyone actually tracks, and it
understates the bar by roughly its dividend yield every year. On a Canadian
equity ETF that is around two to three points annually - more than enough
to turn a losing strategy into an apparent winner. The third option exists
so the gap can be measured, not because it makes a fair comparison.

---

## 4 bis. Analysing a return stream

Set **Source** to "Return stream" to skip prices and signals entirely and
analyse a track record directly: a fund's monthly history, a composite, a
GIPS table, or the output of an engine that lives elsewhere.

```csv
Date,Strategy,Benchmark
2020-01-31,0.0213,0.0185
2020-02-29,-0.0154,-0.0210
```

Long format (`date, name, return`) is also recognized. Daily, weekly,
monthly or quarterly: the frequency is inferred from the spacing of the
dates and drives the annualization, so a monthly file is annualized at 12
periods and not 252. Values may be decimals (0.0213) or percentages (2.13);
the scale is detected and reported, with a manual override if the guess is
wrong.

The app flags the mistakes that quietly invalidate this kind of analysis: a
file of index levels rather than periodic returns, a scale that produces
impossible single-period moves, too few observations for the annualized
figures to mean much.

You get the full statistics, drawdown table, period-return grid, fold
stability, Monte Carlo, and the tearsheet. Positions, frictions, cost
sensitivity and parameter sweeps do not apply: there is no portfolio being
simulated, only a realized stream.

**The reader is a separate, self-contained module**
(`qbt/returns_input.py`) that does not call the shared file loader used by
prices, exogenous series or target weights. It exists on its own precisely
so it can be far more forgiving without touching anything the simulation
depends on -- a return-stream file is rarely a clean export, and this
reader is built to survive that:

- **Excel or CSV, detected from the file's actual bytes**, not just its
  name, so an upload never fails because a wrapper somewhere lost the
  filename.
- **Several encodings and separators tried in turn**, since a French-locale
  export is often semicolon-separated in CP-1252, not comma-separated
  UTF-8.
- **The date column is found by content, not by name.** Every column is
  tried as dates; whichever parses the most wins, with a small preference
  for a date-like header. A column named "Period" or "Mois" is found the
  same way one literally named "Date" is.
- **Both date conventions are tried and compared** for the genuinely
  ambiguous case -- every day in the file at 12 or under -- scored by
  which reading produces more realistic spacing. Which one was used is
  always reported, and the **Date order** control in the sidebar overrides
  it when the guess is wrong. Nothing can rule that out for a file with no
  day over 12 anywhere in it, since at that point both readings are
  equally valid arithmetic and only the person who made the file knows
  which is true.
- **Numbers are cleaned before being given up on**: percent signs, currency
  symbols, parenthesised negatives, and both the North American
  (1,234.56) and European (1 234,56) conventions, decided once per column
  from the punctuation actually present.
- **A column that still won't parse is named in the warnings**, with a
  sample of the values that defeated it, instead of silently vanishing.

A multi-sheet workbook with no sheet chosen tries sheets in order and keeps
the first that yields real data, so a cover page as sheet one no longer
needs to be found and skipped by hand.

---

## 4 ter. Period reporting

The Results tab and the tearsheet both carry two standard tables.

**Trailing periods** - 1M, 3M, 6M, YTD, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y and
since inception, each against the benchmark with the excess alongside.

Every window is measured from the last observation *at or before* the
cutoff, not the first one after it. On daily data the difference is a
session; on a monthly series it is a whole period, and taking the first
point inside the window would make YTD start at the January close and
silently drop January's own return.
Anything longer than a year is annualized; anything shorter stays
cumulative, and an `Annualized` column records which convention each row
uses. Annualizing a three-month figure implies that quarter repeats four
times, which is precisely the extrapolation that makes a good quarter look
like a track record.

A window the history does not cover is left out entirely rather than
measured over whatever exists and labelled as though it were complete - a
"10Y" row computed from six years of data is worse than no row.

**Calendar years** - the return for each year, with the benchmark and the
excess, plus a bar chart. A partial first or last year is flagged in a
`Partial` column, since a strategy that started in October will otherwise
show a suspiciously calm first year.

---

## 4 quater. Frequency awareness

Every statistic adapts to the sampling of the data it is given, because a
figure computed on monthly returns is a monthly figure regardless of what
the label says.

- **VaR and CVaR** are labelled with the actual frequency: "VaR 95%
  (monthly)" on a monthly stream, not "(daily)".
- **Best / Worst / % Positive** report the natural bucket - months for
  daily, weekly and monthly data, quarters for quarterly data. A stream
  coarser than monthly is never resampled into months, which would invent
  buckets that were never observed.
- **Drawdown durations** are counted in observations and named accordingly:
  sessions, weeks, months or quarters. A `Calendar Days` column sits
  alongside, which is the one figure independent of sampling. An eighteen-
  month drawdown previously read as "18 days".
- **The monthly heatmap** is omitted entirely for quarterly or annual
  streams rather than drawn with fabricated cells.
- **Annualization** follows the detected frequency: 12 periods for monthly
  data, 4 for quarterly, 252 for daily.

---

## 4 sexies. Signals: reading the model between trades

The **Signals** tab shows the quantity the model ranks on, per instrument,
through time. Weights say what was chosen; the score says how close the
runner-up came and how a name is trending while it sits out of the book,
which is what makes the next rebalance foreseeable rather than a surprise.

It carries: where each name stands right now, with score, rank, target
weight, held weight and status; the score and the rank through time; the
current spread across the universe; and both tables as CSV.

**Score decides the ranking. It does not decide eligibility.** Most models
apply a filter as well -- an absolute threshold, a trend test, a
percentile screen -- so a name can rank first and still be excluded.
Quantitative Momentum screens on raw momentum before ranking on the blended
score, so the second-highest blended score is routinely left out. That is
the model working, and the status column is there so it reads as a decision
rather than a contradiction.

Seventeen of the nineteen models expose a score. Buy & Hold and Fixed
Weights do not, because they rank nothing, and the tab says so instead of
inventing one.

---

## 5. Engine assumptions

The simulation holds **share counts and a cash balance in currency**, not
weights, and steps through one session at a time. Nothing in the loop reads
a future row. Weights are an output, derived from units times price over
value, rather than the thing being simulated.

That distinction is what makes the rest of this section checkable. Every
fill in the trade log carries the units traded and the price they filled
at, so the execution assumption can be audited instead of trusted; and the
reported value reconciles to units times close plus cash to machine
precision on every session.



They are explicit because they determine how credible the result is.

1. **No look-ahead.** The strategy produces target weights at the close of
   day *t*; the engine executes them at *t + lag*, one business day by
   default. The no-look-ahead property is verifiable: changing the last
   price in the history does not change any earlier return.
2. **The trading day is a choice.** Under Execution, a monthly rebalance
   can land on the last trading day, the first, the 15th, the third Friday
   or the last Monday. Quarterly can run Mar/Jun/Sep/Dec, Jan/Apr/Jul/Oct
   or Feb/May/Aug/Nov; annual can run in any month, so "every July" is one
   selection. Defaults reproduce the period-end calendar exactly, and
   stored configurations keep their dates to the day.

   This exists to be varied. The Robustness tab runs the same strategy
   across every plausible trading day and reports the spread. A monthly
   system tested only on month-end has been tested on one day out of
   twenty; if moving to the 15th materially changes the answer, the answer
   partly belonged to the calendar.

3. **Rebalance dates are signal dates.** A monthly or quarterly calendar
   marks the day the signal is *derived*, at that session's close. The trade
   lands `execution_lag` sessions later: with the default of one, a
   quarter-end signal trades on the first session of the next quarter. The
   trade log shows execution dates, so a quarterly strategy shows trades on
   the 1st, not the 31st.
4. **Execution price.** By default trades settle at the close. Switching to
   "Open (marked at the close)" splits the day in two: the overnight move
   from the prior close to the open is earned on the old weights, the
   intraday move from open to close on the new ones. That is the more
   realistic assumption for an order placed after a prior-close signal, and
   it stops the trade day from silently capturing an overnight gap the
   portfolio was never positioned for. Opening prices come from Yahoo
   Finance; uploaded files fall back to close execution.
5. **Warm-up.** Indicators are blind until they have enough history: a
   200-day average produces nothing for its first 200 sessions. Those
   sessions are not neutral - the portfolio sits in cash *earning the cash
   rate*, which lifts the reported return, stretches the measured period and
   dilutes volatility and drawdown. With "Trim the warm-up period" on, the
   record starts on the first day capital is actually at risk, and the
   benchmark is cut to the same date so the comparison stays honest. Only
   the leading stretch is removed: a deliberate move to cash mid-period is a
   decision and is kept.
6. **Dividends.** Two conventions, chosen under Price convention. *Total
   return* uses dividend-adjusted prices, so payments are folded into the
   price series and compound inside the position from the moment they are
   paid. *Price return + cash dividends* keeps prices ex-dividend and
   credits each payment as cash on its ex-date, where it sits uninvested
   until the next rebalance. Same cash in, different timing - and for a
   strategy that is often partly in cash or rebalances rarely, the gap is
   real. The two are mutually exclusive by construction: crediting dividends
   on top of adjusted prices would count every payment twice, and the app
   refuses that combination rather than silently producing it.
7. **Weights drift between rebalances.** Positions evolve with prices.
   Assuming an implicit daily rebalance is the mistake that most often
   inflates published results.
8. **Frictions on actual turnover, per instrument.** Commission is a flat
   rate, never scaled by size or by anything below -- it is a broker fee,
   not a liquidity cost. Slippage has a flat component (default 25 bps)
   plus two optional refinements:

   - **Per-instrument multipliers.** A manual override list scales the flat
     rate for specific tickers -- for a name known to be thinner than its
     peers, or one with no volume data to drive the model below.
   - **Volume-scaled market impact.** Turned on under Frictions, this adds
     a cost that grows with the *square root* of participation (trade size
     over the instrument's trailing 20-session average volume), the
     standard institutional approximation: a $10K order and a $10M order in
     the same name no longer cost the same rate, and the same order costs
     more in a thinner ETF than in a deep one. Calibrated by one number --
     the extra cost, in bps, for an order equal to 10% of average volume.
     Doubling participation multiplies the impact term by roughly `sqrt(2)`,
     not by 2. Volume is fetched automatically once this is turned on, and
     an instrument with no volume data simply falls back to the flat rate.

   The trailing volume average is shifted by one session, so a trade is
   always priced against liquidity known *before* it happened, never
   against that day's own not-yet-complete print.

9. **Management fees pay the same frictions as everything else.** An
   annual rate accrues daily on the marked value and is deducted at each
   month-end, which is how a fee is actually billed. A fully invested book
   has no idle cash to pay from, so the deduction sells holdings pro rata,
   exactly as a fund liquidates units to meet its own fee -- and, by
   default, that sale pays the same commission and slippage any other
   trade would, tagged **"Fee liquidation"** in the trade log rather than
   "Rebalance" so it reads as an explanation, not a surprise. A checkbox
   under Frictions turns this off to reproduce the simpler, cost-free
   liquidation.

   Realized fee drag runs a little above the headline rate even without
   trading costs, because the fee compounds against a growing balance:
   1.00% per year costs about 1.07% of CAGR over a decade at 8%.

10. **Whole units are optional.** Off by default, since most brokers now
   support fractional shares. Turned on, every order rounds down to a whole
   unit, which leaves idle cash and stops the book sitting exactly on its
   targets -- realistic for a small account, and visible in the cash weight.

11. **Cash is remunerated.** Either at a fixed rate, or by the return of a
   cash-equivalent ETF (PSA.TO, BIL): the opportunity cost of sitting out of
   the market is counted.
12. **Adjustments under 0.5% of weight are ignored** (`min_trade_weight`),
   so the engine does not charge for trades no manager would place. **Full
   exits are exempt.** Closing a position is not a micro adjustment: if the
   threshold could block it, a residual the model no longer wants would
   survive, and under rotation those residuals accumulate until the book
   holds more names than the strategy ever selected -- a "top 3" quietly
   drifting to eight positions.

   For the same reason a rebalance **sells before it buys**, using the
   proceeds to fund the purchases. Scaling a whole order down when cash is
   short would shrink the sales too, leaving positions half-closed.

 (`min_trade_weight`), so
   the engine does not charge for trades no manager would place.
13. **Survivorship bias is not handled automatically.** A universe built
   today from ETFs that exist today carries that bias. The "Data" diagnostic
   flags histories shorter than the tested period.

---

## 4 quinquies. Asset-class budgets and sleeves

Under **Portfolio construction** in the sidebar. The model still picks the
holdings; a sleeve decides how much of the portfolio it gets to pick for.

### By asset class

Assign each instrument a class, give each class a budget, and the model runs
*inside* each class separately. Sixty in equities, thirty in fixed income,
ten in real assets means the model picks the best bonds among bonds rather
than discovering that equities out-ranked every bond and putting everything
there.

Preset universes arrive already tagged. A hand-typed universe comes back as
"Unclassified" for every symbol, which is deliberate: guessing a class from
a ticker would silently misallocate a budget, and being asked is better than
being wrong.

### Blend of strategies

Split the book between two to four models, each with its own budget and its
own parameters. A sleeve holds at most its share, and whatever its model
leaves in cash stays inside that sleeve rather than being handed to the
others -- so a defensive signal in one model cannot become extra risk in
another.

Blending is not free diversification. Two momentum models on the same
universe will hold many of the same names at the same time, and the blend
will look a lot like either one. The gain comes from models that disagree:
a breakout system and a mean-reversion system, or models on different
horizons. The correlation matrix in the Data tab and the fold table in
Robustness are the places to check whether the blend is doing anything the
parts were not.

### Core + strategy

A fixed sleeve held permanently, and the model running on the rest. Set the
core share and list its holdings as `XIC.TO:60, XBB.TO:40`. Percentages or
fractions both work, since only the ratios matter -- the sleeve is scaled to
its share either way.

**Keep core holdings out of the strategy universe** decides whether the two
sleeves can overlap. On, the model never picks a name the core already
holds, so a core of `XIC.TO:100` at fifty percent means XIC.TO sits at
exactly fifty percent, always. Off, the model may add to it and the combined
position can exceed the core share -- a 50% core plus a model that also
likes XIC.TO can reach 67%. On is the default, because "half in XIC.TO, half
run by the model" almost always means the first thing.

If the core covers the entire universe, the model has nothing left to pick
from and its share stays in cash, with a warning.

### Two rules that decide how this behaves

**The record starts when the model does.** A permanently held core is
invested from session one, so total exposure never reveals the model's
warm-up: a 200-day average leaves the book sitting at fifty percent core and
fifty percent cash for its first two hundred sessions, and measuring that
stretch reports a period the strategy had no hand in. With warm-up trimming
on, the record starts the day the model first takes a position, and the
benchmark is cut to the same date. Only the leading stretch goes: once the
model is live, a move to cash is a decision and is kept.

**Cash inside a sleeve stays in that sleeve.** If the bond model rejects
every bond, that thirty percent sits in cash. It is not handed to equities.
Spilling it over would convert a defensive signal into extra equity risk at
exactly the wrong moment, and would mean the bond budget was never a budget.

**Budgets are ceilings, not floors.** A sleeve holds at most its share. A
model that goes half to cash leaves the portfolio under-invested rather than
levering the rest -- which is what "no more than sixty percent in equities"
has to mean to mean anything.

Budgets summing above the leverage ceiling are scaled back proportionally
and flagged. Budgets summing below one hundred leave the remainder in cash,
also flagged. The Positions tab reports each sleeve's budget, what it
actually held on average, and how much of its own budget sat in cash.

Nothing downstream changes: sleeves resolve to an ordinary weight frame
before the engine sees it, so drift, execution lag, frictions, dividends and
warm-up behave exactly as they do for a single strategy. A single sleeve at
one hundred percent reproduces the plain strategy to the last decimal.

---

## 5 bis. Building a strategy in the app

The **Builder** tab writes a strategy as an expression, with no Python and
no redeploy. Test it against the loaded universe, then select **Custom
Formula** as the model and paste the expressions in.

A strategy is two expressions plus sizing:

- **Score** - higher is better; the universe is ranked by it each day and
  the top N are held. `pctrank(mom(price, 126))`
- **Filter** (optional) - must be true for a name to be eligible.
  `price > sma(price, 200)`

Available: `price`; `sma ema mom` for trend; `vol dvol` for risk;
`rsi er` for oscillators and quality; `zscore mean std mmax mmin shift` for
statistics; `pctrank rank` across the universe; `ifelse clip abs log sign`
for shaping. Imported exogenous series appear under their own names. `x` is
any series, `n` a window in sessions.

Some working examples:

```
pctrank(mom(price, 126))                                  momentum rank
price > sma(price, 200)                                   trend filter
0.7 * pctrank(mom(price,126)) + 0.3 * pctrank(-vol(price,60))
ifelse(rsi(price,14) < 35 and price > sma(price,200), 1, 0)
price > mmax(shift(price, 1), 60)                         breakout
```

The Builder reports whether the expression is boolean or continuous, how
much of the price grid it covers, and the first date it produces a value -
a long window costs history, and the backtest cannot start before it.

**On safety.** Expressions are parsed to a syntax tree and walked node by
node against a whitelist; they are never handed to `eval` in any meaningful
sense. There is no attribute access, no subscripting, no imports, no
lambdas, no comprehensions, and no name that is not a declared indicator.
This matters because the app is reachable at a public URL: running
arbitrary user code there would let anyone past the login page read the
secrets file or open network connections from inside the container. A
malformed or hostile expression produces no position at all rather than a
silent partial one.

For anything the expression language cannot say - a stateful rule, an
optimizer, a custom data join - write a real strategy in Python instead,
as below.

---

## 5 ter. Published strategies included

Alongside the generic models, four systems from the literature. Each
docstring in `qbt/strategies/library.py` records where the implementation
departs from the source, because a strategy that quietly differs from the
paper it names is worse than one that admits it.

| Model | Idea |
|---|---|
| Quantitative Momentum (Gray & Vogel) | Rank on 12-2 momentum, keep the leaders, then prefer those whose gain arrived as a smooth drift rather than a few jumps. Smooth momentum has been found to persist where jumpy momentum reverses. |
| Turtle Breakout (Donchian) | Buy a break to a new N-day high, exit on a break to an M-day low, size each position by its recent range so every holding carries similar risk. |
| Time-Series Momentum (Moskowitz, Ooi & Pedersen) | Judge each instrument against itself rather than against the others, and scale positions to a common volatility. |
| Accelerating Dual Momentum | Blend short, medium and long lookbacks instead of trusting one, with an absolute threshold that moves to cash when nothing clears it. |
| MomentumOG | Cumulative return over the lookback, less how far the price sits above its own average over that window. A name that has run up but is stretched far above its mean scores below one that got there steadily. |
| Trend-Gated Target Weights (HIDE-style) | Fixed target weights per asset class, each independently gated by two trend signals worth half its allocation each: full weight, half weight, or cash. Never short. |

### On MomentumOG

The score is `mom(price, n-1) - (price / sma(price, n) - 1)`: cumulative
return over the window, minus the price's percentage distance above its own
mean over that same window. The off-by-one is faithful to the source rather
than incidental -- a rolling buffer of n prices yields n-1 returns for the
momentum leg while the average uses all n.

Both terms contain the latest price, so the score reduces to
`last x (1/first - 1/mean)`. That is what gives it its character: it rewards
a low starting point and penalises a high mean, so a name that has already
run far ahead of its own average ranks below one that arrived steadily. It
is momentum with the froth taken out.

The implementation was checked against a literal transcription of the
original loop at four lookbacks; the largest difference was 7e-15.

### On the HIDE-style model

Alpha Architect's HIDE holds 50% intermediate Treasuries, 25% REITs and 25%
commodities, rebalances monthly on trend signals, and goes to cash rather
than short. Its published material states the allocation and the three
exposure states -- full risk, half risk, risk-off -- but **not the signal
rules themselves**.

Two binary signals worth half the weight each is what produces exactly those
three states, and time-series momentum paired with a long moving average is
the combination Alpha Architect uses in its published trend research. That
is what this model implements, with both windows exposed as parameters.

**The momentum leg is an excess-return test.** The asset must beat cash over
the same window, not merely rise. This is not a refinement. Through 2010-21
cash paid almost nothing and the two tests are indistinguishable; through
2022-24 cash paid four to five percent, and an asset up three percent passes
the absolute test while failing the excess one. A threshold typed in once is
a guess at a number that moves through a rate cycle, which is exactly when a
defensive strategy is being asked to earn its keep.

The hurdle follows **whatever Cash remuneration is set to in the Data
panel** -- a proxy ETF such as BIL (the HIDE preset's default) or, when no
proxy is chosen, the fixed cash rate under Frictions. The same series the
portfolio earns on its idle cash is the series each asset has to beat, which
is the point: the hurdle and the opportunity cost are one number, not two
that can drift apart.

The sidebar states which it resolved to, and warns when no proxy is set and
the cash rate is zero -- in that case an excess-return test is arithmetically
identical to comparing against zero, and the label would otherwise promise
something it is not delivering.

Two overrides remain: "Zero (absolute return)" for the plain test, and
"Fixed rate set here" to pin a number independent of the Data panel. Worth
running the default against zero once, just to see how much the choice moves
the result over a period containing 2022.

So: a faithful reconstruction of a documented *structure*, not a replica of
the fund. It will not track HIDE's returns, and it is not meant to. The
"HIDE target asset classes" preset loads SCHR, VNQ and BCI to pair with it;
note BCI launched in 2017, so an earlier start date drops it from the
universe.

Two honest caveats. The Turtle rule sizes on average true range, which needs
daily highs and lows; this app carries closes only, so the range is
approximated from close-to-close moves. That understates the range on wide
intraday days and leaves positions slightly larger than the published rule
would give -- the behaviour is unaffected, the leverage is a touch higher.
And Quantitative Momentum in the book is a stock-selection system run over
hundreds of names; applied to a handful of ETFs it is the same logic on a
universe far too small to sort into deciles.

---

## 6. Adding a strategy

A strategy is a pure function `(prices, params) -> target weights`. Open
`qbt/strategies/library.py` and add:

```python
@register(
    key="my_strategy",
    label="My Model",
    description="What the model does, in one or two sentences.",
    params=[
        Param("window", "Window", "int", 60, 10, 250, 5),
        Param("threshold", "Threshold (%)", "float", 2.0, 0.0, 20.0, 0.5),
    ],
)
def _my_strategy(px, p):
    signal = px > sma(px, p["window"]) * (1 + p["threshold"] / 100)
    return size_equal(signal.astype(float))
```

To consume exogenous series, add a third argument `ex` to the function: the
engine injects the imported series, already shifted and aligned. The
two-argument signature stays valid for price-only models.

Nothing else needs to change: the interface detects the strategy and builds
its controls from the `Param` list. Rules to respect:

- the value at *t* depends only on information available at *t*;
- a row of weights never sums above 1 (the balance goes to cash);
- return a DataFrame aligned to `px`.

Indicators already available in `qbt/strategies/base.py`: `sma`, `ema`,
`rsi`, `total_return`, `realized_vol`, `downside_vol`, `efficiency_ratio`,
`zscore`, `size_equal`, `size_inverse_vol`, `apply_vol_target`.

---

## 7. Usage without the interface

```python
from qbt import RunConfig, run_from_config
from qbt import metrics as M

cfg = RunConfig.from_yaml(open("configs/dual_momentum_canada.yaml").read())
res, bench, prices = run_from_config(cfg)

print(M.summary(res.returns, res.equity, bench.returns))
res.weights.tail()          # holdings held
res.trades                  # trade log
```

Every backtest run in the interface exports its YAML configuration: that is
the link between exploring on screen and reproducing in a script.

---

## 8. Built-in robustness tests

| Test | Question asked |
|---|---|
| Successive folds | Does behavior hold within each sub-period? |
| In-sample / out-of-sample | Does the second half resemble the first? |
| Parameter surface | Is the result a plateau or a lone spike? |
| Cost sensitivity | At what cost level does the strategy stop paying off? |
| Trading-day sweep | Does the result survive rebalancing on a different day of the period? |
| Block-resampled Monte Carlo | How much of the result depends on the order of returns? |
| Expected Sharpe by chance | What Sharpe would *n* trials produce with no real edge? |

That last point deserves attention: after 200 combinations tested, a Sharpe
of 0.4 is achievable on pure noise. The displayed gap is the model's net
edge.

---

## 9. Layout

```
app.py                     Streamlit interface
qbt/
  config.py                dataclasses and YAML
  data.py                  loading, cleaning, diagnostics
  exog.py                  exogenous series, publication lag
  external.py              imported target weights
  allocation.py            sleeves: asset-class budgets and composition
  monitor.py               market monitor analytics
  tvchart.py               TradingView Lightweight Charts integration
  excel_export.py          complete multi-sheet workbook export
  formula.py               sandboxed expression evaluator
  presets.py               preset universes
  returns_input.py         imported return streams
  engine.py                day-by-day simulation
  metrics.py                performance and risk
  charts.py                Plotly charts (dark in-app theme + light print theme)
  report.py                tearsheet HTML report builder
  robustness.py            robustness tests
  strategies/
    base.py                registry, indicators, sizing
    library.py              built-in strategies
configs/                   example configurations
deploy/                    publish scripts, Hugging Face notes
Dockerfile                 container hosts
render.yaml                Render
example_prices.csv                    price template
example_economic_data.csv             macro-series template
example_fundamental_factor.csv        cross-sectional factor template
example_target_weights.csv            target-weights template
.streamlit/config.toml     theme
```
