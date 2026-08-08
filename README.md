# Quant Backtest Studio

Web app for backtesting systematic strategies. Yahoo Finance or CSV/Excel
file data, a verifiable simulation engine, a full robustness suite,
exogenous-data-driven signals, imported target weights, and one-click exports
(tearsheet, holdings, workbook).

The `qbt/` package is usable standalone, with no interface. The Streamlit
interface (`app.py`) is just a front end: all logic lives in pure functions.

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
| Full workbook | XLSX | Statistics, Series, Holdings, Current Holdings, Monthly Returns, Trades in one file |
| Configuration | YAML | Exact reproduction of the run (imported files are not included; their name, settings, and lag are) |

The tearsheet is the fastest way to share a result: it is a single
self-contained HTML file (Plotly loaded from a CDN), styled for print. Open
it in a browser and use Print -> Save as PDF for a clean PDF, or share the
HTML file directly.

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

---

## 5. Engine assumptions

They are explicit because they determine how credible the result is.

1. **No look-ahead.** The strategy produces target weights at the close of
   day *t*; the engine executes them at *t + lag*, one business day by
   default. The no-look-ahead property is verifiable: changing the last
   price in the history does not change any earlier return.
2. **Execution price.** By default trades settle at the close. Switching to
   "Open (marked at the close)" splits the day in two: the overnight move
   from the prior close to the open is earned on the old weights, the
   intraday move from open to close on the new ones. That is the more
   realistic assumption for an order placed after a prior-close signal, and
   it stops the trade day from silently capturing an overnight gap the
   portfolio was never positioned for. Opening prices come from Yahoo
   Finance; uploaded files fall back to close execution.
3. **Warm-up.** Indicators are blind until they have enough history: a
   200-day average produces nothing for its first 200 sessions. Those
   sessions are not neutral - the portfolio sits in cash *earning the cash
   rate*, which lifts the reported return, stretches the measured period and
   dilutes volatility and drawdown. With "Trim the warm-up period" on, the
   record starts on the first day capital is actually at risk, and the
   benchmark is cut to the same date so the comparison stays honest. Only
   the leading stretch is removed: a deliberate move to cash mid-period is a
   decision and is kept.
4. **Dividends.** Two conventions, chosen under Price convention. *Total
   return* uses dividend-adjusted prices, so payments are folded into the
   price series and compound inside the position from the moment they are
   paid. *Price return + cash dividends* keeps prices ex-dividend and
   credits each payment as cash on its ex-date, where it sits uninvested
   until the next rebalance. Same cash in, different timing - and for a
   strategy that is often partly in cash or rebalances rarely, the gap is
   real. The two are mutually exclusive by construction: crediting dividends
   on top of adjusted prices would count every payment twice, and the app
   refuses that combination rather than silently producing it.
5. **Weights drift between rebalances.** Positions evolve with prices.
   Assuming an implicit daily rebalance is the mistake that most often
   inflates published results.
6. **Frictions on actual turnover.** Cost = sum(|target weight - current
   weight|) x (commission + slippage). Default: 5 bps + 25 bps, a
   conservative blended assumption for Canadian ETFs.
7. **Cash is remunerated.** Either at a fixed rate, or by the return of a
   cash-equivalent ETF (PSA.TO, BIL): the opportunity cost of sitting out of
   the market is counted.
8. **Adjustments under 0.5% of weight are ignored** (`min_trade_weight`), so
   the engine does not charge for trades no manager would place.
9. **Survivorship bias is not handled automatically.** A universe built
   today from ETFs that exist today carries that bias. The "Data" diagnostic
   flags histories shorter than the tested period.

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
