# Design system

For the redesigned app in `web/`. Calm, dark-first, precise. Typography and
alignment carry the design; there is one accent; green and red mean gain and
loss and nothing else; every figure is tabular; there is one chart language;
nothing is decorative.

The reference implementation is the Backtest Results screen
(`web/src/screens/Results.tsx`). When this document and that screen
disagree, fix whichever is wrong and keep them in step.

## 1. Principles

1. **Figures first.** A screen leads with numbers and charts. Explanation
   sits beside them in a quieter voice and folds when it is long.
2. **Structure is alignment and hairlines.** No cards, no shadows, no
   gradients, no filled panels behind content. A group is a heading, a
   hairline, and space.
3. **One accent, spent on meaning.** Brass marks what you can act on, what
   is selected, where focus is, and the subject series of a chart.
4. **Green and red are data.** Only a gain or a loss is green or red. A
   warning is not red; a primary button is not green.
5. **The browser formats nothing.** Every figure arrives from the server as
   the string to display. The front end aligns and colours.
6. **Interactions that only change the view stay in the browser** and take
   under 100 ms: log scale, range, series visibility, tabs, folding.

## 2. Colour

Defined once, in `web/src/styles/tokens.css`. No other file contains a
colour value.

| Token | Value | Role |
|---|---|---|
| `--bg` | `#0d1217` | Page ground: ink, cooled toward blue |
| `--surface` | `#121920` | Control panel, table row hover, skeletons |
| `--raised` | `#18212a` | Selected chips, disabled buttons |
| `--line` | `#25313c` | Structural hairlines: panel edge, table header rule, input borders |
| `--line-soft` | `#1a242d` | Row rules, chart grid lines |
| `--text` | `#e7ecf1` | Primary text and figures (15.8:1 on `--bg`) |
| `--text-2` | `#a7b2bd` | Labels, notes, table headers (8.7:1) |
| `--text-3` | `#7c8996` | Tertiary: benchmark sub-values, axis ticks, help (5.3:1; 5.0:1 on `--surface`) |
| `--accent` | `#d8b46a` | Brass. Primary action, selection, focus ring, links, subject series |
| `--accent-strong` | `#e6c682` | Hover on accent |
| `--accent-wash` | 12% accent | Selected segment background |
| `--on-accent` | `#171207` | Text on an accent fill |
| `--gain` | `#45b88f` | Positive figures, positive bars, positive heat cells (7.6:1) |
| `--loss` | `#ea6f5f` | Negative figures, drawdown, negative bars and cells (6.2:1) |
| `--warn` | `#c9a45c` | Flags (data and settings warnings). Never `--loss` |
| `--series-2` … `--series-6` | slate, blue, violet, teal, clay | Chart context series. Never green or red |

All text pairs meet WCAG AA (4.5:1) at every size used. `--text-3` is the
floor: do not use anything dimmer for text.

Brass replaces the earlier National Bank red, because a red accent beside a
red loss colour makes a chart unreadable. Changing the accent is a change to
three tokens.

A light theme is not built. The tokens are roles, so a light set can be
added under `:root[data-theme="light"]` without touching components.

## 3. Typography

One family: **IBM Plex Sans**, weights 400, 500, 600, self-hosted
(`@fontsource/ibm-plex-sans`, Latin subset). No monospace face: figures
align because `font-variant-numeric: tabular-nums` is set on `body`.

| Token | Size | Use |
|---|---|---|
| `--fs-0` | 11 px | Chart axis ticks only |
| `--fs-1` | 12 px | Labels, table headers, notes, help, readout |
| `--fs-2` | 13 px | Body, table cells, controls (the base size) |
| `--fs-3` | 15 px | Section headings |
| `--fs-4` | 18 px | Run title |
| `--fs-5` | 24 px | KPI figures (18 px on phones) |

Rules:

- Sentence case everywhere. No all-caps labels, no letter-spaced eyebrows.
- Weight carries hierarchy before size does: 600 for headings and the run
  title, 500 for chart titles, tab labels and KPI figures, 400 otherwise.
- Figures are right-aligned in tables and never wrap.
- Running text (notes, descriptions) is capped at 78 characters a line.
- Labels name the thing in the user's words ("Execution lag", not
  "exec_lag"). Units go in a suffix beside the input, not in the label.

## 4. Space and shape

- A 4 px grid: `--s-1` 4, `--s-2` 8, `--s-3` 12, `--s-4` 16, `--s-5` 24,
  `--s-6` 32, `--s-7` 48.
- Between sections of a screen: `--s-5`. Between a heading and its content:
  `--s-2`. Inside the control panel: `--s-3`.
- Row and control height: 28 px (`--row`); 36 px controls on phones.
- Corner radius: 3 px on controls and focus rings. Content has no radius
  because it has no boxes.
- Borders are 1 px. The only 2 px rules are the active tab and workspace.

## 5. Layout

```
+---------------------------------------------------------------+
| Quant Backtest Studio   Backtest  Markets  Research           |  top bar, 44 px
+----------------+----------------------------------------------+
| Data           | Run title   period  instruments  vs bench    |  run line
|  ...           | CAGR | Vol | Sharpe | Max DD | Calmar | Sort |  KPI strip
| Signal         | Results  Compare  Signals  Positions ...     |  tabs (sticky)
|  ...           |                                              |
| Execution    > | screen content, one scroll                   |
| Frictions    > |                                              |
| [Run backtest] |                                              |
+----------------+----------------------------------------------+
   328 px            fluid, padded 24 px
```

- **Top bar**: product name, the three workspaces as text tabs.
- **Control panel** (left, 328 px, its own scroll): collapsible groups in a
  two-column grid of fields; a collapsed group shows a one-line summary of
  its settings. The Run button is fixed at the bottom of the panel, with the
  run error and the "Settings changed since this run" line above it.
  Ctrl+Enter runs from anywhere in the panel.
- **Run header**: the run's label as the page title, its context as a quiet
  line, the time the run took (or "from cache"), then the KPI strip.
- **KPI strip**: one ruled row of six cells, hairline between cells, no
  boxes. Label, figure, benchmark beneath. The figure takes the sign colour
  only where the server says the metric has a sign.
- **Tabs**: sticky under the run header. A tab not yet rebuilt is dimmer and
  opens a pointer to the classic app. A visited tab stays mounted, so
  returning to it is instant.
- **Screen**: a single column of sections with `--s-5` between them; a
  screen may place sections side by side (`.pair`, `.trio`) where a single
  column would waste width.

## 6. Tables

- Semantic `<table>`; the first column is a row header. Header row in
  `--text-2`, 12 px, weight 500, with a `--line` rule beneath. Rows are
  28 px with a `--line-soft` rule. No zebra striping, no vertical rules.
- The server marks each column `left` or `right`; figures are right.
  The first and last columns have no outer padding, so a table's edges align
  with the headings and charts around it.
- Sign colour applies only to columns the server marks `signed`; emphasised
  columns (for example Excess) are weight 600.
- A table wider than its space scrolls horizontally inside itself with the
  first column pinned. The page never scrolls sideways.
- A table of up to about 40 rows is shown whole. Longer tables (trade log,
  ledgers) get their own vertical scroll with a pinned header; build that
  with TanStack Table and Virtual when the first such screen is rebuilt.
- An empty table is one quiet sentence, never an empty grid.

## 7. Charts

One library (Apache ECharts, loaded on demand), one wrapper
(`web/src/charts/core.tsx`). A chart component never names a colour.

- **Series colour**: the subject is brass and 1.8 px; the benchmark is
  slate; further series follow `--series-3` to `--series-6` at 1.2 px.
  Green and red are used only where the mark itself is a gain or a loss:
  signed bars, the subject's drawdown, heat cells.
- **Axes**: values on the right, where the latest figure is. Tick text
  11 px in `--text-3`. Horizontal grid lines only, in `--line-soft`. No axis
  titles: the chart title says what is plotted and in what unit.
- **Time axis**: trading days as categories (no weekend gaps), labelled by
  year on the first session of each year.
- **Title**: 13 px, weight 500, above the plot, stating what is plotted and
  the anchoring where relevant. Controls sit on the same line, right.
- **Time panel** (the signature element): value and drawdown share one time
  axis and one crosshair. A readout line above the plot shows the date under
  the cursor and each series' value and drawdown in fixed columns; with no
  cursor it shows the last date. It replaces a floating tooltip. The readout
  entries are also the legend: pressing one hides or shows that series.
- **Range** (1Y, 3Y, 5Y, 10Y, Max) and **Log scale** are segmented controls
  and a checkbox in the chart header. Log scale draws log10 of the value on
  a linear axis so grid lines are evenly spaced and labelled with the value.
- **Other charts** use a small bordered tooltip (surface background, no
  shadow) with the label and the figure.
- **Month-by-year returns** are an HTML table, not a canvas: text is
  selectable and read by screen readers. Cell shade follows sign and size,
  capped at the 95th percentile of absolute values.
- **Heights**: time panel 440 px (340 on phones); small charts 220 px (200).
- No animation on draw. No gradients, no markers on lines, no 3D except the
  parameter surface, which is the one view that needs it.

## 8. Controls

- Native `input`, `select`, `textarea`, `button`. Every control has a
  visible label above it; help is a "?" that reveals a sentence beneath.
- Number boxes commit on blur or Enter, clamp to their range, and step with
  the arrow keys.
- A choice of two or three is a segmented control; more is a select; a set
  of symbols is a row of toggle chips.
- File inputs accept a drop or a click, show the file name with "Remove",
  and report what was read in a note beneath.
- Buttons say what they do: "Run backtest", "Remove", "Try again".
- Primary button: accent fill, one per view. Secondary: 1 px outline.

## 9. States

| State | Treatment |
|---|---|
| Nothing run yet | The start screen: one sentence saying what to do, the three routes, and the model list |
| Running | A 2 px accent bar across the top; the Run button reads "Running… 1.2 s"; the previous result stays on screen |
| First run, nothing to show yet | Plain blocks in `--surface` where the strip, chart and rows will be, with a status line |
| Loading a tab | The same blocks with "Loading this view…" |
| Run failed | The engine's message as a flag above the Run button; the previous result stays |
| Settings changed since the run | "Settings changed since this run." above the Run button; the KPI strip dims slightly |
| Run no longer held by the server | The message and a "Run it again" button |
| Data or settings warning | A flag: `--warn` left rule and wash, beside the thing it concerns |
| Not rebuilt yet | A short statement and "Open the classic app" |
| Empty table or chart | One sentence in `--text-3` |

Messages say what happened and what to do next. They do not apologise.

## 10. Phones and narrow windows (under 900 px)

- The control panel becomes a full-height sheet under the top bar, opened
  with **Settings** and closed with **Close**, Escape, or by running. Its Run
  button stays fixed at the bottom, above the home indicator.
- The product name is dropped from the top bar; the workspaces remain.
- The KPI strip becomes three columns by two rows; figures are 18 px.
- Every side-by-side arrangement becomes a single column.
- Tabs scroll horizontally; the tab strip stays sticky.
- Inputs are 36 px high with 16 px text (smaller text makes iOS zoom the
  page on focus).
- Charts do not capture scrolling: zoom is the range control, not a pinch.
- Wide tables scroll inside themselves with the first column pinned.
- Nothing depends on hover. Help opens on tap; the chart readout follows a
  touch and returns to the last date when it ends.

## 11. Accessibility

- Keyboard: everything reachable in reading order; a "Skip to results" link;
  tabs move with the arrow keys; Escape closes the settings sheet;
  Ctrl+Enter runs.
- Focus: a 2 px accent ring with a 2 px gap on every focusable element
  (`--focus`). Never removed.
- Contrast: AA for all text (section 2). Sign is never carried by colour
  alone: figures keep their plus or minus.
- Charts have a text label naming the series and the period; the figures
  they plot are also in a table on the same screen.
- Motion: only the progress bar moves. `prefers-reduced-motion` stops it.

## 12. What not to add

Cards with shadows, coloured section backgrounds, icons used as decoration,
all-caps labels, a monospace face for figures, a second accent, animated
chart entrances, hover lifts, a tooltip where the readout would do.
