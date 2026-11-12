// The Backtest control panel: the controls a run is described with.
//
// Rebuilt so far: data source and files, universe, benchmark, signal
// origin, model and parameters, execution and frictions. Controls not yet
// rebuilt keep their defaults (the list is in docs/HANDOFF.md); the classic
// app still has all of them.
import { useMemo } from "react";
import type { ParamMeta, Spec } from "../api";
import { useStudio } from "../state";
import { Check, FileDrop, Group, NumberBox, Segmented, Select, TextArea, TextBox } from "./fields";

const FILES = ".csv,.xlsx,.xls,.xlsm,.txt";
const pct = (v: number) => Math.round(v * 1e6) / 1e4;

function Param({ p, value, series, onChange }: {
  p: ParamMeta; value: unknown; series: string[]; onChange: (v: unknown) => void;
}) {
  if (p.kind === "int" || p.kind === "float") {
    return <NumberBox label={p.label} help={p.help} value={Number(value ?? p.default)}
                      min={p.min} max={p.max} step={p.step} onChange={onChange} />;
  }
  if (p.kind === "bool") return <Check label={p.label} help={p.help} checked={Boolean(value ?? p.default)} onChange={onChange} />;
  if (p.kind === "series") {
    const opts = series.length ? series : ["No series imported"];
    return <Select label={p.label} help={p.help} wide value={String(value || opts[0])}
                   options={opts.map((o) => ({ value: o, label: o }))}
                   onChange={(v) => onChange(v === "No series imported" ? "" : v)} />;
  }
  if (p.kind === "choice" && p.choices.length) {
    return <Select label={p.label} help={p.help} wide value={String(value ?? p.default)}
                   options={p.choices.map((o) => ({ value: o, label: o }))} onChange={onChange} />;
  }
  if (p.kind === "formula") return <TextArea label={p.label} help={p.help} rows={3} value={String(value ?? p.default ?? "")} onChange={onChange} />;
  return <TextBox label={p.label} help={p.help} wide value={String(value ?? p.default ?? "")} onChange={onChange} />;
}

export function ControlPanel({ onRan }: { onRan?: () => void }) {
  const s = useStudio();
  const { spec, set, meta, uploads, symbols } = s;
  const model = meta.models.find((m) => m.key === spec.strategy) ?? meta.models[0];
  const universe = spec.universe ?? symbols;
  const exogCols = uploads.exog?.columns ?? [];

  const benchOptions = useMemo(() => [
    { value: "", label: "None" },
    ...symbols.map((t) => ({ value: t, label: t })),
  ], [symbols]);

  const applyPreset = (name: string) => {
    const p = meta.presets.find((x) => x.name === name);
    if (!p) { set({ preset: "Custom" }); return; }
    set({ preset: name, tickers: p.tickers, universe: null, benchmark: p.benchmark, cash_proxy: p.cash });
  };

  const toggleSymbol = (t: string) => {
    const next = universe.includes(t) ? universe.filter((x) => x !== t) : symbols.filter((x) => x === t || universe.includes(x));
    set({ universe: next.length === symbols.length ? null : next });
  };

  const canRun = spec.source === "yfinance" ? symbols.length > 0 : !!uploads.prices?.instruments.length;
  const run = () => { s.start(); onRan?.(); };

  return (
    <form className="controls" aria-label="Backtest settings"
          onSubmit={(e) => { e.preventDefault(); if (canRun) run(); }}
          onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && canRun) { e.preventDefault(); run(); } }}>
      <div className="controls-scroll">
        <Group title="Data">
          <Segmented label="Price source" value={spec.source}
                     options={[{ value: "yfinance", label: "Yahoo Finance" }, { value: "upload", label: "File" }]}
                     onChange={(v) => set({ source: v, universe: null })} />
          {spec.source === "yfinance" ? (
            <>
              <Select label="Preset universe" wide value={spec.preset}
                      help="Loads a ready-made set of symbols. Everything stays editable afterwards, and the benchmark and cash proxy are filled in to match."
                      options={[{ value: "Custom", label: "Custom" }, ...meta.presets.map((p) => ({ value: p.name, label: p.name }))]}
                      onChange={applyPreset} />
              <TextArea label="Symbols" rows={4} value={spec.tickers.join("\n")}
                        help="One per line or comma-separated. Add .TO for Toronto, .V for TSX-V, no suffix for U.S. tickers."
                        onChange={(v) => set({
                          preset: "Custom", universe: null,
                          tickers: v.replace(/[,;]/g, "\n").split("\n").map((t) => t.toUpperCase()),
                        })} />
              <TextBox label="Start" type="date" value={spec.start ?? ""} onChange={(v) => set({ start: v || null })} />
              <TextBox label="End" type="date" value={spec.end ?? ""} onChange={(v) => set({ end: v || null })} />
            </>
          ) : (
            <>
              <FileDrop label="Price files" accept={FILES} multiple names={uploads.prices?.names ?? []}
                        busy={s.uploading === "prices"} error={uploads.prices?.error}
                        hint="A date column, then prices. Long format (Date, Ticker, Open, High, Low, Close, Volume, Dividends) is recommended; wide files with one column per instrument work too."
                        onFiles={(f) => s.upload("prices", f)} onClear={() => s.clearUpload("prices")} />
              {uploads.prices && !uploads.prices.error && (
                <p className="note wide">
                  {uploads.prices.instruments.length} instrument(s) from {uploads.prices.sources} source(s): {uploads.prices.fields.join(", ")}.
                </p>
              )}
              {uploads.prices?.warnings.map((w) => <p key={w} className="flag wide">{w}</p>)}
              <Select label="Prices in the file are" wide value={spec.file_convention}
                      help="Total return: the close already includes dividends. Price return: the close excludes them and each dividend is credited as cash on its ex-date. Detect: price return when the file has dividends, total return otherwise."
                      options={Object.entries(meta.file_conventions).map(([value, label]) => ({ value, label }))}
                      onChange={(v) => set({ file_convention: v })} />
            </>
          )}
          {symbols.filter(Boolean).length > 0 && (
            <div className="field wide">
              <div className="field-label"><span id="univ-label">Investable universe</span>
                <span className="count num">{universe.length} of {symbols.filter(Boolean).length}</span></div>
              <div className="chips" role="group" aria-labelledby="univ-label">
                {symbols.filter(Boolean).map((t) => (
                  <button key={t} type="button" className="chip" aria-pressed={universe.includes(t)} onClick={() => toggleSymbol(t)}>{t}</button>
                ))}
              </div>
            </div>
          )}
          <Select label="Comparison benchmark" wide value={spec.benchmark ?? ""} options={benchOptions}
                  onChange={(v) => set({ benchmark: v || null })} />
          {spec.benchmark && (
            <Select label="Benchmark convention" wide value={spec.bench_mode}
                    help="Set independently of the strategy. A benchmark measured on price return only understates the bar by roughly its dividend yield each year."
                    options={meta.bench_modes.map((m) => ({ value: m, label: m }))} onChange={(v) => set({ bench_mode: v })} />
          )}
          {spec.source === "yfinance" && (
            <Select label="Price convention" wide value={spec.adjusted ? "total" : "price"}
                    options={[{ value: "total", label: "Total return (dividends reinvested)" }, { value: "price", label: "Price return + cash dividends" }]}
                    onChange={(v) => set({ adjusted: v === "total" })} />
          )}
        </Group>

        <Group title="Signal">
          <Segmented label="Signal origin" value={spec.mode}
                     options={[{ value: "builtin", label: "Built-in model" }, { value: "external_weights", label: "Imported weights" }]}
                     onChange={(v) => set({ mode: v })} />
          <FileDrop label="Economic, fundamental or signal data" accept={FILES} multiple
                    names={uploads.exog?.names ?? []} busy={s.uploading === "exog"} error={uploads.exog?.error}
                    onFiles={(f) => s.upload("exog", f)} onClear={() => s.clearUpload("exog")} />
          {uploads.exog && !uploads.exog.error && (
            <>
              <p className="note wide">{uploads.exog.columns.length} series loaded, from {uploads.exog.start} to {uploads.exog.end}.</p>
              <NumberBox label="Publication lag" suffix="days" min={0} max={120} step={1} value={spec.exog_lag}
                         help="Delay between a data point's reference date and its release. A monthly series typically needs 20 to 45 days. This lag is what keeps the backtest from using a data point before it was published."
                         onChange={(v) => set({ exog_lag: Math.round(v) })} />
            </>
          )}
          {spec.mode === "external_weights" ? (
            <>
              <FileDrop label="Target weights file" accept=".csv,.xlsx,.xls,.txt" names={uploads.weights?.names ?? []}
                        busy={s.uploading === "weights"} error={uploads.weights?.error}
                        hint="A date column, then one column per symbol holding the target weight. Fractions or percentages, either works."
                        onFiles={(f) => s.upload("weights", f)} onClear={() => s.clearUpload("weights")} />
              <Select label="Row handling" value={spec.weights_normalize}
                      options={["None", "Scale to 100%"].map((o) => ({ value: o, label: o }))} onChange={(v) => set({ weights_normalize: v })} />
              <Select label="Rebalance calendar" value={spec.weights_calendar}
                      options={["File dates", "Engine calendar"].map((o) => ({ value: o, label: o }))} onChange={(v) => set({ weights_calendar: v })} />
            </>
          ) : (
            <>
              <Select label="Model" wide value={spec.strategy}
                      options={meta.models.map((m) => ({ value: m.key, label: m.label }))}
                      onChange={(v) => set({ strategy: v, params: {} })} />
              <p className="note wide">{model.description}</p>
              {model.needs_exog && !exogCols.length && (
                <p className="flag wide">This model reads exogenous series. Upload a file above, or it will have no signal.</p>
              )}
              {model.params.map((p) => (
                <Param key={`${model.key}.${p.key}`} p={p} value={spec.params[p.key]} series={exogCols}
                       onChange={(v) => s.setParam(p.key, v)} />
              ))}
            </>
          )}
        </Group>

        <Group title="Execution" defaultOpen={false}
               summary={`${meta.rebalance[spec.rebalance]?.split(" ")[0]}, lag ${spec.execution_lag}`}>
          <Select label="Rebalance" wide value={spec.rebalance}
                  options={Object.entries(meta.rebalance).map(([value, label]) => ({ value, label }))} onChange={(v) => set({ rebalance: v })} />
          <NumberBox label="Execution lag" suffix="days" min={0} max={5} step={1} value={spec.execution_lag}
                     help="1 = signal at the close, executed the next session. 0 assumes execution at the price that produced the signal."
                     onChange={(v) => set({ execution_lag: Math.round(v) })} />
          <Select label="Execution price" value={spec.execute_at_open ? "open" : "close"}
                  options={[{ value: "close", label: "Close" }, { value: "open", label: "Open" }]}
                  onChange={(v) => set({ execute_at_open: v === "open" })} />
          <NumberBox label="Initial capital" suffix="$" min={1000} max={1e9} step={10000} value={spec.initial_capital}
                     onChange={(v) => set({ initial_capital: v })} />
          <Check label="Trim the warm-up period" checked={spec.trim_warmup}
                 help="Start the record on the first day capital is actually at risk, for the strategy and the benchmark alike."
                 onChange={(v) => set({ trim_warmup: v })} />
        </Group>

        <Group title="Frictions" defaultOpen={false}
               summary={`${spec.commission_bps + spec.slippage_bps} bps a trade`}>
          <NumberBox label="Commission" suffix="bps" min={0} max={200} step={1} value={spec.commission_bps} onChange={(v) => set({ commission_bps: v })} />
          <NumberBox label="Slippage" suffix="bps" min={0} max={500} step={5} value={spec.slippage_bps}
                     help="Half the bid-ask spread, charged on every trade regardless of size."
                     onChange={(v) => set({ slippage_bps: v })} />
          <Select label="Market impact" wide value={spec.impact_model}
                  options={[{ value: "flat", label: "None" },
                            { value: "sqrt_vol", label: "Square-root law, scaled by volatility" },
                            { value: "sqrt", label: "Square-root, fixed cost at 10% of volume" }]}
                  onChange={(v) => set({ impact_model: v })} />
          <NumberBox label="Cash rate" suffix="% a year" min={0} max={15} step={0.25} value={pct(spec.cash_rate_pa)}
                     onChange={(v) => set({ cash_rate_pa: v / 100 })} />
          <NumberBox label="Management fee" suffix="% a year" min={0} max={10} step={0.05} value={pct(spec.management_fee_pa)}
                     onChange={(v) => set({ management_fee_pa: v / 100 })} />
        </Group>

        <p className="controls-more">
          Blended benchmarks, sleeves, currencies, comparison strategies and saved presets are not in this version yet.{" "}
          <a href={meta.legacy_url} target="_blank" rel="noreferrer">Open the classic app</a> for those.
        </p>
      </div>

      <div className="controls-foot">
        {s.runError && <p className="flag" role="alert">{s.runError}</p>}
        {s.stale && !s.running && <p className="stale">Settings changed since this run.</p>}
        <button type="submit" className="primary" disabled={!canRun || s.running}>
          {s.running ? `Running… ${s.elapsed.toFixed(1)} s` : "Run backtest"}
        </button>
        {!canRun && <p className="field-help">{spec.source === "upload" ? "Choose a price file to run." : "Enter at least one symbol to run."}</p>}
      </div>
    </form>
  );
}

export type { Spec };
