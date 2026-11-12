// Draws any chart the server describes (server/present.py `chart_block`):
// lines, areas, stacks, bands, bars, points, histograms and grids. The
// description carries what is plotted; colour and type come from the
// tokens, so every chart in the app speaks the same language.
import { useMemo, useState } from "react";
import type { Block } from "../live";
import { axisText, ChartFrame, useChart } from "./core";
import type { Palette } from "./core";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type T = Record<string, any>;

const nf = (v: number, d = 2) => v.toLocaleString("en-CA", { minimumFractionDigits: d, maximumFractionDigits: d });

function fmtY(b: Block, v: number): string {
  if (b.yformat && String(b.yformat).includes("%")) return `${nf(v * 100, 1)}%`;
  const a = Math.abs(v);
  return `${nf(v, a >= 100 ? 0 : a >= 10 ? 1 : 2)}${b.ysuffix ?? ""}`;
}

// ---------------------------------------------------------------- grids
function HeatGrid({ t, title, xt, yt }: { t: T; title: string; xt?: string; yt?: string }) {
  const z: (number | null)[][] = t.z;
  const flat = z.flat().filter((v): v is number => v != null);
  const sorted = flat.map(Math.abs).sort((a, b) => a - b);
  const cap = sorted.length ? sorted[Math.floor(sorted.length * 0.95)] || 1 : 1;
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const mid = t.scale === "rank" ? (t.mid ?? (lo + hi) / 2) : 0;
  const span = Math.max(hi - mid, mid - lo) || 1;
  const cell = (v: number) => {
    if (t.scale === "corr") return { cls: v >= 0 ? "heat-acc" : "heat-alt", a: Math.min(1, Math.abs(v)) * 55 };
    if (t.scale === "rank") return { cls: v >= mid ? "heat-acc" : "heat-neg", a: (Math.abs(v - mid) / span) * 50 };
    return { cls: v > 0 ? "heat-pos" : v < 0 ? "heat-neg" : "", a: Math.min(1, Math.abs(v) / cap) * 50 };
  };
  return (
    <ChartFrame title={title}>
      <div className="scroll-x">
        <table className="grid-heat">
          <thead><tr><th scope="col" className="sticky-col">{yt && xt ? `${yt} \\ ${xt}` : ""}</th>
            {(t.x as string[]).map((x, i) => <th key={i} scope="col">{x}</th>)}</tr></thead>
          <tbody>
            {(t.y as string[]).map((y, r) => (
              <tr key={r}>
                <th scope="row" className="sticky-col">{y}</th>
                {(t.x as string[]).map((_, c) => {
                  const v = z[r]?.[c];
                  if (v == null) return <td key={c} />;
                  const k = cell(v);
                  const shown = t.text?.[r]?.[c] ?? v.toFixed(t.decimals ?? 2);
                  return <td key={c} className={k.cls} style={{ "--a": `${Math.round(k.a)}%` } as React.CSSProperties}>{String(shown)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartFrame>
  );
}

// --------------------------------------------------------------- canvas
function Plot({ b }: { b: Block }) {
  const traces = b.traces as T[];
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const named = traces.filter((t) => t.name && !t.no_legend && (!t.hidden_line || t.stack) && t.t !== "hist");
  const log = !!b.ylog;
  const ty = (v: number | null) => (v == null ? null : log ? (v > 0 ? Math.log10(v) : null) : v);

  const build = useMemo(() => (p: Palette) => {
    const cat = b.xtype === "category";
    const date = b.xtype === "date";
    const hist = traces.find((t) => t.t === "hist");
    const cats: string[] = hist
      ? (hist.counts as number[]).map((_, i) => ((hist.edges[i] + hist.edges[i + 1]) / 2).toFixed(2))
      : cat ? (traces.find((t) => t.x)?.x ?? []).map(String) : [];
    const pts = (t: T, ys: (number | null)[]) =>
      cat ? ys : (t.x as (string | number)[]).map((x, i) => [x, ys[i]]);
    const ramp = [p.gain, p.series[4], p.series[5], p.loss];
    const bandColour = (i: number, n: number) => (n <= 2 ? [p.gain, p.loss][i] : n === 3 ? [p.loss, p.series[5], p.gain][i] : ramp[i % 4]);
    let k = -1;
    const series: T[] = [];
    traces.forEach((t, idx) => {
      if (hidden.has(t.name)) { if (!t.hidden_line && !t.band_to_prev) k += 1; return; }
      if (t.t === "hist") {
        series.push({ type: "bar", data: t.counts, barCategoryGap: "8%", itemStyle: { color: p.series[1] }, emphasis: { disabled: true } });
        return;
      }
      if (t.t === "bar") {
        k += 1;
        const colour = t.quiet ? p.text3 : p.series[k % p.series.length];
        series.push({
          type: "bar", name: t.name, barMaxWidth: 26, emphasis: { disabled: true },
          itemStyle: { color: colour, opacity: t.opacity ?? 1 },
          data: (t.y as (number | null)[]).map((v, i) => {
            const item: T = { value: cat ? v : [t.x[i], v] };
            if (t.signed) item.itemStyle = { color: (v ?? 0) >= 0 ? p.gain : p.loss };
            return item;
          }),
        });
        return;
      }
      if (t.t === "points") {
        k += 1;
        series.push({
          type: "scatter", name: t.name, symbolSize: 9, itemStyle: { color: p.accent },
          data: (t.x as number[]).map((x, i) => ({ value: [x, t.y[i]], name: t.labels?.[i] ?? "" })),
          label: { show: !!t.labels, formatter: (q: T) => q.name, position: "top", color: p.text2, fontFamily: p.font, fontSize: 11 },
          emphasis: { disabled: true },
        });
        return;
      }
      // lines, areas, stacks and bands
      if (t.hidden_line && !t.stack && traces[idx + 1]?.band_to_prev) return;   // upper edge: drawn with the band
      if (t.band_to_prev) {
        const upper = traces[idx - 1];
        const lowY = (t.y as (number | null)[]).map(ty);
        const diff = lowY.map((v, i) => { const u = ty(upper.y[i]); return v == null || u == null ? null : u - v; });
        series.push({ type: "line", data: pts(t, lowY), stack: `band${idx}`, showSymbol: false, silent: true,
                      lineStyle: { opacity: 0 }, tooltip: { show: false } });
        series.push({ type: "line", name: t.name, data: pts(t, diff), stack: `band${idx}`, showSymbol: false, silent: true,
                      lineStyle: { opacity: 0 }, areaStyle: { color: p.accent, opacity: 0.14 }, tooltip: { show: false } });
        return;
      }
      k += 1;
      // Cash is the absence of a position: it is drawn quiet and does not take a series colour.
      const cash = t.stack && t.name === "Cash";
      if (cash) k -= 1;
      const colour = cash ? p.text3 : t.tone === "neg" ? p.loss : p.series[k % p.series.length];
      const first = k === 0;
      series.push({
        type: "line", name: t.name, data: pts(t, (t.y as (number | null)[]).map(ty)), showSymbol: false,
        connectNulls: false, emphasis: { disabled: true },
        stack: t.stack ? "stack" : undefined,
        lineStyle: { color: colour, width: t.stack ? 0 : first ? 1.7 : 1.2, type: t.dash ? "dashed" : "solid" },
        itemStyle: { color: colour },
        areaStyle: t.stack ? { color: colour, opacity: 0.75 } : t.area ? { color: colour, opacity: 0.16, origin: "auto" } : undefined,
      });
    });
    const marks: T[] = [
      ...(b.hlines as number[]).map((y) => ({ yAxis: ty(y), lineStyle: { color: p.text3, type: "dashed", width: 1 }, label: { show: false } })),
      ...(b.vlines as T[]).map((v) => ({
        xAxis: hist ? Math.max(0, cats.findIndex((c) => Number(c) >= v.x)) : v.x,
        lineStyle: { color: v.tone === "neg" ? p.loss : p.text3, type: "dashed", width: 1 },
        label: { show: !!v.label, formatter: String(v.label ?? "").replace(/<[^>]+>/g, ""), color: v.tone === "neg" ? p.loss : p.text2,
                 fontFamily: p.font, fontSize: 11, position: "insideEndTop" } })),
    ];
    if (series.length && marks.length) series[0].markLine = { silent: true, symbol: "none", data: marks };
    const bands = b.bands as T[];
    if (series.length && bands.length) {
      const n = (b.band_labels as string[]).length;
      series[0].markArea = { silent: true, data: bands.map((q) => [
        { xAxis: q.x0, itemStyle: { color: bandColour(q.i, n), opacity: 0.16 } }, { xAxis: q.x1 }]) };
    }
    const range = b.yrange as number[] | null;
    return {
      grid: { left: 8, right: 52, top: 12, bottom: b.xtitle ? 38 : 24 },
      // Ctrl + wheel zooms a time or value axis; the page keeps the plain wheel.
      dataZoom: !hist && !cat ? [{ type: "inside", zoomOnMouseWheel: "ctrl", moveOnMouseMove: false,
                                  moveOnMouseWheel: false, preventDefaultMouseMove: false }] : undefined,
      tooltip: {
        trigger: traces.some((t) => t.t === "points") ? "item" : "axis",
        backgroundColor: p.surface, borderColor: p.line, borderWidth: 1, padding: [6, 8],
        textStyle: { color: p.text, fontFamily: p.font, fontSize: 12 }, extraCssText: "box-shadow:none;border-radius:3px;",
        axisPointer: { type: series.some((s) => s.type === "bar") ? "shadow" : "line",
                       lineStyle: { color: p.text3, type: "dashed" }, shadowStyle: { color: "rgba(255,255,255,0.04)" } },
        valueFormatter: (v: unknown) => {
          const n = Array.isArray(v) ? v[v.length - 1] : v;
          return typeof n === "number" ? fmtY(b, log ? Math.pow(10, n) : n) : "n/a";
        },
      },
      xAxis: {
        type: hist || cat ? "category" : date ? "time" : "value", data: hist || cat ? cats : undefined,
        name: b.xtitle || undefined, nameLocation: "middle", nameGap: 24, nameTextStyle: axisText(p),
        axisLine: { lineStyle: { color: p.line } }, axisTick: { show: false }, splitLine: { show: false },
        axisLabel: { ...axisText(p), hideOverlap: true,
                     formatter: hist ? (v: string) => `${Number(v).toFixed(1)}${b.xsuffix ?? ""}`
                       : (!date && !cat) ? (v: number) => `${v}${b.xsuffix ?? ""}` : undefined },
        scale: true,
      },
      yAxis: {
        type: "value", position: "right", scale: !series.some((s) => s.type === "bar" || s.stack),
        inverse: !!b.yreversed, min: range && !log ? range[0] : undefined, max: range && !log ? range[1] : undefined,
        splitNumber: 4, axisLabel: { ...axisText(p), formatter: (v: number) => fmtY(b, log ? Math.pow(10, v) : v) },
        splitLine: { lineStyle: { color: p.lineSoft } },
      },
      series,
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b, hidden]);

  const ref = useChart(build, [build]);
  const height = Math.max(200, Math.min(460, Number(b.height) || 320)) - 40;
  let li = -1;
  return (
    <ChartFrame title={b.title}>
      {(named.length > 1 || (b.band_labels?.length ?? 0) > 0) && (
        <div className="legend">
          {traces.map((t) => {
            if (!named.includes(t)) return null;
            const cash = t.stack && t.name === "Cash";
            if (!cash) li += 1;
            return (
              <button key={t.name} type="button" className="legend-item" aria-pressed={!hidden.has(t.name)}
                      onClick={() => setHidden((h) => { const n = new Set(h); if (n.has(t.name)) n.delete(t.name); else n.add(t.name); return n; })}>
                <i className={`swatch swatch-${cash ? "quiet" : t.tone === "neg" ? "neg" : li % 6}`} aria-hidden="true" />{t.name}
              </button>
            );
          })}
          {(b.band_labels as string[] | undefined)?.map((l, i, all) => (
            <span key={l} className="legend-item"><i className={`swatch band band-${all.length}-${i}`} aria-hidden="true" />{l}</span>
          ))}
        </div>
      )}
      <div className="plot" style={{ height }} ref={ref} role="img" aria-label={b.title} />
    </ChartFrame>
  );
}

export function Figure({ b }: { b: Block }) {
  const traces = b.traces as T[];
  const grid = traces.find((t) => t.t === "heat" || t.t === "surface");
  if (grid) {
    const t = grid.t === "surface"
      ? { ...grid, x: (grid.x as number[]).map(String), y: (grid.y as number[]).map(String), scale: "rank" }
      : grid;
    return <HeatGrid t={t} title={grid.t === "surface" && b.ztitle ? `${b.title} (${b.ztitle})` : b.title} xt={b.xtitle} yt={b.ytitle} />;
  }
  if (!traces.length) return <ChartFrame title={b.title}><p className="empty-line">Nothing to plot.</p></ChartFrame>;
  return <Plot b={b} />;
}
