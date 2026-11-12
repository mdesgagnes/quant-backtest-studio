// The remaining chart types of the Results screen: a single line with a
// reference level, signed bars, a histogram, and the month-by-year grid.
import { useMemo } from "react";
import type { ChartBlock } from "../api";
import { axisText, ChartFrame, useChart, yearAxis } from "./core";
import type { Palette } from "./core";

const num = (v: number, d = 2) => v.toLocaleString("en-CA", { minimumFractionDigits: d, maximumFractionDigits: d });

const tip = (p: Palette) => ({
  backgroundColor: p.surface, borderColor: p.line, borderWidth: 1, padding: [6, 8],
  textStyle: { color: p.text, fontFamily: p.font, fontSize: 12 }, extraCssText: "box-shadow:none;border-radius:3px;",
});

export function LineChart({ block }: { block: ChartBlock }) {
  const dates = block.x ?? [];
  const s = block.series?.[0];
  const build = useMemo(() => (p: Palette) => ({
    grid: { left: 8, right: 44, top: 10, bottom: 22 },
    tooltip: { trigger: "axis", ...tip(p),
               axisPointer: { type: "line", lineStyle: { color: p.text3, type: "dashed" } },
               formatter: (a: { axisValue: string; value: number | null }[]) =>
                 `${a[0].axisValue}<br><b>${a[0].value == null ? "n/a" : num(a[0].value)}</b>` },
    xAxis: yearAxis(dates, p),
    yAxis: { type: "value", position: "right", scale: true, axisLabel: axisText(p),
             splitLine: { lineStyle: { color: p.lineSoft } }, splitNumber: 4 },
    series: [{ type: "line", data: s?.y ?? [], showSymbol: false, sampling: "lttb",
               lineStyle: { color: p.accent, width: 1.4 }, itemStyle: { color: p.accent },
               emphasis: { disabled: true },
               markLine: block.ref == null ? undefined : {
                 silent: true, symbol: "none", label: { show: false },
                 lineStyle: { color: p.text3, type: "dashed", width: 1 }, data: [{ yAxis: block.ref }] } }],
  }), [block, dates, s]);
  const ref = useChart(build, [build]);
  return <ChartFrame title={block.title}><div className="plot plot-s" ref={ref} role="img" aria-label={block.title} /></ChartFrame>;
}

export function Bars({ block }: { block: ChartBlock }) {
  const suffix = block.suffix ?? "";
  const build = useMemo(() => (p: Palette) => ({
    grid: { left: 8, right: 44, top: 10, bottom: 22 },
    tooltip: { trigger: "axis", ...tip(p), axisPointer: { type: "shadow", shadowStyle: { color: "rgba(255,255,255,0.04)" } },
               formatter: (a: { axisValue: string; value: number | null }[]) =>
                 `${a[0].axisValue}<br><b>${a[0].value == null ? "n/a" : (a[0].value > 0 ? "+" : "") + num(a[0].value)}${suffix}</b>` },
    xAxis: { type: "category", data: block.labels ?? [], axisTick: { show: false },
             axisLine: { lineStyle: { color: p.line } }, axisLabel: { ...axisText(p), hideOverlap: true } },
    yAxis: { type: "value", position: "right", splitNumber: 4,
             axisLabel: { ...axisText(p), formatter: (v: number) => `${v}${suffix}` },
             splitLine: { lineStyle: { color: p.lineSoft } } },
    series: [{ type: "bar", barMaxWidth: 22, emphasis: { disabled: true },
               data: (block.values ?? []).map((v) => ({ value: v, itemStyle: { color: (v ?? 0) >= 0 ? p.gain : p.loss } })) }],
  }), [block, suffix]);
  const ref = useChart(build, [build]);
  return <ChartFrame title={block.title}><div className="plot plot-s" ref={ref} role="img" aria-label={block.title} /></ChartFrame>;
}

export function Distribution({ block }: { block: ChartBlock }) {
  const edges = block.edges ?? [];
  const counts = block.counts ?? [];
  const var95 = block.var95 ?? 0;
  const build = useMemo(() => (p: Palette) => {
    const mids = counts.map((_, i) => (edges[i] + edges[i + 1]) / 2);
    const varBin = Math.max(0, mids.findIndex((m) => m >= var95));
    return {
      grid: { left: 8, right: 44, top: 22, bottom: 22 },
      tooltip: { trigger: "axis", ...tip(p), axisPointer: { type: "shadow", shadowStyle: { color: "rgba(255,255,255,0.04)" } },
                 formatter: (a: { dataIndex: number; value: number }[]) => {
                   const i = a[0].dataIndex;
                   return `${num(edges[i])}% to ${num(edges[i + 1])}%<br><b>${a[0].value}</b> of ${block.n} periods`;
                 } },
      xAxis: { type: "category", data: mids.map((m) => m.toFixed(2)), axisTick: { show: false },
               axisLine: { lineStyle: { color: p.line } },
               axisLabel: { ...axisText(p), hideOverlap: true,
                            interval: Math.max(1, Math.floor(mids.length / 7)),
                            formatter: (v: string) => `${Number(v).toFixed(1)}%` } },
      yAxis: { type: "value", position: "right", splitNumber: 3, axisLabel: axisText(p),
               splitLine: { lineStyle: { color: p.lineSoft } } },
      series: [{ type: "bar", barCategoryGap: "8%", emphasis: { disabled: true },
                 data: counts, itemStyle: { color: p.series[1] },
                 markLine: { silent: true, symbol: "none",
                             lineStyle: { color: p.loss, type: "dashed", width: 1 },
                             label: { formatter: `VaR 95%: ${num(var95)}%`, color: p.loss, fontFamily: p.font,
                                      fontSize: 11, position: "insideEndTop" },
                             data: [{ xAxis: varBin }] } }],
    };
  }, [block, counts, edges, var95]);
  const ref = useChart(build, [build]);
  return <ChartFrame title={block.title}><div className="plot plot-s" ref={ref} role="img" aria-label={`${block.title}. 5% of periods fell below ${num(var95)}%.`} /></ChartFrame>;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// Month-by-year returns as a real table: selectable text, aligned figures,
// readable by a screen reader. Shade follows sign and size, capped at the
// 95th percentile of absolute values so one crash month does not flatten
// the rest.
export function MonthlyGrid({ block }: { block: ChartBlock }) {
  const z = block.z ?? [];
  const years = block.years ?? [];
  const cap = useMemo(() => {
    const all = z.flat().filter((v): v is number => v != null).map(Math.abs).sort((a, b) => a - b);
    return all.length ? all[Math.floor(all.length * 0.95)] || 1 : 1;
  }, [z]);
  if (!years.length) {
    return <ChartFrame title={block.title}><p className="empty-line">Monthly returns need data at least as fine as monthly.</p></ChartFrame>;
  }
  return (
    <ChartFrame title={`${block.title}, %`}>
      <div className="scroll-x">
        <table className="grid-heat">
          <thead><tr><th scope="col" className="sticky-col">Year</th>{MONTHS.map((m) => <th key={m} scope="col">{m}</th>)}</tr></thead>
          <tbody>
            {years.map((y, r) => (
              <tr key={y}>
                <th scope="row" className="sticky-col">{y}</th>
                {MONTHS.map((m, c) => {
                  const v = z[r]?.[c];
                  if (v == null) return <td key={m} className="blank" />;
                  const a = Math.round(Math.min(1, Math.abs(v) / cap) * 50);
                  return (
                    <td key={m} className={v > 0 ? "heat-pos" : v < 0 ? "heat-neg" : ""}
                        style={{ "--a": `${a}%` } as React.CSSProperties}>{v.toFixed(1)}</td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartFrame>
  );
}
