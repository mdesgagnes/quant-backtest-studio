// Value and drawdown on one time axis, read from one crosshair.
//
// The readout line above the plot replaces a floating tooltip: it shows the
// date under the cursor and each series' value and drawdown, in fixed
// columns, so the eye stays in one place while the cursor moves. With no
// cursor it shows the last date. Log scale, range and series visibility are
// changed here in the browser; none of them asks the server for anything.
import { useMemo, useRef, useState } from "react";
import type { EChartsType } from "echarts/core";
import type { ChartBlock } from "../api";
import { axisText, useChart, yearAxis } from "./core";
import type { Palette } from "./core";

const RANGES: [string, number | null][] = [["1Y", 252], ["3Y", 756], ["5Y", 1260], ["10Y", 2520], ["Max", null]];

const fmt = (v: number | null | undefined, digits = 1) =>
  v == null ? "n/a" : v.toLocaleString("en-CA", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export function TimePanel({ value, drawdown }: { value: ChartBlock; drawdown?: ChartBlock }) {
  const dates = value.x ?? [];
  const series = value.series ?? [];
  const dd = drawdown?.series ?? [];
  const [log, setLog] = useState(value.log ?? true);
  const [range, setRange] = useState("Max");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const readout = useRef<HTMLDivElement | null>(null);

  const logs = useMemo(() => series.map((s) => s.y.map((v) => (v != null && v > 0 ? Math.log10(v) : null))), [series]);

  const span = RANGES.find(([k]) => k === range)?.[1] ?? null;
  const from = span == null ? 0 : Math.max(0, dates.length - span);

  const write = (i: number) => {
    const root = readout.current;
    if (!root || !dates.length) return;
    const idx = Math.min(Math.max(i, 0), dates.length - 1);
    root.querySelector("[data-date]")!.textContent = dates[idx];
    series.forEach((s, k) => {
      const a = root.querySelector(`[data-v="${k}"]`);
      const b = root.querySelector(`[data-d="${k}"]`);
      if (a) a.textContent = fmt(s.y[idx]);
      if (b) b.textContent = dd[k] ? `${fmt(dd[k].y[idx])}%` : "";
    });
  };

  const build = useMemo(() => (p: Palette) => {
    const colour = (k: number) => p.series[k % p.series.length];
    const shown = (name: string) => !hidden.has(name);
    const zoom = { start: (from / Math.max(dates.length - 1, 1)) * 100, end: 100 };
    return {
      grid: [
        { left: 8, right: 52, top: 8, height: drawdown ? "62%" : "86%" },
        ...(drawdown ? [{ left: 8, right: 52, top: "72%", bottom: 22 }] : []),
      ],
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      tooltip: {
        trigger: "axis", showContent: false,
        axisPointer: { type: "line", lineStyle: { color: p.text3, width: 1, type: "dashed" }, label: { show: false } },
      },
      dataZoom: [{ type: "inside", xAxisIndex: drawdown ? [0, 1] : [0], ...zoom,
                   zoomOnMouseWheel: "ctrl", moveOnMouseMove: false, moveOnMouseWheel: false,
                   preventDefaultMouseMove: false }],
      xAxis: [
        { ...yearAxis(dates, p, !drawdown), gridIndex: 0 },
        ...(drawdown ? [{ ...yearAxis(dates, p, true), gridIndex: 1 }] : []),
      ],
      yAxis: [
        // Log scale is drawn as log10 of the value on a linear axis, so the
        // grid lines are evenly spaced and labelled with the value itself.
        { type: "value", gridIndex: 0, position: "right", scale: true, splitNumber: 5,
          axisLabel: { ...axisText(p), formatter: (v: number) => fmt(log ? Math.pow(10, v) : v, 0) },
          splitLine: { lineStyle: { color: p.lineSoft } }, axisLine: { show: false },
          minorSplitLine: { show: false } },
        ...(drawdown ? [{
          type: "value", gridIndex: 1, position: "right", max: 0, splitNumber: 2,
          axisLabel: { ...axisText(p), formatter: (v: number) => `${fmt(v, 0)}%` },
          splitLine: { lineStyle: { color: p.lineSoft } }, axisLine: { show: false } }] : []),
      ],
      series: [
        ...series.filter((s) => shown(s.name)).map((s) => {
          const k = series.indexOf(s);
          return { type: "line", name: s.name, xAxisIndex: 0, yAxisIndex: 0,
                   data: log ? logs[series.indexOf(s)] : s.y,
                   showSymbol: false, connectNulls: false, silent: true,
                   lineStyle: { color: colour(k), width: k === 0 ? 1.8 : 1.2 },
                   itemStyle: { color: colour(k) }, emphasis: { disabled: true }, z: k === 0 ? 3 : 2 };
        }),
        ...(drawdown ? dd.filter((s) => shown(s.name)).map((s) => {
          const k = dd.indexOf(s);
          return { type: "line", name: `${s.name} drawdown`, data: s.y, xAxisIndex: 1, yAxisIndex: 1,
                   showSymbol: false, silent: true,
                   lineStyle: { color: k === 0 ? p.loss : colour(k), width: 1.1 },
                   areaStyle: k === 0 ? { color: p.lossWash, origin: "auto" } : undefined,
                   emphasis: { disabled: true }, z: k === 0 ? 3 : 2 };
        }) : []),
      ],
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, drawdown, log, from, hidden]);

  const ref = useChart(build, [build], (chart: EChartsType) => {
    const onMove = (e: unknown) => {
      const info = (e as { axesInfo?: { value: number }[] }).axesInfo?.[0];
      if (info) write(info.value);
    };
    const onOut = () => write(dates.length - 1);
    chart.on("updateAxisPointer", onMove);
    chart.getZr().on("globalout", onOut);
    write(dates.length - 1);
    return () => { chart.off("updateAxisPointer", onMove); chart.getZr()?.off("globalout", onOut); };
  });

  const toggle = (name: string) => setHidden((h) => {
    const n = new Set(h);
    if (n.has(name)) n.delete(name); else if (n.size < series.length - 1) n.add(name);
    return n;
  });

  return (
    <figure className="chart timepanel" aria-label={`${value.title} and ${drawdown?.title ?? ""}`}>
      <figcaption className="chart-head">
        <span className="chart-title">{value.title}{drawdown ? ", with drawdown from prior peak" : ""}</span>
        <span className="chart-tools">
          <span className="seg" role="group" aria-label="Period shown">
            {RANGES.filter(([, n]) => n == null || n < dates.length).map(([k]) => (
              <button key={k} type="button" aria-pressed={range === k} onClick={() => setRange(k)}>{k}</button>
            ))}
          </span>
          {value.log_toggle && (
            <label className="switch" title={value.log_help}>
              <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} />
              <span>Log scale</span>
            </label>
          )}
        </span>
      </figcaption>
      <div className="readout" ref={readout} aria-live="off">
        <span className="readout-date" data-date />
        {series.map((s, k) => (
          <button key={s.name} type="button" className="readout-item" data-k={k}
                  aria-pressed={!hidden.has(s.name)} onClick={() => toggle(s.name)}
                  title={hidden.has(s.name) ? "Show this series" : "Hide this series"}>
            <i className={`swatch swatch-${k}`} aria-hidden="true" />
            <span className="readout-name">{s.name}</span>
            <span className="readout-v num" data-v={k} />
            <span className="readout-d num" data-d={k} />
          </button>
        ))}
      </div>
      <div className="plot plot-time" ref={ref} role="img"
           aria-label={`${series.map((s) => s.name).join(" and ")}, ${dates[0]} to ${dates[dates.length - 1]}`} />
    </figure>
  );
}
