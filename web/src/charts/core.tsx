// One chart language for the whole app. Every chart goes through `useChart`
// and takes its colours and type from the design tokens, so no component
// carries a colour of its own. ECharts is loaded on demand (its own chunk),
// which keeps it out of the first paint.
import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { EChartsType } from "echarts/core";

type Core = typeof import("echarts/core");
let loading: Promise<Core> | null = null;

export function loadCharts(): Promise<Core> {
  if (!loading) loading = import("./bundle").then((m) => m.default as unknown as Core);
  return loading;
}

export interface Palette {
  text: string; text2: string; text3: string; line: string; lineSoft: string;
  surface: string; accent: string; gain: string; loss: string; lossWash: string;
  series: string[]; font: string;
}

export function palette(): Palette {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    text: v("--text"), text2: v("--text-2"), text3: v("--text-3"),
    line: v("--line"), lineSoft: v("--line-soft"), surface: v("--surface"),
    accent: v("--accent"), gain: v("--gain"), loss: v("--loss"), lossWash: v("--loss-wash"),
    // The subject first, then context. Never the sign colours.
    series: [v("--accent"), v("--series-2"), v("--series-3"), v("--series-4"),
             v("--series-5"), v("--series-6")],
    font: v("--font"),
  };
}

export const axisText = (p: Palette) => ({ color: p.text3, fontFamily: p.font, fontSize: 11 });

// Category axis of trading days labelled by year: a label on the first
// session of each calendar year (thinned when the span is long).
export function yearAxis(dates: string[], p: Palette, show = true) {
  const firstOfYear = new Set<number>();
  let years = 0;
  dates.forEach((d, i) => {
    if (i === 0 || d.slice(0, 4) !== dates[i - 1].slice(0, 4)) { firstOfYear.add(i); years += 1; }
  });
  const every = years > 14 ? 2 : 1;
  return {
    type: "category" as const, data: dates, boundaryGap: false,
    axisLine: { lineStyle: { color: p.line } }, axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: {
      show, ...axisText(p), hideOverlap: true,
      interval: (i: number) => firstOfYear.has(i) && (Number(dates[i].slice(0, 4)) % every === 0),
      formatter: (d: string) => d.slice(0, 4),
    },
  };
}

export function useChart(
  build: ((p: Palette) => Record<string, unknown>) | null,
  deps: unknown[],
  onReady?: (chart: EChartsType) => void | (() => void),
): RefObject<HTMLDivElement | null> {
  const el = useRef<HTMLDivElement | null>(null);
  const chart = useRef<EChartsType | null>(null);

  useEffect(() => {
    let dead = false;
    let cleanup: void | (() => void);
    let observer: ResizeObserver | null = null;
    if (!build || !el.current) return;
    loadCharts().then((core) => {
      if (dead || !el.current) return;
      const c = chart.current ?? core.init(el.current, undefined, { renderer: "canvas" });
      chart.current = c;
      c.setOption({ animation: false, ...build(palette()) }, { notMerge: true });
      cleanup = onReady?.(c);
      observer = new ResizeObserver(() => c.resize());
      observer.observe(el.current);
    });
    return () => { dead = true; observer?.disconnect(); if (cleanup) cleanup(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => () => { chart.current?.dispose(); chart.current = null; }, []);
  return el;
}

export function ChartFrame({ title, children, aside, label }: {
  title: string; children: React.ReactNode; aside?: React.ReactNode; label?: string;
}) {
  return (
    <figure className="chart" aria-label={label ?? title}>
      <figcaption className="chart-head">
        <span className="chart-title">{title}</span>
        {aside}
      </figcaption>
      {children}
    </figure>
  );
}
