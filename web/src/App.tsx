// The shell: top bar with the workspaces, the control panel, the run's
// context and KPI strip, the tabs, and the screen. On a phone the control
// panel becomes a sheet opened from the top bar.
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { Meta } from "./api";
import { BlockList, KpiStrip } from "./blocks/Blocks";
import { ControlPanel } from "./controls/ControlPanel";
import { Results } from "./screens/Results";
import { StudioProvider, useStudio } from "./state";

const WORKSPACES = ["Backtest", "Markets", "Research"] as const;
type Workspace = (typeof WORKSPACES)[number];

// Screens with an arrangement of their own. Others fall back to BlockList.
const LAYOUTS: Record<string, (p: { blocks: import("./api").Block[] }) => React.ReactNode> = {
  results: Results,
};

export function App() {
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta, staleTime: Infinity, retry: 1 });
  if (meta.isPending) return <div className="boot" role="status">Loading…</div>;
  if (meta.isError) {
    return (
      <div className="boot">
        <p className="flag" role="alert">{meta.error.message}</p>
        <button type="button" className="secondary" onClick={() => meta.refetch()}>Try again</button>
      </div>
    );
  }
  return <StudioProvider meta={meta.data}><Shell meta={meta.data} /></StudioProvider>;
}

function Shell({ meta }: { meta: Meta }) {
  const [workspace, setWorkspace] = useState<Workspace>("Backtest");
  const [sheet, setSheet] = useState(false);
  const s = useStudio();
  const main = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSheet(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className={`app ${sheet ? "sheet-open" : ""}`}>
      <a className="skip" href="#main">Skip to results</a>
      <header className="topbar">
        <span className="brand">Quant Backtest Studio</span>
        <nav className="workspaces" aria-label="Workspaces">
          {WORKSPACES.map((w) => (
            <button key={w} type="button" aria-current={workspace === w ? "page" : undefined}
                    onClick={() => setWorkspace(w)}>{w}</button>
          ))}
        </nav>
        {workspace === "Backtest" && (
          <button type="button" className="secondary sheet-toggle" aria-expanded={sheet}
                  aria-controls="controls" onClick={() => setSheet(!sheet)}>
            {sheet ? "Close" : "Settings"}
          </button>
        )}
      </header>

      {workspace === "Backtest" ? (
        <div className="body">
          <aside className="side" id="controls" aria-label="Settings">
            <ControlPanel onRan={() => { setSheet(false); main.current?.scrollTo({ top: 0 }); }} />
          </aside>
          <main className="main" id="main" ref={main} tabIndex={-1}>
            <Backtest meta={meta} />
          </main>
        </div>
      ) : (
        <main className="main solo" id="main">
          <Classic name={workspace} url={meta.legacy_url} />
        </main>
      )}
      {s.running && <div className="progress" role="status" aria-label="Running the backtest" />}
    </div>
  );
}

function Classic({ name, url, what }: { name: string; url: string; what?: string }) {
  return (
    <div className="empty">
      <h2>{name} has not been rebuilt yet</h2>
      <p>{what ?? `The ${name} workspace`} still lives in the classic app, with every feature it had.</p>
      <a className="button secondary" href={url} target="_blank" rel="noreferrer">Open the classic app</a>
    </div>
  );
}

function Backtest({ meta }: { meta: Meta }) {
  const s = useStudio();
  const [tab, setTabName] = useState(meta.tabs[0].name);
  const [seen, setSeen] = useState<Set<string>>(() => new Set([meta.tabs[0].name]));
  const setTab = (name: string) => { setTabName(name); setSeen((v) => (v.has(name) ? v : new Set(v).add(name))); };
  const current = meta.tabs.find((t) => t.name === tab) ?? meta.tabs[0];
  const run = s.run;

  if (!run) return s.running ? <Skeleton label={`Running the backtest… ${s.elapsed.toFixed(1)} s`} /> : <Start meta={meta} />;

  return (
    <>
      <header className={`runhead ${s.stale ? "is-stale" : ""}`}>
        <div className="runline">
          <h1>{run.context[0]}</h1>
          <ul className="context">{run.context.slice(1).map((c) => <li key={c}>{c}</li>)}</ul>
          <span className="runmeta num" title={`Run on ${run.stamp}`}>
            {run.cached ? "from cache" : `${run.seconds.toFixed(2)} s`}
          </span>
        </div>
        <KpiStrip items={run.kpis} />
        {run.notices.map((n) => <p key={n} className="flag">{n}</p>)}
      </header>

      <div className="tabs" role="tablist" aria-label="Backtest views"
           onKeyDown={(e) => {
             const i = meta.tabs.findIndex((t) => t.name === tab);
             const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
             if (!d) return;
             const next = meta.tabs[(i + d + meta.tabs.length) % meta.tabs.length].name;
             setTab(next);
             (e.currentTarget.querySelector(`[data-tab="${next}"]`) as HTMLElement | null)?.focus();
           }}>
        {meta.tabs.map((t) => (
          <button key={t.name} type="button" role="tab" data-tab={t.name} id={`tab-${t.name}`}
                  aria-selected={t.name === tab} aria-controls={`panel-${t.name}`} tabIndex={t.name === tab ? 0 : -1}
                  className={t.screen ? "" : "pending"} onClick={() => setTab(t.name)}>{t.name}</button>
        ))}
      </div>

      {/* A tab stays mounted once opened, so coming back to it costs nothing. */}
      {meta.tabs.filter((t) => seen.has(t.name) || t.name === current.name).map((t) => (
        <div key={`${run.id}:${t.name}`} className="screen" role="tabpanel" id={`panel-${t.name}`}
             aria-labelledby={`tab-${t.name}`} hidden={t.name !== current.name}>
          {t.screen
            ? <Screen runId={run.id} name={t.screen} />
            : <Classic name={t.name} url={meta.legacy_url} what={`The ${t.name} tab`} />}
        </div>
      ))}
    </>
  );
}

function Screen({ runId, name }: { runId: string; name: string }) {
  const s = useStudio();
  const q = useQuery({
    queryKey: ["screen", runId, name], queryFn: () => api.screen(runId, name),
    staleTime: Infinity, gcTime: 30 * 60 * 1000, retry: false,
  });
  if (q.isPending) return <Skeleton label="Loading this view…" />;
  if (q.isError) {
    return (
      <div className="empty">
        <p className="flag" role="alert">{q.error.message}</p>
        <button type="button" className="secondary" onClick={() => (q.error.message.includes("no longer held") ? s.start() : q.refetch())}>
          {q.error.message.includes("no longer held") ? "Run it again" : "Try again"}
        </button>
      </div>
    );
  }
  const Layout = LAYOUTS[name];
  return Layout ? <Layout blocks={q.data.blocks} /> : <BlockList blocks={q.data.blocks} />;
}

function Skeleton({ label }: { label: string }) {
  return (
    <div className="skeleton" role="status">
      <p>{label}</p>
      <div className="sk sk-strip" /><div className="sk sk-chart" /><div className="sk sk-row" /><div className="sk sk-row" />
    </div>
  );
}

function Start({ meta }: { meta: Meta }) {
  const routes: [string, string][] = [
    ["Simulate a strategy", "Pull ETFs from Yahoo Finance or upload a price file, choose a model, and the engine handles drift, frictions and execution."],
    ["Test an allocation you already have", "Upload target weights from a spreadsheet or committee and measure them under the same frictions as any built-in model."],
    ["Analyse a track record", "Upload periodic returns or pick Yahoo tickers, and get the statistics with no simulation at all. This lives in the classic app for now."],
  ];
  return (
    <div className="start">
      <p className="start-lead">Pick a source in Settings, then run the backtest. Everything here fills in from that one action.</p>
      <div className="start-grid">
        <section>
          <h3 className="h">Where to start</h3>
          <dl className="routes">{routes.map(([t, b]) => <div key={t}><dt>{t}</dt><dd>{b}</dd></div>)}</dl>
        </section>
        <section>
          <h3 className="h">{meta.models.length} models</h3>
          <ul className="models">
            {meta.models.map((m) => (
              <li key={m.key}><details><summary>{m.label}</summary><p>{m.description}</p></details></li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
