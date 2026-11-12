// The shell: the workspace bar, the control panel and the page. Everything
// inside them is drawn from what the server sends (see live.tsx and
// blocks/Render.tsx). On a phone the control panel is a sheet opened from
// the top bar.
import { useEffect, useRef, useState } from "react";
import { Blocks } from "./blocks/Render";
import { LiveProvider, useLive } from "./live";
import type { Block } from "./live";

export function App() {
  return <LiveProvider><Shell /></LiveProvider>;
}

function Shell() {
  const { page, busy, error, change, retry } = useLive();
  const [sheet, setSheet] = useState(false);
  const main = useRef<HTMLElement | null>(null);
  const wasBusy = useRef(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSheet(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // A run or load started from the sheet closes it, so the result is seen.
  const title = page?.main.find((b) => b.kind === "context")?.lead;
  useEffect(() => { if (wasBusy.current && !busy && title) setSheet(false); wasBusy.current = busy; }, [busy, title]);

  if (!page) {
    return (
      <div className="boot" role="status">
        {error ? (<><p className="flag" role="alert">{error}</p><button type="button" className="secondary" onClick={retry}>Try again</button></>) : "Loading…"}
      </div>
    );
  }

  const ws = page.sidebar.find((b) => b.kind === "control" && b.id === "workspace");
  const side = page.sidebar.filter((b) => b !== ws && !(b.kind === "heading" && b.text === "Settings"));
  const hasSide = side.length > 0;

  return (
    <div className={`app ${sheet ? "sheet-open" : ""}`}>
      <a className="skip" href="#main">Skip to content</a>
      <header className="topbar">
        <span className="brand">Quant Backtest Studio</span>
        {ws && (
          <nav className="workspaces" aria-label="Workspaces">
            {(ws.options as string[]).map((w, i) => (
              <button key={w} type="button" aria-current={ws.value === i ? "page" : undefined}
                      onClick={() => { if (ws.value !== i) change(ws.id, i); }}>{w}</button>
            ))}
          </nav>
        )}
        {hasSide && (
          <button type="button" className="secondary sheet-toggle" aria-expanded={sheet} aria-controls="controls"
                  onClick={() => setSheet(!sheet)}>{sheet ? "Close" : "Settings"}</button>
        )}
      </header>
      <div className={`body ${hasSide ? "" : "no-side"}`}>
        {hasSide && (
          <aside className="side" id="controls" aria-label="Settings">
            <SidePanel blocks={side} />
          </aside>
        )}
        <main className="main" id="main" ref={main} tabIndex={-1}>
          {error && (
            <div className="page-error">
              <p className="flag" role="alert">{error}</p>
              <button type="button" className="secondary" onClick={retry}>Try again</button>
            </div>
          )}
          <Blocks blocks={page.main} />
        </main>
      </div>
      {busy && <div className="progress" role="status" aria-label="Working" />}
    </div>
  );
}

// The panel's main action (Run backtest, Load markets, Load universe) stays
// fixed at the bottom; everything above it scrolls.
function SidePanel({ blocks }: { blocks: Block[] }) {
  const isAction = (b: Block) => b.kind === "control" && b.control === "button" && /^(Run|Load)\b/.test(b.label);
  const action = [...blocks].reverse().find(isAction);
  return (
    <div className="controls">
      <div className="controls-scroll"><Blocks blocks={blocks.filter((b) => b !== action)} side /></div>
      {action && <div className="controls-foot"><Blocks blocks={[action]} side /></div>}
    </div>
  );
}
