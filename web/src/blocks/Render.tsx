// Draws any page the server sends. One component per kind of block
// (server/present.py). A screen needs no code here unless it brings a kind
// of block that does not exist yet.
import { useEffect, useId, useState } from "react";
import type { ReactNode } from "react";
import { Figure } from "../charts/Figure";
import { useLive } from "../live";
import type { Block } from "../live";
import { Results } from "../screens/Results";
import { KpiStrip, Note } from "./Blocks";
import { Control, Upload } from "./Controls";

// Markup the server has already cleaned (server/present.py `clean_html`).
function Html({ html, className }: { html: string; className?: string }) {
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

function Spark({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const pts = values.map((v, i) => `${(i * 78) / (values.length - 1) + 1},${19 - ((v - lo) / span) * 17}`).join(" ");
  return (
    <svg className={`spark ${values[values.length - 1] >= values[0] ? "pos" : "neg"}`} width="80" height="20" viewBox="0 0 80 20" aria-hidden="true">
      <polyline fill="none" strokeWidth="1.3" points={pts} />
    </svg>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Cell = Record<string, any>;

function cellClass(col: Block, cell: Cell, first: boolean) {
  return [col.align === "right" ? "r num" : "", cell.tone ?? "", cell.strong ? "strong" : "",
          cell.heat ? `heat-${cell.heat[0]}` : "", first ? "sticky-col" : ""].join(" ");
}

function Table({ b }: { b: Block }) {
  const cols = b.columns as Block[];
  const rows = b.rows as Cell[][];
  const [sort, setSort] = useState<{ col: number; dir: 1 | -1 } | null>(null);
  if (!cols.length || !rows.length) return <p className="empty-line">Nothing to show here.</p>;
  let shown = rows;
  if (sort) {
    const num = (c: Cell) => { const n = Number(String(c.t ?? "").replace(/[%$,x\s]|bps|pts/g, "")); return Number.isFinite(n) && c.t !== "" ? n : null; };
    shown = [...rows].sort((a, z) => {
      const x = a[sort.col], y = z[sort.col];
      const nx = x.b ?? num(x), ny = y.b ?? num(y);
      if (nx != null && ny != null) return (nx - ny) * sort.dir;
      if (nx != null) return -1;
      if (ny != null) return 1;
      return String(x.t ?? "").localeCompare(String(y.t ?? "")) * sort.dir;
    });
  }
  return (
    <>
      <div className={`scroll-x ${b.scroll ? "scroll-y" : ""}`}>
        <table className="data">
          <thead>
            <tr>{cols.map((c, i) => (
              <th key={i} scope="col" className={`${c.align === "right" ? "r" : ""} ${i === 0 ? "sticky-col" : ""}`}
                  aria-sort={sort?.col === i ? (sort.dir === 1 ? "ascending" : "descending") : undefined} title={c.help || undefined}>
                <button type="button" className="th" onClick={() => setSort((s) => (s?.col === i ? (s.dir === 1 ? { col: i, dir: -1 } : null) : { col: i, dir: 1 }))}>
                  {c.name}{sort?.col === i ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
                </button>
              </th>
            ))}</tr>
          </thead>
          <tbody>
            {shown.map((row, r) => (
              <tr key={r}>
                {row.map((cell, i) => {
                  const col = cols[i];
                  const style = cell.heat ? ({ "--a": `${cell.heat[1]}%` } as React.CSSProperties) : undefined;
                  const body: ReactNode = cell.s ? <Spark values={cell.s} />
                    : col.bar && cell.b != null ? (
                      <span className="bar"><span className="bar-fill" style={{ width: `${Math.max(0, Math.min(100, ((cell.b - col.bar[0]) / (col.bar[1] - col.bar[0])) * 100))}%` }} /><span className="bar-t">{cell.t}</span></span>
                    ) : cell.t;
                  return i === 0
                    ? <th key={i} scope="row" className={cellClass(col, cell, true)} style={style}>{body}</th>
                    : <td key={i} className={cellClass(col, cell, false)} style={style}>{body}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {b.clipped && <p className="caption">Showing the first {b.clipped[0].toLocaleString()} of {b.clipped[1].toLocaleString()} rows. The download has all of them.</p>}
    </>
  );
}

// An editable table. Edits are sent when a cell loses focus; pasting a
// column of values from a spreadsheet fills downward from the cell.
function Editor({ b }: { b: Block }) {
  const { change } = useLive();
  const cols = b.columns as Block[];
  const [rows, setRows] = useState<unknown[][]>(() => (b.rows as Cell[][]).map((r) => r.map((c) => c.v ?? null)));
  useEffect(() => setRows((b.rows as Cell[][]).map((r) => r.map((c) => c.v ?? null))), [b.rows]);
  const send = (next: unknown[][]) => { setRows(next); change(b.id, next); };
  const coerce = (col: Block, raw: string): unknown => {
    if (col.edit?.type === "number") { const n = Number(raw.replace(/[,\s]/g, "")); return raw.trim() === "" || !Number.isFinite(n) ? null : n; }
    return raw === "" ? null : raw;
  };
  const setCell = (r: number, c: number, raw: string, commit: boolean) => {
    const next = rows.map((row) => [...row]);
    next[r][c] = commit ? coerce(cols[c], raw) : raw;
    if (commit) send(next); else setRows(next);
  };
  const paste = (r: number, c: number, text: string) => {
    const lines = text.replace(/\r/g, "").split("\n").filter((l, i, a) => l !== "" || i < a.length - 1);
    if (lines.length < 2 && !text.includes("\t")) return false;
    const next = rows.map((row) => [...row]);
    lines.forEach((line, i) => line.split("\t").forEach((raw, j) => {
      if (next[r + i] && cols[c + j]?.edit) next[r + i][c + j] = coerce(cols[c + j], raw);
    }));
    send(next);
    return true;
  };
  return (
    <div className="scroll-x">
      <table className="data editor">
        <thead><tr>{cols.map((c, i) => <th key={i} scope="col" className={c.align === "right" ? "r" : ""}>{c.name}</th>)}{b.dynamic && <th />}</tr></thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((v, c) => {
                const col = cols[c];
                if (!col.edit) return <td key={c} className={col.align === "right" ? "r num" : ""}>{(b.rows as Cell[][])[r]?.[c]?.t ?? String(v ?? "")}</td>;
                if (col.edit.type === "select") {
                  return (
                    <td key={c}><select aria-label={col.name} value={String(v ?? "")} onChange={(e) => setCell(r, c, e.target.value, true)}>
                      <option value="" />{(col.edit.options as string[]).map((o) => <option key={o} value={o}>{o}</option>)}
                    </select></td>
                  );
                }
                return (
                  <td key={c}><input aria-label={`${col.name}, row ${r + 1}`} className={col.edit.type === "number" ? "num r" : ""}
                    inputMode={col.edit.type === "number" ? "decimal" : undefined} value={v == null ? "" : String(v)}
                    onChange={(e) => setCell(r, c, e.target.value, false)}
                    onBlur={(e) => setCell(r, c, e.target.value, true)}
                    onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                    onPaste={(e) => { if (paste(r, c, e.clipboardData.getData("text"))) e.preventDefault(); }} /></td>
                );
              })}
              {b.dynamic && <td><button type="button" className="link" onClick={() => send(rows.filter((_, i) => i !== r))}>Remove</button></td>}
            </tr>
          ))}
        </tbody>
      </table>
      {b.dynamic && <button type="button" className="secondary add-row" onClick={() => send([...rows, cols.map(() => null)])}>Add a row</button>}
    </div>
  );
}

function Panel({ b, side }: { b: Block; side?: boolean }) {
  const [open, setOpen] = useState<boolean>(b.open);
  const id = useId();
  return (
    <section className={side ? "group" : "panel"}>
      <button type="button" className={side ? "group-head" : "panel-head"} aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
        <span className="chev" aria-hidden="true" />{b.label}
      </button>
      {open && <div className={side ? "group-body" : "panel-body"} id={id}><Blocks blocks={b.blocks} side={side} /></div>}
    </section>
  );
}

function Tabs({ b }: { b: Block }) {
  const { openTab } = useLive();
  const labels = b.labels as string[];
  return (
    <>
      <div className="tabs" role="tablist" aria-label="Views"
           onKeyDown={(e) => {
             const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
             if (!d) return;
             const next = labels[(labels.indexOf(b.active) + d + labels.length) % labels.length];
             openTab(next);
             (e.currentTarget.querySelector(`[data-tab="${CSS.escape(next)}"]`) as HTMLElement | null)?.focus();
           }}>
        {labels.map((l) => (
          <button key={l} type="button" role="tab" data-tab={l} aria-selected={l === b.active}
                  tabIndex={l === b.active ? 0 : -1} onClick={() => openTab(l)}>{l}</button>
        ))}
      </div>
      <div className="screen" role="tabpanel" aria-label={b.active} key={b.active}>
        <Blocks blocks={b.blocks} />
      </div>
    </>
  );
}

function Frame({ b }: { b: Block }) {
  return <iframe className="frame" title="Price chart" sandbox="allow-scripts" srcDoc={b.html} style={{ height: b.height }} />;
}

function One({ b, side }: { b: Block; side?: boolean }) {
  switch (b.kind) {
    case "title": return <h1 className="page-title">{b.sub || b.text}</h1>;
    case "context":
      return (
        <div className="runline">
          <span className="run-lead">{b.lead}</span>
          <ul className="context">{(b.items as string[]).map((h, i) => <li key={i} dangerouslySetInnerHTML={{ __html: h }} />)}</ul>
        </div>
      );
    case "heading": return side ? <h2 className="side-h">{b.text}</h2> : <h3 className="h">{b.text}</h3>;
    case "kpis": return <KpiStrip items={b.items} />;
    case "note": case "flag": return <Note block={b as never} />;
    case "html": return <Html html={b.html} className={`prose ${b.cls ? `x-${b.cls}` : ""}`} />;
    case "caption": return <Html html={b.html} className="caption" />;
    case "code": return <pre className="code"><code>{b.text}</code></pre>;
    case "alert": return <Html html={b.html} className={`alert alert-${b.level}`} />;
    case "table": return <Table b={b} />;
    case "editor": return <Editor b={b} />;
    case "chart": return <Figure b={b} />;
    case "image": return <img className="figure-img" src={b.src} alt="" />;
    case "frame": return <Frame b={b} />;
    case "download": return <a className="button secondary download" href={`/api/download/${encodeURIComponent(b.id)}`} download={b.file_name}>{b.label}</a>;
    case "upload": return <Upload b={b} />;
    case "control": return <Control b={b} side={side} />;
    case "panel": return <Panel b={b} side={side} />;
    case "form": return <div className="form"><Blocks blocks={b.blocks} side={side} /></div>;
    case "chat": return <div className={`chat chat-${b.role}`}><Blocks blocks={b.blocks} /></div>;
    case "tabs": return <Tabs b={b} />;
    case "results": return <Results blocks={b.blocks} />;
    case "loading": return <div className="skeleton" role="status"><p>Loading this view…</p><div className="sk sk-chart" /><div className="sk sk-row" /></div>;
    case "row": {
      const cols = (b.cols as Block[][]);
      const allKpi = cols.every((c) => c.length > 0 && c.every((x) => x.kind === "kpis"));
      if (allKpi) return <KpiStrip items={cols.flatMap((c) => c.flatMap((x) => x.items))} />;
      const live = cols.filter((c) => c.length > 0);
      const w = (b.weights as number[]).filter((_, i) => cols[i].length > 0);
      return (
        <div className={`row ${side ? "row-side" : ""}`} style={{ gridTemplateColumns: w.map((x) => `minmax(0, ${x}fr)`).join(" ") }}>
          {live.map((c, i) => <div key={i} className="col"><Blocks blocks={c} side={side} /></div>)}
        </div>
      );
    }
    default: return null;
  }
}

// Consecutive KPI blocks become one strip.
export function Blocks({ blocks, side }: { blocks: Block[]; side?: boolean }) {
  const out: ReactNode[] = [];
  let run: Block[] = [];
  const flush = (key: string) => {
    if (run.length) { out.push(<KpiStrip key={key} items={run.flatMap((x) => x.items)} />); run = []; }
  };
  blocks.forEach((b, i) => {
    if (b.kind === "kpis") { run.push(b); return; }
    flush(`k${i}`);
    out.push(<One key={`${i}:${b.kind}:${b.id ?? ""}`} b={b} side={side} />);
  });
  flush("k-end");
  return <>{out}</>;
}
