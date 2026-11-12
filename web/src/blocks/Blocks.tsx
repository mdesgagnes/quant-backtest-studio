// The block vocabulary. A screen is a list of blocks from the server
// (server/views/common.py); these components draw them. A new screen needs
// no new component unless it introduces a new kind of block.
import { useId, useState } from "react";
import type { ReactNode } from "react";
import type { Block, ChartBlock, KpiBlock, NoteBlock, PanelBlock, TableBlock } from "../api";
import { Bars, Distribution, LineChart, MonthlyGrid } from "../charts/Simple";
import { TimePanel } from "../charts/TimePanel";

// The server's notes use <b>, <i> and <code> only. Anything else is shown
// as text, so a note can never inject markup.
function Rich({ html }: { html: string }) {
  const parts = html.split(/(<\/?(?:b|i|code)>|<br\s*\/?>)/g);
  const out: ReactNode[] = [];
  const stack: string[] = [];
  let buf: ReactNode[] = [];
  const frames: ReactNode[][] = [];
  parts.forEach((p, i) => {
    const open = /^<(b|i|code)>$/.exec(p);
    const close = /^<\/(b|i|code)>$/.exec(p);
    if (open) { stack.push(open[1]); frames.push(buf); buf = []; }
    else if (close && stack[stack.length - 1] === close[1]) {
      const Tag = stack.pop() as "b" | "i" | "code";
      const inner = buf; buf = frames.pop() ?? [];
      buf.push(<Tag key={i}>{inner}</Tag>);
    }
    else if (/^<br/.test(p)) buf.push(<br key={i} />);
    else if (p) buf.push(decode(p));
  });
  while (frames.length) { const inner = buf; buf = frames.pop()!; buf.push(...inner); }
  out.push(...buf);
  return <>{out}</>;
}

const ENT: Record<string, string> = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&rarr;": "→", "&nbsp;": " ", "&middot;": "·" };
const decode = (s: string) => s.replace(/&(amp|lt|gt|rarr|nbsp|middot);/g, (m) => ENT[m]);

// A long note shows its first sentence; the rest is one click away
// (the rule app.py applies in `note`).
export function Note({ block }: { block: NoteBlock }) {
  const [open, setOpen] = useState(false);
  const text = block.html;
  const cut = text.indexOf(". ");
  const head = cut >= 0 ? text.slice(0, cut) : text;
  const rest = cut >= 0 ? text.slice(cut + 2) : "";
  const balanced = (head.match(/<b>/g)?.length ?? 0) === (head.match(/<\/b>/g)?.length ?? 0);
  const foldable = block.kind === "note" && cut >= 0 && text.length > 170 && head.length <= 160 && rest.trim() && balanced;
  if (!foldable) return <p className={block.kind}><Rich html={text} /></p>;
  return (
    <p className="note">
      <Rich html={`${head}.`} />{" "}
      {open ? <Rich html={rest} /> : (
        <button type="button" className="link" onClick={() => setOpen(true)} aria-expanded={false}>More</button>
      )}
    </p>
  );
}

export function KpiStrip({ items }: { items: KpiBlock[] }) {
  return (
    <dl className="kpis">
      {items.map((k) => (
        <div className="kpi" key={k.label}>
          <dt>{k.label}</dt>
          <dd className={`num kpi-v ${k.tone}`}>{k.value}</dd>
          {k.sub && <dd className="num kpi-sub">{k.sub}</dd>}
        </div>
      ))}
    </dl>
  );
}

export function DataTable({ block, caption }: { block: TableBlock; caption?: string }) {
  if (!block.rows.length) return <p className="empty-line">Nothing to show for this run.</p>;
  return (
    <div className="scroll-x">
      <table className="data">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>{block.columns.map((c, i) => (
            <th key={c.name} scope="col" className={`${c.align === "right" ? "r" : ""} ${i === 0 ? "sticky-col" : ""}`}>{c.name}</th>
          ))}</tr>
        </thead>
        <tbody>
          {block.rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, i) => {
                const c = block.columns[i];
                const cls = `${c.align === "right" ? "r num" : ""} ${cell[1] ?? ""} ${c.strong ? "strong" : ""} ${i === 0 ? "sticky-col" : ""}`;
                return i === 0 ? <th key={i} scope="row" className={cls}>{cell[0]}</th> : <td key={i} className={cls}>{cell[0]}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Panel({ block }: { block: PanelBlock }) {
  const [open, setOpen] = useState(block.open);
  const id = useId();
  return (
    <section className="panel">
      <button type="button" className="panel-head" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
        <span className="chev" aria-hidden="true" />{block.label}
      </button>
      {open && <div className="panel-body" id={id}><BlockList blocks={block.blocks} /></div>}
    </section>
  );
}

export function Chart({ block }: { block: ChartBlock }) {
  switch (block.chart) {
    case "equity": case "drawdown": return <TimePanel value={block} />;
    case "line": return <LineChart block={block} />;
    case "bars": return <Bars block={block} />;
    case "distribution": return <Distribution block={block} />;
    case "monthly_heatmap": return <MonthlyGrid block={block} />;
    default: return <p className="flag">This chart type ({block.chart}) is not drawn yet.</p>;
  }
}

// Draws any list of blocks in order. Consecutive KPIs become one strip.
export function BlockList({ blocks }: { blocks: Block[] }) {
  const out: ReactNode[] = [];
  let kpis: KpiBlock[] = [];
  const flush = (key: string) => { if (kpis.length) { out.push(<KpiStrip key={key} items={kpis} />); kpis = []; } };
  blocks.forEach((b, i) => {
    if (b.kind === "kpi") { kpis.push(b); return; }
    flush(`k${i}`);
    out.push(<One key={i} block={b} />);
  });
  flush("k-end");
  return <>{out}</>;
}

export function One({ block }: { block: Block }) {
  switch (block.kind) {
    case "note": case "flag": return <Note block={block} />;
    case "heading": return <h3 className="h">{block.text}</h3>;
    case "caption": return <p className="caption">{block.text}</p>;
    case "table": return <DataTable block={block} />;
    case "chart": return <Chart block={block} />;
    case "panel": return <Panel block={block} />;
    case "kpi": return <KpiStrip items={[block]} />;
  }
}
