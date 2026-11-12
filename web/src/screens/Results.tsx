// Backtest > Results: the reference screen.
//
// The content comes from the server as blocks, in the order the Streamlit
// tab showed them (server/views/results.py). This file only arranges them:
// which blocks sit side by side on a wide screen. Every other screen can
// start as `<BlockList blocks={...} />` and get an arrangement like this one
// only where the default single column wastes space.
import type { Block, ChartBlock, KpiBlock, TableBlock } from "../api";
import { BlockList, Chart, DataTable, KpiStrip, Note, One } from "../blocks/Blocks";
import { TimePanel } from "../charts/TimePanel";

const chart = (blocks: Block[], kind: string) =>
  blocks.find((b): b is ChartBlock => b.kind === "chart" && b.chart === kind);

// The blocks between one heading and the next.
function section(blocks: Block[], title: string): Block[] {
  const start = blocks.findIndex((b) => b.kind === "heading" && b.text === title);
  if (start < 0) return [];
  const rest = blocks.slice(start + 1);
  const end = rest.findIndex((b) => b.kind === "heading");
  return end < 0 ? rest : rest.slice(0, end);
}

function Section({ title, blocks, className }: { title: string; blocks: Block[]; className?: string }) {
  if (!blocks.length) return null;
  return (
    <section className={`section ${className ?? ""}`}>
      <h3 className="h">{title}</h3>
      {blocks.map((b, i) =>
        b.kind === "table" ? <DataTable key={i} block={b as TableBlock} caption={title} /> : <One key={i} block={b} />)}
    </section>
  );
}

export function Results({ blocks }: { blocks: Block[] }) {
  const firstHeading = blocks.findIndex((b) => b.kind === "heading");
  const lead = firstHeading < 0 ? blocks : blocks.slice(0, firstHeading);
  const notes = lead.filter((b) => b.kind === "note" || b.kind === "flag");
  const kpis = lead.filter((b): b is KpiBlock => b.kind === "kpi");
  const equity = chart(lead, "equity");
  const drawdown = chart(lead, "drawdown");
  const others = lead.filter((b): b is ChartBlock => b.kind === "chart" && b !== equity && b !== drawdown);

  const stats = section(blocks, "Full statistics");
  const statsTable = stats.filter((b) => b.kind === "table");
  const statsRest = stats.filter((b) => b.kind !== "table");

  // Anything this layout does not place by name still appears, in order.
  const placed = new Set(["Full statistics", "Trailing periods", "Calendar years", "Main drawdown episodes"]);
  const extra: Block[] = [];
  let keep = false;
  blocks.slice(Math.max(firstHeading, 0)).forEach((b) => {
    if (b.kind === "heading") keep = !placed.has(b.text);
    if (keep) extra.push(b);
  });

  return (
    <div className="results">
      {kpis.length > 0 && <KpiStrip items={kpis} />}
      {notes.length > 0 && (
        <div className="lead-notes">{notes.map((b, i) => <Note key={i} block={b as never} />)}</div>
      )}
      {equity && <TimePanel value={equity} drawdown={drawdown} />}
      <div className="trio">{others.map((b, i) => <Chart key={i} block={b} />)}</div>
      <div className="pair pair-wide-left">
        <div className="stack">
          <Section title="Trailing periods" blocks={section(blocks, "Trailing periods")} />
          <Section title="Main drawdown episodes" blocks={section(blocks, "Main drawdown episodes")} />
        </div>
        <Section title="Full statistics" blocks={statsTable} />
      </div>
      <Section title="Calendar years" blocks={section(blocks, "Calendar years")} className="calendar" />
      <BlockList blocks={statsRest} />
      <BlockList blocks={extra} />
    </div>
  );
}
