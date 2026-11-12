// The contract with server/app.py. The server decides every figure and its
// formatting; these types describe what it sends, nothing is recomputed here.

export type Tone = "pos" | "neg" | "dim" | "";

export interface ParamMeta {
  key: string; label: string;
  kind: "int" | "float" | "bool" | "choice" | "series" | "formula" | string;
  min: number | null; max: number | null; step: number | null;
  default: unknown; choices: string[]; help: string;
}
export interface ModelMeta {
  key: string; label: string; description: string;
  needs_exog: boolean; needs_cash: boolean; params: ParamMeta[];
}
export interface PresetMeta {
  name: string; tickers: string[]; benchmark: string | null; cash: string | null; note: string;
}
export interface Meta {
  models: ModelMeta[];
  presets: PresetMeta[];
  rebalance: Record<string, string>;
  bench_modes: string[];
  bench_special: { blend: string; strategy: string };
  file_conventions: Record<string, string>;
  defaults: Spec;
  tabs: { name: string; screen: string | null }[];
  legacy_url: string;
}

// One field per control of the Backtest panel (server/pipeline.py RunSpec).
export interface Spec {
  source: "yfinance" | "upload";
  tickers: string[]; start: string | null; end: string | null;
  universe: string[] | null; preset: string;
  benchmark: string | null; bench_mode: string;
  adjusted: boolean; file_convention: string; cash_proxy: string | null;
  exog_lag: number; mode: "builtin" | "external_weights";
  weights_normalize: string; weights_calendar: string;
  strategy: string; params: Record<string, unknown>;
  rebalance: string; execution_lag: number; execute_at_open: boolean;
  trim_warmup: boolean; initial_capital: number;
  commission_bps: number; slippage_bps: number; impact_model: string;
  cash_rate_pa: number; management_fee_pa: number;
  [other: string]: unknown;
}

export interface FileIds { prices: string[]; exog: string[]; weights: string | null }

export interface PriceUpload {
  ids: string[]; names: string[]; instruments: string[]; fields: string[];
  has_open: boolean; has_volume: boolean; sources: number; warnings: string[];
  convention: string | null; error: string | null;
}
export interface ExogUpload {
  ids: string[]; names: string[]; columns: string[];
  start: string | null; end: string | null; error: string | null;
}
export interface WeightsUpload { ids: string[]; names: string[]; columns: string[]; rows: number; error: string | null }

export interface KpiBlock { kind: "kpi"; label: string; value: string; sub: string; tone: Tone }
export interface NoteBlock { kind: "note" | "flag"; html: string }
export interface HeadingBlock { kind: "heading"; text: string }
export interface CaptionBlock { kind: "caption"; text: string }
export interface Column { name: string; align: "left" | "right"; signed: boolean; strong: boolean }
export type Cell = [string] | [string, Tone];
export interface TableBlock { kind: "table"; key: string; columns: Column[]; rows: Cell[][] }
export interface Series { name: string; y: (number | null)[] }
export interface ChartBlock {
  kind: "chart"; chart: string; title: string; suffix?: string; ref?: number;
  log?: boolean; log_toggle?: boolean; log_help?: string;
  x?: string[]; series?: Series[];
  years?: string[]; z?: (number | null)[][];
  counts?: number[]; edges?: number[]; var95?: number; n?: number;
  labels?: string[]; values?: (number | null)[];
}
export interface PanelBlock { kind: "panel"; label: string; open: boolean; blocks: Block[] }
export type Block = KpiBlock | NoteBlock | HeadingBlock | CaptionBlock | TableBlock | ChartBlock | PanelBlock;

export interface RunSummary {
  id: string; cached: boolean; seconds: number; stamp: string; label: string;
  context: string[]; kpis: KpiBlock[]; notices: string[]; screens: string[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(0, "The server did not answer. Check that it is running, then try again.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, String(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => call<Meta>("/meta"),
  upload: <T>(kind: "prices" | "exog" | "weights", files: File[]) => {
    const body = new FormData();
    files.forEach((f) => body.append("files", f, f.name));
    return call<T>(`/uploads/${kind}`, { method: "POST", body });
  },
  run: (spec: Spec, files: FileIds) =>
    call<RunSummary>("/runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, files }),
    }),
  getRun: (id: string) => call<RunSummary>(`/runs/${id}`),
  screen: (id: string, name: string) => call<{ blocks: Block[] }>(`/runs/${id}/screens/${name}`),
};
