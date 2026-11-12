// Session state of the Backtest workspace: the controls, the uploaded
// files, and the run on screen. Changing a control never touches the run;
// only "Run backtest" does, exactly as in the Streamlit app.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api, ApiError } from "./api";
import type { ExogUpload, FileIds, Meta, PriceUpload, RunSummary, Spec, WeightsUpload } from "./api";

const KEY = "qbt.session.v1";

export interface Uploads { prices: PriceUpload | null; exog: ExogUpload | null; weights: WeightsUpload | null }

interface Saved { spec: Spec; uploads: Uploads; runId: string | null; runKey: string | null }

export interface Studio {
  meta: Meta;
  spec: Spec;
  set: (patch: Partial<Spec>) => void;
  setParam: (key: string, value: unknown) => void;
  uploads: Uploads;
  upload: (kind: "prices" | "exog" | "weights", files: File[]) => Promise<void>;
  clearUpload: (kind: "prices" | "exog" | "weights") => void;
  uploading: string | null;
  symbols: string[];          // what the universe and benchmark can be chosen from
  run: RunSummary | null;
  running: boolean;
  runError: string | null;
  elapsed: number;            // seconds since Run was pressed, while running
  stale: boolean;             // controls differ from the run on screen
  start: () => void;
  reset: () => void;
}

const Ctx = createContext<Studio | null>(null);
export const useStudio = () => {
  const v = useContext(Ctx);
  if (!v) throw new Error("useStudio outside StudioProvider");
  return v;
};

function load(): Saved | null {
  try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch { return null; }
}

const fileIds = (u: Uploads): FileIds => ({
  prices: u.prices?.ids ?? [], exog: u.exog?.ids ?? [], weights: u.weights?.ids[0] ?? null,
});
const keyOf = (spec: Spec, u: Uploads) => JSON.stringify([spec, fileIds(u)]);

export function StudioProvider({ meta, children }: { meta: Meta; children: ReactNode }) {
  const saved = useRef(load()).current;
  const [spec, setSpec] = useState<Spec>(() => ({ ...meta.defaults, ...(saved?.spec ?? {}) }));
  const [uploads, setUploads] = useState<Uploads>(saved?.uploads ?? { prices: null, exog: null, weights: null });
  const [run, setRun] = useState<RunSummary | null>(null);
  const [runKey, setRunKey] = useState<string | null>(saved?.runKey ?? null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  // Bring back the run that was on screen, if the server still holds it.
  useEffect(() => {
    if (!saved?.runId) return;
    api.getRun(saved.runId).then(setRun).catch(() => setRunKey(null));
  }, [saved]);

  useEffect(() => {
    const data: Saved = { spec, uploads, runId: run?.id ?? null, runKey };
    try { localStorage.setItem(KEY, JSON.stringify(data)); } catch { /* storage full or blocked */ }
  }, [spec, uploads, run, runKey]);

  useEffect(() => {
    if (!running) return;
    const t0 = performance.now();
    const id = window.setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    return () => window.clearInterval(id);
  }, [running]);

  const set = useCallback((patch: Partial<Spec>) => setSpec((s) => ({ ...s, ...patch })), []);
  const setParam = useCallback((k: string, v: unknown) =>
    setSpec((s) => ({ ...s, params: { ...s.params, [k]: v } })), []);

  const upload = useCallback(async (kind: "prices" | "exog" | "weights", files: File[]) => {
    if (!files.length) return;
    setUploading(kind);
    try {
      const res = await api.upload<PriceUpload & ExogUpload & WeightsUpload>(kind, files);
      setUploads((u) => ({ ...u, [kind]: res }));
      if (kind === "prices") setSpec((s) => ({ ...s, universe: null }));
    } catch (e) {
      const message = e instanceof ApiError ? e.message : "The upload failed.";
      setUploads((u) => ({ ...u, [kind]: { ids: [], names: files.map((f) => f.name), error: message } as never }));
    } finally {
      setUploading(null);
    }
  }, []);

  const clearUpload = useCallback((kind: "prices" | "exog" | "weights") =>
    setUploads((u) => ({ ...u, [kind]: null })), []);

  const symbols = useMemo(
    () => (spec.source === "upload" ? uploads.prices?.instruments ?? []
      : Array.from(new Set(spec.tickers.map((t) => t.trim()).filter(Boolean)))),
    [spec.source, spec.tickers, uploads.prices]);

  const start = useCallback(() => {
    if (running) return;
    setRunning(true); setRunError(null); setElapsed(0);
    const key = keyOf(spec, uploads);
    api.run(spec, fileIds(uploads))
      .then((r) => { setRun(r); setRunKey(key); })
      .catch((e: unknown) => setRunError(e instanceof Error ? e.message : "The backtest stopped."))
      .finally(() => setRunning(false));
  }, [running, spec, uploads]);

  const reset = useCallback(() => {
    setSpec(meta.defaults); setUploads({ prices: null, exog: null, weights: null });
  }, [meta]);

  const stale = !!run && runKey !== null && runKey !== keyOf(spec, uploads);

  const value: Studio = {
    meta, spec, set, setParam, uploads, upload, clearUpload, uploading, symbols,
    run, running, runError, elapsed, stale, start, reset,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
