// The connection to the server. The server holds the session (controls,
// uploads, the run) and answers every change with the page to draw: the
// control panel, the header, and the tab on screen. This module queues what
// the user changed, sends it, and keeps the last page.
//
// Changes made while a request is in flight are sent together afterwards,
// so fast typing or several quick clicks never race each other.
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

export type Tone = "pos" | "neg" | "dim" | "";
// Blocks are described in server/present.py. They are deliberately loose
// here: a block is its `kind` plus what that kind carries.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Block = { kind: string } & Record<string, any>;
export interface Page { sidebar: Block[]; main: Block[]; seconds: number; legacy_url: string }

interface Pending { events: Map<string, unknown>; clicks: Set<string>; submits: Map<string, string>; tab?: string }

export interface Live {
  page: Page | null;
  busy: boolean;
  busyFor: number;
  error: string | null;
  change: (id: string, value: unknown) => void;
  click: (id: string) => void;
  submit: (id: string, text: string) => void;
  upload: (id: string, files: File[]) => Promise<void>;
  clearUpload: (id: string) => void;
  openTab: (name: string) => void;
  retry: () => void;
}

const Ctx = createContext<Live | null>(null);
export const useLive = () => {
  const v = useContext(Ctx);
  if (!v) throw new Error("useLive outside LiveProvider");
  return v;
};

const fresh = (): Pending => ({ events: new Map(), clicks: new Set(), submits: new Map() });

function tabsOf(main: Block[]): Block | undefined {
  return main.find((b) => b.kind === "tabs");
}

export function LiveProvider({ children }: { children: ReactNode }) {
  const [page, setPage] = useState<Page | null>(null);
  const [busy, setBusy] = useState(false);
  const [busyFor, setBusyFor] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<Pending>(fresh());
  const flying = useRef(false);
  const tab = useRef<string | undefined>(undefined);
  // Tabs already drawn for the current state: going back to one is instant.
  const seen = useRef<Map<string, Block[]>>(new Map());

  const flush = useCallback(async () => {
    if (flying.current) return;
    flying.current = true;
    setBusy(true);
    while (true) {
      const p = pending.current;
      pending.current = fresh();
      const changed = p.events.size > 0 || p.clicks.size > 0 || p.submits.size > 0;
      if (changed) seen.current.clear();
      try {
        const res = await fetch("/api/render", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tab: tab.current,
            events: Array.from(p.events, ([id, value]) => ({ id, value })),
            clicks: Array.from(p.clicks), submits: Object.fromEntries(p.submits),
          }),
        });
        if (!res.ok) {
          let detail = res.statusText;
          try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
          throw new Error(String(detail));
        }
        const next = (await res.json()) as Page;
        const t = tabsOf(next.main);
        if (t) { tab.current = t.active; seen.current.set(t.active, t.blocks); }
        setPage(next);
        setError(null);
      } catch (e) {
        setError(e instanceof TypeError
          ? "The server did not answer. Check that it is running, then try again."
          : e instanceof Error ? e.message : "Something went wrong.");
      }
      const q = pending.current;
      if (!(q.events.size || q.clicks.size || q.submits.size || q.tab !== undefined)) break;
      if (q.tab !== undefined) tab.current = q.tab;
    }
    flying.current = false;
    setBusy(false);
  }, []);

  useEffect(() => { void flush(); }, [flush]);

  useEffect(() => {
    if (!busy) { setBusyFor(0); return; }
    const t0 = performance.now();
    const id = window.setInterval(() => setBusyFor((performance.now() - t0) / 1000), 100);
    return () => window.clearInterval(id);
  }, [busy]);

  const change = useCallback((id: string, value: unknown) => { pending.current.events.set(id, value); void flush(); }, [flush]);
  const click = useCallback((id: string) => { pending.current.clicks.add(id); void flush(); }, [flush]);
  const submit = useCallback((id: string, text: string) => { pending.current.submits.set(id, text); void flush(); }, [flush]);

  const openTab = useCallback((name: string) => {
    const cached = seen.current.get(name);
    tab.current = name;
    setPage((p) => {
      if (!p) return p;
      return { ...p, main: p.main.map((b) => (b.kind === "tabs"
        ? { ...b, active: name, blocks: cached ?? [{ kind: "loading" }] } : b)) };
    });
    if (!cached) { pending.current.tab = name; void flush(); }
  }, [flush]);

  const upload = useCallback(async (id: string, files: File[]) => {
    if (!files.length) return;
    const body = new FormData();
    files.forEach((f) => body.append("files", f, f.name));
    setBusy(true);
    try {
      const res = await fetch(`/api/files/${encodeURIComponent(id)}`, { method: "POST", body });
      if (!res.ok) throw new Error("The file could not be sent.");
      seen.current.clear();
    } catch (e) {
      setError(e instanceof Error ? e.message : "The file could not be sent.");
    }
    pending.current.tab = tab.current;
    void flush();
  }, [flush]);

  const clearUpload = useCallback((id: string) => {
    void fetch(`/api/files/${encodeURIComponent(id)}`, { method: "DELETE" }).then(() => {
      seen.current.clear(); pending.current.tab = tab.current; void flush();
    });
  }, [flush]);

  const retry = useCallback(() => { pending.current.tab = tab.current; void flush(); }, [flush]);

  return (
    <Ctx.Provider value={{ page, busy, busyFor, error, change, click, submit, upload, clearUpload, openTab, retry }}>
      {children}
    </Ctx.Provider>
  );
}
