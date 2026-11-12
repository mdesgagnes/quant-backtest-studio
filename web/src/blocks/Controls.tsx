// One component per kind of control the server can ask for. Each shows the
// server's value, lets the user change it, and reports the change; the
// server answers with the page to draw. Text and numbers are reported when
// the user leaves the field or presses Enter, not on every keystroke.
import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useLive } from "../live";
import type { Block } from "../live";

function Labelled({ b, children, htmlFor }: { b: Block; children: ReactNode; htmlFor?: string }) {
  const [show, setShow] = useState(false);
  return (
    <div className="field">
      <div className={`field-label ${b.hide_label ? "sr-only" : ""}`}>
        <label htmlFor={htmlFor}>{b.label}</label>
        {b.help && !b.hide_label && (
          <button type="button" className="hint" aria-expanded={show} aria-label={`About ${b.label}`}
                  onClick={() => setShow(!show)}>?</button>
        )}
      </div>
      {children}
      {b.help && show && <p className="field-help">{b.help}</p>}
    </div>
  );
}

// Local text that follows the server's value unless the user is editing.
function useDraft(server: string): [string, (v: string) => void, () => boolean] {
  const [text, setText] = useState(server);
  const last = useRef(server);
  useEffect(() => { if (server !== last.current) { last.current = server; setText(server); } }, [server]);
  const dirty = () => { const d = text !== last.current; last.current = text; return d; };
  return [text, setText, dirty];
}

function Select({ b }: { b: Block }) {
  const { change } = useLive();
  const id = useId();
  return (
    <Labelled b={b} htmlFor={id}>
      <select id={id} value={b.value ?? ""} onChange={(e) => change(b.id, Number(e.target.value))}>
        {b.value == null && <option value="">Choose…</option>}
        {(b.options as string[]).map((o, i) => <option key={i} value={i}>{o}</option>)}
      </select>
    </Labelled>
  );
}

function Segments({ b }: { b: Block }) {
  const { change } = useLive();
  const opts = b.options as string[];
  return (
    <Labelled b={b}>
      <div className={`seg ${b.style === "pills" ? "seg-wrap" : "seg-fill"}`} role="radiogroup" aria-label={b.label}>
        {opts.map((o, i) => (
          <button key={i} type="button" role="radio" aria-checked={b.value === i}
                  onClick={() => change(b.id, b.value === i && b.clearable ? null : i)}>{o}</button>
        ))}
      </div>
    </Labelled>
  );
}

function Multi({ b }: { b: Block }) {
  const { change } = useLive();
  const opts = b.options as string[];
  const value = b.value as number[];
  const max = b.max_selections as number | undefined;
  const toggle = (i: number) => {
    const next = value.includes(i) ? value.filter((x) => x !== i) : [...value, i];
    if (max && next.length > max) return;
    change(b.id, next);
  };
  if (opts.length <= 16) {
    return (
      <Labelled b={b}>
        <div className="chips" role="group" aria-label={b.label}>
          {opts.map((o, i) => (
            <button key={i} type="button" className="chip" aria-pressed={value.includes(i)} onClick={() => toggle(i)}>{o}</button>
          ))}
          {!opts.length && <span className="empty-line">Nothing to choose from yet.</span>}
        </div>
      </Labelled>
    );
  }
  return (
    <Labelled b={b}>
      <details className="picker">
        <summary>{value.length ? `${value.length} of ${opts.length} selected` : (b.placeholder || "Choose…")}</summary>
        <div className="picker-list">
          {opts.map((o, i) => (
            <label key={i} className="check"><input type="checkbox" checked={value.includes(i)} onChange={() => toggle(i)} /><span>{o}</span></label>
          ))}
        </div>
      </details>
    </Labelled>
  );
}

function NumberField({ b }: { b: Block }) {
  const { change } = useLive();
  const id = useId();
  const [text, setText, dirty] = useDraft(String(b.value));
  const commit = (raw: string) => {
    let v = Number(raw.replace(",", "."));
    if (!Number.isFinite(v)) { setText(String(b.value)); return; }
    if (b.is_int) v = Math.round(v);
    if (b.min != null) v = Math.max(b.min, v);
    if (b.max != null) v = Math.min(b.max, v);
    setText(String(v));
    if (dirty() || v !== b.value) change(b.id, v);
  };
  const nudge = (dir: number) => {
    const step = Number(b.step) || 1;
    const v = Math.round((Number(text) + dir * step) * 1e6) / 1e6;
    setText(String(v)); commit(String(v));
  };
  return (
    <Labelled b={b} htmlFor={id}>
      <span className="input-wrap">
        <input id={id} inputMode="decimal" className="num r" value={text}
               onChange={(e) => setText(e.target.value)} onBlur={() => commit(text)}
               onKeyDown={(e) => {
                 if (e.key === "Enter") { e.preventDefault(); commit(text); }
                 if (e.key === "ArrowUp") { e.preventDefault(); nudge(1); }
                 if (e.key === "ArrowDown") { e.preventDefault(); nudge(-1); }
               }} />
        <span className="stepper" aria-hidden="true">
          <button type="button" tabIndex={-1} onClick={() => nudge(-1)}>−</button>
          <button type="button" tabIndex={-1} onClick={() => nudge(1)}>+</button>
        </span>
      </span>
    </Labelled>
  );
}

function Slider({ b }: { b: Block }) {
  const { change } = useLive();
  const id = useId();
  const [v, setV] = useState<number>(b.value);
  useEffect(() => setV(b.value), [b.value]);
  const send = () => { if (v !== b.value) change(b.id, v); };
  return (
    <Labelled b={b} htmlFor={id}>
      <span className="input-wrap">
        <input id={id} type="range" min={b.min} max={b.max} step={b.step} value={v}
               onChange={(e) => setV(Number(e.target.value))} onPointerUp={send} onKeyUp={send} onBlur={send} />
        <output className="num slider-v">{v}</output>
      </span>
    </Labelled>
  );
}

function Text({ b }: { b: Block }) {
  const { change } = useLive();
  const id = useId();
  const [text, setText, dirty] = useDraft(String(b.value ?? ""));
  const commit = () => { if (dirty()) change(b.id, text); };
  const area = b.control === "text_area";
  return (
    <Labelled b={b} htmlFor={id}>
      {area
        ? <textarea id={id} rows={Math.max(2, Math.round((b.height ?? 80) / 22))} spellCheck={false} value={text}
                    placeholder={b.placeholder ?? ""} onChange={(e) => setText(e.target.value)} onBlur={commit} />
        : <input id={id} type={b.secret ? "password" : "text"} value={text} placeholder={b.placeholder ?? ""}
                 onChange={(e) => setText(e.target.value)} onBlur={commit}
                 onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } }} />}
    </Labelled>
  );
}

function DateField({ b }: { b: Block }) {
  const { change } = useLive();
  const id = useId();
  return (
    <Labelled b={b} htmlFor={id}>
      <input id={id} type="date" value={b.value ?? ""} min={b.min ?? undefined} max={b.max ?? undefined}
             onChange={(e) => { if (e.target.value) change(b.id, e.target.value); }} />
    </Labelled>
  );
}

function Check({ b }: { b: Block }) {
  const { change } = useLive();
  const id = useId();
  return (
    <div className="field">
      <label className={b.control === "toggle" ? "switch" : "check"} htmlFor={id} title={b.help || undefined}>
        <input id={id} type="checkbox" checked={!!b.value} onChange={(e) => change(b.id, e.target.checked)} />
        <span>{b.label}</span>
      </label>
    </div>
  );
}

function Press({ b, primary }: { b: Block; primary?: boolean }) {
  const { click, busy, busyFor } = useLive();
  const mine = useRef(false);
  useEffect(() => { if (!busy) mine.current = false; }, [busy]);
  const main = primary || b.primary;
  return (
    <button type="button" className={main ? "primary" : "secondary"} disabled={b.disabled || (busy && main)}
            title={b.help || undefined} onClick={() => { mine.current = true; click(b.id); }}>
      {busy && mine.current && busyFor > 0.4 ? `Working… ${busyFor.toFixed(1)} s` : b.label}
    </button>
  );
}

function ChatBox({ b }: { b: Block }) {
  const { submit } = useLive();
  const [text, setText] = useState("");
  const send = () => { if (text.trim()) { submit(b.id, text.trim()); setText(""); } };
  return (
    <form className="chatbox" onSubmit={(e) => { e.preventDefault(); send(); }}>
      <input aria-label="Message" value={text} placeholder={b.label} onChange={(e) => setText(e.target.value)} />
      <button type="submit" className="secondary">Send</button>
    </form>
  );
}

export function Control({ b, side }: { b: Block; side?: boolean }) {
  switch (b.control) {
    case "selectbox": case "select_slider": return <Select b={b} />;
    case "radio":
      return (b.options as string[]).length <= 3 && (b.options as string[]).join("").length < 44 ? <Segments b={b} /> : <Select b={b} />;
    case "button_group": return <Segments b={b} />;
    case "multiselect": return <Multi b={b} />;
    case "number_input": return <NumberField b={b} />;
    case "slider": return <Slider b={b} />;
    case "text_input": case "text_area": return <Text b={b} />;
    case "date_input": return <DateField b={b} />;
    case "checkbox": case "toggle": return <Check b={b} />;
    case "button": return <Press b={b} primary={side && /^(Run|Load)\b/.test(b.label)} />;
    case "chat_input": return <ChatBox b={b} />;
    default: return null;
  }
}

export function Upload({ b }: { b: Block }) {
  const { upload, clearUpload } = useLive();
  const id = useId();
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const names = b.names as string[];
  const send = async (files: File[]) => { setBusy(true); await upload(b.id, b.multiple ? files : files.slice(0, 1)); setBusy(false); };
  return (
    <Labelled b={b} htmlFor={id}>
      <div className={`drop ${over ? "over" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); void send(Array.from(e.dataTransfer.files)); }}>
        <input id={id} type="file" className="sr-only" multiple={b.multiple}
               accept={(b.types as string[]).map((t) => `.${t}`).join(",")}
               onChange={(e) => { void send(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
        {busy ? <span>Reading the file…</span> : names.length ? (
          <span className="drop-files">
            <span className="drop-names">{names.join(", ")}</span>
            <button type="button" className="link" onClick={() => clearUpload(b.id)}>Remove</button>
          </span>
        ) : <label htmlFor={id}><span className="link">Choose {b.multiple ? "files" : "a file"}</span> or drop here</label>}
      </div>
    </Labelled>
  );
}
