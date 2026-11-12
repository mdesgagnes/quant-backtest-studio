// Form primitives of the control panel. Each is a native control with a
// visible label, so keyboard, focus and screen readers work without extra
// code. `help` becomes a description the control points at.
import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";

export function Field({ label, help, children, wide }: {
  label: string; help?: string; wide?: boolean;
  children: (ids: { id: string; describedBy?: string }) => ReactNode;
}) {
  const id = useId();
  const [show, setShow] = useState(false);
  return (
    <div className={`field ${wide ? "wide" : ""}`}>
      <div className="field-label">
        <label htmlFor={id}>{label}</label>
        {help && (
          <button type="button" className="hint" aria-expanded={show} aria-label={`About ${label}`}
                  onClick={() => setShow(!show)}>?</button>
        )}
      </div>
      {children({ id, describedBy: help ? `${id}-h` : undefined })}
      {help && <p id={`${id}-h`} className="field-help" hidden={!show}>{help}</p>}
    </div>
  );
}

export function Select<T extends string>({ label, help, value, options, onChange, wide }: {
  label: string; help?: string; wide?: boolean; value: T;
  options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <Field label={label} help={help} wide={wide}>{({ id, describedBy }) => (
      <select id={id} aria-describedby={describedBy} value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    )}</Field>
  );
}

// A number box that keeps what is being typed and commits a valid number
// on blur or Enter, clamped to its range.
export function NumberBox({ label, help, value, min, max, step, onChange, suffix }: {
  label: string; help?: string; value: number; min?: number | null; max?: number | null;
  step?: number | null; onChange: (v: number) => void; suffix?: string;
}) {
  const [text, setText] = useState(String(value));
  const last = useRef(value);
  useEffect(() => { if (value !== last.current) { last.current = value; setText(String(value)); } }, [value]);
  const commit = () => {
    let v = Number(text.replace(",", "."));
    if (!Number.isFinite(v)) { setText(String(value)); return; }
    if (min != null) v = Math.max(min, v);
    if (max != null) v = Math.min(max, v);
    last.current = v; setText(String(v)); if (v !== value) onChange(v);
  };
  return (
    <Field label={label} help={help}>{({ id, describedBy }) => (
      <span className="input-wrap">
        <input id={id} aria-describedby={describedBy} inputMode="decimal" className="num r" value={text}
               onChange={(e) => setText(e.target.value)} onBlur={commit}
               onKeyDown={(e) => {
                 if (e.key === "Enter") commit();
                 if ((e.key === "ArrowUp" || e.key === "ArrowDown") && step) {
                   e.preventDefault();
                   const v = Number(text) + (e.key === "ArrowUp" ? step : -step);
                   setText(String(Math.round(v * 1e6) / 1e6));
                 }
               }} />
        {suffix && <span className="suffix" aria-hidden="true">{suffix}</span>}
      </span>
    )}</Field>
  );
}

export function TextBox({ label, help, value, onChange, type = "text", wide }: {
  label: string; help?: string; value: string; onChange: (v: string) => void; type?: string; wide?: boolean;
}) {
  return (
    <Field label={label} help={help} wide={wide}>{({ id, describedBy }) => (
      <input id={id} type={type} aria-describedby={describedBy} value={value} onChange={(e) => onChange(e.target.value)} />
    )}</Field>
  );
}

export function TextArea({ label, help, value, onChange, rows = 4 }: {
  label: string; help?: string; value: string; onChange: (v: string) => void; rows?: number;
}) {
  return (
    <Field label={label} help={help} wide>{({ id, describedBy }) => (
      <textarea id={id} rows={rows} spellCheck={false} aria-describedby={describedBy} value={value}
                onChange={(e) => onChange(e.target.value)} />
    )}</Field>
  );
}

export function Check({ label, help, checked, onChange }: {
  label: string; help?: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="field wide">
      <label className="check" htmlFor={id} title={help}>
        <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span>{label}</span>
      </label>
    </div>
  );
}

export function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="field wide">
      <div className="seg seg-fill" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button key={o.value} type="button" role="radio" aria-checked={value === o.value}
                  onClick={() => onChange(o.value)}>{o.label}</button>
        ))}
      </div>
    </div>
  );
}

export function FileDrop({ label, accept, multiple, names, busy, error, onFiles, onClear, hint }: {
  label: string; accept: string; multiple?: boolean; names: string[]; busy: boolean;
  error?: string | null; hint?: string; onFiles: (f: File[]) => void; onClear: () => void;
}) {
  const id = useId();
  const [over, setOver] = useState(false);
  return (
    <div className="field wide">
      <div className="field-label"><label htmlFor={id}>{label}</label></div>
      <div className={`drop ${over ? "over" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); onFiles(Array.from(e.dataTransfer.files)); }}>
        <input id={id} type="file" accept={accept} multiple={multiple} className="sr-only"
               onChange={(e) => { onFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
        {busy ? <span className="drop-text">Reading the file…</span>
          : names.length ? (
            <span className="drop-files">
              <span className="drop-names">{names.join(", ")}</span>
              <button type="button" className="link" onClick={onClear}>Remove</button>
            </span>
          ) : <label htmlFor={id} className="drop-text"><span className="link">Choose a file</span> or drop it here</label>}
      </div>
      {hint && !names.length && !busy && <p className="field-help">{hint}</p>}
      {error && <p className="flag" role="alert">{error}</p>}
    </div>
  );
}

export function Group({ title, children, defaultOpen = true, summary }: {
  title: string; children: ReactNode; defaultOpen?: boolean; summary?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <section className="group">
      <h2>
        <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
          <span className="chev" aria-hidden="true" />
          <span className="group-title">{title}</span>
          {!open && summary && <span className="group-sum">{summary}</span>}
        </button>
      </h2>
      {open && <div className="group-body" id={id}>{children}</div>}
    </section>
  );
}
