"""A recording stand-in for the Streamlit API.

The classic app's screen logic (`app.py`, `qbt/snapshot_ui.py`,
`qbt/groups_ui.py`, `qbt/lab/ui.py`) is written against `st.*`. Rather than
rewrite five thousand lines of it and risk drift, the new server runs that
same code with `st` bound to this module's proxy. Each call records a node
(a note, a table, a control, a chart) instead of drawing; the server sends
the nodes to the new front end, which draws them its own way.

What this buys: every screen, control and export behaves exactly as the
classic app's does, because it is the same code. What it costs: a control
inside a screen is a round trip to the server (typically well under a
second), not a browser-only change.

Semantics reproduced from Streamlit, because the code depends on them:

- a widget with a `key` keeps its value in `session_state[key]`;
- a widget without one is identified by its kind, label and default, so it
  resets when its default changes;
- a button is True only on the request that pressed it;
- `st.stop()` ends the run, `st.rerun()` starts it again;
- `st.cache_data` memoises by arguments, with a time to live.

Two additions the classic app could not have: tabs can be skipped when not
on screen (`Tab.body()`), and downloads are registered rather than built.
"""
from __future__ import annotations

import contextvars
import datetime as dt
import hashlib
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import pandas as pd


class StopRun(Exception):
    pass


class Rerun(Exception):
    pass


class Upload:
    """What `st.file_uploader` returns for one file."""

    def __init__(self, name: str, data: bytes):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


class Session:
    """Everything kept between requests for one browser."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}          # st.session_state (and keyed widgets)
        self.auto: Dict[str, Any] = {}           # values of widgets without a key
        self.uploads: Dict[str, List[Upload]] = {}
        self.downloads: Dict[str, Dict[str, Any]] = {}
        self.registry: Dict[str, Dict[str, Any]] = {}   # widgets of the last render
        self.clicked: set = set()
        self.submitted: Dict[str, str] = {}      # chat inputs sent with this request
        self.memo: Dict[Any, Any] = {}           # expensive results, per run
        self.memo_stamp: Any = None
        self.lock = threading.RLock()
        self.touched = time.time()


class SessionState(dict):
    """`st.session_state`: a dict that also allows attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Config:
    """`st.column_config.*`: plain descriptions the serialiser reads."""

    def __getattr__(self, kind: str):
        def make(label: Optional[str] = None, **kw):
            return {"type": kind, "label": label, **kw}
        return make


def _digest(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8", "replace"))
        h.update(b"|")
    return h.hexdigest()[:10]


class Node(dict):
    """One recorded element. Containers hold `children`."""

    def add(self, child: "Node") -> "Node":
        self.setdefault("children", []).append(child)
        return child


class Recorder:
    def __init__(self, session: Session, active_tab: Optional[str] = None,
                 all_tabs: bool = False, tab_needs: Optional[Dict[str, List[str]]] = None):
        self.session = session
        self.sidebar = Node(kind="root", area="sidebar")
        self.main = Node(kind="root", area="main")
        self.stack: List[Node] = [self.main]
        self.active_tab = active_tab
        self.all_tabs = all_tabs
        self.tab_needs = tab_needs or {}
        self.tabsets: List[Node] = []
        self.counts: Dict[str, int] = {}
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.muted = 0
        self.skip: set = set()      # tabs drawn by a dedicated screen instead

    # -- identity -------------------------------------------------------
    def area(self, node: Node) -> str:
        return node.get("area", "main")

    def widget_id(self, area: str, kind: str, label: str, key: Optional[str], default: Any) -> str:
        if key is not None:
            return str(key)
        base = f"{area}|{kind}|{label}|{_digest(default)}"
        n = self.counts.get(base, 0)
        self.counts[base] = n + 1
        return f"{base}|{n}"


_current: contextvars.ContextVar[Optional[Recorder]] = contextvars.ContextVar("qbt_recorder", default=None)


def recorder() -> Recorder:
    rec = _current.get()
    if rec is None:
        raise RuntimeError("No screen is being rendered: st.* was called outside a render.")
    return rec


class DG:
    """A place things are drawn into: the page, the sidebar, a column, an
    expander, a tab. Mirrors Streamlit's DeltaGenerator."""

    def __init__(self, node: Optional[Node] = None, root: Optional[str] = None):
        self._node = node
        self._root = root

    # -- plumbing -------------------------------------------------------
    def _target(self) -> Node:
        rec = recorder()
        if self._node is not None:
            return self._node
        if self._root == "sidebar":
            return rec.sidebar
        return rec.stack[-1]

    def _area(self) -> str:
        return self._target().get("area", "main")

    def _put(self, kind: str, **fields) -> Node:
        target = self._target()
        return target.add(Node(kind=kind, area=target.get("area", "main"), **fields))

    def _container(self, kind: str, **fields) -> "DG":
        return DG(self._put(kind, children=[], **fields))

    def __enter__(self) -> "DG":
        recorder().stack.append(self._target())
        return self

    def __exit__(self, *exc) -> bool:
        recorder().stack.pop()
        return False

    # -- widgets --------------------------------------------------------
    def _value(self, wid: str, key: Optional[str], default: Any) -> Any:
        s = recorder().session
        if key is not None:
            return s.state[key] if key in s.state else default
        return s.auto.get(wid, default)

    def _store(self, wid: str, key: Optional[str], value: Any) -> None:
        s = recorder().session
        if key is not None:
            s.state[key] = value
        else:
            s.auto[wid] = value

    def _widget(self, kind: str, label: str, key: Optional[str], default: Any,
                value: Any, **fields) -> Node:
        rec = recorder()
        wid = fields.pop("wid")
        node = self._put(kind, id=wid, label=str(label), value=value, **fields)
        rec.registry[wid] = {"kind": kind, "key": key,
                             "options": fields.get("options"),
                             "int": fields.get("is_int", False),
                             "columns": fields.get("columns")}
        return node

    def _choice(self, kind: str, label: str, options: Sequence[Any], index: Optional[int],
                key: Optional[str], format_func: Callable[[Any], Any], **extra) -> Any:
        opts = list(options)
        default = opts[index] if (index is not None and opts and 0 <= index < len(opts)) else None
        wid = recorder().widget_id(self._area(), kind, label, key, (default, len(opts)))
        value = self._value(wid, key, default)
        if value is not None and not _contains(opts, value):
            value = default
        self._store(wid, key, value)
        self._widget(kind, label, key, default, value, wid=wid, options=opts,
                     option_labels=[str(format_func(o)) for o in opts], **extra)
        return value

    def selectbox(self, label, options, index=0, format_func=str, key=None, help=None,
                  label_visibility="visible", **_kw):
        return self._choice("selectbox", label, options, index, key, format_func,
                            help=help, label_visibility=label_visibility)

    def radio(self, label, options, index=0, format_func=str, key=None, help=None,
              horizontal=False, label_visibility="visible", **_kw):
        return self._choice("radio", label, options, index, key, format_func,
                            help=help, horizontal=horizontal, label_visibility=label_visibility)

    def pills(self, label, options, selection_mode="single", default=None, format_func=str,
              key=None, label_visibility="visible", **_kw):
        opts = list(options)
        idx = opts.index(default) if default in opts else None
        return self._choice("button_group", label, opts, idx, key, format_func,
                            style="pills", label_visibility=label_visibility, clearable=True)

    def segmented_control(self, label, options, default=None, format_func=str, key=None,
                          label_visibility="visible", **_kw):
        opts = list(options)
        idx = opts.index(default) if default in opts else None
        return self._choice("button_group", label, opts, idx, key, format_func,
                            style="segmented", label_visibility=label_visibility, clearable=True)

    def select_slider(self, label, options, value=None, format_func=str, key=None, help=None, **_kw):
        opts = list(options)
        idx = opts.index(value) if value in opts else 0
        return self._choice("select_slider", label, opts, idx, key, format_func, help=help)

    def multiselect(self, label, options, default=None, format_func=str, key=None, help=None,
                    max_selections=None, placeholder=None, label_visibility="visible", **_kw):
        opts = list(options)
        dflt = [d for d in (default or []) if _contains(opts, d)]
        wid = recorder().widget_id(self._area(), "multiselect", label, key, (dflt, len(opts)))
        value = self._value(wid, key, dflt)
        value = [v for v in (value or []) if _contains(opts, v)]
        if max_selections:
            value = value[:max_selections]
        self._store(wid, key, value)
        self._widget("multiselect", label, key, dflt, value, wid=wid, options=opts,
                     option_labels=[str(format_func(o)) for o in opts], help=help,
                     max_selections=max_selections, placeholder=placeholder,
                     label_visibility=label_visibility)
        return list(value)

    def _number(self, kind, label, min_value, max_value, value, step, key, help, **extra):
        nums = [v for v in (min_value, max_value, value, step) if v is not None]
        is_int = bool(nums) and all(isinstance(v, int) and not isinstance(v, bool) for v in nums)
        if value is None:
            value = min_value if min_value is not None else (0 if is_int else 0.0)
        wid = recorder().widget_id(self._area(), kind, label, key, (value, min_value, max_value))
        got = self._value(wid, key, value)
        try:
            got = int(round(float(got))) if is_int else float(got)
        except (TypeError, ValueError):
            got = value
        if min_value is not None:
            got = max(min_value, got)
        if max_value is not None:
            got = min(max_value, got)
        self._store(wid, key, got)
        self._widget(kind, label, key, value, got, wid=wid, min=min_value, max=max_value,
                     step=step if step is not None else (1 if is_int else 0.01),
                     is_int=is_int, help=help, **extra)
        return got

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None,
                     key=None, help=None, **_kw):
        return self._number("number_input", label, min_value, max_value, value, step, key, help)

    def slider(self, label, min_value=0, max_value=100, value=None, step=None, key=None,
               help=None, **_kw):
        return self._number("slider", label, min_value, max_value, value, step, key, help)

    def _text(self, kind, label, value, key, help, **extra):
        wid = recorder().widget_id(self._area(), kind, label, key, value)
        got = self._value(wid, key, value)
        got = "" if got is None else str(got)
        self._store(wid, key, got)
        self._widget(kind, label, key, value, got, wid=wid, help=help, **extra)
        return got

    def text_input(self, label, value="", key=None, help=None, placeholder=None, type="default",
                   label_visibility="visible", **_kw):
        return self._text("text_input", label, value, key, help, placeholder=placeholder,
                          secret=(type == "password"), label_visibility=label_visibility)

    def text_area(self, label, value="", height=None, key=None, help=None, placeholder=None,
                  label_visibility="visible", **_kw):
        return self._text("text_area", label, value, key, help, placeholder=placeholder,
                          height=height, label_visibility=label_visibility)

    def date_input(self, label, value=None, min_value=None, max_value=None, key=None, help=None, **_kw):
        default = _as_date(value) if value is not None else dt.date.today()
        wid = recorder().widget_id(self._area(), "date_input", label, key, default)
        got = _as_date(self._value(wid, key, default)) or default
        lo, hi = _as_date(min_value), _as_date(max_value)
        if lo and got < lo:
            got = lo
        if hi and got > hi:
            got = hi
        self._store(wid, key, got)
        self._widget("date_input", label, key, default, got, wid=wid, min=lo, max=hi, help=help)
        return got

    def _bool(self, kind, label, value, key, help):
        wid = recorder().widget_id(self._area(), kind, label, key, bool(value))
        got = bool(self._value(wid, key, bool(value)))
        self._store(wid, key, got)
        self._widget(kind, label, key, bool(value), got, wid=wid, help=help)
        return got

    def checkbox(self, label, value=False, key=None, help=None, **_kw):
        return self._bool("checkbox", label, value, key, help)

    def toggle(self, label, value=False, key=None, help=None, **_kw):
        return self._bool("toggle", label, value, key, help)

    def button(self, label, key=None, help=None, type="secondary", disabled=False, **_kw):
        rec = recorder()
        wid = rec.widget_id(self._area(), "button", label, key, None)
        self._widget("button", label, key, None, None, wid=wid, help=help,
                     primary=(type == "primary"), disabled=bool(disabled))
        return (wid in rec.session.clicked) and not disabled and not rec.muted

    def form_submit_button(self, label="Submit", **kw):
        return self.button(label, key=kw.get("key") or f"form_submit:{label}")

    def chat_input(self, placeholder="", key=None, **_kw):
        rec = recorder()
        wid = rec.widget_id(self._area(), "chat_input", placeholder, key, None)
        self._widget("chat_input", placeholder, key, None, None, wid=wid)
        return rec.session.submitted.get(wid) or None

    def file_uploader(self, label, type=None, accept_multiple_files=False, key=None, help=None, **_kw):
        rec = recorder()
        wid = str(key or label)
        files = list(rec.session.uploads.get(wid) or [])
        self._put("file_uploader", id=wid, label=str(label), types=list(type or []),
                  multiple=bool(accept_multiple_files), help=help,
                  names=[f.name for f in files])
        if accept_multiple_files:
            return files
        return files[0] if files else None

    def download_button(self, label, data, file_name=None, mime=None, key=None, help=None, **_kw):
        rec = recorder()
        wid = str(key or label)
        rec.session.downloads[wid] = {"label": label, "data": data,
                                      "file_name": file_name, "mime": mime}
        self._put("download", id=wid, label=str(label), file_name=file_name, help=help)
        return False

    def data_editor(self, data, num_rows="fixed", key=None, column_config=None, disabled=None,
                    hide_index=None, height=None, **_kw):
        rec = recorder()
        base = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        wid = rec.widget_id(self._area(), "data_editor", "", key, list(base.columns))
        s = rec.session
        edited = s.auto.get("editor:" + wid)
        frame = base
        if edited is not None:
            frame = pd.DataFrame(edited, columns=list(base.columns))
            for c in base.columns:                       # keep the base column types
                if pd.api.types.is_numeric_dtype(base[c]) or (
                        (column_config or {}).get(c, {}) or {}).get("type") == "NumberColumn":
                    frame[c] = pd.to_numeric(frame[c], errors="coerce")
        self._put("data_editor", id=wid, frame=frame, column_config=column_config or {},
                  disabled=(list(base.columns) if disabled is True else list(disabled or [])),
                  dynamic=(num_rows == "dynamic"), hide_index=True, height=height)
        rec.registry[wid] = {"kind": "data_editor", "key": key, "columns": list(base.columns)}
        return frame

    # -- display --------------------------------------------------------
    def markdown(self, body="", unsafe_allow_html=False, **_kw):
        self._put("markdown", body=str(body), html=bool(unsafe_allow_html))

    def write(self, *args, **_kw):
        for a in args:
            if isinstance(a, (pd.DataFrame, pd.Series)):
                self.dataframe(a)
            else:
                self._put("markdown", body=str(a), html=False)

    def caption(self, body, **_kw):
        self._put("caption", body=str(body))

    def code(self, body, language=None, **_kw):
        self._put("code", body=str(body), language=language)

    def error(self, body, **_kw):
        self._put("error", body=str(body))

    def warning(self, body, **_kw):
        self._put("warning", body=str(body))

    def info(self, body, **_kw):
        self._put("info", body=str(body))

    def success(self, body, **_kw):
        self._put("success", body=str(body))

    def dataframe(self, data=None, use_container_width=None, hide_index=None, height=None,
                  column_config=None, **_kw):
        self._put("dataframe", data=data, hide_index=bool(hide_index), height=height,
                  column_config=column_config or {})

    table = dataframe

    def plotly_chart(self, figure, use_container_width=None, config=None, key=None, **_kw):
        if figure is not None:
            self._put("plotly_chart", figure=figure)

    def image(self, image, **_kw):
        self._put("image", data=image)

    def html_frame(self, html: str, height: int) -> None:
        """A self-contained document shown in a sandboxed frame (the
        TradingView price chart)."""
        self._put("iframe", html=html, height=int(height))

    # -- layout ---------------------------------------------------------
    def columns(self, spec, gap=None, **_kw):
        weights = [1] * spec if isinstance(spec, int) else list(spec)
        row = self._put("row", children=[], weights=weights)
        return [DG(row.add(Node(kind="column", area=row["area"], children=[])))
                for _ in weights]

    def expander(self, label, expanded=False, **_kw):
        return self._container("expander", label=str(label), expanded=bool(expanded))

    def container(self, **_kw):
        return self._container("container")

    def form(self, key=None, **_kw):
        return self._container("form", id=str(key))

    def chat_message(self, role, **_kw):
        return self._container("chat_message", label=str(role))

    def tabs(self, labels, **_kw):
        rec = recorder()
        node = self._put("tabs", children=[], labels=[str(x) for x in labels])
        rec.tabsets.append(node)
        return [Tab(node.add(Node(kind="tab", area=node["area"], label=str(x), children=[],
                                  index=i, open=False)), node)
                for i, x in enumerate(labels)]

    def spinner(self, *_a, **_kw):
        return _Null()

    def empty(self):
        return self._container("container")

    def set_page_config(self, **_kw):
        return None


class Tab(DG):
    def __init__(self, node: Node, tabset: Node):
        super().__init__(node)
        self._tabset = tabset

    def _wanted(self) -> str:
        """'show', 'mute' (run for the values it leaves behind) or 'skip'."""
        rec = recorder()
        if rec.all_tabs:
            return "show"
        labels = self._tabset["labels"]
        active = rec.active_tab if rec.active_tab in labels else labels[0]
        self._tabset["active"] = active
        me = self._node["label"]
        if me == active:
            return "skip" if f"{len(labels)}:{me}" in rec.skip else "show"
        if me in rec.tab_needs.get(f"{len(labels)}:{active}", []):
            return "mute"
        return "skip"

    def body(self) -> Iterator[None]:
        """`for _ in tab.body():` runs the tab's code only when it is on
        screen, or when the tab on screen needs what this one computes."""
        rec = recorder()
        mode = self._wanted()
        if mode == "skip":
            return
        self._node["open"] = (mode == "show")
        rec.stack.append(self._node)
        if mode == "mute":
            rec.muted += 1
        try:
            yield None
        finally:
            if mode == "mute":
                rec.muted -= 1
            rec.stack.pop()

    def __enter__(self):
        self._node["open"] = True
        return super().__enter__()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _contains(options: List[Any], value: Any) -> bool:
    try:
        return any(o is value or o == value for o in options)
    except Exception:
        return False


def _as_date(v: Any) -> Optional[dt.date]:
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    try:
        return pd.Timestamp(v).date()
    except Exception:
        return None


# ----------------------------------------------------------------------
# cache_data
# ----------------------------------------------------------------------
_cache: Dict[Any, Any] = {}
_cache_lock = threading.RLock()


def arg_key(v: Any) -> Any:
    """A hashable stand-in for an argument, by content where that is cheap."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, bytes):
        return ("bytes", len(v), hashlib.sha1(v).hexdigest())
    if isinstance(v, (tuple, list)):
        return tuple(arg_key(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((str(k), arg_key(x)) for k, x in v.items()))
    if isinstance(v, (pd.DataFrame, pd.Series, pd.Index)):
        try:
            return ("pd", v.shape, int(pd.util.hash_pandas_object(v, index=True).sum()))
        except Exception:
            return ("pd-id", id(v))
    if isinstance(v, (dt.date, dt.datetime)):
        return str(v)
    if hasattr(v, "__dataclass_fields__"):
        return ("dc", type(v).__name__,
                tuple(arg_key(getattr(v, f)) for f in v.__dataclass_fields__))
    return ("obj", type(v).__name__, id(v))


def cache_data(func=None, *, ttl: Optional[float] = None, show_spinner: Any = None, **_kw):
    def wrap(fn):
        name = f"{fn.__module__}.{fn.__qualname__}"

        def cached(*args, **kwargs):
            key = (name, arg_key(args), arg_key(kwargs))
            now = time.time()
            with _cache_lock:
                hit = _cache.get(key)
                if hit is not None and (ttl is None or now - hit[0] < ttl):
                    return hit[1]
            out = fn(*args, **kwargs)
            with _cache_lock:
                _cache[key] = (now, out)
            return out
        cached.__wrapped__ = fn
        cached.clear = lambda: None
        return cached
    if func is not None:
        return wrap(func)
    return wrap


cache_data.clear = lambda: _cache.clear()          # type: ignore[attr-defined]


def memo(session_key: Any, fn: Callable[[], Any]) -> Any:
    """An expensive result kept for the life of the run on screen."""
    s = recorder().session
    if session_key not in s.memo:
        s.memo[session_key] = fn()
    return s.memo[session_key]


# ----------------------------------------------------------------------
# The module-level `st`
# ----------------------------------------------------------------------
class _Secrets:
    def get(self, key, default=None):
        import os
        return os.environ.get("QBT_" + str(key).upper(), default)


class _St(DG):
    """What `import streamlit as st` is replaced by."""

    column_config = _Config()
    secrets = _Secrets()
    cache_data = staticmethod(cache_data)

    def __init__(self):
        super().__init__(None, None)
        object.__setattr__(self, "sidebar", DG(None, "sidebar"))

    def __setattr__(self, name, value):
        # app.py replaces st.dataframe with a wrapper that sets alignment
        # and height. The serialiser does both, so the swap is ignored
        # (it would otherwise nest once per request).
        if name.startswith("_"):
            object.__setattr__(self, name, value)

    @property
    def session_state(self) -> Dict[str, Any]:
        s = recorder().session
        if not isinstance(s.state, SessionState):
            s.state = SessionState(s.state)
        return s.state

    def stop(self):
        raise StopRun()

    def rerun(self):
        raise Rerun()


st = _St()


class _WhenRendering:
    """What the view modules inside qbt get as `st`: the recorder while the
    new server is rendering, the real Streamlit otherwise. So the classic
    app, and tests that run it, behave exactly as before in the same
    process."""

    def __getattr__(self, name: str) -> Any:
        if _current.get() is not None:
            return getattr(st, name)
        import streamlit
        return getattr(streamlit, name)


st_when_rendering = _WhenRendering()


def render(session: Session, script: Callable[[], None], active_tab: Optional[str] = None,
           all_tabs: bool = False, tab_needs: Optional[Dict[str, List[str]]] = None,
           skip: Optional[set] = None) -> Recorder:
    """Runs `script` once (again on `st.rerun()`), recording what it draws."""
    rec: Optional[Recorder] = None
    for _ in range(6):
        rec = Recorder(session, active_tab, all_tabs, tab_needs)
        rec.skip = set(skip or ())
        token = _current.set(rec)
        try:
            script()
        except StopRun:
            pass
        except Rerun:
            session.clicked.clear()
            session.submitted.clear()
            continue
        finally:
            _current.reset(token)
        break
    session.clicked.clear()
    session.submitted.clear()
    # Controls of tabs drawn earlier stay known: the browser keeps those tabs
    # and can send a change from one without it being redrawn first.
    if len(session.registry) > 4000:
        session.registry = {}
    session.registry.update(rec.registry)
    session.touched = time.time()
    return rec
