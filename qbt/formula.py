"""Formula evaluator for user-defined signals.

Lets a strategy be written as an expression over price-derived indicators,
without writing Python and without the app ever running arbitrary code.

    sma(price, 50) > sma(price, 200)
    0.6 * pctrank(mom(price, 126)) + 0.4 * pctrank(-vol(price, 60))
    ifelse(rsi(price, 14) < 30, 1, 0)

Everything evaluates on DataFrames indexed by date with one column per
instrument, so an expression is applied to the whole universe at once and
returns a value per instrument per day.

Why not `eval`
--------------
The app is reachable at a public URL. `eval` or `exec` on user input would
let anyone who reaches the login page run arbitrary code inside the
container: read the secrets file, open network connections, exfiltrate
whatever the process can see. Instead the expression is parsed to a syntax
tree and walked node by node, and anything not on the whitelist below is
refused. There is no attribute access, no subscripting, no imports, no
lambdas, no comprehensions and no name that is not a declared indicator, so
there is no route from an expression to the interpreter.
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .strategies.base import (
    sma as _sma, ema as _ema, rsi as _rsi, total_return as _mom,
    realized_vol as _vol, downside_vol as _dvol,
    efficiency_ratio as _er, zscore as _z,
)

MAX_LENGTH = 2000
MAX_WINDOW = 2000
MAX_NODES = 400

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.Call, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.IfExp,
    ast.BitAnd, ast.BitOr, ast.Invert,
)


class _Elementwise(ast.NodeTransformer):
    """Rewrites `and` / `or` / `not` into elementwise operations.

    Python's `and` asks whether its left operand is true, and a DataFrame
    refuses to answer -- "the truth value is ambiguous". Every combined
    condition therefore fails at evaluation, which is not what anyone means
    by `price > sma(price, 200) and rsi(price, 14) < 30`: they mean the two
    masks intersected, row by row.

    Rewriting the tree rather than asking users to type `&` keeps the
    language readable and avoids the precedence trap that makes `a > b & c
    > d` parse as `a > (b & c) > d`. Parentheses are implicit in the tree,
    so the rewrite cannot introduce that bug.
    """

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        result = node.values[0]
        for right in node.values[1:]:
            result = ast.BinOp(left=result, op=op, right=right)
        return ast.copy_location(result, node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(
                ast.UnaryOp(op=ast.Invert(), operand=node.operand), node)
        return node


# ----------------------------------------------------------------------
# Indicator namespace
# ----------------------------------------------------------------------
def _win(n: Any) -> int:
    """Windows must be plain, sane integers."""
    try:
        n = int(n)
    except Exception:
        raise ValueError("A window must be a whole number.")
    if n < 1 or n > MAX_WINDOW:
        raise ValueError(f"A window must be between 1 and {MAX_WINDOW}.")
    return n


def _frame(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    raise ValueError("This function expects a series such as `price`.")


def _pctrank(x) -> pd.DataFrame:
    return _frame(x).rank(axis=1, pct=True, na_option="keep")


def _rank(x) -> pd.DataFrame:
    return _frame(x).rank(axis=1, ascending=False, na_option="keep",
                          method="first")


def _ifelse(cond, a, b):
    """Elementwise choice; `a` and `b` may be frames or plain numbers.

    The condition itself supplies the grid, so `ifelse(rsi(price,14) < 30,
    1, 0)` works with two bare scalars.
    """
    if not isinstance(cond, pd.DataFrame):
        return a if bool(cond) else b
    c = cond.astype(bool)
    base = next((v for v in (a, b) if isinstance(v, pd.DataFrame)), cond)

    def _as_frame(v):
        if isinstance(v, pd.DataFrame):
            return v.reindex(index=base.index, columns=base.columns)
        return pd.DataFrame(float(v), index=base.index, columns=base.columns)

    return _as_frame(a).where(c.reindex(index=base.index, columns=base.columns),
                              _as_frame(b))


def _clip(x, lo=None, hi=None):
    return _frame(x).clip(lower=lo, upper=hi)


def build_namespace(price: pd.DataFrame,
                    exog: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """The complete set of names an expression may use."""
    ns: Dict[str, Any] = {
        # data
        "price": price,
        # trend and momentum
        "sma": lambda x, n: _sma(_frame(x), _win(n)),
        "ema": lambda x, n: _ema(_frame(x), _win(n)),
        "mom": lambda x, n: _mom(_frame(x), _win(n)),
        "roc": lambda x, n: _mom(_frame(x), _win(n)),
        # risk
        "vol": lambda x, n: _vol(_frame(x), _win(n)),
        "dvol": lambda x, n: _dvol(_frame(x), _win(n)),
        # oscillators and quality
        "rsi": lambda x, n: _rsi(_frame(x), _win(n)),
        "er": lambda x, n: _er(_frame(x), _win(n)),
        # statistics
        "zscore": lambda x, n: _z(_frame(x), _win(n)),
        "mean": lambda x, n: _frame(x).rolling(_win(n), min_periods=_win(n)).mean(),
        "std": lambda x, n: _frame(x).rolling(_win(n), min_periods=_win(n)).std(ddof=1),
        "mmax": lambda x, n: _frame(x).rolling(_win(n), min_periods=_win(n)).max(),
        "mmin": lambda x, n: _frame(x).rolling(_win(n), min_periods=_win(n)).min(),
        "shift": lambda x, n: _frame(x).shift(_win(n)),
        # cross-sectional
        "pctrank": _pctrank,
        "rank": _rank,
        # shaping
        "ifelse": _ifelse,
        "clip": _clip,
        "abs": lambda x: _frame(x).abs(),
        "log": lambda x: np.log(_frame(x).where(_frame(x) > 0)),
        "sign": lambda x: np.sign(_frame(x)),
    }
    if exog is not None and not exog.empty:
        for col in exog.columns:
            key = sanitize(str(col))
            if key and key not in ns:
                ns[key] = exog[[col]] if False else exog[col].to_frame().reindex(
                    columns=[col])
                # a macro series is one column: broadcast it across the universe
                ns[key] = pd.DataFrame(
                    np.repeat(exog[[col]].to_numpy(), price.shape[1], axis=1),
                    index=exog.index, columns=price.columns)
    return ns


def sanitize(name: str) -> str:
    """Turns a column header into a usable identifier."""
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.strip())
    if out and out[0].isdigit():
        out = "s_" + out
    return out.strip("_")


def available_names(price_cols: List[str],
                    exog_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """What the editor should offer, grouped for display."""
    groups = {
        "Data": ["price"],
        "Trend": ["sma(x, n)", "ema(x, n)", "mom(x, n)"],
        "Risk": ["vol(x, n)", "dvol(x, n)"],
        "Oscillators": ["rsi(x, n)", "er(x, n)"],
        "Statistics": ["zscore(x, n)", "mean(x, n)", "std(x, n)",
                       "mmax(x, n)", "mmin(x, n)", "shift(x, n)"],
        "Cross-section": ["pctrank(x)", "rank(x)"],
        "Shaping": ["ifelse(cond, a, b)", "clip(x, lo, hi)", "abs(x)",
                    "log(x)", "sign(x)"],
    }
    if exog_cols:
        groups["Imported series"] = [sanitize(str(c)) for c in exog_cols]
    return groups


# ----------------------------------------------------------------------
# Safe evaluation
# ----------------------------------------------------------------------
class FormulaError(ValueError):
    """Raised with a message meant to be shown to the user."""


def _check(tree: ast.AST, allowed: set) -> None:
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise FormulaError("Expression is too long.")
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError(
                f"`{type(node).__name__}` is not allowed here. Expressions may "
                f"only combine the listed indicators with arithmetic and "
                f"comparisons.")
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise FormulaError(f"Unknown name `{node.id}`.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError("Only the listed functions may be called.")
            if node.func.id not in allowed:
                raise FormulaError(f"Unknown function `{node.func.id}`.")
            if node.keywords:
                raise FormulaError("Named arguments are not supported.")


def evaluate(expression: str, namespace: Dict[str, Any]) -> Any:
    """Parses and evaluates one expression against the namespace."""
    text = (expression or "").strip()
    if not text:
        raise FormulaError("Expression is empty.")
    if len(text) > MAX_LENGTH:
        raise FormulaError("Expression is too long.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Syntax error: {exc.msg}")

    _check(tree, set(namespace))
    # Rewrite after the whitelist check, so the check still sees exactly
    # what the user wrote.
    tree = ast.fix_missing_locations(_Elementwise().visit(tree))
    code = compile(tree, "<formula>", "eval")
    try:
        # __builtins__ emptied: nothing from the interpreter is reachable,
        # only the indicator names explicitly placed in the namespace.
        return eval(code, {"__builtins__": {}}, dict(namespace))
    except FormulaError:
        raise
    except ZeroDivisionError:
        raise FormulaError("Division by zero.")
    except Exception as exc:
        raise FormulaError(f"Could not evaluate: {exc}")


def evaluate_frame(expression: str, price: pd.DataFrame,
                   exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Evaluates and guarantees a DataFrame aligned to the price grid."""
    ns = build_namespace(price, exog)
    out = evaluate(expression, ns)
    if isinstance(out, (int, float, bool, np.floating, np.integer, np.bool_)):
        out = pd.DataFrame(float(out), index=price.index, columns=price.columns)
    if not isinstance(out, pd.DataFrame):
        raise FormulaError(
            "The expression must produce a value per instrument, not a single "
            "number. Combine it with `price` or an indicator.")
    out = out.reindex(index=price.index, columns=price.columns)
    return out.replace([np.inf, -np.inf], np.nan)


def describe(expression: str, price: pd.DataFrame,
             exog: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Evaluates and reports what came out, for the editor's preview."""
    frame = evaluate_frame(expression, price, exog)
    vals = frame.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    first = frame.dropna(how="all")
    boolean = bool(np.isin(finite, (0.0, 1.0)).all()) if finite.size else False
    return {
        "frame": frame,
        "is_boolean": boolean,
        "coverage": float(np.isfinite(vals).mean()) if vals.size else 0.0,
        "first_valid": first.index[0] if len(first) else None,
        "min": float(np.nanmin(finite)) if finite.size else np.nan,
        "max": float(np.nanmax(finite)) if finite.size else np.nan,
        "median": float(np.nanmedian(finite)) if finite.size else np.nan,
    }
