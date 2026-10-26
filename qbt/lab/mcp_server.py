"""MCP server: the lab's tools, callable by Claude (or any MCP client).

Speaks the Model Context Protocol over stdio with newline-delimited JSON-RPC
2.0, using only the standard library -- no `mcp` package needed, so it runs in
the same conda environment as the app. Implements `initialize`, `ping`,
`tools/list` and `tools/call`, which is all a tool server needs.

Run it:

    python -m qbt.lab mcp

Register it in Claude Code:

    claude mcp add qbt-lab -- C:/Users/marcd/miniconda3/envs/quant/python.exe -m qbt.lab mcp

or in Claude Desktop's config (`mcpServers`):

    {"qbt-lab": {"command": "C:/Users/marcd/miniconda3/envs/quant/python.exe",
                 "args": ["-m", "qbt.lab", "mcp"],
                 "cwd": "C:/Users/marcd/Documents/quant-backtest-studio-fresh"}}

Research only: there is no tool that places, or could place, an order.

stdout carries the protocol and nothing else. Anything a library prints is
redirected to stderr for the lifetime of the server.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from . import LAB_VERSION
from . import tools as T

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "Research lab on top of the Quant Backtest Studio engine. Typical flow: load_data -> add_hypothesis -> "
    "factor_screen / factor_report -> run_backtest (pass hypothesis_id so trials are counted) -> "
    "validate_run (pass a grid to get overfitting checks) -> save_run_card. Always report the validation "
    "grade and warnings alongside any performance number, and never present a backtest as a forecast. "
    "This server cannot trade.")


def _result(id_: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message in, one response out (None for notifications)."""
    method = msg.get("method")
    id_ = msg.get("id")
    params = msg.get("params") or {}
    if method is None:                       # a response to us; nothing to do
        return None
    if id_ is None:                          # notification
        return None
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _result(id_, {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": {"name": "qbt-lab", "version": LAB_VERSION},
                             "instructions": INSTRUCTIONS})
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {"tools": [{"name": t["name"], "description": t["description"],
                                        "inputSchema": t["inputSchema"]} for t in T.TOOLS.values()]})
    if method == "tools/call":
        name = params.get("name")
        tool = T.TOOLS.get(name)
        if tool is None:
            return _error(id_, -32602, f"Unknown tool: {name}")
        try:
            out = tool["fn"](**(params.get("arguments") or {}))
        except TypeError as exc:             # wrong argument names
            out = {"error": f"Bad arguments for {name}: {exc}"}
        text = json.dumps(out, indent=1, default=str, allow_nan=False)
        if len(text) > 60000:
            text = text[:60000] + "\n... [truncated]"
        return _result(id_, {"content": [{"type": "text", "text": text}], "isError": "error" in out})
    return _error(id_, -32601, f"Method not found: {method}")


def serve(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    out = stdout or sys.stdout
    real_stdout = sys.stdout
    sys.stdout = sys.stderr                  # library prints must not corrupt the protocol
    try:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                out.write(json.dumps(_error(None, -32700, "Parse error")) + "\n")
                out.flush()
                continue
            batch = msg if isinstance(msg, list) else [msg]
            replies = [r for r in (handle(m) for m in batch if isinstance(m, dict)) if r is not None]
            if replies:
                out.write(json.dumps(replies if isinstance(msg, list) else replies[0]) + "\n")
                out.flush()
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    serve()
