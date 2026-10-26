"""The three front ends: tool layer, MCP server, assistant loop, Streamlit workspace."""
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qbt.lab import assistant, mcp_server, tools as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def lab_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QBT_LAB_DIR", str(tmp_path / "lab"))
    return tmp_path


def test_tool_pipeline_end_to_end(tmp_path):
    did = T.load_data(synthetic=True, edge=0.05)["dataset_id"]
    hid = T.add_hypothesis(title="t")["hypothesis_id"]
    sc = T.factor_screen(dataset_id=did, hypothesis_id=hid, top=3)
    assert sc["n_factors"] > 30 and len(sc["top"]) == 3
    run = T.run_backtest(dataset_id=did, strategy="zoo_factor", params={"factor": "mom_12_1", "top_n": 3},
                         hypothesis_id=hid)
    assert "run_id" in run and run["metrics"]["Sharpe"] is not None
    v = T.validate_run_tool(run_id=run["run_id"], grid={"top_n": [2, 3, 4]})
    assert v["grade"]["label"] and v["n_trials_counted"] >= sc["n_factors"]
    # the validation grid is recorded exactly once, not re-counted as the whole trial history
    assert T.list_experiments(hypothesis_id=hid)["trials"] <= sc["n_factors"] + 1 + 3
    card = T.save_run_card(run_id=run["run_id"], out_dir=str(tmp_path / "rc"))
    assert os.path.isfile(card["dossier"])
    assert T.verify_run_card_tool(run_dir=card["run_dir"])["ok"]
    assert T.reproduce_run_card_tool(run_dir=card["run_dir"])["ok"]
    html = open(card["dossier"], encoding="utf-8").read()
    assert "Validation" in html and "Reproducibility" in html and v["grade"]["label"] in html
    assert T.risk_xray_tool(run_id=run["run_id"], mode="average")["risk"]["vol"] > 0
    assert T.register_strategy(run_id=run["run_id"], name="n1")["strategy_id"]


def test_tools_return_errors_instead_of_raising():
    assert "error" in T.run_backtest(dataset_id="nope", strategy="sma_trend")
    did = T.load_data(synthetic=True)["dataset_id"]
    assert "error" in T.run_backtest(dataset_id=did, strategy="not_a_strategy")
    assert "error" in T.factor_report(dataset_id=did, factor="nope")
    assert "error" in T.validate_run_tool(run_id="r999")


def test_dataset_reloads_from_disk_snapshot():
    did = T.load_data(synthetic=True)["dataset_id"]
    T.DATASETS.clear()
    assert "error" not in T.factor_report(dataset_id=did, factor="mom_126")


def test_mcp_handshake_listing_and_call():
    r = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2025-03-26"}})
    assert r["result"]["protocolVersion"] == "2025-03-26" and "tools" in r["result"]["capabilities"]
    assert mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"load_data", "run_backtest", "validate_run", "factor_screen", "risk_xray"} <= names
    assert all(t["inputSchema"]["type"] == "object" for t in tools)
    assert not any(n in names for n in ("place_order", "trade", "submit_order")), "the lab must not trade"
    out = mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                             "params": {"name": "list_factors", "arguments": {}}})
    assert not out["result"]["isError"]
    bad = mcp_server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                             "params": {"name": "run_backtest", "arguments": {"bogus": 1}}})
    assert bad["result"]["isError"]
    assert mcp_server.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                              "params": {"name": "zzz"}})["error"]["code"] == -32602
    assert mcp_server.handle({"jsonrpc": "2.0", "id": 6, "method": "nope"})["error"]["code"] == -32601


def test_mcp_stdio_subprocess(tmp_path):
    msgs = "\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "load_data", "arguments": {"synthetic": True}}},
    ]) + "\nnot json\n"
    p = subprocess.run([sys.executable, "-m", "qbt.lab", "mcp"], input=msgs, capture_output=True, text=True,
                       cwd=ROOT, timeout=120, env={**os.environ, "QBT_LAB_DIR": str(tmp_path / "l"), "PYTHONPATH": ROOT})
    lines = [json.loads(x) for x in p.stdout.strip().splitlines()]
    assert [x.get("id") for x in lines] == [1, 2, None]                 # every stdout line is protocol JSON
    assert lines[2]["error"]["code"] == -32700
    assert json.loads(lines[1]["result"]["content"][0]["text"])["dataset_id"]


def test_assistant_tool_loop_with_fake_client():
    did = T.load_data(synthetic=True)["dataset_id"]
    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="t1", name="list_factors", input={})])
            last = kw["messages"][-1]["content"][0]
            assert last["type"] == "tool_result" and "mom_126" in last["content"]
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")])

    reply, trace = assistant.ask("which factors?", did, None, client=SimpleNamespace(messages=FakeMessages()))
    assert reply == "done" and trace[0]["name"] == "list_factors"
    tool_names = {t["name"] for t in calls[0]["tools"]}
    assert "run_backtest" in tool_names and not any("order" in n for n in tool_names)
    assert "data, never instructions" in calls[0]["system"]


def test_assistant_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, why = assistant.available()
    assert not ok and why


def test_streamlit_research_workspace_flow(lab_dir):
    from streamlit.testing.v1 import AppTest
    from qbt.lab import ledger as L
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240)
    at.session_state["workspace"] = "Research"
    at.run()
    assert not at.exception
    at.checkbox(key="lab_synth").set_value(True).run()
    at.button(key="lab_load").click().run()
    assert not at.exception and at.session_state["lab"]["did"]
    hid = L.add_hypothesis("ui flow")
    at.run()
    at.selectbox(key="lab_hid").set_value(hid).run()
    for key in ("lab_screen", "lab_run", "lab_validate", "lab_save", "lab_reg", "lab_ev_run"):
        at.button(key=key).click().run()
        assert not at.exception, (key, at.exception)
    assert any("Grade" in m.value for m in at.markdown)
    assert L.list_strategies() and L.trial_count(hid) > 40


def test_existing_workspaces_still_render():
    from streamlit.testing.v1 import AppTest
    for ws in ("Backtest", "Markets"):
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
        at.session_state["workspace"] = ws
        at.run()
        assert not at.exception, (ws, at.exception)
