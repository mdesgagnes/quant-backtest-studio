"""The HTTP API of the redesigned app, end to end on the golden fixture."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient                      # noqa: E402

from server.app import app                                     # noqa: E402

FIX = os.path.join(ROOT, "tests", "golden", "fixtures")
EXPECTED = os.path.join(ROOT, "tests", "golden", "expected", "momentum_file")
client = TestClient(app)


def _upload(kind, name):
    with open(os.path.join(FIX, name), "rb") as fh:
        r = client.post(f"/api/uploads/{kind}", files=[("files", (name, fh.read()))])
    assert r.status_code == 200, r.text
    return r.json()


def _run(spec, files):
    return client.post("/api/runs", json={"spec": spec, "files": files})


def test_meta_lists_every_model_and_tab():
    m = client.get("/api/meta").json()
    assert len(m["models"]) == 21
    assert [t["name"] for t in m["tabs"]][0] == "Results"
    assert m["tabs"][0]["screen"] == "results"
    assert m["defaults"]["commission_bps"] == 5.0


def test_price_upload_reports_what_was_read():
    up = _upload("prices", "prices_long.csv")
    assert up["instruments"] == ["XEI.TO", "XIC.TO", "ZEB.TO", "ZLB.TO"]
    assert up["has_open"] and up["has_volume"] and up["error"] is None
    assert up["convention"] == "price"


def test_run_matches_the_recorded_headline_and_is_cached():
    up = _upload("prices", "prices_long.csv")
    spec = {"source": "upload", "tickers": [], "start": None}
    r = _run(spec, {"prices": up["ids"]})
    assert r.status_code == 200, r.text
    first = r.json()
    with open(os.path.join(EXPECTED, "screens.json"), encoding="utf-8") as fh:
        recorded = json.load(fh)
    strip = next(b for b in recorded["page"] if b["kind"] == "context")
    assert first["context"] == strip["items"]
    want = [b for b in recorded["tab: Results"] if b["kind"] == "kpi"]
    assert [(k["label"], k["value"], k["sub"], k["tone"]) for k in first["kpis"]] == \
           [(k["label"], k["value"], k["sub"], k["tone"]) for k in want]

    again = _run(spec, {"prices": up["ids"]}).json()
    assert again["id"] == first["id"] and again["cached"] is True

    blocks = client.get(f"/api/runs/{first['id']}/screens/results").json()["blocks"]
    kinds = [b["kind"] for b in blocks]
    assert kinds.count("chart") == 6 and kinds.count("table") == 4 and "panel" in kinds
    stats = next(b for b in blocks if b["kind"] == "table" and b["key"] == "statistics")
    assert stats["columns"][0] == {"name": "Metric", "align": "left",
                                   "signed": False, "strong": False}
    assert stats["columns"][1]["align"] == "right"
    json.dumps(blocks, allow_nan=False)            # strict JSON: no NaN reaches the browser


def test_errors_are_the_users_to_read():
    r = _run({"source": "upload", "tickers": [], "start": None}, {})
    assert r.status_code == 422 and "No file loaded" in r.json()["detail"]
    assert _run({"no_such_control": 1}, {}).status_code == 422
    assert client.get("/api/runs/nope/screens/results").status_code == 404
    assert client.get("/api/runs/nope/screens/positions").status_code == 404
