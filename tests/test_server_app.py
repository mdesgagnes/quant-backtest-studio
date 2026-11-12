"""The redesigned app's own API: the page it draws, uploads, downloads, the
password gate and configuration loading. Figures are covered by the golden
tests; this file covers the plumbing around them."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest                                                  # noqa: E402
from fastapi.testclient import TestClient                      # noqa: E402

from server.app import app                                     # noqa: E402

FIX = os.path.join(ROOT, "tests", "golden", "fixtures")


def walk(blocks, out=None):
    out = [] if out is None else out
    for b in blocks:
        out.append(b)
        if "blocks" in b:
            walk(b["blocks"], out)
        for col in b.get("cols", []):
            walk(col, out)
    return out


class Browser:
    """A session: its own cookie jar, and helpers to find and press things."""

    def __init__(self):
        self.c = TestClient(app)
        self.tab = None
        self.page = None
        self.render()

    def render(self, **body):
        if self.tab:
            body.setdefault("tab", self.tab)
        r = self.c.post("/api/render", json=body)
        assert r.status_code == 200, r.text
        self.page = r.json()
        return self.page

    def all(self, area=None):
        p = self.page
        return walk(p[area]) if area else walk(p["sidebar"]) + walk(p["main"])

    def control(self, label, n=0):
        return [b for b in self.all() if b["kind"] == "control" and b["label"] == label][n]

    def set(self, label, value, n=0):
        return self.render(events=[{"id": self.control(label, n)["id"], "value": value}])

    def press(self, label):
        return self.render(clicks=[self.control(label)["id"]])

    def upload(self, wid, name, data=None):
        if data is None:
            with open(os.path.join(FIX, name), "rb") as fh:
                data = fh.read()
        r = self.c.post(f"/api/files/{wid}", files=[("files", (name, data))])
        assert r.status_code == 200, r.text

    def text(self):
        return " ".join(f'{b.get("html") or b.get("text") or b.get("label") or ""} {b.get("sub") or ""}'
                        for b in self.all())


@pytest.fixture(scope="module")
def ran():
    b = Browser()
    b.upload("price_files", "prices_long.csv")
    b.set("Source", 1)
    b.press("Run backtest")
    return b


def test_first_page_is_the_start_screen():
    b = Browser()
    ws = next(x for x in b.all("sidebar") if x.get("id") == "workspace")
    assert ws["options"] == ["Backtest", "Markets", "Research"] and ws["value"] == 0
    assert any(x["kind"] == "control" and x["label"] == "Run backtest" for x in b.all("sidebar"))
    assert "Pick a source on the left" in b.text()


def test_run_shows_the_dedicated_results_screen(ran):
    tabs = next(x for x in ran.page["main"] if x["kind"] == "tabs")
    assert tabs["labels"][0] == "Results" and len(tabs["labels"]) == 10
    assert [x["kind"] for x in tabs["blocks"]] == ["results"]
    kpis = [x for x in tabs["blocks"][0]["blocks"] if x["kind"] == "kpi"]
    assert [(k["label"], k["value"]) for k in kpis][:2] == [("CAGR", "5.47%"), ("Volatility", "10.58%")]
    context = next(x for x in ran.page["main"] if x["kind"] == "context")
    assert context["lead"] == "Dual Momentum (Relative + Absolute)"


def test_every_backtest_tab_draws_and_downloads_work(ran):
    labels = next(x for x in ran.page["main"] if x["kind"] == "tabs")["labels"]
    seen = set()
    for name in labels[1:]:
        ran.tab = name
        ran.render()
        tabs = next(x for x in ran.page["main"] if x["kind"] == "tabs")
        assert tabs["active"] == name and tabs["blocks"], name
        seen |= {x["kind"] for x in walk(tabs["blocks"])}
    assert {"table", "chart", "kpis", "note", "control", "download", "editor", "panel"} <= seen
    ran.tab = "Positions"
    ran.render()
    d = next(x for x in ran.all("main") if x["kind"] == "download" and x["file_name"] == "current_holdings.csv")
    r = ran.c.get("/api/download/" + d["id"])
    assert r.status_code == 200 and r.text.startswith("Instrument,Weight,Value ($)")
    assert "attachment" in r.headers["content-disposition"]
    ran.tab = None


def test_a_control_inside_a_tab_changes_what_the_tab_shows(ran):
    ran.tab = "Robustness"
    ran.render()
    folds = lambda: next(len(x["rows"]) for x in ran.all("main") if x["kind"] == "table")   # noqa: E731
    assert folds() == 5
    ran.set("Number of folds", 7)
    assert folds() == 7
    ran.set("Number of folds", 5)
    ran.tab = None


def test_sessions_do_not_see_each_other(ran):
    other = Browser()
    assert not any(x["kind"] == "tabs" for x in other.page["main"])
    assert other.c.get("/api/download/dlholdings").status_code == 404


def test_loading_a_configuration_replaces_every_control():
    yaml = b"""label: probe
data: {source: yfinance, tickers: [SPY, TLT], start: '2015-01-01', benchmark: SPY}
strategy: {mode: builtin, name: dual_momentum, params: {lookback: 63, top_n: 1}}
engine: {rebalance: Q, execution_lag: 2}
costs: {commission_bps: 1.0, slippage_bps: 3.0, cash_rate_pa: 0.03, management_fee_pa: 0.01}
"""
    b = Browser()
    b.upload("cfgup", "probe.yaml", yaml)
    b.render()
    b.press("Apply configuration")
    c = b.control
    assert c("Symbols (one per line or comma-separated)")["value"] == "SPY\nTLT"
    assert c("Start")["value"] == "2015-01-01"
    bench = c("Comparison benchmark")
    assert bench["options"][bench["value"]] == "SPY"
    assert c("Momentum Window (days)")["value"] == 63
    assert c("Number of Positions")["value"] == 1
    assert c("Execution lag (days)")["value"] == 2
    assert c("Commission (bps)")["value"] == 1.0
    assert c("Slippage, half-spread (bps)")["value"] == 3.0
    assert c("Cash rate (annual %)")["value"] == 3.0
    assert c("Management fee (annual %)")["value"] == 1.0
    reb = c("Rebalance")
    assert reb["options"][reb["value"]].startswith("Quarterly")


def test_password_gate(monkeypatch):
    monkeypatch.setenv("QBT_PASSWORD", "open-sesame")
    b = Browser()
    assert "Restricted access" in b.text()
    assert not any(x.get("label") == "Run backtest" for x in b.all())
    assert b.c.post("/api/files/price_files", files=[("files", ("a.csv", b"x"))]).status_code == 401
    assert b.c.get("/api/download/anything").status_code == 401
    b.set("Password", "wrong")
    b.press("Open")
    assert "Incorrect password." in b.text()
    b.set("Password", "open-sesame")
    b.press("Open")
    assert any(x.get("label") == "Run backtest" for x in b.all())


def test_switching_workspace(ran):
    b = Browser()
    ws = next(x for x in b.all("sidebar") if x.get("id") == "workspace")
    b.render(events=[{"id": ws["id"], "value": 2}])
    assert next(x for x in b.page["main"] if x["kind"] == "title")["text"] == "Research Lab"
    assert next(x for x in b.page["main"] if x["kind"] == "tabs")["labels"][0] == "Start here"
    b.render(events=[{"id": ws["id"], "value": 1}])
    assert any(x.get("label") == "Load markets" for x in b.all("sidebar"))


def test_a_control_on_a_tab_the_browser_kept_still_works(ran):
    """The browser shows a tab it already has without asking again, so a
    change can arrive for a tab that was not the last one drawn."""
    ran.tab = "Robustness"
    ran.render()
    folds = ran.control("Number of folds")
    ran.tab = "Data"
    ran.render()
    ran.tab = "Robustness"
    ran.render(events=[{"id": folds["id"], "value": 4}])
    assert next(len(x["rows"]) for x in ran.all("main") if x["kind"] == "table") == 4
    ran.render(events=[{"id": folds["id"], "value": 5}])
    ran.tab = None
