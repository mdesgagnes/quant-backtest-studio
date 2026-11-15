"""Calibration report: synthetic checks (from the test suite) and real-data
sanity checks in the expected direction. Writes docs/CALIBRATION_REPORT.md.

    python scripts/calibration.py

A failed real-data check is reported as FAIL with the numbers; thresholds
are never tuned to force a pass.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from qbt.security_review import config as CF  # noqa: E402
from qbt.security_review import engine as E  # noqa: E402
from qbt.security_review import stats as S  # noqa: E402
from qbt.security_review.bundle import load_bundle  # noqa: E402


def synthetic():
    xml = os.path.join(ROOT, "data", "cache", "calibration_junit.xml")
    os.makedirs(os.path.dirname(xml), exist_ok=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    os.path.join(ROOT, "tests", "test_security_review.py"), f"--junitxml={xml}"],
                   cwd=ROOT, capture_output=True)
    rows = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        failed = tc.find("failure") is not None or tc.find("error") is not None
        doc = tc.get("name")
        rows.append((doc, "FAIL" if failed else "PASS"))
    return rows


def main():
    cfg = CF.load_all()
    b = load_bundle(cfg, cfg.universe.all_tickers())
    reg = E.regime_labels(cfg, b)
    checks = []

    def add(name, expect, ok, detail):
        checks.append({"Check": name, "Expected": expect, "Result": "PASS" if ok else "FAIL",
                       "Evidence": detail})

    # 1. USMV / SPLV: lower volatility and downside capture than SPY
    rv = E.run(E.Request("balanced", "lower_volatility", ["SPY", "USMV", "SPLV", "RSP", "QQQ"]),
               cfg, b, reg)
    vol = {c: r.standalone["Volatility"] for c, r in rv.results.items()}
    dc = {c: r.relative["Downside capture"] for c, r in rv.results.items()}
    for c in ("USMV", "SPLV"):
        add(f"{c} lower volatility than SPY", f"vol({c}) < vol(SPY)", vol[c] < vol["SPY"],
            f"{vol[c]:.2%} vs {vol['SPY']:.2%} ({rv.period})")
        add(f"{c} lower downside capture than SPY", "downside capture < 1", dc[c] < 1,
            f"{dc[c]:.2f}")
    # 2. RSP smaller mega-cap tilt than SPY
    mt = {c: rv.results[c].metrics["megacap_tilt"] for c in ("RSP", "SPY")}
    sz = {c: rv.results[c].factor["coef"].get("size", np.nan) for c in ("RSP", "SPY")}
    add("RSP smaller mega-cap tilt than SPY", "megacap_tilt(RSP) < megacap_tilt(SPY)",
        mt["RSP"] < mt["SPY"],
        f"{mt['RSP']:+.2f} vs {mt['SPY']:+.2f}; size loading {sz['RSP']:+.2f} vs {sz['SPY']:+.2f}. "
        "Note: RSP is a leg of the QQQ-RSP spread, so part of this is mechanical; the size loading "
        "is independent evidence.")
    # 3. QQQ higher upside capture than SPY
    uc = rv.results["QQQ"].relative["Upside capture"]
    add("QQQ higher upside capture than SPY", "upside capture > 1", uc > 1, f"{uc:.2f}")
    # 4. XSB.TO smaller loss than XLB.TO in the 2022 rate shock
    rv2 = E.run(E.Request("balanced", "rising_rate_protection", ["XSB.TO", "XLB.TO"],
                          alignment="each", robustness=False), cfg, b, reg)
    s = {c: rv2.results[c].stress.set_index("Key").loc["rate_shock_2022", "Candidate return"]
         for c in ("XSB.TO", "XLB.TO")}
    add("XSB.TO smaller 2022 loss than XLB.TO", "ret(XSB) > ret(XLB)", s["XSB.TO"] > s["XLB.TO"],
        f"{s['XSB.TO']:+.2%} vs {s['XLB.TO']:+.2%}")
    # 5. Managed futures positive in 2022 while Balanced was negative
    rv3 = E.run(E.Request("balanced", "stock_bond_correlation_hedge", ["DBMF", "KMLM"],
                          alignment="each", robustness=False), cfg, b, reg)
    for c in ("DBMF", "KMLM"):
        st = rv3.results[c].stress.set_index("Key").loc["rate_shock_2022"]
        add(f"{c} positive in 2022 while Balanced negative", "cand > 0 and portfolio < 0",
            st["Candidate return"] > 0 and st["Portfolio return, without"] < 0,
            f"{c} {st['Candidate return']:+.2%}, Balanced {st['Portfolio return, without']:+.2%}")
    # 6. Adding IEF / TLT lowers Balanced volatility over the full period
    rv4 = E.run(E.Request("balanced", "lower_volatility", ["IEF", "TLT"], alignment="each",
                          robustness=False), cfg, b, reg)
    for c in ("IEF", "TLT"):
        r = rv4.results[c]
        add(f"Adding {c} lowers Balanced volatility", "d_vol < 0", r.metrics["d_vol"] < 0,
            f"{r.metrics['d_vol'] * 100:+.2f} pts over {r.start:%Y-%m} to {r.end:%Y-%m}")
    # 7. Extra: same-sleeve separation, equivalence, income flag
    sc = rv.ranking.set_index("Candidate")["Score"]
    add("US equity ETFs get different scores", "score range > 0.3", sc.max() - sc.min() > 0.3,
        ", ".join(f"{k} {v:+.2f}" for k, v in sc.items()))
    rv5 = E.run(E.Request("income", "income_generation", ["XBB.TO", "ZAG.TO", "ZWC.TO", "XEI.TO"],
                          robustness=False), cfg, b, reg)
    pairs = {frozenset((e["A"], e["B"])) for e in rv5.equivalents}
    add("XBB.TO and ZAG.TO flagged functionally equivalent", "flagged",
        frozenset(("XBB.TO", "ZAG.TO")) in pairs, "; ".join(f"{e['A']}~{e['B']} ({e['Evidence']})"
                                                         for e in rv5.equivalents) or "none")
    fl = " ".join(rv5.results["ZWC.TO"].flags)
    add("ZWC.TO distribution flagged (option writing)", "flag present", "option writing" in fl, fl[:160])
    corr = rv3.results["DBMF"].metrics["corr_to_portfolio"]
    add("DBMF low correlation to Balanced", "corr < 0.3", corr < 0.3, f"{corr:+.2f}")

    syn = synthetic()
    lines = [
        "# Calibration report", "",
        f"Generated by `python scripts/calibration.py` on {date.today().isoformat()}, "
        f"prices to {b.as_of:%Y-%m-%d} (Yahoo Finance adjusted close, CAD).", "",
        "## 1. Synthetic checks (known answers)", "",
        "From `tests/test_security_review.py`: hand-computed metric values on small series, edge "
        "cases, and calibration on synthetic data whose answer is known by construction "
        "(0.7 x benchmark has beta 0.7 and lower volatility; a negatively correlated series lowers "
        "portfolio volatility; a near-copy is flagged functionally equivalent).", "",
        "| Test | Result |", "|---|---|"]
    lines += [f"| `{n}` | {r} |" for n, r in syn]
    npass = sum(r == "PASS" for _, r in syn)
    lines += ["", f"**{npass} of {len(syn)} passed.**", "",
              "## 2. Real-data sanity checks (expected direction)", "",
              "| Check | Expected | Result | Evidence |", "|---|---|---|---|"]
    lines += [f"| {c['Check']} | {c['Expected']} | {c['Result']} | {c['Evidence']} |" for c in checks]
    nok = sum(c["Result"] == "PASS" for c in checks)
    lines += ["", f"**{nok} of {len(checks)} passed.**", ""]
    fails = [c for c in checks if c["Result"] == "FAIL"]
    lines += ["## 3. Failures and explanations", ""]
    if not fails:
        lines += ["None: every check went in the expected direction.", ""]
    else:
        lines += [f"- **{c['Check']}**: {c['Evidence']}. See the explanation below." for c in fails]
        lines += ["", "EXPLANATIONS_PLACEHOLDER", ""]
    out = os.path.join(ROOT, "docs", "CALIBRATION_REPORT.md")
    existing = open(out, encoding="utf-8").read() if os.path.exists(out) else ""
    if "## 4. Notes" in existing:            # keep hand-written notes across re-runs
        lines += [existing[existing.index("## 4. Notes"):]]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
