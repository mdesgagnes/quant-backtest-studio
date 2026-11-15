"""Run a Security review from a YAML request file, without the interface.

    python -m qbt.security_review examples/demo_1_balanced_lower_volatility.yaml

The request file holds the same inputs as the page:

    profile: balanced                 # a key of config/profiles.yaml
    objective: lower_volatility       # a key of config/objectives.yaml
    candidates: [USMV, SPLV, SPY]
    mode: add                         # add | replace
    weight: 0.10                      # Add mode
    replace_holding: VFV.TO           # Replace mode
    alignment: common                 # common | each
    sweep: DBMF                       # optional: also run a profile sweep for this ticker
    output: examples/output/demo.xlsx # optional; default next to the request file

It prints the ranking and each verdict's rationale, and writes the same
Excel workbook the page offers.
"""
from __future__ import annotations

import os
import sys
import warnings

import pandas as pd
import yaml

from . import config as CF
from . import engine as E
from . import export as X
from .bundle import load_bundle


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    warnings.filterwarnings("ignore")
    path = argv[0]
    with open(path, "r", encoding="utf-8") as fh:
        q = yaml.safe_load(fh) or {}
    cfg = CF.load_all()
    cands = [str(c).upper() for c in q.get("candidates") or []]
    if q.get("sweep"):
        cands = list(dict.fromkeys(cands + [str(q["sweep"]).upper()]))
    b = load_bundle(cfg, cands + cfg.universe.all_tickers())
    reg = E.regime_labels(cfg, b)
    req = E.Request(q["profile"], q["objective"], cands if cands else [], q.get("benchmark"),
                    q.get("mode", "add"), q.get("weight"), q.get("replace_holding"),
                    q.get("rebalance"), q.get("alignment", "common"))
    rv = E.run(req, cfg, b, reg) if q.get("candidates") else None
    sw = E.sweep(cfg, b, str(q["sweep"]).upper(), q["objective"], q.get("mode", "add"),
                 q.get("weight"), q.get("replace_holding"), reg) if q.get("sweep") else None
    if rv is None and sw is not None:
        rv = E.run(E.Request(q["profile"], q["objective"], [str(q["sweep"]).upper()],
                             mode=q.get("mode", "add"), weight=q.get("weight"),
                             replace_holding=q.get("replace_holding")), cfg, b, reg)
    pd.set_option("display.width", 200)
    print(f"\n{rv.profile.describe()} | {rv.objective_label} | {rv.period}\n")
    print(rv.ranking.to_string(index=False))
    for c in rv.ranking["Candidate"]:
        r = rv.results[c]
        print(f"\n{c}: {r.decision['verdict']} (confidence {r.decision['confidence']}, "
              f"score {r.decision['composite']:+.2f})")
        for f in r.rationale["facts"]:
            print("  - " + f)
        print("  Trade-off: " + r.rationale["tradeoff"])
        for cv in r.rationale["caveats"]:
            print("  Caveat: " + cv)
    if rv.equivalents:
        print("\nFunctionally equivalent: " + "; ".join(f"{e['A']} ~ {e['B']}" for e in rv.equivalents))
    if sw is not None:
        print("\nProfile sweep:")
        print(sw[["Profile", "Verdict", "Confidence", "Score", "Change in volatility",
                  "Change in max drawdown"]].to_string(index=False))
    out = q.get("output") or os.path.splitext(path)[0] + ".xlsx"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(X.workbook(rv, cfg, sw))
    print(f"\nWorkbook: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
