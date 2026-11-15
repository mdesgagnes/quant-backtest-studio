"""Run a Security review from a YAML request file, without the interface.

    python -m qbt.security_review examples/demo_1_balanced_lower_volatility.yaml

The request file holds the same inputs as the page:

    candidates: [USMV, SPLV, SPY]
    candidate_returns: my_fund.csv    # optional: return streams analysed as securities
    # What to add them to: one of these (several items are compared separately)
    profile: balanced                 # a key of config/profiles.yaml (or profiles: [a, b])
    benchmarks: [XIC.TO, VFV.TO]      # each ticker on its own
    blend: {XIC.TO: 60, XBB.TO: 40}   # one blend of tickers (blend_name: optional label)
    portfolio_returns: my_portfolio.csv   # uploaded portfolio return stream(s)
    returns_currency: CAD             # CAD | USD, for uploaded files
    benchmark: sleeve                 # optional: relative statistics vs each sleeve benchmark
                                      # or a ticker (default: the portfolio itself)
    objective: lower_volatility       # optional; a key of config/objectives.yaml
    mode: add                         # add | replace
    weight: 0.10                      # Add mode: share of the whole portfolio
    fund_from: [bonds]                # optional: reduce only these (bonds, equities or
                                      # tickers), pro rata; default every holding
    replace_holding: VFV.TO           # Replace mode
    alignment: common                 # common | each
    sweep: DBMF                       # optional: also run a profile sweep for this ticker
    output: examples/output/demo.xlsx # optional; default next to the request file

File paths are relative to the request file. It prints one sentence per
security and portfolio, the summary table, and writes the same Excel
workbook the page offers.
"""
from __future__ import annotations

import os
import sys
import warnings

import pandas as pd
import yaml

from .. import returns_input as RI
from . import config as CF
from . import engine as E
from . import export as X
from . import report as RP
from .bundle import load_bundle, with_streams


def _streams(path: str):
    with open(path, "rb") as fh:
        raw, meta = RI.load_return_stream(fh.read(), path)
    df, rep = RI.prepare_returns(raw, "auto", meta)
    out = {}
    for c in df.columns:
        s = df[c].dropna()
        if rep.frequency != "monthly":
            s = RI.compound_to(s.to_frame(), "Monthly").iloc[:, 0]
        out[str(c).strip().upper()] = s
    return out


def _bases(q, streams):
    if q.get("benchmarks"):
        return [CF.custom_profile(str(t).upper(), {str(t).upper(): 1.0}) for t in q["benchmarks"]]
    if q.get("blend"):
        return [CF.custom_profile(q.get("blend_name", "Blend"), q["blend"])]
    if q.get("portfolio_returns"):
        return [CF.custom_profile(k, {k: 1.0}) for k in streams["portfolio"]]
    return list(q.get("profiles") or [q.get("profile", "balanced")])


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
    here = os.path.dirname(os.path.abspath(path))
    streams = {k: _streams(os.path.join(here, q[f])) for k, f in
               (("portfolio", "portfolio_returns"), ("candidate", "candidate_returns")) if q.get(f)}
    uploads = {n: s for d in streams.values() for n, s in d.items()}
    cands = [str(c).upper() for c in q.get("candidates") or []] + list(streams.get("candidate", {}))
    if q.get("sweep"):
        cands = list(dict.fromkeys(cands + [str(q["sweep"]).upper()]))
    bases = _bases(q, streams)
    held = [t for p in bases if isinstance(p, CF.Profile) for t in p.weights]
    b = load_bundle(cfg, [t for t in cands + held if t not in uploads])
    b = with_streams(b, uploads, q.get("returns_currency", "CAD"))
    reg = E.regime_labels(cfg, b)
    objective = q.get("objective")
    req = E.Request(bases[0], objective, cands, q.get("benchmark"),
                    q.get("mode", "add"), q.get("weight"), q.get("replace_holding"),
                    q.get("rebalance"), q.get("alignment", "common"),
                    fund_from=q.get("fund_from"))
    rvs = E.run_many(req, bases, cfg, b, reg)
    sw = (E.sweep(cfg, b, str(q["sweep"]).upper(), objective or next(iter(cfg.objectives.objectives)),
                  q.get("mode", "add"), q.get("weight"), q.get("replace_holding"), reg,
                  fund_from=q.get("fund_from"))
          if q.get("sweep") else None)
    pd.set_option("display.width", 220)
    print()
    for rv in rvs:
        for c in rv.results:
            print("- " + RP.headline(rv, c))
        for c, why in rv.skipped.items():
            print(f"- {c} not analysed against {RP.base_label(rv)}: {why}")
    print()
    print(RP.summary_table(rvs).to_string(index=False))
    if sw is not None:
        print("\nProfile sweep:")
        print(sw[["Profile", "Verdict", "Confidence", "Score", "Change in volatility",
                  "Change in max drawdown"]].to_string(index=False))
    out = q.get("output") or os.path.splitext(path)[0] + ".xlsx"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(X.full_report(rvs, cfg, None, sw))
    print(f"\nWorkbook: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
