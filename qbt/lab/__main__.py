"""Command line: `python -m qbt.lab <command>`.

    demo      end-to-end run on synthetic data, offline (a smoke test you can read)
    mcp       start the MCP server on stdio
    tools     list the tools and their arguments
    call      call any tool:  python -m qbt.lab call run_backtest '{"dataset_id": "...", "strategy": "sma_trend"}'
    research  one-shot study of a real universe: screen factors, backtest the strategy, validate, write the
              run card and dossier
"""
from __future__ import annotations

import argparse
import json
import sys

from . import tools as T


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def cmd_demo(args) -> int:
    d = T.load_data(synthetic=True, edge=0.05)
    did = d["dataset_id"]
    hid = T.add_hypothesis(title="Demo: 12-1 momentum on a synthetic universe")["hypothesis_id"]
    sc = T.factor_screen(dataset_id=did, hypothesis_id=hid, top=5)
    print(f"Screened {sc['n_factors']} factors; {sc['survivors_after_fdr_10pct']} survive FDR at 10%.")
    for r in sc["top"][:5]:
        print(f"  {r['Factor']:<32} IC {r['Mean IC']:+.3f}  t {r['t-stat']:+.2f}  p_bh {r['p_bh']:.3f}")
    run = T.run_backtest(dataset_id=did, strategy="zoo_factor",
                         params={"factor": "mom_12_1", "top_n": 3}, hypothesis_id=hid)
    print("\nSharpe", round(run["metrics"]["Sharpe"], 2), "| CAGR", f"{run['metrics']['CAGR']:.1%}")
    v = T.validate_run_tool(run_id=run["run_id"], grid={"top_n": [2, 3, 4], "factor": ["mom_12_1", "mom_6_1", "mom_126"]})
    print(f"Grade: {v['grade']['label']} ({v['grade']['passes']} pass, {v['grade']['warnings']} warn, "
          f"{v['grade']['fails']} fail; {v['n_trials_counted']} trials counted)")
    for c in v["checks"]:
        print(f"  {c['status'].upper():<5} {c['check']}")
    card = T.save_run_card(run_id=run["run_id"])
    print("\nDossier:", card["dossier"])
    return 0


def cmd_research(args) -> int:
    d = T.load_data(tickers=args.tickers, start=args.start, end=args.end, benchmark=args.benchmark)
    if "error" in d:
        _print(d)
        return 1
    did = d["dataset_id"]
    hid = T.add_hypothesis(title=args.title or f"{args.strategy} on {','.join(args.tickers[:4])}")["hypothesis_id"]
    sc = T.factor_screen(dataset_id=did, hypothesis_id=hid, top=8)
    print(f"{sc['n_factors']} factors screened, {sc['survivors_after_fdr_10pct']} survive FDR at 10%")
    params = json.loads(args.params) if args.params else {}
    run = T.run_backtest(dataset_id=did, strategy=args.strategy, params=params, hypothesis_id=hid)
    if "error" in run:
        _print(run)
        return 1
    grid = json.loads(args.grid) if args.grid else None
    v = T.validate_run_tool(run_id=run["run_id"], grid=grid)
    print(f"Sharpe {run['metrics']['Sharpe']:.2f} | Grade {v['grade']['label']}")
    card = T.save_run_card(run_id=run["run_id"])
    print("Dossier:", card["dossier"])
    return 0


def cmd_tools(args) -> int:
    for t in T.TOOLS.values():
        print(f"{t['name']:<20} {t['description'][:90]}")
    return 0


def cmd_call(args) -> int:
    if args.name not in T.TOOLS:
        print(f"Unknown tool {args.name}. See: python -m qbt.lab tools", file=sys.stderr)
        return 2
    _print(T.TOOLS[args.name]["fn"](**json.loads(args.args or "{}")))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qbt.lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    sub.add_parser("tools").set_defaults(fn=cmd_tools)
    sub.add_parser("mcp").set_defaults(fn=lambda a: __import__("qbt.lab.mcp_server", fromlist=["serve"]).serve() or 0)
    c = sub.add_parser("call")
    c.add_argument("name")
    c.add_argument("args", nargs="?")
    c.set_defaults(fn=cmd_call)
    r = sub.add_parser("research")
    r.add_argument("tickers", nargs="+")
    r.add_argument("--start", default="2012-01-01")
    r.add_argument("--end")
    r.add_argument("--benchmark")
    r.add_argument("--strategy", default="zoo_factor")
    r.add_argument("--params", help="JSON")
    r.add_argument("--grid", help="JSON, e.g. '{\"top_n\":[2,3,4]}'")
    r.add_argument("--title")
    r.set_defaults(fn=cmd_research)
    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
