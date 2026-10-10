"""Command line. Each stage checks its own prerequisites and never launches another stage implicitly.

    python -m ipcch_alphaearth preflight --run RUN_DIR [--hash-inputs]
    python -m ipcch_alphaearth prepare --run RUN_DIR --local-only
    python -m ipcch_alphaearth benchmark --out DIR [--prepared DIR]
    python -m ipcch_alphaearth develop|freeze-map|predict|report|replay --run RUN_DIR   (project-data fitting release only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ipcch_alphaearth")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--run", required=True)
    p.add_argument("--hash-inputs", action="store_true")
    p.add_argument("--kind", choices=("developer", "formal"), default="developer",
                   help="developer runs are disposable; formal runs are frozen evidence (set only at creation)")
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True)
    p.add_argument("--local-only", action="store_true", help="P0: no AlphaEarth descriptors (EE is never contacted)")
    p = sub.add_parser("benchmark")
    p.add_argument("--out", required=True)
    p.add_argument("--prepared", help="prepared stage directory (for the enumerated fit-pool sizes)")
    p = sub.add_parser("ingest-extraction", help="bind a COMPLETE D30 production extraction to a run (local only)")
    p.add_argument("--run", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--outputs", required=True)
    for name in ("develop", "freeze-map", "predict", "report", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
    args = parser.parse_args(argv)

    if args.command == "preflight":
        from ipcch_alphaearth.preflight import run_preflight

        report = run_preflight(Path(args.run), hash_inputs=args.hash_inputs, kind=args.kind)
        print(json.dumps({"pass": report["pass"], "runtime_matches_lock": report["runtime"]["matches_lock"],
                          "fit_ceiling": report["fit_ceiling"]}, indent=2))
        return 0
    if args.command == "prepare":
        from ipcch_alphaearth.artifacts import require_run_dir
        from ipcch_alphaearth.contract import load_contract, load_inputs
        from ipcch_alphaearth.prepare import run_prepare

        run_dir = require_run_dir(args.run)
        if not args.local_only:
            from ipcch_alphaearth.descriptors import attach_descriptors

            attach_descriptors(run_dir, load_contract())
            return 0
        run_prepare(run_dir, load_contract(), load_inputs(), local_only=True)
        return 0
    if args.command == "benchmark":
        from ipcch_alphaearth.benchmark import run_benchmark

        print(json.dumps(run_benchmark(Path(args.out), Path(args.prepared) if args.prepared else None), indent=2))
        return 0
    if args.command == "ingest-extraction":
        from ipcch_alphaearth.artifacts import require_run_dir
        from ipcch_alphaearth.contract import load_contract
        from ipcch_alphaearth.descriptors import ingest_extraction

        run_dir = require_run_dir(args.run)
        print(json.dumps(ingest_extraction(run_dir, load_contract(), Path(args.plan), Path(args.manifest), Path(args.outputs)), default=str))
        return 0
    if args.command in ("develop", "freeze-map", "predict", "report", "replay"):
        from ipcch_alphaearth import backbone, pipeline, replay, report
        from ipcch_alphaearth.artifacts import require_run_dir
        from ipcch_alphaearth.contract import load_contract
        from ipcch_alphaearth.runinputs import load_run_inputs
        from ipcch_alphaearth.runtime import environment_identity

        run_dir = require_run_dir(args.run)
        contract = load_contract()
        inputs = load_run_inputs(run_dir, contract)  # refuses before the P1 descriptor stage exists
        ws, env = pipeline.Workspace(run_dir), environment_identity()
        if args.command == "develop":
            result = pipeline.run_develop(inputs, contract, ws, backbone.ModelStore(run_dir / "models"), env)
        elif args.command == "freeze-map":
            result = pipeline.run_freeze_map(inputs, contract, ws)
        elif args.command == "predict":
            result = pipeline.run_predict(inputs, contract, ws, backbone.ModelStore(run_dir / "models"), env)
        elif args.command == "report":
            result = report.run_report(run_dir, contract, sorted(inputs.horizons))
        else:
            result = replay.run_replay(inputs, contract, run_dir, run_dir / "models")
        print(json.dumps({k: v for k, v in result.items() if k in ("stage", "selections", "summary", "pass", "n_checks", "failed",
                                                                   "qualifying_horizons")}, indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
