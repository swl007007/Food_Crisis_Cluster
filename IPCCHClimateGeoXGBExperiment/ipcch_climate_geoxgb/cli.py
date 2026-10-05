"""Command-line entrypoint: ``python -m ipcch_climate_geoxgb <command>``.

Run with the package directory on the import path, e.g. from the repository
root ``PYTHONPATH=IPCCHClimateGeoXGBExperiment python -m ipcch_climate_geoxgb validate-config``.

Exit codes: 0 success, 1 contract failure, 2 runtime does not match the lock,
3 scientific phase not implemented yet (nothing written), 4 R41 technical failure,
5 replay found a failed check.
"""

from __future__ import annotations

import argparse
import json
import sys

from ipcch_climate_geoxgb import __version__
from ipcch_climate_geoxgb.errors import ContractError, NotImplementedPhaseError, TechnicalError

#: Scientific phases; each fails explicitly until its implementation lands.
PENDING_PHASES: dict[str, str] = {}


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_validate_config(_args) -> int:
    from ipcch_climate_geoxgb.contract import validate_all  # noqa: PLC0415

    _print({"status": "passed", "versions": validate_all()})
    return 0


def cmd_runtime_probe(_args) -> int:
    from ipcch_climate_geoxgb.runtime import probe_runtime  # noqa: PLC0415

    result = probe_runtime()
    _print(result)
    return 0 if result["matches_lock"] else 2


def cmd_preflight(args) -> int:
    from ipcch_climate_geoxgb.artifacts import new_run_dir, write_json  # noqa: PLC0415
    from ipcch_climate_geoxgb.contract import validate_all  # noqa: PLC0415
    from ipcch_climate_geoxgb.preflight import run_preflight  # noqa: PLC0415
    from ipcch_climate_geoxgb.runtime import probe_runtime  # noqa: PLC0415

    runtime = probe_runtime()
    if not runtime["matches_lock"]:
        _print({"status": "failed", "reason": "runtime does not match lock", "runtime": runtime})
        return 2
    versions = validate_all()
    run_dir = new_run_dir(args.run_id)
    out = run_dir / "preflight"
    out.mkdir()
    report = {"package_version": __version__, "config_versions": versions, "runtime": runtime}
    try:
        report.update(run_preflight())
    except ContractError as error:
        report.update({"status": "failed", "error": str(error)})
        write_json(out / "preflight-report.json", report)
        _print({"status": "failed", "error": str(error), "report": str(out / "preflight-report.json")})
        return 1
    digest = write_json(out / "preflight-report.json", report)
    _print({"status": report["status"], "report": str(out / "preflight-report.json"), "sha256": digest})
    return 0


def cmd_prepare(args) -> int:
    from ipcch_climate_geoxgb.artifacts import new_run_dir  # noqa: PLC0415
    from ipcch_climate_geoxgb.prepare import run_prepare  # noqa: PLC0415
    from ipcch_climate_geoxgb.runtime import probe_runtime  # noqa: PLC0415

    runtime = probe_runtime()
    if not runtime["matches_lock"]:
        _print({"status": "failed", "reason": "runtime does not match lock", "runtime": runtime})
        return 2
    manifest = run_prepare(new_run_dir(args.run_id))
    _print({"status": "passed", "manifest_sha256": manifest["manifest_sha256"], "ledger": manifest["ledger"],
            "folds": manifest["folds"], "stage1_split": manifest["stage1_split"]})
    return 0


def existing_run(run_id: str):
    from ipcch_climate_geoxgb import RUNS_DIR  # noqa: PLC0415

    run_dir = RUNS_DIR / run_id
    if not (run_dir / "prepared" / "prepared-manifest.json").is_file():
        raise ContractError(f"run {run_id!r} has no completed prepare stage")
    if (run_dir / "RUN_INCOMPLETE.json").exists():
        raise ContractError(f"run {run_id!r} is marked incomplete (R41); start a new run id after the fix")
    return run_dir


def cmd_learn_map(args) -> int:
    from ipcch_climate_geoxgb.learnmap import run_learn_map  # noqa: PLC0415

    summary = run_learn_map(existing_run(args.run_id))
    _print({"status": "passed", "summary_sha256": summary["summary_sha256"],
            "winners": {h: v["winner"] for h, v in summary["horizons"].items()},
            "model_store": summary["model_store"]})
    return 0


def cmd_predict(args) -> int:
    from ipcch_climate_geoxgb.predict import run_predict  # noqa: PLC0415

    run_dir = existing_run(args.run_id)
    if not all((run_dir / "stage1" / f"frozen_h{h:02d}.json").is_file() for h in (1, 3, 6, 12)):
        raise ContractError(f"run {args.run_id!r} has no frozen Stage1 maps for every H")
    summary = run_predict(run_dir)
    _print({"status": "passed", "summary_sha256": summary["summary_sha256"],
            "horizons": summary["horizons"], "model_store": summary["model_store"]})
    return 0


def cmd_report(args) -> int:
    from ipcch_climate_geoxgb.report import run_report  # noqa: PLC0415

    run_dir = existing_run(args.run_id)
    if not (run_dir / "stage3" / "stage3-summary.json").is_file():
        raise ContractError(f"run {args.run_id!r} has no completed Stage3")
    report = run_report(run_dir)
    _print({"status": "passed", "report_sha256": report["report_sha256"]})
    return 0


def cmd_replay(args) -> int:
    from ipcch_climate_geoxgb.artifacts import write_json  # noqa: PLC0415
    from ipcch_climate_geoxgb.contract import load_experiment_contract  # noqa: PLC0415
    from ipcch_climate_geoxgb.replay import replay_run  # noqa: PLC0415

    run_dir = existing_run(args.run_id)
    result = replay_run(run_dir, load_experiment_contract())
    digest = write_json(run_dir / "report" / f"replay-{args.label}.json", result)
    _print({"status": result["status"], "failures": result["failures"][:20], "replay_sha256": digest,
            "checks_passed": sum(result["passed_checks"].values())})
    return 0 if result["status"] == "passed" else 5


def cmd_pending(args) -> int:
    raise NotImplementedPhaseError(
        f"'{args.command}' belongs to {PENDING_PHASES[args.command]} and is not implemented "
        "in this foundation build; no artifacts were written"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipcch_climate_geoxgb", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config", help="validate config/*.json").set_defaults(func=cmd_validate_config)
    sub.add_parser("runtime-probe", help="compare the interpreter with the lock").set_defaults(
        func=cmd_runtime_probe
    )
    pre = sub.add_parser("preflight", help="read-only input/geometry/cache preflight")
    pre.add_argument("--run-id", required=True, help="new run directory under runs/")
    pre.set_defaults(func=cmd_preflight)
    prep = sub.add_parser("prepare", help="P1: QC ledger, rich601 matrices, F/S split, calendars")
    prep.add_argument("--run-id", required=True, help="new run directory under runs/")
    prep.set_defaults(func=cmd_prepare)
    learn = sub.add_parser("learn-map", help="P3: Stage1 search for all H x G x L, R44 selection, frozen maps")
    learn.add_argument("--run-id", required=True, help="existing run with a completed prepare stage")
    learn.set_defaults(func=cmd_learn_map)
    pred = sub.add_parser("predict", help="P4: rolling Stage3 with frozen maps, gates, pooled and persistence")
    pred.add_argument("--run-id", required=True, help="existing run with prepare and learn-map completed")
    pred.set_defaults(func=cmd_predict)
    rep = sub.add_parser("report", help="P5: metrics, paired deltas, coverage, R49 bootstrap from saved predictions")
    rep.add_argument("--run-id", required=True)
    rep.set_defaults(func=cmd_report)
    rpl = sub.add_parser("replay", help="P5: independent recomputation of a completed run from saved evidence")
    rpl.add_argument("--run-id", required=True)
    rpl.add_argument("--label", default="independent", help="name of the replay record under report/")
    rpl.set_defaults(func=cmd_replay)
    for name, phase in PENDING_PHASES.items():
        sub.add_parser(name, help=f"not implemented yet: {phase}").set_defaults(func=cmd_pending)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Unported phases accept (and ignore) any arguments so they always reach
    # the explicit not-implemented failure; other commands stay strict.
    args, extra = parser.parse_known_args(argv)
    if extra and args.command not in PENDING_PHASES:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        return args.func(args)
    except NotImplementedPhaseError as error:
        print(f"NOT IMPLEMENTED: {error}", file=sys.stderr)
        return 3
    except ContractError as error:
        print(f"CONTRACT FAILURE: {error}", file=sys.stderr)
        return 1
    except TechnicalError as error:
        print(f"TECHNICAL FAILURE (R41, run incomplete): {error}", file=sys.stderr)
        return 4
