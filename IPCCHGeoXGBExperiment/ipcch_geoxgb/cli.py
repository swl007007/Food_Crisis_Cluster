"""Command-line entrypoint: ``python -m ipcch_geoxgb <command>``.

Run with the package directory on the import path, e.g. from the repository
root ``PYTHONPATH=IPCCHGeoXGBExperiment python -m ipcch_geoxgb validate-config``.

Exit codes: 0 success, 1 contract failure, 2 runtime does not match the lock,
3 scientific phase not implemented yet (nothing written).
"""

from __future__ import annotations

import argparse
import json
import sys

from ipcch_geoxgb import __version__
from ipcch_geoxgb.errors import ContractError, NotImplementedPhaseError

#: Scientific phases; each fails explicitly until its implementation lands.
PENDING_PHASES = {
    "prepare": "P1 (population targets, calendar, rich561)",
    "learn-map": "P3 (direct Stage1 maps and frozen winners)",
    "predict": "P4 (rolling Stage3 and paired baselines)",
    "report": "P5 (reporting and independent replay)",
}


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_validate_config(_args) -> int:
    from ipcch_geoxgb.contract import validate_all  # noqa: PLC0415

    _print({"status": "passed", "versions": validate_all()})
    return 0


def cmd_runtime_probe(_args) -> int:
    from ipcch_geoxgb.runtime import probe_runtime  # noqa: PLC0415

    result = probe_runtime()
    _print(result)
    return 0 if result["matches_lock"] else 2


def cmd_preflight(args) -> int:
    from ipcch_geoxgb.artifacts import new_run_dir, write_json  # noqa: PLC0415
    from ipcch_geoxgb.contract import validate_all  # noqa: PLC0415
    from ipcch_geoxgb.preflight import run_preflight  # noqa: PLC0415
    from ipcch_geoxgb.runtime import probe_runtime  # noqa: PLC0415

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


def cmd_pending(args) -> int:
    raise NotImplementedPhaseError(
        f"'{args.command}' belongs to {PENDING_PHASES[args.command]} and is not implemented "
        "in this foundation build; no artifacts were written"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipcch_geoxgb", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config", help="validate config/*.json").set_defaults(func=cmd_validate_config)
    sub.add_parser("runtime-probe", help="compare the interpreter with the lock").set_defaults(
        func=cmd_runtime_probe
    )
    pre = sub.add_parser("preflight", help="read-only input/geometry/cache preflight")
    pre.add_argument("--run-id", required=True, help="new run directory under runs/")
    pre.set_defaults(func=cmd_preflight)
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
