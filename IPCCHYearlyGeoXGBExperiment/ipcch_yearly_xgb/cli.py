"""Command line: ``run_experiment.py [--config DIR] --run-dir DIR <command>`` (or ``-m ipcch_yearly_xgb``).

Commands: validate-config, probe, timing, preflight, predict, report, replay.
``preflight`` and ``timing`` never fit project data. The config directory is
fixed to the package's frozen config; a different --config is refused. Run
directories are new and immutable and belong outside Dropbox.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ipcch_yearly_xgb import CONFIG_DIR
from ipcch_yearly_xgb import contract as C
from ipcch_yearly_xgb.artifacts import record_incomplete, write_json
from ipcch_yearly_xgb.errors import ContractError, TechnicalError


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _horizons(contract):
    from ipcch_yearly_xgb import sources
    return {h: sources.load_horizon(h, contract["recipes"][str(h)]) for h in contract["horizons_months"]}


def _enumerate(contract, horizons, calendar):
    from ipcch_yearly_xgb import planning
    out, problems = {}, []
    exp = contract["expected"]
    for h, hz in horizons.items():
        r = planning.enumerate_horizon(hz, calendar, contract)
        problems += [f"H{h}:{k}" for k in planning.compare(r, exp["per_horizon"][str(h)])]
        if len(hz.regions) != contract["maps"]["terminal_regions"][str(h)] or len(hz.region_of) != 3264:
            problems.append(f"H{h}: map regions/areas")
        main = sum(b["eval_keys"] for b in r["blocks"] if b["period"] == "main")
        supp = sum(b["eval_keys"] for b in r["blocks"] if b["period"] == "supplementary")
        dmain = sum(b["diagnostic_keys"] for b in r["blocks"] if b["period"] == "main")
        dsupp = sum(b["diagnostic_keys"] for b in r["blocks"] if b["period"] == "supplementary")
        if (main, supp, dmain, dsupp) != (exp["main_full_keys"][str(h)], exp["supplementary_full_keys"],
                                          exp["main_diagnostic_keys"][str(h)], exp["supplementary_diagnostic_keys"]):
            problems.append(f"H{h}: cohort counts {(main, supp, dmain, dsupp)}")
        out[str(h)] = r
    total = sum(r["scalar_fits"] for r in out.values())
    if total != exp["total_scalar_fits"] or sum(r["main"]["scalar_fits"] for r in out.values()) != exp["main_scalar_fits"]:
        problems.append(f"total fits {total}")
    nblocks = sum(len(r["blocks"]) for r in out.values())
    if nblocks != contract["calendar"]["current_blocks"]:
        problems.append(f"current blocks {nblocks}")
    if len(calendar) != contract["calendar"]["planned_folds"] or int((calendar["eval_keys"] > 0).sum()) != \
            contract["calendar"]["scored_folds"]:
        problems.append("fold calendar counts")
    return out, total, problems


def _check_config(a) -> None:
    if a.config and Path(a.config).resolve() != CONFIG_DIR.resolve():
        raise ContractError(f"--config must be the frozen package config {CONFIG_DIR}")


def cmd_validate(a) -> int:
    _check_config(a)
    _print({"status": "passed", "versions": C.validate_all()})
    return 0


def cmd_probe(a) -> int:
    from ipcch_yearly_xgb import runtime
    info = runtime.probe()
    _print({**info, "fit_source_sha256": runtime.code_digest(), "source_inventory": runtime.source_inventory()})
    return 0 if info["matches_lock"] else 2


def cmd_timing(a) -> int:
    from ipcch_yearly_xgb import runtime, sources, timing
    contract = C.load_contract()
    out = Path(a.run_dir)
    out.mkdir(parents=True, exist_ok=False)
    info = runtime.probe()
    if not info["matches_lock"]:
        raise ContractError(f"runtime mismatch: {info['mismatches']}")
    enumeration, _, problems = _enumerate(contract, _horizons(contract), sources.load_calendar())
    if problems:
        raise ContractError(f"enumeration mismatch: {problems}")
    res = timing.probe(contract, enumeration)
    write_json(out / "timing.json", {"runtime": info, **res, "note": "synthetic data only"})
    _print({"estimate_seconds": res["estimate_seconds"]})
    return 0


def cmd_preflight(a) -> int:
    from ipcch_yearly_xgb import runtime, sources
    from ipcch_yearly_xgb.artifacts import new_run_dir
    _check_config(a)
    contract = C.load_contract()
    run_dir = new_run_dir(Path(a.run_dir).name, Path(a.run_dir).parent)
    try:
        info = runtime.probe()
        if not info["matches_lock"]:
            raise ContractError(f"runtime mismatch: {info['mismatches']}")
        verified = sources.verify()
        calendar = sources.load_calendar()
        horizons = _horizons(contract)
        enumeration, total, problems = _enumerate(contract, horizons, calendar)
        if problems:
            raise ContractError(f"enumeration mismatch: {problems}")
        disk = shutil.disk_usage(run_dir)
        rep = {"stage": "preflight", "runtime": info, "env": runtime.environment_identity(),
               "source_inventory": runtime.source_inventory(), "inputs_verified": verified,
               "lineage": {h: hz.lineage | {"map_sha256": hz.map_sha256} for h, hz in horizons.items()},
               "enumeration": enumeration, "total_scalar_fits": total, "disk_free_gb": round(disk.free / 2**30, 1),
               "contract": contract["contract_version"]}
        (run_dir / "preflight").mkdir()
        digest = write_json(run_dir / "preflight" / "preflight.json", rep)
    except Exception as error:
        record_incomplete(run_dir, "preflight", {"stage": "preflight"}, error)
        raise
    _print({"status": "passed", "sha256": digest, "total_scalar_fits": total, "disk_free_gb": rep["disk_free_gb"]})
    return 0


def _context(run_dir: Path):
    from ipcch_yearly_xgb import runtime, sources
    if not (run_dir / "preflight" / "preflight.json").is_file():
        raise ContractError("run has no passed preflight")
    if (run_dir / "RUN_INCOMPLETE.json").exists():
        raise ContractError("run is marked incomplete")
    contract = C.load_contract()
    env = runtime.environment_identity()
    pre = json.loads((run_dir / "preflight" / "preflight.json").read_text(encoding="utf-8"))
    if pre["env"] != env:
        raise ContractError("runtime/fit-source identity differs from the run's preflight")
    sources.verify()
    return contract, env, runtime.source_inventory(), _horizons(contract), sources.load_calendar()


def cmd_predict(a) -> int:
    from ipcch_yearly_xgb import run
    _check_config(a)
    run_dir = Path(a.run_dir)
    contract, env, inv, horizons, calendar = _context(run_dir)
    res = run.run_predict(run_dir, horizons, calendar, contract, env, inv)
    _print({"status": "passed", "store": res["store_counts"], "seconds": res["elapsed_seconds"],
            "horizons": res["horizons"]})
    return 0


def cmd_report(a) -> int:
    from ipcch_yearly_xgb import report
    _check_config(a)
    run_dir = Path(a.run_dir)
    try:
        res = report.run_report(run_dir, C.load_contract())
    except Exception as error:
        record_incomplete(run_dir, "report", {"stage": "report"}, error)
        raise
    _print({"status": "passed", **res})
    return 0


def cmd_replay(a) -> int:
    from ipcch_yearly_xgb import replay
    _check_config(a)
    run_dir = Path(a.run_dir)
    contract, env, inv, horizons, calendar = _context(run_dir)
    res = replay.run_replay(run_dir, horizons, calendar, contract, env, inv,
                            expect_fits=contract["expected"]["total_scalar_fits"])
    _print({k: v for k, v in res.items() if k != "failures"} | {"first_failures": res["failures"][:10]})
    return 0 if res["status"] == "passed" else 1


COMMANDS = {"validate-config": cmd_validate, "probe": cmd_probe, "timing": cmd_timing, "preflight": cmd_preflight,
            "predict": cmd_predict, "report": cmd_report, "replay": cmd_replay}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ipcch_yearly_xgb")
    p.add_argument("--config", default=None, help="frozen config directory (must be the package config)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("command", choices=sorted(COMMANDS))
    a = p.parse_args(argv)
    if a.command in ("timing", "preflight", "predict", "report", "replay") and not a.run_dir:
        p.error("--run-dir is required")
    try:
        return COMMANDS[a.command](a)
    except (ContractError, TechnicalError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
