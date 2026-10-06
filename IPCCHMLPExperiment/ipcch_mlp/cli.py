"""Command line: ``python -m ipcch_mlp <command> --run-dir DIR``.

Commands: validate-config, probe, timing, preflight, develop, predict, report,
replay. ``preflight`` and ``timing`` never touch project-data fitting; project
stages refuse to run until the runtime-lock device is frozen. Run directories
are new and immutable; scratch runs belong outside Dropbox.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import shutil
import sys
import time
from pathlib import Path

from ipcch_mlp import contract as C
from ipcch_mlp.artifacts import record_incomplete, write_json
from ipcch_mlp.errors import ContractError, TechnicalError

DEFAULT_RUNS = Path(r"C:\Users\swl00\AppData\Local\Temp\ipcch-mlp-runs")


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _ram() -> dict:
    if platform.system() != "Windows":
        return {}

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"total_gb": round(m.ullTotalPhys / 2**30, 1), "available_gb": round(m.ullAvailPhys / 2**30, 1)}


def _horizons(contract):
    from ipcch_mlp import sources
    return {h: sources.load_horizon(h) for h in contract["horizons_months"]}


def _enumerate(contract, horizons):
    from ipcch_mlp import planning, sources
    cal = sources.load_calendar()
    out, problems = {}, []
    for h, hz in horizons.items():
        r = planning.enumerate_horizon(hz, cal[cal["horizon_months"] == h], contract)
        exp = contract["expected"]["per_horizon"][str(h)]
        problems += [f"H{h} {k}: {r.get(k)} != {v}" for k, v in exp.items() if r.get(k) != v]
        if len(hz.regions) != contract["maps"]["terminal_regions"][str(h)] or len(hz.region_of) != contract["maps"]["mapped_areas"]:
            problems.append(f"H{h}: map regions/areas differ")
        out[str(h)] = r
    total = 3 * sum(r["scalar_fits_per_seed"] for r in out.values()) + contract["expected"]["development_scalar_fits"]
    if total != contract["expected"]["total_scalar_fits"]:
        problems.append(f"total {total} != {contract['expected']['total_scalar_fits']}")
    if max(r["global_pool_max_rows"] for r in out.values()) != contract["expected"]["max_global_pool_rows"]:
        problems.append("maximum global pool differs")
    return out, total, problems


def cmd_validate(_a) -> int:
    _print({"status": "passed", "versions": C.validate_all()})
    return 0


def cmd_probe(_a) -> int:
    from ipcch_mlp import runtime
    info = runtime.probe()
    _print(info)
    return 0 if info["matches_lock"] else 2


def cmd_timing(a) -> int:
    from ipcch_mlp import runtime, sources, timing
    contract = C.load_contract()
    lock = C.load_runtime_lock()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.parallel:
        device = a.device or runtime.frozen_device(lock)
        res = timing.probe_parallel(device, lock["numerics"]["cpu_threads"], a.parallel)
        write_json(out / "timing-parallel.json", {"device": device, "processes": a.parallel, **res,
                                                   "note": "synthetic data only"})
        _print({"device": device, "wall_1": round(res["1"]["wall_seconds"], 1),
                f"wall_{a.parallel}": round(res[str(a.parallel)]["wall_seconds"], 1),
                "throughput_speedup": round(res["throughput_speedup"], 2),
                "identical": res["identical_to_single"] and res[str(a.parallel)]["identical_across_processes"]})
        return 0
    info = runtime.probe(lock)
    if not info["matches_lock"]:
        raise ContractError(f"runtime mismatch: {info['mismatches']}")
    horizons = _horizons(contract)
    enumeration, total, problems = _enumerate(contract, horizons)
    if problems:
        raise ContractError(f"enumeration mismatch: {problems}")
    devices = ["cpu"] + (["cuda"] if info["cuda_available"] else [])
    results, estimates = {}, {}
    for d in devices:
        started = time.time()
        results[d] = timing.probe_device(d, contract, lock["numerics"]["cpu_threads"])
        results[d]["probe_wall_seconds"] = round(time.time() - started, 1)
        estimates[d] = timing.estimate(results[d]["cases"], enumeration, contract["development"]["fit_keys"])
    choice = timing.choose_device(results, estimates)
    payload = {"runtime": info, "results": results, "estimates_seconds": estimates, "chosen_device": choice or None,
               "rule": "faster eligible device by the upper full-run estimate; ties choose CPU; synthetic data only",
               "total_scalar_fits": total}
    write_json(out / "timing.json", payload)
    _print({"chosen_device": choice or None,
            "estimate_hours": {d: {k: [round(x / 3600, 2) for x in (v["stage3_seconds_three_seeds"] if k != "develop_seconds" else v)]
                                   for k, v in e.items()} for d, e in estimates.items()},
            "eligible": {d: r["eligible"] for d, r in results.items()}})
    return 0 if choice else 1


def _new_run(a) -> Path:
    from ipcch_mlp.artifacts import new_run_dir
    return new_run_dir(Path(a.run_dir).name, Path(a.run_dir).parent)


def cmd_preflight(a) -> int:
    from ipcch_mlp import runtime, sources
    contract = C.load_contract()
    lock = C.load_runtime_lock()
    run_dir = _new_run(a)
    try:
        info = runtime.probe(lock)
        if not info["matches_lock"]:
            raise ContractError(f"runtime mismatch: {info['mismatches']}")
        verified = sources.verify()
        horizons = _horizons(contract)
        enumeration, total, problems = _enumerate(contract, horizons)
        if problems:
            raise ContractError(f"enumeration mismatch: {problems}")
        disk = shutil.disk_usage(run_dir)
        report = {"stage": "P0-preflight", "runtime": info, "device_frozen": lock["device"],
                  "sources": verified, "enumeration": {h: {k: v for k, v in r.items() if not k.endswith("_sizes")}
                                                       for h, r in enumeration.items()},
                  "total_scalar_fits": total, "disk_free_gb": round(disk.free / 2**30, 1), "ram": _ram(),
                  "gpu": info["cuda_device"], "contract": contract["contract_version"]}
        (run_dir / "preflight").mkdir()
        digest = write_json(run_dir / "preflight" / "preflight.json", report)
    except Exception as error:
        record_incomplete(run_dir, "preflight", {"stage": "preflight"}, error)
        raise
    _print({"status": "passed", "sha256": digest, "total_scalar_fits": total, "disk_free_gb": report["disk_free_gb"],
            "device_frozen": lock["device"]})
    return 0


def _project_context(run_dir: Path, need_preflight: bool = True):
    from ipcch_mlp import runtime, sources
    from ipcch_mlp.quartets import Engine
    from ipcch_mlp.store import ModelStore
    contract = C.load_contract()
    lock = C.load_runtime_lock()
    device = runtime.frozen_device(lock)
    if need_preflight and not (run_dir / "preflight" / "preflight.json").is_file():
        raise ContractError("run has no passed preflight")
    if (run_dir / "RUN_INCOMPLETE.json").exists():
        raise ContractError("run is marked incomplete")
    runtime.configure(device, lock["numerics"]["cpu_threads"])
    env = runtime.environment_identity(device, lock)
    sources.verify()
    horizons = _horizons(contract)
    return contract, env, device, horizons


def real_context() -> dict:
    """Worker loader for parallel mode: same checks as the parent, in the child process."""
    from ipcch_mlp import runtime, sources
    contract = C.load_contract()
    lock = C.load_runtime_lock()
    device = runtime.frozen_device(lock)
    runtime.configure(device, lock["numerics"]["cpu_threads"])
    env = runtime.environment_identity(device, lock)
    return {"contract": contract, "env": env, "device": device, "horizons": _horizons(contract),
            "split": sources.load_split(), "calendar": sources.load_calendar()}


REAL_LOADER = "ipcch_mlp.cli:real_context"


def _stage(run_dir: Path, name: str, fn):
    try:
        return fn()
    except Exception as error:
        record_incomplete(run_dir, name, {"stage": name}, error)
        raise


def cmd_develop(a) -> int:
    from ipcch_mlp import develop, sources
    from ipcch_mlp.quartets import Engine
    from ipcch_mlp.store import ModelStore
    run_dir = Path(a.run_dir)
    contract, env, device, horizons = _project_context(run_dir)
    t0 = time.time()
    if a.serial:
        store = ModelStore(run_dir / "models", run_dir / "model_requests.jsonl")
        engine = Engine(store, contract, env, device)
        res = _stage(run_dir, "develop", lambda: develop.run_develop(run_dir, engine, contract, horizons,
                                                                      sources.load_split()))
    else:
        from ipcch_mlp import parallel
        res = _stage(run_dir, "develop", lambda: parallel.run_develop_parallel(run_dir, contract, REAL_LOADER))
    _print({"status": "passed", "mode": "serial" if a.serial else "replicate-parallel",
            "winners": {h: v["winner"] for h, v in res["horizons"].items()}, "seconds": round(time.time() - t0, 1)})
    return 0


def cmd_predict(a) -> int:
    from ipcch_mlp import develop, sources, stage3
    from ipcch_mlp.quartets import Engine
    from ipcch_mlp.store import ModelStore
    run_dir = Path(a.run_dir)
    contract, env, device, horizons = _project_context(run_dir)
    winners = develop.load_winners(run_dir)
    t0 = time.time()
    if a.serial:
        store = ModelStore(run_dir / "models", run_dir / "model_requests.jsonl")
        engine = Engine(store, contract, env, device)
        res = _stage(run_dir, "stage3", lambda: stage3.run_stage3(run_dir, engine, contract, horizons,
                                                                   sources.load_calendar(), winners))
    else:
        from ipcch_mlp import parallel
        res = _stage(run_dir, "stage3", lambda: parallel.run_stage3_parallel(run_dir, contract, REAL_LOADER, winners))
    _print({"status": "passed", "mode": "serial" if a.serial else "replicate-parallel",
            "seconds": round(time.time() - t0, 1),
            "new_fits": {r: v["new_scalar_fits"] for r, v in res["replicates"].items()}})
    return 0


def cmd_report(a) -> int:
    from ipcch_mlp import report
    run_dir = Path(a.run_dir)
    contract = C.load_contract()
    res = _stage(run_dir, "report", lambda: report.run_report(run_dir, contract))
    _print({"status": "passed", **res})
    return 0


def cmd_replay(a) -> int:
    from ipcch_mlp import replay, sources
    run_dir = Path(a.run_dir)
    contract, env, device, horizons = _project_context(run_dir)
    res = replay.run_replay(run_dir, contract, env, device, horizons, sources.load_split(), sources.load_calendar())
    _print({k: v for k, v in res.items() if k != "failures"} | {"first_failures": res["failures"][:10]})
    return 0 if res["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ipcch_mlp")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config").set_defaults(func=cmd_validate)
    sub.add_parser("probe").set_defaults(func=cmd_probe)
    t = sub.add_parser("timing")
    t.add_argument("--out", required=True)
    t.add_argument("--parallel", type=int, default=0, help="probe N concurrent processes (synthetic data)")
    t.add_argument("--device", choices=("cpu", "cuda"), default=None, help="parallel probe device (default: frozen)")
    t.set_defaults(func=cmd_timing)
    for name, fn in (("preflight", cmd_preflight), ("develop", cmd_develop), ("predict", cmd_predict),
                     ("report", cmd_report), ("replay", cmd_replay)):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", required=True)
        s.set_defaults(func=fn)
        if name in ("develop", "predict"):
            s.add_argument("--serial", action="store_true",
                           help="run replicates one after another in this process (default: replicate-parallel)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ContractError, TechnicalError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
