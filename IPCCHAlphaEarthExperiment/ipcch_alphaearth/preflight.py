"""Preflight: runtime lock, frozen contract, input availability (and optional fresh hashes), code inventory.

Never fits, never contacts EE, never launches another stage. Creates the immutable run directory on first use.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ipcch_alphaearth import REPO_ROOT
from ipcch_alphaearth.artifacts import check_config_identity, is_dropbox_path, read_json, sha256_file, write_json
from ipcch_alphaearth.contract import config_identity, load_contract, load_inputs
from ipcch_alphaearth.errors import ContractError
from ipcch_alphaearth.runtime import probe_numerical
from ipcch_alphaearth.schedule import all_blocks, check_fit_ceiling
from ipcch_alphaearth.sources import code_identity, code_inventory


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()

    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": run("status", "--porcelain", "--", "IPCCHAlphaEarthExperiment")}


def init_run(run_dir: Path, kind: str = "developer") -> Path:
    """Create (or reopen, refusing config drift) a run. ``kind``: developer (disposable) or formal (frozen evidence)."""
    if kind not in ("developer", "formal"):
        raise ContractError(f"unknown run kind {kind!r}")
    run_dir = Path(run_dir).expanduser().resolve()
    if is_dropbox_path(run_dir):
        raise ContractError(f"{run_dir} is inside Dropbox; runs must live on the WSL filesystem")
    if (run_dir / "run.json").exists():
        check_config_identity(run_dir)
        retained = read_json(run_dir / "run.json").get("kind", "developer")
        if retained != kind:
            raise ContractError(f"run {run_dir.name} is a {retained} run; a run's kind is fixed at creation (never relabelled)")
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ContractError(f"{run_dir} exists and is not an initialized run")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run.json", {"run_id": run_dir.name, "kind": kind, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "config_sha256": config_identity(), "code_identity": code_identity(), "git": git_state()})
    return run_dir


def run_preflight(run_dir: Path, hash_inputs: bool = False, kind: str = "developer") -> dict:
    run_dir = init_run(run_dir, kind)
    contract = load_contract()
    inputs = load_inputs()
    runtime = probe_numerical()
    blocks = all_blocks(contract)
    present = {}
    for name, item in inputs["inputs"].items():
        path = Path(item["path"])
        entry = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
        if hash_inputs and path.exists():
            entry["sha256"] = sha256_file(path)
            entry["matches_pinned"] = entry["sha256"] == item["sha256"]
        present[name] = entry
    report = {
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": read_json(run_dir / "run.json"),
        "runtime": runtime,
        "contract_version": contract["contract_version"],
        "config_sha256": config_identity(),
        "fit_ceiling": check_fit_ceiling(contract, blocks),
        "inputs": present,
        "code_inventory": code_inventory(),
        "git": git_state(),
        "no_fit_no_ee": True,
    }
    problems = []
    if not runtime["matches_lock"]:
        problems += runtime["mismatches"]
    problems += [f"input {n} missing" for n, e in present.items() if not e["exists"]]
    problems += [f"input {n} hash mismatch" for n, e in present.items() if e.get("matches_pinned") is False]
    report["problems"] = problems
    report["pass"] = not problems
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (run_dir / "preflight").mkdir(exist_ok=True)
    write_json(run_dir / "preflight" / f"preflight-{stamp}.json", report)
    if problems:
        raise ContractError("preflight failed: " + "; ".join(problems))
    return report
