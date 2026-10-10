"""Stage pinned inputs into ``<run>/sources`` (byte copy outside Dropbox, fresh SHA256) and code inventory.

A staged file whose fresh hash differs from its pinned value stops staging. Multi-file bundles (the shapefile
components) are staged together under ``sources/<bundle>/`` with their original filenames, so readers that locate
sibling components by basename keep working. Reusing a run's staged sources requires the retained manifest's
request identity (inputs version, layout, every name/source path/pinned hash/bundle) to equal the current
request; any drift refuses reuse instead of returning the old inventory. Downstream stages read only staged
copies, so later Dropbox edits cannot change a run's inputs.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ipcch_alphaearth import EXPERIMENT_ROOT, REPO_ROOT
from ipcch_alphaearth.artifacts import digest, read_json, sha256_file, write_json
from ipcch_alphaearth.errors import ContractError

SOURCE_MANIFEST = "source-manifest.json"
LAYOUT_VERSION = "sources-layout-v2-bundles"


def staged_relpath(name: str, item: dict) -> Path:
    original = Path(item["path"]).name
    if item.get("bundle"):
        return Path("sources") / item["bundle"] / original
    return Path("sources") / f"{name}__{original}"


def request_identity(inputs_cfg: dict) -> dict:
    """What a run's staged sources must correspond to (compared on every reuse)."""
    entries = {name: {"path": item["path"], "sha256": item.get("sha256"), "bundle": item.get("bundle")}
               for name, item in sorted(inputs_cfg["inputs"].items())}
    return {"inputs_version": inputs_cfg["inputs_version"], "layout": LAYOUT_VERSION, "entries": entries,
            "digest": digest({"v": inputs_cfg["inputs_version"], "layout": LAYOUT_VERSION, "entries": entries})}


def _check_bundles(inputs_cfg: dict) -> None:
    seen = {}
    for name, item in inputs_cfg["inputs"].items():
        rel = staged_relpath(name, item)
        if rel in seen:
            raise ContractError(f"inputs {seen[rel]} and {name} would stage to the same path {rel}")
        seen[rel] = name


def stage_inputs(run_dir: Path, inputs_cfg: dict, retries: int = 3) -> dict:
    """Copy every pinned input once; on reuse, refuse any request/layout drift and re-verify staged bytes."""
    run_dir = Path(run_dir)
    out = run_dir / "sources"
    manifest_path = out / SOURCE_MANIFEST
    request = request_identity(inputs_cfg)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        retained = manifest.get("request_identity", {}).get("digest")
        if retained != request["digest"]:
            raise ContractError("staged sources were made for a different input request/layout "
                                f"(retained {retained}, requested {request['digest']}); create a new run")
        verify_staged(run_dir, manifest)
        return manifest
    _check_bundles(inputs_cfg)
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, item in inputs_cfg["inputs"].items():
        src = Path(item["path"])
        if not src.exists():
            raise ContractError(f"input {name} missing: {src}")
        rel = staged_relpath(name, item)
        dst = run_dir / rel
        if dst.exists():
            raise ContractError(f"refusing to overwrite staged file {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                shutil.copyfile(src, dst)
                break
            except OSError as error:  # Dropbox sync can raise transient Input/output errors under WSL
                if attempt == retries - 1:
                    raise ContractError(f"copy of {name} failed after {retries} attempts: {error}") from error
                time.sleep(5)
        sha = sha256_file(dst)
        if item.get("sha256") and sha != item["sha256"]:
            raise ContractError(f"staged {name} sha256 {sha} differs from pinned {item['sha256']}")
        entries[name] = {"source_path": str(src), "staged": str(rel), "bytes": dst.stat().st_size, "sha256": sha,
                         "pinned_sha256": item.get("sha256"), "bundle": item.get("bundle"), "role": item.get("role")}
    manifest = {"inputs_version": inputs_cfg["inputs_version"], "layout": LAYOUT_VERSION, "request_identity": request,
                "staged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": entries}
    write_json(manifest_path, manifest)
    return manifest


def verify_staged(run_dir: Path, manifest: dict) -> None:
    for name, e in manifest["inputs"].items():
        path = Path(run_dir) / e["staged"]
        if not path.exists() or sha256_file(path) != e["sha256"]:
            raise ContractError(f"staged source {name} changed or is missing: {path}")


def staged_path(run_dir: Path, manifest: dict, name: str) -> Path:
    return Path(run_dir) / manifest["inputs"][name]["staged"]


def code_inventory() -> dict:
    """SHA256 of every package/config/script file of this experiment (the fitting-source identity)."""
    files = {}
    for sub in ("ipcch_alphaearth", "config", "scripts"):
        for path in sorted((EXPERIMENT_ROOT / sub).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".json"):
                files[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    return files


def code_identity() -> str:
    return digest(code_inventory())
