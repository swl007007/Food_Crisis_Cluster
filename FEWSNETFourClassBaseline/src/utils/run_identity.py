"""Identities that bind every stage output to its exact inputs, code and runtime.

Stages never continue into existing output. Downstream stages accept an upstream
output only when its recorded identity equals the current one and every recorded
output file still matches its hash (audit finding A02).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PACKAGE / "feature-schema.json"
CODE_ROOTS = ("app", "scripts", "src")
CODE_FILES = ("config.py", "config_visual.py", "feature-schema.json", "run_all.sh")
#: Read-only checking code. It never writes a run artifact, so it is identified
#: separately (verifier identity) and excluded from the PRODUCER identity that binds
#: runs; a later verifier fix therefore does not misattribute or invalidate a run.
VERIFIER_FILES = ("scripts/verify_fourclass.py",)
RUNTIME_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "geopandas", "shapely", "polars")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity() -> dict:
    files = [PACKAGE / name for name in CODE_FILES]
    for root in CODE_ROOTS:
        files += [p for p in (PACKAGE / root).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix in (".py", ".sh", ".json")]
    listing = sorted((p.relative_to(PACKAGE).as_posix(), file_sha256(p)) for p in files
                     if p.relative_to(PACKAGE).as_posix() not in VERIFIER_FILES)
    digest = hashlib.sha256(json.dumps(listing).encode()).hexdigest()
    return {"sha256": digest, "files": len(listing)}


def verifier_identity() -> dict:
    return {rel: file_sha256(PACKAGE / rel) for rel in VERIFIER_FILES}


def code_identity_at(rev: str = "HEAD") -> dict:
    """The same identity computed from committed Git blobs (audit A01): a run is bound
    to committed code only if its recorded identity equals this for the audited commit."""
    import subprocess
    repo = PACKAGE.parent
    prefix = PACKAGE.name + "/"
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, prefix], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    wanted = []
    for full in names:
        rel = full[len(prefix):]
        top = rel.split("/")[0]
        if rel in VERIFIER_FILES:
            continue
        if rel in CODE_FILES or (top in CODE_ROOTS and "__pycache__" not in rel
                                 and Path(rel).suffix in (".py", ".sh", ".json")):
            wanted.append(rel)
    listing = []
    for rel in wanted:
        blob = subprocess.run(["git", "show", f"{rev}:{prefix}{rel}"], cwd=repo,
                              capture_output=True, check=True).stdout
        listing.append((rel, hashlib.sha256(blob).hexdigest()))
    listing.sort()
    return {"sha256": hashlib.sha256(json.dumps(listing).encode()).hexdigest(), "files": len(listing)}


def git_head() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE.parent, capture_output=True,
                          text=True, check=True).stdout.strip()


def runtime_identity() -> dict:
    from importlib.metadata import version
    return {"python": platform.python_version(), **{name: version(name) for name in RUNTIME_PACKAGES}}


def output_hashes(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): file_sha256(p)
            for p in sorted(Path(directory).rglob("*"))
            if p.is_file() and p.name not in ("outputs.json", "identity.json") and not p.name.startswith(".")}


def verify_outputs(directory: Path, recorded: dict) -> list:
    """Return problems: missing, changed or unrecorded files."""
    directory = Path(directory)
    problems = [f"missing {rel}" for rel in recorded if not (directory / rel).is_file()]
    problems += [f"changed {rel}" for rel, sha in recorded.items()
                 if (directory / rel).is_file() and file_sha256(directory / rel) != sha]
    actual = output_hashes(directory)
    problems += [f"unrecorded {rel}" for rel in actual if rel not in recorded]
    return problems


#: Files every completed stage unit MUST record (audit A04): an accepted record must
#: list each of these, and each listed file must exist with its recorded hash.
REQUIRED_PREPARED = ("manifests/sources.json", "manifests/runtime.json", "manifests/preflight.json",
                     "manifests/schedule.json", "manifests/features.json", "manifests/geometry.json",
                     "ledgers/observations.csv", "ledgers/baselines.csv",
                     "snapshot_h4.parquet", "snapshot_h8.parquet", "snapshot_h12.parquet",
                     "geometry/polygon_contiguity_info.pkl", "geometry/FEWSNET_admin_code_lat_lon.csv")
REQUIRED_STAGE1_FOLD = ("candidate.json", "correspondence_table.csv", "target_predictions.csv",
                        "heldout_scores.csv", "fold_membership.csv.gz", "s_branch.pkl",
                        "branch_table.npy", "X_branch_id.npy", "command.json", "run.log")
REQUIRED_STAGE3_FOLD = ("local_support.csv", "imputer_statistics.csv.gz", "training_keys.csv.gz",
                        "models/pooled.pkl.xz")


def check_inventory(base: Path, recorded: dict, required) -> list:
    """Problems if a required file is not recorded, or a recorded file is missing/changed."""
    base = Path(base)
    problems = [f"required {rel} not recorded" for rel in required if rel not in recorded]
    problems += [f"missing {rel}" for rel in recorded if not (base / rel).is_file()]
    problems += [f"changed {rel}" for rel, sha in recorded.items()
                 if (base / rel).is_file() and file_sha256(base / rel) != sha]
    return problems


def write_json_atomic(path: Path, payload) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def refuse_existing(path: Path, what: str) -> None:
    if Path(path).exists():
        raise FileExistsError(f"{path} exists: {what} never continues into existing output; "
                              "use a fresh run directory")


def require_prepared(run: Path) -> dict:
    """Accept a preparation only if its identity equals the current code/runtime and
    every recorded output is present and unchanged."""
    prepared = Path(run) / "prepared"
    marker = prepared / "manifests" / "identity.json"
    if not marker.is_file():
        raise RuntimeError(f"{prepared}: no completion record; preparation is incomplete")
    identity = json.loads(marker.read_text(encoding="utf-8"))
    outputs_path = prepared / "manifests" / "outputs.json"
    if file_sha256(outputs_path) != identity["outputs_sha256"]:
        raise RuntimeError("prepared outputs.json differs from its completion record")
    recorded = json.loads(outputs_path.read_text(encoding="utf-8"))
    problems = verify_outputs(prepared, recorded) + check_inventory(prepared, recorded, REQUIRED_PREPARED)
    if identity.get("stage") != "prepare":
        problems.append("completion record is not a preparation record")
    if problems:
        raise RuntimeError(f"prepared outputs do not match their record: {problems[:5]}")
    if identity["code"] != code_identity() or identity["runtime"] != runtime_identity():
        raise RuntimeError("prepared run was produced by different package code or runtime")
    return identity
