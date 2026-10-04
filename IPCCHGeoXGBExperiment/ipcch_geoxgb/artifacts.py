"""Hashing, JSON output and run-directory discipline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ipcch_geoxgb import RUNS_DIR
from ipcch_geoxgb.errors import ContractError

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """Stream a file's SHA256 without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path | str, payload: object) -> str:
    """Write JSON once (never overwrite) and return the written file's SHA256."""
    path = Path(path)
    if path.exists():
        raise ContractError(f"refusing to overwrite existing artifact {path}")
    text = json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_file(path)


def new_run_dir(run_id: str, runs_root: Path | str = RUNS_DIR) -> Path:
    """Create a fresh run directory; existing run IDs are immutable."""
    if not _RUN_ID.match(run_id):
        raise ContractError(f"invalid run id {run_id!r}")
    runs_root = Path(runs_root).resolve()
    target = (runs_root / run_id).resolve()
    if target.parent != runs_root:
        raise ContractError(f"run directory {target} escapes {runs_root}")
    if target.exists():
        raise ContractError(f"run directory already exists: {target}")
    target.mkdir(parents=True)
    return target
