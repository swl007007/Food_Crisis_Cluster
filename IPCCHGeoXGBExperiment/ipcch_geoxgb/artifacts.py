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


def record_incomplete(run_dir: Path | str, stage: str, context: dict, error: BaseException) -> Path:
    """R41: durable incomplete-run record with context, cause and retained partial evidence.

    Writes ``<run>/<stage>/INCOMPLETE.json`` and ``<run>/RUN_INCOMPLETE.json``;
    the run is not resumed or counted complete afterwards. Partial files are
    listed, not deleted.
    """
    import time  # noqa: PLC0415
    import traceback  # noqa: PLC0415

    run_dir = Path(run_dir)
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    partial = sorted(str(p.relative_to(run_dir)).replace("\\", "/") for p in stage_dir.rglob("*") if p.is_file())
    payload = {
        "status": "incomplete",
        "stage": stage,
        "context": context,
        "error_type": type(error).__name__,
        "error": str(error),
        "notes": list(getattr(error, "__notes__", [])),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        "partial_evidence": partial,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    text = json.dumps(payload, indent=2, default=str) + "\n"
    for path in (stage_dir / "INCOMPLETE.json", run_dir / "RUN_INCOMPLETE.json"):
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    return stage_dir / "INCOMPLETE.json"


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
