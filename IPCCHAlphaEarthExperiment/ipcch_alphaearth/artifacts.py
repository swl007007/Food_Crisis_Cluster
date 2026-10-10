"""Hashing, deterministic serialization, immutable run directories and INCOMPLETE records.

Hash/JSON/incomplete idioms adapted from IPCCHGeoXGBExperiment/ipcch_geoxgb/artifacts.py and
IPCCHYearlyGeoXGBExperiment/ipcch_yearly_xgb/modelstore.py (array_digest) at 6798df2.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth.errors import ContractError, TechnicalError

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
#: Path fragments that mark a Dropbox-synced location (design section 13: runs stay outside Dropbox).
_DROPBOX_MARKERS = ("dropbox",)


def sha256_file(path: Path | str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_digest(array) -> str:
    """SHA256 of dtype, shape and C-ordered bytes; object arrays are refused (their bytes are pointers)."""
    arr = np.ascontiguousarray(np.asarray(array))
    if arr.dtype.hasobject:
        raise TechnicalError("array_digest needs a numeric/fixed-width array")
    header = f"{arr.dtype.str}|{arr.shape}|".encode()
    return hashlib.sha256(header + arr.tobytes()).hexdigest()


def list_sha256(values) -> str:
    """Same convention as the sibling compact contract: newline-joined UTF-8."""
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path | str, payload: object, overwrite: bool = False) -> str:
    """Write JSON (never overwrite unless explicitly allowed) and return the file SHA256."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ContractError(f"refusing to overwrite existing artifact {path}")
    text = json.dumps(payload, indent=2, default=_json_default) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    tmp.replace(path)
    return sha256_file(path)


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv_gz(path: Path | str, frame: pd.DataFrame) -> str:
    """Deterministic gzip CSV (mtime 0, round-trip float precision through repr), never overwritten."""
    path = Path(path)
    if path.exists():
        raise ContractError(f"refusing to overwrite existing artifact {path}")
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format=None)
    data = buffer.getvalue().encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(data)
    return sha256_file(path)


def read_csv_gz(path: Path | str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def write_npy(path: Path | str, array: np.ndarray) -> str:
    path = Path(path)
    if path.exists():
        raise ContractError(f"refusing to overwrite existing artifact {path}")
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    return sha256_file(path)


def is_dropbox_path(path: Path | str) -> bool:
    text = str(Path(path).resolve()).lower()
    return any(marker in text for marker in _DROPBOX_MARKERS)


def new_run_dir(run_id: str, runs_root: Path | str) -> Path:
    """Create a fresh run directory outside Dropbox; existing run IDs are immutable."""
    if not _RUN_ID.match(run_id):
        raise ContractError(f"invalid run id {run_id!r}")
    root = Path(runs_root).resolve()
    if is_dropbox_path(root):
        raise ContractError(f"run root {root} is inside Dropbox; runs must live on the WSL filesystem")
    target = (root / run_id).resolve()
    if target.parent != root:
        raise ContractError(f"run directory {target} escapes {root}")
    if target.exists():
        raise ContractError(f"run directory already exists: {target}")
    target.mkdir(parents=True)
    return target


def require_run_dir(run_dir: Path | str, config_sha256: dict | None = None) -> Path:
    """Open an initialized run; refuses Dropbox paths and any drift from the run's frozen config identity.

    ``config_sha256`` defaults to the current ``config/`` identity; tests may pass an explicit identity.
    """
    run_dir = Path(run_dir).resolve()
    if is_dropbox_path(run_dir):
        raise ContractError(f"run directory {run_dir} is inside Dropbox")
    if not (run_dir / "run.json").exists():
        raise ContractError(f"{run_dir} is not an initialized run (missing run.json)")
    check_config_identity(run_dir, config_sha256)
    return run_dir


def check_config_identity(run_dir: Path, config_sha256: dict | None = None) -> None:
    if config_sha256 is None:
        from ipcch_alphaearth.contract import config_identity  # noqa: PLC0415

        config_sha256 = config_identity()
    retained = read_json(Path(run_dir) / "run.json").get("config_sha256")
    if retained != config_sha256:
        drift = sorted(k for k in set(retained or {}) | set(config_sha256) if (retained or {}).get(k) != config_sha256.get(k))
        raise ContractError(f"run {Path(run_dir).name} was created with a different frozen configuration ({drift}); "
                            "create a new run instead of reusing it")


def record_incomplete(run_dir: Path | str, stage: str, context: dict, error: BaseException) -> Path:
    """Durable ``<run>/<stage>/INCOMPLETE.json`` + ``<run>/RUN_INCOMPLETE.json``; partial files are listed, not deleted."""
    run_dir = Path(run_dir)
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    partial = sorted(str(p.relative_to(run_dir)) for p in stage_dir.rglob("*") if p.is_file())
    payload = {
        "status": "incomplete",
        "stage": stage,
        "context": context,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        "partial_evidence": partial,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    text = json.dumps(payload, indent=2, default=_json_default) + "\n"
    for path in (stage_dir / "INCOMPLETE.json", run_dir / "RUN_INCOMPLETE.json"):
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    return stage_dir / "INCOMPLETE.json"


def stage_complete(run_dir: Path, stage: str) -> bool:
    return (Path(run_dir) / stage / "COMPLETE.json").exists() and not (Path(run_dir) / stage / "INCOMPLETE.json").exists()


def require_stage(run_dir: Path, stage: str) -> None:
    """Stage prerequisites fail clearly; they never auto-launch the earlier stage."""
    if not stage_complete(run_dir, stage):
        raise ContractError(f"prerequisite stage '{stage}' is not COMPLETE in {run_dir}; run it explicitly first")


def inventory(root: Path) -> list[dict]:
    """Sorted file inventory (relative path, bytes, sha256)."""
    root = Path(root)
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows
