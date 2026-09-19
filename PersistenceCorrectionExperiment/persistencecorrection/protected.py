"""Protected-artifact hashing for the persistence-correction experiment (PRD R26).

This is a thin wrapper over the Step 3 gate
(``Step3ExpertCorrectionExperiment/step3correction/protected.py``).  The Step 3
helpers are reused by import - never copied and never edited - and the path list
is extended with the frozen artifacts this task additionally reads:

* ``paper_reproducibility_package/stage3_results/georf_fs{1,2}/predictions_monthly.csv``
  (already covered by the Step 3 list; re-declared here so the intent is explicit
  and so a future narrowing of the Step 3 list cannot silently drop them);
* the four ``refined/cluster_mapping_k40_*_refined_contig3.csv`` month maps under
  ``stage3_results/georf_fs{1,2}/refined/`` - the Step 3 list covers fs1 only.

The union is de-duplicated while preserving order, so the extension can never
hash the same file twice or reorder the Step 3 entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from . import ROOT  # noqa: F401  (import for the step3correction sys.path bootstrap)

from step3correction.protected import (  # noqa: E402
    ARCHIVE,
    FEWSNET_SOURCE,
    PACKAGE,
    PANEL_SOURCE,
    REFINED_MAP_NAMES,
    assert_unchanged,
    hash_protected as _hash_paths,
    package_stage3_path,
    refined_map_path,
    sha256,
)
from step3correction.protected import protected_paths as step3_protected_paths  # noqa: E402

#: Scopes this experiment reads frozen Stage 3 artifacts for (PRD R7).
SCOPES = (1, 2)


class ProtectedPathError(ValueError):
    """Raised when an output path would escape this experiment's output tree."""


EXPERIMENT_DIR = ROOT / "PersistenceCorrectionExperiment"
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"


def additional_protected_paths() -> List[Path]:
    """Frozen inputs this experiment reads on top of the Step 3 list."""
    paths: List[Path] = []
    for scope in SCOPES:
        paths.append(package_stage3_path(scope, "predictions_monthly.csv"))
        for key in REFINED_MAP_NAMES:
            paths.append(refined_map_path(scope, key))
    return paths


def protected_paths() -> List[Path]:
    """Step 3 protected inputs plus this experiment's additions, de-duplicated."""
    ordered: List[Path] = []
    seen = set()
    for path in list(step3_protected_paths()) + additional_protected_paths():
        key = str(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    return ordered


def hash_protected(paths: Iterable[Path] | None = None) -> Dict[str, str]:
    """Hash every protected input, raising when one is missing."""
    return _hash_paths(protected_paths() if paths is None else paths)


def resolve_output_path(relative: str | Path) -> Path:
    """Resolve ``relative`` inside this experiment's output tree, or refuse.

    Mirrors the Step 3 rule: an output path outside the experiment's own
    ``outputs/`` directory raises before any directory is created (PRD R26).
    """
    candidate = Path(relative)
    resolved = (candidate if candidate.is_absolute() else OUTPUT_ROOT / candidate).resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ProtectedPathError(
            f"Refusing to write outside {root}: {resolved}"
        )
    return resolved


def write_baseline_report(destination: str | Path = "protected_hashes_baseline.json") -> Path:
    """Record the current protected-hash set under ``outputs/``."""
    target = resolve_output_path(destination)
    digests = hash_protected()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protected_file_count": len(digests),
        "step3_path_count": len(step3_protected_paths()),
        "additional_path_count": len(additional_protected_paths()),
        "protected_hashes": digests,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


__all__ = [
    "ARCHIVE",
    "FEWSNET_SOURCE",
    "OUTPUT_ROOT",
    "PACKAGE",
    "PANEL_SOURCE",
    "ProtectedPathError",
    "SCOPES",
    "additional_protected_paths",
    "assert_unchanged",
    "hash_protected",
    "package_stage3_path",
    "protected_paths",
    "refined_map_path",
    "resolve_output_path",
    "sha256",
    "write_baseline_report",
]
