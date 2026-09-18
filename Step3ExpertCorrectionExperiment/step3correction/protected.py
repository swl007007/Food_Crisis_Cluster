"""Protected-artifact hashing for the Step 3 expert selective-correction experiment.

Nothing under ``paper_reproducibility_package/``, ``archived/``, ``other_outputs/``
or any existing ``result_*`` directory may be created, modified or deleted by this
experiment.  This module records SHA-256 digests of every protected input that the
experiment reads so a before/after comparison can prove they are untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "Step3ExpertCorrectionExperiment"
PACKAGE = ROOT / "paper_reproducibility_package"
ARCHIVE = ROOT / "archived/release_20260624_reproducibility_inputs"
LEGACY_EVALUATOR = ROOT / (
    "archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final/"
    "fewsnet_baseline_evaluation.py"
)

# External source data (outside the repository).
FEWSNET_SOURCE = ROOT.parents[2] / "1.Source Data/Outcome/FEWSNET_IPC/FEWSNET.csv"
PANEL_SOURCE = ROOT.parents[2] / "1.Source Data/FEWSNET_forecast_unadjusted_bm.csv"

# Frozen Stage 3 contig3 month maps (Phase 3 inputs) and their Phase 2 parents.
REFINED_MAP_NAMES = {
    "general": "cluster_mapping_k40_nc17_general_refined_contig3.csv",
    "m2": "cluster_mapping_k40_nc13_m2_refined_contig3.csv",
    "m6": "cluster_mapping_k40_nc11_m6_refined_contig3.csv",
    "m10": "cluster_mapping_k40_nc16_m10_refined_contig3.csv",
}
STAGE2_MAP_NAMES = {
    "general": "cluster_mapping_k40_nc17_general.csv",
    "m2": "cluster_mapping_k40_nc13_m2.csv",
    "m6": "cluster_mapping_k40_nc11_m6.csv",
    "m10": "cluster_mapping_k40_nc16_m10.csv",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it fully in memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def refined_map_path(scope: int, key: str) -> Path:
    """Return the frozen contig3 partition map for ``scope`` and month key."""
    return PACKAGE / f"stage3_results/georf_fs{scope}/refined" / REFINED_MAP_NAMES[key]


def stage2_map_path(key: str) -> Path:
    """Return the Phase 2 parent map for a month key."""
    return PACKAGE / "stage2_cluster_maps/georf" / STAGE2_MAP_NAMES[key]


def package_stage3_path(scope: int, name: str) -> Path:
    """Return a frozen package Stage 3 artifact path."""
    return PACKAGE / f"stage3_results/georf_fs{scope}" / name


def archive_stage3_path(scope: int, name: str) -> Path:
    """Return the archived twin of a package Stage 3 artifact."""
    return ARCHIVE / f"result_partition_k40_compare_GF_fs{scope}" / name


def archived_expert_baseline_path(scope: int) -> Path:
    """Return the archived FEWS NET expert baseline summary for fs1/fs2."""
    return ARCHIVE / "fewsnet_baseline_results" / f"fewsnet_baseline_results_fs{scope}.csv"


def protected_paths() -> List[Path]:
    """Return every protected file this experiment reads, in a stable order."""
    paths: List[Path] = [FEWSNET_SOURCE, PANEL_SOURCE, LEGACY_EVALUATOR]
    for scope in (1, 2, 3):
        for name in ("predictions_monthly.csv", "metrics_monthly.csv", "run_manifest.json"):
            paths.append(package_stage3_path(scope, name))
            paths.append(archive_stage3_path(scope, name))
    for scope in (1, 2):
        paths.append(archived_expert_baseline_path(scope))
    for key in REFINED_MAP_NAMES:
        paths.append(refined_map_path(1, key))
        paths.append(stage2_map_path(key))
    return paths


def hash_protected(paths: Iterable[Path] | None = None) -> Dict[str, str]:
    """Hash all protected inputs, raising when one is missing."""
    selected = list(protected_paths() if paths is None else paths)
    digests: Dict[str, str] = {}
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(f"Protected input missing: {path}")
        digests[str(path)] = sha256(path)
    return digests


def assert_unchanged(before: Dict[str, str]) -> None:
    """Re-hash the recorded inputs and fail loudly on any drift."""
    after = hash_protected([Path(p) for p in before])
    drift = {p: (before[p], after[p]) for p in before if before[p] != after[p]}
    if drift:
        raise RuntimeError(f"Protected artifact hash drift detected: {drift}")


def write_hash_report(destination: Path, before: Dict[str, str], after: Dict[str, str]) -> None:
    """Persist a before/after protected-hash report."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "unchanged": before == after,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
