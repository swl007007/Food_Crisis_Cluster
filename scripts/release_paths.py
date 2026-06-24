"""Central path contract for the paper reproducibility release layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPRODUCIBILITY_INPUT_NAMES = frozenset(
    {
        "GeoRFExperiment",
        "GeoDTExperiment",
        "main_ablation_exclude_updated_stage3_fixed_partitions",
        "fewsnet_baseline_results",
        "result_partition_k40_compare_GF_fs1",
        "result_partition_k40_compare_GF_fs2",
        "result_partition_k40_compare_GF_fs3",
        "result_partition_k40_compare_DT_fs1",
        "result_partition_k40_compare_DT_fs2",
        "result_partition_k40_compare_DT_fs3",
        "result_partition_k40_compare_GF_thresholded_fs1",
        "result_partition_k40_compare_GF_thresholded_fs2",
        "result_partition_k40_compare_GF_thresholded_fs3",
    }
)


@dataclass(frozen=True)
class ReleasePaths:
    """Resolve release artifact paths after the clean-root archive move."""

    repo_root: Path

    @classmethod
    def default(cls) -> "ReleasePaths":
        return cls(repo_root=Path(__file__).resolve().parents[1])

    @property
    def reproducibility_inputs_root(self) -> Path:
        return self.repo_root / "archived" / "release_20260624_reproducibility_inputs"

    @property
    def legacy_workspace_root(self) -> Path:
        return self.repo_root / "archived" / "release_20260624_legacy_workspace"

    @property
    def local_residue_root(self) -> Path:
        return self.repo_root / "archived" / "local_workspace_residue_20260624"

    @property
    def final_artifacts_root(self) -> Path:
        return self.repo_root / "final_artifacts_in_paper_updated"

    @property
    def package_root(self) -> Path:
        return self.repo_root / "paper_reproducibility_package"

    @property
    def ablation_root(self) -> Path:
        return self.reproducibility_inputs_root / "main_ablation_exclude_updated_stage3_fixed_partitions"

    @property
    def fewsnet_baseline_root(self) -> Path:
        return self.reproducibility_inputs_root / "fewsnet_baseline_results"

    def experiment_root(self, token: str) -> Path:
        normalized = token.upper()
        if normalized == "GF":
            return self.reproducibility_inputs_root / "GeoRFExperiment"
        if normalized == "DT":
            return self.reproducibility_inputs_root / "GeoDTExperiment"
        raise ValueError(f"Unknown experiment token: {token}")

    def stage3_root(self, token: str, scope: int) -> Path:
        normalized = token.upper()
        if normalized not in {"GF", "DT"}:
            raise ValueError(f"Unknown Stage 3 token: {token}")
        if scope not in {1, 2, 3}:
            raise ValueError(f"Stage 3 scope must be 1, 2, or 3, got {scope}")
        return self.reproducibility_inputs_root / f"result_partition_k40_compare_{normalized}_fs{scope}"

    def thresholded_georf_root(self, scope: int) -> Path:
        if scope not in {1, 2, 3}:
            raise ValueError(f"Thresholded GeoRF scope must be 1, 2, or 3, got {scope}")
        return self.reproducibility_inputs_root / f"result_partition_k40_compare_GF_thresholded_fs{scope}"

    def resolve_repo_reference(self, value: str | Path) -> Path:
        """Resolve manifest paths from old root layout or new archive layout."""
        raw = str(value).replace("\\", "/")
        if len(raw) >= 3 and raw[1:3] == ":/":
            if os.name == "nt":
                return Path(raw)
            return Path("/mnt") / raw[0].lower() / raw[3:]

        path = Path(raw)
        if path.is_absolute():
            return path

        root_candidate = self.repo_root / path
        if root_candidate.exists():
            return root_candidate

        if path.parts and path.parts[0] in REPRODUCIBILITY_INPUT_NAMES:
            archived_candidate = self.reproducibility_inputs_root / path
            if archived_candidate.exists():
                return archived_candidate
            return archived_candidate

        return root_candidate

    def require_file(self, path: Path, label: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        return path

    def require_dir(self, path: Path, label: str) -> Path:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")
        return path
