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
                path = Path(raw)
            else:
                path = Path("/mnt") / raw[0].lower() / raw[3:]
            return self._resolve_absolute_reference(path)

        path = Path(raw)
        if path.is_absolute():
            return self._resolve_absolute_reference(path)

        root_candidate = self.repo_root / path
        if root_candidate.exists():
            return root_candidate

        archived_candidate = self._archived_input_candidate(path)
        if archived_candidate is not None:
            return archived_candidate

        return root_candidate

    def _resolve_absolute_reference(self, path: Path) -> Path:
        """Resolve an absolute path, redirecting moved repo inputs to the archive."""
        if path.exists():
            return path

        relative_path = self._relative_to_repo_root(path)
        if relative_path is not None:
            archived_candidate = self._archived_input_candidate(relative_path)
            if archived_candidate is not None:
                return archived_candidate

        return path

    def _relative_to_repo_root(self, path: Path) -> Path | None:
        """Return path relative to repo_root when the absolute path is inside it."""
        try:
            return path.relative_to(self.repo_root)
        except ValueError:
            pass

        try:
            return path.resolve(strict=False).relative_to(
                self.repo_root.resolve(strict=False)
            )
        except (OSError, ValueError):
            return None

    def _archived_input_candidate(self, relative_path: Path) -> Path | None:
        """Return archived path for reproducibility inputs formerly at repo root."""
        if relative_path.parts and relative_path.parts[0] in REPRODUCIBILITY_INPUT_NAMES:
            return self.reproducibility_inputs_root / relative_path
        return None

    def require_file(self, path: Path, label: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        return path

    def require_dir(self, path: Path, label: str) -> Path:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")
        return path
