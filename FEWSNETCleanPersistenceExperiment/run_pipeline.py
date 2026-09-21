"""Stage 1 candidate worker, the recipe/role schedule and the Stage 2 consensus map.

Scope of this module: run the released GeoRF partition learner once per Stage 1
candidate job, in an isolated process, on origin-aligned inputs produced by
``prepare_data.py``; emit exactly the candidate score/assignment artifacts the
released Stage 2 helpers consume; then build each arm/role's frozen general
consensus map and its bounded geographic completion. Stage 3 forecasting,
calibration and recipe selection are NOT implemented here.

Contract anchors (see prd.md / design.md / research/*):

* R1/A1  - fresh run root, pinned release, run-local extraction, protected sources
           untouched; effective settings read back from the fitted estimator.
* R2/D4  - reuse the released F1/no-SMOTE core unchanged; keep its zero-feature
           class-recovery rows inside RF fitting only, never in real support counts.
* R24/D20 - training targets in [O-35 calendar months, O) with O = T - H; the
           released within-area split; nonempty real fitting/validation/target
           support; no minimum-history rule.
* D18/D22 - candidate identity is (arm, role window, target year, target month,
           forecasting scope); identical same-arm jobs in overlapping windows run
           once and are referenced by both roles.
* D23     - one max_plus imputer per candidate, fitted on real fitting rows only and
           reused unchanged for validation and target rows.
* R55/D51 - administrative identifiers stay out of the model-input schema; they are
           carried separately as grouping/routing metadata.
* R59/D55 - one general map per (arm, role window): all approved months and all of
           fs1/fs2/fs3 pooled; ``month_ind`` disabled; no monthly alternatives.
* R60/D56 - the released weight/Gaussian/normalization/k=40/eigengap formulas applied
           exactly, with no bandwidth, k or nc search and no forced nc>=2.
* R61/D57 - one core graph: components from positive off-diagonal affinity, largest
           by node count (ties by smallest canonical code), and both eigengap
           selection and spectral fitting on that same ordered submatrix.
* R62/D58 - node universe is the sorted union of areas that at least one eligible,
           normally completed plan assigns a non-``s-1`` partition.
* R63/D59 - bounded completion: haversine 1NN onto fitted-core donors, <=100 km,
           ties by smallest donor code; everything else stays partition -1.
* R64/D60 - no post-consensus contiguity smoothing, before or after completion.

Nothing here selects a recipe or scores a forecast.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_data as pdata  # noqa: E402


# --------------------------------------------------------------------------------------
# 1. Pinned release and run-local layout
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ZIP = REPO_ROOT / "GeoRFBaseline" / "releases" / "georf-baseline-v0.1.0.zip"
RELEASE_SHA256 = "39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0"
RELEASE_VERSION = "0.1.0-f1-nosmote"
RELEASE_SOURCE_COMMIT = "2dfa121a9398de9a1918ba9c0af34b31ecbb117a"

#: Stage 1 process seeds (research/runtime-and-preflight.md). The RF seed stays 5 and
#: is owned by the released RFmodel constructor, independently of these.
STAGE1_PROCESS_SEED = 42

#: Read back from the pinned config after import, never from our own kwargs.
REPORTED_CONFIG_KEYS: Tuple[str, ...] = (
    "MODEL_CHOICE", "MODE", "NUM_CLASS", "MIN_DEPTH", "MAX_DEPTH", "N_JOBS",
    "MIN_BRANCH_SAMPLE_SIZE", "MIN_SCAN_CLASS_SAMPLE", "FLEX_OPTION", "FLEX_RATIO",
    "FLEX_TYPE", "MIN_GROUP_POS_SAMPLE_SIZE_FLEX", "SIGLVL", "ES_THRD", "MD_THRD",
    "CONTIGUITY", "CONTIGUITY_TYPE", "REFINE_TIMES", "MIN_COMPONENT_SIZE",
    "USE_ADJACENCY_MATRIX", "ADJACENCY_POLYGON_ID_COLUMN",
    "GOVERNING_METRIC", "CRISIS_FOCUSED_OPTIMIZATION", "MIN_CLASS_1_IMPROVEMENT_THRESHOLD",
    "VAL_RATIO", "TRAIN_RATIO", "GROUP_SPLIT", "FEATURE_DROP", "feature_drop",
    "DISABLE_BASELINE_CV_MAP", "PRESERVE_ISOLATED_POLYGONS", "VIS_DEBUG_MODE",
)

RF_PARAM_KEYS: Tuple[str, ...] = (
    "n_estimators", "max_depth", "random_state", "n_jobs", "criterion",
    "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf", "max_features",
    "max_leaf_nodes", "min_impurity_decrease", "bootstrap", "oob_score", "warm_start",
    "class_weight", "ccp_alpha", "max_samples", "monotonic_cst",
)

#: Diagnostics deliberately bypassed through the release's own ImportError branch.
#: `create_pre_partition_diagnostics_cv` fits five extra cross-validation forests and
#: returns a value the release never uses. That alone does not make it side-effect
#: free: it also resets NumPy's global RNG. Equivalence is therefore established
#: empirically by verify_diagnostic_bypass.py, whose retained evidence shows every
#: scientific artifact byte-identical with the diagnostic on and off for the tested
#: candidate. GeoRF.fit:334-396 treats its absence as a supported condition, so no
#: source patch is required or applied.
DIAGNOSTIC_MODULE = "src.diagnostics.pre_partition_diagnostic"

#: The released Stage 1 rolling split restricts training rows to groups present in the
#: candidate's target month (src/customize/customize.py:439-445). D61 removes that
#: restriction for Stage 3 only, so Stage 1 keeps the released behavior here and
#: records both counts so the alternative remains measurable.
TRAIN_GROUP_FILTER = "released: training rows restricted to groups present in the target month"

STANDING_LIMITATIONS: Tuple[str, ...] = (
    "Retrospective run on already-inspected historical years; not a fresh holdout.",
    "Source vintages, Bloomberg units/roll conventions and publication timing remain "
    "unverified (D33/D49/D31); origin cutoffs constrain fitted artifacts, not source lineage.",
    "Stage 1 retains the released zero-feature class-recovery rows inside RF fitting; "
    "no SMOTE is not the absence of all artificial observations.",
    "Job counts are planning arithmetic before D20 support exclusions, not runtime.",
)


class PipelineError(RuntimeError):
    """A gate failed. The affected job/stage is recorded as failed, never as a result."""


# --------------------------------------------------------------------------------------
# 2. Small utilities
# --------------------------------------------------------------------------------------


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk_bytes: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unserializable value of type {type(value)!r}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=_json_default)
        handle.write("\n")


def read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def runtime_identity() -> Dict[str, object]:
    identity = pdata.runtime_identity()
    identity["thread_environment"] = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED")
    }
    return identity


# --------------------------------------------------------------------------------------
# 3. Verified run-local baseline extraction and import isolation
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class BaselineRuntime:
    """A verified, unpatched, run-local copy of the pinned release."""

    root: Path
    release_sha256: str
    manifest_version: str
    manifest_source_commit: str
    payload_files_verified: int
    module_locations: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        data = dict(self.__dict__)
        data["root"] = str(self.root)
        data["patched"] = False
        data["patch_note"] = (
            "No source patch is applied. IPCCH's polygon-refinement patch is specific to "
            "its own correspondence-table flow and is deliberately not imported."
        )
        return data


def extract_baseline(zip_path: Path, destination: Path) -> BaselineRuntime:
    """Verify then extract the pinned release into a fresh directory.

    Fatal in order: release SHA256, ZIP CRC of every member, a single top-level
    ``GeoRFBaseline/``, and every MANIFEST payload hash both inside the archive and
    again on disk after extraction.
    """
    if destination.exists() and any(destination.iterdir()):
        raise PipelineError(f"{destination} already exists and is not empty")

    actual = sha256_file(zip_path)
    if actual != RELEASE_SHA256:
        raise PipelineError(f"release hash mismatch: expected {RELEASE_SHA256}, got {actual}")

    with zipfile.ZipFile(zip_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise PipelineError(f"ZIP CRC failure at {corrupt}")
        tops = {name.split("/")[0] for name in archive.namelist()}
        if tops != {"GeoRFBaseline"}:
            raise PipelineError(f"expected one top-level GeoRFBaseline/, found {sorted(tops)}")
        manifest = json.loads(archive.read("GeoRFBaseline/MANIFEST.json"))
        payload = manifest["files_sha256"]
        for relative, expected in payload.items():
            try:
                data = archive.read(f"GeoRFBaseline/{relative}")
            except KeyError as exc:
                raise PipelineError(f"manifest lists a missing member: {relative}") from exc
            if sha256_bytes(data) != expected:
                raise PipelineError(f"in-archive payload hash mismatch: {relative}")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)

    root = destination / "GeoRFBaseline"
    verify_extracted_baseline(root)
    if manifest.get("version") != RELEASE_VERSION:
        raise PipelineError(f"unexpected release version {manifest.get('version')!r}")
    if manifest.get("source_commit") != RELEASE_SOURCE_COMMIT:
        raise PipelineError(f"unexpected source commit {manifest.get('source_commit')!r}")
    return BaselineRuntime(
        root=root,
        release_sha256=actual,
        manifest_version=str(manifest.get("version", "")),
        manifest_source_commit=str(manifest.get("source_commit", "")),
        payload_files_verified=len(payload),
    )


def verify_extracted_baseline(root: Path) -> BaselineRuntime:
    """Re-verify every MANIFEST payload hash on disk (cheap; run per worker)."""
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise PipelineError(f"extracted baseline has no MANIFEST.json at {root}")
    manifest = read_json(manifest_path)
    payload = manifest["files_sha256"]
    for relative, expected in payload.items():
        on_disk = root / relative
        if not on_disk.is_file():
            raise PipelineError(f"extracted baseline lost {relative}")
        if sha256_file(on_disk) != expected:
            raise PipelineError(f"extracted baseline payload modified: {relative}")
    return BaselineRuntime(
        root=root,
        release_sha256=RELEASE_SHA256,
        manifest_version=str(manifest.get("version", "")),
        manifest_source_commit=str(manifest.get("source_commit", "")),
        payload_files_verified=len(payload),
    )


@contextmanager
def baseline_imports(runtime: BaselineRuntime):
    """Import the pinned copy with it first on ``sys.path`` and assert the pin held.

    Two traps are closed explicitly:

    * ``src`` has no ``__init__.py``, so it is a namespace package whose ``__path__``
      is assembled from every ``sys.path`` entry holding a ``src/`` directory. The
      repository root has one, so a submodule absent from the release would silently
      resolve against unpinned code. ``src.__path__`` is pinned to the extracted copy.
    * ``from config import *`` copied ``FEATURE_DROP`` and its lowercase ``feature_drop``
      alias into the GeoRF module namespace at import time, and
      ``GeoRF._get_feature_drop_config`` reads *that* namespace. Both names are cleared
      in both places; our ordered schemas are already exact (D30/D52-D54), so the
      release's drop list must not remove columns behind our back.
    """
    root = str(runtime.root.resolve())
    saved_path = list(sys.path)
    saved_env = os.environ.get("PYTHONPATH")
    saved_modules = {
        name: module for name, module in sys.modules.items()
        if name in ("config", "config_visual") or name.split(".")[0] == "src"
    }
    for name in list(saved_modules):
        del sys.modules[name]
    os.environ.pop("PYTHONPATH", None)
    sys.path.insert(0, root)
    try:
        import config  # noqa: PLC0415

        import src  # noqa: PLC0415
        baseline_src = str((Path(root) / "src").resolve())
        src.__path__ = [baseline_src]

        # Force the release's own supported "diagnostic module not available" branch
        # instead of running five unused cross-validation forests per candidate.
        stub = type(sys)(DIAGNOSTIC_MODULE)
        stub.__file__ = str(Path(root) / "src" / "diagnostics" / "pre_partition_diagnostic.py")
        stub.__doc__ = "Run-local stub: diagnostics disabled; see DIAGNOSTIC_MODULE."
        sys.modules[DIAGNOSTIC_MODULE] = stub

        import src.model.GeoRF as georf_module  # noqa: PLC0415

        disabled_drop = {"enable": False, "cols": [], "patterns": []}
        for module in (config, georf_module):
            for attribute in ("FEATURE_DROP", "feature_drop"):
                if hasattr(module, attribute):
                    setattr(module, attribute, disabled_drop)

        locations = {"config": getattr(config, "__file__", ""), "src": baseline_src}
        for name, module in list(sys.modules.items()):
            if name != "config" and name.split(".")[0] != "src":
                continue
            if name == DIAGNOSTIC_MODULE:
                continue
            location = getattr(module, "__file__", None)
            if location:
                locations[name] = location
        for name, location in locations.items():
            if not location or not Path(location).resolve().is_relative_to(Path(root)):
                raise PipelineError(
                    f"module {name} resolved to {location!r}, outside the pinned baseline "
                    f"at {root}; a repository-root module is shadowing the release"
                )
        runtime.module_locations = locations
        yield config, georf_module
    finally:
        sys.path[:] = saved_path
        if saved_env is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved_env
        for name in list(sys.modules):
            if name in ("config", "config_visual") or name.split(".")[0] == "src":
                del sys.modules[name]
        sys.modules.update(saved_modules)


@contextmanager
def working_directory(path: Path):
    """GeoRF ignores its constructor directory, so its fit must run with cwd moved."""
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


# --------------------------------------------------------------------------------------
# 4. Run-local geometry (polygon contiguity info, built once per run)
# --------------------------------------------------------------------------------------


def build_polygon_contiguity_info(
    runtime: BaselineRuntime, areas: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    shapefile: Path, cache_dir: Path, verbose: bool = True,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Reproduce the released polygon grouping over the frozen master area universe.

    src/preprocess/preprocess.py:334-424 builds this from the whole panel before the
    monthly loop, so it is identical for every candidate and is built once per run.
    Group IDs are the admin codes themselves; ``polygon_group_mapping`` maps polygon
    index -> [admin code] exactly as the release does.
    """
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise PipelineError("master coordinates contain non-finite values")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with baseline_imports(runtime):
        from src.adjacency.adjacency_utils import load_or_create_adjacency_matrix  # noqa: PLC0415
        from src.customize.customize import PolygonGroupGenerator  # noqa: PLC0415

        adj_raw, polygon_id_mapping, _ = load_or_create_adjacency_matrix(
            shapefile_path=str(shapefile),
            polygon_id_column="admin_code",
            cache_dir=str(cache_dir),
            force_regenerate=False,
        )
        admin_to_polygon_idx = {int(code): idx for idx, code in enumerate(areas)}
        admin_code_to_adj_idx = {int(code): idx for idx, code in polygon_id_mapping.items()}

        adjacency_dict: Dict[int, np.ndarray] = {}
        for polygon_idx, code in enumerate(areas):
            adj_idx = admin_code_to_adj_idx.get(int(code))
            neighbors: List[int] = []
            if adj_idx is not None and adj_idx in adj_raw:
                for neighbor_adj_idx in adj_raw[adj_idx]:
                    neighbor_code = polygon_id_mapping.get(neighbor_adj_idx)
                    if neighbor_code is None:
                        continue
                    mapped = admin_to_polygon_idx.get(int(neighbor_code))
                    if mapped is not None:
                        neighbors.append(mapped)
            adjacency_dict[polygon_idx] = np.array(neighbors, dtype=int)

        generator = PolygonGroupGenerator(
            polygon_centroids=np.column_stack([lat, lon]),
            polygon_group_mapping={i: [int(areas[i])] for i in range(len(areas))},
            neighbor_distance_threshold=0.8,
            adjacency_dict=adjacency_dict,
        )
        info = generator.get_contiguity_info()

    covered = sum(1 for code in areas if int(code) in admin_code_to_adj_idx)
    degrees = np.array([len(v) for v in adjacency_dict.values()])
    report = {
        "shapefile": str(shapefile),
        "polygon_id_column": "admin_code",
        "shapefile_polygons": int(len(polygon_id_mapping)),
        "master_areas": int(len(areas)),
        "master_areas_present_in_shapefile": int(covered),
        "master_areas_absent_from_shapefile": int(len(areas) - covered),
        "neighbor_distance_threshold": 0.8,
        "adjacency_degree": {
            "min": int(degrees.min()), "max": int(degrees.max()),
            "mean": float(degrees.mean()),
            "isolated_polygons": int((degrees == 0).sum()),
        },
        "note": (
            "Group IDs are admin codes; polygon_group_mapping is index -> [admin code], "
            "as in the released setup_spatial_groups."
        ),
    }
    if verbose:
        print(
            f"[geometry] {report['shapefile_polygons']} shapefile polygons; "
            f"{covered}/{len(areas)} master areas matched; "
            f"mean degree {report['adjacency_degree']['mean']:.2f}",
            flush=True,
        )
    if covered == 0:
        raise PipelineError("no master area matched the shapefile admin_code column")
    return info, report


# --------------------------------------------------------------------------------------
# 5. Candidate schedule (D18/D22/D55)
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CandidateJob:
    """One Stage 1 candidate: an arm and an exact (target year, month, scope)."""

    arm: str
    year: int
    month: int
    scope: int
    roles: Tuple[str, ...]

    @property
    def horizon(self) -> int:
        return pdata.FORECASTING_SCOPES[self.scope]

    @property
    def variant(self) -> str:
        return f"GeoRF-{self.arm}"

    @property
    def name(self) -> str:
        return f"{self.variant}_{self.year}_{self.month:02d}_fs{self.scope}"

    @property
    def target_month(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def identity(self) -> Dict[str, object]:
        return {
            "name": self.name, "arm": self.arm, "variant": self.variant,
            "year": self.year, "month": self.month, "forecasting_scope": self.scope,
            "horizon_months": self.horizon, "target_month": self.target_month,
            "map_roles": list(self.roles),
        }


DEVELOPMENT_ROLES: Tuple[str, ...] = ("calibration", "selection")
ALL_ROLES: Tuple[str, ...] = ("calibration", "selection", "final")

#: Every status a scheduled candidate may legitimately end in. Checked as an
#: allow-list, so an unrecognised status blocks a map build instead of passing
#: silently as if the candidate had completed.
TERMINAL_CANDIDATE_STATUSES: Tuple[str, ...] = (
    "completed", "excluded_insufficient_support", "failed", "not_run",
)


def candidate_jobs(arm: str, roles: Sequence[str]) -> List[CandidateJob]:
    """Unique candidates for one arm across the given map-role windows.

    2016 Feb/Jun/Oct x three scopes fall inside both the calibration (2014-2016) and
    selection (2016-2018) windows, and 2018 repeats between selection and final
    (2018-2020). D22/R28 deduplicate only *identical same-arm* jobs, so each runs once
    and is referenced by every role ledger that contains it; window overlap never mixes
    arms or extends a map's information cutoff.
    """
    by_key: Dict[Tuple[int, int, int], List[str]] = {}
    for role in roles:
        for year, month, scope in pdata.role_candidate_jobs(role):
            by_key.setdefault((year, month, scope), []).append(role)
    return [
        CandidateJob(arm=arm, year=year, month=month, scope=scope, roles=tuple(roles_here))
        for (year, month, scope), roles_here in sorted(by_key.items())
    ]


def development_jobs(arm: str) -> List[CandidateJob]:
    """The 51 unique development candidates for one arm (calibration + selection)."""
    return candidate_jobs(arm, DEVELOPMENT_ROLES)


def all_candidate_jobs(arm: str) -> List[CandidateJob]:
    """Every candidate for one arm, including the final 2018-2020 window."""
    return candidate_jobs(arm, ALL_ROLES)


def new_final_jobs(arm: str) -> List[CandidateJob]:
    """Final-window candidates this arm has not already run for a development role.

    The job *name* is role-independent, so an overlapping 2018 candidate is reused
    rather than refitted; only the genuinely new dates are returned.
    """
    existing = {job.name for job in development_jobs(arm)}
    return [
        job for job in all_candidate_jobs(arm)
        if "final" in job.roles and job.name not in existing
    ]


def role_jobs(arm: str, role: str) -> List[CandidateJob]:
    source = all_candidate_jobs(arm) if role == "final" else development_jobs(arm)
    return [job for job in source if role in job.roles]


# --------------------------------------------------------------------------------------
# 6. Row selection under the released mask (D20)
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class CandidateRows:
    train_rows: np.ndarray
    target_rows: np.ndarray
    evidence: Dict[str, object]


def select_candidate_rows(
    sources: pdata.PreparedSources, horizon: int, year: int, month: int
) -> CandidateRows:
    """Labeled training and target rows for one candidate.

    O = T - H. Training targets lie in [O-35 calendar months, O): the configured window
    is 36 but its effective calendar span is 35 and origin-month labels are excluded
    (D20 preserves that convention rather than correcting it). Training rows are then
    restricted to groups present in the target month, which is the released Stage 1
    behavior; both counts are recorded.
    """
    t_index = pdata.month_index(year, month)
    o_index = t_index - horizon
    window_start = o_index - pdata.TRAIN_WINDOW_CALENDAR_MONTHS
    all_t = sources.target_month_idx

    target_rows = np.flatnonzero(all_t == t_index)
    temporal_rows = np.flatnonzero((all_t >= window_start) & (all_t < o_index))

    areas = sources.areas
    target_groups = np.unique(areas[sources.target_area_idx[target_rows]])
    temporal_groups = areas[sources.target_area_idx[temporal_rows]]
    train_rows = temporal_rows[np.isin(temporal_groups, target_groups)]

    train_labels = sources.target_label[train_rows]
    target_labels = sources.target_label[target_rows]
    train_months = np.unique(all_t[train_rows])
    evidence = {
        "target_month": pdata.month_label(t_index),
        "origin_month": pdata.month_label(o_index),
        "horizon_months": horizon,
        "training_window": (
            f"[{pdata.month_label(window_start)}, {pdata.month_label(o_index)}) "
            f"= {pdata.TRAIN_WINDOW_CALENDAR_MONTHS} calendar months, origin month excluded"
        ),
        "train_group_filter": TRAIN_GROUP_FILTER,
        "rows_in_temporal_window": int(temporal_rows.size),
        "rows_after_group_filter": int(train_rows.size),
        "rows_removed_by_group_filter": int(temporal_rows.size - train_rows.size),
        "target_rows": int(target_rows.size),
        "train_observed_months": [pdata.month_label(int(m)) for m in train_months],
        "train_observed_month_count": int(train_months.size),
        "train_areas": int(np.unique(areas[sources.target_area_idx[train_rows]]).size),
        "target_areas": int(target_groups.size),
        "train_class_counts": {
            "0": int((train_labels == 0).sum()), "1": int((train_labels == 1).sum()),
        },
        "target_class_counts": {
            "0": int((target_labels == 0).sum()), "1": int((target_labels == 1).sum()),
        },
    }
    return CandidateRows(train_rows=train_rows, target_rows=target_rows, evidence=evidence)


def build_candidate_matrix(
    sources: pdata.PreparedSources, arm: str, horizon: int, rows: np.ndarray
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Origin-aligned, pre-imputation model inputs for one arm on the given rows.

    Recipes are ordered column subsets of the single updated superset, so a recipe can
    never drift from another recipe or between Stage 1 and Stage 3.
    """
    if arm == pdata.REFERENCE_ARM:
        return pdata.build_reference_matrix(sources, horizon, rows=rows)
    columns = pdata.recipe_columns(arm)
    superset, superset_columns = pdata.build_updated_superset(sources, horizon, rows=rows)
    index_of = {name: position for position, name in enumerate(superset_columns)}
    matrix = superset[:, [index_of[name] for name in columns]]
    del superset
    expected = pdata.DECLARED_RECIPE_WIDTHS[arm]
    if matrix.shape[1] != expected:
        raise PipelineError(f"recipe {arm} width {matrix.shape[1]} != declared {expected}")
    return matrix, columns


# --------------------------------------------------------------------------------------
# 7. Released metric helpers (aggregate class-1 F1 on the candidate's target month)
# --------------------------------------------------------------------------------------


def parse_split_gate_trace(stdout_path: Path) -> List[Dict[str, object]]:
    """Recover every class-1 F1 split decision from the released fit log.

    ``partition_opt.select_f1_children`` prints one line per evaluated split
    (transformation.py:731). Keeping the exact parent/candidate values makes a
    no-split candidate auditable instead of merely asserted.
    """
    marker = "F1 performance gate: "
    trace: List[Dict[str, object]] = []
    if not stdout_path.exists():
        return trace
    with open(stdout_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            body = line.split(marker, 1)[1].strip()
            fields: Dict[str, object] = {}
            for item in body.split(","):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                key, value = key.strip(), value.strip()
                if key == "accepted":
                    fields[key] = value == "True"
                else:
                    fields[key] = float(value)
            if {"parent", "candidate", "accepted"} <= set(fields):
                fields["gain"] = float(fields["candidate"]) - float(fields["parent"])
                trace.append(fields)
    return trace


def class_wise_prf(metrics_module, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Class-wise precision/recall/F1 through the released metric functions.

    ``RFmodel.predict_test`` accumulates the same per-branch counts and calls the same
    ``get_prf``; computing them once over the full prediction vector is identical and
    additionally yields the per-row predictions needed to recompute F1 independently.
    """
    true_class, total_class, pred_total = metrics_module.get_class_wise_accuracy(
        y_true, y_pred, prf=True
    )
    pre, rec, f1, total = metrics_module.get_prf(true_class, total_class, pred_total)
    return {
        "precision": [float(v) for v in pre],
        "recall": [float(v) for v in rec],
        "f1": [float(v) for v in f1],
        "total_class": [float(v) for v in total],
        "true_class": [float(v) for v in true_class],
        "pred_total": [float(v) for v in pred_total],
    }


# --------------------------------------------------------------------------------------
# 8. The Stage 1 worker (one isolated process per candidate)
# --------------------------------------------------------------------------------------


def run_worker(job_dir: Path) -> int:
    """Fit one candidate. Never called in-process by the orchestrator."""
    job = read_json(job_dir / "job.json")
    started = time.time()

    random.seed(STAGE1_PROCESS_SEED)
    np.random.seed(STAGE1_PROCESS_SEED)

    runtime = verify_extracted_baseline(Path(job["baseline_root"]))
    sources = pdata.load_prepared_sources(Path(job["prepared_cache"]))

    arm = str(job["arm"])
    horizon = int(job["horizon_months"])
    selection = select_candidate_rows(sources, horizon, int(job["year"]), int(job["month"]))
    if list(selection.evidence["train_observed_months"]) != list(
        job["rows"]["train_observed_months"]
    ):
        raise PipelineError("worker row selection disagrees with the orchestrator's evidence")

    train_rows, target_rows = selection.train_rows, selection.target_rows
    rows_all = np.concatenate([train_rows, target_rows])
    build_started = time.time()
    matrix, columns = build_candidate_matrix(sources, arm, horizon, rows_all)
    build_seconds = time.time() - build_started
    n_train = int(train_rows.size)
    X_train_raw = matrix[:n_train]
    X_test_raw = matrix[n_train:]
    del matrix

    areas = sources.areas
    groups_train = areas[sources.target_area_idx[train_rows]].astype(np.int64)
    groups_test = areas[sources.target_area_idx[target_rows]].astype(np.int64)
    y_train = sources.target_label[train_rows].astype(np.int64)
    y_test = sources.target_label[target_rows].astype(np.int64)

    result: Dict[str, object] = {
        "identity": {key: job[key] for key in (
            "name", "arm", "variant", "year", "month", "forecasting_scope",
            "horizon_months", "target_month", "map_roles")},
        "started_utc": _utc_now(),
        "rows": selection.evidence,
        "schema": {
            "n_columns": int(len(columns)),
            "declared_width": (
                len(pdata.REFERENCE_COLUMNS) if arm == pdata.REFERENCE_ARM
                else pdata.DECLARED_RECIPE_WIDTHS[arm]
            ),
            "columns_sha256": sha256_bytes("\n".join(columns).encode("utf-8")),
            "feature_build_seconds": round(build_seconds, 2),
            "note": (
                "Administrative, country and partition identifiers are metadata only "
                "and are absent from this schema (D51)."
            ),
        },
        "baseline": runtime.to_dict(),
        "runtime": runtime_identity(),
        "process_seeds": {"python_random": STAGE1_PROCESS_SEED, "numpy": STAGE1_PROCESS_SEED},
    }

    with baseline_imports(runtime) as (config, georf_module):
        from src.utils.split import group_aware_train_val_split  # noqa: PLC0415
        from src.model.model_RF import RFmodel  # noqa: PLC0415
        from src.helper.helper import get_X_branch_id_by_group  # noqa: PLC0415
        import src.metrics.metrics as metrics_module  # noqa: PLC0415

        group_split = dict(getattr(config, "GROUP_SPLIT", {}))
        if not group_split.get("enable", False):
            raise PipelineError("released GROUP_SPLIT is disabled; the within-area split is required")

        # D20: the released within-area assignment, determined BEFORE imputation so the
        # same split can be handed to GeoRF.fit through its explicit interface.
        split_started = time.time()
        split_result = group_aware_train_val_split(
            groups=groups_train,
            val_ratio=float(config.VAL_RATIO),
            min_val_per_group=int(group_split.get("min_val_per_group", 1)),
            random_state=group_split.get("random_state", 42),
            skip_singleton_groups=bool(group_split.get("skip_singleton_groups", True)),
        )
        x_set = np.asarray(split_result["X_set"], dtype=int)
        fitting_mask = x_set == 0
        validation_mask = x_set == 1
        coverage = split_result["coverage"]

        support = {
            "real_fitting_rows": int(fitting_mask.sum()),
            "real_validation_rows": int(validation_mask.sum()),
            "labeled_target_rows": int(y_test.size),
            "fitting_class_counts": {
                "0": int((y_train[fitting_mask] == 0).sum()),
                "1": int((y_train[fitting_mask] == 1).sum()),
            },
            "validation_class_counts": {
                "0": int((y_train[validation_mask] == 0).sum()),
                "1": int((y_train[validation_mask] == 1).sum()),
            },
            "groups_with_validation": int((coverage["val_count"] > 0).sum()),
            "singleton_groups_train_only": int((coverage["total_count"] == 1).sum()),
            "split_settings": {
                "val_ratio": float(config.VAL_RATIO),
                "min_val_per_group": int(group_split.get("min_val_per_group", 1)),
                "random_state": group_split.get("random_state", 42),
                "skip_singleton_groups": bool(group_split.get("skip_singleton_groups", True)),
                "row_order": "master file order (area-major), as produced by prepare_data",
            },
            "artificial_rows_in_support": 0,
            "artificial_row_note": (
                "The release appends one zero-feature row per class inside RF fitting "
                "(model_RF.get_new_forest); those rows are never counted here and never "
                "reach the imputer (D4/D20)."
            ),
            "split_seconds": round(time.time() - split_started, 2),
        }
        result["support"] = support
        coverage.to_csv(job_dir / "val_coverage_by_group.csv", index=False)

        missing = [
            key for key, value in (
                ("real_fitting_rows", support["real_fitting_rows"]),
                ("real_validation_rows", support["real_validation_rows"]),
                ("labeled_target_rows", support["labeled_target_rows"]),
            ) if value == 0
        ]
        if missing:
            result["status"] = "excluded_insufficient_support"
            result["exclusion_reason"] = (
                f"D20 requires nonempty real support; empty: {missing}"
            )
            result["seconds"] = round(time.time() - started, 2)
            write_json(job_dir / "candidate.json", result)
            return 0

        # D23: one max_plus imputer per candidate, fitted on real fitting rows only and
        # reused unchanged for the validation and target rows.
        imputer = pdata.MaxPlusImputer()
        imputer.fit(X_train_raw, fitting_mask=fitting_mask, columns=columns)
        X_train = imputer.transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)
        del X_train_raw, X_test_raw
        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise PipelineError("imputed Stage 1 matrices still hold NaN/inf")
        fill_table = pd.DataFrame(imputer.column_stats_)
        fill_table.to_csv(job_dir / "imputer_fill_values.csv", index=False)
        result["imputation"] = {
            key: value for key, value in imputer.manifest().items()
            if key != "ordered_column_statistics"
        }
        result["imputation"]["fit_rows_are_real_fitting_rows_only"] = True

        with open(Path(job["polygon_contiguity_info"]), "rb") as handle:
            polygon_contiguity_info = pickle.load(handle)

        effective_config = {
            "config_module": {
                key: getattr(config, key, None) for key in REPORTED_CONFIG_KEYS
            },
            "georf_module": {
                key: getattr(georf_module, key, None)
                for key in ("FEATURE_DROP", "feature_drop", "MIN_DEPTH", "MAX_DEPTH",
                            "N_JOBS", "NUM_CLASS", "CONTIGUITY", "CONTIGUITY_TYPE",
                            "MIN_COMPONENT_SIZE", "MODEL_CHOICE", "VIS_DEBUG_MODE")
            },
            "diagnostics_bypassed": DIAGNOSTIC_MODULE,
        }
        result["effective_config"] = effective_config
        result["module_locations"] = dict(runtime.module_locations)

        work_dir = job_dir / "work"
        stdout_path = job_dir / "georf_fit_stdout.txt"
        fit_started = time.time()
        with working_directory(work_dir) as cwd:
            with open(stdout_path, "w", encoding="utf-8", errors="backslashreplace") as sink:
                with contextlib.redirect_stdout(sink):
                    model = georf_module.GeoRF(
                        min_model_depth=config.MIN_DEPTH,
                        max_model_depth=config.MAX_DEPTH,
                    )
                    model.fit(
                        X_train, y_train, groups_train,
                        X_set=x_set,
                        split={"X_set": x_set},
                        contiguity_type="polygon",
                        polygon_contiguity_info=polygon_contiguity_info,
                        feature_names=list(columns),
                        print_to_file=False,
                        track_partition_metrics=False,
                        VIS_DEBUG_MODE=False,
                    )
            fit_seconds = time.time() - fit_started
            model_dir = Path(cwd) / model.model_dir

            required = {
                "s_branch": model_dir / "space_partitions" / "s_branch.pkl",
                "branch_table": model_dir / "space_partitions" / "branch_table.npy",
                "X_branch_id": model_dir / "space_partitions" / "X_branch_id.npy",
                "feature_reference": model_dir / "feature_name_reference.csv",
                "checkpoints": model_dir / "checkpoints",
            }
            absent = [name for name, path in required.items() if not path.exists()]
            if absent:
                raise PipelineError(f"mandatory Stage 1 artifacts missing: {absent}")

            s_branch = pd.read_pickle(required["s_branch"])
            branch_table = np.load(required["branch_table"])
            saved_branch_ids = np.load(required["X_branch_id"], allow_pickle=False)
            if saved_branch_ids.shape[0] != n_train:
                raise PipelineError(
                    f"saved X_branch_id has {saved_branch_ids.shape[0]} rows; "
                    f"{n_train} training rows were supplied"
                )
            reference_names = pd.read_csv(required["feature_reference"])["feature_name"].astype(str).tolist()
            if reference_names != list(columns):
                raise PipelineError("the persisted feature reference differs from the frozen schema")

            # -- scoring: released evaluate() numerics without its SHAP/rendering work --
            eval_started = time.time()
            test_branch_id = get_X_branch_id_by_group(groups_test, s_branch)
            y_pred_partitioned = model.model.predict_georf(
                X_test, groups_test, s_branch, X_branch_id=test_branch_id
            ).astype(np.int64)
            partitioned = class_wise_prf(metrics_module, y_test, y_pred_partitioned)

            if bool(job.get("verify_predict_test", False)):
                released = model.model.predict_test(
                    X_test, y_test, groups_test, s_branch, X_branch_id=test_branch_id
                )
                released_f1 = [float(v) for v in released[2]]
                if not np.allclose(released_f1, partitioned["f1"], rtol=0, atol=0):
                    raise PipelineError(
                        f"predict_test F1 {released_f1} != recomputed {partitioned['f1']}"
                    )
                result["predict_test_cross_check"] = {
                    "released_f1": released_f1, "recomputed_f1": partitioned["f1"],
                    "identical": True,
                }

            baseline_wrapper = RFmodel(
                model.dir_ckpt, model.n_trees_unit, max_depth=model.max_depth,
                num_class=model.num_class, random_state=model.random_state,
                n_jobs=model.n_jobs, sample_weights_by_class=model.sample_weights_by_class,
            )
            baseline_wrapper.train(
                np.array(model._base_training_X, copy=True),
                np.array(model._base_training_y, copy=True),
                branch_id="eval",
                sample_weights_by_class=model.sample_weights_by_class,
            )
            y_pred_base = baseline_wrapper.predict(X_test)
            base = class_wise_prf(metrics_module, y_test, y_pred_base)
            eval_seconds = time.time() - eval_started

            model.model.load("")
            estimator_params = {
                key: value for key, value in model.model.model.get_params().items()
                if key in RF_PARAM_KEYS
            }
            base_params = {
                key: value for key, value in baseline_wrapper.model.get_params().items()
                if key in RF_PARAM_KEYS
            }

        # -- artifacts kept outside the disposable working directory -------------------
        branch_strings = np.asarray(saved_branch_ids, dtype=str)
        correspondence = pd.DataFrame({
            "FEWSNET_admin_code": groups_train,
            "partition_id": np.where(branch_strings == "", "root", branch_strings),
        }).drop_duplicates().sort_values("FEWSNET_admin_code").reset_index(drop=True)
        duplicated = correspondence["FEWSNET_admin_code"].duplicated().sum()
        if duplicated:
            raise PipelineError(
                f"{duplicated} admin codes received more than one partition_id"
            )
        correspondence.to_csv(job_dir / "correspondence_table.csv", index=False)

        pd.DataFrame({
            "FEWSNET_admin_code": groups_test,
            "target_month": job["target_month"],
            "horizon_months": horizon,
            "y_true": y_test,
            "y_pred_partitioned": y_pred_partitioned,
            "y_pred_base": np.asarray(y_pred_base, dtype=np.int64),
            "branch_id": np.where(test_branch_id == "", "root", test_branch_id),
        }).to_csv(job_dir / "target_predictions.csv", index=False)

        shutil.copy2(required["s_branch"], job_dir / "s_branch.pkl")
        np.save(job_dir / "branch_table.npy", branch_table)
        for name in ("log_print.txt", "model_eval.log"):
            source = model_dir / name
            if source.exists():
                shutil.copy2(source, job_dir / name)

        partition_ids = sorted(correspondence["partition_id"].unique().tolist())
        result["partitions"] = {
            "terminal_partition_ids": partition_ids,
            "n_terminal_partitions": len(partition_ids),
            # Accepted splits in a binary partition tree = terminal partitions - 1.
            # branch_table counts accepted *nodes* and already holds the root, so its
            # sum is 1 for an unsplit model and must not be read as a split count.
            "accepted_splits": len(partition_ids) - 1,
            "branch_table_active_nodes": int(np.asarray(branch_table).sum()),
            "branch_table_shape": list(np.asarray(branch_table).shape),
            "split_gate_trace": parse_split_gate_trace(stdout_path),
            "split_gate": (
                "released class-1 F1 gate: accept only a strict gain > 0.01 on the "
                "parent's own validation rows; parent wins ties (partition_opt:867-891)"
            ),
            "correspondence_source": (
                "space_partitions/X_branch_id.npy as saved after group-consistency "
                "regeneration and before GeoRF.fit's grid-only refinement, which is the "
                "same array the released app reads (main_model_GF.py:630-634)"
            ),
            "assigned_areas": int(len(correspondence)),
            "root_only": partition_ids == ["root"],
            "area_counts": {
                str(key): int(value)
                for key, value in correspondence["partition_id"].value_counts().items()
            },
        }
        result["scores"] = {
            "f1(1)": partitioned["f1"][1],
            "f1_base(1)": base["f1"][1],
            "f1(0)": partitioned["f1"][0],
            "f1_base(0)": base["f1"][0],
            "precision(1)": partitioned["precision"][1],
            "recall(1)": partitioned["recall"][1],
            "precision_base(1)": base["precision"][1],
            "recall_base(1)": base["recall"][1],
            "num_samples(0)": int((y_test == 0).sum()),
            "num_samples(1)": int((y_test == 1).sum()),
            "partitioned_counts": partitioned,
            "base_counts": base,
            "note": (
                "Class-1 F1 on the candidate's own target month; the Stage 2 weight is "
                "max(logit(f1(1)) - logit(f1_base(1)), 0) (D56). Computed through the "
                "released get_class_wise_accuracy/get_prf on stored per-row predictions."
            ),
        }
        result["model"] = {
            "georf_instance": {
                "min_model_depth": model.min_model_depth,
                "max_model_depth": model.max_model_depth,
                "n_trees_unit": model.n_trees_unit,
                "num_class": model.num_class,
                "max_depth": model.max_depth,
                "random_state": model.random_state,
                "n_jobs": model.n_jobs,
                "mode": model.mode,
                "drop_list_": list(model.drop_list_),
                "model_dir": model.model_dir,
            },
            "root_estimator_params": estimator_params,
            "base_estimator_params": base_params,
            "fit_call": {
                "contiguity_type": "polygon",
                "print_to_file": False,
                "track_partition_metrics": False,
                "VIS_DEBUG_MODE": False,
                "explicit_split": "split={'X_set': ...} and X_set=... (no second split)",
            },
        }
        result["timings"] = {
            "feature_build_seconds": round(build_seconds, 2),
            "fit_seconds": round(fit_seconds, 2),
            "evaluate_seconds": round(eval_seconds, 2),
        }

    # GeoRF.fit configures the root logger with a FileHandler inside the working tree
    # (GeoRF.py:177-183). Windows refuses to delete a file with an open handle, so the
    # handlers are closed before the tree is removed.
    import logging  # noqa: PLC0415

    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()
    logging.shutdown()

    # The released fit log is several megabytes per candidate (mostly per-polygon
    # contiguity lines). It is retained in full, compressed, rather than truncated.
    if stdout_path.exists():
        import gzip  # noqa: PLC0415

        with open(stdout_path, "rb") as raw, gzip.open(
            stdout_path.with_suffix(".txt.gz"), "wb", compresslevel=6
        ) as packed:
            shutil.copyfileobj(raw, packed)
        stdout_path.unlink()

    if not bool(job.get("keep_work", False)):
        # Stage 1 checkpoints are branch RF pickles measured in hundreds of megabytes
        # per candidate. Stage 2 needs only the scores and the correspondence table,
        # and Stage 3 refits its own models, so the working tree is removed and the
        # retained evidence (s_branch, branch_table, correspondence, predictions,
        # logs) is what reproduces the map.
        shutil.rmtree(work_dir, ignore_errors=True)
        result["work_directory"] = "removed after artifact extraction (--keep-work to retain)"
    else:
        result["work_directory"] = str(work_dir)

    result["status"] = "completed"
    result["seconds"] = round(time.time() - started, 2)
    result["finished_utc"] = _utc_now()
    write_json(job_dir / "candidate.json", result)
    return 0


# --------------------------------------------------------------------------------------
# 9. Orchestration
# --------------------------------------------------------------------------------------


class RunContext:
    """Run root, append-only log and an atomically rewritten manifest."""

    def __init__(self, run_dir: Path, create: bool) -> None:
        self.root = run_dir
        self.manifest_path = run_dir / "run_manifest.json"
        self.log_path = run_dir / "run.log"
        if create:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise PipelineError(
                    f"run root {run_dir} already exists and is not empty; use a fresh root"
                )
            run_dir.mkdir(parents=True, exist_ok=True)
            self.manifest: Dict[str, object] = {
                "experiment": "FEWSNETCleanPersistenceExperiment",
                "component": "run_pipeline",
                "scope": "stage1_candidates_and_schedule",
                "created_utc": _utc_now(),
                "completed_stages": [],
                "limitations": list(STANDING_LIMITATIONS),
            }
            self.save()
        else:
            if not self.manifest_path.exists():
                raise PipelineError(f"{run_dir} is not an initialized run root (run --stage setup)")
            self.manifest = read_json(self.manifest_path)

    def save(self) -> None:
        self.manifest["updated_utc"] = _utc_now()
        write_json(self.manifest_path, self.manifest)

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def stage(self, name: str, payload: Dict[str, object]) -> None:
        stages = dict(self.manifest.get("stages", {}))
        stages[name] = payload
        self.manifest["stages"] = stages
        completed = list(self.manifest.get("completed_stages", []))
        if name not in completed:
            completed.append(name)
        self.manifest["completed_stages"] = completed
        self.save()


def stage_setup(context: RunContext, prepared_dir: Path, data_root: Path) -> None:
    """Extract and verify the release, then build the run-local geometry once."""
    started = time.time()
    prepared_manifest = read_json(prepared_dir / "manifests" / "run_manifest.json")
    if "materialize" not in prepared_manifest.get("completed_stages", []):
        raise PipelineError(f"{prepared_dir} has not completed prepare_data materialize")
    cache_path = prepared_dir / "cache" / pdata.CACHE_FILENAME
    if not cache_path.is_file():
        raise PipelineError(f"prepared source cache missing: {cache_path}")

    runtime = extract_baseline(BASELINE_ZIP, context.root / "baseline")
    context.log(
        f"baseline verified: {runtime.payload_files_verified} payload hashes, "
        f"version {runtime.manifest_version}, commit {runtime.manifest_source_commit[:12]}"
    )

    sources = pdata.load_prepared_sources(cache_path)
    shapefile = data_root / str(pdata.PINNED_SOURCES["fews_shapefile"]["relpath"])
    if not shapefile.is_file():
        raise PipelineError(f"FEWS shapefile not found: {shapefile}")
    sidecars = {}
    for suffix in pdata.SHAPEFILE_SIDECAR_SUFFIXES:
        sidecar = shapefile.with_suffix(suffix)
        if sidecar.is_file():
            sidecars[sidecar.name] = sha256_file(sidecar)

    geometry_dir = context.root / "geometry"
    info, report = build_polygon_contiguity_info(
        runtime, sources.areas, sources.static["lat"], sources.static["lon"],
        shapefile, geometry_dir,
    )
    with open(geometry_dir / "polygon_contiguity_info.pkl", "wb") as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lat_lon = pd.DataFrame({
        "FEWSNET_admin_code": sources.areas.astype(np.int64),
        "lat": sources.static["lat"], "lon": sources.static["lon"],
    })
    lat_lon.to_csv(geometry_dir / "FEWSNET_admin_code_lat_lon.csv", index=False)

    context.manifest["prepared_run"] = {
        "run_dir": str(prepared_dir),
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "source_hashes": prepared_manifest.get("source_hashes", {}),
        "prepare_runtime": prepared_manifest.get("runtime", {}),
    }
    context.manifest["baseline"] = runtime.to_dict()
    context.manifest["geometry"] = {
        **report,
        "shapefile_sha256": sha256_file(shapefile),
        "shapefile_sidecar_sha256": sidecars,
        "adjacency_cache": str(geometry_dir / "polygon_adjacency_cache.pkl"),
        "contiguity_info": str(geometry_dir / "polygon_contiguity_info.pkl"),
    }
    context.manifest["runtime"] = runtime_identity()
    context.manifest["schedule"] = pdata.build_schedule()
    context.stage("setup", {"seconds": round(time.time() - started, 1)})
    context.log(f"setup complete in {time.time() - started:.1f}s")


def _prepared_cache(context: RunContext) -> Path:
    prepared = context.manifest.get("prepared_run")
    if not prepared:
        raise PipelineError("run root has no prepared-source binding; run --stage setup")
    return Path(str(prepared["cache"]))


def run_candidate(
    context: RunContext, sources: pdata.PreparedSources, job: CandidateJob,
    *, keep_work: bool = False, verify_predict_test: bool = False,
) -> Dict[str, object]:
    """Materialize one job's inputs and fit it in an isolated process."""
    job_dir = context.root / "stage1" / job.arm / job.name
    result_path = job_dir / "candidate.json"
    if result_path.exists():
        existing = read_json(result_path)
        context.log(f"[skip] {job.name}: already {existing.get('status')}")
        return existing

    job_dir.mkdir(parents=True, exist_ok=True)
    selection = select_candidate_rows(sources, job.horizon, job.year, job.month)
    payload = dict(job.identity())
    payload.update({
        "baseline_root": str(context.root / "baseline" / "GeoRFBaseline"),
        "prepared_cache": str(_prepared_cache(context)),
        "polygon_contiguity_info": str(
            context.root / "geometry" / "polygon_contiguity_info.pkl"
        ),
        "rows": selection.evidence,
        "keep_work": bool(keep_work),
        "verify_predict_test": bool(verify_predict_test),
        "created_utc": _utc_now(),
    })
    write_json(job_dir / "job.json", payload)

    environment = dict(os.environ)
    environment.update({
        "PYTHONHASHSEED": str(STAGE1_PROCESS_SEED),
        "PYTHONIOENCODING": "utf-8",
    })
    environment.pop("PYTHONPATH", None)
    command = [sys.executable, "-B", str(Path(__file__).resolve()),
               "--stage", "worker", "--job-dir", str(job_dir)]
    started = time.time()
    with open(job_dir / "worker.log", "w", encoding="utf-8", errors="backslashreplace") as log:
        completed = subprocess.run(
            command, cwd=str(job_dir), env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    seconds = time.time() - started

    if completed.returncode != 0 or not result_path.exists():
        failure = {
            "identity": job.identity(),
            "status": "failed",
            "returncode": completed.returncode,
            "seconds": round(seconds, 2),
            "rows": selection.evidence,
            "reason": (
                "worker process failed; see worker.log. An execution failure is not "
                "D19 no-split evidence and not a support exclusion (R23)."
            ),
        }
        write_json(job_dir / "failure.json", failure)
        context.log(f"[fail] {job.name}: returncode {completed.returncode} after {seconds:.1f}s")
        return failure

    result = read_json(result_path)
    result["wall_seconds"] = round(seconds, 2)
    write_json(result_path, result)
    status = result.get("status")
    if status == "completed":
        scores = result["scores"]
        context.log(
            f"[done] {job.name}: {seconds:.1f}s, "
            f"{result['partitions']['n_terminal_partitions']} partitions, "
            f"f1(1)={scores['f1(1)']:.6f} f1_base(1)={scores['f1_base(1)']:.6f}"
        )
    else:
        context.log(f"[skip] {job.name}: {status} ({result.get('exclusion_reason')})")
    return result


def write_stage2_inputs(context: RunContext, arm: str, role: str) -> Dict[str, object]:
    """Emit exactly the artifacts the released Stage 2 helpers read.

    ``linked_tables/main_index.csv`` (name, variant, year, month, forecasting_scope,
    f1(1), f1_base(1)) and ``linked_tables/partitions/{name}_partition.csv`` over the
    complete 0..5717 admin range with ``s-1`` for unassigned areas are what
    ``step4_similarity_matrix.py`` consumes, together with
    ``FEWSNET_admin_code_lat_lon.csv`` in the same experiment directory. Only eligible,
    normally completed candidates enter the index (D58); exclusions and failures are
    recorded separately and are never silently dropped.
    """
    jobs = role_jobs(arm, role)
    target = context.root / "stage2_inputs" / arm / role
    partitions_dir = target / "linked_tables" / "partitions"
    partitions_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []
    for job in jobs:
        job_dir = context.root / "stage1" / arm / job.name
        result_path = job_dir / "candidate.json"
        if not result_path.exists():
            # A crashed worker writes failure.json and no candidate.json. Reading only
            # candidate.json would misreport a real failure as "never run".
            failure_path = job_dir / "failure.json"
            if failure_path.exists():
                failure = read_json(failure_path)
                status_rows.append({
                    **job.identity(), "status": "failed",
                    # The writer stores "reason"; reading "error" would silently drop
                    # the explanation while still blocking the build.
                    "reason": str(failure.get("reason", "worker failure")),
                    "returncode": failure.get("returncode"),
                })
            else:
                status_rows.append({**job.identity(), "status": "not_run"})
            continue
        result = read_json(result_path)
        status = str(result.get("status"))
        status_rows.append({
            **job.identity(),
            "status": status,
            "reason": result.get("exclusion_reason", ""),
            "real_fitting_rows": result.get("support", {}).get("real_fitting_rows"),
            "real_validation_rows": result.get("support", {}).get("real_validation_rows"),
            "labeled_target_rows": result.get("support", {}).get("labeled_target_rows"),
            "n_terminal_partitions": result.get("partitions", {}).get("n_terminal_partitions"),
            "f1(1)": result.get("scores", {}).get("f1(1)"),
            "f1_base(1)": result.get("scores", {}).get("f1_base(1)"),
            "seconds": result.get("seconds"),
        })
        if status != "completed":
            continue
        correspondence = pd.read_csv(
            job_dir / "correspondence_table.csv", dtype={"partition_id": str}
        )
        complete = pd.DataFrame({"FEWSNET_admin_code": range(0, 5718)}).merge(
            correspondence, on="FEWSNET_admin_code", how="left"
        )
        complete["partition_id"] = complete["partition_id"].fillna("s-1")
        complete.to_csv(partitions_dir / f"{job.name}_partition.csv", index=False)
        rows.append({
            "name": job.name, "variant": job.variant, "year": job.year,
            "month": f"{job.month:02d}", "forecasting_scope": f"fs{job.scope}",
            "f1(1)": result["scores"]["f1(1)"], "f1_base(1)": result["scores"]["f1_base(1)"],
        })

    pd.DataFrame(status_rows).to_csv(target / "candidate_status.csv", index=False)
    main_index = pd.DataFrame(rows, columns=[
        "name", "variant", "year", "month", "forecasting_scope", "f1(1)", "f1_base(1)",
    ])
    main_index.to_csv(target / "linked_tables" / "main_index.csv", index=False)
    shutil.copy2(
        context.root / "geometry" / "FEWSNET_admin_code_lat_lon.csv",
        target / "FEWSNET_admin_code_lat_lon.csv",
    )

    eligible = len(rows)
    weights_positive = int(sum(
        1 for row in rows
        if _logit_clip(row["f1(1)"]) - _logit_clip(row["f1_base(1)"]) > 0
    ))
    # R23/R24 distinguish a documented D20 support exclusion from missing or failed
    # required evidence. A validly excluded candidate is a completed scientific outcome
    # and must not block the map; only failures and never-run jobs do.
    statuses = Counter(str(row.get("status")) for row in status_rows)
    excluded = int(statuses.get("excluded_insufficient_support", 0))
    failed = int(statuses.get("failed", 0))
    not_run = int(statuses.get("not_run", 0))
    # Allow-list rather than deny-list: an unrecognised status must block the map,
    # otherwise a future status string would silently pass as if it were completed.
    unaccounted = {
        status: int(count) for status, count in statuses.items()
        if status not in TERMINAL_CANDIDATE_STATUSES
    }
    missing_statuses = len(jobs) - sum(statuses.values())
    summary = {
        "arm": arm,
        "map_role": role,
        "information_cutoff": "%04d-%02d" % pdata.MAP_ROLES[role]["cutoff"],
        "candidate_years": list(pdata.MAP_ROLES[role]["candidate_years"]),
        "scheduled_candidates": len(jobs),
        "completed_eligible_candidates": eligible,
        "validly_excluded_candidates": excluded,
        "failed_candidates": failed,
        "not_run_candidates": not_run,
        "unaccounted_status_candidates": unaccounted,
        "candidates_without_any_status": int(missing_statuses),
        "candidate_status_counts": {key: int(value) for key, value in sorted(statuses.items())},
        "candidates_with_positive_stage2_weight": weights_positive,
        "month_ind": False,
        "map_granularity": "general (D55): one map per arm and role, all months and scopes pooled",
        "stage2_ready": (
            eligible > 0 and failed == 0 and not_run == 0
            and not unaccounted and missing_statuses == 0
        ),
        "stage2_blocked_reason": (
            "" if (eligible > 0 and failed == 0 and not_run == 0
                   and not unaccounted and missing_statuses == 0)
            else "; ".join(filter(None, [
                "no eligible completed candidate" if eligible == 0 else "",
                f"{failed} failed candidates" if failed else "",
                f"{not_run} not-run candidates" if not_run else "",
                f"unaccounted statuses {unaccounted}" if unaccounted else "",
                f"{missing_statuses} candidates without any status"
                if missing_statuses else "",
            ]))
        ),
        "note": (
            "Weights are reported for transparency only; this module does not build the "
            "consensus graph or the map. Zero eligible candidates or any failed required "
            "evidence stops that map build (R23), it is not a no-split result. A "
            "documented D20 support exclusion is a completed outcome and does not block it."
        ),
    }
    write_json(target / "stage2_input_summary.json", summary)
    return summary


def _logit_clip(value: Optional[float], eps: float = 1e-6) -> float:
    """The released Stage 2 weight transform (step4_similarity_matrix.py:50-64)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    clipped = min(max(float(value), eps), 1 - eps)
    return float(np.log(clipped / (1 - clipped)))


def run_jobs(
    context: RunContext, jobs: Sequence[CandidateJob], *, keep_work: bool = False,
    verify_predict_test: bool = False,
) -> Dict[str, object]:
    sources = pdata.load_prepared_sources(_prepared_cache(context))
    outcomes: List[Dict[str, object]] = []
    started = time.time()
    for position, job in enumerate(jobs, start=1):
        context.log(f"[{position}/{len(jobs)}] {job.name}")
        outcomes.append(run_candidate(
            context, sources, job, keep_work=keep_work,
            verify_predict_test=verify_predict_test,
        ))
    seconds = [
        float(outcome.get("wall_seconds", outcome.get("seconds", 0.0)) or 0.0)
        for outcome in outcomes
    ]
    summary = {
        "jobs": len(jobs),
        "completed": sum(1 for o in outcomes if o.get("status") == "completed"),
        "excluded": sum(1 for o in outcomes
                        if o.get("status") == "excluded_insufficient_support"),
        "failed": sum(1 for o in outcomes if o.get("status") == "failed"),
        "total_seconds": round(time.time() - started, 1),
        "per_job_seconds": {
            "min": round(min(seconds), 1) if seconds else None,
            "median": round(float(np.median(seconds)), 1) if seconds else None,
            "max": round(max(seconds), 1) if seconds else None,
            "mean": round(float(np.mean(seconds)), 1) if seconds else None,
        },
    }
    if seconds:
        summary["projected_663_job_hours"] = round(float(np.mean(seconds)) * 663 / 3600.0, 2)
    return summary


# --------------------------------------------------------------------------------------
# 10. Stage 2 - consensus map and bounded geographic completion (D55-D60)
# --------------------------------------------------------------------------------------

#: D56 graph parameters. These are frozen, not searched: sigma is the released
#: haversine Gaussian bandwidth in degrees and k is a graph-neighbour count, NOT a
#: required number of partitions.
SIGMA_DEGREES = 5.0
KNN_K = 40
#: D56.5 keeps the released spectral seed; D56.4's eigsh has no released v0, so the
#: implementation pins one (see ``_eigengap_recommendation``) rather than searching seeds.
SPECTRAL_SEED = 42
EIGSH_MAXITER = 5000
EIGSH_MAX_EIGENVALUES = 20
#: D57: k_eigen = max(2, min(20, n-2)) only satisfies k < n once n >= 3.
MIN_COMPONENT_NODES = 3
#: D59 completion bound and sphere radius.
COMPLETION_MAX_KM = 100.0
EARTH_RADIUS_KM = 6371.0

OUT_OF_SCOPE_LABEL = "s-1"
UNASSIGNED_PARTITION = -1

#: Why an area holds the partition it holds. Kept distinct end-to-end so that
#: "learned", "completed" and "pooled by default" are never conflated (R62/R63).
SUPPORT_SPECTRAL_CORE = "spectral_core"
SUPPORT_COMPLETED = "completed_nearest_donor"
SUPPORT_UNASSIGNED_FAR = "unassigned_nearest_donor_beyond_100km"
SUPPORT_UNASSIGNED_NO_COORD = "unassigned_missing_or_invalid_coordinates"
SUPPORT_POOLED_NO_SPLIT = "pooled_valid_unsplit"

#: Graph support is orthogonal to how an area got its partition: R62/R63 require
#: "other component" and "never in the graph" to stay separately identifiable, because
#: they mean different things. An other-component area *did* have candidate evidence
#: and lost only D57's largest-component selection; a never-in-graph area was never
#: assigned a non-``s-1`` partition by any eligible plan.
GRAPH_IN_CORE = "in_fitted_core"
GRAPH_OTHER_COMPONENT = "in_graph_other_component"
GRAPH_NEVER_IN_GRAPH = "never_in_graph"


@dataclasses.dataclass(frozen=True)
class ConsensusPlan:
    """One eligible Stage 1 candidate as a weighted co-membership vote."""

    name: str
    f1: float
    f1_base: float
    weight: float
    labels: np.ndarray  # int32 over the node universe; -1 == not assigned by this plan

    @property
    def f1_base_is_zero(self) -> bool:
        """True when the 1e-6 clip, not the data, produced this plan's weight.

        With ``f1_base`` exactly 0 the clip sends ``logit(f1_base)`` to about -13.8,
        so the plan outranks well-behaved plans by an order of magnitude regardless
        of its own quality. D56/R60 mandate the released formula, so this is recorded
        rather than corrected; see IMPLEMENTATION_LOG.md L3.
        """
        return float(self.f1_base) == 0.0


def load_consensus_plans(
    stage2_dir: Path,
) -> Tuple[List[ConsensusPlan], np.ndarray, Dict[str, object]]:
    """Read one role's Stage 1 ledger into weighted plans over the D58 node universe.

    Returns the plans, the sorted node universe and a support ledger. Every eligible
    plan is returned, including zero-weight ones: under D58 they still contribute
    graph nodes, and under D56 they contribute no edges.
    """
    main_index = pd.read_csv(stage2_dir / "linked_tables" / "main_index.csv")
    partitions_dir = stage2_dir / "linked_tables" / "partitions"
    if main_index.empty:
        raise PipelineError(
            f"{stage2_dir} has no eligible candidates; an empty pool stops the map "
            "build (R23) and is not a no-split result"
        )

    # D58 / full_universe=False: a node is in scope exactly when some eligible plan
    # gives it a real partition. 's-1' is absence of assignment, not a shared residual.
    frames: Dict[str, pd.DataFrame] = {}
    universe: set = set()
    for name in main_index["name"].astype(str):
        path = partitions_dir / f"{name}_partition.csv"
        if not path.is_file():
            raise PipelineError(f"required partition artifact missing: {path}")
        frame = pd.read_csv(path, dtype={"partition_id": str})
        # A null partition_id is absence of assignment, exactly like 's-1'. Comparing
        # the raw column would let NaN survive, and str(NaN) == 'nan' would then be
        # interned as a real partition, manufacturing co-membership between every
        # unassigned area. D58 forbids missing assignments creating edges.
        frame["partition_id"] = frame["partition_id"].where(
            frame["partition_id"].notna(), OUT_OF_SCOPE_LABEL
        )
        frames[name] = frame
        assigned = frame.loc[frame["partition_id"] != OUT_OF_SCOPE_LABEL, "FEWSNET_admin_code"]
        universe.update(assigned.astype(int).tolist())
    if not universe:
        raise PipelineError(f"{stage2_dir}: every eligible plan marks every area s-1")

    codes = np.array(sorted(universe), dtype=np.int64)
    index_of = {int(code): position for position, code in enumerate(codes)}

    required = {"name", "f1(1)", "f1_base(1)"}
    absent = required - set(main_index.columns)
    if absent:
        raise PipelineError(f"main_index.csv is missing columns: {sorted(absent)}")

    plans: List[ConsensusPlan] = []
    nodes_from_zero_weight_only: set = set(codes.tolist())
    for _, row in main_index.iterrows():
        name = str(row["name"])
        f1 = float(row["f1(1)"])
        f1_base = float(row["f1_base(1)"])
        weight = max(_logit_clip(f1) - _logit_clip(f1_base), 0.0)

        labels = np.full(codes.size, UNASSIGNED_PARTITION, dtype=np.int32)
        frame = frames[name]
        local = frame["partition_id"].astype(str)
        # Released convention: partition ids are interned per plan in first-seen order.
        interned: Dict[str, int] = {}
        for code, pid in zip(frame["FEWSNET_admin_code"].astype(int), local):
            position = index_of.get(int(code))
            if position is None or pid == OUT_OF_SCOPE_LABEL:
                continue
            if pid not in interned:
                interned[pid] = len(interned)
            labels[position] = interned[pid]
        if weight > 0:
            nodes_from_zero_weight_only.difference_update(
                codes[labels != UNASSIGNED_PARTITION].tolist()
            )
        plans.append(ConsensusPlan(name, f1, f1_base, weight, labels))

    positive = [plan for plan in plans if plan.weight > 0]
    total_weight = float(sum(plan.weight for plan in positive))
    ledger = {
        "eligible_plans": len(plans),
        "positive_weight_plans": len(positive),
        "zero_weight_plans": len(plans) - len(positive),
        "total_positive_weight": total_weight,
        "node_universe": int(codes.size),
        "nodes_supported_only_by_zero_weight_plans": int(len(nodes_from_zero_weight_only)),
        "plans_with_f1_base_exactly_zero": sum(1 for plan in positive if plan.f1_base_is_zero),
        "weight_share_of_f1_base_zero_plans": (
            round(
                sum(plan.weight for plan in positive if plan.f1_base_is_zero) / total_weight, 6
            ) if total_weight > 0 else None
        ),
        "plan_weights": [
            {
                "name": plan.name,
                "f1(1)": plan.f1,
                "f1_base(1)": plan.f1_base,
                "weight": plan.weight,
                "weight_share": (
                    round(plan.weight / total_weight, 6) if total_weight > 0 else None
                ),
                "f1_base_is_zero": plan.f1_base_is_zero,
                "assigned_nodes": int((plan.labels != UNASSIGNED_PARTITION).sum()),
                "distinct_partitions": int(
                    np.unique(plan.labels[plan.labels != UNASSIGNED_PARTITION]).size
                ),
            }
            for plan in sorted(plans, key=lambda item: -item.weight)
        ],
    }
    return plans, codes, ledger


def accumulate_co_membership(plans: Sequence[ConsensusPlan], n_nodes: int) -> np.ndarray:
    """Sum each positive-weight plan's within-partition co-membership (D56.2).

    Mirrors ``step4_similarity_matrix.accumulate_similarity``: zero/negative weights
    and unassigned (-1) areas contribute nothing, so they can never manufacture an edge.
    """
    similarity = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for plan in plans:
        if plan.weight <= 0:
            continue
        assigned = plan.labels[plan.labels != UNASSIGNED_PARTITION]
        if assigned.size == 0:
            continue
        for label in np.unique(assigned):
            members = np.where(plan.labels == label)[0]
            if members.size:
                similarity[np.ix_(members, members)] += np.float32(plan.weight)
    return similarity


def gaussian_spatial_weight(
    lat: np.ndarray, lon: np.ndarray, sigma_degrees: float
) -> np.ndarray:
    """The released haversine Gaussian in degrees with a unit diagonal (D56.2)."""
    from sklearn.metrics.pairwise import haversine_distances  # noqa: PLC0415

    coords = np.radians(np.column_stack([lat, lon]).astype(np.float64))
    degrees = np.degrees(haversine_distances(coords, coords))
    weight = np.exp(-(degrees ** 2) / (2.0 * sigma_degrees ** 2))
    np.fill_diagonal(weight, 1.0)
    return weight.astype(np.float32)


def normalize_by_max(matrix: np.ndarray) -> np.ndarray:
    """Divide by the matrix-wide maximum when positive; otherwise leave unchanged."""
    maximum = float(matrix.max())
    if maximum <= 0:
        return matrix
    return (matrix / maximum).astype(np.float32)


def sparsify_top_k(matrix: np.ndarray, k: int):
    """Row-wise top-k with a symmetric union (D56.3).

    Keeps the released conventions exactly: the self-entry competes for a slot, all
    indices are retained when n <= k, and the symmetric maximum means the final degree
    is not bounded by k.
    """
    from scipy.sparse import csr_matrix  # noqa: PLC0415

    n = matrix.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i in range(n):
        values = matrix[i]
        top = np.argpartition(values, -k)[-k:] if n > k else np.arange(n)
        rows.extend([i] * int(top.size))
        cols.extend(top.tolist())
        data.extend(values[top].tolist())
    knn = csr_matrix((data, (rows, cols)), shape=(n, n))
    return knn.maximum(knn.transpose())


def select_core_component(sparse_affinity, codes: np.ndarray) -> Dict[str, object]:
    """Largest positive-edge component, ties by smallest canonical code (D57).

    Self-entries are removed first: an area is not connected to another area by its
    own diagonal, so a node whose only positive entry is its own is isolated.
    """
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    off_diagonal = csr_matrix(sparse_affinity, copy=True)
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    off_diagonal.data = (off_diagonal.data > 0).astype(np.int8)
    off_diagonal.eliminate_zeros()

    count, membership = connected_components(off_diagonal, directed=False)
    sizes = np.bincount(membership, minlength=count)
    largest = int(sizes.max())
    # `codes` is sorted, so the first component reaching the maximum size also holds
    # the smallest canonical code among the tied components.
    tied = [int(cid) for cid in np.where(sizes == largest)[0]]
    chosen = min(tied, key=lambda cid: int(codes[np.where(membership == cid)[0][0]]))
    members = np.where(membership == chosen)[0]
    return {
        "component_count": int(count),
        # R61/C:706-708 require *all* component sizes so equal support can be checked
        # directly; a truncated list hides how fragmented the graph actually is.
        "component_sizes": sorted((int(size) for size in sizes), reverse=True),
        "largest_component_size": largest,
        "tied_components": len(tied),
        "selected_component_id": chosen,
        "selected_component_first_code": int(codes[members[0]]),
        "member_indices": members,
        "isolated_nodes": int((sizes[membership] == 1).sum()),
    }


def _eigengap_recommendation(sub_affinity) -> Dict[str, object]:
    """Normalized-Laplacian largest-consecutive-eigengap nc on the core (D56.4/D57).

    The released ``compute_eigengap`` passes no ``v0``, so ARPACK would start from a
    fresh random vector on every call. D56.5 requires pinned initialization, so one
    deterministic draw from a fixed seed is used instead. That is an initialization
    pin, not a seed search: no alternative start is ever tried.

    A constant ``1/sqrt(n)`` start was rejected: on a regular graph that vector is the
    normalized Laplacian's own zero-eigenvalue eigenvector, and ARPACK deflates it to
    nothing ("ARPACK error -9: Starting vector is zero"). A fixed-seed normal draw has
    the same reproducibility and no degenerate case.
    """
    from scipy.sparse import diags  # noqa: PLC0415
    from scipy.sparse.linalg import eigsh  # noqa: PLC0415

    n = int(sub_affinity.shape[0])
    if n < MIN_COMPONENT_NODES:
        raise PipelineError(
            f"core component has {n} nodes; the retained eigensolver formula needs "
            f"at least {MIN_COMPONENT_NODES} (R61). This is an incomplete build, not "
            "a no-split result"
        )
    k = max(2, min(EIGSH_MAX_EIGENVALUES, n - 2))

    degrees = np.asarray(sub_affinity.sum(axis=1)).reshape(-1).astype(np.float64)
    degrees = np.where(degrees == 0, 1e-10, degrees)
    d_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    laplacian = diags(np.ones(n)) - d_inv_sqrt @ sub_affinity @ d_inv_sqrt

    v0 = np.random.default_rng(SPECTRAL_SEED).standard_normal(n)
    eigenvalues, _ = eigsh(laplacian, k=k, which="SM", maxiter=EIGSH_MAXITER, v0=v0)
    eigenvalues = np.sort(eigenvalues)
    gaps = np.diff(eigenvalues)
    if gaps.size == 0:
        raise PipelineError("not enough eigenvalues for eigengap analysis")
    # argmax retains the first equal maximum, as D56.4 requires.
    nc = int(np.argmax(gaps)) + 1
    return {
        "component_nodes": n,
        "k_eigen": int(k),
        "eigenvalues": [float(value) for value in eigenvalues],
        "eigengaps": [float(value) for value in gaps],
        # Algebraic connectivity of the core. The core is connected by construction, so
        # this is > 0, but a value near machine zero means it is connected only through
        # a near-bottleneck and the leading split rides that bridge. Diagnostic only:
        # it never feeds nc selection.
        "fiedler_value": float(eigenvalues[1]),
        "selected_nc": nc,
        "selection_rule": "argmax(diff(sorted eigenvalues)) + 1, first equal maximum",
        "eigsh": {
            "which": "SM", "maxiter": EIGSH_MAXITER,
            "v0": f"default_rng({SPECTRAL_SEED}).standard_normal(n), pinned; "
                  "released code passes no v0",
        },
    }


def _haversine_km(a_lat: np.ndarray, a_lon: np.ndarray,
                  b_lat: np.ndarray, b_lon: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import haversine_distances  # noqa: PLC0415

    left = np.radians(np.column_stack([a_lat, a_lon]).astype(np.float64))
    right = np.radians(np.column_stack([b_lat, b_lon]).astype(np.float64))
    return haversine_distances(left, right) * EARTH_RADIUS_KM


def complete_geographically(
    donor_codes: np.ndarray, donor_labels: np.ndarray,
    recipient_codes: np.ndarray, coordinates: pd.DataFrame,
) -> pd.DataFrame:
    """Bounded nearest-donor completion (D59/R63).

    Donors are exactly the fitted core with their learned labels; recipients are every
    other master area. A recipient takes its 1NN donor's partition when that donor is
    within 100 km, and otherwise stays unassigned for pooled prediction. Newly
    completed areas never become donors, so there is no propagation.
    """
    lookup = coordinates.set_index("FEWSNET_admin_code")
    donor_lat = lookup.loc[donor_codes, "lat"].to_numpy(dtype=np.float64)
    donor_lon = lookup.loc[donor_codes, "lon"].to_numpy(dtype=np.float64)
    # Finiteness alone is not validity: haversine_distances happily wraps an
    # out-of-range longitude, so lon=360 would sit 1.6e-12 km from lon=0 and inherit
    # that donor's partition. Reuse the preparation stage's geographic bounds check.
    if not pdata._valid_coordinates(donor_lat, donor_lon).all():
        raise PipelineError(
            "fitted core contains areas without valid coordinates; the core's master "
            "geometry is a required input (R63), not a completion failure"
        )
    if not np.all(np.diff(donor_codes) > 0):
        raise PipelineError("donor codes must be strictly ascending for canonical ties")

    records: List[Dict[str, object]] = []
    known = set(lookup.index.astype(int).tolist())
    chunk = 512
    for start in range(0, recipient_codes.size, chunk):
        block = recipient_codes[start:start + chunk]
        present = np.array([int(code) in known for code in block])
        lat = np.full(block.size, np.nan)
        lon = np.full(block.size, np.nan)
        if present.any():
            lat[present] = lookup.loc[block[present], "lat"].to_numpy(dtype=np.float64)
            lon[present] = lookup.loc[block[present], "lon"].to_numpy(dtype=np.float64)
        usable = present & pdata._valid_coordinates(lat, lon)

        distances = None
        if usable.any():
            distances = _haversine_km(lat[usable], lon[usable], donor_lat, donor_lon)

        cursor = 0
        for position, code in enumerate(block):
            if not usable[position]:
                records.append({
                    "FEWSNET_admin_code": int(code),
                    "partition_id": UNASSIGNED_PARTITION,
                    "assignment_source": SUPPORT_UNASSIGNED_NO_COORD,
                    "donor_code": None, "donor_distance_km": None,
                })
                continue
            row = distances[cursor]
            cursor += 1
            # Donor codes ascend, so argmin already resolves exact ties by smallest code.
            nearest = int(np.argmin(row))
            distance = float(row[nearest])
            if distance <= COMPLETION_MAX_KM:
                records.append({
                    "FEWSNET_admin_code": int(code),
                    "partition_id": int(donor_labels[nearest]),
                    "assignment_source": SUPPORT_COMPLETED,
                    "donor_code": int(donor_codes[nearest]),
                    "donor_distance_km": distance,
                })
            else:
                records.append({
                    "FEWSNET_admin_code": int(code),
                    "partition_id": UNASSIGNED_PARTITION,
                    "assignment_source": SUPPORT_UNASSIGNED_FAR,
                    "donor_code": int(donor_codes[nearest]),
                    "donor_distance_km": distance,
                })
    return pd.DataFrame.from_records(records)


def classify_graph_support(
    areas: np.ndarray, core_codes: np.ndarray, graph_codes: np.ndarray
) -> np.ndarray:
    """Label each area in-core / in-graph-other-component / never-in-graph (R62/R63).

    Factored out of the map builder so tests exercise the function production uses,
    rather than a copy of its expression that could pass while production is wrong.
    """
    areas = np.asarray(areas)
    in_core = np.isin(areas, core_codes)
    in_graph = np.isin(areas, graph_codes)
    return np.where(
        in_core, GRAPH_IN_CORE,
        np.where(in_graph, GRAPH_OTHER_COMPONENT, GRAPH_NEVER_IN_GRAPH),
    )


def build_consensus_map(context: RunContext, arm: str, role: str) -> Dict[str, object]:
    """Build and freeze one arm/role general map (D55-D60).

    Order matters: D19's all-nonpositive-weight branch is decided *before* any graph
    is built, so a genuinely unsplit pool is never run through a degenerate spectral
    pipeline and never reported as a solver failure.
    """
    started = time.time()
    stage2_dir = context.root / "stage2_inputs" / arm / role
    summary_path = stage2_dir / "stage2_input_summary.json"
    if not summary_path.is_file():
        raise PipelineError(
            f"{stage2_dir} has no Stage 1 ledger; run the full (unlimited) Stage 1 pool "
            "for this arm first (R23/A19)"
        )
    input_summary = read_json(summary_path)
    if not input_summary.get("stage2_ready"):
        raise PipelineError(
            f"{arm}/{role}: {input_summary.get('stage2_blocked_reason')}; missing or "
            "failed required evidence stops the map build (R23). Documented D20 support "
            "exclusions do not block it."
        )

    out_dir = context.root / "stage2" / arm / role
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PipelineError(f"{out_dir} already holds a frozen map; use a fresh run root")
    out_dir.mkdir(parents=True, exist_ok=True)

    plans, codes, ledger = load_consensus_plans(stage2_dir)
    coordinates = pd.read_csv(context.root / "geometry" / "FEWSNET_admin_code_lat_lon.csv")
    master_codes = np.array(sorted(coordinates["FEWSNET_admin_code"].astype(int)), dtype=np.int64)

    pd.DataFrame(ledger["plan_weights"]).to_csv(out_dir / "plan_weights.csv", index=False)

    evidence: Dict[str, object] = {
        "arm": arm, "map_role": role,
        "information_cutoff": input_summary["information_cutoff"],
        "map_granularity": input_summary["map_granularity"],
        "month_ind": False,
        "parameters": {
            "sigma_degrees": SIGMA_DEGREES, "knn_k": KNN_K,
            "spectral_seed": SPECTRAL_SEED,
            "completion_max_km": COMPLETION_MAX_KM,
            "earth_radius_km": EARTH_RADIUS_KM,
            "post_consensus_smoothing": "disabled (D60/R64)",
        },
        "candidate_ledger": ledger,
        "runtime": runtime_identity(),
    }

    # --- D19: decided before any graph exists -------------------------------------
    if ledger["positive_weight_plans"] == 0:
        assignments = pd.DataFrame({
            "FEWSNET_admin_code": master_codes,
            "partition_id": 0,
            "assignment_source": SUPPORT_POOLED_NO_SPLIT,
            # No graph exists in this branch, so no area can be in a core or a
            # non-core component; the D58 universe is still reported in the ledger.
            "graph_support": GRAPH_NEVER_IN_GRAPH,
            "donor_code": None, "donor_distance_km": None,
        })
        evidence.update({
            "outcome": "valid_unsplit_all_nonpositive_weights",
            "outcome_reason": (
                "every eligible completed candidate has weight <= 0 (D19); no graph is "
                "built and no spectral evidence is claimed"
            ),
            "n_clusters": 1,
        })
    else:
        similarity = accumulate_co_membership(plans, codes.size)
        lookup = coordinates.set_index("FEWSNET_admin_code")
        spatial = gaussian_spatial_weight(
            lookup.loc[codes, "lat"].to_numpy(dtype=np.float64),
            lookup.loc[codes, "lon"].to_numpy(dtype=np.float64),
            SIGMA_DEGREES,
        )
        affinity = normalize_by_max(similarity * spatial)
        del similarity, spatial
        sparse_affinity = sparsify_top_k(affinity, KNN_K)
        del affinity

        core = select_core_component(sparse_affinity, codes)
        members = core.pop("member_indices")
        sub_affinity = sparse_affinity[members][:, members]
        core_codes = codes[members]
        # One matrix identity covers both selection and fitting: R61 requires them to
        # run on the exact same ordered submatrix, and this is what proves it.
        # Canonical dtypes: SciPy may pick int32 or int64 index arrays depending on
        # size, and the identity must not depend on that.
        core_identity = sha256_bytes(
            np.asarray(sub_affinity.indptr, dtype=np.int64).tobytes()
            + np.asarray(sub_affinity.indices, dtype=np.int64).tobytes()
            + np.asarray(sub_affinity.data, dtype=np.float32).tobytes()
            + np.asarray(core_codes, dtype=np.int64).tobytes()
        )
        eigen = _eigengap_recommendation(sub_affinity)
        nc = int(eigen["selected_nc"])

        if nc == 1:
            core_labels = np.zeros(core_codes.size, dtype=np.int64)
            outcome = "valid_unsplit_eigengap_nc1"
            reason = (
                "eigengap selection returned nc=1 on a supported positive-weight core "
                "(D56); routed to the pooled model by D62/R66, distinct from D19"
            )
        else:
            from sklearn.cluster import SpectralClustering  # noqa: PLC0415

            spectral = SpectralClustering(
                n_clusters=nc, affinity="precomputed", assign_labels="kmeans",
                random_state=SPECTRAL_SEED, n_jobs=-1,
            )
            core_labels = spectral.fit_predict(sub_affinity).astype(np.int64)
            outcome = "spectral_partition"
            reason = f"eigengap selected nc={nc} and spectral clustering fitted the same core"

        core_frame = pd.DataFrame({
            "FEWSNET_admin_code": core_codes,
            "partition_id": core_labels,
            "assignment_source": SUPPORT_SPECTRAL_CORE,
            "donor_code": None, "donor_distance_km": None,
        })
        recipients = np.setdiff1d(master_codes, core_codes, assume_unique=True)
        completed = complete_geographically(core_codes, core_labels, recipients, coordinates)
        assignments = pd.concat([core_frame, completed], ignore_index=True)
        assignments = assignments.sort_values("FEWSNET_admin_code").reset_index(drop=True)

        never_in_graph = np.setdiff1d(master_codes, codes, assume_unique=True)
        assignments.insert(3, "graph_support", classify_graph_support(
            assignments["FEWSNET_admin_code"].to_numpy(), core_codes, codes,
        ))
        evidence.update({
            "outcome": outcome, "outcome_reason": reason, "n_clusters": nc,
            "graph": {
                **core,
                "core_matrix_sha256": core_identity,
                "core_nodes": int(core_codes.size),
                "nodes_outside_core": int(codes.size - core_codes.size),
                "areas_never_in_graph": int(never_in_graph.size),
            },
            "eigengap": eigen,
            "core_partition_sizes": {
                str(int(label)): int(count) for label, count
                in zip(*np.unique(core_labels, return_counts=True))
            },
        })

    if len(assignments) != master_codes.size:
        raise PipelineError(
            f"{arm}/{role}: frozen map covers {len(assignments)} of {master_codes.size} "
            "master areas; full cohort accounting is required (R63)"
        )
    assignments.to_csv(out_dir / "consensus_map.csv", index=False)

    counts = assignments["assignment_source"].value_counts().to_dict()
    graph_counts = assignments["graph_support"].value_counts().to_dict()
    evidence["coverage"] = {
        "master_areas": int(master_codes.size),
        "by_assignment_source": {str(key): int(value) for key, value in counts.items()},
        "by_graph_support": {str(key): int(value) for key, value in graph_counts.items()},
        "assigned_areas": int((assignments["partition_id"] != UNASSIGNED_PARTITION).sum()),
        "unassigned_pooled_areas": int(
            (assignments["partition_id"] == UNASSIGNED_PARTITION).sum()
        ),
        # R63: an area that lost only D57's largest-component selection is a different
        # kind of pooled row from one no plan ever assigned. Never collapse the two.
        "unassigned_from_other_component": int(
            ((assignments["partition_id"] == UNASSIGNED_PARTITION)
             & (assignments["graph_support"] == GRAPH_OTHER_COMPONENT)).sum()
        ),
        "unassigned_never_in_graph": int(
            ((assignments["partition_id"] == UNASSIGNED_PARTITION)
             & (assignments["graph_support"] == GRAPH_NEVER_IN_GRAPH)).sum()
        ),
    }
    if "donor_distance_km" in assignments:
        donors = assignments["donor_distance_km"].dropna()
        if not donors.empty:
            evidence["coverage"]["completion_distance_km"] = {
                "min": float(donors.min()), "median": float(donors.median()),
                "max": float(donors.max()),
            }
    evidence["consensus_map_sha256"] = sha256_file(out_dir / "consensus_map.csv")
    evidence["seconds"] = round(time.time() - started, 1)
    write_json(out_dir / "consensus_evidence.json", evidence)
    return evidence


def stage_consensus(
    context: RunContext, arms: Sequence[str], *, roles: Sequence[str] = DEVELOPMENT_ROLES
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for arm in arms:
        for role in roles:
            context.log(f"consensus: {arm}/{role}")
            evidence = build_consensus_map(context, arm, role)
            summary[f"{arm}/{role}"] = {
                "outcome": evidence["outcome"],
                "n_clusters": evidence["n_clusters"],
                "coverage": evidence["coverage"],
                "seconds": evidence["seconds"],
            }
            context.log(
                f"  {evidence['outcome']} nc={evidence['n_clusters']} "
                f"assigned={evidence['coverage']['assigned_areas']}/"
                f"{evidence['coverage']['master_areas']}"
            )
    return summary


# --------------------------------------------------------------------------------------
# 11. Stage 3 - rolling pooled/partitioned predictions (D61/D62)
# --------------------------------------------------------------------------------------

#: The released Stage 3 comparator's hyperparameters
#: (compare_partitioned_vs_pooled_rf_k40_nc4.py:73,80-85). RANDOM_STATE=5 is the same
#: seed Stage 1's fitted estimators report, so both stages train identically seeded
#: forests. n_jobs is a throughput setting and is recorded, not pinned for science.
STAGE3_RF_PARAMS: Dict[str, object] = {
    "n_estimators": 100, "max_depth": None, "random_state": 5, "n_jobs": 1,
}
#: D61.3 keeps the released local-support gate unchanged.
MIN_LOCAL_TRAINING_ROWS = 50

def class1_probability(model, X: np.ndarray) -> np.ndarray:
    """Class-1 probability, tolerating an estimator fitted on a single class.

    D61.3 gates *local* models on both classes; the global pool has no such gate, so a
    pooled forest can legitimately see one class. ``predict_proba(...)[:, 1]`` then
    raises IndexError because only one column exists. Reading the column through
    ``classes_`` returns 0.0 when class 1 was never observed, which is the correct
    probability for that estimator - not an artificial observation or a new gate.
    """
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    # Only a genuinely single-class fit is tolerated. A malformed estimator - no
    # classes_, labels outside {0, 1}, or a column count that disagrees with classes_ -
    # is a defect, and returning zeros for it would hide the defect behind a plausible
    # probability.
    if not classes or not set(classes) <= {0, 1}:
        raise PipelineError(
            f"estimator exposes unusable classes_={classes!r}; expected a subset of "
            "{0, 1} for this binary task"
        )
    if proba.shape[1] != len(classes):
        raise PipelineError(
            f"predict_proba returned {proba.shape[1]} columns for {len(classes)} "
            "classes; the estimator is malformed"
        )
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=float)
    return proba[:, classes.index(1)].astype(float)


ROUTE_LOCAL = "local_partition_rf"
ROUTE_POOLED_UNASSIGNED = "pooled_unassigned_area"
ROUTE_POOLED_UNSUPPORTED = "pooled_partition_below_support_gate"
ROUTE_POOLED_UNSEEN = "pooled_partition_absent_from_training_pool"
ROUTE_POOLED_NO_SPLIT = "pooled_valid_unsplit_map"


@dataclasses.dataclass(frozen=True)
class Stage3Fold:
    """One (arm, map role, target month, horizon) prediction fold."""

    arm: str
    role: str
    year: int
    month: int
    scope: int

    @property
    def horizon(self) -> int:
        return int(pdata.HORIZONS[self.scope - 1])

    @property
    def name(self) -> str:
        return f"{self.arm}_{self.role}_{self.year}_{self.month:02d}_fs{self.scope}"

    def identity(self) -> Dict[str, object]:
        return {
            "arm": self.arm, "map_role": self.role, "year": self.year,
            "month": self.month, "forecasting_scope": self.scope,
            "horizon_months": self.horizon, "fold": self.name,
        }


def development_folds(arm: str) -> List[Stage3Fold]:
    """The 18 approved development folds: 2 roles x 3 target months x 3 scopes (D17)."""
    folds: List[Stage3Fold] = []
    for role in DEVELOPMENT_ROLES:
        for year, month in pdata.MAP_ROLES[role]["prediction_targets"]:
            for scope in (1, 2, 3):
                folds.append(Stage3Fold(arm, role, int(year), int(month), scope))
    return folds


def final_folds(arm: str) -> List[Stage3Fold]:
    """D40's horizon-specific final evaluation folds: 11 + 10 + 9 = 30 per arm.

    fs1 starts 2021-06, fs2 2021-10 and fs3 2022-02; all end 2024-10 on the observed
    February/June/October schedule. The windows deliberately differ in length, so the
    per-horizon cohorts are not interchangeable.
    """
    folds: List[Stage3Fold] = []
    for scope, horizon in enumerate(pdata.HORIZONS, start=1):
        for year, month in pdata.final_target_dates(horizon):
            folds.append(Stage3Fold(arm, "final", int(year), int(month), scope))
    return folds


def select_stage3_rows(
    sources: pdata.PreparedSources, horizon: int, year: int, month: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """The full D61/R65 training pool and the target rows.

    Deliberately *not* ``select_candidate_rows``: R65 removes the released
    target-month area/group filter for Stage 3 only, so every eligible historical row
    trains the model even when its area is absent from this target month. The Stage 1
    filter is preserved separately (IMPLEMENTATION_LOG.md L1).
    """
    t_index = pdata.month_index(year, month)
    o_index = t_index - horizon
    window_start = o_index - pdata.TRAIN_WINDOW_CALENDAR_MONTHS
    all_t = sources.target_month_idx

    target_rows = np.flatnonzero(all_t == t_index)
    train_rows = np.flatnonzero((all_t >= window_start) & (all_t < o_index))
    if train_rows.size == 0:
        raise PipelineError(
            f"empty global training pool for {year}-{month:02d} H={horizon}; this is "
            "incomplete execution, not a statistical fallback (R65)"
        )

    labels = sources.target_label[train_rows]
    evidence = {
        "target_month": pdata.month_label(t_index),
        "origin_month": pdata.month_label(o_index),
        "horizon_months": horizon,
        "training_window": (
            f"[{pdata.month_label(window_start)}, {pdata.month_label(o_index)})"
        ),
        "train_group_filter": "removed for Stage 3 by R65/D61; Stage 1 keeps it",
        "train_rows": int(train_rows.size),
        "train_areas": int(np.unique(sources.areas[sources.target_area_idx[train_rows]]).size),
        "train_class_counts": {
            "0": int((labels == 0).sum()), "1": int((labels == 1).sum()),
        },
        "target_rows": int(target_rows.size),
    }
    return train_rows, target_rows, evidence


def load_frozen_map(context: RunContext, arm: str, role: str) -> Tuple[pd.DataFrame, Dict]:
    """Read one frozen Stage 2 map and verify it against its own recorded digest."""
    map_dir = context.root / "stage2" / arm / role
    path = map_dir / "consensus_map.csv"
    if not path.is_file():
        raise PipelineError(f"no frozen map at {path}; run --stage consensus first")
    evidence = read_json(map_dir / "consensus_evidence.json")
    digest = sha256_file(path)
    if digest != evidence.get("consensus_map_sha256"):
        raise PipelineError(
            f"{path} does not match its frozen digest; the map was modified after "
            "freezing and cannot be used for prediction"
        )
    frame = pd.read_csv(path)
    return frame, evidence


def run_stage3_fold(
    context: RunContext, sources: pdata.PreparedSources, fold: Stage3Fold,
    frozen_map: pd.DataFrame, map_evidence: Dict,
) -> Dict[str, object]:
    """Fit this fold's pooled and local RFs and emit per-row predictions.

    Both prediction streams come from the same training pool and the same shared
    imputer, so a pooled-versus-partitioned difference can only come from the map.
    """
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    started = time.time()
    horizon = fold.horizon
    train_rows, target_rows, evidence = select_stage3_rows(
        sources, horizon, fold.year, fold.month
    )

    matrix, columns = build_candidate_matrix(
        sources, fold.arm, horizon, np.concatenate([train_rows, target_rows])
    )
    n_train = int(train_rows.size)
    # D61.1: one imputer, fitted on the pool only, shared by every fit and prediction.
    imputer = pdata.MaxPlusImputer().fit(matrix[:n_train])
    X_train = imputer.transform(matrix[:n_train])
    X_target = imputer.transform(matrix[n_train:])
    del matrix

    y_train = sources.target_label[train_rows].astype(int)
    train_areas = sources.areas[sources.target_area_idx[train_rows]]
    target_areas = sources.areas[sources.target_area_idx[target_rows]]

    partition_of = dict(zip(
        frozen_map["FEWSNET_admin_code"].astype(int),
        frozen_map["partition_id"].astype(int),
    ))
    train_partition = np.array(
        [partition_of.get(int(code), UNASSIGNED_PARTITION) for code in train_areas]
    )
    target_partition = np.array(
        [partition_of.get(int(code), UNASSIGNED_PARTITION) for code in target_areas]
    )

    pooled = RandomForestClassifier(**STAGE3_RF_PARAMS)
    pooled.fit(X_train, y_train)

    valid_unsplit = str(map_evidence.get("outcome", "")).startswith("valid_unsplit")
    locals_by_partition: Dict[int, object] = {}
    support: List[Dict[str, object]] = []
    if not valid_unsplit:
        # D61.2/D61.3: one local RF per partition that clears the released gate.
        for partition in sorted({int(p) for p in np.unique(train_partition)
                                 if int(p) != UNASSIGNED_PARTITION}):
            mask = train_partition == partition
            rows = int(mask.sum())
            classes = np.unique(y_train[mask])
            eligible = rows >= MIN_LOCAL_TRAINING_ROWS and classes.size >= 2
            if eligible:
                model = RandomForestClassifier(**STAGE3_RF_PARAMS)
                model.fit(X_train[mask], y_train[mask])
                locals_by_partition[partition] = model
            support.append({
                "partition_id": partition, "training_rows": rows,
                "classes_present": int(classes.size), "local_model_fitted": eligible,
                "reason": "" if eligible else (
                    "below_50_rows" if rows < MIN_LOCAL_TRAINING_ROWS else "single_class"
                ),
            })

    # --- prediction routing (D61.4 / D62) -----------------------------------------
    n_target = int(target_rows.size)
    hard = np.full(n_target, -1, dtype=int)
    prob = np.full(n_target, np.nan, dtype=float)
    route = np.empty(n_target, dtype=object)

    # D62 evidence: the pooled stream is retained for *every* target row, including
    # locally routed ones, so the pooled-versus-partitioned comparison and any
    # single-partition equality claim can be checked directly from the artifacts.
    pooled_hard = pooled.predict(X_target).astype(int)
    pooled_prob = class1_probability(pooled, X_target)

    def apply_pooled(mask: np.ndarray, reason: str) -> None:
        if not mask.any():
            return
        hard[mask] = pooled_hard[mask]
        prob[mask] = pooled_prob[mask]
        route[mask] = reason

    if valid_unsplit:
        # D62: a valid unsplit map routes the whole partitioned stream to the full-pool
        # RF. This is an explicit routing choice, not a claim the two training subsets
        # were equal.
        apply_pooled(np.ones(n_target, dtype=bool), ROUTE_POOLED_NO_SPLIT)
    else:
        for partition in np.unique(target_partition):
            mask = target_partition == partition
            if int(partition) == UNASSIGNED_PARTITION:
                apply_pooled(mask, ROUTE_POOLED_UNASSIGNED)
                continue
            model = locals_by_partition.get(int(partition))
            if model is not None:
                hard[mask] = model.predict(X_target[mask])
                prob[mask] = class1_probability(model, X_target[mask])
                route[mask] = ROUTE_LOCAL
                continue
            # The released comparator leaves a target partition that never appeared in
            # training at its zero-initialized value, silently predicting "no crisis".
            # D61.4 directs correcting that here, in experiment code only.
            seen = int(partition) in {int(p) for p in np.unique(train_partition)}
            apply_pooled(mask, ROUTE_POOLED_UNSUPPORTED if seen else ROUTE_POOLED_UNSEEN)

    if (hard < 0).any() or np.isnan(prob).any():
        raise PipelineError(f"{fold.name}: {int((hard < 0).sum())} target rows unrouted")

    metadata = pdata.build_row_metadata(sources, horizon, target_rows)
    metadata["arm"] = fold.arm
    metadata["map_role"] = fold.role
    metadata["forecasting_scope"] = fold.scope
    metadata["partition_id"] = target_partition
    metadata["map_assignment_source"] = [
        frozen_map.set_index("FEWSNET_admin_code")["assignment_source"].get(int(code), "")
        for code in target_areas
    ]
    metadata["model_route"] = route
    metadata["rf_hard_prediction"] = hard
    metadata["rf_prob_class1"] = prob
    metadata["pooled_hard_prediction"] = pooled_hard
    metadata["pooled_prob_class1"] = pooled_prob

    fold_dir = context.root / "stage3" / fold.arm / fold.name
    fold_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(fold_dir / "predictions.csv", index=False)
    if support:
        pd.DataFrame(support).to_csv(fold_dir / "local_support.csv", index=False)

    # A61/design.md: the training-key ledger and the shared imputer's ordered statistics
    # are required evidence, not diagnostics. Without them the pool cannot be
    # reconstructed and the imputer's identity cannot be traced to the real pool.
    pd.DataFrame({
        "FEWSNET_admin_code": train_areas.astype(np.int64),
        "target_month": [
            pdata.month_label(int(index))
            for index in sources.target_month_idx[train_rows]
        ],
        "target_label": y_train,
        "partition_id": train_partition,
    }).to_csv(fold_dir / "training_keys.csv", index=False)
    imputer_stats = pd.DataFrame([
        {"column_order": position, "column": columns[position], **{
            key: value for key, value in stat.items() if key != "column"
        }}
        for position, stat in enumerate(imputer.column_stats_)
    ])
    imputer_stats.to_csv(fold_dir / "imputer_statistics.csv", index=False)

    routes = pd.Series(route).value_counts().to_dict()
    summary = {
        **fold.identity(),
        "rows": evidence,
        "features": len(columns),
        "map": {
            "outcome": map_evidence.get("outcome"),
            "n_clusters": map_evidence.get("n_clusters"),
            "consensus_map_sha256": map_evidence.get("consensus_map_sha256"),
        },
        "rf_params": dict(STAGE3_RF_PARAMS),
        "local_models_fitted": len(locals_by_partition),
        "local_partitions_considered": len(support),
        "model_routes": {str(key): int(value) for key, value in routes.items()},
        "persistence": {
            "available": int(metadata["persistence_available"].sum()),
            "unavailable": int((~metadata["persistence_available"]).sum()),
        },
        "imputer": {
            "strategy": pdata.IMPUTER_STRATEGY,
            "fitted_on": "the D61/R65 training pool only (real rows)",
            "columns": len(imputer.column_stats_),
            "all_missing_columns": int(sum(
                1 for stat in imputer.column_stats_ if stat["all_missing_in_fitting_rows"]
            )),
            "statistics_sha256": sha256_file(fold_dir / "imputer_statistics.csv"),
        },
        "training_keys_sha256": sha256_file(fold_dir / "training_keys.csv"),
        "pooled_stream": {
            "retained_for_all_targets": True,
            "note": (
                "pooled_hard_prediction / pooled_prob_class1 cover every target row, "
                "including locally routed ones, so the pooled-versus-partitioned "
                "comparison is reconstructable from the artifacts alone"
            ),
            "rows_where_partitioned_differs_from_pooled": int((hard != pooled_hard).sum()),
        },
        "predictions_sha256": sha256_file(fold_dir / "predictions.csv"),
        "seconds": round(time.time() - started, 1),
    }
    write_json(fold_dir / "fold.json", summary)
    return summary


def stage_predictions(
    context: RunContext, arms: Sequence[str], *, final: bool = False
) -> Dict[str, object]:
    sources = pdata.load_prepared_sources(_prepared_cache(context))
    roles = ("final",) if final else DEVELOPMENT_ROLES
    summary: Dict[str, object] = {}
    for arm in arms:
        maps = {role: load_frozen_map(context, arm, role) for role in roles}
        folds = final_folds(arm) if final else development_folds(arm)
        outcomes: List[Dict[str, object]] = []
        for position, fold in enumerate(folds, start=1):
            context.log(f"[{position}/{len(folds)}] stage3 {fold.name}")
            frozen_map, map_evidence = maps[fold.role]
            outcome = run_stage3_fold(context, sources, fold, frozen_map, map_evidence)
            context.log(
                f"  {outcome['rows']['target_rows']} target rows, "
                f"{outcome['local_models_fitted']} local RFs, {outcome['seconds']}s"
            )
            outcomes.append(outcome)
        summary[arm] = {
            "folds": len(outcomes),
            "total_seconds": round(sum(o["seconds"] for o in outcomes), 1),
            "target_rows": int(sum(o["rows"]["target_rows"] for o in outcomes)),
            "local_models_fitted": int(sum(o["local_models_fitted"] for o in outcomes)),
        }
    return summary


# --------------------------------------------------------------------------------------
# 12. Calibration and threshold freeze (D16/D17, D37/D38)
# --------------------------------------------------------------------------------------

#: D17/R21: calibrators fit on 2018 only. The reused module defaults to (2018, 2019),
#: which is the historical runner's window, so it is always overridden explicitly.
CALIBRATION_FIT_YEARS: Tuple[int, ...] = (2018,)
PROBABILITY_VARIANTS: Tuple[str, ...] = ("raw", "calibrated")


def _reuse_path() -> None:
    """Put PersistenceCorrectionExperiment on the path for its approved reuse."""
    package = REPO_ROOT / "PersistenceCorrectionExperiment"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))


def load_fold_predictions(
    context: RunContext, arm: str, role: str
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Concatenate one arm/role's Stage 3 folds, verifying each against its digest.

    Returns the rows *and* the per-fold input identities, so whatever is frozen from
    them can record exactly which predictions produced it (R42).
    """
    frames: List[pd.DataFrame] = []
    identities: Dict[str, str] = {}
    folds = final_folds(arm) if role == "final" else development_folds(arm)
    for fold in folds:
        if fold.role != role:
            continue
        fold_dir = context.root / "stage3" / arm / fold.name
        path = fold_dir / "predictions.csv"
        if not path.is_file():
            raise PipelineError(f"missing Stage 3 predictions: {path}")
        digest = sha256_file(path)
        recorded = read_json(fold_dir / "fold.json").get("predictions_sha256")
        if digest != recorded:
            raise PipelineError(f"{path} was modified after its fold completed")
        identities[fold.name] = digest
        frame = pd.read_csv(path)
        frame["fold"] = fold.name
        frames.append(frame)
    if not frames:
        raise PipelineError(f"no {role} folds for arm {arm!r}")
    return pd.concat(frames, ignore_index=True), identities


def _calibration_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Shape Stage 3 output into the reused calibrator's expected columns.

    D37: the calibrator is fitted on every row with a valid label and probability,
    including rows whose persistence is unavailable. It estimates target crisis
    probability, not a persistence-error indicator, so persistence never filters it.
    """
    _reuse_path()
    from persistencecorrection import calibration as cal  # noqa: PLC0415

    frame = pd.DataFrame({
        cal.PROB_COLUMN: predictions["rf_prob_class1"].astype(float),
        cal.TRUTH_COLUMN: predictions["target_label"].astype(int),
        cal.PARTITION_COLUMN: predictions["partition_id"].astype(int),
        cal.MONTH_COLUMN: pd.to_datetime(predictions["target_month"] + "-01"),
    })
    return frame


def stage_final(context: RunContext, arms: Sequence[str]) -> Dict[str, object]:
    """Build the final maps and score D40's evaluation windows for the frozen arms.

    Refuses to run without a frozen recipe-selection manifest: D22's winner must be
    chosen from 2020 evidence *before* any final-window row is scored, and the final
    stage is where that ordering could silently be violated.
    """
    freeze_path = context.root / "report" / "recipe_selection.json"
    if not freeze_path.is_file():
        raise PipelineError(
            "no frozen recipe selection; run report_results.py --stage select-recipe "
            "before scoring any final-window row (D22/R42)"
        )
    frozen = read_json(freeze_path)
    # The freeze file's existence is not its authority. A diagnostic ranking over a
    # subset must never be able to nominate the winner the final stage scores.
    if not frozen.get("authoritative"):
        raise PipelineError(
            f"{freeze_path} is not an authoritative D22 selection; rerun "
            "report_results.py --stage select-recipe over the full recipe inventory"
        )
    required_inventory = [name for name, _ in pdata.RECIPE_MANIFEST]
    if sorted(frozen.get("scored_arms", [])) != sorted(required_inventory):
        raise PipelineError(
            f"{freeze_path} scored {sorted(frozen.get('scored_arms', []))}, not the "
            f"approved inventory {sorted(required_inventory)}"
        )
    winner = str(frozen["winner"])
    if winner not in required_inventory:
        raise PipelineError(
            f"frozen winner {winner!r} is not one of the approved updated recipes"
        )
    expected = {winner, pdata.REFERENCE_ARM}
    if set(arms) != expected:
        raise PipelineError(
            f"final arms must be exactly the frozen winner and the reference "
            f"({sorted(expected)}), got {sorted(arms)}"
        )

    summary: Dict[str, object] = {
        "frozen_winner": winner,
        "winner_score": frozen["winner_score"],
        "recipe_selection_sha256": sha256_file(freeze_path),
        "final_windows": {
            f"fs{scope}": {
                "horizon_months": horizon,
                "target_dates": [
                    "%04d-%02d" % pair for pair in pdata.final_target_dates(horizon)
                ],
            }
            for scope, horizon in enumerate(pdata.HORIZONS, start=1)
        },
    }
    sources = pdata.load_prepared_sources(_prepared_cache(context))
    for arm in arms:
        jobs = new_final_jobs(arm)
        context.log(f"final Stage 1 for {arm}: {len(jobs)} new candidates")
        summary[f"stage1:{arm}"] = run_jobs(context, jobs)
        summary[f"stage2_inputs:{arm}"] = write_stage2_inputs(context, arm, "final")
    summary["consensus"] = stage_consensus(context, arms, roles=("final",))
    del sources
    summary["predictions"] = stage_predictions(context, arms, final=True)
    return summary


def stage_calibrate(context: RunContext, arm: str) -> Dict[str, object]:
    """Fit and freeze one month-pooled calibrator per horizon and calendar month."""
    _reuse_path()
    import dataclasses as dc  # noqa: PLC0415

    from persistencecorrection import calibration as cal  # noqa: PLC0415

    predictions, input_identities = load_fold_predictions(context, arm, "calibration")
    out_dir = context.root / "calibration" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    # A frozen calibrator defines the probability transformation the frozen thresholds
    # were selected against. Silently replacing it would invalidate those thresholds
    # while leaving them in place, so refuse before writing anything.
    existing = [path.name for path in out_dir.glob("calibrators_h*.json")]
    if existing:
        raise PipelineError(
            f"{out_dir} already holds frozen calibrators {sorted(existing)}; an "
            "authorized repair must preserve them under a distinct run root"
        )

    summary: Dict[str, object] = {"arm": arm, "fit_years": list(CALIBRATION_FIT_YEARS)}
    for horizon in pdata.HORIZONS:
        block = predictions[predictions["horizon_months"] == horizon]
        if block.empty:
            raise PipelineError(f"{arm}: no calibration rows for horizon {horizon}")
        calibrators = cal.fit_calibrators(
            _calibration_frame(block), scope=horizon, fit_years=CALIBRATION_FIT_YEARS,
        )
        # R20/D16: route by (horizon, calendar month) only. The reused CalibratorSet
        # prefers a partition-specific calibrator whenever one exists, which is exactly
        # the "local partition-specific override" R20 forbids, so the group table is
        # dropped before the set is frozen rather than merely ignored at call time.
        pooled_only = dc.replace(calibrators, groups={})
        write_json(out_dir / f"calibrators_h{horizon}.json", {
            "arm": arm, "horizon_months": horizon,
            "routing": "month_pooled only (R20/D16); partition groups dropped",
            "routing_note": (
                "Every row routes to its calendar-month pool by design. The reused "
                "module reports 'group_absent_from_fit_window' as the fallback reason; "
                "here the groups were deliberately removed, not missing from the data."
            ),
            "dropped_group_calibrators": len(calibrators.groups),
            "fit_rows": int(len(block)),
            # R42: bind the frozen calibrator to the exact predictions that produced it.
            "input_predictions_sha256": input_identities,
            "calibrator_set": pooled_only.to_dict(),
            "month_pool_reports": pooled_only.month_pool_reports,
        })
        summary[f"h{horizon}"] = {
            "fit_rows": int(len(block)),
            "calendar_months": sorted(int(m) for m in pooled_only.month_pooled),
            "kinds": {
                str(month): calibrator.kind
                for month, calibrator in sorted(pooled_only.month_pooled.items())
            },
            "dropped_group_calibrators": len(calibrators.groups),
        }
    write_json(out_dir / "calibration_summary.json", summary)
    return summary


def load_calibrators(context: RunContext, arm: str, horizon: int):
    _reuse_path()
    from persistencecorrection import calibration as cal  # noqa: PLC0415

    path = context.root / "calibration" / arm / f"calibrators_h{horizon}.json"
    if not path.is_file():
        raise PipelineError(f"no frozen calibrators at {path}; run --stage calibrate")
    return cal.CalibratorSet.from_dict(read_json(path)["calibrator_set"])


def stage_thresholds(context: RunContext, arm: str) -> Dict[str, object]:
    """Select and freeze one raw and one calibrated threshold per horizon (D38)."""
    _reuse_path()
    from persistencecorrection.selection import crisis_f1, select_threshold  # noqa: PLC0415

    predictions, input_identities = load_fold_predictions(context, arm, "selection")
    out_dir = context.root / "thresholds" / arm
    freeze_path = out_dir / "frozen_thresholds.json"
    # Check the freeze guard BEFORE writing any output: a rejected rerun must not
    # leave fresh traces beside the old frozen thresholds.
    if freeze_path.exists():
        raise PipelineError(f"{freeze_path} exists; frozen thresholds are immutable")
    out_dir.mkdir(parents=True, exist_ok=True)

    calibrator_identities = {
        f"h{horizon}": sha256_file(
            context.root / "calibration" / arm / f"calibrators_h{horizon}.json"
        )
        for horizon in pdata.HORIZONS
        if (context.root / "calibration" / arm / f"calibrators_h{horizon}.json").is_file()
    }
    if len(calibrator_identities) != len(pdata.HORIZONS):
        raise PipelineError(f"{arm}: frozen calibrators missing; run --stage calibrate")

    summary: Dict[str, object] = {"arm": arm}
    frozen: Dict[str, object] = {}
    for horizon in pdata.HORIZONS:
        block = predictions[predictions["horizon_months"] == horizon]
        # D36/D38: selection runs on the common valid-persistence support only, pooling
        # all three 2020 months before F1 so this is one pooled TP/FP/FN, not a mean.
        support = block[block["persistence_available"]].copy()
        if support.empty:
            raise PipelineError(f"{arm} h{horizon}: no valid-persistence 2020 support")
        persistence = support["persistence"].to_numpy().astype(int)
        truth = support["target_label"].to_numpy().astype(int)

        calibrators = load_calibrators(context, arm, horizon)
        calibrated, route, reason = calibrators.transform(
            support["rf_prob_class1"].to_numpy(dtype=float),
            pd.to_datetime(support["target_month"] + "-01").dt.month.to_numpy(),
            support["partition_id"].to_numpy().astype(int),
        )
        probabilities = {
            "raw": support["rf_prob_class1"].to_numpy(dtype=float),
            "calibrated": calibrated,
        }

        for variant in PROBABILITY_VARIANTS:
            result = select_threshold(persistence, probabilities[variant], truth)
            # A valid no-correction outcome is tau=None: predictions stay at persistence
            # and the development gain is zero. It must never be coerced to a number.
            frozen[f"{variant}_h{horizon}"] = {
                "arm": arm, "horizon_months": horizon, "variant": variant,
                "tau": result["tau"],
                "no_correction": result["tau"] is None,
                "selected_f1": result["selected_f1"],
                "baseline_persistence_f1": result["baseline_persistence_f1"],
                "gain": result["selected_f1"] - result["baseline_persistence_f1"],
                "n_candidates": result["n_candidates"],
                "support_rows": int(len(support)),
                "target_months": sorted(support["target_month"].unique().tolist()),
            }
            pd.DataFrame(result["trace"], columns=["tau", "class1_f1"]).to_csv(
                out_dir / f"trace_{variant}_h{horizon}.csv", index=False
            )
        summary[f"h{horizon}"] = {
            "support_rows": int(len(support)),
            "excluded_no_persistence": int((~block["persistence_available"]).sum()),
            "persistence_f1": crisis_f1(truth, persistence),
            "calibration_routes": {
                str(key): int(value)
                for key, value in pd.Series(route).value_counts().to_dict().items()
            },
            "calibration_route_reasons": sorted({str(item) for item in reason}),
            **{
                variant: {
                    "tau": frozen[f"{variant}_h{horizon}"]["tau"],
                    "gain": round(float(frozen[f"{variant}_h{horizon}"]["gain"]), 6),
                }
                for variant in PROBABILITY_VARIANTS
            },
        }

    write_json(freeze_path, {
        "arm": arm,
        "frozen_utc": _utc_now(),
        "selection_window": "2020-02, 2020-06, 2020-10 (D17/D38)",
        "rule": "strict p > tau, up-only 0->1; ties keep the smallest tau (D37/D38)",
        # R42 requires the input identities that produced the frozen decision, not just
        # a verification that today's files still hash the same.
        "input_predictions_sha256": input_identities,
        "input_calibrators_sha256": calibrator_identities,
        "thresholds": frozen,
        "runtime": runtime_identity(),
    })
    summary["frozen_thresholds_sha256"] = sha256_file(freeze_path)
    return summary


# --------------------------------------------------------------------------------------
# 13. CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FEWS NET clean persistence experiment: Stage 1 candidates and schedule."
    )
    parser.add_argument("--stage", required=True,
                        choices=("setup", "candidate", "pilot", "development",
                                 "stage2-inputs", "consensus", "predictions",
                                 "calibrate", "thresholds", "final", "schedule",
                                 "worker"))
    parser.add_argument("--run-dir", default=None, help="Experiment-local pipeline run root.")
    parser.add_argument("--prepared-dir", default=None,
                        help="Completed prepare_data run root (setup only).")
    parser.add_argument("--data-root", default=None, help="Override the pinned source root.")
    parser.add_argument("--arm", default=pdata.REFERENCE_ARM,
                        help="Feature arm ('reference' or a frozen recipe name).")
    parser.add_argument("--arms", default=None,
                        help="Comma-separated arms for the development stage.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--scope", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N scheduled jobs (bounded probing).")
    parser.add_argument("--keep-work", action="store_true",
                        help="Retain each job's GeoRF working tree and checkpoints.")
    parser.add_argument("--verify-predict-test", action="store_true",
                        help="Cross-check the recomputed F1 against RFmodel.predict_test.")
    parser.add_argument("--job-dir", default=None, help="Worker stage only.")
    return parser.parse_args(argv)


def _known_arms() -> List[str]:
    return [name for name, _ in pdata.RECIPE_MANIFEST] + [pdata.REFERENCE_ARM]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.stage == "worker":
        if not args.job_dir:
            raise SystemExit("--job-dir is required for the worker stage")
        return run_worker(Path(args.job_dir))

    if args.stage == "schedule":
        schedule = pdata.build_schedule()
        jobs = development_jobs(args.arm)
        overlap = [job.name for job in jobs if len(job.roles) > 1]
        print(json.dumps({
            "arm": args.arm,
            "unique_development_jobs": len(jobs),
            "jobs_in_both_role_windows": len(overlap),
            "calibration_jobs": len(role_jobs(args.arm, "calibration")),
            "selection_jobs": len(role_jobs(args.arm, "selection")),
            "stage1_accounting": schedule["stage1"],
        }, indent=2))
        return 0

    if not args.run_dir:
        raise SystemExit("--run-dir is required")
    # Absolute paths throughout: every worker runs with its own cwd, so a relative run
    # root recorded in the manifest would not resolve inside a job directory.
    run_dir = Path(args.run_dir).resolve()

    if args.stage == "setup":
        if not args.prepared_dir:
            raise SystemExit("--prepared-dir is required for setup")
        context = RunContext(run_dir, create=True)
        stage_setup(
            context, Path(args.prepared_dir).resolve(),
            pdata.resolve_data_root(args.data_root),
        )
        return 0

    context = RunContext(run_dir, create=False)
    if "setup" not in context.manifest.get("completed_stages", []):
        raise SystemExit(f"{run_dir} has not completed --stage setup")

    if args.stage == "candidate":
        if args.arm not in _known_arms():
            raise SystemExit(f"unknown arm {args.arm!r}; known arms are {_known_arms()}")
        if not (args.year and args.month and args.scope):
            raise SystemExit("--year, --month and --scope are required for a single candidate")
        matches = [
            job for job in development_jobs(args.arm)
            if (job.year, job.month, job.scope) == (args.year, args.month, args.scope)
        ]
        if not matches:
            raise SystemExit(
                f"({args.year}, {args.month}, fs{args.scope}) is not a scheduled development "
                "candidate; see --stage schedule"
            )
        summary = run_jobs(context, matches, keep_work=args.keep_work,
                           verify_predict_test=args.verify_predict_test)
        context.stage(f"candidate:{matches[0].name}", summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.stage == "consensus":
        arms = [a.strip() for a in args.arms.split(",")] if args.arms else [args.arm]
        unknown = [a for a in arms if a not in _known_arms()]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; known arms are {_known_arms()}")
        summary = stage_consensus(context, arms)
        context.stage("consensus:" + ",".join(arms), summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.stage == "predictions":
        arms = [a.strip() for a in args.arms.split(",")] if args.arms else [args.arm]
        unknown = [a for a in arms if a not in _known_arms()]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; known arms are {_known_arms()}")
        summary = stage_predictions(context, arms)
        context.stage("predictions:" + ",".join(arms), summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.stage == "final":
        arms = [a.strip() for a in args.arms.split(",")] if args.arms else []
        unknown = [a for a in arms if a not in _known_arms()]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}")
        summary = stage_final(context, arms)
        context.stage("final:" + ",".join(arms), summary)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.stage in ("calibrate", "thresholds"):
        if args.arm not in _known_arms():
            raise SystemExit(f"unknown arm {args.arm!r}")
        handler = stage_calibrate if args.stage == "calibrate" else stage_thresholds
        summary = handler(context, args.arm)
        context.stage(f"{args.stage}:{args.arm}", summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.stage == "pilot":
        arm = args.arm
        if arm not in _known_arms():
            raise SystemExit(f"unknown arm {arm!r}")
        jobs = development_jobs(arm)
        if args.limit:
            jobs = jobs[: args.limit]
        summary = run_jobs(context, jobs, keep_work=args.keep_work,
                           verify_predict_test=args.verify_predict_test)
        summary["arm"] = arm
        summary["partial"] = bool(args.limit)
        if not args.limit:
            summary["stage2_inputs"] = {
                role: write_stage2_inputs(context, arm, role) for role in DEVELOPMENT_ROLES
            }
        else:
            context.log(
                "partial pilot: Stage 2 input ledgers are NOT written; a partial candidate "
                "pool is not a formal map build (R23/A19)"
            )
        context.stage(f"pilot:{arm}", summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.stage == "development":
        arms = [a.strip() for a in (args.arms or "").split(",") if a.strip()]
        if not arms:
            raise SystemExit("--arms is required for the development stage")
        unknown = [a for a in arms if a not in _known_arms()]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}")
        summaries: Dict[str, object] = {}
        for arm in arms:
            jobs = development_jobs(arm)
            if args.limit:
                jobs = jobs[: args.limit]
            summary = run_jobs(context, jobs, keep_work=args.keep_work)
            if not args.limit:
                summary["stage2_inputs"] = {
                    role: write_stage2_inputs(context, arm, role) for role in DEVELOPMENT_ROLES
                }
            summaries[arm] = summary
            context.stage(f"development:{arm}", summary)
        print(json.dumps(summaries, indent=2))
        return 0

    if args.stage == "stage2-inputs":
        arms = [a.strip() for a in (args.arms or args.arm).split(",") if a.strip()]
        summaries = {
            arm: {role: write_stage2_inputs(context, arm, role) for role in DEVELOPMENT_ROLES}
            for arm in arms
        }
        print(json.dumps(summaries, indent=2))
        return 0

    raise SystemExit(f"unhandled stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
