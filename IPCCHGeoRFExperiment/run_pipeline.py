#!/usr/bin/env python3
"""IPCCH GeoRF runner — preparation, Stage 1 and Stage 3.

Scope of this module (design.md "Stage1 integration and original-outcome
support", "Map, donor completion and singleton scoring" and "Stage3 rolling
forecast execution"; implement.md phases 3-4):

    preflight -> target ledger -> feature matrix -> geography ->
    original-outcome split -> ONE GeoRF.fit -> learned-map reconciliation ->
    eligible-donor completion -> supplementary singleton scoring ->
    122 scheduled main folds (+ partial 2026) x four arms -> predictions.csv.gz

Reporting is **not** implemented here: metrics, cohorts and bootstrap intervals
belong to ``report_results.py``, which reads the saved predictions. A completed
run therefore ends at ``status = "stage3_complete"``; it is never called
``complete``, because the acceptance criteria that word refers to cover the
reporter as well.

Degeneracy note, recorded because it changes how the output must be read: on the
pinned data Stage 1 accepted **no split** (class-1 F1 gain 0.007434 against the
inherited strict >.01 gate), leaving one terminal branch. The partitioned arm is
still fitted and reported exactly as R5 specifies, but the contrast it supports
is then a training-set-size comparison (one local model over the assigned areas
versus a pooled model over the whole universe), NOT evidence of spatial
structure — see ``stage3/partitioned_vs_pooled_contrast.json``.

Invocation (Windows-native paths for CLI arguments)::

    python3.12.exe -B IPCCHGeoRFExperiment/run_pipeline.py ^
        --source-root "C:\\...\\1.Source Data\\assembled_IPCCH" ^
        --run-id ipcch-v1-<timestamp>

Every artifact lands under ``IPCCHGeoRFExperiment/runs/<run_id>/``, which is
git-ignored. The pinned CSV, the pinned release ZIP and the repository's own
``src/``/``config.py`` are only ever read.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_runtime as brt  # noqa: E402
import prepare_data as pdata  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
DEFAULT_RUNS_DIR = EXPERIMENT_DIR / "runs"
DEFAULT_RELEASE_ZIP = REPO_ROOT / "GeoRFBaseline" / "releases" / "georf-baseline-v0.1.0.zip"

#: R4/Q2r — nothing learned by Stage1 may look past this month.
PARTITION_INFORMATION_CUTOFF = "2022-12"

#: Q8r — inclusive great-circle cap for BOTH donor choice and attachment.
DONOR_MAX_KM = 100.0
EARTH_RADIUS_KM = 6371.0

#: Q7a — every learned arm classifies with p1 > .5; an exact .5 is class 0.
DECISION_THRESHOLD = 0.5

#: R4's main target schedule: 35/33/30/24 months = 122 scheduled folds. Every
#: origin O = T - H is therefore at or after 2023-01, i.e. strictly after the
#: 2022-12 partition-information cutoff.
MAIN_TARGET_SCHEDULE = {
    1: ("2023-02", "2025-12"),
    3: ("2023-04", "2025-12"),
    6: ("2023-07", "2025-12"),
    12: ("2024-01", "2025-12"),
}
EXPECTED_MAIN_FOLDS = 122

#: R4 keeps the incomplete 2026 source as a SEPARATE period; it is opportunistic
#: (only target months that actually carry a valid label) and never merged into
#: the main schedule.
PARTIAL_PERIOD_YEARS = (2026,)
PERIOD_MAIN = "main"
PERIOD_PARTIAL = "partial_2026"

#: R3 — exactly 36 CALENDAR months [O-35, O], not 36 non-missing observations.
TRAIN_WINDOW_MONTHS = 36

#: R5 — a local partition below this many training rows (or with a single class)
#: falls back to the SAME pooled RF. Matches the baseline's MIN_PARTITION_SAMPLES.
MIN_PARTITION_TRAIN_ROWS = 50

#: R5 — Stage 3 RF is the released configuration with n_jobs pinned to 1. The
#: baseline comparison script already declares exactly these values in
#: ``RF_PARAMS``; the runner asserts the two agree instead of trusting either.
STAGE3_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 5,
    "n_jobs": 1,
}

#: R5/Q7b — every parameter is passed explicitly so nothing depends on a library
#: default, and the fitted booster is read back and checked (Q7b: "record complete
#: effective parameters/booster configuration at execution").
STAGE3_XGB_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 400,
    "max_depth": 6,
    "min_child_weight": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "gamma": 0,
    "max_delta_step": 0,
    "scale_pos_weight": 1,
    "base_score": 0.5,
    "random_state": 5,
    "n_jobs": 1,
    "max_bin": 256,
    "grow_policy": "depthwise",
    "num_parallel_tree": 1,
    "eval_metric": "logloss",
}
XGBOOST_PINNED_VERSION = "3.0.0"

#: Column order of ``stage3/predictions.csv.gz``. Checked at runtime against
#: ``report_results.REQUIRED_PREDICTION_COLUMNS`` rather than assumed; the extra
#: columns here are the reporter's documented optional ones.
PREDICTION_COLUMNS = (
    "admin_code",
    "country_en",
    "ISO3",
    "target_month",
    "origin_month",
    "horizon_months",
    "period",
    "fold_id",
    "ipcch_food_crisis",
    "prob_partitioned_rf",
    "pred_partitioned_rf",
    "prob_pooled_rf",
    "pred_pooled_rf",
    "prob_xgb",
    "pred_xgb",
    "persistence_pred",
    "persistence_source_month",
    "persistence_age_months",
    "branch_id",
    "partition_code",
    "assignment_source",
    "donor_admin_code",
    "donor_distance_km",
    "model_route",
    "model_fallback_reason",
)

#: Actual per-row model routes (distinct from the AREA's assignment_source).
ROUTE_LOCAL_PREFIX = "partition:"
ROUTE_POOLED = "pooled_rf"

#: Files whose content identifies the executable experiment code (A1).
EXPERIMENT_SOURCE_FILES = (
    "prepare_data.py",
    "baseline_runtime.py",
    "run_pipeline.py",
    "report_results.py",
)

# ==========================================================================
# Ablation knobs (task 09-21-ipcch-ch-gate-ablation)
# ==========================================================================
#
# Two knobs, set per cell from the CLI and otherwise inert. With both at their
# defaults this module behaves exactly as it did for run ipcch-v1-20260920d.

#: Cadre Harmonise areas. Measured against country_area_id_lookup.csv: 1,308 of
#: 6,227 areas across 19 countries, and the rule splits the Central African
#: Republic (72 of its 298 areas), which is accepted rather than special-cased.
CH_ADMIN_CODE_FLOOR = 100000

#: None keeps every area; "non_ch" drops admin_code >= floor; "ch_only" keeps only
#: those. Applied after the source gate, never before it.
COHORT_FILTER: str | None = None

#: None leaves the pinned config's MIN_CLASS_1_IMPROVEMENT_THRESHOLD (0.01) alone.
#: A float overrides it in every consuming namespace and is verified by readback.
SPLIT_GATE_OVERRIDE: float | None = None

#: Read back from the pinned config after import, never from our own kwargs.
REPORTED_CONFIG_KEYS = (
    "MODEL_CHOICE",
    "MODE",
    "NUM_CLASS",
    "MIN_DEPTH",
    "MAX_DEPTH",
    "N_JOBS",
    "MIN_BRANCH_SAMPLE_SIZE",
    "MIN_SCAN_CLASS_SAMPLE",
    "FLEX_OPTION",
    "FLEX_RATIO",
    "FLEX_TYPE",
    "MIN_GROUP_POS_SAMPLE_SIZE_FLEX",
    "SIGLVL",
    "ES_THRD",
    "MD_THRD",
    "CONTIGUITY",
    "CONTIGUITY_TYPE",
    "REFINE_TIMES",
    "MIN_COMPONENT_SIZE",
    "GOVERNING_METRIC",
    "CRISIS_FOCUSED_OPTIMIZATION",
    "MIN_CLASS_1_IMPROVEMENT_THRESHOLD",
    "VAL_RATIO",
    "TRAIN_RATIO",
    "GROUP_SPLIT",
    "FEATURE_DROP",
    "feature_drop",
    "DISABLE_BASELINE_CV_MAP",
    "PRESERVE_ISOLATED_POLYGONS",
    "STEP_SIZE",
)


class PipelineError(RuntimeError):
    """A gate failed. The run is marked failed and its artifacts are kept."""


# ==========================================================================
# Run scaffolding
# ==========================================================================


class RunContext:
    """Run directory, append-only log and an atomically replaced manifest."""

    SUBDIRECTORIES = ("baseline", "data", "geography", "stage1", "stage3")

    def __init__(self, runs_dir: Path, run_id: str):
        if not run_id or any(ch in run_id for ch in '/\\:*?"<>|'):
            raise PipelineError(f"invalid run id: {run_id!r}")
        self.run_id = run_id
        self.root = Path(runs_dir) / run_id
        if self.root.exists():
            # Never reuse an id: a failed run stays immutable evidence
            # (design.md "Stops, status and rollback").
            raise PipelineError(
                f"run id {run_id!r} already exists at {self.root}; choose a new id"
            )
        self.root.mkdir(parents=True)
        for name in self.SUBDIRECTORIES:
            (self.root / name).mkdir()
        self.log_path = self.root / "run.log"
        self.manifest_path = self.root / "manifest.json"
        self.started = time.time()
        self.manifest: dict = {
            "run_id": run_id,
            "status": "running",
            "scope": "stage1_and_stage3",
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "stages": {},
            "limitations": list(STANDING_LIMITATIONS),
        }
        self.save()

    # -- logging ---------------------------------------------------------

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    # -- manifest --------------------------------------------------------

    def save(self) -> None:
        self.manifest["updated_utc"] = _utc_now()
        self.manifest["elapsed_seconds"] = round(time.time() - self.started, 3)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2, default=_json_default)
        os.replace(temporary, self.manifest_path)

    def stage(self, name: str, payload: dict) -> None:
        self.manifest["stages"][name] = payload
        self.save()

    def fail(self, reason: str, detail: str = "") -> None:
        self.manifest["status"] = "failed"
        self.manifest["failure"] = {"reason": reason, "detail": detail}
        self.save()

    def finish(self) -> None:
        # Deliberately NOT "complete": the acceptance criteria that word refers
        # to include the reporter, which is a separate entry point.
        self.manifest["status"] = "stage3_complete"
        self.save()

    # -- paths -----------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def geography_dir(self) -> Path:
        return self.root / "geography"

    @property
    def stage1_dir(self) -> Path:
        return self.root / "stage1"

    @property
    def stage3_dir(self) -> Path:
        return self.root / "stage3"

    @property
    def baseline_dir(self) -> Path:
        return self.root / "baseline"


STANDING_LIMITATIONS = (
    "source publication timing is unverified; features assume observation-month "
    "availability (R3)",
    "topology repair does not establish administrative identity, and the upstream "
    "geometry builder allowed unrestricted nearest-neighbour fallback (R2)",
    "Stage1 internal validation uses per-area chronological cutoffs; it is a "
    "development score, not a globally forward forecast test (R4/Q5a)",
    "singleton scores are supplementary post-map diagnostics only (R4/Q5v)",
    "inferred (nearest-donor) membership is not observed or validated membership "
    "(R4/Q8r)",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity() -> dict:
    """Record the actual interpreter and package build, not a requirements pin."""
    import importlib.metadata as metadata  # noqa: PLC0415

    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "geopandas",
        "shapely",
        "pyproj",
        "xgboost",
        "psutil",
    ):
        try:
            versions[package] = metadata.version(package)
        except Exception:  # noqa: BLE001 - absence is itself the evidence
            versions[package] = None
    return {
        "executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cwd_at_start": str(Path.cwd()),
    }


# ==========================================================================
# Stage 1 design matrix
# ==========================================================================


@dataclass
class Stage1Design:
    """Model rows for the single pooled fit, plus their provenance.

    ``frame`` keeps feature and metadata columns together so every row can be
    traced back to one original outcome; ``X``/``y``/``groups``/``x_set`` are the
    exact arrays handed to ``GeoRF.fit``.
    """

    frame: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    x_set: np.ndarray
    singleton_frame: pd.DataFrame
    singleton_X: np.ndarray
    audit: dict = field(default_factory=dict)


def build_stage1_design(
    matrix: "pdata.FeatureMatrix",
    split: "pdata.Stage1Split",
    cutoff: str = PARTITION_INFORMATION_CUTOFF,
) -> Stage1Design:
    """Attach split roles to the four horizon views of each original outcome.

    The split was made on ORIGINAL outcomes (R4/Q5h), so this join is the only
    place the four views appear; all four inherit one role and therefore land on
    the same side. Rows outside the 2014-2022 pool carry no role and are dropped:
    they belong to Stage 3, which is not implemented here.
    """
    frame = matrix.frame
    for column in ("admin_code", "target_month", "origin_month", "horizon_months"):
        if column not in frame.columns:
            raise PipelineError(f"feature matrix lacks {column}")

    roles = split.outcomes.copy()
    roles["target_month"] = pd.to_datetime(roles["target_month"]).dt.strftime("%Y-%m")
    roles["admin_code"] = roles["admin_code"].astype(np.int64)
    keys = ["admin_code", "target_month"]
    if roles.duplicated(keys).any():
        raise PipelineError("split outcomes are not unique on (admin_code, target_month)")

    merged = frame.merge(
        roles[keys + ["split_role"]], on=keys, how="left", validate="many_to_one"
    )
    if len(merged) != len(frame):
        raise PipelineError("role join changed the feature row count")

    learn = merged[merged["split_role"].isin(["fit", "validation"])].reset_index(drop=True)
    singles = merged[merged["split_role"] == "singleton"].reset_index(drop=True)

    horizons = sorted(int(h) for h in pdata.ACTIVE_HORIZONS)
    for name, subset in (("learning", learn), ("singleton", singles)):
        counts = subset.groupby(keys)["horizon_months"].agg(["size", "nunique"])
        if len(counts) and (counts["size"] != len(horizons)).any():
            raise PipelineError(f"{name} rows: an outcome does not have exactly 4 views")
        if len(counts) and (counts["nunique"] != len(horizons)).any():
            raise PipelineError(f"{name} rows: an outcome has duplicate horizons")

    expected_learn = (split.audit["fit"] + split.audit["validation"]) * len(horizons)
    if len(learn) != expected_learn:
        raise PipelineError(
            f"expected {expected_learn} learning rows, assembled {len(learn)}"
        )
    expected_single = split.audit["singleton_areas"] * len(horizons)
    if len(singles) != expected_single:
        raise PipelineError(
            f"expected {expected_single} singleton rows, assembled {len(singles)}"
        )

    # A1/A3 gate: nothing Stage1 sees may postdate the frozen cutoff.
    for column in ("target_month", "origin_month"):
        latest = learn[column].max()
        if latest > cutoff:
            raise PipelineError(
                f"Stage1 {column} reaches {latest}, past the {cutoff} cutoff"
            )

    feature_columns = list(matrix.feature_columns)
    X = learn[feature_columns].to_numpy(dtype=np.float64, copy=True)
    y = learn["ipcch_food_crisis"].to_numpy(dtype=np.int64, copy=True)
    groups = learn["admin_code"].to_numpy(dtype=np.int64, copy=True)
    x_set = np.where(learn["split_role"].to_numpy() == "validation", 1, 0).astype(int)
    singleton_X = singles[feature_columns].to_numpy(dtype=np.float64, copy=True)

    if not np.isin(y, (0, 1)).all():
        raise PipelineError("Stage1 y is not binary")
    if not np.isin(x_set, (0, 1)).all():
        raise PipelineError("X_set must be 0/1 only")
    if not (len(X) == len(y) == len(groups) == len(x_set)):
        raise PipelineError("Stage1 array lengths disagree")
    if x_set.sum() == 0 or (x_set == 0).sum() == 0:
        raise PipelineError("Stage1 split has an empty side")

    fit_keys = set(map(tuple, learn.loc[x_set == 0, keys].to_numpy()))
    val_keys = set(map(tuple, learn.loc[x_set == 1, keys].to_numpy()))
    if fit_keys & val_keys:
        raise PipelineError("an original outcome has views on both split sides")

    audit = {
        "learning_rows": int(len(learn)),
        "fit_rows": int((x_set == 0).sum()),
        "validation_rows": int((x_set == 1).sum()),
        "fit_outcomes": int(len(fit_keys)),
        "validation_outcomes": int(len(val_keys)),
        "singleton_rows": int(len(singles)),
        "singleton_outcomes": int(len(singles) // len(horizons)),
        "active_groups": int(pd.unique(groups).size),
        "groups_with_fit_rows": int(pd.unique(groups[x_set == 0]).size),
        "groups_with_validation_rows": int(pd.unique(groups[x_set == 1]).size),
        "horizons": horizons,
        "target_month_min": str(learn["target_month"].min()),
        "target_month_max": str(learn["target_month"].max()),
        "origin_month_min": str(learn["origin_month"].min()),
        "origin_month_max": str(learn["origin_month"].max()),
        "information_cutoff": cutoff,
        "positive_rate_fit": float(y[x_set == 0].mean()),
        "positive_rate_validation": float(y[x_set == 1].mean()),
        "feature_columns": len(feature_columns),
    }
    return Stage1Design(
        frame=learn,
        X=X,
        y=y,
        groups=groups,
        x_set=x_set,
        singleton_frame=singles,
        singleton_X=singleton_X,
        audit=audit,
    )


def apply_stage1_imputation(imputer, X: np.ndarray, x_set: np.ndarray, *extra):
    """Fit ONE imputer on genuine fitting rows, then transform with stored values.

    Q6c is specific about the failure this avoids: ``comp_impute`` refits and
    discards its state on every call, so calling it on a held-out matrix would
    let validation/singleton extrema set the fill for the training columns. Here
    ``fit`` sees ``x_set == 0`` rows only and every other matrix is transformed
    with the values already stored on the instance.

    Pseudo rows are not involved: the baseline RF appends its per-class zero rows
    inside ``get_new_forest``, i.e. strictly after this transform.
    """
    x_set = np.asarray(x_set)
    fit_rows = X[x_set == 0]
    if len(fit_rows) == 0:
        raise PipelineError("cannot fit the imputer: no fitting rows")
    imputer.fit(fit_rows)
    outputs = [np.asarray(imputer.transform(X), dtype=np.float64)]
    for matrix in extra:
        matrix = np.asarray(matrix)
        if matrix.size == 0:
            outputs.append(matrix.reshape(0, X.shape[1]).astype(np.float64))
        else:
            outputs.append(np.asarray(imputer.transform(matrix), dtype=np.float64))
    return imputer, outputs


def imputer_fill_table(imputer, feature_columns) -> pd.DataFrame:
    """Serialise the fitted fill values so a rerun can be checked against them."""
    rows = []
    for index, name in enumerate(feature_columns):
        stats = getattr(imputer, "column_stats_", {}).get(index, {})
        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "fill_value": getattr(imputer, "impute_values_", {}).get(index),
                "fit_min": stats.get("min"),
                "fit_max": stats.get("max"),
                "fit_has_missing": stats.get("has_missing"),
                "all_missing_in_fit": index in getattr(imputer, "column_stats_", {})
                and pd.isna(stats.get("max")),
            }
        )
    return pd.DataFrame(rows)


# ==========================================================================
# Learned map extraction and reconciliation
# ==========================================================================


@dataclass
class LearnedMap:
    """Explicit terminal membership, separated from default-root placeholders.

    ``branch_by_area`` holds only areas with a genuine terminal assignment.
    ``placeholder_areas`` are areas that sit in a branch which *was* split but
    that were never placed in either child: an inherited helper would report them
    as root, but that is a default, not learned membership (R4), so they fall
    through to donor completion like any other unassigned area.
    """

    branch_by_area: dict
    placeholder_areas: tuple
    terminal_branches: tuple
    branch_columns: tuple
    members_by_branch: dict
    audit: dict


def extract_learned_map(s_branch: pd.DataFrame) -> LearnedMap:
    """Read explicit membership out of ``s_branch``, preserving branch strings.

    ``s_branch`` columns are branch labels in creation order (root ``''`` first,
    then each accepted split's two children), and each column lists the group ids
    placed in that branch, padded with ``-1``. Column labels are kept as text so
    ``'0'``, ``'00'`` and ``'000'`` stay distinct — the leading zeros are lineage.
    """
    columns = [str(column) for column in s_branch.columns]
    if not columns or columns[0] != "":
        raise PipelineError(f"s_branch does not start at the root branch: {columns[:3]}")
    for column in columns[1:]:
        if set(column) - {"0", "1"} or column == "":
            raise PipelineError(f"unexpected branch label in s_branch: {column!r}")

    members: dict[str, set] = {}
    for position, column in enumerate(columns):
        values = pd.to_numeric(
            pd.Series(s_branch.iloc[:, position].to_numpy()), errors="coerce"
        ).to_numpy()
        members[column] = {int(v) for v in values[np.isfinite(values)] if int(v) >= 0}

    column_set = set(columns)

    def has_children(branch: str) -> bool:
        return (branch + "0") in column_set or (branch + "1") in column_set

    # Later columns are deeper, so the last hit is the deepest explicit placement.
    assigned: dict[int, str] = {}
    for column in columns:
        for area in members[column]:
            assigned[area] = column

    branch_by_area: dict[int, str] = {}
    placeholders: list[int] = []
    for area, branch in assigned.items():
        if has_children(branch):
            placeholders.append(area)
        else:
            branch_by_area[area] = branch

    terminal = sorted({branch for branch in branch_by_area.values()})
    audit = {
        "branch_columns": columns,
        "accepted_splits": int((len(columns) - 1) // 2),
        "terminal_branches": terminal,
        "terminal_branch_count": len(terminal),
        "max_branch_depth": max((len(b) for b in columns), default=0),
        "areas_in_root_column": len(members[""]),
        "learned_areas": len(branch_by_area),
        "placeholder_areas": len(placeholders),
        "members_per_terminal_branch": {
            branch: sum(1 for value in branch_by_area.values() if value == branch)
            for branch in terminal
        },
        "placeholder_note": (
            "an area sitting in a branch that was split but which was never placed "
            "in either child; the inherited helper would call it root, which is a "
            "default and not learned membership (R4)"
        ),
    }
    return LearnedMap(
        branch_by_area=branch_by_area,
        placeholder_areas=tuple(sorted(placeholders)),
        terminal_branches=tuple(terminal),
        branch_columns=tuple(columns),
        members_by_branch=members,
        audit=audit,
    )


def checkpoint_exists(checkpoint_dir: Path, branch: str) -> bool:
    """Match ``model_RF._checkpoint_candidates``: ``rf_<branch>`` with/without .pkl."""
    base = Path(checkpoint_dir) / f"rf_{branch}"
    return base.is_file() or Path(str(base) + ".pkl").is_file()


def reconcile_learned_map(
    learned: LearnedMap,
    groups: np.ndarray,
    saved_branch_ids: np.ndarray,
    checkpoint_dir: Path,
) -> dict:
    """Cross-check s_branch, the saved row assignments and checkpoint routing.

    Three artefacts have to agree before the map may be frozen (implement.md
    phase 3 gate): explicit ``s_branch`` membership, the regenerated
    ``X_branch_id.npy``, and an actual model file for every branch a row routes
    to. A branch checkpoint may hold an adopted PARENT model — that is model
    provenance, not a map error, and is recorded rather than rejected.
    """
    groups = np.asarray(groups, dtype=np.int64)
    saved = np.asarray([str(value) for value in np.asarray(saved_branch_ids).ravel()])
    if len(saved) != len(groups):
        raise PipelineError(
            f"X_branch_id has {len(saved)} rows but X_group has {len(groups)}"
        )

    active = set(int(value) for value in np.unique(groups))
    missing_from_root = sorted(active - learned.members_by_branch[""])
    if missing_from_root:
        raise PipelineError(
            f"{len(missing_from_root)} fitted groups are absent from the root branch, "
            f"e.g. {missing_from_root[:5]}"
        )

    # One branch string per group, or the row map fragmented an admin unit.
    per_group = pd.DataFrame({"group": groups, "branch": saved})
    distinct = per_group.groupby("group")["branch"].nunique()
    fragmented = sorted(int(g) for g in distinct[distinct > 1].index)
    if fragmented:
        raise PipelineError(
            f"{len(fragmented)} groups have more than one branch in X_branch_id, "
            f"e.g. {fragmented[:5]}"
        )
    row_branch = per_group.groupby("group")["branch"].first().to_dict()

    disagreements = []
    for area, branch in sorted(learned.branch_by_area.items()):
        actual = row_branch.get(area)
        if actual != branch:
            disagreements.append(
                {"admin_code": int(area), "s_branch": branch, "X_branch_id": actual}
            )
    if disagreements:
        raise PipelineError(
            f"{len(disagreements)} learned areas disagree between s_branch and "
            f"X_branch_id, e.g. {disagreements[:3]}"
        )

    # Placeholders must show up as the inherited default, which is exactly why
    # they cannot be read as membership.
    placeholder_rows = {
        int(area): row_branch.get(int(area)) for area in learned.placeholder_areas
    }
    unexpected = {
        area: branch for area, branch in placeholder_rows.items() if branch != ""
    }

    checkpoint_dir = Path(checkpoint_dir)
    routed = sorted(set(row_branch.values()))
    missing_checkpoints = [
        branch for branch in routed if not checkpoint_exists(checkpoint_dir, branch)
    ]
    if not checkpoint_exists(checkpoint_dir, ""):
        missing_checkpoints.append("<root>")
    if missing_checkpoints:
        raise PipelineError(
            f"no checkpoint for routed branches: {missing_checkpoints}"
        )

    available = sorted(
        path.name for path in checkpoint_dir.iterdir() if path.name.startswith("rf_")
    )
    routed_files = {f"rf_{branch}" for branch in routed} | {"rf_"}
    orphans = [
        name for name in available if name.split(".")[0] not in routed_files
    ]
    return {
        "active_groups": len(active),
        "learned_areas": len(learned.branch_by_area),
        "placeholder_areas": len(learned.placeholder_areas),
        "placeholder_row_branches": placeholder_rows,
        "placeholder_rows_not_default_root": unexpected,
        "row_branches_used": routed,
        "checkpoints_present": available,
        "checkpoints_not_routed": orphans,
        "checkpoints_missing": [],
        "fragmented_groups": [],
        "notes": [
            "a child checkpoint may contain an adopted parent model when the F1 "
            "gate kept the parent for that side; spatial branch and model-fit "
            "provenance are different facts",
            "checkpoints_not_routed are candidate children that were trained and "
            "then REJECTED by the F1 gate; no row routes to them and nothing "
            "downstream may load them",
        ],
    }


# ==========================================================================
# Donor completion (Q5d / Q8r)
# ==========================================================================


def haversine_km(lat1, lon1, lat2, lon2, radius: float = EARTH_RADIUS_KM):
    """Great-circle distance in km between reference coordinates.

    Q8r defines the 100 km cap on the keyed reference coordinates, not on polygon
    centroids: 735 areas differ between the two by more than 1e-6 deg, worst
    0.677 deg (~75 km), which is decision-changing at that cap.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64))
                              for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    inner = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def eligible_donor_table(learned: LearnedMap, split: "pdata.Stage1Split",
                         coordinates: pd.DataFrame) -> pd.DataFrame:
    """Q5d: >=1 original fitting AND >=1 original validation outcome, valid
    coordinates, and exactly one explicit learned assignment.

    Retained root/ancestor membership qualifies when it is real; a default-root
    placeholder never does, which is why ``learned.branch_by_area`` is the only
    admissible source here.
    """
    outcomes = split.outcomes
    fit_counts = (
        outcomes[outcomes["split_role"] == "fit"].groupby("admin_code").size()
    )
    val_counts = (
        outcomes[outcomes["split_role"] == "validation"].groupby("admin_code").size()
    )
    coords = coordinates.set_index(pdata.REFERENCE_ID_COLUMN)

    rows = []
    for area, branch in sorted(learned.branch_by_area.items()):
        n_fit = int(fit_counts.get(area, 0))
        n_val = int(val_counts.get(area, 0))
        if area not in coords.index:
            continue
        lat = float(coords.at[area, "ref_lat"])
        lon = float(coords.at[area, "ref_lon"])
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        if n_fit >= 1 and n_val >= 1:
            rows.append(
                {
                    "admin_code": int(area),
                    "branch_id": branch,
                    "fit_outcomes": n_fit,
                    "validation_outcomes": n_val,
                    "ref_lat": lat,
                    "ref_lon": lon,
                }
            )
    table = pd.DataFrame(
        rows,
        columns=["admin_code", "branch_id", "fit_outcomes", "validation_outcomes",
                 "ref_lat", "ref_lon"],
    )
    return table.sort_values("admin_code").reset_index(drop=True)


def nearest_donor(recipient_coords: np.ndarray, donor_coords: np.ndarray,
                  chunk: int = 256):
    """1-NN over eligible donors only; ties resolve to the lowest donor index.

    ``donor_coords`` must already be ordered by ascending area id, so
    ``argmin``'s first-minimum rule *is* the stable tie order required by Q8r.
    """
    n = len(recipient_coords)
    index = np.full(n, -1, dtype=np.int64)
    distance = np.full(n, np.inf, dtype=np.float64)
    if n == 0 or len(donor_coords) == 0:
        return index, distance
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = recipient_coords[start:stop]
        matrix = haversine_km(
            block[:, 0][:, None],
            block[:, 1][:, None],
            donor_coords[None, :, 0],
            donor_coords[None, :, 1],
        )
        best = np.argmin(matrix, axis=1)
        index[start:stop] = best
        distance[start:stop] = matrix[np.arange(stop - start), best]
    return index, distance


def complete_assignments(
    universe: np.ndarray,
    coordinates: pd.DataFrame,
    learned: LearnedMap,
    donors: pd.DataFrame,
    max_km: float = DONOR_MAX_KM,
) -> pd.DataFrame:
    """Freeze one assignment per area over the whole ID universe.

    Learned labels are preserved untouched. Everything else — placeholders,
    zero-label areas, singleton areas — looks for its nearest ELIGIBLE donor.
    There is no chaining: a donor-completed area never becomes a donor, because
    the candidate set is fixed to ``donors`` before the search starts.
    Beyond the inclusive cap the area stays unassigned and will predict with the
    pooled/root model.
    """
    universe = np.asarray(sorted(int(v) for v in universe), dtype=np.int64)
    coords = coordinates.set_index(pdata.REFERENCE_ID_COLUMN)
    missing = [int(a) for a in universe if a not in coords.index]
    if missing:
        raise PipelineError(f"{len(missing)} areas lack reference coordinates")

    donors = donors.sort_values("admin_code").reset_index(drop=True)
    donor_ids = donors["admin_code"].to_numpy(dtype=np.int64)
    if len(donor_ids) == 0:
        raise PipelineError("no eligible donor survived Q5d; cannot complete the map")
    if not np.all(np.diff(donor_ids) > 0):
        raise PipelineError("donor ids must be unique and ascending for stable ties")
    donor_coords = donors[["ref_lat", "ref_lon"]].to_numpy(dtype=np.float64)
    donor_branch = donors["branch_id"].to_numpy()

    recipients = np.asarray(
        [a for a in universe if a not in learned.branch_by_area], dtype=np.int64
    )
    recipient_coords = (
        coords.loc[recipients][["ref_lat", "ref_lon"]].to_numpy(dtype=np.float64)
        if len(recipients)
        else np.zeros((0, 2))
    )
    if len(recipient_coords) and not np.isfinite(recipient_coords).all():
        bad = recipients[~np.isfinite(recipient_coords).all(axis=1)]
        raise PipelineError(f"{len(bad)} recipients have non-finite coordinates: {bad[:5]}")
    best_index, best_distance = nearest_donor(recipient_coords, donor_coords)
    nearest_by_area = {
        int(area): (int(donor_ids[i]), float(d), str(donor_branch[i]))
        for area, i, d in zip(recipients, best_index, best_distance)
        if i >= 0
    }

    placeholder = set(int(a) for a in learned.placeholder_areas)
    partition_code = {branch: code for code, branch in enumerate(learned.terminal_branches)}

    rows = []
    for area in universe:
        area = int(area)
        if area in learned.branch_by_area:
            branch = learned.branch_by_area[area]
            rows.append(
                {
                    "admin_code": area,
                    "branch_id": branch,
                    "partition_code": partition_code[branch],
                    "assignment_source": "learned",
                    "donor_admin_code": -1,
                    "donor_distance_km": np.nan,
                    "nearest_eligible_admin_code": -1,
                    "nearest_eligible_distance_km": np.nan,
                    "model_route": f"partition:{branch}",
                    "fallback_reason": "",
                    "was_placeholder_root": False,
                }
            )
            continue
        donor, distance, branch = nearest_by_area.get(area, (-1, float("inf"), ""))
        within = donor >= 0 and distance <= max_km
        rows.append(
            {
                "admin_code": area,
                "branch_id": branch if within else "",
                "partition_code": partition_code[branch] if within else -1,
                "assignment_source": "nearest_donor" if within else "unresolved",
                "donor_admin_code": donor if within else -1,
                "donor_distance_km": float(distance) if within else np.nan,
                "nearest_eligible_admin_code": donor,
                "nearest_eligible_distance_km": (
                    float(distance) if math.isfinite(distance) else np.nan
                ),
                "model_route": f"partition:{branch}" if within else "pooled_root",
                "fallback_reason": (
                    "" if within else f"nearest eligible donor beyond {max_km:g} km"
                ),
                "was_placeholder_root": area in placeholder,
            }
        )

    table = pd.DataFrame(rows)
    if len(table) != len(universe):
        raise PipelineError("assignment table does not cover the universe exactly")
    if table["admin_code"].duplicated().any():
        raise PipelineError("assignment table has duplicate areas")
    donor_set = set(int(v) for v in donor_ids)
    chained = table[
        (table["assignment_source"] == "nearest_donor")
        & (~table["donor_admin_code"].isin(donor_set))
    ]
    if len(chained):
        raise PipelineError("a donor outside the eligible set was used (chaining)")
    over_cap = table[
        (table["assignment_source"] == "nearest_donor")
        & (table["donor_distance_km"] > max_km)
    ]
    if len(over_cap):
        raise PipelineError(f"{len(over_cap)} assignments exceed the {max_km} km cap")
    return table


def assignment_summary(table: pd.DataFrame, max_km: float = DONOR_MAX_KM) -> dict:
    donor_rows = table[table["assignment_source"] == "nearest_donor"]
    distances = donor_rows["donor_distance_km"].to_numpy(dtype=np.float64)
    quantiles = {}
    if len(distances):
        for q in (0, 25, 50, 75, 90, 95, 100):
            quantiles[f"p{q}"] = float(np.percentile(distances, q))
    unresolved = table[table["assignment_source"] == "unresolved"]
    unresolved_nearest = unresolved["nearest_eligible_distance_km"].to_numpy(
        dtype=np.float64
    )
    return {
        "universe": int(len(table)),
        "learned": int((table["assignment_source"] == "learned").sum()),
        "nearest_donor": int(len(donor_rows)),
        "unresolved": int(len(unresolved)),
        "assigned_total": int((table["partition_code"] >= 0).sum()),
        "placeholder_root_areas_recompleted": int(
            (table["was_placeholder_root"] & (table["assignment_source"] != "learned")).sum()
        ),
        "donor_distance_km": quantiles,
        "donor_distance_mean_km": float(distances.mean()) if len(distances) else None,
        "donor_within_10km": int((distances <= 10).sum()),
        "donor_within_50km": int((distances <= 50).sum()),
        "donor_at_cap_inclusive": int((distances == max_km).sum()),
        "unresolved_nearest_eligible_km": {
            "min": float(np.nanmin(unresolved_nearest)) if len(unresolved) else None,
            "median": float(np.nanmedian(unresolved_nearest)) if len(unresolved) else None,
            "max": float(np.nanmax(unresolved_nearest)) if len(unresolved) else None,
        },
        "partition_code_counts": {
            int(code): int(count)
            for code, count in table["partition_code"].value_counts().sort_index().items()
        },
        "cap_km": max_km,
    }


# ==========================================================================
# Supplementary singleton scoring (Q5s / Q5m / Q5v)
# ==========================================================================


def singleton_branch_ids(frame: pd.DataFrame, assignments: pd.DataFrame) -> np.ndarray:
    """Route each singleton row through its own area's frozen assignment.

    Q5m: the recipient uses the donor partition's MODEL on its OWN features.
    An unresolved recipient uses the saved root model, which is the same pooled
    fallback Stage 3 will use.
    """
    lookup = assignments.set_index("admin_code")["branch_id"].to_dict()
    missing = sorted(
        {int(a) for a in frame["admin_code"].unique() if int(a) not in lookup}
    )
    if missing:
        raise PipelineError(f"{len(missing)} singleton areas are outside the frozen map")
    return np.array(
        [str(lookup[int(area)]) for area in frame["admin_code"].to_numpy()],
        dtype=object,
    )


def confusion_counts(truth: np.ndarray, predicted: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    f1_denominator = 2 * tp + fp + fn
    return {
        "n": int(len(truth)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positives": int((truth == 1).sum()),
        "predicted_positives": int((predicted == 1).sum()),
        # Q9a: a zero denominator is undefined with a reason, never a zero score.
        "class1_f1": (2 * tp / f1_denominator) if f1_denominator else None,
        "class1_precision": (tp / (tp + fp)) if (tp + fp) else None,
        "class1_recall": (tp / (tp + fn)) if (tp + fn) else None,
        "undefined_reason": "" if f1_denominator else "no positive truth or prediction",
    }


# ==========================================================================
# Stage 3 — four-arm rolling forecasts (R3 window, R4 schedule, R5 arms)
# ==========================================================================


def verify_prediction_schema() -> dict:
    """Check ``PREDICTION_COLUMNS`` against the reporter's own constant.

    The prediction file is the interface between this runner and
    ``report_results``; guessing its column names would move a contract failure
    from here (cheap) to the end of a long run (expensive). The import is late
    so a reporter edit cannot break Stage 1.
    """
    try:
        from report_results import (  # noqa: PLC0415
            OPTIONAL_PREDICTION_COLUMNS,
            REQUIRED_PREDICTION_COLUMNS,
        )
    except Exception as error:  # noqa: BLE001 - the contract must be checkable
        raise PipelineError(
            f"cannot import the reporter's prediction schema: {error}"
        ) from error

    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in PREDICTION_COLUMNS]
    if missing:
        raise PipelineError(
            f"the prediction schema is missing required reporter columns: {missing}"
        )
    unknown = [
        c
        for c in PREDICTION_COLUMNS
        if c not in REQUIRED_PREDICTION_COLUMNS
        and c not in OPTIONAL_PREDICTION_COLUMNS
    ]
    return {
        "required": list(REQUIRED_PREDICTION_COLUMNS),
        "written": list(PREDICTION_COLUMNS),
        "extra_columns_written": unknown,
        "note": (
            "extras are the reporter's documented optional columns; they are "
            "cross-checked by it, never required"
        ),
    }


@dataclass(frozen=True)
class Fold:
    """One (horizon, target month) refit. ``origin_ord`` is O = T - H."""

    fold_id: str
    period: str
    horizon: int
    target_ord: int
    origin_ord: int

    @property
    def target_month(self) -> str:
        return pdata.month_label([self.target_ord])[0]

    @property
    def origin_month(self) -> str:
        return pdata.month_label([self.origin_ord])[0]

    @property
    def window_start_ord(self) -> int:
        return self.origin_ord - (TRAIN_WINDOW_MONTHS - 1)


def build_fold_schedule(
    available_target_ords: dict | None = None,
    schedule: dict = MAIN_TARGET_SCHEDULE,
    partial_years: tuple = PARTIAL_PERIOD_YEARS,
) -> list:
    """R4's 122 main folds, plus opportunistic partial-2026 folds.

    EVERY main month is scheduled, including months that turn out to hold no
    valid label: an empty test month is recorded with zero support, which is
    explicitly not a zero F1 and not a failure (R4). The partial period is the
    opposite — it exists only where labels are actually available — because the
    2026 source is incomplete and was never part of the approved main schedule.
    """
    folds: list[Fold] = []
    for horizon in sorted(schedule):
        first, last = schedule[horizon]
        lo = int(pdata.month_ordinal(int(first[:4]), int(first[5:7])))
        hi = int(pdata.month_ordinal(int(last[:4]), int(last[5:7])))
        if hi < lo:
            raise PipelineError(f"h{horizon} schedule ends before it starts")
        for target_ord in range(lo, hi + 1):
            origin_ord = target_ord - horizon
            folds.append(
                Fold(
                    fold_id=f"h{horizon}_{pdata.month_label([target_ord])[0]}",
                    period=PERIOD_MAIN,
                    horizon=horizon,
                    target_ord=target_ord,
                    origin_ord=origin_ord,
                )
            )

    # R4/A3 safety property first: no origin may reach back into the information
    # Stage 1 learned from. Checked before the count so a bad schedule reports
    # the leak rather than the arithmetic.
    cutoff = int(
        pdata.month_ordinal(
            int(PARTITION_INFORMATION_CUTOFF[:4]), int(PARTITION_INFORMATION_CUTOFF[5:7])
        )
    )
    early = [f.fold_id for f in folds if f.origin_ord <= cutoff]
    if early:
        raise PipelineError(
            f"{len(early)} folds have an origin at or before the "
            f"{PARTITION_INFORMATION_CUTOFF} partition cutoff, e.g. {early[:3]}"
        )

    main_count = len(folds)
    if main_count != EXPECTED_MAIN_FOLDS:
        raise PipelineError(
            f"expected {EXPECTED_MAIN_FOLDS} scheduled main folds, built {main_count}"
        )

    if available_target_ords:
        for horizon in sorted(schedule):
            hi = int(
                pdata.month_ordinal(
                    int(schedule[horizon][1][:4]), int(schedule[horizon][1][5:7])
                )
            )
            for target_ord in sorted(available_target_ords.get(horizon, ())):
                if target_ord <= hi:
                    continue
                if (target_ord // 12) not in partial_years:
                    continue
                folds.append(
                    Fold(
                        fold_id=f"h{horizon}_{pdata.month_label([target_ord])[0]}",
                        period=PERIOD_PARTIAL,
                        horizon=horizon,
                        target_ord=target_ord,
                        origin_ord=target_ord - horizon,
                    )
                )

    identities = [f.fold_id for f in folds]
    if len(set(identities)) != len(identities):
        raise PipelineError("fold identities are not unique")
    late = [f.fold_id for f in folds if f.origin_ord <= cutoff]
    if late:
        raise PipelineError(
            f"{len(late)} partial folds have an origin at or before the "
            f"{PARTITION_INFORMATION_CUTOFF} partition cutoff, e.g. {late[:3]}"
        )
    return folds


@dataclass
class Stage3Panel:
    """Every supervised row Stage 3 can use, with its frozen map provenance.

    The feature matrix is reused as-is: R3 already built each row against its
    OWN origin, so a Stage 3 fold only ever *selects* rows. Nothing here is
    recomputed against the fold's origin, which is exactly the leak R3 forbids.
    """

    metadata: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    admin: np.ndarray
    target_ord: np.ndarray
    origin_ord: np.ndarray
    horizon: np.ndarray
    partition_code: np.ndarray
    branch_id: np.ndarray
    persistence: dict
    branch_by_code: dict
    feature_columns: tuple
    audit: dict = field(default_factory=dict)


def persistence_lookup(valid_labels: pd.DataFrame, admin: np.ndarray,
                       origin_ord: np.ndarray) -> dict:
    """Q4 persistence: the latest valid same-area R1 label with month <= O.

    Computed straight from the R1 ledger and deliberately NOT from anything a
    model saw: persistence "refreshes a lookup without fitting" (R4/Q4). There
    is no maximum age and no 36-month truncation — Stage 3's training window is
    a fitting window, not a history horizon.
    """
    labels = valid_labels
    hist_area = labels["admin_code"].to_numpy(dtype=np.int64)
    hist_ord = pdata.month_ordinal(
        labels["year"].to_numpy(), labels["month"].to_numpy()
    ).astype(np.int64)
    hist_label = labels["ipcch_food_crisis"].to_numpy(dtype=np.int64)

    slot = pdata._as_of_index(hist_area, hist_ord, admin, origin_ord)
    available = slot >= 0
    safe = np.clip(slot, 0, None)
    value = np.where(available, hist_label[safe].astype(np.float64), np.nan)
    source_ord = np.where(available, hist_ord[safe], -1)
    age = np.where(available, (origin_ord - source_ord).astype(np.float64), np.nan)
    if np.any(available & (age < 0)):
        raise PipelineError("a persistence label is dated after its own origin")
    return {
        "available": available,
        "value": value,
        "source_ord": source_ord,
        "age": age,
    }


def build_stage3_panel(matrix, assignments: pd.DataFrame,
                       valid_labels: pd.DataFrame) -> Stage3Panel:
    """Attach the FROZEN Stage 1 map and independent persistence to every row."""
    frame = matrix.frame
    metadata = frame[list(matrix.metadata_columns)].copy()
    admin = metadata["admin_code"].to_numpy(dtype=np.int64)
    target_ord = pdata.month_ordinal(
        metadata["target_month"].str.slice(0, 4).astype(int),
        metadata["target_month"].str.slice(5, 7).astype(int),
    ).astype(np.int64)
    origin_ord = pdata.month_ordinal(
        metadata["origin_month"].str.slice(0, 4).astype(int),
        metadata["origin_month"].str.slice(5, 7).astype(int),
    ).astype(np.int64)
    horizon = frame["horizon_months"].to_numpy(dtype=np.float64).astype(np.int64)
    if not np.array_equal(target_ord - origin_ord, horizon):
        raise PipelineError("a feature row violates O = T - H")
    if not np.isin(horizon, pdata.ACTIVE_HORIZONS).all():
        raise PipelineError("a feature row carries a horizon outside 1/3/6/12")

    required = {"admin_code", "branch_id", "partition_code", "assignment_source",
                "donor_admin_code", "donor_distance_km"}
    missing = required - set(assignments.columns)
    if missing:
        raise PipelineError(f"the frozen map lacks columns: {sorted(missing)}")
    if assignments["admin_code"].duplicated().any():
        raise PipelineError("the frozen map has duplicate areas")
    unmapped = sorted(set(int(a) for a in np.unique(admin))
                      - set(int(a) for a in assignments["admin_code"]))
    if unmapped:
        raise PipelineError(
            f"{len(unmapped)} scored areas are outside the frozen map, "
            f"e.g. {unmapped[:5]}"
        )

    joined = metadata[["admin_code"]].merge(
        assignments[sorted(required)], on="admin_code", how="left",
        validate="many_to_one",
    )
    if len(joined) != len(metadata):
        raise PipelineError("the map join changed the row count")
    partition_code = joined["partition_code"].to_numpy(dtype=np.int64)
    branch_id = joined["branch_id"].fillna("").astype(str).to_numpy()
    for column in ("assignment_source", "donor_admin_code", "donor_distance_km"):
        metadata[column] = joined[column].to_numpy()
    metadata["branch_id"] = branch_id
    metadata["partition_code"] = partition_code

    branch_by_code = {}
    for code, branch in zip(assignments["partition_code"], assignments["branch_id"]):
        code = int(code)
        branch = "" if pd.isna(branch) else str(branch)
        if code < 0:
            continue
        if branch_by_code.setdefault(code, branch) != branch:
            raise PipelineError(f"partition code {code} maps to two branch strings")

    persistence = persistence_lookup(valid_labels, admin, origin_ord)

    # Cross-check: Q6a's history feature and Q4's persistence are the same
    # quantity by definition (latest valid same-area label at O). They are
    # produced by different code paths, so disagreement is a real defect.
    feature_label = frame["last_observed_label"].to_numpy(dtype=np.float64)
    feature_known = ~np.isnan(feature_label)
    if not np.array_equal(feature_known, persistence["available"]):
        raise PipelineError(
            "Q6a history availability disagrees with the independent Q4 "
            "persistence lookup"
        )
    if np.any(feature_known & (feature_label != persistence["value"])):
        raise PipelineError(
            "Q6a history value disagrees with the independent Q4 persistence lookup"
        )

    X = frame[list(matrix.feature_columns)].to_numpy(dtype=np.float64, copy=True)
    y = metadata["ipcch_food_crisis"].to_numpy(dtype=np.int64)
    if not np.isin(y, (0, 1)).all():
        raise PipelineError("Stage3 panel truth is not binary")

    audit = {
        "rows": int(len(metadata)),
        "areas": int(pd.unique(admin).size),
        "countries": int(metadata["country_en"].nunique()),
        "target_month_min": str(metadata["target_month"].min()),
        "target_month_max": str(metadata["target_month"].max()),
        "rows_by_horizon": {
            str(int(h)): int((horizon == h).sum()) for h in sorted(set(horizon.tolist()))
        },
        "rows_by_assignment_source": {
            str(k): int(v)
            for k, v in metadata["assignment_source"].value_counts().items()
        },
        "persistence_available_rows": int(persistence["available"].sum()),
        "partition_codes": sorted({int(c) for c in np.unique(partition_code)}),
        "branch_by_code": branch_by_code,
    }
    return Stage3Panel(
        metadata=metadata,
        X=X,
        y=y,
        admin=admin,
        target_ord=target_ord,
        origin_ord=origin_ord,
        horizon=horizon,
        partition_code=partition_code,
        branch_id=branch_id,
        persistence=persistence,
        branch_by_code=branch_by_code,
        feature_columns=tuple(matrix.feature_columns),
        audit=audit,
    )


@dataclass
class Stage3Helpers:
    """The baseline's Stage 3 fit/probability functions, injected once.

    design.md: reuse the released helpers rather than reimplementing the local
    fallback and single-class probability rules. They are passed in so the
    contract checks can drive the same routing logic without the pinned copy.
    """

    train_pooled: object
    train_partitioned: object
    predict_partitioned_probability: object
    predict_class1_probability: object
    min_partition_rows: int = MIN_PARTITION_TRAIN_ROWS
    source: str = ""


def verify_xgb_configuration(model, expected: dict = STAGE3_XGB_PARAMS) -> dict:
    """Read the fitted booster back and check it against Q7b.

    Q7b asks for the *effective* configuration at execution, so the check reads
    ``save_config()`` (what the booster actually trained with) rather than the
    keyword arguments that were passed in.
    """
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    learner = config["learner"]
    gradient = learner["gradient_booster"]
    tree = gradient["tree_train_param"]

    effective = {
        "xgboost_config_version": config.get("version"),
        "objective": learner["objective"]["name"],
        "booster": learner["learner_train_param"]["booster"],
        "device": learner["generic_param"].get("device", ""),
        "seed": int(learner["generic_param"]["seed"]),
        "n_jobs": int(learner["generic_param"]["n_jobs"]),
        "base_score": float(learner["learner_model_param"]["base_score"]),
        "num_feature": int(learner["learner_model_param"]["num_feature"]),
        "num_boosted_rounds": int(booster.num_boosted_rounds()),
        "num_trees": int(gradient["gbtree_model_param"]["num_trees"]),
        "num_parallel_tree": int(gradient["gbtree_model_param"]["num_parallel_tree"]),
        "tree_method": gradient["gbtree_train_param"]["tree_method"],
        "updater": gradient["gbtree_train_param"]["updater"],
        "max_depth": int(tree["max_depth"]),
        "min_child_weight": float(tree["min_child_weight"]),
        "learning_rate": float(tree["learning_rate"]),
        "subsample": float(tree["subsample"]),
        "colsample_bytree": float(tree["colsample_bytree"]),
        "reg_alpha": float(tree["reg_alpha"]),
        "reg_lambda": float(tree["reg_lambda"]),
        "gamma": float(tree["min_split_loss"]),
        "max_delta_step": float(tree["max_delta_step"]),
        "grow_policy": tree["grow_policy"],
        "max_bin": int(tree["max_bin"]),
        "scale_pos_weight": float(
            learner["objective"].get("reg_loss_param", {}).get("scale_pos_weight", 1.0)
        ),
        "eval_metric": [
            entry.get("name", "")
            for entry in learner.get("metrics", [])
            if isinstance(entry, dict)
        ],
        "missing": (
            "nan" if model.missing is None or np.isnan(model.missing) else model.missing
        ),
    }

    # The booster stores tree parameters in SINGLE precision (0.05 comes back as
    # 0.0500000007), so an exact comparison would fail on a correct model.
    checks = {
        "objective": (expected["objective"], "exact"),
        "booster": (expected["booster"], "exact"),
        "device": (expected["device"], "exact"),
        "tree_method": (expected["tree_method"], "exact"),
        "grow_policy": (expected["grow_policy"], "exact"),
        "seed": (int(expected["random_state"]), "exact"),
        "n_jobs": (int(expected["n_jobs"]), "exact"),
        "max_depth": (int(expected["max_depth"]), "exact"),
        "max_bin": (int(expected["max_bin"]), "exact"),
        "num_parallel_tree": (int(expected["num_parallel_tree"]), "exact"),
        "num_boosted_rounds": (int(expected["n_estimators"]), "exact"),
        "base_score": (float(expected["base_score"]), "float32"),
        "min_child_weight": (float(expected["min_child_weight"]), "float32"),
        "learning_rate": (float(expected["learning_rate"]), "float32"),
        "subsample": (float(expected["subsample"]), "float32"),
        "colsample_bytree": (float(expected["colsample_bytree"]), "float32"),
        "reg_alpha": (float(expected["reg_alpha"]), "float32"),
        "reg_lambda": (float(expected["reg_lambda"]), "float32"),
        "gamma": (float(expected["gamma"]), "float32"),
        "max_delta_step": (float(expected["max_delta_step"]), "float32"),
        "scale_pos_weight": (float(expected["scale_pos_weight"]), "float32"),
    }
    mismatches = []
    for key, (want, kind) in checks.items():
        got = effective.get(key)
        if kind == "float32":
            ok = got is not None and abs(float(got) - want) <= 1e-6 * max(1.0, abs(want))
        else:
            ok = got == want
        if not ok:
            mismatches.append({"parameter": key, "expected": want, "booster": got})

    if effective["eval_metric"] != [expected["eval_metric"]]:
        mismatches.append(
            {
                "parameter": "eval_metric",
                "expected": [expected["eval_metric"]],
                "booster": effective["eval_metric"],
            }
        )
    if mismatches:
        raise PipelineError(f"the fitted XGB booster contradicts Q7b: {mismatches}")

    return {
        "constructor_params": {
            key: _json_default(value) if isinstance(value, np.generic) else value
            for key, value in model.get_params().items()
        },
        "booster_effective": effective,
        "booster_config": config,
        "checked": sorted(checks) + ["eval_metric"],
        "note": (
            "n_estimators has no booster-config entry; the number of rounds the "
            "fitted booster actually holds is the equivalent fact. Tree parameters "
            "round-trip through float32, so they are compared with a 1e-6 tolerance."
        ),
    }


def local_model_routes(models: dict, group_test: np.ndarray,
                       min_rows: int = MIN_PARTITION_TRAIN_ROWS) -> tuple:
    """Per-row model route and fallback reason, mirroring the baseline helper.

    ``predict_partitioned_probability`` serves a row from its partition's model
    when that model exists, and from the pooled model otherwise (``None`` model,
    or a partition absent from the training pool, or ``-1``). Reproducing that
    decision here gives every row an explicit route which is then *verified*
    against the probabilities the helper actually returned.
    """
    n = len(group_test)
    route = np.full(n, "", dtype=object)
    reason = np.full(n, "", dtype=object)
    handled = np.zeros(n, dtype=bool)
    for code, model in models.items():
        code = int(code)
        mask = group_test == code
        if not mask.any():
            continue
        if model is not None:
            # The partition CODE identifies the route unambiguously; the branch
            # string can legitimately be "" (the root branch) and every row
            # already carries it in its own ``branch_id`` column.
            route[mask] = f"{ROUTE_LOCAL_PREFIX}{code}"
        else:
            route[mask] = ROUTE_POOLED
            reason[mask] = (
                f"local partition {code} had fewer than {min_rows} training rows "
                "or a single class"
            )
        handled[mask] = True
    unresolved = (~handled) & (group_test < 0)
    route[unresolved] = ROUTE_POOLED
    reason[unresolved] = "area unassigned by the frozen map (no donor within 100 km)"
    unseen = (~handled) & (group_test >= 0)
    route[unseen] = ROUTE_POOLED
    reason[unseen] = "partition unseen in this fold's training pool"
    if (route == "").any():
        raise PipelineError("a test row received no model route")
    return route, reason


def fit_fold(fold: Fold, panel: Stage3Panel, helpers: Stage3Helpers,
             imputer_factory, xgb_factory, threshold: float = DECISION_THRESHOLD) -> dict:
    """Refit all three learned arms from scratch for one (origin, horizon).

    R3/R5 in one place: the training pool is the 36 CALENDAR months [O-35, O]
    for this horizon over every eligible area; each of those rows keeps its own
    ``T'-H`` predictors; one imputer is fitted on that pool and reused by the
    pooled RF and every local RF; XGB sees the same rows pre-imputation with
    native NaN; and all three classify with p1 > .5.
    """
    horizon = fold.horizon
    origin, target = fold.origin_ord, fold.target_ord
    window_start = fold.window_start_ord

    same_horizon = panel.horizon == horizon
    train_mask = same_horizon & (panel.target_ord >= window_start) & (
        panel.target_ord <= origin
    )
    test_mask = same_horizon & (panel.target_ord == target)

    record = {
        "fold_id": fold.fold_id,
        "period": fold.period,
        "horizon_months": horizon,
        "target_month": fold.target_month,
        "origin_month": fold.origin_month,
        "train_window_start": pdata.month_label([window_start])[0],
        "train_window_end": fold.origin_month,
        "train_window_months": TRAIN_WINDOW_MONTHS,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "fitted": False,
        "skip_reason": "",
    }

    if record["train_rows"] == 0:
        # R4/design.md: an empty GLOBAL training pool is a reported stop, never
        # a fabricated model.
        raise PipelineError(
            f"fold {fold.fold_id} has an empty training pool over "
            f"[{record['train_window_start']}, {record['train_window_end']}]"
        )
    if record["test_rows"] == 0:
        # R4: zero support, explicitly not a zero F1 and not a failure.
        record["skip_reason"] = "no valid label in this target month"
        return {"record": record, "rows": None, "training_keys": None,
                "imputer_fills": None, "xgb_model": None}

    # --- own-origin / no-post-origin-truth gates --------------------------
    if panel.target_ord[train_mask].max() > origin:
        raise PipelineError(f"fold {fold.fold_id} trains on a post-origin target month")
    if panel.origin_ord[train_mask].max() >= origin:
        raise PipelineError(
            f"fold {fold.fold_id} has a training row whose own origin reaches the "
            "fit origin; every historical row must use its own T-H predictors"
        )
    if (panel.target_ord[test_mask] != target).any():
        raise PipelineError(f"fold {fold.fold_id} test rows are not one target month")
    if (panel.origin_ord[test_mask] != origin).any():
        raise PipelineError(f"fold {fold.fold_id} test rows do not share O = T - H")
    overlap = set(zip(panel.admin[train_mask], panel.target_ord[train_mask])) & set(
        zip(panel.admin[test_mask], panel.target_ord[test_mask])
    )
    if overlap:
        raise PipelineError(
            f"fold {fold.fold_id} trains on its own test area-months"
        )

    X_train = panel.X[train_mask]
    y_train = panel.y[train_mask]
    g_train = panel.partition_code[train_mask]
    X_test = panel.X[test_mask]
    g_test = panel.partition_code[test_mask]

    started = time.time()

    # --- Q6c: ONE imputer per (origin, horizon), fitted on training only ----
    imputer = imputer_factory()
    imputer.fit(X_train)
    X_train_imputed = np.asarray(imputer.transform(X_train), dtype=np.float64)
    X_test_imputed = np.asarray(imputer.transform(X_test), dtype=np.float64)
    if not np.isfinite(X_train_imputed).all() or not np.isfinite(X_test_imputed).all():
        raise PipelineError(f"fold {fold.fold_id} still holds NaN after imputation")

    # --- pooled RF, local RFs (same transform), pooled XGB (native NaN) -----
    rf_started = time.time()
    pooled = helpers.train_pooled(X_train_imputed, y_train)
    local_models = helpers.train_partitioned(
        X_train_imputed, y_train, g_train, min_samples=helpers.min_partition_rows
    )
    rf_seconds = time.time() - rf_started

    xgb_started = time.time()
    xgb_model = xgb_factory()
    xgb_model.fit(X_train, y_train)  # Q6c/Q7b: pre-imputation, native NaN
    xgb_seconds = time.time() - xgb_started

    prob_pooled = np.asarray(
        helpers.predict_class1_probability(pooled, X_test_imputed), dtype=np.float64
    )
    prob_partitioned = np.asarray(
        helpers.predict_partitioned_probability(
            local_models, pooled, X_test_imputed, g_test
        ),
        dtype=np.float64,
    )
    prob_xgb = np.asarray(
        helpers.predict_class1_probability(xgb_model, X_test), dtype=np.float64
    )

    route, reason = local_model_routes(
        local_models, g_test, helpers.min_partition_rows
    )

    # Verify the route labels against the probabilities the helper returned:
    # a pooled-routed row must carry EXACTLY the pooled probability, and a
    # locally-routed row must carry exactly that local model's probability.
    pooled_routed = route == ROUTE_POOLED
    if pooled_routed.any() and not np.array_equal(
        prob_partitioned[pooled_routed], prob_pooled[pooled_routed]
    ):
        raise PipelineError(
            f"fold {fold.fold_id}: a pooled-routed row does not carry the pooled "
            "probability; the route labels and the partitioned helper disagree"
        )
    for code, model in local_models.items():
        if model is None:
            continue
        mask = g_test == int(code)
        if not mask.any():
            continue
        expected = np.asarray(
            helpers.predict_class1_probability(model, X_test_imputed[mask]),
            dtype=np.float64,
        )
        if not np.array_equal(prob_partitioned[mask], expected):
            raise PipelineError(
                f"fold {fold.fold_id}: partition {code} rows do not carry that "
                "local model's probability"
            )

    for name, probability in (
        ("partitioned", prob_partitioned),
        ("pooled", prob_pooled),
        ("xgb", prob_xgb),
    ):
        if len(probability) != record["test_rows"]:
            raise PipelineError(f"fold {fold.fold_id}: {name} lost test rows")
        if not np.isfinite(probability).all():
            raise PipelineError(f"fold {fold.fold_id}: {name} produced a non-finite p1")
        if (probability < 0).any() or (probability > 1).any():
            raise PipelineError(f"fold {fold.fold_id}: {name} p1 outside [0, 1]")

    # Q7a: strict; an exact .5 is class 0.
    pred_partitioned = (prob_partitioned > threshold).astype(np.int64)
    pred_pooled = (prob_pooled > threshold).astype(np.int64)
    pred_xgb = (prob_xgb > threshold).astype(np.int64)

    # --- rows --------------------------------------------------------------
    metadata = panel.metadata.loc[test_mask].reset_index(drop=True)
    rows = pd.DataFrame(
        {
            "admin_code": metadata["admin_code"].to_numpy(dtype=np.int64),
            "country_en": metadata["country_en"].astype(str).to_numpy(),
            "ISO3": metadata["ISO3"].fillna("").astype(str).to_numpy(),
            "target_month": fold.target_month,
            "origin_month": fold.origin_month,
            "horizon_months": horizon,
            "period": fold.period,
            "fold_id": fold.fold_id,
            "ipcch_food_crisis": panel.y[test_mask],
            "prob_partitioned_rf": prob_partitioned,
            "pred_partitioned_rf": pred_partitioned,
            "prob_pooled_rf": prob_pooled,
            "pred_pooled_rf": pred_pooled,
            "prob_xgb": prob_xgb,
            "pred_xgb": pred_xgb,
            "persistence_pred": panel.persistence["value"][test_mask],
            "persistence_source_month": pdata.month_label(
                panel.persistence["source_ord"][test_mask]
            ),
            "persistence_age_months": panel.persistence["age"][test_mask],
            "branch_id": metadata["branch_id"].astype(str).to_numpy(),
            "partition_code": g_test,
            "assignment_source": metadata["assignment_source"].astype(str).to_numpy(),
            "donor_admin_code": metadata["donor_admin_code"].to_numpy(),
            "donor_distance_km": metadata["donor_distance_km"].to_numpy(),
            "model_route": route,
            "model_fallback_reason": reason,
        }
    )
    rows = rows[list(PREDICTION_COLUMNS)]

    training_keys = pd.DataFrame(
        {
            "fold_id": fold.fold_id,
            "admin_code": panel.admin[train_mask],
            "target_month": pdata.month_label(panel.target_ord[train_mask]),
            "origin_month": pdata.month_label(panel.origin_ord[train_mask]),
            "horizon_months": horizon,
            "ipcch_food_crisis": y_train,
            "partition_code": g_train,
        }
    )

    fills = imputer_fill_table(imputer, panel.feature_columns)
    fills.insert(0, "fold_id", fold.fold_id)

    # --- contrast audit -----------------------------------------------------
    # The two RF arms are NOT the same model even when the map holds a single
    # partition: local training excludes rows in unassigned areas, so the local
    # forest sees a strictly smaller pool than the pooled forest. Everything
    # below MEASURES that difference; nothing forces the arms to agree.
    unassigned_train = int((g_train < 0).sum())
    assigned_areas = int(pd.unique(panel.admin[train_mask][g_train >= 0]).size)
    fitted_codes = sorted(int(c) for c, m in local_models.items() if m is not None)
    fallback_codes = sorted(int(c) for c, m in local_models.items() if m is None)
    same = prob_partitioned == prob_pooled
    record.update(
        {
            "fitted": True,
            "train_areas": int(pd.unique(panel.admin[train_mask]).size),
            "train_positives": int(y_train.sum()),
            "train_positive_rate": float(y_train.mean()),
            "train_classes": int(np.unique(y_train).size),
            "test_areas": int(pd.unique(panel.admin[test_mask]).size),
            "test_positives": int(panel.y[test_mask].sum()),
            "test_persistence_available": int(
                panel.persistence["available"][test_mask].sum()
            ),
            "partitions_in_training": int(np.unique(g_train[g_train >= 0]).size),
            "local_models_fitted": len(fitted_codes),
            "local_models_fallback": len(fallback_codes),
            "local_fitted_codes": fitted_codes,
            "local_fallback_codes": fallback_codes,
            "unassigned_train_rows_excluded_from_local": unassigned_train,
            "train_areas_in_local_pools": assigned_areas,
            "local_train_rows": {
                str(int(code)): int((g_train == int(code)).sum())
                for code in np.unique(g_train[g_train >= 0])
            },
            "test_rows_by_route": {
                str(k): int(v) for k, v in pd.Series(route).value_counts().items()
            },
            "test_rows_unassigned": int((g_test < 0).sum()),
            "partitioned_equals_pooled_rows": int(same.sum()),
            "partitioned_identical_to_pooled": bool(same.all()),
            "predicted_positives": {
                "partitioned_rf": int(pred_partitioned.sum()),
                "pooled_rf": int(pred_pooled.sum()),
                "xgb": int(pred_xgb.sum()),
            },
            "rf_seconds": round(rf_seconds, 2),
            "xgb_seconds": round(xgb_seconds, 2),
            "seconds": round(time.time() - started, 2),
        }
    )

    # Two ROUTING checks, not identity requirements. Each asserts only what is
    # arithmetically forced by which model served the row; neither constrains a
    # locally-served row, whose probability is expected to differ.
    #
    # (1) An unassigned test row is served by the pooled model in BOTH arms.
    unassigned_test = g_test < 0
    if unassigned_test.any() and not np.array_equal(
        prob_partitioned[unassigned_test], prob_pooled[unassigned_test]
    ):
        raise PipelineError(
            f"fold {fold.fold_id}: an unassigned test row differs between the "
            "partitioned and pooled arms, but both must use the pooled model"
        )
    # (2) If one local model happened to train on the ENTIRE pooled pool -- same
    # rows, same seed -- it is literally the pooled forest. This never holds
    # while any unassigned area contributes a training row.
    covers_pool = (
        len(fitted_codes) == 1
        and unassigned_train == 0
        and int((g_train == fitted_codes[0]).sum()) == len(y_train)
    )
    record["local_model_covers_entire_pool"] = bool(covers_pool)
    if covers_pool and not same.all():
        raise PipelineError(
            f"fold {fold.fold_id}: one local model trained on the entire pool with "
            "the same seed, so it must be the same forest as the pooled model"
        )

    return {
        "record": record,
        "rows": rows,
        "training_keys": training_keys,
        "imputer_fills": fills,
        "xgb_model": xgb_model,
    }


def stage3_coverage(predictions: pd.DataFrame, folds: list,
                    records: list) -> dict:
    """Per-arm coverage on the scheduled keys, and the E_all / E_persist sizes.

    Q4b: E_all is every valid scheduled test key; E_persist is its
    history-available subset. A missing LEARNED prediction is a failure here,
    not a shrunk cohort, so the check is on completeness rather than on masks.
    """
    expected = int(sum(r["test_rows"] for r in records))
    if len(predictions) != expected:
        raise PipelineError(
            f"wrote {len(predictions)} prediction rows for {expected} scheduled "
            "test rows"
        )
    keys = ["admin_code", "target_month", "horizon_months"]
    if predictions.duplicated(keys).any():
        raise PipelineError("prediction keys are not unique")

    coverage = {}
    for arm in ("partitioned_rf", "pooled_rf", "xgb"):
        prob = predictions[f"prob_{arm}"]
        pred = predictions[f"pred_{arm}"]
        missing = int(prob.isna().sum() + pred.isna().sum())
        if missing:
            raise PipelineError(
                f"{missing} scheduled keys have no {arm} prediction; a missing "
                "learned prediction is a failure (Q4b)"
            )
        ties = int(((prob == DECISION_THRESHOLD) & (pred == 1)).sum())
        if ties:
            raise PipelineError(f"{ties} rows classify an exact .5 as class 1 (Q7a)")
        coverage[arm] = {
            "rows": int(len(predictions)),
            "predicted_positive": int(pred.sum()),
            "missing": 0,
        }

    persist_available = predictions["persistence_pred"].notna()
    coverage["persistence"] = {
        "rows": int(persist_available.sum()),
        "predicted_positive": int(predictions.loc[persist_available,
                                                  "persistence_pred"].sum()),
        "missing": int((~persist_available).sum()),
    }

    by_period = {}
    for period, block in predictions.groupby("period"):
        available = block["persistence_pred"].notna()
        by_period[str(period)] = {
            "E_all": int(len(block)),
            "E_persist": int(available.sum()),
            "persistence_coverage": (
                float(available.mean()) if len(block) else None
            ),
            "by_horizon": {
                str(int(h)): {
                    "E_all": int(len(part)),
                    "E_persist": int(part["persistence_pred"].notna().sum()),
                    "positives": int(part["ipcch_food_crisis"].sum()),
                    "areas": int(part["admin_code"].nunique()),
                    "countries": int(part["country_en"].nunique()),
                }
                for h, part in block.groupby("horizon_months")
            },
        }

    return {
        "scheduled_folds": len(folds),
        "folds_with_rows": int(sum(1 for r in records if r["test_rows"] > 0)),
        "folds_empty": int(sum(1 for r in records if r["test_rows"] == 0)),
        "arms": coverage,
        "cohorts": by_period,
        "note": (
            "E_persist is the history-available subset of E_all; rows without "
            "history keep an explicit missing persistence and stay in E_all (Q4b)"
        ),
    }


# ==========================================================================
# Pipeline
# ==========================================================================


def run(args) -> int:
    context = RunContext(Path(args.runs_dir), args.run_id)
    context.log(f"run directory: {context.root}")
    try:
        _execute(context, args)
    except Exception as error:  # noqa: BLE001 - the run must record why it stopped
        detail = traceback.format_exc()
        context.log(f"FAILED: {error}")
        context.log(detail)
        context.fail(str(error), detail)
        return 1
    context.log("stage 1 and stage 3 finished; reporting is a separate entry point")
    return 0


def _execute(context: RunContext, args) -> None:
    source_root = Path(args.source_root)
    source_csv = source_root / "raw" / "IPCCH_2026_completed.csv"
    release_zip = Path(args.release_zip)

    # -- 0. preflight ----------------------------------------------------
    started = time.time()
    context.log("preflight: hashing inputs and recording the runtime")
    if not source_csv.is_file():
        raise PipelineError(f"pinned source CSV not found: {source_csv}")
    runtime = runtime_identity()
    experiment_hashes = {
        name: sha256_file(EXPERIMENT_DIR / name)
        for name in EXPERIMENT_SOURCE_FILES
        if (EXPERIMENT_DIR / name).is_file()
    }
    baseline = brt.extract_baseline(release_zip, context.baseline_dir)
    baseline = brt.apply_polygon_refinement_patch(baseline)
    context.log(
        f"baseline extracted and patched: {baseline.payload_files_verified} payload "
        f"files verified"
    )
    context.manifest["runtime"] = runtime
    context.manifest["code"] = {
        "experiment_files_sha256": experiment_hashes,
        "baseline": baseline.to_dict(),
        "release_zip": str(release_zip),
    }
    context.manifest["source"] = {
        "source_root": str(source_root),
        "csv": str(source_csv),
        "csv_sha256": sha256_file(source_csv),
        "expected_csv_sha256": pdata.SOURCE_SHA256,
    }
    if context.manifest["source"]["csv_sha256"] != pdata.SOURCE_SHA256:
        raise PipelineError("pinned source CSV hash mismatch")
    context.stage("preflight", {"seconds": round(time.time() - started, 1)})

    # -- 1. target ledger ------------------------------------------------
    started = time.time()
    context.log("R1: building the target ledger")
    ledger = pdata.build_target_ledger(source_csv, verify_hash=False)
    gate = pdata.check_target_gate(ledger)
    if not gate["gate_pass"]:
        raise PipelineError(f"target gate failed: {gate['gate_mismatches']}")

    # Ablation knob: restrict the cohort AFTER the source gate, never before. The gate
    # compares against hard-coded audited counts, so a restricted ledger would fail it
    # by construction and we would lose the proof that this run read the same pinned
    # source. Filtering here keeps that proof and makes the restriction a declared,
    # measured step.
    if COHORT_FILTER is not None:
        before = ledger.summary()
        is_ch = ledger.frame["admin_code"] >= CH_ADMIN_CODE_FLOOR
        keep = ~is_ch if COHORT_FILTER == "non_ch" else is_ch
        dropped = ledger.frame[~keep]
        ledger = dataclass_replace(
            ledger, frame=ledger.frame[keep].reset_index(drop=True)
        )
        after = ledger.summary()
        cohort_evidence = {
            "filter": COHORT_FILTER,
            "rule": f"admin_code >= {CH_ADMIN_CODE_FLOOR} is Cadre Harmonise",
            "applied_after": "check_target_gate, which still validated the full source",
            "kept": {k: after[k] for k in ("valid", "positive", "negative", "areas_total")},
            "dropped": {
                "rows": int(len(dropped)),
                "areas": int(dropped["admin_code"].nunique()),
                "valid": int(before["valid"] - after["valid"]),
                "positive": int(before["positive"] - after["positive"]),
                "negative": int(before["negative"] - after["negative"]),
            },
            "full_source_gate": {
                k: before[k] for k in ("valid", "positive", "negative", "areas_total")
            },
        }
        _write_json(context.data_dir / "cohort_filter.json", cohort_evidence)
        context.log(
            f"cohort {COHORT_FILTER}: kept {after['valid']} valid rows over "
            f"{after['areas_total']} areas; dropped {cohort_evidence['dropped']['valid']} "
            f"rows over {cohort_evidence['dropped']['areas']} areas"
        )
        gate = dict(gate, cohort_filter=cohort_evidence)

    valid = ledger.valid()
    valid.to_csv(context.data_dir / "target_ledger_valid.csv.gz", index=False)
    # The 1.18M rows rejected for "no P1-P4" are the empty scaffold; only the
    # rows that had a complete-looking distribution and still failed are worth
    # keeping row by row.
    interesting = ledger.frame[
        (ledger.frame["target_valid"] == 0)
        & (ledger.frame["target_invalid_reason"] != "missing_phase_1_to_4")
    ]
    interesting.to_csv(context.data_dir / "target_ledger_rejected.csv.gz", index=False)
    _write_json(context.data_dir / "target_gate.json", gate)
    context.log(
        f"R1 gate passed: {gate['valid']} valid / {gate['positive']} positive "
        f"in {gate['areas_total']} areas"
    )
    context.stage(
        "target",
        {"seconds": round(time.time() - started, 1), "gate": gate},
    )

    # -- 2. feature matrix -----------------------------------------------
    started = time.time()
    context.log("R3: assembling the 93-column feature matrix")
    matrix = pdata.build_feature_matrix(ledger, source_csv)
    if len(matrix.feature_columns) != pdata.EXPECTED_FEATURE_COUNT:
        raise PipelineError("feature schema width changed")
    # Values as .npy and metadata as CSV: exact float round-trip, and far cheaper
    # than a 170k x 101 gzipped CSV. The schema file makes the pair self-describing.
    np.save(context.data_dir / "feature_values.npy",
            matrix.frame[list(matrix.feature_columns)].to_numpy(dtype=np.float64))
    matrix.frame[list(matrix.metadata_columns)].to_csv(
        context.data_dir / "feature_metadata.csv.gz", index=False
    )
    _write_json(
        context.data_dir / "feature_schema.json",
        {
            "feature_columns": list(matrix.feature_columns),
            "metadata_columns": list(matrix.metadata_columns),
            "blocks": {k: list(v) for k, v in pdata.FEATURE_BLOCKS.items()},
            "rows": int(len(matrix.frame)),
            "values_file": "feature_values.npy",
            "metadata_file": "feature_metadata.csv.gz",
        },
    )
    _write_json(
        context.data_dir / "feature_audit.json",
        {
            "audit": matrix.audit,
            "infinity_audit": matrix.infinity_audit,
            "nan_rates": matrix.nan_rates(),
        },
    )
    context.log(
        f"R3: {len(matrix.frame)} rows x {len(matrix.feature_columns)} columns, "
        f"origins {matrix.audit['origin_month_min']}..{matrix.audit['origin_month_max']}"
    )
    context.stage(
        "features",
        {
            "seconds": round(time.time() - started, 1),
            "rows": int(len(matrix.frame)),
            "columns": len(matrix.feature_columns),
            "audit": matrix.audit,
        },
    )

    # -- 3. geography ----------------------------------------------------
    started = time.time()
    context.log("R2/Q8g: validating, repairing and building adjacency")
    geography = pdata.prepare_geography(
        source_root,
        context.geography_dir,
        baseline_root=baseline.root,
        build_adjacency=True,
        compare_adjacency=True,
    )
    if geography.adjacency is None:
        raise PipelineError("adjacency was not built; polygon contiguity needs it")
    repair = geography.audit["repair"]
    context.log(
        f"R2: {repair['unchanged_valid']} geometries untouched, "
        f"{repair['repaired_polygonal']} repaired in place, "
        f"{repair['repaired_collection_areal_extracted']} areal components extracted "
        f"under the D1 scope extension; {len(geography.adjacency.area_ids)} polygons "
        f"in the adjacency ({geography.adjacency.audit['isolated_polygons']} isolated)"
    )
    context.stage(
        "geography",
        {
            "seconds": round(time.time() - started, 1),
            "repair": geography.audit["repair"],
            "area_universe": geography.audit["area_universe"],
            "adjacency": {
                key: value
                for key, value in geography.adjacency.audit.items()
                if key != "neighbor_counts"
            },
        },
    )

    # -- 4. original-outcome split ---------------------------------------
    started = time.time()
    context.log("R4/Q5a: splitting original 2014-2022 outcomes")
    universe = np.asarray(
        sorted(int(v) for v in geography.reference_coordinates[pdata.REFERENCE_ID_COLUMN]),
        dtype=np.int64,
    )
    split = pdata.build_stage1_split(ledger, all_area_ids=universe)
    split_gate = pdata.check_stage1_split_gate(split)
    if COHORT_FILTER is None:
        # Full cohort: the audited-count gate must pass, which is what proves this run
        # reproduces the baseline.
        if not split_gate["gate_pass"]:
            raise PipelineError(f"split gate failed: {split_gate['gate_mismatches']}")
    else:
        # Restricted cohort: these expectations are audited constants for the FULL
        # source, so a declared restriction mismatches them by construction, exactly as
        # it would have mismatched check_target_gate. The comparison is recorded as
        # not-applicable rather than silently dropped, and the realized counts are kept
        # so the restricted split is still fully measured.
        split_gate = dict(
            split_gate,
            gate_pass=None,
            gate_applicable=False,
            gate_skipped_reason=(
                f"cohort filter {COHORT_FILTER!r} is active; the audited split counts "
                "describe the unrestricted source and cannot apply. The unrestricted "
                "target gate still ran and passed earlier in this run."
            ),
        )
        context.log(
            f"R4 split gate not applicable under cohort {COHORT_FILTER}; realized "
            f"{split_gate['fit']} fit / {split_gate['validation']} validation outcomes, "
            f"{split_gate['singleton_areas']} singletons"
        )
    split.outcomes.to_csv(context.stage1_dir / "split_outcomes.csv.gz", index=False)
    _write_json(context.stage1_dir / "split_gate.json", split_gate)
    if COHORT_FILTER is None:
        context.log(
            f"R4 gate passed: {split_gate['fit']} fit / {split_gate['validation']} "
            f"validation outcomes, {split_gate['singleton_areas']} singletons"
        )

    design = build_stage1_design(matrix, split)
    _write_json(context.stage1_dir / "design_audit.json", design.audit)
    design.frame[list(matrix.metadata_columns) + ["split_role"]].to_csv(
        context.stage1_dir / "design_rows.csv.gz", index=False
    )
    design.singleton_frame[list(matrix.metadata_columns) + ["split_role"]].to_csv(
        context.stage1_dir / "singleton_rows.csv.gz", index=False
    )
    context.log(
        f"Stage1 design: {design.audit['learning_rows']} model rows "
        f"({design.audit['fit_rows']} fit / {design.audit['validation_rows']} validation) "
        f"across {design.audit['active_groups']} groups; targets up to "
        f"{design.audit['target_month_max']}"
    )
    context.stage(
        "split",
        {
            "seconds": round(time.time() - started, 1),
            "gate": split_gate,
            "design": design.audit,
        },
    )

    # -- 5. Stage 1 fit ---------------------------------------------------
    started = time.time()
    context.log("Stage1: entering the pinned baseline")
    fit_payload = _run_stage1_fit(context, baseline, geography, design, matrix, split)
    stage1_stage = {"seconds": round(time.time() - started, 1), **fit_payload["stage"]}
    # R3/A2: the three-namespace readback is required evidence for EVERY cell, so it
    # has to reach the manifest. Only fit_payload["stage"] is persisted, so merge it in.
    if "split_gate" in fit_payload:
        stage1_stage["split_gate"] = fit_payload["split_gate"]
    context.stage("stage1_fit", stage1_stage)

    # -- 6. map, donors, singletons --------------------------------------
    context.manifest["stages"]["assignment"] = fit_payload["assignment_stage"]
    context.manifest["stages"]["singletons"] = fit_payload["singleton_stage"]
    context.save()

    # -- 7. Stage 3 rolling four-arm forecasts ---------------------------
    started = time.time()
    context.log("Stage3: rolling four-arm forecasts")
    stage3 = _run_stage3(
        context,
        baseline,
        matrix=matrix,
        ledger=ledger,
        assignments=fit_payload["assignments"],
        learned_audit=fit_payload["stage"]["learned_map"],
    )
    context.stage("stage3", {"seconds": round(time.time() - started, 1), **stage3})
    context.finish()


def _run_stage1_fit(context: RunContext, baseline, geography, design: Stage1Design,
                    matrix, split: "pdata.Stage1Split") -> dict:
    """The one GeoRF.fit, then everything that must read its saved state."""
    adjacency = geography.adjacency
    groups = design.groups
    polygon_group_mapping = {
        int(index): int(group)
        for index, group in adjacency.polygon_group_mapping.items()
    }
    mapped_groups = set(polygon_group_mapping.values())
    unmapped = sorted(set(int(g) for g in np.unique(groups)) - mapped_groups)
    if unmapped:
        raise PipelineError(
            f"{len(unmapped)} fitted groups are absent from the polygon mapping, "
            f"e.g. {unmapped[:5]}"
        )
    if len(polygon_group_mapping) != len(adjacency.polygon_centroids):
        raise PipelineError("polygon mapping does not cover every polygon")

    stdout_path = context.stage1_dir / "georf_fit_stdout.txt"
    result: dict = {}

    with brt.baseline_imports(baseline) as (config, georf_module):
        # design.md: the IPCCH schema is already exact, so the baseline's own
        # drop list must not remove columns behind our back. `from config import *`
        # copied the binding into the GeoRF module at import time, and the module
        # also reads the lowercase `feature_drop` alias, so both are neutralised
        # in the module namespace where `_get_feature_drop_config` looks.
        disabled_drop = {"enable": False, "cols": [], "patterns": []}
        georf_module.FEATURE_DROP = disabled_drop
        georf_module.feature_drop = disabled_drop

        # Ablation knob: the class-1 F1 split gate. Same `from config import *`
        # hazard as FEATURE_DROP, but worse, because it binds into THREE namespaces:
        # src/tests/sig_test.py:12 and src/partition/transformation.py:13 each copy it
        # from config, and transformation.py:17 then re-exports sig_test's copy over
        # its own. Patching `config` alone would leave the learner gating at 0.01
        # while REPORTED_CONFIG_KEYS faithfully reports the requested value.
        # The readback is unconditional. A default-gate cell needs run-bound proof that
        # it gated at 0.01 just as much as an overridden cell needs proof it gated at
        # 0.005 - "we did not touch it" is an assumption, not evidence, and a zero-split
        # outcome is consistent with 0.01 without demonstrating it.
        import src.partition.transformation as transformation_module  # noqa: PLC0415
        import src.tests.sig_test as sig_test_module  # noqa: PLC0415

        gate = SPLIT_GATE_OVERRIDE
        if gate is not None:
            for module in (config, sig_test_module, transformation_module):
                module.MIN_CLASS_1_IMPROVEMENT_THRESHOLD = float(gate)
        readback = {
            "config": getattr(config, "MIN_CLASS_1_IMPROVEMENT_THRESHOLD", None),
            "src.tests.sig_test": getattr(
                sig_test_module, "MIN_CLASS_1_IMPROVEMENT_THRESHOLD", None
            ),
            "src.partition.transformation": getattr(
                transformation_module, "MIN_CLASS_1_IMPROVEMENT_THRESHOLD", None
            ),
        }
        expected = float(gate) if gate is not None else float(
            getattr(config, "MIN_CLASS_1_IMPROVEMENT_THRESHOLD")
        )
        disagreeing = {
            name: value for name, value in readback.items() if value != expected
        }
        if disagreeing:
            raise PipelineError(
                f"split gate disagrees across consuming namespaces: {disagreeing}; "
                f"expected {expected} "
                f"({'override' if gate is not None else 'pinned default'})"
            )
        result["split_gate"] = {
            "source": "cli_override" if gate is not None else "pinned_config_default",
            "effective": expected,
            "readback": readback,
        }

        from src.customize.customize import OutOfRangeImputer  # noqa: PLC0415

        # Read the effective values back from the modules. The GeoRF namespace is
        # reported separately because `from config import *` copied the bindings at
        # import time, so a later `config.X = ...` does not reach `_get_feature_drop
        # _config`, which reads GeoRF's own globals.
        effective_config = {
            "config_module": {key: getattr(config, key, None) for key in REPORTED_CONFIG_KEYS},
            "georf_module": {
                key: getattr(georf_module, key, None)
                for key in ("FEATURE_DROP", "feature_drop", "MIN_DEPTH", "MAX_DEPTH",
                            "N_JOBS", "NUM_CLASS", "CONTIGUITY", "CONTIGUITY_TYPE",
                            "MIN_COMPONENT_SIZE", "MODEL_CHOICE")
            },
        }
        module_locations = dict(baseline.module_locations)
        module_locations["src.customize.customize"] = sys.modules[
            "src.customize.customize"
        ].__file__
        for name, location in module_locations.items():
            if not str(location).startswith(str(baseline.root)):
                raise PipelineError(f"module {name} escaped the pinned copy: {location}")

        # Q6c: one imputer, fitted on genuine fitting views only.
        imputer = OutOfRangeImputer(strategy="max_plus", multiplier=100.0)
        imputer, (X_imputed, singleton_imputed) = apply_stage1_imputation(
            imputer, design.X, design.x_set, design.singleton_X
        )
        fills = imputer_fill_table(imputer, matrix.feature_columns)
        fills.to_csv(context.stage1_dir / "imputer_fill_values.csv", index=False)
        context.log(
            f"Q6c: imputer fitted on {int((design.x_set == 0).sum())} fitting rows; "
            f"{int(fills['all_missing_in_fit'].sum())} columns were all-missing"
        )
        if not np.isfinite(X_imputed).all():
            raise PipelineError("imputed Stage1 matrix still holds NaN/inf")

        # Pre-fit validation of everything GeoRF will not check for us.
        if X_imputed.shape != (len(design.y), len(matrix.feature_columns)):
            raise PipelineError("imputed matrix shape disagrees with the schema")
        if not np.isin(design.x_set, (0, 1)).all():
            raise PipelineError("X_set must be 0/1")
        if len(groups) != len(design.y):
            raise PipelineError("X_group length disagrees with y")

        polygon_contiguity_info = {
            "polygon_centroids": adjacency.polygon_centroids,
            "polygon_group_mapping": polygon_group_mapping,
            "neighbor_distance_threshold": None,
            "adjacency_dict": adjacency.adjacency_dict,
        }

        context.log(
            f"Stage1 fit: X={X_imputed.shape}, groups={pd.unique(groups).size}, "
            f"MIN_DEPTH={config.MIN_DEPTH}, MAX_DEPTH={config.MAX_DEPTH}, "
            f"N_JOBS={config.N_JOBS} (inherited, not pinned)"
        )

        fit_started = time.time()
        with brt.working_directory(context.stage1_dir) as stage1_cwd:
            with open(stdout_path, "w", encoding="utf-8", errors="backslashreplace") as sink:
                with contextlib.redirect_stdout(sink):
                    model = georf_module.GeoRF(
                        min_model_depth=config.MIN_DEPTH,
                        max_model_depth=config.MAX_DEPTH,
                    )
                    model.fit(
                        X_imputed,
                        design.y,
                        groups,
                        # design.md prescribes split={"X_set": ...} (which also
                        # length-checks it); the same array is passed positionally
                        # as X_set so either reading of the contract holds.
                        X_set=design.x_set,
                        split={"X_set": design.x_set},
                        contiguity_type="polygon",
                        polygon_contiguity_info=polygon_contiguity_info,
                        feature_names=list(matrix.feature_columns),
                        print_to_file=False,
                        track_partition_metrics=False,
                        VIS_DEBUG_MODE=False,
                    )
            fit_seconds = time.time() - fit_started
            model_dir = Path(stage1_cwd) / model.model_dir
            context.log(f"GeoRF.fit returned in {fit_seconds:.1f}s -> {model.model_dir}")

            # Gate: inherited code catches rendering and diagnostic exceptions, so
            # a quiet run proves nothing. Assert the artefacts instead.
            required = {
                "s_branch": model_dir / "space_partitions" / "s_branch.pkl",
                "branch_table": model_dir / "space_partitions" / "branch_table.npy",
                "X_branch_id": model_dir / "space_partitions" / "X_branch_id.npy",
                "feature_reference": model_dir / "feature_name_reference.csv",
                "val_coverage": model_dir / "val_coverage_by_group.csv",
                "root_checkpoint_dir": model_dir / "checkpoints",
            }
            absent = [name for name, path in required.items() if not path.exists()]
            if absent:
                raise PipelineError(f"mandatory Stage1 artefacts missing: {absent}")

            s_branch = pd.read_pickle(required["s_branch"])
            branch_table = np.load(required["branch_table"])
            saved_branch_ids = np.load(required["X_branch_id"], allow_pickle=False)

            learned = extract_learned_map(s_branch)
            reconciliation = reconcile_learned_map(
                learned, groups, saved_branch_ids, model_dir / "checkpoints"
            )
            context.log(
                f"map: {learned.audit['terminal_branch_count']} terminal branches "
                f"(depth<={learned.audit['max_branch_depth']}), "
                f"{learned.audit['learned_areas']} learned areas, "
                f"{learned.audit['placeholder_areas']} default-root placeholders"
            )
            if learned.audit["accepted_splits"] == 0:
                # A null partition is a legitimate result of the frozen gate, but it
                # must never be mistaken for a partitioned run downstream.
                context.log(
                    "NULL PARTITION: no split passed the strict >.01 class-1 F1 gate; "
                    "the learned map is the root branch alone and every area, learned "
                    "or completed, routes to the same root model"
                )
                context.manifest["stage1_outcome"] = "no_partition_learned"
                context.manifest["limitations"] = list(
                    context.manifest["limitations"]
                ) + [
                    "Stage1 accepted no split. With a single terminal partition a "
                    "partitioned-vs-pooled contrast no longer isolates spatial "
                    "heterogeneity: it reduces to a local model trained on the "
                    "assigned areas versus a pooled model trained on the whole "
                    "universe. See stage3/partitioned_vs_pooled_contrast.json"
                ]

            donors = eligible_donor_table(
                learned, split, geography.reference_coordinates
            )
            context.log(f"Q5d: {len(donors)} eligible donors")
            assignments = complete_assignments(
                universe=geography.reference_coordinates[pdata.REFERENCE_ID_COLUMN]
                .to_numpy(dtype=np.int64),
                coordinates=geography.reference_coordinates,
                learned=learned,
                donors=donors,
            )
            summary = assignment_summary(assignments)
            context.log(
                f"Q8r: {summary['learned']} learned + {summary['nearest_donor']} "
                f"donor-completed + {summary['unresolved']} unresolved "
                f"= {summary['universe']}"
            )

            # -- supplementary singleton scoring (after the map is frozen) --
            singleton_stage = _score_singletons(
                context, model, design, singleton_imputed, assignments
            )

            # Actual estimator settings, read back from the saved root model.
            model.model.load("")
            estimator_params = {
                key: value
                for key, value in model.model.model.get_params().items()
                if key
                in (
                    "n_estimators",
                    "max_depth",
                    "random_state",
                    "n_jobs",
                    "criterion",
                    "min_samples_split",
                    "min_samples_leaf",
                    "max_features",
                    "bootstrap",
                    "class_weight",
                    "ccp_alpha",
                )
            }

        # -- persist everything outside the temporary cwd ------------------
        assignments.to_csv(context.stage1_dir / "area_assignments.csv", index=False)
        donors.to_csv(context.stage1_dir / "eligible_donors.csv", index=False)
        pd.DataFrame(
            sorted(learned.branch_by_area.items()), columns=["admin_code", "branch_id"]
        ).to_csv(context.stage1_dir / "learned_map.csv", index=False)
        _write_json(
            context.stage1_dir / "partition_codes.json",
            {
                "branch_to_code": {
                    branch: index for index, branch in enumerate(learned.terminal_branches)
                },
                "unresolved_code": -1,
                "note": "stable serialization of the branch string, not a reclustering",
            },
        )
        _write_json(
            context.stage1_dir / "map_reconciliation.json",
            {"learned": learned.audit, "reconciliation": reconciliation},
        )
        _write_json(context.stage1_dir / "assignment_summary.json", summary)
        _write_json(
            context.stage1_dir / "fit_configuration.json",
            {
                "effective_config": effective_config,
                "georf_instance": {
                    "min_model_depth": model.min_model_depth,
                    "max_model_depth": model.max_model_depth,
                    "n_trees_unit": model.n_trees_unit,
                    "num_class": model.num_class,
                    "max_depth": model.max_depth,
                    "random_state": model.random_state,
                    "n_jobs": model.n_jobs,
                    "mode": model.mode,
                    "name": model.name,
                    "drop_list_": list(model.drop_list_),
                    "model_dir": model.model_dir,
                },
                "root_estimator_params": estimator_params,
                "module_locations": module_locations,
                "call": {
                    "contiguity_type": "polygon",
                    "print_to_file": False,
                    "track_partition_metrics": False,
                    "VIS_DEBUG_MODE": False,
                    "polygon_contiguity_info_keys": sorted(polygon_contiguity_info),
                },
                "branch_table_shape": list(branch_table.shape),
                "branch_table_accepted_nodes": int(branch_table.sum()),
                "fit_seconds": round(fit_seconds, 1),
            },
        )

    result["stage"] = {
        "model_dir": str(model_dir),
        "fit_seconds": round(fit_seconds, 1),
        "rows": int(len(design.y)),
        "learned_map": learned.audit,
        "reconciliation": {
            key: value
            for key, value in reconciliation.items()
            if key != "placeholder_row_branches"
        },
        "root_estimator_params": estimator_params,
        "effective_config": effective_config,
    }
    result["assignment_stage"] = {"summary": summary, "eligible_donors": int(len(donors))}
    result["singleton_stage"] = singleton_stage
    result["assignments"] = assignments
    return result


def _score_singletons(context: RunContext, model, design: Stage1Design,
                      singleton_X: np.ndarray, assignments: pd.DataFrame) -> dict:
    """Score the held-out singleton views through their frozen assignment.

    Q5v: these are supplementary post-map diagnostics. They never touched the
    fit, the q statistics or the parent/child F1 gate, and they do not feed back
    into the map.
    """
    frame = design.singleton_frame
    if len(frame) == 0:
        return {"rows": 0, "note": "no singleton outcomes"}

    branch_ids = singleton_branch_ids(frame, assignments)
    probabilities = model.model.predict_proba_georf(
        singleton_X,
        frame["admin_code"].to_numpy(dtype=np.int64),
        model.s_branch,
        X_branch_id=np.asarray(branch_ids, dtype=object),
    )
    predicted = (probabilities > DECISION_THRESHOLD).astype(np.int64)
    truth = frame["ipcch_food_crisis"].to_numpy(dtype=np.int64)

    route = assignments.set_index("admin_code")
    out = frame[
        ["admin_code", "country_en", "target_month", "origin_month", "horizon_months",
         "ipcch_food_crisis"]
    ].copy()
    out["branch_id"] = branch_ids
    out["prob_partitioned_rf"] = probabilities
    out["pred_partitioned_rf"] = predicted
    for column in ("partition_code", "assignment_source", "donor_admin_code",
                   "donor_distance_km", "model_route"):
        out[column] = route.loc[out["admin_code"], column].to_numpy()
    out.to_csv(context.stage1_dir / "singleton_scores.csv.gz", index=False)

    overall = confusion_counts(truth, predicted)
    by_horizon = {
        str(int(h)): confusion_counts(
            truth[out["horizon_months"].to_numpy() == h],
            predicted[out["horizon_months"].to_numpy() == h],
        )
        for h in sorted(out["horizon_months"].unique())
    }
    by_source = {
        str(source): confusion_counts(
            truth[out["assignment_source"].to_numpy() == source],
            predicted[out["assignment_source"].to_numpy() == source],
        )
        for source in sorted(out["assignment_source"].unique())
    }
    payload = {
        "rows": int(len(out)),
        "areas": int(out["admin_code"].nunique()),
        "threshold": DECISION_THRESHOLD,
        "overall": overall,
        "by_horizon": by_horizon,
        "by_assignment_source": by_source,
        "status": "supplementary development diagnostic (Q5v); not Stage3 evidence",
    }
    _write_json(context.stage1_dir / "singleton_diagnostics.json", payload)
    context.log(
        f"Q5v: scored {payload['rows']} singleton views over {payload['areas']} areas; "
        f"class1 F1 = {overall['class1_f1']}"
    )
    return payload


def _load_stage3_helpers(baseline) -> tuple:
    """Load the baseline's Stage 3 fit/probability helpers from the pinned copy.

    design.md prescribes reuse of these functions; importing the released module
    also lets the run ASSERT that its ``RF_PARAMS``/``MIN_PARTITION_SAMPLES``
    equal the values R5 approves, instead of quietly re-declaring them here.
    """
    import importlib.util  # noqa: PLC0415

    path = Path(baseline.root) / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py"
    if not path.is_file():
        raise PipelineError(f"the pinned baseline has no Stage 3 helper at {path}")
    spec = importlib.util.spec_from_file_location("ipcch_baseline_stage3", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001
        sys.modules.pop(spec.name, None)
        raise PipelineError(f"cannot import the baseline Stage 3 helpers: {error}") from error

    if dict(module.RF_PARAMS) != STAGE3_RF_PARAMS:
        raise PipelineError(
            f"the baseline's RF_PARAMS {module.RF_PARAMS} do not equal the approved "
            f"Stage 3 configuration {STAGE3_RF_PARAMS}"
        )
    if int(module.MIN_PARTITION_SAMPLES) != MIN_PARTITION_TRAIN_ROWS:
        raise PipelineError(
            "the baseline's MIN_PARTITION_SAMPLES does not equal the approved 50"
        )
    # The helper module pulls in more `src.*` modules than the Stage 1 entry
    # did; every one of them must still resolve inside the pinned copy.
    root = Path(baseline.root).resolve()
    escaped = {}
    for name, loaded in list(sys.modules.items()):
        if name != "config" and name.split(".")[0] != "src":
            continue
        location = getattr(loaded, "__file__", None)
        if location and not Path(location).resolve().is_relative_to(root):
            escaped[name] = location
    if escaped:
        raise PipelineError(f"modules escaped the pinned baseline copy: {escaped}")

    helpers = Stage3Helpers(
        train_pooled=module.train_pooled_model,
        train_partitioned=module.train_partitioned_model,
        predict_partitioned_probability=module.predict_partitioned_probability,
        predict_class1_probability=module.predict_class1_probability,
        min_partition_rows=int(module.MIN_PARTITION_SAMPLES),
        source=str(path),
    )
    return helpers, module


def _run_stage3(context: RunContext, baseline, matrix, ledger,
                assignments: pd.DataFrame, learned_audit: dict) -> dict:
    """122 scheduled main folds plus partial-2026, four arms, one frozen map."""
    schema = verify_prediction_schema()
    _write_json(context.stage3_dir / "prediction_schema.json", schema)

    import xgboost  # noqa: PLC0415

    if str(xgboost.__version__) != XGBOOST_PINNED_VERSION:
        raise PipelineError(
            f"Q7b pins XGBoost {XGBOOST_PINNED_VERSION}; the runtime has "
            f"{xgboost.__version__}"
        )

    valid_labels = ledger.valid()
    panel = build_stage3_panel(matrix, assignments, valid_labels)
    _write_json(context.stage3_dir / "panel_audit.json", panel.audit)
    context.log(
        f"Stage3 panel: {panel.audit['rows']} rows over {panel.audit['areas']} areas; "
        f"persistence history on {panel.audit['persistence_available_rows']} rows"
    )

    available = {
        int(h): set(int(t) for t in panel.target_ord[panel.horizon == h])
        for h in pdata.ACTIVE_HORIZONS
    }
    folds = build_fold_schedule(available_target_ords=available)
    main_folds = [f for f in folds if f.period == PERIOD_MAIN]
    context.log(
        f"R4 schedule: {len(main_folds)} main folds + "
        f"{len(folds) - len(main_folds)} partial-{PARTIAL_PERIOD_YEARS[0]} folds"
    )

    degenerate = int(learned_audit.get("terminal_branch_count", 0)) <= 1
    if degenerate:
        assigned = int((assignments["partition_code"] >= 0).sum())
        context.log(
            "DEGENERATE CONTRAST: Stage 1 accepted no split, so the map holds ONE "
            "terminal partition. The partitioned-minus-pooled delta therefore does "
            "not isolate spatial heterogeneity: it reduces to one local model "
            f"trained on {assigned} assigned areas versus one pooled model trained "
            f"on all {len(assignments)}. A nonzero delta is a training-set-size "
            "effect, not evidence of spatial structure."
        )

    def imputer_factory():
        from src.customize.customize import OutOfRangeImputer  # noqa: PLC0415

        return OutOfRangeImputer(strategy="max_plus", multiplier=100.0)

    def xgb_factory():
        return xgboost.XGBClassifier(missing=np.nan, **STAGE3_XGB_PARAMS)

    records: list[dict] = []
    frames: list[pd.DataFrame] = []
    key_frames: list[pd.DataFrame] = []
    fill_frames: list[pd.DataFrame] = []
    xgb_report = None
    started = time.time()

    stdout_path = context.stage3_dir / "fold_stdout.txt"
    with brt.baseline_imports(baseline) as (config, _georf):
        helpers, helper_module = _load_stage3_helpers(baseline)
        context.log(f"Stage3 helpers: {helpers.source}")
        try:
            with open(stdout_path, "w", encoding="utf-8", errors="backslashreplace") as sink:
                with contextlib.redirect_stdout(sink):
                    for index, fold in enumerate(folds, start=1):
                        outcome = fit_fold(
                            fold, panel, helpers, imputer_factory, xgb_factory
                        )
                        records.append(outcome["record"])
                        if outcome["rows"] is not None:
                            frames.append(outcome["rows"])
                            key_frames.append(outcome["training_keys"])
                            fill_frames.append(outcome["imputer_fills"])
                            if xgb_report is None:
                                xgb_report = verify_xgb_configuration(
                                    outcome["xgb_model"]
                                )
                        if index % 10 == 0 or index == len(folds):
                            sink.flush()
                            context.log(
                                f"  fold {index}/{len(folds)} {fold.fold_id} "
                                f"({round(time.time() - started)}s elapsed)"
                            )
        finally:
            sys.modules.pop("ipcch_baseline_stage3", None)
            del helper_module

    fit_seconds = time.time() - started
    if xgb_report is None:
        raise PipelineError("no fold produced a fitted XGB model to verify")
    _write_json(context.stage3_dir / "xgb_effective_configuration.json", xgb_report)

    predictions = pd.concat(frames, ignore_index=True)[list(PREDICTION_COLUMNS)]
    predictions.to_csv(
        context.stage3_dir / "predictions.csv.gz", index=False
    )
    pd.concat(key_frames, ignore_index=True).to_csv(
        context.stage3_dir / "fold_training_keys.csv.gz", index=False
    )
    pd.concat(fill_frames, ignore_index=True).to_csv(
        context.stage3_dir / "fold_imputer_fill_values.csv.gz", index=False
    )
    fold_frame = pd.DataFrame(
        [{k: (json.dumps(v, default=_json_default) if isinstance(v, (dict, list)) else v)
          for k, v in record.items()} for record in records]
    )
    fold_frame.to_csv(context.stage3_dir / "folds.csv", index=False)

    coverage = stage3_coverage(predictions, folds, records)
    _write_json(context.stage3_dir / "coverage.json", coverage)

    fitted = [r for r in records if r["fitted"]]
    probability_gap = (
        predictions["prob_partitioned_rf"] - predictions["prob_pooled_rf"]
    ).abs()
    assignment_counts = (
        assignments["assignment_source"].value_counts().to_dict()
    )
    areas_assigned = int((assignments["partition_code"] >= 0).sum())
    areas_universe = int(len(assignments))
    contrast = {
        "stage1_terminal_branches": int(learned_audit.get("terminal_branch_count", 0)),
        "single_partition_map": degenerate,
        "what_this_contrast_measures": (
            "With a SINGLE learned partition the partitioned-minus-pooled delta no "
            f"longer isolates spatial heterogeneity. It reduces to: one local model "
            f"trained on the {areas_assigned} areas the frozen map assigns to a "
            f"partition, versus one pooled model trained on all {areas_universe} "
            "areas in the universe. A nonzero delta here is a training-set-size "
            "effect and MUST NOT be read as evidence of spatial structure."
        )
        if degenerate
        else (
            "the map holds more than one terminal partition, so the contrast is a "
            "genuine partitioned-versus-pooled comparison"
        ),
        "areas_assigned_to_a_partition": areas_assigned,
        "areas_in_universe": areas_universe,
        "areas_by_assignment_source": {
            str(k): int(v) for k, v in assignment_counts.items()
        },
        "rows_total": int(len(predictions)),
        "test_rows_served_by_pooled_in_both_arms": int(
            (predictions["model_route"] == ROUTE_POOLED).sum()
        ),
        "test_rows_served_by_a_local_model": int(
            (predictions["model_route"] != ROUTE_POOLED).sum()
        ),
        "rows_with_equal_probability": int(
            sum(r["partitioned_equals_pooled_rows"] for r in fitted)
        ),
        "folds_with_all_probabilities_equal": int(
            sum(1 for r in fitted if r["partitioned_identical_to_pooled"])
        ),
        "hard_label_disagreements": int(
            (predictions["pred_partitioned_rf"] != predictions["pred_pooled_rf"]).sum()
        ),
        "unassigned_train_rows_excluded_from_local": int(
            sum(r["unassigned_train_rows_excluded_from_local"] for r in fitted)
        ),
        "probability_gap": {
            "max": float(probability_gap.max()),
            "mean": float(probability_gap.mean()),
            "rows_above_0_05": int((probability_gap > 0.05).sum()),
        },
        "why_the_arms_differ": (
            "Rows in unassigned areas belong to no partition, so the local model "
            "does not train on them while the pooled model does (R5: unassigned "
            "partitions fall back to the pooled RF; the released "
            "train_partitioned_model excludes partition -1 from training). "
            "Different training sets and the same seed give different forests, so "
            "the two arms legitimately disagree. The run asserts only what routing "
            "forces: every unassigned TEST row is served by the pooled model in "
            "both arms, and a local model that trained on the entire pool is the "
            "pooled forest. Nothing forces a locally-served row to agree."
        ),
        "stage1_gate": (
            "no split passed the inherited strict >.01 class-1 F1 gate, so the "
            "learned map is the root branch alone"
        )
        if degenerate
        else "",
    }
    _write_json(context.stage3_dir / "partitioned_vs_pooled_contrast.json", contrast)
    if degenerate:
        context.manifest["limitations"] = list(context.manifest["limitations"]) + [
            "Stage 3 ran the partitioned arm against a SINGLE-partition map. The "
            f"partitioned-minus-pooled delta reduces to a local model trained on "
            f"{areas_assigned} areas versus a pooled model trained on all "
            f"{areas_universe}; a nonzero delta is a training-set-size effect and "
            "is not evidence of spatial structure (see "
            "stage3/partitioned_vs_pooled_contrast.json)"
        ]

    settings = {
        "train_window_months": TRAIN_WINDOW_MONTHS,
        "train_window_rule": "exactly the calendar months [O-35, O], inclusive",
        "main_target_schedule": MAIN_TARGET_SCHEDULE,
        "partial_period_years": list(PARTIAL_PERIOD_YEARS),
        "decision_threshold": DECISION_THRESHOLD,
        "decision_rule": "p1 > threshold; an exact threshold is class 0 (Q7a)",
        "rf_params": STAGE3_RF_PARAMS,
        "xgb_params": STAGE3_XGB_PARAMS,
        "xgb_version": str(xgboost.__version__),
        "min_partition_train_rows": MIN_PARTITION_TRAIN_ROWS,
        "baseline_helpers": helpers.source,
        "map_source": "stage1/area_assignments.csv (frozen before any Stage 3 fit)",
        "pseudo_rows": "none in Stage 3; Stage 1's per-class zero rows are not reused",
        "sample_weights": "unit weights for every arm; no SMOTE, no class rebalancing",
        "imputation": (
            "one OutOfRangeImputer(max_plus, x100) per (origin, horizon), fitted on "
            "the common training pool only; pooled and every local RF reuse it; XGB "
            "receives the pre-imputation matrix with native NaN"
        ),
        "fit_seconds": round(fit_seconds, 1),
    }
    _write_json(context.stage3_dir / "stage3_settings.json", settings)

    context.log(
        f"Stage3: {coverage['folds_with_rows']} fitted folds / "
        f"{coverage['folds_empty']} empty of {len(folds)} scheduled; "
        f"{len(predictions)} prediction rows in {round(fit_seconds)}s"
    )
    return {
        "folds_scheduled": len(folds),
        "folds_main": len(main_folds),
        "folds_partial": len(folds) - len(main_folds),
        "folds_fitted": coverage["folds_with_rows"],
        "folds_empty": coverage["folds_empty"],
        "prediction_rows": int(len(predictions)),
        "coverage": coverage,
        "contrast": contrast,
        "settings": settings,
        "xgb_booster_effective": xgb_report["booster_effective"],
    }


def _write_json(path: Path, payload) -> None:
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    os.replace(temporary, path)


# ==========================================================================
# CLI
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IPCCH GeoRF Stage 1 runner (no Stage 3, no reporting)"
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="pinned assembled_IPCCH folder (raw/, spatial/, country lookup)",
    )
    parser.add_argument("--run-id", required=True, help="fresh run identity")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--release-zip", default=str(DEFAULT_RELEASE_ZIP))
    parser.add_argument(
        "--cohort", choices=("all", "non_ch", "ch_only"), default="all",
        help="restrict the cohort by the Cadre Harmonise rule "
             f"(admin_code >= {CH_ADMIN_CODE_FLOOR}); applied after the source gate",
    )
    parser.add_argument(
        "--split-gate", type=float, default=None,
        help="override MIN_CLASS_1_IMPROVEMENT_THRESHOLD in every consuming "
             "namespace; omit to use the pinned config value",
    )
    return parser


def main(argv=None) -> int:
    global COHORT_FILTER, SPLIT_GATE_OVERRIDE

    args = build_parser().parse_args(argv)
    COHORT_FILTER = None if args.cohort == "all" else args.cohort
    SPLIT_GATE_OVERRIDE = args.split_gate
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
