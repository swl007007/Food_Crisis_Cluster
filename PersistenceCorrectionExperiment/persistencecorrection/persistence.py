"""Layer-1 persistence series for the persistence-correction experiment (PRD R8, R6).

Definition
----------
For target month ``T`` and horizon ``H`` months, persistence is the **observed**
IPC phase binarised to crisis (``fews_ipc >= 3``) taken from the source row dated
exactly ``T - H``, joined on ``(admin_code, T - H)``::

    y_base(T) = 1[ fews_ipc(T - H) >= 3 ]

``H`` is 4 months for fs1 and 8 months for fs2 (PRD R7); no other scope is defined
here.  The series has no free parameters.

Why a calendar join and never a record shift
--------------------------------------------
``1.Source Data/Outcome/FEWSNET_IPC/FEWSNET.csv`` is **not** a monthly panel.
Every admin unit carries exactly 53 records on a shared release grid: quarterly
(Jan/Apr/Jul/Oct) 2009-07..2015-10, then tri-annual (February/June/October)
2016-02..2024-10.  A per-admin ``shift(4)``/``shift(8)`` therefore resolves to
12-16 / 24-32 *calendar* months on the tri-annual era - the defect documented in
``Step3ExpertCorrectionExperiment/README.md`` and
``docs/notes/2026-09-18_benchmark_and_direction_review.md``.  This module joins by
calendar month only; there is no shift-based code path.

Conventions, unchanged from the historical evaluator
----------------------------------------------------
* Binarise ``fews_ipc >= 3`` **before** the join, with a raw-missing phase mapped
  to ``0``.  A fractional or null phase therefore cannot survive the join as
  anything other than ``0``/``1``.
* An origin month absent from the release grid stays **missing**.  Nothing is ever
  imputed, forward-filled or back-filled - there is no imputation code path in this
  module, and :func:`attach_persistence` halts instead of tolerating a gap.
* Join coverage on the evaluation support must be exactly ``1.0`` (PRD R8).

Reuse
-----
The source loader, admin-month key normalisation, the legacy-column firewall and
the positive-class metric helper are imported from
``Step3ExpertCorrectionExperiment/step3correction/expert.py`` rather than copied,
so persistence and the expert series can never diverge on key semantics.  The
calendar join deliberately does **not** route through ``ExpertTable.for_scope``
(``expert.py:93``), whose column firewall structurally rejects any non-expert
series; persistence gets its own path, as required by ``design.md``.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .protected import FEWSNET_SOURCE, package_stage3_path, sha256

from step3correction.expert import (  # noqa: E402
    ExpertContractError,
    assert_no_legacy_expert_columns,
    counts_and_scores,
    load_expert_history,
)

KEYS = ["admin_code", "month_start"]

#: Forecast horizon in calendar months per forecasting scope (PRD R7).
PERSISTENCE_HORIZON_MONTHS: Dict[int, int] = {1: 4, 2: 8}

#: Column names emitted by :func:`attach_persistence`.
PERSISTENCE_COLUMNS = [
    "persistence",
    "persistence_source_month",
    "persistence_phase_raw",
    "persistence_phase_missing",
    "persistence_horizon_months",
]

PERSISTENCE_ALIGNMENT = "calendar"
PERSISTENCE_CONVENTION = (
    "binarise the observed phase fews_ipc>=3 with raw-missing->0, then "
    "calendar-join the observation dated exactly O = T-H (fs1 H=4, fs2 H=8) on "
    "(admin_code, T-H); an absent origin month stays missing and is never "
    "imputed, and coverage on the evaluation support must be exactly 1.0"
)

#: PRD R6, reproduced verbatim so the assumption travels with every run manifest.
R6_AVAILABILITY_ASSUMPTION = (
    "**R6 - Availability convention.** `fews_ipc(D)` is treated as available to a "
    "forecaster at month D. This is an explicit assumption, justified by CS/ML1/ML2 "
    "arriving in one publication (`1.Source Data/Outcome/FEWSNET_IPC/scrape_fewsnet.py:4,22`): "
    "accepting the expert baseline as a forecast made at D requires accepting the same "
    "row's CS at D. The repo contradicts itself here (`src/preprocess/preprocess.py:267-273` "
    "assumes available; `EthiopiaForecastingExperiment/aligned_refit.py:122-127` and "
    "`.trellis/spec/backend/local-forecasting-experiments.md:73` assume not). Record the "
    "contradiction; do **not** modify the Ethiopia spec or its tests."
)

#: The recorded contradiction (PRD R6). Neither side is edited by this experiment.
R6_RECORDED_CONTRADICTION = {
    "assumes_available_at_origin": "src/preprocess/preprocess.py:267-273",
    "assumes_not_available_at_origin": [
        "EthiopiaForecastingExperiment/aligned_refit.py:122-127",
        ".trellis/spec/backend/local-forecasting-experiments.md:73",
    ],
    "resolution_for_this_experiment": (
        "This experiment adopts the available-at-D convention. The departure from the "
        "strictly-before-origin rule in the local-forecasting-experiments spec is "
        "deliberate and scoped to this package; neither the spec nor the Ethiopia "
        "experiment is modified."
    ),
}

#: Default evaluation support: the frozen Stage 3 2021-2024 predictions (PRD R24).
SUPPORT_ADMIN_COLUMN = "FEWSNET_admin_code"
SUPPORT_MONTH_COLUMN = "month_start"

#: Reference crisis-class F1 of persistence on that support
#: (docs/notes/2026-09-18_benchmark_and_direction_review.md, section 3).
REFERENCE_PERSISTENCE_F1: Dict[int, float] = {1: 0.7761, 2: 0.7085}
REFERENCE_F1_TOLERANCE = 1e-4


class PersistenceContractError(RuntimeError):
    """Raised when the persistence data contract is violated; the run must halt."""


@dataclass(frozen=True)
class ObservedPhaseTable:
    """Binarised observed-phase history, keyed by ``(admin_code, month_start)``.

    ``crisis`` is ``fews_ipc >= 3`` with a raw-missing phase mapped to ``0``, and it
    is computed by the shared loader *before* any join takes place.
    """

    frame: pd.DataFrame
    source_path: Path

    def observations(self) -> pd.DataFrame:
        """Return the observed-phase columns only."""
        return self.frame[KEYS + ["crisis", "phase_raw", "phase_missing"]].copy()

    def targets_for_scope(self, scope: int) -> pd.DataFrame:
        """Re-key the observations by the target month they serve: ``T = O + H``."""
        horizon = require_horizon(scope)
        observed = self.frame
        targets = pd.DataFrame(
            {
                "admin_code": observed["admin_code"].to_numpy(),
                "_target_month": (
                    observed["month_start"] + pd.DateOffset(months=horizon)
                ).to_numpy(),
                "persistence": observed["crisis"].to_numpy(),
                "persistence_source_month": observed["month_start"].to_numpy(),
                "persistence_phase_raw": observed["phase_raw"].to_numpy(),
                "persistence_phase_missing": observed["phase_missing"].to_numpy(),
            }
        )
        if targets.duplicated(["admin_code", "_target_month"]).any():
            raise PersistenceContractError(
                f"fs{scope} persistence alignment produced duplicate "
                "(admin_code, target month) keys"
            )
        return targets


def require_horizon(scope: int) -> int:
    """Return the calendar horizon for ``scope``, refusing undefined scopes."""
    if scope not in PERSISTENCE_HORIZON_MONTHS:
        raise PersistenceContractError(
            f"Persistence is only defined for fs1/fs2 in this experiment, got fs{scope}"
        )
    return PERSISTENCE_HORIZON_MONTHS[scope]


def load_observed_phase_history(
    source_path: Path | str = FEWSNET_SOURCE,
) -> ObservedPhaseTable:
    """Load the observed IPC-phase history and binarise it before any join.

    Key normalisation, the all-null key-row drop and the ``fews_ipc >= 3``
    binarisation are performed by the shared Step 3 loader
    (``step3correction.expert.load_expert_history``), which exposes the binarised
    observed phase as ``source_truth``.  Only the observed-phase columns are kept;
    the expert projection columns and the firewalled legacy record-shift columns
    are dropped here and their absence is re-asserted.
    """
    source_path = Path(source_path)
    try:
        loaded = load_expert_history(source_path)
    except ExpertContractError as error:
        raise PersistenceContractError(
            f"Persistence source contract violated in {source_path}: {error}"
        ) from error

    raw = loaded.frame
    frame = pd.DataFrame(
        {
            "admin_code": raw["admin_code"].to_numpy(),
            "month_start": pd.to_datetime(raw["month_start"]).to_numpy(),
            # source_truth == (fews_ipc >= 3) with raw-missing -> 0, pre-join.
            "crisis": raw["source_truth"].astype(int).to_numpy(),
            "phase_raw": raw["fews_ipc"].to_numpy(),
            "phase_missing": raw["fews_ipc"].isna().astype(int).to_numpy(),
        }
    )
    assert_no_legacy_expert_columns(frame, "persistence observed-phase table")
    if frame[KEYS].isna().any().any():
        raise PersistenceContractError("Null admin-month key in persistence source")
    if frame.duplicated(KEYS).any():
        raise PersistenceContractError("Duplicate admin-month key in persistence source")
    if not frame["crisis"].isin([0, 1]).all():
        raise PersistenceContractError("Binarised observed phase is not 0/1")
    return ObservedPhaseTable(
        frame=frame.sort_values(KEYS).reset_index(drop=True), source_path=source_path
    )


def attach_persistence(
    support: pd.DataFrame,
    table: ObservedPhaseTable,
    scope: int,
    *,
    admin_column: str = SUPPORT_ADMIN_COLUMN,
    month_column: str = SUPPORT_MONTH_COLUMN,
) -> pd.DataFrame:
    """Attach the layer-1 persistence series to ``support``.

    Halts unless the join preserves the row count, every matched row is dated
    exactly ``T - H``, and coverage is exactly ``1.0``.  There is no fallback and
    no imputation: a single unmatched target row aborts the run (PRD R8).
    """
    horizon = require_horizon(scope)
    if admin_column not in support.columns or month_column not in support.columns:
        raise PersistenceContractError(
            f"Support is missing key columns {admin_column!r}/{month_column!r}"
        )

    frame = support.copy()
    frame[month_column] = pd.to_datetime(frame[month_column], errors="raise")
    codes = pd.to_numeric(frame[admin_column], errors="raise")
    if codes.isna().any() or (codes % 1 != 0).any():
        raise PersistenceContractError("Invalid admin code in persistence support")
    frame[admin_column] = codes.astype("int64")
    if frame.duplicated([admin_column, month_column]).any():
        raise PersistenceContractError(
            "Duplicate (admin_code, target month) keys in persistence support"
        )

    targets = table.targets_for_scope(scope)
    merged = frame.merge(
        targets,
        left_on=[admin_column, month_column],
        right_on=["admin_code", "_target_month"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["_target_month", "admin_code"])
    if len(merged) != len(frame):
        raise PersistenceContractError(
            f"fs{scope} persistence join changed the row count "
            f"({len(merged)} vs {len(frame)})"
        )

    missing = merged["persistence_source_month"].isna()
    if bool(missing.any()):
        sample = merged.loc[missing, [admin_column, month_column]].head(20)
        raise PersistenceContractError(
            f"fs{scope} persistence coverage is "
            f"{1.0 - float(missing.mean()):.6f}, not 1.0: "
            f"{int(missing.sum())} of {len(merged)} target rows have no observation at "
            f"T-{horizon} months. Persistence is never imputed; halting.\n{sample}"
        )

    source_month = pd.to_datetime(merged["persistence_source_month"])
    target_month = pd.to_datetime(merged[month_column])
    lag = (target_month.dt.year - source_month.dt.year) * 12 + (
        target_month.dt.month - source_month.dt.month
    )
    off_origin = lag.ne(horizon)
    if bool(off_origin.any()):
        raise PersistenceContractError(
            f"fs{scope} persistence source months are not all at T-{horizon}: "
            f"observed lags {sorted(set(lag[off_origin].tolist()))}"
        )

    merged["persistence"] = merged["persistence"].astype(int)
    merged["persistence_phase_missing"] = merged["persistence_phase_missing"].astype(int)
    merged["persistence_horizon_months"] = horizon
    if not merged["persistence"].isin([0, 1]).all():
        raise PersistenceContractError("Persistence values are not 0/1 after the join")
    assert_no_legacy_expert_columns(merged, f"fs{scope} persistence series")
    return merged


def persistence_coverage(
    frame: pd.DataFrame, scope: int, *, label: str = "all rows"
) -> Dict[str, object]:
    """Report persistence coverage and raw-missing-phase counts over ``frame``."""
    available = pd.to_datetime(frame["persistence_source_month"]).notna()
    rows = int(len(frame))
    return {
        "scope": scope,
        "support": label,
        "horizon_months": require_horizon(scope),
        "rows": rows,
        "rows_with_persistence": int(available.sum()),
        "coverage": (float(available.sum()) / rows) if rows else 0.0,
        "rows_with_raw_missing_phase": int(
            pd.to_numeric(frame["persistence_phase_missing"]).sum()
        ),
        "persistence_positive_rate": float(frame["persistence"].mean()) if rows else 0.0,
        "distinct_target_months": int(pd.to_datetime(frame[SUPPORT_MONTH_COLUMN]).nunique())
        if SUPPORT_MONTH_COLUMN in frame.columns
        else None,
    }


def persistence_metrics(
    frame: pd.DataFrame, *, truth_column: str = "y_true"
) -> Dict[str, object]:
    """Crisis-class counts and scores for the persistence series (PRD R17)."""
    try:
        counts, scores = counts_and_scores(frame[truth_column], frame["persistence"])
    except ExpertContractError as error:
        raise PersistenceContractError(f"Persistence metric contract: {error}") from error
    tp, fp, fn, tn = counts
    precision, recall, f1 = scores
    return {
        "rows": int(len(frame)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_class1": precision,
        "recall_class1": recall,
        "f1_class1": f1,
    }


def check_reference_f1(scope: int, f1: float, *, tolerance: float = REFERENCE_F1_TOLERANCE):
    """Compare a computed F1 against the Phase 1 stop/go reference value."""
    reference = REFERENCE_PERSISTENCE_F1[scope]
    delta = float(f1) - reference
    return {
        "scope": scope,
        "reference_f1_class1": reference,
        "computed_f1_class1": float(f1),
        "abs_delta": abs(delta),
        "tolerance": tolerance,
        "reproduced": abs(delta) <= tolerance,
        "reference_source": (
            "docs/notes/2026-09-18_benchmark_and_direction_review.md section 3"
        ),
    }


def load_stage3_support(scope: int) -> pd.DataFrame:
    """Load the frozen Stage 3 2021-2024 evaluation support for ``scope`` (read-only)."""
    path = package_stage3_path(scope, "predictions_monthly.csv")
    frame = pd.read_csv(path)
    frame[SUPPORT_MONTH_COLUMN] = pd.to_datetime(frame[SUPPORT_MONTH_COLUMN])
    return frame


# ---------------------------------------------------------------------------
# Arbitrary support (Phase 3 onwards)
# ---------------------------------------------------------------------------
#
# ``attach_persistence`` was already support-agnostic: it takes any frame with an
# admin-code column and a target-month column.  What follows only adds *loaders*
# so the same contract can be applied to a Phase 2 probability artifact (the
# 2018-2020 and 2021-2024 ``predictions_monthly.csv`` files) instead of only to
# the frozen Stage 3 2021-2024 support.  No contract is relaxed: the calendar
# join at ``T-H``, the coverage-1.0 halt, the no-imputation rule and the per-row
# provenance are the same code path in every case.

#: Columns a Phase 2 probability artifact must expose to serve as a support.
PROBABILITY_SUPPORT_COLUMNS = [
    SUPPORT_ADMIN_COLUMN,
    SUPPORT_MONTH_COLUMN,
    "partition_id",
    "y_true",
    "y_prob_partitioned",
]


def load_probability_support(path: Path | str) -> pd.DataFrame:
    """Load a Phase 2 ``predictions_monthly.csv`` as a persistence support.

    Read-only.  Validates the required columns up front so a schema drift fails
    here rather than silently downstream.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    missing = [
        column for column in PROBABILITY_SUPPORT_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise PersistenceContractError(
            f"Probability support {path} is missing column(s) {missing}"
        )
    frame[SUPPORT_MONTH_COLUMN] = pd.to_datetime(frame[SUPPORT_MONTH_COLUMN])
    return frame


def persistence_for_probability_file(
    path: Path | str,
    table: ObservedPhaseTable,
    scope: int,
) -> pd.DataFrame:
    """Attach persistence to a Phase 2 probability artifact (arbitrary support).

    Identical contract to :func:`attach_persistence`; this is only the loader
    plus the join, so a support outside the frozen 2021-2024 window (for example
    the 2018-2020 probabilities) cannot take a weaker path.
    """
    return attach_persistence(load_probability_support(path), table, scope)


def environment_report() -> Dict[str, str]:
    """Record the interpreter and library versions used for a run."""
    return {
        "python": platform.python_version(),
        "executable": __import__("sys").executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def build_persistence_manifest(
    table: ObservedPhaseTable,
    per_scope: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    """Assemble the Phase 1 run manifest, carrying PRD R6 verbatim (AC1)."""
    return {
        "experiment": "PersistenceCorrectionExperiment",
        "phase": "1 - persistence series",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_report(),
        "source": {
            "path": str(table.source_path),
            "sha256": sha256(table.source_path),
            "rows": int(len(table.frame)),
            "admin_units": int(table.frame["admin_code"].nunique()),
            "distinct_months": int(table.frame["month_start"].nunique()),
            "cadence": (
                "not monthly: quarterly 2009-07..2015-10, then tri-annual "
                "(Feb/Jun/Oct) 2016-02..2024-10"
            ),
        },
        "persistence": {
            "definition": "y_base(T) = 1[fews_ipc(T-H) >= 3]",
            "alignment": PERSISTENCE_ALIGNMENT,
            "horizon_months": dict(PERSISTENCE_HORIZON_MONTHS),
            "convention": PERSISTENCE_CONVENTION,
            "imputation": "none; a missing origin observation halts the run",
            "record_shift_used": False,
        },
        "availability_assumption_r6_verbatim": R6_AVAILABILITY_ASSUMPTION,
        "availability_assumption_recorded_contradiction": R6_RECORDED_CONTRADICTION,
        "scopes": {str(scope): dict(payload) for scope, payload in sorted(per_scope.items())},
    }


def _write_json(destination: Path, payload: object) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return destination


def _default_support_provider(scope: int) -> Tuple[pd.DataFrame, str]:
    """Phase 1 default: the frozen Stage 3 2021-2024 support and its path."""
    return (
        load_stage3_support(scope),
        str(package_stage3_path(scope, "predictions_monthly.csv")),
    )


def build_persistence_series(
    output_dir: Path,
    scopes: Iterable[int] = (1, 2),
    *,
    source_path: Path | str = FEWSNET_SOURCE,
    support_provider: Optional[Callable[[int], Tuple[pd.DataFrame, str]]] = None,
    support_label: str = "stage3 2021-2024 support",
    check_reference: bool = True,
) -> Dict[str, object]:
    """Phase 1 entrypoint: build, validate and persist the persistence series.

    Writes one ``persistence_series_fs{scope}.csv`` per scope plus
    ``persistence_run_manifest.json`` into ``output_dir`` (which the caller must
    have resolved inside this experiment's output tree).

    ``support_provider`` lets a later phase run the *same* contract over a
    different support (for example the Phase 2 2018-2020 probabilities); it must
    return ``(frame, path_str)``.  The default reproduces Phase 1 exactly.
    ``check_reference`` compares the resulting F1 against the 2021-2024 reference
    and is meaningless on any other support, so callers passing their own support
    must set it to ``False``.
    """
    output_dir = Path(output_dir)
    table = load_observed_phase_history(source_path)
    provider = _default_support_provider if support_provider is None else support_provider
    per_scope: Dict[int, Dict[str, object]] = {}
    frames: Dict[int, pd.DataFrame] = {}

    for scope in scopes:
        support, support_path = provider(scope)
        series = attach_persistence(support, table, scope)
        frames[scope] = series
        coverage = persistence_coverage(series, scope, label=support_label)
        metrics = persistence_metrics(series)
        reference = (
            check_reference_f1(scope, metrics["f1_class1"]) if check_reference else None
        )
        per_scope[scope] = {
            "coverage": coverage,
            "metrics": metrics,
            "reference_check": reference,
            "support_path": support_path,
            "output_csv": str(output_dir / f"persistence_series_fs{scope}.csv"),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        columns = [SUPPORT_ADMIN_COLUMN, SUPPORT_MONTH_COLUMN, "y_true"] + PERSISTENCE_COLUMNS
        series[columns].to_csv(output_dir / f"persistence_series_fs{scope}.csv", index=False)

    manifest = build_persistence_manifest(table, per_scope)
    _write_json(output_dir / "persistence_run_manifest.json", manifest)
    return {"manifest": manifest, "frames": frames}


__all__: List[str] = [
    "ObservedPhaseTable",
    "PERSISTENCE_COLUMNS",
    "PROBABILITY_SUPPORT_COLUMNS",
    "SUPPORT_ADMIN_COLUMN",
    "SUPPORT_MONTH_COLUMN",
    "load_probability_support",
    "persistence_for_probability_file",
    "PERSISTENCE_CONVENTION",
    "PERSISTENCE_HORIZON_MONTHS",
    "PersistenceContractError",
    "R6_AVAILABILITY_ASSUMPTION",
    "R6_RECORDED_CONTRADICTION",
    "REFERENCE_PERSISTENCE_F1",
    "attach_persistence",
    "build_persistence_manifest",
    "build_persistence_series",
    "check_reference_f1",
    "load_observed_phase_history",
    "load_stage3_support",
    "persistence_coverage",
    "persistence_metrics",
    "require_horizon",
]
