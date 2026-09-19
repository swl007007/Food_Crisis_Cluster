"""Phase 3: probability calibration for the persistence-correction experiment.

What this module does
---------------------
It maps the partitioned GeoRF crisis score ``y_prob_partitioned`` onto a
calibrated probability ``p_cal`` using a **frozen, static** set of calibrators
fitted once on 2018-2019 and applied unchanged everywhere else (DECISIONS_LOG
C6).  Nothing here selects a threshold, applies an override or computes a test
metric; Phase 3 ends at a reported gate.

Windows (DECISIONS_LOG C5, which supersedes PRD R12)
----------------------------------------------------
R12 asked for calibrators cross-fitted on out-of-fold predictions inside the
36-month training window.  ``scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py``
emits test-month predictions only, so that data does not exist and producing it
would mean editing or duplicating a script R27 freezes.  The replacement split,
fixed before any test data was touched:

===============  ==============  ===============================================
window           labeled months  role
===============  ==============  ===============================================
2018-2019        6               fit the calibrators (in-sample for them)
2020             3               select ``tau`` in Phase 4; out-of-sample here
2021-2024        12              apply once in Phase 5; **never read for a metric**
===============  ==============  ===============================================

The two pre-test windows are disjoint, which preserves R12's actual purpose:
calibration fitting never touches the threshold-selection data.
:func:`fit_calibrators` enforces the fit window structurally - a row outside
``FIT_YEARS`` raises :class:`CalibrationContractError` before anything is fitted.

Grouping (PRD R11)
------------------
Stage 3 ran with ``--month-ind``, so ``partition_id`` is not a stable identifier:
it indexes 13 (Feb) / 11 (Jun) / 16 (Oct) different partitionings selected by
calendar month, plus a ``-1`` sentinel for admin units the month map does not
cover (``compare_partitioned_vs_pooled_rf_k40_nc4.py:270-271, 391-392``), whose
rows are predicted by the pooled model.  Calibrating on ``partition_id`` alone
would silently merge three different partition definitions under one integer, so
every calibrator is keyed by ``(calendar_month, partition_id)``.

Fallback chain, in order, with every step counted and reported
--------------------------------------------------------------
1. fewer than :data:`MIN_GROUP_ROWS` fit rows -> that calendar month's pooled
   calibrator (``min_rows``);
2. a single observed class in the fit rows -> the month pool (``single_class``).
   Platt cannot serve this case: ``LogisticRegression`` raises on one class, and
   isotonic would emit a constant 0 or 1.  The attempt is made and its failure
   recorded rather than assumed;
3. fewer than :data:`MIN_DISTINCT_PROBS` distinct scores -> Platt
   (``platt_too_few_distinct_probabilities``), which degenerates gracefully to
   the group base rate;
4. otherwise isotonic regression.

At apply time a ``(month, partition_id)`` absent from the fit window - in this
data every ``partition_id == -1`` bucket - routes to the month pool
(``group_absent_from_fit_window``).  A month pool that is itself degenerate
falls back to the identity transform, which is recorded, never silent.

Purity
------
:meth:`CalibratorSet.transform` takes three arrays - score, calendar month,
partition id - and nothing else.  It cannot see a label, and
:func:`apply_calibrators` passes it nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import ROOT

#: Labeled years the calibrators may be fitted on (DECISIONS_LOG C5).
FIT_YEARS: Tuple[int, ...] = (2018, 2019)
#: Labeled year Phase 4 selects ``tau`` on. Out-of-sample for the calibrators.
SELECTION_YEARS: Tuple[int, ...] = (2020,)
#: Labeled years Phase 5 adjudicates on. Phase 3 emits them and measures nothing.
TEST_YEARS: Tuple[int, ...] = (2021, 2022, 2023, 2024)

#: Minimum fit rows for a group to get its own calibrator (PRD R11; the Step 3
#: abstention floor, reused rather than re-invented).
MIN_GROUP_ROWS = 50
#: Minimum distinct scores before isotonic regression is meaningful.
MIN_DISTINCT_PROBS = 3

PROB_COLUMN = "y_prob_partitioned"
TRUTH_COLUMN = "y_true"
PARTITION_COLUMN = "partition_id"
MONTH_COLUMN = "month_start"
CALIBRATED_COLUMN = "p_cal"
ROUTE_COLUMN = "calibration_route"
ROUTE_REASON_COLUMN = "calibration_route_reason"

#: The unmapped-admin sentinel emitted by the Stage 3 comparison script.
UNMAPPED_PARTITION_ID = -1

ISOTONIC = "isotonic"
PLATT = "platt"
IDENTITY = "identity"

ROUTE_GROUP = "group"
ROUTE_MONTH_POOLED = "month_pooled"

#: Every fallback reason this module can emit, so counts are exhaustive.
FALLBACK_REASONS = (
    "min_rows",
    "single_class",
    "platt_failed",
    "isotonic_failed",
    "group_absent_from_fit_window",
)


class CalibrationContractError(RuntimeError):
    """Raised when the calibration data contract is violated; the run must halt."""


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibrator:
    """A frozen, serialisable, label-free probability map.

    ``kind`` is ``isotonic`` (piecewise-linear interpolation over the fitted
    knots, exactly reproducing ``IsotonicRegression.predict`` with
    ``out_of_bounds='clip'``), ``platt`` (a sigmoid) or ``identity``.
    """

    kind: str
    params: Dict[str, object]
    n_fit_rows: int
    n_fit_positive: int

    def transform(self, prob: np.ndarray) -> np.ndarray:
        values = np.asarray(prob, dtype=float)
        if not np.isfinite(values).all():
            raise CalibrationContractError("Non-finite score passed to a calibrator")
        if self.kind == IDENTITY:
            out = values
        elif self.kind == ISOTONIC:
            knots_x = np.asarray(self.params["x_thresholds"], dtype=float)
            knots_y = np.asarray(self.params["y_thresholds"], dtype=float)
            out = np.interp(values, knots_x, knots_y)
        elif self.kind == PLATT:
            coef = float(self.params["coef"])
            intercept = float(self.params["intercept"])
            out = 1.0 / (1.0 + np.exp(-(coef * values + intercept)))
        else:
            raise CalibrationContractError(f"Unknown calibrator kind {self.kind!r}")
        return np.clip(out, 0.0, 1.0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "params": self.params,
            "n_fit_rows": int(self.n_fit_rows),
            "n_fit_positive": int(self.n_fit_positive),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Calibrator":
        return cls(
            kind=str(payload["kind"]),
            params=dict(payload["params"]),  # type: ignore[arg-type]
            n_fit_rows=int(payload["n_fit_rows"]),
            n_fit_positive=int(payload["n_fit_positive"]),
        )


def identity_calibrator(n_fit_rows: int = 0, n_fit_positive: int = 0) -> Calibrator:
    """The no-op transform, used only when even a month pool is degenerate."""
    return Calibrator(IDENTITY, {}, n_fit_rows, n_fit_positive)


def fit_isotonic(prob: np.ndarray, truth: np.ndarray) -> Calibrator:
    """Fit an isotonic calibrator and store its knots (not the estimator)."""
    model = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(np.asarray(prob, dtype=float), np.asarray(truth, dtype=float))
    return Calibrator(
        kind=ISOTONIC,
        params={
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        },
        n_fit_rows=int(len(prob)),
        n_fit_positive=int(np.sum(np.asarray(truth) == 1)),
    )


def fit_platt(prob: np.ndarray, truth: np.ndarray) -> Calibrator:
    """Fit a Platt/sigmoid calibrator; raises when the labels are single-class."""
    values = np.asarray(prob, dtype=float).reshape(-1, 1)
    labels = np.asarray(truth).astype(int)
    model = LogisticRegression(solver="lbfgs", max_iter=1000).fit(values, labels)
    return Calibrator(
        kind=PLATT,
        params={
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        },
        n_fit_rows=int(len(labels)),
        n_fit_positive=int(np.sum(labels == 1)),
    )


def fit_group_calibrator(
    prob: np.ndarray,
    truth: np.ndarray,
    *,
    min_group_rows: int = MIN_GROUP_ROWS,
    min_distinct_probs: int = MIN_DISTINCT_PROBS,
) -> Tuple[Optional[Calibrator], str, Dict[str, object]]:
    """Run the documented fallback chain for one group.

    Returns ``(calibrator_or_None, reason, diagnostics)``.  ``None`` means the
    caller must route the group to its calendar month's pooled calibrator, and
    ``reason`` says why.
    """
    prob = np.asarray(prob, dtype=float)
    truth = np.asarray(truth).astype(int)
    distinct_probs = int(np.unique(prob).size)
    diagnostics: Dict[str, object] = {
        "n_rows": int(len(prob)),
        "n_positive": int(np.sum(truth == 1)),
        "n_distinct_probs": distinct_probs,
        "platt_attempt_error": None,
    }

    if len(prob) < min_group_rows:
        return None, "min_rows", diagnostics

    if np.unique(truth).size < 2:
        # Isotonic would emit a constant 0/1 and Platt cannot fit one class.
        # Attempt it so the fallback is an observed consequence, not a claim.
        try:
            calibrator = fit_platt(prob, truth)
        except Exception as error:  # pragma: no cover - sklearn raises here
            diagnostics["platt_attempt_error"] = f"{type(error).__name__}: {error}"
            return None, "single_class", diagnostics
        return calibrator, "platt_single_class", diagnostics

    if distinct_probs < min_distinct_probs:
        try:
            calibrator = fit_platt(prob, truth)
        except Exception as error:
            diagnostics["platt_attempt_error"] = f"{type(error).__name__}: {error}"
            return None, "platt_failed", diagnostics
        return calibrator, "platt_too_few_distinct_probabilities", diagnostics

    try:
        calibrator = fit_isotonic(prob, truth)
    except Exception as error:  # pragma: no cover - defensive
        diagnostics["isotonic_attempt_error"] = f"{type(error).__name__}: {error}"
        return None, "isotonic_failed", diagnostics
    return calibrator, ISOTONIC, diagnostics


# ---------------------------------------------------------------------------
# The frozen calibrator set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibratorSet:
    """All calibrators for one scope, plus the routing table and its provenance."""

    scope: int
    fit_years: Tuple[int, ...]
    fit_months: Tuple[str, ...]
    min_group_rows: int
    min_distinct_probs: int
    month_pooled: Dict[int, Calibrator]
    groups: Dict[Tuple[int, int], Calibrator]
    group_reports: List[Dict[str, object]] = field(default_factory=list)
    month_pool_reports: List[Dict[str, object]] = field(default_factory=list)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        """Canonical, library-independent representation (knots and coefficients)."""
        return {
            "schema": "persistencecorrection.calibration/v1",
            "scope": int(self.scope),
            "fit_years": [int(year) for year in self.fit_years],
            "fit_months": list(self.fit_months),
            "min_group_rows": int(self.min_group_rows),
            "min_distinct_probs": int(self.min_distinct_probs),
            "prob_column": PROB_COLUMN,
            "group_key": ["calendar_month", PARTITION_COLUMN],
            "month_pooled": {
                str(month): calibrator.to_dict()
                for month, calibrator in sorted(self.month_pooled.items())
            },
            "groups": {
                f"{month}|{partition}": calibrator.to_dict()
                for (month, partition), calibrator in sorted(self.groups.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibratorSet":
        groups: Dict[Tuple[int, int], Calibrator] = {}
        for key, value in payload["groups"].items():  # type: ignore[union-attr]
            month_text, partition_text = str(key).split("|")
            groups[(int(month_text), int(partition_text))] = Calibrator.from_dict(value)
        return cls(
            scope=int(payload["scope"]),  # type: ignore[arg-type]
            fit_years=tuple(int(year) for year in payload["fit_years"]),  # type: ignore[union-attr]
            fit_months=tuple(str(month) for month in payload["fit_months"]),  # type: ignore[union-attr]
            min_group_rows=int(payload["min_group_rows"]),  # type: ignore[arg-type]
            min_distinct_probs=int(payload["min_distinct_probs"]),  # type: ignore[arg-type]
            month_pooled={
                int(month): Calibrator.from_dict(value)
                for month, value in payload["month_pooled"].items()  # type: ignore[union-attr]
            },
            groups=groups,
        )

    def canonical_json(self) -> str:
        """Stable text form; the hashed artifact (PRD/ C6 'frozen with a hash')."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    # -- application ------------------------------------------------------

    def transform(
        self,
        prob: np.ndarray,
        calendar_month: np.ndarray,
        partition_id: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrate ``prob``. Label-free by construction: no truth argument exists.

        Returns ``(calibrated, route, route_reason)`` as aligned arrays.
        """
        prob = np.asarray(prob, dtype=float)
        months = np.asarray(calendar_month).astype(int)
        partitions = np.asarray(partition_id).astype(int)
        if not (len(prob) == len(months) == len(partitions)):
            raise CalibrationContractError("transform() received misaligned arrays")
        if prob.size and (np.nanmin(prob) < 0.0 or np.nanmax(prob) > 1.0):
            raise CalibrationContractError("Score outside [0, 1] passed to transform()")

        calibrated = np.full(len(prob), np.nan, dtype=float)
        route = np.empty(len(prob), dtype=object)
        reason = np.empty(len(prob), dtype=object)

        for month in np.unique(months):
            month_mask = months == month
            pooled = self.month_pooled.get(int(month))
            if pooled is None:
                raise CalibrationContractError(
                    f"No pooled calibrator for calendar month {int(month)}; "
                    "the fit window does not cover this month"
                )
            for partition in np.unique(partitions[month_mask]):
                mask = month_mask & (partitions == partition)
                key = (int(month), int(partition))
                own = self.groups.get(key)
                if own is not None:
                    calibrated[mask] = own.transform(prob[mask])
                    route[mask] = ROUTE_GROUP
                    reason[mask] = own.kind
                else:
                    calibrated[mask] = pooled.transform(prob[mask])
                    route[mask] = ROUTE_MONTH_POOLED
                    reason[mask] = self._fallback_reason(key)
        if np.isnan(calibrated).any():
            raise CalibrationContractError("transform() left rows uncalibrated")
        return calibrated, route, reason

    def _fallback_reason(self, key: Tuple[int, int]) -> str:
        for report in self.group_reports:
            if (int(report["calendar_month"]), int(report["partition_id"])) == key:
                return str(report["reason"])
        return "group_absent_from_fit_window"


def assert_fit_window(frame: pd.DataFrame, fit_years: Sequence[int] = FIT_YEARS) -> None:
    """Halt unless every row is inside the declared calibration-fit window.

    This is the structural guarantee behind AC3: a 2020 row (Phase 4 selection)
    or a 2021-2024 row (Phase 5 test) cannot enter a calibrator fit, because the
    fit refuses to start.
    """
    if MONTH_COLUMN not in frame.columns:
        raise CalibrationContractError(f"Fit frame is missing {MONTH_COLUMN!r}")
    years = pd.to_datetime(frame[MONTH_COLUMN]).dt.year
    allowed = set(int(year) for year in fit_years)
    intruders = sorted(set(int(year) for year in years.unique()) - allowed)
    if intruders:
        raise CalibrationContractError(
            "Calibration fit window is "
            f"{sorted(allowed)}; refusing to fit on rows from {intruders}. "
            "No selection-window or test-window row may enter a calibrator fit."
        )


def fit_calibrators(
    frame: pd.DataFrame,
    scope: int,
    *,
    fit_years: Sequence[int] = FIT_YEARS,
    prob_column: str = PROB_COLUMN,
    truth_column: str = TRUTH_COLUMN,
    min_group_rows: int = MIN_GROUP_ROWS,
    min_distinct_probs: int = MIN_DISTINCT_PROBS,
) -> CalibratorSet:
    """Fit the frozen calibrator set for ``scope`` on the fit window only."""
    assert_fit_window(frame, fit_years)
    for column in (prob_column, truth_column, PARTITION_COLUMN, MONTH_COLUMN):
        if column not in frame.columns:
            raise CalibrationContractError(f"Fit frame is missing {column!r}")

    work = frame.copy()
    work[MONTH_COLUMN] = pd.to_datetime(work[MONTH_COLUMN])
    work["calendar_month"] = work[MONTH_COLUMN].dt.month
    prob = pd.to_numeric(work[prob_column], errors="raise").to_numpy(dtype=float)
    truth = pd.to_numeric(work[truth_column], errors="raise").to_numpy()
    if not np.isfinite(prob).all():
        raise CalibrationContractError("Non-finite score in the calibration fit window")
    if not np.isin(truth, (0, 1)).all():
        raise CalibrationContractError("Calibration fit labels are not 0/1")
    work["_prob"] = prob
    work["_truth"] = truth.astype(int)

    month_pooled: Dict[int, Calibrator] = {}
    month_pool_reports: List[Dict[str, object]] = []
    for month, block in work.groupby("calendar_month", sort=True):
        calibrator, reason, diagnostics = fit_group_calibrator(
            block["_prob"].to_numpy(),
            block["_truth"].to_numpy(),
            min_group_rows=1,  # a month pool is the last resort; never size-gated
            min_distinct_probs=min_distinct_probs,
        )
        if calibrator is None:
            calibrator = identity_calibrator(len(block), int(block["_truth"].sum()))
            reason = f"identity_after_{reason}"
        month_pooled[int(month)] = calibrator
        month_pool_reports.append(
            {
                "scope": int(scope),
                "calendar_month": int(month),
                "partition_id": "MONTH_POOL",
                "n_fit_rows": int(diagnostics["n_rows"]),
                "n_fit_positive": int(diagnostics["n_positive"]),
                "fit_positive_rate": float(diagnostics["n_positive"])
                / float(diagnostics["n_rows"]),
                "n_distinct_probs": int(diagnostics["n_distinct_probs"]),
                "route": "month_pool_definition",
                "calibrator_kind": calibrator.kind,
                "reason": reason,
                "is_fallback": False,
                "platt_attempt_error": diagnostics.get("platt_attempt_error"),
            }
        )

    groups: Dict[Tuple[int, int], Calibrator] = {}
    reports: List[Dict[str, object]] = []
    for (month, partition), block in work.groupby(
        ["calendar_month", PARTITION_COLUMN], sort=True
    ):
        calibrator, reason, diagnostics = fit_group_calibrator(
            block["_prob"].to_numpy(),
            block["_truth"].to_numpy(),
            min_group_rows=min_group_rows,
            min_distinct_probs=min_distinct_probs,
        )
        key = (int(month), int(partition))
        if calibrator is not None:
            groups[key] = calibrator
        reports.append(
            {
                "scope": int(scope),
                "calendar_month": int(month),
                "partition_id": int(partition),
                "n_fit_rows": int(diagnostics["n_rows"]),
                "n_fit_positive": int(diagnostics["n_positive"]),
                "fit_positive_rate": float(diagnostics["n_positive"])
                / float(diagnostics["n_rows"]),
                "n_distinct_probs": int(diagnostics["n_distinct_probs"]),
                "route": ROUTE_GROUP if calibrator is not None else ROUTE_MONTH_POOLED,
                "calibrator_kind": calibrator.kind if calibrator is not None else None,
                "reason": reason,
                "is_fallback": calibrator is None,
                "platt_attempt_error": diagnostics.get("platt_attempt_error"),
            }
        )

    fit_months = tuple(
        sorted(work[MONTH_COLUMN].dt.strftime("%Y-%m").unique().tolist())
    )
    return CalibratorSet(
        scope=int(scope),
        fit_years=tuple(int(year) for year in fit_years),
        fit_months=fit_months,
        min_group_rows=int(min_group_rows),
        min_distinct_probs=int(min_distinct_probs),
        month_pooled=month_pooled,
        groups=groups,
        group_reports=reports,
        month_pool_reports=month_pool_reports,
    )


def apply_calibrators(
    frame: pd.DataFrame,
    calibrators: CalibratorSet,
    *,
    prob_column: str = PROB_COLUMN,
    output_column: str = CALIBRATED_COLUMN,
) -> pd.DataFrame:
    """Apply a frozen calibrator set. A pure transform: no label is consulted.

    Only ``prob_column``, ``month_start`` and ``partition_id`` are read; the
    result is independent of whether the frame carries a truth column at all.
    """
    for column in (prob_column, PARTITION_COLUMN, MONTH_COLUMN):
        if column not in frame.columns:
            raise CalibrationContractError(f"Apply frame is missing {column!r}")
    out = frame.copy()
    months = pd.to_datetime(out[MONTH_COLUMN]).dt.month.to_numpy()
    partitions = pd.to_numeric(out[PARTITION_COLUMN], errors="raise").to_numpy()
    prob = pd.to_numeric(out[prob_column], errors="raise").to_numpy(dtype=float)
    calibrated, route, reason = calibrators.transform(prob, months, partitions)
    out[output_column] = calibrated
    out[ROUTE_COLUMN] = route
    out[ROUTE_REASON_COLUMN] = reason
    return out


def fallback_summary(
    calibrators: CalibratorSet, applied: Mapping[str, pd.DataFrame]
) -> Dict[str, object]:
    """Count every fallback, at fit time and at apply time (AC3)."""
    fit_fallbacks: Dict[str, int] = {}
    for report in calibrators.group_reports:
        if report["is_fallback"]:
            key = str(report["reason"])
            fit_fallbacks[key] = fit_fallbacks.get(key, 0) + 1
    group_sizes = [int(report["n_fit_rows"]) for report in calibrators.group_reports]
    apply_counts: Dict[str, Dict[str, object]] = {}
    for label, frame in applied.items():
        counts = (
            frame.groupby([ROUTE_COLUMN, ROUTE_REASON_COLUMN], sort=True)
            .size()
            .reset_index(name="rows")
        )
        apply_counts[label] = {
            "rows": int(len(frame)),
            "rows_by_route": {
                f"{row[ROUTE_COLUMN]}:{row[ROUTE_REASON_COLUMN]}": int(row["rows"])
                for _, row in counts.iterrows()
            },
            "rows_month_pooled": int(
                (frame[ROUTE_COLUMN] == ROUTE_MONTH_POOLED).sum()
            ),
        }
    return {
        "scope": calibrators.scope,
        "groups_total": len(calibrators.group_reports),
        "groups_with_own_calibrator": len(calibrators.groups),
        "groups_routed_to_month_pool": sum(
            1 for report in calibrators.group_reports if report["is_fallback"]
        ),
        "fit_fallback_reason_counts": fit_fallbacks,
        "min_group_rows_threshold": calibrators.min_group_rows,
        "observed_min_group_rows": min(group_sizes) if group_sizes else None,
        "observed_median_group_rows": (
            int(np.median(group_sizes)) if group_sizes else None
        ),
        "calibrator_kind_counts": {
            kind: sum(
                1
                for calibrator in calibrators.groups.values()
                if calibrator.kind == kind
            )
            for kind in sorted({c.kind for c in calibrators.groups.values()})
        },
        "month_pool_kinds": {
            str(month): calibrator.kind
            for month, calibrator in sorted(calibrators.month_pooled.items())
        },
        "apply_windows": apply_counts,
    }


# ---------------------------------------------------------------------------
# Reliability (PRD R5 / AC2)
# ---------------------------------------------------------------------------


def _import_reference_reliability_bins():
    """Reuse the repository's reliability binning by import where possible.

    ``scripts/paper_artifacts/analyze_georf_probability_uncertainty.py:115-148``
    is the existing implementation; it is read-only and never modified.  If it
    cannot be imported the caller gets ``None`` and must say so in the manifest
    rather than silently using a look-alike.
    """
    import sys

    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from scripts.paper_artifacts.analyze_georf_probability_uncertainty import (  # noqa: E402
            reliability_bins,
        )
    except Exception:  # pragma: no cover - environment dependent
        return None
    return reliability_bins


def reliability_bin_table(
    frame: pd.DataFrame, prob_column: str, *, n_bins: int = 10
) -> pd.DataFrame:
    """Ten equal-width reliability bins, via the repository's own implementation."""
    binner = _import_reference_reliability_bins()
    if binner is None:  # pragma: no cover - environment dependent
        raise CalibrationContractError(
            "Could not import reliability_bins from "
            "scripts/paper_artifacts/analyze_georf_probability_uncertainty.py"
        )
    return binner(frame, prob_column, n_bins=n_bins)


def persistence_group_reliability(
    frame: pd.DataFrame,
    prob_column: str,
    *,
    scope: int,
    window: str,
    stage: str,
    sample_status: str,
    truth_column: str = TRUTH_COLUMN,
    persistence_column: str = "persistence",
    tolerance: float = 0.05,
) -> List[Dict[str, object]]:
    """PRD R5: mean predicted probability against the actual crisis rate.

    One row per persistence group (``persist = 0`` and ``persist = 1``) plus an
    ``all`` row.  ``stage`` is ``pre`` or ``post``; ``sample_status`` records
    whether the window was in-sample for the calibrator.
    """
    rows: List[Dict[str, object]] = []
    groups: List[Tuple[str, pd.DataFrame]] = [
        (str(value), frame.loc[frame[persistence_column] == value])
        for value in sorted(frame[persistence_column].unique())
    ]
    groups.append(("all", frame))
    for label, block in groups:
        if block.empty:
            continue
        mean_prob = float(pd.to_numeric(block[prob_column]).mean())
        crisis_rate = float(pd.to_numeric(block[truth_column]).mean())
        gap = mean_prob - crisis_rate
        rows.append(
            {
                "scope": int(scope),
                "window": window,
                "sample_status": sample_status,
                "stage": stage,
                "probability_column": prob_column,
                "persistence_group": label,
                "n": int(len(block)),
                "mean_predicted_probability": mean_prob,
                "observed_crisis_rate": crisis_rate,
                "signed_gap_pred_minus_actual": gap,
                "abs_gap": abs(gap),
                "tolerance": tolerance,
                "within_tolerance": bool(abs(gap) <= tolerance),
            }
        )
    return rows


def brier_score(truth: pd.Series, prob: pd.Series) -> float:
    """Mean squared error of the probability forecast."""
    values = pd.to_numeric(prob, errors="coerce").to_numpy(dtype=float)
    labels = pd.to_numeric(truth, errors="coerce").to_numpy(dtype=float)
    return float(np.mean((values - labels) ** 2))


__all__ = [
    "CALIBRATED_COLUMN",
    "Calibrator",
    "CalibratorSet",
    "CalibrationContractError",
    "FALLBACK_REASONS",
    "FIT_YEARS",
    "MIN_DISTINCT_PROBS",
    "MIN_GROUP_ROWS",
    "PROB_COLUMN",
    "ROUTE_COLUMN",
    "ROUTE_REASON_COLUMN",
    "ROUTE_GROUP",
    "ROUTE_MONTH_POOLED",
    "SELECTION_YEARS",
    "TEST_YEARS",
    "UNMAPPED_PARTITION_ID",
    "apply_calibrators",
    "assert_fit_window",
    "brier_score",
    "fallback_summary",
    "fit_calibrators",
    "fit_group_calibrator",
    "fit_isotonic",
    "fit_platt",
    "identity_calibrator",
    "persistence_group_reliability",
    "reliability_bin_table",
]
