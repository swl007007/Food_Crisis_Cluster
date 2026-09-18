"""Per-partition expert-error learners for the Step 3 correction experiment.

Each existing Stage 3 partition fits one Random Forest on
``w = 1[y != e]`` - "was the expert wrong?" - using the original main feature
matrix ``X`` in its documented order **plus** the binary expert judgement ``e``
as the final column.  The produced score ``q = P(expert_wrong = 1)`` is *not* a
crisis probability and is never calibrated as one.

Deliberate differences from ``scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py``:

* No SMOTE, no class weighting, no correction-specific hyperparameter grid.
* No pooled fallback.  A partition that is absent, unmapped, too small or
  single-class **abstains**; its rows keep the expert prediction.  Using the
  crisis-predicting pooled model as a wrong-score source is forbidden.

Runtime exceptions and invalid inputs are errors, not ordinary abstention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Matches the current main RF exactly (see the reference Stage 3 script).
RF_PARAMS: Dict[str, object] = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 5,
    "n_jobs": 1,
}
MIN_PARTITION_SAMPLES = 50
UNMAPPED_PARTITION_ID = -1

ABSTAIN_UNMAPPED = "unmapped_partition"
ABSTAIN_NO_MODEL = "no_partition_model"
ABSTAIN_TOO_FEW_SAMPLES = "fewer_than_%d_usable_rows" % MIN_PARTITION_SAMPLES
ABSTAIN_SINGLE_CLASS = "single_wrong_label_class"
ELIGIBLE = "eligible"


class CorrectionInputError(RuntimeError):
    """Raised for invalid correction inputs; never downgraded to abstention."""


def wrong_label(truth: np.ndarray, expert: np.ndarray) -> np.ndarray:
    """Return the correction target ``w = 1[y != e]``."""
    y = np.asarray(truth)
    e = np.asarray(expert)
    if y.shape != e.shape:
        raise CorrectionInputError(f"truth/expert shape mismatch: {y.shape} vs {e.shape}")
    if not np.isin(y, [0, 1]).all():
        raise CorrectionInputError("truth must be binary 0/1")
    if not np.isin(e, [0, 1]).all():
        raise CorrectionInputError("expert must be binary 0/1")
    return (y.astype(int) != e.astype(int)).astype(int)


def build_design_matrix(X: np.ndarray, expert: np.ndarray) -> np.ndarray:
    """Append the binary expert judgement as the final model input column."""
    features = np.asarray(X, dtype=float)
    e = np.asarray(expert, dtype=float).reshape(-1, 1)
    if features.shape[0] != e.shape[0]:
        raise CorrectionInputError(
            f"X/expert row mismatch: {features.shape[0]} vs {e.shape[0]}"
        )
    if not np.isfinite(features).all():
        raise CorrectionInputError("Correction features contain non-finite values")
    return np.hstack([features, e])


@dataclass
class PartitionModelReport:
    """Trainability outcome for one partition at one fitting stage."""

    partition_id: int
    n_rows: int
    n_wrong: int
    trained: bool
    reason: str

    def as_record(self, stage: str) -> Dict[str, object]:
        """Flatten for the fold tuning artifact."""
        return {
            "stage": stage,
            "partition_id": int(self.partition_id),
            "n_rows": int(self.n_rows),
            "n_wrong": int(self.n_wrong),
            "trained": bool(self.trained),
            "reason": self.reason,
        }


@dataclass
class CorrectionEnsemble:
    """Fitted per-partition wrong-label models plus their abstention reports."""

    models: Dict[int, RandomForestClassifier]
    reports: List[PartitionModelReport]
    stage: str

    @property
    def n_trained(self) -> int:
        """Number of partitions with a usable wrong-label model."""
        return len(self.models)

    def report_records(self) -> List[Dict[str, object]]:
        """Return flattened per-partition trainability records."""
        return [report.as_record(self.stage) for report in self.reports]


def fit_correction_ensemble(
    *,
    X: np.ndarray,
    expert: np.ndarray,
    truth: np.ndarray,
    partitions: np.ndarray,
    stage: str,
    min_samples: int = MIN_PARTITION_SAMPLES,
    rf_params: Optional[Dict[str, object]] = None,
) -> CorrectionEnsemble:
    """Fit one wrong-label RF per partition, abstaining instead of falling back."""
    design = build_design_matrix(X, expert)
    target = wrong_label(truth, expert)
    groups = np.asarray(partitions)
    if groups.shape[0] != design.shape[0]:
        raise CorrectionInputError(
            f"partition/row mismatch: {groups.shape[0]} vs {design.shape[0]}"
        )
    params = dict(RF_PARAMS if rf_params is None else rf_params)

    models: Dict[int, RandomForestClassifier] = {}
    reports: List[PartitionModelReport] = []
    for partition_id in np.unique(groups):
        partition_id = int(partition_id)
        if partition_id == UNMAPPED_PARTITION_ID:
            mask = groups == partition_id
            reports.append(
                PartitionModelReport(
                    partition_id, int(mask.sum()), int(target[mask].sum()), False,
                    ABSTAIN_UNMAPPED,
                )
            )
            continue
        mask = groups == partition_id
        n_rows = int(mask.sum())
        n_wrong = int(target[mask].sum())
        if n_rows < min_samples:
            reports.append(
                PartitionModelReport(
                    partition_id, n_rows, n_wrong, False, ABSTAIN_TOO_FEW_SAMPLES
                )
            )
            continue
        if np.unique(target[mask]).size < 2:
            reports.append(
                PartitionModelReport(
                    partition_id, n_rows, n_wrong, False, ABSTAIN_SINGLE_CLASS
                )
            )
            continue
        model = RandomForestClassifier(**params)
        model.fit(design[mask], target[mask])
        models[partition_id] = model
        reports.append(PartitionModelReport(partition_id, n_rows, n_wrong, True, ELIGIBLE))
    return CorrectionEnsemble(models=models, reports=reports, stage=stage)


def wrong_scores(
    ensemble: CorrectionEnsemble,
    *,
    X: np.ndarray,
    expert: np.ndarray,
    partitions: np.ndarray,
):
    """Score rows with their own partition model; abstain elsewhere.

    Returns ``(scores, eligible, reasons)``.  Abstaining rows carry ``NaN`` -
    explicitly "unavailable", never a pooled crisis probability.
    """
    design = build_design_matrix(X, expert)
    groups = np.asarray(partitions)
    n_rows = design.shape[0]
    if groups.shape[0] != n_rows:
        raise CorrectionInputError(
            f"partition/row mismatch: {groups.shape[0]} vs {n_rows}"
        )

    scores = np.full(n_rows, np.nan, dtype=float)
    eligible = np.zeros(n_rows, dtype=bool)
    reasons = np.full(n_rows, ABSTAIN_NO_MODEL, dtype=object)
    reasons[groups == UNMAPPED_PARTITION_ID] = ABSTAIN_UNMAPPED

    for partition_id, model in ensemble.models.items():
        mask = groups == partition_id
        if not mask.any():
            continue
        proba = model.predict_proba(design[mask])
        classes = np.asarray(model.classes_)
        wrong_index = np.where(classes == 1)[0]
        if wrong_index.size == 0:
            raise CorrectionInputError(
                f"Partition {partition_id} model lacks a wrong-label class"
            )
        scores[mask] = proba[:, wrong_index[0]].astype(float)
        eligible[mask] = True
        reasons[mask] = ELIGIBLE
    return scores, eligible, reasons
