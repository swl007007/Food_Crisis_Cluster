"""Validation-gated selective-correction rule selection.

One shared wrong-score threshold per outer fold, with two independently enabled
flip directions (``0->1`` and ``1->0``).  A direction is enabled only when, on the
fold's validation rows, it proposes at least :data:`MIN_PROPOSED_FLIPS` flips
spanning at least :data:`MIN_DISTINCT_MONTHS` distinct target months with
correction precision at least :data:`MIN_CORRECTION_PRECISION`.  After direction
filtering, the candidate must yield *strictly* greater crisis-class F1 than the
expert-only reference; ties keep expert-only.

These gates are fixed by the approved design and must never be relaxed to
manufacture a visible improvement.  Test labels never enter this module.

Two direction modes are supported, selected per run and never mixed:

``both`` (Variant A, the default)
    Both flip directions may be enabled, each on its own gates.  This is the
    original approved mechanism and its committed run is the reference result.

``up-only`` (Variant B)
    ``enable_1_to_0`` is forced ``False`` *before* candidate scoring, so only
    ``0->1`` flips can ever be proposed, scored or applied.

    Justification - asymmetric cost.  A ``1->0`` flip switches an
    already-issued crisis warning off.  In food-security early warning a missed
    crisis carries materially higher cost than a false alarm, so a mechanism
    that can silence the expert's own crisis calls is the more dangerous half of
    the rule.  This argument is independent of any test result.

    Epistemic status.  Variant B was specified after Variant A's test results
    were known, so its test metric is a post-hoc, test-informed figure and is
    NOT an out-of-sample estimate.  Variant A remains the only genuinely
    out-of-sample result for this mechanism.  The oracle decomposition that
    corroborates the restriction (fs1 ``0->1`` +0.0078 vs ``1->0`` +0.0034; fs2
    +0.0171 vs +0.0020) was computed with test labels and is recorded only as
    post-hoc corroboration - never as the reason for the restriction and never
    as evidence that Variant B generalises.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MIN_PROPOSED_FLIPS = 20
MIN_DISTINCT_MONTHS = 2
MIN_CORRECTION_PRECISION = 0.75
THRESHOLD_ROUNDING_DECIMALS = 2

# Direction modes.  ``both`` is Variant A (unchanged default); ``up-only`` is
# Variant B, which forces ``enable_1_to_0 = False`` before candidate scoring on
# asymmetric-cost grounds (see the module docstring).  No other mode exists and
# no mode relaxes any gate above.
DIRECTION_MODE_BOTH = "both"
DIRECTION_MODE_UP_ONLY = "up-only"
DIRECTION_MODES = (DIRECTION_MODE_BOTH, DIRECTION_MODE_UP_ONLY)

NO_CORRECTION_REASONS = {
    "no_validation_rows": "validation support is empty",
    "no_eligible_scores": "no finite wrong scores on eligible validation rows",
    "no_candidates": "no candidate thresholds after rounding",
    "no_strict_improvement": "no candidate strictly exceeded expert-only validation F1",
}


class DirectionContractError(RuntimeError):
    """Raised when a forbidden flip direction is proposed or applied."""


def resolve_direction_mode(direction_mode: str) -> str:
    """Validate and return a direction mode, rejecting anything unknown."""
    mode = str(direction_mode)
    if mode not in DIRECTION_MODES:
        raise ValueError(f"direction_mode must be one of {DIRECTION_MODES}, got {mode!r}")
    return mode


def allows_1_to_0(direction_mode: str) -> bool:
    """Return whether the mode permits the ``1->0`` direction at all."""
    return resolve_direction_mode(direction_mode) == DIRECTION_MODE_BOTH


def assert_direction_contract(
    expert: Sequence[int],
    predictions: Sequence[int],
    direction_mode: str,
    *,
    context: str,
) -> None:
    """Halt loudly if a flip violates the run's direction mode.

    Under Variant B a ``1->0`` flip reaching any output is a contract error, not
    a tolerable anomaly, so this is an unconditional hard failure.
    """
    if allows_1_to_0(direction_mode):
        return
    expert_int = np.asarray(expert, dtype=int)
    pred_int = np.asarray(predictions, dtype=int)
    if expert_int.shape != pred_int.shape:
        raise DirectionContractError(
            f"{context}: expert/prediction shape mismatch "
            f"{expert_int.shape} vs {pred_int.shape}"
        )
    violations = int(((expert_int == 1) & (pred_int == 0)).sum())
    if violations:
        raise DirectionContractError(
            f"{context}: {violations} rows flipped 1->0 under direction mode "
            f"{DIRECTION_MODE_UP_ONLY!r}, which forbids that direction"
        )


def crisis_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Positive-class F1 with the archived zero-denominator convention (0.0)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def candidate_thresholds(wrong_scores: Sequence[float]) -> np.ndarray:
    """Sorted unique two-decimal candidates, ascending, from eligible scores.

    Candidates are rounded (following the Ethiopia selective mechanism) but are
    always applied to the original unrounded scores with strict ``q > threshold``.
    """
    scores = np.asarray(wrong_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([], dtype=float)
    return np.sort(np.unique(np.round(scores, THRESHOLD_ROUNDING_DECIMALS)))


def proposed_flip_mask(
    wrong_scores: np.ndarray, eligible: np.ndarray, threshold: float
) -> np.ndarray:
    """Strict ``q > threshold`` on eligible rows only."""
    scores = np.asarray(wrong_scores, dtype=float)
    mask = np.asarray(eligible, dtype=bool) & np.isfinite(scores)
    return mask & (scores > float(threshold))


@dataclass
class DirectionGate:
    """Directional gate evidence for one candidate threshold."""

    direction: str
    proposed: int
    distinct_months: int
    fixes: int
    precision: float
    enabled: bool
    # True when the run's direction mode forbids this direction outright, so the
    # approved gates were never allowed to enable it.  Recorded on the gate
    # rather than in ``as_record`` so Variant A's artifact schema is unchanged;
    # the run-level ``direction_mode`` column carries the same information.
    forced_disabled: bool = False

    def forced_off(self) -> "DirectionGate":
        """Return this gate with the direction forcibly disabled by the mode."""
        return replace(self, enabled=False, forced_disabled=True)

    def as_record(self, prefix: str) -> Dict[str, object]:
        """Flatten for CSV export with a per-direction column prefix."""
        return {
            f"{prefix}_proposed": self.proposed,
            f"{prefix}_distinct_months": self.distinct_months,
            f"{prefix}_fixes": self.fixes,
            f"{prefix}_precision": self.precision,
            f"{prefix}_enabled": bool(self.enabled),
        }


@dataclass
class SelectedRule:
    """The frozen rule applied to the fold's target month."""

    corrected: bool
    threshold: Optional[float]
    enable_0_to_1: bool
    enable_1_to_0: bool
    expert_only_validation_f1: float
    selected_validation_f1: float
    status: str
    reason: str = ""
    # Kept off ``as_record`` deliberately: Variant A's committed ``fold_tuning``
    # schema stays byte-identical, and the runner adds an explicit
    # ``direction_mode`` provenance column for any non-default mode.
    direction_mode: str = DIRECTION_MODE_BOTH
    candidates: List[Dict[str, object]] = field(default_factory=list)

    def as_record(self) -> Dict[str, object]:
        """Flatten the selection outcome for the fold tuning artifact."""
        return {
            "correction_selected": bool(self.corrected),
            "selected_threshold": self.threshold,
            "enable_0_to_1": bool(self.enable_0_to_1),
            "enable_1_to_0": bool(self.enable_1_to_0),
            "expert_only_validation_f1": self.expert_only_validation_f1,
            "selected_validation_f1": self.selected_validation_f1,
            "selection_status": self.status,
            "selection_reason": self.reason,
        }


def evaluate_direction(
    *,
    direction: str,
    expert_value: int,
    expert: np.ndarray,
    truth: np.ndarray,
    months: np.ndarray,
    proposed: np.ndarray,
) -> DirectionGate:
    """Score one flip direction's gates on original validation rows."""
    selected = proposed & (np.asarray(expert, dtype=int) == expert_value)
    count = int(selected.sum())
    if count == 0:
        return DirectionGate(direction, 0, 0, 0, 0.0, False)
    distinct_months = int(pd.Series(np.asarray(months)[selected]).nunique())
    fixes = int((np.asarray(truth, dtype=int)[selected] != expert_value).sum())
    precision = fixes / count
    enabled = (
        count >= MIN_PROPOSED_FLIPS
        and distinct_months >= MIN_DISTINCT_MONTHS
        and precision >= MIN_CORRECTION_PRECISION
    )
    return DirectionGate(direction, count, distinct_months, fixes, precision, enabled)


def apply_rule(
    expert: np.ndarray,
    wrong_scores: np.ndarray,
    eligible: np.ndarray,
    threshold: Optional[float],
    enable_0_to_1: bool,
    enable_1_to_0: bool,
) -> np.ndarray:
    """Return final predictions: flip only where an enabled direction fires."""
    expert_int = np.asarray(expert, dtype=int)
    predictions = expert_int.copy()
    if threshold is None or not (enable_0_to_1 or enable_1_to_0):
        return predictions
    proposed = proposed_flip_mask(wrong_scores, eligible, threshold)
    if enable_0_to_1:
        flip = proposed & (expert_int == 0)
        predictions[flip] = 1
    if enable_1_to_0:
        flip = proposed & (expert_int == 1)
        predictions[flip] = 0
    return predictions


def select_rule(
    *,
    truth: Sequence[int],
    expert: Sequence[int],
    wrong_scores: Sequence[float],
    eligible: Sequence[bool],
    months: Sequence,
    direction_mode: str = DIRECTION_MODE_BOTH,
) -> SelectedRule:
    """Select the fold's shared threshold and direction flags from validation rows.

    Abstaining rows stay in the F1 support (they simply retain the expert label)
    but can never be proposed flips.

    ``direction_mode`` is ``both`` (Variant A) or ``up-only`` (Variant B).  Under
    ``up-only`` the ``1->0`` gate is forced off *before* candidate scoring, so
    every candidate's validation F1 already reflects the restriction.  Nothing
    else changes: the 20-flip / 2-month / 0.75-precision gates, the strict
    ``q > threshold`` comparison, the single shared threshold, the strict-F1
    objective and the expert-only tie-break are all untouched.
    """
    mode = resolve_direction_mode(direction_mode)
    down_allowed = allows_1_to_0(mode)
    truth_arr = np.asarray(truth, dtype=int)
    expert_arr = np.asarray(expert, dtype=int)
    score_arr = np.asarray(wrong_scores, dtype=float)
    eligible_arr = np.asarray(eligible, dtype=bool) & np.isfinite(score_arr)
    month_arr = np.asarray(months)

    lengths = {len(truth_arr), len(expert_arr), len(score_arr), len(eligible_arr), len(month_arr)}
    if len(lengths) != 1:
        raise ValueError(f"Validation arrays have inconsistent lengths: {lengths}")

    expert_f1 = crisis_f1(truth_arr, expert_arr)
    base = SelectedRule(
        corrected=False,
        threshold=None,
        enable_0_to_1=False,
        enable_1_to_0=False,
        expert_only_validation_f1=expert_f1,
        selected_validation_f1=expert_f1,
        status="no_correction",
        direction_mode=mode,
    )
    if len(truth_arr) == 0:
        base.reason = NO_CORRECTION_REASONS["no_validation_rows"]
        return base
    if not eligible_arr.any():
        base.reason = NO_CORRECTION_REASONS["no_eligible_scores"]
        return base

    candidates = candidate_thresholds(score_arr[eligible_arr])
    if candidates.size == 0:
        base.reason = NO_CORRECTION_REASONS["no_candidates"]
        return base

    best = base
    best_f1 = expert_f1
    records: List[Dict[str, object]] = []
    for threshold in candidates:
        proposed = proposed_flip_mask(score_arr, eligible_arr, threshold)
        gate_up = evaluate_direction(
            direction="0_to_1",
            expert_value=0,
            expert=expert_arr,
            truth=truth_arr,
            months=month_arr,
            proposed=proposed,
        )
        gate_down = evaluate_direction(
            direction="1_to_0",
            expert_value=1,
            expert=expert_arr,
            truth=truth_arr,
            months=month_arr,
            proposed=proposed,
        )
        if not down_allowed:
            # Variant B: the direction is disabled a priori on asymmetric-cost
            # grounds, before this candidate's validation F1 is computed, so the
            # gates can never turn it back on.
            gate_down = gate_down.forced_off()
        predictions = apply_rule(
            expert_arr, score_arr, eligible_arr, threshold, gate_up.enabled, gate_down.enabled
        )
        candidate_f1 = crisis_f1(truth_arr, predictions)
        record: Dict[str, object] = {
            "threshold": float(threshold),
            "validation_f1": candidate_f1,
            "expert_only_validation_f1": expert_f1,
            "any_direction_enabled": bool(gate_up.enabled or gate_down.enabled),
        }
        record.update(gate_up.as_record("dir_0_to_1"))
        record.update(gate_down.as_record("dir_1_to_0"))
        records.append(record)

        # Ascending iteration plus strict improvement keeps ties with the
        # earliest candidate, and keeps expert-only on an exact tie.
        if (gate_up.enabled or gate_down.enabled) and candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best = SelectedRule(
                corrected=True,
                threshold=float(threshold),
                enable_0_to_1=gate_up.enabled,
                enable_1_to_0=gate_down.enabled,
                expert_only_validation_f1=expert_f1,
                selected_validation_f1=candidate_f1,
                status="corrected",
                reason="strict validation crisis-F1 improvement over expert-only",
                direction_mode=mode,
            )

    if not best.corrected:
        best = base
        best.reason = NO_CORRECTION_REASONS["no_strict_improvement"]
    if not down_allowed and best.enable_1_to_0:
        raise DirectionContractError(
            f"direction mode {mode!r} forbids 1->0 but the selected rule enabled it"
        )
    best.candidates = records
    return best
