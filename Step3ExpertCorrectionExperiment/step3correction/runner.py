"""Isolated fs1/fs2 expert selective-correction runner (Step 3 experiment).

Method identifiers written by this runner:

* ``partitioned_selective_correction`` - **Variant A** (``--direction-mode both``,
  the default): the **calendar-aligned** FEWS NET expert estimate published at
  ``O = T - H``, plus one validation-gated per-partition correction layer
  selected on ``V = [O - 12 months, O)``, with both flip directions eligible.
* ``partitioned_selective_correction_up_only`` - **Variant B**
  (``--direction-mode up-only``): identical in every respect except that the
  ``1->0`` direction is forced off before candidate scoring, on **asymmetric-cost**
  grounds (a ``1->0`` flip silences an already-issued crisis warning, and a missed
  food-security crisis costs materially more than a false alarm).  Variant B was
  specified after Variant A's test results were known, so its test metric is a
  **post-hoc, test-informed** figure and **not an out-of-sample estimate**;
  Variant A remains the only genuinely out-of-sample result for this mechanism.
  The two variants are separate methods: Variant A is never re-run, modified or
  overwritten by a Variant B run.
* ``pooled`` - the unchanged, **reused** frozen Stage 3 pooled baseline.
* ``partitioned`` - retained only for fs3, where correction is disabled.

The runner never touches production entrypoints: it imports
``src.preprocess``/``src.feature``/``src.customize`` read-only and writes every
artifact inside ``Step3ExpertCorrectionExperiment/outputs/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import correction, selection, windows
from .baselines import KEYS, ReusedBaseline, load_reused_baseline
from .expert import (
    EXPERT_ALIGNMENT,
    EXPERT_CONVENTION,
    LEGACY_EXPERT_CONVENTION,
    SCOPE_HORIZON_MONTHS,
    ExpertContractError,
    assert_no_legacy_expert_columns,
    counts_and_scores,
    coverage_report,
    load_expert_history,
    require_source_alignment,
    verify_archived_expert_baselines,
)
from .features import FeaturePanel, build_feature_panel, fit_window_imputer, transform_with
from .protected import (
    EXPERIMENT_DIR,
    FEWSNET_SOURCE,
    PANEL_SOURCE,
    assert_unchanged,
    hash_protected,
    refined_map_path,
    sha256,
    stage2_map_path,
    write_hash_report,
)

CORRECTION_METHOD = "partitioned_selective_correction"
CORRECTION_METHOD_UP_ONLY = "partitioned_selective_correction_up_only"
DIRECTION_MODE_METHOD = {
    selection.DIRECTION_MODE_BOTH: CORRECTION_METHOD,
    selection.DIRECTION_MODE_UP_ONLY: CORRECTION_METHOD_UP_ONLY,
}
POOLED_METHOD = "pooled"
UNCORRECTED_METHOD = "partitioned"
CORRECTION_SCOPES = (1, 2)
UNCORRECTED_SCOPES = (3,)
APPROVED_OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"
MONTH_MAP_KEY = {2: "m2", 6: "m6", 10: "m10"}
DEFAULT_TARGET_MONTHS = tuple(
    f"{year}-{month:02d}" for year in range(2021, 2025) for month in (2, 6, 10)
)


class RunContractError(RuntimeError):
    """Raised when a run-level contract is violated and the run must halt."""


def correction_method_id(direction_mode: str) -> str:
    """Return the explicit method identifier for a direction mode."""
    return DIRECTION_MODE_METHOD[selection.resolve_direction_mode(direction_mode)]


@dataclass
class FoldResult:
    """Everything produced for one (scope, target month) fold."""

    audit_rows: pd.DataFrame
    validation_rows: pd.DataFrame
    candidates: pd.DataFrame
    partition_reports: pd.DataFrame
    fold_record: Dict[str, object]


def resolve_output_dir(out_dir: Path | str, *, allow_outside_repo: bool = False) -> Path:
    """Validate the output path *before* creating anything.

    Runs must be isolated under ``Step3ExpertCorrectionExperiment/outputs/`` and
    run directories are immutable: an existing directory is a hard error.
    """
    path = Path(out_dir).resolve()
    inside = path == APPROVED_OUTPUT_ROOT or APPROVED_OUTPUT_ROOT in path.parents
    if not inside and not allow_outside_repo:
        raise ValueError(
            f"Correction output must stay under {APPROVED_OUTPUT_ROOT}, got {path}"
        )
    if path.exists():
        raise FileExistsError(f"Correction run directory already exists: {path}")
    return path


def month_map_path(scope: int, target_month: pd.Period) -> Path:
    """Return the frozen contig3 month map used by the original main run."""
    key = MONTH_MAP_KEY.get(target_month.month, "general")
    return refined_map_path(scope, key)


def load_partition_assignment(map_path: Path) -> pd.DataFrame:
    """Load a frozen ``FEWSNET_admin_code -> cluster_id`` map."""
    frame = pd.read_csv(map_path)
    missing = [c for c in ("FEWSNET_admin_code", "cluster_id") if c not in frame.columns]
    if missing:
        raise RunContractError(f"Partition map {map_path} missing columns {missing}")
    frame = frame[["FEWSNET_admin_code", "cluster_id"]].copy()
    frame["FEWSNET_admin_code"] = pd.to_numeric(
        frame["FEWSNET_admin_code"], errors="raise"
    ).astype("int64")
    if frame.duplicated("FEWSNET_admin_code").any():
        raise RunContractError(f"Partition map {map_path} has duplicate admin codes")
    return frame


def assign_partitions(admin_codes: np.ndarray, assignment: pd.DataFrame) -> np.ndarray:
    """Map admin codes to partition ids, using ``-1`` for unmapped units."""
    lookup = assignment.set_index("FEWSNET_admin_code")["cluster_id"]
    mapped = pd.Series(admin_codes).map(lookup)
    return mapped.fillna(correction.UNMAPPED_PARTITION_ID).astype(int).to_numpy()


def _expert_lookup(expert_frame: pd.DataFrame, scope: int) -> pd.DataFrame:
    """Index the scope's **calendar-aligned** expert columns by ``(admin, month)``.

    The legacy record-shift series is a pipeline-validation artifact only, so the
    firewall is re-asserted here: it is structurally impossible for a
    ``legacy_record_shift_*`` column to reach the correction layer.
    """
    assert_no_legacy_expert_columns(expert_frame, f"fs{scope} correction input")
    frame = expert_frame.set_index(KEYS)
    assert_no_legacy_expert_columns(frame, f"fs{scope} correction input")
    return frame


def _attach_expert(
    admin_codes: np.ndarray,
    dates: pd.Series,
    lookup: pd.DataFrame,
    scope: int,
    *,
    context: str,
    require_available: bool,
) -> pd.DataFrame:
    """Join the reconstructed expert onto rows, halting on missing source matches."""
    index = pd.MultiIndex.from_arrays(
        [pd.Series(admin_codes).astype("int64"), pd.to_datetime(pd.Series(dates))],
        names=KEYS,
    )
    matched = lookup.reindex(index)
    unmatched = matched["source_truth"].isna()
    if unmatched.any():
        raise ExpertContractError(
            f"{context}: {int(unmatched.sum())} rows have no FEWS NET source match; "
            "halting rather than imputing or shrinking support"
        )
    expert = matched[f"expert_{scope}"]
    if require_available and expert.isna().any():
        raise ExpertContractError(
            f"{context}: {int(expert.isna().sum())} rows lack an available expert "
            "estimate on required support; halting"
        )
    return matched.reset_index(drop=True)


def _verify_truth_agreement(y: np.ndarray, source_truth: pd.Series, context: str) -> None:
    """Assert the main binary target equals nonmissing ``fews_ipc >= 3``."""
    mismatch = np.asarray(y, dtype=int) != source_truth.to_numpy(dtype=int)
    if mismatch.any():
        raise ExpertContractError(
            f"{context}: {int(mismatch.sum())} rows disagree with source fews_ipc>=3 truth"
        )


def run_fold(
    *,
    scope: int,
    target_month: pd.Period,
    panel: FeaturePanel,
    expert_lookup: pd.DataFrame,
    baseline: ReusedBaseline,
    map_path: Path,
    direction_mode: str = selection.DIRECTION_MODE_BOTH,
) -> FoldResult:
    """Execute one (scope, target month) correction fold end to end.

    ``direction_mode`` selects Variant A (``both``) or Variant B (``up-only``).
    Only the permitted flip directions change; every window, gate, learner and
    abstention rule is shared.
    """
    mode = selection.resolve_direction_mode(direction_mode)
    method = correction_method_id(mode)
    horizon = SCOPE_HORIZON_MONTHS[scope]
    fold_id = f"fs{scope}_{target_month}"
    fold_windows = windows.resolve_fold_windows(target_month, horizon)

    dates = panel.dates
    assignment = load_partition_assignment(map_path)
    partitions = assign_partitions(panel.admin_codes, assignment)

    test = windows.target_mask(dates, fold_windows)
    if not test.any():
        raise RunContractError(f"{fold_id}: target month has no panel rows")
    outer = windows.outer_mask(dates, fold_windows)
    eligible_groups = windows.group_eligibility_mask(partitions, partitions[test])
    outer_eligible = outer & eligible_groups
    validation = outer_eligible & windows.validation_mask(dates, fold_windows)
    initial_fit = outer_eligible & windows.fit_mask(dates, fold_windows)
    purge_gap = outer_eligible & windows.gap_mask(dates, fold_windows)

    # --- expert attachment and contract checks ------------------------------
    test_expert = _attach_expert(
        panel.admin_codes[test], dates[test], expert_lookup, scope,
        context=f"{fold_id} test support", require_available=True,
    )
    _verify_truth_agreement(panel.y[test], test_expert["source_truth"], f"{fold_id} test support")

    outer_expert = _attach_expert(
        panel.admin_codes[outer_eligible], dates[outer_eligible], expert_lookup, scope,
        context=f"{fold_id} outer window", require_available=False,
    )
    _verify_truth_agreement(
        panel.y[outer_eligible], outer_expert["source_truth"], f"{fold_id} outer window"
    )

    outer_index = np.flatnonzero(outer_eligible)
    expert_available = outer_expert[f"expert_{scope}"].notna().to_numpy()
    validation_index = outer_index[validation[outer_index] & expert_available]
    fit_index = outer_index[initial_fit[outer_index] & expert_available]
    refit_index = outer_index[expert_available]
    test_index = np.flatnonzero(test)

    outer_expert_by_row = pd.Series(
        outer_expert[f"expert_{scope}"].to_numpy(), index=outer_index
    )

    def expert_of(indices: np.ndarray) -> np.ndarray:
        return outer_expert_by_row.loc[indices].to_numpy(dtype=int)

    # --- align with the frozen support -------------------------------------
    archived = baseline.predictions.loc[
        baseline.predictions["month_start"].eq(target_month.to_timestamp())
    ].reset_index(drop=True)
    panel_test = pd.DataFrame(
        {
            "admin_code": panel.admin_codes[test].astype("int64"),
            "month_start": pd.to_datetime(dates[test]).to_numpy(),
            "y_true": panel.y[test].astype(int),
            "partition_id": partitions[test].astype(int),
        }
    ).sort_values(KEYS).reset_index(drop=True)
    if panel_test.duplicated(KEYS).any():
        raise RunContractError(f"{fold_id}: duplicate panel admin-month keys on target month")
    pd.testing.assert_frame_equal(panel_test[KEYS], archived[KEYS])
    if not panel_test["y_true"].equals(archived["y_true"].astype(int)):
        raise RunContractError(f"{fold_id}: panel truth differs from frozen support truth")
    if not panel_test["partition_id"].equals(archived["partition_id"].astype(int)):
        raise RunContractError(
            f"{fold_id}: frozen month map does not reproduce archived partition ids"
        )

    # --- stage 1: horizon-isolated fit, then validation scoring -------------
    candidates = pd.DataFrame()
    partition_records: List[Dict[str, object]] = []
    if fit_index.size == 0 or validation_index.size == 0:
        rule = selection.SelectedRule(
            corrected=False,
            threshold=None,
            enable_0_to_1=False,
            enable_1_to_0=False,
            expert_only_validation_f1=float("nan"),
            selected_validation_f1=float("nan"),
            status="no_correction",
            reason="insufficient fit or validation support after temporal isolation",
            direction_mode=mode,
        )
        validation_rows = pd.DataFrame()
        fit_ensemble_trained = 0
    else:
        imputer = fit_window_imputer(panel.X[fit_index])
        X_fit = transform_with(imputer, panel.X[fit_index])
        X_val = transform_with(imputer, panel.X[validation_index])
        fit_ensemble = correction.fit_correction_ensemble(
            X=X_fit,
            expert=expert_of(fit_index),
            truth=panel.y[fit_index],
            partitions=partitions[fit_index],
            stage="validation_fit",
        )
        fit_ensemble_trained = fit_ensemble.n_trained
        partition_records.extend(fit_ensemble.report_records())
        val_scores, val_eligible, val_reasons = correction.wrong_scores(
            fit_ensemble,
            X=X_val,
            expert=expert_of(validation_index),
            partitions=partitions[validation_index],
        )
        val_months = pd.to_datetime(dates[validation_index]).dt.to_period("M").astype(str)
        rule = selection.select_rule(
            truth=panel.y[validation_index],
            expert=expert_of(validation_index),
            wrong_scores=val_scores,
            eligible=val_eligible,
            months=val_months.to_numpy(),
            direction_mode=mode,
        )
        candidates = pd.DataFrame(rule.candidates)
        if not candidates.empty:
            candidates.insert(0, "fold_id", fold_id)
            candidates.insert(1, "scope", scope)
            candidates.insert(2, "target_month", str(target_month))
        validation_rows = pd.DataFrame(
            {
                "fold_id": fold_id,
                "row_role": "validation",
                "scope": scope,
                "target_month": str(target_month),
                "admin_code": panel.admin_codes[validation_index].astype("int64"),
                "label_month": pd.to_datetime(dates[validation_index]).to_numpy(),
                "partition_id": partitions[validation_index],
                "y_true": panel.y[validation_index],
                "expert": expert_of(validation_index),
                "wrong_score": val_scores,
                "score_eligible": val_eligible,
                "abstain_reason": val_reasons,
            }
        )

    # --- stage 2: refit on every eligible outer row, then apply frozen rule --
    refit_imputer = fit_window_imputer(panel.X[refit_index])
    X_refit = transform_with(refit_imputer, panel.X[refit_index])
    X_test = transform_with(refit_imputer, panel.X[test_index])
    refit_ensemble = correction.fit_correction_ensemble(
        X=X_refit,
        expert=expert_of(refit_index),
        truth=panel.y[refit_index],
        partitions=partitions[refit_index],
        stage="final_refit",
    )
    partition_records.extend(refit_ensemble.report_records())

    test_expert_values = test_expert[f"expert_{scope}"].to_numpy(dtype=int)
    test_scores, test_eligible, test_reasons = correction.wrong_scores(
        refit_ensemble,
        X=X_test,
        expert=test_expert_values,
        partitions=partitions[test_index],
    )
    final_predictions = selection.apply_rule(
        test_expert_values,
        test_scores,
        test_eligible,
        rule.threshold,
        rule.enable_0_to_1,
        rule.enable_1_to_0,
    )
    # Variant B: a 1->0 flip reaching the output is a contract error, checked
    # here on the actual test predictions rather than trusted from the flags.
    selection.assert_direction_contract(
        test_expert_values, final_predictions, mode, context=f"{fold_id} test predictions"
    )
    applied_flip = final_predictions != test_expert_values

    audit_rows = pd.DataFrame(
        {
            "fold_id": fold_id,
            "row_role": "test",
            "scope": scope,
            "horizon_months": horizon,
            "admin_code": panel.admin_codes[test_index].astype("int64"),
            "month_start": pd.to_datetime(dates[test_index]).to_numpy(),
            "partition_id": partitions[test_index],
            "y_true": panel.y[test_index],
            "expert": test_expert_values,
            "expert_source_month": test_expert[f"expert_{scope}_source_month"].to_numpy(),
            "expert_phase_raw": test_expert[f"expert_{scope}_phase_raw"].to_numpy(),
            "expert_phase_missing": test_expert[f"expert_{scope}_phase_missing"].to_numpy(),
            "wrong_score": test_scores,
            "score_eligible": test_eligible,
            "abstain_reason": test_reasons,
            "selected_threshold": rule.threshold,
            "enable_0_to_1": rule.enable_0_to_1,
            "enable_1_to_0": rule.enable_1_to_0,
            "applied_flip": applied_flip,
            f"y_pred_{method}": final_predictions,
        }
    )
    archived_pooled = archived.set_index(KEYS)["y_pred_pooled"]
    audit_rows["y_pred_pooled"] = archived_pooled.reindex(
        pd.MultiIndex.from_arrays(
            [audit_rows["admin_code"], audit_rows["month_start"]], names=KEYS
        )
    ).to_numpy(dtype=int)
    # Audit-only, computed strictly after selection.
    audit_rows["flip_fixed"] = applied_flip & (final_predictions == audit_rows["y_true"])
    audit_rows["flip_damaged"] = applied_flip & (final_predictions != audit_rows["y_true"])

    fold_record: Dict[str, object] = {
        "fold_id": fold_id,
        "scope": scope,
        "target_month": str(target_month),
        "method": method,
        "partition_map": str(map_path),
        "partition_map_sha256": sha256(map_path),
        "stage2_parent_map": str(stage2_map_path(MONTH_MAP_KEY.get(target_month.month, "general"))),
        "stage2_parent_map_sha256": sha256(
            stage2_map_path(MONTH_MAP_KEY.get(target_month.month, "general"))
        ),
        "outer_rows_eligible": int(outer_eligible.sum()),
        "outer_rows_expert_available": int(refit_index.size),
        "outer_observed_months": windows.describe_observed_months(dates, outer_eligible)[
            "n_months"
        ],
        "fit_rows": int(fit_index.size),
        "fit_observed_months": int(
            pd.to_datetime(dates[fit_index]).dt.to_period("M").nunique()
        )
        if fit_index.size
        else 0,
        "gap_rows_withheld": int(purge_gap.sum()),
        "gap_observed_months": windows.describe_observed_months(dates, purge_gap)["n_months"],
        "validation_rows": int(validation_index.size),
        "validation_distinct_months": int(
            pd.to_datetime(dates[validation_index]).dt.to_period("M").nunique()
        )
        if validation_index.size
        else 0,
        "validation_observed_month_list": ";".join(
            sorted(
                {
                    str(value)
                    for value in pd.to_datetime(dates[validation_index]).dt.to_period("M")
                }
            )
        )
        if validation_index.size
        else "",
        "test_rows": int(test_index.size),
        "validation_fit_partitions_trained": int(fit_ensemble_trained),
        "final_refit_partitions_trained": int(refit_ensemble.n_trained),
        "test_rows_score_eligible": int(test_eligible.sum()),
        "test_rows_flipped": int(applied_flip.sum()),
        "gate_min_proposed_flips": selection.MIN_PROPOSED_FLIPS,
        "gate_min_distinct_months": selection.MIN_DISTINCT_MONTHS,
        "gate_min_correction_precision": selection.MIN_CORRECTION_PRECISION,
    }
    fold_record.update(fold_windows.as_record())
    fold_record.update(rule.as_record())
    return FoldResult(
        audit_rows=audit_rows,
        validation_rows=validation_rows,
        candidates=candidates,
        partition_reports=pd.DataFrame(
            [{"fold_id": fold_id, "scope": scope, "target_month": str(target_month), **record}
             for record in partition_records]
        ),
        fold_record=fold_record,
    )


def monthly_metrics(
    audit: pd.DataFrame,
    baseline: ReusedBaseline,
    scope: int,
    method: str = CORRECTION_METHOD,
) -> pd.DataFrame:
    """Score the two reported methods on identical admin-month support.

    ``method`` is the variant's explicit correction method identifier; ``pooled``
    is always the reused frozen baseline and is re-verified against the archive.
    """
    rows: List[Dict[str, object]] = []
    for month, group in audit.groupby("month_start"):
        for model, column in (
            (method, f"y_pred_{method}"),
            (POOLED_METHOD, "y_pred_pooled"),
        ):
            counts, scores = counts_and_scores(group["y_true"], group[column])
            rows.append(
                {
                    "scope": scope,
                    "test_month": month.strftime("%Y-%m"),
                    "model": model,
                    "precision": scores[0],
                    "recall": scores[1],
                    "f1": scores[2],
                    "n": int(len(group)),
                    "tp": counts[0],
                    "fp": counts[1],
                    "fn": counts[2],
                    "tn": counts[3],
                    "correction_enabled": model == method,
                }
            )
    frame = pd.DataFrame(rows)
    # Pooled is reused verbatim: confirm the recomputed rows match the archive.
    for record in frame[frame["model"].eq(POOLED_METHOD)].to_dict("records"):
        match = baseline.metrics.loc[
            baseline.metrics["model"].eq("pooled")
            & baseline.metrics["test_month"].eq(record["test_month"])
        ]
        if len(match) != 1:
            raise RunContractError(f"Missing archived pooled metrics for {record['test_month']}")
        np.testing.assert_allclose(
            [record["precision"], record["recall"], record["f1"]],
            match.iloc[0][["precision", "recall", "f1"]].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )
    return frame


def uncorrected_scope_frames(scope: int) -> Dict[str, pd.DataFrame]:
    """Return fs3's reused, explicitly uncorrected predictions and metrics."""
    baseline = load_reused_baseline(scope)
    predictions = baseline.predictions.copy()
    predictions["scope"] = scope
    predictions["method_partitioned"] = UNCORRECTED_METHOD
    predictions["correction_enabled"] = False
    predictions["correction_applicable"] = False
    metrics = baseline.metrics.copy()
    metrics["scope"] = scope
    metrics["correction_enabled"] = False
    metrics["correction_applicable"] = False
    return {"predictions": predictions, "metrics": metrics, "checks": baseline.checks}


VARIANT_A_REFERENCE_RUN = "full_fs1_fs2_20260918"
ASYMMETRIC_COST_JUSTIFICATION = (
    "Asymmetric cost, which is independent of any test result: a 1->0 flip "
    "switches an already-issued crisis warning off, and in food-security early "
    "warning a missed crisis carries materially higher cost than a false alarm."
)
VARIANT_B_POST_HOC_CORROBORATION = {
    "statistic": "per-fold oracle F1 headroom over expert-only, by direction",
    "fs1": {"0_to_1_only": 0.0078, "1_to_0_only": 0.0034},
    "fs2": {"0_to_1_only": 0.0171, "1_to_0_only": 0.0020},
    "computed_with_test_labels": True,
    "role": (
        "post-hoc corroboration only. This decomposition was computed with test "
        "labels and is NOT the reason for the restriction (see "
        "asymmetric_cost_justification) and is NOT evidence that Variant B "
        "generalises."
    ),
}


def _variant_manifest(mode: str, method: str) -> Dict[str, object]:
    """Describe the run's variant, including its epistemic status."""
    if mode == selection.DIRECTION_MODE_BOTH:
        return {
            "variant": "A",
            "direction_mode": mode,
            "method_id": method,
            "directions_permitted": ["0->1", "1->0"],
            "epistemic_status": (
                "Variant A is the original approved mechanism and the only "
                "genuinely out-of-sample result for it: its direction set was "
                "fixed before any test metric was computed."
            ),
        }
    return {
        "variant": "B",
        "direction_mode": mode,
        "method_id": method,
        "directions_permitted": ["0->1"],
        "enable_1_to_0": "forced False before candidate scoring; never proposed or applied",
        "asymmetric_cost_justification": ASYMMETRIC_COST_JUSTIFICATION,
        "post_hoc_corroboration": VARIANT_B_POST_HOC_CORROBORATION,
        "epistemic_status": (
            "Variant B's direction restriction was chosen AFTER Variant A's test "
            "results were known. Its test metric is therefore a post-hoc, "
            "test-informed figure and is NOT an out-of-sample estimate. Variant A "
            f"(reference run {VARIANT_A_REFERENCE_RUN}) remains the only genuinely "
            "out-of-sample result for this mechanism. A Variant B number does not "
            "validate the approach, and any comparison against pooled must carry "
            "this caveat."
        ),
        "variant_a_reference_run": VARIANT_A_REFERENCE_RUN,
        "variant_a_untouched": (
            "Variant A's method id, code path and committed run directory are "
            "unchanged; this run writes a separate directory and method id."
        ),
        "gates_unchanged": (
            "20 proposed flips / >=2 distinct validation months / >=0.75 "
            "correction precision / strictly greater validation crisis F1 with "
            "ties to expert-only / strict q > threshold / one shared threshold"
        ),
    }


def run(
    *,
    scopes: Sequence[int],
    target_months: Sequence[str],
    out_dir: Path | str,
    panel_path: Path | str = PANEL_SOURCE,
    fewsnet_path: Path | str = FEWSNET_SOURCE,
    include_uncorrected_fs3: bool = True,
    allow_outside_repo: bool = False,
    direction_mode: str = selection.DIRECTION_MODE_BOTH,
) -> Path:
    """Run the isolated fs1/fs2 correction experiment and write all artifacts.

    ``direction_mode`` picks the variant: ``both`` (Variant A, default) or
    ``up-only`` (Variant B).  A Variant B run writes a distinct method id into a
    distinct run directory and never touches Variant A's artifacts.
    """
    mode = selection.resolve_direction_mode(direction_mode)
    method = correction_method_id(mode)
    for scope in scopes:
        if scope not in CORRECTION_SCOPES:
            raise ValueError(f"Correction scopes are {CORRECTION_SCOPES}, got fs{scope}")
    run_dir = resolve_output_dir(out_dir, allow_outside_repo=allow_outside_repo)

    hashes_before = hash_protected()
    expert_table = load_expert_history(fewsnet_path)
    # AC1a: the legacy record-shift series must still reproduce the archived
    # baselines, proving the loader and conventions are faithful.  AC1b: the
    # calendar-aligned series that the correction layer actually consumes must
    # sit at exactly ``T - H`` on every available row.
    baseline_verification = verify_archived_expert_baselines(expert_table)
    alignment_audit = {
        scope: require_source_alignment(expert_table.frame, scope) for scope in scopes
    }
    expert_coverage = {
        f"fs{scope}": coverage_report(
            expert_table.frame, scope, label="full reconstructed history"
        )
        for scope in scopes
    }

    run_dir.mkdir(parents=True)
    periods = [pd.Period(str(month), freq="M") for month in target_months]

    all_audit: List[pd.DataFrame] = []
    all_validation: List[pd.DataFrame] = []
    all_candidates: List[pd.DataFrame] = []
    all_partitions: List[pd.DataFrame] = []
    fold_records: List[Dict[str, object]] = []
    all_metrics: List[pd.DataFrame] = []
    scope_checks: Dict[str, object] = {}
    feature_provenance: Dict[str, object] = {}

    for scope in scopes:
        baseline = load_reused_baseline(scope)
        scope_checks[f"fs{scope}"] = baseline.checks
        panel = build_feature_panel(
            forecasting_scope=scope,
            panel_path=panel_path,
            working_dir=run_dir / "feature_workdir",
        )
        feature_provenance[f"fs{scope}"] = panel.provenance()
        pd.DataFrame(
            {
                "feature_index": np.arange(len(panel.feature_names)),
                "feature_name": panel.feature_names,
            }
        ).to_csv(run_dir / f"feature_order_fs{scope}.csv", index=False)
        expert_lookup = _expert_lookup(expert_table.for_scope(scope), scope)
        scope_audit: List[pd.DataFrame] = []
        for period in periods:
            result = run_fold(
                scope=scope,
                target_month=period,
                panel=panel,
                expert_lookup=expert_lookup,
                baseline=baseline,
                map_path=month_map_path(scope, period),
                direction_mode=mode,
            )
            scope_audit.append(result.audit_rows)
            all_audit.append(result.audit_rows)
            if not result.validation_rows.empty:
                all_validation.append(result.validation_rows)
            if not result.candidates.empty:
                all_candidates.append(result.candidates)
            if not result.partition_reports.empty:
                all_partitions.append(result.partition_reports)
            fold_records.append(result.fold_record)
        scope_frame = pd.concat(scope_audit, ignore_index=True)
        all_metrics.append(monthly_metrics(scope_frame, baseline, scope, method=method))

    def _tag(frame: pd.DataFrame) -> pd.DataFrame:
        """Stamp the direction mode on artifacts of a non-default variant.

        Variant A keeps its committed artifact schema byte-for-byte, so the
        provenance column is added only for a restricted mode.  Variant A is
        still identifiable everywhere through the method id in the
        ``y_pred_<method>`` column name, the metrics ``model`` value and the
        ``fold_tuning.method`` value.
        """
        if mode == selection.DIRECTION_MODE_BOTH or frame.empty:
            return frame
        tagged = frame.copy()
        tagged["direction_mode"] = mode
        tagged["enable_1_to_0_forced_disabled"] = True
        return tagged

    audit = pd.concat(all_audit, ignore_index=True)
    # Run-level re-check: independent of the per-fold assertion, prove from the
    # concatenated artifacts that a restricted mode neither enabled nor applied a
    # forbidden direction in any fold.
    if mode != selection.DIRECTION_MODE_BOTH:
        enabled_down = [
            record["fold_id"] for record in fold_records if record.get("enable_1_to_0")
        ]
        if enabled_down:
            raise selection.DirectionContractError(
                f"direction mode {mode!r} forbids 1->0 but folds enabled it: {enabled_down}"
            )
        selection.assert_direction_contract(
            audit["expert"].to_numpy(),
            audit[f"y_pred_{method}"].to_numpy(),
            mode,
            context="run-level test audit",
        )
    _tag(audit).to_csv(run_dir / "predictions_monthly_correction.csv", index=False)
    _tag(pd.concat(all_metrics, ignore_index=True)).to_csv(
        run_dir / "metrics_monthly_correction.csv", index=False
    )
    _tag(pd.DataFrame(fold_records)).to_csv(run_dir / "fold_tuning.csv", index=False)
    if all_candidates:
        _tag(pd.concat(all_candidates, ignore_index=True)).to_csv(
            run_dir / "fold_threshold_candidates.csv", index=False
        )
    if all_validation:
        _tag(pd.concat(all_validation, ignore_index=True)).to_csv(
            run_dir / "validation_rows.csv", index=False
        )
    if all_partitions:
        _tag(pd.concat(all_partitions, ignore_index=True)).to_csv(
            run_dir / "partition_trainability.csv", index=False
        )

    fs3_checks = None
    if include_uncorrected_fs3:
        for scope in UNCORRECTED_SCOPES:
            frames = uncorrected_scope_frames(scope)
            frames["predictions"].to_csv(
                run_dir / f"predictions_monthly_fs{scope}_uncorrected.csv", index=False
            )
            frames["metrics"].to_csv(
                run_dir / f"metrics_monthly_fs{scope}_uncorrected.csv", index=False
            )
            fs3_checks = frames["checks"]

    hashes_after = hash_protected()
    assert_unchanged(hashes_before)
    write_hash_report(run_dir / "protected_hashes.json", hashes_before, hashes_after)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Step 3 partitioned expert selective correction (isolated)",
        "methods": {
            "fs1_fs2_reported": [method, POOLED_METHOD],
            "fs3_reported": [UNCORRECTED_METHOD, POOLED_METHOD],
            "fs3_correction": "not applicable (explicitly uncorrected)",
            "expert_only": "internal selection/audit reference only; not a results series",
        },
        "variant": _variant_manifest(mode, method),
        "scopes_run": list(scopes),
        "target_months": [str(p) for p in periods],
        "sources": {
            "panel": {"path": str(panel_path), "sha256": sha256(panel_path)},
            "fewsnet": {"path": str(fewsnet_path), "sha256": sha256(fewsnet_path)},
        },
        "expert": {
            "expert_alignment": EXPERT_ALIGNMENT,
            "convention": EXPERT_CONVENTION,
            "origin_rule": "expert for target T is the projection published at O = T - H",
            "expert_horizon_verified": all(
                audit_entry["expert_horizon_verified"]
                for audit_entry in alignment_audit.values()
            ),
            "observed_lag_value_counts": {
                f"fs{scope}": audit_entry["observed_lag_value_counts"]
                for scope, audit_entry in alignment_audit.items()
            },
            "source_alignment_audit": alignment_audit,
            "coverage": expert_coverage,
            "legacy_record_shift_series": {
                "role": "pipeline-validation artifact only; never an input to the "
                        "correction layer (firewalled by column prefix "
                        "'legacy_record_shift_' and ExpertTable.for_scope)",
                "convention": LEGACY_EXPERT_CONVENTION,
                "archived_baseline_verification": baseline_verification,
            },
            "deferred_known_issue": (
                "The archived paper FEWS NET baseline was computed from the legacy "
                "record-shift series and therefore inherits its 12-32 month effective "
                "lag. Correcting or regenerating that frozen paper artifact is "
                "explicitly out of scope for this task and is recorded here only."
            ),
        },
        "reused_baselines": {
            "note": "pooled (fs1/fs2) and both fs3 methods are reused frozen archive "
                    "predictions, not newly trained",
            "per_scope_checks": scope_checks,
            "fs3_checks": fs3_checks,
        },
        "rf_params": correction.RF_PARAMS,
        "correction_learner": {
            "target": "w = 1[y != expert]",
            "inputs": "main feature order plus binary expert judgement as final column",
            "min_partition_samples": correction.MIN_PARTITION_SAMPLES,
            "resampling": "none (no SMOTE, no class weighting, no hyperparameter search)",
            "abstention": "absent/unmapped/too-small/single-class partitions retain expert",
            "score_semantics": "P(expert wrong); NOT a calibrated crisis probability",
        },
        "selection": {
            "threshold_candidates": "sorted unique wrong scores rounded to 2 decimals",
            "comparison": "strict q > threshold on unrounded scores",
            "gates": {
                "min_proposed_flips": selection.MIN_PROPOSED_FLIPS,
                "min_distinct_months": selection.MIN_DISTINCT_MONTHS,
                "min_correction_precision": selection.MIN_CORRECTION_PRECISION,
            },
            "objective": "strictly greater validation crisis-class F1 than expert-only; "
                         "ties keep expert-only",
        },
        "windows": {
            "configured_train_window_months": windows.MAIN_TRAIN_WINDOW_MONTHS,
            "outer_window_timestamps": windows.OUTER_WINDOW_TIMESTAMPS,
            "outer_window": "[O - 35 months, O) with O = T - H (historical off-by-one preserved)",
            "validation_calendar_months": windows.VALIDATION_CALENDAR_MONTHS,
            "validation_window": "[O - 12 months, O)",
            "validation_window_rationale": (
                "R8 revised 2026-09-18: source label months are tri-annual and four "
                "months apart while O % 4 == 2, so [O-6, O) held exactly one observed "
                "label month in 24/24 folds and made min_distinct_months>=2 "
                "unsatisfiable. [O-12, O) holds exactly three (O-4, O-8, O-12) with "
                "every approved gate unchanged."
            ),
            "fit_cutoff": "label month strictly before V_start - H",
            "fit_isolation_consequence": (
                "with V = 12 calendar months the horizon-isolated first-stage fit "
                "retains about 4 observed label months for fs1 and 3 for fs2; "
                "partitions under the 50-row minimum abstain as normal"
            ),
        },
        "features": feature_provenance,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "differs_from_historical_package_environment": True,
            "consequence": "pooled/fs3 are reused rather than retrained",
        },
        "protected_artifacts_unchanged": hashes_before == hashes_after,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    """Return the isolated runner's CLI parser."""
    parser = argparse.ArgumentParser(
        description="Step 3 fs1/fs2 partitioned expert selective-correction experiment",
    )
    parser.add_argument("--run-id", required=True, help="Immutable run identifier")
    parser.add_argument(
        "--scopes", nargs="+", type=int, default=list(CORRECTION_SCOPES),
        choices=list(CORRECTION_SCOPES), help="Forecasting scopes to correct (fs1/fs2 only)",
    )
    parser.add_argument(
        "--target-months", nargs="+", default=list(DEFAULT_TARGET_MONTHS),
        help="Evaluation target months (YYYY-MM)",
    )
    parser.add_argument("--panel", default=str(PANEL_SOURCE), help="Main panel CSV path")
    parser.add_argument("--fewsnet", default=str(FEWSNET_SOURCE), help="FEWSNET.csv path")
    parser.add_argument(
        "--acknowledge-unverified-expert-horizon", action="store_true",
        help=(
            "INERT and UNNECESSARY. Retained only so previously documented commands "
            "still parse. The expert is now calendar-aligned at exactly O = T-H "
            "(R1 revised 2026-09-18), so the horizon is verified on every available "
            "row and there is nothing to acknowledge. This flag cannot bypass the "
            "alignment gate: any stale or leakage-direction lag halts the run."
        ),
    )
    parser.add_argument(
        "--direction-mode", default=selection.DIRECTION_MODE_BOTH,
        choices=list(selection.DIRECTION_MODES),
        help=(
            "Permitted flip directions. 'both' (default) is Variant A, method "
            f"{CORRECTION_METHOD}. 'up-only' is Variant B, method "
            f"{CORRECTION_METHOD_UP_ONLY}: it forces enable_1_to_0=False before "
            "candidate scoring on asymmetric-cost grounds (a 1->0 flip silences "
            "an already-issued crisis warning, and a missed crisis costs more "
            "than a false alarm). Variant B was specified after Variant A's test "
            "results were known, so its test metric is post-hoc and "
            "test-informed, NOT an out-of-sample estimate."
        ),
    )
    parser.add_argument(
        "--no-fs3", action="store_true", help="Skip copying the reused uncorrected fs3 outputs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for the isolated correction experiment."""
    args = build_parser().parse_args(argv)
    if args.acknowledge_unverified_expert_horizon:
        print(
            "NOTICE: --acknowledge-unverified-expert-horizon is inert and unnecessary. "
            "The expert is calendar-aligned at exactly O = T-H, so the publication "
            "horizon is verified; the flag changes nothing and cannot bypass the gate."
        )
    run_dir = run(
        scopes=args.scopes,
        target_months=args.target_months,
        out_dir=APPROVED_OUTPUT_ROOT / args.run_id,
        panel_path=args.panel,
        fewsnet_path=args.fewsnet,
        include_uncorrected_fs3=not args.no_fs3,
        direction_mode=args.direction_mode,
    )
    if args.direction_mode != selection.DIRECTION_MODE_BOTH:
        print(
            "NOTICE: Variant B (--direction-mode up-only, method "
            f"{CORRECTION_METHOD_UP_ONLY}). The 1->0 direction is forced off on "
            "asymmetric-cost grounds. This variant was specified after Variant A's "
            "test results were known, so its test metric is post-hoc and "
            "test-informed, NOT an out-of-sample estimate; Variant A "
            f"({VARIANT_A_REFERENCE_RUN}) remains the only genuinely out-of-sample "
            "result for this mechanism."
        )
    print(f"Correction run written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
