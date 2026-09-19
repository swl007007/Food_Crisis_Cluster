#!/usr/bin/env python3
"""Independent recomputation and reporting for Variant B (``up-only``).

Reads only saved run artifacts - it never refits a model - and recomputes, from
first principles:

* overall and per-scope crisis-class metrics for expert-only, Variant B,
  Variant A (the committed reference run) and the reused pooled baseline;
* flip counts with their fixed/damaged split and test flip precision, by
  direction;
* the Variant B direction contract (``enable_1_to_0`` False in every fold and no
  ``1->0`` flip anywhere in the output);
* each fold's selected rule, re-derived from ``validation_rows.csv`` alone, and
  each fold's final test predictions, re-derived from the audit row fields alone;
* validation-vs-test flip precision per enabled direction.

Epistemic note carried into the report: Variant B's direction restriction was
chosen after Variant A's test results were known.  Its test metric is a post-hoc,
test-informed figure, **not** an out-of-sample estimate.  Variant A remains the
only genuinely out-of-sample result for this mechanism.  The restriction itself
is justified by asymmetric cost - a ``1->0`` flip silences an already-issued
crisis warning, and a missed food-security crisis costs materially more than a
false alarm - which is independent of any test result.

Run from the repository root:

    PYTHONPATH="$PWD/Step3ExpertCorrectionExperiment" \
      .venv-geodt-diagnostic/bin/python \
      Step3ExpertCorrectionExperiment/analyze_variant_b.py \
      --report Step3ExpertCorrectionExperiment/logs/variant_b_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from step3correction import protected, runner, selection

OUTPUTS = protected.EXPERIMENT_DIR / "outputs"
VARIANT_A_RUN = OUTPUTS / "full_fs1_fs2_20260918"
VARIANT_B_RUN = OUTPUTS / "full_fs1_fs2_up_only_20260918"

EPISTEMIC_STATUS = (
    "Variant B's direction restriction was chosen AFTER Variant A's test results "
    "were known. Its test metric is a post-hoc, test-informed figure and is NOT "
    "an out-of-sample estimate. Variant A (full_fs1_fs2_20260918) remains the "
    "only genuinely out-of-sample result for this mechanism. The restriction is "
    "justified by asymmetric cost (a 1->0 flip silences an already-issued crisis "
    "warning; a missed crisis costs materially more than a false alarm), which is "
    "independent of any test result. Any comparison against pooled must carry "
    "this caveat."
)


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Crisis-class precision/recall/F1 with the archived zero convention."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": selection.crisis_f1(y, p),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": int(len(y)),
    }


def flip_summary(audit: pd.DataFrame, method_column: str) -> dict:
    """Flip counts, fixed/damaged split and test flip precision by direction."""
    flipped = audit.loc[audit["applied_flip"].astype(bool)]
    out: dict = {}
    for name, subset in (
        ("all", flipped),
        ("0_to_1", flipped.loc[flipped["expert"].eq(0)]),
        ("1_to_0", flipped.loc[flipped["expert"].eq(1)]),
    ):
        count = int(len(subset))
        fixed = int((subset[method_column] == subset["y_true"]).sum())
        out[name] = {
            "flips": count,
            "fixed": fixed,
            "damaged": count - fixed,
            "test_flip_precision": (fixed / count) if count else None,
        }
    return out


def recompute_final_predictions(audit: pd.DataFrame, method_column: str) -> dict:
    """Re-derive every final test label from the saved audit fields alone."""
    mismatches = 0
    for fold_id, group in audit.groupby("fold_id"):
        threshold = group["selected_threshold"].iloc[0]
        threshold = None if pd.isna(threshold) else float(threshold)
        recomputed = selection.apply_rule(
            group["expert"].to_numpy(dtype=int),
            group["wrong_score"].to_numpy(dtype=float),
            group["score_eligible"].to_numpy(dtype=bool),
            threshold,
            bool(group["enable_0_to_1"].iloc[0]),
            bool(group["enable_1_to_0"].iloc[0]),
        )
        mismatches += int((recomputed != group[method_column].to_numpy(dtype=int)).sum())
    return {"rows": int(len(audit)), "mismatched_rows": mismatches}


def recompute_selection(run_dir: Path, direction_mode: str) -> list:
    """Re-run the selector on saved validation rows and compare to fold_tuning."""
    validation = pd.read_csv(run_dir / "validation_rows.csv")
    folds = pd.read_csv(run_dir / "fold_tuning.csv").set_index("fold_id")
    results = []
    for fold_id, group in validation.groupby("fold_id"):
        months = pd.to_datetime(group["label_month"]).dt.to_period("M").astype(str)
        rule = selection.select_rule(
            truth=group["y_true"].to_numpy(dtype=int),
            expert=group["expert"].to_numpy(dtype=int),
            wrong_scores=group["wrong_score"].to_numpy(dtype=float),
            eligible=group["score_eligible"].to_numpy(dtype=bool),
            months=months.to_numpy(),
            direction_mode=direction_mode,
        )
        saved = folds.loc[fold_id]
        saved_threshold = None if pd.isna(saved["selected_threshold"]) else float(
            saved["selected_threshold"]
        )
        agrees = (
            bool(saved["correction_selected"]) == rule.corrected
            and bool(saved["enable_0_to_1"]) == rule.enable_0_to_1
            and bool(saved["enable_1_to_0"]) == rule.enable_1_to_0
            and (
                (saved_threshold is None and rule.threshold is None)
                or (
                    saved_threshold is not None
                    and rule.threshold is not None
                    and abs(saved_threshold - rule.threshold) < 1e-12
                )
            )
            and abs(float(saved["selected_validation_f1"]) - rule.selected_validation_f1) < 1e-12
            and abs(
                float(saved["expert_only_validation_f1"]) - rule.expert_only_validation_f1
            ) < 1e-12
        )
        # Validation-side flip precision for the rule actually selected.
        val_precision = None
        val_flips = None
        if rule.corrected:
            proposed = selection.proposed_flip_mask(
                group["wrong_score"].to_numpy(dtype=float),
                group["score_eligible"].to_numpy(dtype=bool),
                rule.threshold,
            )
            enabled = np.zeros(len(group), dtype=bool)
            if rule.enable_0_to_1:
                enabled |= proposed & group["expert"].eq(0).to_numpy()
            if rule.enable_1_to_0:
                enabled |= proposed & group["expert"].eq(1).to_numpy()
            val_flips = int(enabled.sum())
            if val_flips:
                fixes = int(
                    (
                        group["y_true"].to_numpy(dtype=int)[enabled]
                        != group["expert"].to_numpy(dtype=int)[enabled]
                    ).sum()
                )
                val_precision = fixes / val_flips
        results.append(
            {
                "fold_id": fold_id,
                "selection_status": saved["selection_status"],
                "selected_threshold": saved_threshold,
                "enable_0_to_1": bool(saved["enable_0_to_1"]),
                "enable_1_to_0": bool(saved["enable_1_to_0"]),
                "expert_only_validation_f1": float(saved["expert_only_validation_f1"]),
                "selected_validation_f1": float(saved["selected_validation_f1"]),
                "validation_flips": val_flips,
                "validation_flip_precision": val_precision,
                "test_rows_flipped": int(saved["test_rows_flipped"]),
                "independently_reproduced": bool(agrees),
            }
        )
    return results


def main(argv=None) -> int:
    """Recompute and print the Variant B report."""
    parser = argparse.ArgumentParser(description="Variant B independent recomputation")
    parser.add_argument("--report", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    method_a = f"y_pred_{runner.CORRECTION_METHOD}"
    method_b = f"y_pred_{runner.CORRECTION_METHOD_UP_ONLY}"
    audit_a = pd.read_csv(VARIANT_A_RUN / "predictions_monthly_correction.csv")
    audit_b = pd.read_csv(VARIANT_B_RUN / "predictions_monthly_correction.csv")

    report: dict = {"epistemic_status": EPISTEMIC_STATUS}

    # --- contract: Variant B never enables or applies 1->0 ------------------
    folds_b = pd.read_csv(VARIANT_B_RUN / "fold_tuning.csv")
    down_flips = int(
        (audit_b["applied_flip"].astype(bool) & audit_b["expert"].eq(1)).sum()
    )
    report["variant_b_direction_contract"] = {
        "folds": int(len(folds_b)),
        "folds_with_enable_1_to_0_true": int(folds_b["enable_1_to_0"].astype(bool).sum()),
        "folds_marked_forced_disabled": int(
            folds_b["enable_1_to_0_forced_disabled"].astype(bool).sum()
        ),
        "test_rows_flipped_1_to_0": down_flips,
        "method_ids_in_fold_tuning": sorted(folds_b["method"].unique().tolist()),
        "direction_modes_in_fold_tuning": sorted(folds_b["direction_mode"].unique().tolist()),
        "variant_a_method_column_absent": method_a not in audit_b.columns,
        "pass": bool(
            down_flips == 0
            and not folds_b["enable_1_to_0"].astype(bool).any()
            and method_a not in audit_b.columns
        ),
    }
    selection.assert_direction_contract(
        audit_b["expert"].to_numpy(),
        audit_b[method_b].to_numpy(),
        selection.DIRECTION_MODE_UP_ONLY,
        context="variant B saved audit",
    )

    # --- support identity ---------------------------------------------------
    keys = ["scope", "admin_code", "month_start"]
    left = audit_a[keys + ["y_true", "expert", "y_pred_pooled"]].sort_values(keys)
    right = audit_b[keys + ["y_true", "expert", "y_pred_pooled"]].sort_values(keys)
    report["identical_support"] = {
        "rows": int(len(left)),
        "keys_equal": bool(
            left[keys].reset_index(drop=True).equals(right[keys].reset_index(drop=True))
        ),
        "truth_equal": bool(
            left["y_true"].reset_index(drop=True).equals(right["y_true"].reset_index(drop=True))
        ),
        "expert_equal": bool(
            left["expert"].reset_index(drop=True).equals(right["expert"].reset_index(drop=True))
        ),
        "pooled_equal": bool(
            left["y_pred_pooled"].reset_index(drop=True).equals(
                right["y_pred_pooled"].reset_index(drop=True)
            )
        ),
    }

    # --- headline metrics ---------------------------------------------------
    per_scope: dict = {}
    for scope in (1, 2):
        a = audit_a.loc[audit_a["scope"].eq(scope)].sort_values(["admin_code", "month_start"])
        b = audit_b.loc[audit_b["scope"].eq(scope)].sort_values(["admin_code", "month_start"])
        y = b["y_true"].to_numpy(dtype=int)
        per_scope[f"fs{scope}"] = {
            "expert_only": scores(y, b["expert"].to_numpy(dtype=int)),
            "variant_b_up_only": scores(y, b[method_b].to_numpy(dtype=int)),
            "variant_a_both_directions": scores(
                a["y_true"].to_numpy(dtype=int), a[method_a].to_numpy(dtype=int)
            ),
            "pooled": scores(y, b["y_pred_pooled"].to_numpy(dtype=int)),
            "flips_variant_b": flip_summary(b, method_b),
            "flips_variant_a": flip_summary(a, method_a),
        }
        for label in ("variant_b_up_only", "variant_a_both_directions", "pooled"):
            per_scope[f"fs{scope}"][label]["f1_minus_expert_only"] = (
                per_scope[f"fs{scope}"][label]["f1"]
                - per_scope[f"fs{scope}"]["expert_only"]["f1"]
            )
    report["per_scope"] = per_scope

    # --- independent recomputation -----------------------------------------
    report["recomputed_final_predictions"] = {
        "variant_b": recompute_final_predictions(audit_b, method_b),
        "variant_a": recompute_final_predictions(audit_a, method_a),
    }
    report["recomputed_selection_variant_b"] = recompute_selection(
        VARIANT_B_RUN, selection.DIRECTION_MODE_UP_ONLY
    )

    # --- validation vs test flip precision, pooled over corrected folds -----
    corrected = [f for f in report["recomputed_selection_variant_b"] if f["enable_0_to_1"]]
    val_flips = sum(f["validation_flips"] or 0 for f in corrected)
    val_fixes = sum(
        round((f["validation_flip_precision"] or 0.0) * (f["validation_flips"] or 0))
        for f in corrected
    )
    test_all = flip_summary(audit_b, method_b)["0_to_1"]
    report["validation_vs_test_flip_precision"] = {
        "corrected_folds": len(corrected),
        "validation_flips": val_flips,
        "validation_flip_precision": (val_fixes / val_flips) if val_flips else None,
        "test_flips": test_all["flips"],
        "test_flip_precision": test_all["test_flip_precision"],
    }

    report["protected_hashes"] = {
        "count": len(protected.hash_protected()),
        "unchanged_vs_variant_a_run": json.loads(
            (VARIANT_A_RUN / "protected_hashes.json").read_text()
        )["protected_hashes_after"]
        == protected.hash_protected(),
        "variant_b_run_report_unchanged_flag": json.loads(
            (VARIANT_B_RUN / "protected_hashes.json").read_text()
        )["unchanged"],
    }

    print(json.dumps(report, indent=2, default=str))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
