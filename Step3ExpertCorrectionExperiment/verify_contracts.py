#!/usr/bin/env python3
"""Pre-flight contract evidence for the Step 3 expert selective-correction experiment.

Performs every check that must pass *before* any correction fitting, and prints
machine-checkable evidence.  No model is trained and no protected artifact is
written.

Items, in order:

1. hash every protected input
2. PIPELINE VALIDATION - the *legacy* record-shift series still reproduces the
   archived FEWS NET baselines exactly (AC1a)
3. EXPERIMENT EXPERT - the calendar-aligned series sits at exactly ``O = T - H``
   on every available row, and coverage is reported (AC1b)
3b. calendar vs legacy expert on the 2021-2024 main evaluation support
4. frozen contig3 month maps reproduce the archived partition ids
5. reused pooled / fs3 baselines pass every equality and metric check
6. ``V = [O - 12 months, O)`` holds exactly three observed label months in all
   24 folds, so the approved ``distinct months >= 2`` gate is satisfiable (AC4)
7. protected artifacts unchanged

Run from the repository root:

    .venv-geodt-diagnostic/bin/python Step3ExpertCorrectionExperiment/verify_contracts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step3correction import baselines, expert, protected, runner, selection, windows  # noqa: E402


def main(argv=None) -> int:
    """Run and report every pre-flight contract check."""
    parser = argparse.ArgumentParser(description="Step 3 correction pre-flight checks")
    parser.add_argument(
        "--report", default=None, help="Optional JSON path for the evidence report"
    )
    args = parser.parse_args(argv)

    evidence: dict[str, object] = {}
    hashes_before = protected.hash_protected()
    evidence["protected_inputs_hashed"] = len(hashes_before)
    print(f"[1] Hashed {len(hashes_before)} protected inputs")

    # --- [2] PIPELINE VALIDATION: legacy record-shift archived reproduction --
    table = expert.load_expert_history()
    reproduction = expert.verify_archived_expert_baselines(table)
    evidence["legacy_record_shift_archived_baseline_quarters"] = reproduction
    print(
        "[2] PIPELINE VALIDATION (legacy record-shift series, not the experiment "
        "expert): archived FEWS NET baselines reproduced at atol=1e-12: "
        f"fs1={reproduction[1]['quarters_reproduced']}/39, "
        f"fs2={reproduction[2]['quarters_reproduced']}/39"
    )

    # --- [3] EXPERIMENT EXPERT: exact H-month calendar alignment (blocking) --
    alignment: dict[str, object] = {}
    coverage: dict[str, object] = {}
    alignment_ok = True
    for scope in (1, 2):
        horizon = expert.SCOPE_HORIZON_MONTHS[scope]
        # require_source_alignment raises on any stale or leakage-direction lag.
        audit = expert.require_source_alignment(table.frame, scope)
        alignment[f"fs{scope}"] = audit
        report = expert.coverage_report(
            table.frame, scope, label="full reconstructed history"
        )
        coverage[f"fs{scope}"] = report
        exact = audit["observed_lag_value_counts"] == {horizon: audit["rows_with_expert"]}
        alignment_ok = alignment_ok and exact and audit["expert_horizon_verified"]
        assert exact, (
            f"fs{scope} calendar alignment is not a single bucket at {horizon}: "
            f"{audit['observed_lag_value_counts']}"
        )
        print(
            f"[3] EXPERIMENT EXPERT fs{scope} calendar alignment at O=T-{horizon}: "
            f"{'PASS' if exact else 'FAIL'}. "
            f"rows_at_declared_origin={audit['rows_at_declared_origin']} "
            f"off={audit['rows_off_declared_origin']} "
            f"unavailable={audit['rows_expert_unavailable']} "
            f"observed calendar lags={audit['observed_lag_value_counts']} "
            f"full-history coverage={report['coverage']:.4f}"
        )
    evidence["source_alignment"] = alignment
    evidence["expert_coverage"] = coverage
    evidence["expert_alignment_exact"] = bool(alignment_ok)

    # --- [3b] calendar vs legacy expert on the main evaluation support ------
    main_support = table.frame.loc[
        table.frame["month_start"].between(
            pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-01")
        )
        & table.frame["fews_ipc"].notna()
    ]
    support_report: dict[str, object] = {"rows": int(len(main_support))}
    for scope in (1, 2):
        entry: dict[str, object] = {}
        for label, column in (
            ("calendar_aligned", f"expert_{scope}"),
            ("legacy_record_shift", expert.legacy_column(scope)),
        ):
            series = main_support.dropna(subset=[column])
            counts, scores = expert.counts_and_scores(
                series["source_truth"], series[column]
            )
            entry[label] = {
                "coverage": float(main_support[column].notna().mean()),
                "rows": int(len(series)),
                "precision": scores[0],
                "recall": scores[1],
                "f1": scores[2],
                "tp": counts[0],
                "fp": counts[1],
                "fn": counts[2],
            }
        support_report[f"fs{scope}"] = entry
        print(
            f"[3b] fs{scope} on 2021-2024 targets (n={support_report['rows']}): "
            f"calendar coverage={entry['calendar_aligned']['coverage']:.4f} "
            f"F1={entry['calendar_aligned']['f1']:.4f} "
            f"(P={entry['calendar_aligned']['precision']:.4f} "
            f"R={entry['calendar_aligned']['recall']:.4f}) | "
            f"legacy F1={entry['legacy_record_shift']['f1']:.4f} "
            "[legacy = pipeline-validation artifact, inherited by the archived "
            "paper baseline; correcting that artifact is out of scope]"
        )
    evidence["expert_main_support_comparison"] = support_report

    # --- frozen partition maps reproduce archived assignments ---------------
    map_checks = []
    for scope in (1, 2, 3):
        frozen = baselines.load_reused_baseline(scope)
        predictions = frozen.predictions.copy()
        predictions["month"] = predictions["month_start"].dt.month
        for month, group in predictions.groupby("month"):
            period = pd.Period(f"2021-{month:02d}", freq="M")
            map_path = runner.month_map_path(scope, period)
            assignment = runner.load_partition_assignment(map_path)
            mapped = runner.assign_partitions(group["admin_code"].to_numpy(), assignment)
            matches = bool(np.array_equal(mapped, group["partition_id"].to_numpy(dtype=int)))
            map_checks.append(
                {
                    "scope": scope,
                    "month": int(month),
                    "map": map_path.name,
                    "sha256": protected.sha256(map_path),
                    "reproduces_archived_partition_ids": matches,
                }
            )
    evidence["partition_map_checks"] = map_checks
    all_maps_ok = all(check["reproduces_archived_partition_ids"] for check in map_checks)
    print(
        f"[4] Frozen contig3 month maps reproduce archived partition ids: {all_maps_ok} "
        f"({len(map_checks)} scope-month checks)"
    )

    # --- reused pooled / fs3 baselines --------------------------------------
    evidence["reused_baselines"] = {f"fs{scope}": baselines.load_reused_baseline(scope).checks
                                    for scope in (1, 2, 3)}
    print("[5] Reused pooled/fs3 baselines: archive==package hashes, per-row equality, "
          "and 24 archived metric rows reproduced for each of fs1/fs2/fs3")

    # --- [6] V=[O-12,O) observed label months under the approved gates ------
    labeled = pd.Series(
        pd.to_datetime(
            sorted(
                pd.read_csv(protected.PANEL_SOURCE, usecols=["date", "fews_ipc_crisis"])
                .pipe(lambda frame: frame[frame["fews_ipc_crisis"].notna()])["date"]
                .unique()
            )
        )
    )
    fold_rows = []
    for scope in (1, 2):
        horizon = expert.SCOPE_HORIZON_MONTHS[scope]
        for month in runner.DEFAULT_TARGET_MONTHS:
            fold = windows.resolve_fold_windows(pd.Period(month, freq="M"), horizon)
            validation = windows.describe_observed_months(
                labeled, windows.validation_mask(labeled, fold)
            )
            expected = [
                str((fold.origin - pd.DateOffset(months=offset)).to_period("M"))
                for offset in (12, 8, 4)
            ]
            fold_rows.append(
                {
                    "scope": scope,
                    "target_month": month,
                    "origin": str(fold.origin.to_period("M")),
                    "validation_start": str(fold.validation_start.date()),
                    "validation_end_exclusive": str(fold.validation_end.date()),
                    "outer_months": windows.describe_observed_months(
                        labeled, windows.outer_mask(labeled, fold)
                    )["n_months"],
                    "fit_months": windows.describe_observed_months(
                        labeled, windows.fit_mask(labeled, fold)
                    )["n_months"],
                    "gap_months": windows.describe_observed_months(
                        labeled, windows.gap_mask(labeled, fold)
                    )["n_months"],
                    "validation_months": validation["n_months"],
                    "validation_month_list": ";".join(validation["months"]),
                    "validation_months_are_O_minus_4_8_12": validation["months"] == expected,
                }
            )
    folds = pd.DataFrame(fold_rows)
    evidence["observed_validation_months"] = folds.to_dict("records")
    satisfiable = int((folds["validation_months"] >= selection.MIN_DISTINCT_MONTHS).sum())
    exactly_three = int((folds["validation_months"] == 3).sum())
    identity_ok = bool(folds["validation_months_are_O_minus_4_8_12"].all())
    assert exactly_three == len(folds), (
        "V=[O-12,O) must hold exactly three observed label months in every fold; got "
        f"{folds['validation_months'].value_counts().to_dict()}"
    )
    assert identity_ok, "validation months are not exactly {O-4, O-8, O-12}"
    print(
        f"[6] Observed label months inside V=[O-{windows.VALIDATION_CALENDAR_MONTHS},O): "
        f"exactly 3 in {exactly_three}/{len(folds)} folds "
        f"(all equal to O-4, O-8, O-12: {identity_ok}); "
        f"{satisfiable}/{len(folds)} folds can satisfy the "
        f"min_distinct_months>={selection.MIN_DISTINCT_MONTHS} directional gate. "
        f"Horizon-isolated fit observed months: fs1="
        f"{sorted(set(folds.loc[folds.scope.eq(1), 'fit_months']))} "
        f"fs2={sorted(set(folds.loc[folds.scope.eq(2), 'fit_months']))}"
    )

    protected.assert_unchanged(hashes_before)
    print("[7] Protected artifacts unchanged")
    evidence["protected_artifacts_unchanged"] = True

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
        print(f"Evidence written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
