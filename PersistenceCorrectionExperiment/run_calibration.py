#!/usr/bin/env python3
"""Phase 3 entrypoint: fit, freeze and apply the probability calibrators.

Run from the repository root::

    PYTHONPATH="$PWD/PersistenceCorrectionExperiment" python3 \
        PersistenceCorrectionExperiment/run_calibration.py --run-id phase3

What it does, in order
----------------------
1. hashes the 35 protected inputs;
2. builds the layer-1 persistence series on the **Phase 2 probability support**
   for both the 2018-2020 and the 2021-2024 windows (same calendar join at
   ``T-H``, same coverage-1.0 halt, same no-imputation rule as Phase 1);
3. fits the calibrators on **2018-2019 only** and freezes them to disk with a
   SHA-256 digest (DECISIONS_LOG C5/C6);
4. applies the frozen set to 2018-2019 (in-sample), 2020 (out-of-sample) and
   2021-2024 (emitted for Phase 4/5 to consume);
5. writes pre/post reliability by persistence group for 2018-2019 and 2020 only;
6. prints the PRD R5 gate figures on the 2020 out-of-sample rows.

What it deliberately does **not** do
------------------------------------
No metric is computed against the 2021-2024 labels, no threshold is selected and
no override is applied.  The R5 gate is *reported*, not adjudicated: the exit
code is 0 whether or not the 0.05 criterion is met, so that the decision stays
with a human (Phase 3 ends at a stop/go gate owned by the task author).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from persistencecorrection import calibration as C  # noqa: E402
from persistencecorrection import persistence as P  # noqa: E402
from persistencecorrection import protected  # noqa: E402

PHASE2_ROOT = protected.OUTPUT_ROOT / "phase2_probabilities"

#: (window label, directory template, sample status relative to the calibrator).
WINDOWS = (
    ("fit_2018_2019", "probs_2018_2020_fs{scope}", "in_sample_for_calibrator"),
    ("selection_2020", "probs_2018_2020_fs{scope}", "out_of_sample"),
    ("test_2021_2024", "probs_2021_2024_fs{scope}", "out_of_sample"),
)

WINDOW_YEARS = {
    "fit_2018_2019": C.FIT_YEARS,
    "selection_2020": C.SELECTION_YEARS,
    "test_2021_2024": C.TEST_YEARS,
}

#: Windows Phase 3 is allowed to score. The test window is emitted, never measured.
MEASURABLE_WINDOWS = ("fit_2018_2019", "selection_2020")

SERIES_COLUMNS = [
    P.SUPPORT_ADMIN_COLUMN,
    P.SUPPORT_MONTH_COLUMN,
    "partition_id",
    "y_true",
    "y_prob_pooled",
    "y_prob_partitioned",
    "y_pred_pooled",
    "y_pred_partitioned",
] + P.PERSISTENCE_COLUMNS


def probability_path(scope: int, template: str) -> Path:
    return PHASE2_ROOT / template.format(scope=scope) / "predictions_monthly.csv"


def load_window(scope: int, label: str, table) -> pd.DataFrame:
    """Persistence-joined Phase 2 probabilities for one window of one scope."""
    template = dict((name, tmpl) for name, tmpl, _ in WINDOWS)[label]
    frame = P.persistence_for_probability_file(
        probability_path(scope, template), table, scope
    )
    years = pd.to_datetime(frame[P.SUPPORT_MONTH_COLUMN]).dt.year
    selected = frame.loc[years.isin(WINDOW_YEARS[label])].reset_index(drop=True)
    if selected.empty:
        raise C.CalibrationContractError(f"Window {label} is empty for fs{scope}")
    return selected


def environment_report() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="immutable output directory name")
    parser.add_argument("--scopes", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args(argv)

    output_dir = protected.resolve_output_path(args.run_id)
    if output_dir.exists():
        raise FileExistsError(
            f"Run directory already exists (run IDs are immutable): {output_dir}"
        )

    before = protected.hash_protected()
    output_dir.mkdir(parents=True, exist_ok=False)

    table = P.load_observed_phase_history()
    reliability_rows: list[dict] = []
    bin_rows: list[pd.DataFrame] = []
    group_rows: list[pd.DataFrame] = []
    manifest_scopes: dict[str, object] = {}
    gate_rows: list[dict] = []

    for scope in args.scopes:
        windows = {
            label: load_window(scope, label, table) for label, _, _ in WINDOWS
        }

        # -- AC3: independently verify the fit window is disjoint ----------
        # assert_fit_window() already refuses to fit outside FIT_YEARS. This is
        # the second, key-level check: no admin-month key used for fitting may
        # reappear in the selection or test window.
        keys = {
            label: set(
                zip(
                    frame[P.SUPPORT_ADMIN_COLUMN].astype("int64"),
                    pd.to_datetime(frame[P.SUPPORT_MONTH_COLUMN]).dt.strftime("%Y-%m"),
                )
            )
            for label, frame in windows.items()
        }
        overlaps = {
            f"fit_vs_{label}": sorted(keys["fit_2018_2019"] & keys[label])[:5]
            for label in ("selection_2020", "test_2021_2024")
        }
        if any(overlaps.values()):
            raise C.CalibrationContractError(
                f"fs{scope} calibration-fit keys leak into a later window: {overlaps}"
            )

        # -- fit and freeze (2018-2019 only) ------------------------------
        calibrators = C.fit_calibrators(windows["fit_2018_2019"], scope)
        frozen_path = output_dir / f"calibrators_fs{scope}.json"
        frozen_path.write_text(calibrators.canonical_json(), encoding="utf-8")
        digest = calibrators.digest()

        # Determinism check: refitting the same input must give the same hash.
        refit_digest = C.fit_calibrators(windows["fit_2018_2019"], scope).digest()
        # Round-trip check: the artifact on disk must rebuild an identical set.
        reloaded = C.CalibratorSet.from_dict(json.loads(frozen_path.read_text("utf-8")))
        reload_digest = reloaded.digest()

        # -- apply everywhere ----------------------------------------------
        applied = {
            label: C.apply_calibrators(frame, calibrators)
            for label, frame in windows.items()
        }
        for label, frame in applied.items():
            coverage = P.persistence_coverage(frame, scope, label=label)
            frame[SERIES_COLUMNS + [C.CALIBRATED_COLUMN, C.ROUTE_COLUMN,
                                    C.ROUTE_REASON_COLUMN]].to_csv(
                output_dir / f"calibrated_{label}_fs{scope}.csv", index=False
            )
            manifest_scopes.setdefault(str(scope), {}).setdefault("windows", {})[
                label
            ] = {
                "rows": int(len(frame)),
                "months": sorted(
                    pd.to_datetime(frame[P.SUPPORT_MONTH_COLUMN])
                    .dt.strftime("%Y-%m")
                    .unique()
                    .tolist()
                ),
                "persistence_coverage": coverage["coverage"],
                "persistence_rows_with_raw_missing_phase": coverage[
                    "rows_with_raw_missing_phase"
                ],
                "source": str(
                    probability_path(
                        scope, dict((n, t) for n, t, _ in WINDOWS)[label]
                    )
                ),
                "scored_by_phase3": label in MEASURABLE_WINDOWS,
                "brier_pre_calibration": (
                    C.brier_score(frame["y_true"], frame[C.PROB_COLUMN])
                    if label in MEASURABLE_WINDOWS
                    else None
                ),
                "brier_post_calibration": (
                    C.brier_score(frame["y_true"], frame[C.CALIBRATED_COLUMN])
                    if label in MEASURABLE_WINDOWS
                    else None
                ),
            }

        # -- reliability, measurable windows only ---------------------------
        for label, _, sample_status in WINDOWS:
            if label not in MEASURABLE_WINDOWS:
                continue
            frame = applied[label]
            for stage, column in (("pre", C.PROB_COLUMN), ("post", C.CALIBRATED_COLUMN)):
                reliability_rows.extend(
                    C.persistence_group_reliability(
                        frame,
                        column,
                        scope=scope,
                        window=label,
                        stage=stage,
                        sample_status=sample_status,
                    )
                )
                for persist_value in sorted(frame["persistence"].unique()):
                    block = frame.loc[frame["persistence"] == persist_value]
                    bins = C.reliability_bin_table(block, column)
                    bins.insert(0, "persistence_group", int(persist_value))
                    bins.insert(0, "stage", stage)
                    bins.insert(0, "window", label)
                    bins.insert(0, "scope", scope)
                    bin_rows.append(bins)

        # -- the PRD R5 gate, on 2020 out-of-sample rows only ---------------
        gate_rows.extend(
            row
            for row in reliability_rows
            if row["scope"] == scope
            and row["window"] == "selection_2020"
            and row["stage"] == "post"
            and row["persistence_group"] != "all"
        )

        # -- group and fallback bookkeeping ---------------------------------
        reports = pd.DataFrame(
            calibrators.group_reports + calibrators.month_pool_reports
        )
        group_rows.append(reports)
        summary = C.fallback_summary(
            calibrators,
            {label: applied[label] for label, _, _ in WINDOWS},
        )
        manifest_scopes.setdefault(str(scope), {}).update(
            {
                "calibrator_file": str(frozen_path),
                "calibrator_sha256": digest,
                "refit_sha256": refit_digest,
                "refit_is_deterministic": refit_digest == digest,
                "reloaded_from_disk_sha256": reload_digest,
                "round_trip_identical": reload_digest == digest,
                "fit_months": list(calibrators.fit_months),
                "fit_rows": int(len(windows["fit_2018_2019"])),
                "fit_key_overlap_with_later_windows": {
                    key: len(value) for key, value in overlaps.items()
                },
                "fallbacks": summary,
            }
        )

    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(output_dir / "reliability_persistence_groups.csv", index=False)
    pd.concat(bin_rows, ignore_index=True).to_csv(
        output_dir / "reliability_bins.csv", index=False
    )
    pd.concat(group_rows, ignore_index=True).to_csv(
        output_dir / "calibration_groups.csv", index=False
    )

    gate = pd.DataFrame(gate_rows)
    gate.to_csv(output_dir / "gate_r5_2020_out_of_sample.csv", index=False)

    after = protected.hash_protected()
    protected.assert_unchanged(before)
    from step3correction.protected import write_hash_report  # noqa: E402

    write_hash_report(output_dir / "protected_hashes.json", before, after)

    manifest = {
        "experiment": "PersistenceCorrectionExperiment",
        "phase": "3 - probability calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_report(),
        "windows": {
            "fit": {"years": list(C.FIT_YEARS), "role": "fit the calibrators"},
            "selection": {
                "years": list(C.SELECTION_YEARS),
                "role": "Phase 4 threshold selection; out-of-sample for the calibrator",
            },
            "test": {
                "years": list(C.TEST_YEARS),
                "role": (
                    "Phase 5 adjudication; emitted calibrated but NOT scored in "
                    "Phase 3 - no metric was computed against its labels"
                ),
            },
            "supersedes": (
                "DECISIONS_LOG C5 replaces PRD R12's cross-fitted training-window "
                "calibration, which the frozen Stage 3 script cannot produce"
            ),
        },
        "calibration": {
            "group_key": ["calendar_month", "partition_id"],
            "why_month_is_in_the_key": (
                "Stage 3 ran with --month-ind, so partition_id indexes 13 (Feb) / "
                "11 (Jun) / 16 (Oct) different partitionings plus a -1 unmapped "
                "sentinel; grouping on partition_id alone would merge them"
            ),
            "primary_method": "isotonic regression (y in [0,1], increasing, clipped)",
            "degenerate_method": "Platt / sigmoid",
            "min_group_rows": C.MIN_GROUP_ROWS,
            "min_distinct_probs": C.MIN_DISTINCT_PROBS,
            "fallback_order": [
                "min_rows -> that calendar month's pooled calibrator",
                "single_class -> Platt attempted, fails, month pool",
                "too_few_distinct_probabilities -> Platt",
                "otherwise -> isotonic",
                "apply-time group absent from the fit window -> month pool",
            ],
            "static_and_frozen": (
                "DECISIONS_LOG C6: fitted once on 2018-2019 and applied unchanged "
                "to 2020 and 2021-2024; the stability of the calibration "
                "relationship from 2019 to 2024 is an assumption, not a finding"
            ),
            "probability_source": C.PROB_COLUMN,
            "reliability_binning_source": (
                "imported from scripts/paper_artifacts/"
                "analyze_georf_probability_uncertainty.py:115-148 (reliability_bins); "
                "read-only, not modified"
            ),
        },
        "gate_r5": {
            "criterion": (
                "|mean calibrated probability - observed crisis rate| < 0.05 in "
                "every persistence group"
            ),
            "evaluated_on": "2020 out-of-sample rows",
            "adjudication": "reported only; Phase 3 does not decide and does not "
            "start Phase 4",
        },
        "availability_assumption_r6_verbatim": P.R6_AVAILABILITY_ASSUMPTION,
        "persistence": {
            "definition": "y_base(T) = 1[fews_ipc(T-H) >= 3]",
            "alignment": P.PERSISTENCE_ALIGNMENT,
            "convention": P.PERSISTENCE_CONVENTION,
            "imputation": "none; a missing origin observation halts the run",
        },
        "protected_inputs_hashed": len(before),
        "protected_inputs_unchanged": before == after,
        "scopes": manifest_scopes,
    }
    (output_dir / "calibration_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # -- console report ----------------------------------------------------
    print(f"output: {output_dir}")
    for scope in args.scopes:
        payload = manifest_scopes[str(scope)]
        summary = payload["fallbacks"]
        print(f"\n=== fs{scope} ===")
        print(
            f"  calibrators: {summary['groups_with_own_calibrator']}"
            f"/{summary['groups_total']} groups own, "
            f"{summary['groups_routed_to_month_pool']} routed to a month pool; "
            f"kinds {summary['calibrator_kind_counts']}"
        )
        print(
            f"  group rows: min {summary['observed_min_group_rows']}, "
            f"median {summary['observed_median_group_rows']} "
            f"(threshold {summary['min_group_rows_threshold']})"
        )
        print(f"  fit-time fallbacks: {summary['fit_fallback_reason_counts'] or 'none'}")
        for label, counts in summary["apply_windows"].items():
            print(
                f"  apply {label}: {counts['rows']} rows, "
                f"{counts['rows_month_pooled']} via a month pool "
                f"-> {counts['rows_by_route']}"
            )
        print(
            f"  frozen sha256 {payload['calibrator_sha256'][:16]}... "
            f"deterministic={payload['refit_is_deterministic']} "
            f"round_trip={payload['round_trip_identical']}"
        )

    print("\n=== reliability by persistence group (PRD R5) ===")
    print(
        reliability.loc[reliability["persistence_group"] != "all"]
        .pivot_table(
            index=["scope", "window", "sample_status", "persistence_group"],
            columns="stage",
            values=["mean_predicted_probability", "observed_crisis_rate", "abs_gap"],
        )
        .round(4)
        .to_string()
    )

    print("\n=== GATE FIGURES: 2020 out-of-sample, post-calibration ===")
    for _, row in gate.iterrows():
        verdict = "within" if row["within_tolerance"] else "OUTSIDE"
        print(
            f"  fs{int(row['scope'])} persist={row['persistence_group']} "
            f"n={int(row['n'])} mean_p_cal={row['mean_predicted_probability']:.4f} "
            f"crisis_rate={row['observed_crisis_rate']:.4f} "
            f"|gap|={row['abs_gap']:.4f} {verdict} 0.05"
        )
    print(
        "\nPhase 3 reports these figures and stops. The R5 stop/go decision and "
        "Phase 4 are explicitly out of scope for this run."
    )
    print(f"protected inputs hashed: {len(before)}; unchanged: {before == after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
