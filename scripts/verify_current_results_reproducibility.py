"""Verify the current no-leak result bundle is organized and reproducible.

This script is intentionally read-only. It checks the live result folders,
intermediate experiment folders, fixed-partition ablation outputs, final paper
artifacts, and local archive pointers without regenerating model outputs.
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper_artifacts.audit_final_artifact_sources import (
    build_audit_rows,
    write_audit_outputs,
)
from scripts.release_paths import ReleasePaths

PATHS = ReleasePaths(repo_root=REPO_ROOT)
SCOPES = {1: 4, 2: 8, 3: 12}

MAIN_RESULTS = {
    "GF": {
        "display": "GeoRF",
        "model_type": "RF",
        "experiment": PATHS.experiment_root("GF"),
        "results_subdir": "GeoRFResults",
        "visual_prefix": "result_GeoRF",
        "expected_general_clusters": 17,
    },
    "DT": {
        "display": "GeoDT",
        "model_type": "DT",
        "experiment": PATHS.experiment_root("DT"),
        "results_subdir": "GeoDTResults",
        "visual_prefix": "result_GeoDT",
        "expected_general_clusters": 15,
    },
}

FEATURE_GROUPS = (
    "weather_exclude",
    "agri_exclude",
    "conflict_exclude",
    "econ_exclude",
    "food_prices_exclude",
    "geographic_exclude",
    "secondary_exclude",
    "lag_exclude",
)

FINAL_ARTIFACTS = (
    "01_main_results/main_month_ind_cont3.xlsx",
    "01_main_results/ablation_feature_exclude.xlsx",
    "01_main_results/georf_monthly_performance.png",
    "01_main_results/global_cluster_map_2x2_georf_refined.png",
    "01_main_results/global_cluster_map_2x2_geodt_refined.png",
    "01_main_results/predictions_2024_feb_jun_oct.png",
    "01_main_results/table1_season_performance.csv",
    "01_main_results/table2_region_performance.csv",
    "01_main_results/table2_region_performance_partitioned_pooled_fewsnet.csv",
    "01_main_results/monthly_performance_manifest.json",
    "02_methods_and_temporal_scope/feature_engineering.png",
    "02_methods_and_temporal_scope/walkthrough.png",
    "03_class_prevalence/fewsnet_crisis_stack_2018.png",
    "04_error_analysis/error_rate_seasonal.csv",
    "04_error_analysis/error_rate_seasonal_crisis.csv",
    "04_error_analysis/error_rate_seasonal_noncrisis.csv",
    "04_error_analysis/error_rate_seasonal_3x3.png",
    "04_error_analysis/error_rate_seasonal_3x3_crisis.png",
    "04_error_analysis/error_rate_seasonal_3x3_noncrisis.png",
    "08_geodt_diagnostics/geodt_branch_1_vs_001_locations_2024-10_fs1_global.png",
    "08_geodt_diagnostics/geodt_branch_tree_compare_2024-10_fs1_001_vs_1.png",
)


class Verifier:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.sections: list[str] = []

    def ok(self, message: str) -> None:
        self.sections.append(f"OK: {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.sections.append(f"WARN: {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        self.sections.append(f"FAIL: {message}")

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.ok(message)
        else:
            self.fail(message)


def repo_path_from_manifest(value: str | Path) -> Path:
    """Resolve Windows, repo-relative, or archived manifest paths."""
    return PATHS.resolve_repo_reference(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def csv_summary(path: Path) -> tuple[list[str], int, set[str], set[str]]:
    """Return header, row count, model values, and test months for a CSV."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        row_count = 0
        models: set[str] = set()
        months: set[str] = set()
        for row in reader:
            row_count += 1
            if row.get("model"):
                models.add(row["model"])
            if row.get("test_month"):
                months.add(row["test_month"])
    return header, row_count, models, months


def audit_rows_are_clean(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Return whether audit rows contain no paper-facing unclean statuses."""
    failures: list[str] = []
    for row in rows:
        status = row.get("status")
        artifact_group = row.get("artifact_group", "<unknown>")
        artifact_type = row.get("artifact_type", "<unknown>")
        if status in {"invalid_phase_change", "needs_regeneration"}:
            failures.append(f"{artifact_group} ({artifact_type}) is {status}: {row.get('reason', '')}")
    return not failures, failures


def verify_main_stage3(verifier: Verifier) -> None:
    for token, config in MAIN_RESULTS.items():
        for scope, lag in SCOPES.items():
            out_dir = PATHS.stage3_root(token, scope)
            verifier.check(out_dir.is_dir(), f"{out_dir.name} exists")
            for filename in (
                "metrics_monthly.csv",
                "metrics_polygon_overall.csv",
                "predictions_monthly.csv",
                "run_manifest.json",
            ):
                verifier.check((out_dir / filename).is_file(), f"{out_dir.name}/{filename} exists")

            manifest_path = out_dir / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = load_json(manifest_path)
            verifier.check(
                manifest.get("partition_learning_years") == "2018-2020",
                f"{out_dir.name} uses 2018-2020 partition learning",
            )
            verifier.check(
                manifest.get("evaluation_years") == "2021-2024",
                f"{out_dir.name} evaluates 2021-2024",
            )
            verifier.check(
                manifest.get("data_path", "").endswith("FEWSNET_forecast_unadjusted_bm.csv"),
                f"{out_dir.name} uses global unadjusted FEWSNET source",
            )
            verifier.check(
                manifest.get("forecasting_scope") == scope
                and manifest.get("active_lag_months") == lag,
                f"{out_dir.name} scope/lag contract is fs{scope}/lag-{lag}",
            )
            verifier.check(
                manifest.get("model_type") == config["model_type"],
                f"{out_dir.name} model type is {config['model_type']}",
            )
            verifier.check(
                int(manifest.get("n_predictions", 0)) > 0,
                f"{out_dir.name} has positive prediction count",
            )

            partition_map = repo_path_from_manifest(manifest.get("partition_map_path", ""))
            verifier.check(
                partition_map.is_file(),
                f"{out_dir.name} refined partition map exists: {partition_map.name}",
            )

            metrics_path = out_dir / "metrics_monthly.csv"
            if metrics_path.exists():
                header, rows, models, months = csv_summary(metrics_path)
                required = {"test_month", "model", "precision", "recall", "f1"}
                verifier.check(required.issubset(header), f"{out_dir.name} metrics schema is complete")
                verifier.check({"partitioned", "pooled"}.issubset(models), f"{out_dir.name} has partitioned and pooled metrics")
                verifier.check(len(months) == 12, f"{out_dir.name} has 12 evaluation months in metrics")
                verifier.check(rows >= 24, f"{out_dir.name} has at least 24 monthly metric rows")


def verify_experiment_dirs(verifier: Verifier) -> None:
    for config in MAIN_RESULTS.values():
        experiment = config["experiment"]
        results_dir = experiment / config["results_subdir"]
        verifier.check(experiment.is_dir(), f"{experiment.name} exists")
        verifier.check(results_dir.is_dir(), f"{experiment.name}/{config['results_subdir']} exists")

        visual_dirs = sorted(results_dir.glob(f"{config['visual_prefix']}_*_fs*_*-*_visual"))
        verifier.check(
            len(visual_dirs) == 27,
            f"{experiment.name} has 27 Stage 1 monthly visual archives",
        )

        knn_dir = experiment / "knn_sparsification_results"
        manifest_path = knn_dir / "cluster_mapping_manifest.json"
        verifier.check(manifest_path.is_file(), f"{experiment.name} cluster mapping manifest exists")
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            for key in ("general", "m02", "m06", "m10"):
                entry = manifest.get(key, {})
                path = repo_path_from_manifest(entry.get("path", ""))
                verifier.check(path.is_file(), f"{experiment.name} {key} cluster map exists")
                verifier.check(
                    int(entry.get("n_clusters", 0)) > 0,
                    f"{experiment.name} {key} cluster count is positive",
                )
            general_clusters = manifest.get("general", {}).get("n_clusters")
            verifier.check(
                general_clusters == config["expected_general_clusters"],
                f"{experiment.name} general cluster count matches Stage 3 ({general_clusters})",
            )


def verify_ablation_outputs(verifier: Verifier) -> None:
    root = PATHS.ablation_root
    verifier.check(root.is_dir(), "fixed-partition feature-exclude ablation root exists")

    dataset_manifest_path = root / "input_datasets" / "feature_exclude_dataset_manifest.json"
    verifier.check(dataset_manifest_path.is_file(), "feature-exclude dataset manifest exists")
    if dataset_manifest_path.exists():
        dataset_manifest = load_json(dataset_manifest_path)
        datasets = dataset_manifest.get("datasets", {})
        verifier.check(set(FEATURE_GROUPS).issubset(datasets), "all feature-exclude datasets are recorded")
        for group in FEATURE_GROUPS:
            entry = datasets.get(group, {})
            dataset_path = repo_path_from_manifest(entry.get("path", root / "input_datasets" / f"{group}.csv"))
            verifier.check(dataset_path.is_file(), f"{group} input dataset exists")

    run_manifest_path = root / "ablation_run_manifest.json"
    verifier.check(run_manifest_path.is_file(), "ablation run manifest exists")
    if run_manifest_path.exists():
        run_manifest = load_json(run_manifest_path)
        records = run_manifest.get("records", [])
        completed = [
            record
            for record in records
            if record.get("status") in {"completed", "skipped_existing"}
        ]
        verifier.check(len(completed) == 24, "ablation run manifest has 24 completed/skipped records")

    for group in FEATURE_GROUPS:
        for scope in SCOPES:
            out_dir = root / group / f"result_partition_k40_compare_GF_fs{scope}"
            verifier.check(out_dir.is_dir(), f"{group} fs{scope} output exists")
            for filename in (
                "metrics_monthly.csv",
                "metrics_polygon_overall.csv",
                "predictions_monthly.csv",
                "run_manifest.json",
            ):
                verifier.check((out_dir / filename).is_file(), f"{group} fs{scope}/{filename} exists")
            metrics_path = out_dir / "metrics_monthly.csv"
            if metrics_path.exists():
                _, _, models, months = csv_summary(metrics_path)
                verifier.check({"partitioned", "pooled"}.issubset(models), f"{group} fs{scope} has both model rows")
                verifier.check(len(months) == 12, f"{group} fs{scope} has 12 evaluation months")


def verify_final_artifacts(verifier: Verifier) -> None:
    old_final = REPO_ROOT / "final_artifacts_in_paper"
    current_final = PATHS.final_artifacts_root
    if old_final.exists():
        verifier.warn("legacy final_artifacts_in_paper/ exists; current manifest uses final_artifacts_in_paper_updated/")
    verifier.check(current_final.is_dir(), "current final_artifacts_in_paper_updated directory exists")
    for filename in FINAL_ARTIFACTS:
        path = current_final / filename
        verifier.check(path.is_file(), f"final artifact exists: {filename}")
        if path.suffix == ".xlsx" and path.exists():
            verifier.check(zipfile.is_zipfile(path), f"final workbook is a valid xlsx zip: {filename}")

    monthly_manifest = current_final / "01_main_results" / "monthly_performance_manifest.json"
    if monthly_manifest.exists():
        manifest = load_json(monthly_manifest)
        for source in manifest.get("source_paths", {}).get("model_metrics", []):
            verifier.check(
                PATHS.resolve_repo_reference(source).is_file(),
                f"monthly performance source exists: {source}",
            )


def verify_artifact_source_audit(verifier: Verifier) -> None:
    """Regenerate source audit reports and fail if paper-facing provenance is unclean."""
    rows = build_audit_rows()
    audit_csv, audit_md = write_audit_outputs(rows, PATHS.final_artifacts_root)
    verifier.check(audit_csv.is_file(), "artifact source audit CSV exists")
    verifier.check(audit_md.is_file(), "artifact source audit Markdown exists")

    ok, failures = audit_rows_are_clean(rows)
    verifier.check(ok, "paper-facing artifact source audit has no phase-change or unresolved regeneration rows")
    for failure in failures:
        verifier.fail(f"artifact source audit failure: {failure}")

    thresholded = [
        row
        for row in rows
        if row.get("artifact_type") == "provider_manifest"
        and "result_partition_k40_compare_GF_thresholded" in row.get("artifact_group", "")
    ]
    verifier.check(len(thresholded) == 3, "all three thresholded GeoRF provider manifests audited")
    for row in thresholded:
        verifier.check(
            str(row.get("source_path", "")).endswith("FEWSNET_forecast_unadjusted_bm.csv"),
            f"{row.get('artifact_group')} uses clean FEWSNET source",
        )


def verify_cleanup_archive(verifier: Verifier) -> None:
    archive = REPO_ROOT / "archived" / "no_leak_partition_learning_2018_2020_20260616"
    verifier.check((archive / "README.md").is_file(), "no-leak root CSV archive README exists")
    csv_count = len(list(archive.glob("*/*.csv")))
    verifier.check(csv_count == 24, "no-leak root CSV archive contains 24 CSV files")

    stray_root_csvs = list(REPO_ROOT.glob("results_df*gp_fs*.csv")) + list(REPO_ROOT.glob("y_pred_test*.csv"))
    verifier.check(not stray_root_csvs, "no root-level results_df/y_pred_test CSV files remain")
    stray_root_visuals = list(REPO_ROOT.glob("result_Geo*_fs*_*-*_visual"))
    verifier.check(not stray_root_visuals, "no root-level Stage 1 visual archives remain")


def main() -> int:
    verifier = Verifier()
    verify_main_stage3(verifier)
    verify_experiment_dirs(verifier)
    verify_ablation_outputs(verifier)
    verify_final_artifacts(verifier)
    verify_artifact_source_audit(verifier)
    verify_cleanup_archive(verifier)

    print("Current results reproducibility verification")
    print("=" * 52)
    for line in verifier.sections:
        print(line)

    if verifier.warnings:
        print("\nWarnings:")
        for warning in verifier.warnings:
            print(f"- {warning}")

    if verifier.failures:
        print("\nFailures:")
        for failure in verifier.failures:
            print(f"- {failure}")
        return 1

    print("\nVerification passed: current result bundle is organized and reproducibility metadata is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
