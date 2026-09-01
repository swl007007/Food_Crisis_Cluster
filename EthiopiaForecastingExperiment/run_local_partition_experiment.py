#!/usr/bin/env python3
"""Run the isolated Ethiopia fs0-fs3 local partition experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SCOPES = {0: 1, 1: 4, 2: 8, 3: 12}
TARGET_MONTHS = tuple(
    f"{year}-{month:02d}"
    for year in range(2021, 2025)
    for month in (2, 6, 10)
)
STAGE1_MONTHS = tuple(sorted({int(target_month[-2:]) for target_month in TARGET_MONTHS}))
EXPECTED_ETH_ROWS = 187_200
EXPECTED_ETH_ADMINS = 1_040
EXPECTED_ETH_MONTHS = 180
COVERAGE_THRESHOLD = 0.90


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_admin_codes(values: Iterable[object]) -> pd.Series:
    """Normalize numeric-looking FEWS NET admin codes without changing membership."""
    return pd.Series(values, dtype="string").str.strip().str.replace(r"\.0$", "", regex=True)


def filter_ethiopia_panel(
    panel: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_admins: int | None = None,
    expected_months: int | None = None,
) -> pd.DataFrame:
    """Create and validate the exact ``ISO3 == 'ETH'`` panel cohort."""
    required = {"ISO3", "FEWSNET_admin_code", "date"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")

    eth = panel.loc[panel["ISO3"].eq("ETH")].copy()
    if eth.empty:
        raise ValueError("Exact ISO3 == 'ETH' filter returned no rows")
    if eth["FEWSNET_admin_code"].isna().any():
        raise ValueError("Null Ethiopia admin code")

    eth["date"] = pd.to_datetime(eth["date"]).dt.to_period("M").dt.to_timestamp()
    if eth.duplicated(["FEWSNET_admin_code", "date"]).any():
        raise ValueError("Duplicate Ethiopia admin-month key")
    eth = eth.sort_values(["FEWSNET_admin_code", "date"]).reset_index(drop=True)

    observed = {
        "rows": len(eth),
        "admins": eth["FEWSNET_admin_code"].nunique(),
        "months": eth["date"].dt.to_period("M").nunique(),
    }
    expected = {
        "rows": expected_rows,
        "admins": expected_admins,
        "months": expected_months,
    }
    for name, value in expected.items():
        if value is not None and observed[name] != value:
            raise ValueError(f"Unexpected Ethiopia {name}: {observed[name]} != {value}")
    return eth


def _phase3_binary(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(numeric.notna(), (numeric >= 3).astype(float), np.nan), index=values.index)


def _filter_fewsnet_ethiopia(fewsnet: pd.DataFrame) -> pd.DataFrame:
    required = {
        "country",
        "admin_code",
        "year_month",
        "fews_ipc",
        "fews_proj_near",
        "fews_proj_med",
    }
    missing = required - set(fewsnet.columns)
    if missing:
        raise ValueError(f"FEWS NET input missing required columns: {sorted(missing)}")
    eth = fewsnet.loc[fewsnet["country"].eq("Ethiopia")].copy()
    eth["FEWSNET_admin_code"] = normalize_admin_codes(eth["admin_code"]).to_numpy()
    year_month = eth["year_month"].astype(str).str.replace("_", "-", regex=False)
    eth["period"] = pd.PeriodIndex(year_month, freq="M")
    if eth["FEWSNET_admin_code"].isna().any():
        raise ValueError("Null Ethiopia FEWS NET admin code")
    if eth.duplicated(["FEWSNET_admin_code", "period"]).any():
        raise ValueError("Duplicate Ethiopia FEWS NET admin-month key")
    return eth


def calendar_join_fewsnet(
    fewsnet: pd.DataFrame,
    *,
    target_month: str,
    scope: int,
    cohort_codes: Sequence[object],
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Join native FEWS NET projections by explicit target and source months."""
    if scope not in (1, 2):
        raise ValueError("FEWS NET baseline is available only for fs1 and fs2")
    lag_months, projection_column = {
        1: (4, "fews_proj_near"),
        2: (8, "fews_proj_med"),
    }[scope]
    target = pd.Period(target_month, freq="M")
    source = target - lag_months
    eth = _filter_fewsnet_ethiopia(fewsnet)

    codes = normalize_admin_codes(cohort_codes).drop_duplicates().sort_values()
    base = pd.DataFrame({"FEWSNET_admin_code": codes.to_numpy()})
    actual = eth.loc[
        eth["period"].eq(target),
        ["FEWSNET_admin_code", "fews_ipc"],
    ].rename(columns={"fews_ipc": "actual_phase"})
    projection = eth.loc[
        eth["period"].eq(source),
        ["FEWSNET_admin_code", projection_column],
    ].rename(columns={projection_column: "projection_phase"})
    joined = base.merge(actual, on="FEWSNET_admin_code", how="left").merge(
        projection,
        on="FEWSNET_admin_code",
        how="left",
    )
    joined["y_true_fewsnet"] = _phase3_binary(joined["actual_phase"])
    joined["y_pred_fewsnet"] = _phase3_binary(joined["projection_phase"])
    joined["test_month"] = str(target)
    available = int(joined["projection_phase"].notna().sum())
    total = int(len(joined))
    coverage = {
        "available": available,
        "total": total,
        "fraction": float(available / total) if total else 0.0,
    }
    return joined, coverage


def binary_metrics(y_true: Sequence[object], y_pred: Sequence[object]) -> dict[str, int | float]:
    """Compute class-1 precision, recall, and F1 with confusion counts."""
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    true = frame["y_true"].astype(int).to_numpy()
    pred = frame["y_pred"].astype(int).to_numpy()
    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n": int(len(frame)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _metric_row(
    *,
    scope: int,
    lag_months: int,
    test_month: str,
    model: str,
    metrics: dict[str, int | float] | None,
    coverage: dict[str, int | float],
    status: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scope": f"fs{scope}",
        "lag_months": lag_months,
        "test_month": test_month,
        "model": model,
        "coverage_n": coverage["available"],
        "coverage_total": coverage["total"],
        "coverage": coverage["fraction"],
        "status": status,
    }
    row.update(metrics or {key: np.nan for key in ("precision", "recall", "f1", "n", "tp", "fp", "fn", "tn")})
    return row


def evaluate_scope_month(
    predictions: pd.DataFrame,
    *,
    scope: int,
    lag_months: int,
    fewsnet_eth: pd.DataFrame | None,
    cohort_codes: Sequence[object],
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Recompute final and fixed-0.5 monthly metrics on the required support."""
    if predictions["month_start"].nunique() != 1:
        raise ValueError("Predictions must contain exactly one target month")
    frame = predictions.copy()
    frame["FEWSNET_admin_code"] = normalize_admin_codes(frame["FEWSNET_admin_code"]).to_numpy()
    test_month = str(pd.Period(pd.to_datetime(frame["month_start"].iloc[0]), freq="M"))
    final_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []

    if scope in (1, 2):
        if fewsnet_eth is None:
            raise ValueError("FEWS NET input is required for fs1/fs2")
        baseline, coverage = calendar_join_fewsnet(
            fewsnet_eth,
            target_month=test_month,
            scope=scope,
            cohort_codes=cohort_codes,
        )
        frame = frame.merge(
            baseline[["FEWSNET_admin_code", "y_true_fewsnet", "y_pred_fewsnet"]],
            on="FEWSNET_admin_code",
            how="left",
            validate="one_to_one",
        )
        common = frame[["y_true", "y_true_fewsnet", "y_pred_fewsnet"]].notna().all(axis=1)
        supported = frame.loc[common].copy()
        if not supported.empty and not np.array_equal(
            supported["y_true"].astype(int).to_numpy(),
            supported["y_true_fewsnet"].astype(int).to_numpy(),
        ):
            raise ValueError("Model and FEWS NET target labels disagree on common keys")
        model_status = "common_support"
    else:
        coverage = {"available": 0, "total": len(set(normalize_admin_codes(cohort_codes))), "fraction": 0.0}
        supported = frame.loc[frame["y_true"].notna()].copy()
        model_status = "model_support"

    for model, column in (
        ("pooled", "y_pred_pooled_thresholded"),
        ("partitioned", "y_pred_partitioned_thresholded"),
    ):
        final_rows.append(
            _metric_row(
                scope=scope,
                lag_months=lag_months,
                test_month=test_month,
                model=model,
                metrics=binary_metrics(supported["y_true"], supported[column]),
                coverage=coverage,
                status=model_status,
            )
        )
    for model, column in (
        ("pooled", "y_pred_pooled"),
        ("partitioned", "y_pred_partitioned"),
    ):
        fixed_rows.append(
            _metric_row(
                scope=scope,
                lag_months=lag_months,
                test_month=test_month,
                model=model,
                metrics=binary_metrics(supported["y_true"], supported[column]),
                coverage=coverage,
                status=model_status,
            )
        )

    if scope in (1, 2):
        if coverage["fraction"] < coverage_threshold:
            fews_metrics = None
            status = "suppressed_low_coverage"
        elif supported.empty:
            fews_metrics = None
            status = "unavailable_no_common_support"
        else:
            fews_metrics = binary_metrics(supported["y_true_fewsnet"], supported["y_pred_fewsnet"])
            status = "available"
    else:
        fews_metrics = None
        status = "unavailable_for_scope"
    final_rows.append(
        _metric_row(
            scope=scope,
            lag_months=lag_months,
            test_month=test_month,
            model="fewsnet",
            metrics=fews_metrics,
            coverage=coverage,
            status=status,
        )
    )
    return final_rows, fixed_rows


def create_run_directory(output_root: Path, run_id: str) -> Path:
    """Create a new isolated run directory and refuse overwrites."""
    repo_root = Path(__file__).resolve().parents[1]
    approved_root = (repo_root / "EthiopiaForecastingExperiment" / "outputs" / "local_partition_experiment").resolve()
    output_root = output_root.resolve()
    if output_root.is_relative_to(repo_root) and not output_root.is_relative_to(approved_root):
        raise ValueError(f"Output path must stay under the approved experiment output root: {approved_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir()
    return run_dir


def build_stage1_command(
    *,
    python_executable: Path,
    repo_root: Path,
    panel_path: Path,
    scope: int,
    year: int,
    month: int,
) -> list[str]:
    """Build one isolated Stage 1 scope-month command."""
    return [
        str(python_executable),
        "-B",
        str(repo_root / "app" / "main_model_GF.py"),
        "--start_year",
        str(year),
        "--end_year",
        str(year),
        "--forecasting_scope",
        str(scope),
        "--desired_terms",
        f"{year}-{month:02d}",
        "--data",
        str(panel_path),
        "--strict-lag-only",
        "--random-seed",
        "5",
    ]


def build_stage2_commands(
    *,
    python_executable: Path,
    repo_root: Path,
    stage2_dir: Path,
) -> list[list[str]]:
    """Build the shared general/m2/m6/m10 consensus command sequence."""
    scripts = repo_root / "scripts"
    base = [str(python_executable), "-B"]
    commands = [
        base
        + [
            str(scripts / "step1_merge_results.py"),
            "--experiment-dir",
            str(stage2_dir),
            "--model-type",
            "georf",
        ],
        base
        + [
            str(scripts / "step3_create_linked_tables.py"),
            "--experiment-dir",
            str(stage2_dir),
        ],
    ]
    for month, suffix in ((None, "general"), (2, "m2"), (6, "m6"), (10, "m10")):
        similarity_dir = "similarity_matrices" if month is None else f"similarity_matrices_m{month:02d}"
        step4 = base + [
            str(scripts / "step4_similarity_matrix.py"),
            "--experiment-dir",
            str(stage2_dir),
        ]
        if month is not None:
            step4 += ["--month", str(month)]
        commands.extend(
            [
                step4,
                base
                + [
                    str(scripts / "step5_sparsification.py"),
                    "--experiment-dir",
                    str(stage2_dir),
                    "--similarity-dir",
                    similarity_dir,
                    "--suffix",
                    suffix,
                ],
                base
                + [
                    str(scripts / "step6_complete_clustering_pipeline.py"),
                    "--experiment-dir",
                    str(stage2_dir),
                    "--similarity-dir",
                    similarity_dir,
                    "--suffix",
                    suffix,
                ],
            ]
        )
    return commands


def collect_stage1_outputs(
    *,
    cell_dir: Path,
    stage2_results_dir: Path,
    scope: int,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Collect the minimal Stage 1 artifacts consumed by Stage 2."""
    metrics_path = cell_dir / f"results_df_gp_fs{scope}_{year}_{year}.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    selected = metrics.loc[
        metrics["year"].astype(int).eq(year)
        & metrics["month"].astype(int).eq(month)
    ].copy()
    if len(selected) != 1:
        raise ValueError(
            f"Expected one Stage 1 metrics row for fs{scope} {year}-{month:02d}, got {len(selected)}"
        )

    correspondence_name = f"correspondence_table_{year}-{month:02d}.csv"
    candidates = list(cell_dir.rglob(correspondence_name))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {correspondence_name} under {cell_dir}, got {len(candidates)}"
        )
    archive = stage2_results_dir / f"result_GeoRF_{year}_fs{scope}_{year}-{month:02d}_visual"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(candidates[0], archive / correspondence_name)
    return selected.reset_index(drop=True)


def build_stage3_command(
    *,
    python_executable: Path,
    repo_root: Path,
    panel_path: Path,
    partition_map: Path,
    out_dir: Path,
    scope: int,
    target_month: str,
) -> list[str]:
    """Build one matched Stage 3 fold command."""
    return [
        str(python_executable),
        "-B",
        str(repo_root / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py"),
        "--data",
        str(panel_path),
        "--partition-map",
        str(partition_map),
        "--out-dir",
        str(out_dir),
        "--start-month",
        target_month,
        "--end-month",
        target_month,
        "--train-window",
        "36",
        "--forecasting-scope",
        str(scope),
        "--strict-lag-only",
        "--enable-symmetric-validation-threshold",
        "--threshold-validation-months",
        "6",
    ]


def plot_monthly_metrics(metrics: pd.DataFrame, output_path: Path) -> int:
    """Render the confirmed 4x3 scope-by-metric performance figure."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True, sharey="col")
    styles = {
        "pooled": {"label": "Pooled", "color": "#4C78A8", "marker": "o", "linestyle": "--"},
        "partitioned": {"label": "Partitioned", "color": "#F58518", "marker": "s", "linestyle": "-"},
        "fewsnet": {"label": "FEWS NET", "color": "#54A24B", "marker": "^", "linestyle": ":"},
    }
    metric_names = ("precision", "recall", "f1")
    for row_index, scope in enumerate(("fs0", "fs1", "fs2", "fs3")):
        scope_rows = metrics.loc[metrics["scope"].eq(scope)].copy()
        scope_rows["test_month"] = scope_rows["test_month"].astype(str)
        for column_index, metric in enumerate(metric_names):
            axis = axes[row_index, column_index]
            for model, style in styles.items():
                series = scope_rows.loc[scope_rows["model"].eq(model)].sort_values("test_month")
                if model == "fewsnet":
                    series = series.loc[series["status"].eq("available")]
                if series.empty:
                    continue
                axis.plot(series["test_month"], series[metric], **style)
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.25)
            axis.set_title(f"{scope} - {metric.capitalize()}")
            if column_index == 0:
                axis.set_ylabel("Score")
            if row_index == 3:
                axis.set_xlabel("Target month")
                axis.tick_params(axis="x", rotation=45)
            if scope in ("fs0", "fs3"):
                axis.text(
                    0.02,
                    0.04,
                    "FEWS NET unavailable",
                    transform=axis.transAxes,
                    fontsize=9,
                    color="dimgray",
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Ethiopia local partition experiment: monthly class-1 performance", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return int(axes.size)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
) -> None:
    """Run one pipeline command and keep its full output in a run-local log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {' '.join(command)}\n\n")
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )


def load_partition_maps(
    stage2_dir: Path,
    cohort_codes: Sequence[object],
) -> dict[str, Path]:
    """Load and validate the four shared Stage 2 mapping providers."""
    manifest_path = stage2_dir / "knn_sparsification_results" / "cluster_mapping_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_codes = set(normalize_admin_codes(cohort_codes).tolist())
    mapping_paths: dict[str, Path] = {}
    for key in ("general", "m02", "m06", "m10"):
        entry = manifest.get(key)
        if not entry or not entry.get("path"):
            raise ValueError(f"Stage 2 manifest missing mapping: {key}")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = stage2_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        mapping = pd.read_csv(path)
        observed_codes = set(normalize_admin_codes(mapping["FEWSNET_admin_code"]).tolist())
        if observed_codes != expected_codes:
            raise ValueError(
                f"Stage 2 {key} mapping cohort mismatch: {len(observed_codes)} != {len(expected_codes)}"
            )
        mapping_paths[key] = path
    return mapping_paths


def run_experiment(args: argparse.Namespace) -> Path:
    """Execute the approved Ethiopia-only Stage 1-3 experiment."""
    repo_root = Path(__file__).resolve().parents[1]
    python_executable = Path(sys.executable)
    panel_path = args.panel.resolve()
    fewsnet_path = args.fewsnet.resolve()
    for source in (panel_path, fewsnet_path):
        if not source.is_file():
            raise FileNotFoundError(source)

    run_dir = create_run_directory(args.output_root.resolve(), args.run_id)
    input_dir = run_dir / "input"
    manifest_dir = run_dir / "manifests"
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    stage2_results = stage2_dir / "GeoRFResults"
    stage3_dir = run_dir / "stage3"
    for path in (input_dir, manifest_dir, stage1_dir, stage2_results, stage3_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_hashes_before = {
        "panel": sha256_file(panel_path),
        "fewsnet": sha256_file(fewsnet_path),
    }
    print("Loading and filtering authoritative Ethiopia cohort...", flush=True)
    panel = pd.read_csv(panel_path, low_memory=False)
    eth_panel = filter_ethiopia_panel(
        panel,
        expected_rows=EXPECTED_ETH_ROWS,
        expected_admins=EXPECTED_ETH_ADMINS,
        expected_months=EXPECTED_ETH_MONTHS,
    )
    del panel
    eth_panel_path = input_dir / "ethiopia_panel.csv"
    eth_panel.to_csv(eth_panel_path, index=False)
    cohort_codes = normalize_admin_codes(eth_panel["FEWSNET_admin_code"]).drop_duplicates().sort_values()

    fewsnet = pd.read_csv(fewsnet_path, low_memory=False)
    fewsnet_eth = _filter_fewsnet_ethiopia(fewsnet)
    del fewsnet
    if len(fewsnet_eth) != 55_120 or fewsnet_eth["FEWSNET_admin_code"].nunique() != EXPECTED_ETH_ADMINS:
        raise ValueError(
            f"Unexpected Ethiopia FEWS NET slice: rows={len(fewsnet_eth)}, "
            f"admins={fewsnet_eth['FEWSNET_admin_code'].nunique()}"
        )
    if set(fewsnet_eth["FEWSNET_admin_code"]) != set(cohort_codes):
        raise ValueError("Authoritative panel and FEWS NET Ethiopia admin-code sets differ")

    cohort_manifest = {
        "panel_source_path": str(panel_path),
        "panel_source_sha256": source_hashes_before["panel"],
        "fewsnet_source_path": str(fewsnet_path),
        "fewsnet_source_sha256": source_hashes_before["fewsnet"],
        "filter": "ISO3 == 'ETH'",
        "fewsnet_filter": "country == 'Ethiopia'",
        "canonical_key": "FEWSNET_admin_code",
        "row_count": len(eth_panel),
        "admin_count": eth_panel["FEWSNET_admin_code"].nunique(),
        "month_count": eth_panel["date"].dt.to_period("M").nunique(),
        "date_min": str(eth_panel["date"].min().date()),
        "date_max": str(eth_panel["date"].max().date()),
        "duplicate_admin_month_keys": 0,
        "null_admin_keys": 0,
        "fewsnet_row_count": len(fewsnet_eth),
        "fewsnet_admin_count": fewsnet_eth["FEWSNET_admin_code"].nunique(),
        "admin_code_sets_equal": True,
        "filtered_panel_path": str(eth_panel_path),
        "filtered_panel_sha256": sha256_file(eth_panel_path),
    }
    write_json(manifest_dir / "cohort.json", cohort_manifest)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "5"
    env["NO_LEAK_PARTITION_LEARNING_YEARS"] = "2018-2020"
    env["NO_LEAK_EVALUATION_YEARS"] = "2021-2024"

    partition_learning_years = range(2018, 2021)
    expected_stage1_plans = len(SCOPES) * len(partition_learning_years) * len(STAGE1_MONTHS)
    plan_rows: list[dict[str, object]] = []
    print(
        f"Starting Stage 1: {expected_stage1_plans} Ethiopia partition-learning plans...",
        flush=True,
    )
    for scope, lag_months in SCOPES.items():
        for year in partition_learning_years:
            yearly_metrics = []
            for month in STAGE1_MONTHS:
                cell_dir = stage1_dir / "work" / f"fs{scope}" / f"{year}-{month:02d}"
                cell_dir.mkdir(parents=True)
                log_path = stage1_dir / "logs" / f"fs{scope}_{year}-{month:02d}.log"
                command = build_stage1_command(
                    python_executable=python_executable,
                    repo_root=repo_root,
                    panel_path=eth_panel_path,
                    scope=scope,
                    year=year,
                    month=month,
                )
                run_logged(command, cwd=cell_dir, log_path=log_path, env=env)
                row = collect_stage1_outputs(
                    cell_dir=cell_dir,
                    stage2_results_dir=stage2_results,
                    scope=scope,
                    year=year,
                    month=month,
                )
                correspondence_path = (
                    stage2_results
                    / f"result_GeoRF_{year}_fs{scope}_{year}-{month:02d}_visual"
                    / f"correspondence_table_{year}-{month:02d}.csv"
                )
                correspondence = pd.read_csv(correspondence_path, dtype={"partition_id": "string"})
                if set(normalize_admin_codes(correspondence["FEWSNET_admin_code"])) != set(cohort_codes):
                    raise ValueError(f"Stage 1 cohort mismatch for fs{scope} {year}-{month:02d}")
                yearly_metrics.append(row)
                plan_rows.append(
                    {
                        "scope": f"fs{scope}",
                        "lag_months": lag_months,
                        "year": year,
                        "month": month,
                        "metrics_log": str(log_path),
                        "correspondence_path": str(correspondence_path),
                        "correspondence_sha256": sha256_file(correspondence_path),
                    }
                )
                shutil.rmtree(cell_dir)
                print(
                    f"Stage 1 complete: fs{scope} {year}-{month:02d} "
                    f"({len(plan_rows)}/{expected_stage1_plans})",
                    flush=True,
                )
            yearly = pd.concat(yearly_metrics, ignore_index=True).sort_values(["year", "month"])
            if len(yearly) != len(STAGE1_MONTHS):
                raise ValueError(
                    f"Stage 1 expected {len(STAGE1_MONTHS)} rows for fs{scope} {year}, "
                    f"got {len(yearly)}"
                )
            yearly.to_csv(stage2_results / f"results_df_gp_fs{scope}_{year}_{year}.csv", index=False)
    plans = pd.DataFrame(plan_rows)
    if len(plans) != expected_stage1_plans:
        raise ValueError(
            f"Stage 1 plan count mismatch: {len(plans)} != {expected_stage1_plans}"
        )
    plans.to_csv(stage1_dir / "plan_index.csv", index=False)

    print("Starting Stage 2 shared consensus and stabilization...", flush=True)
    for index, command in enumerate(
        build_stage2_commands(
            python_executable=python_executable,
            repo_root=repo_root,
            stage2_dir=stage2_dir,
        ),
        start=1,
    ):
        run_logged(
            command,
            cwd=repo_root,
            log_path=stage2_dir / "logs" / f"{index:02d}_{Path(command[2]).stem}.log",
            env=env,
        )
    mapping_paths = load_partition_maps(stage2_dir, cohort_codes)

    print("Starting Stage 3: 48 matched validation folds...", flush=True)
    all_thresholds = []
    predictions_by_scope: dict[int, pd.DataFrame] = {}
    fold_count = 0
    for scope, lag_months in SCOPES.items():
        scope_dir = stage3_dir / f"fs{scope}"
        scope_dir.mkdir(parents=True, exist_ok=True)
        scope_predictions = []
        scope_thresholds = []
        for target_month in TARGET_MONTHS:
            month_number = int(target_month[-2:])
            mapping_key = {2: "m02", 6: "m06", 10: "m10"}[month_number]
            fold_dir = stage3_dir / "work" / f"fs{scope}" / target_month
            command = build_stage3_command(
                python_executable=python_executable,
                repo_root=repo_root,
                panel_path=eth_panel_path,
                partition_map=mapping_paths[mapping_key],
                out_dir=fold_dir,
                scope=scope,
                target_month=target_month,
            )
            log_path = scope_dir / "logs" / f"{target_month}.log"
            run_logged(command, cwd=repo_root, log_path=log_path, env=env)
            predictions = pd.read_csv(fold_dir / "predictions_monthly.csv")
            if predictions["month_start"].astype(str).str[:7].nunique() != 1:
                raise ValueError(f"Stage 3 fold produced multiple months: fs{scope} {target_month}")
            scope_predictions.append(predictions)
            thresholds = pd.read_csv(fold_dir / "thresholds_by_fold.csv")
            thresholds.insert(0, "scope", f"fs{scope}")
            thresholds.insert(1, "lag_months", lag_months)
            scope_thresholds.append(thresholds)
            shutil.copy2(fold_dir / "run_manifest.json", scope_dir / f"run_manifest_{target_month}.json")
            shutil.rmtree(fold_dir)
            fold_count += 1
            print(f"Stage 3 complete: fs{scope} {target_month} ({fold_count}/48)", flush=True)
        predictions_scope = pd.concat(scope_predictions, ignore_index=True)
        if predictions_scope.duplicated(["FEWSNET_admin_code", "month_start"]).any():
            raise ValueError(f"Duplicate Stage 3 prediction keys for fs{scope}")
        predictions_scope.to_csv(scope_dir / "predictions_monthly.csv", index=False)
        thresholds_scope = pd.concat(scope_thresholds, ignore_index=True)
        thresholds_scope.to_csv(scope_dir / "thresholds_by_fold.csv", index=False)
        predictions_by_scope[scope] = predictions_scope
        all_thresholds.append(thresholds_scope)

    metric_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    for scope, lag_months in SCOPES.items():
        predictions = predictions_by_scope[scope]
        for target_month, month_rows in predictions.groupby(predictions["month_start"].astype(str).str[:7]):
            final, fixed = evaluate_scope_month(
                month_rows,
                scope=scope,
                lag_months=lag_months,
                fewsnet_eth=fewsnet_eth,
                cohort_codes=cohort_codes,
                coverage_threshold=COVERAGE_THRESHOLD,
            )
            metric_rows.extend(final)
            fixed_rows.extend(fixed)
    metrics = pd.DataFrame(metric_rows).sort_values(["scope", "test_month", "model"])
    fixed_metrics = pd.DataFrame(fixed_rows).sort_values(["scope", "test_month", "model"])
    if len(metrics) != 144 or metrics.duplicated(["scope", "test_month", "model"]).any():
        raise ValueError("Final monthly metrics key/count contract failed")
    if len(fixed_metrics) != 96 or fixed_metrics.duplicated(["scope", "test_month", "model"]).any():
        raise ValueError("Fixed-0.5 monthly metrics key/count contract failed")
    metrics.to_csv(run_dir / "metrics_monthly.csv", index=False)
    fixed_metrics.to_csv(run_dir / "metrics_fixed_05.csv", index=False)
    pd.concat(all_thresholds, ignore_index=True).to_csv(run_dir / "thresholds_by_fold.csv", index=False)
    figure_path = run_dir / "ethiopia_monthly_performance.png"
    panel_count = plot_monthly_metrics(metrics, figure_path)

    source_hashes_after = {
        "panel": sha256_file(panel_path),
        "fewsnet": sha256_file(fewsnet_path),
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("Source file hash changed during experiment")
    code_paths = [
        repo_root / "app" / "main_model_GF.py",
        repo_root / "src" / "feature" / "strict_lag.py",
        repo_root / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py",
        repo_root / "scripts" / "step1_merge_results.py",
        repo_root / "scripts" / "step3_create_linked_tables.py",
        repo_root / "scripts" / "step4_similarity_matrix.py",
        repo_root / "scripts" / "step5_sparsification.py",
        repo_root / "scripts" / "step6_complete_clustering_pipeline.py",
        Path(__file__).resolve(),
    ]
    run_manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "cohort_manifest": str(manifest_dir / "cohort.json"),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "code_hashes": {str(path): sha256_file(path) for path in code_paths},
        "scopes": {f"fs{scope}": lag for scope, lag in SCOPES.items()},
        "target_months": list(TARGET_MONTHS),
        "partition_learning_years": [2018, 2019, 2020],
        "stage1_plan_count": len(plans),
        "stage3_fold_count": fold_count,
        "partition_maps": {key: str(path) for key, path in mapping_paths.items()},
        "partition_map_hashes": {key: sha256_file(path) for key, path in mapping_paths.items()},
        "georf_smote_seed": 5,
        "spectral_seed": 42,
        "strict_lag_only": True,
        "threshold_policy": {
            "mode": "symmetric_validation_only",
            "validation_months": 6,
            "metric": "class_1_f1",
            "test_labels_used": False,
            "fixed_0_5_role": "diagnostic_only",
        },
        "fewsnet_policy": {
            "fs1": "fews_proj_near(T-4) vs fews_ipc(T)",
            "fs2": "fews_proj_med(T-8) vs fews_ipc(T)",
            "fs0": "unavailable",
            "fs3": "unavailable",
            "coverage_threshold": COVERAGE_THRESHOLD,
            "common_admin_month_support": True,
        },
        "metrics_rows": len(metrics),
        "fixed_metrics_rows": len(fixed_metrics),
        "figure": str(figure_path),
        "figure_panels": panel_count,
        "fewsnet_status_counts": metrics.loc[metrics["model"].eq("fewsnet"), "status"].value_counts().to_dict(),
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    return run_dir


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root.parents[2] / "1.Source Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=source_root / "FEWSNET_forecast_unadjusted_bm.csv")
    parser.add_argument(
        "--fewsnet",
        type=Path,
        default=source_root / "Outcome" / "FEWSNET_IPC" / "FEWSNET.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "local_partition_experiment",
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%dT%H%M%S"))
    return parser.parse_args()


if __name__ == "__main__":
    completed_run = run_experiment(parse_args())
    print(f"Experiment complete: {completed_run}")
