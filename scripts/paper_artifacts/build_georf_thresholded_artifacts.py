#!/usr/bin/env python3
"""Build paper-facing artifacts for GeoRF validation-selected thresholding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from paper_horizon_labels import HORIZON_LABELS
except ModuleNotFoundError:
    from scripts.paper_artifacts.paper_horizon_labels import HORIZON_LABELS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "12_thresholded_georf_results"
HORIZONS = HORIZON_LABELS


def load_thresholded_results(source_dir: Path, scopes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frames = []
    threshold_frames = []
    for scope in scopes:
        result_dir = source_dir / f"result_partition_k40_compare_GF_thresholded_{scope}"
        metrics = pd.read_csv(result_dir / "metrics_monthly.csv")
        thresholds = pd.read_csv(result_dir / "threshold_provenance.csv")
        metrics["scope"] = scope
        metrics["forecasting_horizon"] = HORIZONS[scope]
        thresholds["scope"] = scope
        thresholds["forecasting_horizon"] = HORIZONS[scope]
        metrics_frames.append(metrics)
        threshold_frames.append(thresholds)
    return pd.concat(metrics_frames, ignore_index=True), pd.concat(threshold_frames, ignore_index=True)


def load_provider_manifests(source_dir: Path, scopes: list[str]) -> dict[str, dict]:
    """Return selected provenance fields from thresholded provider manifests."""
    details: dict[str, dict] = {}
    for scope in scopes:
        result_dir = source_dir / f"result_partition_k40_compare_GF_thresholded_{scope}"
        manifest_path = result_dir / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        details[scope] = {
            "data_path": str(manifest.get("data_path", "")),
            "month_ind_enabled": bool(manifest.get("month_ind_enabled", False)),
            "partition_map_path": manifest.get("partition_map_path"),
            "partition_map_m2_path": manifest.get("partition_map_m2_path"),
            "partition_map_m6_path": manifest.get("partition_map_m6_path"),
            "partition_map_m10_path": manifest.get("partition_map_m10_path"),
            "partition_map_hashes": manifest.get("partition_map_hashes", {}),
            "smote_available": manifest.get("smote_available"),
            "imblearn_version": manifest.get("imblearn_version"),
            "python_executable": manifest.get("python_executable"),
        }
    return details


def _aggregate_model_metrics(group: pd.DataFrame) -> pd.Series:
    tp = group["tp"].sum()
    fp = group["fp"].sum()
    fn = group["fn"].sum()
    tn = group["tn"].sum()
    return pd.Series(
        {
            "support": int(group["n"].sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": group["precision"].mean(),
            "recall": group["recall"].mean(),
            "f1": group["f1"].mean(),
        }
    )


def build_horizon_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["scope", "forecasting_horizon", "model"]
    for keys, group in metrics.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row.update(_aggregate_model_metrics(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def build_compact_table(horizon_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = ["precision", "recall", "f1"]
    for (scope, horizon), sub in horizon_metrics.groupby(["scope", "forecasting_horizon"], sort=True):
        row = {"scope": scope, "forecasting_horizon": horizon}
        by_model = {model: frame.iloc[0] for model, frame in sub.groupby("model")}
        for model in ["pooled", "partitioned", "partitioned_thresholded"]:
            model_row = by_model.get(model)
            for metric in metric_names:
                row[f"{model}_{metric}"] = model_row[metric] if model_row is not None else float("nan")
        for metric in metric_names:
            row[f"delta_thresholded_minus_partitioned_{metric}"] = (
                row[f"partitioned_thresholded_{metric}"] - row[f"partitioned_{metric}"]
            )
            row[f"delta_thresholded_minus_pooled_{metric}"] = (
                row[f"partitioned_thresholded_{metric}"] - row[f"pooled_{metric}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_for_markdown(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in formatted.columns:
        if column not in {"scope", "forecasting_horizon"}:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return formatted


def write_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_note(output_dir: Path) -> None:
    (output_dir / "georf_thresholded_note.md").write_text(
        "\n".join(
            [
                "# GeoRF Validation-Selected Max-F1 Thresholding",
                "",
                "中文审查说明：",
                "",
                "该 appendix 只针对 GeoRF partitioned/local RF 模型的 thresholded diagnostic。",
                "Threshold 在每个 rolling training window 的 validation subset 上选择，目标是最大化 class-1 F1。",
                "选出的 threshold 只应用于随后 held-out target month 的 test probabilities；没有使用 test labels 选择 threshold。",
                "Horizon-level precision、recall 和 F1 使用 target-month macro mean，与 01_main_results 的主表口径一致。",
                "原始 pooled 和 partitioned hard-prediction 结果保留，用于对照。",
                "这些结果先写入独立 artifact folder，尚不覆盖主文 01-11 artifacts。",
                "",
                "Appendix text (English):",
                "",
                "We evaluate a validation-selected probability threshold for the GeoRF partitioned RF model.",
                "For each forecasting horizon and target month, the threshold is selected on a validation subset from the rolling training window by maximizing class-1 F1, then applied to the held-out target-month probabilities.",
                "The procedure does not use test labels for threshold selection.",
                "Horizon-level precision, recall, and F1 are reported as target-month macro means, matching the aggregation convention used in the main results table.",
                "Pooled and original partitioned hard-prediction results are retained as comparators.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, thresholds = load_thresholded_results(args.source_dir, args.scopes)
    horizon_metrics = build_horizon_metrics(metrics)
    compact = build_compact_table(horizon_metrics)

    horizon_metrics.to_csv(args.output_dir / "georf_thresholded_horizon_metrics.csv", index=False)
    metrics.to_csv(args.output_dir / "georf_thresholded_monthly_metrics.csv", index=False)
    thresholds.to_csv(args.output_dir / "georf_thresholded_threshold_provenance.csv", index=False)
    compact.to_csv(args.output_dir / "georf_thresholded_compact_table.csv", index=False)
    write_markdown_table(format_for_markdown(compact), args.output_dir / "georf_thresholded_compact_table.md")
    write_note(args.output_dir)
    provider_details = load_provider_manifests(args.source_dir, args.scopes)
    provider_sources = {
        scope: str(details.get("data_path", ""))
        for scope, details in provider_details.items()
    }
    (args.output_dir / "artifact_source_manifest.json").write_text(
        json.dumps(
            {
                "artifact_group": "12_thresholded_georf_results",
                "source_dir": str(args.source_dir),
                "scopes": list(args.scopes),
                "source_data_paths": provider_sources,
                "provider_manifests": {
                    scope: str(args.source_dir / f"result_partition_k40_compare_GF_thresholded_{scope}" / "run_manifest.json")
                    for scope in args.scopes
                },
                "provider_details": provider_details,
                "metric_aggregation": "target_month_macro_mean_for_precision_recall_f1",
                "count_aggregation": "summed_across_target_months_for_support_tp_fp_fn_tn",
                "n_monthly_metric_rows": int(len(metrics)),
                "n_threshold_rows": int(len(thresholds)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote GeoRF thresholded artifacts to {args.output_dir}")
    print(f"Rows: monthly_metrics={len(metrics)}, thresholds={len(thresholds)}, compact={len(compact)}")


if __name__ == "__main__":
    main()
