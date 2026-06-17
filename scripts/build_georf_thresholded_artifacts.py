#!/usr/bin/env python3
"""Build paper-facing artifacts for GeoRF validation-selected thresholding."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "12_thresholded_georf_results"
HORIZONS = {
    "fs1": "4-month lag",
    "fs2": "8-month lag",
    "fs3": "12-month lag",
}


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


def _aggregate_model_metrics(group: pd.DataFrame) -> pd.Series:
    tp = group["tp"].sum()
    fp = group["fp"].sum()
    fn = group["fn"].sum()
    tn = group["tn"].sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return pd.Series(
        {
            "support": int(group["n"].sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def build_horizon_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["scope", "forecasting_horizon", "model"], sort=True)
        .apply(_aggregate_model_metrics)
        .reset_index()
    )
    return grouped


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
                "原始 pooled 和 partitioned hard-prediction 结果保留，用于对照。",
                "这些结果先写入独立 artifact folder，尚不覆盖主文 01-11 artifacts。",
                "",
                "Appendix text (English):",
                "",
                "We evaluate a validation-selected probability threshold for the GeoRF partitioned RF model.",
                "For each forecasting horizon and target month, the threshold is selected on a validation subset from the rolling training window by maximizing class-1 F1, then applied to the held-out target-month probabilities.",
                "The procedure does not use test labels for threshold selection.",
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

    print(f"Wrote GeoRF thresholded artifacts to {args.output_dir}")
    print(f"Rows: monthly_metrics={len(metrics)}, thresholds={len(thresholds)}, compact={len(compact)}")


if __name__ == "__main__":
    main()
