#!/usr/bin/env python3
"""Build GeoRF threshold-free and fixed-operating-point diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "11_threshold_free_metrics"
HORIZONS = {
    "fs1": "4-month lag",
    "fs2": "8-month lag",
    "fs3": "12-month lag",
}
MODEL_PROB_COLUMNS = {
    "pooled": "y_prob_pooled",
    "partitioned": "y_prob_partitioned",
}
FIXED_PRECISION_TARGETS = (0.75, 0.80)
FIXED_RECALL_TARGETS = (0.50, 0.60)


def _clean_probability_inputs(y_true: pd.Series, y_prob: pd.Series) -> tuple[pd.Series, pd.Series]:
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_prob, errors="coerce")
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int)
    p = p.loc[valid].clip(0, 1).astype(float)
    return y.reset_index(drop=True), p.reset_index(drop=True)


def pr_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    y, p = _clean_probability_inputs(y_true, y_prob)
    if y.empty or int(y.sum()) == 0:
        return np.nan
    return float(average_precision_score(y, p))


def precision_recall_points(y_true: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    y, p = _clean_probability_inputs(y_true, y_prob)
    if y.empty or int(y.sum()) == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "predicted_positive"])

    rows = []
    positives = int(y.sum())
    for threshold in sorted(p.unique(), reverse=True):
        pred = p >= threshold
        predicted_positive = int(pred.sum())
        if predicted_positive == 0:
            continue
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                "recall": float(tp / positives) if positives else np.nan,
                "predicted_positive": predicted_positive,
            }
        )
    return pd.DataFrame(rows)


def recall_at_fixed_precision(points: pd.DataFrame, fixed_precision: float) -> float:
    feasible = points[points["precision"] >= fixed_precision]
    if feasible.empty:
        return np.nan
    return float(feasible["recall"].max())


def precision_at_fixed_recall(points: pd.DataFrame, fixed_recall: float) -> float:
    feasible = points[points["recall"] >= fixed_recall]
    if feasible.empty:
        return np.nan
    return float(feasible["precision"].max())


def _target_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def compute_model_metrics(df: pd.DataFrame, prob_col: str) -> dict[str, float]:
    y_true, y_prob = _clean_probability_inputs(df["y_true"], df[prob_col])
    points = precision_recall_points(y_true, y_prob)
    metrics: dict[str, float] = {
        "support": int(len(y_true)),
        "positive_cases": int(y_true.sum()) if not y_true.empty else 0,
        "pr_auc": pr_auc(y_true, y_prob),
    }
    for target in FIXED_PRECISION_TARGETS:
        metrics[f"recall_at_precision_{_target_label(target)}"] = recall_at_fixed_precision(points, target)
    for target in FIXED_RECALL_TARGETS:
        metrics[f"precision_at_recall_{_target_label(target)}"] = precision_at_fixed_recall(points, target)
    return metrics


def load_prediction_files(source_dir: Path, scopes: list[str]) -> pd.DataFrame:
    required = {"FEWSNET_admin_code", "month_start", "y_true", "y_prob_pooled", "y_prob_partitioned"}
    frames = []
    for scope in scopes:
        path = source_dir / f"result_partition_k40_compare_GF_{scope}" / "predictions_monthly.csv"
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df["scope"] = scope
        df["forecasting_horizon"] = HORIZONS.get(scope, scope)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, horizon), sub in predictions.groupby(["scope", "forecasting_horizon"], sort=True):
        for model, prob_col in MODEL_PROB_COLUMNS.items():
            rows.append(
                {
                    "scope": scope,
                    "forecasting_horizon": horizon,
                    "model": model,
                    **compute_model_metrics(sub, prob_col),
                }
            )
    return pd.DataFrame(rows)


def build_compact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "pr_auc",
        "recall_at_precision_0_75",
        "recall_at_precision_0_80",
        "precision_at_recall_0_50",
        "precision_at_recall_0_60",
    ]
    rows = []
    for (scope, horizon), sub in metrics.groupby(["scope", "forecasting_horizon"], sort=True):
        pooled = sub[sub["model"] == "pooled"].iloc[0]
        partitioned = sub[sub["model"] == "partitioned"].iloc[0]
        row: dict[str, object] = {"scope": scope, "forecasting_horizon": horizon}
        for metric in metric_columns:
            row[f"pooled_{metric}"] = pooled[metric]
            row[f"partitioned_{metric}"] = partitioned[metric]
            row[f"delta_{metric}"] = partitioned[metric] - pooled[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def format_compact_for_paper(compact: pd.DataFrame) -> pd.DataFrame:
    table = compact.copy()
    for column in table.columns:
        if column not in {"scope", "forecasting_horizon"}:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return table


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


def write_note(output_dir: Path) -> Path:
    path = output_dir / "georf_threshold_free_metrics_note.md"
    path.write_text(
        "\n".join(
            [
                "# GeoRF Threshold-Free and Fixed Operating-Point Metrics",
                "",
                "中文审查说明：",
                "",
                "该 appendix 只针对 GeoRF pooled 和 partitioned/local RF 模型。",
                "所有指标都从现有 Stage 3 `y_prob_pooled` 和 `y_prob_partitioned` 概率输出计算，不重跑模型，也不改变主文 binary prediction rule。",
                "PR-AUC 使用 average precision，衡量 crisis probability ranking 的整体 precision-recall 表现。",
                "Recall at fixed precision 和 precision at fixed recall 是 post hoc operating-point diagnostics，用于展示现有概率排序在指定 precision 或 recall 约束下可达到的 tradeoff。",
                "这些指标不表示本文已经进行了 threshold tuning；主结果仍然使用当前 hard predictions 的 precision、recall 和 F1。",
                "如果某个 operating point 不可达到，表中保留空值。",
                "",
                "Appendix text (English):",
                "",
                "We report additional threshold-free and fixed operating-point diagnostics for the GeoRF pooled and partitioned RF models.",
                "All metrics are computed from existing Stage 3 crisis-class probabilities and do not require model retraining or a different threshold-selection procedure.",
                "PR-AUC is computed as average precision and summarizes the probability ranking across the precision-recall curve.",
                "Recall at fixed precision and precision at fixed recall are post hoc operating-point diagnostics showing feasible tradeoffs under the existing probability scores.",
                "The main binary results remain based on the implemented hard predictions; these appendix metrics are complementary ranking and sensitivity diagnostics.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_prediction_files(args.source_dir, args.scopes)
    metrics = build_metric_table(predictions)
    compact = build_compact_table(metrics)
    compact_paper = format_compact_for_paper(compact)

    metrics.to_csv(args.output_dir / "georf_threshold_free_metrics.csv", index=False)
    compact.to_csv(args.output_dir / "georf_threshold_free_metrics_compact_table.csv", index=False)
    write_markdown_table(compact_paper, args.output_dir / "georf_threshold_free_metrics_compact_table.md")
    write_note(args.output_dir)

    print(f"Wrote GeoRF threshold-free metrics to {args.output_dir}")
    print(f"Rows: predictions={len(predictions)}, metrics={len(metrics)}, compact={len(compact)}")


if __name__ == "__main__":
    main()
