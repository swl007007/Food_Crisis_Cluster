#!/usr/bin/env python3
"""Plot GeoRF pooled vs. partitioned precision-recall curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_georf_threshold_free_metrics as threshold_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "11_threshold_free_metrics"
DEFAULT_CURVE_POINTS = DEFAULT_OUTPUT_DIR / "georf_precision_recall_curve_points.csv"
DEFAULT_PNG = DEFAULT_OUTPUT_DIR / "georf_precision_recall_curves.png"
DEFAULT_PDF = DEFAULT_OUTPUT_DIR / "georf_precision_recall_curves.pdf"
DEFAULT_COMPACT_TABLE = DEFAULT_OUTPUT_DIR / "georf_threshold_free_metrics_compact_table.csv"

MODEL_STYLES = {
    "pooled": {"label": "Pooled RF", "color": "#1f77b4", "linestyle": "--", "marker": "o"},
    "partitioned": {"label": "Partitioned RF", "color": "#d62728", "linestyle": "-", "marker": "s"},
}


def build_precision_recall_curve_points(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, horizon), sub in predictions.groupby(["scope", "forecasting_horizon"], sort=True):
        for model, prob_col in threshold_metrics.MODEL_PROB_COLUMNS.items():
            y_true, y_prob = threshold_metrics._clean_probability_inputs(sub["y_true"], sub[prob_col])
            if y_true.empty or int(y_true.sum()) == 0:
                continue
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            ap = float(average_precision_score(y_true, y_prob))
            threshold_values = list(thresholds.astype(float)) + [np.nan]
            for point_precision, point_recall, threshold in zip(precision, recall, threshold_values):
                rows.append(
                    {
                        "scope": scope,
                        "forecasting_horizon": horizon,
                        "model": model,
                        "recall": float(point_recall),
                        "precision": float(point_precision),
                        "threshold": threshold,
                        "average_precision": ap,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "scope",
            "forecasting_horizon",
            "model",
            "recall",
            "precision",
            "threshold",
            "average_precision",
        ],
    )


def _target_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def _load_or_build_compact_table(output_dir: Path, compact_table: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    if compact_table.exists():
        return pd.read_csv(compact_table)
    metrics = threshold_metrics.build_metric_table(predictions)
    return threshold_metrics.build_compact_table(metrics)


def _plot_operating_points(ax, compact_row: pd.Series, model: str, color: str, marker: str) -> None:
    for fixed_precision in threshold_metrics.FIXED_PRECISION_TARGETS:
        label = _target_label(fixed_precision)
        recall = compact_row[f"{model}_recall_at_precision_{label}"]
        if pd.notna(recall):
            ax.scatter(
                recall,
                fixed_precision,
                color=color,
                marker=marker,
                s=34,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
    for fixed_recall in threshold_metrics.FIXED_RECALL_TARGETS:
        label = _target_label(fixed_recall)
        precision = compact_row[f"{model}_precision_at_recall_{label}"]
        if pd.notna(precision):
            ax.scatter(
                fixed_recall,
                precision,
                facecolor="white",
                edgecolor=color,
                marker=marker,
                s=42,
                linewidth=1.1,
                zorder=4,
            )


def plot_precision_recall_curves(
    curve_points: pd.DataFrame,
    compact: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    dpi: int = 300,
) -> None:
    scopes = [scope for scope in threshold_metrics.HORIZONS if scope in set(curve_points["scope"])]
    fig, axes = plt.subplots(1, len(scopes), figsize=(5.4 * len(scopes), 4.6), sharex=True, sharey=True)
    if len(scopes) == 1:
        axes = [axes]

    for ax, scope in zip(axes, scopes):
        scope_points = curve_points[curve_points["scope"] == scope]
        compact_row = compact[compact["scope"] == scope].iloc[0]
        for model, style in MODEL_STYLES.items():
            model_points = scope_points[scope_points["model"] == model]
            if model_points.empty:
                continue
            ap = model_points["average_precision"].iloc[0]
            ax.step(
                model_points["recall"],
                model_points["precision"],
                where="post",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=f"{style['label']} (AP={ap:.3f})",
            )
            _plot_operating_points(ax, compact_row, model, style["color"], style["marker"])

        ax.axhline(0.75, color="0.75", linewidth=0.8, linestyle=":")
        ax.axhline(0.80, color="0.75", linewidth=0.8, linestyle=":")
        ax.axvline(0.50, color="0.80", linewidth=0.8, linestyle=":")
        ax.axvline(0.60, color="0.80", linewidth=0.8, linestyle=":")
        ax.set_title(f"{scope}: {threshold_metrics.HORIZONS[scope]}", fontsize=12)
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.grid(True, color="0.90", linewidth=0.7)
        ax.legend(loc="lower left", fontsize=8, frameon=True)

    axes[0].set_ylabel("Precision")
    for ax in axes:
        ax.set_xlabel("Recall")
    fig.suptitle("GeoRF Precision-Recall Curves: Pooled vs. Partitioned", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Filled markers: recall at fixed precision 0.75/0.80. Open markers: precision at fixed recall 0.50/0.60.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_precision_recall_curve_artifacts(
    source_dir: Path = REPO_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scopes: tuple[str, ...] = ("fs1", "fs2", "fs3"),
    dpi: int = 300,
    compact_table: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = threshold_metrics.load_prediction_files(source_dir, list(scopes))
    curve_points = build_precision_recall_curve_points(predictions)
    compact_path = compact_table or output_dir / DEFAULT_COMPACT_TABLE.name
    compact = _load_or_build_compact_table(output_dir, compact_path, predictions)

    curve_points_path = output_dir / DEFAULT_CURVE_POINTS.name
    output_png = output_dir / DEFAULT_PNG.name
    output_pdf = output_dir / DEFAULT_PDF.name
    curve_points.to_csv(curve_points_path, index=False)
    plot_precision_recall_curves(curve_points, compact, output_png, output_pdf, dpi=dpi)
    return {"curve_points": curve_points_path, "png": output_png, "pdf": output_pdf}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compact-table", type=Path, default=None)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    outputs = write_precision_recall_curve_artifacts(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        scopes=tuple(args.scopes),
        dpi=args.dpi,
        compact_table=args.compact_table,
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
