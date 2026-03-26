import os
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
MIN_YEAR = 2021

FS_TO_LAG = {1: 4, 2: 8, 3: 12}

RESULT_DIRS = {
    ("GeoRF", 1): BASE / "result_partition_k40_compare_GF_fs1",
    ("GeoRF", 2): BASE / "result_partition_k40_compare_GF_fs2",
    ("GeoRF", 3): BASE / "result_partition_k40_compare_GF_fs3",
    ("GeoXGB", 1): BASE / "result_partition_k40_compare_XGB_fs1",
    ("GeoXGB", 2): BASE / "result_partition_k40_compare_XGB_fs2",
    ("GeoXGB", 3): BASE / "result_partition_k40_compare_XGB_fs3",
    ("GeoDT", 1): BASE / "result_partition_k40_compare_DT_fs1",
    ("GeoDT", 2): BASE / "result_partition_k40_compare_DT_fs2",
    ("GeoDT", 3): BASE / "result_partition_k40_compare_DT_fs3",
}

FEWSNET_FILES = {
    1: BASE / "fewsnet_baseline_results" / "fewsnet_baseline_results_fs1.csv",
    2: BASE / "fewsnet_baseline_results" / "fewsnet_baseline_results_fs2.csv",
}


def load_model_quarterly(result_dir):
    path = result_dir / "metrics_monthly.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    part = df[df["model"] == "partitioned"].copy()
    part["test_month"] = pd.to_datetime(part["test_month"])
    part = part[part["test_month"].dt.year >= MIN_YEAR]
    part["year"] = part["test_month"].dt.year
    part["quarter"] = ((part["test_month"].dt.month - 1) // 3 + 1).astype(int)
    quarterly = part.groupby(["year", "quarter"], as_index=False).agg(
        {"precision": "mean", "recall": "mean", "f1": "mean"}
    )
    quarterly["year_quarter"] = quarterly["year"].astype(str) + "-Q" + quarterly["quarter"].astype(str)
    quarterly = quarterly.sort_values(["year", "quarter"]).reset_index(drop=True)
    return quarterly


def load_fewsnet(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["year"] >= MIN_YEAR].copy()
    df = df.rename(columns={"precision(1)": "precision", "recall(1)": "recall", "f1(1)": "f1"})
    df["year_quarter"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(int).astype(str)
    df = df.sort_values(["year", "quarter"]).reset_index(drop=True)
    return df


model_data = {}
for (model, fs), rdir in RESULT_DIRS.items():
    qdf = load_model_quarterly(rdir)
    if qdf is not None:
        model_data[(model, fs)] = qdf

fewsnet_data = {}
for fs, path in FEWSNET_FILES.items():
    qdf = load_fewsnet(path)
    if qdf is not None:
        fewsnet_data[fs] = qdf

if fewsnet_data and 3 not in fewsnet_data and 2 in fewsnet_data:
    fewsnet_data[3] = fewsnet_data[2].copy()

MODELS = ["GeoRF", "GeoXGB", "GeoDT"]
METRICS = [("precision", "Precision (Class 1)"),
           ("recall", "Recall (Class 1)"),
           ("f1", "F1 Score (Class 1)")]
SCOPES = [1, 2, 3]

COLORS = {"GeoRF": "#1f77b4", "GeoXGB": "#2ca02c", "GeoDT": "#9467bd", "FEWSNET": "#ff7f0e"}
MARKERS = {"GeoRF": "s", "GeoXGB": "^", "GeoDT": "D", "FEWSNET": "o"}

fig, axes = plt.subplots(len(SCOPES), len(METRICS), figsize=(20, 4.5 * len(SCOPES)),
                         constrained_layout=True)

for row, fs in enumerate(SCOPES):
    lag = FS_TO_LAG[fs]

    all_periods = set()
    for m in MODELS:
        df = model_data.get((m, fs))
        if df is not None:
            all_periods.update(df["year_quarter"].tolist())
    fdf = fewsnet_data.get(fs)
    if fdf is not None:
        all_periods.update(fdf["year_quarter"].tolist())

    common = sorted(all_periods)

    for col, (metric_col, metric_label) in enumerate(METRICS):
        ax = axes[row, col]

        for m in MODELS:
            df = model_data.get((m, fs))
            if df is None:
                continue
            merged = pd.DataFrame({"year_quarter": common}).merge(
                df[["year_quarter", metric_col]], on="year_quarter", how="left"
            )
            ax.plot(range(len(common)), merged[metric_col].values,
                    marker=MARKERS[m], color=COLORS[m], label=m,
                    linewidth=2, markersize=5, alpha=0.85)

        if fdf is not None:
            merged = pd.DataFrame({"year_quarter": common}).merge(
                fdf[["year_quarter", metric_col]], on="year_quarter", how="left"
            )
            ext_label = "FEWSNET (8mo-ext)" if fs == 3 else "FEWSNET"
            ls = "--" if fs == 3 else "-"
            ax.plot(range(len(common)), merged[metric_col].values,
                    marker=MARKERS["FEWSNET"], color=COLORS["FEWSNET"],
                    label=ext_label, linewidth=2, markersize=5, alpha=0.7 if fs == 3 else 0.85,
                    linestyle=ls)

        if row == 0:
            ax.set_title(metric_label, fontsize=14, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"Lag {lag} months", fontsize=12, fontweight="bold")
        if row == len(SCOPES) - 1:
            ax.set_xlabel("Time Period", fontsize=11)
            step = max(1, len(common) // 8)
            ax.set_xticks(range(0, len(common), step))
            ax.set_xticklabels([common[i] for i in range(0, len(common), step)],
                               rotation=45, ha="right")
        else:
            ax.set_xticks([])

        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        if row == 0 and col == len(METRICS) - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

fig.suptitle(
    "Partitioned Model vs FEWSNET Baseline — Class 1 Performance (2021–2024)",
    fontsize=16, fontweight="bold", y=1.01
)

out_dir = BASE / "other_outputs"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "model_comparison_line_chart.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close()
