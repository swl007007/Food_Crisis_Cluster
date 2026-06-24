#!/usr/bin/env python3
"""Build GeoRF probability, Brier, reliability, and bootstrap uncertainty diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from paper_horizon_labels import HORIZON_LABELS
except ModuleNotFoundError:
    from scripts.paper_artifacts.paper_horizon_labels import HORIZON_LABELS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated"
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
HORIZONS = HORIZON_LABELS
MODEL_COLUMNS = {
    "pooled": ("y_pred_pooled", "y_prob_pooled"),
    "partitioned": ("y_pred_partitioned", "y_prob_partitioned"),
}
METRICS = ("precision", "recall", "f1", "brier")


def resolve_path(path: Path, platform_name: str | None = None) -> Path:
    """Resolve Windows paths when running under WSL."""
    platform = os.name if platform_name is None else platform_name
    raw = str(path)
    if platform == "nt":
        wsl_match = re.match(r"^/mnt/([A-Za-z])/(.*)$", raw)
        if wsl_match:
            drive, rest = wsl_match.groups()
            return Path(f"{drive.upper()}:\\{rest.replace('/', '\\')}")
        return Path(raw)

    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if match:
        drive, rest = match.groups()
        return Path("/mnt") / drive.lower() / rest.replace("\\", "/")
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_admin_code(series: pd.Series) -> pd.Series:
    """Normalize FEWSNET admin codes to stable strings."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Return mean squared probability error."""
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_prob, errors="coerce")
    valid = y.notna() & p.notna()
    if int(valid.sum()) == 0:
        return np.nan
    return float(np.mean((p[valid].clip(0, 1) - y[valid]) ** 2))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def compute_model_metrics(df: pd.DataFrame, pred_col: str, prob_col: str) -> dict[str, float]:
    """Compute hard-label class-1 metrics and Brier score for one model."""
    valid = df[["y_true", pred_col, prob_col]].notna().all(axis=1)
    sub = df.loc[valid].copy()
    if sub.empty:
        return {
            "support": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "brier": np.nan,
        }

    y_true = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
    y_pred = pd.to_numeric(sub[pred_col], errors="coerce").astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan
    return {
        "support": int(len(sub)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier_score(sub["y_true"], sub[prob_col]),
    }


def reliability_bins(df: pd.DataFrame, prob_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Summarize predicted probabilities by equal-width reliability bins."""
    valid = df[["y_true", prob_col]].notna().all(axis=1)
    sub = df.loc[valid, ["y_true", prob_col]].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "bin_id",
                "bin_lower",
                "bin_upper",
                "n",
                "mean_predicted_probability",
                "observed_crisis_rate",
                "calibration_gap",
            ]
        )

    probabilities = pd.to_numeric(sub[prob_col], errors="coerce").clip(0, 1)
    sub["bin_id"] = np.minimum(np.floor(probabilities * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_id, group in sub.groupby("bin_id"):
        lower = bin_id / n_bins
        upper = (bin_id + 1) / n_bins
        mean_prob = float(pd.to_numeric(group[prob_col], errors="coerce").clip(0, 1).mean())
        observed = float(pd.to_numeric(group["y_true"], errors="coerce").mean())
        rows.append(
            {
                "bin_id": int(bin_id),
                "bin_lower": lower,
                "bin_upper": upper,
                "n": int(len(group)),
                "mean_predicted_probability": mean_prob,
                "observed_crisis_rate": observed,
                "calibration_gap": observed - mean_prob,
            }
        )
    return pd.DataFrame(rows).sort_values("bin_id").reset_index(drop=True)


def resample_clusters(df: pd.DataFrame, cluster_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Resample whole clusters with replacement and keep all rows per selected cluster."""
    clusters = np.array(sorted(df[cluster_col].dropna().unique()))
    if clusters.size == 0:
        return df.iloc[0:0].copy()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    parts = []
    for draw_id, cluster in enumerate(sampled):
        part = df[df[cluster_col] == cluster].copy()
        part["_bootstrap_draw"] = draw_id
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _paired_metric_values(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    pooled = compute_model_metrics(df, *MODEL_COLUMNS["pooled"])
    partitioned = compute_model_metrics(df, *MODEL_COLUMNS["partitioned"])
    delta = {metric: partitioned[metric] - pooled[metric] for metric in METRICS}
    return {"pooled": pooled, "partitioned": partitioned, "delta": delta}


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    low, high = np.percentile(arr, [2.5, 97.5])
    return float(low), float(high)


def build_bootstrap_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cluster_col: str = "ADMIN0",
) -> pd.DataFrame:
    """Build paired country-clustered bootstrap CIs for pooled, partitioned, and delta metrics."""
    point = _paired_metric_values(df)
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = {
        "pooled": {metric: [] for metric in METRICS},
        "partitioned": {metric: [] for metric in METRICS},
        "delta": {metric: [] for metric in METRICS},
    }
    for _ in range(n_bootstrap):
        sample = resample_clusters(df, cluster_col, rng)
        values = _paired_metric_values(sample)
        for model_name in draws:
            for metric in METRICS:
                draws[model_name][metric].append(values[model_name][metric])

    rows = []
    for metric in METRICS:
        pooled_low, pooled_high = _percentile_interval(draws["pooled"][metric])
        part_low, part_high = _percentile_interval(draws["partitioned"][metric])
        delta_low, delta_high = _percentile_interval(draws["delta"][metric])
        rows.append(
            {
                "metric": metric,
                "support": int(len(df)),
                "n_countries": int(df[cluster_col].nunique()),
                "pooled_point": point["pooled"][metric],
                "pooled_ci_low": pooled_low,
                "pooled_ci_high": pooled_high,
                "partitioned_point": point["partitioned"][metric],
                "partitioned_ci_low": part_low,
                "partitioned_ci_high": part_high,
                "delta_point": point["delta"][metric],
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
            }
        )
    return pd.DataFrame(rows)


def build_region_bootstrap_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    min_countries: int = 3,
) -> pd.DataFrame:
    """Build country-clustered bootstrap CIs separately by region."""
    frames = []
    for region in sorted(df["region"].dropna().unique()):
        sub = df[df["region"] == region].copy()
        if sub["ADMIN0"].nunique() < min_countries:
            continue
        ci = build_bootstrap_ci(sub, n_bootstrap=n_bootstrap, seed=seed, cluster_col="ADMIN0")
        ci.insert(0, "region", region)
        frames.append(ci)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def probability_uncertainty(probabilities: pd.Series) -> pd.Series:
    """Return threshold-distance uncertainty on 0..1 scale."""
    p = pd.to_numeric(probabilities, errors="coerce").clip(0, 1)
    return (1.0 - (p - 0.5).abs() * 2.0).clip(0, 1)


def probability_entropy(probabilities: pd.Series) -> pd.Series:
    """Return binary entropy in bits."""
    p = pd.to_numeric(probabilities, errors="coerce").clip(1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_region_map() -> dict[str, str]:
    """Load shared paper region map."""
    module_path = REPO_ROOT / "scripts" / "paper_artifacts" / "plot_region_class_prevalence.py"
    spec = importlib.util.spec_from_file_location("plot_region_class_prevalence_for_probability", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import region map from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.REGION_MAP)


def load_region_lookup(shapefile_path: Path) -> pd.DataFrame:
    """Load admin-code to country and region lookup from FEWSNET boundaries."""
    import geopandas as gpd

    gdf = gpd.read_file(resolve_path(shapefile_path))
    for column in ("FEWSNET_admin_code", "uid", "admin_code", "adm_code", "FNID"):
        if column in gdf.columns:
            gdf = gdf.rename(columns={column: "FEWSNET_admin_code"})
            break
    if "FEWSNET_admin_code" not in gdf.columns:
        raise ValueError(f"No FEWSNET admin-code column found in {shapefile_path}")
    if "ADMIN0" not in gdf.columns:
        raise ValueError(f"ADMIN0 column not found in {shapefile_path}")
    lookup = gdf[["FEWSNET_admin_code", "ADMIN0"]].copy()
    lookup["FEWSNET_admin_code"] = normalize_admin_code(lookup["FEWSNET_admin_code"])
    lookup["region"] = lookup["ADMIN0"].map(load_region_map()).fillna("Other")
    return lookup.drop_duplicates("FEWSNET_admin_code").reset_index(drop=True)


def load_prediction_files(source_dir: Path, scopes: list[str]) -> pd.DataFrame:
    """Load GeoRF fs prediction files and validate probability columns."""
    frames = []
    required = {
        "FEWSNET_admin_code",
        "month_start",
        "y_true",
        "y_pred_pooled",
        "y_pred_partitioned",
        "y_prob_pooled",
        "y_prob_partitioned",
    }
    for scope in scopes:
        path = source_dir / f"result_partition_k40_compare_GF_{scope}" / "predictions_monthly.csv"
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df["scope"] = scope
        df["forecasting_horizon"] = HORIZONS.get(scope, scope)
        df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
        df["month_start"] = pd.to_datetime(df["month_start"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_reliability_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Build reliability-bin table for each horizon and model."""
    frames = []
    for (scope, horizon), sub in df.groupby(["scope", "forecasting_horizon"], sort=True):
        for model_name, (_, prob_col) in MODEL_COLUMNS.items():
            bins = reliability_bins(sub, prob_col, n_bins=n_bins)
            metrics = compute_model_metrics(sub, *MODEL_COLUMNS[model_name])
            bins.insert(0, "model", model_name)
            bins.insert(0, "forecasting_horizon", horizon)
            bins.insert(0, "scope", scope)
            bins["brier_score"] = metrics["brier"]
            frames.append(bins)
    return pd.concat(frames, ignore_index=True)


def build_uncertainty_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive probability uncertainty summaries."""
    rows = []
    for (scope, horizon), sub in df.groupby(["scope", "forecasting_horizon"], sort=True):
        for model_name, (_, prob_col) in MODEL_COLUMNS.items():
            uncertainty = probability_uncertainty(sub[prob_col])
            entropy = probability_entropy(sub[prob_col])
            rows.append(
                {
                    "scope": scope,
                    "forecasting_horizon": horizon,
                    "model": model_name,
                    "support": int(sub[prob_col].notna().sum()),
                    "mean_probability": float(pd.to_numeric(sub[prob_col], errors="coerce").mean()),
                    "mean_threshold_distance_uncertainty": float(uncertainty.mean()),
                    "mean_binary_entropy": float(entropy.mean()),
                    "brier_score": compute_model_metrics(sub, *MODEL_COLUMNS[model_name])["brier"],
                }
            )
    return pd.DataFrame(rows)


def build_all_bootstrap_ci(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Build whole-sample bootstrap CIs by horizon."""
    frames = []
    for (scope, horizon), sub in df.groupby(["scope", "forecasting_horizon"], sort=True):
        ci = build_bootstrap_ci(sub, n_bootstrap=n_bootstrap, seed=seed, cluster_col="ADMIN0")
        ci.insert(0, "forecasting_horizon", horizon)
        ci.insert(0, "scope", scope)
        frames.append(ci)
    return pd.concat(frames, ignore_index=True)


def format_ci_cell(row: pd.Series) -> str:
    """Format a point estimate and percentile interval for compact appendix tables."""
    return f"{row['delta_point']:.3f} [{row['delta_ci_low']:.3f}, {row['delta_ci_high']:.3f}]"


def build_compact_bootstrap_table(bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Build a 1-row-per-horizon compact delta table from bootstrap CI output."""
    metric_order = ["precision", "recall", "f1", "brier"]
    rows = []
    for (scope, horizon), sub in bootstrap.groupby(["scope", "forecasting_horizon"], sort=True):
        row = {
            "scope": scope,
            "forecasting_horizon": horizon,
        }
        for metric in metric_order:
            metric_rows = sub[sub["metric"] == metric]
            row[f"delta_{metric}"] = format_ci_cell(metric_rows.iloc[0]) if not metric_rows.empty else ""
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write a compact Markdown table."""
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all_region_bootstrap_ci(df: pd.DataFrame, n_bootstrap: int, seed: int, min_countries: int) -> pd.DataFrame:
    """Build region-specific bootstrap CIs by horizon."""
    frames = []
    for (scope, horizon), sub in df.groupby(["scope", "forecasting_horizon"], sort=True):
        ci = build_region_bootstrap_ci(sub, n_bootstrap=n_bootstrap, seed=seed, min_countries=min_countries)
        if ci.empty:
            continue
        ci.insert(0, "forecasting_horizon", horizon)
        ci.insert(0, "scope", scope)
        frames.append(ci)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_reliability(reliability: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plot reliability curves by horizon and model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = list(dict.fromkeys(reliability["forecasting_horizon"].tolist()))
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), sharex=True, sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    colors = {"pooled": "#4c78a8", "partitioned": "#d95f02"}
    for ax, horizon in zip(axes, horizons):
        sub_h = reliability[reliability["forecasting_horizon"] == horizon]
        for model_name, sub in sub_h.groupby("model"):
            ax.plot(
                sub["mean_predicted_probability"],
                sub["observed_crisis_rate"],
                marker="o",
                label=model_name,
                color=colors.get(model_name),
            )
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
        ax.set_title(horizon)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed crisis rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("GeoRF Probability Reliability Diagnostics", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_note(output_dir: Path, n_bootstrap: int, min_countries: int) -> Path:
    """Write reviewer-facing Chinese note and English appendix text."""
    path = output_dir / "georf_probability_uncertainty_note.md"
    path.write_text(
        "\n".join(
            [
                "# GeoRF Probability and Uncertainty Diagnostics",
                "",
                "中文审查说明：",
                "",
                "该 appendix 只针对 GeoRF pooled 和 partitioned/local RF 模型。",
                "Stage 3 现在导出 class-1 probabilities，同时保留原有 binary predictions。",
                "标准 Stage 3 评估没有额外 threshold tuning；binary classification 使用 classifier 默认 hard prediction rule。",
                "Brier score 和 reliability bins 使用 raw RF probabilities，不额外拟合 Platt scaling 或 isotonic calibration。",
                "Brier score 越低越好；表中的 delta 仍定义为 partitioned minus pooled，因此 Brier delta 为负表示 partitioned 更好。",
                f"Paired bootstrap confidence intervals 使用 country-clustered resampling，默认 {n_bootstrap} 次重复；region-specific CI 至少需要 {min_countries} 个 countries。",
                "这些 uncertainty summaries 是对已评估 polygon-month predictions 的不确定性诊断，不是未来事件的预测区间。",
                "",
                "Appendix text (English):",
                "",
                "We export class-1 probabilities for the GeoRF pooled and partitioned RF models and report probability diagnostics on the evaluated polygon-month observations.",
                "The standard Stage 3 evaluation does not perform separate threshold tuning; hard classifications follow the classifier default decision rule.",
                "Brier scores and reliability bins are computed from raw RF probabilities, with no additional calibration model fitted in this appendix.",
                "Lower Brier scores are better; deltas are reported as partitioned minus pooled, so negative Brier deltas favor the partitioned model.",
                "Paired confidence intervals use country-clustered bootstrap resampling so pooled and partitioned predictions are evaluated on the same resampled polygon-month observations.",
                "Region-specific intervals repeat the same procedure within regions when enough countries are available.",
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
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-region-countries", type=int, default=3)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_prediction_files(args.source_dir, args.scopes)
    region_lookup = load_region_lookup(args.shapefile)
    df = predictions.merge(region_lookup, on="FEWSNET_admin_code", how="left")
    df = df[df["ADMIN0"].notna()].copy()

    bootstrap = build_all_bootstrap_ci(df, n_bootstrap=args.n_bootstrap, seed=args.seed)
    region_bootstrap = build_all_region_bootstrap_ci(
        df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        min_countries=args.min_region_countries,
    )
    reliability = build_reliability_table(df, n_bins=args.bins)
    uncertainty = build_uncertainty_summary(df)
    compact = build_compact_bootstrap_table(bootstrap)

    bootstrap.to_csv(args.output_dir / "georf_probability_bootstrap_ci.csv", index=False)
    region_bootstrap.to_csv(args.output_dir / "georf_probability_bootstrap_region_ci.csv", index=False)
    reliability.to_csv(args.output_dir / "georf_probability_brier_reliability.csv", index=False)
    uncertainty.to_csv(args.output_dir / "georf_probability_uncertainty_summary.csv", index=False)
    compact.to_csv(args.output_dir / "georf_probability_bootstrap_compact_table.csv", index=False)
    write_markdown_table(compact, args.output_dir / "georf_probability_bootstrap_compact_table.md")
    plot_reliability(reliability, args.output_dir / "georf_probability_reliability.png", dpi=args.dpi)
    write_note(args.output_dir, args.n_bootstrap, args.min_region_countries)

    print(f"Wrote GeoRF probability diagnostics to {args.output_dir}")
    print(f"Rows: bootstrap={len(bootstrap)}, region={len(region_bootstrap)}, reliability={len(reliability)}")


if __name__ == "__main__":
    main()
