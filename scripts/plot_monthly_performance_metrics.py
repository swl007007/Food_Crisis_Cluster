#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_ROOT = Path("main_ablation_results/march2026_main_backup_month_ind_cont3")
DEFAULT_FEWSNET_ROOT = Path("fewsnet_baseline_results_backup")
DEFAULT_OUTPUT_DIR = DEFAULT_ABLATION_ROOT / "monthly_performance_plots"

MODEL_CONFIG = {
    "geodt": {
        "label": "GeoDT",
        "token": "DT",
        "color": "#d62728",
        "filename": "geodt_monthly_performance.png",
    },
    "georf": {
        "label": "GeoRF",
        "token": "GF",
        "color": "#1f77b4",
        "filename": "georf_monthly_performance.png",
    },
}
SCOPES = ("fs1", "fs2", "fs3")
METRICS = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
MODEL_SERIES = ("partitioned", "pooled")
FEWSNET_COLOR = "#2ca02c"
FEWSNET_REUSED_LABEL = "FEWSNET baseline (fs2 reused for fs3)"
QUARTER_TO_MONTH = {"1": "02", "2": "06", "4": "10"}

MODEL_COLUMNS = {"test_month", "model", "precision", "recall", "f1"}
FEWSNET_COLUMNS = {"year", "quarter", "precision(1)", "recall(1)", "f1(1)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GeoDT and GeoRF monthly performance metrics with FEWSNET baselines."
    )
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--fewsnet-root", type=Path, default=DEFAULT_FEWSNET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=("geodt", "georf", "all"), default="all")
    parser.add_argument(
        "--no-extend-fewsnet",
        dest="extend_fewsnet",
        action="store_false",
        default=True,
        help="Do not reuse FEWSNET fs2 baseline values for fs3.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    raw = str(path)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if match:
        drive, rest = match.groups()
        return Path("/mnt") / drive.lower() / rest.replace("\\", "/")
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def selected_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(MODEL_CONFIG)
    return [model_arg]


def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def read_required_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, required, path)
    return df


def model_metrics_path(ablation_root: Path, model_key: str, scope: str) -> Path:
    token = MODEL_CONFIG[model_key]["token"]
    return ablation_root / f"result_partition_k40_compare_{token}_{scope}" / "metrics_monthly.csv"


def fewsnet_path(fewsnet_root: Path, scope: str) -> Path:
    source_scope = "fs2" if scope == "fs3" else scope
    return fewsnet_root / f"fewsnet_baseline_results_{source_scope}.csv"


def load_model_metrics(ablation_root: Path, models: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    source_paths = []
    for model_key in models:
        config = MODEL_CONFIG[model_key]
        for scope in SCOPES:
            path = model_metrics_path(ablation_root, model_key, scope)
            df = read_required_csv(path, MODEL_COLUMNS)
            df = df.copy()
            df["test_month"] = pd.to_datetime(df["test_month"], format="%Y-%m", errors="coerce")
            if df["test_month"].isna().any():
                raise ValueError(f"Invalid test_month values in {path}")
            df["test_month"] = df["test_month"].dt.strftime("%Y-%m")
            df["model_family"] = config["label"]
            df["model_key"] = model_key
            df["source_token"] = config["token"]
            df["scope"] = scope
            df["source_file"] = str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
            frames.append(df)
            source_paths.append(df["source_file"].iloc[0])
    return pd.concat(frames, ignore_index=True), source_paths


def load_fewsnet_baselines(fewsnet_root: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    baselines = {}
    source_paths = []
    for source_scope in ("fs1", "fs2"):
        path = fewsnet_path(fewsnet_root, source_scope)
        df = read_required_csv(path, FEWSNET_COLUMNS).copy()
        df["quarter"] = df["quarter"].astype(str)
        df = df[df["quarter"].isin(QUARTER_TO_MONTH)].copy()
        df["aligned_test_month"] = (
            df["year"].astype(int).astype(str) + "-" + df["quarter"].map(QUARTER_TO_MONTH)
        )
        df = df.rename(
            columns={"precision(1)": "precision", "recall(1)": "recall", "f1(1)": "f1"}
        )
        df["baseline_scope"] = source_scope
        df["source_file"] = str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
        baselines[source_scope] = df
        source_paths.append(df["source_file"].iloc[0] if len(df) else str(path))
    return baselines, source_paths


def value_for_month(df: pd.DataFrame, month: str, metric: str) -> float:
    row = df[df["test_month"] == month]
    if row.empty:
        return np.nan
    value = pd.to_numeric(row.iloc[0][metric], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def baseline_value_for_month(baseline_df: pd.DataFrame, month: str, metric: str) -> float:
    row = baseline_df[baseline_df["aligned_test_month"] == month]
    if row.empty:
        return np.nan
    value = pd.to_numeric(row.iloc[0][metric], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def add_missing(missing_points: list[dict], source_kind: str, model_family, scope, label, metric, month, reason) -> None:
    missing_points.append(
        {
            "source_kind": source_kind,
            "model_family": model_family,
            "scope": scope,
            "series_label": label,
            "metric": metric,
            "test_month": month,
            "reason": reason,
        }
    )


def build_plot_payload(
    model_df: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
    models: list[str],
    extend_fewsnet: bool = True,
) -> tuple[dict, list[dict]]:
    payload = {}
    missing_points = []
    for model_key in models:
        config = MODEL_CONFIG[model_key]
        payload[model_key] = {}
        for scope in SCOPES:
            scoped = model_df[(model_df["model_key"] == model_key) & (model_df["scope"] == scope)]
            months = sorted(scoped["test_month"].dropna().unique(), key=lambda m: pd.Period(m, freq="M"))
            payload[model_key][scope] = {"months": months, "metrics": {}}
            baseline_scope = "fs2" if scope == "fs3" and extend_fewsnet else scope
            baseline = baselines.get(baseline_scope)
            for metric, _ in METRICS:
                payload[model_key][scope]["metrics"][metric] = {}
                for series in MODEL_SERIES:
                    series_df = scoped[scoped["model"] == series]
                    values = []
                    for month in months:
                        value = value_for_month(series_df, month, metric)
                        values.append(value)
                        if np.isnan(value):
                            add_missing(
                                missing_points,
                                "model",
                                config["label"],
                                scope,
                                series,
                                metric,
                                month,
                                "missing row or metric value",
                            )
                    payload[model_key][scope]["metrics"][metric][series] = values
                fewsnet_label = (
                    FEWSNET_REUSED_LABEL
                    if scope == "fs3" and extend_fewsnet
                    else "FEWSNET baseline"
                )
                values = []
                for month in months:
                    if scope == "fs3" and not extend_fewsnet:
                        value = np.nan
                    elif baseline is not None:
                        value = baseline_value_for_month(baseline, month, metric)
                    else:
                        value = np.nan
                    values.append(value)
                    if np.isnan(value):
                        add_missing(
                            missing_points,
                            "fewsnet",
                            None,
                            scope,
                            fewsnet_label,
                            metric,
                            month,
                            "FEWSNET fs3 extension disabled"
                            if scope == "fs3" and not extend_fewsnet
                            else "missing aligned baseline",
                        )
                payload[model_key][scope]["metrics"][metric]["fewsnet"] = values
    return payload, missing_points


def render_model_figure(
    model_key: str,
    payload: dict,
    output_path: Path | None = None,
    extend_fewsnet: bool = True,
) -> None:
    config = MODEL_CONFIG[model_key]
    fig, axes = plt.subplots(len(SCOPES), len(METRICS), figsize=(17, 10), sharey=True)
    for row, scope in enumerate(SCOPES):
        months = payload[model_key][scope]["months"]
        x = np.arange(len(months))
        for col, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row][col]
            metric_payload = payload[model_key][scope]["metrics"][metric]
            ax.plot(
                x,
                metric_payload["partitioned"],
                color=config["color"],
                linestyle="-",
                marker="o",
                linewidth=2,
                label="partitioned",
            )
            ax.plot(
                x,
                metric_payload["pooled"],
                color=config["color"],
                linestyle="--",
                marker="s",
                linewidth=2,
                alpha=0.75,
                label="pooled",
            )
            if scope != "fs3" or extend_fewsnet:
                ax.plot(
                    x,
                    metric_payload["fewsnet"],
                    color=FEWSNET_COLOR,
                    linestyle="-.",
                    marker="^",
                    linewidth=2,
                    label=FEWSNET_REUSED_LABEL if scope == "fs3" else "FEWSNET baseline",
                )
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(metric_label)
            if col == 0:
                ax.set_ylabel(f"{scope}\nMetric value")
            if row == len(SCOPES) - 1:
                ax.set_xlabel("Test month")
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=45, ha="right")
            if scope == "fs3" and extend_fewsnet:
                ax.text(
                    0.01,
                    0.04,
                    "FEWSNET fs2 reused for fs3",
                    transform=ax.transAxes,
                    fontsize=8,
                    color=FEWSNET_COLOR,
                    va="bottom",
                )
    legend_handles = [
        Line2D([0], [0], color=config["color"], linestyle="-", marker="o", linewidth=2, label="partitioned"),
        Line2D([0], [0], color=config["color"], linestyle="--", marker="s", linewidth=2, alpha=0.75, label="pooled"),
        Line2D([0], [0], color=FEWSNET_COLOR, linestyle="-.", marker="^", linewidth=2, label="FEWSNET baseline"),
    ]
    if extend_fewsnet:
        legend_handles.append(
            Line2D([0], [0], color=FEWSNET_COLOR, linestyle="-.", marker="^", linewidth=2, label=FEWSNET_REUSED_LABEL)
        )
    fig.suptitle(f"{config['label']} monthly crisis-class performance", fontsize=16)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validation_summary(payload: dict, models: list[str], mode: str, extend_fewsnet: bool = True) -> dict:
    summary = {
        "mode": mode,
        "models": {},
        "fewsnet_fs3_extended": extend_fewsnet,
        "fewsnet_fs3_label": FEWSNET_REUSED_LABEL if extend_fewsnet else None,
    }
    for model_key in models:
        summary["models"][model_key] = {}
        for scope in SCOPES:
            months = payload[model_key][scope]["months"]
            chronological = months == sorted(months, key=lambda m: pd.Period(m, freq="M"))
            subplot_series = {metric: 3 for metric, _ in METRICS}
            summary["models"][model_key][scope] = {
                "months": months,
                "chronological": chronological,
                "series_per_subplot": subplot_series,
            }
    return summary


def make_manifest(
    ablation_root: Path,
    fewsnet_root: Path,
    output_dir: Path,
    models: list[str],
    model_sources: list[str],
    fewsnet_sources: list[str],
    missing_points: list[dict],
    payload: dict,
    mode: str,
    extend_fewsnet: bool = True,
) -> dict:
    generated_artifacts = []
    if mode == "full":
        generated_artifacts = [str(output_dir / MODEL_CONFIG[m]["filename"]) for m in models]
        generated_artifacts.append(str(output_dir / "monthly_performance_manifest.json"))
    return {
        "workflow_mode": "baseline comparison",
        "status": "exploratory diagnostics",
        "entry_point": "scripts/plot_monthly_performance_metrics.py",
        "model_selection": models,
        "source_roots": {"ablation_root": str(ablation_root), "fewsnet_root": str(fewsnet_root)},
        "source_paths": {"model_metrics": model_sources, "fewsnet_baselines": fewsnet_sources},
        "column_contract": {
            "model_metrics": sorted(MODEL_COLUMNS),
            "fewsnet_baselines": sorted(FEWSNET_COLUMNS),
        },
        "scope_contract": {
            "rows": list(SCOPES),
            "fs3_fewsnet_source": "fs2" if extend_fewsnet else None,
        },
        "series_contract": {
            "partitioned": "solid model-color line from model == partitioned",
            "pooled": "dashed same-family model-color line from model == pooled",
            "fewsnet": "distinct comparison color from FEWSNET baseline files",
        },
        "fewsnet_time_alignment": {
            "1": "YYYY-02",
            "2": "YYYY-06",
            "4": "YYYY-10",
            "ignored_quarters": ["3"],
        },
        "fewsnet_fs3_assumption": (
            "FEWSNET has no native fs3 baseline; fs2 values are reused for fs3 as a labeled comparison proxy."
            if extend_fewsnet
            else "FEWSNET fs3 is not plotted."
        ),
        "generated_artifacts": generated_artifacts,
        "missing_points": missing_points,
        "validation_summary": validation_summary(payload, models, mode, extend_fewsnet),
    }


def print_summary(manifest: dict) -> None:
    print("Monthly performance plot validation")
    print(f"Mode: {manifest['validation_summary']['mode']}")
    print(f"Models: {', '.join(manifest['validation_summary']['models'])}")
    print(f"Model selection: {', '.join(manifest['model_selection'])}")
    print(f"FEWSNET fs3 extended: {manifest['validation_summary']['fewsnet_fs3_extended']}")
    print(f"FEWSNET fs3 label: {manifest['validation_summary']['fewsnet_fs3_label']}")
    print(f"Missing plotted points: {len(manifest['missing_points'])}")
    for artifact in manifest["generated_artifacts"]:
        print(f"Generated: {artifact}")


def main() -> int:
    args = parse_args()
    if args.dry_run and args.smoke:
        raise ValueError("Use only one of --dry-run or --smoke")

    ablation_root = resolve_path(args.ablation_root)
    fewsnet_root = resolve_path(args.fewsnet_root)
    output_dir = resolve_path(args.output_dir)
    models = selected_models(args.model)

    model_df, model_sources = load_model_metrics(ablation_root, models)
    baselines, fewsnet_sources = load_fewsnet_baselines(fewsnet_root)
    payload, missing_points = build_plot_payload(
        model_df,
        baselines,
        models,
        extend_fewsnet=args.extend_fewsnet,
    )

    mode = "dry-run" if args.dry_run else "smoke" if args.smoke else "full"
    if args.smoke:
        for model_key in models:
            render_model_figure(model_key, payload, extend_fewsnet=args.extend_fewsnet)

    manifest = make_manifest(
        ablation_root,
        fewsnet_root,
        output_dir,
        models,
        model_sources,
        fewsnet_sources,
        missing_points,
        payload,
        mode,
        extend_fewsnet=args.extend_fewsnet,
    )

    if not args.dry_run and not args.smoke:
        output_dir.mkdir(parents=True, exist_ok=True)
        for model_key in models:
            render_model_figure(
                model_key,
                payload,
                output_dir / MODEL_CONFIG[model_key]["filename"],
                extend_fewsnet=args.extend_fewsnet,
            )
        manifest_path = output_dir / "monthly_performance_manifest.json"
        manifest["generated_artifacts"] = [str(output_dir / MODEL_CONFIG[m]["filename"]) for m in models]
        manifest["generated_artifacts"].append(str(manifest_path))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
