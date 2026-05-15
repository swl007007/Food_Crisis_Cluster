#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_ROOT = Path("main_ablation_results/march2026_main_backup_month_ind_cont3")
DEFAULT_OUTPUT_DIR_NAME = "phase_change_monthly_performance"
STANDARD_OUTPUT_DIR_NAME = "monthly_performance_plots"

MODEL_CONFIG = {
    "georf": {
        "label": "GeoRF",
        "phase_label": "GeoRF(phase change)",
        "token": "GF",
        "color": "#1f77b4",
        "filename": "georf_phase_change_monthly_performance.png",
    },
    "geodt": {
        "label": "GeoDT",
        "phase_label": "GeoDT(phase change)",
        "token": "DT",
        "color": "#d62728",
        "filename": "geodt_phase_change_monthly_performance.png",
    },
}
SCOPES = ("fs1", "fs2", "fs3")
SCOPE_LAG_MONTHS = {"fs1": 4, "fs2": 8, "fs3": 12}
REQUIRED_COLUMNS = (
    "FEWSNET_admin_code",
    "month_start",
    "y_true",
    "y_pred_pooled",
    "y_pred_partitioned",
)
OPTIONAL_AUDIT_COLUMNS = ("partition_id",)
METRICS = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
SERIES_COLUMNS = {
    "partitioned": "y_pred_partitioned",
    "pooled": "y_pred_pooled",
}
OUTPUT_FILES = {
    "monthly_metrics": "metrics_monthly_phase_change.csv",
    "summary": "summary_phase_change.xlsx",
    "audit": "filtered_predictions_phase_change.csv",
    "manifest": "phase_change_manifest.json",
}
SMOKE_PLOT_TEMPLATE = "smoke_{model}_{scope}_phase_change_monthly_performance.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate phase-change-only monthly performance diagnostics from row-level predictions."
    )
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", choices=("georf", "geodt", "all"), default="all")
    parser.add_argument("--scope", choices=("fs1", "fs2", "fs3", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-audit", action=argparse.BooleanOptionalAction, default=True)
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


def relative_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def selected_models(model_arg: str) -> list[str]:
    return list(MODEL_CONFIG) if model_arg == "all" else [model_arg]


def selected_scopes(scope_arg: str) -> list[str]:
    return list(SCOPES) if scope_arg == "all" else [scope_arg]


def validate_output_dir(ablation_root: Path, output_dir: Path) -> None:
    standard_output = (ablation_root / STANDARD_OUTPUT_DIR_NAME).resolve()
    requested = output_dir.resolve()
    if requested == standard_output:
        raise ValueError(f"Output directory must not be the standard diagnostics directory: {standard_output}")


def prediction_path(ablation_root: Path, model_key: str, scope: str) -> Path:
    token = MODEL_CONFIG[model_key]["token"]
    return ablation_root / f"result_partition_k40_compare_{token}_{scope}" / "predictions_monthly.csv"


def discover_included_sources(ablation_root: Path, models: list[str], scopes: list[str]) -> list[dict[str, Any]]:
    sources = []
    for model_key in models:
        for scope in scopes:
            path = prediction_path(ablation_root, model_key, scope)
            if path.exists():
                config = MODEL_CONFIG[model_key]
                sources.append(
                    {
                        "path": path,
                        "model_key": model_key,
                        "model_label": config["label"],
                        "source_token": config["token"],
                        "scope": scope,
                    }
                )
    if not sources:
        raise FileNotFoundError("No GeoRF or GeoDT prediction files found for the requested selection")
    return sources


def discover_excluded_sources(ablation_root: Path) -> list[str]:
    patterns = (
        "result_partition_k40_compare_XGB_fs*/predictions_monthly.csv",
        "result_partition_k40_compare_GeoXGB_fs*/predictions_monthly.csv",
        "*XGBoost*/predictions_monthly.csv",
    )
    paths = set()
    for pattern in patterns:
        paths.update(ablation_root.glob(pattern))
    return [relative_path(path) for path in sorted(paths)]


def require_columns(df: pd.DataFrame, source: Path) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def validate_binary_column(df: pd.DataFrame, column: str, source: Path) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Column {column} contains non-binary or missing values in {source}")
    unique_values = set(values.astype(int).unique())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Column {column} must contain only 0/1 values in {source}; found {sorted(unique_values)}")
    df[column] = values.astype(int)


def validate_and_prepare_source(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    require_columns(df, source)
    df = df.copy()
    parsed_dates = pd.to_datetime(df["month_start"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"Column month_start contains unparseable dates in {source}")
    df["month_start"] = parsed_dates.dt.to_period("M").dt.to_timestamp()
    for column in ("y_true", "y_pred_pooled", "y_pred_partitioned"):
        validate_binary_column(df, column, source)
    return df


def load_sources(sources: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    source_manifest = []
    for source in sources:
        path = source["path"]
        df = pd.read_csv(path)
        df = validate_and_prepare_source(df, path)
        df["model_key"] = source["model_key"]
        df["model_label"] = source["model_label"]
        df["source_token"] = source["source_token"]
        df["scope"] = source["scope"]
        df["source_file"] = relative_path(path)
        frames.append(df)
        source_manifest.append(
            {
                "path": relative_path(path),
                "model_key": source["model_key"],
                "model_label": source["model_label"],
                "source_token": source["source_token"],
                "scope": source["scope"],
                "row_count_before": int(len(df)),
                "months": [m.strftime("%Y-%m-%d") for m in sorted(df["month_start"].unique())],
            }
        )
    return pd.concat(frames, ignore_index=True), source_manifest


def detect_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    keys = ["model_key", "scope", "FEWSNET_admin_code", "month_start"]
    duplicate_mask = df.duplicated(keys, keep=False)
    duplicate_rows = df.loc[duplicate_mask, keys]
    examples = []
    if not duplicate_rows.empty:
        counts = duplicate_rows.groupby(keys).size().reset_index(name="row_count")
        for record in counts.head(25).to_dict("records"):
            record["month_start"] = record["month_start"].strftime("%Y-%m-%d")
            examples.append(record)
    return {
        "has_duplicates": bool(duplicate_mask.any()),
        "duplicate_row_count": int(duplicate_mask.sum()),
        "duplicate_group_count": int(len(examples)) if duplicate_rows.empty else int(duplicate_rows.groupby(keys).ngroups),
        "examples": examples,
    }


def add_phase_change_fields(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["model_key", "scope", "FEWSNET_admin_code", "month_start"]
    df = df.sort_values(sort_columns).reset_index(drop=True).copy()
    groups = df.groupby(["model_key", "scope", "FEWSNET_admin_code"], sort=False)
    df["previous_month_start"] = groups["month_start"].shift(1)
    df["previous_y_true"] = groups["y_true"].shift(1)
    df["is_first_observation"] = df["previous_y_true"].isna()
    df["phase_change"] = (~df["is_first_observation"]) & (df["y_true"] != df["previous_y_true"])
    return df


def build_row_counts(df: pd.DataFrame) -> dict[str, Any]:
    counts = {}
    for (model_key, scope), group in df.groupby(["model_key", "scope"], sort=True):
        before = int(len(group))
        first = int(group["is_first_observation"].sum())
        after = int(group["phase_change"].sum())
        counts[f"{model_key}_{scope}"] = {
            "model_key": model_key,
            "model_label": MODEL_CONFIG[model_key]["label"],
            "scope": scope,
            "before_filter": before,
            "first_observation_excluded": first,
            "non_change_excluded": before - first - after,
            "after_filter": after,
        }
    return counts


def metric_values(y_true: pd.Series, y_pred: pd.Series) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    true = y_true.astype(int)
    pred = y_pred.astype(int)
    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    support = int((true == 1).sum())
    phase_change_rows = int(len(true))
    predicted_positive = tp + fp
    actual_positive = tp + fn
    undefined = []

    precision = np.nan
    if predicted_positive == 0:
        undefined.append(("precision", "zero predicted positive rows (tp + fp = 0)"))
    else:
        precision = tp / predicted_positive

    recall = np.nan
    if actual_positive == 0:
        undefined.append(("recall", "zero actual positive rows (tp + fn = 0)"))
    else:
        recall = tp / actual_positive

    f1 = np.nan
    if np.isnan(precision) or np.isnan(recall):
        undefined.append(("f1", "precision or recall is undefined"))
    elif precision + recall == 0:
        undefined.append(("f1", "zero precision plus recall denominator"))
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return (
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "phase_change_rows": phase_change_rows,
            "true_positive_count": tp,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "support": support,
            "predicted_positive_count": predicted_positive,
        },
        undefined,
    )


def add_zero_denominator_records(
    records: list[dict[str, Any]],
    undefined: list[tuple[str, str]],
    level: str,
    model_key: str,
    model_label: str,
    scope: str,
    series: str,
    month_start: pd.Timestamp | None,
) -> None:
    for metric, reason in undefined:
        record = {
            "level": level,
            "model_key": model_key,
            "model_label": model_label,
            "scope": scope,
            "series": series,
            "metric": metric,
            "reason": reason,
        }
        if month_start is not None:
            record["month_start"] = month_start.strftime("%Y-%m-%d")
        records.append(record)


def compute_monthly_metrics(phase_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = []
    zero_denominator_records = []
    if phase_rows.empty:
        return pd.DataFrame(), zero_denominator_records
    grouping = ["model_key", "model_label", "scope", "month_start"]
    for (model_key, model_label, scope, month_start), group in phase_rows.groupby(grouping, sort=True):
        for series, column in SERIES_COLUMNS.items():
            values, undefined = metric_values(group["y_true"], group[column])
            records.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "scope": scope,
                    "month_start": month_start.strftime("%Y-%m-%d"),
                    "series": series,
                    **values,
                }
            )
            add_zero_denominator_records(
                zero_denominator_records,
                undefined,
                "monthly",
                model_key,
                model_label,
                scope,
                series,
                month_start,
            )
    return pd.DataFrame(records), zero_denominator_records


def compute_summary_metrics(phase_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = []
    zero_denominator_records = []
    if phase_rows.empty:
        return pd.DataFrame(), zero_denominator_records
    grouping = ["model_key", "model_label", "scope"]
    for (model_key, model_label, scope), group in phase_rows.groupby(grouping, sort=True):
        for series, column in SERIES_COLUMNS.items():
            values, undefined = metric_values(group["y_true"], group[column])
            records.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_CONFIG[model_key]["phase_label"],
                    "base_model_label": model_label,
                    "scope": scope,
                    "lag_months": SCOPE_LAG_MONTHS[scope],
                    "series": series,
                    **values,
                }
            )
            add_zero_denominator_records(
                zero_denominator_records,
                undefined,
                "summary",
                model_key,
                MODEL_CONFIG[model_key]["phase_label"],
                scope,
                series,
                None,
            )
    return pd.DataFrame(records), zero_denominator_records


def find_no_phase_change_months(df: pd.DataFrame, phase_rows: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    phase_months = set()
    if not phase_rows.empty:
        phase_months = set(
            (row.model_key, row.scope, row.month_start)
            for row in phase_rows[["model_key", "scope", "month_start"]].drop_duplicates().itertuples(index=False)
        )
    for row in df[["model_key", "scope", "month_start"]].drop_duplicates().itertuples(index=False):
        if (row.model_key, row.scope, row.month_start) not in phase_months:
            records.append(
                {
                    "model_key": row.model_key,
                    "scope": row.scope,
                    "month_start": row.month_start.strftime("%Y-%m-%d"),
                    "reason": "no retained phase-change rows",
                }
            )
    return records


def render_phase_change_plot(
    metrics_df: pd.DataFrame,
    model_key: str,
    scopes: list[str],
    output_path: Path | None = None,
    smoke: bool = False,
) -> None:
    config = MODEL_CONFIG[model_key]
    fig, axes = plt.subplots(len(scopes), len(METRICS), figsize=(17, 3.2 * len(scopes) + 1.4), squeeze=False)
    for row_index, scope in enumerate(scopes):
        scoped = metrics_df[(metrics_df["model_key"] == model_key) & (metrics_df["scope"] == scope)]
        months = sorted(scoped["month_start"].dropna().unique(), key=lambda m: pd.Period(m, freq="M"))
        x = np.arange(len(months))
        for col_index, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row_index][col_index]
            if months:
                for series, linestyle, marker in (("partitioned", "-", "o"), ("pooled", "--", "s")):
                    series_df = scoped[scoped["series"] == series].set_index("month_start")
                    values = [pd.to_numeric(series_df.loc[month, metric], errors="coerce") if month in series_df.index else np.nan for month in months]
                    ax.plot(
                        x,
                        values,
                        color=config["color"],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2,
                        alpha=0.9 if series == "partitioned" else 0.75,
                        label=series,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(months, rotation=45, ha="right")
            else:
                ax.text(0.5, 0.5, "No phase-change rows", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            if row_index == 0:
                ax.set_title(metric_label)
            if col_index == 0:
                ax.set_ylabel(f"{scope}\nMetric value")
            if row_index == len(scopes) - 1:
                ax.set_xlabel("Test month")
    legend_handles = [
        Line2D([0], [0], color=config["color"], linestyle="-", marker="o", linewidth=2, label="partitioned"),
        Line2D([0], [0], color=config["color"], linestyle="--", marker="s", linewidth=2, alpha=0.75, label="pooled"),
    ]
    mode_label = "smoke" if smoke else "exploratory"
    fig.suptitle(f"{config['label']} phase-change-only monthly crisis-class performance ({mode_label})", fontsize=16)
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def prepare_artifact_paths(output_dir: Path, models: list[str], write_audit: bool) -> dict[str, list[str]]:
    full = [str(output_dir / MODEL_CONFIG[model]["filename"]) for model in models]
    full.extend(
        [
            str(output_dir / OUTPUT_FILES["monthly_metrics"]),
            str(output_dir / OUTPUT_FILES["summary"]),
            str(output_dir / OUTPUT_FILES["manifest"]),
        ]
    )
    if write_audit:
        full.append(str(output_dir / OUTPUT_FILES["audit"]))
    return {"full": full}


def serialize_manifest(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serialize_manifest(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_manifest(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_manifest(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_manifest(
    mode: str,
    ablation_root: Path,
    output_dir: Path,
    models: list[str],
    scopes: list[str],
    source_manifest: list[dict[str, Any]],
    excluded_sources: list[str],
    row_counts: dict[str, Any],
    duplicate_flags: dict[str, Any],
    no_phase_change_months: list[dict[str, Any]],
    zero_denominator_metrics: list[dict[str, Any]],
    generated_artifacts: list[str],
    planned_artifacts: list[str],
) -> dict[str, Any]:
    return {
        "workflow_mode": "exploratory diagnostics / baseline-comparison analysis",
        "status": "exploratory",
        "mode": mode,
        "entry_point": "scripts/plot_phase_change_monthly_performance.py",
        "source_root": str(ablation_root),
        "output_dir": str(output_dir),
        "model_selection": models,
        "scope_selection": scopes,
        "included_source_files": source_manifest,
        "excluded_source_files": excluded_sources,
        "column_contract": {
            "required": list(REQUIRED_COLUMNS),
            "optional_audit": list(OPTIONAL_AUDIT_COLUMNS),
            "spatial_key": "FEWSNET_admin_code",
            "test_month": "month_start",
            "true_label": "y_true",
            "prediction_columns": SERIES_COLUMNS,
        },
        "filter_definition": {
            "grouping": ["model_key", "scope", "FEWSNET_admin_code"],
            "ordering": "month_start ascending within each group",
            "previous_period": "previous available test month, not strict previous calendar month",
            "first_observation": "excluded because no previous y_true exists",
            "phase_change": "retain rows where y_true differs from previous_y_true",
        },
        "row_counts": row_counts,
        "duplicate_flags": duplicate_flags,
        "no_phase_change_months": no_phase_change_months,
        "zero_denominator_metrics": zero_denominator_metrics,
        "generated_artifacts": generated_artifacts,
        "planned_artifacts": planned_artifacts,
    }


def print_console_summary(manifest: dict[str, Any], summary_df: pd.DataFrame) -> None:
    print("Phase-change monthly performance analysis")
    print(f"Mode: {manifest['mode']}")
    print(f"Included source files: {len(manifest['included_source_files'])}")
    print(f"Excluded XGB source files: {len(manifest['excluded_source_files'])}")
    print(f"Output directory: {manifest['output_dir']}")
    for key in sorted(manifest["row_counts"]):
        counts = manifest["row_counts"][key]
        print(
            f"{key}: before={counts['before_filter']} first_excluded={counts['first_observation_excluded']} "
            f"non_change_excluded={counts['non_change_excluded']} after={counts['after_filter']}"
        )
    print(f"Duplicate row groups flagged: {manifest['duplicate_flags']['duplicate_group_count']}")
    print(f"Zero-denominator metrics: {len(manifest['zero_denominator_metrics'])}")
    if manifest["mode"] == "dry-run":
        for artifact in manifest["planned_artifacts"]:
            print(f"Planned: {artifact}")
    else:
        for artifact in manifest["generated_artifacts"]:
            print(f"Generated: {artifact}")
    if manifest["mode"] == "smoke" and not summary_df.empty:
        row = summary_df.head(1).replace({np.nan: None}).to_dict("records")[0]
        print("Draft summary row:")
        print(json.dumps(serialize_manifest(row), sort_keys=True))


def write_full_outputs(
    output_dir: Path,
    models: list[str],
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    phase_rows: pd.DataFrame,
    manifest: dict[str, Any],
    write_audit: bool,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_artifacts = []
    for model_key in models:
        path = output_dir / MODEL_CONFIG[model_key]["filename"]
        render_phase_change_plot(metrics_df, model_key, list(SCOPES), path)
        generated_artifacts.append(str(path))
    metrics_path = output_dir / OUTPUT_FILES["monthly_metrics"]
    summary_path = output_dir / OUTPUT_FILES["summary"]
    manifest_path = output_dir / OUTPUT_FILES["manifest"]
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_excel(summary_path, index=False, sheet_name="phase_change_summary")
    generated_artifacts.extend([str(metrics_path), str(summary_path)])
    if write_audit:
        audit_path = output_dir / OUTPUT_FILES["audit"]
        audit_columns = [
            "model_key",
            "model_label",
            "source_token",
            "scope",
            "source_file",
            "FEWSNET_admin_code",
            "month_start",
            "previous_month_start",
            "previous_y_true",
            "is_first_observation",
            "phase_change",
            "y_true",
            "y_pred_pooled",
            "y_pred_partitioned",
        ]
        if "partition_id" in phase_rows.columns:
            audit_columns.insert(7, "partition_id")
        audit_df = phase_rows[audit_columns].copy()
        audit_df["month_start"] = audit_df["month_start"].dt.strftime("%Y-%m-%d")
        audit_df["previous_month_start"] = audit_df["previous_month_start"].dt.strftime("%Y-%m-%d")
        audit_df["previous_y_true"] = audit_df["previous_y_true"].astype(int)
        audit_df.to_csv(audit_path, index=False)
        generated_artifacts.append(str(audit_path))
    manifest["generated_artifacts"] = generated_artifacts + [str(manifest_path)]
    manifest_path.write_text(json.dumps(serialize_manifest(manifest), indent=2), encoding="utf-8")
    generated_artifacts.append(str(manifest_path))
    return generated_artifacts


def run_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.dry_run and args.smoke:
        raise ValueError("Use only one of --dry-run or --smoke")
    if args.smoke and (args.model == "all" or args.scope == "all"):
        raise ValueError("Smoke mode requires one model and one scope, for example --model georf --scope fs1")

    ablation_root = resolve_path(args.ablation_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else ablation_root / DEFAULT_OUTPUT_DIR_NAME
    validate_output_dir(ablation_root, output_dir)
    models = selected_models(args.model)
    scopes = selected_scopes(args.scope)
    mode = "dry-run" if args.dry_run else "smoke" if args.smoke else "full"

    sources = discover_included_sources(ablation_root, models, scopes)
    excluded_sources = discover_excluded_sources(ablation_root)
    source_df, source_manifest = load_sources(sources)
    duplicate_flags = detect_duplicates(source_df)
    prepared_df = add_phase_change_fields(source_df)
    phase_rows = prepared_df[prepared_df["phase_change"]].copy()
    if not phase_rows.empty:
        phase_rows["previous_y_true"] = phase_rows["previous_y_true"].astype(int)
    row_counts = build_row_counts(prepared_df)
    for source in source_manifest:
        key = f"{source['model_key']}_{source['scope']}"
        source.update(row_counts[key])
    metrics_df, monthly_zero_denominators = compute_monthly_metrics(phase_rows)
    summary_df, summary_zero_denominators = compute_summary_metrics(phase_rows)
    zero_denominator_metrics = monthly_zero_denominators + summary_zero_denominators
    no_phase_change_months = find_no_phase_change_months(prepared_df, phase_rows)
    planned_artifacts = prepare_artifact_paths(output_dir, models, args.write_audit)["full"]
    generated_artifacts: list[str] = []

    manifest = build_manifest(
        mode,
        ablation_root,
        output_dir,
        models,
        scopes,
        source_manifest,
        excluded_sources,
        row_counts,
        duplicate_flags,
        no_phase_change_months,
        zero_denominator_metrics,
        generated_artifacts,
        planned_artifacts,
    )

    if mode == "smoke":
        output_dir.mkdir(parents=True, exist_ok=True)
        smoke_path = output_dir / SMOKE_PLOT_TEMPLATE.format(model=models[0], scope=scopes[0])
        render_phase_change_plot(metrics_df, models[0], scopes, smoke_path, smoke=True)
        generated_artifacts.append(str(smoke_path))
        manifest["generated_artifacts"] = generated_artifacts
    elif mode == "full":
        write_full_outputs(output_dir, models, metrics_df, summary_df, phase_rows, manifest, args.write_audit)

    return manifest, metrics_df, summary_df, phase_rows


def main() -> int:
    args = parse_args()
    manifest, _, summary_df, _ = run_analysis(args)
    print_console_summary(manifest, summary_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
