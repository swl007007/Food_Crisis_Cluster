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
STANDARD_OUTPUT_DIR_NAME = "monthly_performance_plots"

MODEL_CONFIG = {
    "georf": {
        "label": "GeoRF",
        "token": "GF",
        "color": "#1f77b4",
        "summary_labels": {
            "any_phase_change": "GeoRF(phase change)",
            "crisis_onset": "GeoRF(crisis onset)",
        },
    },
    "geodt": {
        "label": "GeoDT",
        "token": "DT",
        "color": "#d62728",
        "summary_labels": {
            "any_phase_change": "GeoDT(phase change)",
            "crisis_onset": "GeoDT(crisis onset)",
        },
    },
}
FILTER_MODE_CONFIG = {
    "any_phase_change": {
        "label": "phase-change-only",
        "short_label": "phase change",
        "retained_column": "any_phase_change",
        "default_output_dir_name": "phase_change_monthly_performance",
        "standard_forbidden_dirs": {STANDARD_OUTPUT_DIR_NAME, "crisis_onset_analysis"},
        "model_plot_filenames": {
            "georf": "georf_phase_change_monthly_performance.png",
            "geodt": "geodt_phase_change_monthly_performance.png",
        },
        "monthly_metrics_filename": "metrics_monthly_phase_change.csv",
        "summary_filename": "summary_phase_change.xlsx",
        "audit_filename": "filtered_predictions_phase_change.csv",
        "manifest_filename": "phase_change_manifest.json",
        "smoke_plot_template": "smoke_{model}_{scope}_phase_change_monthly_performance.png",
        "summary_sheet_name": "phase_change_summary",
        "row_count_field": "phase_change_rows",
        "excluded_count_field": "non_change_excluded",
        "excluded_count_label": "non_change_excluded",
        "no_rows_manifest_key": "no_phase_change_months",
        "no_rows_reason": "no retained phase-change rows",
        "filter_definition": "retain rows where y_true differs from previous_y_true",
        "plot_title_label": "phase-change-only",
        "plot_empty_text": "No phase-change rows",
    },
    "crisis_onset": {
        "label": "crisis-onset-only",
        "short_label": "crisis onset",
        "retained_column": "crisis_onset",
        "default_output_dir_name": "crisis_onset_analysis",
        "standard_forbidden_dirs": {STANDARD_OUTPUT_DIR_NAME, "phase_change_monthly_performance"},
        "model_plot_filenames": {
            "georf": "georf_crisis_onset_monthly_performance.png",
            "geodt": "geodt_crisis_onset_monthly_performance.png",
        },
        "monthly_metrics_filename": "metrics_monthly_crisis_onset.csv",
        "summary_filename": "summary_crisis_onset.xlsx",
        "audit_filename": "filtered_predictions_crisis_onset.csv",
        "manifest_filename": "crisis_onset_manifest.json",
        "smoke_plot_template": "smoke_{model}_{scope}_crisis_onset_monthly_performance.png",
        "summary_sheet_name": "crisis_onset_summary",
        "row_count_field": "retained_rows",
        "excluded_count_field": "non_onset_excluded",
        "excluded_count_label": "non_onset_excluded",
        "no_rows_manifest_key": "no_crisis_onset_months",
        "no_rows_reason": "no retained crisis-onset rows",
        "filter_definition": "retain rows where previous_y_true = 0 and y_true = 1",
        "plot_title_label": "crisis-onset-only",
        "plot_empty_text": "No crisis-onset rows",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate transition-filtered monthly performance diagnostics from row-level predictions."
    )
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--filter-mode", choices=tuple(FILTER_MODE_CONFIG), default="any_phase_change")
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


def mode_config(filter_mode: str) -> dict[str, Any]:
    return FILTER_MODE_CONFIG[filter_mode]


def default_output_dir(ablation_root: Path, filter_mode: str) -> Path:
    return ablation_root / mode_config(filter_mode)["default_output_dir_name"]


def validate_output_dir(ablation_root: Path, output_dir: Path, filter_mode: str) -> None:
    requested = output_dir.resolve()
    for dirname in mode_config(filter_mode)["standard_forbidden_dirs"]:
        forbidden = (ablation_root / dirname).resolve()
        if requested == forbidden:
            raise ValueError(f"Output directory is not valid for filter mode {filter_mode}: {forbidden}")


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


def add_filter_fields(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["model_key", "scope", "FEWSNET_admin_code", "month_start"]
    df = df.sort_values(sort_columns).reset_index(drop=True).copy()
    groups = df.groupby(["model_key", "scope", "FEWSNET_admin_code"], sort=False)
    df["previous_month_start"] = groups["month_start"].shift(1)
    df["previous_y_true"] = groups["y_true"].shift(1)
    df["is_first_observation"] = df["previous_y_true"].isna()
    comparable = ~df["is_first_observation"]
    df["any_phase_change"] = comparable & (df["y_true"] != df["previous_y_true"])
    df["crisis_onset"] = comparable & (df["previous_y_true"] == 0) & (df["y_true"] == 1)
    df["crisis_recovery"] = comparable & (df["previous_y_true"] == 1) & (df["y_true"] == 0)
    df["phase_change"] = df["any_phase_change"]
    return df


def retained_rows_for_mode(df: pd.DataFrame, filter_mode: str) -> pd.DataFrame:
    retained_column = mode_config(filter_mode)["retained_column"]
    retained_rows = df[df[retained_column]].copy()
    retained_rows["filter_mode"] = filter_mode
    retained_rows["retained_by_filter"] = True
    if not retained_rows.empty:
        retained_rows["previous_y_true"] = retained_rows["previous_y_true"].astype(int)
    return retained_rows


def build_row_counts(df: pd.DataFrame, filter_mode: str) -> dict[str, Any]:
    config = mode_config(filter_mode)
    retained_column = config["retained_column"]
    excluded_field = config["excluded_count_field"]
    counts = {}
    for (model_key, scope), group in df.groupby(["model_key", "scope"], sort=True):
        before = int(len(group))
        first = int(group["is_first_observation"].sum())
        retained = int(group[retained_column].sum())
        non_retained = before - first - retained
        record = {
            "model_key": model_key,
            "model_label": MODEL_CONFIG[model_key]["label"],
            "scope": scope,
            "filter_mode": filter_mode,
            "before_filter": before,
            "first_observation_excluded": first,
            excluded_field: non_retained,
            "retained_rows": retained,
            "after_filter": retained,
        }
        if filter_mode == "any_phase_change":
            record["phase_change_rows"] = retained
        if filter_mode == "crisis_onset":
            record["crisis_recovery_excluded"] = int(group["crisis_recovery"].sum())
            record["stable_state_excluded"] = int((~group["is_first_observation"] & ~group["any_phase_change"]).sum())
        counts[f"{model_key}_{scope}"] = record
    return counts


def metric_values(y_true: pd.Series, y_pred: pd.Series, filter_mode: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    true = y_true.astype(int)
    pred = y_pred.astype(int)
    tp = int(((true == 1) & (pred == 1)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    support = int((true == 1).sum())
    retained_count = int(len(true))
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

    values = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "support": support,
        "predicted_positive_count": predicted_positive,
    }
    values[mode_config(filter_mode)["row_count_field"]] = retained_count
    if filter_mode == "crisis_onset":
        values["retained_rows"] = retained_count
    return values, undefined


def add_zero_denominator_records(
    records: list[dict[str, Any]],
    undefined: list[tuple[str, str]],
    level: str,
    filter_mode: str,
    model_key: str,
    model_label: str,
    scope: str,
    series: str,
    month_start: pd.Timestamp | None,
) -> None:
    for metric, reason in undefined:
        record = {
            "level": level,
            "filter_mode": filter_mode,
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


def compute_monthly_metrics(retained_rows: pd.DataFrame, filter_mode: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = []
    zero_denominator_records = []
    if retained_rows.empty:
        return pd.DataFrame(), zero_denominator_records
    grouping = ["model_key", "model_label", "scope", "month_start"]
    for (model_key, model_label, scope, month_start), group in retained_rows.groupby(grouping, sort=True):
        for series, column in SERIES_COLUMNS.items():
            values, undefined = metric_values(group["y_true"], group[column], filter_mode)
            records.append(
                {
                    "filter_mode": filter_mode,
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
                filter_mode,
                model_key,
                model_label,
                scope,
                series,
                month_start,
            )
    return pd.DataFrame(records), zero_denominator_records


def compute_summary_metrics(retained_rows: pd.DataFrame, filter_mode: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = []
    zero_denominator_records = []
    if retained_rows.empty:
        return pd.DataFrame(), zero_denominator_records
    grouping = ["model_key", "model_label", "scope"]
    for (model_key, model_label, scope), group in retained_rows.groupby(grouping, sort=True):
        for series, column in SERIES_COLUMNS.items():
            values, undefined = metric_values(group["y_true"], group[column], filter_mode)
            summary_label = MODEL_CONFIG[model_key]["summary_labels"][filter_mode]
            records.append(
                {
                    "filter_mode": filter_mode,
                    "model_key": model_key,
                    "model_label": summary_label,
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
                filter_mode,
                model_key,
                summary_label,
                scope,
                series,
                None,
            )
    return pd.DataFrame(records), zero_denominator_records


def find_no_retained_months(df: pd.DataFrame, retained_rows: pd.DataFrame, filter_mode: str) -> list[dict[str, Any]]:
    records = []
    retained_months = set()
    if not retained_rows.empty:
        retained_months = set(
            (row.model_key, row.scope, row.month_start)
            for row in retained_rows[["model_key", "scope", "month_start"]].drop_duplicates().itertuples(index=False)
        )
    for row in df[["model_key", "scope", "month_start"]].drop_duplicates().itertuples(index=False):
        if (row.model_key, row.scope, row.month_start) not in retained_months:
            records.append(
                {
                    "filter_mode": filter_mode,
                    "model_key": row.model_key,
                    "scope": row.scope,
                    "month_start": row.month_start.strftime("%Y-%m-%d"),
                    "reason": mode_config(filter_mode)["no_rows_reason"],
                }
            )
    return records


def render_transition_plot(
    metrics_df: pd.DataFrame,
    model_key: str,
    scopes: list[str],
    filter_mode: str,
    output_path: Path | None = None,
    smoke: bool = False,
) -> None:
    model = MODEL_CONFIG[model_key]
    config = mode_config(filter_mode)
    fig, axes = plt.subplots(len(scopes), len(METRICS), figsize=(17, 3.2 * len(scopes) + 1.4), squeeze=False)
    for row_index, scope in enumerate(scopes):
        scoped = metrics_df[(metrics_df["model_key"] == model_key) & (metrics_df["scope"] == scope)]
        months = sorted(scoped["month_start"].dropna().unique(), key=lambda m: pd.Period(m, freq="M")) if not scoped.empty else []
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
                        color=model["color"],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2,
                        alpha=0.9 if series == "partitioned" else 0.75,
                        label=series,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(months, rotation=45, ha="right")
            else:
                ax.text(0.5, 0.5, config["plot_empty_text"], ha="center", va="center", transform=ax.transAxes)
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
        Line2D([0], [0], color=model["color"], linestyle="-", marker="o", linewidth=2, label="partitioned"),
        Line2D([0], [0], color=model["color"], linestyle="--", marker="s", linewidth=2, alpha=0.75, label="pooled"),
    ]
    mode_label = "smoke" if smoke else "exploratory"
    fig.suptitle(f"{model['label']} {config['plot_title_label']} monthly crisis-class performance ({mode_label})", fontsize=16)
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def prepare_artifact_paths(output_dir: Path, models: list[str], write_audit: bool, filter_mode: str) -> dict[str, list[str]]:
    config = mode_config(filter_mode)
    full = [str(output_dir / config["model_plot_filenames"][model]) for model in models]
    full.extend(
        [
            str(output_dir / config["monthly_metrics_filename"]),
            str(output_dir / config["summary_filename"]),
            str(output_dir / config["manifest_filename"]),
        ]
    )
    if write_audit:
        full.append(str(output_dir / config["audit_filename"]))
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
    filter_mode: str,
    ablation_root: Path,
    output_dir: Path,
    models: list[str],
    scopes: list[str],
    source_manifest: list[dict[str, Any]],
    row_counts: dict[str, Any],
    duplicate_flags: dict[str, Any],
    no_retained_months: list[dict[str, Any]],
    zero_denominator_metrics: list[dict[str, Any]],
    generated_artifacts: list[str],
    planned_artifacts: list[str],
) -> dict[str, Any]:
    config = mode_config(filter_mode)
    manifest = {
        "workflow_mode": "exploratory diagnostics / baseline-comparison analysis",
        "status": "exploratory",
        "mode": mode,
        "filter_mode": filter_mode,
        "filter_label": config["label"],
        "entry_point": "scripts/plot_phase_change_monthly_performance.py",
        "source_root": str(ablation_root),
        "output_dir": str(output_dir),
        "output_profile": {
            "default_output_dir_name": config["default_output_dir_name"],
            "monthly_metrics_filename": config["monthly_metrics_filename"],
            "summary_filename": config["summary_filename"],
            "audit_filename": config["audit_filename"],
            "manifest_filename": config["manifest_filename"],
        },
        "model_selection": models,
        "scope_selection": scopes,
        "included_source_files": source_manifest,
        "column_contract": {
            "required": list(REQUIRED_COLUMNS),
            "optional_audit": list(OPTIONAL_AUDIT_COLUMNS),
            "spatial_key": "FEWSNET_admin_code",
            "test_month": "month_start",
            "true_label": "y_true",
            "prediction_columns": SERIES_COLUMNS,
        },
        "filter_definition": {
            "mode": filter_mode,
            "grouping": ["model_key", "scope", "FEWSNET_admin_code"],
            "ordering": "month_start ascending within each group",
            "previous_period": "previous available test month, not strict previous calendar month",
            "first_observation": "excluded because no previous y_true exists",
            "retained_rows": config["filter_definition"],
        },
        "row_counts": row_counts,
        "duplicate_flags": duplicate_flags,
        config["no_rows_manifest_key"]: no_retained_months,
        "zero_denominator_metrics": zero_denominator_metrics,
        "generated_artifacts": generated_artifacts,
        "planned_artifacts": planned_artifacts,
    }
    if filter_mode == "any_phase_change":
        manifest["no_retained_months"] = no_retained_months
    return manifest


def print_console_summary(manifest: dict[str, Any], summary_df: pd.DataFrame) -> None:
    print("Monthly performance transition analysis")
    print(f"Mode: {manifest['mode']}")
    print(f"Filter mode: {manifest['filter_mode']}")
    print(f"Included source files: {len(manifest['included_source_files'])}")
    print(f"Model selection: {', '.join(manifest['model_selection'])}")
    print(f"Output directory: {manifest['output_dir']}")
    for key in sorted(manifest["row_counts"]):
        counts = manifest["row_counts"][key]
        extra = ""
        if manifest["filter_mode"] == "crisis_onset":
            extra = f" crisis_recovery_excluded={counts['crisis_recovery_excluded']}"
        print(
            f"{key}: before={counts['before_filter']} first_excluded={counts['first_observation_excluded']} "
            f"{FILTER_MODE_CONFIG[manifest['filter_mode']]['excluded_count_label']}={counts[FILTER_MODE_CONFIG[manifest['filter_mode']]['excluded_count_field']]} "
            f"retained={counts['retained_rows']}{extra}"
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
    retained_rows: pd.DataFrame,
    manifest: dict[str, Any],
    write_audit: bool,
    filter_mode: str,
) -> list[str]:
    config = mode_config(filter_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_artifacts = []
    for model_key in models:
        path = output_dir / config["model_plot_filenames"][model_key]
        render_transition_plot(metrics_df, model_key, list(SCOPES), filter_mode, path)
        generated_artifacts.append(str(path))
    metrics_path = output_dir / config["monthly_metrics_filename"]
    summary_path = output_dir / config["summary_filename"]
    manifest_path = output_dir / config["manifest_filename"]
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_excel(summary_path, index=False, sheet_name=config["summary_sheet_name"])
    generated_artifacts.extend([str(metrics_path), str(summary_path)])
    if write_audit:
        audit_path = output_dir / config["audit_filename"]
        audit_columns = [
            "model_key",
            "model_label",
            "source_token",
            "scope",
            "source_file",
            "filter_mode",
            "retained_by_filter",
            "FEWSNET_admin_code",
            "month_start",
            "previous_month_start",
            "previous_y_true",
            "is_first_observation",
            "any_phase_change",
            "crisis_onset",
            "crisis_recovery",
            "phase_change",
            "y_true",
            "y_pred_pooled",
            "y_pred_partitioned",
        ]
        if "partition_id" in retained_rows.columns:
            audit_columns.insert(9, "partition_id")
        audit_df = retained_rows[audit_columns].copy()
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

    filter_mode = args.filter_mode
    ablation_root = resolve_path(args.ablation_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else default_output_dir(ablation_root, filter_mode)
    validate_output_dir(ablation_root, output_dir, filter_mode)
    models = selected_models(args.model)
    scopes = selected_scopes(args.scope)
    mode = "dry-run" if args.dry_run else "smoke" if args.smoke else "full"

    sources = discover_included_sources(ablation_root, models, scopes)
    source_df, source_manifest = load_sources(sources)
    duplicate_flags = detect_duplicates(source_df)
    prepared_df = add_filter_fields(source_df)
    retained_rows = retained_rows_for_mode(prepared_df, filter_mode)
    row_counts = build_row_counts(prepared_df, filter_mode)
    for source in source_manifest:
        key = f"{source['model_key']}_{source['scope']}"
        source.update(row_counts[key])
    metrics_df, monthly_zero_denominators = compute_monthly_metrics(retained_rows, filter_mode)
    summary_df, summary_zero_denominators = compute_summary_metrics(retained_rows, filter_mode)
    zero_denominator_metrics = monthly_zero_denominators + summary_zero_denominators
    no_retained_months = find_no_retained_months(prepared_df, retained_rows, filter_mode)
    planned_artifacts = prepare_artifact_paths(output_dir, models, args.write_audit, filter_mode)["full"]
    generated_artifacts: list[str] = []

    manifest = build_manifest(
        mode,
        filter_mode,
        ablation_root,
        output_dir,
        models,
        scopes,
        source_manifest,
        row_counts,
        duplicate_flags,
        no_retained_months,
        zero_denominator_metrics,
        generated_artifacts,
        planned_artifacts,
    )

    if mode == "smoke":
        output_dir.mkdir(parents=True, exist_ok=True)
        smoke_path = output_dir / mode_config(filter_mode)["smoke_plot_template"].format(model=models[0], scope=scopes[0])
        render_transition_plot(metrics_df, models[0], scopes, filter_mode, smoke_path, smoke=True)
        generated_artifacts.append(str(smoke_path))
        manifest["generated_artifacts"] = generated_artifacts
    elif mode == "full":
        write_full_outputs(output_dir, models, metrics_df, summary_df, retained_rows, manifest, args.write_audit, filter_mode)

    return manifest, metrics_df, summary_df, retained_rows


def main() -> int:
    args = parse_args()
    manifest, _, summary_df, _ = run_analysis(args)
    print_console_summary(manifest, summary_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
