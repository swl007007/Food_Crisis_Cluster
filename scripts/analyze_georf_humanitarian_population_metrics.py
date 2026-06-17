#!/usr/bin/env python3
"""Build GeoRF population-weighted humanitarian metric diagnostics."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from paper_horizon_labels import HORIZON_LABELS
except ModuleNotFoundError:
    from scripts.paper_horizon_labels import HORIZON_LABELS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEWSNET = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWSNET.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated" / "09_humanitarian_metrics"
HORIZONS = HORIZON_LABELS
MODEL_PRED_COLUMNS = {
    "pooled": "y_pred_pooled",
    "partitioned": "y_pred_partitioned",
}
POPULATION_METRICS = [
    "population_at_risk",
    "true_alert_population",
    "missed_crisis_population",
    "false_alert_population",
    "true_noncrisis_population",
    "population_weighted_recall",
    "population_weighted_precision",
]


def resolve_path(path: Path, platform_name: str | None = None) -> Path:
    """Resolve Windows paths when running from WSL."""
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


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def load_population_lookup(fewsnet_path: Path) -> pd.DataFrame:
    """Load raw FEWSNET population by admin code and target month."""
    path = resolve_path(fewsnet_path)
    raw = pd.read_csv(path, usecols=["admin_code", "year", "month", "pop"])
    raw["admin_code"] = normalize_admin_code(raw["admin_code"])
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["month"] = pd.to_numeric(raw["month"], errors="coerce")
    raw["month_start"] = pd.to_datetime(
        {"year": raw["year"], "month": raw["month"], "day": 1},
        errors="coerce",
    )
    raw["pop"] = pd.to_numeric(raw["pop"], errors="coerce")
    lookup = raw[["admin_code", "month_start", "pop"]].dropna(subset=["admin_code", "month_start"]).copy()

    duplicated = lookup.duplicated(["admin_code", "month_start"], keep=False)
    if bool(duplicated.any()):
        examples = lookup.loc[duplicated, ["admin_code", "month_start"]].head(5)
        raise ValueError(f"Duplicate population keys in {path}: {examples.to_dict('records')}")
    return lookup.reset_index(drop=True)


def load_prediction_files(source_dir: Path, scopes: list[str]) -> pd.DataFrame:
    """Load GeoRF Stage 3 prediction files."""
    frames = []
    required = {
        "FEWSNET_admin_code",
        "month_start",
        "y_true",
        "y_pred_pooled",
        "y_pred_partitioned",
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


def join_population(predictions: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Attach population to predictions and require full coverage."""
    pred = predictions.copy()
    pred["admin_code"] = normalize_admin_code(pred["FEWSNET_admin_code"])
    pop = population.copy()
    pop["admin_code"] = normalize_admin_code(pop["admin_code"])
    merged = pred.merge(pop, on=["admin_code", "month_start"], how="left", validate="many_to_one")
    missing = merged["pop"].isna()
    if bool(missing.any()):
        examples = merged.loc[missing, ["admin_code", "month_start"]].head(10)
        raise ValueError(f"Missing population for {int(missing.sum())} prediction rows: {examples.to_dict('records')}")
    if bool((merged["pop"] < 0).any()):
        raise ValueError("Population contains negative values after join")
    return merged


def compute_population_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Compute population-month confusion totals and weighted precision/recall."""
    valid = df[["y_true", pred_col, "pop"]].notna().all(axis=1)
    sub = df.loc[valid].copy()
    if sub.empty:
        return {
            "support": 0,
            "population_at_risk": 0.0,
            "true_alert_population": 0.0,
            "missed_crisis_population": 0.0,
            "false_alert_population": 0.0,
            "true_noncrisis_population": 0.0,
            "population_weighted_recall": np.nan,
            "population_weighted_precision": np.nan,
        }

    y_true = pd.to_numeric(sub["y_true"], errors="coerce").astype(int)
    y_pred = pd.to_numeric(sub[pred_col], errors="coerce").astype(int)
    pop = pd.to_numeric(sub["pop"], errors="coerce").astype(float)

    true_alert = float(pop[(y_true == 1) & (y_pred == 1)].sum())
    missed_crisis = float(pop[(y_true == 1) & (y_pred == 0)].sum())
    false_alert = float(pop[(y_true == 0) & (y_pred == 1)].sum())
    true_noncrisis = float(pop[(y_true == 0) & (y_pred == 0)].sum())
    return {
        "support": int(len(sub)),
        "population_at_risk": float(pop.sum()),
        "true_alert_population": true_alert,
        "missed_crisis_population": missed_crisis,
        "false_alert_population": false_alert,
        "true_noncrisis_population": true_noncrisis,
        "population_weighted_recall": safe_divide(true_alert, true_alert + missed_crisis),
        "population_weighted_precision": safe_divide(true_alert, true_alert + false_alert),
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build long-format horizon-level population metric table."""
    rows = []
    for (scope, horizon), sub in df.groupby(["scope", "forecasting_horizon"], sort=True):
        for model, pred_col in MODEL_PRED_COLUMNS.items():
            metrics = compute_population_metrics(sub, pred_col)
            rows.append({"scope": scope, "forecasting_horizon": horizon, "model": model, **metrics})
    return pd.DataFrame(rows)


def build_month_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build long-format month-level population metric table."""
    rows = []
    for (scope, horizon, month_start), sub in df.groupby(["scope", "forecasting_horizon", "month_start"], sort=True):
        for model, pred_col in MODEL_PRED_COLUMNS.items():
            metrics = compute_population_metrics(sub, pred_col)
            rows.append(
                {
                    "scope": scope,
                    "forecasting_horizon": horizon,
                    "month_start": month_start.strftime("%Y-%m-%d"),
                    "model": model,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-horizon compact table with pooled, partitioned, and delta values."""
    rows = []
    for (scope, horizon), sub in summary.groupby(["scope", "forecasting_horizon"], sort=True):
        pooled = sub[sub["model"] == "pooled"].iloc[0]
        partitioned = sub[sub["model"] == "partitioned"].iloc[0]
        row: dict[str, object] = {"scope": scope, "forecasting_horizon": horizon}
        for metric in [
            "missed_crisis_population",
            "false_alert_population",
            "population_weighted_recall",
            "population_weighted_precision",
        ]:
            row[f"pooled_{metric}"] = pooled[metric]
            row[f"partitioned_{metric}"] = partitioned[metric]
            row[f"delta_{metric}"] = partitioned[metric] - pooled[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write a Markdown table without optional pandas dependencies."""
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_compact_for_paper(compact: pd.DataFrame) -> pd.DataFrame:
    """Round compact table for paper-facing Markdown output."""
    table = compact.copy()
    population_columns = [column for column in table.columns if column.endswith("_population")]
    rate_columns = [column for column in table.columns if column.endswith("_recall") or column.endswith("_precision")]
    for column in population_columns:
        table[column] = table[column].map(lambda value: f"{value:,.0f}")
    for column in rate_columns:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    return table


def plot_population_bars(summary: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    """Plot missed-crisis and false-alert population totals by horizon and model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = list(dict.fromkeys(summary["forecasting_horizon"].tolist()))
    metrics = [
        ("missed_crisis_population", "Missed-crisis population-months"),
        ("false_alert_population", "False-alert population-months"),
    ]
    colors = {"pooled": "#4c78a8", "partitioned": "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    x = np.arange(len(horizons))
    width = 0.34
    for ax, (metric, title) in zip(axes, metrics):
        for offset, model in [(-width / 2, "pooled"), (width / 2, "partitioned")]:
            values = []
            for horizon in horizons:
                row = summary[(summary["forecasting_horizon"] == horizon) & (summary["model"] == model)].iloc[0]
                values.append(row[metric] / 1_000_000)
            ax.bar(x + offset, values, width=width, label=model, color=colors[model])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylabel("Population-months (millions)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("GeoRF Population-Weighted Humanitarian Metrics", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_note(output_dir: Path) -> Path:
    """Write reviewer-facing Chinese note and English appendix text."""
    path = output_dir / "georf_humanitarian_population_note.md"
    path.write_text(
        "\n".join(
            [
                "# GeoRF Population-Weighted Humanitarian Metrics",
                "",
                "中文审查说明：",
                "",
                "该 appendix 只针对 GeoRF pooled 和 partitioned/local RF 模型。",
                "Population 来自 raw FEWSNET.csv 的 `pop` 字段，并按 `admin_code` 和 evaluated target month 与 Stage 3 polygon-month predictions 合并。",
                "所有 population totals 都是 evaluated polygon-month observations 上的 population-month totals，不解释为 unique affected people。",
                "Missed-crisis population 定义为实际 crisis 但模型预测 non-crisis 的 population-month 总和。",
                "False-alert population 定义为实际 non-crisis 但模型预测 crisis 的 population-month 总和。",
                "Population-weighted recall 使用 true-alert population 除以实际 crisis population；population-weighted precision 使用 true-alert population 除以预测 crisis population。",
                "表中的 delta 定义为 partitioned minus pooled；missed-crisis / false-alert population 的负 delta 表示 partitioned 更低，recall / precision 的正 delta 表示 partitioned 更高。",
                "",
                "Appendix text (English):",
                "",
                "We report population-weighted humanitarian diagnostics for the GeoRF pooled and partitioned RF models.",
                "Population is taken from the raw FEWSNET `pop` field and merged to evaluated polygon-month predictions by FEWSNET admin code and target month.",
                "Population totals are population-month totals over evaluated polygon-month observations, not estimates of unique affected people.",
                "Missed-crisis population is the population in observations where a crisis occurred but the model predicted non-crisis.",
                "False-alert population is the population in observations where no crisis occurred but the model predicted crisis.",
                "Population-weighted recall divides true-alert population by the total actual-crisis population, and population-weighted precision divides true-alert population by the total predicted-crisis population.",
                "Deltas are reported as partitioned minus pooled.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--fewsnet", type=Path, default=DEFAULT_FEWSNET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scopes", nargs="+", default=["fs1", "fs2", "fs3"], choices=["fs1", "fs2", "fs3"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_prediction_files(args.source_dir, args.scopes)
    population = load_population_lookup(args.fewsnet)
    df = join_population(predictions, population)
    summary = build_summary_table(df)
    by_month = build_month_table(df)
    compact = build_compact_table(summary)
    compact_paper = format_compact_for_paper(compact)

    summary.to_csv(args.output_dir / "georf_humanitarian_population_summary.csv", index=False)
    by_month.to_csv(args.output_dir / "georf_humanitarian_population_by_month.csv", index=False)
    compact.to_csv(args.output_dir / "georf_humanitarian_population_compact_table.csv", index=False)
    write_markdown_table(compact_paper, args.output_dir / "georf_humanitarian_population_compact_table.md")
    plot_population_bars(summary, args.output_dir / "georf_humanitarian_population_bars.png", dpi=args.dpi)
    write_note(args.output_dir)

    print(f"Wrote GeoRF humanitarian population metrics to {args.output_dir}")
    print(f"Rows: predictions={len(predictions)}, joined={len(df)}, summary={len(summary)}, month={len(by_month)}")


if __name__ == "__main__":
    main()
