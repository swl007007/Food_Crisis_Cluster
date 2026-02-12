from __future__ import annotations

import argparse
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    results_subdir: str
    results_pattern: str
    variant: str


MODEL_CONFIG: dict[str, ModelConfig] = {
    "georf": ModelConfig(
        model_type="georf",
        results_subdir="GeoRFResults",
        results_pattern="results_df_gp_fs*.csv",
        variant="GeoRF",
    ),
    "geoxgb": ModelConfig(
        model_type="geoxgb",
        results_subdir="GeoXgboostResults",
        results_pattern="results_df_xgb_gp_fs*.csv",
        variant="GeoXGB",
    ),
    "geodt": ModelConfig(
        model_type="geodt",
        results_subdir="GeoDTResults",
        results_pattern="results_df_dt_gp_fs*.csv",
        variant="GeoDT",
    ),
}

RESULTS_FILENAME_RE = re.compile(r"fs(\d+)_(\d{4})_(\d{4})")
RESULT_DIR_RE = re.compile(
    r"result_Geo(?:RF|XGB|DT)_(\d{4})_fs(\d+)_(\d{4})-(\d{2})_visual"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge correspondence tables with model F1 metrics."
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory (GeoRFExperiment, GeoXGBExperiment, GeoDTExperiment).",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(MODEL_CONFIG.keys()),
        help="Model type for path and naming rules.",
    )
    return parser.parse_args()


def resolve_paths(experiment_dir: str, config: ModelConfig) -> tuple[Path, Path]:
    exp_dir = Path(experiment_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    results_dir = exp_dir / config.results_subdir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    return exp_dir, results_dir


def extract_results_metadata(file_path: Path, variant: str) -> tuple[str, str, str]:
    match = RESULTS_FILENAME_RE.search(file_path.name)
    if not match:
        raise ValueError(f"Cannot parse forecasting scope/year from: {file_path.name}")
    forecasting_scope = f"fs{match.group(1)}"
    year = match.group(2)
    return variant, forecasting_scope, year


def load_results_table(results_dir: Path, config: ModelConfig) -> pd.DataFrame:
    files = sorted(results_dir.glob(config.results_pattern))
    if not files:
        raise FileNotFoundError(
            f"No results files found in {results_dir} matching {config.results_pattern}"
        )

    print(f"Found {len(files)} results files")
    required_columns = ["year", "month", "f1(1)", "f1_base(1)"]
    frames: list[pd.DataFrame] = []

    for file_path in files:
        model, forecasting_scope, _ = extract_results_metadata(file_path, config.variant)
        df = pd.read_csv(file_path)
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {file_path}")

        selected = df[required_columns].copy()
        selected["model"] = model
        selected["forecasting_scope"] = forecasting_scope
        selected = selected[
            ["model", "year", "month", "forecasting_scope", "f1(1)", "f1_base(1)"]
        ]
        frames.append(pd.DataFrame(selected))
        print(f"Loaded {file_path.name}: {len(selected)} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["year"].astype(str)
    combined["month"] = combined["month"].astype(str).str.zfill(2)
    combined["forecasting_scope"] = combined["forecasting_scope"].astype(str)
    return combined


def extract_corr_metadata(file_path: Path, variant: str) -> tuple[str, str, str, str]:
    match = RESULT_DIR_RE.search(str(file_path))
    if not match:
        raise ValueError(f"Cannot parse metadata from correspondence path: {file_path}")
    year = match.group(1)
    forecasting_scope = f"fs{match.group(2)}"
    month = match.group(4)
    return variant, year, month, forecasting_scope


def load_correspondence_tables(results_dir: Path, variant: str) -> list[dict[str, Any]]:
    corr_files = sorted(results_dir.glob("**/correspondence_table_*.csv"))
    if not corr_files:
        raise FileNotFoundError(f"No correspondence tables found in: {results_dir}")

    tables: list[dict[str, Any]] = []
    for file_path in corr_files:
        item_variant, year, month, forecasting_scope = extract_corr_metadata(
            file_path, variant
        )
        name = f"{item_variant}_{year}_{month}_{forecasting_scope}"
        df = pd.read_csv(file_path)
        tables.append(
            {
                "name": name,
                "variant": item_variant,
                "year": year,
                "month": month,
                "forecasting_scope": forecasting_scope,
                "file_path": str(file_path),
                "dataframe": df,
            }
        )
    print(f"Loaded {len(tables)} correspondence tables")
    return tables


def merge_results_with_correspondence(
    results_df: pd.DataFrame, correspondence_tables: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    merged: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []

    for item in correspondence_tables:
        variant = item["variant"]
        year = item["year"]
        month = item["month"]
        forecasting_scope = item["forecasting_scope"]
        corr_df = item["dataframe"].copy()

        matched = results_df[
            (results_df["model"] == variant)
            & (results_df["year"] == year)
            & (results_df["month"] == month)
            & (results_df["forecasting_scope"] == forecasting_scope)
        ]
        matched_df = pd.DataFrame(matched)
        n_matches = len(matched_df)

        if n_matches == 1:
            f1_value = float(matched_df["f1(1)"].iloc[0])
            f1_base_value = float(matched_df["f1_base(1)"].iloc[0])
            corr_df["f1(1)"] = f1_value
            corr_df["f1_base(1)"] = f1_base_value
            corr_df["model"] = variant
            corr_df["year"] = year
            corr_df["month"] = month
            corr_df["forecasting_scope"] = forecasting_scope
            merged.append(
                {
                    "name": item["name"],
                    "variant": variant,
                    "year": year,
                    "month": month,
                    "forecasting_scope": forecasting_scope,
                    "dataframe": corr_df,
                    "merge_status": "success",
                }
            )
            status = "success"
        elif n_matches == 0:
            status = "no_match"
        else:
            status = "multiple_matches"

        stats.append({"name": item["name"], "status": status, "n_matches": n_matches})

    return merged, pd.DataFrame(stats)


def save_outputs(experiment_dir: Path, merged_tables: list[dict[str, Any]]) -> None:
    pickle_path = experiment_dir / "merged_correspondence_tables.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(merged_tables, f)
    print(f"Saved pickle: {pickle_path}")

    csv_dir = experiment_dir / "merged_correspondence_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for item in merged_tables:
        out_path = csv_dir / f"{item['name']}_merged.csv"
        item["dataframe"].to_csv(out_path, index=False)
    print(f"Saved merged CSVs: {csv_dir}")


def main() -> int:
    args = parse_args()
    config = MODEL_CONFIG[args.model_type]

    try:
        experiment_dir, results_dir = resolve_paths(args.experiment_dir, config)
        print(f"Experiment directory: {experiment_dir}")
        print(f"Results directory: {results_dir}")

        results_df = load_results_table(results_dir, config)
        print(f"Combined results rows: {len(results_df)}")

        correspondence_tables = load_correspondence_tables(results_dir, config.variant)
        merged_tables, merge_stats = merge_results_with_correspondence(
            results_df, correspondence_tables
        )

        if merge_stats.empty:
            raise RuntimeError("No correspondence tables were processed")

        print("Merge status counts:")
        print(merge_stats["status"].value_counts())
        print(f"Successfully merged: {len(merged_tables)} / {len(correspondence_tables)}")

        if not merged_tables:
            raise RuntimeError("No correspondence tables merged successfully")

        save_outputs(experiment_dir, merged_tables)
        print("Step 1 merge completed")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
