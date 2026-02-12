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
    variant: str


MODEL_CONFIG: dict[str, ModelConfig] = {
    "georf": ModelConfig("georf", "GeoRFResults", "GeoRF"),
    "geoxgb": ModelConfig("geoxgb", "GeoXgboostResults", "GeoXGB"),
    "geodt": ModelConfig("geodt", "GeoDTResults", "GeoDT"),
}

RESULT_DIR_RE = re.compile(
    r"result_Geo(?:RF|XGB|DT)_(\d{4})_fs(\d+)_(\d{4})-(\d{2})_visual"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load correspondence tables into pickle.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory path")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(MODEL_CONFIG.keys()),
        help="Model type for path mapping",
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


def extract_metadata(file_path: Path, variant: str) -> tuple[str, str, str, str]:
    match = RESULT_DIR_RE.search(str(file_path))
    if not match:
        raise ValueError(f"Cannot parse metadata from path: {file_path}")
    year = match.group(1)
    month = match.group(4)
    forecasting_scope = f"fs{match.group(2)}"
    return variant, year, month, forecasting_scope


def load_correspondence_tables(results_dir: Path, variant: str) -> list[dict[str, Any]]:
    files = sorted(results_dir.glob("**/correspondence_table_*.csv"))
    if not files:
        raise FileNotFoundError(f"No correspondence tables found under {results_dir}")

    loaded: list[dict[str, Any]] = []
    for file_path in files:
        item_variant, year, month, forecasting_scope = extract_metadata(file_path, variant)
        name = f"{item_variant}_{year}_{month}_{forecasting_scope}"
        df = pd.read_csv(file_path, dtype={"partition_id": "str"})
        part = df["partition_id"].astype(str)
        df["partition_id"] = part.where(part.str.startswith("s"), "s" + part)
        loaded.append(
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
        print(f"Loaded {name} ({len(df)} rows)")
    return loaded


def save_pickle(experiment_dir: Path, tables: list[dict[str, Any]]) -> Path:
    out_path = experiment_dir / "correspondence_tables_loaded.pkl"
    with out_path.open("wb") as f:
        pickle.dump(tables, f)
    return out_path


def main() -> int:
    args = parse_args()
    config = MODEL_CONFIG[args.model_type]

    try:
        experiment_dir, results_dir = resolve_paths(args.experiment_dir, config)
        print(f"Experiment directory: {experiment_dir}")
        print(f"Results directory: {results_dir}")
        tables = load_correspondence_tables(results_dir, config.variant)
        print(f"Total loaded correspondence tables: {len(tables)}")
        output_file = save_pickle(experiment_dir, tables)
        print(f"Saved correspondence tables: {output_file}")
        print("Step 2 loading completed")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
