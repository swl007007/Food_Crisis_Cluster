"""Create fresh feature-exclude datasets from current FEWSNET unadjusted BM CSV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\FEWSNET_forecast_unadjusted_bm.csv"
)
if not DEFAULT_SOURCE.exists():
    DEFAULT_SOURCE = Path(
        "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"
        "/FEWSNET_forecast_unadjusted_bm.csv"
    )
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "main_ablation_exclude_updated_stage3_fixed_partitions"
    / "input_datasets"
)

FEATURE_GROUP_COLUMNS = {
    "weather_exclude": [
        "Rainf_f_tavg_mean",
        "Rainf_zscore",
        "Tair_f_tavg_mean",
        "Tair_zscore",
    ],
    "agri_exclude": [
        "AEZ_10000",
        "AEZ_12000",
        "AEZ_15000",
        "AEZ_17000",
        "AEZ_19000",
        "AEZ_25000",
        "AEZ_31000",
        "AEZ_32000",
        "AEZ_33000",
        "AEZ_34000",
        "AEZ_36000",
        "AEZ_38000",
        "AEZ_4000",
        "AEZ_40000",
        "AEZ_43000",
        "AEZ_7000",
        "AEZ_9000",
        "EVI",
        "crop",
        "distance_to_river",
        "gpp_mean",
        "range",
    ],
    "conflict_exclude": [
        "distance_to_nearest_acled",
        "event_count_battles",
        "event_count_battles_w10",
        "event_count_battles_w5",
        "event_count_explosions",
        "event_count_explosions_w10",
        "event_count_explosions_w5",
        "event_count_violence",
        "event_count_violence_w10",
        "event_count_violence_w5",
        "sum_fatalities_battles",
        "sum_fatalities_battles_w10",
        "sum_fatalities_battles_w5",
        "sum_fatalities_explosions",
        "sum_fatalities_explosions_w10",
        "sum_fatalities_explosions_w5",
        "sum_fatalities_violence",
        "sum_fatalities_violence_w10",
        "sum_fatalities_violence_w5",
    ],
    "econ_exclude": [
        "CC",
        "CPI",
        "Food_CPI",
        "Food_food_inflation",
        "GDP",
        "gini",
        "market_access",
        "market_distance",
        "nightlight",
        "nightlight_sd",
        "pop",
    ],
    "food_prices_exclude": [
        "FAO_price",
        "WFP_Price",
        "WFP_Price_std",
    ],
    "geographic_exclude": [
        "elevation",
        "ruggedness",
        "sg_cec_5-15cm",
        "sg_cfvo_5-15cm",
        "sg_nitrogen_5-15cm",
        "sg_phh2o_5-15cm",
        "sg_soc_5-15cm",
        "slope",
    ],
    "secondary_exclude": [
        "distance_to_nearest_acled",
        "event_count_battles",
        "event_count_explosions",
        "event_count_violence",
        "sum_fatalities_battles",
        "sum_fatalities_explosions",
        "sum_fatalities_violence",
        "event_count_battles_w5",
        "event_count_explosions_w5",
        "event_count_violence_w5",
        "sum_fatalities_battles_w5",
        "sum_fatalities_explosions_w5",
        "sum_fatalities_violence_w5",
        "event_count_battles_w10",
        "event_count_explosions_w10",
        "event_count_violence_w10",
        "sum_fatalities_battles_w10",
        "sum_fatalities_explosions_w10",
        "sum_fatalities_violence_w10",
        "nightlight",
        "nightlight_sd",
        "market_distance",
        "FAO_price",
        "market_access",
        "CPI",
        "GDP",
        "CC",
        "gini",
        "WFP_Price",
        "WFP_Price_std",
        "fews_ha",
        "fews_proj_near",
        "fews_proj_near_ha",
        "fews_proj_med",
        "fews_proj_med_ha",
        "pop",
        "fews_proj_med_adjusted",
        "Food_CPI",
        "Food_food_inflation",
    ],
    "lag_exclude": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(FEATURE_GROUP_COLUMNS),
        help="Generate one dataset. Repeat for multiple datasets. Defaults to all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.source, nrows=0)
    source_columns = list(header.columns)
    source_set = set(source_columns)
    manifest_path = args.out_dir / "feature_exclude_dataset_manifest.json"
    existing_datasets = {}
    if args.dataset and manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        existing_datasets = existing_manifest.get("datasets", {})

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "source": str(args.source),
        "out_dir": str(args.out_dir),
        "source_column_count": len(source_columns),
        "datasets": existing_datasets,
    }

    selected_datasets = args.dataset or list(FEATURE_GROUP_COLUMNS)
    for dataset_name in selected_datasets:
        drop_columns = FEATURE_GROUP_COLUMNS[dataset_name]
        missing = [column for column in drop_columns if column not in source_set]
        if missing:
            raise ValueError(f"{dataset_name} has missing source columns: {missing}")

        keep_columns = [column for column in source_columns if column not in set(drop_columns)]
        out_path = args.out_dir / f"{dataset_name}.csv"
        print(f"Writing {out_path} ({len(keep_columns)} columns; drop {len(drop_columns)})")
        pd.read_csv(args.source, usecols=keep_columns, low_memory=False).to_csv(
            out_path,
            index=False,
        )
        manifest["datasets"][dataset_name] = {
            "path": str(out_path),
            "column_count": len(keep_columns),
            "dropped_columns": drop_columns,
        }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
