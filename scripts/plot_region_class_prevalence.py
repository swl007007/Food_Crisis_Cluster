#!/usr/bin/env python3
"""Plot test-period class prevalence by FEWSNET region."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = REPO_ROOT / "result_partition_k40_compare_GF_fs1" / "predictions_monthly.csv"
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated"
DEFAULT_CSV = "region_class_prevalence_2021_2024.csv"
DEFAULT_FIGURE = "region_class_prevalence_2021_2024.png"

CRISIS_COLOR = "#d73027"
NON_CRISIS_COLOR = "#2ca25f"
UNVALIDATED_COLOR = "#bdbdbd"
UNVALIDATED_REGION = "Middle East"
UNVALIDATED_MONTHS = {
    "2021-10",
    "2022-02",
    "2022-06",
    "2022-10",
    "2023-02",
}
REGION_ORDER = [
    "East Africa",
    "West Africa",
    "Southern Africa",
    "Middle East",
    "Latin America",
    "Other",
]

REGION_MAP = {
    "Burundi": "East Africa",
    "Djibouti": "East Africa",
    "Eritrea": "East Africa",
    "Ethiopia": "East Africa",
    "Kenya": "East Africa",
    "Rwanda": "East Africa",
    "Somalia": "East Africa",
    "South Sudan": "East Africa",
    "Sudan": "East Africa",
    "Tanzania": "East Africa",
    "Uganda": "East Africa",
    "Burkina Faso": "West Africa",
    "Chad": "West Africa",
    "Gambia": "West Africa",
    "Ghana": "West Africa",
    "Guinea": "West Africa",
    "Liberia": "West Africa",
    "Mali": "West Africa",
    "Mauritania": "West Africa",
    "Niger": "West Africa",
    "Nigeria": "West Africa",
    "Senegal": "West Africa",
    "Sierra Leone": "West Africa",
    "Togo": "West Africa",
    "Benin": "West Africa",
    "Cote d'Ivoire": "West Africa",
    "Ivory Coast": "West Africa",
    "Angola": "Southern Africa",
    "Botswana": "Southern Africa",
    "Lesotho": "Southern Africa",
    "Madagascar": "Southern Africa",
    "Malawi": "Southern Africa",
    "Mozambique": "Southern Africa",
    "Namibia": "Southern Africa",
    "South Africa": "Southern Africa",
    "Swaziland": "Southern Africa",
    "Eswatini": "Southern Africa",
    "Zambia": "Southern Africa",
    "Zimbabwe": "Southern Africa",
    "Afghanistan": "Middle East",
    "Iraq": "Middle East",
    "Palestine": "Middle East",
    "Syria": "Middle East",
    "Yemen": "Middle East",
    "Jordan": "Middle East",
    "Lebanon": "Middle East",
    "El Salvador": "Latin America",
    "Guatemala": "Latin America",
    "Haiti": "Latin America",
    "Honduras": "Latin America",
    "Nicaragua": "Latin America",
    "Colombia": "Latin America",
    "Ecuador": "Latin America",
    "Peru": "Latin America",
    "Bolivia": "Latin America",
    "Venezuela": "Latin America",
}


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


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_region_lookup(shapefile_path: Path) -> pd.DataFrame:
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
    lookup["region"] = lookup["ADMIN0"].map(REGION_MAP).fillna("Other")
    return lookup.drop_duplicates("FEWSNET_admin_code")


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(path))
    required = {"FEWSNET_admin_code", "month_start", "y_true"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df[list(required)].copy()


def build_prevalence_table(predictions: pd.DataFrame, region_lookup: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
    df["target_month"] = pd.to_datetime(df["month_start"], errors="coerce").dt.strftime("%Y-%m")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df = df[df["target_month"].notna() & df["y_true"].isin([0, 1])].copy()

    lookup = region_lookup[["FEWSNET_admin_code", "region"]].copy()
    lookup["FEWSNET_admin_code"] = normalize_admin_code(lookup["FEWSNET_admin_code"])
    merged = df.merge(lookup, on="FEWSNET_admin_code", how="left")
    merged["region"] = merged["region"].fillna("Other")

    grouped = (
        merged.groupby(["region", "target_month"], as_index=False)
        .agg(
            crisis_count=("y_true", lambda values: int((values == 1).sum())),
            non_crisis_count=("y_true", lambda values: int((values == 0).sum())),
        )
    )
    grouped["total"] = grouped["crisis_count"] + grouped["non_crisis_count"]
    grouped["crisis_prevalence"] = grouped["crisis_count"] / grouped["total"]
    grouped["crisis_prevalence"] = grouped["crisis_prevalence"].round(6)
    grouped["validation_status"] = "validated"
    unvalidated = grouped["region"].eq(UNVALIDATED_REGION) & grouped["target_month"].isin(UNVALIDATED_MONTHS)
    grouped.loc[unvalidated, "validation_status"] = "data not validated"

    ordered_regions = [region for region in REGION_ORDER if region in set(grouped["region"])]
    remaining = sorted(set(grouped["region"]) - set(ordered_regions))
    region_rank = {region: idx for idx, region in enumerate(ordered_regions + remaining)}
    grouped["_region_rank"] = grouped["region"].map(region_rank)
    grouped = grouped.sort_values(["_region_rank", "target_month"]).drop(columns="_region_rank")
    return grouped[
        [
            "region",
            "target_month",
            "crisis_count",
            "non_crisis_count",
            "total",
            "crisis_prevalence",
            "validation_status",
        ]
    ].reset_index(drop=True)


def render_prevalence_figure(table: pd.DataFrame, output_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    regions = [region for region in REGION_ORDER if region in set(table["region"])]
    regions.extend(sorted(set(table["region"]) - set(regions)))
    months = sorted(table["target_month"].unique(), key=lambda value: pd.Period(value, freq="M"))

    ncols = 3
    nrows = math.ceil(len(regions) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    x = range(len(months))

    for ax, region in zip(axes_flat, regions):
        sub = table[table["region"] == region].set_index("target_month").reindex(months).fillna(0)
        crisis = sub["crisis_count"].astype(int).to_numpy()
        non_crisis = sub["non_crisis_count"].astype(int).to_numpy()
        total_crisis = int(crisis.sum())
        total = int(crisis.sum() + non_crisis.sum())
        prevalence = total_crisis / total if total else 0
        ax.stackplot(x, crisis, non_crisis, colors=[CRISIS_COLOR, NON_CRISIS_COLOR], alpha=0.95)
        unvalidated_positions = [
            idx
            for idx, month in enumerate(months)
            if str(sub.loc[month, "validation_status"]) == "data not validated"
        ]
        for idx in unvalidated_positions:
            ax.axvspan(idx - 0.5, idx + 0.5, color=UNVALIDATED_COLOR, alpha=0.28, linewidth=0)
        if unvalidated_positions:
            start, end = min(unvalidated_positions), max(unvalidated_positions)
            y_top = max((crisis + non_crisis).max(), 1)
            ax.text(
                (start + end) / 2,
                y_top * 0.92,
                "data not validated",
                ha="center",
                va="top",
                fontsize=9,
                color="#4d4d4d",
            )
        ax.set_title(f"{region} (crisis prevalence={prevalence:.1%})", fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(list(x))
        ax.set_xticklabels(months, rotation=45, ha="right")
        ax.set_ylabel("Count")

    for ax in axes_flat[len(regions) :]:
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=CRISIS_COLOR, label="Crisis (value=1)"),
        mpatches.Patch(color=NON_CRISIS_COLOR, label="Non-crisis (value=0)"),
        mpatches.Patch(color=UNVALIDATED_COLOR, alpha=0.28, label="Data not validated"),
    ]
    fig.suptitle("Test-period class prevalence by FEWSNET region (2021-2024)", fontsize=16)
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-name", default=DEFAULT_CSV)
    parser.add_argument("--figure-name", default=DEFAULT_FIGURE)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    predictions = load_predictions(args.predictions)
    region_lookup = load_region_lookup(args.shapefile)
    table = build_prevalence_table(predictions, region_lookup)

    csv_path = output_dir / args.csv_name
    figure_path = output_dir / args.figure_name
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    render_prevalence_figure(table, figure_path, args.dpi)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {figure_path}")
    print(f"Rows: {len(table)}; regions: {', '.join(table['region'].drop_duplicates())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
