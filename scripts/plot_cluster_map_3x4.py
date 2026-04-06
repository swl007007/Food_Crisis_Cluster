#!/usr/bin/env python3
"""
Create a 3x4 spatial cluster map figure from model-specific cluster mapping CSVs.

Rows are fixed as:
1) GeoDT
2) GeoRF
3) GeoXGB

Columns are fixed as:
1) general
2) m2
3) m6
4) m10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "cluster_map_3x4.png"
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)

MODEL_ORDER = ("GeoDT", "GeoRF", "GeoXGB")
MODEL_DIRS = {
    "GeoDT": REPO_ROOT / "GeoDTExperiment" / "knn_sparsification_results",
    "GeoRF": REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results",
    "GeoXGB": REPO_ROOT / "GeoXGBExperiment" / "knn_sparsification_results",
}
COLUMN_ORDER = ("general", "m2", "m6", "m10")
FILE_PATTERN = re.compile(
    r"cluster_mapping_k40_nc(?P<nc>\d+)_(?P<tag>general|m\d+)\.csv$"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot a 3x4 panel of cluster maps for GeoDT/GeoRF/GeoXGB."
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SHAPEFILE,
        help="Path to FEWS NET shapefile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    return parser.parse_args()


def normalize_admin_code(series: pd.Series) -> pd.Series:
    """Normalize admin-code values so CSV and shapefile keys align."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def load_shapefile(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Load shapefile and standardize FEWSNET admin-code column."""
    gdf = gpd.read_file(shapefile_path)
    candidates = ("FEWSNET_admin_code", "admin_code", "adm_code", "FNID")
    found = next((c for c in candidates if c in gdf.columns), None)
    if found is None:
        raise ValueError(
            f"Admin-code column not found in shapefile. Tried: {candidates}. "
            f"Available columns: {list(gdf.columns)}"
        )

    if found != "FEWSNET_admin_code":
        gdf = gdf.rename(columns={found: "FEWSNET_admin_code"})

    gdf["FEWSNET_admin_code"] = normalize_admin_code(gdf["FEWSNET_admin_code"])
    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        gdf["geometry"] = gdf.geometry.buffer(0)

    return gdf


def discover_csvs() -> Dict[str, Dict[str, Path]]:
    """Find mapping CSVs and index them by model and tag."""
    result: Dict[str, Dict[str, Path]] = {model: {} for model in MODEL_ORDER}
    for model in MODEL_ORDER:
        for csv_path in sorted(MODEL_DIRS[model].glob("cluster_mapping_k40_nc*_*.csv")):
            match = FILE_PATTERN.match(csv_path.name)
            if not match:
                continue
            tag = match.group("tag")
            if tag in COLUMN_ORDER:
                result[model][tag] = csv_path
    return result


def merged_map(
    base_gdf: gpd.GeoDataFrame,
    csv_path: Path,
) -> gpd.GeoDataFrame:
    """Merge one cluster-mapping CSV with base geometry."""
    df = pd.read_csv(csv_path, usecols=["FEWSNET_admin_code", "cluster_id"])
    df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
    merged = base_gdf.merge(df, on="FEWSNET_admin_code", how="inner")
    if merged.empty:
        raise ValueError(f"No matched records after merge: {csv_path}")
    merged["cluster_id"] = pd.to_numeric(merged["cluster_id"], errors="coerce")
    merged = merged.dropna(subset=["cluster_id"]).copy()
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    return merged


def plot_grid(
    base_gdf: gpd.GeoDataFrame,
    csv_map: Dict[str, Dict[str, Path]],
    output_path: Path,
    dpi: int,
) -> None:
    """Create and save the 3x4 panel figure."""
    panel_data: Dict[tuple[str, str], tuple[gpd.GeoDataFrame, Path]] = {}
    all_partition_ids: set[int] = set()
    for model in MODEL_ORDER:
        for tag in COLUMN_ORDER:
            csv_path = csv_map.get(model, {}).get(tag)
            if csv_path is None:
                continue
            merged = merged_map(base_gdf, csv_path).rename(
                columns={"cluster_id": "partition_id"}
            )
            panel_data[(model, tag)] = (merged, csv_path)
            all_partition_ids.update(merged["partition_id"].unique().tolist())

    if not all_partition_ids:
        raise ValueError("No cluster mappings were loaded. Check CSV discovery paths.")

    unique_partition_ids = sorted(all_partition_ids)
    pid_to_idx = {pid: idx for idx, pid in enumerate(unique_partition_ids)}
    cmap = plt.get_cmap("tab20", len(unique_partition_ids))
    boundary_layer = base_gdf[["geometry"]]

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))

    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, tag in enumerate(COLUMN_ORDER):
            ax = axes[row_idx, col_idx]
            panel = panel_data.get((model, tag))
            if panel is None:
                ax.text(0.5, 0.5, f"Missing\n{model} {tag}", ha="center", va="center")
                ax.set_axis_off()
                continue

            merged, csv_path = panel
            merged = merged.copy()
            merged["partition_idx"] = merged["partition_id"].map(pid_to_idx)
            merged.plot(
                ax=ax,
                column="partition_idx",
                cmap=cmap,
                edgecolor="none",
                linewidth=0.0,
                legend=False,
                categorical=True,
            )
            # Thin administrative boundaries from the shapefile.
            boundary_layer.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.08,
                alpha=0.45,
            )
            ax.set_title(csv_path.stem, fontsize=10, pad=6)
            ax.set_axis_off()

        axes[row_idx, 0].text(
            -0.10,
            0.5,
            model,
            rotation=90,
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    legend_handles = [
        mpatches.Patch(
            facecolor=cmap(pid_to_idx[pid]),
            edgecolor="black",
            linewidth=0.2,
            label=str(pid),
        )
        for pid in unique_partition_ids
    ]
    fig.legend(
        handles=legend_handles,
        title="partition_id",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(12, len(legend_handles)),
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )

    fig.suptitle(
        "Cluster Mapping (k=40): GeoDT / GeoRF / GeoXGB\n"
        "Columns: general, m2, m6, m10",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.03, 0.08, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the 3x4 map plotting workflow."""
    args = parse_args()
    base_gdf = load_shapefile(args.shapefile)
    csv_map = discover_csvs()
    plot_grid(base_gdf, csv_map, args.output, args.dpi)
    print(f"Saved 3x4 figure: {args.output}")


if __name__ == "__main__":
    main()
