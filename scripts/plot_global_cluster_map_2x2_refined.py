#!/usr/bin/env python3
"""Create separate fs1 2x2 Global partition map figures for GeoRF and GeoDT."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "main_ablation_results" / "march2026_main_backup_month_ind_cont3"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)

MODEL_SPECS = {
    "GeoRF": {"dir_token": "GF", "output": "global_cluster_map_2x2_georf_refined.png"},
    "GeoDT": {"dir_token": "DT", "output": "global_cluster_map_2x2_geodt_refined.png"},
}
PANEL_ORDER = ("general", "m2", "m6", "m10")
PANEL_TITLES = {
    "general": "General",
    "m2": "Month-specific: m2",
    "m6": "Month-specific: m6",
    "m10": "Month-specific: m10",
}
TAG_PATTERN = re.compile(r"cluster_mapping_k40_nc(?P<nc>\d+)_(?P<tag>general|m2|m6|m10)_refined")

REGION_ORDER = (
    "West Africa",
    "East Africa",
    "Central Africa",
    "Southern Africa",
    "Middle East & Afghanistan",
    "Latin America",
)
REGION_COLORS = {
    "West Africa": "#2b8cbe",
    "East Africa": "#31a354",
    "Central Africa": "#f16913",
    "Southern Africa": "#807dba",
    "Middle East & Afghanistan": "#bf812d",
    "Latin America": "#ef3b2c",
}
REGION_ABBREVIATIONS = {
    "West Africa": "WA",
    "East Africa": "EA",
    "Central Africa": "CA",
    "Southern Africa": "SA",
    "Middle East & Afghanistan": "MEA",
    "Latin America": "LA",
}
REGION_SUBPALETTES = {
    "West Africa": ["#2b8cbe", "#4eb3d3", "#7bccc4", "#a8ddb5", "#43a2ca", "#74a9cf", "#3690c0"],
    "East Africa": ["#31a354", "#74c476", "#a1d99b", "#41ab5d", "#78c679", "#addd8e", "#2ca25f", "#66c2a4", "#99d8c9"],
    "Central Africa": ["#fdae6b", "#fd8d3c", "#f16913", "#fdd0a2", "#e6550d", "#fdae61"],
    "Southern Africa": ["#807dba", "#9e9ac8", "#bcbddc", "#756bb1", "#8c6bb1", "#b2abd2"],
    "Middle East & Afghanistan": ["#d8b365", "#c7a76c", "#bf812d", "#dfc27d", "#a6611a"],
    "Latin America": ["#ef3b2c", "#fb6a4a", "#fc9272", "#de2d26", "#fcae91"],
}
ISO_TO_REGION = {
    "BF": "West Africa",
    "ML": "West Africa",
    "NE": "West Africa",
    "NG": "West Africa",
    "BI": "East Africa",
    "ET": "East Africa",
    "KE": "East Africa",
    "SD": "East Africa",
    "SO": "East Africa",
    "SS": "East Africa",
    "UG": "East Africa",
    "CD": "Central Africa",
    "CM": "Central Africa",
    "TD": "Central Africa",
    "MG": "Southern Africa",
    "MW": "Southern Africa",
    "MZ": "Southern Africa",
    "ZW": "Southern Africa",
    "AF": "Middle East & Afghanistan",
    "YE": "Middle East & Afghanistan",
    "GT": "Latin America",
    "HT": "Latin America",
}

try:
    BASEMAP_SOURCE = cx.providers.CartoDB.PositronNoLabels
except AttributeError:
    BASEMAP_SOURCE = cx.providers.OpenStreetMap.Mapnik


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot separate fs1 2x2 Global refined partition maps for GeoRF and GeoDT."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source folder containing result_partition_k40_compare_{GF,DT}_fs1/refined directories.",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SHAPEFILE,
        help="Global FEWSNET shapefile path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated GeoRF and GeoDT figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Disable contextily basemap tiles.",
    )
    return parser.parse_args()


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_shapefile(shapefile_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shapefile_path)
    candidates = ("FEWSNET_admin_code", "admin_code", "adm_code", "FNID")
    found = next((column for column in candidates if column in gdf.columns), None)
    if found is None:
        raise ValueError(
            f"Admin-code column not found in shapefile. Tried: {candidates}. "
            f"Available columns: {list(gdf.columns)}"
        )
    if found != "FEWSNET_admin_code":
        gdf = gdf.rename(columns={found: "FEWSNET_admin_code"})

    if "ISO" not in gdf.columns:
        raise ValueError(f"ISO column not found in shapefile. Available columns: {list(gdf.columns)}")

    gdf["FEWSNET_admin_code"] = normalize_admin_code(gdf["FEWSNET_admin_code"])
    gdf["ISO"] = gdf["ISO"].astype(str).str.strip().str.upper()
    gdf["region_group"] = gdf["ISO"].map(ISO_TO_REGION)
    missing_regions = sorted(gdf.loc[gdf["region_group"].isna(), "ISO"].dropna().unique().tolist())
    if missing_regions:
        raise ValueError(f"No region group assigned for ISO values: {missing_regions}")

    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def choose_mapping(refined_dir: Path, tag: str) -> Path:
    candidates = []
    for csv_path in sorted(refined_dir.glob(f"cluster_mapping_k40_nc*_{tag}_*.csv")):
        match = TAG_PATTERN.search(csv_path.name)
        if not match or match.group("tag") != tag:
            continue
        candidates.append((int(match.group("nc")), len(csv_path.name), csv_path))
    if not candidates:
        raise FileNotFoundError(f"No {tag} cluster mapping CSV found in {refined_dir}")
    return max(candidates)[2]


def discover_model_csvs(source_dir: Path, model: str) -> Dict[str, Path]:
    dir_token = MODEL_SPECS[model]["dir_token"]
    refined_dir = source_dir / f"result_partition_k40_compare_{dir_token}_fs1" / "refined"
    return {panel: choose_mapping(refined_dir, panel) for panel in PANEL_ORDER}


def merged_map(base_gdf: gpd.GeoDataFrame, csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path, usecols=["FEWSNET_admin_code", "cluster_id"])
    df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
    merged = base_gdf.merge(df, on="FEWSNET_admin_code", how="inner")
    if merged.empty:
        raise ValueError(f"No matched records after merge: {csv_path}")
    merged["cluster_id"] = pd.to_numeric(merged["cluster_id"], errors="coerce")
    merged = merged.dropna(subset=["cluster_id"]).copy()
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    return merged


def dominant_cluster_regions(panel_data: Dict[str, gpd.GeoDataFrame]) -> Dict[Tuple[str, int], str]:
    result: Dict[Tuple[str, int], str] = {}
    for panel, gdf in panel_data.items():
        counts = (
            gdf.groupby(["cluster_id", "region_group"])
            .size()
            .reset_index(name="count")
            .sort_values(["cluster_id", "count", "region_group"], ascending=[True, False, True])
        )
        dominant = counts.drop_duplicates("cluster_id")
        for row in dominant.itertuples(index=False):
            result[(panel, int(row.cluster_id))] = str(row.region_group)
    return result


def build_partition_palette(
    panel_data: Dict[str, gpd.GeoDataFrame],
    cluster_regions: Dict[Tuple[str, int], str],
) -> Tuple[ListedColormap, Dict[Tuple[str, int], int], Dict[str, Dict[int, str]], Dict[Tuple[str, int], str]]:
    colors = []
    key_to_idx: Dict[Tuple[str, int], int] = {}
    key_to_color: Dict[Tuple[str, int], str] = {}
    summary: Dict[str, Dict[int, str]] = {}
    for panel in PANEL_ORDER:
        panel_clusters = sorted(panel_data[panel]["cluster_id"].unique().tolist())
        summary[panel] = {int(cluster_id): cluster_regions[(panel, int(cluster_id))] for cluster_id in panel_clusters}
        region_to_clusters: Dict[str, list[int]] = {region: [] for region in REGION_ORDER}
        for cluster_id in panel_clusters:
            region_to_clusters[cluster_regions[(panel, int(cluster_id))]].append(int(cluster_id))
        for region in REGION_ORDER:
            clusters = region_to_clusters[region]
            subpalette = REGION_SUBPALETTES[region]
            for idx, cluster_id in enumerate(clusters):
                if idx >= len(subpalette):
                    raise ValueError(
                        f"Not enough colors for {panel} {region}: "
                        f"{len(clusters)} clusters, {len(subpalette)} colors"
                    )
                color = subpalette[idx]
                key_to_idx[(panel, cluster_id)] = len(colors)
                key_to_color[(panel, cluster_id)] = color
                colors.append(color)
    return ListedColormap(colors), key_to_idx, summary, key_to_color


def add_partition_color_index(gdf: gpd.GeoDataFrame, panel: str, key_to_idx: Dict[Tuple[str, int], int]) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["partition_color_idx"] = gdf["cluster_id"].map(lambda cluster_id: key_to_idx[(panel, int(cluster_id))])
    return gdf


def plot_model_grid(
    model: str,
    base_gdf: gpd.GeoDataFrame,
    csvs: Dict[str, Path],
    output_path: Path,
    dpi: int,
    add_basemap: bool,
) -> Dict[str, Dict[int, str]]:
    panel_data = {panel: merged_map(base_gdf, csvs[panel]) for panel in PANEL_ORDER}
    cluster_regions = dominant_cluster_regions(panel_data)
    cmap, key_to_idx, summary, key_to_color = build_partition_palette(panel_data, cluster_regions)

    plot_data = {
        panel: add_partition_color_index(gdf, panel, key_to_idx).to_crs(epsg=3857)
        for panel, gdf in panel_data.items()
    }
    boundary_layer = base_gdf[["geometry"]].to_crs(epsg=3857)
    total_bounds = boundary_layer.total_bounds

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.ravel()

    for ax, panel in zip(axes_flat, PANEL_ORDER):
        gdf = plot_data[panel]
        gdf.plot(
            ax=ax,
            column="partition_color_idx",
            cmap=cmap,
            edgecolor="white",
            linewidth=0.10,
            legend=False,
            categorical=True,
            alpha=0.9 if add_basemap else 1.0,
            zorder=2,
        )
        boundary_layer.boundary.plot(ax=ax, color="#222222", linewidth=0.08, alpha=0.45, zorder=3)
        ax.set_xlim(total_bounds[0], total_bounds[2])
        ax.set_ylim(total_bounds[1], total_bounds[3])
        if add_basemap:
            try:
                cx.add_basemap(ax, source=BASEMAP_SOURCE, zoom="auto", attribution=False, zorder=1)
            except Exception as exc:
                print(f"WARNING: basemap failed for {model} {panel}: {exc}")
        ax.set_title(PANEL_TITLES[panel], fontsize=12, fontweight="bold", pad=6)
        ax.set_axis_off()

    color_to_labels: Dict[str, list[str]] = {}
    for panel in PANEL_ORDER:
        for cluster_id in sorted(panel_data[panel]["cluster_id"].unique().tolist()):
            color = key_to_color[(panel, int(cluster_id))]
            region = summary[panel][int(cluster_id)]
            color_to_labels.setdefault(color, []).append(
                f"{panel} c{int(cluster_id)} ({REGION_ABBREVIATIONS[region]})"
            )

    legend_handles = [
        mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=0.2,
            label="; ".join(labels),
        )
        for color, labels in color_to_labels.items()
    ]
    fig.legend(
        handles=legend_handles,
        title="Shared color partition groups",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=True,
        fontsize=5.5,
        title_fontsize=8,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.35,
    )
    fig.suptitle(
        f"{model} fs1 Global Refined Partition Mapping (k=40)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.02, 0.20, 0.98, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {model} figure: {output_path}")
    return summary


def print_region_summary(base_gdf: gpd.GeoDataFrame) -> None:
    summary = (
        base_gdf[["ISO", "ADMIN0", "region_group"]]
        .drop_duplicates()
        .sort_values(["region_group", "ADMIN0"])
    )
    print("\nRegion grouping:")
    for region in REGION_ORDER:
        countries = summary.loc[summary["region_group"] == region, "ADMIN0"].tolist()
        print(f"  {region}: {', '.join(countries)}")


def main() -> None:
    args = parse_args()
    base_gdf = load_shapefile(args.shapefile)
    print(f"Loaded shapefile: {args.shapefile} ({len(base_gdf)} polygons)")
    print_region_summary(base_gdf)

    for model in MODEL_SPECS:
        csvs = discover_model_csvs(args.source_dir, model)
        print(f"\n{model} fs1 mappings:")
        for panel in PANEL_ORDER:
            print(f"  {panel:7s} -> {csvs[panel]}")
        output_path = args.output_dir / MODEL_SPECS[model]["output"]
        summary = plot_model_grid(
            model=model,
            base_gdf=base_gdf,
            csvs=csvs,
            output_path=output_path,
            dpi=args.dpi,
            add_basemap=not args.no_basemap,
        )
        print(f"{model} partition dominant regions:")
        for panel in PANEL_ORDER:
            print(f"  {panel:7s}: {summary[panel]}")


if __name__ == "__main__":
    main()
