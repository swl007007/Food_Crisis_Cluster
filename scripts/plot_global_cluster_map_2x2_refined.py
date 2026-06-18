#!/usr/bin/env python3
"""Create separate 4-month-horizon 2x2 Global partition map figures for GeoRF and GeoDT."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

try:
    from paper_horizon_labels import label_for_scope
except ModuleNotFoundError:
    from scripts.paper_horizon_labels import label_for_scope


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT
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
TAG_PATTERN = re.compile(
    r"cluster_mapping_k40_nc(?P<nc>\d+)_(?P<tag>general|m2|m6|m10)(?:_refined.*)?\.csv$"
)

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
LATAM_COUNTRIES = ("Guatemala", "Haiti")
AFRICA_MIN_X_M = -2_226_000.0
REGION_SUBPALETTES = {
    "West Africa": ["#2b8cbe", "#4eb3d3", "#7bccc4", "#a8ddb5", "#43a2ca", "#74a9cf", "#3690c0"],
    "East Africa": ["#31a354", "#74c476", "#a1d99b", "#41ab5d", "#78c679", "#addd8e", "#2ca25f", "#66c2a4", "#99d8c9", "#006d2c"],
    "Central Africa": ["#fdae6b", "#fd8d3c", "#f16913", "#fdd0a2", "#e6550d", "#fdae61"],
    "Southern Africa": ["#807dba", "#9e9ac8", "#bcbddc", "#756bb1", "#8c6bb1", "#b2abd2"],
    "Middle East & Afghanistan": ["#d8b365", "#c7a76c", "#bf812d", "#dfc27d", "#a6611a"],
    "Latin America": ["#ef3b2c", "#fb6a4a", "#fc9272", "#de2d26", "#fcae91"],
}
HATCH_PATTERNS = (
    "",
    "///",
    "\\\\\\",
    "xxx",
    "...",
    "++",
    "--",
    "||",
    "oo",
    "**",
    "//////",
    "\\\\\\\\\\\\",
    "xxxx",
    "....",
    "++++",
    "----",
    "||||",
    "OOOO",
    "****",
    "////",
)


@dataclass(frozen=True)
class PartitionStyle:
    """Matplotlib style assigned to one panel-local cluster."""

    facecolor: str
    hatch: str


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
        description="Plot separate 4-month horizon 2x2 Global refined partition maps for GeoRF and GeoDT."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Source folder. Defaults to the repo root and prefers "
            "GeoRFExperiment/GeoDTExperiment knn_sparsification_results. "
            "Also supports "
            "result_partition_k40_compare_{GF,DT}_fs1/refined directories."
        ),
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
    for csv_path in sorted(refined_dir.glob(f"cluster_mapping_k40_nc*_{tag}*.csv")):
        match = TAG_PATTERN.search(csv_path.name)
        if not match or match.group("tag") != tag:
            continue
        candidates.append((csv_path.stat().st_mtime, int(match.group("nc")), csv_path.name, csv_path))
    if not candidates:
        raise FileNotFoundError(f"No {tag} cluster mapping CSV found in {refined_dir}")
    return max(candidates)[3]


def discover_model_csvs(source_dir: Path, model: str) -> Dict[str, Path]:
    experiment_dir = source_dir / f"{model}Experiment" / "knn_sparsification_results"
    if experiment_dir.exists():
        return {panel: choose_mapping(experiment_dir, panel) for panel in PANEL_ORDER}

    direct_experiment_dir = source_dir / "knn_sparsification_results"
    if source_dir.name == f"{model}Experiment" and direct_experiment_dir.exists():
        return {panel: choose_mapping(direct_experiment_dir, panel) for panel in PANEL_ORDER}

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


def build_partition_styles(
    panel_data: Dict[str, gpd.GeoDataFrame],
    cluster_regions: Dict[Tuple[str, int], str],
) -> Tuple[Dict[Tuple[str, int], PartitionStyle], Dict[str, Dict[int, str]]]:
    key_to_style: Dict[Tuple[str, int], PartitionStyle] = {}
    summary: Dict[str, Dict[int, str]] = {}
    for panel in PANEL_ORDER:
        panel_clusters = sorted(panel_data[panel]["cluster_id"].unique().tolist())
        summary[panel] = {int(cluster_id): cluster_regions[(panel, int(cluster_id))] for cluster_id in panel_clusters}
        region_to_clusters: Dict[str, list[int]] = {region: [] for region in REGION_ORDER}
        for cluster_id in panel_clusters:
            region_to_clusters[cluster_regions[(panel, int(cluster_id))]].append(int(cluster_id))
        for region in REGION_ORDER:
            clusters = region_to_clusters[region]
            for cluster_id in clusters:
                key_to_style[(panel, cluster_id)] = PartitionStyle(
                    facecolor="#ffffff",
                    hatch=hatch_for_cluster_id(cluster_id),
                )
    return key_to_style, summary


def hatch_for_cluster_id(cluster_id: int) -> str:
    return HATCH_PATTERNS[int(cluster_id) % len(HATCH_PATTERNS)]


def compact_cluster_labels(cluster_ids: Iterable[int]) -> list[str]:
    return [f"c{int(cluster_id)}" for cluster_id in sorted(cluster_ids)]


def plot_partition_layer(
    ax,
    gdf: gpd.GeoDataFrame,
    panel: str,
    key_to_style: Dict[Tuple[str, int], PartitionStyle],
    boundary_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    for cluster_id in sorted(gdf["cluster_id"].unique().tolist()):
        style = key_to_style[(panel, int(cluster_id))]
        cluster_subset = gdf[gdf["cluster_id"].eq(cluster_id)]
        for region in REGION_ORDER:
            subset = cluster_subset[cluster_subset["region_group"].eq(region)]
            if subset.empty:
                continue
            subset.plot(
                ax=ax,
                color=REGION_COLORS[region],
                edgecolor="#4d4d4d",
                linewidth=0.10,
                hatch=style.hatch,
                alpha=0.92,
                zorder=2,
            )
    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.boundary.plot(ax=ax, color="#222222", linewidth=0.08, alpha=0.45, zorder=3)


def add_latam_inset(
    parent_ax,
    latam_gdf: gpd.GeoDataFrame,
    panel: str,
    key_to_style: Dict[Tuple[str, int], PartitionStyle],
) -> None:
    if latam_gdf.empty:
        return
    inset = parent_ax.inset_axes([0.01, 0.01, 0.30, 0.28])
    plot_partition_layer(inset, latam_gdf, panel, key_to_style, latam_gdf)
    minx, miny, maxx, maxy = latam_gdf.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.12
    inset.set_xlim(minx - pad_x, maxx + pad_x)
    inset.set_ylim(miny - pad_y, maxy + pad_y)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("0.35")
        spine.set_linewidth(0.9)
    inset.set_title("Latin America (FEWSNET)", fontsize=7, pad=1.5)
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.92)


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
    key_to_style, summary = build_partition_styles(panel_data, cluster_regions)

    plot_data = {panel: gdf.to_crs(epsg=3857) for panel, gdf in panel_data.items()}
    boundary_layer = base_gdf[["geometry"]].to_crs(epsg=3857)
    main_boundary = boundary_layer[boundary_layer.geometry.centroid.x >= AFRICA_MIN_X_M]
    total_bounds = main_boundary.total_bounds

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.ravel()

    for ax, panel in zip(axes_flat, PANEL_ORDER):
        gdf = plot_data[panel]
        main_gdf = gdf[gdf.geometry.centroid.x >= AFRICA_MIN_X_M]
        plot_partition_layer(ax, main_gdf, panel, key_to_style, main_boundary)
        ax.set_xlim(total_bounds[0], total_bounds[2])
        ax.set_ylim(total_bounds[1], total_bounds[3])
        if add_basemap:
            try:
                cx.add_basemap(ax, source=BASEMAP_SOURCE, zoom="auto", attribution=False, zorder=1)
            except Exception as exc:
                print(f"WARNING: basemap failed for {model} {panel}: {exc}")
        ax.set_title(PANEL_TITLES[panel], fontsize=12, fontweight="bold", pad=6)
        ax.set_axis_off()
        if "ADMIN0" in panel_data[panel].columns:
            latam_gdf = panel_data[panel][panel_data[panel]["ADMIN0"].isin(LATAM_COUNTRIES)].copy()
            add_latam_inset(ax, latam_gdf, panel, key_to_style)

    legend_clusters = sorted(
        {int(cluster_id) for panel in PANEL_ORDER for cluster_id in panel_data[panel]["cluster_id"].unique().tolist()}
    )

    legend_handles = [
        mpatches.Patch(
            facecolor="#ffffff",
            hatch=hatch_for_cluster_id(cluster_id),
            edgecolor="black",
            linewidth=0.35,
            label=f"c{cluster_id}",
        )
        for cluster_id in legend_clusters
    ]
    region_handles = [
        mpatches.Patch(
            facecolor=REGION_COLORS[region],
            edgecolor="black",
            linewidth=0.35,
            label=REGION_ABBREVIATIONS[region],
        )
        for region in REGION_ORDER
    ]
    fig.legend(
        handles=legend_handles + region_handles,
        title="Partition ID (texture) and region (color)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.040),
        ncol=min(10, max(1, len(legend_handles))),
        frameon=True,
        fontsize=8.5,
        title_fontsize=10,
        columnspacing=1.1,
        handlelength=1.8,
        handleheight=1.0,
        handletextpad=0.45,
    )
    fig.text(
        0.5,
        0.012,
        "Cluster IDs are interpreted within each panel; color indicates geographic region.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.suptitle(
        f"{model} {label_for_scope('fs1')} Global Refined Partition Mapping (k=40)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.02, 0.16, 0.98, 0.94))
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
        print(f"\n{model} {label_for_scope('fs1')} mappings:")
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
