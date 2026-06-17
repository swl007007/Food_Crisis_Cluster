#!/usr/bin/env python3
"""Plot GeoDT branch 1 and branch 011 locations for the 2024-10 4-month-horizon diagnostic pair."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

try:
    from paper_horizon_labels import label_for_scope
except ModuleNotFoundError:
    from scripts.paper_horizon_labels import label_for_scope


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRESPONDENCE = REPO_ROOT / "result_GeoDT_0" / "correspondence_table_2024-10.csv"
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "geodt_branch_1_vs_001_locations_2024-10_fs1_global.png"

DEFAULT_BRANCHES = ("1", "001")
BRANCH_COLORS = {"1": "#d73027", "001": "#2166ac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected GeoDT branch spatial locations from correspondence_table_2024-10.csv."
    )
    parser.add_argument(
        "--correspondence",
        type=Path,
        default=DEFAULT_CORRESPONDENCE,
        help="Path to correspondence_table_2024-10.csv.",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SHAPEFILE,
        help="Path to FEWSNET global admin-boundary shapefile.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--branches", nargs=2, default=DEFAULT_BRANCHES, help="Two branch IDs to compare.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_branch_geometries(correspondence_path: Path, shapefile_path: Path, branches: tuple[str, str]) -> gpd.GeoDataFrame:
    correspondence = pd.read_csv(correspondence_path, usecols=["FEWSNET_admin_code", "partition_id"], dtype=str)
    correspondence["FEWSNET_admin_code"] = normalize_admin_code(correspondence["FEWSNET_admin_code"])
    correspondence["partition_id"] = correspondence["partition_id"].astype(str).str.strip()
    correspondence = correspondence[correspondence["partition_id"].ne("nan")].copy()

    gdf = gpd.read_file(shapefile_path)
    if "admin_code" not in gdf.columns:
        raise ValueError(f"admin_code column not found in shapefile. Available columns: {list(gdf.columns)}")
    gdf["admin_code"] = normalize_admin_code(gdf["admin_code"])

    merged = gdf.merge(correspondence, left_on="admin_code", right_on="FEWSNET_admin_code", how="inner")
    if len(merged) != len(correspondence):
        raise ValueError(f"Matched {len(merged)} shapefile rows for {len(correspondence)} correspondence rows.")
    selected = merged[merged["partition_id"].isin(branches)].copy()
    missing = sorted(set(branches) - set(selected["partition_id"].unique().tolist()))
    if missing:
        raise ValueError(f"Selected branch partition_id values not found after join: {missing}")
    invalid_count = (~selected.geometry.is_valid).sum()
    if invalid_count > 0:
        selected["geometry"] = selected.geometry.buffer(0)
    return merged, selected


def set_panel_extent(ax: plt.Axes, gdf: gpd.GeoDataFrame, pad_fraction: float = 0.08) -> None:
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = max(maxx - minx, 1e-6)
    dy = max(maxy - miny, 1e-6)
    ax.set_xlim(minx - dx * pad_fraction, maxx + dx * pad_fraction)
    ax.set_ylim(miny - dy * pad_fraction, maxy + dy * pad_fraction)


def plot_branch_panel(
    ax: plt.Axes,
    context_gdf: gpd.GeoDataFrame,
    selected_gdf: gpd.GeoDataFrame,
    branch_ids: Iterable[int],
    title: str,
) -> None:
    context_gdf.plot(ax=ax, color="#eeeeee", edgecolor="#9a9a9a", linewidth=0.25, zorder=1)
    for branch_id in branch_ids:
        branch_gdf = selected_gdf[selected_gdf["partition_id"] == branch_id]
        branch_gdf.plot(
            ax=ax,
            color=BRANCH_COLORS.get(branch_id, "#756bb1"),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
            zorder=2,
        )
    set_panel_extent(ax, context_gdf)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_branch_locations(
    merged: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    branches: tuple[str, str],
    output_path: Path,
    dpi: int,
) -> None:
    context = merged.copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    plot_branch_panel(
        axes[0],
        context,
        selected,
        branches,
        f"GeoDT selected branches\nBranch {branches[0]} vs Branch {branches[1]}",
    )
    plot_branch_panel(axes[1], context, selected, (branches[0],), f"Branch {branches[0]} only")
    plot_branch_panel(axes[2], context, selected, (branches[1],), f"Branch {branches[1]} only")

    counts = selected["partition_id"].value_counts().to_dict()
    legend_handles = [
        mpatches.Patch(
            facecolor=BRANCH_COLORS.get(branch_id, "#756bb1"),
            edgecolor="white",
            label=f"Branch {branch_id} (n={counts.get(branch_id, 0)})",
        )
        for branch_id in branches
    ]
    legend_handles.append(
        mpatches.Patch(
            facecolor="#eeeeee",
            edgecolor="#9a9a9a",
            label=f"Other global 2024-10 GeoDT {label_for_scope('fs1')} areas",
        )
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, fontsize=10)
    fig.suptitle(
        f"Global spatial locations of GeoDT branch-specific local DecisionTree comparison pair (2024-10, {label_for_scope('fs1')})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0.02, 0.10, 0.98, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    branches = tuple(str(branch).strip() for branch in args.branches)
    merged, selected = load_branch_geometries(args.correspondence, args.shapefile, branches)
    plot_branch_locations(merged, selected, branches, args.output, args.dpi)
    counts = selected["partition_id"].value_counts().sort_index()
    print(f"Joined correspondence rows: {len(merged)}")
    for branch_id, count in counts.items():
        print(f"Branch {branch_id}: {int(count)} polygons")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
