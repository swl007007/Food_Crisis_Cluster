#!/usr/bin/env python3
"""Plot GeoRF m2 pre/post adjacency-refinement partitions and reassigned polygons."""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRE_MAPPING = REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc13_m2.csv"
DEFAULT_POST_MAPPING = (
    REPO_ROOT
    / "result_partition_k40_compare_GF_fs1"
    / "refined"
    / "cluster_mapping_k40_nc13_m2_refined_contig3.csv"
)
DEFAULT_REFINEMENT_LOG = (
    REPO_ROOT
    / "result_partition_k40_compare_GF_fs1"
    / "refined"
    / "refine_summary_cluster_mapping_k40_nc13_m2.txt"
)
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated"
DEFAULT_FIGURE_NAME = "georf_m2_adjacency_refinement_1x3.png"
DEFAULT_SUMMARY_NAME = "georf_m2_adjacency_refinement_summary.csv"
DEFAULT_NOTE_NAME = "georf_m2_adjacency_refinement_note.md"
LATAM_REGION = "Latin America"
PLOT_SIMPLIFY_TOLERANCE_M = 5_000

CLUSTER_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#2CA02C",
    "#D62728",
]
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


@dataclass(frozen=True)
class ClusterStyle:
    """Matplotlib style assigned to one GeoRF m2 cluster."""

    facecolor: str
    hatch: str


def resolve_path(path: Path) -> Path:
    raw = str(path)
    if os.name == "nt":
        wsl_match = re.match(r"^[\\/]+mnt[\\/]+([A-Za-z])[\\/]+(.*)$", raw)
        if wsl_match:
            drive, rest = wsl_match.groups()
            return Path(f"{drive.upper()}:\\{rest.replace('/', '\\')}")
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if match:
        drive, rest = match.groups()
        if os.name == "nt":
            return Path(raw)
        return Path("/mnt") / drive.lower() / rest.replace("\\", "/")
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(path))
    required = {"FEWSNET_admin_code", "cluster_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df[["FEWSNET_admin_code", "cluster_id"]].copy()
    out["FEWSNET_admin_code"] = normalize_admin_code(out["FEWSNET_admin_code"])
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="raise").astype(int)
    return out.drop_duplicates("FEWSNET_admin_code").sort_values("FEWSNET_admin_code").reset_index(drop=True)


def build_reassignment_table(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    left = pre[["FEWSNET_admin_code", "cluster_id"]].copy()
    right = post[["FEWSNET_admin_code", "cluster_id"]].copy()
    left["FEWSNET_admin_code"] = normalize_admin_code(left["FEWSNET_admin_code"])
    right["FEWSNET_admin_code"] = normalize_admin_code(right["FEWSNET_admin_code"])
    left["cluster_id"] = pd.to_numeric(left["cluster_id"], errors="raise").astype(int)
    right["cluster_id"] = pd.to_numeric(right["cluster_id"], errors="raise").astype(int)

    merged = left.merge(
        right,
        on="FEWSNET_admin_code",
        how="inner",
        suffixes=("_before", "_after"),
    )
    if merged.empty:
        raise ValueError("No common FEWSNET_admin_code values between pre- and post-refinement mappings")
    merged["reassigned"] = merged["cluster_id_before"].ne(merged["cluster_id_after"])
    return merged.sort_values("FEWSNET_admin_code").reset_index(drop=True)


def parse_refinement_log(path: Path) -> dict:
    text = resolve_path(path).read_text(encoding="utf-8")
    iterations_match = re.search(r"^Iterations:\s*(\d+)", text, flags=re.MULTILINE)
    total_match = re.search(r"Total reassigned:\s*(\d+)\s+polygons", text)
    per_iteration = [
        int(value)
        for value in re.findall(r"Iteration\s+\d+/\d+:\s*(\d+)\s+polygons reassigned", text)
    ]
    return {
        "iterations": int(iterations_match.group(1)) if iterations_match else len(per_iteration),
        "total_reassigned": int(total_match.group(1)) if total_match else sum(per_iteration),
        "per_iteration_reassigned": per_iteration,
    }


def summarize_reassignment(table: pd.DataFrame, iterations: int, iteration_move_count: int | None = None) -> dict:
    n_polygons = int(len(table))
    n_final_changed = int(table["reassigned"].sum())
    if iteration_move_count is None:
        iteration_move_count = n_final_changed
    return {
        "mapping": "GeoRF m2",
        "iterations": int(iterations),
        "n_polygons": n_polygons,
        "n_final_changed_polygons": n_final_changed,
        "final_changed_pct": round(n_final_changed / n_polygons * 100, 3) if n_polygons else 0.0,
        "n_iteration_reassignment_moves": int(iteration_move_count),
        "n_clusters_before": int(table["cluster_id_before"].nunique()),
        "n_clusters_after": int(table["cluster_id_after"].nunique()),
    }


def load_shapefile(shapefile_path: Path):
    import geopandas as gpd

    gdf = gpd.read_file(resolve_path(shapefile_path))
    candidates = ("FEWSNET_admin_code", "admin_code", "adm_code", "FNID", "uid")
    found = next((column for column in candidates if column in gdf.columns), None)
    if found is None:
        raise ValueError(f"Admin-code column not found in shapefile. Tried: {candidates}")
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
    invalid_count = int((~gdf.geometry.is_valid).sum())
    if invalid_count:
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def merge_geometries(base_gdf, table: pd.DataFrame, cluster_column: str):
    df = table[["FEWSNET_admin_code", cluster_column]].copy()
    df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
    merged = base_gdf.merge(df, on="FEWSNET_admin_code", how="inner")
    if merged.empty:
        raise ValueError(f"No matched polygons for {cluster_column}")
    return merged


def compact_cluster_label(cluster_id: int) -> str:
    return f"c{int(cluster_id)}"


def cluster_style_map(cluster_ids: Iterable[int]) -> dict[int, ClusterStyle]:
    clusters = sorted({int(cluster_id) for cluster_id in cluster_ids})
    capacity = len(CLUSTER_PALETTE) * len(HATCH_PATTERNS)
    if len(clusters) > capacity:
        raise ValueError(f"Style set supports {capacity} clusters, got {len(clusters)}")
    return {
        cluster_id: ClusterStyle(
            facecolor="#ffffff",
            hatch=HATCH_PATTERNS[idx % len(HATCH_PATTERNS)],
        )
        for idx, cluster_id in enumerate(clusters)
    }


def split_main_and_latam_layers(gdf):
    latam = gdf[gdf["region_group"].eq(LATAM_REGION)].copy()
    main = gdf[~gdf["region_group"].eq(LATAM_REGION)].copy()
    return main, latam


def set_padded_bounds(ax, gdf, pad_fraction: float = 0.035) -> None:
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * pad_fraction
    pad_y = (maxy - miny) * pad_fraction
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)


def clean_geometry_for_dissolve(gdf):
    cleaned = gdf.copy()
    try:
        cleaned["geometry"] = cleaned.geometry.make_valid()
    except AttributeError:
        cleaned["geometry"] = cleaned.geometry.buffer(0)
    cleaned = cleaned[cleaned.geometry.notna() & ~cleaned.geometry.is_empty].copy()
    return cleaned


def dissolve_plot_layer(gdf, by: list[str]):
    cleaned = clean_geometry_for_dissolve(gdf)
    if cleaned.empty:
        return cleaned
    try:
        return cleaned.dissolve(by=by, as_index=False, method="coverage")
    except TypeError:
        return cleaned.dissolve(by=by, as_index=False)
    except Exception:
        cleaned["geometry"] = cleaned.geometry.buffer(0)
        return cleaned.dissolve(by=by, as_index=False)


def build_admin0_context(base_gdf):
    admin0_column = "ADMIN0" if "ADMIN0" in base_gdf.columns else "ISO"
    admin0 = base_gdf[[admin0_column, "geometry"]].dissolve(by=admin0_column, as_index=False)
    admin0["geometry"] = admin0.geometry.simplify(
        PLOT_SIMPLIFY_TOLERANCE_M,
        preserve_topology=True,
    )
    return admin0


def load_admin0_basemap(fallback_gdf, target_crs):
    import geopandas as gpd

    try:
        from cartopy.io import shapereader

        path = shapereader.natural_earth(
            resolution="110m",
            category="cultural",
            name="admin_0_countries",
        )
        admin0 = gpd.read_file(path).to_crs(target_crs)
        admin0["geometry"] = admin0.geometry.simplify(
            PLOT_SIMPLIFY_TOLERANCE_M,
            preserve_topology=True,
        )
        return admin0
    except Exception as exc:
        print(f"WARNING: Natural Earth admin0 basemap unavailable; using FEWSNET countries only: {exc}")
        return build_admin0_context(fallback_gdf).to_crs(target_crs)


def plot_admin0_context(ax, enabled: bool, admin0_gdf, extent_gdf) -> None:
    if not enabled or admin0_gdf is None or admin0_gdf.empty or extent_gdf.empty:
        return
    ax.set_facecolor("#dbe3e6")
    minx, miny, maxx, maxy = extent_gdf.total_bounds
    subset = admin0_gdf.cx[minx:maxx, miny:maxy]
    if subset.empty:
        return
    subset.plot(
        ax=ax,
        color="#f8f6f0",
        edgecolor="#d0c7c2",
        linewidth=0.25,
        alpha=0.95,
        zorder=1,
    )


def plot_admin0_outline(ax, enabled: bool, admin0_gdf, extent_gdf) -> None:
    if not enabled or admin0_gdf is None or admin0_gdf.empty or extent_gdf.empty:
        return
    minx, miny, maxx, maxy = extent_gdf.total_bounds
    subset = admin0_gdf.cx[minx:maxx, miny:maxy]
    if subset.empty:
        return
    subset.boundary.plot(ax=ax, color="#4d4d4d", linewidth=0.35, alpha=0.75, zorder=5)


def plot_cluster_partitions(ax, gdf, cluster_column: str, style_lookup: dict[int, ClusterStyle]) -> None:
    for region in REGION_ORDER:
        subset = gdf[gdf["region_group"].eq(region)]
        if subset.empty:
            continue
        subset = dissolve_plot_layer(subset, ["region_group"])
        subset.plot(
            ax=ax,
            color=REGION_COLORS[region],
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
            zorder=3,
        )
    for cluster_id, style in style_lookup.items():
        cluster_subset = gdf[gdf[cluster_column].eq(cluster_id)]
        if cluster_subset.empty:
            continue
        cluster_subset = dissolve_plot_layer(cluster_subset, [cluster_column])
        cluster_subset.plot(
            ax=ax,
            color="none",
            edgecolor="#4d4d4d",
            linewidth=0.0,
            hatch=style.hatch,
            zorder=4,
        )


def add_latam_inset(
    parent_ax,
    latam_gdf,
    cluster_column: str | None,
    style_lookup: dict[int, ClusterStyle],
    add_basemap_flag: bool,
    admin0_gdf,
    reassigned_only: bool = False,
    highlight_gdf=None,
) -> None:
    if latam_gdf.empty:
        return
    inset = parent_ax.inset_axes([0.01, 0.01, 0.30, 0.30])
    plot_admin0_context(inset, add_basemap_flag, admin0_gdf, latam_gdf)
    if reassigned_only and highlight_gdf is not None and not highlight_gdf.empty:
        highlight_gdf.plot(
            ax=inset,
            color="#d73027",
            edgecolor="#111111",
            linewidth=0.20,
            hatch="xxx",
            zorder=3,
        )
    elif cluster_column is not None:
        plot_cluster_partitions(inset, latam_gdf, cluster_column, style_lookup)
    plot_admin0_outline(inset, add_basemap_flag, admin0_gdf, latam_gdf)
    set_padded_bounds(inset, latam_gdf, pad_fraction=0.08)
    inset.set_title("Latin America (FEWSNET)", fontsize=7, pad=1.5)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("0.35")
        spine.set_linewidth(0.9)
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.92)


def plot_refinement_figure(
    base_gdf,
    reassignment_table: pd.DataFrame,
    output_path: Path,
    summary: dict,
    dpi: int,
    add_basemap_flag: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    all_clusters = pd.concat(
        [reassignment_table["cluster_id_before"], reassignment_table["cluster_id_after"]],
        ignore_index=True,
    )
    style_lookup = cluster_style_map(all_clusters)
    plot_base = base_gdf.to_crs(epsg=3857).copy()
    plot_base["geometry"] = plot_base.geometry.simplify(
        PLOT_SIMPLIFY_TOLERANCE_M,
        preserve_topology=True,
    )
    before = merge_geometries(plot_base, reassignment_table, "cluster_id_before")
    after = merge_geometries(plot_base, reassignment_table, "cluster_id_after")
    reassigned = merge_geometries(
        plot_base,
        reassignment_table[reassignment_table["reassigned"]],
        "cluster_id_after",
    )
    mapped_base = plot_base.merge(
        reassignment_table[["FEWSNET_admin_code"]],
        on="FEWSNET_admin_code",
        how="inner",
    )
    mapped_main, mapped_latam = split_main_and_latam_layers(mapped_base)
    before_main, before_latam = split_main_and_latam_layers(before)
    after_main, after_latam = split_main_and_latam_layers(after)
    reassigned_main, reassigned_latam = split_main_and_latam_layers(reassigned)
    admin0_context = load_admin0_basemap(plot_base, plot_base.crs) if add_basemap_flag else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    panels = [
        (axes[0], before_main, before_latam, "cluster_id_before", "Before adjacency refinement"),
        (axes[1], after_main, after_latam, "cluster_id_after", "After adjacency refinement"),
    ]
    for ax, gdf, latam_gdf, cluster_column, title in panels:
        plot_admin0_context(ax, add_basemap_flag, admin0_context, mapped_main)
        plot_cluster_partitions(ax, gdf, cluster_column, style_lookup)
        plot_admin0_outline(ax, add_basemap_flag, admin0_context, mapped_main)
        set_padded_bounds(ax, mapped_main)
        add_latam_inset(ax, latam_gdf, cluster_column, style_lookup, add_basemap_flag, admin0_context)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_axis_off()

    ax = axes[2]
    plot_admin0_context(ax, add_basemap_flag, admin0_context, mapped_main)
    if not reassigned_main.empty:
        reassigned_main.plot(
            ax=ax,
            color="#d73027",
            edgecolor="#111111",
            linewidth=0.22,
            hatch="xxx",
            zorder=3,
    )
    plot_admin0_outline(ax, add_basemap_flag, admin0_context, mapped_main)
    set_padded_bounds(ax, mapped_main)
    add_latam_inset(
        ax,
        mapped_latam,
        None,
        style_lookup,
        add_basemap_flag,
        admin0_context,
        reassigned_only=True,
        highlight_gdf=reassigned_latam,
    )
    ax.set_title("Reassigned polygons", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        -0.035,
        f"{summary['n_final_changed_polygons']} / {summary['n_polygons']} polygons changed final cluster "
        f"({summary['final_changed_pct']:.2f}%)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.set_axis_off()

    handles = [
        mpatches.Patch(
            facecolor="#ffffff",
            hatch=style_lookup[cluster_id].hatch,
            edgecolor="black",
            linewidth=0.35,
            label=compact_cluster_label(cluster_id),
        )
        for cluster_id in sorted(style_lookup)
    ]
    handles.append(
        mpatches.Patch(
            facecolor="#d73027",
            hatch="xxx",
            edgecolor="black",
            linewidth=0.35,
            label="Reassigned",
        )
    )
    handles.extend(
        mpatches.Patch(
            facecolor=REGION_COLORS[region],
            edgecolor="black",
            linewidth=0.35,
            label=REGION_ABBREVIATIONS[region],
        )
        for region in REGION_ORDER
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        fontsize=9,
        handlelength=1.8,
        handleheight=1.0,
        columnspacing=1.0,
        handletextpad=0.45,
        bbox_to_anchor=(0.5, -0.010),
    )
    fig.suptitle(
        "GeoRF m2 Adjacency Refinement (Pre-Stage 3 Local RF)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_note(output_dir: Path, summary: dict, log_info: dict) -> Path:
    note_path = output_dir / DEFAULT_NOTE_NAME
    per_iteration = ", ".join(str(value) for value in log_info["per_iteration_reassigned"])
    note_path.write_text(
        "\n".join(
            [
                "# GeoRF m2 Adjacency Refinement Note",
                "",
                "中文审查说明：",
                "",
                "该图使用 GeoRF 的 m2 consensus partition 作为一个实际 Stage 3 local RF",
                "评估之前的 adjacency refinement 示例。左图展示 refinement 前的 cluster",
                "assignment，中图展示 refinement 后的 assignment，右图只高亮发生 reassignment",
                "的 polygons。",
                "",
                f"该 m2 mapping 共包含 {summary['n_polygons']} 个 polygons，",
                f"3 次 deterministic refinement iteration 后共有 {summary['n_final_changed_polygons']} 个",
                f"polygons 的最终 cluster assignment 发生变化，占 {summary['final_changed_pct']:.2f}%。",
                f"每次 iteration 的 reassignment move 数量为：{per_iteration}，",
                f"iteration-level moves 合计 {summary['n_iteration_reassignment_moves']}。",
                "",
                "该图只汇报当前已经实现并用于 Stage 3 fixed-partition evaluation 的",
                "adjacency refinement，不额外声称新的 split acceptance 或模型重训机制。",
                "",
                "Appendix text (English):",
                "",
                "As an example of the adjacency-refinement step applied before the Stage 3 local",
                "RF evaluation, we show the GeoRF m2 consensus partition before and after",
                "refinement. The refinement log records",
                f"{summary['n_iteration_reassignment_moves']} polygon-iteration reassignment moves across",
                f"three deterministic iterations. In the final pre/post comparison,",
                f"{summary['n_final_changed_polygons']} of {summary['n_polygons']} polygons "
                f"({summary['final_changed_pct']:.2f}%) changed cluster assignment. The right panel highlights the polygons",
                "whose cluster assignment changed between the pre- and post-refinement maps.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-mapping", type=Path, default=DEFAULT_PRE_MAPPING)
    parser.add_argument("--post-mapping", type=Path, default=DEFAULT_POST_MAPPING)
    parser.add_argument("--refinement-log", type=Path, default=DEFAULT_REFINEMENT_LOG)
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Disable contextily basemap tiles.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pre = load_mapping(args.pre_mapping)
    post = load_mapping(args.post_mapping)
    reassignment_table = build_reassignment_table(pre, post)
    log_info = parse_refinement_log(args.refinement_log)
    summary = summarize_reassignment(
        reassignment_table,
        iterations=log_info["iterations"],
        iteration_move_count=log_info["total_reassigned"],
    )

    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    write_note(output_dir, summary, log_info)

    base_gdf = load_shapefile(args.shapefile)
    figure_path = output_dir / DEFAULT_FIGURE_NAME
    plot_refinement_figure(
        base_gdf,
        reassignment_table,
        figure_path,
        summary,
        args.dpi,
        add_basemap_flag=not args.no_basemap,
    )

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote figure: {figure_path}")
    print(
        "Final changed polygons: "
        f"{summary['n_final_changed_polygons']} / {summary['n_polygons']} "
        f"(iteration-level moves: {summary['n_iteration_reassignment_moves']})"
    )


if __name__ == "__main__":
    main()
