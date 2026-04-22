#!/usr/bin/env python3
"""
Visualize 2024 GeoRF partitioned crisis predictions vs actual outcomes
(Feb / Jun / Oct) as a single figure with a 2 row x 3 column grid:
  Row 0: Actual (y_true)
  Row 1: Predicted (y_pred_partitioned)
  Columns: February / June / October 2024

Each panel also carries a small Latin America thumbnail (Guatemala + Haiti, the
FEWSNET-covered Latin American zones) in the bottom-left corner so those small
areas are preserved when the main view is cropped to Africa + Middle East.

Reference: scripts/plot_cluster_map.py (same shapefile + basemap approach).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

try:
    import contextily as ctx
except ImportError:
    ctx = None
    print("WARNING: contextily not available. Install with: pip install contextily")
    print("Proceeding without basemap.")


PREDICTIONS_FILE = Path(
    r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\regional_ablation_results\Nigeria_experiments_local\result_partition_k40_compare_GF_fs2\predictions_monthly.csv'
)
SHAPEFILE = Path(
    r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis'
    r'\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries'
    r'\Nigeria.shp'
)
OUTPUT_FILE = Path('predictions_2024_feb_jun_oct.png')
DPI = 300

PRED_COLUMN = 'y_pred_partitioned'
TRUE_COLUMN = 'y_true'
TARGET_MONTHS = [
    ('2024-02-01', 'February 2024'),
    ('2024-06-01', 'June 2024'),
    ('2024-10-01', 'October 2024'),
]

ROWS = [
    ('Actual', TRUE_COLUMN),
    ('Predicted', PRED_COLUMN),
]

CLASS_COLORS = {
    0: '#2ca02c',  # non-crisis - green
    1: '#d62728',  # crisis     - red
}
CLASS_LABELS = {
    0: 'Non-crisis (0)',
    1: 'Crisis (1)',
}

# FEWSNET Latin America coverage. These are the only two Latin American
# countries present in FEWS_Admin_LZ_v3.shp.
LATAM_COUNTRIES = ('Guatemala', 'Haiti')


def load_predictions(pred_file: Path) -> pd.DataFrame:
    print(f"Loading predictions from {pred_file}...")
    df = pd.read_csv(pred_file)
    df['month_start'] = pd.to_datetime(df['month_start']).dt.strftime('%Y-%m-%d')
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return df


def load_shapefile(shapefile_path: Path) -> gpd.GeoDataFrame:
    print(f"\nLoading shapefile from {shapefile_path.name}...")
    gdf = gpd.read_file(shapefile_path)
    print(f"  Loaded: {len(gdf)} features  CRS: {gdf.crs}")

    uid_variations = ['FEWSNET_admin_code', 'admin_code', 'adm_code', 'FNID']
    found_col = next((v for v in uid_variations if v in gdf.columns), None)
    if found_col is None:
        raise ValueError(
            f"Could not find admin code column. Tried: {uid_variations}. "
            f"Available: {list(gdf.columns)}"
        )
    if found_col != 'FEWSNET_admin_code':
        gdf = gdf.rename(columns={found_col: 'FEWSNET_admin_code'})
        print(f"  Renamed '{found_col}' -> 'FEWSNET_admin_code'")

    invalid = (~gdf.geometry.is_valid).sum()
    if invalid:
        print(f"  Fixing {invalid} invalid geometries")
        gdf['geometry'] = gdf.geometry.buffer(0)
    return gdf


def _plot_choropleth(ax, gdf_subset: gpd.GeoDataFrame, value_col: str) -> gpd.GeoDataFrame:
    """Render a 0/1 choropleth with fixed green/red color mapping.

    Returns the subset of rows that were actually drawn (y non-null) so the
    caller can compute stats without re-filtering.
    """
    assigned = gdf_subset[gdf_subset[value_col].notna()].copy()
    if len(assigned) == 0:
        return assigned

    assigned['_class'] = assigned[value_col].astype(int)
    present_classes = sorted(assigned['_class'].unique().tolist())
    cmap = ListedColormap([CLASS_COLORS[c] for c in present_classes])

    code_map = {c: i for i, c in enumerate(present_classes)}
    assigned['_cmap_code'] = assigned['_class'].map(code_map)

    assigned.plot(
        ax=ax,
        column='_cmap_code',
        cmap=cmap,
        edgecolor='white',
        linewidth=0.15,
        legend=False,
        categorical=True,
        vmin=-0.5,
        vmax=len(present_classes) - 0.5,
    )
    return assigned


def add_latam_inset(parent_ax, latam_gdf: gpd.GeoDataFrame, value_col: str) -> None:
    """Add a small Latin America inset (Guatemala + Haiti) at bottom-left.

    The inset is kept in EPSG:4326 (no basemap) because Guatemala and Haiti
    are non-contiguous; a tiled basemap on a small inset just renders as empty
    ocean between the two.
    """
    if latam_gdf is None or len(latam_gdf) == 0:
        return

    inset = parent_ax.inset_axes([0.01, 0.01, 0.30, 0.28])
    _plot_choropleth(inset, latam_gdf, value_col)

    minx, miny, maxx, maxy = latam_gdf.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.12
    inset.set_xlim(minx - pad_x, maxx + pad_x)
    inset.set_ylim(miny - pad_y, maxy + pad_y)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor('0.35')
        spine.set_linewidth(0.9)
    inset.set_title('Latin America (FEWSNET)', fontsize=7, pad=1.5)
    inset.patch.set_facecolor('white')
    inset.patch.set_alpha(0.92)


def plot_panel(
    ax,
    gdf_main: gpd.GeoDataFrame,
    gdf_latam: gpd.GeoDataFrame,
    value_col: str,
    title: str,
) -> None:
    assigned = _plot_choropleth(ax, gdf_main, value_col)

    if ctx is not None:
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                alpha=0.4,
                attribution=False,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  WARNING: basemap failed ({title}): {exc}")

    n_total = len(assigned)
    if n_total:
        n_crisis = int((assigned['_class'] == 1).sum())
        pct_crisis = n_crisis / n_total * 100
        stats = f"Crisis: {n_crisis:,}/{n_total:,} ({pct_crisis:.1f}%)"
    else:
        stats = "No data available"
    # Stats go to the bottom-right corner to leave bottom-left free for the
    # Latin America inset.
    ax.text(
        0.98, 0.02, stats,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.6'),
    )

    ax.set_title(title, fontsize=12, weight='bold', pad=6)
    ax.axis('off')

    add_latam_inset(ax, gdf_latam, value_col)


def plot_predictions(
    merged_3857: gpd.GeoDataFrame,
    merged_4326: gpd.GeoDataFrame,
    output_file: Path,
    dpi: int = 300,
) -> None:
    print("\nCreating 2x3 actual-vs-predicted map figure...")

    # Crop main panels to Africa + Middle East (shared extent for all 6 panels).
    # EPSG:3857 longitude ~-20 deg = ~-2.226e6 m; keeps Haiti/Guatemala (Latin
    # America) out of the main frame so the African extent isn't squashed by
    # empty Atlantic space. Latin America is restored via the per-panel inset.
    AFRICA_MIN_X_M = -2_226_000.0
    ref_col = ROWS[0][1]
    has_data = merged_3857[merged_3857[ref_col].notna()]
    centroids = has_data.geometry.centroid
    in_region = has_data[centroids.x >= AFRICA_MIN_X_M]
    minx, miny, maxx, maxy = in_region.total_bounds
    pad_x = (maxx - minx) * 0.03
    pad_y = (maxy - miny) * 0.03

    if 'ADMIN0' in merged_4326.columns:
        latam_all = merged_4326[merged_4326['ADMIN0'].isin(LATAM_COUNTRIES)].copy()
        if len(latam_all):
            coverage = latam_all['ADMIN0'].value_counts().to_dict()
            print(f"  LatAm coverage: {coverage}")
        else:
            print("  WARNING: no rows matched LatAm countries; inset will be empty.")
    else:
        print("  WARNING: shapefile has no ADMIN0 column; LatAm inset disabled.")
        latam_all = merged_4326.iloc[0:0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax in axes.flat:
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    for col_idx, (month_key, month_label) in enumerate(TARGET_MONTHS):
        for row_idx, (row_label, value_col) in enumerate(ROWS):
            ax = axes[row_idx, col_idx]
            subset_main = merged_3857[merged_3857['month_start'] == month_key]
            subset_latam = latam_all[latam_all['month_start'] == month_key]
            panel_title = f"{row_label} - {month_label}"
            print(f"  {panel_title}: main={len(subset_main):,}  "
                  f"latam={len(subset_latam):,}")
            plot_panel(ax, subset_main, subset_latam, value_col, panel_title)

    for row_idx, (row_label, _) in enumerate(ROWS):
        axes[row_idx, 0].text(
            -0.04, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=15,
            weight='bold',
            rotation=90,
            ha='center',
            va='center',
        )

    legend_elements = [
        mpatches.Patch(facecolor=CLASS_COLORS[0], edgecolor='white', label=CLASS_LABELS[0]),
        mpatches.Patch(facecolor=CLASS_COLORS[1], edgecolor='white', label=CLASS_LABELS[1]),
        mpatches.Patch(facecolor='#e0e0e0', edgecolor='white', label='No data'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        'GeoRF: Actual vs Partitioned Predicted Food Crisis - 2024\n'
        '(result_partition_k40_compare_GF_fs2)',
        fontsize=15,
        weight='bold',
        y=0.98,
    )

    plt.tight_layout(rect=[0.02, 0.04, 1, 0.95])
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close(fig)


def main() -> None:
    print("=" * 80)
    print("2024 ACTUAL vs PREDICTED MAP (Feb / Jun / Oct)")
    print("=" * 80)

    df = load_predictions(PREDICTIONS_FILE)

    target_keys = [m[0] for m in TARGET_MONTHS]
    df = df[df['month_start'].isin(target_keys)].copy()
    print(f"\nFiltered to target months -> {len(df):,} rows")
    for key, label in TARGET_MONTHS:
        sub = df[df['month_start'] == key]
        if len(sub) == 0:
            raise ValueError(f"No rows for {label} ({key}) in {PREDICTIONS_FILE}")
        n_crisis_true = int((sub[TRUE_COLUMN] == 1).sum())
        n_crisis_pred = int((sub[PRED_COLUMN] == 1).sum())
        print(f"  {label}: {len(sub):,} rows | "
              f"actual crisis={n_crisis_true:,} "
              f"({n_crisis_true / len(sub) * 100:.1f}%) | "
              f"pred crisis={n_crisis_pred:,} "
              f"({n_crisis_pred / len(sub) * 100:.1f}%)")

    gdf = load_shapefile(SHAPEFILE)

    print("\nMerging predictions with geometries...")
    merge_cols = ['FEWSNET_admin_code', 'month_start', TRUE_COLUMN, PRED_COLUMN]
    merged_4326 = gdf.merge(df[merge_cols], on='FEWSNET_admin_code', how='inner')
    print(f"  Matched: {len(merged_4326):,} polygon-month rows "
          f"(from {len(gdf):,} unique polygons)")
    if len(merged_4326) == 0:
        raise ValueError("No features matched! Check FEWSNET_admin_code alignment.")

    if ctx is not None:
        print("  Reprojecting main panels to EPSG:3857 for basemap...")
        merged_3857 = merged_4326.to_crs(epsg=3857)
    else:
        merged_3857 = merged_4326

    plot_predictions(merged_3857, merged_4326, OUTPUT_FILE, DPI)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == '__main__':
    main()
