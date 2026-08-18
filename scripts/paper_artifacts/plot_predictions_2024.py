#!/usr/bin/env python3
"""
Visualize 2024 GeoRF partitioned crisis predictions vs actual outcomes
(Feb / Jun / Oct) as a single figure with a 2 row x 3 column grid:
  Row 0: Actual (y_true)
  Row 1: Predicted (y_pred_partitioned)
  Columns: February / June / October 2024

Each panel also carries separate Guatemala and Haiti thumbnails in the
bottom-left corner so both FEWSNET-covered Latin American countries remain
legible without widening the Africa + Middle East main extent.

The background is an offline vector country layer rather than network tiles.
"""

import argparse
import importlib.util
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = REPO_ROOT / 'result_partition_k40_compare_GF_fs2' / 'predictions_monthly.csv'
SHAPEFILE = Path(
    r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis'
    r'\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries'
    r'\FEWS_Admin_LZ_v3.shp'
)
OUTPUT_FILE = Path('predictions_2024_feb_jun_oct.png')
DPI = 300
MAIN_CRS = "EPSG:3857"
BASEMAP_OCEAN_COLOR = "#dfe8eb"
BASEMAP_LAND_COLOR = "#f6f3eb"
BASEMAP_BORDER_COLOR = "#aaa29d"
NO_DATA_COLOR = "#c9c9c9"

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Plot 2024 February, June, and October actual vs predicted GeoRF maps.'
    )
    parser.add_argument('--predictions', type=Path, default=PREDICTIONS_FILE)
    parser.add_argument('--shapefile', type=Path, default=SHAPEFILE)
    parser.add_argument('--output-file', type=Path, default=OUTPUT_FILE)
    parser.add_argument('--dpi', type=int, default=DPI)
    parser.add_argument('--title-source', type=str, default='result_partition_k40_compare_GF_fs2')
    parser.add_argument(
        '--admin0-shapefile',
        type=Path,
        default=None,
        help=(
            'Optional offline country-boundary shapefile for geographic context. '
            'If omitted, a local Natural Earth fixture is used when available, '
            'otherwise FEWSNET countries are dissolved as a fallback.'
        ),
    )
    parser.add_argument('--no-basemap', action='store_true')
    return parser.parse_args(argv)

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
COUNTRY_INSETS = (
    ('Guatemala', 'GTM', (0.012, 0.025, 0.17, 0.21)),
    ('Haiti', 'HTI', (0.012, 0.245, 0.13, 0.145)),
)


def load_predictions(pred_file: Path) -> pd.DataFrame:
    print(f"Loading predictions from {pred_file}...")
    df = pd.read_csv(pred_file)
    df['month_start'] = pd.to_datetime(df['month_start']).dt.strftime('%Y-%m-%d')
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return df


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


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
    gdf['FEWSNET_admin_code'] = normalize_admin_code(gdf['FEWSNET_admin_code'])

    invalid = (~gdf.geometry.is_valid).sum()
    if invalid:
        print(f"  Fixing {invalid} invalid geometries")
        gdf['geometry'] = gdf.geometry.buffer(0)
    return gdf


def discover_admin0_shapefile() -> Path | None:
    """Return an installed offline Natural Earth low-resolution shapefile."""
    spec = importlib.util.find_spec('pyogrio')
    if spec is None or spec.origin is None:
        return None
    candidate = (
        Path(spec.origin).resolve().parent
        / 'tests'
        / 'fixtures'
        / 'naturalearth_lowres'
        / 'naturalearth_lowres.shp'
    )
    return candidate if candidate.exists() else None


def _clean_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop empty geometry and repair invalid polygons for plotting."""
    cleaned = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    invalid = ~cleaned.geometry.is_valid
    if invalid.any():
        cleaned.loc[invalid, 'geometry'] = cleaned.loc[invalid].geometry.buffer(0)
    return cleaned


def build_admin0_context(base_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build an offline country layer from the FEWSNET administrative polygons."""
    admin0_col = next((col for col in ('ADMIN0', 'ISO') if col in base_gdf.columns), None)
    if admin0_col is None:
        raise ValueError('Cannot build ADMIN0 context: shapefile has no ADMIN0 or ISO column.')
    cleaned = _clean_geometry(base_gdf[[admin0_col, 'geometry']])
    return cleaned.dissolve(by=admin0_col, as_index=False)


def load_admin0_source(
    admin0_shapefile: Path | None,
    fallback_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Load a local country basemap, falling back to dissolved FEWSNET countries."""
    selected_path = admin0_shapefile or discover_admin0_shapefile()
    if selected_path is not None:
        selected_path = Path(selected_path)
        if not selected_path.exists():
            raise FileNotFoundError(f'ADMIN0 shapefile not found: {selected_path}')
        print(f"  Loading offline ADMIN0 basemap: {selected_path}")
        return _clean_geometry(gpd.read_file(selected_path))

    print('  WARNING: offline global ADMIN0 basemap unavailable; using FEWSNET countries.')
    return build_admin0_context(fallback_gdf)


def prepare_admin0_basemap(
    admin0_source: gpd.GeoDataFrame,
    target_crs,
) -> gpd.GeoDataFrame:
    """Project and lightly simplify the country context for rendering."""
    admin0 = admin0_source.to_crs(target_crs)
    crs = admin0.crs
    tolerance = 0.02 if crs is not None and crs.is_geographic else 3_000.0
    admin0 = admin0.copy()
    admin0['geometry'] = admin0.geometry.simplify(tolerance, preserve_topology=True)
    return admin0


def plot_admin0_context(
    ax,
    enabled: bool,
    admin0_gdf: gpd.GeoDataFrame,
    extent_gdf: gpd.GeoDataFrame,
) -> None:
    """Draw pale land and country borders below the thematic polygons."""
    if not enabled or admin0_gdf.empty or extent_gdf.empty:
        return
    ax.set_facecolor(BASEMAP_OCEAN_COLOR)
    minx, miny, maxx, maxy = extent_gdf.total_bounds
    subset = admin0_gdf.cx[minx:maxx, miny:maxy]
    if subset.empty:
        return
    subset.plot(
        ax=ax,
        color=BASEMAP_LAND_COLOR,
        edgecolor=BASEMAP_BORDER_COLOR,
        linewidth=0.36,
        zorder=1,
    )


def plot_admin0_outline(
    ax,
    enabled: bool,
    admin0_gdf: gpd.GeoDataFrame,
    extent_gdf: gpd.GeoDataFrame,
) -> None:
    """Redraw country boundaries above the thematic polygons."""
    if not enabled or admin0_gdf.empty or extent_gdf.empty:
        return
    minx, miny, maxx, maxy = extent_gdf.total_bounds
    subset = admin0_gdf.cx[minx:maxx, miny:maxy]
    if subset.empty:
        return
    subset.boundary.plot(
        ax=ax,
        color='#555555',
        linewidth=0.50,
        alpha=0.78,
        zorder=5,
    )


def _plot_no_data_coverage(
    ax,
    coverage_gdf: gpd.GeoDataFrame,
    data_gdf: gpd.GeoDataFrame,
    value_col: str,
) -> None:
    """Draw only FEWSNET polygons that lack an observation for this panel."""
    if coverage_gdf.empty:
        return
    assigned_codes = data_gdf.loc[
        data_gdf[value_col].notna(),
        'FEWSNET_admin_code',
    ]
    missing = coverage_gdf[
        ~coverage_gdf['FEWSNET_admin_code'].isin(assigned_codes)
    ]
    if missing.empty:
        return
    missing.plot(
        ax=ax,
        color=NO_DATA_COLOR,
        edgecolor='white',
        linewidth=0.12,
        zorder=2,
    )


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
        zorder=3,
    )
    return assigned


def add_latam_inset(
    parent_ax,
    latam_gdf: gpd.GeoDataFrame,
    latam_coverage: gpd.GeoDataFrame,
    value_col: str,
    admin0_gdf: gpd.GeoDataFrame,
    add_basemap: bool,
) -> None:
    """Add independent Guatemala and Haiti insets with country-specific bounds."""
    if latam_coverage is None or len(latam_coverage) == 0:
        return

    for country, short_label, inset_position in COUNTRY_INSETS:
        coverage = latam_coverage[latam_coverage['ADMIN0'].eq(country)].copy()
        if coverage.empty:
            continue
        data = latam_gdf[latam_gdf['ADMIN0'].eq(country)].copy()
        inset = parent_ax.inset_axes(inset_position)
        plot_admin0_context(inset, add_basemap, admin0_gdf, coverage)
        _plot_no_data_coverage(inset, coverage, data, value_col)
        _plot_choropleth(inset, data, value_col)
        plot_admin0_outline(inset, add_basemap, admin0_gdf, coverage)

        minx, miny, maxx, maxy = coverage.total_bounds
        width = maxx - minx
        height = maxy - miny
        pad_x = max(width * 0.09, 0.10)
        pad_y = max(height * 0.11, 0.08)
        inset.set_xlim(minx - pad_x, maxx + pad_x)
        inset.set_ylim(miny - pad_y, maxy + pad_y)
        inset.set_aspect('equal', adjustable='box')
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor('0.35')
            spine.set_linewidth(0.65)
        inset.text(
            0.05,
            0.95,
            short_label,
            transform=inset.transAxes,
            ha='left',
            va='top',
            fontsize=7.5,
            weight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=0.6),
        )


def plot_panel(
    ax,
    gdf_main: gpd.GeoDataFrame,
    main_coverage: gpd.GeoDataFrame,
    gdf_latam: gpd.GeoDataFrame,
    latam_coverage: gpd.GeoDataFrame,
    stats_gdf: gpd.GeoDataFrame,
    admin0_main: gpd.GeoDataFrame,
    admin0_latam: gpd.GeoDataFrame,
    value_col: str,
    add_basemap: bool,
) -> None:
    plot_admin0_context(ax, add_basemap, admin0_main, main_coverage)
    _plot_no_data_coverage(ax, main_coverage, gdf_main, value_col)
    _plot_choropleth(ax, gdf_main, value_col)
    plot_admin0_outline(ax, add_basemap, admin0_main, main_coverage)

    stats_assigned = stats_gdf[stats_gdf[value_col].notna()]
    n_total = len(stats_assigned)
    if n_total:
        n_crisis = int((stats_assigned[value_col].astype(int) == 1).sum())
        pct_crisis = n_crisis / n_total * 100
        stats = f"Crisis: {n_crisis:,}/{n_total:,} ({pct_crisis:.1f}%)"
    else:
        stats = "No data available"
    # Stats go to the bottom-right corner to leave bottom-left free for the
    # Latin America inset.
    ax.text(
        0.98, 0.02, stats,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(
            boxstyle='round,pad=0.22',
            facecolor='white',
            alpha=0.88,
            edgecolor='0.6',
            linewidth=0.6,
        ),
    )

    ax.axis('off')

    add_latam_inset(
        ax,
        gdf_latam,
        latam_coverage,
        value_col,
        admin0_latam,
        add_basemap,
    )


def plot_predictions(
    merged_4326: gpd.GeoDataFrame,
    base_4326: gpd.GeoDataFrame,
    output_file: Path,
    dpi: int = 300,
    title_source: str = 'result_partition_k40_compare_GF_fs2',
    add_basemap: bool = True,
    admin0_shapefile: Path | None = None,
) -> None:
    print("\nCreating 2x3 actual-vs-predicted map figure...")
    _ = title_source  # Retained for CLI compatibility after removing the figure title.

    if merged_4326.crs is None or base_4326.crs is None:
        raise ValueError('Both prediction and base geometries must have a CRS.')
    merged_4326 = merged_4326.to_crs(epsg=4326)
    base_4326 = base_4326.to_crs(epsg=4326)

    if 'ADMIN0' not in base_4326.columns or 'ADMIN0' not in merged_4326.columns:
        raise ValueError('ADMIN0 is required to separate the Guatemala and Haiti insets.')

    main_coverage_4326 = base_4326[
        ~base_4326['ADMIN0'].isin(LATAM_COUNTRIES)
    ].copy()
    latam_coverage = base_4326[
        base_4326['ADMIN0'].isin(LATAM_COUNTRIES)
    ].copy()
    merged_main_4326 = merged_4326[
        ~merged_4326['ADMIN0'].isin(LATAM_COUNTRIES)
    ].copy()
    latam_all = merged_4326[
        merged_4326['ADMIN0'].isin(LATAM_COUNTRIES)
    ].copy()

    coverage = latam_coverage['ADMIN0'].value_counts().to_dict()
    print(f"  LatAm coverage: {coverage}")

    main_coverage = main_coverage_4326.to_crs(MAIN_CRS)
    merged_main = merged_main_4326.to_crs(MAIN_CRS)
    admin0_source = load_admin0_source(admin0_shapefile, base_4326)
    admin0_main = prepare_admin0_basemap(admin0_source, MAIN_CRS)
    admin0_latam = prepare_admin0_basemap(admin0_source, latam_all.crs)

    minx, miny, maxx, maxy = main_coverage.total_bounds
    pad_x = (maxx - minx) * 0.015
    pad_y = (maxy - miny) * 0.02
    main_bounds = (
        minx - pad_x,
        miny - pad_y,
        maxx + pad_x,
        maxy + pad_y,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        gridspec_kw={'wspace': 0.015, 'hspace': 0.03},
    )
    fig.patch.set_facecolor('white')

    for col_idx, (month_key, month_label) in enumerate(TARGET_MONTHS):
        stats_subset = merged_4326[merged_4326['month_start'] == month_key]
        subset_main = merged_main[merged_main['month_start'] == month_key]
        subset_latam = latam_all[latam_all['month_start'] == month_key]
        for row_idx, (row_label, value_col) in enumerate(ROWS):
            ax = axes[row_idx, col_idx]
            panel_title = f"{row_label} - {month_label}"
            print(f"  {panel_title}: main={len(subset_main):,}  "
                  f"latam={len(subset_latam):,}")
            plot_panel(
                ax,
                subset_main,
                main_coverage,
                subset_latam,
                latam_coverage,
                stats_subset,
                admin0_main,
                admin0_latam,
                value_col,
                add_basemap,
            )
            ax.set_xlim(main_bounds[0], main_bounds[2])
            ax.set_ylim(main_bounds[1], main_bounds[3])
            ax.set_aspect('equal', adjustable='box')
        axes[0, col_idx].set_title(month_label, fontsize=13, weight='bold', pad=4)

    for row_idx, (row_label, _) in enumerate(ROWS):
        axes[row_idx, 0].text(
            -0.035, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=14,
            weight='bold',
            rotation=90,
            ha='center',
            va='center',
        )

    legend_elements = [
        mpatches.Patch(facecolor=CLASS_COLORS[0], edgecolor='white', label=CLASS_LABELS[0]),
        mpatches.Patch(facecolor=CLASS_COLORS[1], edgecolor='white', label=CLASS_LABELS[1]),
        mpatches.Patch(facecolor=NO_DATA_COLOR, edgecolor='white', label='No data'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )

    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.94,
        bottom=0.095,
        wspace=0.015,
        hspace=0.03,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0.04)
    print(f"Saved: {output_file}")
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    print("=" * 80)
    print("2024 ACTUAL vs PREDICTED MAP (Feb / Jun / Oct)")
    print("=" * 80)

    df = load_predictions(args.predictions)
    df['FEWSNET_admin_code'] = normalize_admin_code(df['FEWSNET_admin_code'])

    target_keys = [m[0] for m in TARGET_MONTHS]
    df = df[df['month_start'].isin(target_keys)].copy()
    print(f"\nFiltered to target months -> {len(df):,} rows")
    for key, label in TARGET_MONTHS:
        sub = df[df['month_start'] == key]
        if len(sub) == 0:
            raise ValueError(f"No rows for {label} ({key}) in {args.predictions}")
        n_crisis_true = int((sub[TRUE_COLUMN] == 1).sum())
        n_crisis_pred = int((sub[PRED_COLUMN] == 1).sum())
        print(f"  {label}: {len(sub):,} rows | "
              f"actual crisis={n_crisis_true:,} "
              f"({n_crisis_true / len(sub) * 100:.1f}%) | "
              f"pred crisis={n_crisis_pred:,} "
              f"({n_crisis_pred / len(sub) * 100:.1f}%)")

    gdf = load_shapefile(args.shapefile)
    if gdf.crs is None:
        raise ValueError(f'Shapefile has no CRS: {args.shapefile}')
    base_4326 = gdf.to_crs(epsg=4326)

    print("\nMerging predictions with geometries...")
    merge_cols = ['FEWSNET_admin_code', 'month_start', TRUE_COLUMN, PRED_COLUMN]
    merged_4326 = base_4326.merge(df[merge_cols], on='FEWSNET_admin_code', how='inner')
    print(f"  Matched: {len(merged_4326):,} polygon-month rows "
          f"(from {len(gdf):,} unique polygons)")
    if len(merged_4326) == 0:
        raise ValueError("No features matched! Check FEWSNET_admin_code alignment.")

    plot_predictions(
        merged_4326,
        base_4326,
        args.output_file,
        args.dpi,
        args.title_source,
        add_basemap=not args.no_basemap,
        admin0_shapefile=args.admin0_shapefile,
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Output: {args.output_file.resolve()}")


if __name__ == '__main__':
    main()
