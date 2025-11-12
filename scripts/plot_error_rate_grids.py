#!/usr/bin/env python3
"""
Error Rate Choropleth Grid Generator
Loads prediction CSVs, computes polygon-level error rates, and generates
yearly and seasonal choropleth grids (4×3 yearly, 3×3 seasonal).
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import numpy as np

try:
    import contextily as ctx
except ImportError:
    ctx = None

warnings.filterwarnings('ignore', category=FutureWarning)


def map_month_to_season(month: int) -> str:
    """Map month to season (DJFM, AMJJ, ASON). Dec-Mar → DJFM, Apr-Jul → AMJJ, Aug-Nov → ASON."""
    if month in [12, 1, 2, 3]:
        return "DJFM"
    elif month in [4, 5, 6, 7]:
        return "AMJJ"
    elif month in [8, 9, 10, 11]:
        return "ASON"


def adjust_year_for_djf(row: pd.Series) -> int:
    """For DJF season, December uses next year's label."""
    if row['season'] == 'DJFM' and row['month'] == 12:
        return row['year'] + 1
    return row['year']


def load_predictions(
    pred_dir: Path,
    file_glob: str,
    scope_regex: str,
    uid_col: str,
    date_col: str,
    y_true_col: str,
    y_pred_col: str,
    start_date: str,
    verbose: bool
) -> pd.DataFrame:
    """Load and combine all prediction CSVs, extract scope, filter by date."""

    csv_files = sorted(pred_dir.glob(file_glob))
    if not csv_files:
        # Show available files to help debug
        all_csv = list(pred_dir.glob('*.csv'))
        if all_csv and verbose:
            print(f"No files matching '{file_glob}' in {pred_dir}")
            print(f"Available CSV files: {', '.join([f.name for f in all_csv[:5]])}")
        raise FileNotFoundError(f"No files matching '{file_glob}' in {pred_dir}")

    if verbose:
        print(f"Found {len(csv_files)} prediction files:")
        for f in csv_files[:5]:
            print(f"  - {f.name}")
        if len(csv_files) > 5:
            print(f"  ... and {len(csv_files) - 5} more")

    pattern = re.compile(scope_regex)
    frames = []

    for fpath in csv_files:
        match = pattern.search(fpath.name)
        if not match:
            if verbose:
                print(f"  Skipping {fpath.name} (no scope match)")
            continue

        scope = f"fs{match.group('scope')}"

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            if verbose:
                print(f"  ERROR reading {fpath.name}: {e}")
            continue

        # Find uid column - try multiple common variations
        uid_variations = [uid_col, 'uid', 'admin_code', 'adm_code', 'FEWSNET_admin_code']
        found_uid = None
        for var in uid_variations:
            if var in df.columns:
                found_uid = var
                break

        if found_uid is None:
            if verbose:
                print(f"  ERROR in {fpath.name}: no uid column found (tried {uid_variations})")
            continue

        # Rename to standard uid_col
        if found_uid != uid_col:
            df = df.rename(columns={found_uid: uid_col})

        # Handle date column - construct from year/month if needed
        if date_col not in df.columns:
            if 'year' in df.columns and 'month' in df.columns:
                # Construct date from year/month (use day=1)
                df[date_col] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                if verbose and fpath == csv_files[0]:  # Only print once
                    print(f"  Constructing '{date_col}' from year/month columns")
            else:
                if verbose:
                    print(f"  ERROR in {fpath.name}: no '{date_col}' or year/month columns")
                continue

        # Check required columns
        required = [uid_col, date_col, y_true_col, y_pred_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            if verbose:
                print(f"  ERROR in {fpath.name}: missing columns {missing}")
            continue

        df = df[[uid_col, date_col, y_true_col, y_pred_col]].copy()
        df['scope'] = scope
        frames.append(df)

        if verbose and len(frames) <= 3:  # Print first few
            print(f"  Loaded {fpath.name} → {scope}: {len(df)} rows")

    if verbose and len(frames) > 3:
        print(f"  ... loaded {len(frames)} files total")

    if not frames:
        raise ValueError("No valid prediction data loaded")

    combined = pd.concat(frames, ignore_index=True)

    # Parse dates (may already be datetime if constructed above)
    if not pd.api.types.is_datetime64_any_dtype(combined[date_col]):
        combined[date_col] = pd.to_datetime(combined[date_col], errors='coerce')
        invalid_dates = combined[date_col].isna().sum()
        if invalid_dates > 0:
            if verbose:
                print(f"WARNING: {invalid_dates} rows with invalid dates, dropping")
            combined = combined.dropna(subset=[date_col])

    # Filter by start date
    start = pd.to_datetime(start_date)
    combined = combined[combined[date_col] >= start].copy()

    if len(combined) == 0:
        raise ValueError(f"No data after filtering for dates >= {start_date}")

    # Derive year, month, season
    combined['year'] = combined[date_col].dt.year
    combined['month'] = combined[date_col].dt.month
    combined['season'] = combined['month'].apply(map_month_to_season)

    # Adjust year for December in DJF
    combined['year'] = combined.apply(adjust_year_for_djf, axis=1)

    if verbose:
        print(f"\nCombined: {len(combined)} rows from {combined['year'].min()} to {combined['year'].max()}")
        print(f"Scopes: {sorted(combined['scope'].unique())}")

    return combined


def aggregate_error_rate(
    df: pd.DataFrame,
    by: List[str],
    uid_col: str,
    y_true_col: str,
    y_pred_col: str,
    verbose: bool
) -> pd.DataFrame:
    """
    Aggregate error rate by grouping columns.
    Returns DataFrame with columns: by + [uid_col, 'err_rate', 'n']
    """

    df = df.copy()
    df['error'] = (df[y_pred_col] != df[y_true_col]).astype(int)

    group_cols = by + [uid_col]

    agg = df.groupby(group_cols, as_index=False).agg(
        n=('error', 'count'),
        err_rate=('error', 'mean')
    )

    # Clamp to [0, 1]
    agg['err_rate'] = agg['err_rate'].clip(0.0, 1.0)

    # Filter out empty groups
    agg = agg[agg['n'] > 0].copy()

    if verbose:
        print(f"\nAggregated by {by}:")
        print(f"  Total groups: {len(agg)}")
        print(f"  Error rate range: [{agg['err_rate'].min():.3f}, {agg['err_rate'].max():.3f}]")

    return agg


def prepare_polys(
    polygons_path: Path,
    uid_col: str,
    verbose: bool
) -> gpd.GeoDataFrame:
    """Load polygons, ensure valid geometries and CRS."""

    polys = gpd.read_file(polygons_path)

    if verbose:
        print(f"\nLoaded polygons from {polygons_path.name}: {len(polys)} features")

    # Find uid column with variations
    uid_variations = [uid_col, 'uid', 'admin_code', 'adm_code', 'FEWSNET_admin_code']
    found_uid = None
    for var in uid_variations:
        if var in polys.columns:
            found_uid = var
            break

    if found_uid is None:
        raise ValueError(f"Polygon uid column not found. Tried: {uid_variations}")

    if found_uid != uid_col:
        polys = polys.rename(columns={found_uid: uid_col})
        if verbose:
            print(f"  Renamed polygon column '{found_uid}' → '{uid_col}'")

    # Ensure valid geometries
    invalid = (~polys.geometry.is_valid).sum()
    if invalid > 0:
        if verbose:
            print(f"  Fixing {invalid} invalid geometries")
        polys['geometry'] = polys.geometry.buffer(0)

    # Set CRS if missing
    if polys.crs is None:
        polys = polys.set_crs('EPSG:4326', allow_override=True)
        if verbose:
            print(f"  Set CRS to EPSG:4326")

    return polys


def render_grid(
    agg_df: pd.DataFrame,
    polys: gpd.GeoDataFrame,
    uid_col: str,
    row_values: List[Any],
    col_values: List[str],
    row_label: str,
    col_label: str,
    value_col: str,
    vmin: float,
    vmax: float,
    missing_color: str,
    dpi: int,
    cmap_name: str,
    output_path: Path,
    add_basemap: bool,
    verbose: bool
) -> Dict[str, int]:
    """
    Render grid of choropleth maps (4×3 yearly, 3×3 seasonal).
    Returns dict of subplot counts for reporting.
    """

    n_rows = len(row_values)
    n_cols = len(col_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Setup colormap
    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = BoundaryNorm(
        boundaries=np.linspace(vmin, vmax, 11),  # 0%, 10%, ..., 100%
        ncolors=cmap.N,
        clip=True
    )

    # Reproject for basemap if needed
    if add_basemap and ctx is not None:
        polys_plot = polys.to_crs(epsg=3857)
    else:
        polys_plot = polys

    subplot_counts = {}

    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axes[i, j]

            # Filter data for this subplot
            subset = agg_df[
                (agg_df[row_label] == row_val) &
                (agg_df[col_label] == col_val)
            ].copy()

            title = f"{row_label.capitalize()} {row_val} — {col_val}"

            if len(subset) == 0:
                # No data
                polys_plot.plot(ax=ax, color=missing_color, edgecolor='white', linewidth=0.3, alpha=0.6)

                # Add basemap
                if add_basemap and ctx is not None:
                    try:
                        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
                    except Exception:
                        pass  # Silently fail if basemap unavailable

                ax.set_title(title, fontsize=10, weight='bold')
                ax.axis('off')
                ax.text(
                    0.5, 0.5, 'No data',
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=12, color='gray'
                )
                subplot_counts[f"{row_val}_{col_val}"] = 0
                if verbose:
                    print(f"  [{i},{j}] {title}: NO DATA")
                continue

            # Merge with polygons
            merged = polys.merge(subset, on=uid_col, how='inner')

            if len(merged) == 0:
                # No matching polygons
                polys_plot.plot(ax=ax, color=missing_color, edgecolor='white', linewidth=0.3, alpha=0.6)

                # Add basemap
                if add_basemap and ctx is not None:
                    try:
                        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
                    except Exception:
                        pass

                ax.set_title(title, fontsize=10, weight='bold')
                ax.axis('off')
                ax.text(
                    0.5, 0.5, 'No matches',
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=12, color='gray'
                )
                subplot_counts[f"{row_val}_{col_val}"] = 0
                if verbose:
                    print(f"  [{i},{j}] {title}: NO MATCHES")
                continue

            # Reproject merged data if using basemap
            if add_basemap and ctx is not None:
                merged = merged.to_crs(epsg=3857)

            # Plot base (all polygons in missing color)
            polys_plot.plot(ax=ax, color=missing_color, edgecolor='white', linewidth=0.3, alpha=0.6)

            # Plot data polygons
            merged.plot(
                ax=ax,
                column=value_col,
                cmap=cmap,
                norm=norm,
                edgecolor='white',
                linewidth=0.3,
                legend=False,
                alpha=0.8
            )

            # Add basemap underneath
            if add_basemap and ctx is not None:
                try:
                    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
                except Exception:
                    pass  # Silently fail if basemap unavailable

            ax.set_title(title, fontsize=10, weight='bold')
            ax.axis('off')

            subplot_counts[f"{row_val}_{col_val}"] = len(merged)

            if verbose:
                print(f"  [{i},{j}] {title}: {len(merged)} polygons, "
                      f"err_rate={merged[value_col].mean():.2%}")

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation='horizontal',
        fraction=0.02,
        pad=0.02,
        aspect=40
    )
    cbar.set_label('Error Rate', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Format ticks as percentages
    ticks = np.linspace(vmin, vmax, 11)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved: {output_path}")

    return subplot_counts


def main():
    # Default paths
    DEFAULT_PRED_DIR = Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\results')
    DEFAULT_POLYGONS = Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp')

    parser = argparse.ArgumentParser(
        description="Generate error rate choropleth grids (yearly & seasonal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--pred-dir', type=Path, default=DEFAULT_PRED_DIR,
                        help='Directory containing prediction CSVs')
    parser.add_argument('--polygons', type=Path, default=DEFAULT_POLYGONS,
                        help='Path to polygon file (.gpkg or .shp)')
    parser.add_argument('--uid-col', type=str, default='adm_code',
                        help='UID column name (tries common variations: adm_code, admin_code, FEWSNET_admin_code)')
    parser.add_argument('--date-col', type=str, default='date',
                        help='Date column name')
    parser.add_argument('--y-true-col', type=str, default='fews_ipc_crisis_true',
                        help='True label column')
    parser.add_argument('--y-pred-col', type=str, default='fews_ipc_crisis_pred',
                        help='Predicted label column')
    parser.add_argument('--file-glob', type=str, default='y_pred_test_*gp_fs*_*_*.csv',
                        help='File pattern to match (y_pred_test_gp_fs{scope}_{year}_{year}.csv or y_pred_test_xgb_gp_fs{scope}_{year}_{year}.csv)')
    parser.add_argument('--scope-regex', type=str, default=r'fs(?P<scope>\d+)',
                        help='Regex to extract scope from filename')
    parser.add_argument('--scopes', nargs='+', default=['fs1', 'fs2', 'fs3'],
                        help='Scopes to display')
    parser.add_argument('--start', type=str, default='2021-01-01',
                        help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--out-dir', type=Path, default=Path('.'),
                        help='Output directory')
    parser.add_argument('--vmin', type=float, default=0.0,
                        help='Color scale minimum')
    parser.add_argument('--vmax', type=float, default=1.0,
                        help='Color scale maximum')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure DPI (default: 300 for high quality)')
    parser.add_argument('--missing-color', type=str, default='#dddddd',
                        help='Color for missing data')
    parser.add_argument('--cmap', type=str, default='YlOrRd',
                        help='Matplotlib colormap')
    parser.add_argument('--basemap', action='store_true', default=True,
                        help='Add grey basemap for geographic context (default: True)')
    parser.add_argument('--no-basemap', dest='basemap', action='store_false',
                        help='Disable basemap')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Setup output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    print("=" * 60)
    print("ERROR RATE CHOROPLETH GRID GENERATOR")
    print("=" * 60)

    # Check basemap availability
    if args.basemap and ctx is None:
        print("\nWARNING: contextily not available. Install with: pip install contextily")
        print("Proceeding without basemap.\n")
        args.basemap = False
    elif args.basemap and args.verbose:
        print("\nBasemap enabled: CartoDB.Positron\n")

    # Load predictions
    df = load_predictions(
        pred_dir=args.pred_dir,
        file_glob=args.file_glob,
        scope_regex=args.scope_regex,
        uid_col=args.uid_col,
        date_col=args.date_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        start_date=args.start,
        verbose=args.verbose
    )

    # Load polygons
    polys = prepare_polys(
        polygons_path=args.polygons,
        uid_col=args.uid_col,
        verbose=args.verbose
    )

    # Aggregate yearly
    print("\n" + "=" * 60)
    print("YEARLY AGGREGATION")
    print("=" * 60)

    yearly = aggregate_error_rate(
        df=df,
        by=['scope', 'year'],
        uid_col=args.uid_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        verbose=args.verbose
    )

    yearly_csv = args.out_dir / 'error_rate_yearly.csv'
    yearly.to_csv(yearly_csv, index=False)
    print(f"Saved: {yearly_csv}")

    # Aggregate seasonal
    print("\n" + "=" * 60)
    print("SEASONAL AGGREGATION")
    print("=" * 60)

    seasonal = aggregate_error_rate(
        df=df,
        by=['scope', 'season'],
        uid_col=args.uid_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        verbose=args.verbose
    )

    seasonal_csv = args.out_dir / 'error_rate_seasonal.csv'
    seasonal.to_csv(seasonal_csv, index=False)
    print(f"Saved: {seasonal_csv}")

    # Filter to requested scopes
    yearly_filt = yearly[yearly['scope'].isin(args.scopes)].copy()
    seasonal_filt = seasonal[seasonal['scope'].isin(args.scopes)].copy()

    # Get sorted years (up to 4)
    years = sorted(yearly_filt['year'].unique())[-4:]
    seasons = ['DJFM', 'AMJJ', 'ASON']

    if len(years) == 0:
        print("\nWARNING: No years available for plotting")
    else:
        # Render yearly grid
        print("\n" + "=" * 60)
        print("RENDERING YEARLY GRID")
        print("=" * 60)

        yearly_png = args.out_dir / 'error_rate_yearly_4x3.png'
        yearly_counts = render_grid(
            agg_df=yearly_filt,
            polys=polys,
            uid_col=args.uid_col,
            row_values=years,
            col_values=args.scopes,
            row_label='year',
            col_label='scope',
            value_col='err_rate',
            vmin=args.vmin,
            vmax=args.vmax,
            missing_color=args.missing_color,
            dpi=args.dpi,
            cmap_name=args.cmap,
            output_path=yearly_png,
            add_basemap=args.basemap,
            verbose=args.verbose
        )

    if len(seasonal_filt) == 0:
        print("\nWARNING: No seasonal data available for plotting")
    else:
        # Render seasonal grid
        print("\n" + "=" * 60)
        print("RENDERING SEASONAL GRID")
        print("=" * 60)

        seasonal_png = args.out_dir / 'error_rate_seasonal_3x3.png'
        seasonal_counts = render_grid(
            agg_df=seasonal_filt,
            polys=polys,
            uid_col=args.uid_col,
            row_values=seasons,
            col_values=args.scopes,
            row_label='season',
            col_label='scope',
            value_col='err_rate',
            vmin=args.vmin,
            vmax=args.vmax,
            missing_color=args.missing_color,
            dpi=args.dpi,
            cmap_name=args.cmap,
            output_path=seasonal_png,
            add_basemap=args.basemap,
            verbose=args.verbose
        )

    # Generate report (convert numpy types to Python native for JSON serialization)
    report = {
        'files_loaded': len(list(args.pred_dir.glob(args.file_glob))),
        'rows_after_filter': int(len(df)),
        'years_plotted': [int(y) for y in years],
        'scopes_used': args.scopes,
        'season_counts': {
            season: int(len(seasonal_filt[seasonal_filt['season'] == season]))
            for season in seasons
        },
        'yearly_polygons': {k: int(v) for k, v in yearly_counts.items()} if len(years) > 0 else {},
        'seasonal_polygons': {k: int(v) for k, v in seasonal_counts.items()} if len(seasonal_filt) > 0 else {}
    }

    report_json = args.out_dir / 'error_rate_extract_report.json'
    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Report: {report_json}")
    print(f"Years plotted: {report['years_plotted']}")
    print(f"Scopes used: {report['scopes_used']}")


if __name__ == '__main__':
    main()
