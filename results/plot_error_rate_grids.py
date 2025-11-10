#!/usr/bin/env python3
"""Error rate choropleth grid generator: yearly and seasonal 4×3 maps"""
import argparse, json, re, sys, warnings
from glob import glob
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(42)

def load_predictions(pred_dir, file_glob, scope_regex, uid_col, date_col, y_true_col, y_pred_col, start_date, verbose=False):
    """Load and combine prediction CSVs with scope extraction"""
    pattern = Path(pred_dir) / file_glob
    files = sorted(glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    if verbose:
        print(f"Found {len(files)} CSV files")

    scope_pattern, dfs = re.compile(scope_regex), []
    for fpath in files:
        fname = Path(fpath).name
        match = scope_pattern.search(fname)
        if not match:
            if verbose: print(f"  Skipping {fname} (no scope match)")
            continue
        scope = 'fs' + match.group('scope')
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            if verbose: print(f"  Error reading {fname}: {e}")
            continue
        required = [uid_col, y_true_col, y_pred_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {fname}")
        df['scope'] = scope
        if date_col and date_col in df.columns:
            df['date'] = pd.to_datetime(df[date_col])
            df['year'], df['month'] = df['date'].dt.year, df['date'].dt.month
        elif 'year' in df.columns and 'month' in df.columns:
            df['year'], df['month'] = df['year'].astype(int), df['month'].astype(int)
        else:
            raise ValueError(f"Cannot derive date from {fname}: need '{date_col}' or 'year'+'month'")
        dfs.append(df[[uid_col, 'year', 'month', y_pred_col, y_true_col, 'scope']])
        if verbose: print(f"  Loaded {fname}: {len(df)} rows → scope={scope}")

    if not dfs:
        raise ValueError("No valid CSV files loaded")
    combined = pd.concat(dfs, ignore_index=True)
    start_dt = pd.to_datetime(start_date)
    combined = combined[(combined['year'] > start_dt.year) |
                       ((combined['year'] == start_dt.year) & (combined['month'] >= start_dt.month))].copy()
    if len(combined) == 0:
        raise ValueError(f"No data after filtering for start_date={start_date}")

    # DJF special: December belongs to next year's DJF
    def assign_season(row):
        m, y = row['month'], row['year']
        if m == 12: return pd.Series({'season': 'DJF', 'season_year': y+1})
        elif m in [1,2]: return pd.Series({'season': 'DJF', 'season_year': y})
        elif m in [3,4,5]: return pd.Series({'season': 'MAM', 'season_year': y})
        elif m in [6,7,8]: return pd.Series({'season': 'JJA', 'season_year': y})
        else: return pd.Series({'season': 'SON', 'season_year': y})

    season_data = combined.apply(assign_season, axis=1)
    combined['season'], combined['season_year'] = season_data['season'], season_data['season_year']
    if verbose:
        print(f"Combined: {len(combined)} rows after {start_date} filter")
        print(f"  Years: {sorted(combined['year'].unique())}, Scopes: {sorted(combined['scope'].unique())}")
    return combined

def aggregate_error_rate(df, groupby_cols, uid_col, y_pred_col, y_true_col, verbose=False):
    """Aggregate error rates by grouping columns"""
    df = df.copy()
    df['error'] = (df[y_pred_col] != df[y_true_col]).astype(int)
    agg_df = df.groupby(groupby_cols + [uid_col]).agg(err_rate=('error', 'mean'), n=('error', 'count')).reset_index()
    agg_df['err_rate'] = agg_df['err_rate'].clip(0.0, 1.0)
    assert agg_df['err_rate'].between(0, 1).all() and (agg_df['n'] > 0).all()
    if verbose:
        print(f"Aggregated by {groupby_cols}: {len(agg_df)} rows, mean err={agg_df['err_rate'].mean():.1%}")
    return agg_df

def prepare_polys(polygons_path, uid_col, target_crs='EPSG:4326', verbose=False):
    """Load and prepare polygon GeoDataFrame"""
    gdf = gpd.read_file(polygons_path)
    if uid_col not in gdf.columns:
        raise ValueError(f"UID column '{uid_col}' not in {polygons_path}. Available: {list(gdf.columns)}")
    invalid = ~gdf.geometry.is_valid
    if invalid.sum() > 0:
        if verbose: print(f"  Fixing {invalid.sum()} invalid geometries")
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].buffer(0)
    if gdf.crs is None:
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    if verbose: print(f"Loaded polygons: {len(gdf)} features, CRS={gdf.crs}")
    return gdf

def render_grid(gdf_list, titles, nrows, ncols, value_col, vmin, vmax, cmap, missing_color, dpi, output_path, verbose=False):
    """Render 4×3 choropleth grid with shared colorbar"""
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=dpi)
    axes = axes.flatten()
    for idx, (gdf, title, ax) in enumerate(zip(gdf_list, titles, axes)):
        if gdf is None or len(gdf) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')
            continue
        data_mask = gdf[value_col].notna()
        if data_mask.sum() > 0:
            gdf[data_mask].plot(ax=ax, column=value_col, cmap=cmap, vmin=vmin, vmax=vmax,
                              edgecolor='white', linewidth=0.3, legend=False)
        no_data_mask = ~data_mask
        if no_data_mask.sum() > 0:
            gdf[no_data_mask].plot(ax=ax, color=missing_color, alpha=0.7, hatch='///',
                                  edgecolor='white', linewidth=0.1, legend=False)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')
        if verbose and idx == 0:
            print(f"  Subplot '{title}': {data_mask.sum()} data, {no_data_mask.sum()} missing")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, aspect=30,
                       format=PercentFormatter(xmax=1.0, decimals=0))
    cbar.set_label('Error Rate', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    if verbose: print(f"Saved grid: {output_path}")


def main():
    p = argparse.ArgumentParser(description='Generate error rate choropleth grids (yearly/seasonal)',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--pred-dir', required=True, help='Directory containing prediction CSVs')
    p.add_argument('--polygons', required=True, help='Path to polygon shapefile/geopackage')
    p.add_argument('--uid-col', default='adm_code', help='UID column name')
    p.add_argument('--poly-uid-col', default=None, help='UID column in polygons (defaults to --uid-col)')
    p.add_argument('--date-col', default=None, help='Date column (None=use year+month)')
    p.add_argument('--scopes', nargs='+', default=['fs1', 'fs2', 'fs3'], help='Scopes to plot')
    p.add_argument('--start', default='2021-01-01', help='Start date filter')
    p.add_argument('--out-dir', default='.', help='Output directory')
    p.add_argument('--vmin', type=float, default=0.0, help='Colorbar min')
    p.add_argument('--vmax', type=float, default=1.0, help='Colorbar max')
    p.add_argument('--dpi', type=int, default=200, help='DPI')
    p.add_argument('--cmap', default='Reds', help='Colormap')
    p.add_argument('--missing-color', default='#dddddd', help='Missing data color')
    p.add_argument('--verbose', action='store_true', help='Verbose')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poly_uid_col = args.poly_uid_col or args.uid_col
    if args.verbose: print("="*60 + "\nERROR RATE CHOROPLETH GRID GENERATOR\n" + "="*60)

    df = load_predictions(args.pred_dir, 'y_pred_test_gp_fs*_*.csv', r'fs(?P<scope>\d+)',
                         args.uid_col, args.date_col, 'fews_ipc_crisis_true',
                         'fews_ipc_crisis_pred', args.start, args.verbose)

    df_yearly = aggregate_error_rate(df, ['scope', 'year'], args.uid_col,
                                     'fews_ipc_crisis_pred', 'fews_ipc_crisis_true', args.verbose)
    df_yearly.to_csv(out_dir / 'error_rate_yearly.csv', index=False)

    df_seasonal = aggregate_error_rate(df, ['scope', 'season', 'season_year'], args.uid_col,
                                       'fews_ipc_crisis_pred', 'fews_ipc_crisis_true', args.verbose)
    df_seasonal.to_csv(out_dir / 'error_rate_seasonal.csv', index=False)

    gdf_base = prepare_polys(args.polygons, poly_uid_col, verbose=args.verbose)

    available_scopes = sorted(df['scope'].unique())
    if args.verbose and (missing := [s for s in args.scopes if s not in available_scopes]):
        print(f"WARNING: Requested scopes not in data: {missing}")
    scopes = [s for s in args.scopes if s in available_scopes]
    if not scopes:
        raise ValueError(f"No valid scopes. Requested: {args.scopes}, Available: {available_scopes}")

    years = sorted(df_yearly['year'].unique())[-4:]
    gdf_list_yearly, titles_yearly = [], []
    for year in years:
        for scope in scopes:
            subset = df_yearly[(df_yearly['year'] == year) & (df_yearly['scope'] == scope)]
            gdf_list_yearly.append(gdf_base.merge(subset, left_on=poly_uid_col, right_on=args.uid_col, how='inner'))
            titles_yearly.append(f'Year {year} — {scope}')

    render_grid(gdf_list_yearly, titles_yearly, len(years), len(scopes), 'err_rate',
               args.vmin, args.vmax, args.cmap, args.missing_color, args.dpi,
               out_dir / 'error_rate_yearly_4x3.png', args.verbose)

    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    season_years = df_seasonal.groupby('season')['season_year'].max().to_dict()
    gdf_list_seasonal, titles_seasonal = [], []
    for season in seasons:
        sy = season_years.get(season)
        for scope in scopes:
            if sy:
                subset = df_seasonal[(df_seasonal['season'] == season) &
                                   (df_seasonal['season_year'] == sy) &
                                   (df_seasonal['scope'] == scope)]
                gdf_merged = gdf_base.merge(subset, left_on=poly_uid_col, right_on=args.uid_col, how='inner')
            else:
                gdf_merged = None
            gdf_list_seasonal.append(gdf_merged)
            titles_seasonal.append(f'{season}{f" ({sy})" if sy else ""} — {scope}')

    render_grid(gdf_list_seasonal, titles_seasonal, 4, len(scopes), 'err_rate',
               args.vmin, args.vmax, args.cmap, args.missing_color, args.dpi,
               out_dir / 'error_rate_seasonal_4x3.png', args.verbose)

    report = {'files_loaded': len(glob(str(Path(args.pred_dir) / 'y_pred_test_gp_fs*_*.csv'))),
              'rows_after_filter': len(df), 'years_plotted': years, 'scopes_used': scopes,
              'season_counts': {s: len(df_seasonal[df_seasonal['season'] == s]) for s in seasons}}
    with open(out_dir / 'error_rate_extract_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    if args.verbose:
        print("="*60 + "\nCOMPLETED\n" + "="*60 + f"\nOutputs: {out_dir}\n  - error_rate_yearly.csv"
              "\n  - error_rate_seasonal.csv\n  - error_rate_yearly_4x3.png"
              "\n  - error_rate_seasonal_4x3.png\n  - error_rate_extract_report.json")
    return 0

if __name__ == '__main__':
    sys.exit(main())
