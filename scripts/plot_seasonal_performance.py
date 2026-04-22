#!/usr/bin/env python3
"""
Seasonal Performance Maps & Tables Generator
=============================================
Loads predictions_monthly.csv files (fs1/fs2/fs3), generates:
  - 3 seasonal choropleth maps (all / crisis-only / noncrisis-only)
  - Table 1: model performance by season
  - Table 2: model performance by region (builtin FEWSNET mapping)

Reference: scripts/plot_error_rate_grids.py
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm

try:
    import contextily as ctx
except ImportError:
    ctx = None

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Reusable helpers (adapted from plot_error_rate_grids.py)
# ---------------------------------------------------------------------------

def map_month_to_season(month: int) -> str:
    if month in [12, 1, 2, 3]:
        return "DJFM"
    elif month in [4, 5, 6, 7]:
        return "AMJJ"
    elif month in [8, 9, 10, 11]:
        return "ASON"


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'support': len(y_true)}


def _prf(tp: int, fp: int, fn: int, tn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if not (np.isnan(precision) or np.isnan(recall) or precision + recall == 0)
          else np.nan)
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else np.nan
    return precision, recall, f1, accuracy


LATAM_COUNTRIES = ('Guatemala', 'Haiti')


REGION_MAP = {
    'Burundi': 'East Africa', 'Djibouti': 'East Africa', 'Eritrea': 'East Africa',
    'Ethiopia': 'East Africa', 'Kenya': 'East Africa', 'Rwanda': 'East Africa',
    'Somalia': 'East Africa', 'South Sudan': 'East Africa', 'Sudan': 'East Africa',
    'Tanzania': 'East Africa', 'Uganda': 'East Africa',
    'Burkina Faso': 'West Africa', 'Chad': 'West Africa', 'Gambia': 'West Africa',
    'Ghana': 'West Africa', 'Guinea': 'West Africa', 'Liberia': 'West Africa',
    'Mali': 'West Africa', 'Mauritania': 'West Africa', 'Niger': 'West Africa',
    'Nigeria': 'West Africa', 'Senegal': 'West Africa', 'Sierra Leone': 'West Africa',
    'Togo': 'West Africa', 'Benin': 'West Africa', "Cote d'Ivoire": 'West Africa',
    'Ivory Coast': 'West Africa',
    'Angola': 'Southern Africa', 'Botswana': 'Southern Africa', 'Lesotho': 'Southern Africa',
    'Madagascar': 'Southern Africa', 'Malawi': 'Southern Africa', 'Mozambique': 'Southern Africa',
    'Namibia': 'Southern Africa', 'South Africa': 'Southern Africa',
    'Swaziland': 'Southern Africa', 'Eswatini': 'Southern Africa',
    'Zambia': 'Southern Africa', 'Zimbabwe': 'Southern Africa',
    'Egypt': 'North Africa', 'Libya': 'North Africa', 'Morocco': 'North Africa',
    'Tunisia': 'North Africa', 'Algeria': 'North Africa',
    'Afghanistan': 'Middle East', 'Iraq': 'Middle East', 'Palestine': 'Middle East',
    'Syria': 'Middle East', 'Yemen': 'Middle East', 'Jordan': 'Middle East',
    'Lebanon': 'Middle East',
    'Bangladesh': 'South Asia', 'Nepal': 'South Asia', 'Pakistan': 'South Asia',
    'India': 'South Asia', 'Sri Lanka': 'South Asia',
    'Myanmar': 'Southeast Asia', 'Philippines': 'Southeast Asia',
    'Cambodia': 'Southeast Asia', 'Laos': 'Southeast Asia',
    'Thailand': 'Southeast Asia', 'Vietnam': 'Southeast Asia',
    'Indonesia': 'Southeast Asia',
    'El Salvador': 'Latin America', 'Guatemala': 'Latin America',
    'Haiti': 'Latin America', 'Honduras': 'Latin America',
    'Nicaragua': 'Latin America', 'Colombia': 'Latin America',
    'Ecuador': 'Latin America', 'Peru': 'Latin America',
    'Bolivia': 'Latin America', 'Venezuela': 'Latin America',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_scopes(
    fs1_path: Path, fs2_path: Path, fs3_path: Path,
    y_pred_col: str = 'y_pred_partitioned',
) -> pd.DataFrame:
    """Load the three scope CSVs and combine into one DataFrame."""
    frames = []
    for path, scope in [(fs1_path, 'fs1'), (fs2_path, 'fs2'), (fs3_path, 'fs3')]:
        df = pd.read_csv(path)
        df['scope'] = scope
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Normalise column names to the standard used downstream
    combined = combined.rename(columns={
        'FEWSNET_admin_code': 'uid',
        'month_start': 'date',
        'y_true': 'y_true',
    })

    # Keep a single prediction column named y_pred
    if y_pred_col in combined.columns:
        combined = combined.rename(columns={y_pred_col: 'y_pred'})
    else:
        raise ValueError(f"Prediction column '{y_pred_col}' not found. "
                         f"Available: {list(combined.columns)}")

    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined['year'] = combined['date'].dt.year
    combined['month'] = combined['date'].dt.month
    combined['season'] = combined['month'].apply(map_month_to_season)

    # Adjust Dec -> next year for DJFM
    mask = (combined['season'] == 'DJFM') & (combined['month'] == 12)
    combined.loc[mask, 'year'] = combined.loc[mask, 'year'] + 1

    print(f"Loaded {len(combined):,} rows  |  "
          f"date range {combined['date'].min().date()} - {combined['date'].max().date()}  |  "
          f"scopes {sorted(combined['scope'].unique())}")
    return combined


def load_polys(polygons_path: Path) -> gpd.GeoDataFrame:
    polys = gpd.read_file(polygons_path)

    for col in ['FEWSNET_admin_code', 'uid', 'admin_code', 'adm_code']:
        if col in polys.columns:
            polys = polys.rename(columns={col: 'uid'})
            break
    if 'uid' not in polys.columns:
        raise ValueError(f"No uid column in polygons: {list(polys.columns)}")

    invalid = (~polys.geometry.is_valid).sum()
    if invalid > 0:
        polys['geometry'] = polys.geometry.buffer(0)
    if polys.crs is None:
        polys = polys.set_crs('EPSG:4326', allow_override=True)

    return polys


# ---------------------------------------------------------------------------
# Error-rate aggregation (polygon-level, for choropleth)
# ---------------------------------------------------------------------------

def aggregate_error_rate(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    df = df.copy()
    df['error'] = (df['y_pred'] != df['y_true']).astype(int)
    agg = df.groupby(by + ['uid'], as_index=False).agg(
        n=('error', 'count'),
        err_rate=('error', 'mean'),
    )
    agg['err_rate'] = agg['err_rate'].clip(0.0, 1.0)
    return agg[agg['n'] > 0].copy()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _add_latam_inset(
    parent_ax,
    latam_polys: gpd.GeoDataFrame,
    sub_df: pd.DataFrame,
    cmap,
    norm,
    missing_color: str,
) -> None:
    """Bottom-left inset showing FEWSNET Latin America (Guatemala + Haiti).

    Kept in EPSG:4326 (no basemap) because Guatemala and Haiti are
    non-contiguous; a tiled basemap on a small inset would render mostly as
    empty ocean between the two countries.
    """
    if latam_polys is None or len(latam_polys) == 0:
        return

    inset = parent_ax.inset_axes([0.01, 0.01, 0.30, 0.28])
    latam_polys.plot(ax=inset, color=missing_color, edgecolor='white',
                     linewidth=0.3, alpha=0.6)
    if sub_df is not None and len(sub_df):
        merged_latam = latam_polys.merge(sub_df, on='uid', how='inner')
        if len(merged_latam):
            merged_latam.plot(ax=inset, column='err_rate', cmap=cmap, norm=norm,
                              edgecolor='white', linewidth=0.3,
                              legend=False, alpha=0.85)

    minx, miny, maxx, maxy = latam_polys.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.12
    inset.set_xlim(minx - pad_x, maxx + pad_x)
    inset.set_ylim(miny - pad_y, maxy + pad_y)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor('0.35')
        spine.set_linewidth(0.9)
    inset.set_title('Latin America', fontsize=6, pad=1.5)
    inset.patch.set_facecolor('white')
    inset.patch.set_alpha(0.92)


def render_seasonal_grid(
    agg_df: pd.DataFrame,
    polys: gpd.GeoDataFrame,
    scopes: List[str],
    output_path: Path,
    title_suffix: str = '',
    cmap_name: str = 'YlOrRd',
    vmin: float = 0.0,
    vmax: float = 1.0,
    missing_color: str = '#dddddd',
    dpi: int = 300,
    add_basemap: bool = True,
):
    seasons = ['DJFM', 'AMJJ', 'ASON']
    n_rows, n_cols = len(seasons), len(scopes)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = BoundaryNorm(np.linspace(vmin, vmax, 11), cmap.N, clip=True)

    polys_plot = polys.to_crs(epsg=3857) if (add_basemap and ctx) else polys

    if 'ADMIN0' in polys.columns:
        latam_polys = polys[polys['ADMIN0'].isin(LATAM_COUNTRIES)].copy()
    else:
        latam_polys = polys.iloc[0:0]

    # Main-panel extent: crop to Africa + Middle East so the small Latin America
    # zones don't squash the African extent. LatAm is restored in the inset.
    AFRICA_MIN_X_M = -2_226_000.0 if (add_basemap and ctx) else -20.0
    in_region = polys_plot[polys_plot.geometry.centroid.x >= AFRICA_MIN_X_M]
    if len(in_region):
        minx, miny, maxx, maxy = in_region.total_bounds
        pad_x = (maxx - minx) * 0.03
        pad_y = (maxy - miny) * 0.03
        main_xlim = (minx - pad_x, maxx + pad_x)
        main_ylim = (miny - pad_y, maxy + pad_y)
    else:
        main_xlim = main_ylim = None

    for i, season in enumerate(seasons):
        for j, scope in enumerate(scopes):
            ax = axes[i, j]
            sub = agg_df[(agg_df['season'] == season) & (agg_df['scope'] == scope)]
            title = f"{season} - {scope}{title_suffix}"

            if len(sub) == 0:
                polys_plot.plot(ax=ax, color=missing_color, edgecolor='white',
                                linewidth=0.3, alpha=0.6)
                ax.set_title(title, fontsize=10, weight='bold')
                ax.axis('off')
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                if main_xlim is not None:
                    ax.set_xlim(*main_xlim)
                    ax.set_ylim(*main_ylim)
                _add_latam_inset(ax, latam_polys, sub, cmap, norm, missing_color)
                continue

            merged = polys.merge(sub, on='uid', how='inner')
            if add_basemap and ctx:
                merged = merged.to_crs(epsg=3857)

            polys_plot.plot(ax=ax, color=missing_color, edgecolor='white',
                            linewidth=0.3, alpha=0.6)
            merged.plot(ax=ax, column='err_rate', cmap=cmap, norm=norm,
                        edgecolor='white', linewidth=0.3, legend=False, alpha=0.8)

            if main_xlim is not None:
                ax.set_xlim(*main_xlim)
                ax.set_ylim(*main_ylim)

            if add_basemap and ctx:
                try:
                    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                                    alpha=0.5, attribution=False)
                except Exception:
                    pass

            ax.set_title(title, fontsize=10, weight='bold')
            ax.axis('off')
            _add_latam_inset(ax, latam_polys, sub, cmap, norm, missing_color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.02, pad=0.02, aspect=40)
    cbar.set_label('Error Rate', fontsize=12, weight='bold')
    ticks = np.linspace(vmin, vmax, 11)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved map: {output_path}")


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def compute_season_table(df: pd.DataFrame, scopes: List[str]) -> pd.DataFrame:
    """Table 1: precision / recall / F1 / accuracy by scope x season."""
    seasons = ['DJFM', 'AMJJ', 'ASON']
    rows = []
    for scope in scopes:
        for season in seasons:
            sub = df[(df['scope'] == scope) & (df['season'] == season)]
            if len(sub) == 0:
                continue
            c = _confusion(sub['y_true'].values, sub['y_pred'].values)
            p, r, f1, acc = _prf(c['tp'], c['fp'], c['fn'], c['tn'])
            rows.append({
                'scope': scope,
                'season': season,
                'support': c['support'],
                'tp': c['tp'], 'fp': c['fp'], 'fn': c['fn'], 'tn': c['tn'],
                'precision': round(p, 4) if not np.isnan(p) else np.nan,
                'recall': round(r, 4) if not np.isnan(r) else np.nan,
                'f1': round(f1, 4) if not np.isnan(f1) else np.nan,
                'accuracy': round(acc, 4) if not np.isnan(acc) else np.nan,
            })
    return pd.DataFrame(rows)


def compute_region_table(
    df: pd.DataFrame,
    polys: gpd.GeoDataFrame,
    scopes: List[str],
) -> pd.DataFrame:
    """Table 2: precision / recall / F1 / accuracy by scope x region."""
    # Merge with polys for ADMIN0
    polys_sub = polys[['uid', 'ADMIN0']].copy()
    merged = df.merge(polys_sub, on='uid', how='left')
    merged = merged[merged['ADMIN0'].notna()].copy()
    merged['region'] = merged['ADMIN0'].map(REGION_MAP).fillna('Other')

    rows = []
    for scope in scopes:
        for region in sorted(merged['region'].unique()):
            sub = merged[(merged['scope'] == scope) & (merged['region'] == region)]
            if len(sub) == 0:
                continue
            c = _confusion(sub['y_true'].values, sub['y_pred'].values)
            p, r, f1, acc = _prf(c['tp'], c['fp'], c['fn'], c['tn'])
            rows.append({
                'scope': scope,
                'region': region,
                'support': c['support'],
                'tp': c['tp'], 'fp': c['fp'], 'fn': c['fn'], 'tn': c['tn'],
                'precision': round(p, 4) if not np.isnan(p) else np.nan,
                'recall': round(r, 4) if not np.isnan(r) else np.nan,
                'f1': round(f1, 4) if not np.isnan(f1) else np.nan,
                'accuracy': round(acc, 4) if not np.isnan(acc) else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE = Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code'
                r'\Step5_Geo_RF_trial\Food_Crisis_Cluster')
    DATA_DIR = BASE / 'main_ablation_results' / 'march2026_main_backup_month_ind_cont3'
    DEFAULT_POLYGONS = Path(
        r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data'
        r'\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
    )

    parser = argparse.ArgumentParser(
        description='Generate seasonal choropleth maps and performance tables',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--fs1', type=Path,
                        default=DATA_DIR / 'result_partition_k40_compare_GF_fs1' / 'predictions_monthly.csv')
    parser.add_argument('--fs2', type=Path,
                        default=DATA_DIR / 'result_partition_k40_compare_GF_fs2' / 'predictions_monthly.csv')
    parser.add_argument('--fs3', type=Path,
                        default=DATA_DIR / 'result_partition_k40_compare_GF_fs3' / 'predictions_monthly.csv')
    parser.add_argument('--polygons', type=Path, default=DEFAULT_POLYGONS)
    parser.add_argument('--y-pred-col', type=str, default='y_pred_partitioned',
                        help='Which prediction column to use')
    parser.add_argument('--out-dir', type=Path,
                        default=DATA_DIR / 'seasonal_performance_GF')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--no-basemap', dest='basemap', action='store_false', default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scopes = ['fs1', 'fs2', 'fs3']

    if args.basemap and ctx is None:
        print("WARNING: contextily not available, proceeding without basemap")
        args.basemap = False

    # ---- Load data ---------------------------------------------------------
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_all_scopes(args.fs1, args.fs2, args.fs3, y_pred_col=args.y_pred_col)
    polys = load_polys(args.polygons)

    # ---- Seasonal maps -----------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING SEASONAL MAPS")
    print("=" * 60)

    for filter_type, label, suffix in [
        ('all',        '',                  ''),
        ('crisis',     ' (Crisis only)',    '_crisis'),
        ('noncrisis',  ' (Non-crisis only)', '_noncrisis'),
    ]:
        if filter_type == 'all':
            subset = df
        elif filter_type == 'crisis':
            subset = df[df['y_true'] == 1]
        else:
            subset = df[df['y_true'] == 0]

        print(f"\n  Filter: {filter_type}  ->  {len(subset):,} rows")

        agg = aggregate_error_rate(subset, by=['scope', 'season'])

        out_png = args.out_dir / f'error_rate_seasonal_3x3{suffix}.png'
        render_seasonal_grid(
            agg, polys, scopes, out_png,
            title_suffix=label,
            dpi=args.dpi,
            add_basemap=args.basemap,
        )

        out_csv = args.out_dir / f'error_rate_seasonal{suffix}.csv'
        agg.to_csv(out_csv, index=False)
        print(f"  Saved csv: {out_csv}")

    # ---- Table 1: season performance ---------------------------------------
    print("\n" + "=" * 60)
    print("TABLE 1: MODEL PERFORMANCE BY SEASON")
    print("=" * 60)

    tbl1 = compute_season_table(df, scopes)
    tbl1_path = args.out_dir / 'table1_season_performance.csv'
    tbl1.to_csv(tbl1_path, index=False)
    print(f"Saved: {tbl1_path}")
    print(tbl1.to_string(index=False))

    # ---- Table 2: region performance ---------------------------------------
    print("\n" + "=" * 60)
    print("TABLE 2: MODEL PERFORMANCE BY REGION")
    print("=" * 60)

    tbl2 = compute_region_table(df, polys, scopes)
    tbl2_path = args.out_dir / 'table2_region_performance.csv'
    tbl2.to_csv(tbl2_path, index=False)
    print(f"Saved: {tbl2_path}")
    print(tbl2.to_string(index=False))

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"All outputs in: {args.out_dir}")


if __name__ == '__main__':
    main()
