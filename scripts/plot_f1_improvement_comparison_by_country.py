"""
Plot 2x2 panel of F1 improvement maps comparing GeoRF vs XGBoost at lag1 and lag2.
Aggregated by country level.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
from pathlib import Path
import numpy as np

# File paths
GEORF_LAG1 = r'result_partition_k40_nc4_compare/metrics_polygon_overall_lag1.csv'
GEORF_LAG2 = r'result_partition_k40_nc4_compare/metrics_polygon_overall_lag2.csv'
XGB_LAG1 = r'result_partition_k40_nc4_compare_XGB/metrics_polygon_overall_lag1.csv'
XGB_LAG2 = r'result_partition_k40_nc4_compare_XGB/metrics_polygon_overall_lag2.csv'
SHAPEFILE = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
OUTPUT = 'f1_improvement_comparison_by_country_2x2.png'

def load_and_aggregate_by_country(csv_path, shapefile_path):
    """Load metrics CSV, merge with shapefile, and aggregate by country."""
    # Load metrics
    metrics_df = pd.read_csv(csv_path)
    print(f"Loaded {len(metrics_df)} admin units from {csv_path}")

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Ensure admin_code is string in both dataframes
    metrics_df['FEWSNET_admin_code'] = metrics_df['FEWSNET_admin_code'].astype(str)
    gdf['admin_code'] = gdf['admin_code'].astype(str)

    # Merge to get country information for each admin unit
    merged = gdf.merge(
        metrics_df[['FEWSNET_admin_code', 'f1_diff', 'n']],
        left_on='admin_code',
        right_on='FEWSNET_admin_code',
        how='inner'
    )

    print(f"Merged: {len(merged)} admin units matched")

    # Calculate weighted mean F1 improvement by country (weighted by sample size 'n')
    country_stats = merged.groupby('ISO').apply(
        lambda x: pd.Series({
            'f1_diff_mean': np.average(x['f1_diff'], weights=x['n']) if x['n'].sum() > 0 else np.nan,
            'total_n': x['n'].sum(),
            'n_admin_units': len(x),
            'adm0_name': x['adm0_name'].iloc[0]
        })
    ).reset_index()

    print(f"\nAggregated to {len(country_stats)} countries")
    print(country_stats[['ISO', 'adm0_name', 'f1_diff_mean', 'n_admin_units', 'total_n']])

    # Dissolve geometries by country
    country_gdf = gdf.dissolve(by='ISO', as_index=False)

    # Merge country-level F1 improvements
    country_gdf = country_gdf.merge(
        country_stats[['ISO', 'f1_diff_mean', 'total_n', 'n_admin_units']],
        on='ISO',
        how='left'
    )

    return country_gdf

def plot_map(ax, gdf, title, vmin, vmax):
    """Plot a single F1 improvement map at country level."""
    # Create diverging colormap centered at 0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot with semi-transparent, thin boundaries
    gdf.plot(
        column='f1_diff_mean',
        ax=ax,
        cmap='RdBu',
        norm=norm,
        edgecolor=(0, 0, 0, 0.5),  # Black with 50% transparency
        linewidth=0.5,
        legend=False,
        missing_kwds={'color': 'lightgrey', 'edgecolor': (0, 0, 0, 0.5), 'linewidth': 0.5}
    )

    # Add basemap
    try:
        # Convert to Web Mercator for basemap
        gdf_web = gdf.to_crs(epsg=3857)
        ctx.add_basemap(
            ax,
            crs=gdf.crs.to_string(),
            source=ctx.providers.CartoDB.Positron,
            alpha=0.5
        )
    except Exception as e:
        print(f"Could not add basemap: {e}")

    # Add country labels
    for idx, row in gdf.iterrows():
        if pd.notna(row['f1_diff_mean']):
            # Get centroid for label placement
            centroid = row.geometry.centroid
            ax.annotate(
                text=row['ISO'],
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    return norm

def main():
    """Create 2x2 panel plot aggregated by country."""
    print("Loading and aggregating data by country...\n")

    # Load all datasets and aggregate by country
    print("="*60)
    print("GeoRF Lag1:")
    print("="*60)
    georf_lag1 = load_and_aggregate_by_country(GEORF_LAG1, SHAPEFILE)

    print("\n" + "="*60)
    print("GeoRF Lag2:")
    print("="*60)
    georf_lag2 = load_and_aggregate_by_country(GEORF_LAG2, SHAPEFILE)

    print("\n" + "="*60)
    print("XGBoost Lag1:")
    print("="*60)
    xgb_lag1 = load_and_aggregate_by_country(XGB_LAG1, SHAPEFILE)

    print("\n" + "="*60)
    print("XGBoost Lag2:")
    print("="*60)
    xgb_lag2 = load_and_aggregate_by_country(XGB_LAG2, SHAPEFILE)

    # Calculate global min/max for consistent color scale
    all_values = pd.concat([
        georf_lag1['f1_diff_mean'].dropna(),
        georf_lag2['f1_diff_mean'].dropna(),
        xgb_lag1['f1_diff_mean'].dropna(),
        xgb_lag2['f1_diff_mean'].dropna()
    ])
    vmin = all_values.min()
    vmax = all_values.max()

    # Make symmetric around 0 for better diverging colormap
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    print(f"\n\nF1 diff range: [{vmin:.3f}, {vmax:.3f}]")

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('F1 Score Improvement by Country: Partitioned vs Pooled Models',
                 fontsize=18, fontweight='bold', y=0.98)

    # Plot each panel
    print("\nCreating plots...")
    norm = plot_map(axes[0, 0], georf_lag1, 'GeoRF - Lag 1 (4-month forecasting)', vmin, vmax)
    plot_map(axes[0, 1], georf_lag2, 'GeoRF - Lag 2 (8-month forecasting)', vmin, vmax)
    plot_map(axes[1, 0], xgb_lag1, 'XGBoost - Lag 1 (4-month forecasting)', vmin, vmax)
    plot_map(axes[1, 1], xgb_lag2, 'XGBoost - Lag 2 (8-month forecasting)', vmin, vmax)

    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Mean F1 Score Improvement\n(Partitioned - Pooled)',
                   fontsize=12, fontweight='bold')

    # Add interpretation text
    fig.text(0.5, 0.02,
             'Blue = Partitioning improves F1 score | Red = Partitioning decreases F1 score | Gray = No data\n' +
             'Values are weighted means across administrative units within each country',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.04, 0.92, 0.97])

    # Save
    print(f"\nSaving to {OUTPUT}...")
    plt.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
    print("Done!")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS BY COUNTRY")
    print("="*60)

    for name, data in [('GeoRF Lag1', georf_lag1), ('GeoRF Lag2', georf_lag2),
                       ('XGB Lag1', xgb_lag1), ('XGB Lag2', xgb_lag2)]:
        f1_diff = data['f1_diff_mean'].dropna()
        print(f"\n{name}:")
        print(f"  Mean F1 improvement: {f1_diff.mean():.4f}")
        print(f"  Median F1 improvement: {f1_diff.median():.4f}")
        print(f"  Min F1 improvement: {f1_diff.min():.4f}")
        print(f"  Max F1 improvement: {f1_diff.max():.4f}")
        print(f"  % countries with improvement (>0): {(f1_diff > 0).sum() / len(f1_diff) * 100:.1f}%")
        print(f"  % countries with decline (<0): {(f1_diff < 0).sum() / len(f1_diff) * 100:.1f}%")
        print(f"  Countries included: {len(f1_diff)}")

    plt.show()

if __name__ == '__main__':
    main()
