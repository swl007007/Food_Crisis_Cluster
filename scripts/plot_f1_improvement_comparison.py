"""
Plot 2x2 panel of F1 improvement maps comparing GeoRF vs XGBoost at lag1 and lag2.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
from pathlib import Path

# File paths
GEORF_LAG1 = r'result_partition_k40_nc4_compare/metrics_polygon_overall_lag1.csv'
GEORF_LAG2 = r'result_partition_k40_nc4_compare/metrics_polygon_overall_lag2.csv'
XGB_LAG1 = r'result_partition_k40_nc4_compare_XGB/metrics_polygon_overall_lag1.csv'
XGB_LAG2 = r'result_partition_k40_nc4_compare_XGB/metrics_polygon_overall_lag2.csv'
SHAPEFILE = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
OUTPUT = 'f1_improvement_comparison_2x2.png'

def load_and_merge_data(csv_path, shapefile_path):
    """Load metrics CSV and merge with shapefile."""
    # Load metrics
    metrics_df = pd.read_csv(csv_path)

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Ensure FEWSNET_admin_code is string in both dataframes
    metrics_df['FEWSNET_admin_code'] = metrics_df['FEWSNET_admin_code'].astype(str)
    gdf['admin_code'] = gdf['admin_code'].astype(str)

    # Merge on admin code
    merged = gdf.merge(
        metrics_df[['FEWSNET_admin_code', 'f1_diff']],
        left_on='admin_code',
        right_on='FEWSNET_admin_code',
        how='left'
    )

    return merged

def plot_map(ax, gdf, title, vmin, vmax):
    """Plot a single F1 improvement map."""
    # Create diverging colormap centered at 0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot with semi-transparent, thin boundaries
    gdf.plot(
        column='f1_diff',
        ax=ax,
        cmap='RdBu',
        norm=norm,
        edgecolor=(0, 0, 0, 0.5),  # Black with 50% transparency
        linewidth=0.1,
        legend=False,
        missing_kwds={'color': 'lightgrey', 'edgecolor': (0, 0, 0, 0.5), 'linewidth': 0.1}
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

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    return norm

def main():
    """Create 2x2 panel plot."""
    print("Loading data...")

    # Load all datasets
    georf_lag1 = load_and_merge_data(GEORF_LAG1, SHAPEFILE)
    georf_lag2 = load_and_merge_data(GEORF_LAG2, SHAPEFILE)
    xgb_lag1 = load_and_merge_data(XGB_LAG1, SHAPEFILE)
    xgb_lag2 = load_and_merge_data(XGB_LAG2, SHAPEFILE)

    # Calculate global min/max for consistent color scale
    all_values = pd.concat([
        georf_lag1['f1_diff'].dropna(),
        georf_lag2['f1_diff'].dropna(),
        xgb_lag1['f1_diff'].dropna(),
        xgb_lag2['f1_diff'].dropna()
    ])
    vmin = all_values.min()
    vmax = all_values.max()

    # Make symmetric around 0 for better diverging colormap
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    print(f"F1 diff range: [{vmin:.3f}, {vmax:.3f}]")

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('F1 Score Improvement: Partitioned vs Pooled Models',
                 fontsize=18, fontweight='bold', y=0.98)

    # Plot each panel
    print("Creating plots...")
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
    cbar.set_label('F1 Score Improvement\n(Partitioned - Pooled)',
                   fontsize=12, fontweight='bold')

    # Add interpretation text
    fig.text(0.5, 0.02,
             'Blue = Partitioning improves F1 score | Red = Partitioning decreases F1 score | Gray = No data',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.97])

    # Save
    print(f"Saving to {OUTPUT}...")
    plt.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
    print("Done!")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for name, data in [('GeoRF Lag1', georf_lag1), ('GeoRF Lag2', georf_lag2),
                       ('XGB Lag1', xgb_lag1), ('XGB Lag2', xgb_lag2)]:
        f1_diff = data['f1_diff'].dropna()
        print(f"\n{name}:")
        print(f"  Mean F1 improvement: {f1_diff.mean():.4f}")
        print(f"  Median F1 improvement: {f1_diff.median():.4f}")
        print(f"  Min F1 improvement: {f1_diff.min():.4f}")
        print(f"  Max F1 improvement: {f1_diff.max():.4f}")
        print(f"  % areas with improvement (>0): {(f1_diff > 0).sum() / len(f1_diff) * 100:.1f}%")
        print(f"  % areas with decline (<0): {(f1_diff < 0).sum() / len(f1_diff) * 100:.1f}%")

    plt.show()

if __name__ == '__main__':
    main()
