#!/usr/bin/env python3
"""
Visualize Clustering Results
Loads final cluster labels and plots them on a choropleth map using FEWS NET boundaries.
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

# Configuration
CLUSTER_FILE = Path('knn_sparsification_results/final_cluster_labels_k40_nc4.npz')
SHAPEFILE = Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp')
OUTPUT_FILE = Path('cluster_map_visualization.png')
DPI = 300

def load_cluster_labels(cluster_file):
    """Load cluster labels from npz file."""
    print(f"Loading cluster labels from {cluster_file}...")
    data = np.load(cluster_file)

    print(f"  Keys: {list(data.keys())}")
    print(f"  cluster_labels shape: {data['cluster_labels'].shape}")
    print(f"  n_clusters: {data['n_clusters']}")
    print(f"  Unique clusters: {np.unique(data['cluster_labels'])}")

    # Create dataframe
    cluster_df = pd.DataFrame({
        'FEWSNET_admin_code': data['admin_codes'],
        'cluster_id': data['cluster_labels']
    })

    # Add outlier flag if available
    if 'outlier_indices' in data.keys():
        outlier_indices = data['outlier_indices']
        cluster_df['is_outlier'] = cluster_df.index.isin(outlier_indices)
    else:
        cluster_df['is_outlier'] = False

    print(f"\n  Created cluster dataframe: {len(cluster_df)} rows")
    print(f"  Cluster distribution:")
    for cluster_id, count in cluster_df['cluster_id'].value_counts().sort_index().items():
        pct = count / len(cluster_df) * 100
        print(f"    Cluster {cluster_id}: {count:,} ({pct:.2f}%)")

    if cluster_df['is_outlier'].any():
        n_outliers = cluster_df['is_outlier'].sum()
        print(f"  Outliers: {n_outliers} ({n_outliers/len(cluster_df)*100:.2f}%)")

    return cluster_df, int(data['n_clusters'])

def load_shapefile(shapefile_path):
    """Load FEWS NET admin boundaries shapefile."""
    print(f"\nLoading shapefile from {shapefile_path.name}...")
    gdf = gpd.read_file(shapefile_path)

    print(f"  Loaded: {len(gdf)} features")
    print(f"  CRS: {gdf.crs}")

    # Find admin code column
    uid_variations = ['FEWSNET_admin_code', 'admin_code', 'adm_code', 'FNID']
    found_col = None
    for var in uid_variations:
        if var in gdf.columns:
            found_col = var
            break

    if found_col is None:
        print(f"  Available columns: {list(gdf.columns)}")
        raise ValueError(f"Could not find admin code column. Tried: {uid_variations}")

    # Standardize column name
    if found_col != 'FEWSNET_admin_code':
        gdf = gdf.rename(columns={found_col: 'FEWSNET_admin_code'})
        print(f"  Renamed '{found_col}' → 'FEWSNET_admin_code'")

    # Ensure valid geometries
    invalid = (~gdf.geometry.is_valid).sum()
    if invalid > 0:
        print(f"  Fixing {invalid} invalid geometries")
        gdf['geometry'] = gdf.geometry.buffer(0)

    return gdf

def plot_clusters(gdf, n_clusters, output_file, dpi=300):
    """Create choropleth map of clusters."""
    print(f"\nCreating cluster map...")

    # Setup figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Reproject to Web Mercator for basemap
    if ctx is not None:
        print("  Reprojecting to EPSG:3857 for basemap...")
        gdf_plot = gdf.to_crs(epsg=3857)
    else:
        gdf_plot = gdf.copy()

    # Define colormap - use distinct colors for clusters
    # Use tab10 for up to 10 clusters, or generate custom colors
    if n_clusters <= 10:
        colors = plt.cm.tab10(range(n_clusters))
    else:
        colors = plt.cm.tab20(range(n_clusters))

    cmap = ListedColormap(colors)

    # Plot clusters
    gdf_plot.plot(
        ax=ax,
        column='cluster_id',
        cmap=cmap,
        edgecolor='white',
        linewidth=0.2,
        legend=False,
        alpha=0.8,
        categorical=True
    )

    # Add basemap underneath
    if ctx is not None:
        print("  Adding grey basemap...")
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                alpha=0.4,
                attribution=False
            )
        except Exception as e:
            print(f"  WARNING: Could not add basemap: {e}")

    # Create legend
    legend_elements = []
    for i in range(n_clusters):
        count = (gdf['cluster_id'] == i).sum()
        pct = count / len(gdf) * 100
        label = f'Cluster {i} (n={count:,}, {pct:.1f}%)'
        legend_elements.append(
            mpatches.Patch(facecolor=colors[i], edgecolor='white', label=label)
        )

    # Add outlier info if available
    if 'is_outlier' in gdf.columns and gdf['is_outlier'].any():
        n_outliers = gdf['is_outlier'].sum()
        outlier_pct = n_outliers / len(gdf) * 100
        ax.text(
            0.02, 0.98,
            f'Outliers: {n_outliers} ({outlier_pct:.2f}%)',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=9,
        framealpha=0.95,
        title='Spatial Clusters',
        edgecolor='black'
    )

    ax.set_title(
        f'Food Crisis Spatial Clustering Results\n'
        f'{n_clusters} Clusters based on Historical Partition Similarity (KNN k=40)',
        fontsize=14,
        weight='bold',
        pad=20
    )
    ax.axis('off')

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    print("=" * 80)
    print("SPATIAL CLUSTERING VISUALIZATION")
    print("=" * 80)

    # Load cluster labels
    cluster_df, n_clusters = load_cluster_labels(CLUSTER_FILE)

    # Load shapefile
    gdf = load_shapefile(SHAPEFILE)

    # Merge
    print(f"\nMerging cluster labels with geometries...")
    original_len = len(gdf)
    merged = gdf.merge(cluster_df, on='FEWSNET_admin_code', how='inner')
    print(f"  Matched: {len(merged)}/{original_len} features ({len(merged)/original_len*100:.1f}%)")

    if len(merged) == 0:
        raise ValueError("No features matched! Check FEWSNET_admin_code alignment.")

    # Plot
    plot_clusters(merged, n_clusters, OUTPUT_FILE, DPI)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Clusters: {n_clusters}")
    print(f"Features plotted: {len(merged):,}")

if __name__ == '__main__':
    main()
