"""
Complete Clustering Pipeline with KNN and Outlier Handling

This script performs:
1. KNN graph construction
2. Eigengap analysis on main component
3. Spectral clustering
4. Geographic KNN for outlier assignment
5. Final cluster label generation
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, load_npz, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
K_NEIGHBORS = 40  # KNN parameter
OUTPUT_DIR = Path('knn_sparsification_results')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("COMPLETE CLUSTERING PIPELINE")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n>> Loading saved KNN graph and data...")

# Load KNN graph
sparse_affinity = load_npz(OUTPUT_DIR / f'knn_graph_k{K_NEIGHBORS}.npz')
N = sparse_affinity.shape[0]

# Load component labels
comp_data = np.load(OUTPUT_DIR / f'connected_components_k{K_NEIGHBORS}.npz')
labels = comp_data['labels']
n_components = int(comp_data['n_components'])

# Load coordinates
matrix_data = np.load('similarity_matrices/similarity_matrices.npz')
lat_lon_data = matrix_data['lat_lon_data']

print(f"  N = {N:,} admin codes")
print(f"  Components: {n_components}")
print(f"  KNN edges: {sparse_affinity.nnz:,}")

# =============================================================================
# STEP 1: SEPARATE MAIN COMPONENT FROM OUTLIERS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: SEPARATING MAIN COMPONENT FROM OUTLIERS")
print("=" * 80)

counts = np.bincount(labels)
main_component_id = np.argmax(counts)

main_indices = np.where(labels == main_component_id)[0]
outlier_indices = np.where(labels != main_component_id)[0]

print(f"\nMain component: {len(main_indices):,} nodes ({len(main_indices)/N*100:.2f}%)")
print(f"Outliers: {len(outlier_indices):,} nodes ({len(outlier_indices)/N*100:.2f}%)")

# Extract subgraph
print(f"\nExtracting main component subgraph...")
sub_affinity = sparse_affinity[main_indices][:, main_indices]

print(f"  Subgraph shape: {sub_affinity.shape}")
print(f"  Subgraph edges: {sub_affinity.nnz:,}")
print(f"  Subgraph density: {100 * sub_affinity.nnz / (len(main_indices) * (len(main_indices)-1)):.4f}%")

# =============================================================================
# STEP 2: EIGENGAP ANALYSIS ON MAIN COMPONENT
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: EIGENGAP ANALYSIS (MAIN COMPONENT)")
print("=" * 80)

print("\nComputing normalized Laplacian eigenvalues...")

degrees_main = np.array(sub_affinity.sum(axis=1)).flatten()
degrees_main[degrees_main == 0] = 1e-10
d_inv_sqrt_main = diags(1.0 / np.sqrt(degrees_main))
identity_main = diags(np.ones(len(main_indices)))
L_main = identity_main - d_inv_sqrt_main @ sub_affinity @ d_inv_sqrt_main

try:
    n_eigs_main = min(15, len(main_indices) - 2)
    vals_main, vecs_main = eigsh(L_main, k=n_eigs_main, which='SM', maxiter=5000)
    vals_main = np.sort(vals_main)

    print(f"✓ Computed {len(vals_main)} eigenvalues")
    print(f"\nFirst 10 eigenvalues:")
    for i, v in enumerate(vals_main[:min(10, len(vals_main))]):
        print(f"  λ_{i+1} = {v:.6f}")

    # Eigengap analysis
    eigengaps_main = np.diff(vals_main)
    max_gap_idx = np.argmax(eigengaps_main)
    suggested_k_main = max_gap_idx + 1

    print(f"\nLargest eigengap: position {max_gap_idx+1} (size: {eigengaps_main[max_gap_idx]:.6f})")
    print(f"RECOMMENDED CLUSTERS: {suggested_k_main}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(vals_main) + 1), vals_main, 'o-',
                linewidth=2, markersize=8, color='darkorange')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Eigenvalues (Main Component)')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(1, len(eigengaps_main) + 1), eigengaps_main,
               color='orange', edgecolor='black', alpha=0.7)
    axes[1].bar(max_gap_idx + 1, eigengaps_main[max_gap_idx],
               color='red', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Gap Index')
    axes[1].set_ylabel('Eigengap Size')
    axes[1].set_title('Eigengap Analysis')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('eigengap_main_component.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: eigengap_main_component.png")

    eigenvalue_success = True

except Exception as e:
    print(f"✗ Error: {e}")
    suggested_k_main = 3
    print(f"Using default: {suggested_k_main} clusters")
    eigenvalue_success = False

# =============================================================================
# STEP 3: SPECTRAL CLUSTERING
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: SPECTRAL CLUSTERING")
print("=" * 80)

N_CLUSTERS = suggested_k_main

print(f"\nClustering main component...")
print(f"  Target clusters: {N_CLUSTERS}")
print(f"  Nodes: {len(main_indices):,}")

try:
    sc = SpectralClustering(
        n_clusters=N_CLUSTERS,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
        n_jobs=-1
    )

    main_labels = sc.fit_predict(sub_affinity)

    print(f"✓ Clustering complete!")

    unique_main, counts_main = np.unique(main_labels, return_counts=True)
    print(f"\nCluster distribution (main component):")
    for cluster_id, count in zip(unique_main, counts_main):
        pct = count / len(main_labels) * 100
        print(f"  Cluster {cluster_id}: {count:,} nodes ({pct:.2f}%)")

    clustering_success = True

except Exception as e:
    print(f"✗ Error: {e}")
    main_labels = None
    clustering_success = False

# =============================================================================
# STEP 4: ASSIGN OUTLIERS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: OUTLIER ASSIGNMENT")
print("=" * 80)

if clustering_success and len(outlier_indices) > 0:
    print(f"\nAssigning {len(outlier_indices)} outliers using geographic KNN...")

    X_train = lat_lon_data[main_indices]
    y_train = main_labels
    X_outliers = lat_lon_data[outlier_indices]

    knn_classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_classifier.fit(X_train, y_train)
    outlier_labels = knn_classifier.predict(X_outliers)

    print(f"✓ Outliers assigned!")

    unique_outlier, counts_outlier = np.unique(outlier_labels, return_counts=True)
    print(f"\nOutlier distribution:")
    for cluster_id, count in zip(unique_outlier, counts_outlier):
        print(f"  → Cluster {cluster_id}: {count} outliers")

elif clustering_success:
    print(f"\nNo outliers to assign!")
    outlier_labels = None
else:
    print(f"\nSkipped (clustering failed)")
    outlier_labels = None

# =============================================================================
# STEP 5: MERGE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: FINAL CLUSTER LABELS")
print("=" * 80)

if clustering_success:
    print("\nCreating final labels...")

    final_labels = np.zeros(N, dtype=int)
    final_labels[main_indices] = main_labels

    if outlier_labels is not None and len(outlier_indices) > 0:
        final_labels[outlier_indices] = outlier_labels

    print(f"✓ Final labels created!")

    unique_final, counts_final = np.unique(final_labels, return_counts=True)
    print(f"\nFINAL DISTRIBUTION:")
    print(f"{'Cluster':<10} {'Count':<10} {'Percentage'}")
    print("-" * 35)
    for cluster_id, count in zip(unique_final, counts_final):
        pct = count / N * 100
        print(f"{cluster_id:<10} {count:<10,} {pct:>6.2f}%")

    print(f"\nTotal: {N:,} nodes across {len(unique_final)} clusters")

else:
    print("\nFailed to generate labels")
    final_labels = None

# =============================================================================
# STEP 6: SAVE RESULTS
# =============================================================================
if final_labels is not None:
    print("\n" + "=" * 80)
    print("STEP 6: SAVING RESULTS")
    print("=" * 80)

    # Save cluster labels
    np.savez_compressed(
        OUTPUT_DIR / f'final_cluster_labels_k{K_NEIGHBORS}_nc{N_CLUSTERS}.npz',
        cluster_labels=final_labels,
        admin_codes=np.arange(N),
        n_clusters=N_CLUSTERS,
        k_neighbors=K_NEIGHBORS,
        main_indices=main_indices,
        outlier_indices=outlier_indices
    )
    print(f"\n✓ Saved: final_cluster_labels_k{K_NEIGHBORS}_nc{N_CLUSTERS}.npz")

    # Save mapping table
    cluster_mapping = pd.DataFrame({
        'FEWSNET_admin_code': np.arange(N),
        'cluster_id': final_labels,
        'latitude': lat_lon_data[:, 0],
        'longitude': lat_lon_data[:, 1],
        'is_outlier': np.isin(np.arange(N), outlier_indices)
    })

    cluster_mapping.to_csv(
        OUTPUT_DIR / f'cluster_mapping_k{K_NEIGHBORS}_nc{N_CLUSTERS}.csv',
        index=False
    )
    print(f"✓ Saved: cluster_mapping_k{K_NEIGHBORS}_nc{N_CLUSTERS}.csv")

    # Update report
    report_file = OUTPUT_DIR / f'knn_analysis_report_k{K_NEIGHBORS}.json'
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
    else:
        report = {}

    report['spectral_clustering'] = {
        'n_clusters': int(N_CLUSTERS),
        'main_component_size': int(len(main_indices)),
        'n_outliers': int(len(outlier_indices)),
        'cluster_sizes': {int(k): int(v) for k, v in zip(unique_final, counts_final)},
        'method': 'spectral_clustering_with_outlier_knn'
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Updated: {report_file.name}")

    # Visualize
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter = axes[0].scatter(lat_lon_data[:, 1], lat_lon_data[:, 0],
                             c=final_labels, cmap='tab10', s=5, alpha=0.6)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title(f'Geographic Clusters (N={len(unique_final)})')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster ID')

    axes[1].bar(unique_final, counts_final, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Admin Codes')
    axes[1].set_title('Cluster Sizes')
    axes[1].grid(True, alpha=0.3, axis='y')

    for cluster_id, count in zip(unique_final, counts_final):
        axes[1].text(cluster_id, count, f'{count:,}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('final_clustering_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: final_clustering_results.png")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
