#!/usr/bin/env python3
"""
Partition Contiguity Refinement Script

Refines spatial partition assignments using adjacency-aware majority voting
to reduce mosaic patterns and enclaves. Uses existing polygon adjacency cache
and partition optimization logic from src/partition/partition_opt.py.

Usage:
    python scripts/refine_partitions_contiguity.py --adj ./src/adjacency/polygon_adjacency_cache.pkl --in cluster_mapping_k40_nc2_m6.csv --out ./result_partition_k40_nc4_compare/refined --iters 2

Author: Claude Code (ML Platform Engineer)
Date: 2026-02-01
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.partition.partition_opt import swap_partition_polygon
from src.adjacency.adjacency_utils import adjacency_dict_to_neighbors_dict


def load_adjacency_cache(cache_path):
    """Load polygon adjacency cache and extract required data."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Adjacency cache not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    required_keys = ['adjacency_dict', 'polygon_id_mapping', 'polygon_centroids']
    for key in required_keys:
        if key not in cache_data:
            raise ValueError(f"Invalid cache: missing key '{key}'")

    return (cache_data['adjacency_dict'],
            cache_data['polygon_id_mapping'],
            cache_data['polygon_centroids'])


def validate_partition_alignment(partition_df, polygon_id_mapping):
    """Verify partition UIDs align with adjacency cache."""
    partition_uids = set(partition_df['FEWSNET_admin_code'].unique())
    cache_uids = set(polygon_id_mapping.keys())

    missing_in_cache = partition_uids - cache_uids
    missing_in_partition = cache_uids - partition_uids

    if missing_in_cache:
        raise ValueError(f"UIDs in partition missing from adjacency cache: {sorted(list(missing_in_cache))[:10]}...")

    # Missing in partition is OK (cache may have extra polygons)
    return True


def count_connected_components(partition_assignments, polygon_neighbors):
    """Count number of spatially connected components per partition using BFS."""
    from collections import deque

    partition_components = {}

    # Only process valid partitions (>= 0)
    valid_partitions = np.unique(partition_assignments[partition_assignments >= 0])

    for partition_id in valid_partitions:
        # Get polygon indices for this partition
        poly_indices = np.where(partition_assignments == partition_id)[0]

        if len(poly_indices) == 0:
            partition_components[int(partition_id)] = 0
            continue

        # BFS to find connected components
        visited = set()
        n_components = 0

        for start_poly in poly_indices:
            if start_poly in visited:
                continue

            # Start new component
            n_components += 1
            queue = deque([start_poly])
            visited.add(start_poly)

            while queue:
                current_poly = queue.popleft()

                if current_poly not in polygon_neighbors:
                    continue

                # Visit all neighbors in same partition
                for neighbor_poly in polygon_neighbors[current_poly]:
                    if (neighbor_poly in poly_indices and
                        neighbor_poly not in visited):
                        visited.add(neighbor_poly)
                        queue.append(neighbor_poly)

        partition_components[int(partition_id)] = n_components

    return partition_components


def refine_partition_csv(input_csv, output_csv, adjacency_dict, polygon_id_mapping,
                         polygon_centroids, refine_iters=2, log_file=None):
    """
    Refine partition assignments using adjacency-aware majority voting.

    Parameters:
    -----------
    input_csv : str
        Path to input partition CSV (FEWSNET_admin_code, cluster_id, lat, lon, is_outlier)
    output_csv : str
        Path to output refined CSV
    adjacency_dict : dict
        Polygon adjacency dictionary from cache
    polygon_id_mapping : dict
        Mapping from FEWSNET_admin_code to polygon index
    polygon_centroids : np.ndarray
        Polygon centroid coordinates
    refine_iters : int
        Number of refinement iterations
    log_file : str, optional
        Path to log file

    Returns:
    --------
    n_reassigned : int
        Total number of polygons reassigned across all iterations
    """

    # Load partition CSV
    partition_df = pd.read_csv(input_csv)

    # Validate alignment
    validate_partition_alignment(partition_df, polygon_id_mapping)

    # Convert adjacency dict to neighbors dict format
    polygon_neighbors = adjacency_dict_to_neighbors_dict(adjacency_dict)

    # Build reverse mapping: polygon_index -> FEWSNET_admin_code
    index_to_uid = {idx: uid for uid, idx in polygon_id_mapping.items()}

    # Initialize partition assignments array indexed by polygon index
    n_polygons = len(polygon_id_mapping)
    partition_assignments = np.full(n_polygons, -1, dtype=int)

    # Map FEWSNET_admin_code to cluster_id
    uid_to_cluster = dict(zip(partition_df['FEWSNET_admin_code'],
                              partition_df['cluster_id']))

    # Fill partition_assignments using polygon indices
    for uid, cluster_id in uid_to_cluster.items():
        if uid in polygon_id_mapping:
            poly_idx = polygon_id_mapping[uid]
            partition_assignments[poly_idx] = cluster_id

    # Validate partition assignments (should be non-negative)
    valid_mask = partition_assignments >= 0
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print(f"WARNING: {n_invalid} polygons have invalid partition assignments (will be skipped)")

    # Count initial components (only for valid partitions)
    initial_components = count_connected_components(partition_assignments, polygon_neighbors)

    # Apply refinement iterations
    total_reassigned = 0

    for iter_num in range(refine_iters):
        partition_before = partition_assignments.copy()

        # Apply one iteration of majority voting with error handling
        try:
            partition_assignments = swap_partition_polygon(
                partition_assignments,
                polygon_neighbors,
                centroids=polygon_centroids
            )
        except Exception as e:
            error_msg = f"WARNING: Refinement error in iteration {iter_num+1}: {e}"
            print(error_msg)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"{error_msg}\n")
            # Use partition_before to continue
            partition_assignments = partition_before
            break

        # Count reassignments
        n_changed = np.sum(partition_before != partition_assignments)
        total_reassigned += n_changed

        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Iteration {iter_num+1}/{refine_iters}: {n_changed} polygons reassigned\n")

    # Count final components
    final_components = count_connected_components(partition_assignments, polygon_neighbors)

    # Update partition_df with refined assignments
    partition_df['cluster_id_original'] = partition_df['cluster_id']

    for idx, row in partition_df.iterrows():
        uid = row['FEWSNET_admin_code']
        if uid in polygon_id_mapping:
            poly_idx = polygon_id_mapping[uid]
            partition_df.at[idx, 'cluster_id'] = partition_assignments[poly_idx]

    # Save refined CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    partition_df.to_csv(output_csv, index=False)

    # Write summary to log
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"\nSummary:\n")
            f.write(f"  Total reassigned: {total_reassigned} polygons\n")
            f.write(f"  Row count: {len(partition_df)} (unchanged)\n")
            f.write(f"  Unique partitions: {partition_df['cluster_id'].nunique()}\n")
            f.write(f"  Connected components (before): {initial_components}\n")
            f.write(f"  Connected components (after):  {final_components}\n")

    return total_reassigned


def main():
    parser = argparse.ArgumentParser(description='Refine partition contiguity using adjacency matrix')
    parser.add_argument('--adj', required=True, help='Path to adjacency cache pickle')
    parser.add_argument('--in', dest='input_csv', required=True, help='Input partition CSV')
    parser.add_argument('--out', required=True, help='Output directory for refined CSV')
    parser.add_argument('--iters', type=int, default=2, help='Number of refinement iterations (default: 2)')

    args = parser.parse_args()

    # Load adjacency cache
    print(f"Loading adjacency cache from: {args.adj}")
    adjacency_dict, polygon_id_mapping, polygon_centroids = load_adjacency_cache(args.adj)
    print(f"  Loaded {len(adjacency_dict)} polygons")

    # Construct output paths
    input_name = Path(args.input_csv).stem
    output_csv = os.path.join(args.out, f"{input_name}_refined_contig{args.iters}.csv")
    log_file = os.path.join(args.out, f"refine_summary_{input_name}.txt")

    # Ensure output dir exists
    os.makedirs(args.out, exist_ok=True)

    # Initialize log
    with open(log_file, 'w') as f:
        f.write(f"Partition Contiguity Refinement Log\n")
        f.write(f"====================================\n")
        f.write(f"Input:  {args.input_csv}\n")
        f.write(f"Output: {output_csv}\n")
        f.write(f"Iterations: {args.iters}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Seed: 42 (deterministic)\n\n")

    # Set random seed for determinism
    np.random.seed(42)

    # Refine partition
    print(f"Refining partition: {args.input_csv}")
    n_reassigned = refine_partition_csv(
        args.input_csv,
        output_csv,
        adjacency_dict,
        polygon_id_mapping,
        polygon_centroids,
        refine_iters=args.iters,
        log_file=log_file
    )

    print(f"  Reassigned: {n_reassigned} polygons")
    print(f"  Output: {output_csv}")
    print(f"  Log: {log_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
