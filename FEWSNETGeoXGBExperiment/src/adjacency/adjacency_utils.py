# @Author: Weilun Shi
# @Date: 2025-08-20
# @Email: swl007007@gmail.com
# @License: MIT License

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_polygon_adjacency_matrix(shapefile_path, polygon_id_column='FEWSNET_ID', cache_file='polygon_adjacency_matrix.pkl'):
    """
    Create a polygon adjacency matrix from a shapefile using true spatial boundaries.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile containing polygon boundaries
    polygon_id_column : str, default='FEWSNET_ID'
        Column name containing polygon identifiers
    cache_file : str, default='polygon_adjacency_matrix.pkl'
        Filename for caching the adjacency matrix
        
    Returns:
    --------
    adjacency_dict : dict
        Dictionary where keys are polygon indices and values are lists of adjacent polygon indices
    polygon_id_mapping : dict
        Dictionary mapping polygon IDs to array indices
    polygon_centroids : numpy.ndarray
        Array of polygon centroids (lat, lon)
    """
    
    logger.info(f"Loading shapefile from: {shapefile_path}")
    
    # Load shapefile
    try:
        gdf = gpd.read_file(shapefile_path)
        logger.info(f"Loaded {len(gdf)} polygons from shapefile")
    except Exception as e:
        logger.error(f"Failed to load shapefile: {e}")
        raise
    
    # Ensure we have the required ID column
    if polygon_id_column not in gdf.columns:
        logger.warning(f"Column '{polygon_id_column}' not found. Available columns: {list(gdf.columns)}")
        # Try common alternatives
        for alt_col in ['ADMIN_ID', 'ID', 'FID', 'OBJECTID']:
            if alt_col in gdf.columns:
                polygon_id_column = alt_col
                logger.info(f"Using alternative ID column: {polygon_id_column}")
                break
        else:
            # Use index as ID
            polygon_id_column = 'INDEX_ID'
            gdf[polygon_id_column] = gdf.index
            logger.info("Using row index as polygon ID")
    
    # Create mapping from polygon ID to array index
    polygon_ids = gdf[polygon_id_column].values
    polygon_id_mapping = {pid: idx for idx, pid in enumerate(polygon_ids)}
    n_polygons = len(polygon_ids)
    
    logger.info(f"Creating adjacency matrix for {n_polygons} polygons")
    
    # Calculate centroids
    centroids = gdf.geometry.centroid
    polygon_centroids = np.array([[point.y, point.x] for point in centroids])  # (lat, lon)
    
    # Initialize adjacency dictionary
    adjacency_dict = {i: [] for i in range(n_polygons)}
    
    # Calculate adjacency using spatial operations
    logger.info("Computing polygon adjacencies using spatial boundaries...")
    
    # Use spatial index for efficiency
    spatial_index = gdf.sindex
    
    for idx, polygon in gdf.iterrows():
        current_geometry = polygon.geometry
        
        # Get potential neighbors using spatial index
        possible_matches_index = list(spatial_index.intersection(current_geometry.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        
        # Check for actual adjacency (shared boundary)
        for neighbor_idx, neighbor_polygon in possible_matches.iterrows():
            if idx != neighbor_idx:  # Don't include self
                neighbor_geometry = neighbor_polygon.geometry
                
                # Check if polygons share a boundary (touches but not just at a point)
                if current_geometry.touches(neighbor_geometry):
                    # Additional check to ensure they share more than just a point
                    intersection = current_geometry.intersection(neighbor_geometry)
                    if hasattr(intersection, 'length') and intersection.length > 0:
                        # They share a boundary line, not just a point
                        current_array_idx = polygon_id_mapping[polygon[polygon_id_column]]
                        neighbor_array_idx = polygon_id_mapping[neighbor_polygon[polygon_id_column]]
                        
                        if neighbor_array_idx not in adjacency_dict[current_array_idx]:
                            adjacency_dict[current_array_idx].append(neighbor_array_idx)
    
    # Convert lists to numpy arrays for consistency
    for key in adjacency_dict:
        adjacency_dict[key] = np.array(adjacency_dict[key], dtype=int)
    
    # Log adjacency statistics
    neighbor_counts = [len(neighbors) for neighbors in adjacency_dict.values()]
    logger.info(f"Adjacency stats: min={min(neighbor_counts)}, max={max(neighbor_counts)}, "
               f"mean={np.mean(neighbor_counts):.1f}")
    
    # Check for isolated polygons
    isolated_polygons = [poly_id for poly_id, neighbors in adjacency_dict.items() if len(neighbors) == 0]
    if isolated_polygons:
        logger.warning(f"Found {len(isolated_polygons)} isolated polygons: {isolated_polygons}")
    
    return adjacency_dict, polygon_id_mapping, polygon_centroids


def load_or_create_adjacency_matrix(shapefile_path=None, polygon_id_column='FEWSNET_ID', 
                                   cache_dir=None, force_regenerate=False):
    """
    Load cached adjacency matrix or create a new one from shapefile.
    
    Parameters:
    -----------
    shapefile_path : str, optional
        Path to the shapefile. If None, uses default FEWS path
    polygon_id_column : str, default='FEWSNET_ID'
        Column name containing polygon identifiers
    cache_dir : str, optional
        Directory to store/load cache files. If None, uses current directory
    force_regenerate : bool, default=False
        If True, regenerate even if cache exists
        
    Returns:
    --------
    adjacency_dict : dict
        Dictionary where keys are polygon indices and values are lists of adjacent polygon indices
    polygon_id_mapping : dict
        Dictionary mapping polygon IDs to array indices
    polygon_centroids : numpy.ndarray
        Array of polygon centroids (lat, lon)
    """
    
    # Set default shapefile path if not provided
    if shapefile_path is None:
        shapefile_path = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = os.path.dirname(os.path.abspath(__file__))
    
    cache_file = os.path.join(cache_dir, 'polygon_adjacency_cache.pkl')
    
    # Check if cache exists and is valid
    if os.path.exists(cache_file) and not force_regenerate:
        try:
            logger.info(f"Loading cached adjacency matrix from: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data
            required_keys = ['adjacency_dict', 'polygon_id_mapping', 'polygon_centroids', 'shapefile_path', 'polygon_id_column']
            if all(key in cache_data for key in required_keys):
                # Check if shapefile and column match
                if (cache_data['shapefile_path'] == shapefile_path and 
                    cache_data['polygon_id_column'] == polygon_id_column):
                    
                    logger.info(f"Using cached adjacency matrix with {len(cache_data['adjacency_dict'])} polygons")
                    return (cache_data['adjacency_dict'], 
                           cache_data['polygon_id_mapping'], 
                           cache_data['polygon_centroids'])
                else:
                    logger.info("Cache shapefile/column mismatch, regenerating...")
            else:
                logger.info("Invalid cache format, regenerating...")
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, regenerating...")
    
    # Generate new adjacency matrix
    logger.info("Generating new adjacency matrix from shapefile...")
    
    # Check if shapefile exists
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    adjacency_dict, polygon_id_mapping, polygon_centroids = create_polygon_adjacency_matrix(
        shapefile_path, polygon_id_column
    )
    
    # Cache the results
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_data = {
            'adjacency_dict': adjacency_dict,
            'polygon_id_mapping': polygon_id_mapping,
            'polygon_centroids': polygon_centroids,
            'shapefile_path': shapefile_path,
            'polygon_id_column': polygon_id_column,
            'creation_time': pd.Timestamp.now().isoformat()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Cached adjacency matrix to: {cache_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
    
    return adjacency_dict, polygon_id_mapping, polygon_centroids


def adjacency_dict_to_neighbors_dict(adjacency_dict):
    """
    Convert adjacency dictionary to the format expected by existing polygon neighbor functions.
    
    Parameters:
    -----------
    adjacency_dict : dict
        Dictionary where keys are polygon indices and values are arrays of adjacent polygon indices
        
    Returns:
    --------
    neighbors_dict : dict
        Dictionary in the format expected by get_polygon_neighbors() and related functions
    """
    neighbors_dict = {}
    for poly_idx, neighbors in adjacency_dict.items():
        neighbors_dict[poly_idx] = neighbors.tolist() if isinstance(neighbors, np.ndarray) else list(neighbors)
    
    return neighbors_dict


def validate_adjacency_matrix(adjacency_dict, polygon_centroids=None, distance_threshold=None):
    """
    Validate adjacency matrix and optionally compare with distance-based neighbors.
    
    Parameters:
    -----------
    adjacency_dict : dict
        Adjacency dictionary to validate
    polygon_centroids : numpy.ndarray, optional
        Centroid coordinates for distance comparison
    distance_threshold : float, optional
        Distance threshold for comparison with distance-based neighbors
        
    Returns:
    --------
    validation_results : dict
        Dictionary containing validation statistics and any issues found
    """
    results = {
        'n_polygons': len(adjacency_dict),
        'neighbor_counts': [],
        'isolated_polygons': [],
        'symmetric_adjacency': True,
        'issues': []
    }
    
    # Basic statistics
    for poly_idx, neighbors in adjacency_dict.items():
        neighbor_count = len(neighbors)
        results['neighbor_counts'].append(neighbor_count)
        
        if neighbor_count == 0:
            results['isolated_polygons'].append(poly_idx)
    
    # Check symmetry (if A is neighbor of B, B should be neighbor of A)
    for poly_idx, neighbors in adjacency_dict.items():
        for neighbor_idx in neighbors:
            if neighbor_idx in adjacency_dict:
                if poly_idx not in adjacency_dict[neighbor_idx]:
                    results['symmetric_adjacency'] = False
                    results['issues'].append(f"Asymmetric adjacency: {poly_idx} -> {neighbor_idx}")
    
    # Calculate statistics
    if results['neighbor_counts']:
        results['min_neighbors'] = min(results['neighbor_counts'])
        results['max_neighbors'] = max(results['neighbor_counts'])
        results['mean_neighbors'] = np.mean(results['neighbor_counts'])
        results['std_neighbors'] = np.std(results['neighbor_counts'])
    
    # Compare with distance-based neighbors if centroids provided
    if polygon_centroids is not None and distance_threshold is not None:
        from src.partition.partition_opt import get_polygon_neighbors
        distance_neighbors = get_polygon_neighbors(polygon_centroids, distance_threshold)
        
        # Compare neighbor counts
        adjacency_neighbors_dict = adjacency_dict_to_neighbors_dict(adjacency_dict)
        
        comparison_stats = {
            'adjacency_total_connections': sum(len(neighbors) for neighbors in adjacency_neighbors_dict.values()),
            'distance_total_connections': sum(len(neighbors) for neighbors in distance_neighbors.values()),
            'polygon_differences': []
        }
        
        for poly_idx in adjacency_neighbors_dict:
            adj_neighbors = set(adjacency_neighbors_dict[poly_idx])
            dist_neighbors = set(distance_neighbors.get(poly_idx, []))
            
            if adj_neighbors != dist_neighbors:
                comparison_stats['polygon_differences'].append({
                    'polygon': poly_idx,
                    'adjacency_count': len(adj_neighbors),
                    'distance_count': len(dist_neighbors),
                    'only_in_adjacency': list(adj_neighbors - dist_neighbors),
                    'only_in_distance': list(dist_neighbors - adj_neighbors)
                })
        
        results['distance_comparison'] = comparison_stats
    
    return results