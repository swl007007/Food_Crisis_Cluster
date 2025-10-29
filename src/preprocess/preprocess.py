
#!/usr/bin/env python3
"""
Replicated and debugged version of main_model_GF_main.ipynb

This script replicates the full functionality of the notebook for food crisis prediction
using GeoRF with polygon-based contiguity support.

Key features:
1. Data preprocessing with polars and pandas
"""

import numpy as np
import pandas as pd
import polars as pl
import os
import sys
import warnings

# Add parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')
# GeoRF imports
from src.customize.customize import *
from src.helper.helper import get_spatial_range
from src.tests.class_wise_metrics import *
from config_visual import *
from src.utils.force_clean import *
from src.utils.lag_schedules import resolve_lag_schedule

ACTIVE_LAGS = resolve_lag_schedule(LAGS_MONTHS, context="config_visual.LAGS_MONTHS")


# Import adjacency matrix utilities
if USE_ADJACENCY_MATRIX:
    from src.adjacency.adjacency_utils import load_or_create_adjacency_matrix


def comp_impute(X, strategy="max_plus", multiplier=100.0):
    """
    Custom imputation function that handles infinite values and missing data.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data array
    strategy : str
        Imputation strategy
    multiplier : float
        Multiplier for out-of-range values
        
    Returns:
    --------
    X : numpy.ndarray
        Imputed data array
    """
    # Handle infinite values
    inf_count = 0
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        
        try:
            col_data_float = col_data.astype(float)
            if np.isinf(col_data_float).any():
                inf_count += 1
                col_data_float[np.isinf(col_data_float)] = np.nan
                X[:, col_idx] = col_data_float
        except Exception:
            continue
    
    if inf_count > 0:
        print(f"Found and replaced infinite values in {inf_count} columns")
    
    # Apply imputation with reduced verbosity
    X_imputed, imputer = impute_missing_values(X, strategy=strategy, multiplier=multiplier, verbose=False)
    
    # Get summary info
    if hasattr(imputer, 'column_stats_'):
        imputed_cols = sum(1 for stats in imputer.column_stats_.values() if stats['has_missing'])
        print(f"Imputed missing values in {imputed_cols} columns")
    
    return X_imputed

def handle_infinite_values(X):
    """
    Handle infinite values by replacing them with NaN.
    XGBoost can handle NaN values natively without extreme value imputation.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data array
        
    Returns:
    --------
    X : numpy.ndarray
        Data array with infinite values replaced by NaN
    """
    # Handle infinite values only - XGBoost handles NaN natively
    inf_count = 0
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        
        try:
            col_data_float = col_data.astype(float)
            if np.isinf(col_data_float).any():
                inf_count += 1
                col_data_float[np.isinf(col_data_float)] = np.nan
                X[:, col_idx] = col_data_float
        except Exception:
            continue
    
    if inf_count > 0:
        print(f"Found and replaced infinite values with NaN in {inf_count} columns")
        print("XGBoost will handle NaN values natively during training")
    
    # Check for missing value statistics
    total_missing = np.isnan(X.astype(float)).sum()
    if total_missing > 0:
        print(f"Total missing values (including NaN): {total_missing} ({100*total_missing/X.size:.2f}% of data)")
        print("XGBoost will use native missing value handling during training")
    
    return X

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the FEWSNET data.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
        
    Returns:
    --------
    df : pandas.DataFrame
        Preprocessed dataframe
    """
    print("Loading data...")
    
    # Load data with polars
    data = pl.read_csv(data_path)
    
    # Drop unnecessary columns
    cols_to_drop = [
        "ISO3", "fews_ipc_adjusted", "fews_proj_med_adjusted",
        "fews_proj_near", "fews_proj_near_ha", "fews_proj_med",
        "fews_proj_med_ha", "ADMIN0", "ADMIN1", "ADMIN2", "ADMIN3"
    ]
    data = data.drop(cols_to_drop)
    
    # Optionally reduce variables
    less_var = True
    if less_var:
        var_extra = [
            'Evap_tavg_mean', 'Qsb_tavg_mean', 'RadT_tavg_mean', 
            'SnowCover_inst_mean', 'SnowDepth_inst_mean', 'Snowf_tavg_mean',
            'SoilMoi00_10cm_tavg_mean', 'SoilMoi10_40cm_tavg_mean', 
            'SoilMoi100_200cm_tavg_mean', 'SoilMoi40_100cm_tavg_mean',
            'LWdown_f_tavg_mean', 'SoilTemp00_10cm_tavg_mean', 
            'SoilTemp10_40cm_tavg_mean', 'SoilTemp100_200cm_tavg_mean',
            'SoilTemp40_100cm_tavg_mean', 'SWdown_f_tavg_mean', 
            'SWE_inst_mean', 'Swnet_tavg_mean', 'Wind_f_tavg_mean',
            'Lwnet_tavg_mean', 'Psurf_f_tavg_mean', 'Qair_f_tavg_mean',
            'Qg_tavg_mean', 'Qh_tavg_mean', 'Qle_tavg_mean', 'Qs_tavg_mean'
        ]
        data = data.drop([col for col in data.columns if col.startswith(tuple(var_extra))])
    
    # Filter for non-null crisis data
    data = data.filter(pl.col("fews_ipc_crisis").is_not_null())
    
    # Encode ISO
    data = data.with_columns([
        pl.col("ISO").cast(pl.Categorical).to_physical().alias("ISO_encoded")
    ])
    
    # Drop unit name and ISO
    data = data.drop(["unit_name", "ISO"])
    
    # Process AEZ columns
    for col in data.columns:
        if col.startswith("AEZ_"):
            data = data.with_columns(
                pl.when(pl.col(col) == "True")
                .then(1)
                .when(pl.col(col) == "False")
                .then(0)
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    # Convert to pandas
    df = data.to_pandas()
    
    # Process dates
    df['date'] = pd.to_datetime(df['date'])
    df['years'] = df['date'].dt.year
    
    # Create lag features matching active schedule
    for lag in ACTIVE_LAGS:
        df[f'fews_ipc_crisis_lag_{lag}'] = (
            df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(lag)
        )
    for lag in ACTIVE_LAGS:
        df[f'fews_ipc_lag_{lag}'] = (
            df.groupby('FEWSNET_admin_code')['fews_ipc'].shift(lag)
        )
    
    # drop fews_ipc
    df = df.drop(columns=['fews_ipc'])
    
    # Create year and month dummies
    for year in df['years'].unique():
        df[f'year_{year}'] = (df['years'] == year).astype(int)
    for month in df['date'].dt.month.unique():
        df[f'month_{month}'] = (df['date'].dt.month == month).astype(int)
    
    # Create AEZ groups
    aez_columns = [col for col in df.columns if col.startswith('AEZ_')]
    df['AEZ_group'] = df.groupby(aez_columns).ngroup()
    df['AEZ_country_group'] = df.groupby(['AEZ_group', 'ISO_encoded']).ngroup()
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def setup_spatial_groups(df, assignment='polygons'):
    """
    Setup spatial grouping based on assignment method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    assignment : str
        Grouping method ('polygons', 'grid', 'country', etc.)
        
    Returns:
    --------
    X_group : numpy.ndarray
        Group assignments
    X_loc : numpy.ndarray
        Location coordinates
    contiguity_info : dict or None
        Contiguity information for polygon-based grouping
    """
    print(f"Setting up spatial groups using: {assignment}")
    
    # Get location coordinates
    X_loc = df[['lat', 'lon']].values
    
    if assignment == 'polygons':
        # CRITICAL FIX: Use unique admin units, not all temporal records
        # The bug was that X_polygon_ids contained all 70,228 temporal records
        # when it should only contain unique admin units for spatial grouping
        
        # Get unique polygons and their centroids (deduplicated)
        polygon_data = df[['FEWSNET_admin_code', 'lat', 'lon']].drop_duplicates()
        polygon_centroids = polygon_data[['lat', 'lon']].values
        
        # Get unique admin codes
        unique_polygons = polygon_data['FEWSNET_admin_code'].unique()
        
        print(f"SPATIAL GROUPING FIX:")
        print(f"  - Total records in df: {len(df):,}")
        print(f"  - Unique admin units: {len(unique_polygons):,}")
        print(f"  - Records per admin unit: {len(df)/len(unique_polygons):.1f}")
        
        # Create X_polygon_ids array that maps each temporal record to its admin unit ID
        # This ensures partitioning works on temporal records but groups by admin units
        X_polygon_ids = df['FEWSNET_admin_code'].values
        
        # Create mapping from FEWSNET_admin_code to polygon index
        admin_to_polygon_idx = {admin_code: idx for idx, admin_code in enumerate(unique_polygons)}
        
        # FIXED: Create correct polygon_group_mapping
        # Map polygon_index -> [admin_code] (each polygon is its own group using admin code as group ID)
        polygon_group_mapping = {i: [unique_polygons[i]] for i in range(len(unique_polygons))}
        
        # Convert admin codes to polygon indices for PolygonGroupGenerator
        polygon_indices = np.array([admin_to_polygon_idx[admin_code] for admin_code in X_polygon_ids])
        
        # Load adjacency matrix if enabled
        adjacency_dict = None
        if USE_ADJACENCY_MATRIX:
            try:
                print("Loading adjacency matrix for polygon-based contiguity...")
                adj_dict_raw, polygon_id_mapping, adj_centroids = load_or_create_adjacency_matrix(
                    shapefile_path=ADJACENCY_SHAPEFILE_PATH,
                    polygon_id_column=ADJACENCY_POLYGON_ID_COLUMN,
                    cache_dir=ADJACENCY_CACHE_DIR,
                    force_regenerate=ADJACENCY_FORCE_REGENERATE
                )
                
                # Create mapping from admin_code to adjacency matrix index
                # The adjacency matrix keys are indices 0,1,2... corresponding to shapefile admin_codes
                admin_code_to_adj_idx = {}
                adj_idx_to_admin_code = {}
                
                for adj_idx, admin_code in polygon_id_mapping.items():
                    admin_code_to_adj_idx[admin_code] = adj_idx
                    adj_idx_to_admin_code[adj_idx] = admin_code
                
                # Map adjacency dictionary to use polygon indices (matching our polygon_indices array)
                adjacency_dict = {}
                for polygon_idx in range(len(unique_polygons)):
                    admin_code = unique_polygons[polygon_idx]
                    if admin_code in admin_code_to_adj_idx:
                        adj_idx = admin_code_to_adj_idx[admin_code]
                        if adj_idx in adj_dict_raw:
                            # Map neighbor adjacency indices back to polygon indices
                            neighbor_admin_codes = [adj_idx_to_admin_code.get(neighbor_adj_idx) for neighbor_adj_idx in adj_dict_raw[adj_idx]]
                            neighbor_polygon_indices = []
                            for neighbor_admin_code in neighbor_admin_codes:
                                if neighbor_admin_code in admin_to_polygon_idx:
                                    neighbor_polygon_indices.append(admin_to_polygon_idx[neighbor_admin_code])
                            adjacency_dict[polygon_idx] = np.array(neighbor_polygon_indices)
                        else:
                            adjacency_dict[polygon_idx] = np.array([])
                    else:
                        adjacency_dict[polygon_idx] = np.array([])
                
                print(f"Adjacency matrix loaded: {len(adj_dict_raw)} polygons in shapefile, {len(adjacency_dict)} mapped to current data")
                print(f"Sample adjacency connections: polygon 0 has {len(adjacency_dict.get(0, []))} neighbors")
                
            except Exception as e:
                print(f"Warning: Failed to load adjacency matrix: {e}")
                print("Falling back to distance-based neighbor calculation")
                adjacency_dict = None
        
        # Create polygon generator with correct mapping and adjacency info
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=polygon_centroids,
            polygon_group_mapping=polygon_group_mapping,
            neighbor_distance_threshold=0.8,
            adjacency_dict=adjacency_dict
        )
        
        # Generate groups (these will be admin codes, not indices!)
        X_group = polygon_gen.get_groups(polygon_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Polygon setup complete: {len(unique_polygons)} polygons, {len(np.unique(X_group))} groups")
        print(f"X_group range: {X_group.min()} to {X_group.max()}")
        print(f"Contiguity info keys: {list(contiguity_info.keys())}")
        
        # Validation
        assert len(np.unique(X_group)) == len(unique_polygons), "Number of groups should equal number of polygons"
        assert set(X_group) == set(unique_polygons), "Group IDs should be admin codes"
        print("Validation passed!")
        
    elif assignment == 'grid':
        xmin, xmax, ymin, ymax = get_spatial_range(X_loc)
        STEP_SIZE = 0.1
        group_gen = GroupGenerator(xmin, xmax, ymin, ymax, STEP_SIZE)
        X_group = group_gen.get_groups(X_loc)
        contiguity_info = None
        
    elif assignment == 'country':
        # Get country groups and calculate mean centroids for each country
        country_data = df[['ISO_encoded', 'lat', 'lon']].groupby('ISO_encoded').mean().reset_index()
        country_centroids = country_data[['lat', 'lon']].values
        unique_countries = country_data['ISO_encoded'].unique()
        
        # Create mapping from country_index -> [country_id]
        country_group_mapping = {i: [unique_countries[i]] for i in range(len(unique_countries))}
        
        # Create country to index mapping
        country_to_idx = {country_id: idx for idx, country_id in enumerate(unique_countries)}
        
        # Convert country IDs to indices for PolygonGroupGenerator
        country_indices = np.array([country_to_idx[country_id] for country_id in df['ISO_encoded'].values])
        
        # Create polygon generator for countries
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=country_centroids,
            polygon_group_mapping=country_group_mapping,
            neighbor_distance_threshold=2.0  # Larger threshold for countries
        )
        
        X_group = polygon_gen.get_groups(country_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Country setup complete: {len(unique_countries)} countries, {len(np.unique(X_group))} groups")
        
    elif assignment == 'AEZ':
        # Get AEZ groups and calculate mean centroids for each AEZ
        aez_data = df[['AEZ_group', 'lat', 'lon']].groupby('AEZ_group').mean().reset_index()
        aez_centroids = aez_data[['lat', 'lon']].values
        unique_aezs = aez_data['AEZ_group'].unique()
        
        # Create mapping from aez_index -> [aez_id]
        aez_group_mapping = {i: [unique_aezs[i]] for i in range(len(unique_aezs))}
        
        # Create AEZ to index mapping
        aez_to_idx = {aez_id: idx for idx, aez_id in enumerate(unique_aezs)}
        
        # Convert AEZ IDs to indices for PolygonGroupGenerator
        aez_indices = np.array([aez_to_idx[aez_id] for aez_id in df['AEZ_group'].values])
        
        # Create polygon generator for AEZs
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=aez_centroids,
            polygon_group_mapping=aez_group_mapping,
            neighbor_distance_threshold=1.5  # Medium threshold for AEZs
        )
        
        X_group = polygon_gen.get_groups(aez_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"AEZ setup complete: {len(unique_aezs)} AEZs, {len(np.unique(X_group))} groups")
        
    elif assignment == 'country_AEZ':
        # Get country-AEZ groups and calculate mean centroids for each combination
        country_aez_data = df[['AEZ_country_group', 'lat', 'lon']].groupby('AEZ_country_group').mean().reset_index()
        country_aez_centroids = country_aez_data[['lat', 'lon']].values
        unique_country_aezs = country_aez_data['AEZ_country_group'].unique()
        
        # Create mapping from country_aez_index -> [country_aez_id]
        country_aez_group_mapping = {i: [unique_country_aezs[i]] for i in range(len(unique_country_aezs))}
        
        # Create country-AEZ to index mapping
        country_aez_to_idx = {ca_id: idx for idx, ca_id in enumerate(unique_country_aezs)}
        
        # Convert country-AEZ IDs to indices for PolygonGroupGenerator
        country_aez_indices = np.array([country_aez_to_idx[ca_id] for ca_id in df['AEZ_country_group'].values])
        
        # Create polygon generator for country-AEZs
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=country_aez_centroids,
            polygon_group_mapping=country_aez_group_mapping,
            neighbor_distance_threshold=1.2  # Smaller threshold for country-AEZ combinations
        )
        
        X_group = polygon_gen.get_groups(country_aez_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Country-AEZ setup complete: {len(unique_country_aezs)} country-AEZ combinations, {len(np.unique(X_group))} groups")
        
    elif assignment == 'geokmeans':
        # Get geo-based K-means clusters
        cluster_id, cluster_info, admin_to_group_map = create_kmeans_groupgenerator_from_admin_codes(
            df, n_clusters=100, random_state=42, features_for_clustering=['lat', 'lon']
        )
        
        # Calculate mean centroids for each cluster
        df_with_clusters = df.copy()
        df_with_clusters['cluster_id'] = cluster_id
        cluster_data = df_with_clusters[['cluster_id', 'lat', 'lon']].groupby('cluster_id').mean().reset_index()
        cluster_centroids = cluster_data[['lat', 'lon']].values
        unique_clusters = cluster_data['cluster_id'].unique()
        
        # Create mapping from cluster_index -> [cluster_id]
        cluster_group_mapping = {i: [unique_clusters[i]] for i in range(len(unique_clusters))}
        
        # Create cluster to index mapping
        cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
        
        # Convert cluster IDs to indices for PolygonGroupGenerator
        cluster_indices = np.array([cluster_to_idx[cluster_id] for cluster_id in cluster_id])
        
        # Create polygon generator for geo-kmeans clusters
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=cluster_centroids,
            polygon_group_mapping=cluster_group_mapping,
            neighbor_distance_threshold=1.0  # Medium threshold for geo-clusters
        )
        
        X_group = polygon_gen.get_groups(cluster_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Geo-KMeans setup complete: {len(unique_clusters)} clusters, {len(np.unique(X_group))} groups")
        
        # Save correspondence table
        correspondence_table = pd.DataFrame(list(admin_to_group_map.items()), 
                                          columns=['FEWSNET_admin_code', 'cluster_id'])
        correspondence_table.to_csv('correspondence_table_geokmeans.csv', index=False)
        
    elif assignment == 'all_kmeans':
        # Get all-features K-means clusters
        X_columns = df.drop(columns=['FEWSNET_admin_code', 'lat', 'lon', 'fews_ipc_crisis']).columns.tolist()
        cluster_id, cluster_info, admin_to_group_map = create_kmeans_groupgenerator_from_admin_codes(
            df, n_clusters=100, random_state=42, features_for_clustering=X_columns
        )
        
        # Calculate mean centroids for each cluster (using lat/lon even though clustering was on all features)
        df_with_clusters = df.copy()
        df_with_clusters['cluster_id'] = cluster_id
        cluster_data = df_with_clusters[['cluster_id', 'lat', 'lon']].groupby('cluster_id').mean().reset_index()
        cluster_centroids = cluster_data[['lat', 'lon']].values
        unique_clusters = cluster_data['cluster_id'].unique()
        
        # Create mapping from cluster_index -> [cluster_id]
        cluster_group_mapping = {i: [unique_clusters[i]] for i in range(len(unique_clusters))}
        
        # Create cluster to index mapping
        cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
        
        # Convert cluster IDs to indices for PolygonGroupGenerator
        cluster_indices = np.array([cluster_to_idx[cluster_id] for cluster_id in cluster_id])
        
        # Create polygon generator for all-features kmeans clusters
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=cluster_centroids,
            polygon_group_mapping=cluster_group_mapping,
            neighbor_distance_threshold=0.8  # Smaller threshold for feature-based clusters
        )
        
        X_group = polygon_gen.get_groups(cluster_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"All-KMeans setup complete: {len(unique_clusters)} clusters, {len(np.unique(X_group))} groups")
        
        # Save correspondence table
        correspondence_table = pd.DataFrame(list(admin_to_group_map.items()), 
                                          columns=['FEWSNET_admin_code', 'cluster_id'])
        correspondence_table.to_csv('correspondence_table_allkmeans.csv', index=False)
        
    else:
        raise ValueError(f"Unknown assignment method: {assignment}")
    
    return X_group, X_loc, contiguity_info
