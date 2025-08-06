#!/usr/bin/env python3
"""
XGBoost version of main_model_GF_replicated_fixed.py

This script replicates the full functionality of the original GeoRF pipeline
but uses XGBoost instead of Random Forest as the base learner.

Key features:
1. Data preprocessing with polars and pandas (same as RF version)
2. Multiple spatial grouping options (polygons, grid, country, AEZ, etc.)
3. Polygon-based contiguity with corrected setup
4. Time-based train-test splitting for temporal validation
5. Single-layer and 2-layer GeoXGB models (XGBoost-based GeoRF)
6. Comprehensive evaluation and result saving
7. XGBoost hyperparameters optimized for acute food crisis prediction

Date: 2024-07-21
"""

import numpy as np
import pandas as pd
import polars as pl
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# GeoRF imports (modified for XGBoost)
from GeoRF_XGB import GeoRF_XGB  # Changed from GeoRF to GeoRF_XGB
from customize import *
from data import load_demo_data
from helper import get_spatial_range
from initialization import train_test_split_all
from customize import train_test_split_rolling_window
from config import *
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# Configuration
DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_IPC_train_lag_forecast_v06252025.csv"

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

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the FEWSNET data.
    Same as RF version - no changes needed.
    """
    print("Loading data...")
    
    # Load data with polars
    data = pl.read_csv(data_path)
    
    # Drop unnecessary columns
    cols_to_drop = [
        "ISO3", "fews_ipc_adjusted", "fews_proj_med_adjusted", "fews_ipc",
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
    
    # Create lag features
    df['fews_ipc_crisis_lag_1'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(1)
    df['fews_ipc_crisis_lag_2'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(2)
    df['fews_ipc_crisis_lag_3'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(3)
    
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
    Same as RF version - no changes needed.
    """
    print(f"Setting up spatial groups using: {assignment}")
    
    # Get location coordinates
    X_loc = df[['lat', 'lon']].values
    
    if assignment == 'polygons':
        X_polygon_ids = df['FEWSNET_admin_code'].values
        
        # Get unique polygons and their centroids
        polygon_data = df[['FEWSNET_admin_code', 'lat', 'lon']].drop_duplicates()
        polygon_centroids = polygon_data[['lat', 'lon']].values
        
        # Get unique admin codes
        unique_polygons = polygon_data['FEWSNET_admin_code'].unique()
        
        # Create mapping from FEWSNET_admin_code to polygon index
        admin_to_polygon_idx = {admin_code: idx for idx, admin_code in enumerate(unique_polygons)}
        
        # FIXED: Create correct polygon_group_mapping
        polygon_group_mapping = {i: [unique_polygons[i]] for i in range(len(unique_polygons))}
        
        # Convert admin codes to polygon indices for PolygonGroupGenerator
        polygon_indices = np.array([admin_to_polygon_idx[admin_code] for admin_code in X_polygon_ids])
        
        # Create polygon generator with correct mapping
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=polygon_centroids,
            polygon_group_mapping=polygon_group_mapping,
            neighbor_distance_threshold=0.8
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

def prepare_features(df, X_group, X_loc, forecasting_scope=4):
    """
    Prepare feature matrices and identify L1/L2 feature indices with forecasting scope logic.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    X_group : numpy.ndarray
        Group assignments
    X_loc : numpy.ndarray
        Location coordinates
    forecasting_scope : int, default=4
        Forecasting scope: 1=3mo lag, 2=6mo lag, 3=9mo lag, 4=12mo lag
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    l1_index : list
        Indices for L1 features
    l2_index : list
        Indices for L2 features
    years : numpy.ndarray
        Year values for temporal splitting
    """
    print(f"Preparing features with forecasting_scope={forecasting_scope}...")
    
    # Define time-variant features for L2
    time_variants = [
        'event_count_battles', 'event_count_explosions', 'event_count_violence',
        'sum_fatalities_battles', 'sum_fatalities_explosions', 'sum_fatalities_violence',
        'event_count_battles_w5', 'event_count_explosions_w5', 'event_count_violence_w5',
        'sum_fatalities_battles_w5', 'sum_fatalities_explosions_w5', 'sum_fatalities_violence_w5',
        'event_count_battles_w10', 'event_count_explosions_w10', 'event_count_violence_w10',
        'sum_fatalities_battles_w10', 'sum_fatalities_explosions_w10', 'sum_fatalities_violence_w10',
        'nightlight', 'nightlight_sd', 'EVI', 'EVI_stdDev', 'FAO_price',
        'Evap_tavg_mean', 'Evap_tavg_stdDev', 'LWdown_f_tavg_mean', 'LWdown_f_tavg_stdDev',
        'Lwnet_tavg_mean', 'Lwnet_tavg_stdDev', 'Psurf_f_tavg_mean', 'Psurf_f_tavg_stdDev',
        'Qair_f_tavg_mean', 'Qair_f_tavg_stdDev', 'Qg_tavg_mean', 'Qg_tavg_stdDev',
        'Qh_tavg_mean', 'Qh_tavg_stdDev', 'Qle_tavg_mean', 'Qle_tavg_stdDev',
        'Qs_tavg_mean', 'Qs_tavg_stdDev', 'Qsb_tavg_mean', 'Qsb_tavg_stdDev',
        'RadT_tavg_mean', 'RadT_tavg_stdDev', 'Rainf_f_tavg_mean', 'Rainf_f_tavg_stdDev',
        'SnowCover_inst_mean', 'SnowCover_inst_stdDev', 'SnowDepth_inst_mean', 'SnowDepth_inst_stdDev',
        'Snowf_tavg_mean', 'Snowf_tavg_stdDev', 'SoilMoi00_10cm_tavg_mean', 'SoilMoi00_10cm_tavg_stdDev',
        'SoilMoi10_40cm_tavg_mean', 'SoilMoi10_40cm_tavg_stdDev', 'SoilMoi100_200cm_tavg_mean', 'SoilMoi100_200cm_tavg_stdDev',
        'SoilMoi40_100cm_tavg_mean', 'SoilMoi40_100cm_tavg_stdDev', 'SoilTemp00_10cm_tavg_mean', 'SoilTemp00_10cm_tavg_stdDev',
        'SoilTemp10_40cm_tavg_mean', 'SoilTemp10_40cm_tavg_stdDev', 'SoilTemp100_200cm_tavg_mean', 'SoilTemp100_200cm_tavg_stdDev',
        'SoilTemp40_100cm_tavg_mean', 'SoilTemp40_100cm_tavg_stdDev', 'SWdown_f_tavg_mean', 'SWdown_f_tavg_stdDev',
        'SWE_inst_mean', 'SWE_inst_stdDev', 'Swnet_tavg_mean', 'Swnet_tavg_stdDev',
        'Tair_f_tavg_mean', 'Tair_f_tavg_stdDev', 'Wind_f_tavg_mean', 'Wind_f_tavg_stdDev',
        'gpp_sd', 'gpp_mean', 'CPI', 'GDP', 'CC', 'gini', 'WFP_Price', 'WFP_Price_std'
    ]
    
    # Determine lag months based on forecasting scope  
    # 1=3mo lag, 2=6mo lag, 3=9mo lag, 4=12mo lag
    lag_months_map = {1: 3, 2: 6, 3: 9, 4: 12}
    lag_months = lag_months_map.get(forecasting_scope, 12)
    print(f"Using {lag_months}-month lag for forecasting scope {forecasting_scope}")
    
    # Create appropriate time variant list based on forecasting scope
    if forecasting_scope == 4:
        # For 12-month lag, use existing m12 variables if available
        time_variants_m12 = [variant + '_m12' for variant in time_variants]
        time_variants_list = time_variants + time_variants_m12
    else:
        # For other forecasting scopes, we'll create lagged features dynamically
        time_variants_list = time_variants.copy()
    
    # Get target variable
    y = df['fews_ipc_crisis'].values
    
    # Create comprehensive correspondence table with X_group mapping
    correspondence_df = df[['FEWSNET_admin_code', 'AEZ_group', 'AEZ_country_group', 'ISO_encoded']].copy()
    correspondence_df['X_group'] = X_group
    correspondence_table = correspondence_df.drop_duplicates()
    correspondence_table.to_csv('correspondence_table.csv', index=False)
    
    # Sort by admin code and date
    df_sorted = df.sort_values(by=['FEWSNET_admin_code', 'date'])
    
    # Create lag features based on forecasting scope
    print(f"Creating lag features for forecasting scope {forecasting_scope} ({lag_months} months)...")
    
    if forecasting_scope != 4:
        # For scopes 1, 2, 3, create lagged features for time-variant variables
        existing_cols_before = set(df_sorted.columns)
        
        for variant in time_variants:
            if variant in df_sorted.columns:
                lagged_col_name = f'{variant}_lag{lag_months}m'
                df_sorted[lagged_col_name] = df_sorted.groupby('FEWSNET_admin_code')[variant].shift(lag_months)
                time_variants_list.append(lagged_col_name)
        
        # Check for duplicates and remove if they already exist in the dataset
        print("Checking for duplicate lagged features...")
        cols_to_remove = []
        for variant in time_variants:
            lagged_col_name = f'{variant}_lag{lag_months}m'
            # Check if similar columns already exist with various naming patterns
            potential_existing = [
                f'{variant}_m{lag_months}',        # e.g., variant_m3, variant_m6
                f'{variant}_lag_{lag_months}',     # e.g., variant_lag_3, variant_lag_6  
                f'{variant}_{lag_months}m',        # e.g., variant_3m, variant_6m
                f'{variant}_l{lag_months}',        # e.g., variant_l3, variant_l6, variant_l9
                f'{variant}_L{lag_months}',        # e.g., variant_L3, variant_L6 (uppercase)
                f'{variant}.l{lag_months}',        # e.g., variant.l3, variant.l6 (with dot)
                f'{variant}.L{lag_months}'         # e.g., variant.L3, variant.L6 (with dot, uppercase)
            ]
            
            for existing_name in potential_existing:
                if existing_name in df_sorted.columns and lagged_col_name in df_sorted.columns:
                    print(f"Found duplicate: {lagged_col_name} vs {existing_name}, keeping {existing_name}")
                    cols_to_remove.append(lagged_col_name)
                    if lagged_col_name in time_variants_list:
                        time_variants_list.remove(lagged_col_name)
                    if existing_name not in time_variants_list:
                        time_variants_list.append(existing_name)
                    break
        
        # Remove duplicate columns
        if cols_to_remove:
            df_sorted = df_sorted.drop(columns=[col for col in cols_to_remove if col in df_sorted.columns])
            print(f"Removed {len(cols_to_remove)} duplicate columns")
    
    # Create additional lag features for crisis variable
    for lag in range(1, 4):
        df_sorted[f'fews_ipc_crisis_lag_{lag}'] = df_sorted.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(lag)
    
    # Drop unnecessary columns
    df_features = df_sorted.drop(columns=['date', 'fews_ipc_crisis', 'AEZ_group', 'ISO_encoded', 'AEZ_country_group'])
    
    # Get feature matrix
    X = df_features.values
    
    # Identify L2 feature indices
    l2_index = []
    for i, col in enumerate(df_features.columns):
        if col in time_variants_list:
            l2_index.append(i)
    
    # L1 features are all others
    l1_index = [i for i in range(X.shape[1]) if i not in l2_index]
    
    # Apply imputation (reduced verbosity)
    print("Applying imputation to missing values...")
    X = comp_impute(X, strategy="max_plus", multiplier=100.0)
    
    # Get years for temporal splitting
    years = df_sorted['years'].values
    # get terms within each year:group by FESNET_admin_code and year, set first month as 1, second months as 2
    months = df_sorted['date'].dt.month.values
    # if months = 1,2,3 then terms =1, if months = 4,5,6 then terms = 2, if months = 7,8,9 then terms = 3, if months = 10,11,12 then terms = 4
    terms = np.zeros_like(months)
    terms[months <= 3] = 1
    terms[(months > 3) & (months <= 6)] = 2
    terms[(months > 6) & (months <= 9)] = 3
    terms[(months > 9) & (months <= 12)] = 4
    
    print(f"Feature preparation complete: {X.shape[1]} features, {len(l1_index)} L1 features, {len(l2_index)} L2 features")
    
    return X, y, l1_index, l2_index, years, terms, df_sorted['date']

def validate_polygon_contiguity(contiguity_info, X_group):
    """
    Validate polygon contiguity setup.
    Same as RF version - no changes needed.
    """
    print("=== Polygon Contiguity Validation ===")
    
    from partition_opt import get_polygon_neighbors
    
    neighbors = get_polygon_neighbors(contiguity_info['polygon_centroids'], 
                                     contiguity_info['neighbor_distance_threshold'])
    neighbor_counts = [len(n) for n in neighbors.values()]
    
    print(f"Neighbor stats: min={min(neighbor_counts)}, max={max(neighbor_counts)}, "
          f"mean={np.mean(neighbor_counts):.1f}")
    
    # Check for isolated polygons
    isolated_polygons = [poly_id for poly_id, neighs in neighbors.items() if len(neighs) == 0]
    if len(isolated_polygons) > 0:
        print(f"Warning: {len(isolated_polygons)} isolated polygons")
    
    # Check centroids
    centroids = contiguity_info['polygon_centroids']
    print(f"Centroid range: lat {centroids[:, 0].min():.2f}-{centroids[:, 0].max():.2f}, "
          f"lon {centroids[:, 1].min():.2f}-{centroids[:, 1].max():.2f}")
    
    # Check group sizes
    group_sizes = pd.Series(X_group).value_counts()
    print(f"Group size stats: min={group_sizes.min()}, max={group_sizes.max()}, "
          f"mean={group_sizes.mean():.1f}")
    
    # Check for small groups
    small_groups = group_sizes[group_sizes < 10]
    if len(small_groups) > 0:
        print(f"Warning: {len(small_groups)} groups have <10 samples")
    
    print("=== End Polygon Validation ===")

def create_correspondence_table(df, years, dates, train_year, quarter, X_branch_id, result_dir):
    """
    Create correspondence table mapping FEWSNET_admin_code to partition_id for rolling window splits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe with FEWSNET_admin_code
    years : numpy.ndarray
        Year values for all data
    dates : pandas.Series
        Date values for precise temporal filtering
    train_year : int
        Year for which training was done (2024)
    quarter : int
        Quarter for which training was done (1-4)
    X_branch_id : numpy.ndarray
        Branch IDs (partition IDs) from trained model
    result_dir : str
        Directory to save correspondence table
    """
    print(f"Creating correspondence table for Q{quarter} {train_year}...")
    
    try:
        import pandas as pd
        
        # Convert dates to pandas datetime if needed
        if not isinstance(dates, pd.Series):
            dates = pd.to_datetime(dates)
        
        # Define quarter end dates for 2024
        quarter_ends = {
            1: pd.Timestamp('2024-03-31'),
            2: pd.Timestamp('2024-06-30'),
            3: pd.Timestamp('2024-09-30'),
            4: pd.Timestamp('2024-12-31')
        }
        
        # Get the training mask (same logic as rolling window)
        test_quarter_end = quarter_ends[quarter]
        train_end_date = test_quarter_end
        train_start_date = train_end_date - pd.DateOffset(years=5)
        train_mask = (dates >= train_start_date) & (dates < train_end_date)
        
        # Check if X_branch_id length matches training data
        if len(X_branch_id) != np.sum(train_mask):
            print(f"Warning: X_branch_id length ({len(X_branch_id)}) doesn't match training data length ({np.sum(train_mask)})")
            print("Using first N samples from df for correspondence table creation")
            # Use the first len(X_branch_id) samples that match the training criteria
            train_indices = np.where(train_mask)[0][:len(X_branch_id)]
            df_train = df.iloc[train_indices].copy()
        else:
            # Get training subset of dataframe
            df_train = df[train_mask].copy()
        
        # Add partition IDs to training data
        df_train = df_train.iloc[:len(X_branch_id)].copy()  # Ensure exact length match
        df_train['partition_id'] = X_branch_id
        
        # Extract unique FEWSNET_admin_code and partition_id pairs
        correspondence_table = df_train[['FEWSNET_admin_code', 'partition_id']].drop_duplicates()
        
        # Sort by admin code for better readability
        correspondence_table = correspondence_table.sort_values('FEWSNET_admin_code')
        
        # Save to result directory with quarter naming
        output_path = os.path.join(result_dir, f'correspondence_table_Q{quarter}_{train_year}.csv')
        correspondence_table.to_csv(output_path, index=False)
        
        print(f"Correspondence table saved to: {output_path}")
        print(f"Table contains {len(correspondence_table)} unique admin_code-partition_id pairs")
        
    except Exception as e:
        print(f"Error creating correspondence table: {e}")
        print("Correspondence table creation failed, but model training/evaluation will continue")

def run_temporal_evaluation(X, y, X_loc, X_group, years, dates, l1_index, l2_index, 
                           assignment, contiguity_info, df, nowcasting=False, max_depth=None, input_terms=None, desire_terms=None,
                           track_partition_metrics=False, enable_metrics_maps=True,
                           # XGBoost hyperparameters
                           learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0,
                           subsample=0.8, colsample_bytree=0.8):
    """
    Run temporal evaluation for 2024 quarters using rolling window approach with XGBoost.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    X_loc : numpy.ndarray
        Location coordinates
    X_group : numpy.ndarray
        Group assignments
    years : numpy.ndarray
        Year values
    dates : pandas.Series or array-like
        Date values for precise temporal splitting
    l1_index : list
        L1 feature indices
    l2_index : list
        L2 feature indices
    assignment : str
        Spatial assignment method
    contiguity_info : dict or None
        Contiguity information
    df : pandas.DataFrame
        Original dataframe with FEWSNET_admin_code
    nowcasting : bool
        Whether to use 2-layer model
    max_depth : int or None
        Maximum depth for XGB models
    input_terms : numpy.ndarray
        Terms within each year (1-4 corresponding to quarters)
    desire_terms : int or None
        Specific quarter to evaluate (1-4), or None for all quarters
    track_partition_metrics : bool
        Whether to enable partition metrics tracking and visualization
    enable_metrics_maps : bool
        Whether to create maps showing F1/accuracy improvements
    learning_rate : float
        XGBoost learning rate
    reg_alpha : float
        XGBoost L1 regularization
    reg_lambda : float
        XGBoost L2 regularization
    subsample : float
        XGBoost subsample ratio
    colsample_bytree : float
        XGBoost column subsampling ratio
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Evaluation results with quarter information
    y_pred_test : pandas.DataFrame
        Prediction results with quarter information
    """
    print(f"Running temporal evaluation with XGBoost (nowcasting={nowcasting})...")
    print(f"XGB hyperparameters: lr={learning_rate}, reg_alpha={reg_alpha}, reg_lambda={reg_lambda}")
    
    # Initialize results tracking
    results_df = pd.DataFrame(columns=[
        'year', 'quarter', 'precision(0)', 'recall(0)', 'f1(0)', 'precision(1)', 'recall(1)', 'f1(1)',
        'precision_base(0)', 'recall_base(0)', 'f1_base(0)', 'precision_base(1)', 'recall_base(1)', 'f1_base(1)',
        'num_samples(0)', 'num_samples(1)'
    ])
    
    y_pred_test = pd.DataFrame(columns=['year', 'quarter', 'month', 'adm_code', 'fews_ipc_crisis_pred', 'fews_ipc_crisis_true'])
    
    # Setup correspondence table path for partition metrics tracking
    correspondence_table_path = None
    if track_partition_metrics:
        print("Setting up partition metrics tracking...")
        
        # Create base correspondence table from the data
        correspondence_table_path = 'correspondence_table_metrics.csv'
        try:
            # Create correspondence table that properly maps X_group to FEWSNET_admin_code
            if not os.path.exists(correspondence_table_path):
                print("Creating correspondence table for partition metrics...")
                
                # Create proper correspondence table based on assignment method
                if assignment == 'polygons':
                    # For polygon assignment, X_group contains FEWSNET_admin_code values directly
                    # Create a direct mapping, but we need to verify the relationship
                    print(f"Debug: Creating polygon correspondence table")
                    print(f"X_group type: {type(X_group[0])}, sample values: {X_group[:5]}")
                    print(f"FEWSNET_admin_code sample values: {df['FEWSNET_admin_code'].head().tolist()}")
                    
                    # For polygon assignment, X_group should already BE the admin codes
                    # So we create a simple identity mapping
                    unique_admin_codes = df['FEWSNET_admin_code'].dropna().unique()
                    unique_groups = np.unique(X_group)
                    
                    print(f"Unique admin codes count: {len(unique_admin_codes)}")
                    print(f"Unique X_group values count: {len(unique_groups)}")
                    
                    # Check if X_group values are actually admin codes
                    admin_codes_set = set(unique_admin_codes)
                    group_codes_set = set(unique_groups)
                    
                    if admin_codes_set == group_codes_set:
                        print("X_group contains admin codes directly - creating identity mapping")
                        corr_df = pd.DataFrame({
                            'FEWSNET_admin_code': unique_admin_codes,
                            'X_group': unique_admin_codes
                        })
                    else:
                        print("X_group doesn't match admin codes - creating index-based mapping")
                        # Create mapping based on actual data relationships
                        unique_entries = []
                        for i in range(len(df)):
                            admin_code = df.iloc[i]['FEWSNET_admin_code']
                            group_id = X_group[i]
                            if not pd.isna(admin_code):
                                unique_entries.append({
                                    'FEWSNET_admin_code': admin_code,
                                    'X_group': group_id
                                })
                        
                        # Remove duplicates and create DataFrame
                        corr_df = pd.DataFrame(unique_entries).drop_duplicates()
                
                elif assignment in ['country', 'AEZ', 'country_AEZ']:
                    # For these assignments, create mapping from admin codes to group IDs
                    mapping_data = []
                    for i in range(len(df)):
                        admin_code = df.iloc[i]['FEWSNET_admin_code']
                        group_id = X_group[i]
                        if not pd.isna(admin_code):
                            mapping_data.append({
                                'FEWSNET_admin_code': admin_code,
                                'X_group': group_id
                            })
                    
                    corr_df = pd.DataFrame(mapping_data).drop_duplicates()
                
                elif assignment in ['geokmeans', 'all_kmeans']:
                    # For kmeans assignments, use the existing correspondence table if available
                    kmeans_table_path = f'correspondence_table_{assignment.replace("_", "")}.csv'
                    if os.path.exists(kmeans_table_path):
                        print(f"Using existing kmeans correspondence table: {kmeans_table_path}")
                        correspondence_table_path = kmeans_table_path
                        corr_df = None  # Don't create new table
                    else:
                        print(f"Warning: Expected kmeans correspondence table not found: {kmeans_table_path}")
                        correspondence_table_path = None
                        corr_df = None
                
                else:
                    # For grid assignment, create a simple mapping
                    mapping_data = []
                    for i in range(len(df)):
                        admin_code = df.iloc[i]['FEWSNET_admin_code']
                        group_id = X_group[i]
                        if not pd.isna(admin_code):
                            mapping_data.append({
                                'FEWSNET_admin_code': admin_code,
                                'X_group': group_id
                            })
                    
                    corr_df = pd.DataFrame(mapping_data).drop_duplicates()
                
                # Save the correspondence table if we created one
                if corr_df is not None and len(corr_df) > 0:
                    corr_df.to_csv(correspondence_table_path, index=False)
                    print(f"Created correspondence table with {len(corr_df)} entries: {correspondence_table_path}")
                    print(f"Sample entries:")
                    print(corr_df.head())
                else:
                    if corr_df is not None:  # Empty DataFrame
                        print("Warning: Could not create correspondence table - no valid data found")
                        correspondence_table_path = None
                    # else: using existing kmeans table, so correspondence_table_path is already set
                        
            else:
                print(f"Using existing correspondence table: {correspondence_table_path}")
                
        except Exception as e:
            print(f"Warning: Could not create correspondence table: {e}")
            print("Maps will not be generated, but CSV metrics will still be saved.")
            correspondence_table_path = None
    
    # Determine contiguity settings
    if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans']:
        contiguity_type = 'polygon'
        polygon_contiguity_info = contiguity_info
    else:
        contiguity_type = 'grid'
        polygon_contiguity_info = None
    
    # Run evaluation for all quarters from 2015 to 2024 using rolling window
    start_year = 2015
    end_year = 2024
    print(f"\nEvaluating all quarters from {start_year} to {end_year} using rolling window approach...")
    
    # Determine which quarters to evaluate based on desire_terms
    if desire_terms is None:
        quarters_to_evaluate = [1, 2, 3, 4]  # Evaluate all quarters
        print(f"Evaluating all quarters (Q1-Q4) for each year from {start_year} to {end_year}")
    else:
        quarters_to_evaluate = [desire_terms]  # Evaluate only specific quarter
        print(f"Evaluating only Q{desire_terms} for each year from {start_year} to {end_year}")
    
    # Create progress bar for quarterly evaluation
    total_evaluations = len(range(start_year, end_year + 1)) * len(quarters_to_evaluate)
    progress_bar = tqdm(
        total=total_evaluations, 
        desc="GeoXGB Quarterly Evaluation", 
        unit="quarter",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Loop through years and quarters
    for test_year in range(start_year, end_year + 1):
        for quarter in quarters_to_evaluate:
            progress_bar.set_description(f"GeoXGB Q{quarter} {test_year}")
            print(f"\n--- Evaluating Q{quarter} {test_year} ---")
            
            # Train-test split with rolling window (5 years before quarter end)
            (Xtrain, ytrain, Xtrain_loc, Xtrain_group,
             Xtest, ytest, Xtest_loc, Xtest_group) = train_test_split_rolling_window(
                X, y, X_loc, X_group, years, dates, test_year=test_year, input_terms=input_terms, need_terms=quarter)
            
            ytrain = ytrain.astype(int)
            ytest = ytest.astype(int)
            
            print(f"Train samples: {len(ytrain)}, Test samples: {len(ytest)}")
            
            # Skip evaluation if no test samples
            if len(ytest) == 0:
                print(f"Warning: No test samples for Q{quarter} {test_year}. Skipping this quarter.")
                continue
            
            if nowcasting:
                # 2-layer model with XGBoost
                Xtrain_L1 = Xtrain[:, l1_index]
                Xtrain_L2 = Xtrain[:, l2_index]
                Xtest_L1 = Xtest[:, l1_index]
                Xtest_L2 = Xtest[:, l2_index]
                
                # Create and train 2-layer GeoXGB model
                geoxgb_2layer = GeoRF_XGB(
                    min_model_depth=MIN_DEPTH,
                    max_model_depth=MAX_DEPTH,
                    n_jobs=N_JOBS,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree
                )
                
                # Train 2-layer model with optional metrics tracking
                if track_partition_metrics:
                    # Note: 2-layer fit doesn't support partition metrics yet, 
                    # but we can extend it later if needed
                    print("Note: Partition metrics tracking not yet supported for 2-layer models")
                    
                geoxgb_2layer.fit_2layer(
                    Xtrain_L1, Xtrain_L2, ytrain, Xtrain_group,
                    val_ratio=VAL_RATIO,
                    contiguity_type=contiguity_type,
                    polygon_contiguity_info=polygon_contiguity_info
                )
                
                # Get predictions
                ypred = geoxgb_2layer.predict_2layer(Xtest_L1, Xtest_L2, Xtest_group, correction_strategy='flip')
                
                # Evaluate
                (pre, rec, f1, pre_base, rec_base, f1_base) = geoxgb_2layer.evaluate_2layer(
                    X_L1_test=Xtest_L1,
                    X_L2_test=Xtest_L2,
                    y_test=ytest,
                    X_group_test=Xtest_group,
                    X_L1_train=Xtrain_L1,
                    X_L2_train=Xtrain_L2,
                    y_train=ytrain,
                    X_group_train=Xtrain_group,
                    correction_strategy='flip',
                    print_to_file=True,
                    contiguity_type=contiguity_type,
                    polygon_contiguity_info=polygon_contiguity_info
                )
                
                print(f"Q{quarter} {test_year} Test - 2-Layer GeoXGB F1: {f1}, 2-Layer Base XGB F1: {f1_base}")
                
                # Extract and save correspondence table for 2-layer model
                try:
                    X_branch_id_path = os.path.join(geoxgb_2layer.dir_space, 'X_branch_id.npy')
                    if os.path.exists(X_branch_id_path):
                        X_branch_id = np.load(X_branch_id_path)
                        create_correspondence_table(df, years, dates, test_year, quarter, X_branch_id, geoxgb_2layer.model_dir)
                except Exception as e:
                    print(f"Warning: Could not create correspondence table for Q{quarter} {test_year}: {e}")
            
            else:
                # Single-layer model with XGBoost
                geoxgb = GeoRF_XGB(
                    min_model_depth=MIN_DEPTH,
                    max_model_depth=MAX_DEPTH,
                    n_jobs=N_JOBS,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree
                )
                
                # Train model with optional partition metrics tracking
                if track_partition_metrics:
                    print(f"Training GeoXGB with partition metrics tracking enabled")
                    print(f"Correspondence table path: {correspondence_table_path}")
                    print(f"Training set shape: {Xtrain.shape}, Groups shape: {Xtrain_group.shape}")
                    print(f"Unique training groups: {len(np.unique(Xtrain_group))}")
                    
                    # Verify correspondence table exists and is readable
                    if correspondence_table_path and os.path.exists(correspondence_table_path):
                        test_df = pd.read_csv(correspondence_table_path)
                        print(f"Correspondence table loaded successfully with {len(test_df)} entries")
                        print(f"Columns: {test_df.columns.tolist()}")
                        print(f"Sample entries:\n{test_df.head()}")
                    else:
                        print(f"Warning: Correspondence table not found at {correspondence_table_path}")
                    
                geoxgb.fit(
                    Xtrain, ytrain, Xtrain_group,
                    val_ratio=VAL_RATIO,
                    contiguity_type=contiguity_type,
                    polygon_contiguity_info=polygon_contiguity_info,
                    track_partition_metrics=track_partition_metrics,
                    correspondence_table_path=correspondence_table_path
                )
                
                # Check if metrics were tracked
                if track_partition_metrics and hasattr(geoxgb, 'metrics_tracker'):
                    if geoxgb.metrics_tracker is not None:
                        print(f"\nPartition metrics tracker found for Q{quarter} {test_year}")
                        
                        # Check if any metrics were actually recorded
                        if hasattr(geoxgb.metrics_tracker, 'all_metrics') and geoxgb.metrics_tracker.all_metrics:
                            print(f"Number of metric records: {len(geoxgb.metrics_tracker.all_metrics)}")
                            
                            # Show some sample metrics
                            for i, record in enumerate(geoxgb.metrics_tracker.all_metrics[:3]):
                                print(f"  Record {i}: Round {record.get('partition_round', 'N/A')}, "
                                      f"Branch {record.get('branch_id', 'N/A')}, "
                                      f"F1 improvement: {record.get('f1_improvement', 'N/A'):.4f}")
                        else:
                            print("No metrics records found in tracker")
                        
                        # Try to get summary
                        try:
                            summary = geoxgb.metrics_tracker.get_improvement_summary()
                            if summary:
                                print(f"\nPartition Metrics Summary for Q{quarter} {test_year}:")
                                print(f"  Total partitions tracked: {summary['total_partitions']}")
                                print(f"  Average F1 improvement: {summary['avg_f1_improvement']:.4f}")
                                print(f"  Average accuracy improvement: {summary['avg_accuracy_improvement']:.4f}")
                                print(f"  Positive F1 improvements: {summary['positive_f1_improvements']}")
                                print(f"  Positive accuracy improvements: {summary['positive_accuracy_improvements']}")
                            else:
                                print("Warning: No partition metrics summary available")
                        except Exception as e:
                            print(f"Error getting metrics summary: {e}")
                            
                        # Check if visualization files were created
                        if hasattr(geoxgb, 'model_dir'):
                            vis_dir = os.path.join(geoxgb.model_dir, 'vis')
                            metrics_dir = os.path.join(geoxgb.model_dir, 'partition_metrics')
                            
                            if os.path.exists(vis_dir):
                                vis_files = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
                                print(f"Visualization files created: {len(vis_files)}")
                                if vis_files:
                                    print(f"  Sample files: {vis_files[:3]}")
                            else:
                                print("No visualization directory found")
                                
                            if os.path.exists(metrics_dir):
                                csv_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
                                print(f"Metrics CSV files created: {len(csv_files)}")
                                if csv_files:
                                    print(f"  Sample files: {csv_files[:3]}")
                            else:
                                print("No metrics directory found")
                    else:
                        print("Warning: Metrics tracker is None")
                else:
                    if track_partition_metrics:
                        print("Warning: Metrics tracker not found on geoxgb object")
                
                # Get predictions
                ypred = geoxgb.predict(Xtest, Xtest_group)
                
                # Evaluate
                (pre, rec, f1, pre_base, rec_base, f1_base) = geoxgb.evaluate(
                    Xtest, ytest, Xtest_group, eval_base=True, print_to_file=True
                )
                
                print(f"Q{quarter} {test_year} Test - GeoXGB F1: {f1}, Base XGB F1: {f1_base}")
                
                # Extract and save correspondence table for single-layer model
                try:
                    X_branch_id_path = os.path.join(geoxgb.dir_space, 'X_branch_id.npy')
                    if os.path.exists(X_branch_id_path):
                        X_branch_id = np.load(X_branch_id_path)
                        create_correspondence_table(df, years, dates, test_year, quarter, X_branch_id, geoxgb.model_dir)
                except Exception as e:
                    print(f"Warning: Could not create correspondence table for Q{quarter} {test_year}: {e}")
        
        # Store results
        nsample_class = np.bincount(ytest)
        
        results_df = pd.concat([results_df, pd.DataFrame({
            'year': [test_year],
            'quarter': [quarter],
            'precision(0)': [pre[0]],
            'precision(1)': [pre[1]],
            'recall(0)': [rec[0]],
            'recall(1)': [rec[1]],
            'f1(0)': [f1[0]],
            'f1(1)': [f1[1]],
            'precision_base(0)': [pre_base[0]],
            'precision_base(1)': [pre_base[1]],
            'recall_base(0)': [rec_base[0]],
            'recall_base(1)': [rec_base[1]],
            'f1_base(0)': [f1_base[0]],
            'f1_base(1)': [f1_base[1]],
            'num_samples(0)': [nsample_class[0]],
            'num_samples(1)': [nsample_class[1]]
        })], ignore_index=True)
        
        # Store predictions - need to handle column indexing issue
        try:
            # Try to get month and admin code from test data
            # This assumes these are in the original columns
            y_pred_test = pd.concat([y_pred_test, pd.DataFrame({
                'year': [test_year] * len(ytest),
                'quarter': [quarter] * len(ytest),
                'month': [quarter * 3] * len(ytest),  # Use quarter end month (3, 6, 9, 12)
                'adm_code': [0] * len(ytest),  # Placeholder - would need actual admin codes
                'fews_ipc_crisis_pred': ypred,
                'fews_ipc_crisis_true': ytest
            })], ignore_index=True)
        except:
            # Fallback with placeholders
            y_pred_test = pd.concat([y_pred_test, pd.DataFrame({
                'year': [test_year] * len(ytest),
                'quarter': [quarter] * len(ytest),
                'month': [quarter * 3] * len(ytest),  # Use quarter end month (3, 6, 9, 12)
                'adm_code': [0] * len(ytest),
                'fews_ipc_crisis_pred': ypred,
                'fews_ipc_crisis_true': ytest
            })], ignore_index=True)
            
            # Update progress bar
            progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    return results_df, y_pred_test

def save_results(results_df, y_pred_test, assignment, nowcasting=False, max_depth=None, desire_terms=None, forecasting_scope=None):
    """
    Save evaluation results to CSV files.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Evaluation results
    y_pred_test : pandas.DataFrame
        Prediction results
    assignment : str
        Spatial assignment method
    nowcasting : bool
        Whether 2-layer model was used
    max_depth : int or None
        Maximum depth setting
    desire_terms : int or None
        Desired terms setting
    forecasting_scope : int or None
        Forecasting scope (1=3mo, 2=6mo, 3=9mo, 4=12mo)
    """
    # Create file names based on assignment
    pred_test_name = 'y_pred_test_xgb_g'  # Added _xgb suffix
    results_df_name = 'results_df_xgb_g'  # Added _xgb suffix
    
    assignment_suffixes = {
        'polygons': 'p',
        'grid': 'g',
        'country': 'c',
        'AEZ': 'ae',
        'country_AEZ': 'cae',
        'geokmeans': 'gk',
        'all_kmeans': 'ak'
    }
    
    if assignment in assignment_suffixes:
        pred_test_name += assignment_suffixes[assignment]
        results_df_name += assignment_suffixes[assignment]
    
    # Add depth suffix if specified
    if max_depth is not None:
        pred_test_name += f'_d{max_depth}'
        results_df_name += f'_d{max_depth}'
    
    if desire_terms is not None:
        pred_test_name += f'_t{desire_terms}'
        results_df_name += f'_t{desire_terms}'
    
    # Add forecasting scope suffix
    if forecasting_scope is not None:
        pred_test_name += f'_fs{forecasting_scope}'
        results_df_name += f'_fs{forecasting_scope}'
    
    # Add nowcasting suffix
    if nowcasting:
        pred_test_name += '_nowcast'
        results_df_name += '_nowcast'
    
    # Add file extension
    pred_test_name += '.csv'
    results_df_name += '.csv'
    
    # Save files
    y_pred_test.to_csv(pred_test_name, index=False)
    results_df.to_csv(results_df_name, index=False)
    
    print(f"Results saved to {results_df_name}")
    print(f"Predictions saved to {pred_test_name}")

def main():
    """
    Main function to run the complete XGBoost pipeline.
    """
    print("=== Starting GeoXGB Food Crisis Prediction Pipeline ===")
    
    # Configuration
    assignment = 'polygons'  # Change this to test different grouping methods
    nowcasting = False       # Set to True for 2-layer model
    max_depth = 6           # XGBoost default, good for preventing overfitting
    desire_terms = None      # None=all quarters, 1=Q1 only, 2=Q2 only, 3=Q3 only, 4=Q4 only
    forecasting_scope = 1    # 1=3mo lag, 2=6mo lag, 3=9mo lag, 4=12mo lag
    
    # XGBoost-specific hyperparameters
    learning_rate = 0.1     # Moderate learning rate for stability
    reg_alpha = 0.1         # L1 regularization for feature selection
    reg_lambda = 1.0        # L2 regularization for stability
    subsample = 0.8         # Prevent overfitting
    colsample_bytree = 0.8  # Prevent overfitting
    
    # Partition Metrics Tracking Configuration
    track_partition_metrics = True  # Enable partition metrics tracking and visualization
    enable_metrics_maps = True      # Create maps showing F1/accuracy improvements
    
    print(f"Configuration:")
    print(f"  - Assignment method: {assignment}")
    print(f"  - Nowcasting (2-layer): {nowcasting}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Desired terms: {desire_terms} ({'All quarters (Q1-Q4)' if desire_terms is None else f'Q{desire_terms} only'})")
    print(f"  - Forecasting scope: {forecasting_scope} ({[3,6,9,12][forecasting_scope-1]}-month lag)")
    print(f"  - Rolling window: 5-year training windows before each test quarter")
    print(f"  - Track partition metrics: {track_partition_metrics}")
    print(f"  - Enable metrics maps: {enable_metrics_maps}")
    print(f"  - XGBoost hyperparameters: lr={learning_rate}, reg_alpha={reg_alpha}, reg_lambda={reg_lambda}")
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(DATA_PATH)
        
        # Step 2: Setup spatial groups
        X_group, X_loc, contiguity_info = setup_spatial_groups(df, assignment)
        
        # Step 3: Prepare features with forecasting scope
        X, y, l1_index, l2_index, years, terms, dates = prepare_features(df, X_group, X_loc, forecasting_scope=forecasting_scope)
        
        # Step 4: Validate polygon contiguity (if applicable)
        if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans'] and contiguity_info is not None:
            validate_polygon_contiguity(contiguity_info, X_group)
        
        # Step 5: Run temporal evaluation with XGBoost
        results_df, y_pred_test = run_temporal_evaluation(
            X, y, X_loc, X_group, years, dates, l1_index, l2_index,
            assignment, contiguity_info, df, nowcasting, max_depth, input_terms=terms, desire_terms=desire_terms,
            track_partition_metrics=track_partition_metrics, enable_metrics_maps=enable_metrics_maps,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )
        
        # Step 6: Save results
        save_results(results_df, y_pred_test, assignment, nowcasting, max_depth, desire_terms=desire_terms, forecasting_scope=forecasting_scope)
        
        # Step 7: Display summary
        print("\n=== XGBoost Evaluation Summary ===")
        if 'quarter' in results_df.columns:
            print("Results by Quarter:")
            print(results_df.groupby(['year', 'quarter'])[['f1(0)', 'f1(1)', 'f1_base(0)', 'f1_base(1)']].mean())
        else:
            print("Results by Year:")
            print(results_df.groupby('year')[['f1(0)', 'f1(1)', 'f1_base(0)', 'f1_base(1)']].mean())
        
        print("\n=== XGBoost Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)