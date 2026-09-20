#!/usr/bin/env python3
"""
Pre-Partitioning Diagnostic System

Generates per-polygon misclassification rate choropleth maps before spatial partitioning
begins. This diagnostic helps anticipate how optimization may partition and aids debugging.

Key Features:
- Computes training-only error rates per polygon
- Handles zero-denominator cases (no training samples)
- Creates choropleth maps for overall and class-1 specific errors
- Provides summary statistics and quality checks
- Integrates seamlessly with GeoRF and XGBoost pipelines

Usage:
    from src.diagnostics.pre_partition_diagnostic import create_pre_partition_diagnostics
    
    create_pre_partition_diagnostics(
        df_train=training_dataframe,
        y_true=actual_labels,
        y_pred=baseline_predictions,
        vis_dir="./vis",
        shapefile_path="path/to/polygons.shp",
        uid_col="FEWSNET_admin_code",
        class_positive=1,
        min_n_threshold=10,
        seed=42
    )
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')



try:
    import geopandas as gpd
    import contextily as ctx
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    print("WARNING: geopandas/contextily not available. Maps will not be generated.")
    GEOSPATIAL_AVAILABLE = False

# Import for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def compute_training_error_rates(df, y_true, y_pred, uid_col='FEWSNET_admin_code', 
                               class_positive=1, fold_split_col=None):
    """
    Compute per-polygon error rates from training data only.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing training data with polygon IDs
    y_true : array-like
        True labels for all samples
    y_pred : array-like  
        Predicted labels for all samples
    uid_col : str, default='FEWSNET_admin_code'
        Column name containing polygon unique identifiers
    class_positive : int, default=1
        Label value for positive class (crisis)
    fold_split_col : str, optional
        Column indicating train/valid/test split. If provided, filters to 'train' only
        
    Returns
    -------
    pandas.DataFrame
        Per-polygon error statistics with columns:
        - uid_col: Polygon identifier
        - pct_err_all: Overall misclassification rate [0-1]
        - pct_err_cls1: Class 1 misclassification rate [0-1]
        - n_train: Number of training samples per polygon
        - n_pos_train: Number of positive training samples per polygon
    """
    print("\n=== Computing Training Error Rates by Polygon ===")
    
    # Create working dataframe
    work_df = df.copy()
    work_df['y_true'] = y_true
    work_df['y_pred'] = y_pred
    
    # Filter to training data only if fold split column provided
    if fold_split_col and fold_split_col in work_df.columns:
        initial_count = len(work_df)
        work_df = work_df[work_df[fold_split_col] == 'train'].copy()
        print(f"  Filtered to training data: {len(work_df):,} samples (from {initial_count:,})")
    else:
        print(f"  Using all {len(work_df):,} samples as training data")
    
    # Remove rows with missing polygon IDs
    work_df = work_df.dropna(subset=[uid_col])
    print(f"  Samples with valid polygon IDs: {len(work_df):,}")
    
    # Compute error signals
    work_df['err_all'] = (work_df['y_true'] != work_df['y_pred']).astype(int)
    work_df['is_positive'] = (work_df['y_true'] == class_positive).astype(int)
    work_df['err_cls1'] = (work_df['is_positive'] & (work_df['y_pred'] != work_df['y_true'])).astype(int)
    
    # Group by polygon and compute statistics
    polygon_stats = work_df.groupby(uid_col).agg({
        'err_all': ['sum', 'count'],
        'err_cls1': 'sum',
        'is_positive': 'sum'
    }).reset_index()
    
    # Flatten column names
    polygon_stats.columns = [uid_col, 'n_err_all', 'n_train', 'n_err_cls1', 'n_pos_train']
    
    # Compute error rates with proper handling of zero denominators
    polygon_stats['pct_err_all'] = np.where(
        polygon_stats['n_train'] > 0,
        polygon_stats['n_err_all'] / polygon_stats['n_train'],
        np.nan
    )
    
    polygon_stats['pct_err_cls1'] = np.where(
        polygon_stats['n_pos_train'] > 0,
        polygon_stats['n_err_cls1'] / polygon_stats['n_pos_train'], 
        np.nan
    )
    
    # Quality checks
    print(f"  Unique polygons processed: {len(polygon_stats):,}")
    print(f"  Polygons with training data: {(polygon_stats['n_train'] > 0).sum():,}")
    print(f"  Polygons with positive samples: {(polygon_stats['n_pos_train'] > 0).sum():,}")
    
    # Summary statistics
    valid_err_all = polygon_stats['pct_err_all'].dropna()
    valid_err_cls1 = polygon_stats['pct_err_cls1'].dropna()
    
    if len(valid_err_all) > 0:
        print(f"  Overall error rate: {valid_err_all.mean():.3f} +- {valid_err_all.std():.3f}")
        print(f"  Error rate range: [{valid_err_all.min():.3f}, {valid_err_all.max():.3f}]")
    
    if len(valid_err_cls1) > 0:
        print(f"  Class 1 error rate: {valid_err_cls1.mean():.3f} +- {valid_err_cls1.std():.3f}")
        print(f"  Class 1 error range: [{valid_err_cls1.min():.3f}, {valid_err_cls1.max():.3f}]")
    
    # Keep only essential columns for output
    result_cols = [uid_col, 'pct_err_all', 'pct_err_cls1', 'n_train', 'n_pos_train']
    return polygon_stats[result_cols]


def save_diagnostic_csv(error_df, vis_dir, uid_col='FEWSNET_admin_code', min_n_threshold=10):
    """
    Save diagnostic results to CSV with quality flags.
    
    Parameters
    ----------
    error_df : pandas.DataFrame
        Per-polygon error statistics from compute_training_error_rates
    vis_dir : str or Path
        Output directory for diagnostic files
    uid_col : str
        Column name containing polygon identifiers
    min_n_threshold : int
        Minimum training samples to avoid low-sample warnings
        
    Returns
    -------
    Path
        Path to saved CSV file
    """
    vis_path = Path(vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)
    
    # Add quality flags
    output_df = error_df.copy()
    output_df['low_sample_flag'] = output_df['n_train'] < min_n_threshold
    output_df['no_pos_samples_flag'] = output_df['n_pos_train'] == 0
    output_df['no_train_flag'] = output_df['n_train'] == 0
    
    # Save CSV
    csv_path = vis_path / "train_error_by_polygon.csv"
    output_df.to_csv(csv_path, index=False)
    
    print(f"\n=== Diagnostic CSV Saved ===")
    print(f"  File: {csv_path}")
    print(f"  Polygons: {len(output_df):,}")
    print(f"  Low sample warnings (n<{min_n_threshold}): {output_df['low_sample_flag'].sum():,}")
    print(f"  No positive samples: {output_df['no_pos_samples_flag'].sum():,}")
    print(f"  No training samples: {output_df['no_train_flag'].sum():,}")
    
    return csv_path


def create_error_choropleth_maps(error_df, vis_dir, shapefile_path=None, 
                                uid_col='FEWSNET_admin_code', crs_target='EPSG:4326',
                                missing_color='lightgray', dpi=200, seed=42, VIS_DEBUG_MODE=True):
    """
    Generate choropleth maps showing error rates by polygon.
    
    Parameters
    ----------
    error_df : pandas.DataFrame
        Per-polygon error statistics
    vis_dir : str or Path
        Output directory for map files
    shapefile_path : str, optional
        Path to polygon shapefile. Uses default FEWSNET path if None
    uid_col : str
        Column name containing polygon identifiers  
    crs_target : str
        Target CRS for maps
    missing_color : str
        Color for polygons with NaN error rates
    dpi : int
        Output image resolution
    seed : int
        Random seed for reproducible colors
        
    Returns
    -------
    tuple
        Paths to (overall_error_map, class1_error_map)
    """
    # Strict gate: do nothing when disabled
    try:
        if not bool(VIS_DEBUG_MODE):
            return None, None
    except Exception:
        return None, None

    if not GEOSPATIAL_AVAILABLE:
        print("Geospatial packages not available. Skipping map generation.")
        return None, None
    
    print("\n=== Generating Error Rate Choropleth Maps ===")
    
    # Set random seed for reproducible colors
    np.random.seed(seed)
    
    # Default shapefile path
    if shapefile_path is None:
        shapefile_path = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
    
    vis_path = Path(vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load shapefile
        print(f"  Loading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        print(f"  Shapefile polygons: {len(gdf):,}")
        
        # === COMPREHENSIVE SHAPEFILE DEBUG INFO ===
        print(f"\n=== SHAPEFILE DEBUG INFO ===")
        print(f"Shapefile path: {shapefile_path}")
        print(f"Shapefile columns: {list(gdf.columns)}")
        print(f"Looking for admin_code column...")
        
        if 'admin_code' in gdf.columns:
            print(f"SUCCESS: Found admin_code column")
            print(f"  Data type: {gdf['admin_code'].dtype}")
            print(f"  Sample values: {gdf['admin_code'].unique()[:10]}")
            print(f"  Value range: {gdf['admin_code'].min()} to {gdf['admin_code'].max()}")
            print(f"  Unique admin codes: {len(gdf['admin_code'].unique()):,}")
        else:
            print(f"ERROR: No admin_code column found!")
            print(f"  Available columns: {list(gdf.columns)}")
            # Try to find similar columns
            possible_cols = [col for col in gdf.columns if 'admin' in col.lower() or 'code' in col.lower() or 'id' in col.lower()]
            if possible_cols:
                print(f"  Possible ID columns: {possible_cols}")
            
        # === ERROR DATA DEBUG INFO ===
        print(f"\n=== ERROR DATA DEBUG INFO ===")
        print(f"Error data shape: {error_df.shape}")
        print(f"Error data columns: {list(error_df.columns)}")
        print(f"Looking for {uid_col} column...")
        
        if uid_col in error_df.columns:
            print(f"SUCCESS: Found {uid_col} column in error data")
            print(f"  Data type: {error_df[uid_col].dtype}")
            print(f"  Sample values: {error_df[uid_col].unique()[:10]}")
            print(f"  Value range: {error_df[uid_col].min()} to {error_df[uid_col].max()}")
            print(f"  Unique admin codes: {len(error_df[uid_col].unique()):,}")
        else:
            print(f"ERROR: No {uid_col} column found in error data!")
            print(f"  Available columns: {list(error_df.columns)}")
        
        # Standardize geometry column and CRS
        original_admin_col = None
        if 'admin_code' in gdf.columns:
            original_admin_col = 'admin_code'
            gdf = gdf.rename(columns={'admin_code': uid_col})
            print(f"\n  Renamed shapefile column: 'admin_code' -> '{uid_col}'")
        elif uid_col not in gdf.columns:
            # Try to find alternative column names
            possible_cols = [col for col in gdf.columns if 'admin' in col.lower() or 'code' in col.lower() or 'id' in col.lower()]
            if possible_cols:
                original_admin_col = possible_cols[0]
                gdf = gdf.rename(columns={original_admin_col: uid_col})
                print(f"\n  Using alternative column: '{original_admin_col}' -> '{uid_col}'")
            else:
                print(f"\nERROR: Cannot find suitable ID column in shapefile!")
                return None, None
        
        # Ensure both columns exist before proceeding
        if uid_col not in gdf.columns:
            print(f"ERROR: {uid_col} column missing from shapefile after processing!")
            return None, None
        
        if uid_col not in error_df.columns:
            print(f"ERROR: {uid_col} column missing from error data!")
            return None, None
        
        # Keep only necessary columns
        gdf = gdf[[uid_col, 'geometry']].copy()
        gdf = gdf.to_crs(crs_target)
        
        # === DATA TYPE HARMONIZATION ===
        print(f"\n=== DATA TYPE HARMONIZATION ===")
        print(f"Before harmonization:")
        print(f"  Shapefile {uid_col} type: {gdf[uid_col].dtype}")
        print(f"  Error data {uid_col} type: {error_df[uid_col].dtype}")
        
        # Convert to consistent types for better matching
        if gdf[uid_col].dtype != error_df[uid_col].dtype:
            print(f"  Converting data types for compatibility...")
            try:
                # Try to convert both to numeric first
                gdf[uid_col] = pd.to_numeric(gdf[uid_col], errors='ignore')
                error_df_copy = error_df.copy()
                error_df_copy[uid_col] = pd.to_numeric(error_df_copy[uid_col], errors='ignore')
                
                # If still different, convert both to string
                if gdf[uid_col].dtype != error_df_copy[uid_col].dtype:
                    gdf[uid_col] = gdf[uid_col].astype(str)
                    error_df_copy[uid_col] = error_df_copy[uid_col].astype(str)
                    print(f"  Converted both to string for compatibility")
                
                # Always update error_df with the processed copy
                error_df = error_df_copy
                print(f"  Data types after conversion: shapefile={gdf[uid_col].dtype}, error={error_df[uid_col].dtype}")
                    
            except Exception as e:
                print(f"  Warning: Data type conversion failed: {e}")
                # Fallback: convert both to string
                gdf[uid_col] = gdf[uid_col].astype(str)
                error_df = error_df.copy()
                error_df[uid_col] = error_df[uid_col].astype(str)
                print(f"  Fallback: Converted both to string")
        
        print(f"After harmonization:")
        print(f"  Shapefile {uid_col} type: {gdf[uid_col].dtype}")
        print(f"  Error data {uid_col} type: {error_df[uid_col].dtype}")
        
        # === MERGE COMPATIBILITY TEST ===
        print(f"\n=== MERGE COMPATIBILITY TEST ===")
        shapefile_codes = set(gdf[uid_col].unique())
        error_codes = set(error_df[uid_col].unique())
        
        print(f"Shapefile admin codes: {len(shapefile_codes):,} unique values")
        print(f"Error data admin codes: {len(error_codes):,} unique values")
        
        # Find overlapping codes
        common_codes = shapefile_codes & error_codes
        print(f"Common admin codes: {len(common_codes):,}")
        
        if len(error_codes) > 0:
            match_rate = len(common_codes) / len(error_codes) * 100
            print(f"Error data match rate: {match_rate:.1f}%")
        else:
            match_rate = 0
            print(f"ERROR: No admin codes in error data!")
        
        if match_rate < 10:
            print(f"WARNING: Very low match rate ({match_rate:.1f}%)!")
            print(f"Sample shapefile codes: {list(shapefile_codes)[:10]}")
            print(f"Sample error codes: {list(error_codes)[:10]}")
        
        # Merge with error data
        print(f"\n=== PERFORMING MERGE ===")
        merged_gdf = gdf.merge(error_df, on=uid_col, how='left')
        merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry')
        
        print(f"Merge results:")
        print(f"  Total polygons after merge: {len(merged_gdf):,}")
        print(f"  Polygons with error data: {merged_gdf['pct_err_all'].notna().sum():,}")
        
        if 'pct_err_all' in merged_gdf.columns:
            successful_matches = merged_gdf['pct_err_all'].notna().sum()
            success_rate = successful_matches / len(merged_gdf) * 100
            print(f"  Successful polygon matches: {success_rate:.1f}%")
        else:
            print(f"  ERROR: pct_err_all column missing after merge!")
        
        # === CREATE DIAGNOSTIC MAPS ===
        print(f"\n=== CREATING DIAGNOSTIC MAPS ===")
        map_paths = []
        
        # Only generate maps if VIS_DEBUG_MODE is enabled
        if not VIS_DEBUG_MODE:
            print("Diagnostic map generation disabled (VIS_DEBUG_MODE=False)")
            return None, None
        
        for metric, title, filename in [
            ('pct_err_all', 'Overall Misclassification Rate by Polygon (Training Data)', 'map_pct_err_all.png'),
            ('pct_err_cls1', 'Class 1 Misclassification Rate by Polygon (Training Data)', 'map_pct_err_class1.png')
        ]:
            print(f"\n--- Creating {metric} map ---")
            
            # Check if metric column exists
            if metric not in merged_gdf.columns:
                print(f"ERROR: {metric} column not found in merged data!")
                print(f"Available columns: {list(merged_gdf.columns)}")
                continue
            
            # Analyze the metric data
            metric_values = merged_gdf[metric]
            data_mask = metric_values.notna()
            valid_data = metric_values.dropna()
            
            print(f"Metric data analysis:")
            print(f"  Column: {metric}")
            print(f"  Total polygons: {len(merged_gdf):,}")
            print(f"  Polygons with valid data: {data_mask.sum():,}")
            print(f"  Polygons with missing data: {(~data_mask).sum():,}")
            
            if len(valid_data) > 0:
                print(f"  Data range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
                print(f"  Mean: {valid_data.mean():.4f}")
                print(f"  Std: {valid_data.std():.4f}")
                print(f"  Non-zero values: {(valid_data > 0).sum():,}")
                print(f"  Sample values: {valid_data.head().tolist()}")
            else:
                print(f"  ERROR: No valid data found for {metric}!")
                print(f"  All values are NaN or missing")
                continue
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot polygons with data
            if data_mask.sum() > 0:
                print(f"  Plotting {data_mask.sum():,} polygons with data...")
                
                # Check if all values are zero
                non_zero_count = (valid_data > 0).sum()
                if non_zero_count == 0:
                    print(f"  WARNING: All error rates are exactly 0.0!")
                    print(f"  This suggests either perfect predictions or data processing error")
                elif non_zero_count < len(valid_data) * 0.1:
                    print(f"  WARNING: Only {non_zero_count} polygons have non-zero error rates")
                
                # Plot polygons with valid data
                try:
                    # Plot without automatic legend
                    merged_gdf[data_mask].plot(
                        column=metric,
                        ax=ax,
                        cmap='Reds',
                        vmin=0.0,
                        vmax=1.0,
                        legend=False
                    )

                    # Manually create colorbar with PercentFormatter
                    sm = mpl.cm.ScalarMappable(
                        norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
                        cmap='Reds'
                    )
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label='Error Rate')
                    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))

                    print(f"  SUCCESS: Plotted polygons with valid data")
                except Exception as e:
                    print(f"  ERROR: Failed to plot polygons with data: {e}")
                    continue
            else:
                print(f"  No polygons with valid data to plot!")
            
            # Plot polygons without data in missing color
            no_data_mask = ~data_mask
            if no_data_mask.sum() > 0:
                merged_gdf[no_data_mask].plot(
                    ax=ax,
                    color=missing_color,
                    alpha=0.7,
                    hatch='///',
                    edgecolor='white',
                    linewidth=0.1
                )
            
            # Styling
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add summary statistics text
            valid_values = merged_gdf[metric].dropna()
            if len(valid_values) > 0:
                stats_text = (f"Valid Polygons: {len(valid_values):,}\n"
                            f"Mean: {valid_values.mean():.1%}\n"
                            f"Std: {valid_values.std():.1%}\n"
                            f"Range: [{valid_values.min():.1%}, {valid_values.max():.1%}]")
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add missing data legend if needed
            if no_data_mask.sum() > 0:
                from matplotlib.patches import Patch
                legend_elements = ax.get_legend().get_patches() if ax.get_legend() else []
                legend_elements.append(Patch(facecolor=missing_color, hatch='///', 
                                          label=f'No Training Data ({no_data_mask.sum():,} polygons)'))
                ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            
            # Save map
            map_path = vis_path / filename
            plt.savefig(map_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            map_paths.append(map_path)
            print(f"    Saved: {map_path}")
        
        return tuple(map_paths)
        
    except Exception as e:
        print(f"Error generating maps: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_pre_partition_diagnostics(df_train, y_true, y_pred, vis_dir, 
                                   shapefile_path=None, uid_col='FEWSNET_admin_code',
                                   class_positive=1, fold_split_col=None, 
                                   min_n_threshold=10, seed=42, VIS_DEBUG_MODE=True):
    """
    Main function to create comprehensive pre-partitioning diagnostics.
    
    This function orchestrates the complete diagnostic workflow:
    1. Computes per-polygon error rates from training data
    2. Saves diagnostic CSV with quality flags
    3. Generates choropleth maps showing spatial error patterns
    4. Provides summary statistics and debug guidance
    
    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataframe with polygon identifiers and features
    y_true : array-like
        True labels for training samples
    y_pred : array-like
        Predicted labels from baseline model
    vis_dir : str or Path
        Output directory for diagnostic files
    shapefile_path : str, optional
        Path to polygon shapefile for mapping
    uid_col : str, default='FEWSNET_admin_code'
        Column containing polygon unique identifiers
    class_positive : int, default=1
        Label value for positive class (crisis)
    fold_split_col : str, optional
        Column indicating train/valid/test split
    min_n_threshold : int, default=10
        Minimum samples per polygon to avoid warnings
    seed : int, default=42
        Random seed for reproducible outputs
        
    Returns
    -------
    dict
        Dictionary containing paths to generated artifacts:
        - 'csv': Path to diagnostic CSV
        - 'map_overall': Path to overall error map
        - 'map_class1': Path to class 1 error map
        - 'summary': Summary text
    """
    print("="*60)
    print("PRE-PARTITIONING DIAGNOSTIC SYSTEM")
    print("="*60)
    
    vis_path = Path(vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)
    
    
    # Step 1: Compute error rates
    error_df = compute_training_error_rates(
        df_train, y_true, y_pred, uid_col=uid_col, 
        class_positive=class_positive, fold_split_col=fold_split_col
    )
    
    # Step 2: Save diagnostic CSV
    csv_path = save_diagnostic_csv(error_df, vis_dir, uid_col=uid_col, 
                                  min_n_threshold=min_n_threshold)
    
    # Step 3: Generate maps
    map_overall, map_class1 = create_error_choropleth_maps(
        error_df, vis_dir, shapefile_path=shapefile_path, uid_col=uid_col, seed=seed, VIS_DEBUG_MODE=VIS_DEBUG_MODE
    )
    
    # Step 4: Generate summary
    summary_lines = []
    summary_lines.append("DIAGNOSTIC SUMMARY")
    summary_lines.append("-" * 50)
    
    # Global error statistics
    valid_overall = error_df['pct_err_all'].dropna()
    valid_class1 = error_df['pct_err_cls1'].dropna()
    
    if len(valid_overall) > 0:
        summary_lines.append(f"Overall Error Rate: {valid_overall.mean():.1%} (+-{valid_overall.std():.1%})")
        summary_lines.append(f"Error Range: [{valid_overall.min():.1%}, {valid_overall.max():.1%}]")
        
        # Identify hotspots (top 10% error rate)
        hotspot_threshold = valid_overall.quantile(0.9)
        hotspots = error_df[error_df['pct_err_all'] >= hotspot_threshold]
        summary_lines.append(f"Error Hotspots (>={hotspot_threshold:.1%}): {len(hotspots)} polygons")
    
    if len(valid_class1) > 0:
        summary_lines.append(f"Class 1 Error Rate: {valid_class1.mean():.1%} (+-{valid_class1.std():.1%})")
        
    # NaN polygon statistics
    nan_overall = error_df['pct_err_all'].isna().sum()
    nan_class1 = error_df['pct_err_cls1'].isna().sum()
    total_polygons = len(error_df)
    
    summary_lines.append(f"Polygons with missing overall error data: {nan_overall}/{total_polygons} ({nan_overall/total_polygons:.1%})")
    summary_lines.append(f"Polygons with missing class 1 error data: {nan_class1}/{total_polygons} ({nan_class1/total_polygons:.1%})")
    
    # Debug guidance
    summary_lines.append("")
    summary_lines.append("DEBUG GUIDANCE:")
    summary_lines.append("- Compare these error patterns against actual partition boundaries")
    summary_lines.append("- Error clusters should correlate with partitioning if optimization aligns with error reduction")
    summary_lines.append("- Unexpected spatial patterns may indicate contiguity issues or objective misalignment")
    summary_lines.append("- High NaN percentage suggests data coverage gaps requiring attention")
    
    summary_text = "\n".join(summary_lines)
    
    # Save summary to file
    summary_path = vis_path / "diagnostic_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    
    # Artifact paths
    artifacts = {
        'csv': csv_path,
        'map_overall': map_overall,
        'map_class1': map_class1,
        'summary': summary_text,
        'summary_file': summary_path
    }
    
    print("\n" + "="*60)
    print("DIAGNOSTIC ARTIFACTS GENERATED")
    print("="*60)
    print(f"CSV: {csv_path}")
    if map_overall:
        print(f"Overall Error Map: {map_overall}")
    if map_class1:
        print(f"Class 1 Error Map: {map_class1}")
    print(f"Summary: {summary_path}")
    
    return artifacts


def create_diagnostic_summary(error_df, vis_dir):
    """
    Create diagnostic summary text file.
    
    Args:
        error_df: Error statistics dataframe
        vis_dir: Output directory
        
    Returns:
        Path to summary file
    """
    # Generate summary
    summary_lines = []
    summary_lines.append("DIAGNOSTIC SUMMARY")
    summary_lines.append("-" * 50)
    
    # Global error statistics
    valid_overall = error_df['pct_err_all'].dropna()
    valid_class1 = error_df['pct_err_cls1'].dropna()
    
    if len(valid_overall) > 0:
        summary_lines.append(f"Overall Error Rate: {valid_overall.mean():.1%} (+-{valid_overall.std():.1%})")
        summary_lines.append(f"Error Range: [{valid_overall.min():.1%}, {valid_overall.max():.1%}]")
        
        # Identify hotspots (top 10% error rate)
        hotspot_threshold = valid_overall.quantile(0.9)
        hotspots = error_df[error_df['pct_err_all'] >= hotspot_threshold]
        summary_lines.append(f"Error Hotspots (>={hotspot_threshold:.1%}): {len(hotspots)} polygons")
    
    if len(valid_class1) > 0:
        summary_lines.append(f"Class 1 Error Rate: {valid_class1.mean():.1%} (+-{valid_class1.std():.1%})")
        
    # NaN polygon statistics
    nan_overall = error_df['pct_err_all'].isna().sum()
    nan_class1 = error_df['pct_err_cls1'].isna().sum()
    total_polygons = len(error_df)
    
    summary_lines.append(f"Polygons with missing overall error data: {nan_overall}/{total_polygons} ({nan_overall/total_polygons:.1%})")
    summary_lines.append(f"Polygons with missing class 1 error data: {nan_class1}/{total_polygons} ({nan_class1/total_polygons:.1%})")
    
    # Guidance
    summary_lines.append("")
    summary_lines.append("DEBUG GUIDANCE:")
    summary_lines.append("- Compare these error patterns against actual partition boundaries")
    summary_lines.append("- Error clusters should correlate with partitioning if optimization aligns with error reduction")
    summary_lines.append("- Unexpected spatial patterns may indicate contiguity issues or objective misalignment")
    summary_lines.append("- High NaN percentage suggests data coverage gaps requiring attention")
    
    # Save summary
    summary_path = Path(vis_dir) / "diagnostic_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        for line in summary_lines:
            f.write(line + '\n')
    
    return str(summary_path)


def stratified_group_kfold(X, y, X_group, n_splits=5, random_state=42):
    """
    Create stratified k-fold splits that keep polygons together.
    
    This ensures that all samples from the same polygon stay in the same fold,
    preventing data leakage while maintaining class stratification.
    
    Args:
        X: Feature matrix (N, features)
        y: Labels (N,)
        X_group: Polygon group IDs (N,)
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        Generator of (train_indices, val_indices) tuples
    """
    
    # Create polygon-level summary for stratification
    polygon_df = pd.DataFrame({
        'polygon_id': X_group,
        'y': y
    })
    
    # Aggregate by polygon: majority class per polygon
    polygon_summary = polygon_df.groupby('polygon_id').agg({
        'y': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]  # Majority class
    }).reset_index()
    
    unique_polygons = polygon_summary['polygon_id'].values
    polygon_classes = polygon_summary['y'].values
    
    # Stratified split at polygon level
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_poly_idx, val_poly_idx in skf.split(unique_polygons, polygon_classes):
        train_polygons = set(unique_polygons[train_poly_idx])
        val_polygons = set(unique_polygons[val_poly_idx])
        
        # Map back to sample indices
        train_sample_mask = np.isin(X_group, list(train_polygons))
        val_sample_mask = np.isin(X_group, list(val_polygons))
        
        train_indices = np.where(train_sample_mask)[0]
        val_indices = np.where(val_sample_mask)[0]
        
        yield train_indices, val_indices


def compute_cv_predictions(X_train, y_train, X_group_train, model_class, model_params, 
                          cv_folds=5, random_state=42):
    """
    Generate cross-validated predictions on training data to avoid overfitting bias.
    
    Args:
        X_train: Training features (test data completely excluded)
        y_train: Training labels (test data completely excluded)  
        X_group_train: Training polygon IDs (test data completely excluded)
        model_class: Model class to instantiate (e.g., RFmodel)
        model_params: Parameters for model initialization
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        y_pred_cv: Cross-validated predictions for all training samples
    """
    print(f"\n=== Cross-Validation Prediction Generation ===")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Unique polygons: {len(np.unique(X_group_train)):,}")
    
    # Verify no test data contamination
    assert len(X_train) == len(y_train) == len(X_group_train), "Training data size mismatch"
    
    # Initialize predictions array
    y_pred_cv = np.full(len(y_train), -1, dtype=int)  # -1 indicates unassigned
    
    fold_num = 0
    for train_idx, val_idx in stratified_group_kfold(X_train, y_train, X_group_train, 
                                                    n_splits=cv_folds, random_state=random_state):
        fold_num += 1
        print(f"  Fold {fold_num}: Train={len(train_idx):,}, Val={len(val_idx):,}")
        
        # Create and train model for this fold
        try:
            # Handle different model initialization patterns
            if hasattr(model_class, '__name__') and 'RF' in model_class.__name__:
                # Random Forest model (from model_RF.py)
                fold_model = model_class(
                    model_params.get('dir_ckpt', './temp_cv'),
                    model_params.get('n_trees_unit', 100),
                    max_depth=model_params.get('max_depth', None)
                )
                fold_model.train(X_train[train_idx], y_train[train_idx], branch_id=f'cv_fold_{fold_num}')
                fold_predictions = fold_model.predict(X_train[val_idx])
                
            else:
                # Generic sklearn-style model
                fold_model = model_class(**model_params)
                fold_model.fit(X_train[train_idx], y_train[train_idx])
                fold_predictions = fold_model.predict(X_train[val_idx])
                
        except Exception as e:
            print(f"    ERROR in fold {fold_num}: {e}")
            # Fallback: random predictions maintaining class distribution
            class_dist = np.bincount(y_train[train_idx]) / len(y_train[train_idx])
            fold_predictions = np.random.choice(
                len(class_dist), size=len(val_idx), p=class_dist
            )
            
        # Store fold predictions
        y_pred_cv[val_idx] = fold_predictions.ravel()
    
    # Verify all samples got predictions
    unassigned = np.sum(y_pred_cv == -1)
    if unassigned > 0:
        print(f"  WARNING: {unassigned} samples without CV predictions")
        # Assign random predictions to unassigned samples
        class_dist = np.bincount(y_train) / len(y_train)
        unassigned_mask = (y_pred_cv == -1)
        y_pred_cv[unassigned_mask] = np.random.choice(
            len(class_dist), size=unassigned, p=class_dist
        )
    
    print(f"  CV predictions complete: {len(y_pred_cv):,} samples")
    return y_pred_cv


def create_pre_partition_diagnostics_cv(X_train, y_train, X_group_train, 
                                       model_class, model_params,
                                       vis_dir="./vis",
                                       shapefile_path=None,
                                       uid_col='FEWSNET_admin_code',
                                       class_positive=1,
                                       min_n_threshold=10,
                                       cv_folds=5,
                                       random_state=42,
                                       seed=None, VIS_DEBUG_MODE=True):
    """
    Generate pre-partitioning diagnostics using cross-validation to prevent overfitting bias.
    
    This function performs k-fold cross-validation on TRAINING DATA ONLY (test data must be 
    completely excluded) to generate realistic error rate estimates for diagnostic visualization.
    
    Args:
        X_train: Training features only (N_train, features) - NO TEST DATA
        y_train: Training labels only (N_train,) - NO TEST DATA  
        X_group_train: Training polygon IDs only (N_train,) - NO TEST DATA
        model_class: Model class to instantiate for CV (e.g., RFmodel, sklearn estimator)
        model_params: Dictionary of model initialization parameters
        vis_dir: Output directory for diagnostic artifacts
        shapefile_path: Path to polygon shapefile for choropleth maps
        uid_col: Polygon ID column name
        class_positive: Positive class label for class-1 specific metrics
        min_n_threshold: Minimum samples per polygon for quality warnings
        cv_folds: Number of cross-validation folds
        random_state: Random seed for CV splits
        seed: Additional random seed (for backward compatibility)
        
    Returns:
        Dictionary of diagnostic artifacts and results
    """
    
    print("\n" + "="*60)
    print("PRE-PARTITIONING DIAGNOSTIC SYSTEM (CROSS-VALIDATION)")
    print("="*60)
    
    # Data validation and test contamination check
    print(f"\n=== Data Validation ===")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Training labels: {len(y_train):,}")
    print(f"  Training polygon IDs: {len(X_group_train):,}")
    print(f"  Unique polygons: {len(np.unique(X_group_train)):,}")
    
    # Ensure consistent sizes
    assert len(X_train) == len(y_train) == len(X_group_train), \
        "Training data size mismatch - potential data leakage!"
    
    # Set random seed
    final_seed = seed if seed is not None else random_state
    np.random.seed(final_seed)
    
    print(f"  Random seed: {final_seed}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Model class: {model_class.__name__ if hasattr(model_class, '__name__') else str(model_class)}")
    
    # Generate cross-validated predictions
    y_pred_cv = compute_cv_predictions(
        X_train, y_train, X_group_train,
        model_class=model_class,
        model_params=model_params,
        cv_folds=cv_folds,
        random_state=final_seed
    )
    
    # Create training dataframe for error computation
    df_train_cv = pd.DataFrame({
        uid_col: X_group_train,
        'y_true': y_train,
        'y_pred': y_pred_cv
    })
    
    # Compute error rates using CV predictions
    error_df = compute_training_error_rates(
        df_train_cv, df_train_cv['y_true'].values, df_train_cv['y_pred'].values, 
        uid_col=uid_col, class_positive=class_positive
    )
    
    # Save diagnostic CSV
    csv_path = save_diagnostic_csv(error_df, vis_dir, uid_col=uid_col, 
                                  min_n_threshold=min_n_threshold)
    
    # Generate choropleth maps
    map_overall = None
    map_class1 = None
    
    if shapefile_path and GEOSPATIAL_AVAILABLE:
        try:
            map_overall, map_class1 = create_error_choropleth_maps(
                error_df, vis_dir, shapefile_path, uid_col=uid_col, seed=final_seed, VIS_DEBUG_MODE=VIS_DEBUG_MODE
            )
        except Exception as e:
            print(f"Error generating maps: {e}")
    
    # Generate summary
    summary_path = create_diagnostic_summary(error_df, vis_dir)
    
    # Compile results
    artifacts = {
        'csv_path': csv_path,
        'map_overall': map_overall,
        'map_class1': map_class1, 
        'summary_path': summary_path,
        'error_df': error_df,
        'cv_predictions': y_pred_cv
    }
    
    print(f"\n{'='*60}")
    print("CV DIAGNOSTIC ARTIFACTS GENERATED")
    print("="*60)
    print(f"CSV: {csv_path}")
    if map_overall:
        print(f"Overall Error Map: {map_overall}")
    if map_class1:
        print(f"Class 1 Error Map: {map_class1}")
    print(f"Summary: {summary_path}")
    
    return artifacts


if __name__ == "__main__":
    # Example usage and testing
    print("Pre-Partitioning Diagnostic System")
    print("Run create_pre_partition_diagnostics() with your data to generate diagnostic maps.")
