#!/usr/bin/env python3
"""
Error Rate Choropleth Grid Generator
Loads prediction CSVs, computes polygon-level error rates, and generates
yearly and seasonal choropleth grids (4x3 yearly, 3x3 seasonal).
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import numpy as np

try:
    import contextily as ctx
except ImportError:
    ctx = None

warnings.filterwarnings('ignore', category=FutureWarning)


def map_month_to_season(month: int) -> str:
    """Map month to season (DJFM, AMJJ, ASON). Dec-Mar -> DJFM, Apr-Jul -> AMJJ, Aug-Nov -> ASON."""
    if month in [12, 1, 2, 3]:
        return "DJFM"
    elif month in [4, 5, 6, 7]:
        return "AMJJ"
    elif month in [8, 9, 10, 11]:
        return "ASON"


def adjust_year_for_djf(row: pd.Series) -> int:
    """For DJF season, December uses next year's label."""
    if row['season'] == 'DJFM' and row['month'] == 12:
        return row['year'] + 1
    return row['year']


def load_predictions(
    pred_dir: Path,
    file_glob: str,
    scope_regex: str,
    uid_col: str,
    date_col: str,
    y_true_col: str,
    y_pred_col: str,
    start_date: str,
    end_date: str,
    verbose: bool
) -> pd.DataFrame:
    """Load and combine all prediction CSVs, extract scope, filter by date range."""

    csv_files = sorted(pred_dir.glob(file_glob))
    if not csv_files:
        # Show available files to help debug
        all_csv = list(pred_dir.glob('*.csv'))
        if all_csv and verbose:
            print(f"No files matching '{file_glob}' in {pred_dir}")
            print(f"Available CSV files: {', '.join([f.name for f in all_csv[:5]])}")
        raise FileNotFoundError(f"No files matching '{file_glob}' in {pred_dir}")

    if verbose:
        print(f"Found {len(csv_files)} prediction files:")
        for f in csv_files[:5]:
            print(f"  - {f.name}")
        if len(csv_files) > 5:
            print(f"  ... and {len(csv_files) - 5} more")

    pattern = re.compile(scope_regex)
    frames = []

    for fpath in csv_files:
        match = pattern.search(fpath.name)
        if not match:
            if verbose:
                print(f"  Skipping {fpath.name} (no scope match)")
            continue

        scope = f"fs{match.group('scope')}"

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            if verbose:
                print(f"  ERROR reading {fpath.name}: {e}")
            continue

        # Find uid column - try multiple common variations
        uid_variations = [uid_col, 'uid', 'admin_code', 'adm_code', 'FEWSNET_admin_code']
        found_uid = None
        for var in uid_variations:
            if var in df.columns:
                found_uid = var
                break

        if found_uid is None:
            if verbose:
                print(f"  ERROR in {fpath.name}: no uid column found (tried {uid_variations})")
            continue

        # Rename to standard uid_col
        if found_uid != uid_col:
            df = df.rename(columns={found_uid: uid_col})

        # Handle date column - construct from year/month if needed
        if date_col not in df.columns:
            if 'year' in df.columns and 'month' in df.columns:
                # Construct date from year/month (use day=1)
                df[date_col] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                if verbose and fpath == csv_files[0]:  # Only print once
                    print(f"  Constructing '{date_col}' from year/month columns")
            else:
                if verbose:
                    print(f"  ERROR in {fpath.name}: no '{date_col}' or year/month columns")
                continue

        # Check required columns
        required = [uid_col, date_col, y_true_col, y_pred_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            if verbose:
                print(f"  ERROR in {fpath.name}: missing columns {missing}")
            continue

        df = df[[uid_col, date_col, y_true_col, y_pred_col]].copy()
        df['scope'] = scope
        frames.append(df)

        if verbose and len(frames) <= 3:  # Print first few
            print(f"  Loaded {fpath.name} -> {scope}: {len(df)} rows")

    if verbose and len(frames) > 3:
        print(f"  ... loaded {len(frames)} files total")

    if not frames:
        raise ValueError("No valid prediction data loaded")

    combined = pd.concat(frames, ignore_index=True)

    # Parse dates (may already be datetime if constructed above)
    if not pd.api.types.is_datetime64_any_dtype(combined[date_col]):
        combined[date_col] = pd.to_datetime(combined[date_col], errors='coerce')
        invalid_dates = combined[date_col].isna().sum()
        if invalid_dates > 0:
            if verbose:
                print(f"WARNING: {invalid_dates} rows with invalid dates, dropping")
            combined = combined.dropna(subset=[date_col])

    # Filter by date range (optional)
    if start_date is not None:
        start = pd.to_datetime(start_date)
        original_len = len(combined)
        combined = combined[combined[date_col] >= start].copy()
        if verbose:
            print(f"Applied start date filter (>= {start_date}): {original_len} -> {len(combined)} rows")
        if len(combined) == 0:
            raise ValueError(f"No data after filtering for dates >= {start_date}")

    if end_date is not None:
        end = pd.to_datetime(end_date)
        original_len = len(combined)
        combined = combined[combined[date_col] <= end].copy()
        if verbose:
            print(f"Applied end date filter (<= {end_date}): {original_len} -> {len(combined)} rows")
        if len(combined) == 0:
            raise ValueError(f"No data after filtering for dates <= {end_date}")

    # Derive year, month, season
    combined['year'] = combined[date_col].dt.year
    combined['month'] = combined[date_col].dt.month
    combined['season'] = combined['month'].apply(map_month_to_season)

    # Adjust year for December in DJF
    combined['year'] = combined.apply(adjust_year_for_djf, axis=1)

    if verbose:
        print(f"\nCombined: {len(combined)} rows from {combined['year'].min()} to {combined['year'].max()}")
        print(f"Date range: {combined[date_col].min()} to {combined[date_col].max()}")
        print(f"Scopes: {sorted(combined['scope'].unique())}")
        if start_date or end_date:
            print(f"[OK] Date filtering applied: {'start=' + str(start_date) if start_date else ''} {'end=' + str(end_date) if end_date else ''}")

    return combined


def apply_truth_filter(
    df: pd.DataFrame,
    filter_type: str,
    y_true_col: str,
    verbose: bool
) -> tuple[pd.DataFrame, str]:
    """
    Filter dataframe based on ground truth label to analyze errors on crisis vs non-crisis cases.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined prediction dataframe
    filter_type : str
        Filter type: 'all' (no filter), 'crisis' (y_true==1), 'noncrisis' (y_true==0)
    y_true_col : str
        Name of ground truth column
    verbose : bool
        Print filtering information

    Returns:
    --------
    tuple[pd.DataFrame, str]
        Filtered dataframe and filename suffix (e.g., '_crisis', '_noncrisis', '')
    """

    original_len = len(df)

    if filter_type == 'all':
        # No filtering
        filtered_df = df.copy()
        suffix = ''
        filter_desc = 'No filter (all data)'
    elif filter_type == 'crisis':
        # Only actual crisis cases
        filtered_df = df[df[y_true_col] == 1].copy()
        suffix = '_crisis'
        filter_desc = 'Actual crisis cases only (fews_ipc_crisis_true==1)'
    elif filter_type == 'noncrisis':
        # Only actual non-crisis cases
        filtered_df = df[df[y_true_col] == 0].copy()
        suffix = '_noncrisis'
        filter_desc = 'Actual non-crisis cases only (fews_ipc_crisis_true==0)'
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    if verbose:
        print(f"\nApplying filter: {filter_desc}")
        print(f"  Original rows: {original_len}")
        print(f"  Filtered rows: {len(filtered_df)}")
        print(f"  Retained: {len(filtered_df)/original_len*100:.1f}%")

    if len(filtered_df) == 0:
        raise ValueError(f"No data remaining after applying filter '{filter_type}'")

    return filtered_df, suffix


def _confusion_for_binary(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> Dict[str, int]:
    """
    Compute confusion matrix for binary classification.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    positive_label : int
        Label for positive class (default: 1)

    Returns:
    --------
    dict : {'tp': int, 'fp': int, 'tn': int, 'fn': int, 'support': int}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Positive class
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))

    support = len(y_true)

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'support': support}


def _safe_prf(tp: int, fp: int, fn: int) -> tuple:
    """
    Compute precision, recall, F1 with zero-division guards.

    Parameters:
    -----------
    tp, fp, fn : int
        Confusion matrix counts

    Returns:
    --------
    tuple : (precision, recall, f1) where undefined values are np.nan
    """
    # Precision = TP / (TP + FP)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan

    # Recall = TP / (TP + FN)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = np.nan

    # F1 = 2 * P * R / (P + R)
    if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan

    return precision, recall, f1


def _get_builtin_region_map() -> Dict[str, str]:
    """
    Returns builtin ADMIN0 -> region mapping for FEWS NET countries.

    Regions: East Africa, West Africa, Southern Africa, North Africa,
             Middle East, South Asia, Southeast Asia, Latin America

    Returns:
    --------
    dict : Mapping from country name to region
    """
    return {
        # East Africa
        'Burundi': 'East Africa',
        'Djibouti': 'East Africa',
        'Eritrea': 'East Africa',
        'Ethiopia': 'East Africa',
        'Kenya': 'East Africa',
        'Rwanda': 'East Africa',
        'Somalia': 'East Africa',
        'South Sudan': 'East Africa',
        'Sudan': 'East Africa',
        'Tanzania': 'East Africa',
        'Uganda': 'East Africa',

        # West Africa
        'Burkina Faso': 'West Africa',
        'Chad': 'West Africa',
        'Gambia': 'West Africa',
        'Ghana': 'West Africa',
        'Guinea': 'West Africa',
        'Liberia': 'West Africa',
        'Mali': 'West Africa',
        'Mauritania': 'West Africa',
        'Niger': 'West Africa',
        'Nigeria': 'West Africa',
        'Senegal': 'West Africa',
        'Sierra Leone': 'West Africa',
        'Togo': 'West Africa',
        'Benin': 'West Africa',
        "Cote d'Ivoire": 'West Africa',
        'Ivory Coast': 'West Africa',

        # Southern Africa
        'Angola': 'Southern Africa',
        'Botswana': 'Southern Africa',
        'Lesotho': 'Southern Africa',
        'Madagascar': 'Southern Africa',
        'Malawi': 'Southern Africa',
        'Mozambique': 'Southern Africa',
        'Namibia': 'Southern Africa',
        'South Africa': 'Southern Africa',
        'Swaziland': 'Southern Africa',
        'Eswatini': 'Southern Africa',
        'Zambia': 'Southern Africa',
        'Zimbabwe': 'Southern Africa',

        # North Africa
        'Egypt': 'North Africa',
        'Libya': 'North Africa',
        'Morocco': 'North Africa',
        'Tunisia': 'North Africa',
        'Algeria': 'North Africa',

        # Middle East
        'Afghanistan': 'Middle East',
        'Iraq': 'Middle East',
        'Palestine': 'Middle East',
        'Syria': 'Middle East',
        'Yemen': 'Middle East',
        'Jordan': 'Middle East',
        'Lebanon': 'Middle East',

        # South Asia
        'Bangladesh': 'South Asia',
        'Nepal': 'South Asia',
        'Pakistan': 'South Asia',
        'India': 'South Asia',
        'Sri Lanka': 'South Asia',

        # Southeast Asia
        'Myanmar': 'Southeast Asia',
        'Philippines': 'Southeast Asia',
        'Cambodia': 'Southeast Asia',
        'Laos': 'Southeast Asia',
        'Thailand': 'Southeast Asia',
        'Vietnam': 'Southeast Asia',
        'Indonesia': 'Southeast Asia',

        # Latin America
        'El Salvador': 'Latin America',
        'Guatemala': 'Latin America',
        'Haiti': 'Latin America',
        'Honduras': 'Latin America',
        'Nicaragua': 'Latin America',
        'Colombia': 'Latin America',
        'Ecuador': 'Latin America',
        'Peru': 'Latin America',
        'Bolivia': 'Latin America',
        'Venezuela': 'Latin America',
    }


def compute_admin0_and_region_metrics(
    df: pd.DataFrame,
    polys: gpd.GeoDataFrame,
    scopes: List[str],
    uid_col: str,
    y_true_col: str,
    y_pred_col: str,
    admin0_col: str = 'ADMIN0',
    positive_label: int = 1,
    region_map_path: Path = None,
    out_dir: Path = Path('.'),
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 metrics by country (ADMIN0) and region for each scope.

    Parameters:
    -----------
    df : pd.DataFrame
        Prediction dataframe with columns [uid_col, y_true_col, y_pred_col, scope]
    polys : gpd.GeoDataFrame
        Polygons with ADMIN0 column
    scopes : List[str]
        List of scopes to process (e.g., ['fs1', 'fs2', 'fs3'])
    uid_col : str
        UID column name for joining
    y_true_col : str
        Ground truth column
    y_pred_col : str
        Prediction column
    admin0_col : str
        Country column name in polys (default: 'ADMIN0')
    positive_label : int
        Positive class label (default: 1)
    region_map_path : Path, optional
        CSV file with [ADMIN0, region] columns
    out_dir : Path
        Output directory for CSVs
    verbose : bool
        Verbose output

    Returns:
    --------
    dict : {'files': list of written file paths, 'admin0_list': list of countries}
    """

    # 1. Merge df with polys to get ADMIN0
    if verbose:
        print(f"\nMerging predictions with polygons to get {admin0_col}...")

    # Find matching uid column in polys
    uid_variations = [uid_col, 'uid', 'admin_code', 'adm_code', 'FEWSNET_admin_code']
    found_uid = None
    for var in uid_variations:
        if var in polys.columns:
            found_uid = var
            break

    if found_uid is None:
        raise ValueError(f"Could not find uid column in polygons. Tried: {uid_variations}")

    if found_uid != uid_col:
        polys = polys.rename(columns={found_uid: uid_col})
        if verbose:
            print(f"  Renamed polygon column '{found_uid}' -> '{uid_col}'")

    # Check if ADMIN0 exists
    if admin0_col not in polys.columns:
        raise ValueError(f"Column '{admin0_col}' not found in polygons. Available: {list(polys.columns)}")

    # Merge - keep only necessary columns from polys
    polys_subset = polys[[uid_col, admin0_col]].copy()
    merged = df.merge(polys_subset, on=uid_col, how='left')

    if verbose:
        print(f"  Merged: {len(merged)} rows")
        null_admin0 = merged[admin0_col].isna().sum()
        if null_admin0 > 0:
            print(f"  WARNING: {null_admin0} rows ({null_admin0/len(merged)*100:.1f}%) have null {admin0_col}")

    # Drop rows with null ADMIN0
    merged = merged[merged[admin0_col].notna()].copy()

    # 2. Print sorted unique ADMIN0 values
    admin0_list = sorted(merged[admin0_col].unique())
    print(f"\n{admin0_col} countries detected ({len(admin0_list)}):")
    print(", ".join(admin0_list))

    # 3. Load region mapping
    if region_map_path and region_map_path.exists():
        if verbose:
            print(f"\nLoading region mapping from: {region_map_path}")
        region_map_df = pd.read_csv(region_map_path)
        if 'ADMIN0' not in region_map_df.columns or 'region' not in region_map_df.columns:
            raise ValueError(f"Region map CSV must have columns 'ADMIN0' and 'region'")
        region_map = dict(zip(region_map_df['ADMIN0'], region_map_df['region']))
        print(f"  Region mapping source: CSV ({len(region_map)} entries)")
    else:
        region_map = _get_builtin_region_map()
        print(f"  Region mapping source: builtin ({len(region_map)} entries)")

    # Add region column
    merged['region'] = merged[admin0_col].map(region_map)
    # Unmapped countries -> "Other"
    unmapped = merged['region'].isna().sum()
    if unmapped > 0:
        unmapped_countries = merged[merged['region'].isna()][admin0_col].unique()
        if verbose:
            print(f"  {unmapped} rows from unmapped countries -> 'Other': {list(unmapped_countries)}")
        merged.loc[merged['region'].isna(), 'region'] = 'Other'

    # 4. Process each scope
    out_dir.mkdir(parents=True, exist_ok=True)
    files_written = []

    for scope in scopes:
        if scope not in merged['scope'].values:
            if verbose:
                print(f"\n  WARNING: Scope '{scope}' not found in data, skipping")
            continue

        scope_df = merged[merged['scope'] == scope].copy()

        if len(scope_df) == 0:
            if verbose:
                print(f"\n  WARNING: No data for scope '{scope}', skipping")
            continue

        if verbose:
            print(f"\nProcessing scope: {scope} ({len(scope_df)} rows)")

        # --- COUNTRY LEVEL ---
        country_metrics = []

        for country in sorted(scope_df[admin0_col].unique()):
            country_data = scope_df[scope_df[admin0_col] == country]

            # Compute confusion matrix
            conf = _confusion_for_binary(
                country_data[y_true_col].values,
                country_data[y_pred_col].values,
                positive_label=positive_label
            )

            # Compute metrics
            precision, recall, f1 = _safe_prf(conf['tp'], conf['fp'], conf['fn'])

            country_metrics.append({
                'scope': scope,
                admin0_col: country,
                'support': conf['support'],
                'tp': conf['tp'],
                'fp': conf['fp'],
                'fn': conf['fn'],
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        # Convert to DataFrame and save
        country_df = pd.DataFrame(country_metrics)
        country_df = country_df.sort_values(['scope', admin0_col]).reset_index(drop=True)

        country_file = out_dir / f'country_metrics_{scope}.csv'
        country_df.to_csv(country_file, index=False)
        files_written.append(str(country_file))

        if verbose:
            print(f"  Wrote {country_file.name} ({len(country_df)} countries)")

        # --- REGION LEVEL ---
        region_metrics = []

        for region in sorted(scope_df['region'].unique()):
            region_data = scope_df[scope_df['region'] == region]

            # Compute confusion matrix
            conf = _confusion_for_binary(
                region_data[y_true_col].values,
                region_data[y_pred_col].values,
                positive_label=positive_label
            )

            # Compute metrics from aggregated counts (micro-averaging)
            precision, recall, f1 = _safe_prf(conf['tp'], conf['fp'], conf['fn'])

            region_metrics.append({
                'scope': scope,
                'region': region,
                'support': conf['support'],
                'tp': conf['tp'],
                'fp': conf['fp'],
                'fn': conf['fn'],
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        # Convert to DataFrame and save
        region_df = pd.DataFrame(region_metrics)
        region_df = region_df.sort_values(['scope', 'region']).reset_index(drop=True)

        region_file = out_dir / f'region_metrics_{scope}.csv'
        region_df.to_csv(region_file, index=False)
        files_written.append(str(region_file))

        if verbose:
            print(f"  Wrote {region_file.name} ({len(region_df)} regions)")

    return {
        'files': files_written,
        'admin0_list': admin0_list,
        'scopes_processed': scopes
    }


def aggregate_error_rate(
    df: pd.DataFrame,
    by: List[str],
    uid_col: str,
    y_true_col: str,
    y_pred_col: str,
    verbose: bool
) -> pd.DataFrame:
    """
    Aggregate error rate by grouping columns.
    Returns DataFrame with columns: by + [uid_col, 'err_rate', 'n']
    """

    df = df.copy()
    df['error'] = (df[y_pred_col] != df[y_true_col]).astype(int)

    group_cols = by + [uid_col]

    agg = df.groupby(group_cols, as_index=False).agg(
        n=('error', 'count'),
        err_rate=('error', 'mean')
    )

    # Clamp to [0, 1]
    agg['err_rate'] = agg['err_rate'].clip(0.0, 1.0)

    # Filter out empty groups
    agg = agg[agg['n'] > 0].copy()

    if verbose:
        print(f"\nAggregated by {by}:")
        print(f"  Total groups: {len(agg)}")
        print(f"  Error rate range: [{agg['err_rate'].min():.3f}, {agg['err_rate'].max():.3f}]")

    return agg


def prepare_polys(
    polygons_path: Path,
    uid_col: str,
    verbose: bool
) -> gpd.GeoDataFrame:
    """Load polygons, ensure valid geometries and CRS."""

    polys = gpd.read_file(polygons_path)

    if verbose:
        print(f"\nLoaded polygons from {polygons_path.name}: {len(polys)} features")

    # Find uid column with variations
    uid_variations = [uid_col, 'uid', 'admin_code', 'adm_code', 'FEWSNET_admin_code']
    found_uid = None
    for var in uid_variations:
        if var in polys.columns:
            found_uid = var
            break

    if found_uid is None:
        raise ValueError(f"Polygon uid column not found. Tried: {uid_variations}")

    if found_uid != uid_col:
        polys = polys.rename(columns={found_uid: uid_col})
        if verbose:
            print(f"  Renamed polygon column '{found_uid}' -> '{uid_col}'")

    # Ensure valid geometries
    invalid = (~polys.geometry.is_valid).sum()
    if invalid > 0:
        if verbose:
            print(f"  Fixing {invalid} invalid geometries")
        polys['geometry'] = polys.geometry.buffer(0)

    # Set CRS if missing
    if polys.crs is None:
        polys = polys.set_crs('EPSG:4326', allow_override=True)
        if verbose:
            print(f"  Set CRS to EPSG:4326")

    return polys


def render_grid(
    agg_df: pd.DataFrame,
    polys: gpd.GeoDataFrame,
    uid_col: str,
    row_values: List[Any],
    col_values: List[str],
    row_label: str,
    col_label: str,
    value_col: str,
    vmin: float,
    vmax: float,
    missing_color: str,
    dpi: int,
    cmap_name: str,
    output_path: Path,
    add_basemap: bool,
    verbose: bool,
    filter_label: str = ''
) -> Dict[str, int]:
    """
    Render grid of choropleth maps (4x3 yearly, 3x3 seasonal).
    Returns dict of subplot counts for reporting.

    Parameters:
    -----------
    filter_label : str, optional
        Label to append to subplot titles indicating filter applied (e.g., ' (Crisis only)')
    """

    n_rows = len(row_values)
    n_cols = len(col_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Setup colormap
    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = BoundaryNorm(
        boundaries=np.linspace(vmin, vmax, 11),  # 0%, 10%, ..., 100%
        ncolors=cmap.N,
        clip=True
    )

    # Reproject for basemap if needed
    if add_basemap and ctx is not None:
        polys_plot = polys.to_crs(epsg=3857)
    else:
        polys_plot = polys

    subplot_counts = {}

    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axes[i, j]

            # Filter data for this subplot
            subset = agg_df[
                (agg_df[row_label] == row_val) &
                (agg_df[col_label] == col_val)
            ].copy()

            title = f"{row_label.capitalize()} {row_val} - {col_val}{filter_label}"

            if len(subset) == 0:
                # No data
                polys_plot.plot(ax=ax, color=missing_color, edgecolor='white', linewidth=0.3, alpha=0.6)

                # Add basemap
                if add_basemap and ctx is not None:
                    try:
                        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
                    except Exception:
                        pass  # Silently fail if basemap unavailable

                ax.set_title(title, fontsize=10, weight='bold')
                ax.axis('off')
                ax.text(
                    0.5, 0.5, 'No data',
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=12, color='gray'
                )
                subplot_counts[f"{row_val}_{col_val}"] = 0
                if verbose:
                    print(f"  [{i},{j}] {title}: NO DATA")
                continue

            # Merge with polygons
            merged = polys.merge(subset, on=uid_col, how='inner')

            if len(merged) == 0:
                # No matching polygons
                polys_plot.plot(ax=ax, color=missing_color, edgecolor='white', linewidth=0.3, alpha=0.6)

                # Add basemap
                if add_basemap and ctx is not None:
                    try:
                        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
                    except Exception:
                        pass

                ax.set_title(title, fontsize=10, weight='bold')
                ax.axis('off')
                ax.text(
                    0.5, 0.5, 'No matches',
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=12, color='gray'
                )
                subplot_counts[f"{row_val}_{col_val}"] = 0
                if verbose:
                    print(f"  [{i},{j}] {title}: NO MATCHES")
                continue

            # Reproject merged data if using basemap
            if add_basemap and ctx is not None:
                merged = merged.to_crs(epsg=3857)

            # Plot base (all polygons in missing color)
            polys_plot.plot(ax=ax, color=missing_color, edgecolor='white', linewidth=0.3, alpha=0.6)

            # Plot data polygons
            merged.plot(
                ax=ax,
                column=value_col,
                cmap=cmap,
                norm=norm,
                edgecolor='white',
                linewidth=0.3,
                legend=False,
                alpha=0.8
            )

            # Add basemap underneath
            if add_basemap and ctx is not None:
                try:
                    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
                except Exception:
                    pass  # Silently fail if basemap unavailable

            ax.set_title(title, fontsize=10, weight='bold')
            ax.axis('off')

            subplot_counts[f"{row_val}_{col_val}"] = len(merged)

            if verbose:
                print(f"  [{i},{j}] {title}: {len(merged)} polygons, "
                      f"err_rate={merged[value_col].mean():.2%}")

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation='horizontal',
        fraction=0.02,
        pad=0.02,
        aspect=40
    )
    cbar.set_label('Error Rate', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Format ticks as percentages
    ticks = np.linspace(vmin, vmax, 11)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nSaved: {output_path}")

    return subplot_counts


def main():
    # Default paths
    DEFAULT_PRED_DIR = Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\results')
    DEFAULT_POLYGONS = Path(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp')

    parser = argparse.ArgumentParser(
        description="Generate error rate choropleth grids (yearly & seasonal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--pred-dir', type=Path, default=DEFAULT_PRED_DIR,
                        help='Directory containing prediction CSVs')
    parser.add_argument('--polygons', type=Path, default=DEFAULT_POLYGONS,
                        help='Path to polygon file (.gpkg or .shp)')
    parser.add_argument('--uid-col', type=str, default='adm_code',
                        help='UID column name (tries common variations: adm_code, admin_code, FEWSNET_admin_code)')
    parser.add_argument('--date-col', type=str, default='date',
                        help='Date column name')
    parser.add_argument('--y-true-col', type=str, default='fews_ipc_crisis_true',
                        help='True label column')
    parser.add_argument('--y-pred-col', type=str, default='fews_ipc_crisis_pred',
                        help='Predicted label column')
    parser.add_argument('--file-glob', type=str, default='y_pred_test_*gp_fs*_*_*.csv',
                        help='File pattern to match (y_pred_test_gp_fs{scope}_{year}_{year}.csv or y_pred_test_xgb_gp_fs{scope}_{year}_{year}.csv)')
    parser.add_argument('--scope-regex', type=str, default=r'fs(?P<scope>\d+)',
                        help='Regex to extract scope from filename')
    parser.add_argument('--scopes', nargs='+', default=['fs1', 'fs2', 'fs3'],
                        help='Scopes to display')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date filter (YYYY-MM-DD), e.g., 2021-01-01. If not specified, includes all data from earliest date.')
    parser.add_argument('--end', type=str, default=None,
                        help='End date filter (YYYY-MM-DD), e.g., 2024-12-31. If not specified, includes all data up to latest date.')
    parser.add_argument('--start-year', type=int, default=None,
                        help='Start year filter (YYYY), e.g., 2021. Alternative to --start for year-level filtering.')
    parser.add_argument('--end-year', type=int, default=None,
                        help='End year filter (YYYY), e.g., 2024. Alternative to --end for year-level filtering.')
    parser.add_argument('--out-dir', type=Path, default=Path('.'),
                        help='Output directory')
    parser.add_argument('--vmin', type=float, default=0.0,
                        help='Color scale minimum')
    parser.add_argument('--vmax', type=float, default=1.0,
                        help='Color scale maximum')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure DPI (default: 300 for high quality)')
    parser.add_argument('--missing-color', type=str, default='#dddddd',
                        help='Color for missing data')
    parser.add_argument('--cmap', type=str, default='YlOrRd',
                        help='Matplotlib colormap')
    parser.add_argument('--basemap', action='store_true', default=True,
                        help='Add grey basemap for geographic context (default: True)')
    parser.add_argument('--no-basemap', dest='basemap', action='store_false',
                        help='Disable basemap')
    parser.add_argument('--filter', type=str, choices=['all', 'crisis', 'noncrisis'], default='all',
                        help="Filter by ground truth label: 'all' (no filter, default), 'crisis' (fews_ipc_crisis_true==1), 'noncrisis' (fews_ipc_crisis_true==0)")
    parser.add_argument('--region-map', type=Path, default=None,
                        help='Optional CSV mapping ADMIN0 to region (columns: ADMIN0,region)')
    parser.add_argument('--metrics-out', type=Path, default=None,
                        help='Output directory for country/region metrics CSVs (default: <out-dir>/metrics)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Setup output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    print("=" * 60)
    print("ERROR RATE CHOROPLETH GRID GENERATOR")
    print("=" * 60)

    # Check basemap availability
    if args.basemap and ctx is None:
        print("\nWARNING: contextily not available. Install with: pip install contextily")
        print("Proceeding without basemap.\n")
        args.basemap = False
    elif args.basemap and args.verbose:
        print("\nBasemap enabled: CartoDB.Positron\n")

    # Handle year-based filtering (convert to date strings)
    start_date = args.start
    end_date = args.end

    if args.start_year is not None:
        if start_date is not None:
            print("WARNING: Both --start and --start-year specified. Using --start-year.")
        start_date = f"{args.start_year}-01-01"
        if args.verbose:
            print(f"Start year {args.start_year} converted to date: {start_date}")

    if args.end_year is not None:
        if end_date is not None:
            print("WARNING: Both --end and --end-year specified. Using --end-year.")
        end_date = f"{args.end_year}-12-31"
        if args.verbose:
            print(f"End year {args.end_year} converted to date: {end_date}")

    # Print date filtering info
    if start_date or end_date:
        print("\nDate Filtering:")
        if start_date:
            print(f"  Start: {start_date}")
        else:
            print(f"  Start: (no filter, from earliest)")
        if end_date:
            print(f"  End: {end_date}")
        else:
            print(f"  End: (no filter, to latest)")
    else:
        print("\nDate Filtering: None (using all available data)")

    # Load predictions
    df = load_predictions(
        pred_dir=args.pred_dir,
        file_glob=args.file_glob,
        scope_regex=args.scope_regex,
        uid_col=args.uid_col,
        date_col=args.date_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        start_date=start_date,
        end_date=end_date,
        verbose=args.verbose
    )

    # Apply truth filter (crisis/noncrisis/all)
    df, filter_suffix = apply_truth_filter(
        df=df,
        filter_type=args.filter,
        y_true_col=args.y_true_col,
        verbose=args.verbose
    )

    # Create human-readable filter label for plot titles
    if args.filter == 'crisis':
        filter_label = ' (Crisis only)'
    elif args.filter == 'noncrisis':
        filter_label = ' (Non-crisis only)'
    else:
        filter_label = ''

    # Load polygons
    polys = prepare_polys(
        polygons_path=args.polygons,
        uid_col=args.uid_col,
        verbose=args.verbose
    )

    # Aggregate yearly
    print("\n" + "=" * 60)
    print("YEARLY AGGREGATION")
    print("=" * 60)

    yearly = aggregate_error_rate(
        df=df,
        by=['scope', 'year'],
        uid_col=args.uid_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        verbose=args.verbose
    )

    yearly_csv = args.out_dir / f'error_rate_yearly{filter_suffix}.csv'
    yearly.to_csv(yearly_csv, index=False)
    print(f"Saved: {yearly_csv}")

    # Aggregate seasonal
    print("\n" + "=" * 60)
    print("SEASONAL AGGREGATION")
    print("=" * 60)

    seasonal = aggregate_error_rate(
        df=df,
        by=['scope', 'season'],
        uid_col=args.uid_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        verbose=args.verbose
    )

    seasonal_csv = args.out_dir / f'error_rate_seasonal{filter_suffix}.csv'
    seasonal.to_csv(seasonal_csv, index=False)
    print(f"Saved: {seasonal_csv}")

    # Compute country and region metrics
    metrics_dir = args.metrics_out if args.metrics_out else (args.out_dir / 'metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("COMPUTING COUNTRY AND REGION METRICS")
    print("=" * 60)
    print(f"Using filtered dataset: {len(df)} rows")
    if start_date or end_date:
        print(f"Date range: {df[args.date_col].min()} to {df[args.date_col].max()}")
        print(f"[OK] Metrics will reflect date filtering: {'start=' + str(start_date) if start_date else ''} {'end=' + str(end_date) if end_date else ''}")

    try:
        metrics_summary = compute_admin0_and_region_metrics(
            df=df,
            polys=polys,
            scopes=args.scopes,
            uid_col=args.uid_col,
            y_true_col=args.y_true_col,
            y_pred_col=args.y_pred_col,
            admin0_col='ADMIN0',
            positive_label=1,
            region_map_path=args.region_map,
            out_dir=metrics_dir,
            verbose=args.verbose
        )

        print(f"\nWrote {len(metrics_summary['files'])} metric files to {metrics_dir}")

    except Exception as e:
        print(f"\nWARNING: Could not compute country/region metrics: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    # Filter to requested scopes
    yearly_filt = yearly[yearly['scope'].isin(args.scopes)].copy()
    seasonal_filt = seasonal[seasonal['scope'].isin(args.scopes)].copy()

    # Get sorted years (up to 4)
    years = sorted(yearly_filt['year'].unique())[-4:]
    seasons = ['DJFM', 'AMJJ', 'ASON']

    if len(years) == 0:
        print("\nWARNING: No years available for plotting")
    else:
        # Render yearly grid
        print("\n" + "=" * 60)
        print("RENDERING YEARLY GRID")
        print("=" * 60)

        yearly_png = args.out_dir / f'error_rate_yearly_4x3{filter_suffix}.png'
        yearly_counts = render_grid(
            agg_df=yearly_filt,
            polys=polys,
            uid_col=args.uid_col,
            row_values=years,
            col_values=args.scopes,
            row_label='year',
            col_label='scope',
            value_col='err_rate',
            vmin=args.vmin,
            vmax=args.vmax,
            missing_color=args.missing_color,
            dpi=args.dpi,
            cmap_name=args.cmap,
            output_path=yearly_png,
            add_basemap=args.basemap,
            verbose=args.verbose,
            filter_label=filter_label
        )

    if len(seasonal_filt) == 0:
        print("\nWARNING: No seasonal data available for plotting")
    else:
        # Render seasonal grid
        print("\n" + "=" * 60)
        print("RENDERING SEASONAL GRID")
        print("=" * 60)

        seasonal_png = args.out_dir / f'error_rate_seasonal_3x3{filter_suffix}.png'
        seasonal_counts = render_grid(
            agg_df=seasonal_filt,
            polys=polys,
            uid_col=args.uid_col,
            row_values=seasons,
            col_values=args.scopes,
            row_label='season',
            col_label='scope',
            value_col='err_rate',
            vmin=args.vmin,
            vmax=args.vmax,
            missing_color=args.missing_color,
            dpi=args.dpi,
            cmap_name=args.cmap,
            output_path=seasonal_png,
            add_basemap=args.basemap,
            verbose=args.verbose,
            filter_label=filter_label
        )

    # Generate report (convert numpy types to Python native for JSON serialization)
    filter_descriptions = {
        'all': 'No filter (all data)',
        'crisis': 'Actual crisis cases only (fews_ipc_crisis_true==1)',
        'noncrisis': 'Actual non-crisis cases only (fews_ipc_crisis_true==0)'
    }

    report = {
        'files_loaded': len(list(args.pred_dir.glob(args.file_glob))),
        'rows_after_filter': int(len(df)),
        'date_range': {
            'min': str(df[args.date_col].min()),
            'max': str(df[args.date_col].max())
        },
        'date_filter_applied': {
            'start_date': start_date,
            'end_date': end_date,
            'start_year': args.start_year,
            'end_year': args.end_year
        },
        'filter_applied': args.filter,
        'filter_description': filter_descriptions.get(args.filter, 'Unknown filter'),
        'years_plotted': [int(y) for y in years],
        'scopes_used': args.scopes,
        'season_counts': {
            season: int(len(seasonal_filt[seasonal_filt['season'] == season]))
            for season in seasons
        },
        'yearly_polygons': {k: int(v) for k, v in yearly_counts.items()} if len(years) > 0 else {},
        'seasonal_polygons': {k: int(v) for k, v in seasonal_counts.items()} if len(seasonal_filt) > 0 else {}
    }

    report_json = args.out_dir / f'error_rate_extract_report{filter_suffix}.json'
    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Report: {report_json}")
    print(f"Data rows used: {report['rows_after_filter']}")
    print(f"Date range in data: {report['date_range']['min']} to {report['date_range']['max']}")
    if start_date or end_date:
        print(f"Date filters applied: {f'start={start_date}' if start_date else ''} {f'end={end_date}' if end_date else ''}")
        print("[OK] All outputs (yearly, seasonal, country, region metrics) use this filtered dataset")
    else:
        print("Date filters: None (all available data used)")
    print(f"Years plotted: {report['years_plotted']}")
    print(f"Scopes used: {report['scopes_used']}")


if __name__ == '__main__':
    main()
