import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import LAGS_MONTHS
from src.utils.lag_schedules import resolve_lag_schedule

ACTIVE_LAGS = resolve_lag_schedule(LAGS_MONTHS, context="config.LAGS_MONTHS")
PROJECT_ROOT = str(REPO_ROOT / 'results')
VIS_OUTPUT_DIR = REPO_ROOT / 'result_GeoRF' / 'vis'
FEWSNET_EXTENSION_SOURCE: dict[int, int] = {}

def load_and_process_data(base_dir=PROJECT_ROOT):
    """
    Load probit, RF, and XGBoost results for comparison using auto-detection.
    
    Expected file patterns:
    - Probit: baseline_probit_results/baseline_probit_results_fs{scope}.csv
    - GeoRF: results_df_gp_fs{scope}_{start_year}_{end_year}.csv  
    - XGBoost: results_df_xgb_gp_fs{scope}_{start_year}_{end_year}.csv
    
    Parameters:
    -----------
    base_dir : str, default='.'
        Base directory to search for files
    
    Returns:
    --------
    probit_data, rf_data, xgboost_data : dict
        Dictionaries with forecasting scope as key and DataFrames as values
    """
    print("Auto-detecting result files...")

    global FEWSNET_EXTENSION_SOURCE
    FEWSNET_EXTENSION_SOURCE.clear()

    # Map forecasting scope indices to canonical lag months
    scope_to_lag = {idx + 1: lag for idx, lag in enumerate(ACTIVE_LAGS)}
    
    # Initialize data storage
    probit_data = {}
    fewsnet_data = {}
    rf_data = {}
    xgboost_data = {}
    
    # 1. Load probit baseline results
    print("\nSearching for probit baseline results...")
    probit_pattern = os.path.join(base_dir, 'baseline_probit_results', 'baseline_probit_results_fs*.csv')
    probit_files = glob.glob(probit_pattern)
    
    # 1.5. Load FEWSNET baseline results
    print("\nSearching for FEWSNET baseline results...")
    fewsnet_pattern = os.path.join(base_dir, 'fewsnet_baseline_results', 'fewsnet_baseline_results_fs*.csv')
    fewsnet_files = glob.glob(fewsnet_pattern)
    
    for file_path in probit_files:
        filename = os.path.basename(file_path)
        # Extract forecasting scope from filename: baseline_probit_results_fs4.csv -> scope=4
        try:
            scope = int(filename.split('_fs')[1].split('.')[0])
        except (ValueError, IndexError) as e:
            print(f"  Warning: Could not parse forecasting scope from {filename}: {e}")
            continue

        if scope not in scope_to_lag:
            raise ValueError(
                f"Unsupported forecasting scope fs{scope} detected in {filename}; expected indices {list(scope_to_lag)} for lags {ACTIVE_LAGS}."
            )

        lag_months = scope_to_lag[scope]

        df = pd.read_csv(file_path)
        print(f"  Found probit results for forecasting scope {scope} ({lag_months}-month): {filename}")

        # Create year-quarter identifier if quarter column exists
        if 'quarter' in df.columns:
            df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        else:
            df['year_quarter'] = df['year'].astype(str)

        df = df.sort_values(['year'] + (['quarter'] if 'quarter' in df.columns else [])).reset_index(drop=True)
        probit_data[lag_months] = df
    
    # 1.5. Process FEWSNET baseline results
    
    for file_path in fewsnet_files:
        filename = os.path.basename(file_path)
        # Extract forecasting scope from filename: fewsnet_baseline_results_fs1.csv -> scope=1
        try:
            scope = int(filename.split('_fs')[1].split('.')[0])
        except (ValueError, IndexError) as e:
            print(f"  Warning: Could not parse forecasting scope from {filename}: {e}")
            continue

        if scope not in scope_to_lag:
            raise ValueError(
                f"Unsupported FEWSNET forecasting scope fs{scope} detected in {filename}; expected indices {list(scope_to_lag)} for lags {ACTIVE_LAGS}."
            )

        lag_months = scope_to_lag[scope]

        df = pd.read_csv(file_path)
        print(f"  Found FEWSNET results for forecasting scope {scope} ({lag_months}-month): {filename}")

        # Create year-quarter identifier if quarter column exists
        if 'quarter' in df.columns:
            df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        else:
            df['year_quarter'] = df['year'].astype(str)

        df = df.sort_values(['year'] + (['quarter'] if 'quarter' in df.columns else [])).reset_index(drop=True)
        fewsnet_data[lag_months] = df

    
    # Extend FEWSNET baseline to longer lag periods using the longest available prediction if needed
    if fewsnet_data:
        existing_lags = sorted(fewsnet_data.keys())
        longest_available = max(existing_lags)
        missing_lags = [lag for lag in ACTIVE_LAGS if lag not in fewsnet_data]

        for extended_lag in missing_lags:
            if extended_lag > longest_available:
                FEWSNET_EXTENSION_SOURCE[extended_lag] = longest_available
                fewsnet_data[extended_lag] = fewsnet_data[longest_available].copy()
                print(
                    f"  Extended FEWSNET results to {extended_lag}-month lag using {longest_available}-month predictions"
                )
    
    # 2. Load GeoRF results
    print("\nSearching for GeoRF results...")
    rf_pattern = os.path.join(base_dir, 'results_df_gp_fs*_*_*.csv')
    rf_files = glob.glob(rf_pattern)
    
    # Group RF files by forecasting scope and combine across year ranges
    rf_by_scope = {}
    for file_path in rf_files:
        filename = os.path.basename(file_path)
        try:
            parts = filename.split('_fs')[1].split('_')
            scope = int(parts[0])
            start_year = int(parts[1])
            end_year = int(parts[2].split('.')[0])
        except (ValueError, IndexError) as e:
            print(f"  Warning: Could not parse filename {filename}: {e}")
            continue

        if scope not in scope_to_lag:
            raise ValueError(
                f"Unsupported GeoRF forecasting scope fs{scope} detected in {filename}; expected indices {list(scope_to_lag)} for lags {ACTIVE_LAGS}."
            )

        lag_months = scope_to_lag[scope]

        df = pd.read_csv(file_path)
        print(f"  Found GeoRF results for scope {scope} ({lag_months}-month), years {start_year}-{end_year}: {filename}")

        if lag_months not in rf_by_scope:
            rf_by_scope[lag_months] = []
        rf_by_scope[lag_months].append(df)
    
    # Combine RF data across year ranges for each scope
    for lag_months, dfs in rf_by_scope.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Create year-quarter identifier
        if 'quarter' in combined_df.columns:
            combined_df['year_quarter'] = combined_df['year'].astype(str) + '-Q' + combined_df['quarter'].astype(str)
        else:
            combined_df['year_quarter'] = combined_df['year'].astype(str)
        
        combined_df = combined_df.sort_values(['year'] + (['quarter'] if 'quarter' in combined_df.columns else [])).reset_index(drop=True)
        rf_data[lag_months] = combined_df
        print(f"  Combined {len(dfs)} files for {lag_months}-month lag: {len(combined_df)} total records")
    
    # 3. Load XGBoost results  
    print("\nSearching for XGBoost results...")
    xgb_pattern = os.path.join(base_dir, 'results_df_xgb_gp_fs*_*_*.csv')
    xgb_files = glob.glob(xgb_pattern)
    
    # Group XGB files by forecasting scope and combine across year ranges
    xgb_by_scope = {}
    for file_path in xgb_files:
        filename = os.path.basename(file_path)
        try:
            parts = filename.split('_fs')[1].split('_')
            scope = int(parts[0])
            start_year = int(parts[1])
            end_year = int(parts[2].split('.')[0])
        except (ValueError, IndexError) as e:
            print(f"  Warning: Could not parse filename {filename}: {e}")
            continue

        if scope not in scope_to_lag:
            raise ValueError(
                f"Unsupported XGBoost forecasting scope fs{scope} detected in {filename}; expected indices {list(scope_to_lag)} for lags {ACTIVE_LAGS}."
            )

        lag_months = scope_to_lag[scope]

        df = pd.read_csv(file_path)
        print(f"  Found XGBoost results for scope {scope} ({lag_months}-month), years {start_year}-{end_year}: {filename}")

        if lag_months not in xgb_by_scope:
            xgb_by_scope[lag_months] = []
        xgb_by_scope[lag_months].append(df)
    
    # Combine XGBoost data across year ranges for each scope
    for lag_months, dfs in xgb_by_scope.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Create year-quarter identifier
        if 'quarter' in combined_df.columns:
            combined_df['year_quarter'] = combined_df['year'].astype(str) + '-Q' + combined_df['quarter'].astype(str)
        else:
            combined_df['year_quarter'] = combined_df['year'].astype(str)
            
        combined_df = combined_df.sort_values(['year'] + (['quarter'] if 'quarter' in combined_df.columns else [])).reset_index(drop=True)
        xgboost_data[lag_months] = combined_df
        print(f"  Combined {len(dfs)} files for {lag_months}-month lag: {len(combined_df)} total records")
    
    # Guard against unsupported lag outputs
    collected_lags = set(probit_data.keys()) | set(fewsnet_data.keys()) | set(rf_data.keys()) | set(xgboost_data.keys())
    unexpected_lags = sorted(collected_lags - set(ACTIVE_LAGS))
    if unexpected_lags:
        raise ValueError(f"Detected unsupported lag months {unexpected_lags}; expected {ACTIVE_LAGS}.")

    # Summary
    print(f"\nData loading summary:")
    print(f"  Active lag schedule: {ACTIVE_LAGS}")
    print(f"  Probit baseline: {list(probit_data.keys())} month lags")
    print(f"  FEWSNET baseline: {list(fewsnet_data.keys())} month lags")
    print(f"  GeoRF: {list(rf_data.keys())} month lags")
    print(f"  XGBoost: {list(xgboost_data.keys())} month lags")

    
    return probit_data, fewsnet_data, rf_data, xgboost_data


def create_comparison_plot(probit_data, fewsnet_data, rf_data, xgboost_data):
    """Create dynamic subplot grid comparing probit, FEWSNET, GeoRF, and XGBoost across available forecasting scopes and class 1 metrics"""
    global FEWSNET_EXTENSION_SOURCE

    # Use canonical lag ordering for plotting
    available_lags = list(ACTIVE_LAGS)

    # Verify at least one dataset provides data
    has_any_data = any(
        lag in dataset
        for lag in available_lags
        for dataset in (probit_data, fewsnet_data, rf_data, xgboost_data)
    )

    if not has_any_data:
        print("No data available for plotting!")
        return None

    # Setup subplot grid (one row per lag, three metrics per row)
    n_rows = len(available_lags)
    n_cols = 3  # precision, recall, f1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create lag labels
    lag_labels = [f'Lag {lag} months' for lag in available_lags]
    
    # Define metrics and their labels
    metrics = ['precision(1)', 'recall(1)', 'f1(1)']
    metric_labels = ['Precision (Class 1)', 'Recall (Class 1)', 'F1 Score (Class 1)']
    
    # Define colors for each model
    colors = {
        'probit': 'red',
        'fewsnet': 'orange',
        'rf': 'blue', 
        'xgboost': 'green'
    }
    
    # Plot for each lag period (row) and metric (column)
    for row, (lag, lag_label) in enumerate(zip(available_lags, lag_labels)):
        for col, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            
            # Check which datasets have this lag period
            has_probit = lag in probit_data
            has_fewsnet = lag in fewsnet_data
            has_rf = lag in rf_data
            has_xgb = lag in xgboost_data
            
            if not (has_probit or has_fewsnet or has_rf or has_xgb):
                # No data for this lag period
                ax.text(0.5, 0.5, f'No data available\nfor {lag}-month lag', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_label} - {lag_label}')
                continue
            
            # Get available datasets for this lag period
            datasets = {}
            if has_probit:
                datasets['probit'] = probit_data[lag]
            if has_fewsnet:
                datasets['fewsnet'] = fewsnet_data[lag]
            if has_rf:
                datasets['rf'] = rf_data[lag]
            if has_xgb:
                datasets['xgboost'] = xgboost_data[lag]
            
            # Find common time periods across available datasets
            all_periods = []
            for name, df in datasets.items():
                all_periods.extend(df['year_quarter'].tolist())
            
            # Get intersection of all available datasets
            if len(datasets) > 1:
                period_sets = [set(df['year_quarter']) for df in datasets.values()]
                common_periods = set.intersection(*period_sets)
            else:
                # Only one dataset available
                common_periods = set(list(datasets.values())[0]['year_quarter'])
            
            common_periods = sorted(list(common_periods))
            
            if not common_periods:
                # No common periods found
                ax.text(0.5, 0.5, f'No common time periods\nfor {lag}-month lag', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_label} - {lag_label}')
                continue
            
            # Filter available datasets to common periods and check for metric availability
            filtered_datasets = {}
            for name, df in datasets.items():
                if metric in df.columns:
                    filtered_df = df[df['year_quarter'].isin(common_periods)]
                    # Sort by year and quarter if quarter exists, otherwise just year
                    sort_cols = ['year'] + (['quarter'] if 'quarter' in filtered_df.columns else [])
                    filtered_df = filtered_df.sort_values(sort_cols).reset_index(drop=True)
                    filtered_datasets[name] = filtered_df
            
            if not filtered_datasets:
                # No datasets have this metric
                ax.text(0.5, 0.5, f'Metric {metric} not available\nfor {lag}-month lag', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_label} - {lag_label}')
                continue
            
            # Get minimum length across all available datasets
            min_length = min(len(df) for df in filtered_datasets.values())
            x_positions = range(min_length)
            
            # Plot each available model
            plotted_models = []
            
            if 'probit' in filtered_datasets:
                probit_values = filtered_datasets['probit'][metric].iloc[:min_length]
                ax.plot(x_positions, probit_values, 'o-', 
                       color=colors['probit'], label='Probit (Baseline)', 
                       linewidth=2, markersize=4, alpha=0.8)
                plotted_models.append('Probit')
            
            if 'fewsnet' in filtered_datasets:
                fewsnet_values = filtered_datasets['fewsnet'][metric].iloc[:min_length]
                extended_from = FEWSNET_EXTENSION_SOURCE.get(lag)
                if extended_from is not None:
                    label = f'FEWSNET ({extended_from}mo-ext)'
                    linestyle = '--'
                    alpha = 0.6
                    plotted_models.append(f'FEWSNET ({extended_from}mo-ext)')
                else:
                    label = 'FEWSNET (Official)'
                    linestyle = '-'
                    alpha = 0.8
                    plotted_models.append('FEWSNET')

                ax.plot(x_positions, fewsnet_values, 'D' + linestyle,
                       color=colors['fewsnet'], label=label,
                       linewidth=2, markersize=4, alpha=alpha)
            
            if 'rf' in filtered_datasets:
                rf_values = filtered_datasets['rf'][metric].iloc[:min_length]
                ax.plot(x_positions, rf_values, 's-', 
                       color=colors['rf'], label='GeoRF', 
                       linewidth=2, markersize=4, alpha=0.8)
                plotted_models.append('GeoRF')
            
            if 'xgboost' in filtered_datasets:
                xgboost_values = filtered_datasets['xgboost'][metric].iloc[:min_length]
                ax.plot(x_positions, xgboost_values, '^-',
                       color=colors['xgboost'], label='XGBoost',
                       linewidth=2, markersize=4, alpha=0.8)
                plotted_models.append('XGBoost')

            # Add vertical line at 2021-Q1 if it exists in the data
            if filtered_datasets:
                # Use any available dataset to find 2021-Q1 position
                sample_df = list(filtered_datasets.values())[0]
                year_quarters = sample_df['year_quarter'].iloc[:min_length].tolist()
                if '2021-Q1' in year_quarters:
                    vline_pos = year_quarters.index('2021-Q1')
                    ax.axvline(x=vline_pos, color='gray', linestyle='--',
                              linewidth=1.5, alpha=0.7, label='2021-Q1' if row == 0 and col == 2 else None)

            # Set subplot title and labels
            if row == 0:
                ax.set_title(f'{metric_label}', fontsize=14, fontweight='bold')
            
            if col == 0:
                ax.set_ylabel(f'{lag_label}', fontsize=12, fontweight='bold')
            
            # Set x-axis labels (only for bottom row)
            if row == n_rows - 1:  # Use dynamic bottom row
                ax.set_xlabel('Time Period', fontsize=11)
                # Set x-tick labels with rotation, showing every few periods to avoid crowding
                if min_length > 0:
                    step = max(1, min_length // 8)  # Show ~8 labels max
                    ax.set_xticks(x_positions[::step])
                    # Use any available dataset for x-labels (they should have common periods)
                    sample_df = list(filtered_datasets.values())[0]
                    ax.set_xticklabels(sample_df['year_quarter'].iloc[:min_length:step], rotation=45, ha='right')
            else:
                ax.set_xticks([])
            
            # Add grid and legend (only for top-right subplot if we have plots)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 2 and plotted_models:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set y-axis limits for better comparison
            ax.set_ylim(0, 1)
            
            # Add models plotted info as subtitle
            if plotted_models:
                models_str = ', '.join(plotted_models)
                ax.text(0.5, 0.95, f'Models: {models_str}', 
                       ha='center', va='top', transform=ax.transAxes, 
                       fontsize=9, style='italic')
    
    # Add main title
    fig.suptitle('Class 1 Performance Comparison Across Forecasting Scopes', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, right=0.88, hspace=0.3, wspace=0.3)
    
    return fig

def print_summary_statistics(probit_data, fewsnet_data, rf_data, xgboost_data):
    """Print class 1 performance statistics for each model and lag (unweighted averages)"""
    
    # Use canonical lag periods
    available_lags = list(ACTIVE_LAGS)
    metrics = ['precision(1)', 'recall(1)', 'f1(1)']
    metric_names = ['Precision (Class 1)', 'Recall (Class 1)', 'F1 Score (Class 1)']
    
    print("\\n" + "="*80)
    print("CLASS 1 PERFORMANCE SUMMARY (Unweighted Averages)")
    print("="*80)
    
    for lag in available_lags:
        print(f"\\n{lag}-month lag forecasting:")
        print("-" * 50)
        
        # Calculate simple means for each metric
        for metric, metric_name in zip(metrics, metric_names):
            probit_avg = probit_data[lag][metric].mean() if lag in probit_data and metric in probit_data[lag].columns else np.nan
            fewsnet_avg = fewsnet_data[lag][metric].mean() if lag in fewsnet_data and metric in fewsnet_data[lag].columns else np.nan
            rf_avg = rf_data[lag][metric].mean() if lag in rf_data and metric in rf_data[lag].columns else np.nan
            xgboost_avg = xgboost_data[lag][metric].mean() if lag in xgboost_data and metric in xgboost_data[lag].columns else np.nan
            
            # Add indicator for extended FEWSNET data
            extended_from = FEWSNET_EXTENSION_SOURCE.get(lag)
            fewsnet_label = "FEWSNET" if extended_from is None else f"FEWSNET ({extended_from}mo-ext)"
            
            print(f"{metric_name:>18}: Probit={probit_avg:.4f}, {fewsnet_label}={fewsnet_avg:.4f}, GeoRF={rf_avg:.4f}, XGBoost={xgboost_avg:.4f}")
        
        # Print data counts
        probit_count = len(probit_data[lag]) if lag in probit_data else 0
        fewsnet_count = len(fewsnet_data[lag]) if lag in fewsnet_data else 0
        rf_count = len(rf_data[lag]) if lag in rf_data else 0
        xgb_count = len(xgboost_data[lag]) if lag in xgboost_data else 0
        
        extended_from = FEWSNET_EXTENSION_SOURCE.get(lag)
        fewsnet_count_label = "FEWSNET" if extended_from is None else f"FEWSNET ({extended_from}mo-ext)"
            
        print(f"{'Sample counts':>18}: Probit={probit_count}, {fewsnet_count_label}={fewsnet_count}, GeoRF={rf_count}, XGBoost={xgb_count}")
        
        # Add note about extended FEWSNET data for longer lags
        if extended_from is not None:
            print(f"{'':>18}  * FEWSNET data extended using {extended_from}-month predictions")

def main():
    base_dir = PROJECT_ROOT
    
    # Load and process data using auto-detection
    print("Auto-detecting and loading comparison data...")
    probit_data, fewsnet_data, rf_data, xgboost_data = load_and_process_data(base_dir)
    
    # Check if any data was found
    if not probit_data and not fewsnet_data and not rf_data and not xgboost_data:
        print("\\nError: No result files found!")
        print("Expected files:")
        print("  - Probit: baseline_probit_results/baseline_probit_results_fs*.csv")
        print("  - FEWSNET: fewsnet_baseline_results/fewsnet_baseline_results_fs*.csv")
        print("  - GeoRF: results_df_gp_fs*_*_*.csv")
        print("  - XGBoost: results_df_xgb_gp_fs*_*_*.csv")
        return
    
    available_lags = list(ACTIVE_LAGS)
    
    # Print data summary
    print("\\nData Summary:")
    print("=" * 50)
    for lag in available_lags:
        has_probit = lag in probit_data
        has_fewsnet = lag in fewsnet_data
        has_rf = lag in rf_data
        has_xgb = lag in xgboost_data
        
        print(f"{lag}-month lag:")
        if has_probit:
            probit_df = probit_data[lag]
            print(f"  Probit: {len(probit_df)} time periods ({probit_df['year_quarter'].min()} to {probit_df['year_quarter'].max()})")
        else:
            print(f"  Probit: No data found")
            
        if has_fewsnet:
            fewsnet_df = fewsnet_data[lag]
            print(f"  FEWSNET: {len(fewsnet_df)} time periods ({fewsnet_df['year_quarter'].min()} to {fewsnet_df['year_quarter'].max()})")
        else:
            print(f"  FEWSNET: No data found")
            
        if has_rf:
            rf_df = rf_data[lag]
            print(f"  GeoRF: {len(rf_df)} time periods ({rf_df['year_quarter'].min()} to {rf_df['year_quarter'].max()})")
        else:
            print(f"  GeoRF: No data found")
            
        if has_xgb:
            xgb_df = xgboost_data[lag]
            print(f"  XGBoost: {len(xgb_df)} time periods ({xgb_df['year_quarter'].min()} to {xgb_df['year_quarter'].max()})")
        else:
            print(f"  XGBoost: No data found")
    # Create comparison plot only if we have data and visuals enabled
    if available_lags:
        print("\\nCreating class 1 performance comparison visualization...")
        fig = create_comparison_plot(probit_data, fewsnet_data, rf_data, xgboost_data)

        if fig is not None:
            VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = VIS_OUTPUT_DIR / f"georf_vs_baseline_lag_{'_'.join(str(lag) for lag in ACTIVE_LAGS)}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")

            print_summary_statistics(probit_data, fewsnet_data, rf_data, xgboost_data)
            plt.show()
        else:
            print("No figure generated; skipping visualization export.")
    else:
        print("\\nNo data available for plotting.")

if __name__ == "__main__":
    main()
