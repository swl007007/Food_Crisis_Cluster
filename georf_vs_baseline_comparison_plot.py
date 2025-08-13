import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_process_data(base_dir):
    """Load probit, RF, and XGBoost results for comparison"""
    
    # Define lag periods in months
    lag_periods = [3, 6, 9, 12]
    
    # Initialize data storage
    probit_data = {}
    rf_data = {}
    xgboost_data = {}
    
    # Load data for each lag period
    for lag in lag_periods:
        # Load probit baseline results
        probit_file = os.path.join(base_dir, f'probit_l{lag}.csv')
        probit_df = pd.read_csv(probit_file)
        
        # Create year-quarter identifier and sort
        probit_df['year_quarter'] = probit_df['year'].astype(str) + '-Q' + probit_df['quarter'].astype(str)
        probit_df = probit_df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # Calculate total samples for weighting
        probit_df['total_samples'] = probit_df['num_samples(0)'] + probit_df['num_samples(1)']
        
        probit_data[lag] = probit_df
        
        # Load RF (GeoRF) results
        rf_file = os.path.join(base_dir, f'RF_l{lag}.csv')
        rf_df = pd.read_csv(rf_file)
        
        # Create year-quarter identifier and sort
        rf_df['year_quarter'] = rf_df['year'].astype(str) + '-Q' + rf_df['quarter'].astype(str)
        rf_df = rf_df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # Calculate total samples for weighting
        rf_df['total_samples'] = rf_df['num_samples(0)'] + rf_df['num_samples(1)']
        
        rf_data[lag] = rf_df
        
        # Load XGBoost results
        xgboost_file = os.path.join(base_dir, f'XGboost_l{lag}.csv')
        xgboost_df = pd.read_csv(xgboost_file)
        
        # Create year-quarter identifier and sort
        xgboost_df['year_quarter'] = xgboost_df['year'].astype(str) + '-Q' + xgboost_df['quarter'].astype(str)
        xgboost_df = xgboost_df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # Calculate total samples for weighting
        xgboost_df['total_samples'] = xgboost_df['num_samples(0)'] + xgboost_df['num_samples(1)']
        
        xgboost_data[lag] = xgboost_df
    
    return probit_data, rf_data, xgboost_data

def calculate_weighted_metrics(df, metric_col):
    """Calculate sample-weighted metric averages"""
    weights = df['total_samples'] / df['total_samples'].sum()
    return (df[metric_col] * weights).sum()

def create_comparison_plot(probit_data, rf_data, xgboost_data):
    """Create 4x3 subplot grid comparing probit, RF, and GeoRF across forecasting scopes and metrics"""
    
    # Setup the 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    # Define lag periods and their labels
    lag_periods = [3, 6, 9, 12]
    lag_labels = ['l3 (3-month)', 'l6 (6-month)', 'l9 (9-month)', 'l12 (12-month)']
    
    # Define metrics and their labels
    metrics = ['precision(1)', 'recall(1)', 'f1(1)']
    metric_labels = ['Precision', 'Recall', 'F1 Score']
    
    # Define colors for each model
    colors = {
        'probit': 'red',
        'rf': 'blue', 
        'xgboost': 'green'
    }
    
    # Plot for each lag period (row) and metric (column)
    for row, (lag, lag_label) in enumerate(zip(lag_periods, lag_labels)):
        for col, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            
            # Get data for this lag period
            probit_df = probit_data[lag]
            rf_df = rf_data[lag]
            xgboost_df = xgboost_data[lag]
            
            # Find common time periods across all datasets
            common_periods = set(probit_df['year_quarter']).intersection(
                set(rf_df['year_quarter']), set(xgboost_df['year_quarter'])
            )
            common_periods = sorted(list(common_periods))
            
            # Filter datasets to common periods
            probit_common = probit_df[probit_df['year_quarter'].isin(common_periods)].sort_values(['year', 'quarter']).reset_index(drop=True)
            rf_common = rf_df[rf_df['year_quarter'].isin(common_periods)].sort_values(['year', 'quarter']).reset_index(drop=True)
            xgboost_common = xgboost_df[xgboost_df['year_quarter'].isin(common_periods)].sort_values(['year', 'quarter']).reset_index(drop=True)
            
            
            # Ensure all data arrays have the same length before plotting
            min_length = min(len(probit_common), len(rf_common), len(xgboost_common))
            
            # Truncate all arrays to the minimum length
            probit_values = probit_common[metric].iloc[:min_length]
            rf_values = rf_common[metric].iloc[:min_length]  
            xgboost_values = xgboost_common[metric].iloc[:min_length]
            x_positions = range(min_length)
            
            # Plot each model
            ax.plot(x_positions, probit_values, 'o-', 
                   color=colors['probit'], label='Probit (Baseline)', 
                   linewidth=2, markersize=4, alpha=0.8)
            
            ax.plot(x_positions, rf_values, 's-', 
                   color=colors['rf'], label='RF (GeoRF)', 
                   linewidth=2, markersize=4, alpha=0.8)
            
            ax.plot(x_positions, xgboost_values, '^-', 
                   color=colors['xgboost'], label='XGBoost', 
                   linewidth=2, markersize=4, alpha=0.8)
            
            # Set subplot title and labels
            if row == 0:
                ax.set_title(f'{metric_label}', fontsize=14, fontweight='bold')
            
            if col == 0:
                ax.set_ylabel(f'{lag_label}\\n{metric_label}', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel(f'{metric_label}', fontsize=11)
            
            # Set x-axis labels (only for bottom row)
            if row == 3:
                ax.set_xlabel('Time Period (Year-Quarter)', fontsize=11)
                # Set x-tick labels with rotation, showing every few quarters to avoid crowding
                step = max(1, min_length // 10)  # Show ~10 labels max
                ax.set_xticks(x_positions[::step])
                ax.set_xticklabels(probit_common['year_quarter'].iloc[:min_length:step], rotation=45, ha='right')
            else:
                ax.set_xticks([])
            
            # Add grid and legend (only for top-right subplot)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set y-axis limits for better comparison
            ax.set_ylim(0, 1)
            
            # Add row label on the left
            if col == 0:
                ax.text(-0.15, 0.5, lag_label, rotation=90, va='center', ha='center',
                       transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    # Add main title
    fig.suptitle('Model Performance Comparison Across Forecasting Scopes', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, right=0.88, hspace=0.3, wspace=0.3)
    
    return fig

def print_summary_statistics(probit_data, rf_data, xgboost_data):
    """Print weighted average performance statistics for each model and lag"""
    
    lag_periods = [3, 6, 9, 12]
    metrics = ['precision(1)', 'recall(1)', 'f1(1)']
    metric_names = ['Precision', 'Recall', 'F1 Score']
    
    print("\\n" + "="*80)
    print("SAMPLE-WEIGHTED PERFORMANCE SUMMARY")
    print("="*80)
    
    for lag in lag_periods:
        print(f"\\n{lag}-month lag forecasting:")
        print("-" * 40)
        
        # Calculate weighted averages for each metric
        for metric, metric_name in zip(metrics, metric_names):
            probit_avg = calculate_weighted_metrics(probit_data[lag], metric)
            rf_avg = calculate_weighted_metrics(rf_data[lag], metric)
            xgboost_avg = calculate_weighted_metrics(xgboost_data[lag], metric)
            
            print(f"{metric_name:>10}: Probit={probit_avg:.4f}, RF={rf_avg:.4f}, XGBoost={xgboost_avg:.4f}")

def main():
    # Set directory containing the data files
    base_dir = "20250808attachments/trial_07_XGboost_and_baseline"
    
    # Load and process data
    print("Loading probit, RF, and XGBoost data...")
    probit_data, rf_data, xgboost_data = load_and_process_data(base_dir)
    
    # Print data summary
    print("\\nData Summary:")
    print("=" * 50)
    for lag in [3, 6, 9, 12]:
        probit_df = probit_data[lag]
        rf_df = rf_data[lag]
        xgboost_df = xgboost_data[lag]
        print(f"{lag}-month lag: {len(probit_df)} time periods")
        print(f"  Time range: {probit_df['year_quarter'].min()} to {probit_df['year_quarter'].max()}")
    
    # Create comparison plot
    print("\\nCreating 4x3 comparison visualization...")
    fig = create_comparison_plot(probit_data, rf_data, xgboost_data)
    
    # Save plot
    output_path = os.path.join(base_dir, 'model_comparison_4x3_grid.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print summary statistics
    print_summary_statistics(probit_data, rf_data, xgboost_data)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()