import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def _resolve_vis_flag(VIS_DEBUG_MODE=None):
    try:
        if VIS_DEBUG_MODE is None:
            try:
                from config_visual import VIS_DEBUG_MODE as Vv
            except ImportError:
                from config import VIS_DEBUG_MODE as Vv
            return bool(Vv)
        return bool(VIS_DEBUG_MODE)
    except Exception:
        return False

def load_and_process_data(base_dir):
    """Load all CSV files and keep time series data"""
    
    # File mapping with lag information
    files = {
        3: 'results_df_gp_fs1_l3.csv',
        6: 'results_df_gp_fs2_l6.csv', 
        9: 'results_df_gp_fs3_l9.csv',
        12: 'results_df_gp_fs4_l12.csv'
    }
    
    all_data = {}
    
    for lag, filename in files.items():
        filepath = os.path.join(base_dir, filename)
        df = pd.read_csv(filepath)
        
        # Create year-quarter identifier for time series
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        
        # Calculate total samples per row for weighting within each time period
        df['total_samples'] = df['num_samples(0)'] + df['num_samples(1)']
        
        # Store the dataframe with lag info
        all_data[lag] = df
    
    return all_data

def create_comparison_plot(data, VIS_DEBUG_MODE=None):
    if not _resolve_vis_flag(VIS_DEBUG_MODE):
        return None
    """Create time series subplot visualization with metrics and comparison lines"""
    
    lags = sorted(data.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # Get all unique time periods and sort them
    all_time_periods = set()
    for lag in lags:
        all_time_periods.update(data[lag]['year_quarter'].tolist())
    time_periods = sorted(list(all_time_periods))
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot precision over time
    for i, lag in enumerate(lags):
        df = data[lag]
        # Sort by year and quarter for proper time series
        df_sorted = df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        ax1.plot(range(len(df_sorted)), df_sorted['precision(1)'], 
                'o-', color=colors[i], label=f'{lag} months lag', markersize=4, linewidth=2)
        ax1.plot(range(len(df_sorted)), df_sorted['precision_base(1)'], 
                '--', color=colors[i], alpha=0.7, linewidth=1.5)
        
        # Set x-axis labels only for the first series to avoid overlap
        if i == 0:
            ax1.set_xticks(range(len(df_sorted)))
            ax1.set_xticklabels(df_sorted['year_quarter'], rotation=45)
    
    ax1.set_title('Precision Over Time')
    ax1.set_xlabel('Year-Quarter')
    ax1.set_ylabel('Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot recall over time
    for i, lag in enumerate(lags):
        df = data[lag]
        df_sorted = df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        ax2.plot(range(len(df_sorted)), df_sorted['recall(1)'], 
                'o-', color=colors[i], label=f'{lag} months lag', markersize=4, linewidth=2)
        ax2.plot(range(len(df_sorted)), df_sorted['recall_base(1)'], 
                '--', color=colors[i], alpha=0.7, linewidth=1.5)
        
        # Set x-axis labels only for the first series to avoid overlap
        if i == 0:
            ax2.set_xticks(range(len(df_sorted)))
            ax2.set_xticklabels(df_sorted['year_quarter'], rotation=45)
    
    ax2.set_title('Recall Over Time')
    ax2.set_xlabel('Year-Quarter')
    ax2.set_ylabel('Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot F1 over time
    for i, lag in enumerate(lags):
        df = data[lag]
        df_sorted = df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        ax3.plot(range(len(df_sorted)), df_sorted['f1(1)'], 
                'o-', color=colors[i], label=f'{lag} months lag', markersize=4, linewidth=2)
        ax3.plot(range(len(df_sorted)), df_sorted['f1_base(1)'], 
                '--', color=colors[i], alpha=0.7, linewidth=1.5)
        
        # Set x-axis labels only for the first series to avoid overlap
        if i == 0:
            ax3.set_xticks(range(len(df_sorted)))
            ax3.set_xticklabels(df_sorted['year_quarter'], rotation=45)
    
    ax3.set_title('F1 Score Over Time')
    ax3.set_xlabel('Year-Quarter')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add legend explanation for dashed lines
    fig.suptitle('Model Performance vs Baseline Over Time (Dashed Lines)', fontsize=14, y=1.02)
    
    plt.tight_layout()
    return fig

def main(VIS_DEBUG_MODE=None):
    if not _resolve_vis_flag(VIS_DEBUG_MODE):
        print("Visualization disabled (VIS_DEBUG_MODE=False); skipping metric comparison plot.")
        return
    # Set base directory  
    base_dir = "trial_06_10years_metric_compare"
    
    # Load and process data
    print("Loading and processing data...")
    all_data = load_and_process_data(base_dir)
    
    # Print data summary
    print("\nData Summary:")
    print("=" * 60)
    for lag in sorted(all_data.keys()):
        df = all_data[lag]
        print(f"Lag {lag} months: {len(df)} time periods")
        print(f"  Time range: {df['year_quarter'].min()} to {df['year_quarter'].max()}")
        print(f"  Avg Precision: {df['precision(1)'].mean():.4f} (baseline: {df['precision_base(1)'].mean():.4f})")
        print(f"  Avg Recall:    {df['recall(1)'].mean():.4f} (baseline: {df['recall_base(1)'].mean():.4f})")
        print(f"  Avg F1 Score:  {df['f1(1)'].mean():.4f} (baseline: {df['f1_base(1)'].mean():.4f})")
        print()
    
    # Create and save plot
    print("Creating time series visualization...")
    fig = create_comparison_plot(all_data, VIS_DEBUG_MODE=True)
    
    # Save plot
    output_path = os.path.join(base_dir, 'metric_comparison_timeseries.png')
    if fig is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        # Show plot
        plt.show()

if __name__ == "__main__":
    main()
