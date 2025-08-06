import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_process_data(base_dir, baseline_results_path):
    """Load GeoRF results and baseline probit results for comparison"""
    
    # File mapping for GeoRF results with lag information
    georf_files = {
        3: 'results_df_gp_fs1_l3.csv',
        6: 'results_df_gp_fs2_l6.csv', 
        9: 'results_df_gp_fs3_l9.csv',
        12: 'results_df_gp_fs4_l12.csv'
    }
    
    # Load GeoRF data
    georf_data = {}
    
    for lag, filename in georf_files.items():
        filepath = os.path.join(base_dir, filename)
        df = pd.read_csv(filepath)
        
        # Create year-quarter identifier for time series
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        
        # Calculate total samples per row for weighting within each time period
        df['total_samples'] = df['num_samples(0)'] + df['num_samples(1)']
        
        # Store the dataframe with lag info
        georf_data[lag] = df
    
    # Load baseline probit results
    baseline_df = pd.read_csv(baseline_results_path)
    
    # Create year-quarter identifier for baseline
    baseline_df['year_quarter'] = baseline_df['year'].astype(str) + '-Q' + baseline_df['quarter'].astype(str)
    
    # Calculate total samples for baseline
    baseline_df['total_samples'] = baseline_df['num_samples(0)'] + baseline_df['num_samples(1)']
    
    # Rename baseline columns to match GeoRF format
    baseline_df = baseline_df.rename(columns={
        'precision(0)': 'precision_base(0)',
        'recall(0)': 'recall_base(0)', 
        'f1(0)': 'f1_base(0)',
        'precision(1)': 'precision_base(1)',
        'recall(1)': 'recall_base(1)',
        'f1(1)': 'f1_base(1)'
    })
    
    return georf_data, baseline_df

def create_comparison_plot(georf_data, baseline_df):
    """Create time series subplot visualization comparing GeoRF with baseline probit"""
    
    lags = sorted(georf_data.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # Sort baseline data by year and quarter
    baseline_sorted = baseline_df.sort_values(['year', 'quarter']).reset_index(drop=True)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot precision over time
    for i, lag in enumerate(lags):
        df = georf_data[lag]
        df_sorted = df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # GeoRF performance (solid lines)
        ax1.plot(range(len(df_sorted)), df_sorted['precision(1)'], 
                'o-', color=colors[i], label=f'GeoRF {lag}m lag', markersize=4, linewidth=2)
        
        # Set x-axis labels only for the first series to avoid overlap
        if i == 0:
            ax1.set_xticks(range(len(df_sorted)))
            ax1.set_xticklabels(df_sorted['year_quarter'], rotation=45)
    
    # Add baseline probit results as time series (dashed line)
    ax1.plot(range(len(baseline_sorted)), baseline_sorted['precision_base(1)'], 
            's--', color='black', label='Baseline Probit', markersize=3, linewidth=2, alpha=0.8)
    
    ax1.set_title('Precision Over Time')
    ax1.set_xlabel('Year-Quarter')
    ax1.set_ylabel('Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot recall over time
    for i, lag in enumerate(lags):
        df = georf_data[lag]
        df_sorted = df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # GeoRF performance (solid lines)
        ax2.plot(range(len(df_sorted)), df_sorted['recall(1)'], 
                'o-', color=colors[i], label=f'GeoRF {lag}m lag', markersize=4, linewidth=2)
        
        # Set x-axis labels only for the first series to avoid overlap
        if i == 0:
            ax2.set_xticks(range(len(df_sorted)))
            ax2.set_xticklabels(df_sorted['year_quarter'], rotation=45)
    
    # Add baseline probit recall time series
    ax2.plot(range(len(baseline_sorted)), baseline_sorted['recall_base(1)'], 
            's--', color='black', label='Baseline Probit', markersize=3, linewidth=2, alpha=0.8)
    
    ax2.set_title('Recall Over Time')
    ax2.set_xlabel('Year-Quarter')
    ax2.set_ylabel('Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot F1 over time
    for i, lag in enumerate(lags):
        df = georf_data[lag]
        df_sorted = df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # GeoRF performance (solid lines)
        ax3.plot(range(len(df_sorted)), df_sorted['f1(1)'], 
                'o-', color=colors[i], label=f'GeoRF {lag}m lag', markersize=4, linewidth=2)
        
        # Set x-axis labels only for the first series to avoid overlap
        if i == 0:
            ax3.set_xticks(range(len(df_sorted)))
            ax3.set_xticklabels(df_sorted['year_quarter'], rotation=45)
    
    # Add baseline probit F1 time series
    ax3.plot(range(len(baseline_sorted)), baseline_sorted['f1_base(1)'], 
            's--', color='black', label='Baseline Probit', markersize=3, linewidth=2, alpha=0.8)
    
    ax3.set_title('F1 Score Over Time')
    ax3.set_xlabel('Year-Quarter')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add main title
    fig.suptitle('GeoRF vs Baseline Probit Regression Performance Over Time', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Calculate sample-weighted averages for return
    weights = baseline_sorted['total_samples'] / baseline_sorted['total_samples'].sum()
    weighted_baseline_precision = (baseline_sorted['precision_base(1)'] * weights).sum()
    weighted_baseline_recall = (baseline_sorted['recall_base(1)'] * weights).sum()
    weighted_baseline_f1 = (baseline_sorted['f1_base(1)'] * weights).sum()
    
    return fig, weighted_baseline_precision, weighted_baseline_recall, weighted_baseline_f1

def main():
    # Set directories
    base_dir = "trial_06_10years_metric_compare"
    baseline_results_path = "baseline_probit_results/baseline_probit_results.csv"
    
    # Load and process data
    print("Loading GeoRF and baseline probit data...")
    georf_data, baseline_df = load_and_process_data(base_dir, baseline_results_path)
    
    # Print data summary
    print("\nGeoRF Data Summary:")
    print("=" * 50)
    for lag in sorted(georf_data.keys()):
        df = georf_data[lag]
        print(f"GeoRF {lag}m lag: {len(df)} time periods")
        print(f"  Time range: {df['year_quarter'].min()} to {df['year_quarter'].max()}")
        print(f"  Avg Precision: {df['precision(1)'].mean():.4f}")
        print(f"  Avg Recall:    {df['recall(1)'].mean():.4f}")
        print(f"  Avg F1 Score:  {df['f1(1)'].mean():.4f}")
        print()
    
    print("Baseline Probit Data Summary:")
    print("=" * 50)
    print(f"Baseline: {len(baseline_df)} time periods")
    print(f"Time range: {baseline_df['year_quarter'].min()} to {baseline_df['year_quarter'].max()}")
    
    # Calculate sample-weighted averages for baseline
    weights = baseline_df['total_samples'] / baseline_df['total_samples'].sum()
    weighted_precision = (baseline_df['precision_base(1)'] * weights).sum()
    weighted_recall = (baseline_df['recall_base(1)'] * weights).sum()
    weighted_f1 = (baseline_df['f1_base(1)'] * weights).sum()
    
    print(f"Weighted Avg Precision: {weighted_precision:.4f}")
    print(f"Weighted Avg Recall:    {weighted_recall:.4f}")
    print(f"Weighted Avg F1 Score:  {weighted_f1:.4f}")
    print()
    
    # Create and save plot
    print("Creating comparison visualization...")
    fig, baseline_precision, baseline_recall, baseline_f1 = create_comparison_plot(georf_data, baseline_df)
    
    # Save plot
    output_path = os.path.join(base_dir, 'georf_vs_baseline_probit_comparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Performance comparison summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"Baseline Probit Regression (Sample-Weighted):")
    print(f"  Precision: {baseline_precision:.4f}")
    print(f"  Recall:    {baseline_recall:.4f}")
    print(f"  F1 Score:  {baseline_f1:.4f}")
    print()
    
    # Compare with each GeoRF lag
    for lag in sorted(georf_data.keys()):
        df = georf_data[lag]
        georf_precision = df['precision(1)'].mean()
        georf_recall = df['recall(1)'].mean()
        georf_f1 = df['f1(1)'].mean()
        
        print(f"GeoRF {lag}m lag vs Baseline:")
        print(f"  Precision: {georf_precision:.4f} vs {baseline_precision:.4f} "
              f"({'+' if georf_precision > baseline_precision else ''}{georf_precision - baseline_precision:.4f})")
        print(f"  Recall:    {georf_recall:.4f} vs {baseline_recall:.4f} "
              f"({'+' if georf_recall > baseline_recall else ''}{georf_recall - baseline_recall:.4f})")
        print(f"  F1 Score:  {georf_f1:.4f} vs {baseline_f1:.4f} "
              f"({'+' if georf_f1 > baseline_f1 else ''}{georf_f1 - baseline_f1:.4f})")
        print()
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()