#!/usr/bin/env python3
"""
Plot F1 CDF comparison between pooled and partitioned models
across different model configurations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "f1_cdf_comparison_pooled_vs_partitioned.png"

# Define folder patterns and model classes
FOLDER_PATTERNS = {
    'Month-Specific (m)': [
        'result_partition_k40_nc4_compare_m_fs1',
        'result_partition_k40_nc4_compare_m_fs2',
    ],
    'No Refinement (rf0g)': [
        'result_partition_k40_nc4_compare_rf0g_fs1',
        'result_partition_k40_nc4_compare_rf0g_fs2',
    ],
    'Refinement x3 (rf3)': [
        'result_partition_k40_nc4_compare_rf3_fs1',
        'result_partition_k40_nc4_compare_rf3_fs2',
    ],
}

def load_metrics_for_model_class(folder_names):
    """Load and combine metrics_monthly.csv from multiple folders."""
    dfs = []

    for folder_name in folder_names:
        folder_path = BASE_DIR / folder_name
        metrics_file = folder_path / 'metrics_monthly.csv'

        if not metrics_file.exists():
            print(f"WARNING: File not found: {metrics_file}")
            continue

        print(f"Loading: {metrics_file.name} from {folder_name}")
        df = pd.read_csv(metrics_file)
        df['source_folder'] = folder_name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No metrics files found for folders: {folder_names}")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined_df)} rows")

    return combined_df

def plot_f1_cdf(ax, df, title):
    """Plot CDF of F1 scores for pooled vs partitioned models."""

    # Check if 'model' column exists
    if 'model' not in df.columns:
        print(f"ERROR: 'model' column not found in dataframe for {title}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Get unique models
    models = df['model'].unique()
    print(f"\n{title}:")
    print(f"  Unique models: {models}")

    # Identify pooled and partitioned models
    pooled_models = [m for m in models if 'pooled' in m.lower() or 'pool' in m.lower()]
    partitioned_models = [m for m in models if 'partition' in m.lower() or 'part' in m.lower()]

    print(f"  Pooled models: {pooled_models}")
    print(f"  Partitioned models: {partitioned_models}")

    # Define colors
    colors = {
        'pooled': '#e74c3c',      # Red
        'partitioned': '#3498db'  # Blue
    }

    # Plot CDFs
    for model_type, model_list in [('pooled', pooled_models), ('partitioned', partitioned_models)]:
        if not model_list:
            print(f"  WARNING: No {model_type} models found")
            continue

        # Combine F1 scores from all models of this type
        f1_scores = []
        for model in model_list:
            model_df = df[df['model'] == model]

            # Try different F1 column names
            f1_col = None
            for col in ['f1_class1', 'f1_score', 'f1', 'F1_class1']:
                if col in model_df.columns:
                    f1_col = col
                    break

            if f1_col is None:
                print(f"  ERROR: F1 column not found for {model}. Available: {list(model_df.columns)}")
                continue

            scores = model_df[f1_col].dropna().values
            f1_scores.extend(scores)
            print(f"    {model}: {len(scores)} scores (mean={np.mean(scores):.3f})")

        if not f1_scores:
            continue

        # Sort F1 scores for CDF
        f1_sorted = np.sort(f1_scores)
        cdf = np.arange(1, len(f1_sorted) + 1) / len(f1_sorted)

        # Plot
        ax.plot(f1_sorted, cdf,
                label=f'{model_type.capitalize()} (n={len(f1_sorted)})',
                color=colors[model_type],
                linewidth=2.5,
                alpha=0.8)

    # Styling
    ax.set_xlabel('F1 Score (Class 1 - Crisis)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add median lines
    for model_type, model_list in [('pooled', pooled_models), ('partitioned', partitioned_models)]:
        if not model_list:
            continue

        f1_scores = []
        for model in model_list:
            model_df = df[df['model'] == model]
            f1_col = None
            for col in ['f1_class1', 'f1_score', 'f1', 'F1_class1']:
                if col in model_df.columns:
                    f1_col = col
                    break
            if f1_col:
                f1_scores.extend(model_df[f1_col].dropna().values)

        if f1_scores:
            median_f1 = np.median(f1_scores)
            ax.axvline(median_f1, color=colors[model_type], linestyle=':', alpha=0.5, linewidth=1.5)
            ax.text(median_f1, 0.05, f'{median_f1:.3f}',
                   rotation=90, va='bottom', ha='right',
                   fontsize=9, color=colors[model_type], fontweight='bold')

def main():
    print("=" * 80)
    print("F1 CDF COMPARISON: POOLED VS PARTITIONED MODELS")
    print("=" * 80)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Process each model class
    for idx, (model_class_name, folder_names) in enumerate(FOLDER_PATTERNS.items()):
        print(f"\n{'=' * 80}")
        print(f"MODEL CLASS: {model_class_name}")
        print(f"{'=' * 80}")

        try:
            # Load and combine data
            df = load_metrics_for_model_class(folder_names)

            # Plot CDF
            plot_f1_cdf(axes[idx], df, model_class_name)

        except Exception as e:
            print(f"ERROR processing {model_class_name}: {e}")
            import traceback
            traceback.print_exc()

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n{'=' * 80}")
    print(f"PLOT SAVED: {OUTPUT_FILE}")
    print(f"{'=' * 80}")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
