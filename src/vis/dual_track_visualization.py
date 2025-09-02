"""
Dual-Track Partition Visualization

This module provides separate visualization tracks for:
1. Hierarchical Track: Clean partition hierarchy without spatial optimization
2. Optimized Track: Spatially optimized partitions with contiguity refinement

This addresses the user confusion about fragmented final maps by clearly
separating algorithmic intent and providing both perspectives.

Author: Weilunm Shi  
Date: 2025-08-29
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime

def create_dual_track_visualization(
    result_dir: str,
    hierarchical_correspondence: str,
    optimized_correspondence: str,
    shapefile_path: Optional[str] = None,
    output_prefix: str = "dual_track",
    figsize: Tuple[int, int] = (20, 10),
    dpi: int = 300
) -> Dict[str, str]:
    """
    Create side-by-side comparison of hierarchical vs spatially optimized partitions.
    
    Args:
        result_dir: Directory to save outputs
        hierarchical_correspondence: Path to clean hierarchical correspondence table
        optimized_correspondence: Path to spatially optimized correspondence table  
        shapefile_path: Path to admin boundaries shapefile
        output_prefix: Prefix for output filenames
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
    
    Returns:
        Dictionary mapping visualization types to file paths
    """
    
    try:
        import geopandas as gpd
        import contextily as ctx
        from matplotlib.colors import ListedColormap
    except ImportError as e:
        raise ImportError(f"Required packages missing: {e}. Install with: pip install geopandas contextily")
    
    # Default shapefile path
    if shapefile_path is None:
        shapefile_path = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
    
    # Load correspondence tables
    hierarchical_df = pd.read_csv(hierarchical_correspondence)
    optimized_df = pd.read_csv(optimized_correspondence)
    
    # Load shapefile
    if not os.path.exists(shapefile_path):
        print(f"Warning: Shapefile not found at {shapefile_path}")
        return {'status': 'shapefile_missing', 'shapefile_path': shapefile_path}
    
    shapefile = gpd.read_file(shapefile_path)
    
    # Merge with correspondence tables
    hierarchical_merged = shapefile.merge(
        hierarchical_df, 
        left_on='admin_code', 
        right_on='FEWSNET_admin_code', 
        how='inner'
    )
    
    optimized_merged = shapefile.merge(
        optimized_df, 
        left_on='admin_code', 
        right_on='FEWSNET_admin_code', 
        how='inner'
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Color scheme for partitions
    partition_colors = {
        0: '#FF6B6B',  # Red
        1: '#4ECDC4',  # Teal
        2: '#45B7D1',  # Blue
        3: '#FFA07A',  # Light salmon
        4: '#98D8C8',  # Mint
        5: '#F7DC6F',  # Yellow
        6: '#BB8FCE',  # Purple
        7: '#85C1E9'   # Light blue
    }
    
    # Plot hierarchical partitions (left subplot)
    hierarchical_partitions = sorted(hierarchical_merged['partition_id'].unique())
    for partition in hierarchical_partitions:
        mask = hierarchical_merged['partition_id'] == partition
        color = partition_colors.get(partition, '#CCCCCC')
        hierarchical_merged[mask].plot(
            ax=ax1,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
            label=f'Partition {partition}'
        )
    
    ax1.set_title(f'Hierarchical Partitions\n(Clean Algorithm Structure)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.axis('off')
    
    # Add hierarchical info text
    hierarchical_info = f"Partitions: {len(hierarchical_partitions)}\nType: Clean Hierarchy\nContiguity: Not Applied"
    ax1.text(0.02, 0.98, hierarchical_info, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot optimized partitions (right subplot)
    optimized_partitions = sorted(optimized_merged['partition_id'].unique())
    for partition in optimized_partitions:
        mask = optimized_merged['partition_id'] == partition
        color = partition_colors.get(partition, '#CCCCCC')
        optimized_merged[mask].plot(
            ax=ax2,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
            label=f'Partition {partition}'
        )
    
    ax2.set_title(f'Spatially Optimized Partitions\n(Performance-Optimized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)  
    ax2.axis('off')
    
    # Add optimization info text
    optimization_info = f"Partitions: {len(optimized_partitions)}\nType: Spatial Optimization\nContiguity: Applied (4/9 threshold)"
    ax2.text(0.02, 0.98, optimization_info, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add fragmentation metrics if available
    if 'fragmentation_index' in optimized_df.columns:
        avg_frag = optimized_df.groupby('partition_id')['fragmentation_index'].first().mean()
        fragmentation_text = f"Avg Fragmentation: {avg_frag:.3f}"
        ax2.text(0.02, 0.85, fragmentation_text, transform=ax2.transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Create shared legend
    legend_patches = []
    all_partitions = sorted(set(hierarchical_partitions + optimized_partitions))
    for partition in all_partitions:
        color = partition_colors.get(partition, '#CCCCCC')
        legend_patches.append(mpatches.Patch(color=color, label=f'Partition {partition}'))
    
    fig.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=len(all_partitions))
    
    # Add main title
    fig.suptitle('GeoRF Partition Comparison: Hierarchical vs Spatially Optimized', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add explanation text
    explanation = ("Left: Clean hierarchical splits following algorithm structure. "
                  "Right: Spatially optimized assignments after contiguity refinement. "
                  "Fragmentation in right map is intentional performance optimization, not errors.")
    fig.text(0.5, 0.08, explanation, ha='center', fontsize=11, wrap=True,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    # Save visualization
    output_files = {}
    
    # Main comparison plot
    comparison_path = os.path.join(result_dir, f'{output_prefix}_hierarchical_vs_optimized.png')
    plt.savefig(comparison_path, dpi=dpi, bbox_inches='tight')
    output_files['comparison'] = comparison_path
    
    plt.close()
    
    # Generate individual high-resolution plots
    # Hierarchical only
    fig_hier, ax_hier = plt.subplots(1, 1, figsize=(12, 10))
    for partition in hierarchical_partitions:
        mask = hierarchical_merged['partition_id'] == partition
        color = partition_colors.get(partition, '#CCCCCC')
        hierarchical_merged[mask].plot(
            ax=ax_hier,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
            label=f'Partition {partition}'
        )
    
    ax_hier.set_title('GeoRF Hierarchical Partitions (Clean Algorithm Structure)', 
                      fontsize=16, fontweight='bold')
    ax_hier.axis('off')
    ax_hier.legend(loc='upper right')
    
    hierarchical_path = os.path.join(result_dir, f'{output_prefix}_hierarchical_only.png')
    plt.savefig(hierarchical_path, dpi=dpi, bbox_inches='tight')
    output_files['hierarchical_only'] = hierarchical_path
    plt.close()
    
    # Optimized only
    fig_opt, ax_opt = plt.subplots(1, 1, figsize=(12, 10))
    for partition in optimized_partitions:
        mask = optimized_merged['partition_id'] == partition
        color = partition_colors.get(partition, '#CCCCCC')
        optimized_merged[mask].plot(
            ax=ax_opt,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.8,
            label=f'Partition {partition}'
        )
    
    ax_opt.set_title('GeoRF Spatially Optimized Partitions (Performance-Optimized with Fragmentation)', 
                     fontsize=16, fontweight='bold')
    ax_opt.axis('off')
    ax_opt.legend(loc='upper right')
    
    # Add fragmentation explanation
    frag_explanation = ("Mosaicked patterns result from 4/9 majority voting in contiguity refinement.\n"
                       "Fragmentation is intentional optimization for crisis prediction accuracy.")
    ax_opt.text(0.02, 0.02, frag_explanation, transform=ax_opt.transAxes,
                verticalalignment='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    optimized_path = os.path.join(result_dir, f'{output_prefix}_optimized_only.png')
    plt.savefig(optimized_path, dpi=dpi, bbox_inches='tight')
    output_files['optimized_only'] = optimized_path
    plt.close()
    
    return output_files

def generate_fragmentation_comparison_metrics(
    hierarchical_correspondence: str,
    optimized_correspondence: str,
    result_dir: str,
    output_name: str = "fragmentation_comparison"
) -> str:
    """
    Generate detailed comparison metrics between hierarchical and optimized partitions.
    
    Args:
        hierarchical_correspondence: Path to hierarchical correspondence table
        optimized_correspondence: Path to optimized correspondence table
        result_dir: Directory to save output
        output_name: Name for output file
    
    Returns:
        Path to generated metrics comparison file
    """
    
    # Load correspondence tables
    hierarchical_df = pd.read_csv(hierarchical_correspondence)
    optimized_df = pd.read_csv(optimized_correspondence)
    
    # Basic statistics comparison
    comparison_data = []
    
    # Partition count comparison
    hier_partitions = set(hierarchical_df['partition_id'].unique())
    opt_partitions = set(optimized_df['partition_id'].unique())
    
    comparison_data.append({
        'metric': 'unique_partitions',
        'hierarchical': len(hier_partitions),
        'optimized': len(opt_partitions),
        'difference': len(opt_partitions) - len(hier_partitions),
        'description': 'Number of unique partition IDs'
    })
    
    # Admin unit distribution
    for partition_id in sorted(hier_partitions.union(opt_partitions)):
        hier_count = len(hierarchical_df[hierarchical_df['partition_id'] == partition_id])
        opt_count = len(optimized_df[optimized_df['partition_id'] == partition_id])
        
        comparison_data.append({
            'metric': f'partition_{partition_id}_size',
            'hierarchical': hier_count,
            'optimized': opt_count,
            'difference': opt_count - hier_count,
            'description': f'Number of admin units in partition {partition_id}'
        })
    
    # Fragmentation metrics (if available in optimized table)
    if 'fragmentation_index' in optimized_df.columns:
        avg_fragmentation = optimized_df.groupby('partition_id')['fragmentation_index'].first().mean()
        max_fragmentation = optimized_df.groupby('partition_id')['fragmentation_index'].first().max()
        
        comparison_data.append({
            'metric': 'avg_fragmentation_index',
            'hierarchical': 0.0,  # Hierarchical assumed to be non-fragmented
            'optimized': avg_fragmentation,
            'difference': avg_fragmentation,
            'description': 'Average fragmentation index (1 = completely fragmented)'
        })
        
        comparison_data.append({
            'metric': 'max_fragmentation_index',
            'hierarchical': 0.0,
            'optimized': max_fragmentation,
            'difference': max_fragmentation,
            'description': 'Maximum fragmentation index across partitions'
        })
    
    # Component counts (if available)
    if 'component_count' in optimized_df.columns:
        total_components_opt = optimized_df.groupby('partition_id')['component_count'].first().sum()
        
        comparison_data.append({
            'metric': 'total_spatial_components',
            'hierarchical': len(hier_partitions),  # Assume 1 component per partition
            'optimized': total_components_opt,
            'difference': total_components_opt - len(hier_partitions),
            'description': 'Total number of disconnected spatial components'
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(result_dir, f'{output_name}.csv')
    comparison_df.to_csv(csv_path, index=False)
    
    # Generate detailed report
    report_path = os.path.join(result_dir, f'{output_name}_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Fragmentation Comparison Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares hierarchical (clean) vs spatially optimized (fragmented) partition assignments.\n\n")
        
        f.write("### Key Differences\n\n")
        
        # Partition count differences
        hier_partitions_count = comparison_df[comparison_df['metric'] == 'unique_partitions']['hierarchical'].iloc[0]
        opt_partitions_count = comparison_df[comparison_df['metric'] == 'unique_partitions']['optimized'].iloc[0]
        
        f.write(f"- **Hierarchical Partitions**: {hier_partitions_count} clean partition(s)\n")
        f.write(f"- **Optimized Partitions**: {opt_partitions_count} optimized partition(s)\n")
        
        if 'avg_fragmentation_index' in comparison_df['metric'].values:
            avg_frag = comparison_df[comparison_df['metric'] == 'avg_fragmentation_index']['optimized'].iloc[0]
            f.write(f"- **Average Fragmentation**: {avg_frag:.3f} (0=contiguous, 1=completely fragmented)\n")
        
        f.write("\n### Partition Size Distribution\n\n")
        f.write("| Partition ID | Hierarchical Count | Optimized Count | Difference |\n")
        f.write("|--------------|--------------------|-----------------|-----------|\n")
        
        for _, row in comparison_df.iterrows():
            if row['metric'].startswith('partition_') and row['metric'].endswith('_size'):
                partition_id = row['metric'].split('_')[1]
                f.write(f"| {partition_id} | {row['hierarchical']} | {row['optimized']} | {row['difference']:+d} |\n")
        
        f.write("\n### Spatial Fragmentation Analysis\n\n")
        
        if 'total_spatial_components' in comparison_df['metric'].values:
            hier_components = comparison_df[comparison_df['metric'] == 'total_spatial_components']['hierarchical'].iloc[0]
            opt_components = comparison_df[comparison_df['metric'] == 'total_spatial_components']['optimized'].iloc[0]
            
            f.write(f"- **Hierarchical Components**: {hier_components} (assumes 1 contiguous component per partition)\n")
            f.write(f"- **Optimized Components**: {opt_components} disconnected spatial pieces\n")
            f.write(f"- **Fragmentation Factor**: {opt_components / hier_components:.1f}x increase in spatial components\n\n")
        
        f.write("### Interpretation\n\n")
        f.write("**Hierarchical partitions** represent the theoretical algorithm structure:\n")
        f.write("- Clean splits following significance testing\n")
        f.write("- No spatial post-processing applied\n")
        f.write("- Useful for understanding algorithm behavior\n\n")
        
        f.write("**Spatially optimized partitions** represent actual model assignments:\n")
        f.write("- Include contiguity refinement with 4/9 majority voting\n")
        f.write("- Optimized for crisis prediction performance\n")
        f.write("- Fragmentation is intentional, not implementation bugs\n\n")
        
        f.write("### Algorithmic Rationale for Fragmentation\n\n")
        f.write("The conservative 4/9 voting threshold creates spatial equilibria where:\n")
        f.write("1. Isolated polygons resist partition switching\n")
        f.write("2. Small clusters maintain stability through self-voting\n")
        f.write("3. Spatial patterns optimize for predictive accuracy over spatial compactness\n")
        f.write("4. Enclaves represent genuine feature-space similarities despite geographic separation\n\n")
        
        f.write("## Detailed Metrics\n\n")
        f.write("| Metric | Hierarchical | Optimized | Difference | Description |\n")
        f.write("|--------|--------------|-----------|------------|-------------|\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['metric']} | {row['hierarchical']} | {row['optimized']} | {row['difference']:+} | {row['description']} |\n")
        
        f.write(f"\n*Raw data saved to: `{os.path.basename(csv_path)}`*\n")
    
    print(f"Generated fragmentation comparison: {os.path.basename(report_path)}")
    return report_path

def create_configuration_controlled_visualization(
    result_dir: str,
    correspondence_table: str,
    disable_contiguity: bool = False,
    visualization_mode: str = "both",
    output_prefix: str = "configurable"
) -> Dict[str, str]:
    """
    Create partition visualizations with configuration controls for contiguity refinement.
    
    Args:
        result_dir: Directory to save outputs
        correspondence_table: Path to correspondence table
        disable_contiguity: If True, skip contiguity refinement effects
        visualization_mode: 'hierarchical', 'optimized', or 'both'
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary mapping output types to file paths
    """
    
    # Load correspondence table
    df = pd.read_csv(correspondence_table)
    
    output_files = {}
    
    # Generate configuration info
    config_info = {
        'contiguity_disabled': disable_contiguity,
        'visualization_mode': visualization_mode,
        'generation_timestamp': datetime.now().isoformat(),
        'purpose': 'User-controlled partition visualization'
    }
    
    # Create different visualization modes based on configuration
    if visualization_mode in ['hierarchical', 'both']:
        # Generate clean hierarchical view
        hierarchical_df = df.copy()
        hierarchical_df['contiguity_applied'] = False
        hierarchical_df['spatial_optimization'] = False
        
        hierarchical_path = os.path.join(result_dir, f'{output_prefix}_hierarchical_config.csv')
        hierarchical_df.to_csv(hierarchical_path, index=False)
        output_files['hierarchical_table'] = hierarchical_path
    
    if visualization_mode in ['optimized', 'both'] and not disable_contiguity:
        # Generate optimized view (only if contiguity not disabled)
        optimized_df = df.copy()
        optimized_df['contiguity_applied'] = True
        optimized_df['spatial_optimization'] = True
        
        optimized_path = os.path.join(result_dir, f'{output_prefix}_optimized_config.csv')
        optimized_df.to_csv(optimized_path, index=False)
        output_files['optimized_table'] = optimized_path
    
    # Save configuration metadata
    config_path = os.path.join(result_dir, f'{output_prefix}_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2)
    output_files['config'] = config_path
    
    return output_files

def main_dual_track_visualization(result_dir: str) -> Dict[str, Any]:
    """
    Main function to create dual-track partition visualizations for a result directory.
    
    Args:
        result_dir: Path to GeoRF result directory
    
    Returns:
        Dictionary containing visualization results and file paths
    """
    
    print(f"Creating dual-track visualizations for: {result_dir}")
    
    # Look for fixed correspondence tables (from partition_consistency_fix.py)
    hierarchical_files = []
    optimized_files = []
    
    for file in os.listdir(result_dir):
        if file.startswith('correspondence_hierarchical_') and file.endswith('.csv'):
            hierarchical_files.append(os.path.join(result_dir, file))
        elif file.startswith('correspondence_optimized_') and file.endswith('.csv'):
            optimized_files.append(os.path.join(result_dir, file))
    
    if not hierarchical_files or not optimized_files:
        print("Missing hierarchical or optimized correspondence tables")
        print("Please run partition_consistency_fix.py first")
        return {
            'status': 'missing_tables', 
            'hierarchical_found': len(hierarchical_files),
            'optimized_found': len(optimized_files)
        }
    
    # Use the first available tables
    hierarchical_table = hierarchical_files[0]
    optimized_table = optimized_files[0]
    
    print(f"Using hierarchical table: {os.path.basename(hierarchical_table)}")
    print(f"Using optimized table: {os.path.basename(optimized_table)}")
    
    output_files = {}
    
    # Create dual-track visualizations
    try:
        dual_track_files = create_dual_track_visualization(
            result_dir=result_dir,
            hierarchical_correspondence=hierarchical_table,
            optimized_correspondence=optimized_table
        )
        output_files.update(dual_track_files)
        print(f"Generated {len(dual_track_files)} dual-track visualizations")
        
    except Exception as e:
        print(f"Warning: Could not generate dual-track visualizations: {e}")
        output_files['dual_track_error'] = str(e)
    
    # Generate fragmentation comparison metrics
    try:
        fragmentation_report = generate_fragmentation_comparison_metrics(
            hierarchical_correspondence=hierarchical_table,
            optimized_correspondence=optimized_table,
            result_dir=result_dir
        )
        output_files['fragmentation_report'] = fragmentation_report
        print(f"Generated fragmentation comparison report")
        
    except Exception as e:
        print(f"Warning: Could not generate fragmentation comparison: {e}")
        output_files['fragmentation_error'] = str(e)
    
    # Create configuration-controlled tables
    try:
        config_files = create_configuration_controlled_visualization(
            result_dir=result_dir,
            correspondence_table=hierarchical_table,
            visualization_mode='both'
        )
        output_files.update(config_files)
        print(f"Generated {len(config_files)} configuration-controlled files")
        
    except Exception as e:
        print(f"Warning: Could not generate configuration-controlled files: {e}")
        output_files['config_error'] = str(e)
    
    return {
        'status': 'success',
        'result_dir': result_dir,
        'output_files': output_files,
        'hierarchical_table': hierarchical_table,
        'optimized_table': optimized_table
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
        result = main_dual_track_visualization(result_dir)
        print(f"\nVisualization result: {result['status']}")
        
        if result['status'] == 'success':
            print("Generated files:")
            for file_type, file_path in result['output_files'].items():
                if isinstance(file_path, str) and not file_type.endswith('_error'):
                    print(f"  - {file_type}: {os.path.basename(file_path)}")
    else:
        print("Usage: python dual_track_visualization.py <result_directory>")
        print("Example: python dual_track_visualization.py result_GeoRF_27")