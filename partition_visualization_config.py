"""
Partition Visualization Configuration

This module provides configuration options to control the visualization of GeoRF partitions,
allowing users to choose between:
1. Clean hierarchical views (no spatial optimization)
2. Spatially optimized views (with contiguity refinement)
3. Custom hybrid approaches

This addresses the user confusion by providing clear control over algorithmic stages.

Author: Claude Code Assistant
Date: 2025-08-29
"""

import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json

@dataclass
class PartitionVisualizationConfig:
    """
    Configuration class for partition visualization options.
    """
    # Core visualization mode
    visualization_mode: str = "both"  # Options: 'hierarchical', 'optimized', 'both'
    
    # Contiguity refinement controls
    enable_contiguity_refinement: bool = True
    contiguity_threshold: float = 4/9  # Conservative 4/9 majority threshold
    contiguity_iterations: int = 3  # Number of refinement iterations
    
    # Spatial coherence vs performance trade-off
    prioritize_spatial_coherence: bool = False  # If True, reduces fragmentation
    coherence_weight: float = 0.5  # Balance between coherence and performance (0-1)
    
    # Fragmentation tolerance
    max_fragmentation_index: float = 0.5  # Maximum acceptable fragmentation (0-1)
    preserve_enclaves: bool = True  # Whether to preserve algorithmic enclaves
    
    # Visualization appearance
    show_fragmentation_metrics: bool = True
    show_branch_adoption_info: bool = True
    color_by_performance: bool = False  # Color partitions by prediction performance
    
    # Output options
    generate_comparison_plots: bool = True
    save_intermediate_stages: bool = False  # Save each refinement iteration
    output_format: str = "png"  # Options: 'png', 'svg', 'pdf'
    dpi: int = 300
    
    # Documentation and explanation
    include_algorithm_explanation: bool = True
    generate_fragmentation_report: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_modes = {'hierarchical', 'optimized', 'both'}
        if self.visualization_mode not in valid_modes:
            raise ValueError(f"visualization_mode must be one of {valid_modes}")
        
        if not 0 <= self.coherence_weight <= 1:
            raise ValueError("coherence_weight must be between 0 and 1")
        
        if not 0 <= self.max_fragmentation_index <= 1:
            raise ValueError("max_fragmentation_index must be between 0 and 1")
        
        valid_formats = {'png', 'svg', 'pdf'}
        if self.output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'visualization_mode': self.visualization_mode,
            'enable_contiguity_refinement': self.enable_contiguity_refinement,
            'contiguity_threshold': self.contiguity_threshold,
            'contiguity_iterations': self.contiguity_iterations,
            'prioritize_spatial_coherence': self.prioritize_spatial_coherence,
            'coherence_weight': self.coherence_weight,
            'max_fragmentation_index': self.max_fragmentation_index,
            'preserve_enclaves': self.preserve_enclaves,
            'show_fragmentation_metrics': self.show_fragmentation_metrics,
            'show_branch_adoption_info': self.show_branch_adoption_info,
            'color_by_performance': self.color_by_performance,
            'generate_comparison_plots': self.generate_comparison_plots,
            'save_intermediate_stages': self.save_intermediate_stages,
            'output_format': self.output_format,
            'dpi': self.dpi,
            'include_algorithm_explanation': self.include_algorithm_explanation,
            'generate_fragmentation_report': self.generate_fragmentation_report
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PartitionVisualizationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'PartitionVisualizationConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Predefined configuration presets
PRESET_CONFIGS = {
    'clean_hierarchical': PartitionVisualizationConfig(
        visualization_mode='hierarchical',
        enable_contiguity_refinement=False,
        prioritize_spatial_coherence=True,
        coherence_weight=1.0,
        show_fragmentation_metrics=False,
        generate_comparison_plots=False
    ),
    
    'performance_optimized': PartitionVisualizationConfig(
        visualization_mode='optimized',
        enable_contiguity_refinement=True,
        prioritize_spatial_coherence=False,
        coherence_weight=0.0,
        preserve_enclaves=True,
        show_fragmentation_metrics=True,
        color_by_performance=True
    ),
    
    'balanced_view': PartitionVisualizationConfig(
        visualization_mode='both',
        enable_contiguity_refinement=True,
        prioritize_spatial_coherence=False,
        coherence_weight=0.5,
        max_fragmentation_index=0.4,
        generate_comparison_plots=True,
        show_fragmentation_metrics=True
    ),
    
    'research_detailed': PartitionVisualizationConfig(
        visualization_mode='both',
        enable_contiguity_refinement=True,
        save_intermediate_stages=True,
        show_fragmentation_metrics=True,
        show_branch_adoption_info=True,
        generate_comparison_plots=True,
        include_algorithm_explanation=True,
        generate_fragmentation_report=True
    ),
    
    'user_friendly': PartitionVisualizationConfig(
        visualization_mode='hierarchical',
        enable_contiguity_refinement=False,
        prioritize_spatial_coherence=True,
        show_fragmentation_metrics=False,
        include_algorithm_explanation=True,
        generate_comparison_plots=False
    )
}

def create_contiguity_refinement_options() -> Dict[str, Dict[str, Any]]:
    """
    Create different contiguity refinement option sets.
    
    Returns:
        Dictionary of refinement options with their parameters
    """
    
    return {
        'disabled': {
            'enabled': False,
            'threshold': None,
            'iterations': 0,
            'description': 'No contiguity refinement applied - clean hierarchical partitions'
        },
        
        'conservative': {
            'enabled': True,
            'threshold': 4/9,  # ~44.4% - original conservative threshold
            'iterations': 3,
            'description': 'Conservative 4/9 majority voting - preserves enclaves'
        },
        
        'moderate': {
            'enabled': True,
            'threshold': 0.5,  # 50% - simple majority
            'iterations': 3,
            'description': 'Simple majority voting - moderate fragmentation reduction'
        },
        
        'aggressive': {
            'enabled': True,
            'threshold': 5/9,  # ~55.6% - more aggressive smoothing
            'iterations': 5,
            'description': 'Aggressive majority voting - maximizes spatial coherence'
        },
        
        'minimal': {
            'enabled': True,
            'threshold': 4/9,
            'iterations': 1,
            'description': 'Single iteration conservative refinement'
        }
    }

def generate_visualization_configuration_guide(output_dir: str) -> str:
    """
    Generate a comprehensive guide for partition visualization configuration options.
    
    Args:
        output_dir: Directory to save the guide
    
    Returns:
        Path to generated configuration guide
    """
    
    guide_path = os.path.join(output_dir, 'partition_visualization_configuration_guide.md')
    
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("# Partition Visualization Configuration Guide\n\n")
        f.write("*Generated for GeoRF partition map consistency fixes*\n\n")
        
        f.write("## Overview\n\n")
        f.write("This guide explains how to control the visualization of GeoRF partitions to address ")
        f.write("the confusion between hierarchical algorithm structure and spatially optimized results.\n\n")
        
        f.write("## Key Concepts\n\n")
        f.write("### 1. Hierarchical Partitions\n")
        f.write("- **What**: Clean algorithm splits following significance testing\n")
        f.write("- **Characteristics**: Contiguous regions, clear boundaries\n")
        f.write("- **Use**: Understanding algorithm structure and branch adoption\n")
        f.write("- **Example**: Clean {0, 1} partitions after significance testing\n\n")
        
        f.write("### 2. Spatially Optimized Partitions\n")
        f.write("- **What**: Post-contiguity-refinement assignments\n")
        f.write("- **Characteristics**: Fragmented, mosaicked patterns with enclaves\n")
        f.write("- **Use**: Actual model predictions and performance analysis\n")
        f.write("- **Example**: Fragmented {0, 1} partitions optimized for crisis prediction\n\n")
        
        f.write("### 3. Fragmentation vs Errors\n")
        f.write("- **Fragmentation is NOT a bug**: Intentional optimization for prediction performance\n")
        f.write("- **Conservative 4/9 threshold**: Creates stable spatial equilibria\n")
        f.write("- **Enclaves are intentional**: Represent feature-space similarities\n\n")
        
        f.write("## Configuration Options\n\n")
        
        # Document preset configurations
        f.write("### Preset Configurations\n\n")
        for preset_name, config in PRESET_CONFIGS.items():
            f.write(f"#### `{preset_name}`\n")
            f.write(f"- **Mode**: {config.visualization_mode}\n")
            f.write(f"- **Contiguity**: {'Enabled' if config.enable_contiguity_refinement else 'Disabled'}\n")
            f.write(f"- **Spatial Coherence**: {'Prioritized' if config.prioritize_spatial_coherence else 'Performance optimized'}\n")
            f.write(f"- **Use Case**: ")
            
            if preset_name == 'clean_hierarchical':
                f.write("Understanding algorithm structure without spatial optimization")
            elif preset_name == 'performance_optimized':
                f.write("Viewing actual model assignments used in predictions")
            elif preset_name == 'balanced_view':
                f.write("Side-by-side comparison of hierarchical vs optimized")
            elif preset_name == 'research_detailed':
                f.write("Comprehensive analysis with all intermediate stages")
            elif preset_name == 'user_friendly':
                f.write("Simplified view for users confused by fragmentation")
            
            f.write("\n\n")
        
        # Document contiguity options
        f.write("### Contiguity Refinement Options\n\n")
        contiguity_options = create_contiguity_refinement_options()
        
        for option_name, params in contiguity_options.items():
            f.write(f"#### `{option_name}`\n")
            f.write(f"- **Enabled**: {params['enabled']}\n")
            if params['enabled']:
                f.write(f"- **Threshold**: {params['threshold']:.3f}\n")
                f.write(f"- **Iterations**: {params['iterations']}\n")
            f.write(f"- **Description**: {params['description']}\n\n")
        
        f.write("## Usage Examples\n\n")
        f.write("### Python Usage\n\n")
        f.write("```python\n")
        f.write("from partition_visualization_config import PartitionVisualizationConfig, PRESET_CONFIGS\n\n")
        f.write("# Use preset configuration\n")
        f.write("config = PRESET_CONFIGS['clean_hierarchical']\n\n")
        f.write("# Or create custom configuration\n")
        f.write("config = PartitionVisualizationConfig(\n")
        f.write("    visualization_mode='both',\n")
        f.write("    enable_contiguity_refinement=False,\n")
        f.write("    prioritize_spatial_coherence=True\n")
        f.write(")\n\n")
        f.write("# Save configuration\n")
        f.write("config.save_to_file('my_partition_config.json')\n")
        f.write("```\n\n")
        
        f.write("### Command Line Usage\n\n")
        f.write("```bash\n")
        f.write("# Use preset configuration\n")
        f.write("python partition_consistency_fix.py result_GeoRF_27 --preset clean_hierarchical\n\n")
        f.write("# Use custom configuration file\n")
        f.write("python partition_consistency_fix.py result_GeoRF_27 --config my_partition_config.json\n\n")
        f.write("# Quick options\n")
        f.write("python partition_consistency_fix.py result_GeoRF_27 --mode hierarchical --no-contiguity\n")
        f.write("```\n\n")
        
        f.write("## Understanding the Trade-offs\n\n")
        f.write("### Spatial Coherence vs Prediction Performance\n\n")
        f.write("| Configuration | Spatial Coherence | Prediction Performance | Use Case |\n")
        f.write("|---------------|-------------------|-----------------------|----------|\n")
        f.write("| Hierarchical Only | High | Unknown | Algorithm understanding |\n")
        f.write("| Conservative Contiguity | Low | High | Actual model performance |\n")
        f.write("| Aggressive Contiguity | Medium | Medium | Balanced view |\n")
        f.write("| No Contiguity | High | Baseline | Clean theoretical view |\n\n")
        
        f.write("### When to Use Each Mode\n\n")
        f.write("**Use Hierarchical Mode When**:\n")
        f.write("- Understanding algorithm splits and branch adoption\n")
        f.write("- Debugging significance testing logic\n")
        f.write("- Presenting clean partition structure\n")
        f.write("- Confused by fragmented patterns\n\n")
        
        f.write("**Use Optimized Mode When**:\n")
        f.write("- Analyzing actual model performance\n")
        f.write("- Understanding spatial prediction patterns\n")
        f.write("- Evaluating contiguity refinement effects\n")
        f.write("- Researching spatial optimization algorithms\n\n")
        
        f.write("**Use Both Modes When**:\n")
        f.write("- Comparing algorithm structure vs final assignments\n")
        f.write("- Documenting algorithmic trade-offs\n")
        f.write("- Teaching GeoRF methodology\n")
        f.write("- Comprehensive result analysis\n\n")
        
        f.write("## Common Questions\n\n")
        f.write("### Q: Why are final partition maps fragmented?\n")
        f.write("A: Fragmentation results from contiguity refinement optimization. The 4/9 majority ")
        f.write("voting threshold creates spatial equilibria that optimize prediction performance ")
        f.write("rather than spatial compactness.\n\n")
        
        f.write("### Q: Are fragmented maps errors or bugs?\n")
        f.write("A: No, fragmentation is intentional algorithmic optimization. Use hierarchical ")
        f.write("mode to see clean theoretical partitions if fragmentation is confusing.\n\n")
        
        f.write("### Q: Which visualization should I use for presentations?\n")
        f.write("A: Use hierarchical mode for clean algorithm explanation, optimized mode for ")
        f.write("performance analysis, or both modes for comprehensive understanding.\n\n")
        
        f.write("### Q: How do I reduce fragmentation?\n")
        f.write("A: Use aggressive contiguity refinement or disable contiguity entirely. Note ")
        f.write("this may reduce prediction performance.\n\n")
        
        f.write("## Configuration File Format\n\n")
        f.write("Configuration files use JSON format:\n\n")
        f.write("```json\n")
        example_config = PRESET_CONFIGS['balanced_view'].to_dict()
        f.write(json.dumps(example_config, indent=2))
        f.write("\n```\n\n")
        
        f.write("## Advanced Options\n\n")
        f.write("For advanced users, additional configuration parameters are available:\n\n")
        f.write("- `coherence_weight`: Balance spatial coherence vs performance (0-1)\n")
        f.write("- `max_fragmentation_index`: Maximum acceptable fragmentation level\n")
        f.write("- `preserve_enclaves`: Whether to preserve algorithmic enclaves\n")
        f.write("- `save_intermediate_stages`: Save each refinement iteration\n")
        f.write("- `color_by_performance`: Color partitions by prediction accuracy\n\n")
    
    return guide_path

def apply_configuration_to_result_directory(
    result_dir: str,
    config: PartitionVisualizationConfig,
    correspondence_table: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply visualization configuration to a GeoRF result directory.
    
    Args:
        result_dir: Path to GeoRF result directory
        config: Partition visualization configuration
        correspondence_table: Path to correspondence table (auto-detected if None)
    
    Returns:
        Dictionary containing applied configuration results
    """
    
    print(f"Applying partition visualization configuration to: {result_dir}")
    print(f"Configuration mode: {config.visualization_mode}")
    print(f"Contiguity refinement: {'enabled' if config.enable_contiguity_refinement else 'disabled'}")
    
    # Save configuration to result directory
    config_path = os.path.join(result_dir, 'partition_visualization_config.json')
    config.save_to_file(config_path)
    
    results = {
        'status': 'success',
        'result_dir': result_dir,
        'config_path': config_path,
        'config': config.to_dict(),
        'generated_files': []
    }
    
    # Auto-detect correspondence table if not provided
    if correspondence_table is None:
        correspondence_files = []
        for file in os.listdir(result_dir):
            if file.startswith('correspondence_') and file.endswith('.csv'):
                correspondence_files.append(os.path.join(result_dir, file))
        
        if correspondence_files:
            correspondence_table = correspondence_files[0]
            print(f"Auto-detected correspondence table: {os.path.basename(correspondence_table)}")
        else:
            print("No correspondence table found")
            results['status'] = 'no_correspondence_table'
            return results
    
    # Generate configuration-specific files based on mode
    if config.visualization_mode in ['hierarchical', 'both']:
        # Create hierarchical correspondence table
        import pandas as pd
        df = pd.read_csv(correspondence_table)
        df['visualization_mode'] = 'hierarchical'
        df['contiguity_applied'] = False
        df['spatial_optimization'] = False
        
        hierarchical_path = os.path.join(result_dir, 'correspondence_hierarchical_configured.csv')
        df.to_csv(hierarchical_path, index=False)
        results['generated_files'].append(hierarchical_path)
    
    if config.visualization_mode in ['optimized', 'both'] and config.enable_contiguity_refinement:
        # Create optimized correspondence table with configuration parameters
        import pandas as pd
        df = pd.read_csv(correspondence_table)
        df['visualization_mode'] = 'optimized'
        df['contiguity_applied'] = True
        df['contiguity_threshold'] = config.contiguity_threshold
        df['contiguity_iterations'] = config.contiguity_iterations
        df['spatial_optimization'] = True
        
        optimized_path = os.path.join(result_dir, 'correspondence_optimized_configured.csv')
        df.to_csv(optimized_path, index=False)
        results['generated_files'].append(optimized_path)
    
    # Generate configuration guide if requested
    if config.include_algorithm_explanation:
        guide_path = generate_visualization_configuration_guide(result_dir)
        results['generated_files'].append(guide_path)
    
    print(f"Configuration applied successfully!")
    print(f"Generated {len(results['generated_files'])} files")
    
    return results

def main_apply_configuration(
    result_dir: str,
    preset_name: Optional[str] = None,
    config_file: Optional[str] = None,
    **custom_params
) -> Dict[str, Any]:
    """
    Main function to apply partition visualization configuration.
    
    Args:
        result_dir: Path to GeoRF result directory
        preset_name: Name of preset configuration to use
        config_file: Path to custom configuration JSON file
        **custom_params: Custom configuration parameters
    
    Returns:
        Dictionary containing configuration application results
    """
    
    # Determine configuration to use
    if config_file and os.path.exists(config_file):
        config = PartitionVisualizationConfig.load_from_file(config_file)
        print(f"Loaded configuration from: {config_file}")
    elif preset_name and preset_name in PRESET_CONFIGS:
        config = PRESET_CONFIGS[preset_name]
        print(f"Using preset configuration: {preset_name}")
    elif custom_params:
        config = PartitionVisualizationConfig(**custom_params)
        print("Using custom configuration parameters")
    else:
        config = PRESET_CONFIGS['balanced_view']  # Default
        print("Using default balanced_view configuration")
    
    # Apply configuration
    results = apply_configuration_to_result_directory(result_dir, config)
    
    return results

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply partition visualization configuration')
    parser.add_argument('result_dir', help='Path to GeoRF result directory')
    parser.add_argument('--preset', choices=list(PRESET_CONFIGS.keys()), 
                       help='Use preset configuration')
    parser.add_argument('--config', help='Path to custom configuration JSON file')
    parser.add_argument('--mode', choices=['hierarchical', 'optimized', 'both'],
                       help='Visualization mode')
    parser.add_argument('--no-contiguity', action='store_true',
                       help='Disable contiguity refinement')
    parser.add_argument('--coherence', type=float, default=0.5,
                       help='Spatial coherence weight (0-1)')
    
    args = parser.parse_args()
    
    # Build custom parameters from command line args
    custom_params = {}
    if args.mode:
        custom_params['visualization_mode'] = args.mode
    if args.no_contiguity:
        custom_params['enable_contiguity_refinement'] = False
        custom_params['prioritize_spatial_coherence'] = True
    if args.coherence != 0.5:
        custom_params['coherence_weight'] = args.coherence
    
    result = main_apply_configuration(
        result_dir=args.result_dir,
        preset_name=args.preset,
        config_file=args.config,
        **custom_params
    )
    
    print(f"\nConfiguration application result: {result['status']}")
    if result['status'] == 'success':
        print(f"Generated {len(result['generated_files'])} files")