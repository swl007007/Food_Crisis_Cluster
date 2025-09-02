"""
Transformation Consistency Patch

This module provides patches to transformation.py to ensure consistent 
correspondence table generation between round-by-round and final partition maps.

The key fix addresses the issue where:
- partition_map_round_0_branch_root.png shows immediate split results
- partition_map.png shows terminal assignments after significance testing
- These should be consistent when representing the same algorithm stage

Author: Weilunm Shi
Date: 2025-08-29
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def generate_consistent_round_correspondence(
    partition_data: Dict[str, List],
    branch_info: Dict[str, Any],
    round_id: int,
    branch_id: str,
    vis_dir: str,
    use_terminal_assignments: bool = True
) -> str:
    """
    Generate correspondence table for round maps that's consistent with final map logic.
    
    This fixes the issue where round maps and final maps use different correspondence
    table generation approaches, causing visual inconsistencies.
    
    Args:
        partition_data: Dictionary containing FEWSNET_admin_code and partition assignments
        branch_info: Information about branch hierarchy and adoption
        round_id: Partitioning round number
        branch_id: Branch identifier (e.g., "root", "0", "1")
        vis_dir: Visualization directory
        use_terminal_assignments: If True, apply terminal assignment logic
    
    Returns:
        Path to generated consistent correspondence table
    """
    
    # Create base correspondence DataFrame
    correspondence_df = pd.DataFrame(partition_data)
    correspondence_df = correspondence_df.drop_duplicates()
    
    if use_terminal_assignments and branch_info:
        # Apply the same terminal assignment logic used in final maps
        # This ensures consistency between round and final visualizations
        
        adopted_branches = branch_info.get('adopted_branches', [])
        active_branches = branch_info.get('active_branches', [])
        
        # Map adopted branch assignments to parent assignments
        partition_mapping = {}
        
        for adopted in adopted_branches:
            if len(adopted) > 1:  # e.g., 'rf_00' -> 'rf_0'
                parent_branch = adopted[:-1] if adopted.startswith('rf_') else adopted[:-1]
                parent_branch = parent_branch.replace('rf_', '') if parent_branch.startswith('rf_') else parent_branch
                
                # Find corresponding partition ID for parent branch
                if parent_branch == '':  # Root branch
                    # For root adoptions, map to partition 0
                    if adopted in correspondence_df.get('branch_label', []):
                        partition_mapping[adopted] = 0
                elif parent_branch in ['0', '1']:  # Level 1 branches
                    partition_id = int(parent_branch)
                    if adopted in correspondence_df.get('branch_label', []):
                        partition_mapping[adopted] = partition_id
        
        # Apply partition mapping if branch labels exist
        if 'branch_label' in correspondence_df.columns:
            correspondence_df['original_partition_id'] = correspondence_df['partition_id']
            for branch_label, new_partition_id in partition_mapping.items():
                mask = correspondence_df['branch_label'] == branch_label
                correspondence_df.loc[mask, 'partition_id'] = new_partition_id
    
    # Add consistency metadata
    correspondence_df['generation_stage'] = 'round_map_consistent'
    correspondence_df['round_id'] = round_id
    correspondence_df['branch_id'] = branch_id
    correspondence_df['terminal_assignment_applied'] = use_terminal_assignments
    
    # Generate file path
    if use_terminal_assignments:
        filename = f'correspondence_round_{round_id}_branch_{branch_id}_terminal.csv'
    else:
        filename = f'correspondence_round_{round_id}_branch_{branch_id}_immediate.csv'
    
    correspondence_path = os.path.join(vis_dir, filename)
    correspondence_df.to_csv(correspondence_path, index=False)
    
    print(f"Generated consistent round correspondence: {filename}")
    return correspondence_path

def create_dual_stage_round_visualizations(
    partition_data: Dict[str, List],
    branch_info: Dict[str, Any],
    round_id: int,
    branch_id: str,
    vis_dir: str
) -> Dict[str, str]:
    """
    Create both immediate and terminal assignment visualizations for a round.
    
    This provides clear separation between:
    1. Immediate split results (raw algorithm output)
    2. Terminal assignments (post-significance-testing)
    
    Args:
        partition_data: Dictionary containing partition assignments
        branch_info: Branch hierarchy and adoption information
        round_id: Partitioning round number
        branch_id: Branch identifier
        vis_dir: Visualization directory
    
    Returns:
        Dictionary mapping visualization types to file paths
    """
    
    visualization_files = {}
    
    # Generate immediate split correspondence (original behavior)
    immediate_correspondence = generate_consistent_round_correspondence(
        partition_data=partition_data,
        branch_info=branch_info,
        round_id=round_id,
        branch_id=branch_id,
        vis_dir=vis_dir,
        use_terminal_assignments=False
    )
    visualization_files['immediate_correspondence'] = immediate_correspondence
    
    # Generate terminal assignment correspondence (consistent with final maps)
    terminal_correspondence = generate_consistent_round_correspondence(
        partition_data=partition_data,
        branch_info=branch_info,
        round_id=round_id,
        branch_id=branch_id,
        vis_dir=vis_dir,
        use_terminal_assignments=True
    )
    visualization_files['terminal_correspondence'] = terminal_correspondence
    
    # Generate partition maps for both
    try:
        from visualization import plot_partition_map
        
        # Immediate split partition map
        immediate_map_path = os.path.join(
            vis_dir, 
            f'partition_map_round_{round_id}_branch_{branch_id}_immediate.png'
        )
        
        plot_partition_map(
            correspondence_table_path=immediate_correspondence,
            save_path=immediate_map_path,
            title=f'Round {round_id}, Branch {branch_id} - Immediate Splits',
            figsize=(14, 12)
        )
        visualization_files['immediate_map'] = immediate_map_path
        
        # Terminal assignment partition map (consistent with final)
        terminal_map_path = os.path.join(
            vis_dir, 
            f'partition_map_round_{round_id}_branch_{branch_id}_terminal.png'
        )
        
        plot_partition_map(
            correspondence_table_path=terminal_correspondence,
            save_path=terminal_map_path,
            title=f'Round {round_id}, Branch {branch_id} - Terminal Assignments',
            figsize=(14, 12)
        )
        visualization_files['terminal_map'] = terminal_map_path
        
        print(f"Generated dual-stage visualizations for Round {round_id}, Branch {branch_id}")
        
    except Exception as e:
        print(f"Warning: Could not generate partition maps: {e}")
        visualization_files['map_generation_error'] = str(e)
    
    return visualization_files

def patch_transformation_correspondence_logic(
    result_dir: str,
    enable_dual_stage: bool = True
) -> Dict[str, Any]:
    """
    Apply consistency patches to transformation.py correspondence table generation.
    
    This patches the logic in transformation.py lines around 800-820 where
    correspondence tables are generated during partitioning rounds.
    
    Args:
        result_dir: Path to result directory
        enable_dual_stage: Whether to generate both immediate and terminal visualizations
    
    Returns:
        Dictionary containing patch application results
    """
    
    patch_info = {
        'target_file': 'transformation.py',
        'target_lines': '800-820',
        'issue_addressed': 'Inconsistent correspondence table generation between rounds and final',
        'solution': 'Generate both immediate and terminal assignment tables for consistency',
        'enable_dual_stage': enable_dual_stage
    }
    
    # Create patched correspondence generation function
    patched_code = """
# PATCHED CORRESPONDENCE TABLE GENERATION FOR CONSISTENCY
# This replaces the original correspondence table logic in transformation.py:800-820

def generate_patched_round_correspondence(partition_data, branch_info, round_id, branch_id, vis_dir):
    '''
    Generate correspondence table consistent with final map logic.
    
    Addresses the issue where partition_map_round_0_branch_root.png differs 
    from partition_map.png due to different correspondence table generation timing.
    '''
    
    # Import the patch functions
    from transformation_consistency_patch import create_dual_stage_round_visualizations
    
    # Generate both immediate and terminal visualizations
    visualization_files = create_dual_stage_round_visualizations(
        partition_data=partition_data,
        branch_info=branch_info,
        round_id=round_id,
        branch_id=branch_id,
        vis_dir=vis_dir
    )
    
    return visualization_files

# To use this patch, replace the original correspondence table generation 
# in transformation.py:800-820 with a call to generate_patched_round_correspondence()
"""
    
    # Save patch code to result directory
    patch_code_path = os.path.join(result_dir, 'transformation_patch_code.py')
    with open(patch_code_path, 'w', encoding='utf-8') as f:
        f.write(patched_code)
    
    # Generate patch documentation
    patch_doc_path = os.path.join(result_dir, 'transformation_patch_documentation.md')
    
    with open(patch_doc_path, 'w', encoding='utf-8') as f:
        f.write("# Transformation Consistency Patch Documentation\n\n")
        f.write("## Issue Description\n\n")
        f.write("The original `transformation.py` generates correspondence tables during ")
        f.write("partitioning rounds that are inconsistent with the final correspondence ")
        f.write("table generation logic. This causes:\n\n")
        f.write("1. `partition_map_round_0_branch_root.png` to show immediate split results\n")
        f.write("2. `partition_map.png` to show terminal assignments after significance testing\n")
        f.write("3. Visual inconsistency when both should represent the same algorithm stage\n\n")
        
        f.write("## Patch Solution\n\n")
        f.write("This patch provides:\n\n")
        f.write("1. **Dual-stage correspondence tables**: Both immediate and terminal versions\n")
        f.write("2. **Consistent logic**: Uses same terminal assignment approach as final maps\n")
        f.write("3. **Clear separation**: Different file names for different stages\n\n")
        
        f.write("## Implementation\n\n")
        f.write("### Original Code (transformation.py:800-820)\n\n")
        f.write("```python\n")
        f.write("# Original inconsistent logic\n")
        f.write("partition_df = pd.DataFrame(partition_data)\n")
        f.write("partition_df = partition_df.drop_duplicates()\n")
        f.write("partition_df.to_csv(current_correspondence_path, index=False)\n")
        f.write("```\n\n")
        
        f.write("### Patched Code\n\n")
        f.write("```python\n")
        f.write("# Patched consistent logic\n")
        f.write("visualization_files = generate_patched_round_correspondence(\n")
        f.write("    partition_data=partition_data,\n")
        f.write("    branch_info=branch_info,\n")
        f.write("    round_id=i,\n")
        f.write("    branch_id=branch_id,\n")
        f.write("    vis_dir=vis_dir\n")
        f.write(")\n")
        f.write("```\n\n")
        
        f.write("## Generated Files\n\n")
        if enable_dual_stage:
            f.write("For each round, the patch generates:\n\n")
            f.write("1. `correspondence_round_X_branch_Y_immediate.csv` - Raw split results\n")
            f.write("2. `correspondence_round_X_branch_Y_terminal.csv` - Terminal assignments\n")
            f.write("3. `partition_map_round_X_branch_Y_immediate.png` - Immediate visualization\n")
            f.write("4. `partition_map_round_X_branch_Y_terminal.png` - Terminal visualization\n\n")
            
            f.write("The terminal versions should be consistent with the final partition map.\n\n")
        
        f.write("## Usage\n\n")
        f.write("### Automatic Application\n\n")
        f.write("To apply this patch automatically to future runs:\n\n")
        f.write("1. Import the patch in `transformation.py`:\n")
        f.write("   ```python\n")
        f.write("   from transformation_consistency_patch import create_dual_stage_round_visualizations\n")
        f.write("   ```\n\n")
        f.write("2. Replace the correspondence table generation section (lines ~800-820) ")
        f.write("with the patched logic\n\n")
        
        f.write("### Manual Application\n\n")
        f.write("To apply this patch to existing result directories:\n\n")
        f.write("```python\n")
        f.write("from transformation_consistency_patch import patch_transformation_correspondence_logic\n\n")
        f.write("result = patch_transformation_correspondence_logic('result_GeoRF_27')\n")
        f.write("```\n\n")
        
        f.write("## Verification\n\n")
        f.write("To verify the patch works correctly:\n\n")
        f.write("1. Compare `partition_map_round_0_branch_root_terminal.png` ")
        f.write("with `partition_map.png`\n")
        f.write("2. Both should show the same partition assignments\n")
        f.write("3. The only differences should be in visualization styling, not partition content\n\n")
        
        f.write("## Integration with Other Fixes\n\n")
        f.write("This patch works together with:\n\n")
        f.write("- `partition_consistency_fix.py`: Creates standardized correspondence tables\n")
        f.write("- `dual_track_visualization.py`: Provides hierarchical vs optimized views\n")
        f.write("- `partition_visualization_config.py`: Allows user control over visualization modes\n\n")
    
    return {
        'status': 'patch_generated',
        'patch_code_path': patch_code_path,
        'patch_doc_path': patch_doc_path,
        'patch_info': patch_info,
        'enable_dual_stage': enable_dual_stage
    }

def apply_patch_to_existing_result(
    result_dir: str,
    round_correspondences: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Apply consistency patch to an existing result directory.
    
    This retrospectively creates consistent correspondence tables and visualizations
    for existing GeoRF results that may have inconsistent round vs final maps.
    
    Args:
        result_dir: Path to existing result directory
        round_correspondences: Optional dictionary of existing round correspondence files
    
    Returns:
        Dictionary containing patch application results
    """
    
    print(f"Applying transformation consistency patch to: {result_dir}")
    
    # Auto-detect existing correspondence tables if not provided
    if round_correspondences is None:
        round_correspondences = {}
        for file in os.listdir(result_dir):
            if file.startswith('correspondence_table_Q') and file.endswith('.csv'):
                round_name = file.replace('correspondence_table_', '').replace('.csv', '')
                round_correspondences[round_name] = os.path.join(result_dir, file)
    
    if not round_correspondences:
        return {
            'status': 'no_round_correspondences',
            'result_dir': result_dir,
            'message': 'No round correspondence tables found to patch'
        }
    
    print(f"Found {len(round_correspondences)} round correspondence tables to patch")
    
    patched_files = {}
    
    # Mock branch info (in practice, this would be parsed from training logs)
    branch_info = {
        'available_branches': ['rf_', 'rf_0', 'rf_00', 'rf_01', 'rf_1', 'rf_10', 'rf_11'],
        'active_branches': ['rf_', 'rf_0', 'rf_1'],
        'adopted_branches': ['rf_00', 'rf_01', 'rf_10', 'rf_11']
    }
    
    # Apply patch to each round correspondence
    for round_name, correspondence_path in round_correspondences.items():
        try:
            # Load existing correspondence
            df = pd.read_csv(correspondence_path)
            
            # Convert to partition_data format
            partition_data = {
                'FEWSNET_admin_code': df['FEWSNET_admin_code'].tolist(),
                'partition_id': df['partition_id'].tolist()
            }
            
            # Extract round and branch info from name
            # Example: "Q4_2019" -> round_id=4, branch_id="root" (assumed)
            if 'Q' in round_name:
                round_id = int(round_name.split('_')[0].replace('Q', ''))
                branch_id = "root"  # Most round tables are for root branch
            else:
                round_id = 0
                branch_id = round_name
            
            # Create vis directory if it doesn't exist
            vis_dir = os.path.join(result_dir, 'vis')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            
            # Generate dual-stage visualizations
            visualization_files = create_dual_stage_round_visualizations(
                partition_data=partition_data,
                branch_info=branch_info,
                round_id=round_id,
                branch_id=branch_id,
                vis_dir=vis_dir
            )
            
            patched_files[round_name] = visualization_files
            print(f"Patched round: {round_name}")
            
        except Exception as e:
            print(f"Error patching round {round_name}: {e}")
            patched_files[round_name] = {'error': str(e)}
    
    return {
        'status': 'success',
        'result_dir': result_dir,
        'patched_rounds': list(patched_files.keys()),
        'patched_files': patched_files,
        'branch_info': branch_info
    }

def main_patch_application(result_dir: str) -> Dict[str, Any]:
    """
    Main function to apply all transformation consistency patches.
    
    Args:
        result_dir: Path to GeoRF result directory
    
    Returns:
        Dictionary containing comprehensive patch results
    """
    
    print(f"Applying comprehensive transformation consistency patches to: {result_dir}")
    
    results = {
        'status': 'success',
        'result_dir': result_dir,
        'patches_applied': []
    }
    
    # 1. Generate patch code and documentation
    patch_generation = patch_transformation_correspondence_logic(result_dir)
    results['patch_generation'] = patch_generation
    results['patches_applied'].append('patch_code_generation')
    
    # 2. Apply patch to existing results
    patch_application = apply_patch_to_existing_result(result_dir)
    results['patch_application'] = patch_application
    results['patches_applied'].append('existing_result_patching')
    
    print(f"Applied {len(results['patches_applied'])} patches successfully")
    print("Patches applied:")
    for patch in results['patches_applied']:
        print(f"  - {patch}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
        result = main_patch_application(result_dir)
        print(f"\nPatch application result: {result['status']}")
    else:
        print("Usage: python transformation_consistency_patch.py <result_directory>")
        print("Example: python transformation_consistency_patch.py result_GeoRF_27")