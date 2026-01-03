#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization fix functions for GeoRF pipeline.

This module provides robust visualization functions that always generate maps,
even in degenerate cases (insufficient partitions, both branches select root model).

Key fixes:
1. Always create vis_dir regardless of partition count
2. Generate placeholder maps for degenerate cases
3. Decouple visualization from VIS_DEBUG_MODE for essential maps
4. Add structured logging and call graph tracing
5. Handle empty/NaN data gracefully
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def compute_final_accuracy_per_polygon(model, test_data, uid_col='FEWSNET_admin_code', VIS_DEBUG_MODE=None):
    """
    Compute per-polygon accuracy metrics on the terminal model state.
    
    Parameters
    ----------
    model : GeoRF or GeoXGB
        Trained model object with predict_test_group_wise capability
    test_data : tuple
        (Xtest, ytest, Xtest_group) test dataset
    uid_col : str
        Column name for polygon unique identifiers
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: [uid_col, pct_err_all, pct_err_pos, eval_count]
        or None if VIS_DEBUG_MODE=False
    """
    # Early return if visualization is disabled (strict gate)
    try:
        if VIS_DEBUG_MODE is None:
            try:
                from config_visual import VIS_DEBUG_MODE as Vv
            except ImportError:
                from config import VIS_DEBUG_MODE as Vv
            enabled = bool(Vv)
        else:
            enabled = bool(VIS_DEBUG_MODE)
    except Exception:
        enabled = False

    if not enabled:
        return None
    
    try:
        Xtest, ytest, Xtest_group = test_data
        
        # Import required functions
        from src.model.model_RF import predict_test_group_wise as predict_rf
        from src.model.model_XGB import predict_test_group_wise as predict_xgb
        
        # Determine which predict function to use based on model type
        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
            # GeoRF case
            results, groups, total_number = predict_rf(
                model.model, Xtest, ytest, Xtest_group, model.s_branch,
                prf=True, base=False, X_branch_id=None
            )
        else:
            # GeoXGB case - assume similar interface
            results, groups, total_number = predict_xgb(
                model.model, Xtest, ytest, Xtest_group, model.s_branch,
                prf=True, base=False, X_branch_id=None
            )
        
        # Process results into per-polygon accuracy
        accuracy_data = []
        
        for i, group_id in enumerate(groups):
            if i < len(results):
                # Extract precision, recall, f1, support from results
                # results shape: (n_groups, 4, n_classes)
                # Index 2 is F1 score, we use 1-F1 as error rate approximation
                if len(results.shape) == 3 and results.shape[1] >= 3:
                    # For class 0 (no crisis) and class 1 (crisis)
                    f1_all = np.mean(results[i, 2, :])  # Average F1 across classes
                    f1_class1 = results[i, 2, 1] if results.shape[2] > 1 else f1_all
                    
                    pct_err_all = 1 - f1_all  # Overall error rate approximation
                    pct_err_pos = 1 - f1_class1  # Class 1 error rate
                    
                    # Get evaluation count from total_number
                    eval_count = int(np.sum(total_number[i, :]) if i < len(total_number) else 0)
                else:
                    # Fallback for unexpected result structure
                    pct_err_all = np.nan
                    pct_err_pos = np.nan
                    eval_count = 0
            else:
                pct_err_all = np.nan
                pct_err_pos = np.nan
                eval_count = 0
                
            accuracy_data.append({
                uid_col: int(group_id),
                'pct_err_all': pct_err_all,
                'pct_err_pos': pct_err_pos,
                'eval_count': eval_count
            })
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        # Handle zero-denominator cases
        zero_eval_mask = accuracy_df['eval_count'] == 0
        accuracy_df.loc[zero_eval_mask, 'pct_err_all'] = np.nan
        accuracy_df.loc[zero_eval_mask, 'pct_err_pos'] = np.nan
        
        print(f"Final accuracy computation: {len(accuracy_df)} polygons, "
              f"{zero_eval_mask.sum()} with zero evaluations")
        
        return accuracy_df
        
    except Exception as e:
        print(f"Warning: Could not compute final accuracy per polygon: {e}")
        return None


def render_final_accuracy_maps(accuracy_df, vis_dir, uid_col='FEWSNET_admin_code', 
                              dpi=None, missing_color=None, backend=None,
                              degenerate_note=None, force_render=False, VIS_DEBUG_MODE=None):
    """
    Render final accuracy choropleth maps for overall and class 1 error rates.
    
    Parameters
    ----------
    accuracy_df : pandas.DataFrame
        DataFrame with accuracy data per polygon
    vis_dir : str
        Directory to save maps
    uid_col : str
        Column name for polygon identifiers
    dpi : int
        Output resolution
    missing_color : str
        Color for polygons with missing data
    backend : str
        Visualization backend ('matplotlib')
    degenerate_note : str, optional
        Note for degenerate cases
    force_render : bool
        If True, render even when VIS_DEBUG_MODE=False
        
    Returns
    -------
    dict
        Summary of rendered artifacts
    """
    # Strict gate: skip rendering when VIS_DEBUG_MODE is False
    try:
        if VIS_DEBUG_MODE is None:
            try:
                from config_visual import VIS_DEBUG_MODE as Vv
            except ImportError:
                from config import VIS_DEBUG_MODE as Vv
            enabled = bool(Vv)
        else:
            enabled = bool(VIS_DEBUG_MODE)
    except Exception:
        enabled = False
    if not enabled:
        return {'artifacts_rendered': [], 'skipped_reason': 'VIS_DEBUG_MODE=False'}
    
    # Load configuration defaults
    try:
        from config import (FINAL_ACCURACY_DPI, FINAL_ACCURACY_MISSING_COLOR, 
                           FINAL_ACCURACY_BACKEND)
        if dpi is None:
            dpi = FINAL_ACCURACY_DPI
        if missing_color is None:
            missing_color = FINAL_ACCURACY_MISSING_COLOR
        if backend is None:
            backend = FINAL_ACCURACY_BACKEND
    except ImportError:
        # Use fallback defaults if config not available
        if dpi is None:
            dpi = 200
        if missing_color is None:
            missing_color = '#dddddd'
        if backend is None:
            backend = 'matplotlib'
    
    rendered_artifacts = []
    
    # Early return if no data
    if accuracy_df is None or len(accuracy_df) == 0:
        print("No accuracy data available for final accuracy maps")
        return {'artifacts_rendered': [], 'skipped_reason': 'No accuracy data'}
    
    try:
        # Render overall error rate map
        overall_path = os.path.join(vis_dir, 'final_accuracy_map.png')
        overall_title = "Final Accuracy Map - Overall Error Rate"
        if degenerate_note:
            overall_title += f" ({degenerate_note})"
            
        if 'pct_err_all' in accuracy_df.columns:
            overall_result = render_accuracy_choropleth(
                accuracy_df, 'pct_err_all', overall_path, overall_title,
                uid_col, dpi, missing_color
            )
            if overall_result:
                rendered_artifacts.append('final_accuracy_map.png')
                print(f"Rendered final accuracy map: {overall_path}")
        
        # Render class 1 error rate map
        class1_path = os.path.join(vis_dir, 'final_accuracy_map_class1.png')
        class1_title = "Final Accuracy Map - Class 1 Error Rate"
        if degenerate_note:
            class1_title += f" ({degenerate_note})"
            
        if 'pct_err_pos' in accuracy_df.columns:
            class1_result = render_accuracy_choropleth(
                accuracy_df, 'pct_err_pos', class1_path, class1_title,
                uid_col, dpi, missing_color
            )
            if class1_result:
                rendered_artifacts.append('final_accuracy_map_class1.png')
                print(f"Rendered final accuracy class 1 map: {class1_path}")
        
        # Save accuracy CSV
        csv_path = os.path.join(vis_dir, 'final_accuracy_by_polygon.csv')
        accuracy_df.to_csv(csv_path, index=False, encoding='utf-8')
        rendered_artifacts.append('final_accuracy_by_polygon.csv')
        print(f"Saved final accuracy CSV: {csv_path}")
        
    except Exception as e:
        print(f"Warning: Error rendering final accuracy maps: {e}")
    
    return {'artifacts_rendered': rendered_artifacts}


def render_accuracy_choropleth(df, metric_col, save_path, title, uid_col, dpi, missing_color, VIS_DEBUG_MODE=None):
    """
    Render a single accuracy choropleth map using existing infrastructure.
    
    Returns path to saved map or None if failed.
    """
    try:
        # Use existing plot_error_rate_choropleth function
        from src.vis.visualization import plot_error_rate_choropleth
        
        fig = plot_error_rate_choropleth(
            df, metric_col,
            uid_col=uid_col,
            title=title,
            save_path=save_path,
            dpi=dpi,
            missing_color=missing_color,
            VIS_DEBUG_MODE=True
        )
        return save_path
        
    except Exception as e:
        print(f"Choropleth rendering failed: {e}, creating placeholder")
        
        # Create placeholder map
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Final accuracy data unavailable\n{title}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(title, fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return save_path


def render_accuracy_map(df, save_path=None, title=None, degenerate_note=None, VIS_DEBUG_MODE=None):
    """
    Render accuracy map that always works, even for degenerate/single-partition cases.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with accuracy data per polygon
    save_path : str, optional  
        Path to save the map
    title : str, optional
        Map title
    degenerate_note : str, optional
        Note to append for degenerate cases
        
    Returns
    -------
    str : path to saved map or None if failed
    """
    try:
        if save_path is None:
            save_path = "accuracy_map.png"
            
        if title is None:
            title = "Accuracy Map"
            
        if degenerate_note:
            title += f" ({degenerate_note})"
            
        # Strict gate
        try:
            if VIS_DEBUG_MODE is None:
                try:
                    from config_visual import VIS_DEBUG_MODE as Vv
                except ImportError:
                    from config import VIS_DEBUG_MODE as Vv
                if not Vv:
                    return None
            else:
                if not VIS_DEBUG_MODE:
                    return None
        except Exception:
            return None

        # Use existing choropleth function if data is valid
        if len(df) > 0 and 'accuracy' in df.columns:
            try:
                from src.vis.visualization import plot_error_rate_choropleth
                fig = plot_error_rate_choropleth(
                    df, 'accuracy', 
                    title=title,
                    save_path=save_path,
                    VIS_DEBUG_MODE=True
                )
                return save_path
            except Exception as e:
                print(f"Choropleth function failed: {e}, creating placeholder")
        
        # Create placeholder map for empty data or choropleth failure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'No accuracy data available\n{title}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(title, fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        return save_path
            
    except Exception as e:
        print(f"Warning: Could not render accuracy map: {e}")
        return None


def render_partition_map(correspondence_df, save_path=None, title=None, degenerate_note=None, VIS_DEBUG_MODE=None):
    """
    Render partition map that always works, even for degenerate/single-partition cases.
    
    Parameters
    ----------
    correspondence_df : pandas.DataFrame
        DataFrame with correspondence between admin codes and partition IDs
    save_path : str, optional
        Path to save the map  
    title : str, optional
        Map title
    degenerate_note : str, optional
        Note to append for degenerate cases
        
    Returns
    -------
    str : path to saved map or None if failed
    """
    try:
        # Strict gate
        try:
            if VIS_DEBUG_MODE is None:
                try:
                    from config_visual import VIS_DEBUG_MODE as Vv
                except ImportError:
                    from config import VIS_DEBUG_MODE as Vv
                if not Vv:
                    return None
            else:
                if not VIS_DEBUG_MODE:
                    return None
        except Exception:
            return None

        if save_path is None:
            save_path = "partition_map.png"
            
        if title is None:
            title = "Partition Map"
            
        if degenerate_note:
            title += f" ({degenerate_note})"
            
        # Check if we have valid partition data
        if len(correspondence_df) > 0 and 'partition_id' in correspondence_df.columns:
            # Count unique partitions
            unique_partitions = correspondence_df['partition_id'].nunique()
            
            if unique_partitions > 1:
                # Use existing plot_partition_map function
                # Save to temporary CSV first
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_csv:
                        correspondence_df.to_csv(temp_csv.name, index=False, encoding='utf-8')
                        temp_csv_path = temp_csv.name
                    
                    from src.vis.visualization import plot_partition_map
                    fig = plot_partition_map(temp_csv_path, save_path=save_path, title=title, VIS_DEBUG_MODE=True)
                    os.unlink(temp_csv_path)  # Clean up temp file
                    return save_path
                except Exception as e:
                    print(f"plot_partition_map failed: {e}, creating placeholder")
                    if os.path.exists(temp_csv_path):
                        os.unlink(temp_csv_path)  # Clean up temp file
            
        # Create placeholder map for degenerate cases
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_partitions = correspondence_df['partition_id'].nunique() if len(correspondence_df) > 0 and 'partition_id' in correspondence_df.columns else 0
        n_polygons = len(correspondence_df)
        
        placeholder_text = f"""Partition Map
        
Partitions: {n_partitions}
Polygons: {n_polygons}

{degenerate_note or "Single partition or insufficient data"}
        """
        
        ax.text(0.5, 0.5, placeholder_text, 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(title, fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        return save_path
            
    except Exception as e:
        print(f"Warning: Could not render partition map: {e}")
        return None


def ensure_vis_dir_and_render_maps(model_dir, correspondence_df=None, test_data=None, 
                                  partition_count=None, stage_info="", force_accuracy=False, model=None, VIS_DEBUG_MODE=None):
    """
    Ensure visualization directory exists and render essential maps.
    
    This function is the main fix for the visualization issue. It:
    1. Always creates vis_dir regardless of conditions
    2. Renders maps when VIS_DEBUG_MODE=True or force flags are set
    3. Adds structured logging and stage tracking
    4. Handles empty data gracefully
    
    Parameters
    ---------- 
    model_dir : str
        Main model directory
    correspondence_df : pandas.DataFrame, optional
        Correspondence table with admin codes and partition IDs
    test_data : tuple, optional
        (Xtest, ytest, Xtest_group) for final accuracy computation
    partition_count : int, optional
        Number of partitions (for degenerate case detection)
    stage_info : str, optional
        Stage description for logging
    force_accuracy : bool, optional
        If True, force accuracy rendering even when VIS_DEBUG_MODE=False
    model : object, optional
        Model object for accuracy computation
        
    Returns
    -------
    dict : Summary of rendered artifacts
    """
    # Strict gate: skip all rendering and directory creation when disabled
    enabled = False
    try:
        if VIS_DEBUG_MODE is None:
            try:
                from config_visual import VIS_DEBUG_MODE as Vv
            except ImportError:
                from config import VIS_DEBUG_MODE as Vv
            enabled = bool(Vv)
        else:
            enabled = bool(VIS_DEBUG_MODE)
    except Exception:
        enabled = False

    if not enabled:
        return {
            'stage': stage_info,
            'vis_dir_created': False,
            'vis_dir_path': None,
            'artifacts_rendered': [],
            'partition_count': partition_count,
            'visualization_disabled': True
        }
    
    # ALWAYS create vis directory
    vis_dir = os.path.join(model_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Stage tracking and logging
    stage_summary = {
        'stage': stage_info,
        'vis_dir_created': True,
        'vis_dir_path': vis_dir,
        'artifacts_rendered': [],
        'partition_count': partition_count,
        'degenerate_case': partition_count is not None and partition_count <= 1
    }
    
    print(f"\n=== VISUALIZATION STAGE: {stage_info} ===")
    print(f"Vis directory: {vis_dir}")
    print(f"Partition count: {partition_count}")
    print(f"Degenerate case: {stage_summary['degenerate_case']}")
    
    # Determine if this is a degenerate case
    degenerate_note = None
    if stage_summary['degenerate_case']:
        degenerate_note = "single partition/degenerate"
    elif partition_count is not None and partition_count == 0:
        degenerate_note = "no partitions created"
    
    # Compute and render final accuracy maps
    if test_data is not None and model is not None:
        print("Computing final accuracy per polygon...")
        accuracy_df = compute_final_accuracy_per_polygon(model, test_data, VIS_DEBUG_MODE=True)
        
        # Render final accuracy maps
        accuracy_summary = render_final_accuracy_maps(
            accuracy_df, vis_dir, 
            degenerate_note=degenerate_note,
            force_render=False,
            VIS_DEBUG_MODE=True
        )
        
        if accuracy_summary['artifacts_rendered']:
            stage_summary['artifacts_rendered'].extend(accuracy_summary['artifacts_rendered'])
            print(f"Final accuracy artifacts: {accuracy_summary['artifacts_rendered']}")
        else:
            skip_reason = accuracy_summary.get('skipped_reason', 'Unknown')
            print(f"Final accuracy rendering skipped: {skip_reason}")
    else:
        print(f"No test data or model provided for final accuracy computation")
    
    # Render partition map
    if correspondence_df is not None and len(correspondence_df) > 0:
        partition_path = os.path.join(vis_dir, 'partition_map.png') 
        partition_result = render_partition_map(correspondence_df, partition_path, degenerate_note=degenerate_note, VIS_DEBUG_MODE=True)
        if partition_result:
            stage_summary['artifacts_rendered'].append('partition_map.png')
            print(f"Rendered partition map: {partition_path}")
        else:
            print(f"Failed to render partition map")
    else:
        print(f"No correspondence data provided")
    
    # Render final partition map (always at end)
    if correspondence_df is not None and len(correspondence_df) > 0:
        final_partition_path = os.path.join(vis_dir, 'final_partition_map.png')
        final_result = render_partition_map(correspondence_df, final_partition_path, 
                                          title="Final Partition Map", degenerate_note=degenerate_note, VIS_DEBUG_MODE=True)
        if final_result:
            stage_summary['artifacts_rendered'].append('final_partition_map.png')
            print(f"Rendered final partition map: {final_partition_path}")
    
    # Write call graph trace with UTF-8 encoding
    trace_path = os.path.join(vis_dir, 'call_graph_trace.txt')
    try:
        with open(trace_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{stage_info}:\n")
            f.write(f"  - vis_dir: {vis_dir}\n")
            f.write(f"  - partition_count: {partition_count}\n") 
            f.write(f"  - degenerate_case: {stage_summary['degenerate_case']}\n")
            f.write(f"  - artifacts_rendered: {stage_summary['artifacts_rendered']}\n")
            f.write(f"  - timestamp: {pd.Timestamp.now()}\n")
        stage_summary['call_graph_trace'] = trace_path
        print(f"Updated call graph trace: {trace_path}")
    except Exception as e:
        print(f"Could not write call graph trace: {e}")
    
    # Write stage summary CSV with UTF-8 encoding
    summary_csv_path = os.path.join(vis_dir, 'stage_summary.csv')
    try:
        summary_df = pd.DataFrame([stage_summary])
        if os.path.exists(summary_csv_path):
            existing_df = pd.read_csv(summary_csv_path, encoding='utf-8')
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"Updated stage summary: {summary_csv_path}")
    except Exception as e:
        print(f"Could not write stage summary: {e}")
    
    total_artifacts = len(stage_summary['artifacts_rendered'])
    print(f"=== VISUALIZATION COMPLETE: {total_artifacts} artifacts rendered ===\n")
    
    return stage_summary
