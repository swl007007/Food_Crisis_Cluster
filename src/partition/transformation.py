# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
# import tensorflow as tf
import pandas as pd
from scipy import stats

from config import *
from src.helper.helper import *
from src.model.train_branch import *
from src.partition.partition_opt import *
from src.tests.sig_test import *

from src.customize.customize import *

# MODULE LEVEL DEBUG: This should print when transformation.py is imported
import sys
sys.stderr.write("*** TRANSFORMATION.PY MODULE LOADED - DEBUG LOGGING ACTIVE ***\n")
sys.stderr.flush()

from src.vis.visualization import *

# Strict visualization gate resolver (prefer config_visual)
def _resolve_vis_flag(VIS_DEBUG_MODE=None):
  try:
    if VIS_DEBUG_MODE is None:
      try:
        from config_visual import VIS_DEBUG_MODE as V
      except ImportError:
        from config import VIS_DEBUG_MODE as V
      return bool(V)
    return bool(VIS_DEBUG_MODE)
  except Exception:
    return False

# import models

def generate_branch_visualization_correspondence(X, X_group, X_branch_id, branch_id,
                                               s0_group_refined, s1_group_refined,
                                               visualization_scope='branch_only'):
  """Generate correspondence rows for partition visualization."""

  partition_data = []

  # Normalise branch identifiers for lookup and child labels
  focus_branch = branch_id or ''
  child_zero = f"{focus_branch}0"
  child_one = f"{focus_branch}1"

  def _extract_valid_groups(groups):
    if groups is None:
      return []

    flat_groups = np.asarray(groups).ravel()
    valid = []
    for value in flat_groups:
      try:
        gid = int(value)
      except (TypeError, ValueError):
        continue
      if gid >= 0:
        valid.append(gid)
    return valid

  # Prepare group assignments derived from the refinement step
  s0_groups = _extract_valid_groups(s0_group_refined)
  s1_groups = _extract_valid_groups(s1_group_refined)

  if visualization_scope == 'branch_only':
    if not s0_groups and not s1_groups:
      return partition_data

    for gid in s0_groups:
      partition_data.append({
        'FEWSNET_admin_code': gid,
        'partition_id': child_zero
      })

    for gid in s1_groups:
      partition_data.append({
        'FEWSNET_admin_code': gid,
        'partition_id': child_one
      })

    return partition_data

  # Fallback to full-scope correspondence: include all records, but respect new assignments
  reassignment_lookup = {gid: child_zero for gid in s0_groups}
  reassignment_lookup.update({gid: child_one for gid in s1_groups})

  for idx in range(len(X)):
    admin_code = X_group[idx]
    current_branch = X_branch_id[idx]

    if admin_code in reassignment_lookup:
      refined_partition = reassignment_lookup[admin_code]
    else:
      refined_partition = current_branch if current_branch != '' else 'root'

    partition_data.append({
      'FEWSNET_admin_code': admin_code,
      'partition_id': refined_partition
    })

  return partition_data

def partition(model, X, y,
                     X_group , X_set, X_id, X_branch_id,
                     #group_loc = None,
                     X_loc = None,#this is optional for spatial smoothing (partition shape refinement)
                     min_depth = MIN_DEPTH,
                     max_depth = MAX_DEPTH,
                     refine_times = REFINE_TIMES,
                     contiguity_type = 'grid',
                     polygon_contiguity_info = None,
                     track_partition_metrics = False,
                     correspondence_table_path = None,
                     model_dir = None,  # Add model_dir parameter
                     **paras):

  '''
  **paras is for model-specific parameters (could be different for deep learning and traditional ML)
  maybe create two different versions?
  '''
  # Strict gate: take visualization flag only from arguments, never from config
  VIS_DEBUG_MODE = paras.get('VIS_DEBUG_MODE', False)
  print("="*50)
  print(f"TRANSFORMATION.PY DEBUG: partition() received VIS_DEBUG_MODE={VIS_DEBUG_MODE}")
  print(f"TRANSFORMATION.PY DEBUG: paras keys = {list(paras.keys())}")
  print("="*50)

  # Resolve model_dir for visualization artifacts and metrics export
  resolved_model_dir = model_dir or getattr(model, 'model_dir', None) or getattr(model, 'path', None)
  if resolved_model_dir is None:
    print("Warning: partition() could not resolve model_dir; visualization exports will be skipped")
  model_dir = resolved_model_dir

  # Initialize metrics tracker if requested
  metrics_tracker = None
  if track_partition_metrics:
    from src.metrics.metrics import PartitionMetricsTracker
    metrics_tracker = PartitionMetricsTracker(correspondence_table_path)
    print("Partition metrics tracking enabled")
  
  #branch_table is also something that can be returned if needed
  branch_table = np.zeros([2**max_depth, max_depth])
  branch_table[:,0:min_depth] = 1 #-1

  #init s list: a dictionary
  #all all cells to initial branch with id==''
  #dict #rows must be consistent
  s_branch, max_size_needed = init_s_branch(n_groups = N_GROUPS)#grid_dim = GRID_DIM

  # Train and save initial base model for the root branch (branch_id = '')
  print("Training initial base model for root branch...")
  train_list = get_id_list(X_branch_id, X_set, '', set_id = 0)
  val_list = get_id_list(X_branch_id, X_set, '', set_id = 1)
  X_train_base = X[train_list]
  y_train_base = y[train_list]
  model.train(X_train_base, y_train_base, branch_id='')
  model.save('')
  print(f"Initial base model trained with {len(y_train_base)} samples")
  
  # Generate baseline visualization for the root model (before any partitioning)
  if metrics_tracker is not None and correspondence_table_path:
    try:
      # Get validation data for baseline visualization
      X_val_base = X[val_list]
      y_val_base = y[val_list]
      X_group_base = X_group[val_list]
      
      # Get baseline predictions (same for before and after since no partitioning yet)
      y_pred_base = base_eval_using_merged_branch_data(model, X_val_base, '')
      
      # Record baseline metrics for round -1 (pre-partitioning)
      metrics_tracker.record_partition_metrics(
          partition_round=-1,
          branch_id='baseline',
          y_true=y_val_base,
          y_pred_before=y_pred_base,
          y_pred_after=y_pred_base,  # Same as before since no partitioning
          X_group=X_group_base,
          partition_type="baseline"
      )
      
      # Generate baseline visualization
      if model_dir:
        import os
        from src.vis.visualization import plot_metrics_improvement_map

        vis_dir = os.path.join(model_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        metrics_dir = os.path.join(model_dir, 'partition_metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        # Save baseline metrics to CSV
        baseline_csv = os.path.join(metrics_dir, "partition_metrics_baseline.csv")
        baseline_metrics = []
        if hasattr(metrics_tracker, 'all_metrics') and metrics_tracker.all_metrics:
          for record in metrics_tracker.all_metrics:
            if record['partition_round'] == -1 and record['branch_id'] == 'baseline':
              baseline_metrics.append(record)

        if baseline_metrics:
          import pandas as pd
          baseline_df = pd.DataFrame(baseline_metrics)
          baseline_df.to_csv(baseline_csv, index=False)

          # Create baseline performance maps (will show current performance levels)
          baseline_f1_path = os.path.join(vis_dir, "baseline_f1_performance_map.png")
          plot_metrics_improvement_map(
              baseline_csv,
              metric_type='f1_before',  # Show current F1 performance
              correspondence_table_path=correspondence_table_path,
              save_path=baseline_f1_path,
              title="Baseline F1 Performance (Before Partitioning)"
          )

          baseline_acc_path = os.path.join(vis_dir, "baseline_accuracy_performance_map.png")
          plot_metrics_improvement_map(
              baseline_csv,
              metric_type='accuracy_before',  # Show current accuracy performance
              correspondence_table_path=correspondence_table_path,
              save_path=baseline_acc_path,
              title="Baseline Accuracy Performance (Before Partitioning)"
          )

          print("Generated baseline performance maps in /vis/ directory")
      else:
        print("Warning: Skipping baseline metric exports because model_dir is undefined")
        
    except Exception as e:
      print(f"Warning: Could not generate baseline visualization: {e}")
  
  print(f"TRANSFORMATION.PY DEBUG: Starting partition loop with max_depth={max_depth}, range={max_depth-1}")
  for i in range(max_depth-1):

    num_branches = 2**i
    init_epoch_number = PRETRAIN_EPOCH + EPOCH_TRAIN * i
    print("Level %d --------------------------------------------------" % (i))

    #removed in the group-based version
    # #update step_size for creating grid
    # step_size = STEP_SIZE / (2**np.floor(i/2))

    for j in range(num_branches):

      if branch_table[j,i] == 0:
        continue

      #get branch_id
      branch_id = get_branch_id_for(i,j)

      #print only
      print("Level %d -- branch: %s --------------------------------------------------" % (len(branch_id), branch_id) )

      #get branch data
      train_list = get_id_list(X_branch_id, X_set, branch_id, set_id = 0)
      val_list = get_id_list(X_branch_id, X_set, branch_id, set_id = 1)

      if val_list[0].shape[0]==0:
        print('error! branch_id: ' + branch_id)
        print('train_list size: %d' % (train_list[0].shape[0]))


      #partition
      #key return: X0_train, y0_train, X0_val, y0_val, X1_train, y1_train, X1_val, y1_val

      #get train val data for branch
      X_val = X[val_list]
      y_val = y[val_list]

      #change model: evaluation function
      y_pred_before = base_eval_using_merged_branch_data(model, X_val, branch_id)
      del X_val

      #get s list
      #gid stores row and column ids for grid cells (here y_val_gid and true_pred_gid are the same set, order should be the same but double check if needed)
      #value stores cell-wise sum of per-class stats
      #need to make results returned by groupby have the same order
      #change stat: get stats function

      (y_val_gid, y_val_value,
       true_pred_gid, true_pred_value) = get_class_wise_stat(y_val, y_pred_before,
                                                             X_group[val_list])
                                                            #  X_loc[np.ix_(val_list[0], GRID_COLS)])#X_val_grid

      y_val_gid = np.asarray(y_val_gid)
      unique_gid_values, unique_gid_indices = np.unique(y_val_gid, return_index=True)

      # TODO: Verify if there are still data of interest left for selected class.
      #y_val_value should have shape (num_groups, n_class)
      # if np.sum(y_val_value) <= MIN_BRANCH_SAMPLE_SIZE:
      if (np.sum(y_val_value) <= MIN_BRANCH_SAMPLE_SIZE): #and i>=min_depth:
        print('selected classes: sample size too small! returning')
        continue
      # print('y_val_gid: ', y_val_gid)

      del y_val
      #s0 and s1 returned by scan are at grid cell level
      #correct
      # s0, s1 = scan(y_val_value, true_pred_value, MIN_SCAN_CLASS_SAMPLE)
      #incorrect

      RETURN_SCAN_SCORE = True
      if RETURN_SCAN_SCORE:
        # the error is calculated in get_c_b, no need to convert true to error here
        s0, s1, gscore, qscore = scan(y_val_value, true_pred_value, MIN_SCAN_CLASS_SAMPLE, return_score = RETURN_SCAN_SCORE)
        # s0, s1, gscore = scan(y_val_value, y_val_value * (1-true_pred_value), MIN_SCAN_CLASS_SAMPLE, return_score = RETURN_SCAN_SCORE)
      else:
        s0, s1 = scan(y_val_value, true_pred_value, MIN_SCAN_CLASS_SAMPLE)
        # s0, s1 = scan(y_val_value, y_val_value * (1-true_pred_value), MIN_SCAN_CLASS_SAMPLE)

      # s0_prev = s0
      # s1_prev = s1
      s0_before_contiguity = s0.copy()
      s1_before_contiguity = s1.copy()
      s0_group_before_contiguity = get_s_list_group_ids(s0_before_contiguity, y_val_gid)
      s1_group_before_contiguity = get_s_list_group_ids(s1_before_contiguity, y_val_gid)
      
      # create a dataframe for current s0 group, s1 group and their corresponding gscore/qscore, turned on if VIS_DEBUG_MODE
      if VIS_DEBUG_MODE:
        print(f"INFO: VIS_DEBUG_MODE=True, attempting to write score_details CSV for round {i} branch {branch_id}")
        try:
          import os
          if not model_dir:
            raise ValueError("model_dir is undefined; cannot write score_details CSV")
          vis_dir = os.path.join(model_dir, 'vis')
          os.makedirs(vis_dir, exist_ok=True)
          score_df_path = os.path.join(vis_dir, f'score_details_round_{i}_branch_{branch_id or "root"}.csv')
          gscore_arr = np.asarray(gscore, dtype=float)

          y_val_gid_unique = y_val_gid[unique_gid_indices]
          gscore_unique = gscore_arr[unique_gid_indices]

          s0_flags = np.isin(y_val_gid_unique, s0_group_before_contiguity).astype(int)
          s1_flags = np.isin(y_val_gid_unique, s1_group_before_contiguity).astype(int)

          print(f"Debug: writing score details for round {i}, branch '{branch_id}' with {len(y_val_gid_unique)} groups")

          score_df_dict = {
            'FEWSNET_admin_code': y_val_gid_unique,
            's0_partition': s0_flags,
            's1_partition': s1_flags,
            'gscore': gscore_unique
          }

          # Expand q-score information safely for CSV export. qscore is one value
          # per class; duplicate it across rows so the DataFrame shape matches.
          if qscore is not None:
            qscore_arr = np.asarray(qscore)

            if qscore_arr.ndim == 0:
              # Scalar risk multiplier - broadcast to all groups.
              score_df_dict['qscore'] = np.full(gscore_unique.shape, float(qscore_arr), dtype=float)
            elif qscore_arr.ndim == 1:
              for cls_idx, cls_q in enumerate(qscore_arr):
                score_df_dict[f'qscore_class_{cls_idx}'] = np.full(gscore_unique.shape, float(cls_q), dtype=float)
            else:
              # Unexpected shape - fall back to per-row max to avoid crashes and log it.
              flattened = float(np.max(qscore_arr))
              score_df_dict['qscore'] = np.full(gscore_unique.shape, flattened, dtype=float)

          score_df = pd.DataFrame(score_df_dict)
          score_df.to_csv(score_df_path, index=False)
          print(f"INFO: WRITE score_details_round_{i}_branch_{branch_id or 'root'}.csv -> {score_df_path} ({len(score_df)} rows)")
        except Exception as e:
          print(f"Warning: Could not save partition score details: {e}")
      else:
        print(f"INFO: VIS_DEBUG_MODE=False, skipping score_details CSV write for round {i} branch {branch_id}")
      
      if CONTIGUITY:
        if contiguity_type == 'polygon' and polygon_contiguity_info is not None:
          # Use polygon-based contiguity refinement
          
          # Store initial state for epoch 0 (before any refinement)
          s0_prev, s1_prev = s0.copy(), s1.copy()
          
          for i_refine in range(refine_times):
            print(f"Contiguity refinement epoch {i_refine + 1}/{refine_times} for branch {branch_id}")
            s0, s1 = get_refined_partitions_dispatcher(
              s0, s1, y_val_gid, None,
              dir=model.path,
              branch_id=branch_id,
              contiguity_type='polygon',
              polygon_centroids=polygon_contiguity_info['polygon_centroids'],
              polygon_group_mapping=polygon_contiguity_info['polygon_group_mapping'],
              neighbor_distance_threshold=polygon_contiguity_info.get('neighbor_distance_threshold'),
              adjacency_dict=polygon_contiguity_info.get('adjacency_dict')
            )
            
            # Generate partition map visualization for each refinement epoch
            try:
              import os
              from src.vis.visualization import plot_partition_map, plot_partition_swaps
              
              # Create vis directory if it doesn't exist (strict gate from caller)
              if not VIS_DEBUG_MODE:
                raise RuntimeError('VIS_DEBUG_MODE=False; skipping contiguity visualization')
              if not model_dir:
                raise ValueError('model_dir is undefined; cannot render contiguity refinement maps')
              vis_dir = os.path.join(model_dir, 'vis')
              os.makedirs(vis_dir, exist_ok=True)
              
              # Create temporary partition mapping for current refinement state
              s0_group_refined = get_s_list_group_ids(s0, y_val_gid)
              s1_group_refined = get_s_list_group_ids(s1, y_val_gid)
                
              
              # Create temporary correspondence table for current refinement state
              current_correspondence_path = os.path.join(vis_dir, f'temp_correspondence_round_{i}_branch_{branch_id or "root"}_refine_{i_refine + 1}.csv')
              
              # Generate correspondence table for branch visualization (only show the branch being split)
              import pandas as pd
              partition_data = generate_branch_visualization_correspondence(
                X, X_group, X_branch_id, branch_id, s0_group_refined, s1_group_refined, 
                visualization_scope='branch_only'  # Only show the current branch being split
              )
              
              # Create DataFrame and save
              partition_df = pd.DataFrame(partition_data)
              partition_df = partition_df.drop_duplicates()
              # CRITICAL: Ensure partition_id is string to preserve "00", "01" format
              if 'partition_id' in partition_df.columns:
                partition_df['partition_id'] = partition_df['partition_id'].astype(str)
              partition_df.to_csv(current_correspondence_path, index=False)
              
              # Generate partition map (strict gate)
              partition_map_path = os.path.join(vis_dir, f'contiguity_refinement_round_{i}_branch_{branch_id or "root"}_epoch_{i_refine + 1}.png')
              plot_partition_map(
                correspondence_table_path=current_correspondence_path,
                save_path=partition_map_path,
                title=f'Contiguity Refinement - Round {i}, Branch {branch_id if branch_id else "root"}, Epoch {i_refine + 1}/{refine_times}',
                figsize=(14, 12),
                VIS_DEBUG_MODE=VIS_DEBUG_MODE
              )
              print(f"Generated contiguity refinement map: {partition_map_path}")
              
              # Generate swap visualization (compare with previous epoch)
              if i_refine > 0 or True:  # Always generate, even for first epoch (compare with initial scan)
                # Create previous state correspondence table
                s0_group_prev = get_s_list_group_ids(s0_prev, y_val_gid)
                s1_group_prev = get_s_list_group_ids(s1_prev, y_val_gid)
                
                prev_correspondence_path = os.path.join(vis_dir, f'temp_correspondence_round_{i}_branch_{branch_id or "root"}_refine_{i_refine}.csv')
                
                # Generate correspondence table for previous state (only show the branch being split)
                prev_partition_data = generate_branch_visualization_correspondence(
                  X, X_group, X_branch_id, branch_id, s0_group_prev, s1_group_prev, 
                  visualization_scope='branch_only'  # Only show the current branch being split
                )
                
                prev_partition_df = pd.DataFrame(prev_partition_data)
                prev_partition_df = prev_partition_df.drop_duplicates()
                # CRITICAL: Ensure partition_id is string to preserve "00", "01" format
                if 'partition_id' in prev_partition_df.columns:
                  prev_partition_df['partition_id'] = prev_partition_df['partition_id'].astype(str)
                prev_partition_df.to_csv(prev_correspondence_path, index=False)
                
                # Generate swap visualization
                swap_map_path = os.path.join(vis_dir, f'contiguity_swaps_round_{i}_branch_{branch_id or "root"}_epoch_{i_refine}_to_{i_refine + 1}.png')
                
                try:
                  swap_fig = plot_partition_swaps(
                    correspondence_before_path=prev_correspondence_path,
                    correspondence_after_path=current_correspondence_path,
                    save_path=swap_map_path,
                    title=f'Partition Swaps - Round {i}, Branch {branch_id if branch_id else "root"}, Epoch {i_refine} -> {i_refine + 1}',
                    figsize=(14, 12),
                    VIS_DEBUG_MODE=VIS_DEBUG_MODE
                  )
                  if swap_fig is not None:
                    print(f"Generated contiguity swap map: {swap_map_path}")
                  else:
                    print(f"No swaps detected for Round {i}, Branch {branch_id}, Epoch {i_refine} -> {i_refine + 1}")
                except Exception as swap_error:
                  print(f"ERROR: Failed to create swap visualization: {swap_error}")
                  import traceback
                  traceback.print_exc()
                
                # Clean up previous correspondence table
                try:
                  os.remove(prev_correspondence_path)
                except:
                  pass
              
              # Clean up current correspondence table
              os.remove(current_correspondence_path)
              
              # Update previous state for next iteration
              s0_prev, s1_prev = s0.copy(), s1.copy()
              
            except Exception as e:
              print(f"Warning: Could not generate contiguity refinement map for Round {i}, Branch {branch_id}, Epoch {i_refine + 1}: {e}")
              
        else:
          # Use grid-based contiguity refinement (original implementation)
          group_loc = generate_groups_loc(X_DIM, STEP_SIZE)
          #group_loc = generate_groups_loc(X_loc, STEP_SIZE)
          # refine_times = REFINE_TIMES
          
          # Store initial state for epoch 0 (before any refinement)
          s0_prev, s1_prev = s0.copy(), s1.copy()
          
          for i_refine in range(refine_times):
            print(f"Contiguity refinement epoch {i_refine + 1}/{refine_times} for branch {branch_id}")
            #!!!s0 and s1 do not contain gid; instead, they contain indices from y_val_true (for gid need to use y_val_gid)
            s0, s1 = get_refined_partitions_dispatcher(s0, s1, y_val_gid, group_loc, dir = model.path, branch_id = branch_id, contiguity_type = 'grid')
            
            # Generate partition map visualization for each refinement epoch (grid-based)
            try:
              import os
              from src.vis.visualization import plot_partition_map, plot_partition_swaps
              
              # Create vis directory if it doesn't exist (strict gate from caller)
              if not VIS_DEBUG_MODE:
                raise RuntimeError('VIS_DEBUG_MODE=False; skipping contiguity visualization')
              if not model_dir:
                raise ValueError('model_dir is undefined; cannot render contiguity refinement maps')
              vis_dir = os.path.join(model_dir, 'vis')
              os.makedirs(vis_dir, exist_ok=True)
              
              # Create temporary partition mapping for current refinement state
              s0_group_refined = get_s_list_group_ids(s0, y_val_gid)
              s1_group_refined = get_s_list_group_ids(s1, y_val_gid)
              
              # Create temporary correspondence table for current refinement state
              current_correspondence_path = os.path.join(vis_dir, f'temp_correspondence_round_{i}_branch_{branch_id or "root"}_refine_{i_refine + 1}.csv')
              
              # Generate correspondence table for branch visualization (only show the branch being split)
              import pandas as pd
              partition_data = generate_branch_visualization_correspondence(
                X, X_group, X_branch_id, branch_id, s0_group_refined, s1_group_refined, 
                visualization_scope='branch_only'  # Only show the current branch being split
              )
              
              # Create DataFrame and save
              partition_df = pd.DataFrame(partition_data)
              partition_df = partition_df.drop_duplicates()
              # CRITICAL: Ensure partition_id is string to preserve "00", "01" format
              if 'partition_id' in partition_df.columns:
                partition_df['partition_id'] = partition_df['partition_id'].astype(str)
              partition_df.to_csv(current_correspondence_path, index=False)
              
              # Generate partition map
              partition_map_path = os.path.join(vis_dir, f'contiguity_refinement_round_{i}_branch_{branch_id or "root"}_epoch_{i_refine + 1}.png')
              
              plot_partition_map(
                correspondence_table_path=current_correspondence_path,
                save_path=partition_map_path,
                title=f'Contiguity Refinement - Round {i}, Branch {branch_id if branch_id else "root"}, Epoch {i_refine + 1}/{refine_times}',
                figsize=(14, 12),
                VIS_DEBUG_MODE=VIS_DEBUG_MODE
              )
              
              print(f"Generated contiguity refinement map: {partition_map_path}")
              
              # Generate swap visualization (compare with previous epoch)
              if i_refine > 0 or True:  # Always generate, even for first epoch (compare with initial scan)
                # Create previous state correspondence table
                s0_group_prev = get_s_list_group_ids(s0_prev, y_val_gid)
                s1_group_prev = get_s_list_group_ids(s1_prev, y_val_gid)
                
                prev_correspondence_path = os.path.join(vis_dir, f'temp_correspondence_round_{i}_branch_{branch_id or "root"}_refine_{i_refine}.csv')
                
                # Generate correspondence table for previous state (only show the branch being split)
                prev_partition_data = generate_branch_visualization_correspondence(
                  X, X_group, X_branch_id, branch_id, s0_group_prev, s1_group_prev, 
                  visualization_scope='branch_only'  # Only show the current branch being split
                )
                
                prev_partition_df = pd.DataFrame(prev_partition_data)
                prev_partition_df = prev_partition_df.drop_duplicates()
                # CRITICAL: Ensure partition_id is string to preserve "00", "01" format
                if 'partition_id' in prev_partition_df.columns:
                  prev_partition_df['partition_id'] = prev_partition_df['partition_id'].astype(str)
                prev_partition_df.to_csv(prev_correspondence_path, index=False)
                
                # Generate swap visualization
                swap_map_path = os.path.join(vis_dir, f'contiguity_swaps_round_{i}_branch_{branch_id or "root"}_epoch_{i_refine}_to_{i_refine + 1}.png')
                
                try:
                  swap_fig = plot_partition_swaps(
                    correspondence_before_path=prev_correspondence_path,
                    correspondence_after_path=current_correspondence_path,
                    save_path=swap_map_path,
                    title=f'Partition Swaps - Round {i}, Branch {branch_id if branch_id else "root"}, Epoch {i_refine} -> {i_refine + 1}',
                    figsize=(14, 12),
                    VIS_DEBUG_MODE=VIS_DEBUG_MODE
                  )
                  
                  if swap_fig is not None:
                    print(f"Generated contiguity swap map: {swap_map_path}")
                  else:
                    print(f"No swaps detected for Round {i}, Branch {branch_id}, Epoch {i_refine} -> {i_refine + 1}")
                    
                except Exception as swap_error:
                  print(f"ERROR: Failed to create swap visualization: {swap_error}")
                  import traceback
                  traceback.print_exc()
                
                # Clean up previous correspondence table
                try:
                  os.remove(prev_correspondence_path)
                except:
                  pass
              
              # Clean up current correspondence table
              os.remove(current_correspondence_path)
              
              # Update previous state for next iteration
              s0_prev, s1_prev = s0.copy(), s1.copy()
              
            except Exception as e:
              print(f"Warning: Could not generate contiguity refinement map for Round {i}, Branch {branch_id}, Epoch {i_refine + 1}: {e}")

      #debug
      # group_loc = generate_groups_loc(X_DIM, STEP_SIZE)
      # #group_loc = generate_groups_loc(X_loc, STEP_SIZE)
      # #s0_group = get_s_list_group_ids(s0, y_val_gid)
      # #s1_group = get_s_list_group_ids(s1, y_val_gid)
      # s0_debug, s1_debug = get_refined_partitions_dispatcher(s0, s1, y_val_gid, group_loc, dir = model.path, branch_id = branch_id, contiguity_type = 'grid')
      # print('#Debug: s0, s1; s0_refine, s1_refine: ', s0.shape, s1.shape, s0_debug.shape, s1_debug.shape)

      s0_group = get_s_list_group_ids(s0, y_val_gid)
      s1_group = get_s_list_group_ids(s1, y_val_gid)
      # if VIS_DEBUG_MODE, save the final s0_group and s1_group after contiguity refinement
      if VIS_DEBUG_MODE:
        print(f"INFO: VIS_DEBUG_MODE=True, attempting to write final_partitions CSV for round {i} branch {branch_id}")
        try:
          import os
          if not model_dir:
            raise ValueError("model_dir is undefined; cannot write final_partitions CSV")
          vis_dir = os.path.join(model_dir, 'vis')
          os.makedirs(vis_dir, exist_ok=True)
          final_partition_df_path = os.path.join(vis_dir, f'final_partitions_round_{i}_branch_{branch_id or "root"}.csv')
          final_gid_unique = y_val_gid[unique_gid_indices]
          final_s0_flags = np.isin(final_gid_unique, s0_group).astype(int)
          final_s1_flags = np.isin(final_gid_unique, s1_group).astype(int)
          print(f"Debug: writing final partitions for round {i}, branch '{branch_id}' with {len(final_gid_unique)} groups")
          final_partition_df = pd.DataFrame({
            'FEWSNET_admin_code': final_gid_unique,
            's0_partition': final_s0_flags,
            's1_partition': final_s1_flags
          })
          final_partition_df.to_csv(final_partition_df_path, index=False)
          print(f"INFO: WRITE final_partitions_round_{i}_branch_{branch_id or 'root'}.csv -> {final_partition_df_path} ({len(final_partition_df)} rows)")
        except Exception as e:
          print(f"Warning: Could not save final partition assignments: {e}")
      else:
        print(f"INFO: VIS_DEBUG_MODE=False, skipping final_partitions CSV write for round {i} branch {branch_id}")
          
      (X0_train, y0_train, X0_val, y0_val,
       X1_train, y1_train, X1_val, y1_val,
       s0_train, s1_train, s0_val, s1_val) = get_branch_data_by_group(X, y, X_group,
                                                                  train_list, val_list, s0_group, s1_group)
      # print('s0: ', s0)
      # print('s1: ', s1)
      # print('s0_group: ', s0_group)
      # print('s1_group: ', s1_group)

      # if (check_split_validity(X0_train, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue
      # if (check_split_validity(X1_train, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue
      # if (check_split_validity(X0_val, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue
      # if (check_split_validity(X1_val, MIN_BRANCH_SAMPLE_SIZE) == 0) and i>=min_depth:
      #   print('sample size too small! returning')
      #   continue

      if (check_split_validity(X0_train, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue
      if (check_split_validity(X1_train, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue
      if (check_split_validity(X0_val, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue
      if (check_split_validity(X1_val, MIN_BRANCH_SAMPLE_SIZE) == 0): #and i>=min_depth:
        print('sample size too small! returning')
        continue

      #train and eval hypothetical branches
      print("Training new branches...")#i and j splits are not used in train_and_eval_two_branch()
      y0_pred, y1_pred = train_and_eval_two_branch(model, X0_train, y0_train, X0_val,
                                                  X1_train, y1_train, X1_val, branch_id)

      sig = 1
      #test only if a new split will give a depth > MIN_DEPTH
      #otherwise directly split
      if (len(branch_id)+1) > min_depth or model.name in ('RF', 'DT'):
        print("Training base branch:")

        #get base branch data
        X_train = auto_hv_stack(X0_train, X1_train)
        X_val = auto_hv_stack(X0_val, X1_val)
        y_train = auto_hv_stack(y0_train, y1_train)
        y_val = auto_hv_stack(y0_val, y1_val)

        #y_pred_before is used for sig test
        y_pred_before = base_eval_using_merged_branch_data(model, X_val, branch_id)

        #align with y_preds' shape, otherwise (y_true - y_pred) will broadcast into 2D arrays (n x n)
        if len(y_val.shape) == 1 and MODE == 'regression':
          #update shapes for regression! The current version assumes y has shape (N,1) where N is the number of data points
          #addition: the expand_dims is also necessary for Reduction.None to function correctly in MSE calcualtion
          y_val = np.expand_dims(y_val, axis = 1)
          y0_val = np.expand_dims(y0_val, axis = 1)
          y1_val = np.expand_dims(y1_val, axis = 1)
          # y_pred = np.expand_dims(y_pred, axis = 1)

        #get stats
        base_score_before = get_score(y_val, y_pred_before)
        split_score0, split_score1 = get_split_score(y0_val, y0_pred, y1_val, y1_pred)

        #additional step for random forest
        #evaluate if previous fuller branch may outperform one sub-branch
        if model.name in ('RF', 'DT'):
          #for quick testing of design
          #not considering efficiency here
          if MODE == 'classification':
            y0_pred_before = base_eval_using_merged_branch_data(model, X0_val, branch_id)
            y1_pred_before = base_eval_using_merged_branch_data(model, X1_val, branch_id)
            split_score0_before, split_score1_before = get_split_score(y0_val, y0_pred_before, y1_val, y1_pred_before)
            print('effects of using only data in partition:')
            print('score before 0: ', np.mean(split_score0_before), 'score after 0: ', np.mean(split_score0))
            print('score before 1: ', np.mean(split_score1_before), 'score after 1: ', np.mean(split_score1))
            if np.mean(split_score0_before) >= np.mean(split_score0) and np.mean(split_score1_before) < np.mean(split_score1):
              #overwrite
              split_score0 = split_score0_before
              model.load(branch_id)
              model.save(branch_id + '0')
              print('overwrite branch', branch_id + '0', ' weights with branch', branch_id)

            elif np.mean(split_score0_before) < np.mean(split_score0) and np.mean(split_score1_before) >= np.mean(split_score1):
              #overwrite
              split_score1 = split_score1_before
              model.load(branch_id)
              model.save(branch_id + '1')
              print('overwrite branch', branch_id + '1', ' weights with branch', branch_id)
            elif ((len(branch_id)+1) <= min_depth and
                  np.mean(split_score0_before) >= np.mean(split_score0) and
                  np.mean(split_score1_before) >= np.mean(split_score1)):
              sig = 0

        #train and eval base branch
        if model.type == 'incremental':
          y_pred = train_and_eval_using_merged_branch_data(model, X_train, y_train, X_val, branch_id)
          base_score = get_score(y_val, y_pred)
        else:
          y_pred = None
          base_score = base_score_before#np.zeros(base_score_before.shape)

        #sig test
        if (len(branch_id)+1) > min_depth:
          #RF model will get into evaluation regardless of min_depth for weight selection
          sig = sig_test(base_score, split_score0, split_score1, base_score_before)
        else:
          print("Smaller than MIN_DEPTH, split directly...")

      else:
        #only DL model may reach here
        print("Smaller than MIN_DEPTH, split directly...")

      # Track metrics before and after partition decision
      if metrics_tracker is not None:
        # Get validation data for this branch
        X_val_combined = auto_hv_stack(X0_val, X1_val)
        y_val_combined = auto_hv_stack(y0_val, y1_val)
        # Get X_group values for validation samples in each partition
        val_indices_s0 = val_list[0][s0_val] if len(s0_val) > 0 else np.array([])
        val_indices_s1 = val_list[0][s1_val] if len(s1_val) > 0 else np.array([])
        
        X_group_s0 = X_group[val_indices_s0] if len(val_indices_s0) > 0 else np.array([])
        X_group_s1 = X_group[val_indices_s1] if len(val_indices_s1) > 0 else np.array([])
        
        X_group_combined = np.hstack([X_group_s0, X_group_s1])
        
        # Predictions before partition (from parent branch)
        y_pred_before_combined = base_eval_using_merged_branch_data(model, X_val_combined, branch_id)
        
        # Predictions after partition (concatenate from both child branches)
        y_pred_after_combined = np.hstack([y0_pred, y1_pred])
        
        # Record metrics for this partition round
        partition_type = "binary_split" if sig == 1 else "no_split"
        metrics_tracker.record_partition_metrics(
            partition_round=i,
            branch_id=branch_id,
            y_true=y_val_combined,
            y_pred_before=y_pred_before_combined,
            y_pred_after=y_pred_after_combined,
            X_group=X_group_combined,
            partition_type=partition_type
        )
        
        # Generate per-round visualization immediately after recording metrics
        # Save to /vis/ directory for debugging purposes
        if correspondence_table_path:
          try:
            import os
            from src.vis.visualization import plot_metrics_improvement_map

            # Use vis directory instead of partition_metrics for easier access
            if not model_dir:
              raise ValueError('model_dir is undefined; cannot render metrics improvement maps')
            vis_dir = os.path.join(model_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)

            # Also create partition_metrics directory for CSV files
            metrics_dir = os.path.join(model_dir, 'partition_metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Save current round metrics to CSV for visualization
            branch_suffix = f"branch{branch_id}" if branch_id else "branchroot"
            temp_csv_file = f"partition_metrics_round{i}_{branch_suffix}.csv"
            temp_csv_path = os.path.join(metrics_dir, temp_csv_file)
            
            # Get current partition data for this specific round and branch
            current_metrics = []
            if hasattr(metrics_tracker, 'all_metrics') and metrics_tracker.all_metrics:
              for record in metrics_tracker.all_metrics:
                if record['partition_round'] == i and record['branch_id'] == branch_id:
                  current_metrics.append(record)
            
            if current_metrics:
              import pandas as pd
              current_df = pd.DataFrame(current_metrics)
              current_df.to_csv(temp_csv_path, index=False)
              
              # Generate F1 improvement map in /vis/ directory  
              f1_map_path = os.path.join(vis_dir, f"round{i}_{branch_suffix}_f1_improvement_map.png")
              f1_map_file = plot_metrics_improvement_map(
                  temp_csv_path,
                  metric_type='f1_improvement',
                  correspondence_table_path=correspondence_table_path,
                  save_path=f1_map_path,
                  title=f"F1 Improvement - Round {i}, Branch {branch_id if branch_id else 'root'}"
              )
              
              # Generate accuracy improvement map in /vis/ directory
              acc_map_path = os.path.join(vis_dir, f"round{i}_{branch_suffix}_accuracy_improvement_map.png")  
              acc_map_file = plot_metrics_improvement_map(
                  temp_csv_path,
                  metric_type='accuracy_improvement', 
                  correspondence_table_path=correspondence_table_path,
                  save_path=acc_map_path,
                  title=f"Accuracy Improvement - Round {i}, Branch {branch_id if branch_id else 'root'}"
              )
              
              print(f"Generated per-round maps in /vis/ for Round {i}, Branch {branch_id}: F1 and Accuracy improvement maps")
              
          except Exception as e:
            print(f"Warning: Could not generate per-round visualization for Round {i}, Branch {branch_id}: {e}")

      #decision
      if sig == 1:
        print("+ Split %s into %s, %s" % (branch_id, branch_id+'0', branch_id+'1') )

        '''
        update X_branch_id
        scan version uses pre-stored s0 and s1 lists for train and val
        '''

        #update branch_id
        X0_train_id, X1_train_id, X0_val_id, X1_val_id = get_branch_X_id(X_id, train_list, val_list, s0_train, s1_train, s0_val, s1_val)
        X0_id = np.hstack([X0_train_id, X0_val_id])
        X1_id = np.hstack([X1_train_id, X1_val_id])

        print('X1_id.shape: ', X1_id.shape)

        if X0_id.shape[0]==0 or X1_id.shape[0]==0:
          print('error in getting branch_id! X0_id size: %d, X1_id size: %d' % (X0_id.shape[0], X1_id.shape[0]))

        X_branch_id = update_branch_id(X0_id, X_branch_id, branch_id + '0')
        X_branch_id = update_branch_id(X1_id, X_branch_id, branch_id + '1')

        # print('X_branch_id 1 count: ', np.where(X_branch_id == '1'))


        # update s table
        # need to make sure ids returned by groupby are consistent

        # FIX: Size arrays exactly to avoid trailing -1 values
        # Create arrays sized exactly to the data we have
        s0_grid_set = y_val_gid[s0] if s0.shape[0] > 0 else np.array([], dtype=np.int32)
        s1_grid_set = y_val_gid[s1] if s1.shape[0] > 0 else np.array([], dtype=np.int32)
        
        # Pad to max_size_needed only if necessary for compatibility, but fill with actual data
        if s0_grid_set.shape[0] < max_size_needed:
            # Pad with actual group IDs from parent branch to maintain inheritance
            parent_groups = s_branch[branch_id] if branch_id in s_branch else np.array([], dtype=np.int32)
            if len(parent_groups) > 0:
                # Take valid parent group IDs (not -1) to fill remaining slots if needed
                valid_parent_groups = parent_groups[parent_groups >= 0]
                if len(valid_parent_groups) > 0:
                    # Extend s0_grid_set to max_size_needed with inherited values
                    padded_s0 = np.full(max_size_needed, -1, dtype=np.int32)  
                    padded_s0[:s0_grid_set.shape[0]] = s0_grid_set
                    s0_grid_set = padded_s0
                else:
                    # No valid parent groups, use exact size
                    pass
            else:
                # No parent branch, use exact size
                pass
        
        if s1_grid_set.shape[0] < max_size_needed:
            # Similar logic for s1_grid_set
            parent_groups = s_branch[branch_id] if branch_id in s_branch else np.array([], dtype=np.int32)
            if len(parent_groups) > 0:
                valid_parent_groups = parent_groups[parent_groups >= 0]
                if len(valid_parent_groups) > 0:
                    padded_s1 = np.full(max_size_needed, -1, dtype=np.int32)
                    padded_s1[:s1_grid_set.shape[0]] = s1_grid_set
                    s1_grid_set = padded_s1
        
        s_branch[branch_id + '0'] = s0_grid_set
        s_branch[branch_id + '1'] = s1_grid_set

        #update branch_table and score_table
        if i+1 < max_depth:
          next_level_row_ids_for_new_branches = [branch_id_to_loop_id(branch_id+'0'), branch_id_to_loop_id(branch_id+'1')]
          branch_table[next_level_row_ids_for_new_branches, i+1] = 1

        # Generate partition map visualization after each partition round (strict gate)
        if VIS_DEBUG_MODE:
          try:
            import os
            import pandas as pd
            from src.vis.partition_map import render_round_map

            if not model_dir:
              raise ValueError('model_dir is undefined; cannot render partition maps')
            vis_dir = os.path.join(model_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)

            partition_data = generate_branch_visualization_correspondence(
              X, X_group, X_branch_id,
              branch_id,
              s0_group, s1_group,
              visualization_scope='branch_only'
            )

            if not partition_data:
              raise ValueError('Scoped branch correspondence is empty; cannot render map')

            partition_df = pd.DataFrame(partition_data)
            parent_scope = set()
            parent_scope.update([gid for gid in s0_group])
            parent_scope.update([gid for gid in s1_group])

            partition_map_path = os.path.join(
              vis_dir,
              f'partition_map_round_{i}_branch_{branch_id or "root"}_scoped.png'
            )

            render_round_map(
              partition_df,
              parent_scope_uids=parent_scope,
              parent_label=branch_id or 'root',
              round_id=i,
              save_path=partition_map_path,
              VIS_DEBUG_MODE=VIS_DEBUG_MODE
            )

            print(f"Generated scoped partition map: {partition_map_path}")

          except Exception as e:
            print(f"Warning: Could not generate scoped partition map for Round {i}, Branch {branch_id}: {e}")

        # vis_partition_training(grid, branch_id)
        # !!!the generate_vis_image() and generate_vis_image_for_all_groups() in the following can be added back to motinor partitioning in the training process
        #generate_vis_image(s_branch, X_branch_id, max_depth = max_depth, dir = model.path, step_size = STEP_SIZE, file_name = branch_id + '_split')
        #accuracy
        grid, vmin, vmax = generate_count_grid(true_pred_value/(y_val_value+0.0001), y_val_gid, class_id = 0, step_size = STEP_SIZE)
        #generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_acc_' + branch_id, vmin = vmin, vmax = vmax)

        #gscore (used in scan statistics)
        # print('#Debug: true_pred_value.shape, y_val_gid.shape: ', true_pred_value.shape, y_val_gid.shape)
        scan_gscore = np.expand_dims(np.hstack([gscore[s0], gscore[s1]]), axis = -1)
        grid, vmin, vmax = generate_count_grid(scan_gscore, np.hstack([s0_group, s1_group]), class_id = 0, step_size = STEP_SIZE)
        #generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_scan_' + branch_id, vmin = vmin, vmax = vmax)
        #gscore rank: ranking of each group in scan
        gscore_argsort = np.argsort(gscore,0)[::-1]
        gscore_rank = np.arange(gscore.shape[0])
        gscore_rank = gscore_rank[gscore_argsort.argsort()]
        #gscore_rank = gscore.shape[0] - 1 - gscore_rank
        scan_gscore_rank = np.expand_dims(np.hstack([gscore_rank[s0], gscore_rank[s1]]), axis = -1)
        grid, vmin, vmax = generate_count_grid(scan_gscore_rank, np.hstack([s0_group, s1_group]), class_id = 0, step_size = STEP_SIZE)
        #generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_rank_' + branch_id, vmin = vmin, vmax = vmax)
        #gscore>0
        scan_gscore_positive = (scan_gscore>0).astype(int)
        grid, vmin, vmax = generate_count_grid(scan_gscore_positive, np.hstack([s0_group, s1_group]), class_id = 0, step_size = STEP_SIZE)
        #generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_binary_' + branch_id, vmin = vmin, vmax = vmax)
        #count
        grid, vmin, vmax = generate_count_grid((y_val_value > MIN_GROUP_POS_SAMPLE_SIZE_FLEX).astype(int), y_val_gid, class_id = 0, step_size = STEP_SIZE)
        #generate_vis_image_for_all_groups(grid, dir = model.path, ext = '_cnt_' + branch_id, vmin = vmin, vmax = vmax)

      else:
        print("= Branch %s not split" % (branch_id) )

  # Save and return metrics if tracking was enabled
  if metrics_tracker is not None:
    # Save metrics to the model directory
    try:
      import os
      if not model_dir:
        raise ValueError('model_dir is undefined; cannot persist partition metrics')
      metrics_dir = os.path.join(model_dir, 'partition_metrics')
      os.makedirs(metrics_dir, exist_ok=True)
      metrics_tracker.save_metrics_to_csv(metrics_dir)
      
      # Create visualization dashboard if correspondence table is available
      if correspondence_table_path and VIS_DEBUG_MODE:
        from src.vis.visualization import plot_partition_metrics_dashboard
        dashboard_outputs = plot_partition_metrics_dashboard(
          metrics_tracker, 
          metrics_dir,
          correspondence_table_path=correspondence_table_path
        )
        print(f"Partition metrics dashboard created in: {metrics_dir}")
        print(f"Created files: {dashboard_outputs}")
        
        # Generate comprehensive summary maps in /vis/ directory for debugging
        try:
          if not model_dir:
            raise ValueError('model_dir is undefined; cannot persist visualization metrics')
          vis_dir = os.path.join(model_dir, 'vis')
          os.makedirs(vis_dir, exist_ok=True)
          
          # Create overall performance summary maps from all collected metrics
          if hasattr(metrics_tracker, 'all_metrics') and metrics_tracker.all_metrics:
            import pandas as pd
            all_metrics_df = pd.DataFrame(metrics_tracker.all_metrics)
            
            # Save comprehensive metrics CSV
            comprehensive_csv = os.path.join(vis_dir, "comprehensive_partition_metrics.csv")
            all_metrics_df.to_csv(comprehensive_csv, index=False)
            
            # Generate comprehensive improvement maps
            if len(all_metrics_df) > 0:
              from src.vis.visualization import plot_metrics_improvement_map
              
              # Overall F1 improvement map across all rounds
              overall_f1_path = os.path.join(vis_dir, "overall_f1_improvement_map.png")
              plot_metrics_improvement_map(
                  comprehensive_csv,
                  metric_type='f1_improvement',
                  correspondence_table_path=correspondence_table_path,
                  save_path=overall_f1_path,
                  title="Overall F1 Improvement Across All Partition Rounds"
              )
              
              # Overall accuracy improvement map across all rounds  
              overall_acc_path = os.path.join(vis_dir, "overall_accuracy_improvement_map.png")
              plot_metrics_improvement_map(
                  comprehensive_csv,
                  metric_type='accuracy_improvement',
                  correspondence_table_path=correspondence_table_path,
                  save_path=overall_acc_path,
                  title="Overall Accuracy Improvement Across All Partition Rounds"
              )
              
              # Final performance maps (after all partitioning)
              final_f1_path = os.path.join(vis_dir, "final_f1_performance_map.png")
              plot_metrics_improvement_map(
                  comprehensive_csv,
                  metric_type='f1_after',
                  correspondence_table_path=correspondence_table_path,
                  save_path=final_f1_path,
                  title="Final F1 Performance (After All Partitioning)"
              )
              
              final_acc_path = os.path.join(vis_dir, "final_accuracy_performance_map.png")
              plot_metrics_improvement_map(
                  comprehensive_csv,
                  metric_type='accuracy_after',
                  correspondence_table_path=correspondence_table_path,
                  save_path=final_acc_path,
                  title="Final Accuracy Performance (After All Partitioning)"
              )
              
              print(f"Generated comprehensive performance maps in /vis/ directory")
              print(f"Maps include: baseline, per-round, overall, and final performance visualizations")
              
        except Exception as vis_e:
          print(f"Warning: Could not create comprehensive /vis/ visualizations: {vis_e}")
        
    except Exception as e:
      print(f"ERROR: Could not save partition metrics: {e}")
      import traceback
      traceback.print_exc()
    
    if track_partition_metrics:
      return X_branch_id, branch_table, s_branch, metrics_tracker
    else:
      return X_branch_id, branch_table, s_branch

  # No metrics tracker: return core outputs
  return X_branch_id, branch_table, s_branch
