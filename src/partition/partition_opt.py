# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

#notes: added semantic segmentation

import numpy as np
# import tensorflow as tf
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

from scipy import ndimage
# from skimage import measure

# from partition_opt import swap_partition_general
from src.customize.customize import generate_groups_loc
# from visualization import vis_partition_group, generate_vis_image_from_grid
from src.vis.visualization import *
from src.helper.helper import *

from config import *

# Crisis-focused scoring functions
def get_class_1_f1_score(y_true, y_pred):
    """Calculate class 1 F1 score for crisis prediction optimization."""
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_true_labels, y_pred_labels = y_true, y_pred
    return f1_score(y_true_labels, y_pred_labels, pos_label=1, zero_division=0)

def get_class_1_precision_score(y_true, y_pred):
    """Calculate class 1 precision score for crisis prediction optimization."""
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_true_labels, y_pred_labels = y_true, y_pred
    return precision_score(y_true_labels, y_pred_labels, pos_label=1, zero_division=0)

def get_class_1_recall_score(y_true, y_pred):
    """Calculate class 1 recall score for crisis prediction optimization."""
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_true_labels, y_pred_labels = y_true, y_pred
    return recall_score(y_true_labels, y_pred_labels, pos_label=1, zero_division=0)

def get_metric_score_array(y_true, y_pred, metric_name=None):
    """Calculate per-sample scores based on governing metric."""
    if metric_name is None:
        metric_name = GOVERNING_METRIC if 'GOVERNING_METRIC' in globals() else 'overall_accuracy'
    
    if metric_name == 'class_1_f1':
        # For F1, we need to calculate per-sample contributions
        # This is an approximation - return 1.0 for correct class 1 predictions, 0.0 otherwise
        if len(y_true.shape) > 1:
            y_true_labels = np.argmax(y_true, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_true_labels, y_pred_labels = y_true, y_pred
        
        # Return 1.0 for correct class 1 predictions, 0.0 for all others
        correct_class_1 = (y_true_labels == 1) & (y_pred_labels == 1)
        return correct_class_1.astype(float)
    
    elif metric_name == 'overall_accuracy':
        # Original behavior
        if len(y_true.shape) > 1:
            return (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).astype(int)
        else:
            return (y_true == y_pred).astype(int)
    
    else:
        # Fallback to overall accuracy
        if len(y_true.shape) > 1:
            return (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).astype(int)
        else:
            return (y_true == y_pred).astype(int)

def groupby_sum(y, y_group, onehot = ONEHOT):
  try:
    y = y.astype(int)
    y_group = y_group.reshape([-1,1])
    
    # Ensure y has the right shape for hstack
    if len(y.shape) == 1:
        y = y.reshape([-1, 1])
    
    y = np.hstack([y_group, y])

    # CRITICAL FIX: Create DataFrame with explicit dtypes to prevent hashtable corruption
    y_df = pd.DataFrame(y, dtype=np.int64)  # Force int64 to avoid int32 hashtable issues
    
    # Use explicit column naming to avoid groupby issues
    y_df.columns = ['group_id'] + [f'col_{i}' for i in range(y_df.shape[1]-1)]
    
    # Perform groupby with explicit handling
    try:
      y_grouped = y_df.groupby('group_id', sort=False).sum()
    except KeyError as e:
      if 'int32' in str(e):
        print("Warning: Pandas hashtable issue detected, using numpy-based groupby fallback")
        # Fallback to numpy-based groupby
        unique_groups, indices = np.unique(y_group.flatten(), return_inverse=True)
        grouped_sums = []
        for i, group_id in enumerate(unique_groups):
          mask = (indices == i)
          group_sum = y[mask, 1:].sum(axis=0)
          grouped_sums.append(group_sum)
        
        return unique_groups, np.array(grouped_sums)
      else:
        raise

    return y_grouped.index.to_numpy(), y_grouped.to_numpy()
    
  except Exception as e:
    print(f"Error in groupby_sum: {e}")
    print(f"Input shapes: y={y.shape}, y_group={y_group.shape}")
    print(f"Input dtypes: y={y.dtype}, y_group={y_group.dtype}")
    raise

def get_class_wise_stat(y_true, y_pred, y_group, mode = MODE, onehot = ONEHOT):

  if mode == 'classification':
    # n_sample = y_true.shape[0]

    if len(y_true.shape)==1:
      # y_true = tf.one_hot(y_true, NUM_CLASS)
      # y_pred = tf.one_hot(y_pred, NUM_CLASS)
      y_true = np.eye(NUM_CLASS)[y_true.astype(int)].astype(int)
      y_pred = np.eye(NUM_CLASS)[y_pred.astype(int)].astype(int)
    # else:
    # #this is to make coding consistent (tf functions might be used in this function when implementing the RF version)
    #   y_true = tf.convert_to_tensor(y_true)
    #   y_pred = tf.convert_to_tensor(y_pred)
    #   # tf.convert_to_tensor(numpy_array, dtype=tf.float32)

    #reshape image or time-series labels
    #can handle shapes of N x m x m x k, where m is img size for semantic segmentation
    #or shapes of N x t x k, where t is the length of a sequence
    if len(y_true.shape)>=3:
      # n_dims = len(y_true.shape)
      # data_point_size = 1
      # for dim in range(1,n_dims-1):
      #   data_point_size *= y_true.shape[dim]
      n_pre = y_true.shape[0]
      y_true = y_true.reshape(-1, NUM_CLASS)
      y_pred = y_pred.reshape(-1, NUM_CLASS)
      # y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
      # y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])
      n_after = y_true.shape[0]

      data_point_size = int(n_after/n_pre)
      y_group = y_group.astype(int)
      y_group = np.repeat(y_group, data_point_size)

    # Crisis-focused optimization: use governing metric instead of overall accuracy
    if 'CRISIS_FOCUSED_OPTIMIZATION' in globals() and CRISIS_FOCUSED_OPTIMIZATION:
        stat = get_metric_score_array(y_true, y_pred, GOVERNING_METRIC)
    else:
        stat = (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).astype(int)
    # stat = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    # stat = stat.numpy()
    # y_true = np.array(y_true)

    #select_class (should not be used before categorical_accuracy,
    #which is not correct when there is only one class)
    #here only use aggregated c and b, so only need to select subset of columns
    if SELECT_CLASS is not None:
      y_true = y_true[:, SELECT_CLASS]

    # TODO: Check what happens if y_true is all zeros (already entered the optimization phase)

    true_pred_w_class = y_true * np.expand_dims(stat, 1)

    #group by groups
    y_true_group, y_true_value = groupby_sum(y_true, y_group)
    true_pred_group, true_pred_value = groupby_sum(true_pred_w_class, y_group)

    return y_true_group, y_true_value, true_pred_group, true_pred_value

  else:
    # #reshape image or time-series labels
    # if len(y_true.shape)>=3:
    #   y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
    #   y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

    #may (or may not) need to revise for regression using scan methods!!!
    # stat = tf.keras.losses.MSE(y_true, y_pred)
    # stat = stat.numpy()
    score = np.square(y_true - y_pred)
    stat_id, stat_value = groupby_sum(stat, y_group)

    return stat_id, stat_value

def get_c_b(y_true_value, true_pred_value):
  '''This calculates the total C and B.
  '''
  c = y_true_value - true_pred_value
  base = y_true_value

  c_tot = np.sum(c,0)
  base_tot = np.sum(base,0)

  b = np.expand_dims(c_tot,0) * (base / np.expand_dims(base_tot,0))
  b = np.nan_to_num(b)

  return c,b

def get_min_max_size(cnt, flex_ratio):
  #this can be optimized with a binary search
  #since the total time cost on this should be minimal, using a linear scan at the moment
  cnt_total = np.sum(cnt)
  # print('cnt.shape, cnt_total: ', cnt.shape, cnt_total)
  cnt_to_i = 0
  min_size = -1
  max_size = -1
  for i in range(cnt.shape[0]):
    cnt_to_i += cnt[i]
    if min_size == -1 and cnt_to_i / cnt_total >= flex_ratio:
      min_size = i
    if max_size == -1 and cnt_to_i / cnt_total >= 1 - flex_ratio:
      max_size = i-1
      break

  if max_size < min_size:
    print('max_size < min_size: ', min_size, max_size)
    max_size = min_size

  return min_size, max_size


def optimize_size(g, set_size, flex_ratio, flex_type = FLEX_TYPE, cnt = None):

  sorted_g_score = np.sort(g)
  sorted_g_score = sorted_g_score[::-1]

  # if cnt is not None:
  #   print('#Debug: np.stack([sorted_g_score, cnt]): ', np.stack([sorted_g_score, cnt]))

  optimal_size = set_size

  if flex_type == 'n_sample' and cnt is not None:
    min_size, max_size = get_min_max_size(cnt.astype(float), flex_ratio)
  elif flex_type == 'n_group_w_sample' and cnt is not None:
    cnt_binary = (cnt > MIN_GROUP_POS_SAMPLE_SIZE_FLEX).astype(float)
    min_size, max_size = get_min_max_size(cnt_binary, flex_ratio)
  else:
    min_size = (np.ceil(set_size * (1 - flex_ratio))).astype(int)
    max_size = (np.ceil(set_size * (1 + flex_ratio))).astype(int)
    if flex_type != 'n_group':
      print('Warning: cnt is None')

  for size in range(min_size, max_size):
    if sorted_g_score[size] <= 0:
      optimal_size = size - 1
      break


  return optimal_size


def get_grid_id_for_largest_component(grid, components, sizes):
  #non-zero component
  largest_component = np.argsort(sizes)[-1]#np.max(sizes, axis=0)

  #get_grid_id_from_components_id
  grid_ids = grid[components == largest_component]
  grid_id = grid_ids[0]

  if grid_id == 0:
    # if sizes.shape[0] == 1:
    #   return -1
    largest_component = np.argsort(sizes)[-2]#np.max(sizes, axis=0)
    grid_ids = grid[components == largest_component]
    grid_id = grid_ids[0]

  return grid_id

def label_multi(grid, connectivity=1, return_num=True):
  cid = 0
  values = np.unique(grid)
  label = np.zeros(grid.shape)
  grid_mask = grid>0

  for value in values:
    if value == 0:
      continue
    label_value, num_labels = ndimage.label(grid==value)
    label_value += cid * (label_value>0)
    label_value = label_value * grid_mask
    label += label_value
    cid += num_labels

  return label, cid


def swap_small_components(s_grid, min_component_size):

  s_grid_mask = (s_grid>0)
  # components, num_labels = ndimage.label(s_grid)
  # components, num_labels = measure.label(s_grid, connectivity=1, return_num=True)
  components, num_labels = label_multi(s_grid)

  sizes = ndimage.sum(s_grid_mask, components, range(num_labels + 1))
  swap_labels = np.where(sizes <= min_component_size)[0]

  with np.printoptions(threshold=np.inf):
    print('swap_small_components:')
    print('grid:')
    print(s_grid)
    print('components:')
    print(components)
    print('num_labels: ', num_labels)
    print('sizes:')
    print(sizes)

  if num_labels==0:
    return s_grid#TODO: check this!!!

  largest_id = get_grid_id_for_largest_component(s_grid, components, sizes)
  print('largest_id:', largest_id)

  for i in range(1, swap_labels.shape[0]):
    s_grid[components == swap_labels[i]] = largest_id

  s_grid = s_grid * s_grid_mask

  return s_grid


#The function is to improve contiguity of spatial partitioning for gridded data
def swap_partition_general(loc_grid, locs, null_value = -1):
  '''Used as a sub-function to improve contiguity.
  locs is an arrary of locations in partition pid (pid is either 0 or 1)
  '''
  i_max, j_max = loc_grid.shape
  loc_grid_new = np.copy(loc_grid)
  for loc in locs:
    loc_i = loc[0]
    loc_j = loc[1]

    if loc_grid[loc_i, loc_j] == null_value:#confirm how many null values are there in refinement-all
      continue

    ##for 8-neighbor
    loc_i_min = max(0, loc_i - 1)
    loc_i_max = min(i_max, loc_i + 2)
    loc_j_min = max(0, loc_j - 1)
    loc_j_max = min(j_max, loc_j + 2)

    local_grid = loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max]
    local_grid = local_grid[local_grid != null_value].reshape(-1)
    center_value = loc_grid[loc_i, loc_j].astype(int)
    if local_grid.shape[0] == 0:
      majority = center_value
    else:
      count_list = np.bincount(local_grid.astype(int))
      if count_list.shape[0] == 2:
        other_value = int(1-center_value)
        if count_list[center_value] / (count_list[other_value] + count_list[center_value]) < 4/9:
          majority = other_value
        else:
          majority = center_value
      else:
        majority = np.bincount(local_grid.astype(int)).argmax()
      # majority = stats.mode(local_grid, keepdims=True)[0][0]

    loc_grid_new[loc_i, loc_j] = majority


  return loc_grid_new

#The function is to improve contiguity of spatial partitioning for gridded data
def swap_partition(loc_grid, locs, pid):
  '''Used as a sub-function to improve contiguity.
  locs is an arrary of locations in partition pid (pid is either 0 or 1)
  '''
  i_max, j_max = loc_grid.shape
  loc_grid_new = np.copy(loc_grid)
  for loc in locs:
    loc_i = loc[0]
    loc_j = loc[1]
    count = 0
    count_eq = 0 #there are -1 values

    ##for 8-neighbor
    loc_i_min = max(0, loc_i - 1)
    loc_i_max = min(i_max, loc_i + 2)
    loc_j_min = max(0, loc_j - 1)
    loc_j_max = min(j_max, loc_j + 2)

    loc_mask = (loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max] >= 0).astype(int)
    count = int(np.sum( (loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max]!=pid).astype(int) * loc_mask ))
    count_eq = int(np.sum( (loc_grid[loc_i_min:loc_i_max, loc_j_min:loc_j_max]==pid).astype(int) ))

    if count > count_eq:
      loc_grid_new[loc_i, loc_j] = np.abs(1-pid)

  return loc_grid_new


def grid_to_partitions(loc_grid, null_value = -1):
  group_id_grid = np.arange(loc_grid.shape[0] * loc_grid.shape[1])
  group_id_grid = group_id_grid.reshape(loc_grid.shape)

  s0 = group_id_grid[loc_grid==0]
  s1 = group_id_grid[loc_grid==1]

  return s0, s1


#customized spatial contiguity check, optional
def get_refined_partitions_dispatcher(s0, s1, y_val_gid, g_loc, dir = None, branch_id = None, 
                          contiguity_type='grid', polygon_centroids=None, 
                          polygon_group_mapping=None, neighbor_distance_threshold=None):
  '''This will refine partitions based on specific user needs, i.e., this function is problem-specific.
     This example will improve the spatial contiguity of the obtained partitions, by swapping partition assignments for an element that is surrounded by elements belonging to another partition.
     s0 and s1 contains group_ids
     g_loc contains locations of groups (can be represented by row id and column id)
     
     Parameters:
     -----------
     contiguity_type : str, default='grid'
         Type of contiguity refinement: 'grid' or 'polygon'
     polygon_centroids : array-like, shape (n_polygons, 2), optional
         Centroid coordinates for polygon-based contiguity
     polygon_group_mapping : dict, optional
         Mapping from polygon indices to group IDs for polygon-based contiguity
     neighbor_distance_threshold : float, optional
         Distance threshold for polygon neighbors
  '''
  
  # Use polygon-based contiguity if specified
  if contiguity_type == 'polygon':
    if polygon_centroids is None or polygon_group_mapping is None:
      print("Warning: polygon_centroids and polygon_group_mapping required for polygon contiguity. Falling back to grid contiguity.")
      return get_refined_partitions_grid(s0, s1, y_val_gid, g_loc, dir, branch_id)
    else:
      return get_refined_partitions_polygon(s0, s1, y_val_gid, polygon_centroids, 
                                          polygon_group_mapping, neighbor_distance_threshold, 
                                          dir, branch_id)
  
  # Use grid-based contiguity (default)
  return get_refined_partitions_grid(s0, s1, y_val_gid, g_loc, dir, branch_id)


def get_refined_partitions_grid(s0, s1, y_val_gid, g_loc, dir=None, branch_id=None):
  '''Grid version of get_refined_partitions using grid-based spatial contiguity.
  
  Parameters:
  -----------
  s0, s1 : array-like
      Partition assignments (indices in y_val_gid)
  y_val_gid : array-like
      Group IDs for validation data
  g_loc : array-like, shape (n_groups, 2)
      Grid locations for each group (row, col coordinates)
  dir : str, optional
      Directory for saving visualizations
  branch_id : str, optional
      Branch identifier for visualization files
      
  Returns:
  --------
  s0_refined, s1_refined : array-like
      Refined partition assignments
  '''
  #s0, s1 do not contain gid directly
  s0_group = get_s_list_group_ids(s0.astype(int), y_val_gid).astype(int)
  s1_group = get_s_list_group_ids(s1.astype(int), y_val_gid).astype(int)

  #need to first got grid-based locations (for efficiency)

  #s0,s1 to grid
  loc_grid = -np.ones([np.max(g_loc[:,0])+1, np.max(g_loc[:,1])+1])#need to default values to -1
  loc_s0 = g_loc[s0_group]#need to test
  loc_s1 = g_loc[s1_group]
  loc_grid[loc_s0[:,0], loc_s0[:,1]] = 0
  loc_grid[loc_s1[:,0], loc_s1[:,1]] = 1

  # print(loc_grid)

  #the generate_vis_image_for_all_groups() in the following 2 if conditions can be added back to compare results
  if dir is not None and branch_id is not None:
    generate_vis_image_for_all_groups(loc_grid, dir = dir, ext = '_debug1_' + branch_id, vmin = -1, vmax = 1)

  #only perform once after each partitioning, should not affect efficiency
  # loc_grid = swap_partition(loc_grid, loc_s0, 0)
  # loc_grid = swap_partition(loc_grid, loc_s1, 1)
  loc_grid = swap_partition_general(loc_grid, np.vstack([loc_s0, loc_s1]))

  if dir is not None:
    generate_vis_image_for_all_groups(loc_grid, dir = dir, ext = '_debug2_' + branch_id, vmin = -1, vmax = 1)
  # print(loc_grid)

  #grid to s0, s1
  s0_group, s1_group = grid_to_partitions(loc_grid)
  gid_to_s_map = get_gid_to_s_map(y_val_gid, s0, s1)
  s0, s1 = s_group_to_s(s0_group, s1_group, gid_to_s_map)

  return s0.astype(int), s1.astype(int)


def get_polygon_neighbors(centroids, neighbor_distance_threshold=None, adjacency_dict=None):
  '''Find neighboring polygons based on either true adjacency matrix or centroid distances.
  
  Parameters:
  -----------
  centroids : array-like, shape (n_polygons, 2)
      Centroid coordinates (lat, lon) for each polygon
  neighbor_distance_threshold : float, optional
      Maximum distance to consider polygons as neighbors.
      If None, uses adaptive threshold based on median distance.
      Only used when adjacency_dict is None.
  adjacency_dict : dict, optional
      Pre-computed adjacency dictionary from true polygon boundaries.
      If provided, this takes precedence over distance-based calculation.
      
  Returns:
  --------
  neighbor_dict : dict
      Dictionary where keys are polygon indices and values are lists of neighbor indices
  '''
  import numpy as np
  
  # If adjacency dictionary is provided, use it directly
  if adjacency_dict is not None:
    from src.adjacency.adjacency_utils import adjacency_dict_to_neighbors_dict
    print("Using true polygon adjacency matrix for neighbor determination")
    return adjacency_dict_to_neighbors_dict(adjacency_dict)
  
  # Fall back to distance-based neighbor calculation
  print("Using centroid distance-based neighbor determination")
  from scipy.spatial.distance import cdist
  
  # Calculate pairwise distances between centroids
  distances = cdist(centroids, centroids, metric='euclidean')
  
  # Set distance threshold if not provided
  if neighbor_distance_threshold is None:
    # Use median of non-zero distances as threshold
    non_zero_distances = distances[distances > 0]
    neighbor_distance_threshold = np.median(non_zero_distances) * 1.5
  
  # Build neighbor dictionary
  neighbor_dict = {}
  for i in range(len(centroids)):
    # Find neighbors within threshold distance (excluding self)
    neighbors = np.where((distances[i] <= neighbor_distance_threshold) & (distances[i] > 0))[0]
    neighbor_dict[i] = neighbors.tolist()
  
  return neighbor_dict


def swap_partition_polygon(partition_assignments, polygon_neighbors, centroids=None):
  '''Polygon version of swap_partition_general using majority voting among neighbors.
  
  Parameters:
  -----------
  partition_assignments : array-like, shape (n_polygons,)
      Current partition assignments (0 or 1) for each polygon
  polygon_neighbors : dict
      Dictionary where keys are polygon indices and values are lists of neighbor indices
  centroids : array-like, shape (n_polygons, 2), optional
      Centroid coordinates for debugging/visualization
      
  Returns:
  --------
  partition_assignments_new : array-like, shape (n_polygons,)
      Refined partition assignments after majority voting
  '''
  import numpy as np
  
  partition_assignments_new = np.copy(partition_assignments)
  
  for poly_idx in range(len(partition_assignments)):
    current_partition = partition_assignments[poly_idx]
    
    # Handle isolated polygons (polygons with no neighbors)
    if poly_idx not in polygon_neighbors or len(polygon_neighbors[poly_idx]) == 0:
      try:
        from config import PRESERVE_ISOLATED_POLYGONS, DIAGNOSTIC_POLYGON_TRACKING
        if PRESERVE_ISOLATED_POLYGONS:
          # Keep isolated polygon with its original assignment
          partition_assignments_new[poly_idx] = current_partition
          if DIAGNOSTIC_POLYGON_TRACKING:
            print(f"Preserving isolated polygon {poly_idx} with partition {current_partition}")
          continue
        else:
          continue  # Original behavior: skip isolated polygons
      except:
        continue  # Fallback to original behavior
    
    # Get neighbor partition assignments
    neighbor_indices = polygon_neighbors[poly_idx]
    neighbor_partitions = partition_assignments[neighbor_indices]
    
    # Include current polygon in the voting
    all_partitions = np.append(neighbor_partitions, current_partition)
    
    # Count partition assignments
    unique_partitions, counts = np.unique(all_partitions, return_counts=True)
    
    if len(unique_partitions) == 1:
      # All neighbors have same partition
      majority_partition = unique_partitions[0]
    elif len(unique_partitions) == 2:
      # Binary case: use threshold similar to grid-based (4/9)
      partition_0_count = counts[unique_partitions == 0][0] if 0 in unique_partitions else 0
      partition_1_count = counts[unique_partitions == 1][0] if 1 in unique_partitions else 0
      total_count = partition_0_count + partition_1_count
      
      current_partition_count = partition_0_count if current_partition == 0 else partition_1_count
      
      # Use same threshold as grid-based: if current partition < 4/9 of total, switch
      if current_partition_count / total_count < 4/9:
        majority_partition = 1 - current_partition
      else:
        majority_partition = current_partition
    else:
      # Multi-partition case: use simple majority
      majority_partition = unique_partitions[np.argmax(counts)]
    
    partition_assignments_new[poly_idx] = majority_partition
  
  return partition_assignments_new


def polygon_partitions_to_groups(partition_assignments, polygon_to_group_map):
  '''Convert polygon partition assignments back to group-level partitions.
  
  Parameters:
  -----------
  partition_assignments : array-like, shape (n_polygons,)
      Partition assignments (0 or 1) for each polygon
  polygon_to_group_map : dict
      Dictionary mapping polygon indices to group IDs
      
  Returns:
  --------
  s0_group : array-like
      Group IDs assigned to partition 0
  s1_group : array-like
      Group IDs assigned to partition 1
  '''
  import numpy as np
  
  s0_groups = []
  s1_groups = []
  
  for poly_idx, partition in enumerate(partition_assignments):
    if poly_idx in polygon_to_group_map:
      group_ids = polygon_to_group_map[poly_idx]
      if isinstance(group_ids, (list, np.ndarray)):
        # Multiple groups per polygon
        if partition == 0:
          s0_groups.extend(group_ids)
        else:
          s1_groups.extend(group_ids)
      else:
        # Single group per polygon
        if partition == 0:
          s0_groups.append(group_ids)
        else:
          s1_groups.append(group_ids)
  
  return np.array(s0_groups), np.array(s1_groups)


def get_refined_partitions_polygon(s0, s1, y_val_gid, polygon_centroids, 
                                  polygon_group_mapping, neighbor_distance_threshold=None,
                                  dir=None, branch_id=None):
  '''Polygon version of get_refined_partitions using centroid-based neighbors.
  
  Parameters:
  -----------
  s0, s1 : array-like
      Partition assignments (indices in y_val_gid)
  y_val_gid : array-like
      Group IDs for validation data
  polygon_centroids : array-like, shape (n_polygons, 2)
      Centroid coordinates (lat, lon) for each polygon
  polygon_group_mapping : dict
      Dictionary mapping polygon indices to group IDs or lists of group IDs
  neighbor_distance_threshold : float, optional
      Distance threshold for determining neighbors
  dir : str, optional
      Directory for saving visualizations
  branch_id : str, optional
      Branch identifier for visualization files
      
  Returns:
  --------
  s0_refined, s1_refined : array-like
      Refined partition assignments
  '''
  import numpy as np
  from src.helper.helper import get_gid_to_s_map
  
  # Track polygon counts for validation
  try:
    from config import DIAGNOSTIC_POLYGON_TRACKING
    if DIAGNOSTIC_POLYGON_TRACKING:
      print(f"get_refined_partitions_polygon - Input: s0={len(s0)}, s1={len(s1)}, y_val_gid={len(y_val_gid)}")
  except:
    pass
  
  # Get group IDs for each partition
  s0_group = get_s_list_group_ids(s0.astype(int), y_val_gid).astype(int)
  s1_group = get_s_list_group_ids(s1.astype(int), y_val_gid).astype(int)
  
  # Track group counts
  try:
    from config import DIAGNOSTIC_POLYGON_TRACKING
    if DIAGNOSTIC_POLYGON_TRACKING:
      print(f"  After get_s_list_group_ids: s0_group={len(s0_group)}, s1_group={len(s1_group)}")
  except:
    pass
  
  # Create reverse mapping from group IDs to polygon indices
  group_to_polygon_map = {}
  for poly_idx, group_ids in polygon_group_mapping.items():
    if isinstance(group_ids, (list, np.ndarray)):
      for group_id in group_ids:
        if group_id not in group_to_polygon_map:
          group_to_polygon_map[group_id] = []
        group_to_polygon_map[group_id].append(poly_idx)
    else:
      if group_ids not in group_to_polygon_map:
        group_to_polygon_map[group_ids] = []
      group_to_polygon_map[group_ids].append(poly_idx)
  
  # Initialize polygon partition assignments
  n_polygons = len(polygon_centroids)
  polygon_partitions = np.full(n_polygons, -1, dtype=int)
  
  # Assign partitions to polygons based on group memberships
  for group_id in s0_group:
    if group_id in group_to_polygon_map:
      for poly_idx in group_to_polygon_map[group_id]:
        polygon_partitions[poly_idx] = 0
  
  for group_id in s1_group:
    if group_id in group_to_polygon_map:
      for poly_idx in group_to_polygon_map[group_id]:
        polygon_partitions[poly_idx] = 1
  
  # Find polygon neighbors (use adjacency matrix if available in polygon_group_mapping dict)
  adjacency_dict = None
  if isinstance(polygon_group_mapping, dict) and 'adjacency_dict' in polygon_group_mapping:
    adjacency_dict = polygon_group_mapping['adjacency_dict']
  polygon_neighbors = get_polygon_neighbors(polygon_centroids, neighbor_distance_threshold, adjacency_dict)
  
  # Apply majority voting refinement
  polygon_partitions_refined = swap_partition_polygon(polygon_partitions, polygon_neighbors, polygon_centroids)
  
  # Convert back to group-level partitions
  s0_group_refined, s1_group_refined = polygon_partitions_to_groups(polygon_partitions_refined, polygon_group_mapping)
  
  # Convert back to original indices
  gid_to_s_map = get_gid_to_s_map(y_val_gid, s0, s1)
  s0_refined, s1_refined = s_group_to_s(s0_group_refined, s1_group_refined, gid_to_s_map)
  
  # Validate output polygon counts
  try:
    from config import DIAGNOSTIC_POLYGON_TRACKING, VALIDATE_POLYGON_COUNTS
    if DIAGNOSTIC_POLYGON_TRACKING:
      print(f"  Output: s0_refined={len(s0_refined)}, s1_refined={len(s1_refined)}")
      print(f"  Total refined: {len(s0_refined) + len(s1_refined)}, Original total: {len(s0) + len(s1)}")
    
    if VALIDATE_POLYGON_COUNTS:
      original_total = len(s0) + len(s1)
      refined_total = len(s0_refined) + len(s1_refined)
      if original_total != refined_total:
        print(f"WARNING: Polygon count mismatch in get_refined_partitions_polygon!")
        print(f"  Original: {original_total}, Refined: {refined_total}, Difference: {original_total - refined_total}")
  except:
    pass
  
  return s0_refined.astype(int), s1_refined.astype(int)


def s_group_to_s(s0_group, s1_group, gid_to_s_map):
  # Handle both dictionary and array-based mapping
  if isinstance(gid_to_s_map, dict):
    # Dictionary-based mapping (for non-contiguous group IDs)
    s0 = np.array([gid_to_s_map[gid] for gid in s0_group if gid in gid_to_s_map])
    s1 = np.array([gid_to_s_map[gid] for gid in s1_group if gid in gid_to_s_map])
  else:
    # Array-based mapping (for contiguous group IDs - legacy)
    s0 = gid_to_s_map[s0_group]
    s1 = gid_to_s_map[s1_group]
  return s0, s1

def get_top_cells(g, flex = FLEX_OPTION, flex_ratio = FLEX_RATIO, flex_type = FLEX_TYPE, cnt = None):
  '''get the top half cells with largest values (return values are row ids)'''

  #this is only to get index to be used for s0, s1
  sorted_g = np.argsort(g,0)#second input might not be needed
  sorted_g = sorted_g[::-1]
  # sorted_g = g.shape[0] - 1 - sorted_g#sorted_g[::-1]
  set_size = np.ceil(sorted_g.shape[0]/2).astype(int)

  if flex:
    if cnt is not None:
      # print('#Debug: np.stack([g, cnt]): ', np.stack([g, cnt]))
      cnt = cnt[sorted_g]
    set_size = optimize_size(g, set_size, flex_ratio, flex_type = flex_type, cnt = cnt)

  s0 = sorted_g[0:set_size]
  s1 = sorted_g[set_size:]



  return s0, s1

def get_score(y_true, y_pred, mode = MODE):

  score = None
  if mode == 'classification':
    if len(y_true.shape)==1:
      # y_true = tf.one_hot(y_true, NUM_CLASS)
      # y_pred = tf.one_hot(y_pred, NUM_CLASS)
      y_true = np.eye(NUM_CLASS)[y_true.astype(int)].astype(int)
      y_pred = np.eye(NUM_CLASS)[y_pred.astype(int)].astype(int)
    # else:
    # #this is to make coding consistent for later parts of the function (where tf functions are used)
    #   y_true = tf.convert_to_tensor(y_true)
    #   y_pred = tf.convert_to_tensor(y_pred)
    #   # tf.convert_to_tensor(numpy_array, dtype=tf.float32)

    #reshape image or time-series labels
    if len(y_true.shape)>=3:
      y_true = y_true.reshape(-1, NUM_CLASS)
      y_pred = y_pred.reshape(-1, NUM_CLASS)
      #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape
      # y_true = tf.reshape(y_true, [-1,NUM_CLASS])#tf.reshape takes numpy arrays
      # y_pred = tf.reshape(y_pred, [-1,NUM_CLASS])

    # Crisis-focused optimization: use governing metric instead of overall accuracy
    if 'CRISIS_FOCUSED_OPTIMIZATION' in globals() and CRISIS_FOCUSED_OPTIMIZATION:
        score = get_metric_score_array(y_true, y_pred, GOVERNING_METRIC)
    else:
        score = (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).astype(int)
    # score = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    # score = score.numpy()

    #select_class
    #here need to remove the rows from non-selected classes
    #which will otherwise affect the diff margin sizes in sig_test()
    if SELECT_CLASS is not None:
      score_select = np.zeros(score.shape)
      for i in range(SELECT_CLASS.shape[0]):
        class_id = int(SELECT_CLASS[i])
        # score_select[y_true.numpy()[:,class_id]==1] = 1
        score_select[y_true[:,class_id]==1] = 1
      score = score[score_select.astype(bool)]
      # TODO: Check what if score is empty?

  else:
    #GeoRF code is not tested for regression yet
    score = np.square(y_true - y_pred)
    #the reduction option is deprecated
    #without reduction it might be ok if each element is still surrounded by []
    # score = tf.keras.losses.MSE(y_true, y_pred, reduction=tf.keras.losses.Reduction.NONE)#reduction=tf.keras.losses.Reduction.NONE
    #check this for regression!!!
    #careful with dimension when using Reduction.None (last dimension must be for prediction-target dimensions): https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction
    # score = - score.numpy()

  return score

def get_split_score(y0_true, y0_pred, y1_true, y1_pred, mode = MODE):
  score0 = get_score(y0_true, y0_pred)
  score1 = get_score(y1_true, y1_pred)

  score = np.hstack([score0, score1])

  return score0, score1

'''Partitioning optimization.'''
def scan(y_true_value, true_pred_value, min_sample,
         flex = FLEX_OPTION,
         flex_type = FLEX_TYPE,
         return_score = False):#connected = True, g_grid = None, X_dim = None, step_size = None,

  '''flex_type: determines if the split balance is based on #groups or #samples in groups'''
  c,b = get_c_b(y_true_value, true_pred_value)

  max_iteration = 1000

  #init q
  q = np.zeros(y_true_value.shape[1])
  q_init = np.nan_to_num(c/b)
  for i in range(q_init.shape[1]):
    q_class = q_init[:,i]
    # s_class, _ = get_top_cells(q_class)
    print('scan init step (ignore immediate "cnt = None" warning)')
    s_class, _ = get_top_cells(q_class)#no cnt-based flex for initialization?
    q[i] = np.sum(c[s_class,i]) / np.sum(b[s_class,i])

  # q = np.random.rand(y_true_value.shape[1])*2
  # q = np.exp(q)

  q_filter = np.sum(b,0) < min_sample
  q[q_filter == 1] = 1
  q[q == 0] = 1
  q = np.expand_dims(q,0)

  #prepare cnt for sample count based flex
  b_cnt = np.sum(b,1)

  log_lr_prev = 0
  for i in range(max_iteration):#coordinate descent
    #update location
    g = c * np.log(q) + b * (1-q)
    g = np.sum(g, 1)
    s0, s1 = get_top_cells(g, flex_type = flex_type, cnt = b_cnt)
    log_lr = np.sum(g[s0])

    #update q
    q = np.nan_to_num(np.sum(c[s0],0) / np.sum(b[s0],0))
    q[q == 0] = 1
    q[q_filter == 1] = 1

    if log_lr < 0:
      print("log_lr < 0: check initialization!")

    # if (log_lr - log_lr_prev) / log_lr_prev < 0.05:
    #   break

    log_lr_prev = log_lr

    s0 = s0.reshape(-1)
    s1 = s1.reshape(-1)
    # if (i == max_iteration - 1) and (connected == True):
      # s0, s1 = get_connected_top_cells(g, g_grid, X_dim, step_size, flex = flex)

    # print('s0', s0)
    # print('s1', s1)

  if return_score:
    return s0, s1, g
    # return s0, s1, g[s0], g[s1]
  else:
    return s0, s1


def get_refined_partitions_all(X_branch_id, s_branch, X_group, dir = None, min_component_size = 10, max_depth = MAX_DEPTH):
  '''This is used to refine partitions with all partition ids (not for smoothing binary partitions during the training process).'''

  unique_branch = np.unique(X_branch_id[X_branch_id != ''])
  branch_id_len = np.array(list(map(lambda x: len(x), unique_branch)))
  unique_branch = unique_branch[np.argsort(branch_id_len).astype(int)]
  #here grid has null/empty value of 0 (in partition_opt null is -1)
  grid, id_map = vis_partition_group(s_branch, unique_branch, step_size=STEP_SIZE, max_depth = max_depth, return_id_map = True)
  id_map[0] = ''#unique branch no longer contains ''
  print('id_map', id_map)
  # print('grid min: ', np.min(grid))
  # print('grid.shape', grid.shape)

  if VIS_DEBUG_MODE:
    generate_vis_image_from_grid(grid, dir, file_name = 'all_refined_before')

  if dir is not None and VIS_DEBUG_MODE:
    np.save(dir + '/' + 'grid' + '_before' + '.npy', grid)#ext

  locs = generate_groups_loc(X_DIM, STEP_SIZE)
  # for refine_i in range(REFINE_TIMES):
    # grid = swap_partition_general(grid, locs, null_value = 0)
  grid = swap_small_components(grid, min_component_size)

  if dir is not None and VIS_DEBUG_MODE:
    generate_vis_image_from_grid(grid, dir, file_name = 'all_refined')
    np.save(dir + '/' + 'grid' + '_after' + '.npy', grid)

  list_branch_id_int = grid.reshape(-1).astype(int)
  # list_group_id = np.arange(grid.shape[0] * grid.shape[1])
  list_branch_id = np.asarray([id_map[int_id] for int_id in list_branch_id_int])
  # np.stack([list_group_id, list_branch_id], axis = -1)
  X_branch_id = list_branch_id[X_group.astype(int)]

  return X_branch_id
