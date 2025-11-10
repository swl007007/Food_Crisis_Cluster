# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
# import tensorflow as tf
import pandas as pd

from config import *

#visualization
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from PIL import Image

# import geopandas as gpd

'''
Functions in visualization normally need to be customized depending on how the groups are generated, and the data type.
'''

# Strict visualization gate resolver
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

#in training visualization
def vis_partition_training(grid, branch_id):
  '''Visualize space-partitionings.'''

  vis_size = 256
  # grid_img = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
  resized_img = Image.fromarray(grid).resize((vis_size, vis_size), Image.NEAREST)
  resized_img_np = np.array(resized_img)

  fig = plt.figure(figsize=(3,3))
  # color_palette = sns.color_palette('deep', s_branch.shape[1])
  im = plt.imshow(resized_img_np, interpolation="none")#cmap=color_palette,
  # cbar = plt.colorbar(im, ticks = np.arange(s_branch.shape[1]))
  # cbar.ax.set_yticklabels(list(s_branch.keys()))
  img_name = branch_id
  if len(branch_id) == 0:
    img_name = 'initial'

  fig.savefig(dir + '/' + img_name + '.png')

  return#grid

def generate_grid_vis(X_dim, step_size):#, max_size_decay

  n_row = np.ceil(X_dim[0]/step_size).astype(int)
  n_col = np.ceil(X_dim[1]/step_size).astype(int)

  grid = np.ones([n_row, n_col])

  return grid, n_row, n_col


def vis_partition(s_branch, unique_branch, max_depth = MAX_DEPTH, step_size = STEP_SIZE):

  #get cell size
  step_size = step_size / (2**np.floor((MAX_DEPTH-1)/2))
  grid, n_row, n_col = generate_grid_vis(X_DIM, step_size)

  color_id = 0
  for branch_id in unique_branch:#list(s_branch.keys()):
    current_depth = len(branch_id)
    if current_depth >= max_depth:
      continue

    if current_depth == 0:
      current_step_size = step_size
    else:
      current_step_size = step_size / (2**np.floor((current_depth-1)/2))

    step_size_ratio = current_step_size / step_size

    # if current_depth == 2 or current_depth == 4:
    #   step_size_ratio = step_size_ratio * 2

    for gid in s_branch[branch_id]:

      if gid is None or (gid == 0):
        break

      # print(gid)
      row_id = gid[0]
      col_id = gid[1]

      #get row ranges in full resolution grid
      row_min = row_id * step_size_ratio
      row_max = row_min + step_size_ratio
      col_min = col_id * step_size_ratio
      col_max = col_min + step_size_ratio

      row_range = np.arange(row_min,row_max).astype(int)
      col_range = np.arange(col_min,col_max).astype(int)

      grid[np.ix_(row_range, col_range)] = color_id

    color_id = color_id + 1

  return grid


def vis_partition_group(s_branch, unique_branch, step_size, max_depth = MAX_DEPTH, return_id_map = False):
  '''id map stores the 1 to 1 matching between branch_id (e.g., '01') and color_id (an integer id for each branch)'''

  #get cell size
  #!!!be careful, this may need to be updated with np.floor + 1 to be consistent with other initializations
  #n_rows = np.ceil(X_DIM[0]/step_size).astype(int)
  #n_cols = np.ceil(X_DIM[1]/step_size).astype(int)
  n_rows = np.floor(X_DIM[0]/step_size).astype(int) + 1
  n_cols = np.floor(X_DIM[1]/step_size).astype(int) + 1
  grid = np.zeros((n_rows, n_cols))

  id_map = {}

  color_id = 1#background color to be 0
  for branch_id in unique_branch:#list(s_branch.keys()):
    current_depth = len(branch_id)
    if current_depth >= max_depth or current_depth == 0:#0 means '', X_branch_id may contain '' as test samples are have not been assigned branches yet
      continue

    # if current_depth == 0:
    #   current_step_size = STEP_SIZE
    # else:
    #   current_step_size = STEP_SIZE / (2**np.floor((current_depth-1)/2))

    # step_size_ratio = current_step_size / step_size
    step_size_ratio = 1

    # if current_depth == 2 or current_depth == 4:
    #   step_size_ratio = step_size_ratio * 2

    # print('s_branch[branch_id]', s_branch[branch_id])

    for gid in s_branch[branch_id]:

      if gid is None or (gid == -1):#(gid == 0):#check this, a valid gid may be 0!!!
        #break#probably some are removed by contiguity (swap small components)
        continue

      # print(gid)
      row_id = np.floor(gid/n_cols).astype(int)
      col_id = (gid % n_cols).astype(int)
      # row_id = gid[0]
      # col_id = gid[1]

      #get row ranges in full resolution grid
      row_min = row_id * step_size_ratio
      row_max = row_min + step_size_ratio
      col_min = col_id * step_size_ratio
      col_max = col_min + step_size_ratio

      row_range = np.arange(row_min,row_max).astype(int)
      col_range = np.arange(col_min,col_max).astype(int)

      grid[np.ix_(row_range, col_range)] = color_id

    id_map[int(color_id)] = branch_id
    color_id = color_id + 1

  if return_id_map:
    return grid, id_map
  else:
    return grid

def generate_vis_image(s_branch, X_branch_id, max_depth, dir, step_size = STEP_SIZE, file_name='all', VIS_DEBUG_MODE=None):
  if not _resolve_vis_flag(VIS_DEBUG_MODE):
    return
    
  print(list(s_branch.keys()))

  unique_branch = np.unique(X_branch_id)
  branch_id_len = np.array(list(map(lambda x: len(x), unique_branch)))
  unique_branch = unique_branch[np.argsort(branch_id_len).astype(int)]
  print(unique_branch)

  from PIL import ImageOps
  VIS_SIZE = 1000
  grid = vis_partition_group(s_branch, unique_branch, step_size=step_size, max_depth = max_depth)
  resized_img = ImageOps.flip(Image.fromarray(grid)).resize((VIS_SIZE, int(VIS_SIZE*(xmax/ymax))), Image.NEAREST)
  resized_img = np.array(resized_img)

  IMG_SIZE = 20
  fig = plt.figure(figsize=(IMG_SIZE,int(IMG_SIZE*(xmax/ymax))))
  color_palette = sns.color_palette('deep', s_branch.shape[1])
  im = plt.imshow(resized_img, interpolation="none")#cmap=color_palette,
  cbar = plt.colorbar(im, ticks = np.arange(s_branch.shape[1]))
  #the following line has issues with .py in some environments (might be version issues)
  # cbar.ax.set_yticklabels(unique_branch.tolist())#list(s_branch.keys())

  # fig.savefig(dir + '/' + 'all.png')
  fig.savefig(dir + '/' + file_name + '.png')
  np.save(dir + '/' + 'grid' + file_name + '.npy', grid)


def generate_vis_image_from_grid(grid, dir, file_name='all', VIS_DEBUG_MODE=None):
  if not _resolve_vis_flag(VIS_DEBUG_MODE):
    return
    
  from PIL import ImageOps
  VIS_SIZE = 1000
  resized_img = ImageOps.flip(Image.fromarray(grid)).resize((VIS_SIZE, int(VIS_SIZE*(xmax/ymax))), Image.NEAREST)
  resized_img = np.array(resized_img)

  IMG_SIZE = 20
  fig = plt.figure(figsize=(IMG_SIZE,int(IMG_SIZE*(xmax/ymax))))
  color_palette = sns.color_palette('deep', len(np.unique(grid.reshape(-1))))
  im = plt.imshow(resized_img, interpolation="none")#cmap=color_palette,
  cbar = plt.colorbar(im, ticks = np.arange(len(np.unique(grid.reshape(-1)))))
  #the following line has issues with .py in some environments (might be version issues)
  # cbar.ax.set_yticklabels(unique_branch.tolist())#list(s_branch.keys())

  # fig.savefig(dir + '/' + 'all.png')
  fig.savefig(dir + '/' + file_name + '.png')


def generate_performance_grid(results, groups, step_size = STEP_SIZE, prf = True, class_id = None, X_dim = X_DIM):
  '''
  Args:
    results: group-wise performances (e.g., prf from the predict_test_group_wise())
    groups: group ids for the results
    prf: True if "results" is for multi-outputs (e.g., prf results with pre, rec, f1, total number); and False if for other scalar outputs, e.g., accuracy, etc.
    class_id: if only wants to show results for one class (e.g., background and one-class)
  '''
  n_rows = np.ceil(X_dim[0]/step_size).astype(int)
  n_cols = np.ceil(X_dim[1]/step_size).astype(int)
  grid = np.empty((n_rows, n_cols))
  grid[:,:] = np.nan

  row_ids = np.floor(groups/n_cols).astype(int)
  col_ids = (groups % n_cols).astype(int)

  if prf:
    if class_id is None:
      class_weights = results[:,3,:] / np.expand_dims(np.sum(results[:,3,:], axis = -1), 1)
      grid[row_ids, col_ids] = np.sum(results[:,2,:] * class_weights, axis = -1)
    else:
      grid[row_ids, col_ids] = results[:, 2, class_id]
  else:
    grid[row_ids, col_ids] = results

  vmin = np.min(grid[row_ids, col_ids])
  vmax = np.max(grid[row_ids, col_ids])

  return grid, vmin, vmax

def generate_count_grid(results, groups, step_size = STEP_SIZE, class_id = None, X_dim = X_DIM):
  '''
  Args:
    results: group-wise performances (e.g., prf from the predict_test_group_wise())
    groups: group ids for the results
    prf: True if "results" is for multi-outputs (e.g., prf results with pre, rec, f1, total number); and False if for other scalar outputs, e.g., accuracy, etc.
    class_id: if only wants to show results for one class (e.g., background and one-class)
  '''
  n_rows = np.ceil(X_dim[0]/step_size).astype(int)
  n_cols = np.ceil(X_dim[1]/step_size).astype(int)
  grid = np.empty((n_rows, n_cols))
  grid[:,:] = np.nan

  row_ids = np.floor(groups/n_cols).astype(int)
  col_ids = (groups % n_cols).astype(int)

  if class_id is None:
    grid[row_ids, col_ids] = np.sum(results, axis = -1)
  else:
    grid[row_ids, col_ids] = results[:, class_id]

  vmin = np.min(grid[row_ids, col_ids])
  vmax = np.max(grid[row_ids, col_ids])

  print('count: results.shape: ', results.shape)
  print('count: grid.shape: ', grid.shape)
  print('count: results: ', results)
  print('count: np.sum(results, axis = -1): ', np.sum(results, axis = -1))

  return grid, vmin, vmax


def get_symmetric_vmin_vmax(vmin, vmax, option = 'always'):
  if option == 'always':
    v = max(abs(vmin), abs(vmax))
    vmin = -v
    vmax = v
  elif vmax>0 and vmin<0:
      v = max(abs(vmin), abs(vmax))
      vmin = -v
      vmax = v

  return vmin, vmax

def generate_diff_grid(grid, groups, step_size = STEP_SIZE, X_dim = X_DIM):
  '''
  Adhoc function used to get vmin and vmax for visualization. The grid is ready from performance grids.
  '''
  n_rows = np.ceil(X_dim[0]/step_size).astype(int)
  n_cols = np.ceil(X_dim[1]/step_size).astype(int)
  row_ids = np.floor(groups/n_cols).astype(int)
  col_ids = (groups % n_cols).astype(int)

  vmin = np.min(grid[row_ids, col_ids])
  vmax = np.max(grid[row_ids, col_ids])

  vmin, vmax = get_symmetric_vmin_vmax(vmin, vmax)

  return grid, vmin, vmax



def get_colors(n_partitions):
    """
    Get optimized color palette for partition visualization based on number of partitions.
    
    Args:
        n_partitions (int): Number of partitions to color
        
    Returns:
        list: List of colors suitable for the given number of partitions
    """
    if n_partitions <= 10:
        return sns.color_palette("tab10", n_partitions)
    elif n_partitions <= 20:
        return sns.color_palette("tab20", n_partitions)
    elif n_partitions <= 30:
        return sns.color_palette("hls", n_partitions)
    else:
        cmap = plt.get_cmap('turbo', n_partitions)
        return cmap.colors

def plot_partition_swaps(correspondence_before_path, correspondence_after_path,
                        shapefile_path=None, 
                        save_path=None, 
                        title="Partition Assignment Swaps", 
                        figsize=(12, 10), 
                        dpi=300,
                        add_basemap=True,
                        basemap_source=None,
                        VIS_DEBUG_MODE=None):
    """
    Plot only the areas that changed partition assignments between two states.
    
    This function highlights administrative units that were reassigned to different 
    partitions during contiguity refinement, making it easy to see what areas 
    were swapped to improve spatial contiguity.
    
    Args:
        correspondence_before_path (str): Path to CSV with partition assignments before refinement
        correspondence_after_path (str): Path to CSV with partition assignments after refinement
        shapefile_path (str, optional): Path to shapefile with admin boundaries
        save_path (str, optional): Output path for saved map
        title (str): Title for the map
        figsize (tuple): Figure size as (width, height)
        dpi (int): Resolution for saved figure
        add_basemap (bool): Whether to add contextily basemap
        basemap_source: Contextily basemap source
        
    Returns:
        matplotlib.figure.Figure: The created figure object, or None if no swaps detected
    """
    try:
        import geopandas as gpd
        import contextily as ctx
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError(f"Required packages missing: {e}. Install with: pip install geopandas contextily")
    
    # Default paths
    if shapefile_path is None:
        shapefile_path = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
    
    if save_path is None:
        save_path = correspondence_after_path.replace('.csv', '_swaps_map.png')
    
    # Load correspondence tables
    try:
        df_before = pd.read_csv(correspondence_before_path)
        df_after = pd.read_csv(correspondence_after_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Correspondence table not found: {e}")
    
    # Validate required columns
    required_cols = ['FEWSNET_admin_code', 'partition_id']
    for df, name in [(df_before, 'before'), (df_after, 'after')]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {name} correspondence table: {missing_cols}")
    
    # Clean data
    df_before = df_before.dropna(subset=['partition_id'])
    df_before = df_before[df_before['partition_id'] != '']
    df_after = df_after.dropna(subset=['partition_id'])
    df_after = df_after[df_after['partition_id'] != '']
    
    # Merge to find differences
    df_before['admin_code'] = df_before['FEWSNET_admin_code'].astype(float).astype(int).astype(str)
    df_after['admin_code'] = df_after['FEWSNET_admin_code'].astype(float).astype(int).astype(str)
    
    # Find swapped areas by comparing partition_id for same admin_code
    merged = df_before.merge(df_after, on='admin_code', suffixes=('_before', '_after'), how='inner')
    
    # Identify areas that changed partition assignment
    swapped_areas = merged[merged['partition_id_before'] != merged['partition_id_after']].copy()
    
    if len(swapped_areas) == 0:
        print("No partition swaps detected between the two states.")
        return None
    
    print(f"Found {len(swapped_areas)} administrative units that changed partition assignment")
    
    # Load shapefile
    try:
        gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        raise FileNotFoundError(f"Cannot load shapefile: {shapefile_path}. Error: {e}")
    
    # Keep only necessary columns and merge with swapped areas
    gdf = gdf[['admin_code', 'geometry']].copy()
    gdf['admin_code'] = gdf['admin_code'].astype(str)
    
    swapped_gdf = swapped_areas.merge(gdf, on='admin_code', how='left')
    
    # Check for missing geometries
    missing_geom = swapped_gdf['geometry'].isna().sum()
    if missing_geom > 0:
        print(f"Warning: {missing_geom} swapped areas have no matching geometry")
    
    # Convert to GeoDataFrame
    swapped_gdf = gpd.GeoDataFrame(swapped_gdf, geometry='geometry')
    swapped_gdf = swapped_gdf.dropna(subset=['geometry'])
    
    if len(swapped_gdf) == 0:
        print("No valid geometries found for swapped areas")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load full admin boundaries for context (light gray background)
    try:
        full_gdf = gpd.read_file(shapefile_path)
        full_gdf = full_gdf[['admin_code', 'geometry']].copy()
        full_gdf.plot(ax=ax, color='lightgray', alpha=0.3, edgecolor='white', linewidth=0.1)
    except:
        pass  # Continue without background if shapefile loading fails
    
    # Create swap direction mapping for colors
    # Convert partition IDs to strings to avoid data type issues
    swapped_gdf['swap_direction'] = (swapped_gdf['partition_id_before'].astype(str) + 
                                    ' → ' + 
                                    swapped_gdf['partition_id_after'].astype(str))
    unique_swaps = swapped_gdf['swap_direction'].unique()
    n_swaps = len(unique_swaps)
    
    # Use distinct colors for different swap directions
    swap_colors = get_colors(n_swaps)
    
    # Create a custom colormap from the selected colors
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(swap_colors)
    
    # Plot swapped areas with distinct colors based on swap direction
    swapped_gdf.plot(
        ax=ax,
        column='swap_direction',
        cmap=cmap,
        legend=False,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add basemap if requested
    if add_basemap:
        try:
            if basemap_source is None:
                basemap_source = ctx.providers.CartoDB.Positron
            
            ctx.add_basemap(
                ax,
                crs=swapped_gdf.crs.to_string(),
                source=basemap_source,
                zoom='auto',
                alpha=0.7
            )
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Create custom legend
    legend_elements = []
    
    for i, swap_direction in enumerate(sorted(unique_swaps)):
        count = len(swapped_gdf[swapped_gdf['swap_direction'] == swap_direction])
        label = f"{swap_direction} ({count} units)"
        legend_elements.append(Patch(facecolor=swap_colors[i], label=label))
    
    # Add legend
    ax.legend(handles=legend_elements, 
             loc='center left', 
             bbox_to_anchor=(1, 0.5), 
             fontsize=10,
             title="Partition Swaps")
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add summary statistics as text
    stats_text = f"Total swapped units: {len(swapped_gdf)}\nSwap directions: {n_swaps}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Partition swaps map saved to: {save_path}")
    except Exception as e:
        print(f"Could not save figure: {e}")
    
    return fig

def generate_vis_image_for_all_groups(grid, dir, ext = '', vmin = None, vmax = None, VIS_DEBUG_MODE=None):
  '''This generates visualization images for all groups.
  Args:
    grid: from generate_grid_vis()
    ext: to append to the default file name
    vmin: min value for clim in matplotlib
    vmax: max value for clim in matplotlib
  '''
  if not _resolve_vis_flag(VIS_DEBUG_MODE):
    return

  # if vmin is None:
  #   vmin = np.min(grid)
  # if vmax is None:
  #   vmax = np.max(grid)

  from PIL import ImageOps
  VIS_SIZE = 1000
  resized_img = ImageOps.flip(Image.fromarray(grid)).resize((VIS_SIZE, int(VIS_SIZE*(xmax/ymax))), Image.NEAREST)
  resized_img = np.array(resized_img)

  resized_img = np.ma.array(resized_img, mask=np.isnan(resized_img))
  cmap = matplotlib.cm.viridis
  cmap.set_bad('white',1.)#mask out nan values

  IMG_SIZE = 20
  fig = plt.figure(figsize=(IMG_SIZE,int(IMG_SIZE*(xmax/ymax))))
  # color_palette = sns.color_palette('deep', s_branch.shape[1])
  im = plt.imshow(resized_img, interpolation="none", cmap = cmap)#cmap=color_palette,
  if vmin is not None:
    # vmin, vmax = get_symmetric_vmin_vmax(vmin, vmax)
    im.set_clim(vmin,vmax)
  cbar = plt.colorbar(im)
  # cbar = plt.colorbar(im, ticks = np.arange(s_branch.shape[1]))
  #the following line has issues with .py in some environments (might be version versions)
  # cbar.ax.set_yticklabels(unique_branch.tolist())#list(s_branch.keys())

  fig.savefig(dir + '/' + 'result_group' + ext + '.png')


def plot_partition_map(correspondence_table_path,
                      shapefile_path=None,
                      save_path=None,
                      title="GeoRF Partition Map",
                      figsize=(12, 10),
                      dpi=300,
                      add_basemap=True,
                      basemap_source=None,
                      VIS_DEBUG_MODE=None):
    """Render partition choropleths with strict legend/data parity and audits."""

    try:
        import geopandas as gpd
        import contextily as ctx
        from matplotlib.colors import ListedColormap, to_hex
        from matplotlib.patches import Patch
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"Required packages missing: {e}. Install with: pip install geopandas contextily")

    np.random.seed(42)

    from pathlib import Path

    try:
        from config_visual import HIDE_UNASSIGNED_PARTITIONS, UNASSIGNED_LABELS
    except ImportError:
        try:
            from config import HIDE_UNASSIGNED_PARTITIONS, UNASSIGNED_LABELS  # type: ignore
        except ImportError:
            HIDE_UNASSIGNED_PARTITIONS = True
            UNASSIGNED_LABELS = [-1]

    _ = HIDE_UNASSIGNED_PARTITIONS  # Preserve flag for compatibility even if unused

    if shapefile_path is None:
        shapefile_path = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'

    if save_path is None:
        save_path = correspondence_table_path.replace('.csv', '_partition_map.png')

    output_dir = Path(save_path).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        assignments_df = pd.read_csv(correspondence_table_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Correspondence table not found: {correspondence_table_path}")

    required_cols = {'FEWSNET_admin_code', 'partition_id'}
    missing_cols = required_cols - set(assignments_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in correspondence table: {sorted(missing_cols)}")

    assignments_df['admin_code'] = assignments_df['FEWSNET_admin_code'].astype('string').str.strip()

    nodata_tokens = {"", "nan", "NaN", "None", "<NA>", "-1"}
    nodata_tokens.update({str(v).strip() for v in UNASSIGNED_LABELS})

    partitions_str = assignments_df['partition_id'].astype('string').str.strip()
    nodata_mask = partitions_str.isna() | partitions_str.isin(nodata_tokens)
    partitions_clean = partitions_str.mask(nodata_mask)

    assignments_df['partition_id'] = pd.Categorical(
        partitions_clean,
        categories=sorted(partitions_clean.dropna().unique()),
        ordered=False
    )

    assignments_df = assignments_df[assignments_df['admin_code'].notna()].copy()
    assignments_df = assignments_df.drop_duplicates(subset=['admin_code'], keep='last')

    label_freqs_in_data = (
        assignments_df.assign(partition_id=assignments_df['partition_id'].astype('string'))
        .groupby('partition_id', dropna=False)
        .size()
        .reset_index(name='count')
    )
    label_freqs_in_data['is_nodata'] = label_freqs_in_data['partition_id'].isna() | label_freqs_in_data['partition_id'].isin(nodata_tokens)
    label_freqs_in_data.to_csv(output_dir / 'label_freqs_in_data.csv', index=False)

    categories_in_data = [
        cat for cat in sorted(partitions_clean.dropna().unique())
        if cat not in nodata_tokens
    ]

    if not categories_in_data:
        raise ValueError("No valid partition labels available after NoData filtering")

    assignments_scope = assignments_df[assignments_df['partition_id'].notna()].copy()
    assignments_scope['partition_id'] = assignments_scope['partition_id'].astype('string')

    total_assignments = len(assignments_scope)

    try:
        polygons_gdf = gpd.read_file(shapefile_path)[['admin_code', 'geometry']].copy()
    except Exception as exc:
        raise FileNotFoundError(f"Cannot load shapefile: {shapefile_path}. Error: {exc}")

    polygons_gdf['admin_code'] = polygons_gdf['admin_code'].astype('string').str.strip()
    polygons_gdf = polygons_gdf.dropna(subset=['geometry'])

    assignments_presence = assignments_scope[['admin_code', 'partition_id']].copy()
    assignments_presence['partition_id'] = assignments_presence['partition_id'].astype('string')
    assignments_presence['in_assignments'] = True

    polygons_presence = polygons_gdf[['admin_code']].copy()
    polygons_presence['in_polygons'] = True

    join_audit = assignments_presence.merge(polygons_presence, on='admin_code', how='outer')
    join_audit['in_assignments'] = join_audit['in_assignments'].fillna(False)
    join_audit['in_polygons'] = join_audit['in_polygons'].fillna(False)
    join_audit['joined'] = join_audit['in_assignments'] & join_audit['in_polygons']
    join_audit['partition_id'] = join_audit['partition_id'].astype('string')
    join_audit.to_csv(output_dir / 'join_audit.csv', index=False)

    try:
        merged = polygons_gdf.merge(
            assignments_scope[['admin_code', 'partition_id']],
            on='admin_code',
            how='inner',
            validate='1:1'
        )
    except Exception as exc:
        raise ValueError(f"Failed to merge polygons with assignments: {exc}")

    if merged.empty:
        raise ValueError("Inner join yielded no records; verify UID alignment")

    merged['partition_id'] = merged['partition_id'].astype('string').str.strip()
    merged = merged[merged['partition_id'].isin(categories_in_data)]

    if merged.empty:
        raise ValueError("All joined partitions filtered out after normalization")

    drawn_categories = [
        cat for cat in categories_in_data
        if (merged['partition_id'] == cat).any()
    ]

    if not drawn_categories:
        raise ValueError("No partition categories remain for rendering")

    merged['partition_id'] = pd.Categorical(merged['partition_id'], categories=drawn_categories, ordered=False)

    gdf_plot = gpd.GeoDataFrame(merged, geometry='geometry', crs=polygons_gdf.crs)

    label_freqs_drawn = (
        gdf_plot['partition_id']
        .astype('string')
        .value_counts(dropna=False)
        .rename_axis('partition_id')
        .reindex(drawn_categories, fill_value=0)
        .reset_index(name='count')
    )
    label_freqs_drawn.to_csv(output_dir / 'label_freqs_drawn.csv', index=False)

    diff_table = (
        label_freqs_in_data.set_index('partition_id')
        .join(label_freqs_drawn.set_index('partition_id'), how='outer', lsuffix='_data', rsuffix='_drawn')
        .fillna(0)
    )
    diff_table['count_data'] = diff_table['count_data'].astype(int)
    diff_table['count_drawn'] = diff_table['count_drawn'].astype(int)
    diff_table['delta'] = diff_table['count_drawn'] - diff_table['count_data']

    n_labels_in_data = len(categories_in_data)
    n_labels_drawn = len(drawn_categories)
    n_missing_labels = len(set(categories_in_data) - set(drawn_categories))
    join_loss_count = max(total_assignments - len(gdf_plot), 0)

    diff_report_path = output_dir / 'labels_diff_report.txt'
    with diff_report_path.open('w', encoding='utf-8') as diff_handle:
        diff_handle.write(
            f"n_labels_in_data={n_labels_in_data}\n"
            f"n_labels_drawn={n_labels_drawn}\n"
            f"n_missing_labels={n_missing_labels}\n"
            f"join_loss_count={join_loss_count}\n"
        )
        diff_handle.write("\nlabel deltas (partition_id, count_data, count_drawn, delta):\n")
        diff_handle.write(diff_table.reset_index().to_string(index=False))
        diff_handle.write("\n")

    palette = get_colors(len(drawn_categories))
    palette = [to_hex(c) for c in palette]
    cmap = ListedColormap(palette)

    fig, ax = plt.subplots(figsize=figsize)

    gdf_plot.plot(
        ax=ax,
        column='partition_id',
        cmap=cmap,
        legend=False,
        alpha=1.0,
        edgecolor='black',
        linewidth=0.2
    )

    if add_basemap:
        try:
            if basemap_source is None:
                basemap_source = ctx.providers.CartoDB.Positron

            ctx.add_basemap(
                ax,
                crs=gdf_plot.crs.to_string(),
                source=basemap_source,
                zoom='auto',
                alpha=0.6
            )
        except Exception as exc:
            print(f"Could not add basemap: {exc}")

    legend_entries = []
    for color, cat in zip(palette, drawn_categories):
        count = int((gdf_plot['partition_id'] == cat).sum())
        legend_entries.append(Patch(facecolor=color, edgecolor='black', label=f"{cat} (n={count})"))

    legend_categories = [entry.get_label().split(' (n=')[0] for entry in legend_entries]
    if set(legend_categories) != set(drawn_categories):
        raise ValueError("Legend/data categories mismatch after construction")

    ax.legend(
        handles=legend_entries,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        title="Partitions"
    )

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    scoped_title = f"{title} • units={len(gdf_plot)} • categories={len(drawn_categories)}"
    ax.set_title(scoped_title, fontsize=16, fontweight='bold', pad=20)

    stats_text = (
        f"labels(data)={n_labels_in_data}\n"
        f"labels(drawn)={n_labels_drawn}\n"
        f"join_loss={join_loss_count}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    try:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"Partition map saved to: {save_path}")
    except Exception as exc:
        print(f"Could not save figure: {exc}")

    return fig
def plot_partition_map_from_result_dir(result_dir, year=None, model_type='GeoRF', VIS_DEBUG_MODE=None, **kwargs):
    """
    Convenience function to plot partition map from a result directory.
    
    Args:
        result_dir (str): Path to result directory (e.g., 'result_GeoRF_27/')
        year (str, optional): Specific year to plot (e.g., '2021'). If None, uses any available
        model_type (str): Model type for file naming ('GeoRF' or 'GeoXGB')
        **kwargs: Additional arguments passed to plot_partition_map()
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    import os
    import glob
    
    # Find correspondence table
    if year:
        if model_type == 'GeoXGB':
            pattern = f"correspondence_table_xgb_{year}.csv"
        else:
            pattern = f"correspondence_table_{year}.csv"
    else:
        if model_type == 'GeoXGB':
            pattern = "correspondence_table_xgb_*.csv"
        else:
            pattern = "correspondence_table_*.csv"
    
    correspondence_files = glob.glob(os.path.join(result_dir, pattern))
    
    if not correspondence_files:
        raise FileNotFoundError(f"No correspondence table found in {result_dir} with pattern {pattern}")
    
    correspondence_path = correspondence_files[0]  # Use first match
    
    # Set default title
    if 'title' not in kwargs:
        dir_name = os.path.basename(result_dir.rstrip('/'))
        file_name = os.path.basename(correspondence_path)
        kwargs['title'] = f"{dir_name} - {file_name}"
    
    return plot_partition_map(correspondence_path, VIS_DEBUG_MODE=VIS_DEBUG_MODE, **kwargs)


def plot_metrics_improvement_map(metrics_csv_path, 
                                metric_type='f1_improvement',
                                correspondence_table_path=None,
                                shapefile_path=None,
                                save_path=None,
                                title=None,
                                figsize=(12, 10),
                                dpi=300,
                                add_basemap=True,
                                colormap='RdBu',
                                center_colormap=True,
                                VIS_DEBUG_MODE=None):
    """
    Plot F1/accuracy improvement on a map using partition metrics and correspondence table.
    
    Args:
        metrics_csv_path (str): Path to CSV with partition metrics 
                               (output from PartitionMetricsTracker.save_metrics_to_csv)
        metric_type (str): Metric to plot ('f1_improvement', 'accuracy_improvement', 
                          'f1_before', 'f1_after', 'accuracy_before', 'accuracy_after')
        correspondence_table_path (str, optional): Path to correspondence table CSV
        shapefile_path (str, optional): Path to admin boundaries shapefile  
        save_path (str, optional): Output path for saved map
        title (str, optional): Map title
        figsize (tuple): Figure size
        dpi (int): Resolution for saved figure
        add_basemap (bool): Whether to add contextily basemap
        colormap (str): Matplotlib colormap name
        center_colormap (bool): Whether to center colormap at zero (for improvements)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    try:
        import geopandas as gpd
        import contextily as ctx
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError(f"Required packages missing: {e}. Install with: pip install geopandas contextily")
    
    # Default paths
    if shapefile_path is None:
        shapefile_path = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
    
    if save_path is None:
        save_path = metrics_csv_path.replace('.csv', f'_{metric_type}_map.png')
        
    if title is None:
        title = f"GeoRF {metric_type.replace('_', ' ').title()} by Administrative Unit"
    
    # Load metrics data
    try:
        metrics_df = pd.read_csv(metrics_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv_path}")
    
    # Validate metric type
    if metric_type not in metrics_df.columns:
        available_metrics = [col for col in metrics_df.columns if any(x in col for x in ['f1', 'accuracy', 'improvement'])]
        raise ValueError(f"Metric '{metric_type}' not found. Available metrics: {available_metrics}")
    
    # If no correspondence table provided, try to infer from metrics data structure
    if correspondence_table_path is None:
        # Look for correspondence table in same directory or result directory
        import os
        metrics_dir = os.path.dirname(metrics_csv_path)
        
        # Try to find correspondence table
        possible_paths = [
            os.path.join(metrics_dir, 'correspondence_table_2021.csv'),
            os.path.join(metrics_dir, 'correspondence_table_2022.csv'), 
            os.path.join(metrics_dir, 'correspondence_table_2023.csv'),
            os.path.join(metrics_dir, 'correspondence_table_2024.csv'),
            os.path.join(metrics_dir, '..', 'correspondence_table.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                correspondence_table_path = path
                break
                
        if correspondence_table_path is None:
            raise FileNotFoundError("No correspondence table found. Please provide correspondence_table_path parameter.")
    
    # Load correspondence table to map X_group to FEWSNET_admin_code
    try:
        corr_df = pd.read_csv(correspondence_table_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Correspondence table not found: {correspondence_table_path}")
    
    # Merge metrics with correspondence table
    if 'X_group' in metrics_df.columns and 'X_group' in corr_df.columns:
        merged_metrics = metrics_df.merge(corr_df, on='X_group', how='left')
    elif 'X_group' in metrics_df.columns and 'FEWSNET_admin_code' in corr_df.columns:
        # If X_group maps to FEWSNET_admin_code directly
        corr_df_renamed = corr_df.rename(columns={'FEWSNET_admin_code': 'X_group'})
        merged_metrics = metrics_df.merge(corr_df_renamed, on='X_group', how='left')
    else:
        # Try direct merge if X_group is actually admin codes
        merged_metrics = metrics_df.copy()
        merged_metrics['FEWSNET_admin_code'] = merged_metrics['X_group']
    
    # Load shapefile
    try:
        gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        raise FileNotFoundError(f"Cannot load shapefile: {shapefile_path}. Error: {e}")
    
    # Merge with geometries
    gdf = gdf[['admin_code', 'geometry']].copy()
    gdf['admin_code'] = gdf['admin_code'].astype(str)
    
    if 'FEWSNET_admin_code' in merged_metrics.columns:
        # Convert FEWSNET_admin_code to int first, then to string to remove decimal points
        merged_metrics['admin_code'] = merged_metrics['FEWSNET_admin_code'].astype(float).astype(int).astype(str)
        
        # Use outer join when configured to preserve polygons
        try:
            from config import USE_OUTER_JOINS
            join_type = 'outer' if USE_OUTER_JOINS else 'left'
        except:
            join_type = 'left'
            
        final_gdf = merged_metrics.merge(gdf, on='admin_code', how=join_type)
    else:
        # Fallback: try using X_group as admin_code  
        merged_metrics['admin_code'] = merged_metrics['X_group'].astype(float).astype(int).astype(str)
        
        # Use outer join when configured to preserve polygons
        try:
            from config import USE_OUTER_JOINS
            join_type = 'outer' if USE_OUTER_JOINS else 'left'
        except:
            join_type = 'left'
            
        final_gdf = merged_metrics.merge(gdf, on='admin_code', how=join_type)
    
    # Convert to GeoDataFrame
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry')
    
    # Track polygon losses before dropna
    initial_count = len(final_gdf)
    final_gdf = final_gdf.dropna(subset=['geometry', metric_type])
    final_count = len(final_gdf)
    
    if initial_count != final_count:
        print(f"Polygon loss in metrics visualization: {initial_count} -> {final_count} ({initial_count-final_count} lost)")
        try:
            from config import DIAGNOSTIC_POLYGON_TRACKING
            if DIAGNOSTIC_POLYGON_TRACKING:
                print("  Lost polygons due to missing geometry or metric data")
        except:
            pass
    
    if len(final_gdf) == 0:
        raise ValueError("No valid geometries found after merging. Check correspondence table and admin codes.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get metric values for colormap
    metric_values = final_gdf[metric_type].values
    vmin, vmax = metric_values.min(), metric_values.max()
    
    # Center colormap at zero for improvement metrics
    if center_colormap and 'improvement' in metric_type:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot metric values
    final_gdf.plot(
        ax=ax,
        column=metric_type,
        cmap=colormap,
        norm=norm,
        legend=False,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.2
    )
    
    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(
                ax,
                crs=final_gdf.crs.to_string(),
                source=ctx.providers.CartoDB.Positron,
                zoom='auto',
                alpha=0.7
            )
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(metric_type.replace('_', ' ').title(), fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add summary statistics
    stats_text = f"""
    Mean: {metric_values.mean():.4f}
    Std: {metric_values.std():.4f}
    Min: {metric_values.min():.4f}
    Max: {metric_values.max():.4f}
    """
    
    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    
    try:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Metrics improvement map saved to: {save_path}")
    except Exception as e:
        print(f"Could not save figure: {e}")
    
    return fig


def plot_partition_metrics_dashboard(metrics_tracker, output_dir, 
                                   correspondence_table_path=None,
                                   shapefile_path=None,
                                   create_maps=True):
    """
    Create a comprehensive dashboard of partition metrics including maps and summaries.
    
    Args:
        metrics_tracker: PartitionMetricsTracker instance with recorded metrics
        output_dir: Directory to save outputs
        correspondence_table_path: Path to correspondence table
        shapefile_path: Path to admin boundaries shapefile
        create_maps: Whether to create map visualizations
        
    Returns:
        dict: Summary of created outputs
    """
    if not VIS_DEBUG_MODE:
        return {'csv_files': [], 'map_files': [], 'summary': {}}
        
    import os
    
    # Save CSV files
    metrics_tracker.save_metrics_to_csv(output_dir)
    
    # Get summary statistics
    summary = metrics_tracker.get_improvement_summary()
    if summary:
        print("\n=== PARTITION METRICS SUMMARY ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
    
    outputs_created = {
        'csv_files': [],
        'map_files': [],
        'summary': summary
    }
    
    # Find created CSV files
    csv_files = [f for f in os.listdir(output_dir) if f.startswith('partition_metrics_') and f.endswith('.csv')]
    outputs_created['csv_files'] = csv_files
    
    # Create maps if requested and possible
    if create_maps and correspondence_table_path:
        for csv_file in csv_files:
            csv_path = os.path.join(output_dir, csv_file)
            
            try:
                # Create F1 improvement map
                f1_map = plot_metrics_improvement_map(
                    csv_path, 
                    metric_type='f1_improvement',
                    correspondence_table_path=correspondence_table_path,
                    shapefile_path=shapefile_path,
                    colormap='RdBu'
                )
                plt.close(f1_map)
                
                # Create accuracy improvement map
                acc_map = plot_metrics_improvement_map(
                    csv_path,
                    metric_type='accuracy_improvement', 
                    correspondence_table_path=correspondence_table_path,
                    shapefile_path=shapefile_path,
                    colormap='RdBu'
                )
                plt.close(acc_map)
                
                # Track created map files
                f1_map_file = csv_file.replace('.csv', '_f1_improvement_map.png')
                acc_map_file = csv_file.replace('.csv', '_accuracy_improvement_map.png')
                outputs_created['map_files'].extend([f1_map_file, acc_map_file])
                
            except Exception as e:
                print(f"Warning: Could not create maps for {csv_file}: {e}")
    
    return outputs_created


def plot_error_rate_choropleth(error_df, metric_col, shapefile_path=None, 
                              uid_col='FEWSNET_admin_code', title=None,
                              save_path=None, figsize=(12, 10), dpi=200,
                              missing_color='lightgray', crs_target='EPSG:4326', VIS_DEBUG_MODE=None):
    """
    Create choropleth map showing error rates by polygon.
    
    This function integrates with the existing visualization infrastructure to create
    consistent choropleth maps for pre-partitioning diagnostics.
    
    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame with polygon error statistics
    metric_col : str
        Column name containing error rates to visualize
    shapefile_path : str, optional
        Path to polygon shapefile. Uses default FEWSNET path if None
    uid_col : str, default='FEWSNET_admin_code'
        Column containing polygon unique identifiers
    title : str, optional
        Map title. Auto-generated if None
    save_path : str, optional
        Path to save the map. If None, displays only
    figsize : tuple, default=(12, 10)
        Figure size in inches
    dpi : int, default=200
        Output resolution for saved maps
    missing_color : str, default='lightgray'
        Color for polygons with missing data
    crs_target : str, default='EPSG:4326'
        Target coordinate reference system
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    try:
        import geopandas as gpd
        import contextily as ctx
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError(f"Required packages missing: {e}. Install with: pip install geopandas contextily")
    
    # Default shapefile path
    if shapefile_path is None:
        shapefile_path = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
    
    # Auto-generate title if not provided
    if title is None:
        metric_name = metric_col.replace('pct_err_', '').replace('_', ' ').title()
        title = f"{metric_name} Error Rate by Polygon"
    
    # Load and prepare shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Standardize column names
    if 'admin_code' in gdf.columns:
        gdf = gdf.rename(columns={'admin_code': uid_col})
    
    gdf = gdf[[uid_col, 'geometry']].copy()
    gdf = gdf.to_crs(crs_target)
    
    # Merge with error data
    merged_gdf = gdf.merge(error_df, on=uid_col, how='left')
    merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot polygons with valid data
    data_mask = merged_gdf[metric_col].notna()
    
    if data_mask.sum() > 0:
        merged_gdf[data_mask].plot(
            column=metric_col,
            ax=ax,
            cmap='Reds',
            vmin=0.0,
            vmax=1.0,
            legend=True,
            legend_kwds={
                'shrink': 0.8,
                'label': 'Error Rate',
                'format': PercentFormatter(xmax=1.0, decimals=0)
            }
        )
    
    # Plot polygons without data
    no_data_mask = ~data_mask
    if no_data_mask.sum() > 0:
        merged_gdf[no_data_mask].plot(
            ax=ax,
            color=missing_color,
            alpha=0.7,
            hatch='///',
            edgecolor='white',
            linewidth=0.1
        )
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    valid_values = merged_gdf[metric_col].dropna()
    if len(valid_values) > 0:
        stats_text = (f"Valid: {len(valid_values):,}\n"
                     f"Mean: {valid_values.mean():.1%}\n"
                     f"Std: {valid_values.std():.1%}\n"
                     f"Range: [{valid_values.min():.1%}, {valid_values.max():.1%}]")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend for missing data if needed
    if no_data_mask.sum() > 0:
        legend_elements = []
        if ax.get_legend():
            legend_elements = ax.get_legend().get_patches()
        legend_elements.append(Patch(facecolor=missing_color, hatch='///', 
                                   label=f'No Data ({no_data_mask.sum():,})'))
        ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Choropleth map saved: {save_path}")
    
    return fig
    # Strict gate: no image creation when disabled
    if not _resolve_vis_flag(VIS_DEBUG_MODE):
        return None
    # Strict gate
    if not _resolve_vis_flag(VIS_DEBUG_MODE):
        return None
    # Strict gate
    if not _resolve_vis_flag(VIS_DEBUG_MODE):
        return None
    # Strict gate
    if not _resolve_vis_flag(VIS_DEBUG_MODE):
        return None
