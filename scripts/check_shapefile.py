import geopandas as gpd

shapefile = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'

gdf = gpd.read_file(shapefile)
print("Columns:", gdf.columns.tolist())
print("\nFirst 5 rows:")
print(gdf.head())
