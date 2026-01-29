import geopandas as gpd
import pandas as pd

shapefile = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'

gdf = gpd.read_file(shapefile)
print("Country-related columns:")
print("\nADMIN0 unique values:", gdf['ADMIN0'].unique()[:10])
print("\nISO unique values:", gdf['ISO'].unique()[:10])
print("\nadm0_name unique values:", gdf['adm0_name'].unique()[:10])

print("\n\nSample rows:")
print(gdf[['admin_code', 'ADMIN0', 'ISO', 'adm0_name']].head(10))
