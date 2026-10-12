# IPCCH AlphaEarth polygon streaming capacity, 2026-10-12

This is a capacity calculation, not a replacement extraction release. The current EE c4 run continues.

The fixed remaining-work snapshot at 05:10:52 UTC contains 4,535 active admins (one partially completed), 38,194 remaining pieces. Calculations below select complete polygons for those active admins; the one partial polygon therefore includes six already completed pieces. All sizes are decimal GB/TB.

Actual compressed GCS object metadata was obtained with 34 sequential read requests, about 6 MB metadata, zero TIFF pixel downloads. The 2017 prefix listing is complete. Polygon selection uses raw UTM TIFF bounds with a 20 m halo, densified frozen piece geometries and conservative fallback for invalid projected geometries. This screening has not been validated for scientific equivalence to EE.

| Per-admin statistic | Known compressed TIFF inputs (GB) |
|---|---:|
| Median | 9.83 |
| 90th percentile | 22.66 |
| 95th percentile | 31.35 |
| 99th percentile | 55.32 |
| Maximum (admin 100228, 159 TIFFs) | 373.27 |

With all parent-asset shards retained instead of spatial screening, median/p95/max are 22.20/57.58/475.53 GB.

Four admins each no larger than the single-admin 95th percentile need at most 125.39 GB of known compressed inputs before shared-file reuse. This is NOT a statement about the 95th percentile of four-admin batches. Four admins each at the single-admin 99th percentile would require 221.28 GB before reuse. The four largest selected admins have disjoint tile sets totaling 777.80 GB; fixed concurrency alone cannot enforce the cap.

Proposed capacity envelope (not implemented): 160 GB shared compressed-file cache, 40 GB reserved for bounded in-progress downloads, scratch and outputs; admission by the bytes of the union of pinned files, not solely worker count. Up to four ordinary admins can run when that shared budget allows. Larger admins run with reduced concurrency; the largest must stream smaller within-admin groups or COG blocks. A physical float32 mosaic is excluded from this capacity model. Statistics are accumulated as raw A/S/Q; decoded blocks are discarded. A 512x512x64 block is 16 MiB int8, 64 MiB float32 or 128 MiB float64; actual working RAM includes several arrays, reprojection buffers and library caches and needs measurement. A 1 GiB/worker target at four workers is plausible from block arithmetic, not measured peak memory.

Twenty remaining candidate parent IDs have no mapping in the supplied GCS index, affecting 252 active admins; their unknown files are not included in the sizes. Of 4,283 fully mapped admins, the numeric join separately reports which fit 40/80/160 GB. Missing parents and boundary/mask equivalence must be resolved before substituting local results.

Network workload matters separately from peak storage: the spatially screened active whole polygons reference 3,102 distinct TIFFs totaling 7.22 TB. Downloading each polygon independently and discarding all cache between polygons totals 55.50 TB (7.69x repeated transfer). A 160 GB shared cache can reduce but is not proven to eliminate repeated downloads. At an assumed sustained 100 MB/s, these two whole-file transfer workloads require about 20.1 h and 154.2 h respectively, excluding compute, queuing and overhead; no actual throughput was measured. COG Range reads can change these totals and were not benchmarked. The bucket is requester-pays.

Observed local environment: ArcGIS Desktop 10.8 Python 2.7 is 32-bit; tiny 64-band signed-int8 TIFF/BigTIFF/COG reads work. Default MosaicToNewRaster can fill validity separately per band and auto-create overviews; default Zonal Statistics uses cell-center rasterization, not fractional area weights. Neither default operation implements frozen D30 without additional controls. Existing ipcch-geo Python 3.12 is 64-bit and already has rasterio/GDAL, NumPy, pyproj and Shapely. Read-only hardware snapshot: D drive about 381 GiB free; C about 186 GiB free; WSL memory about 21 GiB available. Capacity alone does not establish accuracy or speed.

Evidence:
- /tmp/aef_polygon_compressed_capacity.json
- /tmp/aef_admin_tile_sets.json
- /tmp/aef_gcs_2017_object_sizes.json
- /tmp/aef_admin_capacity_summary.json
- /tmp/aef-capability/arcpy-observed.txt
- /tmp/aef-capability/rasterio-observed.txt
- /mnt/c/Users/swl00/Downloads/satellite_embedding_v1_annual_README.md

No production code, frozen rule, cloud export, source output, or active EE run was changed for this study. No real TIFF pixels, local polygon moments, streaming peak or end-to-end speed were tested.
