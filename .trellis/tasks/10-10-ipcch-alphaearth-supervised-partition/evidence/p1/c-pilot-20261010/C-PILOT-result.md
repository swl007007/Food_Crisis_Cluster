# C pilot (release ipcch-alphaearth-p1-C-pilot-20261010): technical stop before any computation

Pre-run checks passed: HEAD e4a04d2 (clean); ee_pilot_c.py f16be55c…, ee_metadata.py d246a6c5… (= release
source_sha256); released body 22a593ea…, 10 requests, limits unchanged; fresh
`/home/swl007007/ipcch-alphaearth-runs/p1-c-pilot-20261010`; `timeout 1200`.

| call | purpose | outcome |
|---|---|---|
| 1 | initialize | ok, 1.20 s |
| 2 | C01 overlap piece | error 0.17 s: "Projection: The CRS of a map projection could not be parsed." |
| 3 | C01 retry (wrapper's single retry) | same error 0.27 s → stop "failed twice" |

Calls 3 of 13; 0 responses; no A/S/Q, no area, nothing normalized; no export/write/fit. The request passed
`crs="EPSG:6933"` with `crsTransform` [10,0,0,0,-10,0]; EE (earthengine-api 1.7.36 service) does not parse that
authority code. This is a technical CRS-support stop, not a scientific result; the D30 recipe is unaffected.

Repair prepared (offline; needs a new release): the identical EPSG:6933 definition is sent as WKT1 (pyproj WKT1_GDAL
without the outer AUTHORITY tag), verified locally to equal EPSG:6933 with bit-identical transforms; the script first
asks EE to project three pinned reference points with that WKT and stops if any coordinate differs from pyproj by
> 1 mm. "could not be parsed" is now a non-retryable rejection (the wasted retry above would not recur). Rev2 manifest
`c_pilot_request_rev2.json` (body 8ef74cf2…; same 10 requests; +1 call for the check → ≤ 14 calls; release_id null).
