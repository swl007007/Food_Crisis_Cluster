# P0 ongoing implementation: bounded supervisor sample

Scope: offline preparation only. These are ordinary repair findings, not a scientific stop or full P0 acceptance. No current configuration drift or incorrect numerical result was observed. Keep working under the existing P0 release; no EE contact or project-data fit.

1. `sources.py:20-22,39`: staged shapefile components have different basenames (`geometry_shp__...shp`, `geometry_shx__...shx`, `geometry_dbf__...dbf`). A direct geopandas read of the actual staged SHP in p0-local-20261010a fails because it looks for the SHP basename's SHX. Preserve the bundle association (e.g. a geometry directory using original filenames). Add a focused actual-staged-bundle readability test. Do not alter original sources or silently overwrite existing run evidence.

2. `preflight.py:29-40`, `artifacts.py:138-144`, `sources.py:29-32`: reusing a run currently accepts changed config identity, and reusing its source manifest returns the old inventory even when requested inputs/version/hashes change. Scratch probes confirmed both paths; the present developer run hashes still match. Validate frozen config identity on all reused run entry paths and compare requested source identity/inventory to the retained manifest before work. Add focused tests for config drift and source-request drift. Keep the fix small; distinguish disposable developer runs from formal frozen evidence and do not relabel existing runs.

Sample source hashes: sources.py 0f773ece0851d5f23cffb8f992308e6762465e5d2f5b536e1ec278ed7104345a; preflight.py 6286d7bfe7d751def12ea9b70ae731d38a6c6024f89fd3b61bb5de48a3c22a2b; prepare.py 9686668c9ca99ab779c2a2714b6057d654a7a0fd27b2478403bb30f8d4fbf074.

Other sampled preparation semantics showed no concrete regression: canonical population-valid keys, no feature-NaN row deletion, keyed origin T-H, invariant static values/missingness, exact 296 feature order, independent ordinary-value-at-origin source comparison. This is not full real-matrix or completed-P1 acceptance.
