# Final convergence evidence — 2026-09-20

Planning-only, after Q9b approval. Three bounded read-only explorations checked
integration, support gates and geography. No source/code changes, transformed
panel, adjacency generation, repair or fitting. The final design/plan remains
in preparation after Q8g approval. Paths below are relative to the repository unless stated otherwise.

## Q8g: geometry repair policy approved

External root: `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\assembled_IPCCH`.

Live facts:
- `spatial/ipcch_admin_geometry.shp`:6,227rows, EPSG:4326,5,446Polygon and
  781MultiPolygon; none empty/missing, finite bounds, but253invalid geometries.
- Main thread reproduced invalid counts using the preferred Windows Python3.12,
  GeoPandas1.0.1 and Shapely2.1.0:212Self-intersection,30Ring Self-intersection,
  5Nested shells,6Too few points in geometry component. WSL python3 lacks GeoPandas;
  it was not installed or substituted into the execution environment.
- Examples from `explain_validity`: area100474,
  `Self-intersection[-9.8945607 8.4986919]`; area100630,
  `Ring Self-intersection[-10.5742701 9.0511333]`.
- Explorer verified string shapefile IDs normalize to the same unique6,227integer
  IDs as the coordinate/country CSVs. Range0..101324 is non-contiguous.
- `spatial/unique_area_id_lat_lon.csv`: all coordinates finite/in range; keyed
  shapefile coordinate attributes differ by at most5.000089231543825e-11.
  Use this reference CSV for donor distance, as already established in
  source-audit.md:43 and brainstorm.md:58-61; do not substitute geometric centroids.
- `country_area_id_lookup.csv`:53nonmissing country names.31Cote d'Ivoire areas
  have missing iso3 but valid country names/code;15Namibia country_code values
  are truly empty but iso3=NAM. Retain these areas; raw ISO completeness is not
  bootstrap-country eligibility. No inferred ISO repair is authorized.

Baseline utility `GeoRFBaseline/src/adjacency/adjacency_utils.py:18-38` returns
adjacency by array index, source-ID-to-index mapping and centroid coordinates.
`:51-64` silently substitutes an ID field/row index if the requested field is
missing; caller must reject that situation. `:73-75` computes centroids in source
CRS. `:98-108` uses touches plus positive shared-intersection length (point contacts
excluded), with no geometry validation/repair. Main thread inspected these lines.
Cache checks at`:171-175` cover path/ID field, not source contents; new run caches
must be bound to actual source components. No cached adjacency was generated here.

**APPROVED on2026-09-20:** during implementation, apply Shapely make_valid to
invalid geometries in an experiment-local copy, preserving raw source, area IDs,
reference coordinates and valid geometries. Save original validity reason,
before/after geometry type, validity and footprint/adjacency-change evidence;
where invalid originals prevent comparison, report that comparison unavailable.
Stop on remaining invalid/empty/non-polygon results or ambiguous area identity;
do not silently extract/drop components, exclude areas, replace boundaries or
disable spatial refinement. Repair establishes usable topology, not correct
administrative identity. Upstream unlimited-nearest-fallback provenance remains
unverified. Alternative requires an agreed valid geometry source or a separate
change to the spatial-learning design; neither is authorized.

## Direct Stage1 integration facts for the later design

- `GeoRFBaseline/src/model/GeoRF.py:44,122,219-259`: direct GeoRF construction/fit
  accepts prepared X/y/X_group and explicit split. Prefer a validated complete
  `split={"X_set": ...}` with0fit/1validation; validate values and all lengths.
  Four horizon views can share the same actual area group. Non-contiguous groups
  are initialized explicitly in `src/partition/transformation.py:161-166`.
- Polygon info fields: `polygon_centroids`, `polygon_group_mapping`, optional
  `neighbor_distance_threshold`/`adjacency_dict` (`transformation.py:399-407`).
  This mapping is polygon index to group, opposite the adjacency utility return;
  neighbors also use polygon indices (`partition_opt.py:725-736`).
- Core absolute `src.*` and `config` imports (`GeoRF.py:17-29`) require import
  isolation; `merge/terminal.py:15-24` can also read config_visual. Record loaded
  module paths, do not assume a GeoRFBaseline-prefixed import isolates dependencies.
- Keep F1 constants and delta>.01 (`config.py:29,32,306,342-346`), no SMOTE
  (`model_RF.py:83-85`). Disable redundant FEATURE_DROP for the final explicit
  IPCCH schema (`config.py:247-254`, `GeoRF.py:287`), without changing frozen files.
- Constructor `dir` does not control output: `GeoRF.py:87-109` calls default
  `create_dir` (`helper.py:308-348`), writing result_GeoRF under process cwd.
  Isolate cwd under the new run. Outer random_state/n_jobs are not proof of actual
  estimator configuration (`GeoRF.py:401`, `model_RF.py:46-54`).
- Baseline/source agreement for GeoRF.py, helper.py, transformation.py and config.py
  was checked against v0.1.0 ZIP by the integration explorer.

## Original-outcome support does not require changing q mathematics

Preserve original key/split/coverage before expansion. The frozen active gates:
- Parent positive validation count must exceed MIN_BRANCH_SAMPLE_SIZE=0
  (`transformation.py:310-314`, `config.py:214`). Complete four-view expansion
  preserves zero/nonzero status.
- All four child fit/validation sets must be nonempty (`transformation.py:693-708`,
  `helper.py:37-47`); the configured row threshold0 adds no higher requirement.
- MIN_SCAN_CLASS_SAMPLE=0 (`config.py:215`, `partition_opt.py:960-985`). Its quantity
  is q error mass, not independent outcome count: D=2TP+FP+FN, A=2TP and b derived
  from c=D-A (`partition_opt.py:134-146,220-225`). Do not divide these values by4.
- FLEX_TYPE=n_group, FLEX_RATIO=.1 (`config.py:216-222`, `partition_opt.py:250-267`);
  horizons do not create more spatial groups. Fewer than2groups produces an empty
  child (`partition_opt.py:842-847`). Spatial component size5 counts cells, not
  temporal rows (`config.py:315`, `partition_opt.py:315-323`).
- `select_f1_children` uses pooled validation confusion counts, parent wins ties,
  strict delta>.01 (`partition_opt.py:867-887`, `transformation.py:715-731`). This
  is a performance gate; horizon predictions can differ. Legacy significance
  tests are outside the active F1 path (`transformation.py:732-802`).
- RF estimator row-level tree defaults are distinct from experiment support;
  keep them and the approved per-class pseudo rows. No higher scientific support
  cutoff or independent-observation inference was approved.

## Export/refinement compatibility and final design resolution

- `GeoRF.py:438-447` saves s_branch, branch_table and regenerated row assignments.
  `helper.py:279-295` resolves groups by last matching branch and gives unknown
  groups the empty root label. Verify true group membership to distinguish actual
  learned root/ancestor membership from default-root placeholders.
- `merge/terminal.py:76-106` retains first assignment on collision and coerces
  malformed labels to root. Do not accept this export alone as donor provenance.
- Child checkpoints can contain adopted parent models (`transformation.py:718-728`);
  spatial branch and model-fit provenance are different. Preserve both evidence.
- Main thread confirmed `GeoRF.py:449-456` describes its final global refinement
  as grid-only, while calling it under the same CONTIGUITY flag used by training.
  `partition_opt.py:1029-1061` indexes a grid with group IDs. This is incompatible
  with directly treating sparse IPCCH area IDs as grid cells. The saved row map
  precedes this step but terminal CSV creation at`GeoRF.py:471` follows it.
  The final design keeps polygon learning/refinement and one consistent final
  learned map, using a local guard that skips this final grid block in polygon mode. Merely
  setting CONTIGUITY=False would also disable approved training refinement and
  is not an adopted solution. No code has been edited.
- DISABLE_BASELINE_CV_MAP only disables one diagnostic; a second CV diagnostic
  remains at`GeoRF.py:373` (`pre_partition_diagnostic.py:919-926`). These extra
  diagnostic fits are not required by the IPCCH protocol; design preserves inherited
  diagnostic behavior to avoid another core patch. Avoid accidentally
  turning their outputs into partition or final-test evidence.

## Next action

Q8g is approved. design.md/implement.md and converged PRD v1.0 now exist;
complete cross-document review, validate manifests and document limits.
Then present the final planning summary for implementation approval. Do not run
task.py start, repair geometry or train in this planning turn.

## Final contract review — 2026-09-20

Two independent read-only checks compared the pre-convergence approvals with
PRD v1.0 and compared PRD/design/implement. No scientific decision, A1–A8 ID or
material execution-contract inconsistency was found. The retention review found
five shortened research paths and a missing brainstorm.md evidence reference;
the main thread restored them. These are document checks, not model/data tests.
All three artifacts retain planning status and require final-summary approval.
