# New package location and dependency boundary — accepted R50

Source inspection on 2026-10-04; the user accepted this boundary as R50 / v0.44.
R51 authorizes infrastructure, freeze and Claude execution with Codex supervision.
At acceptance the directory did not exist; no package code or model fit had been created.

## Current dependency evidence

- `IPCCHGeoRFExperiment/run_pipeline.py:57-60` uses its own directory on sys.path
  for baseline_runtime and prepare_data; `:1104` also imports report_results.
  Its default backend is `GeoRFBaseline/releases/georf-baseline-v0.1.0.zip`
  (`:65,2970`), not the repository-root src tree. The ZIP is executable source:
  baseline_runtime.py:196-225 imports its extracted config and src.model.GeoRF;
  prepare_data.py:1895-1906 loads its adjacency helper. These runtime paths are
  not replaced merely by copying the outer run_pipeline.py.
- `IPCCHPopulationHistoryExperiment/prepare_data.py:42-49` adds the repository
  root and imports IPCCHGeoRFExperiment.prepare_data directly. Its preparation
  at :832-849 calls the old target ledger, country lookup and feature matrix;
  its rich561 code also uses FEATURE_COLUMNS and month_ordinal (:223,391).
  Copying only the history entrypoint leaves live old-package code dependencies.
- `FEWSNETGeoXGBExperiment/src/partition/transformation.py:13-27` imports bare
  config and src modules, including helper, training, scoring and visualization;
  partition_opt.py:22-28 does likewise. Both read bare config_visual dynamically.
  `scripts/run_experiment.py:67-70` prepends the package to sys.path. These names
  can bind the wrong package if copied without adapting the import boundary.
- `src/model/native_xgb.py:30` imports the FEWS experiment plan, while :44-49,
  157-174 and plan.py:18,25 assume integer four-class labels/probabilities.
  `src/experiment/stage3.py:96-97,378-380` and plan.py:10 also retain FEWS-specific
  windows/H; augmentation paths appear at :117-127,155-157,193,222-225. R1-R49
  already prescribe the new regression/data/time contracts instead.

These facts establish direct dependency edges, not a completed transitive import
audit or a finalized implementation file list. Source-package completion/audit
status does not prove acceptance of the new package.

## Accepted R50 package boundary

1. Create a new sibling package `IPCCHGeoXGBExperiment/` under this repository
   only after implementation is authorized. Use the IPCCH GeoRF completed
   package as the data/geography/calendar/evidence foundation. Keep the three
   reference experiment packages and sibling IPCCH repository unchanged.
2. Bring the necessary source functions into the new package as explicit local
   copies with a source manifest; adapt those copies to the accepted contracts.
   IPCCH GeoRF provides population QC/key/calendar/original-feature/reporting
   scaffolding; population-history provides the rich561 recipe; latest FEWS
   GeoXGB provides native booster continuation, scan/recursive routing and
   historical adoption structure. Sibling IPCCH provides the quartet/decoder
   reference. Copy only code required by the new path, not entire old workflows.
3. Record for each reused component its actual source path, inspected source
   byte hash and repository commit or release identity, plus the new local path
   and adaptation description. A commit alone cannot identify uncommitted source
   edits: record the actual source bytes and their relation to that commit.
   Preserve applicable source notices. Freeze the resulting package code/version
   before a scientific run; changing another experiment later must not change it.
4. At runtime, use package-qualified local imports and explicit local settings.
   Do not import executable modules from sibling experiment packages, silently
   load the old GeoRF ZIP backend, or resolve bare repository config/src through
   sys.path precedence. Use ordinary external libraries from the separately
   pinned environment; no shared-framework refactor of the existing packages is
   required. The final bounded source-copy/import list belongs in the execution
   plan, not an inferred permission to copy all transitive legacy dependencies.
5. Raw CSV and already frozen geography remain explicit, read-only external
   inputs. Configure paths and validate identities under R18/R39; the geometry
   may be located under the existing completed run without importing that run's
   Python code. Include a manifest of all additionally required source tables
   and their identities. Do not relocate, rewrite or repair original inputs.
   Code independence does not mean the deliverable embeds all raw data.
6. Rebuild new QC labels, >=.20-dependent history values, feature values, maps,
   models and predictions according to R1-R49. Do not treat old binary labels,
   feature caches, learned maps, donor assignments, fitted models or reported
   scores as new-run artifacts. R39's specified geometry/ID/adjacency reuse is
   the explicit exception for geographic inputs; its provenance limits persist.
7. New outputs and caches stay under the new package's run-specific locations,
   with separate run IDs and manifests. No writes into completed reference runs,
   no shared old-model cache, and no overwriting their configuration or results.
   Follow the existing prepare / learn-map / rolling-predict / report separation
   without introducing a consensus phase or a general plugin framework.

This creates a versionable implementation of the accepted design while retaining
the IPCCH completed package as its foundation. The cost is maintaining a bounded
local code copy and its provenance rather than automatically inheriting later
changes to reference packages. Exact software versions, final delivery contents
are fixed in the execution baseline. R51 supplies ordered implementation authorization.
