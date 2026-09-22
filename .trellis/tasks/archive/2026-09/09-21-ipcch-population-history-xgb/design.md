# IPCCH population-history experiment — design v0.9

Final-review proposal, not implementation authorization. Requirements and exact
math are in prd.md and technical-contract.md. Do not infer new scientific options
from an implementation convenience. Original pipeline/data/results are immutable.

## Minimal file boundary

Create only IPCCHPopulationHistoryExperiment/ after execution approval, with
prepare_data.py, run_pipeline.py, report_results.py, test_contracts.py, README.md
and a local .gitignore for generated runs. No generic adapters, plugin registry,
task scheduler, model service or new dependency. Reuse ordinary pandas/numpy,
sklearn/XGBoost and the existing IPCCH pure helpers. Names below are proposed
interfaces to implement, not commands already run.

Use explicit package imports from the repository root. Avoid importing the old
runner just to fetch constants: its top level changes sys.path and imports bare
module names. Frozen JSON inventories are copied verbatim into the experiment's
configuration artifacts on implementation; no regenerated source-field discovery.
The new scripts can share small pure functions directly without extra framework.

## Preparation and reuse

1. Pin source, country lookup, original helper modules, technical manifests and
   runtime. Reuse prepare_data.build_target_ledger, check_target_gate and
   load_country_lookup; explicitly enforce returned gates and ID coverage.
2. Build original93 from the full valid ledger and full covariate scaffold through
   build_feature_matrix. Use the same source hash for ledger and feature loading.
   Separately attach q3 regression truth by unique original area/target keys;
   never overwrite the binary target consumed by the old helper.
3. Per area, precompute ordered valid distribution months/values and endpoint
   states. Build the new history at each unique (area,origin), reusing it across
   horizon/target rows sharing that origin. Never use a target label except as
   history when its observation month is actually <=that particular origin.
4. Append the ordered468 columns and assert rich width561. Save original93 and
   rich matrix/schema/audits; finite engineered values or explicit NaN only.
   Preserve a dated source ledger and per-row six selected observation keys so
   windows/features can be independently reconstructed without logging every sum.
5. Write a fold manifest for 144 development and122 main scheduled folds.
   Main predictions cannot be scheduled for scoring until the freeze manifest
   exists. No Stage1, Stage2, geometry repair, adjacency or donor completion.

## Runtime and fit loop

Run under the frozen Windows environment. Use baseline_runtime.extract_baseline
and baseline_imports for the existing RF imputer without applying the old polygon
patch. This context imports GeoRF classes but never fits GeoRF or writes partition
artifacts. Verify source paths and baseline payloads before and after use; do not
substitute root scripts with different SMOTE behavior.

Build boolean full/matched train masks once per fold from saved keys, not from
model-specific NaN filtering. RF imputation is fitted once on that matched pool
and reused across its candidates; XGB retains NaNs. Classifier constants and
regression constants use the explicit technical-contract path. Actual training
errors stop rather than fabricate a prediction or silently skip an arm.

Sequential candidates/folds; one thread per estimator. Save each candidate's
development scores, route and support with exact keys. Parameterize objective
explicitly for y/e/q3; use raw classifier scores plus the common crisis-oriented
mapping, and clipped regression output. No SHAP calculation or map rendering is
required for this experiment. Imputer prints go to logs, not performance claims.

The pilot uses one supported h1 development target for all six methods/configs;
its exact-identity artifacts are reused. It measures resource cost and wiring,
not a method-selection shortcut. Nonempty global/matched fitting failures stop
the affected run and preserve its partial evidence.

## Development freeze and final prediction

Pool each candidate's causal development predictions per horizon. Use bounded
threshold pairs and deterministic score ties; save sufficient counts/scores to
replay selection without fitting. Do not select with final labels or calibration
learned from final outcomes. Freeze all six per-horizon configurations and
thresholds plus one primary family. Record logical information cutoff2022-12,
actual execution timestamp and exact file/code hashes; do not backdate execution.

Refit each selected estimator at each main origin using only its allowed current
pool; model parameters change, scientific choices do not. Reuse fullpool XGB
predictions for no-history binary fallback and retain explicit route labels.
Save selected final models/constant specifications and RF fills, alongside
per-row raw scores, transformed scores, thresholds and hard decisions.

## Run evidence layout

```
runs/<fresh-id>/
  manifest.json, commands.json, run.log
  inputs/                    hashes, runtime, helper identity, frozen spec/configs
  baseline/                  verified pristine imputer provider
  data/                      valid/QC ledger, original/rich matrices, ordered schema,
                             country mapping, history-source keys, feature audits
  folds/                     calendar, ordered fitting/evaluation keys, support/routes
  development/               candidate predictions, parameters, threshold/selection ledgers
  freeze.json                selected family/configs/thresholds and bound identities
  main/                      selected models, imputation, keyed predictions
  reports/                   metrics, deltas, flips/share errors, subgroup tables,
                             shared country draws, rejection logs, leave-year-out verdicts
  validation/                independent checks, first/last-fold replay evidence
```

CSV.gz/JSON/NPY and existing estimator serialization suffice. Do not create a new
storage format. A fresh root is created by preparation; later commands may add
their declared stages to that same bound root, never overwrite completed stages.
Failed runs remain evidence. Record any controlled continuation with matching
stage/input identity; never call a partial run complete.

## Reporting and reproducibility

Reuse confusion_counts/class1_metrics, adapt only the new-arm validation and
joint country draw aggregation. Old whole-table validators bind old arms, and
old bootstrap draws separately by horizon; neither is a drop-in new reporter.
Validate unique model/area/target/horizon keys, identical truth/country/persistence
on comparison keys and exact E_history/E_no_history union=E_all.

Report every arm, not just the primary winner; distinguish three claim types and
same-information from full-training-pool contrasts. Frozen scores support an
independent report replay and first/last nonempty main fold at every horizon
under the pinned environment. These verification refits are evidence checks,
not extra candidates; account for them separately from the5916 fit budget.

Before audit closure, promote code, acceptance index and sufficient per-row/
training/selection/draw evidence into reviewed commit artifacts. Raw sources and
large matrices/models remain local with hashes; aggregate tables alone cannot
replace keyed evidence. If an artifact is too large for Git, establish an explicit
accessible, hash-bound audit bundle accepted by the existing controller contract;
do not claim an isolated reviewer can read an ignored run automatically.
