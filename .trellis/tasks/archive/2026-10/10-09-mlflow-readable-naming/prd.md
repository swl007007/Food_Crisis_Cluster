# MLflow readable naming rebuild

## Goal

The local IPCCH MLflow store (experiments `IPCCH` and `IPCCH Summary`) is hard to read:
internal code names in run/model/dataset names, five spellings of the horizon, one arm name
with different meanings across families, `main` meaning different years in different
families, and details that belong in descriptions packed into names. Rebuild the store from
the saved sources with one readable vocabulary so the user and supervisors/PIs can use it as
a dashboard. No retraining, no rescoring.

## Decisions (grill with the user, 2026-10-09)

Approach and audience
- D1 Rebuild, not patch: MLflow 3.17 cannot rename logged models or run-input datasets.
  Move the current store whole to `ipcch-mlflow-backups/` (restorable), re-import the six
  families into a fresh store at the same path, and change the importer so future imports
  use the new names.
- D2 Readers are the user plus supervisors/PIs. All names, tags and descriptions in English.

Vocabulary
- D3 Lead time: names say `3-month`; tag `lead_months` zero-padded (`01/03/06/12`); numeric
  param `lead_months`. Retire `H6`, `h6`, `h06`, `horizon`, `horizon_months`.
- D4 Families (slug / short title used in names): `geoxgb_reference` / GeoXGB reference,
  `geoxgb_yearly_refit` / GeoXGB yearly refit, `geoxgb_climate_swap` / GeoXGB climate swap,
  `geoxgb_maps_2024` / GeoXGB maps to 2024, `geoxgb_window_probe` / GeoXGB window probe,
  `mlp_residual_fixed_maps` / MLP residual on fixed maps. Long titles go in parent run names
  and descriptions; source run IDs move to provenance tags.
- D5 Arms by role, same meaning in every family: `persistence`, `pooled`,
  `partitioned_gated`, `regional_ungated`, `global_base` (MLP without residual); comparison
  panels from the reference run become `reference_<arm>`. Tag `arm_role` =
  baseline / candidate / diagnostic; window probe adds tag `window` = `36_month` /
  `full_history`. The exact mapping of each old arm is verified against the source code;
  if an old arm does not fit a role name, it gets its own name rather than being forced.
- D6 Periods: tag `period` = actual target-month span; tag `period_role` = `primary`,
  `holdout`, `combined`, `year_2025`/`year_2026` (maps_2024 recomputed single-year blocks),
  `selected_months` (window probe). Metric keys use the role.
- D7 Cohorts (long descriptive names, each with a 1-2 sentence definition):
  `all_scored`, `persistence_available`, `regional_model_fitted`,
  `regional_model_fitted_and_persistence_available`, `in_partition_map`,
  `regional_model_fitted_both_windows`, `regional_model_fitted_full_history_only`; gate
  sub-cohorts `gate_used_regional`, `gate_rejected_no_gain`,
  `gate_rejected_too_little_validation` (detailed experiment only).
- D8 Metrics keep `binary.*` / `four_class.*`; only `q3_r2_projected` ->
  `share_phase3plus_r2` (and `_raw`) and `n` -> `n_rows`.
- D9 Contrasts: one form `<A>_minus_<B>`; cross-family contrasts `*_minus_reference_*`.
  The dashboard carries only own-arm minus persistence and minus pooled on crisis F1
  (point delta, plus `ci_low`/`ci_high` where a bootstrap interval was saved).

Objects
- D10 Experiments `IPCCH - detailed runs` and `IPCCH - dashboard`.
- D11 Run names `<family short title> | <arm> | <lead>` (+ `| <window>` for the window probe,
  `| seed N` when a family has several seeds); parent runs use the long family title.
- D12 Dashboard is wide: one run per family x arm x lead (x seed); metric keys
  `<period_role>.<cohort>.<metric>`; the MLP gets an extra `mean of 3 seeds` row per arm and
  lead (mean of the three saved values; no interval).
- D13 Registered model per family x arm x lead, e.g.
  `IPCCH GeoXGB reference | partitioned_gated | 3-month`; versions = seeds; persistence is
  not registered. Models stay external descriptors (not loadable).
- D14 Evaluation datasets named `IPCCH eval | <lead> | <period span> | <cohort>`; one name
  must always mean one digest. New training-data descriptors, one per family x lead
  (context `training`), pointing to the prepared feature matrix and key table that every
  refit draws from, with the window rule and feature set in the description. They describe
  the candidate row pool, not the exact rows of each refit.
- D15 Tags: a short readable set without prefix; every hash, fingerprint, importer version
  and original ID under `_prov.` (nothing dropped).
- D16 Descriptions: four paragraphs (What / Compare with / Status / Caveats + original).
  Parent runs also state the family's already-accepted conclusion, quoted from the accepted
  reports or task closure; no new judgement.
- D17 The dashboard experiment description and `IPCCHMLflow/README.md` give a reading
  guide: vocabulary, comparability rules, copy-paste filters and suggested charts. Gate
  sub-cohorts stay out of the dashboard.

Process
- D18 Trellis task with a proportionate PRD; TDD. The close audit is waived by the user
  up front (recorded as a user waiver, not an audit pass).
- D19 Default rules: no new intervals are computed; seed means carry no interval; the
  `reconcile` command (a one-time repair of the old store) is removed because a fresh store
  never needs it.

## Acceptance Criteria

- [ ] Value mapping: every metric value in the old store (126 detailed records, 492 Summary
      rows) is found, through an explicit old->new name mapping, with the identical value in
      the new store. The only new values are seed-mean rows and dashboard contrasts copied
      from detailed records; each is traceable to its sources in the row document.
- [ ] Counts per family / arm / lead / cohort match the old store; no unmapped old metric.
- [ ] Every dataset name maps to exactly one digest; every metric in the dashboard is bound
      to the dataset of its period and cohort.
- [ ] Repeat import and repeat dashboard apply are no-ops; interrupted runs resume.
- [ ] Tests pass (existing behaviour kept where still relevant, new naming tests added).
- [ ] Browser screenshots of the dashboard, a run page, the Models page and run Inputs show
      readable names; default 100-row dashboard search payload is measured and reported.
- [ ] Old store kept as a backup until the user approves the new one.

## Out of scope

Retraining, rescoring, new scientific claims, model serving, row uploads, changes to the
source runs, automatic push/merge.
