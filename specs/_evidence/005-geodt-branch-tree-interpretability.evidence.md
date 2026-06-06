# Evidence Pack: GeoDT Branch-Specific Local Tree Interpretability Figure

**Feature slug**: `005-geodt-branch-tree-interpretability`  
**Mode**: Mode 1 - Pre-spec Evidence Pass  
**Date**: 2026-06-05  
**Brownfield mode**: ON  
**Status**: Draft evidence for `specs/005-geodt-branch-tree-interpretability/`; no implementation changes made.

## Summary

This evidence pass supports a future exploratory GeoDT diagnostic that compares two least-similar terminal branch-specific DecisionTree models from one archived monthly GeoDT run, preferably `2024-10` with `fs1`, in a `1x2` shallow-rule figure.

Observed: Existing GeoDT `dt_rules_*.csv` artifacts are already specified and implemented as root/global DecisionTree rule exports. They load `branch_id=''` inside `app/main_model_DT.py:_export_dt_iteration_artifacts()` and therefore must not be treated as branch-specific local-tree or final branch-dispatched rule exports.

Observed: Branch-specific GeoDT model dispatch exists separately. GeoDT training saves a root checkpoint and branch-specific checkpoints, stores branch assignments in `space_partitions/`, and prediction dispatch loads a checkpoint for each branch ID in `DTmodel.predict_georf()`.

Inference: The proposed branch-tree visualization should be implemented as a separate exploratory diagnostic that reads branch-specific checkpoints and terminal branch assignment artifacts directly. It should not modify or reinterpret the existing `dt_rules` export contract.

## Evidence Status

Evidence Status: Incomplete but sufficient for pre-spec grounding.

Reason: Source-level evidence establishes the root/global-only meaning of current `dt_rules`, the branch-specific checkpoint naming/dispatch path, likely artifact sources, likely change surface, and artifact-hygiene constraints. Archive-level evidence is incomplete because this pass did not verify a concrete `result_GeoDT_2024_fs1_2024-10_visual` folder containing branch checkpoints and `space_partitions/` artifacts.

## Validation Status

Validation Status: Not Applicable

This Mode 1 pass did not execute the proposed diagnostic, load archived checkpoints, run model training, run smoke tests, or inspect generated figures. It only gathered repository evidence for future Speckit artifacts.

## Acceptance Readiness

Acceptance Readiness: Not Applicable

Implementation acceptance is not in scope for this evidence pass.

## Search Coverage

### Governance and project context inspected

- `AGENTS.md` - repository structure, Python path, batch commands, generated artifact guidance, testing guidance, and Windows CMD escaping guidance.
- `CLAUDE.md` - current workflow, GeoDT role, fs0/fs1-fs3 scope rules, prediction/scenario separation, generated artifacts, and operational limitations.
- `.gitignore` - generated artifacts are ignored, including `*.csv`, `*.png`, `*.pkl`, `result_*`, `deliverables/*`, and `.specify/*`.
- `specs/005-geodt-branch-tree-interpretability/spec.md` - feature spec that must preserve the root/global `dt_rules` boundary while defining branch-specific tree diagnostics.
- `specs/005-geodt-branch-tree-interpretability/plan.md` - implementation plan for archive characterization, branch-specific checkpoint loading, pair selection, and figure/metadata output.
- `specs/005-geodt-branch-tree-interpretability/tasks.md` - task list for implementing and validating the branch-tree interpretability diagnostic.

### Source and workflow files inspected or structurally searched

- `run_batches_2021_2024_visual_monthly.bat`
- `config.py`
- `app/main_model_DT.py`
- `src/utils/dt_rule_export.py`
- `src/model/GeoRF_DT.py`
- `src/model/model_DT.py`
- `src/model/train_branch.py`
- `src/helper/helper.py`
- `src/merge/terminal.py`
- `src/vis/visualization.py`
- `scripts/plot_monthly_performance_metrics.py`
- `scripts/plot_phase_change_monthly_performance.py`
- `scripts/plot_cluster_map_3x4.py`
- `scripts/plot_predictions_2024.py`

### Search terms used

- `_export_dt_iteration_artifacts export_dt_rules SAVE_DT_RULES dt_rules`
- `predict_georf get_X_branch_id_by_group s_branch X_branch_id branch_id checkpoint`
- `plot_tree DecisionTreeClassifier subplot matplotlib figure`
- `SAVE_DT_RULES --no-dt-rules dt_rules archive_folder space_partitions checkpoints correspondence_table`
- `result_GeoDT_2024_fs1_2024-10_visual result_GeoDT_*_visual checkpoints space_partitions s_branch.pkl correspondence_table_ dt_`

### Files intentionally not treated as stable proof

- Generated `result_*`, `*.csv`, `*.png`, `*.pkl`, notebook, and deliverable artifacts remain generated outputs unless explicitly promoted.
- Existing `dt_rules_*.csv` artifacts, if present, are not evidence for branch-specific rules because repository evidence shows they are root/global-only exports.

## Findings

### Existing entrypoints and controls

Observed: Standard GeoDT Stage 1 monthly evaluation is launched through `run_batches_2021_2024_visual_monthly.bat geodt`, which selects `app/main_model_DT.py` as the GeoDT entrypoint.

Observed: The batch launcher accepts `--no-dt-rules`; `grep` found `--no-dt-rules` parsing at `run_batches_2021_2024_visual_monthly.bat:73` and GeoDT `SAVE_DT_RULES` assignment at `run_batches_2021_2024_visual_monthly.bat:277-279`.

Observed: `config.py` parses `SAVE_DT_RULES` and `SAVE_DT_NODE_DUMP` from the environment. These existing controls define the root/global `dt_rules` background behavior that the branch-tree diagnostic must not change.

How this applies: The new branch-tree interpretability diagnostic should not change `SAVE_DT_RULES`, `SAVE_DT_NODE_DUMP`, or `--no-dt-rules` behavior. It should be a separate exploratory workflow or script.

### Existing `dt_rules` semantics are root/global-only

Observed: `app/main_model_DT.py:_export_dt_iteration_artifacts()` calls `model_wrapper.load('')` before exporting a fitted `DecisionTreeClassifier` through `export_dt_rules()`.

Observed: The same function writes files named like `dt_rules/dt_rules_<year>_fs<scope>_<year-month>.csv`, optional `dt_tree_nodes_*.csv`, `dt_rules_manifest.csv`, and `logs/dt_rules_export.log`.

Observed: `specs/005-geodt-branch-tree-interpretability/spec.md` includes requirements that current `dt_rules_*.csv` artifacts load the root/global checkpoint with `branch_id=''` and must not be represented as branch-specific local tree exports or final branch-dispatched prediction-path rules.

Inference: Existing `dt_rules` can provide background/provenance context, but the proposed feature must not use it as the source for the branch-tree figure.

### Branch-specific checkpoint training and naming exists

Observed: `src/model/GeoRF_DT.py:fit()` trains a root DecisionTree through `DTmodel(...).train(..., branch_id='')` and saves it with `self.model.save('')` before partitioning.

Observed: `src/model/train_branch.py:train_and_eval_two_branch()` trains and saves branch-specific children with `model.save(branch_id + '0')` and `model.save(branch_id + '1')`.

Observed: `src/model/model_DT.py:_get_checkpoint_path()` constructs checkpoint names as `dt_` plus the normalized branch suffix. Therefore the root/global checkpoint uses the historic empty suffix `dt_`, while branch-specific checkpoints use names such as `dt_0`, `dt_1`, `dt_00`, or similar binary-path suffixes.

How this applies: The proposed feature should discover branch checkpoint candidates from the selected run's checkpoint directory but must filter candidates by terminal branch usage, not filename presence alone.

### Final prediction dispatch is branch-specific

Observed: `src/model/model_DT.py:predict_georf()` computes or accepts `X_branch_id`, iterates over `np.unique(X_branch_id)`, calls `self.load(branch_id)`, predicts that subset, and writes predictions back into a full prediction vector.

Observed: `src/helper/helper.py:get_X_branch_id_by_group()` maps each record's group ID to a branch ID using `s_branch`, defaulting unmatched groups to the root branch `''`.

Inference: A correct branch-tree diagnostic should identify terminal branches from the same assignment data that prediction dispatch uses. Checkpoint filenames alone are insufficient because unused or non-terminal branch checkpoints may exist.

### Terminal branch assignment artifacts and correspondence tables exist in the training path

Observed: `src/model/GeoRF_DT.py:fit()` saves `s_branch.pkl`, `branch_table.npy`, and `X_branch_id.npy` under `self.dir_space`, corresponding to the run's `space_partitions/` directory.

Observed: `src/merge/terminal.py:build_terminal()` creates a correspondence dataframe containing `FEWSNET_admin_code`, `partition_id`, and optionally `branch_id`, preserving terminal lineage for diagnostics.

Observed: `src/model/GeoRF_DT.py:fit()` calls `build_terminal(X_group, X_branch_id)` and logs/saves label-frequency diagnostics under the run's `vis/` directory when visualization diagnostics run.

How this applies: The proposed feature should prefer `space_partitions/s_branch.pkl` and `space_partitions/X_branch_id.npy` for branch eligibility, with `correspondence_table_*.csv` as a fallback only if it preserves `branch_id` or terminal lineage.

### Archive behavior is not sufficient proof of branch diagnostic availability

Observed: `run_batches_2021_2024_visual_monthly.bat` archives visual outputs, `correspondence_table_*.csv`, `log_print.txt`, and DT-specific `dt_rules/` and `logs/dt_rules_export.log` artifacts.

Observed: Batch grep found archive lines for `correspondence_table_*.csv` at `run_batches_2021_2024_visual_monthly.bat:555-558` and DT rule archives at `run_batches_2021_2024_visual_monthly.bat:582-623`.

Observed: The grep pass did not find corresponding archive-copy lines for `checkpoints/` or `space_partitions/` in the same batch launcher output.

Assumption: Monthly visual archives may not contain branch-specific checkpoint files or `space_partitions/` unless another code path or manual copy preserved them. This must be characterized before implementation.

How this applies: The spec should require an archive-content characterization step before implementation. If preferred archive `result_GeoDT_2024_fs1_2024-10_visual` lacks checkpoints or branch assignments, the diagnostic should either locate the source run folder or fail clearly.

### Feature-name sources need explicit characterization

Observed: `src/model/GeoRF_DT.py:fit()` accepts `feature_names`, caches them, and writes a feature reference when available. It also updates feature names after feature dropping.

Observed: Existing `dt_rule_export.tree_to_dataframe()` falls back to generated `feature_<idx>` labels if feature names are missing or too short.

Inference: For a publication-quality branch-tree comparison, generic `feature_<idx>` labels may be unacceptable. The new feature should fail clearly if it cannot reliably map tree split indices to feature names from the selected run.

### Visualization implementation patterns exist

Observed: Existing scripts use `argparse`, `Path`, `matplotlib`, output directory validation, manifest generation, and fixed output artifacts. Relevant patterns include:

- `scripts/plot_monthly_performance_metrics.py:render_model_figure()` and manifest helpers.
- `scripts/plot_phase_change_monthly_performance.py:validate_output_dir()` and output artifact preparation.
- `scripts/plot_cluster_map_3x4.py:plot_grid()` for multi-panel figure layout.
- `scripts/plot_predictions_2024.py` for plotting helpers and fixed visual encodings.

Inference: A future implementation can follow the existing script-style diagnostics pattern under `scripts/` and use `src/vis/` only for reusable plotting helpers if needed. No new framework is indicated by repository evidence.

### Artifact hygiene and high-risk boundaries

Observed: `.gitignore` ignores `*.csv`, `*.png`, `*.pkl`, `result_*`, `deliverables/*`, and `.specify/*`.

Observed: `.specify/memory/constitution.md` and project guidance emphasize that generated diagnostics must not overwrite production forecast deliverables, scenario outputs, checkpoints, standard evaluation outputs, or partition artifacts.

How this applies: The proposed feature should write to a clearly scoped diagnostics output location and include metadata/provenance. It should not write into standard `result_GeoDT_*` training artifacts except by reading from them.

### Existing tests, commands, and known validation gaps

Observed: `AGENTS.md` lists `python -m src.tests.sig_test` for partition significance threshold changes and `python scripts/test_baseline_cache.py` for cache/lag handling.

Observed: `CLAUDE.md` documents Python 3.12+, conda/pip environment setup, the standard 3-stage batch workflow, and GeoDT single execution with `python app/main_model_DT.py --start_year 2024 --end_year 2024 --forecasting_scope 1`.

Observed: `specs/005-geodt-branch-tree-interpretability/tasks.md` lists a gap for adding a focused characterization test or smoke assertion that current `dt_rules_*.csv` export loads `branch_id=''`, and another gap to characterize whether monthly visual archives contain branch checkpoints and `space_partitions/`.

Inference: Before implementing the branch-tree figure, add or define focused archive-characterization and branch-selection smoke checks rather than relying on broad model training.

## Likely Change Surface

Likely implementation files, if/when this feature is built:

- `scripts/` - a new focused diagnostic script is the most likely entrypoint because existing plotting and analysis utilities live here.
- `src/vis/` - optional reusable rendering helper if the figure logic becomes shared.
- `src/utils/` - optional reusable DecisionTree shallow-signature helper if needed.
- `specs/005-geodt-branch-tree-interpretability/` - canonical feature directory containing `spec.md`, `plan.md`, `tasks.md`, and evidence references.

Files likely read by the diagnostic:

- Selected `result_GeoDT_*_visual/` archive or source run folder.
- `checkpoints/dt_*` branch-specific checkpoint files.
- `space_partitions/s_branch.pkl`.
- `space_partitions/X_branch_id.npy`.
- `space_partitions/branch_table.npy`.
- `correspondence_table_*.csv` when terminal branch labels are present.
- Feature-name reference written by `GeoRF_DT.fit()` or equivalent run metadata.

High-risk or read-only boundaries:

- `app/main_model_DT.py` and existing `dt_rules` export path should not be changed merely to support this exploratory figure.
- `src/model/model_DT.py`, `src/model/GeoRF_DT.py`, `src/model/train_branch.py`, and `src/helper/helper.py` define existing training/dispatch semantics and should not be refactored for the diagnostic unless explicitly authorized.
- Existing `result_GeoDT_*` model artifacts, checkpoints, and partition files should be read-only inputs.
- Batch launchers should not be changed unless the feature explicitly becomes a launcher-supported workflow.

## External Contracts, Data Artifacts, and Generated Outputs

Public APIs / CLI flags / config keys / file formats that must be preserved:

- `run_batches_2021_2024_visual_monthly.bat geodt`
- `run_batches_2021_2024_visual_monthly.bat geodt --no-dt-rules`
- `SAVE_DT_RULES`
- `SAVE_DT_NODE_DUMP`
- `app/main_model_DT.py --start_year --end_year --forecasting_scope`
- `dt_rules/dt_rules_*.csv`
- `dt_rules/dt_tree_nodes_*.csv`
- `dt_rules/dt_rules_manifest.csv`
- `logs/dt_rules_export.log`
- GeoDT checkpoint naming: `dt_`, `dt_0`, `dt_1`, `dt_00`, etc.
- `space_partitions/s_branch.pkl`
- `space_partitions/X_branch_id.npy`
- `space_partitions/branch_table.npy`
- `correspondence_table_*.csv`

Generated outputs proposed for the new feature:

- One publication-usable PNG and/or PDF 1x2 comparison figure.
- One metadata summary containing workflow mode, selected archive folder, run year/month, forecasting scope, model family, selected branches, checkpoint paths, branch assignment source, top-K depth, dissimilarity score, and output paths.

## Assumptions

Observed: The canonical feature slug is `005-geodt-branch-tree-interpretability`, per user correction. The root/global `dt_rules` behavior remains background evidence and boundary context, not the feature name.

Assumption: Preferred archive `result_GeoDT_2024_fs1_2024-10_visual` may be absent or incomplete in the local checkout. This pass did not verify its contents.

Assumption: Branch-specific checkpoints and `space_partitions/` may need to be read from a source run folder rather than a monthly visual archive, because the observed batch archive copy logic explicitly handles `dt_rules/`, logs, visual files, and correspondence tables but did not show checkpoint/space-partition copying.

Assumption: `correspondence_table_*.csv` is only sufficient if it includes reliable terminal branch labels. A table with only partition labels and no branch lineage may be insufficient for checkpoint loading.

Assumption: A new script under `scripts/` is the lowest-risk entrypoint, because existing exploratory/diagnostic plotting workflows already use scripts and generated outputs.

## Open Questions

1. Does `result_GeoDT_2024_fs1_2024-10_visual` exist in the target environment, and if so, does it contain `checkpoints/dt_*` branch checkpoints?
2. Does the selected archive contain `space_partitions/s_branch.pkl`, `space_partitions/X_branch_id.npy`, and `space_partitions/branch_table.npy`?
3. If the preferred archive lacks branch artifacts, which source run directory should be used as the authoritative input?
4. Which feature-name reference file is present in the selected run, and can it be mapped exactly to the loaded DecisionTree split indices?
5. Are branch IDs in `correspondence_table_*.csv` terminal branch IDs, partition labels, or display labels after adoption/inheritance?
6. Should root branch `''` be eligible if final dispatch routes any admin units to root, or should the figure require two non-root terminal branches to make the spatial-partitioning contrast clear?
7. What output directory should hold the generated figure and metadata so they remain ignored and do not overwrite production or paper deliverables?
8. Should the final figure use `sklearn.tree.plot_tree`, a custom shallow text/tree renderer, or a tabular rule-panel renderer? Repository evidence supports matplotlib, but not a dedicated existing branch-tree plotter.
9. Should branch-pair tie-breaking prioritize larger admin coverage, different root split feature, higher sample count, or visual readability?
10. What minimal smoke test should be accepted: artifact-existence inspection, synthetic tiny tree unit test, or a real archived-run dry run?

## Foundation Execution Evidence: T001-T018A

**Date**: 2026-06-05  
**Scope**: Speckit implementation foundation only; no behavior-changing diagnostic implementation was created.  
**Focused test path**: `src/tests/test_geodt_branch_tree_diagnostic.py`.

### T001 Active feature artifacts

Verified active planning artifacts for `specs/005-geodt-branch-tree-interpretability/` are present and aligned for implementation startup:

- `specs/005-geodt-branch-tree-interpretability/spec.md`
- `specs/005-geodt-branch-tree-interpretability/plan.md`
- `specs/005-geodt-branch-tree-interpretability/contracts/geodt-branch-tree-diagnostic-cli.md`
- `specs/005-geodt-branch-tree-interpretability/tasks.md`
- `specs/005-geodt-branch-tree-interpretability/implementation-micro-plan.md`
- `specs/005-geodt-branch-tree-interpretability/research.md`
- `specs/005-geodt-branch-tree-interpretability/data-model.md`
- `specs/005-geodt-branch-tree-interpretability/quickstart.md`

### T002 Focused test path

Selected `src/tests/test_geodt_branch_tree_diagnostic.py` as the focused test path.

**Rationale**: `src/tests/` exists and already contains project-local Python test modules. A top-level `tests/` directory was not present in the local checkout. The selected path keeps GeoDT diagnostic characterization and regression coverage close to the existing repository test layout without introducing a new top-level test convention.

### T003 Diagnostic script candidate path

Verified `scripts/plot_geodt_branch_tree_comparison.py` does not currently exist. The planned diagnostic script path is therefore available for the future T024 implementation task. No script was created during T001-T018A.

### T004 Migration status

No data migration, config migration, dependency migration, production artifact migration, or existing GeoDT artifact migration is required for the foundation phase or the planned exploratory diagnostic. Rollback for future implementation remains deletion of the new diagnostic script, focused test file, and generated diagnostics outputs only; existing GeoDT training artifacts remain read-only inputs.

### T005 Context pointer status

`CLAUDE.md` references `specs/005-geodt-branch-tree-interpretability/plan.md` as the current plan for additional context. `AGENTS.md` contains a generic Speckit context block rather than a conflicting 005 feature path. Older evidence-file guidance under "Copy into spec.md" is historical and does not override the current `spec.md` References section.

### T006 Root/global `dt_rules` characterization

Confirmed from source inspection:

- `config.py` defines `SAVE_DT_RULES` and `SAVE_DT_NODE_DUMP` as environment-derived controls.
- `app/main_model_DT.py` defines DT rule output constants for `dt_rules/`, `dt_rules_manifest.csv`, and `dt_rules_export.log`.
- Existing DT rule export flow resolves the root checkpoint path/hash using branch ID `''` and calls `model_wrapper.load('')` before exporting rules.
- `src/utils/dt_rule_export.py` exports whatever fitted `DecisionTreeClassifier` it receives and falls back to generated `feature_<idx>` labels when feature names are unavailable or too short.

Conclusion: current `dt_rules_*.csv` artifacts are root/global export artifacts and must not be treated as branch-specific local-tree or final branch-dispatched rule outputs for this feature.

### T007 Checkpoint naming and loader characterization

Confirmed from source inspection:

- `src/model/model_DT.py` normalizes branch IDs and maps root branch `''` to checkpoint filename `dt_`.
- Branch-specific IDs such as `0`, `1`, `00`, and `01` map to `dt_0`, `dt_1`, `dt_00`, and `dt_01` candidate filenames.
- Loader candidates include direct and absolute paths plus `.pkl` variants.
- Save behavior writes serialized checkpoints and records path overrides, with an in-memory fallback only for write failures during training.
- `src/model/GeoRF_DT.py` trains and saves the root checkpoint with `self.model.save('')` before partitioning.
- `src/model/train_branch.py` saves child branch checkpoints with `model.save(branch_id + '0')` and `model.save(branch_id + '1')`.

Conclusion: checkpoint filename presence alone is not enough for eligibility; the diagnostic must combine loadability with final terminal assignment evidence.

### T008 Branch assignment semantics characterization

Confirmed from source inspection:

- `src/helper/helper.py:get_group_branch_dict()` maps group IDs to branch IDs from `s_branch` columns.
- `src/helper/helper.py:get_X_branch_id_by_group()` initializes rows to root branch `''` and assigns branch IDs based on matching groups from `s_branch`.
- `src/model/GeoRF_DT.py` saves `s_branch.pkl`, `branch_table.npy`, and regenerated `X_branch_id.npy` under `space_partitions/`.
- `src/merge/terminal.py:build_terminal()` emits `FEWSNET_admin_code`, `partition_id`, and optional `branch_id`, preserving terminal lineage when `keep_branch_id=True`.

Conclusion: `X_branch_id.npy` and `s_branch.pkl` are the primary branch-assignment evidence candidates; correspondence tables are fallback evidence only when reliable terminal branch labels are present.

### T009 Local archive/source availability

Local bounded root inspection found no `result_GeoDT*` or `result_GeoDT*visual*` directories in the repository root. Therefore no real archive/source folder was characterized during T001-T018A. Real-archive dry run remains pending an explicit bounded archive path/root and is non-blocking for the synthetic fixture MVP foundation.

### T010 Feature-name source candidates

Because no local `result_GeoDT*` archive/source folder was present, selected-run feature-name source compatibility could not be verified against a real archive. Source inspection confirmed `GeoRF_DT.fit()` caches and writes feature references when `feature_names` are supplied, while `dt_rule_export` can fall back to generic generated feature labels. The diagnostic must reject incompatible or unreliable feature-name sources for publication-quality branch-tree plotting.

### T011-T014 Synthetic fixture foundation

Created `src/tests/test_geodt_branch_tree_diagnostic.py` with:

- a synthetic complete GeoDT-like archive fixture containing branch checkpoints, `space_partitions/s_branch.pkl`, `space_partitions/X_branch_id.npy`, `branch_table.npy`, `correspondence_table_*.csv`, and `feature_names.txt`;
- a visual-archive plus same-run provider fixture where the visual archive lacks branch artifacts and the provider contains them;
- negative fixtures for root/global-only `dt_rules`, feature-name mismatch, and split-index/feature-name incompatibility coverage.

### T015-T018 RED regression gates

Added focused RED regression tests in `src/tests/test_geodt_branch_tree_diagnostic.py` for:

- root/global-only `dt_rules` archive incompleteness and non-use as branch-tree source;
- no-overwrite behavior for pre-existing diagnostic metadata output;
- bounded archive discovery rejecting missing/unbounded roots;
- same-run artifact-provider metadata recording for visual archive plus source provider.

### T018A Foundation gate validation

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result: RED gate confirmed.

Observed result: 3 fixture-only tests passed and 4 diagnostic-behavior tests failed because `scripts/plot_geodt_branch_tree_comparison.py` does not exist yet. This is the expected pre-implementation failure mode for T015-T018 because T024 has not created the standalone CLI. No behavior-changing implementation was performed.

Failure summary:

- `test_root_global_only_dt_rules_archive_is_incomplete_and_not_used_as_branch_source` failed before diagnostic behavior because the script path is absent.
- `test_no_overwrite_regression_for_existing_production_artifact_paths` failed before diagnostic behavior because the script path is absent.
- `test_bounded_archive_discovery_rejects_unbounded_or_unrelated_search` failed before diagnostic behavior because the script path is absent.
- `test_same_run_artifact_provider_records_selected_archive_and_provider_paths` failed before diagnostic behavior because the script path is absent.

Foundation status: T001-T018A evidence and RED gates are in place. T024 may create the diagnostic CLI next, but T032/T033/T035 remain blocked until T041-T043 are complete.

## US1 RED Test Evidence: T019-T023

**Date**: 2026-06-05  
**Scope**: US1 test-writing tasks only; no diagnostic implementation was created for this RED run.

Extended `src/tests/test_geodt_branch_tree_diagnostic.py` with US1 RED coverage for:

- T019: default figure-generation CLI contract arguments and no mutation of `SAVE_DT_RULES`, `SAVE_DT_NODE_DUMP`, or `ACTIVE_LAGS`;
- T020: archive-list fallback/preflight metadata recording for incomplete and complete candidates;
- T021: checkpoint classification and branch eligibility reporting, including root/global exclusion;
- T022: feature-name source mismatch / split-index bounds reporting;
- T023: figure-generation output expectations for PNG, metadata, branch-specific provenance, readability, and neutral class labels.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result: RED gate confirmed.

Observed result: 3 fixture-only tests passed and 9 diagnostic-behavior tests failed because `scripts/plot_geodt_branch_tree_comparison.py` does not exist yet. This is the expected pre-T024 failure mode. T024 may now create the standalone CLI skeleton, but T032/T033/T035 remain blocked until T041-T043 are complete.

## CLI Skeleton Evidence: T024

**Date**: 2026-06-05
**Scope**: T024 only; created a standalone argparse CLI skeleton at `scripts/plot_geodt_branch_tree_comparison.py` without implementing archive resolution, artifact characterization, selection, rendering, metadata writing, or audit/reproduction behavior.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_default_figure_cli_accepts_contract_arguments_without_training_config_mutation" -q
python3 "scripts/plot_geodt_branch_tree_comparison.py" --help
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T019 CLI contract test passed.
- `--help` exited successfully and listed the required CLI skeleton flags, including archive inputs, output controls, `--k`, `--max-plot-depth`, `--audit-only`, and `--reproduce-from`.
- Full focused test file remains in expected partial-RED state: 4 tests passed and 8 tests failed because later preflight, audit, failure-summary, metadata, and figure-rendering behavior is not implemented yet.

T032/T033/T035 remain blocked until shared selection pipeline tasks T041-T043 are complete.

## Bounded Archive Resolver Evidence: T025

**Date**: 2026-06-05
**Scope**: T025 only; implemented bounded archive input handling for explicit archive path/list/root and deterministic minimum-completeness candidate decisions in `scripts/plot_geodt_branch_tree_comparison.py`.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_bounded_archive_discovery_rejects_unbounded_or_unrelated_search" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T020 archive-list fallback/minimum-completeness metadata test passed.
- Bounded missing archive-root rejection test passed.
- Full focused test file remains expected partial-RED: 7 tests passed and 5 tests failed because later provider, checkpoint/feature validation, no-overwrite, and figure-rendering behavior is not implemented yet.

No arbitrary filesystem search was introduced; archive-root discovery is limited to direct `result_GeoDT*` children of the explicit bounded root.

## Same-Run Artifact Provider Evidence: T026

**Date**: 2026-06-05
**Scope**: T026 only; implemented explicit `--artifact-provider-path` use for minimum-completeness checks and audit recording when a selected visual archive lacks branch checkpoints or feature-name artifacts.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_same_run_artifact_provider_records_selected_archive_and_provider_paths" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" "src/tests/test_geodt_branch_tree_diagnostic.py::test_bounded_archive_discovery_rejects_unbounded_or_unrelated_search" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T026 same-run provider path recording test passed.
- T025 fallback and bounded-root regression tests remained passing.
- Full focused test file remains expected partial-RED: 8 tests passed and 4 tests failed because later no-overwrite, checkpoint classification/eligibility output, feature-name validation, and figure-rendering behavior is not implemented yet.

## Artifact Characterizer Evidence: T027

**Date**: 2026-06-05
**Scope**: T027 only; added structured artifact characterization for archive/provider roots covering `checkpoints/dt_*`, `space_partitions/X_branch_id.npy`, `space_partitions/s_branch.pkl`, `branch_table.npy`, `correspondence_table_*.csv`, `feature_names.txt`, and the root/global `dt_rules` boundary note.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_root_global_only_dt_rules_archive_is_incomplete_and_not_used_as_branch_source" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" "src/tests/test_geodt_branch_tree_diagnostic.py::test_same_run_artifact_provider_records_selected_archive_and_provider_paths" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- Root/global-only `dt_rules` archive remains incomplete for branch diagnostics.
- Minimum-completeness and provider regression tests remained passing.
- Full focused test file remains expected partial-RED: 8 tests passed and 4 tests failed because later no-overwrite, checkpoint classification/eligibility output, feature-name validation, and figure-rendering behavior is not implemented yet.

## Branch Assignment Source Selector Evidence: T028

**Date**: 2026-06-05
**Scope**: T028 only; added assignment source precedence recording for `X_branch_id.npy`, `s_branch.pkl`, and `correspondence_table_*.csv`, including rejected lower-precedence source reasons in the audit artifact characterization.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_checkpoint_classification_and_branch_eligibility_exclude_root_global" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" "src/tests/test_geodt_branch_tree_diagnostic.py::test_same_run_artifact_provider_records_selected_archive_and_provider_paths" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T028 assignment/checkpoint visibility guardrail passed.
- T025/T026 resolver/provider regression tests remained passing.
- Full focused test file remains expected partial-RED: 9 tests passed and 3 tests failed because later no-overwrite, feature-name validation, and figure-rendering behavior is not implemented yet.

## Feature-Name Source Selector Evidence: T029

**Date**: 2026-06-05
**Scope**: T029 only; added feature-name source selection from `feature_names.txt`, feature-name count compatibility checks against branch DecisionTree checkpoint feature counts, and split-index bounds checks for branch checkpoint tree structures.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_feature_name_source_precedence_and_split_index_bounds_are_reported" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_checkpoint_classification_and_branch_eligibility_exclude_root_global" "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" "src/tests/test_geodt_branch_tree_diagnostic.py::test_same_run_artifact_provider_records_selected_archive_and_provider_paths" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T029 feature-name mismatch/bounds test passed.
- Prior artifact, assignment, fallback, and provider tests remained passing.
- Full focused test file remains expected partial-RED: 10 tests passed and 2 tests failed because later no-overwrite and figure-rendering behavior is not implemented yet.

## Checkpoint Parser and Loader Evidence: T030

**Date**: 2026-06-05
**Scope**: T030 only; added checkpoint records with filename, parsed branch ID, root/global vs branch-specific candidate classification, parsing method, load status, and mismatch reason for unloadable checkpoints.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_checkpoint_classification_and_branch_eligibility_exclude_root_global" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_feature_name_source_precedence_and_split_index_bounds_are_reported" "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" "src/tests/test_geodt_branch_tree_diagnostic.py::test_same_run_artifact_provider_records_selected_archive_and_provider_paths" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T030 checkpoint classification guardrail passed.
- T025-T029 regression tests remained passing.
- Full focused test file remains expected partial-RED: 10 tests passed and 2 tests failed because later no-overwrite and figure-rendering behavior is not implemented yet.

## Branch Eligibility Builder Evidence: T031

**Date**: 2026-06-05
**Scope**: T031 only; added branch eligibility records by joining assignment source counts, checkpoint classification/load status, and feature-source compatibility. Records include assigned count type, prediction-row count when available, admin/group count when available, unavailable training-sample marker, and exclusion reasons.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_checkpoint_classification_and_branch_eligibility_exclude_root_global" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_feature_name_source_precedence_and_split_index_bounds_are_reported" "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" "src/tests/test_geodt_branch_tree_diagnostic.py::test_same_run_artifact_provider_records_selected_archive_and_provider_paths" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T031 branch eligibility/count guardrail passed.
- T025-T030 regression tests remained passing.
- Full focused test file remains expected partial-RED: 10 tests passed and 2 tests failed because later no-overwrite and figure-rendering behavior is not implemented yet.

## Shared Selection Pipeline RED Evidence: T037

**Date**: 2026-06-05
**Scope**: T037 test-only task; added RED coverage for top-K branch signature extraction at depths `0` through `K-1`, actual available depth, split-feature set, and neutral leaf class summaries. This is shared selection-pipeline coverage required before US1 figure rendering and reused by US2 audit-only mode.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" -q
```

Result: RED confirmed. The test fails with `AttributeError` because `extract_branch_signature` is not implemented yet; T041 is expected to make this test pass.

## Shared Selection Pipeline RED Evidence: T038

**Date**: 2026-06-05
**Scope**: T038 test-only task; added RED coverage for deterministic branch-pair Jaccard scoring, exact formula expectations, ranking order, and proof that supplemental threshold/direction fields are not used for default ranking.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields" -q
```

Result: RED confirmed. The test fails with `AttributeError` because `score_branch_pairs` is not implemented yet; T042 is expected to make this test pass.

## Shared Selection Pipeline RED Evidence: T039

**Date**: 2026-06-05
**Scope**: T039 test-only task; added RED coverage for deterministic tie-break ordering, readability-gated selected-pair choice, rejected higher-scoring unreadable pair recording, and no-readable-pair failure.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_pair_selection_applies_tie_breaks_readability_gate_and_no_readable_failure" -q
```

Result: RED confirmed. The test fails with `AttributeError` because `select_readable_pair` is not implemented yet; T042/T043 are expected to make this test pass.

## Shared Selection Pipeline Implementation Evidence: T041

**Date**: 2026-06-05
**Scope**: T041 only; implemented `extract_branch_signature()` and a small tree-depth traversal helper in `scripts/plot_geodt_branch_tree_comparison.py`. The implementation records top-K split features by depth, split-feature set, actual available split depth, supplemental threshold/direction audit fields, and neutral `class N` leaf summaries without crisis/non-crisis labeling.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_checkpoint_classification_and_branch_eligibility_exclude_root_global" "src/tests/test_geodt_branch_tree_diagnostic.py::test_feature_name_source_precedence_and_split_index_bounds_are_reported" -q
```

Result:

- T041 focused signature extraction test passed.
- Nearby T031/T029 preflight regression tests remained passing.
- No scoring, readability selection, figure rendering, metadata writing, or audit-only wrapper behavior was implemented in T041.

## Shared Selection Pipeline Implementation Evidence: T042

**Date**: 2026-06-05
**Scope**: T042 only; implemented `score_branch_pairs()` plus contrast descriptor and deterministic tie-break field recording in `scripts/plot_geodt_branch_tree_comparison.py`. The implementation ranks by top-K split-feature Jaccard distance and records that threshold/direction fields are supplemental, not ranking inputs.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" -q
```

Result:

- T042 focused scoring test passed.
- T041 signature extraction regression test remained passing.
- No readability selection, figure rendering, metadata writing, or audit-only wrapper behavior was implemented in T042.

## Shared Selection Pipeline Implementation Evidence: T043

**Date**: 2026-06-05
**Scope**: T043 only; implemented `select_readable_pair()` in `scripts/plot_geodt_branch_tree_comparison.py` to consume scored pair records and external readability records, reject unreadable higher-scoring pairs with reasons, apply deterministic tie-break ordering, and fail with `No readable branch pair` when no candidate passes.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_pair_selection_applies_tie_breaks_readability_gate_and_no_readable_failure" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" "src/tests/test_geodt_branch_tree_diagnostic.py::test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields" -q
```

Result:

- T043 focused readability-gated selection test passed.
- T041 and T042 shared selection-pipeline regression tests remained passing.
- No figure rendering, metadata writing, or audit-only wrapper behavior was implemented in T043.

## Figure Renderer Implementation Evidence: T032

**Date**: 2026-06-05
**Scope**: T032 only; implemented the non-audit figure-generation path in `scripts/plot_geodt_branch_tree_comparison.py`. The renderer consumes the selected highest-ranked readable pair produced by the shared selection pipeline, loads branch-specific checkpoints, renders matched 1x2 shallow DecisionTree panels with neutral `class 0` / `class 1` labels, writes a 300 DPI PNG, and writes minimal renderer metadata required by the existing smoke test. Scoring and readability selection remain in T041-T043 helpers and were not duplicated inside the renderer.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_generation_writes_readable_png_metadata_and_neutral_labels" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" "src/tests/test_geodt_branch_tree_diagnostic.py::test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields" "src/tests/test_geodt_branch_tree_diagnostic.py::test_pair_selection_applies_tie_breaks_readability_gate_and_no_readable_failure" -q
```

Result:

- T032 focused figure-generation test passed.
- T041-T043 shared selection-pipeline regression tests remained passing.
- Full canonical metadata writer and no-overwrite/failure-summary behavior remain pending T033 and T034.

## Metadata Writer Implementation Evidence: T033

**Date**: 2026-06-05
**Scope**: T033 only; expanded the figure-generation metadata JSON writer in `scripts/plot_geodt_branch_tree_comparison.py` to record selected archive/run identity, artifact-provider path, minimum completeness status, branch assignment source, feature-name source, checkpoint provenance, selected branch counts, root/global exclusion, selected pair score, contrast descriptor, readability result, tie-break result, rejected higher-scoring pairs, neutral class-label policy, and output figure paths.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_generation_writes_readable_png_metadata_and_neutral_labels" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" "src/tests/test_geodt_branch_tree_diagnostic.py::test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields" "src/tests/test_geodt_branch_tree_diagnostic.py::test_pair_selection_applies_tie_breaks_readability_gate_and_no_readable_failure" -q
```

Result:

- T033 focused metadata assertions passed.
- T041-T043 shared selection-pipeline regression tests remained passing.
- No-overwrite output conflict handling and structured failure summaries remain pending T034.

## Output Safeguard Implementation Evidence: T034

**Date**: 2026-06-05
**Scope**: T034 only; implemented output path conflict detection, default no-overwrite behavior, and structured failure-summary JSON output for figure-generation artifact conflicts in `scripts/plot_geodt_branch_tree_comparison.py`. The diagnostic now checks planned PNG and metadata paths before rendering and fails without overwriting existing artifacts unless `--overwrite` is supplied.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_no_overwrite_regression_for_existing_production_artifact_paths" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_generation_writes_readable_png_metadata_and_neutral_labels" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" "src/tests/test_geodt_branch_tree_diagnostic.py::test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields" "src/tests/test_geodt_branch_tree_diagnostic.py::test_pair_selection_applies_tie_breaks_readability_gate_and_no_readable_failure" -q
```

Result:

- T034 no-overwrite and structured failure-summary test passed.
- T032/T033 figure and metadata regression test remained passing.
- T041-T043 shared selection-pipeline regression tests remained passing.

## Focused US1 Validation Evidence: T035

**Date**: 2026-06-05
**Scope**: T035 validation only; ran the complete focused GeoDT branch-tree diagnostic test file covering archive preflight, checkpoint classification, feature-name validation, figure rendering, no-overwrite behavior, metadata output, and shared selection-pipeline behavior. No implementation changes were made for T035.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- Focused US1 validation passed: 15 tests passed.
- Archive preflight, bounded discovery, same-run provider, root/global-only archive rejection, checkpoint classification, branch eligibility, feature-name compatibility, figure output, metadata output, no-overwrite failure summary, and T041-T043 shared selection-pipeline behaviors all remained green.
- Known baseline failures: none observed in this focused validation run.
- New regressions: none observed in this focused validation run.

## Synthetic Figure-Generation CLI Smoke Evidence: T036

**Date**: 2026-06-05
**Scope**: T036 validation only; ran a standalone synthetic figure-generation CLI smoke using the reusable focused-test fixture builder. Outputs were created under a temporary diagnostics directory in `/tmp`; no source archive artifacts or production outputs were modified.

Executed:

```bash
python3 - <<'PY'
# Imported build_complete_geodt_archive from src/tests/test_geodt_branch_tree_diagnostic.py,
# created a temporary synthetic archive, then ran scripts/plot_geodt_branch_tree_comparison.py
# with --archive-path, --output-dir, --k 3, and --max-plot-depth 3.
PY
```

Observed command output:

```text
Figure written: /tmp/geodt_t036_s32rxcfa/diagnostics/geodt_branch_tree_compare_2024-10_fs1_0_vs_1.png
Metadata written: /tmp/geodt_t036_s32rxcfa/diagnostics/geodt_branch_tree_compare_2024-10_fs1_0_vs_1_metadata.json
SMOKE_ARCHIVE=/tmp/geodt_t036_s32rxcfa/result_GeoDT_2024_fs1_2024-10
SMOKE_OUTPUT_DIR=/tmp/geodt_t036_s32rxcfa/diagnostics
SMOKE_PNG=/tmp/geodt_t036_s32rxcfa/diagnostics/geodt_branch_tree_compare_2024-10_fs1_0_vs_1.png
SMOKE_METADATA=/tmp/geodt_t036_s32rxcfa/diagnostics/geodt_branch_tree_compare_2024-10_fs1_0_vs_1_metadata.json
```

Result:

- Synthetic figure-generation CLI smoke passed.
- PNG and metadata outputs were generated under a temporary diagnostics directory.
- Metadata confirmed `workflow_mode == "figure-generation"`, selected archive provenance, readable selected pair, and existing output figure paths.
- Known baseline failures: none observed in this smoke run.
- New regressions: none observed in this smoke run.

## Audit-Only Contract RED Evidence: T040

**Date**: 2026-06-05
**Scope**: T040 test-only task; added RED contract coverage that audit-only mode must render no PNG/PDF outputs and must record the same selected archive, selected branch IDs, selected pair score, readability result, and tie-break result as figure-generation mode for the same synthetic input. The test also requires audit metadata to identify reuse of the shared selection pipeline.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_audit_only_writes_selection_parity_metadata_without_rendering_figures" -q
```

Result: RED confirmed. The test fails with `KeyError: 'selected_branch_ids'` because current audit-only mode writes only archive/preflight audit metadata and does not yet wrap the shared T041-T043 selection pipeline. T044 is expected to make this test pass without duplicating scoring/readability logic.

## Audit-Only Mode Implementation Evidence: T044

**Date**: 2026-06-05
**Scope**: T044 only; implemented audit-only selection metadata by wiring `--audit-only` through the existing preflight, eligibility, feature selection, signature extraction, scoring, and readability records produced by the shared T041-T043 selection pipeline. Audit-only writes an audit JSON with selected branch IDs, selected pair score, readability result, tie-break result, rejected higher-scoring pairs, shared selection-pipeline source marker, and no figure output paths. It suppresses PNG/PDF rendering.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_audit_only_writes_selection_parity_metadata_without_rendering_figures" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_preflight_metadata_records_archive_fallback_and_minimum_completeness" "src/tests/test_geodt_branch_tree_diagnostic.py::test_checkpoint_classification_and_branch_eligibility_exclude_root_global" "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_generation_writes_readable_png_metadata_and_neutral_labels" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T044 audit-only selection parity/no-figure contract test passed.
- Audit/preflight and figure-generation regressions remained passing.
- Full focused diagnostic suite passed: 16 tests passed.
- No duplicate scoring/readability implementation was introduced in audit-only mode; it reuses `_build_selection()` and the T041-T043 helpers.
- Known baseline failures: none observed.
- New regressions: none observed.

## US2 Focused Validation Evidence: T045

**Date**: 2026-06-05
**Scope**: T045 validation only; ran focused tests for signature extraction, Jaccard scoring, deterministic tie-breaks, readability-gated selection, audit-only output, and no-figure behavior.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_top_k_signature_extraction_records_depth_features_and_neutral_leaf_summaries" "src/tests/test_geodt_branch_tree_diagnostic.py::test_jaccard_scoring_is_deterministic_and_ignores_supplemental_threshold_direction_fields" "src/tests/test_geodt_branch_tree_diagnostic.py::test_pair_selection_applies_tie_breaks_readability_gate_and_no_readable_failure" "src/tests/test_geodt_branch_tree_diagnostic.py::test_audit_only_writes_selection_parity_metadata_without_rendering_figures" -q
```

Result:

- US2 focused validation passed: 4 tests passed.
- Signature extraction, Jaccard scoring, tie-break/readability selection, audit-only parity, and no-figure behavior all remained green.
- Known baseline failures: none observed.
- New regressions: none observed.

## Synthetic Audit-Only CLI Smoke Evidence: T046

**Date**: 2026-06-05
**Scope**: T046 validation only; ran a standalone synthetic audit-only CLI smoke using the reusable focused-test fixture builder. Outputs were created under a temporary audit diagnostics directory in `/tmp`; no PNG/PDF files were created.

Executed:

```bash
python3 - <<'PY'
# Imported build_complete_geodt_archive from src/tests/test_geodt_branch_tree_diagnostic.py,
# created a temporary synthetic archive, then ran scripts/plot_geodt_branch_tree_comparison.py
# with --audit-only, --archive-path, --output-dir, --k 3, and --max-plot-depth 3.
PY
```

Observed command output:

```text
Archive selection audit written: /tmp/geodt_t046_tzr599p5/audit-diagnostics/geodt_branch_tree_compare_2024-10_fs1_archive_selection_audit.json
SMOKE_ARCHIVE=/tmp/geodt_t046_tzr599p5/result_GeoDT_2024_fs1_2024-10
SMOKE_OUTPUT_DIR=/tmp/geodt_t046_tzr599p5/audit-diagnostics
SMOKE_AUDIT=/tmp/geodt_t046_tzr599p5/audit-diagnostics/geodt_branch_tree_compare_2024-10_fs1_archive_selection_audit.json
```

Result:

- Synthetic audit-only CLI smoke passed.
- Audit JSON was generated under a temporary diagnostics directory.
- No PNG or PDF outputs were created.
- Audit metadata confirmed `workflow_mode == "audit-only"`, selected archive provenance, selected branch IDs, readable selected pair, empty `output_figure_paths`, and `selection_pipeline_source == "shared"`.
- Known baseline failures: none observed.
- New regressions: none observed.

## Reproduction Metadata Schema RED Evidence: T047

**Date**: 2026-06-05
**Scope**: T047 test-only task; added RED coverage requiring figure metadata to contain reproduction inputs: archive discovery input, selected archive, artifact-provider path field, selected branches, checkpoint paths, feature-name source path, K, plotted depth, selected score, dissimilarity formula name/version, and top-K split-feature sets for selected branches.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_metadata_contains_reproduction_schema_inputs" -q
```

Result: RED confirmed. The test fails with `KeyError: 'archive_discovery_input'` because current figure metadata does not yet include the reproduction archive-discovery schema fields. T050 is expected to make this schema test pass as part of metadata loading/resolution support.

## Reproduction Mismatch RED Evidence: T048

**Date**: 2026-06-05
**Scope**: T048 test-only task; added RED coverage requiring `--reproduce-from` to fail with a mismatch report when a checkpoint recorded in prior metadata is missing, without reselecting a replacement pair or rendering PNG/PDF outputs.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_fails_on_missing_recorded_checkpoint_without_reselection" -q
```

Result: RED confirmed. The test currently fails because `--reproduce-from` falls through to normal archive input handling and reports `No bounded archive input was provided` instead of loading metadata and reporting a missing-checkpoint mismatch. T051/T052 are expected to make this test pass.

## Reproduction CLI Contract RED Evidence: T049

**Date**: 2026-06-05
**Scope**: T049 test-only task; added RED CLI contract coverage that `--reproduce-from <metadata.json>` should recover the recorded selected archive, branch IDs, selected pair score, and report `reselected_pair == false` without rendering PNG/PDF outputs.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_cli_recovers_recorded_selection_without_reselection" -q
```

Result: RED confirmed. The test fails because `--reproduce-from` currently exits with `No bounded archive input was provided` instead of loading the metadata path and producing a reproduction report. T050/T051 are expected to make this contract pass.

## Reproduction Metadata Loading Evidence: T050

**Date**: 2026-06-05
**Scope**: T050 only; added reproduction schema fields to figure metadata and implemented `--reproduce-from` routing that loads a metadata JSON path, resolves recorded archive/provider/branch/checkpoint/feature/K/formula inputs into a reproduction report, and records `reselected_pair == false` without rendering PNG/PDF outputs.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_metadata_contains_reproduction_schema_inputs" "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_cli_recovers_recorded_selection_without_reselection" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_generation_writes_readable_png_metadata_and_neutral_labels" "src/tests/test_geodt_branch_tree_diagnostic.py::test_audit_only_writes_selection_parity_metadata_without_rendering_figures" -q
```

Result:

- T047 reproduction schema test passed.
- T049 reproduction CLI contract test passed.
- Figure-generation and audit-only regressions remained passing.
- Reproduction mode now loads metadata and writes a report without default reselection.
- Full artifact mismatch verification remains pending T051/T052.

## Reproduction Verification Evidence: T051

**Date**: 2026-06-05
**Scope**: T051 only; implemented reproduction verification for recorded checkpoint paths during `--reproduce-from`. Reproduction now fails with a `reproduction-mismatch` report when a checkpoint path recorded in metadata is missing and records `reselected_pair == false`, preventing replacement selection. The happy-path reproduction report remains successful when recorded artifacts are present.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_fails_on_missing_recorded_checkpoint_without_reselection" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_cli_recovers_recorded_selection_without_reselection" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- Missing recorded checkpoint mismatch test passed.
- Reproduction happy-path report test passed.
- Full focused diagnostic suite passed: 19 tests passed.
- Reproduction does not reselect replacement pairs when recorded artifacts are missing.
- Known baseline failures: none observed.
- New regressions: none observed.

## Reproduction Report and Mismatch Failure Summary Evidence: T052

**Date**: 2026-06-05
**Scope**: T052 only; implemented structured reproduction failure-summary output for missing recorded checkpoints. Reproduction now writes a `*_reproduction_failure_summary.json` file containing the failure stage, mismatch type, missing recorded artifacts, source metadata path, and `reselected_pair == false`, while still avoiding replacement selection and PNG/PDF rendering.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_fails_on_missing_recorded_checkpoint_without_reselection" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_cli_recovers_recorded_selection_without_reselection" -q
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- T052 missing-checkpoint reproduction failure-summary test passed.
- Reproduction happy-path report test passed.
- Full focused diagnostic suite passed: 19 tests passed.
- Missing recorded checkpoint mismatches now produce a structured failure-summary file in the requested output directory.
- Reproduction still does not reselect replacement pairs or render PNG/PDF outputs on mismatch.
- Known baseline failures: none observed.
- New regressions: none observed.

## Focused US3 Reproduction Validation Evidence: T053

**Date**: 2026-06-05
**Scope**: T053 validation only; ran focused reproduction metadata, missing-checkpoint mismatch, and reproduction CLI contract tests for User Story 3.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py::test_figure_metadata_contains_reproduction_schema_inputs" "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_fails_on_missing_recorded_checkpoint_without_reselection" "src/tests/test_geodt_branch_tree_diagnostic.py::test_reproduce_from_metadata_cli_recovers_recorded_selection_without_reselection" -q
```

Result:

- Focused US3 reproduction validation passed: 3 tests passed.
- Metadata contains the recorded inputs needed for reproduction.
- Missing recorded checkpoint mismatch produces a structured failure summary without replacement selection.
- Reproduction happy path recovers the recorded selected archive, branch IDs, and selected pair score without rendering PNG/PDF outputs.
- Known baseline failures: none observed.
- New regressions: none observed.

## Synthetic Reproduction CLI Smoke Evidence: T054

**Date**: 2026-06-05
**Scope**: T054 validation only; ran a standalone synthetic reproduction smoke using the reusable focused-test fixture builder. Figure generation first produced metadata from a temporary synthetic GeoDT-like archive, then `--reproduce-from` recovered the recorded selection into a reproduction report without rendering PNG/PDF outputs.

Executed:

```bash
python3 - <<'PY'
# Imported build_complete_geodt_archive from src/tests/test_geodt_branch_tree_diagnostic.py,
# created a temporary synthetic archive, ran figure-generation mode to produce metadata,
# then ran scripts/plot_geodt_branch_tree_comparison.py --reproduce-from <metadata>
# with an explicit temporary output directory and verified selected pair parity.
PY
```

Observed command output:

```text
SMOKE_ARCHIVE=/tmp/geodt_t054_sg0fbf7c/result_GeoDT_2024_fs1_2024-10
SMOKE_METADATA=/tmp/geodt_t054_sg0fbf7c/figure-diagnostics/geodt_branch_tree_compare_2024-10_fs1_0_vs_1_metadata.json
SMOKE_REPRODUCTION_REPORT=/tmp/geodt_t054_sg0fbf7c/reproduction-diagnostics/geodt_branch_tree_compare_2024-10_fs1_0_vs_1_metadata_reproduction_report.json
SMOKE_SELECTED_BRANCH_IDS=['0', '1']
SMOKE_SELECTED_PAIR_SCORE=0.0
```

Result:

- Synthetic reproduction CLI smoke passed.
- Reproduction recovered the same selected branch IDs and selected pair score recorded in metadata.
- Reproduction report recorded `reselected_pair == false`.
- No PNG or PDF outputs were rendered during reproduction.
- Known baseline failures: none observed.
- New regressions: none observed.

## Documentation Alignment Evidence: T055-T056

**Date**: 2026-06-05
**Scope**: T055 and T056 documentation only; updated quickstart and CLI contract text to remove planning-only wording and align examples/outputs with implemented behavior.

Changes recorded:

- `specs/005-geodt-branch-tree-interpretability/quickstart.md` now describes the implemented standalone diagnostic, includes explicit `--k 3` and `--max-plot-depth 3` in figure/audit examples, includes an explicit reproduction output directory, and documents reproduction report/failure-summary behavior.
- `specs/005-geodt-branch-tree-interpretability/contracts/geodt-branch-tree-diagnostic-cli.md` now describes the implemented interface, no-render reproduction report path, and structured reproduction failure summary on missing/changed recorded artifacts.

Validation status:

- Documentation-only update; no runtime validation command was required for these tasks.
- The implemented CLI behavior referenced here was already validated by T053 and T054.

## Optional Real-Archive Audit-Only Dry Run Evidence: T057

**Date**: 2026-06-05
**Scope**: T057 optional validation only; attempted a bounded real-archive audit-only dry run for the preferred `2024-10` / `fs1` GeoDT case without modifying legacy training, dispatch, branch training, or archived artifacts.

Environment setup for the bounded run:

- Created/used `.venv-geodt-diagnostic` after the base WSL Python lacked project dependencies and the Windows Python environment had `numpy 2.4` incompatible with `numba` through `shap`.
- Installed missing runtime packages in the venv after user approval: `pyarrow` for `polars.to_pandas()` and `psutil` for monthly evaluation.
- Downgraded only the venv runtime to `pandas<3` after the legacy monthly split check compared `test_month.freq` to `'M'`, while pandas 3 returns a `<MonthEnd>` frequency object.

Bounded GeoDT source run attempted:

```bash
SAVE_DT_RULES=1 SAVE_DT_NODE_DUMP=0 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv-geodt-diagnostic/bin/python app/main_model_DT.py --start_year 2024 --end_year 2024 --forecasting_scope 1 --desired_terms 2024-10
```

Because `app/main_model_DT.py` hard-codes a Windows data path, the actual run used an in-memory `DATA_PATH` override to the WSL-resolvable CSV path:

```text
/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm_NGA.csv
```

Result of bounded source run:

- The run loaded data, prepared features, selected `max_depth=7`, and reached GeoDT model fitting for `2024-10` / `fs1`.
- The run failed before branch-specific partition/checkpoint artifacts were produced due to a legacy partition helper overflow:

```text
OverflowError: Python integer 32768 out of bounds for int16
```

- Partial `result_GeoDT_0` contained only the root/global checkpoint `checkpoints/dt_`, `feature_name_reference.csv`, logs, and validation/diagnostic CSVs. It did not contain branch-specific `dt_0` / `dt_1` checkpoints or `space_partitions/` artifacts needed by the diagnostic.
- This is recorded as a known baseline/runtime blocker for the real training path, not a regression introduced by the branch-tree diagnostic.

Bounded audit-only diagnostic attempted with the available real artifacts:

```bash
.venv-geodt-diagnostic/bin/python scripts/plot_geodt_branch_tree_comparison.py --audit-only --archive-path "deliverables/GeoDTExperiment_archived/GeoDTResults/result_GeoDT_2024_fs1_2024-10_visual" --artifact-provider-path result_GeoDT_0 --output-dir "other_outputs/geodt_branch_tree_diagnostic_real_run"
```

Result:

- The diagnostic correctly rejected the available real inputs because the visual archive lacked checkpoints/partition artifacts and the partial provider contained only root/global `dt_`.
- The rejection explicitly stated that root/global `dt_rules` archives are incomplete for branch diagnostics.
- Captured structured output was written to `other_outputs/geodt_branch_tree_diagnostic_real_run/audit_only_failure_summary.txt`.
- The captured summary recorded:
  - selected visual archive: `deliverables/GeoDTExperiment_archived/GeoDTResults/result_GeoDT_2024_fs1_2024-10_visual`;
  - artifact provider: `result_GeoDT_0`;
  - missing artifacts: at least two loadable branch-specific checkpoints, at least two eligible assigned branch checkpoints, and a compatible feature-name source;
  - checkpoint characterization: only `result_GeoDT_0/checkpoints/dt_` was present and classified as `root/global`;
  - branch eligibility: root/global checkpoint excluded because it is not explicitly terminal and has no assignment count in the selected source;
  - decision: rejected.

Validation status:

- T057 executed and produced the expected bounded failure/rejection result for the available real artifacts.
- No PNG/PDF figure was produced, and no archived artifact was modified.
- Known baseline failures: bounded GeoDT source training failed before branch artifact creation due to the legacy `init_s_branch` integer overflow.
- New regressions: none attributed to the branch-tree diagnostic.

## Source Boundary Validation Evidence: T058

**Date**: 2026-06-05
**Scope**: T058 validation only; confirmed read-only GeoDT source files and protected production artifact directories were not modified for this diagnostic feature.

Executed:

```bash
git diff --name-only -- app/main_model_DT.py src/model/GeoRF_DT.py src/model/model_DT.py src/model/train_branch.py src/helper/helper.py src/merge/terminal.py src/utils/dt_rule_export.py run_batches_2021_2024_visual_monthly.bat
git status --short -- app/main_model_DT.py src/model/GeoRF_DT.py src/model/model_DT.py src/model/train_branch.py src/helper/helper.py src/merge/terminal.py src/utils/dt_rule_export.py run_batches_2021_2024_visual_monthly.bat result_* deliverables other_outputs
```

Result:

- Both commands completed with no output.
- No diffs were present in the read-only GeoDT training, dispatch, branch training, helper, terminal merge, DT rule export, or batch launcher files.
- No protected production artifact directories reported tracked or untracked status changes in this check.
- Existing GeoDT training/evaluation behavior and production artifacts remained untouched.

## Final Focused Validation Evidence: T059

**Date**: 2026-06-05
**Scope**: T059 validation only; ran the full focused GeoDT branch-tree diagnostic test file covering brownfield preflight, archive/provider behavior, assignment and feature-name validation, checkpoint classification, eligibility, shared selection pipeline, figure generation, metadata, audit-only, reproduction, mismatch failure summaries, and output protection.

Executed:

```bash
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
```

Result:

- Final focused diagnostic validation passed: 19 tests passed.
- Known baseline failures: none observed in this focused validation run.
- New regressions: none observed in this focused validation run.

## Rollback and No-Migration Evidence: T060

**Date**: 2026-06-05
**Scope**: T060 migration/rollback documentation only.

Rollback scope:

- Remove the standalone diagnostic script `scripts/plot_geodt_branch_tree_comparison.py`.
- Remove the focused diagnostic tests in `src/tests/test_geodt_branch_tree_diagnostic.py` if rolling back this feature entirely.
- Remove generated diagnostics outputs created by local smoke/validation runs, if any remain outside temporary directories.
- Remove or archive the Speckit artifacts for `specs/005-geodt-branch-tree-interpretability/` only if the feature planning record itself is intentionally withdrawn.

No-migration confirmation:

- No database, schema, config, runtime flag, training pipeline, prediction dispatch, branch training, checkpoint format, partition artifact, `dt_rules` export, batch launcher, production deliverable, or evaluation-output migration was introduced.
- Existing GeoDT artifacts do not require migration or rollback.
- Existing GeoDT training/evaluation behavior is preserved; rollback is limited to removing the new standalone diagnostic/test/docs artifacts and generated diagnostic outputs.

## Deterministic Post-Implementation Validation Evidence

**Date**: 2026-06-05
**Scope**: Session goal validation; ran deterministic lint/format, typecheck, focused tests, import/package smoke, and post-implementation artifact validation for the implemented GeoDT branch-tree diagnostic.

Executed:

```bash
/home/swl007007/.local/bin/ruff format "scripts/plot_geodt_branch_tree_comparison.py" "src/tests/test_geodt_branch_tree_diagnostic.py"
/home/swl007007/.local/bin/ruff check "scripts/plot_geodt_branch_tree_comparison.py" "src/tests/test_geodt_branch_tree_diagnostic.py"
/home/swl007007/.local/bin/ruff format --check "scripts/plot_geodt_branch_tree_comparison.py" "src/tests/test_geodt_branch_tree_diagnostic.py"
/home/swl007007/.local/bin/ty check --python /usr/bin/python3 --extra-search-path "/home/swl007007/.local/lib/python3.12/site-packages" "scripts/plot_geodt_branch_tree_comparison.py" "src/tests/test_geodt_branch_tree_diagnostic.py"
python3 -m pytest "src/tests/test_geodt_branch_tree_diagnostic.py" -q
python3 - <<'PY'
# Imported scripts/plot_geodt_branch_tree_comparison.py, parsed implemented CLI flags,
# ran figure-generation, audit-only, and reproduction subprocess checks against a
# temporary synthetic GeoDT-like archive, and validated expected artifact schemas,
# paths, metadata, and absence of unexpected PNG/PDF/failure outputs by mode.
PY
```

Result:

- Ruff lint passed.
- Ruff format check passed after formatting the diagnostic script and focused test file.
- `ty` typecheck passed when pointed at the active Python user-site dependency path.
- Focused diagnostic tests passed: 19 tests passed.
- Import/parser smoke passed.
- Artifact validation passed for figure-generation metadata/PNG, audit-only JSON with no PNG/PDF outputs, and reproduction report with no reselection and no PNG/PDF outputs.
- Known baseline failures: none observed in these deterministic checks.
- New regressions: none observed in these deterministic checks.

## Suggested References

Include these high-signal references in the future `spec.md` or `plan.md` rather than listing every inspected source file:

- `specs/_evidence/005-geodt-branch-tree-interpretability.evidence.md`
- `specs/005-geodt-branch-tree-interpretability/spec.md`
- `specs/005-geodt-branch-tree-interpretability/plan.md`
- `specs/005-geodt-branch-tree-interpretability/tasks.md`
- `CLAUDE.md`
- `AGENTS.md`
- `.specify/memory/constitution.md`

## Copy into `spec.md`

```markdown
## References

- `specs/_evidence/005-geodt-branch-tree-interpretability.evidence.md` - Brownfield evidence for branch-specific GeoDT tree diagnostic, including root/global `dt_rules` boundary, branch checkpoint dispatch semantics, archive risks, and open questions.
- `specs/005-geodt-branch-tree-interpretability/spec.md` - Migrated contract proving existing `dt_rules_*.csv` artifacts are root/global exports and must not be treated as branch-specific local-tree rules.
- `specs/005-geodt-branch-tree-interpretability/plan.md` - Existing GeoDT DT rule export data flow and branch-semantics boundary.
- `specs/005-geodt-branch-tree-interpretability/tasks.md` - Follow-up gaps for root/global characterization and archive checkpoint/partition availability.
- `CLAUDE.md` - Current GeoDT workflow, forecasting scope rules, artifact boundaries, and operational constraints.
- `AGENTS.md` - Repository structure, batch workflow commands, testing guidance, and Windows CMD-safe batch conventions.
- `.specify/memory/constitution.md` - Pipeline contract, temporal/spatial integrity, validation evidence, and artifact hygiene rules.
```
