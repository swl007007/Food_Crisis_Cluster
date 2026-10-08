# PROGRESS — focused evaluation summary and native catalog

Operational log only; approval authority is the user and the supervisor (wN:p1).

## Lifecycle

- Repair prerequisite: c22112d (round 1), 659f72d (final round). Gates 938a8227 and 14cfd4ce
  waived by the user (not audit pass); final repair run 497d0907 closed without audit
  (archive b779159).
- Parent started by controller: run c9e5302e995a4df188546d4e98145d46, base b779159,
  executor 174ea213 / term_65d3f9b51d7fa2.

## Steps

1. [x] Scratch prototype (sqlite + scratch server on port 5031), MLflow 3.17:
   - LoggedModel names may not contain `.`, `/`, `:`, `%` or quotes → logged model names use
     `{family}-h{H}-{arm}-seed{seed}`; dotted registered-model names are allowed
     (`ipcch.{family}.h{H}.{arm}`).
   - `create_external_model` accepts a `source_run_id` in another experiment; the version's
     run_id becomes that original run. The original run's own tags/inputs/outputs are not
     modified (links live only in logged_models / model_versions rows).
   - Metrics logged with model_id + dataset name/digest appear on the LoggedModel with the
     dataset name; `search_logged_models` by tag and by dataset works. Repeated `log_inputs`
     does not duplicate dataset or model inputs. `get_metric_history` does not echo model_id.
2. [x] Read-only enumeration from the live accepted views reproduces 492 rows, 4,424 finite
   values, 4 NA, 0 absent; every value equals the live run metric.
3. [x] `IPCCHMLflow/summary_catalog.py` (plan/inventory/apply/verify) + 4 fixture tests
   (counts/links/NA, repeat no-op without metric history, interrupted resume, refusals).
   Full suite 30/30. Scratch UI: run Datasets (Evaluation) + Registered models + per-metric
   Models; model page Source run, Datasets used, versions, dataset-keyed metrics.
4. [x] Frozen plan a4d69a0082a2ef84… (evidence/frozen-plan-a4d69a00.json): 492 rows, 4,424
   finite, 4 NA (window H6 new_local_support macro-F1, 4 arms), 9 metric names, 68 registry
   names, 100 versions (76 fresh, 16 diagnostic-local, 8 reused-comparator), 57 dataset
   descriptors over 46 names. P6/MLP/yearly/climate share main E_all/E_persist descriptors
   (same keys and truth); split2024 main and window are separate.
5. [x] Pre-apply inventory (126 runs, sha d0147e5f…) and backup 20261008-pre-summary (DB sha
   97e2b373…, 2,039 files). Apply of a4d69a00 (388 s): 68 registered models, 100 external
   models, 100 versions, 492 rows, 4,424 metrics, 492 inputs; verify: 57 dataset descriptors,
   9 metric names, original 126 runs unchanged, catalog_status complete. Post-apply backup
   20261008-post-summary. Repeat apply: 492 noop, no creates, DB counts identical.
6. [x] Payload (UI 100-row search): IPCCH 3,817,125 B (unchanged from baseline) vs Summary
   710,070 B; warm API 0.06 s vs 0.37–0.62 s. Browser (headless Edge, 3 repeats, first rows):
   original 1.67–1.88 s before / 1.54–1.81 s after; Summary 1.52–1.94 s; filtered Summary
   1.01–1.39 s vs filtered original 1.32–1.57 s; compare ~0.7–1.1 s both. No first-paint
   improvement claimed. Windows: Summary list/row/Models/Registry/version screenshots; bundle
   download via a model tag URI (162,252,800 B, SHA matches model_bundle_sha256).
   Spec: .trellis/spec/backend/local-mlflow-summary-catalog.md. README updated.
7. [ ] Supervisor acceptance; close only on supervisor instruction (no auto audit).
