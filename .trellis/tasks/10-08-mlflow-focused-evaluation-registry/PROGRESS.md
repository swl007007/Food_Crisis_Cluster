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
5. [ ] Inventory + backup; apply; verify; repeat (no-op).
6. [ ] Payload/latency and browser checks; Windows Models/Registry/Inputs; docs; evidence.
