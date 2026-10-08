# Native registry research — preliminary, 2026-10-08

Installed MLflow3.17 supports create_external_model(name,source_run_id,tags,params,model_type,experiment_id), producing a LoggedModel with external flavor and a minimal MLmodel descriptor suitable for registry linking. Source: installed mlflow/tracking/fluent.py:2580-2640. It does not load or copy the original scientific model; catalog metadata can refer to existing parent artifact/model bundle identities. MLflow pyfunc explicitly refuses to load external models (pyfunc/__init__.py:1140-1146). This is a plausible discovery/lineage option, not an inference wrapper.

Read-only scout reported no existing registered models, logged models or evaluation datasets. Model registry entries, logged models, run-input datasets and evaluation datasets serve different UI paths; dataset display/API details need targeted follow-up after intended use is selected. A scout observed a 36-character limit for run-input dataset digest: do not blindly pass the existing 64-character cohort SHA; verify installed validation and preserve full hash separately.

First decision: registry for browsing/lineage/downloads, or directly loadable inference. For this task recommend the former. Exact per-family/H/arm/seed and fold granularity is not adopted yet. No models or datasets created; live store untouched.
