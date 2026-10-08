# Dataset registration options — installed MLflow3.17, 2026-10-08

Read-only API/source research; no registration or store mutations.

## Run-input metadata

MlflowClient.log_inputs(run_id,datasets=[DatasetInput(...)]) accepts mlflow.entities.Dataset(name,digest,source_type,source,schema,profile). There are no row records in this entity. Schema/profile are JSON strings; profile can retain num_rows. It is recovered as run.inputs.dataset_inputs. Native run Inputs visibility must be browser-verified. This does not populate the separate EvaluationDataset catalog.

Installed package evidence (relative to ~/.venvs/ipcch-mlflow/lib/python3.12/site-packages/mlflow): entities/dataset.py:10, entities/dataset_input.py:12, tracking/client.py:2797, store/tracking/sqlalchemy_store.py:1208. data/meta_dataset.py:13,60 offers metadata-only Dataset but lacks constructor profile; use entity API for full metadata.

Digest length max36 (utils/validation.py:83,808). Candidate native digest: first32 hex of the full semantic descriptor SHA256, retaining full digest and cohort/source hashes in tags/artifact. Validate collision/identity rather than treating short hash as sufficient authority. Within an experiment store reuses (name,digest) and retains first metadata (sqlalchemy_store.py:2499,2518).

## Separate Datasets catalog

EvaluationDataset is documented for GenAI inputs/expectations (entities/evaluation_dataset.py:38). create_dataset(name,experiment_id,tags) allows an empty shell, but schema/profile are None until records exist (sqlalchemy_store.py:7775). merge_records requires inputs and optionally expectations (entities/evaluation_dataset.py:218,313). Populating it means an explicit row representation; a tag claiming n does not make a native record count.

## Proposed scientific scope

Register evaluation cohort metadata without copying rows. Share across arms/seeds only when sorted keys, truth semantics/content and target/decoding definition match. Existing cohort key hashes alone do not prove same truth/features. Source-specific feature schema and preprocessing remain separate provenance; do not fabricate training associations. Native metadata granularity and original versus Summary experiment dataset IDs must be documented.

User decision needed: lightweight run Inputs metadata versus standalone Datasets page with explicit actual evaluation rows. Model catalog proposed granularity family x H x arm; seed/source run as versions, fold/region members referenced within a version. Not yet approved.
