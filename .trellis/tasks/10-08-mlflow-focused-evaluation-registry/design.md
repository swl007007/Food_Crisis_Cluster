# Focused MLflow comparison and native catalog — proposed v1

Approved by user “批准” on 2026-10-08. Adopted: separate Summary, external model discovery, row-free run Inputs. Counts below are proposals enumerated from current accepted artifacts.

## 1. Existing authority

Original experiment IPCCH, six source parents/120 evaluation views, remains unchanged. Read accepted `view/evaluation_view.json` and archived parent manifests/predictions through MLflow or its verified local artifact store; do not depend on original Temp folders. Saved reports/metrics are score authority. Prediction rows may be read to establish cohort/truth identity, never to recompute performance. Reuse extractor cohort rules and run-key/source references.

## 2. Compact experiment

Create `IPCCH Summary`. One flat evaluation record per original child plus exact primary period/cohort. Explicit whitelist:

- Non-window: main/supplementary, own E_all/E_persist/local_eligible/local_persist_matched raw panels only.
- Window: selected_dates aggregate own all/mapped/common_local_support/new_local_support panels only.
- Exclude comparator panels, deltas, confidence bounds/draws, by-gate diagnostics, per-date window scores, and redundant split combined/year summaries from the compact metrics. Original details remain linked.
- Include only explicitly saved own panels, not synthetic rows inferred from n or NA metadata. Current whitelist has no saved n=0 panel; split H12 main absent, window new_local_support only H1/H6. Explain these gaps in the experiment description.

Uniform metric names (no period/cohort prefix): binary.accuracy, binary.precision, binary.recall, binary.f1, binary.f2, four_class.accuracy, four_class.macro_f1, q3_r2_projected, n. Keep undefined values absent and their reason in a small source-reference/NA artifact. Native timestamps describe this projection/import, not scientific fit time.

Enumerated records: P6 32; MLP248; yearly96; climate32; window56; split28 =492. Finite values4,424, NA4. The full experiment has only9 metric names. Tags/params carry family,H,arm,seed,period,cohort,original run ID,source path,source fingerprint and dataset identity. Keep row lists and full feature schemas out of run tags/search responses.

Provide native saved view if supported or documented filter/direct link: start with period=main, cohort=E_all, H=1, one seed42; users change H or choose E_persist/local cohorts for matched comparison. Persistence is not in E_all where absent. Supplementary and selected_dates require explicit selection. No global best-model ranking across unmatched data.

Use stable key original evaluation source_key + exact namespace. Copy metrics from saved panel with source paths; assert key uniqueness and consistent counts. Do not use the old reconcile command to remove metrics from IPCCH.

## 3. Models: external catalog, not callable packaging

Use installed `mlflow.create_external_model` and native registered model versions. Registry names group family x H x arm, e.g. ipcch.yearly_geoxgb.h1.geo. Version identity binds original run, seed and accepted artifact/source identity, not automatic score ranking. Preserve seed tag; version number is MLflow allocation, not seed number.

Current proposed inventory:68 names/100 versions (MLP16 names/48 versions, others52/52). Non-persistence views only:76 fresh_trained,16 diagnostic_local,8 reused_comparator. Window base arms stay explicitly reused comparisons; no new-fit claim. Persistence has dataset/summary records but no artificial model version.

One external model describes that arm's historical multi-fold/model recipe. Link original child and source parent, immutable model bundle URI/checksum/member inventory, preprocessing transforms, feature schema, maps, gate/route definitions and source code identity. Several catalog entries may reference one archive; do not duplicate weights or explode every fold/region/quartet into registry entries. Bundle references must explain that the parent archive includes several arms and the linked manifests describe member selection; do not imply every member belongs exclusively to one arm.

Create LoggedModels in the compact experiment with explicit original source-run provenance; verify installed native source_run_id/experiment association in an isolated fixture. If cross-experiment source_run_id is rejected, retain original run URI as provenance and leave native source_run_id unset rather than invent a training run. Native model version sources point to external model descriptors. No production/latest/best alias or deployment stage.

Associate compact metrics with external model_id and dataset name/digest using native Metric fields, enabling dataset-aware model comparisons. Persistence metrics have no model_id. Test native Models and Registry display plus artifact links. A READY external model means catalog descriptor ready, not loadable inference. Explicit external/multi-fold metadata in descriptions.

## 4. Datasets: evaluation metadata in Inputs

Use MlflowClient.log_inputs and entities.Dataset/DatasetInput. No merge_records or separate EvaluationDataset creation. Dataset represents evaluation keys and truth, not a feature matrix or training inputs.

Build a canonical descriptor including dataset version/source authority,H,actual date bounds,period/cohort,sample count,unique sorted keys hash,truth-value digest and target/phase/decoding semantics. Bind row identity against archived prediction keys and values; do not infer truth equality merely from the existing cohort hash. Name/digest reuse across arm/seed only when the whole evaluation descriptor matches. Conservative separation across different source identities is acceptable; no speculative cross-study deduplication.

Schema contains compact evaluation columns only (area key,target time,truth/target definitions as available); profile contains num_rows. Full feature schemas, fitting manifests and transformations are linked separately from model metadata and not inlined in every Inputs descriptor. Mark context=evaluation, historical import. Do not assert an unverified training dataset.

Native digest max36 chars: use first32 hex of full descriptor SHA256, retain full SHA,cohort/truth/source hashes in InputTags and descriptor artifact. Check existing name/digest metadata agrees fully before reuse; stop on collision/conflict. Count dataset identities from the actual canonical descriptors in read-only planning before writes, not by assuming fixed number of feature matrices.

## 5. Controlled writes and performance

Use existing venv/server/store and import lock. Add the smallest dedicated summary/catalog command beside current tools, consuming accepted imports without changing their frozen extraction/fingerprints. No plugin registry, custom DB tables or dashboard.

Before applying: capture source run IDs/metrics/params/tags/artifact manifests, existing model/dataset identities and SQLite+artifact backup. Preflight every record/metric/descriptor and all reference links. Stable projection keys and explicit original fingerprints govern idempotence; conflicts stop. Journal newly assigned run/model/version IDs. Mark the operation complete last, after exactly492 planned Summary records and enumerated model/dataset identities verify. No source history deletion.

The native external-model API can mark a descriptor READY before the whole operation finishes; separate catalog_status=incomplete/complete records the operation, so readiness is not overstated. Repeated execution must reuse IDs and not re-log unchanged metric history. Preserve partially created records and resume exact same plan; avoid complex generalized migration.

Measure same100 run search payload and three request timings on original vs Summary. Target Summary <=1MB; keep compact metadata small. Profile actual browser initial load/filter/compare and report before/after; if browser remains slow, test minimal native filtering/paging before claiming resolution. Do not change MLflow internals or install another front end.

## 6. Repair dependency / execution boundary

Before feature implementation, execute the dedicated repair child `.trellis/tasks/10-08-mlflow-import-recovery-repair` under this task for accepted audit job938a82278edb625c94c67468. Repair only A01 (validate all intended parent/child changes before any write; safe refusal and documented resumable reconciliation) and A02 (include inventory reference parents in plan dependency closure without broadening requested writes). Test late-child conflict leaves original records usable and window-only planning includes P6. Existing data has no demonstrated corruption; no real-store reconcile is needed to repair code.

Commit approved artifacts, verify actual Claude Opus5.5 1M executor, start repair with --remediation-for using the exact registered repo path. Close repair through supported controller; keep evidence and await gate resolution before ordinary main-task start. Do not reset/override a gate or change active executor. Main task then executes summary/catalog/Inputs. Final approval covers this bounded dependency and feature plan, not an independent science audit. No push/merge implied.
