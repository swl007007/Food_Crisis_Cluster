# MLflow focused evaluation and model/dataset registration

## Goal

Provide a responsive IPCCH comparison surface and native model/dataset discovery, using saved results without retraining or losing historical evidence.

## Adopted requirements

- R1: Diagnose UI load and verify the improvement on the same local service. Hiding columns alone is insufficient: installed SearchRuns returns full metrics.
- R2: User adopted a separate `IPCCH Summary` experiment. Each record has at most eight scientific scores plus n; period/cohort are record metadata. Preserve original IPCCH records and detailed reports unchanged and link back.
- R3: User selected native external-model catalog registration for browsing, comparison, lineage and downloads. No inference wrappers, fabricated signatures, model loading, serving or retraining.
- R4: User selected row-free dataset metadata linked through run Inputs. Do not populate the separate GenAI EvaluationDataset catalog or upload input rows.
- R5: Preserve truth/decoding definitions, model branch semantics, source identities, NA reasons and actual cohort comparability. Dataset key equality alone does not prove common truth or features.
- R6: Retain isolated MLflow3.17, localhost hosting and existing durable storage. Use native SDK/UI, no dashboard, new platform or framework.

## Evidence and proposed scope

Read-only measurements: 126 original runs, 20,528 scalar values, 1,528 distinct metric names; default100 search response 3,817,125 bytes. API response is approximately0.36–0.48s; browser rendering has not been profiled, so dominant lag cause remains unproven. See research/ui-load.md.

Proposed whitelist yields492 Summary records, at most9 metric names across the whole experiment,4,424 finite values and4 NA values. Proposed model grouping yields68 registry names/100 external-model versions;20 persistence views have no model entry. These are final-design proposals for summary approval, not live changes.

The prior task close audit938a82278edb625c94c67468 has an open major gate: A01 reconciliation can mutate parent before detecting a child conflict. A02 omits inventory reference dependencies from per-family planning. Both are bounded code defects; the audit independently verified all currently imported child values and found no current data corruption. Resolve them through a dedicated repair child before ordinary feature execution; never clear controller state manually.

**Dependency status (2026-10-08, after execution):** repair child completed in two rounds (c22112d, then final 659f72d; tests 26/26). Gates 938a8227 and 14cfd4ce were WAIVED by the user (status waived, gate_open 0, resolved_by null) and the final repair run 497d0907 was closed without audit under the user's explicit override — a user waiver, not an audit pass. Records: archive/2026-10/10-08-mlflow-import-recovery-repair and archive/2026-10/10-08-mlflow-reconcile-content-preflight (evidence/non-audit-closure.md).

## Acceptance

- Original126 runs, source values/artifacts and historical identities unchanged by the feature operation; no original metric deletion.
- Summary492 records/9 possible metric names, exact source-valued scores, honest missing values and links; discrepancies require reconciliation before writes.
- Default100 Summary search payload <=1 MB versus measured3.817 MB baseline; browser load/filter/compare measured before/after and usability verified. Do not claim browser speedup from API timing alone.
- Native Models/Registry entries and Summary Inputs visible, with correct source/artifact links and no unsupported inference claim; model and dataset identity round-trip verifies.
- Repeat operation creates no duplicate records/models/versions/dataset attachments or metric history; interruption resumes visibly incomplete work.
- Existing import regression tests and focused new checks pass; pre-change backup and targeted post-write readback are retained.

## Authorization and status

User authorized task creation (2026-10-08), selected external catalog (1), adopted IPCCH Summary, and selected dataset Inputs metadata (1). Final summary approved by user “批准” on 2026-10-08, including the bounded repair prerequisite and Claude execution under Codex supervision. Execution is released subject to supported lifecycle/gate checks; no tracking mutations have occurred at release.

## Out of scope

Scientific reruns, new scores, model serving, raw-row uploads, GenAI evaluation catalog, wholesale historical-store migration, new experiment families, deletion of source or old MLflow records, automatic push/merge. Formal training dataset associations are deferred because complete fitting-data semantics are not established by evaluation keys.
