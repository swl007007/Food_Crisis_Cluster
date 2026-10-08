# Local MLflow experiment management

## Goal

Install isolated local MLflow so the user can browse, compare and download the retained results/models of six recent IPCCH studies from one local UI, with reproducible provenance and explicit comparison cohorts.

## Requirements

- R1: Provide a working local MLflow installation and a unified view of existing experiments, not only an installation command.
- R2: Preserve frozen modeling environments, source runs, models, predictions and scientific definitions. Historical ingestion must not retrain models or imply that an imported run was executed by MLflow.
- R3: Preserve provenance and comparability boundaries: repository, original run ID, code/config/input identities, horizons, periods, evaluation cohorts, truth/decoding definitions and metric NA semantics must remain distinguishable.
- R4: Freeze the experiment inventory, comparison layout, storage and import rules before implementation. Adopted scope/storage/host are R6–R8; proposed complete layout and acceptance are in design.md for final review.
- R5: No cloud service, external publication, model deployment or change to scientific training code is implied by this request.
- R6: User selected initial scope option1: recent IPCCH experiments in this repository—P6 GeoXGB, fixed-map MLP, yearly GeoXGB, climate perturbation, and related history-window/split2024 sensitivities. Exclude adjacent IPCCH global/oracle/threshold families and FEWSNET/ETH/GeoRF/GeoDT historical families from this first import.
- R7: User selected full retained experiment-artifact storage: copy reports, configurations, predictions, fitted models and checksum inventories into durable managed storage, preserving originals. Register original input paths and hashes instead of duplicating raw/prepared input datasets. Imported experiment artifacts must remain accessible if the old Temp run directories are removed. This is not a promise of self-contained retraining without original inputs.
- R8: User adopted an isolated WSL virtual environment, local SQLite backend and durable storage under the Linux user home outside Dropbox. Bind the tracking server to localhost only; Windows browser access is intended via localhost:5000 and must be verified. Provide explicit start/stop commands; no cloud service, LAN exposure or modification to frozen Windows model environments.
- R9: Proposed comparison layout: one IPCCH experiment, six original-source parents holding retained artifacts once, and 120 model-arm/horizon/seed evaluation views (126 total records). Explicitly distinguish persistence, reused baselines and diagnostic local predictions from fresh trained models. Main/supplementary/selected-date periods and each evaluation cohort remain separate metric namespaces. Final user review approves this proposed layout.
- R10: Import already-saved finite metrics and provenance without retraining, loading model binaries or inventing absent scores/timestamps. Preserve NA reasons, empty cohorts, conditional CIs and descriptive seed ranges. Shared imported names alone must never imply matched evaluation keys. Source scientific acceptance, lifecycle status and import completion are separate fields.
- R11: Same-source repeat import must be a verified no-op; conflicting content must stop for reconciliation. Full artifact and metric readback verification precedes import-complete status. Provide source manifests, command documentation, a backup procedure and a scratch restore check.

## Status and authorization

- User requested local installation and unified management of completed experiments, with grilling as needed.
- User approved a new task: “可以新建”. Created 2026-10-07; planning only. No MLflow installation, service startup, import or modeling-environment modification yet.
- The previous yearly-XGB work has been accepted and merged to main, but its audit lifecycle remains active. This new task does not close or rebind that run; resolve the lifecycle transition before implementation.
- Initial planning is complete: design.md/implement.md and source-layout research are ready for final review. Proposed execution is Claude Opus5.5 1M under Codex supervision; verify the actual session before lifecycle start. User approved the complete final planning summary with “可以执行” on 2026-10-07. Installation, historical import and acceptance verification are released after the lifecycle transition.

## Acceptance Criteria

- [x] Scope, full retained-artifact copying policy and isolated WSL/SQLite/localhost host adopted (R6–R8).
- [x] User approves final PRD/design/plan, including proposed comparison layout and execution/lifecycle transition (R4,R9).
- [ ] Local MLflow UI and API can list agreed historical experiments and runs.
- [ ] Imported counts, parameters, metrics and provenance reconcile with source evidence; invalid/missing values are represented honestly.
- [ ] Repeating an import does not duplicate runs or overwrite different historical identities silently.
- [ ] Frozen modeling environments and original scientific artifacts remain unchanged; document launch, import and backup/recovery commands.
- [ ] Six parent sources and expected120 child views reconcile with the finite manifest, with no missing-source claims hidden by aggregate counts. Source retained files and each imported finite metric verify against copied originals (R7,R9–R11).
- [ ] Windows and WSL localhost API/UI access, artifact download, manual start/stop and scratch restore succeed; browser-render verification or its specific limitation is documented (R8,R11).

## Out of scope

Adjacent IPCCH/FEWSNET/ETH experiment families, model retraining, automatic instrumentation of training scripts, cloud/LAN access, model serving/registry conversion, custom dashboards and deletion of original files. Backup/retained artifacts do not include duplicate raw inputs or large prepared feature matrices; original inputs remain necessary for retraining.

## Confirmed planning evidence

`research/findings.md` records the initial bounded inventory; `research/report-layout.md` completes the six-family source and report-schema mapping, including both sensitivities. No existing MLflow installation/service was found in the inspected environments; isolated WSL hosting is locally feasible. Wider FEWSNET/adjacent-IPCCH families remain outside the adopted scope. Scientific acceptance is distinct from Trellis lifecycle status.
