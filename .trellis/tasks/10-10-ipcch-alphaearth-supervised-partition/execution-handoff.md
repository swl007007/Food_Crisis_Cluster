# Execution authorization and supervisor handoff — 2026-10-10

## Latest user instruction

The user activated a persistent goal:

> 用herdr技能绑定另一pane的opus 5.5 1m作为执行者，你作为监督者抽查diff，checkpoint和交付物。直到完成spec前运行goal任务。除非出现CRITICAL ERROR否则保持运行。

Operational scope: execute the work defined by the completed PRD/design/implement, with Claude Opus 5.5 (1M context) as executor and Codex as supervisor. This supersedes the prior planning-only hold. Work continues through ordinary fixes and checkpoints without repeated user approval; the supervisor checks the concrete evidence and releases dependent stages. The scientific choices D1–D29 and the bounded experiment remain unchanged. The intent is completion of the spec's deliverables, not merely creating code or obtaining favorable metrics.

The present release is P0 offline implementation/readiness. Subsequent phases are within the goal, but their documented concrete checkpoints remain: inspect identity/resources/outputs before releasing dependent work. No automatic EE/export/fitting launch from a preflight. Do not create billing/project/bucket resources, expand quotas, change IAM, spend an unbounded amount, overwrite prior results, or change science to bypass an obstacle. Use the existing configured infrastructure; report actual resource facts and a bounded proposal. Unavailable credentials/access, a required new financial commitment, destructive action, or inability to preserve the scientific contract must be escalated; ordinary code/test/storage faults should be repaired within scope.

## Ownership

- Executor: the live Opus 5.5 /1M session in Herdr `w14:p2`, to be verified and registered before lifecycle start. Owns implementation, tests, execution, evidence, operational PROGRESS and commits. Do not spawn alternative executors or replace the bound session.
- Supervisor: Codex in `w14:p1`. Owns sampled diff review, checkpoint release and deliverable acceptance, grounded in current artifacts rather than executor reports alone. Findings go to the executor to repair. Scoped review is not a full re-audit of all previously accepted packages.
- Shared repository: you are not alone in the codebase. Preserve others' edits and all old experiment/runtime/store artifacts. Only the task folder, isolated new experiment package, approved result notes and the completed-family MLflow integration are in scope.

## Read first

Read this handoff, then prd.md, design.md and implement.md in full. Read decisions.md for accepted science and PROGRESS.md for prior facts. Curated implement.jsonl/check.jsonl point to the relevant source/runtime/calendar/EE/evaluation evidence and backend contracts. Early statements in research notes that choices were open are history; the final design and D1–D29 control.

Key invariants: four B0/B1/B2/G roles; population-derived truth; compact296 with explicit H1 adaptation; fixed2017 mean64/SD64,80% coverage; own-origin dynamics; inclusive month-end U<=fit O; honest unique B1 OOT providers;2021/2022 forward selection; maps frozen through2022;61quartets/244scalar ceiling; fixed9recipes/H; complete all-H results with no all-four-win gate; four-H corrected country inference and explicit retrospective limitations.

## Start procedure

1. Report real Claude session/model/context/pane/terminal and readiness. No lifecycle or implementation before supervisor registration.
2. After registration, create/use branch `ipcch-alphaearth-supervised-partition` from current main without disturbing unrelated files. Finalize only administrative acceptance/start annotations needed by this authorization, validate the context manifests, inspect the expected docs-only diff and attempt GitNexus detect_changes. If its known LadybugDB failure persists, record it and use Git diff; do not reindex or repair the graph as task scope.
3. Commit all task specification/handoff files before start. Do not add ignored raw data/large artifacts. Keep task status planning until the wrapper starts it. This commit becomes the real base SHA, not a guessed historical value.
4. Use the registered exact lowercase repository spelling for wrapper calls:

   `trellis-audit --repo '/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster' start ipcch-alphaearth-supervised-partition`

5. Report actual run ID/base SHA/executor tuple and native in_progress. Once these match, implement P0 as specified. Never reset lifecycle state, bind an observer or silently change executor.

## Checkpoints and communication

For each checkpoint commit the bounded code/evidence delta and report: SHA/base, changed scope, commands and test results, source/runtime identities, counts/budget reconciliation, findings and technical limitations. Preserve logs and compact evidence within this task; keep large run data outside Dropbox. Do not claim a passing test establishes real EE availability or actual historical release provenance.

Send milestone/checkpoint reports to the supervisor using Herdr's agent prompt surface targeting `w14:p1` without waiting on its turn. Do not send repeated idle/status prompts. A checkpoint hold waits for a supervisor release, not a new user preference vote; continue independent authorized work while waiting if available. On an observation timeout inspect the live process/session before restarting anything.

P0 checkpoint must precede real fits or EE calls. P1 proposals identify actual project/destination, source assets/native grids/ownership, bounded pilot and full-export limits. Preserve the map-freeze checkpoint before main G scoring. Production report/replay and final inventory must precede result acceptance; live MLflow integration follows the accepted result and frozen import plan with backup. Do not push, merge or close lifecycle until separately directed by the supervisor within the user's scope.

Scientific null/fallback results are not critical errors and must be delivered honestly. Never expand the search to force significance. Ordinary correctable findings are repaired and checked on their actual diff, not used to restart unlimited review cycles.
