# GeoXGBoost implementation plan — v1.0

**Current execution override: D28, section 8 below; user accepted the final A3 six-root implementation/run plan. Stage 1 overfitting remains unresolved.** D26/D27 bounded experiments are complete. Original full-pipeline checklists below are historical scope, not authorization to resume Stage2/3 or close this task.

2026-10-01. D24 accepts the bounded design; D25 explicitly authorizes execution after the final planning summary. Implement, verify and run the finite experiment-plan budget, after committing the frozen planning package and starting through the bound Claude executor's audit wrapper. No numerical or scope change is introduced by D25.

## 1. Entry and source boundary

- [x] Resolve the experiment proposal through D24 and consolidate the final PRD/design/plan for review. Preserve the distinction between approval of the design and permission to implement/train.
- [ ] Before an enrolled implementation, commit the approved planning artifacts, inspect `trellis-audit status`, bind the verified actual Claude executor, and start through the audit wrapper. Verify the run, executor, base SHA, task `in_progress`, and exact registered repository path spelling. Do not substitute another session or use native `task.py start/archive` to bypass the controller.
- [ ] Read the backend specs; task-specific D14 replaces the mother's RF imputer contract only inside the new XGB fork. ETH-only expert, SMOTE, 88-column and output-directory rules do not apply to this package.
- [ ] Pin the mother source to `14c89bc150194452361bb495c601de070cd94ce7`. Planned sibling: `FEWSNETGeoXGBExperiment/`. Copy the 55 tracked mother source/package files, excluding `runs/`, caches and ignored local data. Build the copy list from that commit; do not bulk-copy the live directory.
- [ ] Record mother identity separately from the new committed producer SHA. Historical v7 producer is `fe40de375ea11d6684e84c12202c6ddafa134b85`; do not label the current source a same-run v7 producer or an inherited audit pass.
- [ ] Keep source release ZIP external and read-only: `GeoRFBaseline/releases/georf-baseline-v0.1.0.zip`, SHA256 `39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0`. Preserve the existing frozen raw source manifest; source panel SHA256 `611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651` and feature schema SHA256 `51b6f8b21b76a78510522c34e2d1f2a648b7aec768bbcac3dd2318669fa13349` are planning anchors to verify.
- [ ] Run mother/fork in separate processes with each entrypoint's own PACKAGE first on `sys.path`. Check module locations and inherited tests containing literal mother paths. Preserve the mother source, data and results.

## 2. Ordered changes, after authorization

The bound Claude executor owns implementation and may dispatch native `trellis-implement` and `trellis-check` agents with the current task context; the supervisor coordinates planning and scoped verification. Before modifying a function/class/method, run GitNexus upstream impact and report callers/processes/risk; inspect the exact code to change. Run `detect_changes` before any commit; document existing index failures and use source tracing if unavailable. No new generic model registry, plugin framework, experiment database or parallel scheduler is needed.

1. **Preparation and identity.** Reuse `scripts/prepare_fourclass.py`, the frozen feature builder, snapshot schema, keyed ledgers and completion protocol. Extend the label snapshot range from the actual outer/internal fitting schedule; preserve feature formulas and early NaNs. Parameterize the approved finite run table inside the fork, with explicit candidate family identities and development/final roles. Save resolved runtime versions including XGBoost and code/source/schema identities. Existing prepare requires producer/verifier code equal committed HEAD, so authoritative data runs follow a code commit.
2. **Native booster adapter.** Replace the active RF fitting path in the fork with a small native-XGB implementation usable by both stages. Reuse fixed-four metrics. Dense float input, ±inf→NaN, fixed num_class=4, no RF imputer or pseudo-class rows. Fresh global fitting and immutable-parent continuation are separate operations, with fixed rounds and capacity accounting. Reuse checkpoint metadata; save native model plus feature order, fitting keys, parent identity and base-score/prefix evidence. Do not introduce an abstract backend hierarchy.
3. **Stage 1 integration.** Wire the adapter into the fork's `app/main_model_GF.py`, `src/model/GeoRF.py`, `src/model/train_branch.py`, `src/partition/transformation.py` and `partition_opt.py` as required by actual call flow. Fix the root's current double-fit path before continuation could append twice. Reuse split/scan; add approved ratio/seed/threshold identity, support checks, path-round ceiling, whole-parent paired scoring and exact parent inheritance. Keep area IDs and fallback routing intact. Reuse the root for the same-identity E3 pooled prediction.
4. **Stage 2 inputs.** Reuse `scripts/run_stage2.py` and its existing step consumers. Expand candidate identity so L/ratio/seed/threshold cannot overwrite a fold. Select a complete expected subset by configuration and `candidate target < map origin`; pool all H into one general map. Represent `no_prior_candidates` before the consensus call; retain the existing all-zero-weight null route. Missing expected evidence remains an error. For the old-map diagnostic, read committed v7 `stage1/folds/*/correspondence_table.csv` with scores and recorded historical identities; do not depend on ignored handoff mirrors. The existing wrapper validates the whole prepared schedule and copies the whole handoff tree, so explicit subset acceptance is required; truncating files alone is not valid. Add the small duplicate/weight diagnostics to existing output tables, without changing consensus weighting or graph rules.
5. **Stage 3 fitting and gate.** Adapt the existing comparison runner to fit the selected G, replay six internal dates with their own legal fitting origins, fit eligible increments, aggregate paired confusion and apply strict >.01 routing. Preserve all paired rows when an internal local fit falls back. Refit at the external origin; save actual route/model assignment and reasons. Implement the fixed-map independent diagnostic in the same flow using local-trained G followed by L. Share only exact-identity global fits across maps/arms.
6. **Finite development driver.** Prefer extending existing runners; if necessary add one small `scripts/run_development.py` to enumerate the fixed grid and invoke them. It selects G then 24 map/local combinations, reconstructs temporally available new and old maps, emits the full selection table and freezes the winner using the declared rule. No adaptive search or resumption framework; reuse immutable run IDs and existing completion records. The driver may select on development predictions only.
7. **Reporting and verification.** Extend the existing `report_fourclass.py` and `verify_fourclass.py` to accept new arm/model types while preserving cohorts, four-class metrics and paired country bootstrap. Replay actual native saved boosters and keyed routes, reconstruct q/E2/E3 weights and E5 gate scores where relevant. Separate implementation completeness, persistence success, expert performance and historical RF references. Do not relabel v7 as a matched 59-month RF control.
8. **Documentation.** Write the fork README, run recipe and concise task-local `PROGRESS.md` execution ledger. Align root README/PIPELINE_WORKFLOW/CLAUDE only where a changed user-facing workflow requires it; keep legacy root GeoXGB excluded from the main batch workflow. Add only needed schema tracking exceptions to `.gitignore`; keep generated model trees/runs ignored unless explicitly promoted as deliverable evidence.

## 3. Minimal meaningful verification

Use the inherited test runner; add a small set of regression cases for the changed scientific contracts, not a new testing framework or tests that merely repeat config values.

| Check | Evidence that must fail on a real defect |
|---|---|
| Native continuation | Train a small global, continue with a missing-class local subset, verify four probabilities, exact old tree structure/leaf/missing directions and base score, unchanged parent, expected appended rounds; saved reload reproduces predictions. Check zero increment separately. |
| Evaluation roles | Crafted counts separate q from acceptance; parent and all child combinations use identical parent keys; equality at 0 and .01 rejects. Support-failed child retains parent predictions and complete keys. |
| Time and support | A miniature dated panel shows Stage 1 random rows limited by external O, E3 excluded, Stage 2 excludes scores at/after O, Stage 3 each internal V uses its own59-month window. Include no-prior-map, missing rare class, too-few dates and failed-current-fit fallbacks. |
| Gate and routes | A poor internal date cannot vanish when local fitting fails; recompute the aggregate >.01 decision; replay external keyed routes from saved models. A deliberately changed route or fitting key must be detected. |
| End to end | One small labelled candidate → consensus → external fold → report → verifier, including an absent fourth class; compare saved probabilities/argmax and independently recomputed metrics. Production runners create the evidence, not hand-written acceptance fixtures. |

Run relevant inherited fixed-four/baseline/acceptance tests after the adapter changes. The root legacy `python -m src.tests.sig_test` is a binary path and does not verify this fork's macro-F1 rules; the targeted E2 tests above are required. No training compatibility test has run during planning.

Preferred interpreter: `/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe`; mother numerical environment Python3.12.10, numpy2.2.6, pandas2.2.3, sklearn1.6.1, scipy1.15.2, geopandas1.0.1, shapely2.1.0, polars1.27.1. Pin XGBoost3.0.0 and actual native build identity after verification; do not silently replace the runtime with the Linux counting environment.

Inherited command shapes, run from the **future fork** directory after implementation (not executed now):

```text
<Windows Python 3.12> -B tests/test_baseline.py
<Windows Python 3.12> -B scripts/prepare_fourclass.py --run-dir runs/<fresh-id> --preflight-only
<Windows Python 3.12> -B scripts/prepare_fourclass.py --run-dir runs/<fresh-id>
<Windows Python 3.12> -B scripts/run_stage1.py --run-dir runs/<fresh-id>
<Windows Python 3.12> -B scripts/run_stage2.py --run-dir runs/<fresh-id>
<Windows Python 3.12> -B scripts/report_fourclass.py --run-dir runs/<fresh-id>
<Windows Python 3.12> -B scripts/verify_fourclass.py --run-dir runs/<fresh-id>
```

These existing shapes do not yet express development subsets or the six-date gate. Document the final bounded driver/Stage3 CLI once implemented; do not claim these commands alone run the new design. Existing `run_all.sh` omits verifier/tests, so it is not the acceptance command. No authoritative real run before meaningful compatibility/time/routing checks pass.

## 4. Evidence, execution order and rollback

- Retain keyed development/final predictions, internal gate pairs, support counts, actual routes, selected configs, complete expected candidate ledger, map origins/identities, model fitting keys and replayable final checkpoints. Preserve inherited completion-last and hash-bound consumer acceptance; do not build a parallel artifact-integrity system.
- Record all tried configurations and negative results. Compute counts/time/storage after the first three scheduled candidates within the fixed budget. Keep bulky Stage 1 working checkpoints in Windows-accessible scratch outside Dropbox; preserve necessary evidence before cleanup. All run IDs immutable; no forced overwrites.
- Run the approved development budget first. Freeze selected configs/map/rules before final scoring. Final execution scope follows `experiment-plan.md`; approval of implementation alone is not automatically approval of a full training budget.
- Changes to shared code after authoritative preparation require a fresh correctly identified run or demonstrated unaffected evidence; never patch stored metadata to fabricate same-run identity. If a scientific contract must change, return to planning.
- On compatibility or resource failure, keep the failing evidence, report the incomplete step, and stop dependent runs. Do not silently switch to sklearn XGB, RF, imputation, fewer classes, fewer dates or a smaller grid.
- Rollback is isolation: leave the mother and its artifacts untouched; retain any failed fork/run for diagnosis and stop its entrypoint. Do not delete unrelated work or raw inputs.
- Before delivery, verify only intended source/spec changes, run GitNexus `detect_changes`, commit relevant nonignored code/evidence, and use the bound executor's audit close wrapper. Verify the accepted audit result before claiming audited completion; a queue entry is not a pass. Scientific failure remains a valid negative result and cannot be repaired by changing final evaluation rules.

## 5. Planning status

D24 accepts design and D25 authorizes execution. Final planning review and context validation passed. Actual executor binding/start and technical compatibility checks remain prerequisites before product implementation/authoritative experiments; record them in PROGRESS.md. Planning research did not run models or create the code fork.

## 6. D26 amendment (2026-10-01): binary crisis endpoint, Stage 1 first

- [x] Stop the fourclass batch (old Stage 1 dispatcher and minirun-2 progression) by verified PIDs; preserve all artifacts (19 completed roots, the 0xC0000409 crash scratch, 6 interrupted roots).
- [x] Add a crisis scorer (argmax → IPC≥3 collapse; exact crisis-positive F1; crisis scan masses) next to the fixed-four metrics; keep fixed-four as secondary in every record. Implemented at87513eb.
- [x] Stage 1: E1 crisis masses, E2 exact crisis gain on complete parent keys (families unchanged), E3 crisis F1 vs root, E4 weight input = crisis F1. Tests for each; native D26 check reports43 tests passed, recorded at7d3f0ec.
- [x] Re-select G from the saved 72 G predictions by development crisis F1 (no new fits): H4 G1/H8 G4/H12 G2.
- [x] Bounded representative Stage 1 subset on the full-geometry data:12 roots/48 candidates completed. Keyed independent E2/E3 rescore recorded at8b1487f; this is not full checkpoint replay or648-candidate acceptance. User reviewed and authorized only D27 next.

## 7. D27 implementation and bounded execution

- [x] Commit aligned PRD/design/evaluation-contract/experiment-plan/implement/context amendments before product edits (38e13de, coordinator). Audit run ed632775 and bound Claude session unchanged; no register/start reset.
- [x] Load task context in order: jsonl entries, prd.md, design.md, implement.md. Native trellis-implement implemented most of A2 (stopped at ~10.5 min by the runtime rule; changes captured and reviewed; executor finished the reuse guards/G lock) with the smallest existing split/driver/identity/diagnostic changes. Preserve old random paths and saved D26 evidence.
- [x] Add explicit time-block mode (3dab25b): latest three global observed history months for common E1/E2, earlier history for fitting, no per-area reassignment. Root/candidate/completion identity and the expected list distinguish exactly six L1/gt0 candidates from old grids. Keep all fixed model/support/time contracts.
- [x] Add the meaningful split/identity/paired-diagnostic checks in A2 (native trellis-check of the D27 diff: no result-affecting issue; split rebuild and comparison tests added before commit; 52 tests OK at 3dab25b); fix only relevant result/acceptance issues. Run native trellis-check covering D27, record findings and test evidence. Do not substitute a previous D26 check.
- [x] Commit the producer (3dab25b) before preparing a fresh full-data/full-geometry run outside Dropbox (geoxgb-d27-tb3-20261001; G reused from geoxgb-v1-20261001: G1/G4/G2). Reuse identity-checked G selection; do not repeat72fits or cross-expand the six-candidate budget. Record the actual CLI, source hashes, schedule, runtime and code identity.
- [x] Execute six roots (all completed), preserve all outputs/failures, and compare with six completed D26 r80/L1/gt0 controls (stage1_tb3_compare, exit 0). Independently recompute keyed E2/E3 scores; report support, root changes, incremental generalisation and coverage-aware partition diversity.
- [x] Update PROGRESS with separate implementation, experiment-completion and scientific-evidence statuses (overfitting NOT solved; returned for review). Stage1 overfitting stays unresolved unless supported by evidence; improvement in this small comparison is not final scientific success. Return results for review; do not run full648, Stage2/3, final evaluation or audit-close yet.

## 8. D28 Plan B (A3 implementation and six-root execution approved)

- [x] Present the concrete A3 plan and record subsequent user approval (“接受”) before product edits/training.
- [x] Commit approved planning (317529b, coordinator); audit binding/base/run preserved; no task reset.
- [x] Load current jsonl, PRD, design and implement context; dispatched bounded native trellis-implement (finished in ~4.5 min; GitNexus fallback = grep tracing). Trace adapter/train_branch/transformation/identity consumers before editing, preserve parent mode and old outputs; document existing GitNexus fallback if still unavailable.
- [x] Add root-anchored child fitting (98adf48), keeping current-parent E1/E2 and fallback. Record sharing source separately from routing parent, actual local rounds separately from inherited search-budget rounds. Retain original search-opportunity ceiling and support checks.
- [x] Add focused second-level continuation/fallback/search-budget regressions (incl. production-path non-root fallback and exhausted budget; native check: no result-affecting issue; 58 tests OK), six-candidate identity and keyed comparisons; native trellis-check covers final changed logic. Avoid an unrelated verifier rewrite.
- [x] Commit producer (98adf48), prepare fresh full-data run (geoxgb-d28-rootinc-20261001; G from geoxgb-v1-20261001), reuse approved G selection and execute only six r80/L1/gt0 root-mode candidates. Compare with existing D26 controls; verify matching fitting keys/root predictions before mechanism interpretation.
- [x] Update docs/PROGRESS with bounded implementation, run and science results separately (overfitting NOT solved; returned for review). No claim of solved overfitting from E2/tests/zero partitions; no Stage2/3/full648/final/close without subsequent decision.
