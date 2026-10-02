# GeoXGBoost implementation plan — v1.0

**Current status: D30 recent-search six-root experiment completed (§10, d30-recent-search-plan.md; producer 5517fb4, reporter f777ab0); D31 per-area matched-search control completed and inconclusive (§11). D32/A7 Stage1 assignment-evidence export completed (c079b75); D33/A8 completed (69f2cc3); D34/A9 completed (7b2bf6f; check passed); D35/A10 global +20 capacity control completed (be5f485; §15; supervisor check passed); D37/A11 not adopted (§17); D38/A12 persistence-margin root completed, exploratory candidate not adopted (§18); D39/A13 probability diagnostic completed, zero fits (§19, d38-persistence-margin-root-plan.md); no new training after D35 (§14, d34-e1-brier-contrast-plan.md) (§13, d33-shallow-replay-plan.md) (§12, d32-stage1-assignment-plan.md); supersedes the D32 Stage2 four-map proposal. Stage1 overfitting unresolved; no Stage2/3/full648/final/close in this run.**

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

## 9. D29 confirmation diagnostic (A4 implementation/run authorized)

- [x] Record adopted scope: freeze the full candidate before diagnostic-only C; no C-driven gate, pruning, fallback or candidate deletion. Read-only call-flow/support research completed.
- [x] User adopted diagnostic-only confirmation and explicitly delegated non-Stage3 data handling ("没问题stage3以外的数据怎么搞都行"); proceed with A4 six roots, no repeated data-split approval questions.
- [x] Commit approved planning (e292680, coordinator). Preserve original audit run/executor/base, no new register/start or task reset.
- [x] Bound Claude dispatched native trellis-implement (finished ~6 min, no take-over). Reuse split, root-mode driver, prediction and comparison paths; add explicit S/C roles and separate candidate/run identity. Preserve original fitting keys and all legacy modes.
- [x] Isolate C supervision (C removed before GeoRF.fit; frozen digest before/after C scoring) from all search/model decisions; freeze existing map/checkpoints/routes before C scoring. No new generic gate framework. Keep parent comparison/fallback, L1 capacity and search counter unchanged.
- [x] Meaningful tests (ConfirmationDiagnostic; 62 OK; native checks: no result-affecting issue): deterministic label-blind S/C partition with balanced odd groups/singletons; original fitting keys and S∪C identity; production-path C label mutation cannot affect map/model/routes; complete C fallback keys; comparison distinguishes old exposed-C from new isolated-C. Native trellis-check covers current diff.
- [x] Commit producer (ab1ac83; 62 tests OK, exit 0) and pass required checks before authoritative preparation. Prepare fresh six-root run; reuse original v1 G screen, not reuse-of-reuse; no extra G fits. Run exactly A4 six candidates with frozen environment and full geography (geoxgb-d29-confirm-20261001, G from v1 G1/G4/G2, six roots completed, exit 0).
- [x] Verify D28-matched fitting/root/target identity (reporter ef49d26 exit 0; supervisor independent stdlib rescore passed), independently rescore S/C/E3, report support/confusions/structure/E4 without C selection. Record actual commands, producer, failures and evidence.
- [x] Update PRD/spec/PROGRESS with implementation/run/science statuses; stop for review (overfitting NOT solved). No full648, Stage2/3, final evaluation, audit close or claim that overfitting is solved.

## 10. D30 recent-search contrast (authorized)

- [x] User authorized recommended development research; record exact six-root contract in d30-recent-search-plan.md, preserve Stage3 boundary.
- [x] Commit planning (01b993b); preserve original Claude/audit binding.
- [x] Native trellis-implement (producer ~3 min; reporter taken over by executor): reuse D29 mode and confirmation split; unchanged fitting/C, recent-six S only, explicit unused history and independent identity. Preserve old modes.
- [x] Production-path unused/C-label invariance, deterministic recent dates and support fallback tests; native trellis-check on final diff (no result-affecting finding; f777ab0 scoped check).
- [x] Commit producer (5517fb4; 67 tests OK)/pass checks, fresh prepare/G reuse (geoxgb-d30-recent-search-20261001, G1/G4/G2), six roots only (all completed); no72 G refits.
- [x] D29-keyed root/roles/target comparison (reporter f777ab0 exit 0; real-data replay byte-identical; supervisor independent rescore passed), S_recent/C whole/C recent/C older/E3 scores, support/structure/E4 and independent rescore.
- [x] Update all task docs and PROGRESS with results (overfitting NOT solved); no Stage2/3/full648/final/close or claim of solved overfitting without evidence.

## 11. D31 per-area matched-search control (completed)

- [x] Supervisor decision: keep the original per-area-count matched control (not area-blind); record exact A6 contract and key-only feasibility in d31-matched-search-plan.md.
- [x] Supervisor review with corrections (D30 search-score coverage, supervisor-led review boundary, supporting probe). Commit planning; preserve original Claude/audit binding (no register/start/reset).
- [x] Native trellis-implement (producer ~3 min; reporter by executor after the child returned): matched-size sampler (exact RNG rule, k_a from A5 dates incl. zero), `--matched-size-seed` on the D30 path, 18 roots/candidates with search seed separate from split seed; reporter `--mode matchedsize` in the existing comparison (pinned D30 5517fb4 / D29 ab1ac83). Preserve old modes.
- [x] Tests (72 OK): deterministic order-independent exact per-area sampler, seeds differ; production-path unused/C-label invariance; reporter acceptance/rejection; one fixture test driving the reporter row function end to end. Native trellis-check on the final diff (no result-affecting finding).
- [x] Commit producer (75e04ce) and pass tests; fresh prepare (geoxgb-d31-matched-search-20261001), original v1 G reuse (no refits), exactly 18 candidates (all completed).
- [x] Keyed comparison vs D30/D29 (exit 0) with real-data reporter replay (3 modes byte-identical) and supervisor independent rescore (passed); per seed/pair, all-root and split-only means, fallback contributions, date support, ARI descriptive only.
- [x] Update docs/PROGRESS; stop (inconclusive; overfitting NOT solved). Then synthesize D26–D31 for a bounded map-utility proposal for supervisor scientific review (no automatic Stage2/3; no further search-window/threshold/C-split variants on these six targets).

## 12. D32 Stage1 assignment-evidence export (completed)

- [x] User steer: Stage1 root engineering first; Stage2 formula/scope deferred. D32 Stage2 four-map proposal superseded (not implemented). Read-only trace recorded in d32-stage1-assignment-plan.md §1.
- [x] Supervisor review (wording: branch fitting-pool changes are an induced consequence of altered search/partition membership, not a separate fitting change). Commit planning; preserve audit run/base/session (no register/start/reset).
- [x] Native trellis-implement (~2.6 min): `assignment_evidence.csv` in `run_candidate` after freeze (universe gtrain ∪ gtest ∪ C areas; columns per d32-stage1-assignment-plan.md §2); `candidate.json` `routing_export` + `assignment_evidence` (schema d32-v1, sha, counts); completion chain records the file, required only when declared. No change to search/fit/routes/predictions/scores; no Stage2 change.
- [x] Tests via existing production fixtures (77 OK; supervisor precedence fix search→fitting→target→C): no-search with fit/target, searched root-unsplit, searched named branch (root-booster copy if column written), target-only, C isolation; exact joins/counts/spatial mask; routing/probabilities/scores/correspondence unchanged; pinned D28–D31 still accepted. Native trellis-check on final diff (no contract-affecting finding; re-check of the precedence delta clean).
- [x] Code commit c079b75; actual old-vs-new fixture replay (outputs identical except the new file/contract keys); keyed derivation on saved D31 validates tallies (labelled derivation, not producer execution); docs/PROGRESS. No model batch/Stage2/3/final/close. Then return to Stage1 overfitting research under a supervisor-led spec.

## 13. D33 shallow truncation replay of frozen D29 (completed)

- [x] Supervisor decision and approval: one predict-only depth-1 truncation diagnostic; fixed comparators root / depth1 / full; initial zero-fit error-change evidence reproduced (`research/d29-error-changes.md`, `research/d29_error_changes.py`; 0 mismatches vs the supervisor's tally; independently agreed).
- [x] Commit planning (85b352c); preserve audit run/base/session.
- [x] Native trellis-implement (~3.5 min) + executor fixes (labels, retained parent, identity, guards): minimal isolated replay script (data construction via ab1ac83 producer semantics, frozen runtime), truncated-s_branch routing, verify "0"/"1" last saves belong to the root decision, mandatory full-tree gate first (round-trip exact equality; C/E3 probabilities where saved; all saved S/C/E3 hard predictions and routes), per-key S/C/E3 rows + identities (inputs, checkpoints, producer, script commit, runtime), per-H/T reports per d33-shallow-replay-plan.md §4; minimal routing tests. No fits, grid, selection, E4 or Stage2-consumable maps.
- [x] Native trellis-check + re-check (no result-affecting finding); source-equivalence note; code commit 69f2cc3; six predict-only cases (gate 6×23 checks, 0 mismatches); executor independent recomputation 144/144; docs; stopped for supervisor synthesis.

## 14. D34 E1 hard-F1 vs Brier paired contrast (completed; supervisor check passed)

- [x] Supervisor decision A9; exact spec drafted in d34-e1-brier-contrast-plan.md (21 roots fitted once each, shared by hard_f1 and brier_crisis = 42 candidates; v1 G-screen reuse, no new G-screen fits).
- [x] Supervisor review (legacy replay invariance wording; operational final isolation). Commit planning; preserve audit run/base/session.
- [x] Native trellis-implement (producer ~4.3 min; reporter by executor with supervisor-found blocker fixes): 21-root paired mode in run_stage1/prepare, E1 variant through GeoRF.fit → partition, current-parent probabilities for Brier masses, scan diagnostics (zero c/g, distinct g, boundary tie block, side sizes), paired reporter mode; tests: legacy hard-path old/new replay, C-label invariance for both variants, pure Brier mass formula, tie-statistics on synthetic g, identity/completion.
- [x] Native trellis-check (no result-affecting finding); code commit 7b2bf6f; legacy hard-path old/new replay (retained evidence; supervisor spot check passed); fresh run (21/21 roots, ~11 min wall); e1pair report (exit 0); docs. Supervisor independent check and synthesis pending.

## 15. D35 global +20 capacity control (completed)

- [x] Supervisor decision and standalone spec d35-global-increment-control-plan.md; executor review: no result-affecting contradiction (clarified Parquet ≤2020-12 filter for the rebuild; empty S no-search stratum).
- [x] Commit planning (adff5a5); same run/executor; no new roots/G/E1/E2, no Stage2/3/close.
- [x] Native trellis-implement (~4.3 min): one diagnostic script (accept D34 at pinned 7b2bf6f, rebuild fitting/S/C/E3 with ≤2020-12 filter, root replay gate, 21 `continue_booster(root, X_fit, y_fit, L1)` fits, saved UBJ + continuation records, same-key S/C/E3 probabilities/hard predictions, per-H/T table root/global+20/hard/Brier, search_rows strata); small tests (fitting excludes S/C/target; holdout-label invariance; prefix preserved +20 rounds; strata scoring).
- [x] Native trellis-check (no result-affecting finding); producer commit be5f485 (88 tests OK); run (21/21 gates passed, 80 s); factual docs. Independent check and synthesis left to the supervisor.

## 16. D36 transfer diagnostic (analysis only, completed)

- [x] Supervisor analysis (`research/d36_transfer_diagnostic.py`, outputs in `C:\Users\swl00\geoxgb_runs\d36-transfer-diagnostic-20261002\`), executor independent read-only verification (`research/d36_executor_*`), findings `research/d36-transfer-findings.md`. No fits, no production changes, no 2021+ data.

## 17. D37 recency-weighted root control (completed; not adopted)

- [x] Supervisor standalone spec d37-recency-root-plan.md; executor review (one guard gap raised for supervisor decision: rebuilt fitting-feature equivalence).
- [x] Commit planning (cd30c7c, f412956); same run/executor; no new run/close.
- [x] Native trellis-implement (~5.8 min) + supervisor finishing notes: optional `sample_weight=None` on `native_xgb.fit_global` (default path unchanged, weights validated and recorded), one small diagnostic runner (pinned D34 acceptance, rebuild ≤2020-12 with fitting, original-root replay gate, 21 weighted fresh roots, frozen UBJ reload, same-key E3/C rows, persistence-matched and transition-group reports, identities); tests incl. actual old/new default-path fixture replay.
- [x] Native trellis-check (no result-affecting finding); actual old/new default-path replay identical; producer commit 3b53989 (92 tests OK); run 21/21 gates, 174 s; factual docs; stopped for supervisor independent verification.
- [x] Supervisor independent check PASS (evidence in research/d37_*); decision: do not adopt the 24-month recency root.

## 18. D38 fixed weak persistence-margin root (completed; exploratory candidate, not adopted as default)

- [x] Zero-fit prior/support diagnostic (supervisor `research/d38_prior_support.*`, executor `research/d38_executor_spotcheck.*`, findings `research/d38-prior-support-findings.md`); supervisor standalone spec d38-persistence-margin-root-plan.md with executor review tightenings.
- [x] Commit planning (1e4db02); same run/executor; no new run/close.
- [x] Native trellis-implement (~5.9 min): optional `base_margin=None` on `native_xgb.fit_global`/`proba` (default path unchanged); fit_global writes a persistent booster-attribute marker before hashing; two-way proba guard; marked-parent continuation refused; direct float32 .5 for missing origins via the existing phase mapping; one small runner (D37 rebuild/replay/identity reuse, 21 fresh anchored roots, prior-only and post-hoc control (missing-origin rows p_post = p_original exactly; known rows p·q normalised; probabilities kept as float64 representations of native float32 output), same-key E3/C reports).
- [x] Tests incl. actual old/new default-path replay (byte-identical) and synthetic neutral-margin training equivalence; native trellis-check (~4.3 min, no result-affecting finding); producer commit 2d4fe4e (100 tests OK, exit 0); run 21/21 gates, exactly 21 fits, 244 s; factual docs; stopped for supervisor independent verification.
- [x] Supervisor independent check PASS (evidence in research/d38_*); executor exact match of the pre-computed fixed controls (63/63 per-root equal). Training stopped at 21 fits.
- [x] Supervisor synthesis: retain as exploratory root candidate; not default; not propagated to partitions.

## 19. D39 saved-probability diagnostic (completed; analysis only)

- [x] Supervisor spec d39-probability-diagnostic-plan.md; executor review (prior-only 0.5 ties, constant strata identities, point-derived persistence AUC/AP wording).
- [x] Supervisor diagnostic `research/d39_probability_diagnostic.py/.json` on the 21 verified D38 E3 row files; executor independent check `research/d39_executor_check.py/.json` (2,414 comparisons, 0 issues; overall, per-H and 6 individual roots; remaining 15 roots via aggregates only). Findings `research/d39-probability-findings.md`. No fits, thresholds, calibrators, product code or tests.
