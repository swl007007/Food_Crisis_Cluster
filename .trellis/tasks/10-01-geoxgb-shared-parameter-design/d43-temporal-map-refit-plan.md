# D43 / A17: temporal Brier map learning under common forecasting

2026-10-02. Supervisor decision under the user's delegated Stage1 research authority. Design selected; executor review and committed aligned planning required before implementation or fitting. Supersedes the discussion-only status of `research/d43-temporal-map-options.md`; legacy D26/D27 hard-F1 map reuse remains deferred. Same active task, executor, audit run and base. No Stage2/3, full648, final evaluation or close.

## 1. Question and stopping decision

Does a recent time-block Brier map-learning pipeline yield more useful regions than the D34 random Brier pipeline when BOTH maps use the same current forecasting root, fitting pool and local fitting rule?

D27 confounded map learning with a stale forecasting root, hard-F1 E1 and cumulative parent increments. D30 retained random within-date mixing; D42 changed map vintage. This is one remaining bounded pipeline comparison, not a pure causal test of validation chronology. Search-root age, sample size, coverage and reuse of search labels all change together. All 21 existing D34 cases are included to avoid choosing a favourable calendar subset. These are repeatedly used development cases, not untouched tests.

After this run, stop the time-split/map-generation variant sequence and synthesize Stage1 evidence, regardless of outcome. No tb2/tb4, extra seeds, C splits, thresholds, shrinkage grid or outcome-selected reruns. A tiny, horizon-dependent difference is inconclusive and will not justify adoption or a claim that overfitting is solved. Even a useful development gain is only a candidate for later separately specified work; Stage2/3 remain deferred. Numerical replay and scientific improvement are separate decisions.

## 2. Fixed schedule and search roles

Use D34 `geoxgb-d34-e1-brier-20261002`, producer `7b2bf6f`, and D35 `geoxgb-d35-global-increment-20261002`, producer `be5f485`. Exactly H={4,8,12} crossed with the seven targets below. G remains H4=G1/H8=G4/H12=G2; no G screen.

| T | H4 search months | H8 search months | H12 search months |
|---|---|---|---|
|2018-06|2017-02,2017-06,2017-10|2016-10,2017-02,2017-06|2016-06,2016-10,2017-02|
|2018-10|2017-06,2017-10,2018-02|2017-02,2017-06,2017-10|2016-10,2017-02,2017-06|
|2019-02|2017-10,2018-02,2018-06|2017-06,2017-10,2018-02|2017-02,2017-06,2017-10|
|2019-06|2018-02,2018-06,2018-10|2017-10,2018-02,2018-06|2017-06,2017-10,2018-02|
|2019-10|2018-06,2018-10,2019-02|2018-02,2018-06,2018-10|2017-10,2018-02,2018-06|
|2020-02|2018-10,2019-02,2019-06|2018-06,2018-10,2019-02|2018-02,2018-06,2018-10|
|2020-06|2019-02,2019-06,2019-10|2018-10,2019-02,2019-06|2018-06,2018-10,2019-02|

Supervisor preflight: these are the last three distinct historical months of each saved D34 `fold_membership.csv.gz`, excluding `heldout_target`. Recompute from actual keys before fitting; a mismatch stops the run, never substitutes dates.

Reconstruct the original D34 legal pool as FIT union S union C, with unique (area,month) keys in original numeric area/month order; all labels in [O-59,O), O=T-H. Preserve the producer's target-area restriction. Read snapshots with target_month <= 2020-12. Reuse `time_block_split`: all rows in the listed three months are search S_tb for E1/E2, all earlier legal rows are FIT_tb; no per-area reassignment, no random confirmation split. Do not merge E3 into this pool. Existing D34 C labels may enter this new map's search or fitting; C is no longer an isolated diagnostic for this arm and is not scored as confirmation.

Fit one search root per pair on FIT_tb using the fixed G. Learn one candidate with `run_candidate(..., increment_source="root", e1="brier_crisis", confirmation=None)`, L1=20 rounds, gt0, existing geometry, support, depth and path-search budget. E1 uses current-parent Brier probabilities; E2 compares current-parent routes with four-class argmax collapsed crisis F1. Each child starts from the immutable search root, not the forecasting root. Record the search root's actual last label month and support. This time block is not strict rolling-origin validation at each S_tb row's own forecast origin.

## 3. Common forecast and budget

Four primary arms: `root`, `global20`, `random_map_refit`, `temporal_map_refit`.

- Forecast root and FIT are the existing D34 current root and its exact r80 FIT keys. Load them; no new forecasting-root/global20 fits. Root/global20 predictions must reproduce D34/D35 on the same E3 keys before dependent work.
- Random map is D34's Brier `assignment_evidence.csv`; temporal map is the newly frozen candidate's D32 assignment evidence. Use `spatial_partition_id`, not predictive route or booster identity.
- Both map arms use D42 `common_refit`, `route`, `arm_proba` and `save_arm` semantics: every named region (including a searched root-copy/unsplit root region) fits one L1 increment from the CURRENT D34 forecasting root on its CURRENT FIT member rows, in original order. FIT_SUPPORT stays rows500/areas50/dates6/classes2. No E2/C/E3 enable gate at refit. s-1/missing/support-ineligible -> current root. An area without own fitting history may use its eligible regional model.
- Keep existing 162-feature order, NaN handling, numerical environment and unweighted/no-margin fitting. No D38 anchors, D40 threshold policy or D41 shrinkage.
- Budget: exactly 21 temporal search-root fits and 21 temporal searches, plus the three predeclared production-equivalence searches below using saved roots (no new roots for those checks). Existing five split levels (`MAX_DEPTH=6`, loop `range(max_depth-1)`) allow at most 31 split nodes/62 child-fit attempts per search and 32 terminal regions per map. Thus at most 1,488 search child fits across 24 searches. D34's 21 fixed random maps have 166 named regions (supervisor key-only count); new temporal maps have at most 672. At most 838 common regional refits, one per named region per arm per pair; eligibility may reduce this count. Freeze and record each map's counts before refits. These are bounds derived from existing depth and fixed maps, not new capacity settings. No retries with altered seeds/supports.
- Use a sequential runner initially; no new scheduler or worker framework. Estimate 15-35 minutes wall time from D27/D34/D42, not a guarantee. Failed identity/replay/fitting stops subsequent pairs and preserves partial evidence; normal support fallback keeps all evaluation keys.

## 4. Evidence and interpretation

Preserve source hashes, producer/replay identity, current/search root UBJs and metadata, ordered membership/roles/fit-key digests, temporal candidate decisions/maps/checkpoints, refit region members/support/UBJs/prefix evidence, complete E3 rows and completion-last records. Reuse existing identity helpers; no generic acceptance framework. Old runs remain read-only; all new bulky outputs in a fresh directory under C:\Users\swl00\geoxgb_runs.

Score E3 only for the primary comparison: four class probabilities, fixed argmax crisis F1, secondary four-class macro-F1, float64 crisis Brier, confusion counts. Report all-key and identical exact-origin-persistence-available cohorts; persistence is hist_phase_o00, no latest-label replacement. Per H, per target and overall: pooled row metrics AND separate mean-fold deltas, each map minus root/global20 and temporal minus random. Include changed decisions (corrected/spoiled, TP/FP changes), region counts, support and fallback shares alongside deltas. No significance claim from 21 correlated folds and no comparison of this pooled result with D42's 12-pair subset.

Search S_tb scores and its search-root E3 predictions produced by the existing helper are descriptive only, separately labelled; never mix that stale search root with the common forecast arms. Maps and common fits must be frozen before E3 scores influence any decision. E3 labels passed to the existing prediction/reporting helper may be scored after search, but must not enter fit/search/support/route selection.

Report exact S_tb overlap with D34 FIT/S/C by keys. Roughly 80% may be reused in refitting, unlike D34 S: legal at O, but asymmetric and not proof of reduced selection bias. Also report search rows/dates/areas, root age and map coverage. Do not remove recent rows from the common forecast pool to force disjointness. Region count and support differences are consequences of each pipeline, not controlled causal factors.

## 5. Minimal implementation, cases and checks

One isolated `FEWSNETGeoXGBExperiment/scripts/stage1_temporal_map_refit.py` plus focused checks in the existing test file. CLI: `--d34-run PATH --d35-run PATH --out FRESH_PATH`. Hard-code this experiment's schedule and identities in the runner; do not widen existing prepare/main/plan modes. Reuse `gate_root`, filtered `rebuild`, `time_block_split`, `fit_global`, `run_candidate` and D42 helpers. Restore original row order after recombining FIT/S/C, verifying keys and features against the existing reconstruction. Use D34's geometry, not a test fixture's geometry.

Good: both eligible maps fit from identical current root/FIT and preserve its prefix; same E3 keys retained. Base: unsplit map is one real region; no-search/support failure uses current root. Bad: temporal map accidentally refits from FIT_tb/search root, E3 mutation changes learned output, label-month table differs, or a failed pair continues later fitting -> reject/stop.

Small runnable checks must exercise: exact pool/role reconstruction and date table; production search + common refit invariant to E3 label mutation; identical common root/FIT despite different search roles; s-1/unsplit/support fallback via reused D42 checks; failed first pair prevents subsequent pair execution. Saved/reloaded predictions and root/tree prefixes are checked; the changed runner must have native trellis-check. No unrelated framework or mother-package edits.

Before any temporal search, run three fixed assembly-equivalence checks: H4/H8/H12 at T=2018-06. Through the new runner's reconstruction, restore D34 FIT/S/C roles, exclude C from search, load the saved D34 root and original root-fit record, then call the existing Brier/root/L1/gt0 `run_candidate` with the original C tuple for identical diagnostic exports. Compare D32 assignments and routed native UBJ bytes with the saved D34 candidate; compare probability/routing values and scores, not path/timing/gzip header metadata. Any mismatch stops all temporal work. Retain the results as a three-case plumbing check, not a proof for every dataset. No new random-map candidate is substituted for the saved control.

Wrong: compare tb3's stale-root predictions against D34 and call the difference partition quality. Correct: primary comparisons use the same CURRENT forecast root and refit pool; search-root differences are disclosed as part of map learning.

## 6. Execution order

Executor read-only review -> supervisor resolve issues -> align PRD/design/implement/evaluation/experiment-plan pointers and jsonl (respect 32768-byte injection limit), commit planning -> native implement -> native check/tests -> commit producer -> run exactly the above -> supervisor independently verify roles, same-root refits, model replay and keyed scores -> scientific synthesis. GitNexus impact/detect_changes required attempts; existing LadybugDB failure recorded with source tracing, no index repair. Preserve original audit/session; no close or Stage2/3 advancement.

## 7. Run and factual results (2026-10-02; independent verification and synthesis by the supervisor)

**Producer and run**
- Native trellis-implement was stopped at its 10-minute bound; its partial work was adopted. The supervisor's contract fixes were already present: no hash-difference assertion, `legal_pool_membership.csv.gz` per pair, and the refit budget checked before any refit. The executor completed the matched-cohort Brier report via `_pool_block`. Native trellis-check (~2.6 min) found no result-affecting issue. GitNexus impact/detect_changes: LadybugDB read-only error, risk UNKNOWN.
- Producer commit `1be4e3bb2c1e28216b5c395e0215e4576fdcac5b`; after the commit `tests/test_baseline.py` ran 110 tests OK, exit 0.
- Run `C:\Users\swl00\geoxgb_runs\geoxgb-d43-temporal-map-refit-20261002` on the frozen Windows Python, under a `timeout 7000` wrapper; the user moved it to the background, with one live process verified. Exit 0, 926 s. Log `C:\Users\swl00\geoxgb_runs\d43-run.log` (sha256 `5cd0ac27…`).
- **Artifact hashes:** `summary.json` `e6f0eca3…`, `gate.json` `aec0a8a8…`, `identity.json` `6105536…`, `completion.json` `147d2ca1…`.

**Equivalence and budget**
- All three equivalence searches passed (H4/H8/H12 2018-06): assignment evidence, all checkpoint shas and the frozen digest equal the saved D34 candidates. This is a three-case plumbing check, not a proof.
- All 21 pairs passed. Budget used: 21 search roots, 24 searches, 490 child fits (limit 1,488), 306 common refits (limit 838).
- dev_baselines: 15 checked, 6 not covered.

**Maps**
- Random maps: 166 named regions, all eligible.
- Temporal maps: 140 named regions, all eligible. Per pair: 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9, 10, 11, 12, 12. The random maps per pair: 3–12.
- S_tb is 15,734–16,095 rows per pair (336,826 in total). By D34 role: fitting 76.16%, validation 11.78%, confirmation 12.06%.
- Search-root age at O is 16 months for every pair (the last FIT_tb label is 12 months older than the last D34 FIT label; for example H4 2018-06 last FIT_tb label 2016-10 vs D34 2017-10).
- Share of E3 rows routed to root: random .017928 (s-1 2,035 rows), temporal .007577 (s-1 860 rows).

**E3, all keys (113,508 rows):** pooled crisis F1 / crisis Brier / four-class macro-F1

| Arm | Crisis F1 | Crisis Brier | Macro-F1 |
|---|---|---|---|
| root | .550455 | .101890546 | .531864 |
| global20 | .550674 | .101950387 | .531784 |
| random_map_refit | .550919 | .101870102 | .528752 |
| temporal_map_refit | .550304 | .103005187 | .528388 |

Root and global20 reproduce D36.

**Pooled deltas**
- Temporal − random: F1 −.000615, Brier +.001135. Decisions: changed 2,356, corrected 953, spoiled 1,403, TP +251, FP +701.
- Random − root: F1 +.000464, Brier −.0000204.
- Temporal − root: F1 −.000151, Brier +.001115.

**Fold-mean deltas**
- Temporal − random: F1 +.000610, Brier +.001135, macro-F1 +.000857.
- Random − root: F1 +.000513.
- Temporal − root: F1 +.001123, Brier +.001111.
- The sign of the F1 delta differs between the pooled and fold-mean results.

**Persistence-matched (112,795 rows):** F1 / Brier

| Arm | Crisis F1 | Crisis Brier |
|---|---|---|
| root | .551516 | .101884595 |
| global20 | .551734 | .101944273 |
| random | .551979 | .101864403 |
| temporal | .551334 | .103006280 |
| persistence | .586406 | .146407199 (one-hot) |

**By horizon, pooled F1 (root / global20 / random / temporal):**
- H4: .628357 / .628212 / .627861 / .624110
- H8: .532228 / .532364 / .533740 / .533937
- H12: .477156 / .477876 / .476931 / .485699

Fold-mean temporal − random F1: H4 −.003635, H8 −.000663, H12 +.006129. The temporal arm's Brier is higher than the random arm's at every horizon.

**Not done:** the executor did not run the supervisor's `d43_independent_check.py`. There is no adoption, Stage 2/3 or close. The scientific reading is left to the supervisor.
