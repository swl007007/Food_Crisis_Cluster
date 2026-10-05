# D44: restoring the full historical pool does not close the root deficit

Supervisor synthesis, 2026-10-02. Plan `d44-full-pool-root-diagnostic-plan.md`, planning commit `2f6223e`. External diagnostic `C:\Users\swl00\geoxgb_runs\geoxgb-d44-full-pool-root-20261002`: 15 shared H/T pairs, 30 saved-model replays, zero fits, 11 seconds. The comparison is within the same 59-month label window, not longer history. Stage2/3 and final-period evaluation remain deferred.

## Verification and scope

The executor verified both source identities, frozen schema and snapshots, G parameters/rounds, actual fitting-key digests, all E3 keys/truth and exact float32 replay of all 30 saved models. Native scoped review found no material issue; the supervisor caught a duplicate requested feature column before execution, and the executor removed it and added shape/uniqueness checks. Runtime: Python3.12.10, numpy2.2.6, pandas2.2.3, XGBoost3.0.0.

The supervisor independently reconstructed all 15 original-file comparisons, fitting digests and pool differences, rescored per-pair/per-H/pooled metrics on both cohorts, and raw-replayed six models (both arms at T2019-06 for each H). A separate comparison reconciled all joined probabilities, all reported metrics and mean-fold differences with the executor outputs: no discrepancies. The supervisor did not independently replay the other 24 models; their replay was checked through the executor script and artifacts. The initial supervisor script used the wrong checkpoint directory and stopped before writing results; its path was corrected to the recorded checkpoints directory, then rerun successfully.

`r80` is nominal: actual FIT/full-window fractions are 0.7485–0.7983. Per-area validation rounding changes that fraction; dividing validation into S/C does not. The full window contains 15–45 additional rows from 3–4 areas per pair excluded by Stage1's target-area restriction, with none of these areas in E3. Thus this is not a pure sample-size experiment. Different fitting rows also change stochastic subsampling paths even with seed42. G was selected using these development cases; no independent-testing or significance claim is made.

## Same-key results

Exactly H4/H8/H12 × T={2019-02,2019-06,2019-10,2020-02,2020-06}. Crisis F1 means four-class argmax collapsed to IPC>=3. Brier uses the sum of crisis probabilities in float64; lower is better.

Persistence-available cohort: **80,613 rows**.

| Arm | Crisis F1 | Crisis Brier | Fixed-four macro-F1 |
|---|---:|---:|---:|
| r80-FIT root | .566612 | .103692772 | .538337 |
| Full-pool root | .562348 | .104181562 | .523944 |
| Persistence | .592605 | .153622865 | .604247 |

Persistence Brier is a one-hot reference, not calibrated probabilistic output. Full-pool minus r80-FIT: pooled F1 **−.004263493**, mean-fold F1 **−.004051562**, pooled Brier **+.000488790**. Across 1,257 changed binary decisions, 547 errors were corrected and 710 correct decisions spoiled; TP −25, FP +138. These correlated rows are not independent trials.

| H | r80-FIT F1 | Full-pool F1 | Persistence F1 | Full−r80 Brier |
|---|---:|---:|---:|---:|
|4|.640694|.638547|.646614|+.000048549|
|8|.560169|.547594|.555002|+.001249828|
|12|.479039|.482622|.574223|+.000170313|

The H8 r80 root exceeds persistence on this particular 15-pair subset; do not repeat the different 21-pair conclusion as if it applied here. The full-pool root is below persistence at every H. H12 gains F1 slightly while Brier worsens; H4 and especially H8 lose F1. On persistence-matched per-pair scores, full-pool improves 7/15; on all-key scores it improves 8/15. Neither count proves reliability.

All-key cohort: **81,321 rows**. r80-FIT/full-pool F1=.565075800/.560831967; Brier=.103677907/.104161192. Pooled F1 difference=−.004243834 and mean-fold difference=−.004012638. No persistence score is reported on unavailable-origin rows. These 15-case aggregates must not be compared directly with D42's 12 or D43's 21 cases, nor with historical Stage3 ~.75/.8 scores.

## Decision

**Do not adopt a full-pool root replacement on the strength of D44.** Restoring the held-out historical rows is not an observed remedy for the root deficit on these saved models. This does not establish that smaller fitting pools are intrinsically better, identify an optimal split ratio, prove overfitting causally, or say whether spatial partitioning can ever help. No new ratio/seed grid follows.

Stage1's scientific problem remains unresolved. D32's routing/evidence engineering repair and D34's hard-F1 search-degeneracy repair remain valid within their verified scopes; their correctness does not establish forecast improvement. Root-first research remains appropriate, with the accumulated negative controls constraining the next hypothesis. No change to the final primary-model contract, no propagation of the D38 candidate to partitions, and no Stage2/3/full648/final run or audit close is authorized by this checkpoint.

## Evidence

Task research preserves `d44_full_pool_root_diagnostic.py`, `d44_summary.json`, `d44_identity.json`, `d44_supervisor_check.py`, `d44_supervisor_results.json`, `d44_compare_checks.py`, and `d44_comparison_results.json`. Bulky joined rows and original models remain external. Summary SHA256: `2a8dafc680dff46c7921ac52ec35639a3a7de5f9438450d8d7f053f3780fb6bc`. Exact source hashes are in the identity and independent-comparison records.
