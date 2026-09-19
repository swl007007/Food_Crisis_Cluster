# Design: main Step 3 expert selective correction

> **Revision 2026-09-18 (authorized by user):** two contract changes, both driven by the discovery that the FEWS NET source is tri-annual (Feb/Jun/Oct, 53 records per admin) rather than monthly. (1) The expert series is now **calendar-aligned** at `O = T-H`, replacing the legacy record shifts, which resolved to 12–32 calendar months. (2) The validation interval is now `V=[O-12 months, O)`, because the six-month version admitted exactly one observed label month per fold and made the approved `distinct months >= 2` gate unsatisfiable. All selection gates (20 flips / 2 months / 0.75 precision / strict F1) are unchanged. Correcting the paper's archived FEWS NET baseline, which inherits the legacy defect, is explicitly deferred.

## Boundary and data flow

Use the existing main GeoRF Stage 3 entrypoint and partition selection/refinement behavior. For fs1/fs2, replace the partitioned prediction mechanism with:

`original expert e + original features X → partition RF score q=P(expert_wrong=1) → validation gate → e or 1-e`.

Keep the existing pooled branch separate and unchanged. For fs3 use the existing complete path, including its original fallback and sampling. Stage 1/2 remain unchanged. A fold is one scope, target month, and partition configuration; do not pool validation scores across scopes, outer months, or alternative partition configurations.

## Expert data contract

1. Use the original source `Outcome/FEWSNET_IPC/FEWSNET.csv`. Normalize admin/date keys using main-experiment conventions; reject duplicate valid admin-month keys. Build expert history before cutting training/evaluation dates or selecting the main cohort.
2. Sort each admin's full history by year/month. Convert `fews_proj_near` and `fews_proj_med` to binary Phase 3+ first, preserving the original raw-missing-phase → 0 rule. Then join **by calendar origin** (revised 2026-09-18, R1): the expert for target T is the projection published at `O = T-H` — fs1 `fews_proj_near` at `T-4`, fs2 `fews_proj_med` at `T-8` — because a near projection published in month D targets `D+4` and a medium projection targets `D+8`. Missing or absent publications at `O` remain missing; never impute. Still do not adopt the ETH fs3 anchor or any other ETH mapping. Additionally compute the legacy `shift(4)`/`shift(8)` record series as a pipeline-validation artifact only, clearly labelled, and never feed it to the correction layer.
3. Carry the source row's date, phase field, raw-missing flag, binary estimate, target month, and scope. Assert every used source date equals exactly `T-H`; any other lag is a contract error that halts the run. A leakage-direction date (source after `O`) halts unconditionally with no override. Within-month publication timing remains an explicit availability assumption unless actual release dates are supplied: the source carries publication month, not day.
4. Truth is the original main model's binary crisis outcome. Verify agreement with nonmissing target-month `fews_ipc >= 3`; never use contemporaneous truth as an expert input.
5. Exclude shift-induced unavailable expert rows from correction fitting and record counts. On required validation/test support, missing expert matches, duplicate keys, truth mismatches, or incompatible source dates are contract errors: stop the affected run, do not impute, silently shrink the comparison, or fabricate an expert prediction. The original raw-missing-phase → 0 convention in step 2 is preserved and separately audited.
6. Two distinct expert checks. (a) Reproduce archived expert fs1/fs2 baselines **from the legacy record-shift series** on their original support, proving the loader and conventions are faithful. (b) Report calendar-aligned coverage and exact-lag assertions for the series actually used. The two series differ by construction and must never be compared as if interchangeable; the archived paper baseline inherits the legacy defect and is out of scope to correct. Then compare methods on the exact existing main admin-month support. Do not borrow ETH's 90% coverage rule or its 2021-06 exclusion.

## Outer window and validation

Let target month be T, horizon H be 4 or 8 months, and origin O=T-H. Preserve the main splitter's actual temporal range `W=[O-35 months,O)` and its existing partition-group eligibility. Pooled keeps its own existing splitter/eligibility unchanged.

For correction, the validation interval is `V=[O-12 months,O)` intersected with eligible outer rows (revised 2026-09-18, R8: `O ≡ 2 (mod 4)` and source label months are tri-annual, so `[O-6,O)` holds exactly one observed label month in 24/24 folds and cannot satisfy the `distinct months >= 2` gate; `[O-12,O)` holds exactly three — `O-4`, `O-8`, `O-12`). Keep all rows from a month together. Fit one set of partition RFs using outer-window rows whose label month is strictly before `V_start-H`. This conservative single fit is safe at the earliest validation origin and hence at every later validation origin. Rows between that cutoff and V are withheld from the initial fit. Do not substitute random splits, anchor V on the last observed training month, or widen V beyond the twelve approved calendar months. Note the consequence, accepted with the R8 revision: the horizon-isolated first-stage fit retains 4 observed label months for fs1 and 3 for fs2; partitions falling under the 50-row minimum abstain as normal.

Use only fitting data for any learned preprocessing in this correction fit; apply it to V without refitting. Keep original feature definitions and missing-value conventions. Before implementation, trace preprocessing to ensure no fitting statistic uses validation/test data; do not alter the pooled preprocessing as a side effect.

Compute validation wrong scores, select the rule, then refit correction RFs on all eligible W rows using the same parameters and fit-only preprocessing. Apply the frozen rule to T. The available/abstaining partition set may change after refit; every unavailable final model still retains expert. Record both fitting stages' usable sample counts. If fitting/validation evidence is insufficient, select explicit no-correction rather than relaxing temporal rules.

## Partition learners

Use the existing partition maps and assignments (including existing refinement) without relearning from correction labels. Inputs are existing main X in its documented order plus binary e. Target is `w = 1[y != e]`.

RF configuration matches the current main RF: 100 trees, unlimited depth, random_state=5, n_jobs=1; all other defaults follow the recorded environment. Use no SMOTE, class weighting, or correction-specific hyperparameter grid. Minimum usable partition sample count is 50 and both wrong-label classes must exist.

Absent, unmapped, too-small, or single-class partition models abstain. Represent score as unavailable plus an eligibility flag, not as a crisis score from pooled. Such rows remain in validation F1 support but cannot be proposed flips. Persist an abstention reason. Runtime exceptions or invalid data are errors, not ordinary abstention.

## Selection algorithm

- Obtain finite wrong scores on eligible validation rows. Candidate thresholds are the sorted unique scores rounded to two decimal places, following ETH's selective mechanism; apply them to the original unrounded scores with strict `q > threshold`.
- For each candidate, a proposed flip requires an eligible score above threshold. Direction is determined by expert e: 0→1 or 1→0.
- For each direction, count proposed flips, distinct target months, and fixes (`y != e`). Enable it only if count >=20, months >=2, and fixes/count >=0.75. Precision uses original validation rows, never synthetic or weighted counts.
- **Variant B (2026-09-18):** run as a separate, explicitly named method `partitioned_selective_correction_up_only`, retaining Variant A's `partitioned_selective_correction` unchanged. Variant B forces `enable_1_to_0 = False` before candidate scoring, so only `0→1` flips can ever be proposed or applied; a `1→0` flip in Variant B output is a contract error. The restriction is justified by asymmetric cost, not by the oracle decomposition. Do not re-run or overwrite Variant A: its committed run `full_fs1_fs2_20260918` is the reference result.
- Apply only enabled directions. Score crisis-class F1 on all required validation rows, including abstentions, with the original zero-denominator convention.
- Start with the expert-only candidate. Replace it only for strictly greater F1. Iterate thresholds in ascending order so remaining exact ties are deterministic (first candidate wins). Both directions share one threshold; direction enable flags are separate.
- If no candidate strictly improves expert-only, no finite candidates exist, or validation cannot support the gates, select explicit no-correction. A validation set without positive truth cannot achieve strict crisis F1 improvement under the zero convention.
- Final prediction: `1-e` only if the selected rule enables that direction, the final model exists, and `q > threshold`; otherwise e. No threshold retuning, disabling, or fallback based on observed test performance.

The wrong score is not a crisis probability and is not declared probability-calibrated. Do not put it into a column named `y_prob_partitioned` or compute crisis probability metrics from it.

## Outputs and compatibility

Write a new, uniquely named correction run directory; refuse overwriting an existing run. Reuse existing prediction/metric/manifest layouts where their semantics still apply. Use explicit fs1/fs2 method ID `partitioned_selective_correction`, retain `pooled`, and retain `partitioned` at fs3 with correction disabled. Do not silently repurpose legacy partitioned label/probability columns or overwrite old output folders.

Row audit fields must retain keys, scope, partition, truth, original pooled prediction, expert prediction/source provenance, wrong score, eligibility/reason, selected threshold and direction flags, applied flip, and final prediction. Test fixes/damages are audit-only and computed after selection. fs3 correction fields are explicitly not applicable.

A compact fold tuning artifact records every threshold candidate's directional gates and final validation F1, the expert-only reference, selection status, validation range, fit/refit ranges and partition counts. Also persist validation row keys, truth, expert, wrong score and eligibility with an explicit validation role and outer-fold identifier, so gate counts and selection can be independently recomputed; keep these rows out of test metrics. Manifest records exact sources/hashes, environment, RF parameters, feature order, partition provenance, expert convention, and window endpoints. Consolidate these fields into existing analogous artifacts when practical; no separate framework.

Report the two requested methods on identical main support using existing monthly metrics and summary conventions. Expert-only appears only in selection/audit, not an additional results series. Figures/tables must mark fs3 as uncorrected and must not attribute differences solely to partitioning. Frozen paper deliverables require a separate promotion decision.

## Acceptance risks and implementation gates

The experiment expert is calendar-aligned at exactly `O = T-H` (R1, revised 2026-09-18); the legacy record-shift series survives only as a pipeline-validation artifact whose archived equality must still hold. Preserve the missing-phase conversion exactly while exposing provenance. Numerical equality and source availability must be checked against real data before running correction experiments. A failed check is a blocking discrepancy, not authorization to substitute ETH behavior.

Temporal purging and rare expert errors may produce many abstentions. This is intended. Do not weaken the approved 20/2/75% gates to obtain visible improvements. Unchanged pooled is an asymmetric comparator by user choice. No performance gain is promised.

## Authorized isolated execution (2026-09-18)

Create `Step3ExpertCorrectionExperiment/` under the current repository. Implement an isolated runner importing existing main helpers, without changing production entrypoints. Reuse frozen contig3 month maps in `paper_reproducibility_package/stage3_results/georf_fs1/refined/`: their Phase 2 parents are in `paper_reproducibility_package/stage2_cluster_maps/georf/`. These frozen maps reproduce all original main month-specific assignments; do not rebuild Phase 1/2 or repeat map refinement. Hash both source maps and used maps.

Reuse the exact archived pooled predictions for fs1/fs2 and both archived methods for unchanged fs3, with per-row equality and metric checks, rather than refitting an already frozen comparator under an unverifiable historical package environment. Clearly record this as reused baseline predictions, not newly trained pooled/fs3. Newly train only the fs1/fs2 correction branch. Archive and package prediction copies have matching hashes.

The main feature builder writes auxiliary CSVs: execute it only inside the experiment working directory. Capture its pre-imputation X and feature order while preserving main feature definitions; fit the same out-of-range imputation strategy separately inside correction fit/refit windows. Do not use the ETH aligned snapshots or fixed 88 predictors.
