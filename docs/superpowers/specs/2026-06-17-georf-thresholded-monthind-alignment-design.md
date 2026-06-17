# GeoRF Thresholded Month-Specific Partition Alignment Design

## Context

The paper-facing `01_main_results` and `12_thresholded_georf_results` should compare the same GeoRF partitioning regime when reporting pooled, partitioned, and validation-thresholded partitioned results. A read-only audit found that the current 12 artifacts are not aligned with 01:

- `y_true`, pooled predictions, and pooled probabilities match exactly.
- `partition_id` differs for most prediction rows.
- The 01 prediction `partition_id` distribution matches month-specific `m2`, `m6`, and `m10` refined partition maps.
- The 12 prediction `partition_id` distribution matches the general refined partition map.
- Existing run manifests record only the general `partition_map_path`; they do not record `month_ind`, month-specific partition paths, SMOTE availability, Python executable, or `imblearn` version.

The discrepancy is therefore a partition-assignment provenance problem, not evidence that phase-change data leaked into the artifacts. It is also not adequately explained as RF randomness.

## Goal

Regenerate `12_thresholded_georf_results` so the validation-selected threshold diagnostic uses the same month-specific partition assignment regime as `01_main_results`.

## Scope

In scope:

- GeoRF only.
- `result_partition_k40_compare_GF_thresholded_fs1`, `fs2`, and `fs3`.
- `final_artifacts_in_paper_updated/12_thresholded_georf_results`.
- Provider manifest fields needed to prove month-specific partitioning and SMOTE/Python provenance.
- Tests or verification checks that catch future `month_ind` misalignment.

Out of scope:

- Changing 01 main-result providers.
- Changing Stage 1 or Stage 2 partition learning.
- Adding GeoDT, GeoXGB, or pooled-only analyses.
- Reintroducing phase-change outcomes or FEWSNET 12-month baseline comparisons.

## Design

Run the thresholded GeoRF Stage 3 comparison with the same month-specific partition maps used by 01: February targets use the refined `m2` map, June targets use the refined `m6` map, October targets use the refined `m10` map, and any other evaluated target month would fall back to the general map. The thresholding logic remains unchanged: each target month selects a class-1 F1-maximizing threshold on validation data from the training window and applies it only to held-out test probabilities.

The provider manifest will be expanded so it records:

- `month_ind_enabled`.
- `partition_map_m2_path`, `partition_map_m6_path`, and `partition_map_m10_path`.
- File hashes for the general and month-specific partition maps.
- `smote_available`, `imblearn_version`, and `python_executable`.

The artifact builder for 12 will continue to summarize the thresholded provider outputs, but its manifest should make clear which provider manifests and partition maps were used.

## Validation

Verification should include:

- Compare 01 and regenerated 12 prediction rows on `FEWSNET_admin_code` and `month_start`.
- Assert `y_true`, `y_pred_pooled`, and `y_prob_pooled` still match exactly.
- Assert 12 `partition_id` matches the expected refined `m2/m6/m10` map for February, June, and October rows.
- Confirm 12 no longer matches the general map for all target months.
- Confirm no `phase_change` source appears in regenerated paper artifacts or thresholded providers.
- Run the existing reproducibility verifier after regeneration.

## Expected Outcome

After the fix, the 12 thresholded diagnostic will be directly comparable to 01 main results because both use the same month-specific partition assignment regime. Any remaining difference between `partitioned` and `partitioned_thresholded` in 12 will then reflect thresholding, not a change from month-specific to general partitions.
