# Data Model: GeoDT Branch-Specific Local Tree Interpretability Figure

## GeoDT Monthly Run

**Fields**:
- `selected_archive_path`: path to selected visual/archive directory.
- `artifact_provider_path`: optional same-run/source folder path when different from selected archive.
- `model_family`: expected `GeoDT`.
- `run_year_month`: selected monthly run token such as `2024-10`.
- `forecasting_scope`: scope token such as `fs1`.
- `archive_discovery_input`: explicit path, explicit list, discovery root, configured root, or failure reason.
- `discovered_candidate_count`: integer.
- `fallback_selection_reason`: text.
- `candidate_decisions`: list of archive candidate decisions.
- `minimum_artifact_completeness`: complete/incomplete plus details.
- `same_run_compatibility_evidence`: required when artifact provider differs from selected archive.

**Validation rules**:
- Must be GeoDT monthly archive/source identity.
- Must not be considered complete solely because root/global `dt_rules` exist.
- Artifact-provider folder is valid only with same model family, year/month, forecasting scope, and run identity evidence.

## Archive Candidate Decision

**Fields**:
- `candidate_path`
- `parsed_year_month`
- `parsed_forecasting_scope`
- `is_preferred_exact_match`
- `month_distance_from_preferred`
- `minimum_artifact_score_or_status`
- `accepted`: boolean
- `rejection_reason`

**Validation rules**:
- Candidates must come only from explicit path/list/root or configured root.
- Tie-breaking must be deterministic.

## Checkpoint Record

**Fields**:
- `checkpoint_path`
- `filename`
- `parsed_branch_id`
- `classification`: root/global, branch-specific candidate, unusable, unknown.
- `parsing_method`
- `loader_validation_status`
- `checkpoint_feature_count`
- `mismatch_reason`

**Validation rules**:
- Filename pattern alone cannot prove branch-specific terminal eligibility.
- Root/global checkpoint is excluded unless explicitly terminal in dispatch assignments.
- Load failure marks checkpoint unusable.

## Branch Assignment Source

**Fields**:
- `source_path`
- `source_type`: X_branch_id, s_branch, correspondence_table, equivalent, unavailable.
- `precedence_rank`
- `evidence_validation_status`: confirmed, rejected, not applicable.
- `rejection_reason`
- `branch_counts`: mapping branch ID to assigned admin/group and/or prediction-row count.

**Validation rules**:
- Source must be tied to selected run/month/scope.
- Correspondence table must contain reliable terminal branch IDs before use.

## Feature-Name Source

**Fields**:
- `source_path`
- `source_type`: checkpoint/model bundle, pipeline feature column file, training matrix column order, configured feature list, unavailable.
- `feature_names`: ordered list.
- `feature_name_count`
- `checkpoint_feature_count`
- `compatibility_result`
- `rejected_sources`: list with reasons.

**Validation rules**:
- Length must match checkpoint feature count when available.
- All split feature indices in selected trees must be in bounds.
- Incompatible sources cannot be used for plotting.

## Branch Eligibility Record

**Fields**:
- `branch_id`
- `assignment_present`: boolean.
- `assigned_admin_group_count`
- `prediction_row_count`
- `training_sample_count_for_checkpoint`
- `checkpoint_record`
- `feature_name_status`
- `is_root_global`
- `eligibility_status`: eligible, ineligible, failed, skipped.
- `exclusion_reasons`

**Validation rules**:
- Must have final dispatch assignment, nonzero coverage, loadable matching checkpoint, reliable feature names, and root/global eligibility exception only when explicitly terminal.

## Branch Signature

**Fields**:
- `branch_id`
- `k_value`
- `actual_available_depth`
- `split_feature_set`
- `split_nodes`: feature, threshold, direction, depth, node identifier.
- `leaf_class_summary`: optional.
- `class_label_source`: optional.

**Validation rules**:
- Feature set contains non-leaf split feature names at depths `0` through `K-1`.
- If no non-leaf splits exist within plotted depth, readability fails.

## Pair Score

**Fields**:
- `branch_pair`: normalized ordered pair.
- `feature_set_a`
- `feature_set_b`
- `jaccard_distance`
- `supplemental_threshold_direction_fields`
- `tie_break_fields`
- `readability_result`
- `selected`: boolean
- `rejection_reason`

**Validation rules**:
- If both feature sets are empty, pair is not eligible.
- Default ranking uses only Jaccard distance and deterministic tie-breaks.
- Threshold/direction fields do not alter default selection.

## Readability Result

**Fields**:
- `has_non_leaf_split_a`
- `has_non_leaf_split_b`
- `same_plotted_max_depth`
- `same_style`
- `label_wrapping_or_abbreviation`
- `abbreviation_map`
- `minimum_font_size`
- `class_color_legend_status`
- `passed`: boolean
- `failure_reasons`

**Validation rules**:
- No silent truncation.
- Abbreviations preserve unique mapping.
- Font size should not fall below 6 pt or equivalent.

## Metadata Summary

**Fields**:
- All provenance fields required by `spec.md` FR-031.
- Links to figure, audit summary, reproduction report, and failure summary when applicable.

**Validation rules**:
- Must support reproduction mode recovering selected archive, branches, checkpoint paths, feature source, top-K features, score, and figure inputs.

## Failure Summary

**Fields**:
- `failure_stage`
- `missing_or_invalid_artifacts`
- `evidence_mismatch`
- `candidate_archive_decisions`
- `recommended_next_action`
- `partial_outputs_created`

**Validation rules**:
- Must not indicate successful figure generation.
- Must not silently fall back to root/global rules.
