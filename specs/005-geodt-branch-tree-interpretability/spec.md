# Feature Specification: GeoDT Branch-Specific Local Tree Interpretability Figure

**Feature Branch**: `005-geodt-branch-tree-interpretability`  
**Created**: 2026-06-05  
**Status**: Draft  
**Input**: User description: "Create an interpretability diagnostic figure for GeoDT that demonstrates why spatial partitioning matters by selecting two substantially different terminal branch-specific decision trees from one archived monthly GeoDT run and visualizing their shallow decision rules side by side."

## Clarifications

### Session 2026-06-05

- Q: If the preferred visual archive lacks branch checkpoints or partition artifacts, should the diagnostic be allowed to use a matching source run folder as the artifact provider? → A: Allow a same-run/source folder artifact provider when the visual archive lacks required branch checkpoints or partition artifacts, with metadata recording both paths and compatibility evidence.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate Branch-Tree Comparison Figure (Priority: P1)

As a researcher evaluating GeoDT interpretability, I want to generate a side-by-side figure comparing two terminal branch-specific local DecisionTree models from the same archived monthly GeoDT run, so that I can visually demonstrate that GeoDT learns different local decision rules for different spatial partitions rather than relying only on a single global decision tree.

**Why this priority**: This is the core value of the feature. The figure must support the paper/report claim that GeoDT's spatial partitioning can change the decision logic applied to different groups or regions, when the selected branches show contrast under the recorded default dissimilarity score.

**Independent Test**: Run the diagnostic on one eligible archived GeoDT monthly result folder. The test passes if it produces a 1x2 figure with two branch-specific local trees, both loaded from branch checkpoints, both truncated to the same shallow depth, and both labeled with branch metadata.

**Acceptance Scenarios**:

1. **Given** an eligible archived GeoDT monthly result directory with branch checkpoints, branch assignments, and feature names, **When** the diagnostic runs, **Then** it produces a 1x2 PNG/PDF figure comparing two eligible terminal branch trees.
2. **Given** the selected pair has a recorded default top-K split-feature Jaccard dissimilarity score and passes readability checks, **When** the figure is rendered, **Then** each panel displays the selected branch's recorded top-K split features, the metadata records the pair score, and all split labels satisfy the figure readability requirements.
3. **Given** the exact preferred run is unavailable, **When** the diagnostic selects a fallback run using the FR-003A fallback order, **Then** the figure and metadata clearly state the fallback run and why it was selected.

---

### User Story 2 - Audit Candidate Branch Selection Without Rendering the Final Figure (Priority: P2)

As a model analyst, I want to run a branch-selection audit that lists eligible terminal branches, shallow signatures, pairwise dissimilarity scores, rejected candidates, and the selected branch pair, so that I can verify the comparison is systematic rather than cherry-picked.

**Why this priority**: The visualization should be defensible as an interpretability diagnostic. A standalone audit lets analysts validate branch eligibility and pair selection before rendering the final figure.

**Independent Test**: Run audit-only mode as defined by FR-AUDIT-001 through FR-AUDIT-004. The test passes if it performs archive discovery, preflight, archive completeness checks, branch eligibility checks, feature-name source selection, signature extraction, dissimilarity scoring, readability checks, and metadata/audit output without requiring PNG/PDF figure generation.

**Acceptance Scenarios**:

1. **Given** multiple terminal branch checkpoints and dispatch assignments, **When** the audit runs, **Then** it lists all eligible and ineligible branches with reasons.
2. **Given** all candidate branch signatures are extracted, **When** pairwise scoring completes, **Then** the audit records the exact default scoring formula, K value, each candidate pair's feature sets, each candidate pair's Jaccard score, the selected highest-ranked readable pair, and any rejected higher-scoring pair.
3. **Given** the top-scoring pair is rejected for readability or missing plotting information, **When** the audit completes, **Then** it records the rejected pair, score, and reason.

---

### User Story 3 - Reproduce a Previously Generated Branch-Tree Figure from Metadata (Priority: P3)

As a reviewer or future maintainer, I want to reproduce the selected branch comparison using the metadata summary and archived artifacts, so that I can audit the figure's provenance and verify that it used branch-specific checkpoints rather than root/global rule exports.

**Why this priority**: The figure is intended for interpretability in a paper/report. Reproduction from metadata is the strongest guard against confusing branch-specific local checkpoint diagnostics with existing root/global rule CSV exports.

**Independent Test**: Run reproduction mode with a prior metadata summary path as input. The test passes if reproduction mode accepts the metadata path and recovers the same selected archive, selected branch IDs, checkpoint paths, feature-name source, top-K features, and default dissimilarity score. Bitwise-identical image output is not required.

**Acceptance Scenarios**:

1. **Given** a metadata summary from a prior successful run, **When** the reproduction workflow is run, **Then** it resolves the same run folder, branch assignment source, feature-name source, checkpoint paths, branch IDs, K value, and dissimilarity formula. Bitwise-identical image reproduction is not required.
2. **Given** existing `dt_rules_*.csv` files describe only the root/global tree, **When** the metadata is inspected, **Then** it explicitly states that the figure was generated from branch-specific checkpoint models and not from root/global rule CSVs.

---

### Edge Cases

- Preferred archive `result_GeoDT_2024_fs1_2024-10_visual` is unavailable.
  - Use the fallback selection order defined in FR-003A.
  - Clearly state the chosen year/month/forecasting scope, fallback selection reason, and considered candidate outcomes in both figure metadata and audit output.
- No archive path, archive list, or archive discovery root is provided.
  - Use the configured project result/archive root if available.
  - Otherwise fail during preflight with a clear message.
  - Do not perform unbounded recursive filesystem search.
- No available GeoDT archive satisfies the minimum required artifact set in FR-000I through FR-000N.
  - Fail during preflight with a clear diagnostic summary.
  - Do not substitute root/global `dt_rules_*.csv` exports for branch-specific checkpoint analysis.
- Fewer than two eligible terminal branches are assigned by the selected run's final dispatch artifact and backed by loadable checkpoints.
  - Do not fabricate a comparison.
  - Produce a diagnostic note listing branch eligibility failures instead of a misleading 1x2 local-branch comparison.
- Root/global checkpoint `branch_id=''` appears in the checkpoint directory.
  - Exclude it unless the dispatch artifact explicitly assigns at least one admin group, admin unit, or prediction row to `branch_id=''` as a terminal prediction branch.
- Checkpoint files exist but are not assigned to any admin group, admin unit, or prediction row by the selected run's dispatch artifact.
  - Exclude unused branch trees from candidate selection and record the exclusion reason.
- Terminal branch assignment data are missing from the highest-precedence source.
  - Use the next source in the defined branch assignment source precedence only if it can be tied to the selected run/month/scope.
  - If no reliable assignment source exists, fail during preflight with a clear message.
- Feature reference/name file is missing or cannot be mapped reliably to tree split indices under FR-006D through FR-006H.
  - Fail with a clear message rather than plotting unreadable or incorrect feature labels.
- A selected tree has fewer than K levels.
  - Plot all available shallow levels for that tree and record the actual plotted depth used.
- Class distribution or class-to-label mapping is unavailable.
  - Use neutral `class 0` / `class 1` labels or omit class tendency text.
  - Do not label any node or leaf as crisis/non-crisis without a confirmed class-label mapping source.
- Branch signatures are highly similar across all candidates.
  - Still select the highest-ranked readable pair under the default Jaccard dissimilarity score, record the score, apply the contrast descriptor in FR-028B, and avoid overstating visual contrast.
- The highest-scoring pair fails readability checks under FR-016C, FR-016F, FR-024C, FR-024D, or FR-024E.
  - Select the next-highest-scoring readable pair if available.
  - Record the rejected higher-scoring pair, score, and rejection reason.
- Multiple readable branch pairs tie on the default dissimilarity score.
  - Apply deterministic tie-break rules in FR-016G.
  - Record the tie and tie-break result in metadata.
- No branch pair passes readability checks under FR-016C, FR-016F, FR-024C, FR-024D, or FR-024E.
  - Fail with a clear diagnostic summary rather than generating a misleading figure.
- Checkpoint loading fails because of environment, serialization, or package-version mismatch.
  - Report the affected checkpoint path and do not silently fall back to root/global rules.
- Output figure becomes unreadable because feature names are long.
  - Apply wrapping, abbreviation, or font-size adjustment while preserving a unique mapping to original feature names.
  - Document abbreviations in metadata or a figure note.
- Reproduction metadata points to missing or changed artifacts.
  - Fail reproduction mode with a clear mismatch report rather than selecting replacement artifacts.
- Failure occurs after preflight begins.
  - Produce or display a structured failure summary with failure stage, missing or invalid artifacts, evidence mismatch when relevant, candidate archive decisions, and recommended next action.

## Requirements *(mandatory)*

### Existing Behavior Preservation

- **Existing behavior to preserve**: Existing GeoDT training/evaluation behavior must remain unchanged. Existing spatial partitioning, statistical testing, prediction dispatch, checkpoint naming, and root/global `dt_rules_*.csv` export behavior must remain unchanged. Existing generated artifacts must not be overwritten or reinterpreted.
- **Explicitly allowed behavior changes**: The feature may add an exploratory diagnostic workflow that reads archived/source GeoDT artifacts and writes a figure plus metadata summary. It may select branch-specific checkpoints for visualization, compute deterministic shallow-rule dissimilarity, run preflight/audit/reproduction checks, and generate diagnostic notes when required artifacts are unavailable.
- **Unknown legacy behavior requiring clarification**: It is unknown whether the preferred monthly visual archive `result_GeoDT_2024_fs1_2024-10_visual` contains branch-specific checkpoints, `space_partitions/s_branch.pkl`, `X_branch_id.npy`, feature-name references, or correspondence tables with reliable terminal branch IDs. It is also unknown whether branch-specific checkpoint availability is consistent across all monthly GeoDT archives.
- **Backward compatibility requirements**: Existing GeoDT workflows, root/global DT rule exports, model artifacts, generated result directories, partition artifacts, and evaluation outputs must remain compatible with current usage. Existing generated artifacts must not be overwritten or reinterpreted.
- **Public API / CLI / config / data artifact compatibility**: The diagnostic MUST remain compatible with existing GeoDT workflows and artifact conventions. Known examples include batch launchers such as `run_batches_2021_2024_visual_monthly.bat geodt`, model entry points such as `app/main_model_DT.py --start_year --end_year --forecasting_scope`, configuration flags such as `SAVE_DT_RULES` and `SAVE_DT_NODE_DUMP`, checkpoint artifacts such as `checkpoints/dt_*`, partition artifacts such as `space_partitions/s_branch.pkl` and `space_partitions/X_branch_id.npy`, correspondence tables such as `correspondence_table_*.csv`, and root/global rule exports such as `dt_rules/dt_rules_*.csv`. These examples are illustrative of known brownfield conventions, not a requirement to create, rename, or refactor those paths.
- **Verified-path requirement**: Any path used by the diagnostic MUST be discovered from the selected archive, configured explicitly, or confirmed by preflight. The implementation MUST NOT assume that every known example path exists in every monthly archive.
- **Out-of-scope cleanup or refactor**: This feature does not refactor GeoDT training, prediction dispatch, branch training, existing DT rule export, feature engineering, lag schedules, shapefile/map workflows, or generated artifact layout. It does not rename existing `dt_rules` outputs or clean up legacy result directories.

### Preflight Requirements

- **FR-000**: System MUST run a preflight archive characterization step before branch selection or figure generation.
- **FR-000A**: The preflight MUST identify available GeoDT monthly archives, the minimum required artifact set in FR-000I through FR-000N, branch-specific checkpoints, root/global checkpoint, branch assignment sources, feature-name sources, candidate terminal branches, and missing or unusable artifacts.
- **FR-000B**: The preflight MUST classify each discovered checkpoint as root/global, branch-specific candidate, unusable, or unknown.
- **FR-000C**: The preflight MUST fail before plotting if no reliable branch assignment source or feature-name source exists.
- **FR-000D**: The preflight MUST produce an audit record that is included in or linked from the metadata summary.
- **FR-000E**: The preflight MUST explicitly state whether the selected archive contains enough artifacts to support branch-specific local tree comparison under the minimum required artifact set in FR-000I through FR-000N. If it does not, the diagnostic must not fall back to root/global `dt_rules` as a substitute.
- **FR-000F**: The diagnostic MUST accept or document an archive discovery root or explicit archive path/list. If an explicit preferred archive path is provided, preflight MUST evaluate that path first. If only an archive discovery root is provided, preflight MUST search only within that root for GeoDT monthly archive candidates. If neither an explicit archive path nor a discovery root is provided, the diagnostic MUST use a configured project result/archive root if available; otherwise it MUST fail with a clear message rather than recursively searching arbitrary directories.
- **FR-000G**: Archive discovery MUST be bounded and deterministic. The diagnostic MUST NOT search unrelated user directories, temporary folders, or the entire filesystem.
- **FR-000H**: Metadata MUST record the archive discovery input used: explicit archive path, archive list, configured archive root, or failure reason if none was available.

### Minimum Required Artifact Set

- **FR-000I**: For a GeoDT monthly archive to be considered complete for this diagnostic, it MUST contain or provide access to all of the following:
  1. the selected GeoDT monthly archive directory;
  2. at least two loadable branch-specific DecisionTree checkpoints;
  3. one reliable branch assignment source under FR-006B;
  4. one reliable feature-name source under FR-006D through FR-006H;
  5. enough tree structure information in the checkpoints to render non-leaf split nodes and leaf nodes;
  6. enough branch assignment information to compute assigned admin/group count or prediction-row count for each eligible branch.
- **FR-000M**: If the selected monthly visual archive lacks required branch checkpoints, partition artifacts, or feature-name artifacts, it MAY still be considered complete only when preflight identifies a matching same-run/source run folder that provides the missing artifacts and confirms compatibility with the selected archive's model family, year/month, forecasting scope, and run identity.
- **FR-000N**: When a same-run/source run folder provides required artifacts for a selected visual archive, metadata MUST record both the selected visual archive path and the artifact-provider source run path, the artifacts provided by each location, and the evidence used to confirm same-run compatibility.
- **FR-000J**: Class-to-label mapping is NOT required for archive completeness. If class semantics are unavailable, the figure MUST use neutral class labels or omit class tendency text.
- **FR-000K**: Training-sample count is NOT required for archive completeness. If unavailable, metadata MUST mark `training_sample_count_for_checkpoint` as unavailable.
- **FR-000L**: A monthly archive that contains only root/global `dt_rules_*.csv` exports but no usable branch-specific checkpoints MUST NOT be considered complete for this diagnostic.

### Functional Requirements

- **FR-001**: System MUST create an interpretability diagnostic for GeoDT's spatial partitioning mechanism using branch-specific local DecisionTree checkpoints from one archived monthly GeoDT run.
- **FR-002**: System MUST prefer the archived run with year/month `2024-10`, forecasting scope `fs1`, and expected archive folder pattern `result_GeoDT_2024_fs1_2024-10_visual`.
- **FR-003**: If the preferred run is unavailable or does not satisfy the minimum required artifact set in FR-000I through FR-000N, System MUST use the fallback selection order in FR-003A and MUST state the chosen year, month, forecasting scope, archive folder, and fallback selection reason in the figure metadata and audit summary.
- **FR-003A**: Fallback run selection MUST use this ordered rule:
  1. Use the exact preferred archive if available: year/month `2024-10`, forecasting scope `fs1`, expected folder pattern `result_GeoDT_2024_fs1_2024-10_visual`.
  2. Otherwise, choose the available GeoDT archive with the same forecasting scope `fs1` and the smallest absolute month distance from `2024-10`, provided it satisfies the minimum required artifact set in FR-000I through FR-000N.
  3. Otherwise, choose the available GeoDT monthly archive with the smallest absolute month distance from `2024-10`, regardless of forecasting scope, provided it satisfies the minimum required artifact set in FR-000I through FR-000N.
  4. If multiple candidates tie, choose the candidate with the most complete minimum required artifact set in FR-000I through FR-000N.
  5. If no candidate satisfies the minimum required artifact set in FR-000I through FR-000N, fail during preflight with a clear diagnostic summary.
- **FR-003B**: Metadata MUST record all considered fallback candidates, whether each was accepted or rejected, and the reason for the final selected run.
- **FR-004**: System MUST inspect the archived GeoDT result directory for DecisionTree checkpoints named like `dt_`, `dt_0`, `dt_1`, `dt_00`, `dt_01`, or equivalent checkpoint naming used by the run, and classify each discovered checkpoint during preflight.
- **FR-005**: System MUST identify terminal branches that are actually assigned by the selected run's final dispatch artifacts.
- **FR-006**: System MUST use the branch assignment source precedence in FR-006B to map admin groups, admin units, or prediction rows to terminal branch IDs.

### Branch Eligibility and Assignment Source Requirements

- **FR-006A**: A branch is eligible for comparison only if all of the following are true:
  - its branch ID appears in the selected run's final dispatch branch assignment artifact;
  - at least one admin group, admin unit, or prediction row is assigned to it;
  - a matching branch-specific DecisionTree checkpoint exists;
  - the checkpoint loads successfully;
  - feature names can be mapped reliably to the tree's split feature indices under FR-006D through FR-006H;
  - it is not the root/global branch `branch_id=''`, unless the dispatch artifact explicitly assigns at least one admin group, admin unit, or prediction row to `branch_id=''` as a terminal prediction branch.
- **FR-006B**: Branch assignment source precedence MUST be:
  1. final prediction dispatch artifact, such as `space_partitions/X_branch_id.npy`, if present and mappable to rows/groups;
  2. `space_partitions/s_branch.pkl`, if it maps admin groups or prediction rows to terminal branches for the selected run;
  3. `correspondence_table_*.csv`, only if it contains terminal branch IDs for the same run/month/scope;
  4. equivalent run artifact, only if its mapping to final prediction dispatch is documented in metadata;
  5. otherwise fail with a clear preflight error.
- **FR-006C**: Checkpoint filename presence alone MUST NOT be treated as evidence that a branch is terminal or used for prediction dispatch.

### Feature-Name Source Requirements

- **FR-006D**: Feature-name source precedence MUST be:
  1. selected-run feature-name artifact explicitly saved with the branch checkpoint or model bundle;
  2. selected-run pipeline feature column file;
  3. selected-run training matrix column order, if recoverable and tied to the same run/month/scope;
  4. configured feature column list, only if it matches the checkpoint feature count;
  5. otherwise fail preflight.
- **FR-006E**: The selected feature-name source MUST be tied to the selected archive, run month, forecasting scope, or checkpoint bundle. A feature list from a different run/month/scope MUST NOT be used unless metadata explicitly records why it is compatible.
- **FR-006F**: If a checkpoint exposes feature count metadata such as `n_features_in_`, the feature-name source length MUST match that count. If it does not match, the checkpoint MUST be marked unusable for this diagnostic unless a documented model-specific reason explains the mismatch.
- **FR-006G**: If any split feature index used by a selected tree is out of bounds for the selected feature-name source, the checkpoint MUST be marked unusable.
- **FR-006H**: If multiple feature-name sources are found, preflight MUST record which source was selected, which sources were rejected, and the reason for selection.

### Checkpoint Parsing Requirements

- **FR-CHK-001**: The diagnostic MUST parse branch IDs from checkpoint filenames using the selected run's known checkpoint naming convention or existing checkpoint loader behavior when available.
- **FR-CHK-002**: The root/global checkpoint MUST be identified explicitly. A filename such as `dt_` MAY represent the root/global checkpoint only if confirmed by loader behavior, metadata, or brownfield evidence.
- **FR-CHK-003**: A branch-specific checkpoint filename MUST NOT be trusted solely by pattern matching if there is contradictory metadata or loader behavior.
- **FR-CHK-004**: If a checkpoint's parsed branch ID conflicts with dispatch assignment branch IDs or checkpoint metadata, the checkpoint MUST be marked unusable until the mismatch is resolved.
- **FR-CHK-005**: Metadata MUST record the checkpoint filename, parsed branch ID, checkpoint classification, parsing method, and any mismatch reason.

### Branch Signature, Scoring, and Selection Requirements

- **FR-007**: System MUST exclude the root/global tree with `branch_id=''` unless it is explicitly assigned as a terminal prediction branch by the selected run's dispatch artifact.
- **FR-008**: System MUST NOT rely on existing `dt_rules_*.csv` files if those files only export the root/global tree.
- **FR-009**: System MUST load and analyze branch-specific checkpoint models directly when constructing the comparison.
- **FR-010**: System MUST extract a compact shallow signature from each eligible candidate branch tree using split features at depths `0` through `K-1`.
- **FR-011**: Each branch signature MUST include top-level split feature names and MAY include threshold/direction details as supplemental audit fields.
- **FR-012**: Each branch signature SHOULD include shallow leaf class distribution or dominant class when available, using neutral class labels unless class-to-label mapping is confirmed.
- **FR-013**: System MUST compute pairwise dissimilarity between eligible branch signatures.
- **FR-014**: The default branch-pair dissimilarity score MUST be deterministic and MUST be computed as:

  `D(A, B) = JaccardDistance(F_A, F_B)`

  where `F_A` and `F_B` are sets of split feature names appearing in non-leaf nodes at depths `0` through `K-1`.
- **FR-014A**: `K` MUST default to `3`. If either tree is shallower than depth 3, use the available shallow depth for that tree and record the actual depth used.
- **FR-014B**: `JaccardDistance(F_A, F_B)` MUST be computed as `1 - |F_A ∩ F_B| / |F_A ∪ F_B|`. If both feature sets are empty, the pair is not eligible for dissimilarity ranking.
- **FR-015**: Threshold/direction mismatch penalties MUST NOT be used in the default branch-pair ranking. They MAY be computed as supplemental metadata only, but MUST NOT change the selected pair unless a future spec revision explicitly changes the scoring formula.
- **FR-016**: System MUST select two terminal branch trees for comparison only after eligibility, scoring, and readability checks are complete.
- **FR-016A**: The system MUST rank all eligible branch pairs by the default dissimilarity score.
- **FR-016B**: The system SHOULD select the highest-scoring pair that passes figure readability checks under FR-016C, FR-016F, FR-024C, FR-024D, and FR-024E.
- **FR-016G**: If multiple readable branch pairs tie on the default dissimilarity score, the system MUST break ties deterministically using the following order:
  1. larger minimum assigned admin/group count across the two branches;
  2. larger total assigned admin/group count across the two branches;
  3. larger minimum prediction-row count if available;
  4. lexicographic order of normalized branch ID pair;
  5. if still tied, record the tie and select the first pair under deterministic sorted order.
- **FR-016H**: Metadata MUST record any tie, the tie-break fields used, and the final selected pair.
- **FR-016C**: Readability checks MUST include:
  - both trees have at least one non-leaf split within the plotted depth;
  - split feature labels can be rendered legibly after wrapping or abbreviation;
  - both panels can be plotted at the same depth and comparable visual scale;
  - neither panel is empty or visually degenerate.
- **FR-016F**: Comparable visual scale means both panels MUST use the same plotted max depth, same figure dimensions, same tree plotting function or equivalent visual style, and same class-color legend when class coloring is used.
- **FR-016D**: If the highest-scoring pair fails readability checks under FR-016C, FR-016F, FR-024C, FR-024D, or FR-024E, the system MAY select the next-highest-scoring readable pair, but metadata MUST record the rejected pair, its score, and rejection reason.
- **FR-016E**: If no branch pair passes readability checks under FR-016C, FR-016F, FR-024C, FR-024D, or FR-024E, fail with a clear diagnostic summary rather than generating a misleading figure.

### Branch Coverage and Count Requirements

- **FR-017**: System MUST report selected branch IDs and assigned admin/group count for each selected branch.
- **FR-017A**: System SHOULD separately report `prediction_row_count_for_branch` when final dispatch rows are available.
- **FR-017B**: System SHOULD separately report `training_sample_count_for_checkpoint` when the local tree checkpoint or training artifact exposes this value.
- **FR-017C**: System MUST NOT collapse assigned admin/group count, prediction-row count, and training-sample count into a single ambiguous count field.
- **FR-017D**: If any count is unavailable, metadata MUST mark that field as unavailable rather than guessing.

### Figure Requirements

- **FR-018**: System MUST produce a 1x2 subplot figure comparing the top few layers of the two selected branch-specific DecisionTree models.
- **FR-019**: The left panel MUST show Branch A's local DecisionTree rules truncated to the selected shallow plotted depth.
- **FR-020**: The right panel MUST show Branch B's local DecisionTree rules truncated to the same plotted depth and visual style as Branch A, with actual available depth recorded for each tree.
- **FR-021**: Each panel title MUST use the format `Branch <branch_id>: local DT rules`.
- **FR-022**: Each panel SHOULD include assigned admin/group count and class tendency when available and safe to label.
- **FR-022A**: The figure MUST NOT label a branch, node, or leaf as "crisis" or "non-crisis" unless the class-to-label mapping is recovered from run artifacts, pipeline constants, or another explicitly documented source.
- **FR-022B**: If class-to-label mapping cannot be confirmed, the figure MUST use neutral labels such as `class 0` and `class 1`, or omit class tendency text.
- **FR-022C**: Metadata MUST record the class-label source whenever crisis/non-crisis tendency is displayed.
- **FR-023**: The visualization MUST use matched layout and scale across both panels, with comparable visual scale as defined in FR-016F.
- **FR-023A**: The 1x2 figure MUST use a minimum width of 10 inches and a minimum height of 4 inches unless the output format imposes a different layout constraint.
- **FR-023B**: PNG output MUST be rendered at 300 DPI or higher.
- **FR-023C**: PDF output SHOULD preserve vector text where supported by the plotting backend.
- **FR-024**: Split feature names MUST be readable in the final figure under FR-024C, FR-024D, and FR-024E.
- **FR-024A**: Split labels MUST remain legible at 100% zoom after wrapping, abbreviation, or font-size adjustment.
- **FR-024B**: Any abbreviation of feature names MUST be documented in metadata or a figure note if it could make interpretation ambiguous.
- **FR-024C**: No split label may be silently truncated. If a split label is shortened, the abbreviation MUST preserve a unique mapping to the original feature name and be documented in metadata or a figure note.
- **FR-024D**: Rendered split-label font size SHOULD NOT fall below 6 pt in PDF output or the equivalent readable pixel height in PNG output. If the plotting backend cannot report font size, the diagnostic MUST use a configured minimum font size and record it in metadata.
- **FR-024E**: If label wrapping, abbreviation, or minimum font-size constraints still cannot make labels readable within the required layout, the affected pair MUST fail readability checks.
- **FR-025**: Leaves SHOULD use color, shading, or equivalent visual encoding to indicate predicted class or class purity when class outputs are available.
- **FR-025A**: Leaf color or purity encoding MUST be described in the caption, legend, or metadata. If class semantics are unknown, the legend MUST use neutral labels such as `class 0` and `class 1` and MUST avoid crisis/non-crisis terminology.
- **FR-025B**: If class output information is unavailable from the tree object, leaf color encoding MAY be omitted, and metadata MUST record that class output information was unavailable.
- **FR-026**: The figure MUST keep only shallow tree layers and MUST prioritize conceptual contrast over full-tree exhaustiveness.
- **FR-027**: The figure caption or subtitle MUST explain that the two local trees were selected as the highest-ranked readable pair under the default dissimilarity score within the same GeoDT monthly model.
- **FR-028**: The caption or subtitle MUST state that the plotted trees compare local decision rules used by different terminal branches in the selected monthly model and that existing root/global DT rule exports do not demonstrate final branch-dispatched local tree behavior.
- **FR-028A**: If the selected pair's Jaccard dissimilarity score is low or branches are highly similar, the caption MUST avoid claiming strong regional heterogeneity. It may state that the figure shows the highest-ranked readable contrast available in the selected run.
- **FR-028B**: Metadata MUST record the selected pair's score and whether the diagnostic considered the contrast low, moderate, or high using a documented descriptive rule. The default descriptive rule is: high contrast when Jaccard distance is `>= 0.75`; moderate contrast when `0.40 <= Jaccard distance < 0.75`; low contrast when Jaccard distance is `< 0.40`.
- **FR-029**: System MUST produce at least one figure output that satisfies the measurable layout, DPI, and label-readability requirements above.
- **FR-030**: System SHOULD produce both PNG and PDF outputs when feasible.

### Audit-Only Mode Requirements

- **FR-AUDIT-001**: The diagnostic MUST expose an audit-only mode that performs archive discovery, preflight, archive completeness checks, branch eligibility checks, feature-name source selection, signature extraction, dissimilarity scoring, readability checks, and metadata/audit output without rendering the final figure.
- **FR-AUDIT-002**: Audit-only mode MUST produce the same selection decision that figure-generation mode would use, assuming the same inputs and available artifacts.
- **FR-AUDIT-003**: Audit-only mode MUST write or display an audit summary containing eligible branches, ineligible branches, rejected or skipped candidate branches, branch-pair scores, readability decisions, selected pair, fallback selection decision, and root/global exclusion decision.
- **FR-AUDIT-004**: Audit-only mode MUST be independently testable without requiring PNG/PDF output.

### Reproduction Mode Requirements

- **FR-REPRO-001**: The diagnostic MUST expose a reproduction mode that accepts a prior metadata summary path as input.
- **FR-REPRO-002**: Reproduction mode MUST use the metadata summary to resolve the recorded archive discovery input, selected archive folder, selected run year/month, forecasting scope, branch assignment source, feature-name source, selected branch IDs, checkpoint paths, K value, actual plotted depths, and dissimilarity formula.
- **FR-REPRO-003**: Reproduction mode MUST verify that the recovered branch IDs, checkpoint paths, top-K feature sets, and default dissimilarity score match the metadata summary.
- **FR-REPRO-004**: Reproduction mode MUST NOT reselect a different branch pair unless the user explicitly requests a new selection run. By default, it must reproduce the recorded selection.
- **FR-REPRO-005**: Bitwise-identical image reproduction is NOT required. Reproduction succeeds if the selected archive, selected branch IDs, checkpoint paths, feature-name source, top-K feature sets, default dissimilarity score, and figure inputs match the metadata summary.
- **FR-REPRO-006**: If recorded artifacts are missing or changed, reproduction mode MUST fail with a clear mismatch report rather than silently selecting replacement artifacts.

### Output Artifact Location Requirements

- **FR-OUT-001**: The diagnostic MUST accept or document an output directory. If no output directory is provided, outputs SHOULD be written under a diagnostics/output directory inside the selected archive or under a configured project diagnostic output root.
- **FR-OUT-002**: Output filenames SHOULD include selected run year-month, forecasting scope, selected branch IDs, and artifact type.
- **FR-OUT-003**: The diagnostic MUST NOT overwrite existing output files unless overwrite behavior is explicitly requested or configured. If an output path already exists, the diagnostic SHOULD create a unique filename or fail with a clear message.
- **FR-OUT-004**: Metadata MUST record all output artifact paths, including figure files, audit summaries, and reproduction reports.

Suggested filename pattern:

- `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>.png`
- `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>.pdf`
- `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>_metadata.json`
- `geodt_branch_tree_compare_<YYYY-MM>_<fs>_<branchA>_vs_<branchB>_audit.md`

### Metadata, Provenance, and Failure Requirements

- **FR-031**: System MUST produce a metadata summary containing at minimum:
  - workflow mode;
  - archive discovery input;
  - archive discovery root or explicit archive path/list;
  - discovered archive candidate count;
  - selected archive folder;
  - artifact-provider source run folder, if different from selected archive;
  - artifacts provided by each folder and same-run compatibility evidence, if an artifact-provider source is used;
  - selected run year/month;
  - forecasting scope;
  - selected archive minimum required artifact completeness result under FR-000I through FR-000N;
  - fallback selection reason;
  - all considered fallback candidates and rejection reasons;
  - model family;
  - branch assignment source path;
  - branch assignment source type;
  - evidence validation status for branch assignment source;
  - feature-name source path;
  - rejected feature-name sources and reasons;
  - feature-name count;
  - checkpoint feature count, if available;
  - feature-name compatibility result;
  - evidence validation status for feature-name source;
  - selected branch IDs;
  - checkpoint paths used;
  - checkpoint filename, parsed branch ID, checkpoint classification, branch-ID parsing method, and any mismatch reason;
  - evidence validation status for checkpoint branch-ID parsing;
  - whether each checkpoint is branch-specific or root/global;
  - root/global exclusion decision;
  - evidence validation status for the root/global `dt_rules` boundary;
  - K value;
  - actual plotted depth for each tree;
  - dissimilarity formula name and version;
  - selected pair dissimilarity score;
  - contrast descriptor under FR-028B;
  - top-K split features for each selected branch;
  - threshold/direction supplemental audit fields, if computed;
  - assigned admin/group count for each branch;
  - prediction-row count for each branch, if available;
  - training-sample count for each checkpoint, if available;
  - class-label mapping source, if crisis/non-crisis tendency is displayed;
  - evidence validation status for class-label mapping when used;
  - class output availability and leaf color encoding status;
  - readability check result under FR-016C, FR-016F, FR-024C, FR-024D, and FR-024E;
  - rejected higher-scoring pair summary, if any;
  - tie-break result, including tie-break fields used, if any;
  - failed or skipped candidate branch summary;
  - output figure paths;
  - audit summary paths;
  - reproduction report paths, if any;
  - failure summary path, if applicable;
  - package/checkpoint loading notes when relevant.
- **FR-032**: System MUST include a short note explicitly distinguishing this branch-tree comparison from existing `dt_rules` CSV exports that represent only the root/global tree.
- **FR-033**: System MUST identify the affected workflow as an exploratory interpretability diagnostic, not a production forecast, evaluation artifact, fs0-only artifact, or scenario overlay.
- **FR-034**: Reproducibility provenance MUST be sufficient for a reviewer to rerun the diagnostic on the same archived artifacts and recover the same selected run, selected branches, top-K feature sets, dissimilarity score, and plotted checkpoint paths.
- **FR-035**: System MUST NOT overwrite standard forecast deliverables, production model artifacts, archived checkpoints, partition artifacts, existing `dt_rules` exports, or existing evaluation outputs.
- **FR-036**: System MUST fail clearly when checkpoints, branch assignments, feature-name sources, or other parts of the minimum required artifact set in FR-000I through FR-000N are unavailable rather than silently falling back to root/global rules.
- **FR-036A**: When the diagnostic fails after preflight begins, it MUST produce or display a structured failure summary containing the failure stage, missing or invalid artifacts, evidence mismatch if any, candidate archive decisions, and recommended next action.
- **FR-036B**: If failure occurs before an output directory is available, the failure summary MAY be printed to console only, but the diagnostic MUST still avoid silent failure.

### Key Entities *(include if feature involves data)*

- **GeoDT Monthly Run**: One archived monthly GeoDT result directory, optionally paired with a same-run/source artifact-provider folder when the visual archive lacks required branch artifacts. Key attributes include selected archive path, artifact-provider source path when different, year, month, forecasting scope, model family, same-run compatibility evidence, minimum required artifact availability, fallback selection status, archive discovery input, and available result subdirectories.
- **Root/Global DecisionTree Checkpoint**: The DecisionTree trained at `branch_id=''`. It is saved as the root checkpoint and is not sufficient for this diagnostic unless it is explicitly assigned as a terminal prediction branch by the selected run's dispatch artifact.
- **Branch-Specific DecisionTree Checkpoint**: A local DecisionTree model associated with a spatial branch such as `0`, `1`, `00`, or `01`. It is branch-specific only after preflight confirms branch-ID parsing and eligibility; filename pattern alone is not sufficient.
- **Terminal Branch**: A branch ID assigned to at least one final prediction row/admin group by the selected run's dispatch artifact and backed by a loadable branch-specific checkpoint.
- **Branch Assignment Map**: Dispatch-authority artifact or table mapping admin groups, admin units, or prediction rows to terminal branch IDs. Source precedence is final prediction dispatch artifact such as `space_partitions/X_branch_id.npy`, then `space_partitions/s_branch.pkl`, then reliable same-run/month/scope `correspondence_table_*.csv`, then a documented equivalent run artifact.
- **Feature-Name Source**: Artifact or recoverable run metadata that maps DecisionTree split feature indices to human-readable feature names. Source precedence is selected-run checkpoint/model-bundle feature names, selected-run pipeline feature column file, selected-run training matrix column order, compatible configured feature list, then failure.
- **Admin Group / Admin Unit**: A spatial group or administrative unit routed to a terminal GeoDT branch for prediction.
- **Branch Coverage Counts**: Separate audit fields for assigned admin/group count, prediction-row count, and training-sample count. These counts have different meanings and must not be conflated.
- **Branch Signature**: Compact representation of a branch tree's top-K split structure, including split feature names and optional supplemental threshold/direction details and shallow leaf class tendencies.
- **Dissimilarity Score**: Deterministic top-K split feature Jaccard distance used for pair ranking. Supplemental threshold/direction mismatch fields are audit-only and do not alter the default selected pair.
- **Readability Check**: A gate applied after dissimilarity ranking to ensure both trees can be plotted legibly, at comparable scale, and with at least one non-leaf split within the plotted depth according to FR-016C, FR-016F, FR-024C, FR-024D, and FR-024E.
- **Audit Summary**: Standalone preflight and branch-selection record produced without rendering a final figure. It contains archive discovery, archive completeness, branch eligibility, feature-name source selection, pair scoring, readability decisions, selected pair, fallback decision, and root/global exclusion decision.
- **Reproduction Metadata Input**: Prior metadata summary path used by reproduction mode to recover the selected archive, branch IDs, checkpoint paths, feature-name source, K value, actual plotted depths, and default dissimilarity score without reselecting a different pair.
- **Failure Summary**: Structured diagnostic record produced or displayed when the workflow fails after preflight begins, including failure stage, missing or invalid artifacts, evidence mismatch, archive decisions, and recommended next action.
- **Interpretability Figure**: The final 1x2 subplot comparing two selected branch-specific local DecisionTree models from the same monthly GeoDT run.
- **Metadata Summary**: Reproducibility record containing preflight audit, archive discovery input, fallback candidate decisions, selected run, minimum required artifact completeness, branch assignment source, feature-name source, selected branches, checkpoint paths, checkpoint parsing, branch eligibility decisions, evidence validation status, dissimilarity score, top-K features, branch coverage counts, tie-break result, readability results, output paths, failure summary path if applicable, and root/global distinction notes.

### Pipeline & Data Assumptions *(mandatory for model/pipeline changes)*

- **Workflow**: Exploratory interpretability diagnostic for an archived GeoDT model. This is not a production forecast workflow, not a standard evaluation artifact, not an fs0-only artifact, and not a synthetic scenario overlay. Outputs must not be promoted as standard forecast deliverables.
- **Temporal Scope**: Preferred target run is the archived monthly GeoDT model for `2024-10` with forecasting scope `fs1`. If unavailable or incomplete under FR-000I through FR-000N, the fallback order in FR-003A must be used. The selected run's exact year, month, forecasting scope, fallback selection reason, archive discovery input, discovered archive candidate count, considered fallback candidates, and same-run/source artifact-provider folder when used must be recorded. This diagnostic does not create new predictions and does not change feature-month to target-month mapping.
- **Crisis Label**: The diagnostic may display crisis/non-crisis tendency only when the class-to-label mapping is confirmed from selected-run artifacts, pipeline constants, or another documented source. If the mapping cannot be confirmed, the figure must use neutral class labels or omit tendency labels. The diagnostic must not infer crisis semantics solely from sklearn class index order. Leaf color encoding must follow FR-025A and FR-025B.
- **Spatial Inputs**: Spatial branch assignments must come from the selected run's dispatch-authority artifacts using the source precedence in FR-006B. Terminal branch identity must reflect final prediction dispatch, not merely checkpoint filename presence.
- **Geographic Scope**: The diagnostic inherits geographic scope from the selected archived GeoDT run. No new shapefile join or map-rendering assumption should be introduced. If geographic labels are used, their source must be recorded. Do not assume Nigeria-only development shapefiles represent production Sub-Saharan Africa coverage.
- **Threshold Contract**: This diagnostic visualizes trained DecisionTree rules and does not set or change prediction thresholds. If class tendency is reported using model probabilities or leaf distributions, the class-label mapping source and method for assigning tendency must be stated. No scenario-only threshold should be introduced or reported as a standard forecast threshold.
- **Prediction Partition Maps**: The diagnostic must use the terminal branch assignments from the selected run's prediction dispatch artifacts. It must not substitute a different partition map from another target month, forecasting scope, or scenario launcher.
- **Artifacts**: Generated artifacts include one PNG/PDF figure when figure-generation mode succeeds, one metadata summary, audit summaries, reproduction reports when reproduction mode runs, and structured failure summaries when failures occur after preflight begins. Output directory behavior follows FR-OUT-001 through FR-OUT-004. Required provenance fields must align with FR-031, including workflow mode, archive discovery input, selected archive folder, run year/month, forecasting scope, minimum required artifact completeness, fallback selection reason, fallback candidate decisions, model family `GeoDT`, selected branch IDs, branch assignment source, feature-name source validation, checkpoint parsing method, root/global exclusion decision, K value, actual plotted depths, dissimilarity formula name/version, selected pair score, top-K split features, branch coverage counts, evidence validation status, tie-break result, readability results, skipped/rejected candidate summaries, output paths, and failure summary path if applicable. Generated outputs should remain exploratory and should not be committed as source-controlled model artifacts unless explicitly reviewed.
- **Environment**: The diagnostic should be runnable in the project's existing analysis environment. It should handle archived checkpoints with the package versions required by the original pipeline. If run on Windows or WSL, output paths and text should remain portable. Any console output intended for Windows terminals should avoid encoding-sensitive characters where practical.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A successful figure-generation run produces one 1x2 PNG or PDF figure comparing two eligible terminal branch-specific local DecisionTree models from the same selected GeoDT monthly archive.
- **SC-002**: Both plotted trees are loaded from branch-specific checkpoint paths, not from root/global-only `dt_rules_*.csv` exports.
- **SC-003**: The selected branch pair is the highest-ranked readable pair under the default top-K split-feature Jaccard dissimilarity score and deterministic tie-break rules, and any rejected higher-scoring pair or resolved tie is documented.
- **SC-004**: The metadata summary lists the selected run folder, fallback selection reason, selected branch IDs, default dissimilarity score, top-K features for each branch, assigned admin/group counts, prediction-row counts when available, training-sample counts when available, and checkpoint paths used.
- **SC-005**: The figure caption clearly states the selected year/month/forecasting scope and explains that the two local trees were selected as the highest-ranked readable terminal branch pair within the same GeoDT monthly model.
- **SC-006**: The final figure supports interpretation only according to the recorded default dissimilarity score and contrast descriptor. If contrast is low, the output states that the selected run did not provide a strong branch-rule contrast under the default shallow-signature criterion.
- **SC-007**: If the preferred `2024-10 fs1` run is unavailable or incomplete, the output clearly identifies the fallback run selected under FR-003A and does not imply that the figure came from the unavailable or incomplete preferred run.
- **SC-008**: If fewer than two eligible readable terminal branch trees are available, the diagnostic produces a clear explanatory summary and does not create a misleading comparison figure.
- **SC-009**: The diagnostic does not overwrite or modify production forecast artifacts, archived checkpoints, partition artifacts, existing `dt_rules` exports, or existing evaluation outputs.
- **SC-010**: Reproduction mode can recover the selected archive, branch IDs, checkpoint paths, feature-name source, top-K feature sets, default dissimilarity score, and figure inputs from the metadata summary and archived run artifacts. Bitwise-identical image output is not required.
- **SC-011**: PNG output is at least 300 DPI, or PDF output preserves readable vector text where supported.
- **SC-012**: All plotted split labels are readable at 100% zoom, are not silently truncated, and either use unmodified feature names or documented unique abbreviations.
- **SC-013**: A preflight audit exists and identifies available archives, the minimum required artifact set in FR-000I through FR-000N, checkpoint classifications, branch assignment sources, feature-name sources, candidate terminal branches, and missing or unusable artifacts.
- **SC-014**: Fallback selection is reproducible because metadata records archive discovery input, discovered archive candidate count, all considered fallback candidates, acceptance/rejection decisions, and the final selected-run reason.
- **SC-015**: Branch eligibility decisions are documented for eligible, failed, skipped, and rejected candidates.
- **SC-016**: The metadata explicitly states that root/global `dt_rules_*.csv` exports are insufficient for this diagnostic when they represent only the root/global tree.
- **SC-017**: Preflight classifies the selected archive as complete only if it satisfies the minimum required artifact set in FR-000I through FR-000N, including same-run/source artifact-provider compatibility when required artifacts are not located directly in the visual archive.
- **SC-018**: Audit-only mode produces a complete audit summary without rendering a figure and selects the same branch pair as figure-generation mode for the same inputs.
- **SC-019**: Generated output files are written to the configured or documented diagnostic output location and do not overwrite existing artifacts unless explicitly allowed.
- **SC-020**: Evidence-dependent assumptions used by the diagnostic are recorded as confirmed, rejected, or not applicable in metadata.
- **SC-021**: Required-artifact or evidence-mismatch failures produce a structured diagnostic summary rather than a partial or misleading figure.
- **SC-022**: Archive discovery is deterministic and bounded to the explicit archive path/list, configured discovery root, or configured project result/archive root.
- **SC-023**: Feature-name source validation is documented, including selected and rejected sources, feature-name count, checkpoint feature count when available, and compatibility result.

## Assumptions

- Branch-specific DecisionTree checkpoints may be saved using a naming convention that encodes branch IDs, such as `dt_`, `dt_0`, `dt_1`, `dt_00`, or equivalent, but each path and parsed branch ID must still be discovered, configured, or confirmed by preflight.
- The preferred display depth uses `K = 3` by default, with actual plotted depth recorded separately for each selected tree when a tree is shallower.
- Top-K split feature Jaccard distance is the deterministic default branch dissimilarity criterion for the interpretability figure.
- Threshold/direction mismatch fields may be useful audit metadata but do not affect default branch-pair selection.
- If class distributions are available but class-to-label mapping is not confirmed, neutral class labels are sufficient and safer than inferred crisis/non-crisis wording.
- The diagnostic is intended for paper/report interpretation and should prioritize readability, provenance, reproducibility, and explicit contrast scoring over full-tree exhaustiveness.
- The intended interpretation is supported only to the extent that the selected pair's recorded score and plotted rules show meaningful contrast. If contrast is low, the diagnostic should report that the selected run did not provide a strong branch-rule contrast under the default shallow-signature criterion.

## Brownfield Migration Notes

- This feature preserves the migrated brownfield finding that existing `dt_rules_*.csv` artifacts are root/global exports when loaded with `branch_id=''`, subject to evidence validation before use.
- This feature does not expand or rename existing `dt_rules` behavior. It specifies a separate branch-specific interpretability diagnostic that should load branch-specific checkpoints directly.
- Archive characterization has been promoted from a migration note to formal preflight requirements for this feature.

### Evidence Validation Requirements

- **FR-EVID-001**: Before relying on a brownfield artifact convention, the diagnostic or implementation plan MUST validate the convention against repository evidence, selected archive contents, or preflight results.
- **FR-EVID-002**: The following assumptions MUST be treated as evidence-dependent and MUST be confirmed before use:
  1. existing `dt_rules_*.csv` exports represent root/global rules when generated from `branch_id=''`;
  2. `space_partitions/X_branch_id.npy`, when present, is a final prediction dispatch artifact and has higher precedence than `s_branch.pkl`;
  3. `space_partitions/s_branch.pkl` maps admin groups or prediction rows to terminal branch IDs for the selected run;
  4. `correspondence_table_*.csv` contains terminal branch IDs for the same run/month/scope before it is used as an assignment source;
  5. checkpoint filenames such as `dt_`, `dt_0`, `dt_1`, `dt_00`, and `dt_01` can be parsed into branch IDs consistently with the selected run's checkpoint loader;
  6. any configured feature-column file corresponds to the selected run/month/scope.
- **FR-EVID-003**: If an evidence-dependent assumption cannot be confirmed, the diagnostic MUST either use a lower-precedence confirmed source, mark the affected candidate unusable, or fail preflight with a clear evidence-mismatch report.
- **FR-EVID-004**: Metadata MUST record evidence validation status for branch assignment source, feature-name source, checkpoint branch-ID parsing, root/global `dt_rules` boundary, and class-label mapping when used.

## References

- `specs/_evidence/005-geodt-branch-tree-interpretability.evidence.md` - Brownfield evidence for branch-specific GeoDT tree diagnostic, including root/global `dt_rules` boundary, branch checkpoint dispatch semantics, archive risks, and open questions.
- `CLAUDE.md` - Current GeoDT workflow, forecasting scope rules, artifact boundaries, and operational constraints.
- `AGENTS.md` - Repository structure, batch workflow commands, testing guidance, and Windows CMD-safe batch conventions.
- `.specify/memory/constitution.md` - Pipeline contract, temporal/spatial integrity, validation evidence, and artifact hygiene rules.

These references are evidence sources to be validated during implementation/preflight, not substitutes for selected-archive validation.
