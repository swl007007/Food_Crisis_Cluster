# Step 3 partitioned expert selective correction

## Goal and status

Design the main GeoRF Step 3 comparison as partitioned RF selective correction of the original FEWS NET expert estimate versus the unchanged pooled RF. Design was confirmed during grilling on 2026-09-18. The user subsequently authorized implementation and experimental execution: rebuild only Phase 3, reuse Phase 2 partitions, and save all new experiment files/results in a new directory under the repository root. Paper-artifact replacement and commits are not authorized.

## Background

The main Stage 3 evaluates fixed Stage 1/2 partitions on 2021–2024. The ETH selective-correction experiment supplies the correction mechanism, not its expert series or model family. Exact current-code anchors and limitations are in `research.md`.

## Confirmed requirements

- R1 — Expert definition **(revised 2026-09-18 by explicit user decision; supersedes the original record-shift mandate)**: the expert estimate for target month T is the **calendar-aligned** FEWS NET projection published at origin `O = T-H` — `fews_proj_near` at `T-4` for fs1 and `fews_proj_med` at `T-8` for fs2 — because a near projection published in month D targets `D+4` and a medium projection targets `D+8`. Retain the original binary Phase 3+ conversion, per-admin ordering, and raw-missing-phase → 0 convention. The legacy `shift(4)`/`shift(8)` **record** shifts are a defect: the source is tri-annual (Feb/Jun/Oct) from 2016-02, so they resolve to 12–16 (fs1) and 24–32 (fs2) calendar months, with zero rows at the declared origin. Verified 2026-09-18: calendar alignment reaches 100% coverage on 2021–2024 targets and raises crisis F1 from 0.6239→0.8070 (fs1) and 0.5246→0.7630 (fs2); `record shift(1)`/`shift(2)` reproduce the calendar figures exactly, confirming 1 record = 4 calendar months in the tri-annual era. Keep the legacy reconstruction **only** as a data-pipeline validation check against archived baselines, never as the experiment's expert input. Do not import the ETH expert mapping or its fs3 anchor.
- R2 — Architecture: expert binary judgment is the starting prediction. Each partition's RF learns `expert_wrong = (expert != truth)` from the original main-model features plus the expert judgment. Flip only when a validation-approved rule allows it. This is expert plus one learned correction layer, not two learned residual layers or probability calibration.
- R3 — Comparators: retain only the original pooled as the reported comparator. No pooled correction model and no added expert-only leaderboard. Expert-only remains the internal no-correction candidate and audit reference. Describe gains as differences between complete methods, not an isolated partition effect.
- R4 — Scopes: enable correction for fs1/fs2 (4/8 months). fs3 retains the original partitioned-versus-pooled computation and is explicitly marked uncorrected. No fs3 proxy expert; fs0 and GeoDT are outside this task.
- R5 — Learner: preserve main RF parameters. No SMOTE, class weights, or new hyperparameter search for correction RF. Preserve original pooled and fs3 training behavior.
- R6 — Abstention: partitions with fewer than 50 usable training rows or a single wrong-label class retain expert predictions. Do not use the crisis-predicting pooled model as a wrong-score fallback. Apply the same rule in validation and final fitting.
- R7 — Selection: one shared wrong-score threshold per outer fold, with separately enabled 0→1 and 1→0 directions. Each enabled direction needs at least 20 proposed flips over at least two validation months and correction precision >= 0.75. After direction filtering, validation crisis F1 must strictly exceed expert-only. Otherwise make no corrections; expert-only wins ties. Freeze selection for test.
- R8 — Time: retain the exact original outer window, including its historical off-by-one behavior: configured 36 months means `[T-H-35 months, T-H)` (35 monthly timestamps). Reserve the last **twelve** calendar months for validation — `V=[O-12 months, O)`, **revised 2026-09-18 by explicit user decision**: source label months are tri-annual and exactly 4 months apart while `O ≡ 2 (mod 4)`, so a six-calendar-month `V` contains exactly one observed label month in 24/24 folds, making the approved `distinct months >= 2` gate structurally unsatisfiable. Twelve calendar months yield exactly three observed label months per fold with every approved gate left unchanged. Isolate fitting labels by forecast horizon at the validation boundary, then refit on the complete eligible outer window. Insufficient validation yields no correction.
- R9 — Preserve the original pooled predictions, fs3 results, Stage 1/2 partitions, evaluation schedule, and main feature definitions. New results must use distinct paths and explicit method labels; do not overwrite or relabel frozen paper artifacts.

## Acceptance criteria

- AC1 (R1): two separate gates. (a) **Pipeline validation** — reconstruct the legacy record-shift baselines and match archived fs1/fs2 metrics on their original support at 1e-12 tolerance (verified 2026-09-18: 39/39 quarters each). (b) **Experiment expert** — assert the calendar-aligned series has an exact H-month publication lag on every required row, report coverage, and halt on any row whose source date is not exactly `T-H`. Retain per-row source provenance for both and verify truth agreement on main evaluation keys.
- AC2 (R2, R5, R6): show that RF targets are expert errors, inputs include the expert judgment, no correction resampling/weighting occurs, and untrainable partitions retain expert exactly.
- AC3 (R7): independently recompute selected thresholds, directional counts, distinct months, correction precision, strict F1 improvement, and final flips from audit outputs. Test labels never affect selection.
- AC4 (R8): assert exact outer endpoints, twelve-calendar-month validation bounds (three observed label months per fold), and horizon isolation for all fitting labels. No in-sample validation predictions or test-driven fallback.
- AC5 (R3, R4, R9): unchanged pooled predictions and unchanged fs3 labels/metrics on identical admin-month support; fs1/fs2 use the distinct correction method name. No added pooled-correction comparator.
- AC6 (R9): protected input/output hashes remain unchanged. Recompute monthly confusion counts and main summary metrics from saved predictions.

## Out of scope

ETH reruns or migration; new expert definitions; XGBoost; residual stacking; extra baselines; changing Stage 1/2, the historical window, pooled thresholds, or fs3; automatic publication or paper-artifact promotion.

## Review and deferred verification

No unresolved user design choices. Exact raw-data lineage/availability, source hashes, archived numerical reproduction, and runnable checks are implementation acceptance gates, not claims of verification in this design session. `design.md` defines conservative failure behavior for contract violations. Implementation and experiment execution are now authorized by the subsequent user instruction; historical outputs remain protected.
