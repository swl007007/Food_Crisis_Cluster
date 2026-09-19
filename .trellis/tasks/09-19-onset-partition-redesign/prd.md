# Onset-objective partition redesign and 2-layer optimization

Created 2026-09-19. **Status: planning suspended pending a leadership decision** on whether the
FEWS NET suspension window becomes the main line. See
`docs/notes/2026-09-19_suspension-window-briefing.md`.

## Why this task is paused

A design grill on 2026-09-19 settled 18 decisions (below). Partway through, the reason for the
missing 2025-02 / 2025-06 labels was identified: **USAID suspended FEWS NET**, and the team regards
that suspension as a research subject in its own right. The proposed redirection is a pair of
parallel models — a cheap operational model for the normal period and a suspension fallback — argued
on flexibility and cost rather than accuracy.

That redirection would change the test window, the horizon definition, and the entire success
criterion, so the remaining planning is held until leadership decides. **The decisions below are
recorded so the grill is not lost**; those marked *direction-dependent* would need revisiting.

## Decisions that hold regardless of the direction

These follow from defects in the existing pipeline and are valid under either framing.

- **D1 — SMOTE off in both arms.** `train_partitioned_model` applies SMOTE
  (`compare_partitioned_vs_pooled_rf_k40_nc4.py:325`) while `train_pooled_model` does not, so
  "partitioned > pooled" conflates spatial partitioning with oversampling. Turning it off in both
  arms is also the clean ablation. SMOTE is additionally unsuitable here because the second layer
  thresholds a probability and SMOTE changes the training prior.
- **D2 — The partition criterion becomes real class-1 F1 on the `persist=0` subset.** Today it is
  class-1 **recall**: `partition_opt.py:72-82` returns a true-positive indicator, so false positives
  never enter the split criterion, and the real F1 helper at `:40` is dead code. With override flip
  precision already at 0.37-0.44, a criterion that ignores false positives selects partitions in
  exactly the wrong direction.
- **D3 — Stage 1 and Stage 2 use one objective.** They currently disagree: Stage 1 optimises recall,
  Stage 2's consensus similarity uses real F1 improvement
  (`step4_similarity_matrix.py:57-59`). **Dependency:** Stage 2 reads `f1(1)` / `f1_base(1)` from
  Stage 1's `results_df`, so Stage 1 must emit `persist=0` F1 columns before Stage 2 can change.
- **D4 — Implement as a new metric branch, default unchanged.** Add a metric to
  `get_metric_score_array` selected via `GOVERNING_METRIC`; the existing `class_1_f1` path stays
  byte-identical, pinned by a regression test, so the frozen paper artifacts remain reproducible.
- **D5 — The second-layer model trains on the `persist=0` subset only.** The full-sample model is
  dominated by "is this place normally in crisis" and largely relearns persistence; the override's
  actual decision problem is discrimination *within* `persist=0`.
- **D6 — The pooled comparison arm is identical except for partitioning:** same `persist=0` training
  subset, no SMOTE, same features and parameters.
- **D7 — Raw probability is the primary input; calibrated is a pre-registered secondary.** Measured
  in the prior task: the calibrated ceiling (+0.0054 / +0.0140) was *below* the raw ceiling
  (+0.0095 / +0.0331), so per-`(month, partition)` calibration cost the mechanism ceiling.
- **D8 — Up-only is structural.** A `persist=0`-only model can express nothing but `0 -> 1`.
- **D9 — Extend `PersistenceCorrectionExperiment/`**, reusing its verified persistence join,
  structural up-only override, and protected-hash gate.
- **D10 — Implementation and verification must be done by different parties**, and the verifier may
  not import the implementer's modules. This replaces the prior task's decision 26, which broke when
  a rate limit forced the main session to both implement and report (C10 there).
- **D11 — Pre-register a rule, not a value, for branch sample size:** a minimum count of class-1
  positives per branch within `persist=0`, derived from the data structure before any result is
  seen. `SIGLVL` / `ES_THRD` unchanged initially.
- **D12 — Stage 1 month inclusion is governed by a pre-registered lag-coverage floor.** A target
  month's 36-month training window must have at least a stated fraction of its label months carrying
  an observed `T-H`. Measured coverage: fs1 reaches 100% at 2019-02, fs2 at 2019-10; 2016-02 is 0%.
  Equal weighting above the floor, hard exclusion below it — no weighting function, to avoid adding
  a free parameter.

## Direction-dependent decisions (revisit after the leadership decision)

- *D13* — Adjudication on all rows, comparable to persistence and expert. **Still sound**, but the
  expert does not exist in the suspension window, so the comparator set changes there.
- *D14* — Tiered bar: +0.02 pass, scope-specific target equal to the gap to the expert. **Numbers are
  support-dependent**: 2021-2024 gaps are +0.031 / +0.054; 2024-only they are +0.047 / +0.083; in the
  gap there is no expert at all.
- *D15* — fs1 pilot first, fs1 must pass, fs2 confirms. **The fs1/fs2 framing itself does not apply
  in the gap**, where forecasting 2025-10 from the last publication is a 12-month lead.
- *D16* — Windows: partition 2018-2022, threshold 2023, test 2024. Superseded if the test window
  moves to the suspension period.
- *D17* — Fold-level bootstrap plus leave-one-fold-out. **Unusable at 3 test folds** (10 distinct
  multisets); an admin-level cluster bootstrap keeping each admin's months together is the
  replacement, with per-month LOFO demoted to a descriptive check.
- *D18* — Two hard stop points (no partitions produced, or fs1 failing the +0.02 pass line) triggering
  a switch to IPCCH. Carries over in spirit; the trigger conditions need restating once the direction
  is fixed.

## Acceptance criteria

To be written once the direction is decided. Any successor **must carry a fresh pre-registration**:
the prior task's R19-R23 were consumed, and reusing them on the same data would void their meaning.
