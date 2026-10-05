# Configuration selection: source boundaries and accepted R44

Planning only, 2026-10-04. R43 fixes eight G/L recipe pairs per H in v0.37.
The user accepted the selection rule below as R44 in v0.38. No fitting or scoring ran.

## Established boundaries and source facts

- R35 already fixes the 2014-2022 pooled, within-area chronological half F/S
  split. S is reused for spatial scan/split decisions and is internal development
  evidence, not independent forward prediction performance. R34 does not create
  a separate C/E3 or consensus-map ensemble. R42 fixes one G/L pair per H.
- `FEWSNETGeoXGBExperiment/src/experiment/plan.py:220-223` has an exact G tie key
  of rounds, depth, then ID. That only defines a source global-model tie policy;
  it does not decide how the new eight joint G/L maps should be selected.
- `IPCCHPopulationHistoryExperiment/run_pipeline.py:1235-1245,1281-1291,1305-1350`
  instead selects each pooled arm/H from saved 2020-2022 development predictions,
  using a history-only cohort, threshold-pair search and JSON-order ties. Those
  cohort/threshold rules were explicitly excluded by R21; its selection calendar
  is not automatically inherited. The existing code has no spatial quartet map.

## Accepted first-version rule

1. For each H, run all eight G/L recipes against exactly the same frozen R35 F/S
   original-key memberships and feature contract, with the same R45 scan /
   recursion budget. Fit global and eligible local quartets on F only. S is used
   for the adopted scan, smoothing, support checks and complete-routing F1 gates.
   No recipe changes the cohort or gets extra seeds, early stopping or retuning.
2. At the end of each candidate's Stage1 search, predict the entire common S key
   set through that candidate's accepted terminal routing. Parent/global fallback
   rows remain present. Apply the same bounded isotonic projection and >=.20
   decoder; pool TP/FP/FN once across S, with original keys equally weighted.
3. Rank by the highest absolute crisis F1 on S, not the largest improvement over
   that recipe's own global baseline. Do not average country/month scores, pool
   different H, or use q3 R2/other headline metrics as undeclared tie criteria.
4. For exactly equal F1, prefer fewer terminal learned regions; then fewer nominal
   G+L boosting rounds; then lower global depth; then lower local depth; finally
   fixed (G ID, L ID) order. Compare exact confusion-count fractions rather than
   rounded displayed scores. The terminal root counts as one region; outside-map
   global-fallback coverage is not counted as an extra learned region. This is an
   engineering preference for a simpler equal-scoring result, not a significance
   or theoretical complexity claim.
5. A candidate with no accepted split remains eligible with its global quartet's
   S score. Preserve all eight ledger entries. F1=NA is retained with its reason,
   never filled with zero or declared the winner; if every candidate is NA, report
   selection_unavailable and do not freeze a winner. Any candidate's technical
   error follows R41: an incomplete candidate set cannot be silently reduced to
   the successful recipes and called a completed selection.
6. Freeze the winning candidate's map and G/L recipe directly for that H. Do not
   average maps or refit a different Stage1 map using F+S after ranking. Stage3
   global/local boosters still refit on their lawful rolling windows, using the
   frozen winning map/configuration and the already accepted historical gate.
   The matched pooled control uses that fold's same G_H global quartet, so the
   no-partition path remains identical to that control under R36.
7. Save the eight recipes, F/S memberships, keyed raw/projected quartet predictions,
   labels/confusion counts, accepted maps/routes and complete ranking/tie reasons.
   Ranking must be reproducible from saved predictions without fresh training.

## Interpretation and trade-off

This accepted rule deliberately reuses the adopted S development set for both
map search and recipe selection. Selection scores are optimistic internal scores;
they are not independently validated forecasts and do not directly rank the full
Stage3 historical-adoption/36-month-refit policy. All maps/configuration choices
remain within the 2022-12 information boundary; primary performance is measured
by the frozen 2023-2025 rolling evaluation and 2026 is separate.

An extra development rolling-selection layer would align selection more closely
with Stage3 but add many global/local historical fits and another map-availability
contract. Replaying 2020 origins with a map learned through 2022 cannot be called
an independent forward test of that map. The accepted first version does not
add that layer or silently claim it has been performed.

R44 fixes the selection cohort, score and tie order. R45 subsequently fixes
membership depth 4, one scan per parent and 1000 iterations; R46 fixes candidate
size / score ties; R47 fixes the rolling prediction schedule and observed-target
coverage. R48 fixes model reuse and conservative total fit accounting in
fit-reuse-budget.md; this document does not authorize training.
