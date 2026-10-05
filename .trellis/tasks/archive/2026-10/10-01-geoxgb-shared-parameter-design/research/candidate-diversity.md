# Stage 1 candidate quantity and diversity

2026-10-01, planning only. Existing committed reference: `R=FEWSNETFourClassBaseline/runs/fourclass-v7-20260928`. No model fitting or consensus reconstruction was run.

## Observed candidate pool

The saved ledger schedules 108 monthly folds across fs1–fs3; 81 have no target labels and are skipped. Of the 27 completed candidates, 19 have one terminal route and 8 have more than one; only **3 have positive E4 performance weight**. Sources: `R/stage2/consensus.json:2-7`, `R/stage2/plan_weights.csv:2-28`. The latter counts are over completed candidates, not all scheduled folds.

| Scope | Target year | Completed | Single terminal | Multiple terminals | Positive weight |
|---|---:|---:|---:|---:|---:|
| fs1 | 2018 | 3 | 3 | 0 | 0 |
| fs1 | 2019 | 3 | 3 | 0 | 0 |
| fs1 | 2020 | 3 | 1 | 2 | 1 |
| fs2 | 2018 | 3 | 2 | 1 | 1 |
| fs2 | 2019 | 3 | 2 | 1 | 0 |
| fs2 | 2020 | 3 | 0 | 3 | 0 |
| fs3 | 2018 | 3 | 3 | 0 | 0 |
| fs3 | 2019 | 3 | 3 | 0 | 0 |
| fs3 | 2020 | 3 | 2 | 1 | 1 |

The positive candidates are `fs1_2020-06`, `fs2_2018-10`, and `fs3_2020-10`, with weights 0.02258639987290484, 0.0128216406993229, and 0.03793511274630068 (`plan_weights.csv:9,13,28`). Their normalized shares are **30.7955%, 17.4817%, 51.7228%**. Top three account for 100% of the performance weight. The concentration summary `(sum w)^2 / sum(w^2)` is 2.5450; this is not an independent or effective statistical sample size.

## Duplicate checks and their limits

Saved correspondence labels were read as strings to preserve leading zeros, then canonically renamed in sorted area order. The first two positive maps cover the same 5,365 areas and are not exact duplicates (3 versus 7 labels). The third covers 5,286 areas and has 8 labels; it omits area IDs 0–78 found in the other two. Of three possible pairs, only one has identical coverage; that pair has no exact duplicate. The other two duplicate comparisons are undefined, not evidence of structural diversity. No pairwise ARI was computed.

The full consensus reports 5,506 areas and reads all 27 plans (`consensus.json:13,21-26`); the union of positive maps covers 5,365. Zero-weight candidates can still participate in area-inventory construction. This count is not authorization to delete zero-weight or duplicate candidates, or change the existing E4 weighting algorithm.

## Historical budget discussion and current decision

Historical brainstorm below: the five-seed/270-candidate suggestion was never approved. D24 now accepts `../experiment-plan.md`: three seeds and D23's two threshold families, 324 candidates for one cross-H local vector and 648 across both local choices. Leave-one-seed-out runs are outside the accepted first round. The count evidence above is unchanged; design acceptance does not authorize training.

- D20 fixes two split proportions, not the total number of candidate maps. One seed per proportion yields `3 horizons ×9 labelled targets ×2 proportions =54` candidates for a fixed set of per-H configurations.
- A proposed five fixed split seeds gives **270 labelled-target candidates**. This is a proposal, not an accepted seed budget or a claim that 270 maps are independent. The equivalent full monthly schedule has 1,080 entries, 810 empty-target skips under the frozen observed calendar; skipped entries do not fit models or create labels.
- More random splits do not guarantee more useful geographic partitions. Record split/root-only counts, positive-weight counts by H/year/proportion/seed, exact partition signatures with coverage identity, and weight concentration. Do not project the old positive-weight percentage onto XGB as an expected result.
- These diagnostics must use development evidence only. Do not select seed lists, expand trials, lower D12, or change E4 because final Stage 3 scores disappoint. No universal minimum useful-candidate count is established by the observed run.
- A development-only leave-one-seed-out consensus check is a possible stability diagnostic for the bounded plan; it would reuse fitted candidate artifacts and keep the full-pool map as the main map. It is not run or approved here and would not turn correlated candidates into independent evidence.
- D21 clarifies the objective: agreement can reflect stable structure and is not itself a failure. Diversity is not to be maximized independently of development forecasting performance. No minimum number of distinct partitions, artificial disagreement target, or confidence guarantee follows from these counts; the five-seed proposal remains unapproved.

## Reproduction

From repository root:

```bash
python3 .trellis/tasks/10-01-geoxgb-shared-parameter-design/research/count_candidate_diversity_v7.py FEWSNETFourClassBaseline/runs/fourclass-v7-20260928 > /tmp/geoxgb_v7_candidate_counts.json
```

`candidate-diversity-v7.json` preserves the counts, exact coverage differences, canonical partition signatures, and hashes of the three Stage 2 sources plus three positive correspondence maps. The small script is specific to this nonempty v7 positive-weight pool; it is not the proposed production diagnostics implementation. Main-session verification read all 27 plan-weight rows, the consensus summary, and the canonicalization script.
