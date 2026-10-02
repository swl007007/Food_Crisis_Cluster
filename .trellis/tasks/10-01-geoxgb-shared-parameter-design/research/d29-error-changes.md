# D29 root → full-tree error changes (zero fit, 2026-10-01)

**Scope:** six frozen D29 candidates (producer ab1ac83, run `geoxgb-d29-confirm-20261001`). Saved hard predictions only: S `validation_predictions.csv.gz`, C `confirmation_predictions.csv.gz`, E3 `target_predictions.csv`. Binary crisis = four-class code ≥ 2. No fits, no Stage 2/3, no final data.

**Reproduction:** `python3 research/d29_error_changes.py <D29 stage1_rootconf dir> research/d29_error_changes.csv` (stdlib only). It reproduces the supervisor's `/tmp/stage1_d29_error_changes.json` with 504 fields compared and 0 mismatches. The CSV adds a terminal-depth grouping.

**E3, pooled over the six targets × H** (repeated targets, so descriptive and not independent evidence):

| Group | Rows | Corrected | Spoiled | New TP | Lost TP | New FP | Removed FP |
|---|---|---|---|---|---|---|---|
| All | 32373 | 116 | 134 | 16 | 34 | 100 | 100 |
| 0 S rows in area (root, no search) | 1070 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 S row | 9016 | 33 | 35 | 3 | 11 | 24 | 30 |
| 2 S rows | 22287 | 83 | 99 | 13 | 23 | 76 | 70 |
| Terminal depth 1 | 2175 | 5 | 2 | 2 | 0 | 2 | 3 |
| Terminal depth 2 | 8761 | 24 | 40 | 0 | 9 | 31 | 24 |
| Terminal depth 3 | 10453 | 40 | 35 | 4 | 11 | 24 | 36 |
| Terminal depth 4 | 7315 | 39 | 39 | 9 | 11 | 28 | 30 |
| Terminal depth 5 | 2599 | 8 | 18 | 1 | 3 | 15 | 7 |

**Reading:**
- Corrected/spoiled counts are accuracy flips, not F1.
- The full tree loses more crisis TP than it gains (16 vs 34), while FP changes cancel out (100 vs 100).
- No-search areas never flip. Losses also occur in areas with 2 S rows, not only in singleton areas.
- Harm by final depth is non-monotone.
- Grouping by final depth describes where flips sit in the frozen tree. It is **not** the same-row counterfactual of stopping earlier; that needs the D33 truncation replay.
