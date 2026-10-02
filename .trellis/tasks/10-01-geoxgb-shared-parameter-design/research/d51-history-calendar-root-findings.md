# D51: history + calendar (78-feature) root ablation (2026-10-02)

**Scope:** contract `d51-history-calendar-root-plan.md`, planning commit `99ac473`. The primary crisis-F1 endpoint and the final criterion are unchanged.
- **Change:** each of the 21 D34 roots is re-fitted on the 75 schema `history_blocks` features plus the 3 `known_calendar` features (including `target_year`), 78 columns in global schema order. Removed jointly: static 28 + dynamic 41 + legacy 15 = 84.
- **Unchanged:** all original FIT rows, order and labels (including missing-origin rows), W59, the four-class objective, G (H4 G1 / H8 G4 / H12 G2), rounds 200 / 400 / 400, seed 42, colsample and other parameters; unweighted; no margins or local trees.
- **Arms:** the saved full-162 original root, the fresh 78-feature root (h78), and exact-origin persistence.
- **What it is not:** a joint removal that mixes information and feature-search effects (frozen colsample on fewer columns), so no variable, block or causal attribution.

## Producer, tests and run (executor factual record, `d58b242`)

- **Producer** `edffb0d`: task-research runner `research/d51_history_calendar_root.py` (636 lines), git blob `5bd89e384dff16a35b783163ba8d39989769ba28`, sha256 `dd10560c4fbdbc24a0bf0d31f917a51375b2243a083e07f1ab086562747e18ab`. No package or schema edits.
- **Native implement** (about 3.1 min) and **native check** (about 1.4 min, no result-affecting finding, no edits). Synthetic-only selftest OK at HEAD; no package tests.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d51-history-calendar-root-20261002`, frozen Windows Python 3.12.10 with assertions on (`sys.flags.optimize = 0` recorded by a launch of the same interpreter immediately before), exit 0 in 166 s. Log: `research/d51-run.log`.
  - **Fits:** exactly 21, on 1,339,197 FIT rows in total; `num_feature` 78; rounds 200 / 400 / 400; `base_score` 5E-1; all UBJ hashes differ from the originals; 0 reload mismatches.
  - **Gates:** all 21 full-162 original replays passed first; projection, fit and D49/D50 original-arm consistency gates (inventory, matched n/excluded, confusions, original D50 cells) passed; `dev_baselines` 15 checked / 6 not covered, as in D47/D48.

## Verification (supervisor, distinct from the executor record)

`research/d51_supervisor_check.py` → `research/d51_supervisor_results.json`: **PASS**.
- 5,738 checks, 360 metric cells, 1,660,244 FIT/C/E3 rows; no producer imports and no fits.
- Independent full-snapshot keys, order and labels, with all original FIT rows retained; the 78 columns selected by schema names (not producer indices); exact raw replay of both UBJs; params unchanged except dimensionality; base score, classes and rounds.
- All-key and matched metrics, exact paired F1 deltas and fold wins, and independent pair-count within-phase AUC and means.

These are numerical checks, not statistical tests.

## Results (E3, matched exact-origin keys; excluded 142 / 284 / 287 per H)

| H | Pooled crisis F1: original / h78 / persistence | Mean-fold F1 | h78 > original | h78 > persistence | Crisis-call share, original → h78 |
|---|---|---|---|---|---|
| 4 | .629050 / .633073 / .651697 | .626608 / .629004 / .655255 | 2/7 | 1/7 (1 tie) | .1499 → .1744 |
| 8 | .533375 / .557674 / .555614 | .534031 / .545162 / .555568 | 3/7 | 4/7 | .1567 → .2194 |
| 12 | .478355 / .512267 / .549808 | .479043 / .492742 / .547711 | 3/7 | 2/7 | .0867 → .1874 |

- **Probability quality, original → h78 (pooled):** crisis Brier .090807 → .092897, .110801 → .116120, .104087 → .113380; log loss .615151 → .626246, .715178 → .744998, .733590 → .787869. Worse at every H.
- **Mean-fold AP:** lower for h78 at every H (.7328 → .7175, .6503 → .6117, .6445 → .6052). AUC lower at H4/H8, slightly higher at H12 (.8583 → .8601).
- **All-key crisis F1, original / h78:** .628357 / .632423, .532228 / .556649, .477156 / .511301.
- **FIT and C pooled F1** fall at every H (FIT .7258 / .7658 / .6093 → .6768 / .6641 / .5103; C .7156 / .7277 / .5891 → .6652 / .6398 / .4893).

**D50 within-phase E3 AUC (valid-fold means, original / h78):**

| H | Phase 1 | Phase 2 | Phase 3 | Phase 4-or-5 (4/7 valid) |
|---|---|---|---|---|
| 4 | .773484 / .738919 | .655702 / .631413 | .736468 / .705903 | .803153 / .830429 |
| 8 | .824208 / .794334 | .598012 / .616874 | .745844 / .704122 | .938756 / .842284 |
| 12 | .818068 / .801824 | .605804 / .607957 | .728402 / .700038 | .854736 / .768576 |

- **H8 phase 2 per date, original → h78:** 2018-06 .469723 → .510904; 2018-10 .414721 → .410785; 2019-02 .718863 → .626516; 2019-06 .655053 → .720178; 2019-10 .458898 → .646443; 2020-02 .837987 → .792196; 2020-06 .630837 → .611093. Above .5 on 4/7 → 6/7 dates, but only 3/7 dates improve.
- Phase 4-or-5 cells have negatives of only 1–38 and are not robust.

## Supervisor synthesis and decision

**D51 is complete and the 78-feature root is not adopted as a default.** It is preserved as a completed diagnostic only; no automatic feature-subset sequence follows.
- At H8, h78 is above persistence pooled (+.002060), but its mean-fold F1 .545162 is below persistence's .555568, and it beats the original on only 3/7 folds (persistence on 4/7).
- At every H, h78's Brier, log loss and AP are worse and its phase 3 AUC is lower. FIT and C F1 are lower, so a smaller historical gap is not proof that overfitting is resolved.
- The H8 phase 2 mean improves (.598012 → .616874; above .5 on 4/7 → 6/7 dates), yet only 3/7 individual dates improve (including 2018-06 .469723 → .510904 and 2019-10 .458898 → .646443), and 2018-10 stays at .410785. Recovered dates are not cherry-picked.
- The aggregate F1 increase comes with a higher crisis-call share, so it cannot be attributed only to the operating point or to causal noise removal.
- The joint 84-feature removal mixes information and feature-search effects; no variable-level causal attribution. Neither "the covariates are all noise" nor "history-only solves it" follows.
- Stage 1 remains unresolved; no D52.

## Evidence

**Task research holds exact byte copies** (line endings preserved):

| File | Bytes | sha256 |
|---|---|---|
| `d51_summary.json` | 701,621 | `fffa57844526518bb11b8dfe6797a1dbd0506cd3bbb74bff5702e979c2dd0f59` |
| `d51_identity.json` | 42,771 | `739dd0dcf40dca2cd5b9c5353e73e00bfbd9e500441ca8c6a8dd64160fb76639` |
| `d51_feature_map.json` | 11,359 | `43c935e8b4199e1c065c92b903fe709b5ceedbdb7e56cbe75618beda85c60eec` |
| `d51_gate.json` | 60,997 | `445911c7ef99209ca1c3e2af8b56f253c43cad9df48bc273323f4ed5898d1730` |
| `d51_completion.json` | 12,890 | `7283b9b811c4ddcfc0248cd41a2a6cde1f8160c3888126504528ea15c9e1ed1f` |
| `d51_supervisor_check.py` | 13,678 | `6fcb41b6fd7f38aac9a3dc257edd3fb37ec6aa06885ab2ae00e8c5369fc2a177` |
| `d51_supervisor_results.json` | 365,673 | `5331e259a94f6994dd7f5a8d2446604629ecad0a3e8b847cad9c490dcf2760cd` |
| `d51-run.log` | 18,895 | `bdf0b08c5177bbc7829d6d6f16a1c864e2186201f586b51ce2ee506dc1b96058` |

**Kept external** in the run directory (about 133 MB): the 21 `h78_root.ubj` files and `h78_root.json` records and the keyed `rows_{FIT,C,E3}.csv.gz`; their hashes are recorded in `d51_completion.json`.
