# D33 reconstruction source equivalence (ab1ac83 vs replay code), 2026-10-01

**Purpose:** proportionate provenance for the predict-only D33 replay. The replay script imports rebuild helpers from the current package. This note shows that every rebuild dependency is source-equivalent to the D29 producer ab1ac83. The mandatory real-data full-tree gate (exact round-trip equality) remains the empirical check. No runtime integrity framework.

**Byte-identical files** (`git diff --quiet ab1ac83 HEAD`):
- `config.py` (TRAIN_WINDOW_MONTHS, GROUP_SPLIT)
- `feature-schema.json`
- `src/customize/customize.py` (`train_test_split_rolling_window`, incl. the target-month area restriction)
- `src/feature/fourclass_features.py` (`load_schema`, `month_label`)
- `src/helper/helper.py` (`get_X_branch_id_by_group`)
- `src/metrics/fourclass.py`
- `src/model/native_xgb.py` (`from_raw`, `proba`, `keys_sha`)
- `src/utils/run_identity.py`

**Purely additive files** (0 deleted or modified lines):
- `src/experiment/plan.py`: only RECENTSEARCH* and MATCHED* names were added; none is used by the replay.
- `src/utils/split.py`: only `recent_search_months` and `matched_size_sample` were added. `confirmation_split`, `group_aware_train_val_split` and `time_block_split` are AST-identical.

**Changed file:** `app/main_model_GF.py`. The only rebuild dependency in it, `stage1_split`, is AST-identical. It reads only the unchanged config, split and plan names above. The changed `main`/`run_candidate`/import lines are not imported by the replay. `assignment_evidence` (new in D32) feeds metadata only.

**Glue mapped to ab1ac83 `main()`** (native trellis-check):

| Step | ab1ac83 `main()` lines |
|---|---|
| Snapshot load and horizon/feature-order check | 335–339 |
| Sort and arrays | 340–347 |
| Rolling-window split | 349–352 |
| Origin index | 353–354 |
| r80 split, seed 42 | 364 |
| Confirmation split, seed 42 | 375–377 |
| keep = ~C | 459 |
| E3 rows | 352 |

All steps match. `main()`'s check-only guards are represented by an empty-target and `[O-59, O)` window assertion.

**Empirical gate:** a predict-only trial on the real D29 data passed every full-tree check with 0 mismatches (keys, truth, hard predictions, routes, C/E3 root and full probabilities, booster SHAs, fitting keys). The trial output was deleted; the authoritative run follows the code commit.
