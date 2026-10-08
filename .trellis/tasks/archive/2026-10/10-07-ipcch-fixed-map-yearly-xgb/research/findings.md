# Planning evidence — 2026-10-07

No project-data fit or prediction was performed. Two read-only research agents inspected computation and reuse; the main session independently reran `enumerate_yearly.py` and persisted its output as `fit-enumeration.json`.

## Exact workload

21 global +160 local quartets =724 scalar regressors; main584, supplementary adds140. Per H totals200/160/168/196. Current unique requests15 global/115 local, historical15/114, shared9/69. Complete origin/block/date/support counts are in the JSON. Constant q5 targets remain ordinary fits.

The single insufficient unique local pool is H12, origin2021-01, r0011:443keys/164areas/27months. Every current regional pool is supported. Historical support-ready current-key counts by successive blocks are:

| H | 2023 | 2024 | 2025 | 2026 supplement |
| --- | ---: | ---: | ---: | ---: |
| 1 | 414 | 826 | 2,167 | 1,808 |
| 3 | 699 | 1,173 | 2,894 | 1,809 |
| 6 | 489 | 835 | 1,107 | 1,829 |
| 12 | not scheduled | 236 | 1,773 | 901 |

These are eligible upper bounds before gain, not predicted adoption or F1 effects. Historical six-month sets each map to one annual model origin. All empty target months are excluded from globally observed gate dates; notably H1 2026 uses2025-05..10 and H12 2026 uses2024-06..11.

## Source reuse and traps

- `IPCCHGeoXGBExperiment/ipcch_geoxgb/quartet.py:138-211` fits four scalar globals/own-root continuations but records unit weights; `modelstore.py:127-194` rejects non-unit records. Both must be adapted together in the isolated variant. Main session inspected these ranges directly.
- `stage3.py:80-135,173-235` assumes36-month pools and row-specific origins; `:246-265` fits current locals only after gate acceptance. Neither scheduling assumption implements this task's adopted annual protocol/diagnostic.
- `replay.py:343,369-421,453-480,711-715` assumes fit origin T−H or U−H and36-month windows. Keep old replay immutable; create annual-aware independent verification.
- `predict.py:27-71` checks frozen-map lineage against its original contract. New protocol identity must not overwrite the old map's contract version.
- `projection.py`, `metrics.py` and pure report functions supply original scientific semantics. `report.py:205-208` counts region-fold gates; the new report must count annual-block decisions. Default prepared CSV parsing and round-trip prediction parsing must not be silently conflated.
- Pin complete reused/new source inventories: original `learnmap.py:74-84` environment identity only hashes quartet.py, which is insufficient to bind annual control/evaluation behavior.

## Runtime and lifecycle facts

The research agent checked the existing Windows Store Python metadata only: Python3.12.10 and all8 versions in P6 `config/runtime-lock.json` match (numpy2.2.6, pandas2.2.3, xgboost3.0.0, geopandas1.0.1, pyogrio0.11.0, shapely2.1.0, pyproj3.7.1, pytest9.1.0). Recheck at P0; no install or scientific runtime execution occurred during this probe.

GitNexus query hit the existing LadybugDB read-only shadow-page replay error; no index repair was attempted. Bounded source inspection supplied the above evidence. No reusable all-history/split2024 script was located in the bounded searched locations; this does not establish that none exists elsewhere.

Read-only `trellis-audit status` shows no active runs and an existing registration keyed to the lowercase path `/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster`. Preserve that exact spelling for future lifecycle operations. The recorded executor belongs to an older session and was not rebound; current execution identity must be verified after approval. No audit run or task start was initiated.

Remaining operational verification: preflight rehash of all prepared arrays/source artifacts, synthetic weighted-model checks, actual fit timing/storage, and concrete executor identity. No model-quality or gain claims follow from this enumeration.

## Planning convergence checks

PRD R1–R13 retained; R14–R15 consolidate reporting and measurable completion requirements for final review. Design and execution plan are present. `task.py validate .trellis/tasks/10-07-ipcch-fixed-map-yearly-xgb` passed with seven existing references in each context manifest. A stdlib assertion check passed for 21/160 quartets, 724 total/584 main scalar fits, 15 current blocks, all reported full/local cohort counts, one historical annual origin per block, and task status `planning`. These checks validate planning consistency, not implementation. Final user approval remains pending; no task start or scientific fit occurred.

Subsequent approval: user said “确认，可以进入执行。” on 2026-10-07. PRD/design/plan approval status updated accordingly. Before the planning commit, GitNexus `detect_changes(scope="all", repo="Food_Crisis_Cluster")` again returned the LadybugDB read-only shadow-page error. No index repair or scientific change was made; Git path/diff checks cover this documentation-only commit.
