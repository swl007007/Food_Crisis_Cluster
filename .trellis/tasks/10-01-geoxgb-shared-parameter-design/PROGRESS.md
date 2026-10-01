# GeoXGBoost execution progress

2026-10-01. Operational ledger, not approval authority. Approval is D25 in prd.md; numerical contract is experiment-plan.md v1.0.

## Checkpoint

- Approved: D1–D24 design, D25 implementation and bounded experiment execution.
- Mother source: 14c89bc150194452361bb495c601de070cd94ce7.
- Planning/context validation: passed; implement.jsonl 7 entries, check.jsonl 6.
- Working tree before execution: only this task directory untracked; mother unchanged.
- Audit: enrolled exact lowercase repository path, no active runs/open gates. Controller stopped at initial inspection.
- Verified available executor: Claude Opus 5.5, pane wN:p2, session a53ea9d2-1aa3-44e8-8bc4-0dde99258a6c, terminal term_65ccaebe9009a8. Idle in this repository at verification; registration not yet changed.

## Remaining order

1. Commit frozen planning/authorization; bind live executor and audit start; verify base/run/status.
2. Fork 55 source/package files; native continuation and date/route contracts; commit authoritative code.
3. Execute bounded global screening, 648 candidate tasks, 24 development pipelines and two old-map diagnostic arms; record timing/resource probes within budget.
4. Freeze selected configs/map/rules by 2020-12 information; one final retrospective evaluation and country bootstrap.
5. Replay/verify/report implementation completeness and scientific results separately; commit relevant evidence, audit close, verify accepted result.

No product code fork or model training has run at this initial checkpoint. Executor must update this ledger at material milestones and retain failures/negative results.

## 2026-10-01 — Audit start (executor)

- Started from the bound Claude session itself: `trellis-audit --repo '/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster' start geoxgb-shared-parameter-design`.
- Run id `ed632775e46b47598bcfc53ef088bf8e`, phase `active`, base_sha `5e2289d8914a86ce079c95149a13bca2b3610cbb` (= branch `geoxgb-shared-parameter-experiment` HEAD, planning commit).
- Executor recorded by the controller: claude session `a53ea9d2-1aa3-44e8-8bc4-0dde99258a6c`, terminal `term_65ccaebe9009a8`; matches this process (`HERDR_PANE_ID=wN:p2`, `CLAUDE_CODE_SESSION_ID`). Controller `running=True`; it was the only active run at start.
- task.json status after start: `in_progress`. Working tree before product edits: only task.json modified by the wrapper.
- Implementation boundary: new sibling `FEWSNETGeoXGBExperiment/` forked from the 55 tracked files of `FEWSNETFourClassBaseline/` at 14c89bc; mother package, its runs and raw inputs are read-only. Root legacy GeoRF/GeoXGB code is not touched. Bulky Stage 1 scratch goes outside Dropbox.

## 2026-10-01 — Fork and runtime compatibility (executor)

- Fork: `aa7ac82` copies the 55 files tracked under `FEWSNETFourClassBaseline/` at 14c89bc (git archive, no runs/caches); every blob equals its mother blob. `feature-schema.json` was hidden by the root `*.json` ignore rule; `6ae4700` adds `!FEWSNETGeoXGBExperiment/feature-schema.json` and tracks it (blob identical). These two fork commits were made before the GitNexus re-index finished, so `detect_changes` could not cover them (fork absent from the stale graph; both are verbatim-copy/schema-only). From here every commit runs `detect_changes`; fork symbols' callers are verified from source where the graph lacks them.
- Runtime (Windows `python3.12.exe`): Python 3.12.10, numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.15.2, xgboost 3.0.0 (CPU+CUDA build, OpenMP; CPU used), polars 1.27.1, geopandas 1.0.1, shapely 2.1.0; 32 logical CPUs.
- Executor probe (scratch `C:\Users\swl00\geoxgb_scratch\probe\probe_continuation.py`, synthetic NaN/true-zero data): fixed four-class `xgb.train` + `xgb_model=` continuation on a subset without class 3 and on a two-class subset: parent prefix identical in `get_dump(json)` and in model JSON trees incl. `default_left`/leaf weights/`tree_info`; parent bytes unchanged; base_score `5E-1` unchanged; 4 probabilities; UBJ reload predictions exact; margin(child) = margin(parent) + appended-range margin − base_score (max err 1.2e-7, float32); refits byte-deterministic; DMatrix rejects ±inf (so the fork must clean ±inf→NaN).
- Coordinator independent probe (reported 2026-10-01): G1 200 rounds + L1 20 rounds on classes 0/1/2 only — parent JSON byte-identical, all 800 prefix trees and base_score exact in the 880-tree child, 4 probabilities, UBJ reload and zero-increment clone exact. Runtime evidence only, not fork-path coverage; production regression tests still required.
- Timing probe (v7 2014-truncated h4 snapshot, O=2020-06, 79,338 rows / 15 label months, nthread=4): G depth3×200 rounds 5.5 s; depth4×400 rounds 10.0 s; L2 40-round continuation on 11k rows 0.4 s; 2.5 MB UBJ per G4 booster. Budget is CPU-feasible; plan sequential nthread=4 fits with outer process concurrency.

## 2026-10-01 — Tooling limitation: GitNexus

- Index is stale at 3cfb546 (2026-09-21) and does not contain the fork (`MATCH (f:File) WHERE filePath CONTAINS 'FEWSNETGeoXGBExperiment'` → 0). Two `node .gitnexus/run.cjs analyze` attempts from this session failed (first stopped without a completion line; second: `Analysis failed: Invalid transaction type to rollback`). Afterwards `detect_changes` reports `LadybugDB unavailable ... Couldn't replay shadow pages under read-only mode`; `.gitnexus/lbug` (424 MB) is present with a 0-byte `lbug.shadow` and `lbug.wal.checkpoint` of 17:06. I stopped touching the index so as not to worsen it; a read-write re-open/rebuild by the owner may be needed.
- Consequence: impact/detect_changes cannot be produced for fork symbols. Callers of every edited fork symbol are verified from source with grep instead (fork is self-contained): `partition` ← `GeoRF.fit` + tests; `GeoRF.fit` ← `app/main_model_GF.py` (+ unused 2-layer path); `train_and_eval_two_branch` ← `partition`; `select_macro_children` ← `partition` + tests; removed `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` / `src/utils/inventories.py` / `REQUIRED_STAGE*_FOLD` were used only by `run_all.sh`, the old acceptance/verifier and tests, all replaced. Risk: fork-local, low; mother untouched.
