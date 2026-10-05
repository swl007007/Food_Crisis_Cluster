# Architecture evidence, 2026-10-04

Read-only research followed by task-local planning writes only. No training, replay, source modifications or audit acceptance claim.

## Snapshot and discovery

- Food_Crisis_Cluster HEAD observed: `f75d5de2da4895327b9f1f4679488ce5aa808f0d`.
- Sibling IPCCH HEAD observed: `c06bc0f59ba37a672234595b1f63b53d21652134`; status was clean.
- IPCCH absolute root: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/IPCCH`.
- Three read-only default agents explored package identity, IPCCH regressors/decoder, and GeoXGB partition implementation. Parent read both package READMEs, IPCCH GeoRF design, current GeoXGB design/evaluation contract in full and spot-checked the key code below.
- GeoXGB GitNexus query failed on read-only shadow-page replay; used source without reindexing. IPCCH query line numbers drifted; source lines govern.
- `trellis-audit status` attempt returned `attempt to write a readonly database`; no enrollment/start/close was attempted. No controller state is inferred.

## Direct evidence anchors

Paths below are relative to Food_Crisis_Cluster unless explicitly prefixed `../IPCCH/`.

| Evidence | Source |
|---|---|
| Four cumulative shares q2..q5 | `../IPCCH/src/ipcch/forecasting_weight_decay.py:132-139`, `derive_cumulative_targets` |
| Four real-valued scores thresholded with >= | `../IPCCH/src/ipcch/operational_contract.py:258-262`, `decode_phase_predictions` |
| cummax only after binarization; highest phase wins | Same file `:276-292` |
| Four separate XGBRegressor fitting loop | `../IPCCH/scripts/modeling/run_deep_feature_weight_decay_forecasting.py:264-271,322-343` |
| Launch clips/rounds; differs from raw operational score decoder | `../IPCCH/src/ipcch/launch_nowcasting.py:1042-1072` |
| Ready-made regional mapping interface, not partition learning | `../IPCCH/run_region_models.py:90-102,417-420,500` |
| New local GeoXGB package/current study | `FEWSNETGeoXGBExperiment/README.md:3-32` |
| Frozen global trees plus local additions | Same README `:58-62`; latest task `design.md:43` |
| Backend still int labels and four classes | `FEWSNETGeoXGBExperiment/src/model/native_xgb.py:177-204,224`; `src/experiment/plan.py:24-27` |
| Partition rejects non-classification/non-macro mode | `FEWSNETGeoXGBExperiment/src/partition/transformation.py:180-182` |
| Parent/child selector rather than generic significance test | Same file `:806-849`; `src/partition/partition_opt.py:860`, `select_macro_children` |
| F/S/C/E3 roles and crisis F1 empirical gain | `.trellis/tasks/archive/2026-10/10-02-exogenous-transition-forecast-design/evaluation-contract.md:36-46` |
| Consensus weight is positive F1-logit gain; null map fallback | Same contract `:50-58`; `FEWSNETGeoXGBExperiment/scripts/run_stage2.py:87-110,227-269` |
| Latest task completed; audit waiver is not PASS | `.trellis/tasks/archive/2026-10/10-02-exogenous-transition-forecast-design/task.json:6,14`; `research/final-report.md:51,68` |
| IPCCH GeoRF is binary, not continuous share regression | `IPCCHGeoRFExperiment/README.md:1,74-83`; `prepare_data.py:210-222` |
| Exact .20 is negative in old GeoRF binary target | `IPCCHGeoRFExperiment/prepare_data.py:210-222` |
| No split learned; local/pooled training composition differed | `IPCCHGeoRFExperiment/README.md:44-59` |
| IPCCH canonical run and release validation | Same README `:8,146-156`; `validation/artifact_hashes.json:2-5,19` |
| Pooled q3 XGB reference, explicitly no GeoRF/partition | `IPCCHPopulationHistoryExperiment/README.md:12,24` |

The IPCCH GeoRF release identifies `runs/ipcch-v1-20260920d/` and frozen source
`IPCCH_2026_completed.csv` SHA256 `ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673`.
Its base ZIP is `GeoRFBaseline/releases/georf-baseline-v0.1.0.zip`, SHA256
`39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0`.
These are existing manifest declarations inspected during discovery, not hashes recomputed in this turn.

## Foundational originals read by parent

- `IPCCHGeoRFExperiment/README.md` (194 lines).
- `.trellis/tasks/archive/2026-09/09-19-ipcch-binary-georf-pipeline/design.md` (274 lines).
- `FEWSNETGeoXGBExperiment/README.md` (155 lines).
- `.trellis/tasks/archive/2026-10/10-02-exogenous-transition-forecast-design/design.md` (112 lines).
- Same task `evaluation-contract.md` (76 lines).

The source READMEs include historical unarchived task paths; the archived paths above exist.
Earlier approved contracts are evidence of those experiments only. No dates, thresholds,
capacity grids or audit waivers transfer to this new experiment without a decision.
