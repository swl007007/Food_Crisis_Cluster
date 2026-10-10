# Pooled XGB recipe: existing evidence and pending choice

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

2026-10-10. Read-only exploration for the AlphaEarth planning task; no fitting, code edits, cloud jobs or task lifecycle changes. A scout inspected configs, the compact fitting path and saved metadata; the coordinator read both complete JSON configs, `fit_model`, and the explicit effective-config block in the launch metadata. This note distinguishes established recipe facts from the proposed new use.

## Existing compact recipe

All paths in this section are relative to sibling `../IPCCH/`.

| Parameter | q2/q4/q5 | q3 |
|---|---:|---:|
| max_depth | 11 | 9 |
| n_estimators | 200 | 200 |
| learning_rate | 0.1 | 0.1 |
| min_child_weight | 0 | 0 |
| subsample | 1 | 0.5 |
| colsample_bytree | 0.5 | 0.7 |
| gamma | 0 | 0.1 |
| scale_pos_weight | 1 | 1 |
| reg_alpha, runtime default | 0 | 0 |
| reg_lambda, runtime default | 1 | 1 |
| objective | reg:squarederror | reg:squarederror |
| seed | 42 | 42 |
| n_jobs | 16 | 16 |
| early stopping | none | none |

Direct sources: `configs/forecasting_hyperparameters.json:2-9`, `configs/forecasting_hyperparameters_p3.json:2-9`. `scripts/modeling/run_deep_feature_weight_decay_forecasting.py:281-288` selects the q3 config only for `phase3_worse`, sets random_state, and passes sample_weight to XGBRegressor.fit without an evaluation set/callback. Compact annual batching sets n_jobs (`:1103-1111`). There is no horizon-specific config selection in the inspected path (`:1167-1174`).

Explicit effective recipes are also saved in `results/launch/nowcasting_2026_04_compact_cds_v1/runs/compact_baseline/12m/run_metadata.json:58-80` (coordinator inspected); seed, half-life and threads are at `:32-36`. The more recent historical/global extension `results/experiments/compact_climate_weather_oracle_eval_2026_jan_apr_v1/runs/compact_baseline/{0,3,6,12}m/run_metadata.json` records 296 features, seed42, half-life24, n_jobs16 and the same config hashes. The H12 record at `:402-414` identifies 2022–2025 as reused and 2026 as newly fitted. A later Somalia-local extension is a different scope, not a new pooled/global benchmark.

Config SHA256 values, checked against live config bytes by the scout:

- common config: `3742300661466f22eba8198e5d4f9c2a277615ba59562fbf27f251ecc932dd76`;
- q3 config: `cdc0e55aa15bdda932465088568b9ee717d226208ffb32076f438fb274e6317b`.

Frozen hash/runtime constants are in `src/ipcch/compact_features.py:75-77`. The old runtime is Python3.12.3, numpy2.4.4, pandas3.0.3, sklearn1.8.0, xgboost3.2.0; its interpreter is `/home/swl007007/.venvs/ipcch-geo/bin/python`. This does not yet choose this task's runtime.

The scout loaded saved H12/2026 boosters and found 200 rounds, 296 features and square-error regression. UBJ files do not preserve all training hyperparameters; do not use a bare reloaded booster configuration to reconstruct original depth/eta/seed. L1/L2 and CPU auto-to-hist defaults were checked by applying the frozen recipe to an already saved booster in memory under the existing 3.2.0 runtime, not by retraining. Default resolution is runtime evidence; original fit-recipe evidence is the config/runner/metadata combination. The original JSONs do not explicitly set tree_method; resolved CPU updater is grow_quantile_histmaker. New execution must make effective parameters explicit and freeze them with the chosen runtime.

## Temporal and target boundaries

The old fitter uses normalized cumulative shares (`src/ipcch/origin_safe.py:125-133`) and all share-valid historical labels up to each cutoff (`:313-316`). Temporal weights are unnormalized `0.5 ** (age/24)` (`:319-325`), shared across q2-q5. These are predictor-fit weights, separate from the normalized ridge objective adopted in D19; do not normalize XGB weights merely because the correction-layer objective is normalized.

Original horizons are H0/3/6/12, not H1. Old classification truth is reported phase and raw quartet decoding lacks this task's common monotone projection. Reusing these parameters does not authorize importing old training keys/QC, truth, decoder, H0 as H1, or incompatible saved predictions. D3–D7 govern this task's actual historical fit anchors; nominal January anchors must not override the accepted first-partial-year rules.

## Original tuning provenance is not established

The scout found no search ledger, tuning dates, fold keys or selected metric that generated the two JSONs in the inspected configs/current fitter/lightweight metadata/notebooks. `CLAUDE.md:74` only calls q3 separate tuning. `src/ipcch/food_crisis_functions.py:93-103` contains a possible combined tuning metric, but no evidence connects it to these config files. Notebooks that load the JSONs and later evaluate 2022–2024 do not establish when or on which labels the configs were selected. Old Git history/external tuning material was not exhaustively searched.

Therefore it is unknown whether original tuning used 2023+ labels. This is neither proof of leakage nor proof of temporally blind selection. Deep trees alone also do not prove a strong baseline under the new population-derived truth and smaller early fits. Existing reported-phase performance cannot establish new-protocol superiority over persistence.

## Accepted first-experiment choice (D21)

User accepted on 2026-10-10: use the existing two compact recipes as a fixed inherited benchmark, with the same target-specific parameters and seed42 for B0 and B1 at H1/3/6/12; do not add a new base-learner search in this first mechanism experiment. B2 and G must reuse the same B1 fitted backbone/raw predictions, rather than fitting separate favorable base models. AlphaEarth changes the input columns, not the base-model tuning budget. D19's nine regional recipes per H remain the correction/tree recipe grid; the XGB depth is not constrained by D14's partition-tree depth3.

The user explicitly accepted disclosure of the unverified original tuning lineage with this choice. New fits, transforms and map/recipe selection remain restricted to their accepted temporal information sets, but the complete inherited parameter-selection procedure cannot be certified as historically blind. This is the separate D21 limitation, not an inference from D2's retrospective product-version decision. Parameter reuse is not fitted-model reuse, and D9's compatibility requirements for old results still apply.

The alternative was a separate, temporally nested pooled-model selection procedure, with the same search opportunity in B0/B1 and every relevant development boundary. That would require a newly specified grid and additional historical predictions/fits; it must not become an informal sweep on the main evaluation or a choice made solely to increase G−B2. No such tuning procedure is included or has been run. Retaining the existing recipe saves that extra search but may leave the base suboptimal, especially with added AlphaEarth columns or different early training support.
