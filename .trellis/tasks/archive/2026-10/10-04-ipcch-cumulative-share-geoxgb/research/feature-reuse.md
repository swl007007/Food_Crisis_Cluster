# IPCCH feature reuse evidence

Read-only code inspection during grill, 2026-10-04. No training or feature generation executed.
Parent read `IPCCHPopulationHistoryExperiment/README.md` in full, delegated a bounded code/config comparison,
and spot-checked `build_history_block` and the existing arm cohort routing. Paths below are repository-relative.

- `IPCCHGeoRFExperiment/prepare_data.py:409-544` defines original93: 70 raw fields, 15 derivatives,
  target-month sin/cos, three label-history fields, two crisis-recency fields, actual horizon_months.
  `:948-1007` uses own origin T-H for raw values, trailing aggregates, lags and history.
- `IPCCHPopulationHistoryExperiment/config/feature-schema.json` freezes original93 + 468 = rich561.
  `prepare_data.py:222-260` validates names/block order/counts; `:848-871` concatenates original/rich matrices.
- `prepare_data.py:291-345` constructs q2,q3,q4,q5,severity_index,entropy,concentration,severe_fraction.
  Severe fraction is q4/q3 and remains NaN when q3=0.
- `build_history_block`, `prepare_data.py:540-639`, limits each area's observations to its own origin
  using right searchsorted; six observation slots, changes/rates/trends, calendar windows m06/m12/m24/m36/all.
  History uses the full valid ledger, not just fitting/evaluation rows. No cross-area outcome aggregation found.
- `prepare_data.py:474-476` requires >=1 observation for mean/extrema/latest-minus-mean, >=2 for population
  standard deviation, >=3 for slope; insufficient history stays NaN.
- Strict old binary threshold resides in `IPCCHGeoRFExperiment/prepare_data.py:210-222` (`5*P3plus>S`).
  It feeds original93 label/recency features (`:981-1006`) and rich binary histories/events/runs
  (`IPCCHPopulationHistoryExperiment/prepare_data.py:408-441,679-745`). These must be regenerated under >=.
  Frozen old class-count gates at GeoRF `prepare_data.py:31-32,383-395` cannot be transplanted unchanged;
  population preparation calls the old gate at `:834-837`.
- All feature names/order may remain reusable, but cached values and semantic identity do not: the new
  target/threshold/availability contract must be bound alongside the ordered schema.
- Existing population `share_xgb` is q3-only and matched-history-only; see `run_pipeline.py:263-264,297-309`.
  `:619-620` chooses matched training and eval_history for non-fullpool arms. This is not the new common
  full eligible pool contract and must not silently restrict the four-regressor experiment.
- Population XGB uses raw matrix slices/native missing handling; only RF uses fitted imputation
  (`run_pipeline.py:576-637`). No need to inherit RF imputation merely to reuse features.
- Month<=origin is an observation-month alignment assumption, not verified publication/vintage availability.
  No claim about complete upstream lineage or real-time availability follows from these helpers.

User accepted: rich561 as the shared global/local feature recipe, rebuilt using the
new target contract. No automatic inheritance of the older experiment's threshold tuning, correction arm,
matched-history training restriction, model-selection schedule or claims. The parent also read the original
`.trellis/tasks/archive/2026-09/09-21-ipcch-population-history-xgb/technical-contract.md` in full;
sections 1–2 provide the feature formulas. Feature acceptance is recorded in draft v0.15, R21.
The user subsequently accepted retrospective observation-month-end availability (v0.16, R22);
publication/vintage timing remains unverified. The new semantic schema version must be pinned before implementation.
