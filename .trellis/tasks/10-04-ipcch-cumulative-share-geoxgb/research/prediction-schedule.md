# Rolling prediction schedule and coverage — accepted R47

Planning only, 2026-10-04. The user accepted this schedule as R47 / v0.41,
following R46. Two bounded read-only source inspections were used;
no feature build, model fitting, raw-data recount or prediction experiment ran.

## Existing IPCCH package behavior

- `IPCCHGeoRFExperiment/run_pipeline.py:1163-1188` explicitly schedules every
  main calendar month, including months without valid truth. `:1205-1209` asserts
  122 main folds. `:1211-1229` appends only actually available target months in
  the partial year 2026. Available months come from the valid-target panel
  (`:2729-2733`), not the current wall-clock month or a guessed release date.
- Its `fit_fold` returns for empty test rows (`:1628-1632`), but checks empty
  training first (`:1621-1627`). This order is a source fact, not an adopted
  requirement for the new task. The accepted rule below determines a scored fold's
  necessity before requesting global fits, preserving R40 for required fits.
- `IPCCHGeoRFExperiment/prepare_data.py:1094` builds features from ledger.valid();
  `TargetLedger.valid` uses target_valid==1 (`:145-146`). The supervised matrix
  is valid original outcome keys times H (`:935-948`), rather than every area in
  every calendar month. Missing history remains NaN (`:981-990`).
- `IPCCHPopulationHistoryExperiment/prepare_data.py:768-804` uses the same four
  H and 122-fold main calendar. Its rich561 builder consumes valid-target keys
  (`:849,868-871`), and its runner records skipped_empty_test (`run_pipeline.py:
  549-559`). The inspected entry does not define a 2026 supplementary calendar.
  Its history-only arm/score restrictions are excluded by new-task R21.
- The old GeoRF runner retains rows without persistence in its full-cohort
  results (`run_pipeline.py:1954-1955`). Its donor/map requirements do not override
  R36's explicit no-donor and unmapped-area global policy in the new task.

## Accepted R47 retrospective evaluation schedule

| H months | Main target months, inclusive | Scheduled main folds |
|---|---|---:|
| 1 | 2023-02 through 2025-12 | 35 |
| 3 | 2023-04 through 2025-12 | 33 |
| 6 | 2023-07 through 2025-12 | 30 |
| 12 | 2024-01 through 2025-12 | 24 |

1. Keep all 122 main month/H entries in a schedule ledger, with O=T-H and
   origin>=2023-01. For each entry, forecast all original area/target-month keys
   whose true population shares pass this task's QC. Do not restrict these keys
   by country, learned-map membership, local adoption, history or persistence
   availability. Apply the adopted feature, fallback and metric rules unchanged.
2. If a scheduled main month has no QC-valid target keys, record
   `no_valid_target`, support=0 and metrics=NA. Request no current or historical
   gate fits for that empty fold. This is a defined empty evaluation cohort, not
   a model technical failure and not F1=0. A nonempty prediction fold with any
   required global fitting pool empty still stops as incomplete under R40;
   errors on attempted models still follow R41.
3. For supplementary 2026, derive the set of months with at least one QC-valid
   original population outcome from the frozen source; cross those months with
   all four H and use the same prediction-key policy. Keep this period separate.
   Record the observed month set and source identity; do not project an endpoint
   from today's date or automatically append new data. Enumerate all 12 calendar
   months in a supplementary coverage ledger, marking absent ones as having no
   QC-valid outcomes in this source, not as known future forecast failures.
4. Preserve raw/QC and geographic coverage ledgers for invalid or absent target
   observations. Do not impute their truth or add model predictions for unobserved
   area/month combinations in the first version. This is observed-outcome
   retrospective evaluation, not a full monthly forecasting service or proof of
   full geographic prediction coverage. Missing truth differs from missing
   history: a QC-valid target with no past history remains a prediction row.
5. Target validity only defines the retrospective evaluation cohort. The value of
   current T truth must not enter X(i,O), fitting pools, local adoption gates,
   routing choice or hyperparameter selection. Persistence retains its independent
   as-of-O availability flag; matched comparisons use the already adopted common
   keys and full-model scores remain separately reported.

This follows the completed IPCCH package's evaluation scope and retains explicit
empty-month evidence. It avoids fitting current forecasts that cannot be scored,
but cannot establish model behavior for the entirely unlabeled area/month grid.
Missing reports may be selective; observed-key performance is not automatically
population-wide or full-calendar performance. No new source truth is filled.

The current schedule has at most 122 nonempty main fold executions plus 48
supplementary ones. These are current month/H folds, not model fitting counts:
each executed fold can request up to six historical dates under R37, each with
global and eligible regional quartets, plus its current fits. R48 subsequently
fixes exact-identity reuse and conservative fit accounting in fit-reuse-budget.md.
Actual data-dependent unique request counts await authorized preparation. No training is authorized.
