# Current support-gate evidence

Read-only source inspection during grill, 2026-10-04. Two bounded scouts inspected
the IPCCH packages and current FEWS NET GeoXGB independently. Parent spot-checked
the FEWS constants, Stage1 eligibility and Stage3 gate implementation. No training.
Paths below are repository-relative. The source facts are existing engineering
floors, not statistical sufficiency claims. Subsequent user acceptance is recorded
separately in the final section.

## FEWS NET GeoXGB

- `FEWSNETGeoXGBExperiment/src/experiment/plan.py:74-81`: FIT_SUPPORT is rows=500,
  areas=50, dates=6, classes=2; Stage1 validation rows=100, areas=20, dates=3;
  Stage3 gate adds local_fit_dates=3; GATE_DATES=6; strict Stage3 gain >.01.
- `src/partition/transformation.py:812-836` under that package deduplicates
  original fitting/evaluation keys before support checks. Each child is eligible
  independently; an ineligible side retains parent. All ineligible keeps parent.
- `src/model/native_xgb.py:272-282`: dates counts distinct target months, not
  consecutive calendar coverage or per-area minimum months. classes is the number
  of represented four-class codes, not crisis-positive/negative support.
- `src/experiment/stage3.py:330-365`: region gate pools historical validation rows;
  local_fit_dates counts validation dates with a supported local fit. Undefined
  crisis F1 or insufficient support rejects adoption. These are pooled floors,
  not requirements of 100 rows/20 areas on every date.
- `src/experiment/stage3.py:443-457,471-485`: historical unsupported local fits keep
  all validation rows but substitute global predictions. Current unsupported local
  fit also routes global even if the historical adoption gate passed.
- `src/experiment/stage3.py:274-278` and `app/main_model_GF.py:800`: global/root
  support is not governed by the full local FIT_SUPPORT table.
- Regression adaptation must remove class-code casting/softprob/argmax assumptions;
  classes>=2 does not apply unchanged to continuous q2..q5. Validation crisis
  composition and constant handling were adopted in R29/R31; global support
  was separately accepted in R40 below.

## IPCCH package boundaries

- IPCCH GeoRF runner loads `GeoRFBaseline/releases/georf-baseline-v0.1.0.zip`
  (`IPCCHGeoRFExperiment/run_pipeline.py:65,1996,2283,2663`). Scout verified relevant
  `GeoRFBaseline/` support/helper files match ZIP members; root `src/` is not an
  interchangeable source for those historical gates.
- `IPCCHGeoRFExperiment/run_pipeline.py:98-100,1673-1675` and
  `GeoRFBaseline/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:245-262`:
  local classifier needs >=50 rows and >=2 binary classes, otherwise pooled RF.
  It has no historical Stage3 F1-gain gate; new Stage3 gate comes from GeoXGB.
- `GeoRFBaseline/src/partition/transformation.py:693-708` requires nonempty child
  fitting and validation sets; its old classifier support checks are much weaker
  than the newer GeoXGB floors. Exact old cohort capacities are identity checks,
  not transferable minimum sample sizes.
- `IPCCHPopulationHistoryExperiment/run_pipeline.py:342-375`: empty fit fails;
  constant regression target returns an explicit constant prediction. This is a
  pooled-arm source behavior; its constant shortcut was not inherited under R31.
- Neither old binary two-class support nor four-class classes>=2 establishes
  continuous-target variation or adequate crisis validation. No new per-class
  count, unique-area or month floor should be attributed to these packages.

## User-accepted structural floors (v0.22, R28)

User accepted the newer GeoXGB structural floors for local eligibility: fit
500 original area-month keys / 50 areas / 6 target months; validation
100 keys / 20 areas / 3 target months; Stage3 additionally >=3 historical dates
with supported local fitting. Count each horizon independently and never multiply
support by the four regression targets. Do not inherit classes>=2 as a regression
requirement. Global support was separately accepted in R40 (v0.34). The gate-date
calendar was accepted in R37 (v0.31), with up to six latest globally
observed target months and pooled confusion counts; see historical-gate-calendar.md.
Constant targets were subsequently accepted for ordinary squared-error booster
fitting (v0.25, R31), retaining global prefixes for local continuation; the older
constant_target/model=None shortcut is not inherited. This is a planning decision,
not implementation authorization.

## User-accepted additional validation class floors (v0.23, R29)

The user accepted >=20 genuine crisis and >=20 genuine noncrisis original keys,
in addition to all structural floors. Stage1 checks each proposed child region's
validation cohort; Stage3 checks each region's pooled historical validation cohort,
not each date separately. Counts use the new population-derived truth and are
horizon-specific. Insufficient support retains parent/global and does not remove
evaluation rows. This is a newly proposed engineering rule, not a threshold found
in the source packages or a statistical power guarantee. It does not automatically
impose binary class constraints on the continuous regression fitting pool.

## Accepted R40: global/root eligibility and empty fitting pools

User accepted this policy in v0.34. Focused current-source check confirms:

- `IPCCHGeoRFExperiment/run_pipeline.py:1621-1627` explicitly stops on an empty
  global fitting pool. No local-style row/area/month minimum is applied there.
- `FEWSNETGeoXGBExperiment/src/experiment/stage3.py:274-278` likewise rejects an
  empty global pool, then fits and records support rather than applying FIT_SUPPORT.
- FEWS Stage1 has a separate classifier-only root classes>=2 check at
  `FEWSNETGeoXGBExperiment/app/main_model_GF.py:800-804`; the new continuous
  regression backend must not reinterpret it as an approved global support floor.
- `IPCCHPopulationHistoryExperiment/run_pipeline.py:342-343` also rejects empty
  fitting pools. Its subsequent constant shortcut differs from adopted R31.

User-approved policy: for Stage1 global/root and every required historical/current
Stage3 global quartet, fit when the legal QC-valid original-key pool is nonempty.
Do not add the local 500/50/6 floors or crisis class-count requirements to global
regression. Record keys, area count, distinct target months, crisis composition
and per-target constant flags; nonempty is operational eligibility, not a claim
of adequate precision or statistical support. Existing local and validation gates
remain unchanged.

If a required global pool is empty, record the stage/H/origin/window/reason and
stop that run as incomplete. Do not widen the window, delete the fold/history
date, use future models, or substitute persistence as a model prediction. This
protects the specified cohort and global comparison baseline. Whether a date
requires predictions at all belongs to the separately pending prediction schedule;
numeric fit/prediction failures follow the separately accepted R41 stop-on-error
contract in error-policy.md.

This keeps the source package's global coverage policy, with the trade-off that
a small but nonempty global pool can produce a weak model. Imposing a stronger
global minimum would be a new engineering gate and could leave more origins
without a usable global baseline; no empirical justification for another numeric
minimum has been established in this planning work.
