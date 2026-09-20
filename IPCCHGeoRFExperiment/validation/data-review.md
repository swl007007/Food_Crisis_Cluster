# Data / features / split audit
Pinned completion: 0b9e61abade508d7dbe43380f956347d926bbdc0. Source inspected from isolated export; actual run read-only: IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d.

No concrete defect identified in this bounded scope. This is not a whole-task verdict.

Verified with Windows Python 3.12, no model fitting:
- `python3.12.exe -B .../data_check.py`: exit 0. Source SHA matches ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673. Rebuilt R1 ledger from raw source: 42,695 valid, 15,206 positive. All valid row IDs, binary labels, P5-fill flags, decimal totals, normalized components and normalized P3+ strings equal saved ledger. Rebuild uses pinned implementation; focused decimal edge tests complement it.
- Independent keyed source lookup for ALL 170,780 saved feature rows: exact shape 170780x93; raw 70 at own origin, all 15 derivatives, calendar values, O=T-H, latest same-area valid history and latest positive recency equal stored values (numeric tolerance 1e-13; calendar atol 1e-14). History was independently searched per area, including metadata source months.
- Original Stage1 keys exactly equal valid 2014-2022 keys. Every area's chronological half split independently verified: fit 8,561, validation 9,558, singleton 1,472. All 93 stored Stage1 fill values independently recomputed from genuine fit views only.
- ALL scheduled Stage3 window starts and training-row counts verified against [O-35,O]; ALL 126 fitted folds' complete training-key sets and all 93 per-fold RF fill values independently match saved features/training pool.
- `schema_check.py`: exit 0; exact 70 raw names/order matches six approved whitelist blocks in secondary-predictors.md.
- 20 non-geography `test_contracts.py` checks: exit 0, all passed. Test file copied from ignored worktree, not committed evidence; runner fixes tempfile directory to audit scratch.

Code evidence (paths relative pinned export):
- IPCCHGeoRFExperiment/prepare_data.py:237-288 R1 missing P5 only, bounds, population, exact sum and exact 5*P3plus>S threshold; :292-380 ledger source/raw/normalized provenance.
- prepare_data.py:409-587 frozen 70+15+2+3+2+1 schema; :762-805 complete per-area calendar-window sums; :808-838 inclusive history cutoff.
- prepare_data.py:940-1012 row-owned origin lookups and latest history/crisis; :2273-2318 original-outcome half split.
- IPCCHGeoRFExperiment/run_pipeline.py:523-545 training-only Stage1 imputation; :1556-1571 Stage3 same-horizon [O-35,O] selection; :1634-1660 shared RF transform and native-NaN XGB pool.

Limits: did not independently classify every invalid raw target with a second Decimal implementation; rejected-ledger rowwise reconciliation not covered. Did not refit models or inspect geometry, q/F1 effective-support implementation, pseudo-row baseline internals, or model checkpoint matrices. Independent feature checks cover history labels/recency and dates, but do not separately recompute history missing flags or latest-label age (focused tests cover these). Publication-time validity and upstream covariate provenance remain outside observation-month availability contract. Initial checker failures were its own pandas float parsing of decimal-string columns and Windows default GBK reading of UTF-8; corrected checkers passed, no production change.
