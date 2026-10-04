# P1 evidence — population targets, calendar, rich561 (2026-10-04)

Phase status: implemented and checked on a development preparation run. No
model was fitted. The formal P6 run will re-prepare under the frozen code; the
numbers below are data facts of the pinned source, not model results.

## Implementation (package `IPCCHGeoXGBExperiment/ipcch_geoxgb/`)

| Module | Content | Provenance (`config/source-provenance.json`) |
|---|---|---|
| `targets.py` | R20 QC ledger (order unchanged), exact `5*sum(P_k..P5) >= S` phase truth for k=2..5, q2..q5 and p1..p5 normalized, four-class/binary maps | GeoRF prepare_data 101-380, adapted |
| `features.py` | original93 at own origin (raw/derived/calendar/label-history/recency/H); history468 (8 series, slots, changes, trends, windows, support, binary states, threshold distance, crisis windows, events); alias recomputation | GeoRF 409-1079 + PopulationHistory 138-765, 1018-1073, adapted to crisis = phase>=3 |
| `schedule.py` | Stage1 within-area F/S split (+F-before-S assertion), 122-fold main calendar, 2026 supplement + 12-month coverage, R23 window, R37 gate dates | GeoRF 2273-2362; PopulationHistory 768-806, adapted |
| `prepare.py` | orchestration, persistence as-of-O lookup, deterministic gzip, hashed manifest; CLI `prepare --run-id` | new |

Infinity policy (implement.md clarification 3) kept: raw/original93 → NaN with
audit (none in the pinned source); history468 infinity → stop; final X finite or NaN.
Truth `q2..q5` floats come from exact `Q_k/S` at 100 digits; history q-series use
the same values (old code summed p floats; recorded as an adaptation).

## Tests

`python -m pytest -q` (pinned Windows runtime): **98 passed** (70 P0 + 28 P1),
`evidence/P1-pytest.log`. P1 tests cover: exact .20 at q2/q3/q4/q5 including
`S≠1` (0.21/1.05 → phase 3; 0.20999 → phase 2); sum .90/1.10 inclusive and
.89/1.11 rejected; missing P1–P4 vs P5 fill flag; population missing/≤0;
malformed/duplicate keys; preserved reported phase; no-history rows (NaN
features, persistence NA); sparse calendar lag (missing panel month → NaN lag
and NaN 4-month sum, not a row shift); 561 order and alias equality; m06/m12
window edges and std/slope support; history infinity stop; raw infinity audit;
inclusive crisis state in label-history features; future-data perturbation
(panel and labels after O changed) leaves the row's 561 values and persistence
unchanged; persistence = latest observation at or before O (age 0 at O);
122-fold calendar by H and first months; 2026 coverage; F/S halves/singletons;
R23 window and R37 six-date rule.

## Development preparation run `p1-dev-20261004`

`PYTHONPATH=. python -m ipcch_geoxgb prepare --run-id p1-dev-20261004` → exit 0,
57 s. Manifest `evidence/P1-prepared-manifest.json` (sha256 `4fc8684f…7d727`)
lists the SHA256 of every artifact (ignored under `runs/p1-dev-20261004/prepared/`).

- Ledger: 1,219,868 rows; 42,695 valid (6,224 of 6,227 areas); invalid reasons
  missing P1–P4 1,176,232 / sum out of bounds 577 / population ≤0 363 / share
  out of bounds 1; 84 valid rows with P5 filled.
- Phase truth counts 1/2/3/4/5 = 8,404 / 16,484 / 15,163 / 2,599 / 45; crisis
  17,807, non-crisis 24,888. Normalized q exactly 0.20: q2 282, q3 2,601, q4
  1,376, q5 14. Cross-check: the old strict-rule package reported 15,206
  positives; 17,807 − 15,206 = 2,601 = rows with q3 exactly 0.20, so the
  ≥ boundary is the only change. 42,695 valid equals the old audited count.
- rich561 per H: 42,695 × 561 each; 0 infinities, 0 all-NaN columns, 0 rows
  without an origin panel row. Rows without any history at origin H1/H3/H6/H12:
  6,224 / 6,254 / 7,994 / 10,741 (kept, NaN features); persistence available
  36,471 / 36,441 / 34,701 / 31,954. Max 25 valid observations per area.
- Stage1 F/S (2014-01..2022-12): 19,591 outcomes; F 8,561 (crisis 2,564),
  S 9,558 (crisis 3,355); 1,472 singleton areas; 3,264 multi-outcome areas;
  1,491 areas with no outcome (identical split counts to the old package, as
  the rule is unchanged).
- Calendar: 122 main folds (35/33/30/24); 110 non-empty; 12 `no_valid_target`
  folds = target months 2024-12, 2025-11, 2025-12 for every H. 2026 observed
  months 2026-01..2026-04 → 16 supplementary folds; coverage ledger lists all
  12 months.

## Notes for supervision

- 2,601 valid rows sit exactly on q3 = 0.20 (6.1% of valid); the inclusive
  rule therefore materially changes crisis prevalence relative to the old
  package. This is the accepted R12 contract, recorded as a data fact.
- GitNexus impact/detect_changes remain unavailable (LadybugDB shadow-page
  error); all P1 symbols are new and only used inside the package.
