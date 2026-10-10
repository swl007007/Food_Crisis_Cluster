# P0 evidence: offline implementation and local no-fit readiness (2026-10-10)

Scope released: P0 offline only. No EE/GCS contact, no project-data model fit, no MLflow write. Executor Claude
Opus 5.5 /1M `3546490e-8e08-43f2-9f1d-9c72bccda1e8` (w14:p2); supervisor Codex w14:p1. Audit run
`42ee3def6b9c4222bc769712d47ccbe4`, base `50e1f228a8034e65c8883f913e4359e8bec4df39`, branch
`ipcch-alphaearth-supervised-partition`. Source inventory: `p0-source-inventory.sha256` (45 files).

## Implementation (isolated `IPCCHAlphaEarthExperiment/`, package `ipcch_alphaearth`)

Configs: frozen `experiment-contract.json` (validated against D1–D29 constants on every load), `inputs.json` (21 pinned
sources, geometry bundle), `runtime-lock.json`. Modules and attribution: README.md and module docstrings
(P6 copies at 6798df2; vendored compact296 from sibling IPCCH 9b41d7a, byte-identical to the compact manifest
code_sha256). CLI: preflight, prepare --local-only, benchmark, develop, freeze-map, predict, report, replay. All
fitting commands refuse without the P1 descriptor stage (checked).

## Local preparation (real sources, no fit) — run `~/ipcch-alphaearth-runs/p0-local-20261010b`

| Check | Result |
|---|---|
| Staging | 21 inputs byte-copied outside Dropbox and freshly rehashed; all pinned hashes match; geometry bundle under `sources/geometry/` with original filenames (readable by GeoPandas) |
| Canonical truth | 42,695 valid rows / 6,224 admins; rebuilt ledger identical to saved P6 (keys, phase, exact q strings) |
| Keys per H | identical to saved P6 key tables (admin, target, origin, country, phase) for H1/3/6/12 |
| compact296 order | list sha256 e97388f8…f9909, eight groups 29/120/56/28/29/27/5/2 |
| Overlap vs inherited compact | H3/H6/H12 equal on all 42,681 shared keys for all 296 features (0 unequal cells) |
| H1 adaptation | rebuilt at O = T−1; 234 origin-dependent features differ from compact H0, static/identifier equal |
| Direct source lookup | all 80 `value_at_origin` features equal the source cell at (admin, T−H): 3,415,600 row checks |
| Temporal ancestry | post-cutoff perturbation (2022-06, 2024-03) changed 0 earlier-origin cells over 1,864 feature/cutoff cells |
| History/IDP | independent pure-Python replays pass on every row (H1 history values 100,276; IDP 34,052) |
| Retained keys | 14 canonical keys absent from the old reported-phase compact cohort are kept (list: run folder) |
| Missingness | no row dropped for missing features; NaN fraction 6.6–6.8%; popdensity missing for 4,918 admins kept |
| Cohorts | main 17,322 / 16,919 / 16,413 / 14,087; supplement 4,092 per H; no valid key after 2026-04 |
| Calendar | 122 main folds (110 nonempty; empty 2024-12, 2025-11, 2025-12) + 16 supplement |
| Fit enumeration | B1 38 + B0 23 = 61 quartets / 244 scalar fits; 22 bridge slots; no empty backbone/correction pool; 1,153,625 fit rows; min backbone pool n_eff 3,458 |
| Correction pools | dev-2021 pools 8,309 / 8,197 / 6,293 / 3,683 rows reproduce the planning counts; earliest H12 largest-month weight share 0.31 |
| Determinism | developer runs a and b: all 40 prepared artifacts byte-identical |

Run a (`p0-local-20261010a`) is retained as evidence; it now refuses reuse because `inputs.json` changed (bundle
field) — the intended drift refusal.

## Tests (synthetic only): `p0-unittest-final.log` — 71 tests OK (272 s)

Covers the implement.md table: exact .20 truth/decoding, QC reasons, projection-before-clipping, undefined
macro-F1/R²; label U=O allowed / U>O refused, own-origin windows, season O+1 boundary, H1≠H0, anchors independent
of observed labels, empty folds kept; unique providers, 22 bridge slots, future-label perturbation leaves earlier
fits/predictions/pools/maps unchanged (with a later-block power check); A/S/Q merge vs SD averaging, exact .80,
no-data vs failed export, variance cancellation, piece admin/asset binding, complete admin set, orientation-free
geodesic area, finite 64-band mask (NaN/±Inf), ownership without double counting, pre-start EE concurrency;
joint ridge = independent null-space constrained reference (coefficients ≤1e-10, objective ≤1e-12), normalized
loss, unpenalized shared intercept, unweighted regional penalty, inactive slots, one supported leaf = exact B2,
all-unroutable = exact B2, empty pool = declared identity; brute-force best-first reference through two splits with
descriptor/node ties and strict gain floor; exact-Fraction selection; bootstrap multiplicities/undefined draws/K<2;
per-H statuses; store integrity (partial-before-fit, slice binding, digest/record binding); map-freeze hash chain;
replay canonical-cohort/provider/report/bootstrap binding with tamper regressions. Synthetic E2E: regional case
selects partition recipes (H1/H3/H6) with nine recipes × 2021/2022 guard records per H, fits each of the 61
quartets once, zero-fit replay passes; unsupported-split case gives exact B2 everywhere (zero deltas, bound 0).

## Benchmark (synthetic; not measured production cost)

XGB frozen recipe, 16 threads: 148.8 s per quartet at 38,603 × 424 (largest real pool), 78.0 s at 10,000; peak
3.69 GB; extrapolated over the 61 enumerated pools ≈ 1.9 h and ≈ 2.0 GB of boosters. Search: original fixture
stopped at 2 leaves (early-stop scope only); supplement fixture reaches 8 leaves / depth 3 under production
support/penalties: 18,432 cuts in 3.98 s at 15,700 rows × 4,430 admins, 0.6 GB; conservative 28 full paths ≈ 111 s.

## Supervisor samples and repairs (all closed by supervisor at their bounded scope)

Receipts and findings are retained in this folder: prepare (bundle staging, run/request drift), model/search (strict
tie floor, all-unroutable B2, store identity/recovery), pipeline (empty-pool identity, map-freeze chain), report/replay
fixed list (canonical binding, provider identity, full recount, draw contents, retained counts/NA + residual emitted
fields; red and green probes), extraction (piece binding, admin set, ring orientation, finite mask, concurrency).

## GitNexus

Earlier (pre-start) detect_changes failed with the LadybugDB read-only shadow-page error (historical). After the user
repaired GitNexus: impact on the new symbols returns "target not found / UNKNOWN" because the package is not
indexed — not evidence of zero impact; exact in-package callers were used instead. detect_changes on the staged diff
is recorded in the commit message.

## Cloud pre-authorization (recorded verbatim, effective after this checkpoint)

"预授权云端操作，但只授权必要的EE以及最多GCS的查询和下载，不涉及其他IAM。" — necessary EE query/extraction/download and at most
GCS query/download with existing identity/resources. Not IAM, billing/API enablement, new projects/buckets, or GCS
uploads/exports/writes. The drafted `extract_alphaearth.py export` writes to GCS/Drive (a cloud write); for P1 the
first option is direct EE computation returning the same reduced sufficient-statistic table locally in small bounded
batches, to be specified concretely after the metadata checkpoint. Any cloud write needs explicit clarification.

## Limitations / open items for P1

EE script and moment/ownership logic are unexercised against live EE; actual 2017 asset versions, native grids,
footprints, ownership impact, all-masked null-sum behaviour, quota/cost and destination remain P1 checkpoint facts.
Benchmarks are synthetic. Planning limitations (retrospective years, availability proxy, inherited tuning history,
product/pretraining history, approximate boundary weights, admin vintage, conditional inference) are unchanged.
