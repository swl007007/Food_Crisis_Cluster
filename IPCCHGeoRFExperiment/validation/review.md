# IPCCH release review — 2026-09-20

**Disposition: ready for integration after the release corrections in this change.**
No numerical defect was found in the audited scientific outputs. This conclusion
is limited to the checks below; it is not a claim of verified publication timing,
correct upstream geographic identity, or a full end-to-end training replay.

## Frozen scope and independence

- Base: `2dfa121a9398de9a1918ba9c0af34b31ecbb117a`.
- Opus implementation: `3addf9a97d31ee2a869d22df26ececa995acab19`.
- Opus closure / audited source: `0b9e61abade508d7dbe43380f956347d926bbdc0`.
- Authoritative run: `IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d`.
- Source/release/production/test/run-artifact identities: `artifact_hashes.json`.

Tracked source was exported from the fixed closure commit into an independent
temporary directory, without symlinks to source files. The four missing test
files were copied separately and hash-identified; they were not falsely treated
as tracked at the audited commit. Original run artifacts were read-only.
Four fresh, bounded reviewers inspected source and independently checked different
artifact paths; the main reviewer checked source excerpts, reran all tests and
replayed eight real forecast folds. The main reviewer had authored the original
plan, so this is not a fully context-free controller audit. No executor transcript
was used as scientific evidence.

## Findings corrected before integration

1. **Reproduction files absent from Git.** The root `test_*`/`*.json` ignore rules
   excluded all four IPCCH test files, baseline source-manifest/runtime/test files,
   task metadata and referenced planning audits. Narrow exceptions now include
   these existing files. No raw source, full run or fitted model is promoted.
2. **Closure references were stale.** Both archived JSONL context manifests still
   pointed to the former active-task path. All 20 references now resolve to the
   archived task; Trellis context validation passes. Current status banners and
   task commit metadata distinguish completion from the retained planning snapshot.
3. **Coverage denominator was mixed.** 75,580/81,109 is 93.18% for main plus
   partial 2026; 91.55% applies to main only, 59,273/64,741. README now separates them.
4. **Area support was mislabeled.** Valid target labels cover 6,224 areas; 6,227 is the
   geographic universe. README and closure acceptance now distinguish them.
5. **The headline exceeded the comparisons.** Approved intervals compare
   partitioned RF to each baseline; they do not establish pooled-RF/XGB versus
   persistence superiority. The headline and interval-count wording were narrowed.
   Sparse-history concordance and zero missingness are now described as observations,
   without claiming they identify crisis stability or upstream filling/leakage.

Production Python files and the scientific protocol are unchanged by these
corrections; their hashes still match the authoritative run manifest. Original
output tables and their numerical values were preserved. The code/metadata additions
and narrative changes are release corrections, not a new experiment.

## Acceptance evidence

| Criterion | Current evidence |
|---|---|
| A1 | Source SHA verified; ZIP CRC/MANIFEST and all 45 payload entries checked; only the approved polygon guard differs. Source/run geography components match recorded hashes. All 128 run files are hash-bound in artifact_hashes.json. |
| A2 | Independent Fraction arithmetic scanned all 1,219,868 source rows, exactly matching all 42,695 valid keys/labels, 15,206 positives, 84 P5 fills and 2,601 threshold ties. Pinned implementation rebuild also matched normalized component strings. |
| A3 | All 170,780 feature rows obey own-origin alignment. All 126 fitted folds' 1,733,566 training keys and 81,109 predictions were independently reconciled with exact [O-35,O] windows. All 122 main scheduled folds plus 16 partial folds are represented; 12 are empty. |
| A4 | Original split matches 8,561 fit / 9,558 validation / 1,472 singletons; one accepted root only. All 2,963 nonlearning areas' nearest-donor identities/distances independently match; 3,264 learned + 2,367 completed + 596 unresolved. All 6,227 serialized repaired geometries equal an independent reconstruction of the approved repair. |
| A5 | Persistence independently reconstructed from the full valid-label ledger; E_persist reconciled before scoring. No learned prediction is missing. All 5,411 pooled fallback rows have identical partitioned/pooled probabilities. |
| A6 | Independent reconstruction matched 56 metric rows, 40 delta rows, all 48,000 bootstrap statistic values, country-draw multiplicities and all intervals. Reporter regeneration reproduced all 23 report artifacts' computed contents. Eight model refits reproduced probabilities exactly. |
| A7 | No accepted split and unfavorable contrasts retained. Root-only RF differences are explicitly attributed to different local/global training composition; no partition advantage or untested model-superiority claim. |
| A8 | Exact 70 raw-field order checked against approved whitelist. All 170,780 x 93 feature entries checked through independent raw/calendar/history paths. Stage 1's 93 fills and Stage 3's 11,718 fills recomputed from genuine fitting rows. Tests cover missing flags/age and held-out exclusion. |

Independent reports with source-line anchors are retained as `data-review.md`,
`partition-review.md`, `stage3-review.md` and `results-review.md`. These reports
describe the initial audited revision; their release findings are resolved above.

## Tests and actual prediction replay

Interpreter: Windows Python 3.12.10, NumPy 2.2.6, pandas 2.2.3, sklearn 1.6.1,
XGBoost 3.0.0, GeoPandas 1.0.1, Shapely 2.1.0. No dependency installation/substitution.

| Command in isolated export | Result |
|---|---|
| python3.12.exe -B IPCCHGeoRFExperiment/test_contracts.py |33/33, no skips |
| python3.12.exe -B IPCCHGeoRFExperiment/test_stage1_contracts.py |31/31, no skips |
| python3.12.exe -B IPCCHGeoRFExperiment/test_stage3_contracts.py |30/30, no skips |
| python3.12.exe -B IPCCHGeoRFExperiment/test_report_contracts.py |28/28 |
| python3.12.exe -B tests/test_baseline.py, extracted baseline cwd |11/11 |

All exit codes 0; exact invocations/cwd/timing and full logs are in
`final-test-results.json` and `final-*-tests.log`. The actual-helper test initially
skipped in the fresh export; after extracting the verified ZIP into the scratch
run layout it passed, included in the final 30/30 result.

For each horizon, replayed the first and last nonempty MAIN target fold:
h1=2023-02/2025-10, h3=2023-04/2025-10, h6=2023-07/2025-10,
h12=2024-01/2025-10. Fresh pooled RF/local RF/XGB fits used saved, independently
verified feature matrices and frozen assignments. All probabilities and hard
predictions matched saved rows exactly (maximum probability difference 0).
Effective XGB configuration was checked on every replayed booster.
Details: `replay-results.json`. This is eight selected folds, not all 126 refits.

An additional exact-rational source-label check is in `fraction-target-results.json`.
Its initial comparison used YYYY-MM keys against ledger YYYY-MM-01 strings;
normalizing only the key representation resolved that checker error. No target
or source value was changed.

Verbatim audit-script snapshots are in `scripts/`. They are evidence of the review
method, not a new supported application interface. Execution layout was a scratch
root containing the fixed Git export plus those scripts; some scripts also record
the original Windows source/run/scratch paths. To repeat them elsewhere, recreate
that layout and adjust only those input/output locations. The supported portable
entrypoints remain the experiment README's tests, pipeline and reporter.

## Task closure and retained limitations

The task is archived, marked completed, and no longer active. Commit 0b9e61a contains
the archived PRD/design/plan/decisions/closure record; 3addf9a contains implementation.
The historical task.json was ignored at closure, so its later inclusion does not
retroactively create a controller-bound completion snapshot. This review establishes
current release readiness from committed requirements/source plus hash-bound local
artifacts; it does not claim a historical controller close-audit certificate.

`manifest.json` records `stage3_complete`; `reports/report_manifest.json` separately
records completed reporting. Both were verified. Neither original manifest was
rewritten to claim that later checks occurred during the original run.

Stage3 estimators were not serialized by the original design. All-fold source,
keys/fills/probability-route checks plus eight exact replays support reproducibility;
we did not replay every forest/booster or relearn Stage 1. The 11 baseline tests cover
F1/q/checkpoint behavior; actual Stage 1 state/logs establish the rejected split.
The original run verifies XGB effective configuration on its first nonempty fold;
the fixed factory serves all folds, and this review additionally checked eight.

Published timing, upstream covariate processing and true area identity remain
unverified. Geometry footprint changes, isolated polygons, sparse-history limits
and country-composition-only uncertainty remain in README. Main F1 tables are
copied as reviewed summary artifacts; full row-level outputs remain local.

GitNexus detect_changes was available during this review but returned no indexed
symbols for the 88 changed files in the original diff. This is an index-coverage
limitation, not evidence of zero impact; actual source/call paths were inspected.
The review corrections change no production function, class or method.
