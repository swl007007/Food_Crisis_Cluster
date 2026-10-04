# P6 learn-map checkpoint: p6-formal-20261004b (STOP before predict)

Release: `p6-release.md` (supervisor P6 release through the learn-map
checkpoint). Restart after the R41 file-lock stop: `p6-restart-evidence.md`,
`p6-file-lock-restart.md`. Implementation code identity **6798df2** (HEAD
44951cc, package and configs unchanged). No predict, no report, no retuning, no
lifecycle close. Audit run 7ced754e… / base 6c98f73 untouched.

## Identity

| Item | Value |
|---|---|
| preflight | p6-preflight-20261004b passed, report sha256 ca5d72e4… |
| prepared manifest | d4cb9bfd773e84c2… (14 artifacts byte-identical to the failed run a and dev run) |
| runtime | Windows Python 3.12.10, xgboost 3.0.0, numpy 2.2.6, lock ipcch-geoxgb-runtime-v1, quartet_code_sha256 3ec3dbbf… |
| schema | ipcch-geoxgb-rich561-ge020-v1 / deca3a09…; contract ipcch-geoxgb-contract-v1.0 |
| stage1-summary.json | sha256 e90bae9da897d65b34a59254cbf64ff66ea7b2745a55a025eea715e3c5da3488 |
| learn-map | exit 0, 13 m 34 s wall (812 s internal), log runs/p6-logs/learn-map-b.log |

## Stage1 fits and requests

429 requests = 429 fits, 0 hits, **0 failed**. Per H the 4 root_global quartets
(G1–G4) are shared by the two L recipes. Child_local quartets: H1 105, H3 109,
H6 98, H12 101. That makes 1716 scalar boosters, all persisted.

## Frozen maps (all H: accepted split = true, learned areas 3264 = map rows)

| H | Winner | Selection F1 (S half) | Runner-up (Δ) | Terminal regions | Accepted splits | map sha256 | selection sha256 |
|---|---|---|---|---|---|---|---|
| 1 | G1L2 | 0.85099 (5488/6449) | G3L2 0.84885 (−0.00214) | 9 | 8 | 77ef44da88ba551f… | f70e0e747b6d4311… |
| 3 | G3L2 | 0.84291 (2774/3291) | G2L2 0.84053 (−0.00238) | 7 | 6 | 16f354ff369c2663… | 4b13f12a8893193b… |
| 6 | G4L2 | 0.83683 (5544/6625) | G3L2 0.83666 (−0.00017) | 6 | 5 | 04ffe8363e99fa88… | 48c444aa48c59140… |
| 12 | G2L2 | 0.83006 (5544/6679) | G1L2 0.82719 (−0.00287) | 9 | 8 | f61be8bd90c9555d… | 411db67182fb828b… |

Selection status `selected` for every H, with no undefined candidates. All
four winners use L2. The full rankings, every candidate's terminal regions and
accepted splits, the per-node decisions (outcome, eligibility, selected
children, base/best F1, sizes after smoothing, switched areas) and per-region
connectivity (areas, components, largest component, isolated areas and the
full component-size list) are in
`evidence/p6-formal-20261004b-map-diagnostics.json`, read from the saved
artifacts by `evidence/stage1_map_diagnostics.py`. That script also verifies
each map_sha256 against the file.

Accepted nodes: H1 r, r0, r1, r00, r01, r11, r000, r011; H3 r, r0, r1, r01,
r10, r011; H6 r, r0, r1, r10, r11; H12 r, r0, r1, r00, r01, r10, r000, r001.
Other nodes stopped on `rejected_gate` (no strict gain) or
`rejected_no_eligible_child` (R28/R29 support).

Connectivity (regions: areas / components / isolated):
- H1: r0000 231/161/139, r0001 199/45/25, r001 332/147/114,
  r010 388/9/1, r0110 206/7/1, r0111 247/14/8, r10 726/223/183,
  r110 398/114/89, r111 537/136/109
- H3: r00 765/288/225, r010 377/9/1, r0110 217/5/0, r0111 251/19/12,
  r100 302/119/89, r101 429/112/95, r11 923/239/191
- H6: r00 779/292/229, r01 829/21/10, r100 297/126/96, r101 426/109/94,
  r110 388/116/90, r111 545/136/108
- H12: r0000 226/146/118, r0001 189/73/55, r0010 173/88/69,
  r0011 180/55/41, r010 438/9/2, r011 395/19/11, r100 292/141/111,
  r101 434/95/77, r11 937/239/194

## Stage3 request enumeration and distinct-fit budget (no fitting)

`evidence/stage3_enumeration.py` (pinned runtime) produced
`evidence/p6-formal-20261004b-stage3-enumeration.json`. Counts are quartets;
scalar fits are 4 × quartets. Historical locals are exact: each is requested
iff the region has validation keys at gate date U and passes R28 fit support on
[V−35, V]. Current locals are an **upper bound**: support and the ≥3-date
requirement are data-deterministic, but the strict F1 gain > .01 depends on
predictions.

| H | Scheduled / scored folds | Current global | Historical global | Historical local (exact) | Hist. unsupported → global fallback | Current local (upper) | Distinct global | Distinct local exact | + current-only upper | Scalar fits exact / upper |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 39 / 36 | 36 | 216 | 931 | 100 | ≤76 | 43 | 183 | ≤9 | 904 / 940 |
| 3 | 37 / 34 | 34 | 204 | 736 | 4 | ≤82 | 43 | 143 | ≤16 | 744 / 808 |
| 6 | 34 / 31 | 31 | 186 | 614 | 54 | ≤59 | 43 | 129 | ≤17 | 688 / 756 |
| 12 | 28 / 25 | 25 | 150 | 723 | 32 | ≤37 | 43 | 157 | ≤21 | 800 / 884 |
| **Total** | 138 / 126 | 126 | 756 | 3004 | 190 | ≤254 | 172 | 612 | ≤63 | **3136 / 3388** |

- Requests: ≤ 4140 quartet requests in total, served by R48 exact-identity
  reuse. The distinct-fit budget is 3136 exact, at most 3388 scalar fits,
  against the R48 Stage3 ceiling before reuse of 80920.
- Gate region-folds with deterministic gate support: H1 110/324, H3 98/238,
  H6 72/186, H12 62/225. Only these can adopt a current local.
- Unscored folds: in every H, main 2024-12, 2025-11 and 2025-12 have no valid
  target (status `no_valid_target`, matching the prepared calendar), so 110
  main + 16 supplementary folds are scored.
- Validation of the enumerator against production:
  `evidence/stage3_enumeration_selfcheck.py` runs the production learn-map and
  predict on the synthetic adopted-local world (tests/test_p5_e2e, west_areas
  = 56) and compares against the production Stage3 request ledger. Current
  global 8, historical global 48, historical local 96 and distinct global 17
  match exactly; production current locals (16) are within the upper bound.
  Status passed (`evidence/stage3_enumeration_selfcheck.json`). No project
  data was fit for this check.
- Expected Stage3 wall time, from the Stage1 rate (~1.9 s per quartet on
  larger F-half pools): at most about 30 min for ≤ 847 distinct quartets.

## Limitations and observations (no action taken; recipes fixed)

1. Run a, `p6-formal-20261004`, is a preserved R41 stop (WinError 32 at the
   model-dir rename; Dropbox is the suspected, not proven, lock holder). Run b
   ran with Dropbox sync paused. The no-fit probe passed before it, and
   learn-map then completed with 0 failures. Stage3 should also run with sync
   paused.
2. Recipe selection margins are small: H6 winner vs runner-up Δ 0.00017; the
   other H's 0.0021–0.0029. These are single-split S-half F1 values; R44 picks
   the winner deterministically and no tie rule was needed.
3. Several accepted gates have tiny positive gains, as R-contract `gain > 0`
   allows. Examples: H1 r11 0.98840 → 0.98866; H1 r000 base 1/18 → 19/288.
4. Terminal regions are spatially fragmented. Most regions have tens to
   hundreds of components and many isolated areas (e.g. H1 r10: 223
   components, 183 isolated). This is what the single-column scan plus three
   synchronous 4/9 smoothing rounds produce; the contract imposes no
   connectivity requirement. Reported only as a diagnostic.
5. The selection F1s are in-sample Stage1 S-half values, not evaluation
   results. No Stage3/out-of-sample number exists yet.
