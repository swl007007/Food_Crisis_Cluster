# D42 old-vs-current map common refit: findings (2026-10-02)

**Scope:** 12 targets. For each, the latest same-H D34 Brier map with U < O and the current map were both partitioned by `spatial_partition_id` on the current root's fitting rows. Each region meeting FIT_SUPPORT was refitted once from the current root with L1.
- **Arms on the same C/E3 keys:** root, D35 global+20, current_map_refit and old_map_refit.
- **Fits:** 175 regional L1 fits; no new root, G or global+20 fits.
- **Not done:** no consensus, E4, Stage 3 gate, Stage 2/3 or 2021+ data.
- **Limits:** the 12 targets are an overlapping, repeatedly developed subset, not comparable to the 21-root aggregates. The two maps differ in region count, membership and coverage. The old map also encodes the old root's error geography from its search. So differences cannot be attributed to map age alone, and the result does not show that learned geometry beats arbitrary partitions.

## Evidence

- **Producer:** `65b07334f74dc160281f70f95c2624acf4b333ed` (`scripts/stage1_map_transfer.py`; 106 tests OK after commit). Planning commit 3cd4bc7.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d42-map-transfer-20261002`. Exit 0, 133 s, 12/12 gates, 175 fits. Log `C:\Users\swl00\geoxgb_runs\d42-run.log` (sha256 `e82cb326…`).
- **External artifacts:** per-pair region UBJ/JSON files and the `rows_C/E3.csv.gz` row files stay external, referenced by the run's `completion.json` (`bc69d71f…`), `gate.json` (`2e6f0611…`) and `identity.json` (`8d9fcfa5…`).
- **Summary:** copied as `research/d42_summary.json`, line endings normalised; original sha256 `610d7f4c…`.
- **Supervisor input reference:** `research/d42_input_check.py` (sha256 `e7547e5d…`) → `research/d42_input_check.json` (`c353e465…`). It holds all 12 pairs' region pools, digests, support and coverage. The executor confirmed that the run's 175 regions match it exactly in support, eligibility, map hash and fit-key digest.
- **Supervisor independent check (PASS):** `research/d42_independent_check.py` (sha256 `8fb076c1…`) → `research/d42_independent_results.json` (line endings normalised; original `89551a77…`). No production imports.
  - 3,326 checks with no issues.
  - All 175 model prefixes, pools and coverage routes.
  - Raw probability replay on 181,707 keyed C/E3 rows for the four arms (726,828 probability rows).
  - Per-pair all-key and matched scoring, plus pooled all-key and per-H scoring.
  - The checker's first pass had a float32 Brier-sum discrepancy. It was fixed by matching the report's explicit float64 arithmetic; no tolerance was weakened and no production code was edited.

## Results (E3, 12-pair subset only)

**All keys (65,226 rows), pooled:**

| Arm | Crisis F1 | Pooled row-weighted crisis Brier |
|---|---|---|
| Root | .579422 | .104539173 |
| Global+20 | .579555 | .104581077 |
| Current-map refit | .578609 | .104446158 |
| Old-map refit | .579184 | .104429508 |

The runner's `mean_crisis_brier_all` is a different quantity: the mean of per-pair Brier (.104645 / .104686 / .104546 / .104532).

**Fold-mean deltas (per pair, then averaged):**
- **F1 vs root:** global+20 +.000199, current −.000974, old −.000256. Old minus current: +.000718.
- **Brier vs root:** current −.0000991, old −.000113.

**Persistence-matched keys (64,521 rows):**

| Arm | Crisis F1 | Brier |
|---|---|---|
| Root | .581410 | .104563267 |
| Global+20 | .581541 | .104604647 |
| Current-map refit | .580591 | .104469235 |
| Old-map refit | .581171 | .104452403 |
| Persistence | .598496 | (one-hot) .156398692 |

**By horizon, pooled F1 (root / global+20 / current / old):**
- H4: .639703 / .639126 / .637604 / .637917
- H8: .553919 / .554235 / .555169 / .554369
- H12: .499789 / .500946 / .497481 / .500950

**Coverage:** the share of E3 rows routed to root is .013476 for the current map (all `s-1`) and .029482 for the old map (`s-1` 1,074, missing 849). No region fell back for insufficient support. Per-pair region counts and F1 values are in the D42 spec.

**C (descriptive; the old map may have seen current C labels):** pooled F1 .673280 / .673656 / .680221 / .679218.

## Reading (supervisor decision)

- Neither refitted-map arm is adopted.
- E3 effects are weak and heterogeneous across pairs and horizons; every arm stays below persistence on this subset.
- This neither dismisses partitioning in general nor shows that Stage 1 overfitting is solved.
- New fits stop here.
