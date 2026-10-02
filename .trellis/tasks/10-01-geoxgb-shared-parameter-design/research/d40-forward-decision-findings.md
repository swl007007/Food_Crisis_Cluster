# D40 forward decision-rule feasibility: findings (2026-10-02; zero model fits)

**Scope:** the 21 verified D38 `rows_E3.csv.gz` files only (producer 2d4fe4e; input hashes equal D39). Per arm (original, anchored), a single crisis-mass threshold `s = p3 + p4or5 ≥ τ` was learned only from same-H, same-arm saved E3 rows with target month U < O. The rule was frozen in the spec; the planning commit 6efbb47 preceded the run.
- Probabilities and Brier are unchanged.
- No new fits, calibrators, endpoint adoption, arm selection, Stage 2/3 or 2021+ data.
- This is conditional development evidence (D24 retrospective G selection, exposed overlapping folds).

## Evidence

- **Supervisor script:** `research/d40_forward_decision.py` (sha256 `a76f561d…`).
- **Outputs:** `research/d40_summary.json` and `research/d40_thresholds.json`, line endings normalised; original sha256 `b90f4f99…` and `1decee01…`. Originals in `C:\Users\swl00\geoxgb_runs\d40-forward-decision-20261002\`.
- **Bulky per-row policy results:** kept external, `C:\Users\swl00\geoxgb_runs\d40-forward-decision-20261002\policy_rows.csv.gz`, sha256 `ceaf8b56ef9e8c5b78f485dc6bb315439c12badabc0da3018b35d9187eda2abd`.
- **Supervisor self-checks:** largest-τ exact tie; brute-force fixture; invariance under mutation of current or future truth; 15 fallbacks unchanged. The supervisor also confirmed that all 227,016 policy keys (root, arm, area, target_month) are unique.
- **Executor independent check:** `research/d40_executor_check.py` (sha256 `ea7d47de…`) → `research/d40_executor_check.json` (`7b13c9f4…`). Exit 0; 582 checks, 0 issues. It reads the 21 D38 frames itself and does not import or rerun the main script.
  - **Coverage:** strict U < O source lists, dates, hashes and eligibility for all 21 roots.
  - **Thresholds:** all 12 recomputed by a different algorithm (ascending distinct-score scan, `searchsorted` counts, exact `Fraction` F1, ties to the largest τ). τ and its float hex, source F1, source confusion and counts are identical.
  - **Scores:** all 42 root-arm cells, all-key and matched, including persistence; the 15 fallbacks per arm equal argmax exactly.
  - **Also:** Brier unchanged; pooled all-21 and eligible-6 blocks; every one of the 227,016 rows in `policy_rows.csv.gz`.
  - The script contains a redundant first pass over the policy rows that only counts them; it does not affect any check.

## Results

**Eligibility:** 6 roots qualify (H4 2019-10, 2020-02 and 2020-06; H8 2020-02 and 2020-06; H12 2020-06). The other 15 fall back to argmax.

**Thresholds:** all lie between .285 and .409. Source F1 values are in-sample selection values, not validation.

| Root | Original τ | Anchored τ | Sources |
|---|---|---|---|
| H4 2019-10 | .408973 | .366098 | 3 |
| H4 2020-02 | .408211 | .366098 | 4 |
| H4 2020-06 | .313662 | .362108 | 5 |
| H8 2020-02 | .377517 | .284867 | 3 |
| H8 2020-06 | .305982 | .289038 | 4 |
| H12 2020-06 | .404593 | .374201 | 3 |

**Six eligible roots, matched keys (descriptive for this subset only):**
- **Original:** argmax .596973 → policy .583611 (3 better / 3 worse).
- **Anchored:** argmax .598054 → policy .603584 (4 / 2).
- **Persistence on the same keys:** .587843. The anchored argmax already beats it before any policy.

**Per-root policy − argmax (matched F1):**

| Root | Original | Anchored |
|---|---|---|
| H12 2020-06 | +.0224 | +.0091 |
| H4 2019-10 | +.0111 | +.0226 |
| H4 2020-02 | −.0050 | +.0121 |
| H4 2020-06 | −.0704 | −.0424 |
| H8 2020-02 | +.0271 | +.0826 |
| H8 2020-06 | −.0471 | −.0332 |

**All 21 roots including fallbacks, matched:** original .551516 → .548873; anchored .562155 → .565785. Persistence is .586406.

## Reading (supervisor decision)

**Not adopted.**
- The all-21 aggregate remains below persistence, under either arm, with or without the policy. (This is the aggregate over all 21 roots, not a statement that each root is below persistence; clarified 2026-10-02 during D49.)
- The anchored arm's eligible-subset argmax already beat persistence before the policy.
- The changes are unstable: H4 and H8 2020-06 worsen materially in both arms, while H8 2020-02 improves.
- This is decision-policy feasibility evidence only, not a resolution of partition overfitting, and not an endpoint change.
