# D54 / A28: zero-fit fixed-policy local-increment contrast (completed; no policy adopted; supervisor verification passed)

2026-10-02. A separately bounded diagnostic under the user's delegated Stage 1 research authority. Approved by the supervisor with option (a) of §5 and four clarifications (selection disclosure, what the transform preserves, meaning of "gates", Brier normalisation); the planning commit precedes any code, and real rows are scored only after a separate supervisor release. Same active task, executor, audit run and base. After this result: an explicit stop and a research decision memo; no automatic next variant.
- **Unchanged:** the primary crisis-F1 endpoint (four-class argmax → code ≥ 2), the four-class main contract and the final criterion.
- **Out of scope:** fits, model loads, new matrices, map learning, re-search, relearning or changing partition acceptance / local-enablement gates (input validation is still required), half-local arms, any alpha / threshold / blend grid, new coefficients, Stage 2/3, final evaluation, close.

## 1. Question and what it is not

D41 compared the saved full Brier-local (frozen D34 maps and routes) with the root at the original operating point: E3 matched F1 .552179 vs .551516. D38 defined a fixed persistence preference, the post-hoc transform `p_post ∝ p · q` with q = .625 for the origin class and .125 for each other class (uniform q, so no change, when the origin is missing). It was applied to the root only.

D54 asks: **after the same fixed D38 transform is applied to both the saved root and the saved full-local probabilities, do the existing local increments change E3 crisis decisions for better or worse, relative to the transformed root and to persistence?**
- The transform applies the same prior reweighting to both arms and changes no tree. Only the difference of class log-odds (margin vectors up to an additive normalisation) between full and root is preserved; probability differences post(full) − post(root) are not invariant and may change. On saved float probabilities this is not a byte-exact margin replay (as D41 noted for its half arm).
- Zero new fits and zero free parameters: q is the constant already fixed in D38. This interaction itself was chosen after exposed development results (D38–D53), so D54 is not pre-registered and is not immune to selection.
- **Not** partition training on an anchored base and **not** evidence that such training would work. Post-hoc outputs are diagnostic only; raw four-class outputs are retained as the primary model outputs.

## 2. Frozen inputs (read only)

- **D41 rows:** `C:\Users\swl00\geoxgb_runs\d41-local-shrinkage-20261002\rows.csv.gz`, sha256 must equal `rows_sha256` in committed `research/d41_summary.json` (`a36104f6…`). 321,047 C + E3 rows (C 207,539; E3 113,508) with root and full four-class probabilities, `branch_id`, `routing`, `route_type` (local / zero_increment). D41's `persistence_code` is not used as authority.
- **Authoritative origin codes and original probabilities:** the D52 `rows_C.csv.gz` and `rows_E3.csv.gz` (42 files), hash-bound through committed `research/d52_completion.json` (external `completion.json` byte-identical first). Read with `float_precision="round_trip"`.
- **D38 reconciliation (E3 only, option (a)):** D38 `rows_E3.csv.gz` (21 files), sha256 equal to `input_hashes` in `research/d39_probability_diagnostic.json` and `research/d40_summary.json`. D38 `rows_C` are not read; C is still fully bound and aligned through D41 + D52, and no D38 C replay is claimed.

## 3. Alignment gates (fail closed; a discrepancy stops, never a silent fix)

1. Hashes as above; 21 roots, 7 per H; keys `(root, part, area, target_month, horizon)` unique and set-equal between D41 and D52 for C and E3.
2. On every joined row: D41 `p_root_*` equal D52 `p_original_*` exactly, and truth labels equal.
3. Origin codes from D52 only: `persistence_code` ∈ {0, 1, 2, 3} or missing; record any D41 vs D52 code disagreement as a stopping discrepancy.
4. Missing origin is neutral: post = raw for both arms on those rows.
5. Zero-increment identity: on `route_type == zero_increment` rows, `p_full == p_root` exactly, hence post(full) == post(root).
6. Reconciliation: raw root and raw full confusions and probability rows equal D41 exactly (per root and pooled); post(root) on E3 equals D38's saved `p_posthoc` (argmax and probabilities to 1e-12) and D38's per-root post-hoc confusions. The normalised crisis Brier used here differs slightly from D41's legacy raw-mass Brier, so old unnormalised Brier values are not required to match.

## 4. Metrics (descriptive; no significance or adoption rule)

- **Arms:** raw root, raw full, post(root), post(full); persistence on matched keys only.
- **Primary:** E3 matched exact-origin crisis F1 with TP/FP/FN per arm; per H pooled and mean-fold; fold wins for post(full) vs post(root), post(full) vs persistence, and raw full vs raw root (the D41 reference).
- **Local − root change counts:** TP/FP/FN changes for raw and post, split by `route_type` (zero-increment rows must show zero change).
- **Probability:** normalised crisis Brier on s = (p2 + p3)/Σp for all four arms; persistence one-hot Brier as a reference only.
- **All-key E3:** model arms separately with missing-origin counts; nothing compared with persistence on all keys.
- **C:** in-window interpolation context only.
- All 21 roots kept.

## 5. Objections and open points for review

1. **D38 `rows_C` hashes are not in a committed record.** Resolved: option (a) — D38 post-hoc reconciliation on E3 only; C stays bound through D41 + D52 and is context only.
2. **Redundancy check:** D41 measured the local increment at the original operating point; D38 measured the transform on the root only. The interaction (the same increment at the transformed operating point) has not been computed. It is nonredundant but narrow: decisions can change only where post(root) and post(full) straddle a class boundary that matters for code ≥ 2.
3. **No outcome prediction** is made in advance.

## 6. Implementation and order

- **Script:** one standalone task-research runner, `research/d54_fixed_policy_local_contrast.py`, numpy/pandas/stdlib only; synthetic checks for the transform (including missing-origin neutrality and zero-increment identity), key-join failure, and reconciliation failure stopping the run. Exact bytes committed before the run; script git blob equal to HEAD.
- **Output (frozen):** `C:\Users\swl00\geoxgb_runs\geoxgb-d54-fixed-policy-local-contrast-20261002`, no overwrite: compact summary, per-root confusions, change counts, identity and completion JSON.
- **Order:** supervisor review → planning commit (GitNexus attempt) → native implement and native check (each ≤ 10 min) → producer commit → supervisor release → single zero-fit run → factual report → independent supervisor verification → explicit stop and research decision memo.

## 7. Result and research decision (2026-10-02)

- **Producer and run:** producer `6d5619b`; run `C:\Users\swl00\geoxgb_runs\geoxgb-d54-fixed-policy-local-contrast-20261002`, exit 0 in 6 s, zero fits; all gates passed; D41 and D38/D39 E3 reconciliations exact. Executor factual record `16ee1f4`.
- **Supervisor verification:** PASS (`research/d54_supervisor_check.py` / `research/d54_supervisor_results.json`; 8,454 checks).
- **Findings:** [research/d54-fixed-policy-local-contrast-findings.md](research/d54-fixed-policy-local-contrast-findings.md). **Decision memo:** [research/stage1-research-decision-d54.md](research/stage1-research-decision-d54.md).
- **Decision:** no new prediction policy adopted. E3 matched pooled post root .5839738224 → post full .5846255194 vs persistence .5864055300 (mean-fold .5822054687 → .5831214943 vs .5861777298); H4 no decision changes, H8 negative, H12 positive but below persistence; normalised Brier slightly worse. Explicit research stop on adjacent current-recipe variants; no automatic D55. Stage 1 remains open.
