# D51 / A25: history + calendar (78-feature) root ablation (approved)

2026-10-02. Supervisor selection under the user's delegated Stage 1 research authority, after D50. Approved after supervisor review with two precision edits (projection equality wording; coverage-metadata limitation); the planning commit precedes any code, and real fits wait for a reviewed, committed producer and a separate supervisor release. Same active task, executor, audit run and base.
- **Endpoint:** the primary crisis-F1 endpoint (four-class argmax → code ≥ 2) and the final criterion are unchanged.
- **Feature contract:** the fixed 162-feature contract is revised for this one diagnostic arm only; it is not a new default. No expert inputs.
- **Out of scope:** Stage 2/3, final-period reads, full 648, audit close, default adoption, partition-viability claims, other feature subsets or grids.

## 1. Question and what it is not

D50 found within-exact-phase ordering throughout phase 3 and weak, date-unstable ordering in phase 2, especially at H8. Earlier depth, weight, recency and pool changes did not improve on persistence. D51 asks: if each root is re-fitted on only the schema's outcome-history and known-calendar blocks, how do forward E3 crisis F1 and within-phase ordering compare with the full-162 original root and with persistence?

**What the arm is:** exactly the 75 `history_blocks` features plus the 3 `known_calendar` features (`target_year`, `target_month_sin`, `target_month_cos`), 78 in total, in global schema order. Removed: `static_sources` (28) + `dynamic_sources_at_origin` (41) + `legacy_covariate_derived` (15) = 84. The subset is defined by schema blocks only; no feature is chosen or dropped after inspecting results.

**What it is not:**
- This is a **joint removal** of 84 features. The outcome cannot be attributed to any single variable or block, nor to a causal overfitting mechanism.
- G, rounds and `colsample_bytree` stay frozen at values chosen with 162 features. With 78 features the effective per-tree feature search changes (0.8 × 78 vs 0.8 × 162 columns), so this is disclosed and not re-tuned.
- A smaller FIT-to-C or FIT-to-E3 gap alone is not evidence of a better model. Results are read against both the original root and persistence.
- No candidate, date or parameter selection, and no extrapolation to partition viability.

**Pre-spec metadata check:** `research/early-feature-coverage.json` records the first finite month as 2010-01 for all 69 raw static and dynamic sources. This shows only that some observations exist from 2010-01; it does not establish comparable coverage across countries or eras, and missingness can still differ by country and era. No outcomes were scored for this check.

## 2. Frozen design

- **Pairs:** the 21 D34 pairs, H ∈ {4, 8, 12} × T ∈ {2018-06, 2018-10, 2019-02, 2019-06, 2019-10, 2020-02, 2020-06}.
- **Unchanged:** the full ORIGINAL FIT rows, row order and labels, **including missing-origin rows** (the D48 known-origin mask is not inherited; that would change two factors); W59; the four-class `multi:softprob` objective; G = {4: G1, 8: G4, 12: G2}; rounds 200 / 400 / 400; seed 42; unweighted; no margins; no local trees; NaN handling.
- **The only change:** the fresh root's FIT, C and E3 matrices are column projections of the rebuilt full-162 matrices onto the 78 selected schema columns.
- **Arms:** the saved full-162 original root; one fresh 78-feature root per pair (at most 21 fits); exact-origin persistence.

### Feature projection

- The selected indices are derived from the schema groups (`history_blocks` values + `known_calendar`), mapped to positions in `ordered_features`, sorted to preserve global schema order, and asserted to number 78. The 84 removed names equal `static_sources` + `dynamic_sources_at_origin` + `legacy_covariate_derived` exactly.
- Record the exact selected names, the index map and the sha256 of the schema file.
- The original-root replay and persistence use the **full** matrices. Persistence reads `hist_phase_o00` at full-schema index 87. In the 78-column matrix that feature sits at a different index; index 87 is never used on the projected matrix.
- No package or global schema edits.

## 3. Gates and checks (fail closed)

1. **Original replay first, on full-162 data:** D34 pinned acceptance (7b2bf6f) and the D37 `gate_root`, with the original matrices and checkpoints untouched. All 21 roots must pass before their fit.
2. **Projection:** the 78 names and positions match the schema groups; the projected matrices equal the full matrices' selected columns by exact array equality with `equal_nan=True` (NaN payloads need not be bitwise identical) for FIT, C and E3, with the input dtype and row keys/order preserved.
3. **Fit:** one `nx.fit_global(X_fit[:, idx], y_fit, plan.G_CONFIGS[g])`, unweighted and with no margin. Assert:
   - params and rounds equal `booster_params(G)`;
   - `base_score` 0.5, 4 classes, and the model's recorded feature count is 78;
   - no weight or margin block;
   - the FIT keys and labels equal the original FIT keys and labels.
4. **Reload:** the saved UBJ reloads exactly on FIT, C and E3.
5. **Narrow tests** (as a `--selftest` or task-research tests, no package tests unless package code changes):
   - feature projection: names, order, count 78, exact column equality on a synthetic matrix;
   - a synthetic four-class fit-and-reload round trip on 78 columns;
   - a failed gate means no fit.

   Synthetic test fits are separate from the 21-real-fit budget.
6. **Original-arm consistency:** the complete 21-root inventory equals D49/D50; per root the matched n and excluded count equal D49 (`per_root[<root>]["n"]`, `["excluded_missing_origin"]`); the original arm's matched E3 confusions equal the D49 per-root original argmax confusions (`research/d49_summary.json`); and its D50-style cells equal `research/d50_summary.json` original cells (n, P, N, AUC to 1e-12, null reasons). Same root, keys and score, so any difference is a pipeline error.
7. **Stopping:** stop at the first error; no silent reruns; input snapshots read with the ≤ 2020-12 filter only.

## 4. Metrics

- **Primary:** E3 crisis F1 (argmax → code ≥ 2) on the matched exact-origin keys, for the original root, the 78-feature root and persistence, with TP/FP/FN and crisis-call share. Per pair, then per H pooled and mean-fold, with fold wins against both the original and persistence (descriptive).
- **All-key E3:** original and 78-feature arms reported separately, with missing-origin counts. Nothing is compared with persistence on all keys.
- **Alongside:** fixed-four macro-F1, unweighted crisis Brier and four-class log loss (persistence: crisis F1, macro-F1 and one-hot Brier only); within-root crisis AUC/AP on s = (p2 + p3)/Σp, mean-fold only. Reuse the existing scoring helpers.
- **D50 cells:** per root, per exact origin phase (`persistence_code` 0 / 1 / 2 / 3), E3 within-cell crisis AUC for both arms, with n, P, N and null reasons (`empty`, `no_positive`, `no_negative`). All four phases are reported; per-H means over valid folds only, with `n_valid / 7`. No pooled AUC.
- **FIT and C:** reported as role context (in-sample and in-window interpolation).
- No significance tests and no success thresholds.

## 5. Evidence and implementation

- **Script:** one task-research runner, `research/d51_history_calendar_root.py`, borrowing minimal existing helpers (D34 acceptance, D37 gate/rebuild/persistence, D46 scoring, `nx.fit_global`). Its exact bytes are committed before the run, and the script git blob must equal HEAD.
- **External outputs** under `C:\Users\swl00\geoxgb_runs\geoxgb-d51-history-calendar-root-20261002\`: saved four-class probabilities, truth and argmax with keys for FIT, C and E3 (both arms); the fresh UBJs and fit records; the selected feature names, index map and schema hash; fitting keys, labels and config; source gate and reload records; compact summary and identity JSON.
- **Order:** supervisor review → align pointers and commit the planning (GitNexus attempt) → native implement and native check (each ≤ 10 min) → producer commit → supervisor release → single run (at most 21 fits) → factual report → independent supervisor check → stop.
