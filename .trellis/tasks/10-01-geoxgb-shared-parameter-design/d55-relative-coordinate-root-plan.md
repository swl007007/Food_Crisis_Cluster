# D55 / A29: relative-coordinate root contrast (approved; implemented with synthetic checks only; NOT run — task closed incomplete by user cancellation)

2026-10-02. Planning authorised by the supervisor under the user's delegated Stage 1 research authority, as one distinct representation mechanism. This supersedes the D54 research stop only for this written contrast; adjacent grids, Stage 2/3, full 648, final evaluation and close stay stopped. Approved by the supervisor with six clarifications (open points resolved, decode-first tie example, exact screen arithmetic, origin source and D52 alignment, explicit probability metrics, encoded-booster labelling); the planning commit precedes any code, and no fit runs until a separate release. The D54 decision memo carries an addendum pointing to this specifically authorised exception. Same active task, executor, audit run and base.
- **Unchanged:** the external fixed four-class probability axis, the primary endpoint (four-class argmax → code ≥ 2, crisis F1) and the final criterion. Diagnostic only; no adoption.

## 1. Mechanism and limits

**Encoding.** For each labelled row in each role (FIT, C, E3), reference k = the exact-origin code from that role's own origin feature (`rr.persistence_codes(hist_phase_o00)`, phase 5 → code 3) when known — never from truth. The known-mask is saved separately from k, so the missing-origin convention k = 0 can never be mistaken for an observed IPC 1. The training target is z = y XOR k (bitwise on codes 0–3). When the origin is missing, k = 0 is used **only as a coordinate convention** (identity permutation): it is not an imputation, not persistence and not a feature; no row is removed and `hist_phase_o00` stays NaN. The model is the existing `multi:softprob` with `num_class=4` fitted on z. At prediction, the same row's k decodes the probability columns, p_y[j] = q_z[j XOR k].

**Hypothesis.** With finite, fixed capacity (G, rounds), output coordinates shared across origin states (unchanged / within-side flip / cross-side flips) may let trees share stay/transition patterns and transfer better forward. No new information is added; with unlimited capacity the absolute-label model can represent the same functions, so any difference is representational under the fixed boosting path.

**Limits:**
- XOR categories are categorical, not severity distances. z = 3 pools IPC 2↔3 (onset and relief) with IPC 1↔4-or-5 jumps; z = 2 pools IPC 1↔3 and IPC 2↔4-or-5; z = 1 pools IPC 1↔2 and IPC 3↔4-or-5.
- Missing-origin FIT rows (exact-origin known fraction H4 .36–.80, H8 .24–.67, H12 .86–.87) stay in absolute coordinates, so relative and absolute coordinates are mixed in training. If they interact badly, the trees may spend capacity separating them through the NaN branch.
- The class balance of z is not asserted in advance; transformed support is counted and saved. If one class dominates, the loss and argmax may lean towards the reference class.
- Onset and relief drivers may not share.
- No attribution solely to overfitting; distinct from D38 (fixed log-prior in absolute coordinates), D52 (binary target) and D48 (row removal); not a test of partitions.
- Chosen after exposed development results, so it is not fresh or independent evidence. The protocol and screen below are frozen before this contrast is scored.

## 2. Frozen design

- **Pairs:** the 21 original D34 roots, H ∈ {4, 8, 12} × T ∈ {2018-06, 2018-10, 2019-02, 2019-06, 2019-10, 2020-02, 2020-06}.
- **Unchanged:** full 162 features; all original FIT keys and row order, including missing-origin rows; W59; G = {4: G1, 8: G4, 12: G2}; rounds 200 / 400 / 400; seed 42; `multi:softprob`, `num_class=4`; recorded `base_score` 5E-1 exactly as the original roots; no weights, no margins, no local trees. At most 21 new fits.
- **Labels:** original y is kept as metadata; z labels and the reference known-mask are saved and hashed alongside the y labels.
- **Decoding before scoring:** every saved raw q (FIT, C, E3) is decoded with the same row's k into p_y. Hard predictions use `np.argmax(p_y)` on the original axis (ties resolved by original column order). `argmax(q) XOR k` is never used, because with ties it can differ. Example: q = [.4, .4, .1, .1], k = 1 decodes to p_y = [.4, .4, .1, .1] with argmax class 0, while the prohibited shortcut gives argmax(q) XOR 1 = class 1.

## 3. Arms

- **Relative:** the fresh relative-coordinate root, decoded.
- **Original:** the saved D34 root (gate replay).
- **Persistence:** exact-origin `persistence_code ≥ 2`, matched keys only.
- **Reference (not an arm, no selection; E3 only):** the fixed D38 post-hoc root, computed identically from the original saved probabilities with the D38 q (.625 / .125). On E3 matched keys it must equal the verified D38/D54 values (`research/d54_summary.json` post_root), clearly marked as an existing operating-point reference. No coefficient is tuned; it is not computed on C.

## 4. Transfer screen (written before any outcome)

The relative root **clears the screen only if** its matched-E3 crisis F1 strictly exceeds **both** the original root **and** persistence at **each** H ∈ {4, 8, 12}, in **both** pooled and mean-fold aggregation — 12 strict comparisons.
- **Exact arithmetic:** pooled F1 is the exact `Fraction` from integer pooled confusion counts; mean-fold F1 is the mean of the 7 exact per-root F1 `Fraction`s. Both fractions and the 12 booleans are saved; floats are for display only. Equality never passes, and rounding cannot create a pass.
- Failure of any one comparison means the contrast does not clear the screen and the representation sequence stops (no D56).
- Passing permits consideration only; it is not main-model adoption and does not open Stage 3.
- The D38 post-hoc reference need not be beaten for the screen; it is shown for context.
- **Mandatory secondary tradeoffs** (reported, never substituted for the screen): normalised crisis Brier, four-class log loss, fixed-four macro-F1, fold wins against original and persistence.

## 5. Gates and checks (fail closed)

1. **All original gates first:** before **any** fit, for all 21 roots: D34 acceptance (7b2bf6f) and `rr.gate_root` replay; alignment of the original FIT/C/E3 keys, truth, origin codes and original probabilities with the committed hash-bound D52 rows (external D52 `completion.json` byte-equal to `research/d52_completion.json`, then all 63 row hashes); and D49/D54 E3 consistency (gate 6). Pass 2 re-gates each root, requires the FIT-key hash to match pass 1, then fits.
2. **Encoding:** k derived only from `hist_phase_o00` on each role's own matrix (FIT, C, E3), never from truth, with the known-mask saved separately; codes must agree with the D52 `persistence_code` for C/E3 and the D49 matched inventory for E3; the y → z map is checked as a bijection; missing rows use identity; neither the input matrix nor y is mutated.
3. **Fit:** one `nx.fit_global(X_fit, z_fit, plan.G_CONFIGS[g])` with the original parameter invocation (no new override); params and rounds equal `booster_params(G)`; recorded `base_score` verified as 5E-1; 4 classes; no weight or margin. FIT keys equal the original.
4. **Encoded booster labelling:** the relative UBJ is not an ordinary absolute-label root. Its fit record carries a mapping version, the origin source (feature name `hist_phase_o00`, schema position 87), the known-mask / k / y / z hashes and the decoded external class labels, and marks it diagnostic-only and not compatible with existing production consumers. No new package API or marker framework; it is never handed to local continuation.
5. **Reload:** identity is written before any fit; the saved UBJ reloads exactly; both raw q and decoded p_y are saved and reproduce exactly on FIT, C and E3.
6. **Original consistency (before any fit):** original-arm matched E3 confusions equal D49 per-root values; the D38 reference equals D54 post_root on E3 matched.
7. **Synthetic selftest** (outside the 21-fit budget): all 16 (y, k) pairs encode and decode correctly, with the crisis bit identity; missing-origin identity with no mutation; the tie case q = [.4, .4, .1, .1], k = 1 (decode-first class 0, prohibited shortcut class 1); absent transformed classes still give a 4-column softprob; a later-root gate failure causes zero fits; reload exactness.
8. **Stopping:** stop at the first error; no silent reruns; snapshots read with the ≤ 2020-12 filter.

## 6. Metrics and outputs

- **Primary (screen):** matched E3 crisis F1 with TP/FP/FN per arm; per H pooled and mean-fold; the 12 screen comparisons recorded explicitly as pass/fail.
- **Secondary:** exact raw native q and decoded p_y are kept for replay and original-axis argmax. For probability metrics, each decoded row is normalised in float64 to sum 1; both the crisis Brier (on s = p2 + p3 of the normalised row) and the four-class log loss (no clipping; stop on a non-positive true-class probability) use that normalised row. Older raw-mass metrics are not expected to be identical. Fixed-four macro-F1 and fold wins; persistence one-hot Brier as reference. No AUC/AP.
- **Context:** FIT and C (in-sample / interpolation); all-key E3 model arms and missing-origin E3 rows reported separately, with no persistence there.
- **Support:** per root, original y and transformed z class counts by known-mask, with label-date counts. No extra ranking, country or subgroup grids.
- **Outputs (external, frozen):** `C:\Users\swl00\geoxgb_runs\geoxgb-d55-relative-coordinate-root-20261002\`, no overwrite: per-root relative UBJ and fit record (params, z/y/known-mask hashes, support); keyed FIT/C/E3 rows with truth, k, raw q, decoded p_y and original probabilities; gate, identity (before any fit), summary (with the screen table), completion; run log.

## 7. Implementation and order

- **Reuse:** package helpers only — `scripts/stage1_recency_root.py` (`gate_root`, `persistence_codes`, `dev_baseline_check`), `scripts/stage1_class_weight_root.py` (`block`, `delta`, `_pool`, `_mean_fold`, adapted to the normalised probability inputs if needed), `scripts/stage1_rootconf_compare.accept_mode`, `src/model/native_xgb.py` (`fit_global`, `proba`, `raw`, `from_raw`), `src/utils/run_identity.py`. The task-research runners D51/D52 are references for structure (two-pass gating, D49/D50 consistency) but are not importable modules; their needed logic is written minimally in the new runner rather than copying D52's full orchestration. No package, schema or default edits; no new framework.
- **Script:** one task-research runner, `research/d55_relative_coordinate_root.py`; exact bytes committed before the run; script git blob and package code equal HEAD.
- **Order:** supervisor review → planning commit (GitNexus attempt) → native implement and native check (each ≤ 10 min) → producer commit → supervisor review and release → one run on frozen Windows Python (assertions on), stop at first error → factual report → independent supervisor check → findings → stop. No D56.

## 8. Resolved points

1. `k` uses `rr.persistence_codes` (phase 5 → code 3), matching every earlier persistence arm.
2. The D38 post-hoc reference is computed on E3 only.
3. The original parameter invocation is preserved; the recorded `base_score` 5E-1 is verified, with no new override.
