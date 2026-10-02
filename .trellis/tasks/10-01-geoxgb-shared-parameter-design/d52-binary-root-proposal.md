# D52 / A26: independent binary-objective root diagnostic (PROPOSAL — pending user decision; not approved)

2026-10-02. Prepared by the executor at the supervisor's request under the user's delegated Stage 1 research authority. **This is a proposal only.** It is not approved, not in any executable manifest, and nothing is implemented, trained or scored until the user answers the gate question below and the supervisor then releases a run.

## 0. Gate question for the user

D26 (`prd.md:137`) records the user's instruction "可以保留四分类概率但按照二分类评估" (keep the four-class probabilities, evaluate as binary), and fixes native four-class XGB probabilities (`multi:softprob`, `num_class=4`, fixed class axis) as the model. R11 (`prd.md:23`) and A11 (`prd.md:84`) keep a fixed four-class output axis.

**Question:** do you permit one independent, diagnostic-only arm trained with a `binary:logistic` objective (target: original four-class code ≥ 2) across the fixed 21 D34 roots (one binary fit per root), alongside, and without changing, the four-class main contract?
- **Yes:** the plan below is finalised for supervisor review, then native implement/check, producer commit and a single supervisor-released run.
- **No:** D52 is dropped; nothing changes.

Either way the four-class package, schema, models, probabilities, endpoint (four-class argmax → code ≥ 2) and final criterion stay intact. Any later move to a binary main model would need its own separate approval.

## 1. Question and what it is not

Every root in D26–D51 is a four-class `multi:softprob` booster; `binary:logistic` has not been used anywhere in the package or task research. D52 asks how a root trained directly on the binary crisis target compares, on forward E3, with the existing four-class root and persistence.

**Limits:**
- The objective changes together with capacity (4 trees per round → 1 tree per round), the Hessians that `min_child_weight` and `reg_lambda` act on, and the loss/decision geometry. G was selected for the four-class model. No pure objective causality can be claimed.
- A negative result does not show that the objective is never a bottleneck. A positive result is not adoption, transfer or an overfitting resolution.
- No prediction is made in advance about the direction of the crisis-call share.
- Exposed, overlapping development folds; no significance tests or success thresholds.

## 2. Frozen design (if approved)

- **Pairs:** the 21 D34 pairs, H ∈ {4, 8, 12} × T ∈ {2018-06, 2018-10, 2019-02, 2019-06, 2019-10, 2020-02, 2020-06}.
- **Unchanged:** the full 162 features; all original FIT rows, order and labels (including missing-origin rows); W59; G = {4: G1, 8: G4, 12: G2} with rounds 200 / 400 / 400; seed and all other G parameters; unweighted; no margins; no local trees.
- **Binary model:** standalone task-research `xgb.train` with the G parameters copied explicitly, `objective` replaced by `binary:logistic`, `num_class` and `multi_strategy` removed, and **`base_score` set explicitly to 0.5** (XGBoost otherwise estimates the intercept from the labels). Target y_bin = I[original four-class code ≥ 2]. The package `native_xgb`, `plan.py` and the schema are not edited or generalised.
- **Outputs:** native binary p ∈ [0, 1]; hard crisis iff p ≥ 0.5.

### Arms and contrasts

| Arm | Fit | Hard crisis rule | Probability score for Brier / log loss / ranking |
|---|---|---|---|
| Binary root | fresh (≤ 21 fits) | p ≥ .5 | p |
| Original, normalised mass | zero-fit | s ≥ .5, s = (p2 + p3)/Σp in float64 | s |
| Original, pipeline argmax | zero-fit (existing endpoint) | four-class argmax → code ≥ 2 | s (shared with the mass arm) |
| Persistence | — | `persistence_code` ≥ 2 | one-hot (Brier only) |

- **Contrast A (same rule):** binary p ≥ .5 vs original normalised mass s ≥ .5.
- **Contrast B (endpoint):** binary p ≥ .5 vs the existing pipeline argmax, and each vs persistence.
- A and B are reported separately. Probability scoring (Brier, log loss, ranking) is distinguished from hard decisions; the argmax and mass arms share the same score s.
- The D39 per-root `fixed_half_mass` used the raw mass p2 + p3; D52 uses the normalised s. This is noted as a different legacy definition; no flip study is added.

## 3. Metrics

- **Primary:** E3 crisis F1 with TP/FP/FN and crisis-call share on the matched exact-origin keys, for every arm and persistence; per pair, then per H pooled and mean-fold, with fold wins for contrasts A and B.
- **All-key E3:** model arms reported separately with missing-origin counts; nothing compared with persistence on all keys.
- **FIT and C:** role context (in-sample, in-window interpolation).
- **Shared binary probability scores** (binary p; original s), float64: binary crisis Brier, and binary log loss with a fixed clip `eps = np.finfo(np.float64).eps` applied **only** to log loss; clipped counts disclosed. No clipping for F1 or AUC. Persistence: one-hot Brier only.
- **Original-only reference:** the four-class macro-F1 and four-class log loss of the original root. Four-class macro-F1 is not applicable to the binary arm.
- **Ranking:** within-root crisis AUC/AP (binary p; original s), mean-fold only.
- **D50 cells:** per root, per exact origin phase (`persistence_code` 0 / 1 / 2 / 3), E3 within-cell AUC for the binary p and original s, with n, P, N and null reasons; per-H means over valid folds with `n_valid / 7`. No pooled ranking.

## 4. Gates and checks (fail closed)

1. **Original replay first, on full-162 data:** D34 pinned acceptance (7b2bf6f) and the D37 `gate_root`; the original matrices, checkpoints and probabilities untouched.
2. **Original-arm consistency before any real binary fit (required):** the original replay above and these consistency gates must finish and pass before any real binary fitting: the 21-root inventory, matched n / excluded, original argmax and persistence confusions equal D49, and original D50 cells equal D50.
3. **FIT identity:** all original FIT keys and order and the original four-class labels preserved; derived binary labels recorded with their hash.
4. **Binary fit:** params equal the copied G params except for these differences only: `objective` replaced by `binary:logistic`; `num_class` and `multi_strategy` removed; `base_score=0.5` added as an explicit override. Recorded `base_score` 5E-1; configured rounds; feature count 162; prediction shape (n,); no weight or margin.
5. **Reload:** the saved UBJ reloads exactly on FIT, C and E3.
6. **Narrow selftest** (synthetic only, outside the 21-fit budget): binary output shape and range; `base_score` 0.5; target mapping code ≥ 2 → 1; reload round trip; a failed gate means no fit.
7. **Stopping:** stop at the first error; no silent reruns; snapshots read with the ≤ 2020-12 filter only.

## 5. Evidence and order

- **Script (if approved):** one standalone task-research runner, `research/d52_binary_root.py`, reusing the D34 acceptance, D37 gate/rebuild/persistence and D46 scoring helpers where they apply. Exact bytes committed before the run; script git blob equal to HEAD.
- **External outputs** (frozen directory `C:\Users\swl00\geoxgb_runs\geoxgb-d52-binary-root-20261002`, must not pre-exist): binary UBJs and fit records (params, initial score, prediction shape), keyed binary p and original probabilities for FIT, C and E3, labels and hashes, gate/reload records, identity, log and artifact hashes, compact summary.
- **Order:** user decision on §0 → (if yes) supervisor review of this plan → planning commit → native implement and native check (each ≤ 10 min) → producer commit → supervisor release → single run (≤ 21 fits) → factual report → independent supervisor check → stop.
- **Excluded:** ratio, capacity or threshold grids; final data, Stage 2/3, maps, adoption; no automatic D53.
