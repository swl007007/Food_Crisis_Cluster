# D47 / A21: depth-1 (stump) root contrast (completed; not adopted; supervisor verification passed)

2026-10-02. Supervisor selection under the user's delegated Stage 1 research authority. Approved after supervisor review with one wording correction (regularisers held at existing fixed values, not tuned); the planning commit precedes any code or fits. Same active task, executor, audit run and base.
- **Endpoint:** the crisis-F1 endpoint is the PRIMARY scientific endpoint and is unchanged.
- **Out of scope:** Stage 2/3, final-period reads, full 648 and audit close.
- No default adoption, whatever the result.

## 1. Question

No global root shallower than depth 3 has been fitted. G1–G4 are depth 3/4 (`src/experiment/plan.py:34-37`). D33's depth-1 truncation was of the partition tree, not the booster. Depth-1 boosters have appeared only as 20-round increments on top of depth-3/4 roots (L1, D35).

D47 asks what happens to forward E3 crisis F1 when the root is re-fitted with stumps (`max_depth=1`), everything else fixed.

**What changes along with depth:** capacity, optimisation path, sampling paths and effective regularisation all change together. `min_child_weight` and `reg_lambda` are held at their existing fixed, conservative values (they were not tuned), and their effective action can differ with shallower trees. This is therefore not an isolated interaction effect.
- Per-class margins become additive in single features. Probabilities still interact through the softmax.
- Engineered interaction features (for example `hist_origin_phase{k}_x_w12_*`) remain.

**Limits on interpretation:**
- A negative result does not show that interactions carry transferable signal in general.
- A positive result does not show that overfitting is solved.
- A smaller FIT-to-E3 gap can come simply from a worse FIT fit, and the role populations and prevalences differ, so gap reduction is not evidence about overfitting.

## 2. Frozen design

- **Pairs:** the same 21 D34 pairs, H ∈ {4, 8, 12} × T ∈ {2018-06, 2018-10, 2019-02, 2019-06, 2019-10, 2020-02, 2020-06}.
- **Unchanged:** the FIT/C/E3 keys and FIT lineage, the 59-month window, the 162 features, NaN handling, no weights and no margins, the four-class `multi:softprob` objective with argmax → code ≥ 2, the seed, eta, subsampling and regularisers.
- **Rounds:** H4 200; H8/H12 400.
- **Config:** a copied `dict(plan.G_CONFIGS[g], max_depth=1)` with G = {4: G1, 8: G4, 12: G2}. `plan.G_CONFIGS` and `XGB_BASE` must be equal before and after the run.
- **Arms:** the saved original D34 root; one fresh stump root per pair; persistence on the same keys.
- **Budget:** at most 21 fresh fits. No unweighted G refit, depth or rounds search, local fits, maps, D38/D46 combination, Stage 2/3, full 648, final evaluation or close.
- **Stopping:** stop at the first failure; there is no automatic rerun.

## 3. Gates (fail closed)

1. **Original replay before any fit:** D34 pinned acceptance (7b2bf6f) and the D37 `gate_root`, covering the C/E3 full probabilities and labels and the FIT keys and matrices.
2. **Fitted model checks:**
   - the recorded params equal `booster_params(G)` except `max_depth=1`;
   - the training resolved config, recorded separately at fit time, shows `max_depth` 1. A UBJ reload does not keep every training parameter, so the saved tree structure is the depth evidence;
   - every saved tree in the model JSON has depth ≤ 1, with split and leaf counts recorded;
   - `base_score` is 0.5, with 4 classes and the configured rounds;
   - no weights and no base margin are present.
3. **Reload:** the stump UBJ reloaded from file reproduces FIT, C and E3 probabilities exactly.
4. **Narrow tests:**
   - a failed gate means no fit;
   - building the copied config does not mutate the plan defaults.

   Otherwise the existing exact gates are relied on; no sprawling test suite.

## 4. Metrics

**Coverage:** FIT (in-sample), C (in-window historical interpolation) and E3 (forward), per pair and per H, pooled and mean-fold.

**Per arm:**
- crisis F1 with TP/FP/FN, the PRIMARY endpoint (argmax → code ≥ 2), with the derived crisis-call share;
- fixed-four macro-F1;
- unweighted crisis Brier (p2 + p3, float64);
- four-class log loss (float64, no clipping; stop on a non-positive true-class probability).

**E3 cohorts:** all keys and the persistence-matched cohort. Persistence gets crisis F1, macro-F1 and one-hot Brier only, with no log loss.

**Secondary mechanistic diagnostics:** within-root crisis AUC/AP on s = (p2 + p3)/Σp in float64, via sklearn, mean-fold only with eligible counts. These are not a replacement endpoint and not adoption gates.

**Reporting:** actual E3 gains and losses per H and per-fold heterogeneity (fold wins), descriptively. No significance or adoption rule.

## 5. Placement and implementation

**The smallest maintainable placement** is one committed package runner, `FEWSNETGeoXGBExperiment/scripts/stage1_stump_root.py`. It imports the already committed and verified D37 helpers (`stage1_recency_root`: acceptance, gate, rebuild, persistence, dev-baselines check) and the D46 pure scoring helpers (`stage1_class_weight_root`: `block`, `delta`, `crisis_score`, `log_loss_fourclass`, `ranking`). D46's `score_part`/`aggregate` are reused only where they are arm-agnostic. Where D46 hard-codes its arm names, a small local aggregation over the two arms plus persistence is written, not a copy of D46.

Only the stump fit and its depth checks are new, and there is no second copy of the scoring code. The weight function and post-hoc arm of D46 are not used.

**Why a package script rather than an external one:**
- the scoring is shared with the verified D46 code, so not duplicating it reduces divergence risk;
- the package and script identity come from git HEAD, through the existing working-tree-equals-HEAD check;
- the two narrow tests go in `tests/test_baseline.py`.

**Process:** native implement and native check, each under 10 minutes → committed producer with test evidence → supervisor release → a single real run on the frozen Windows Python into a fresh external directory. External evidence: stump UBJs and records, keyed rows, compact summary and identity JSON. No new runtime dependencies.

## 6. Order

Supervisor review → align pointers and commit the planning (GitNexus attempt; LadybugDB failure recorded) → native implement → native check → producer commit → supervisor release → single run → factual report → supervisor verification and synthesis.

## 7. Result (2026-10-02)

- **Producer and run:** producer `5f6eb4307c1bded78c89ff27ab76d04431a5995d` (116 tests OK). Run `C:\Users\swl00\geoxgb_runs\geoxgb-d47-stump-root-20261002`: exit 0 in 189 s, exactly 21 stump fits, all gates passed, every tree depth 1, exact reloads.
- **Supervisor verification:** PASS (`research/d47_supervisor_check.py` / `_results.json`, 172,573 checks).
- **Findings:** [research/d47-stump-root-findings.md](research/d47-stump-root-findings.md).
- **Decision:** the fixed-round depth-1 replacement is not adopted. On E3 it is below both the original and persistence at every H, and the depth/round sequence stops.
- **Stage 1** remains unresolved, and no D48 starts automatically.
