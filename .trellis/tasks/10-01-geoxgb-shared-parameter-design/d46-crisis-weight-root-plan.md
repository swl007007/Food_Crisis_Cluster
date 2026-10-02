# D46 / A20: 2×-crisis class-weighted root training contrast (approved)

2026-10-02. Supervisor design decision under the user's delegated Stage 1 research authority. Approved after supervisor review with one factual correction (D39 per-horizon calibration); the planning commit precedes any code or fits. The fixed crisis-F1 endpoint is retained unchanged. Same active task, executor, audit run and base. No Stage 2/3, final-period reads, full 648 or audit close.

## 1. Motivation and question

D35–D45 changed global capacity, time weights, initial margins, decision rules, local amplitude, map vintage and generation, the fitting pool and boosting prefixes. None changed how the root's trees are fitted with respect to crisis labels.

D46 asks: does re-fitting the root under a fixed 2:1 crisis-class cost add value beyond applying the same cost odds to the existing scores?

**Rejected first, before any scoring:** a current-origin prevalence-ratio adjustment of the saved probabilities. No adjusted scores or fits were produced. It was rejected because D39 undermines its simple premise and because applying a global prior ratio on top of origin-conditioned features risked double counting. D39's mean E3 crisis probability against truth prevalence (.183) was about .180 at H4 and .188 at H8, but .152 at H12. The root was not uniformly close to E3 prevalence, and it was not at the FIT prevalence either. Those were the reasons, not a measured failure.

## 2. Frozen design

- **Pairs:** all 21 D34 pairs, H ∈ {4, 8, 12} × T ∈ {2018-06, 2018-10, 2019-02, 2019-06, 2019-10, 2020-02, 2020-06}.
- **Unchanged from D34:** the FIT/C/E3 keys; G = {4: G1, 8: G4, 12: G2}; seed, rounds, the 162 features, NaN handling, the 59-month window and the four-class `multi:softprob` objective.
- **Weights:** `w = (1 + I[y_fit ≥ 2]) / mean(1 + I[y_fit ≥ 2])`, computed in float64 from the actual FIT labels only. The existing `nx.check_sample_weight` casts to float32. Record the actual float32 sha, sum, min, max and the per-class values. Normal float32 rounding of the mean is accepted. The Kish ESS that `weight_record` stores is not interpreted as an independent sample size.
- **Why 2:1:** a prespecified, moderate and arbitrary choice, not an optimum.
- **Arms:**
  - `original`: the saved D34 root;
  - `weighted`: one fresh root fitted with the weights above;
  - `posthoc2x`: zero fit; the original probabilities with classes 2 and 3 multiplied by 2, then renormalised;
  - plus persistence on the same keys.
- **Budget:** at most 21 authoritative root fits. No unweighted refits, G search, local fits, new windows, weights, seeds or thresholds. No combination with D38 and no propagation to partitions.
- **Stopping:** a gate or fit failure stops the run, keeps partial evidence and triggers no wider rerun. Whatever the outcome, there is no automatic adoption and no further weight sequence.

## 3. Interpretation

**Primary contrasts, on E3 per H:**
- weighted − original;
- weighted − posthoc2x;
- weighted − persistence;
- posthoc2x − original, reported explicitly.

**Reading:**
- Weighted − posthoc2x captures the whole effect of re-fitting under weights: different splits and leaves, regularisation and optimisation acting on weighted Hessians, and so on. It is not unique proof of split allocation and not evidence of causal generalisation.
- The weighted outputs are cost-sensitive scores, not calibrated posteriors. Unweighted Brier and log loss stay visible, with no direction promised.
- An F1 gain alone is not a claim that overfitting is solved.

## 4. Gates and checks (fail closed)

1. **Original-root replay before any fit:** reuse D37 `gate_root` for the C/E3 full probabilities and labels, and the FIT keys and matrices.
2. **Fit:** `nx.fit_global(X_fit, y_fit, G, sample_weight=w)`, with no adapter or model-semantics edits. Assert:
   - the scalar `base_score` is 0.5;
   - 4 classes and the configured rounds;
   - the actual float32 weight bytes match the record.
3. **Reload:** the weighted UBJ reloaded from file must reproduce its probabilities exactly.
4. **Focused tests,** no boilerplate:
   - a failed gate prevents any fit;
   - weights depend only on FIT labels, i.e. mutating S/C/E3 labels leaves the weights and fit inputs unchanged;
   - the post-hoc ×2 control preserves within-group ratios (classes 0/1, classes 2/3) and the crisis-score ranking.
5. **Process:** native implement, then native check, then the committed producer, then the single real run. Each subagent stays within the 10-minute ceiling.

## 5. Metrics

**Coverage:** per pair, and per H both pooled and mean-fold, with the FIT (in-sample), C (in-window historical interpolation) and E3 (forward) roles explicit and prevalence differences noted.

**Per arm:**
- crisis F1 with TP/FP/FN (four-class argmax then code ≥ 2);
- fixed-four macro-F1;
- unweighted crisis Brier (p2 + p3 summed in float64 from the raw native probabilities);
- four-class log loss (float64, no clipping; stop on a non-positive true-class probability).

**E3 cohorts:**
- all keys;
- exact-origin persistence available. Persistence gets crisis F1, macro-F1 and a one-hot Brier reference only, with no log loss.

**Within-root ranking:**
- Crisis AUC and AP only, via the installed `sklearn.metrics` (`roc_auc_score`, `average_precision_score`).
- For ranking, the score for every arm is s = (p2 + p3) / Σ p0..p3, in float64. XGBoost's float32 rows sum only approximately to 1.
- Post-hoc ×2 is then the monotone map 2s/(1 + s), up to floating-point arithmetic. The check confirms the control's ranking agrees with the original's to numerical precision, and rounding ties are not read as learned ranking changes.
- Report mean-fold summaries with eligible counts. Rankings are not pooled across heterogeneous roots.
- The raw-probability Brier and log loss definitions stay unchanged.

## 6. Evidence

- **One minimal committed runner,** `FEWSNETGeoXGBExperiment/scripts/stage1_class_weight_root.py`, reusing D37's gate, rebuild, identity, persistence and scoring helpers. Focused tests go in `tests/test_baseline.py`.
- **External, under `C:\Users\swl00\geoxgb_runs\`:** weighted UBJ files with fit and weight provenance, keyed C/E3 probabilities for all arms, and compact metrics and identity JSON. FIT rows and probabilities are also kept there if needed for clean independent replay. No large files go into git.
- **Afterwards:** the supervisor independently verifies, and synthesis is the supervisor's. Stop after this contrast.

## 7. Order

Supervisor review → align pointers and commit the planning (GitNexus attempt; LadybugDB failure recorded) → native implement → native check → producer commit with test evidence → single run (at most 21 fits) → factual report → supervisor verification and synthesis.
