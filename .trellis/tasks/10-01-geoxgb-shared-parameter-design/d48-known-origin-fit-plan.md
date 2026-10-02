# D48 / A22: known-origin FIT restriction contrast (approved)

2026-10-02. Supervisor decision under the user's delegated Stage 1 research authority. Approved after supervisor review with two wording corrections (row-selection basis, H12 extent); the planning commit precedes any code or fits. Same active task, executor, audit run and base.
- **Endpoint:** the crisis-F1 endpoint is primary and unchanged.
- **Out of scope:** Stage 2/3, final-period reads, full 648 and audit close.
- No default adoption, even if the result is positive.

## 1. Question and what it is not

This is one finite data-pool contrast: re-fit each root only on the ORIGINAL FIT rows whose exact-origin `hist_phase_o00` is finite, then compare it with the saved original root and persistence on origin-known keys.

It is not an identifiable test of missingness causality. D38 showed that exact-origin availability follows the label calendar (quarterly up to 2015, triannual from 2016). At H4/H8 the restriction therefore also cuts to the current calendar regime.

**Everything that changes together:** sample size, calendar regime, era, country and class composition, missingness selection, and the optimisation path. There is no covariate-shift or causal claim.

**H12:** retains 85.6–87.4% of FIT, removing 12.6–14.4%. That is less change than at H4/H8, but still a change.

## 2. Frozen design

- **Pairs:** all 21 D34 pairs, H ∈ {4, 8, 12} × T ∈ {2018-06, 2018-10, 2019-02, 2019-06, 2019-10, 2020-02, 2020-06}.
- **Unchanged:** G = {4: G1, 8: G4, 12: G2}, the 162 features, seed, rounds (H4 200; H8/H12 400), NaN handling, the four-class `multi:softprob` objective and the argmax → code ≥ 2 crisis endpoint.
- **Window:** W59 `[O−59, O)` stays the legal pool boundary; the eligible label dates simply shrink.
- **The only change:** the ORIGINAL FIT rows are masked to those with finite `hist_phase_o00`.
  - No target-outcome values are used to choose rows. Selection uses the availability of the already-known origin-label feature within the original labelled FIT pool.
  - No latest observed label is substituted.
  - No weights or margins.
  - No S/C/E3 rows enter fitting.
- **Arms:** the saved original root; one fresh known-origin root per pair (at most 21 fits); persistence.
- **Excluded:** age/era/weight controls, subgroup selection, round changes, and dropping weak roots. All 21 are kept.
- **Comparison keys:** exactly the origin-known keys of each part.
  - FIT-known is in-sample under both arms.
  - C-known is in-window historical interpolation.
  - E3-known is the forward evaluation.
  - Missing-origin E3 rows are reported only as excluded counts and coverage: no root extrapolation, fallback or default design.
- **Reporting limit:** no all-population performance is advertised, and nothing is compared with the original all-key scores.

## 3. Current support (counts from saved D47 rows; no scores)

| Root | Known / FIT rows | Fraction | Label dates | Areas | Countries | Classes 0 / 1 / 2 / 3 | C origin-known | E3 origin-known (excluded) |
|---|---|---|---|---|---|---|---|---|
| H4 2018-06 | 23,686 / 65,541 | 0.361 | 6 | 5,362 | 20 | 13,267 / 6,868 / 3,136 / 415 | 3,730 / 10,120 | 5,364 / 5,364 (0) |
| H4 2018-10 | 27,825 / 65,908 | 0.422 | 7 | 5,365 | 20 | 15,547 / 8,177 / 3,665 / 436 | 4,407 / 10,193 | 5,364 / 5,365 (1) |
| H4 2019-02 | 31,362 / 61,361 | 0.511 | 8 | 5,365 | 20 | 17,347 / 9,400 / 4,165 / 450 | 5,255 / 10,293 | 5,365 / 5,365 (0) |
| H4 2019-06 | 35,465 / 61,868 | 0.573 | 9 | 5,365 | 20 | 19,442 / 10,716 / 4,819 / 488 | 5,843 / 10,294 | 5,365 / 5,365 (0) |
| H4 2019-10 | 39,577 / 62,374 | 0.635 | 10 | 5,365 | 20 | 21,700 / 11,840 / 5,492 / 545 | 6,478 / 10,295 | 5,365 / 5,365 (0) |
| H4 2020-02 | 46,136 / 62,578 | 0.737 | 11 | 5,365 | 20 | 24,818 / 14,094 / 6,618 / 606 | 5,954 / 7,945 | 5,365 / 5,506 (141) |
| H4 2020-06 | 50,506 / 62,937 | 0.802 | 12 | 5,365 | 20 | 26,906 / 15,325 / 7,655 / 620 | 6,407 / 7,945 | 5,506 / 5,506 (0) |
| **H8 2018-06** | 15,679 / 65,195 | 0.240 | **4** | 5,287 | 20 | 8,838 / 4,326 / 2,264 / 251 | 2,445 / 10,039 | 5,364 / 5,364 (0) |
| **H8 2018-10** | 19,614 / 65,545 | 0.299 | **5** | 5,359 | 20 | 11,062 / 5,634 / 2,604 / 314 | 3,069 / 10,121 | 5,364 / 5,365 (1) |
| H8 2019-02 | 23,759 / 65,908 | 0.360 | 6 | 5,362 | 20 | 13,359 / 6,930 / 3,128 / 342 | 3,736 / 10,193 | 5,364 / 5,365 (1) |
| H8 2019-06 | 27,374 / 61,361 | 0.446 | 7 | 5,363 | 20 | 15,212 / 8,184 / 3,625 / 353 | 4,560 / 10,293 | 5,365 / 5,365 (0) |
| H8 2019-10 | 31,442 / 61,868 | 0.508 | 8 | 5,364 | 20 | 17,270 / 9,513 / 4,274 / 385 | 5,169 / 10,294 | 5,365 / 5,365 (0) |
| H8 2020-02 | 35,509 / 62,374 | 0.569 | 9 | 5,364 | 20 | 19,525 / 10,596 / 4,947 / 441 | 5,838 / 10,295 | 5,365 / 5,506 (141) |
| H8 2020-06 | 41,901 / 62,578 | 0.670 | 10 | 5,365 | 20 | 22,552 / 12,790 / 6,060 / 499 | 5,397 / 7,945 | 5,365 / 5,506 (141) |
| H12 2018-06 | 60,794 / 69,546 | 0.874 | 16 | 5,123 | 19 | 39,989 / 16,193 / 4,349 / 263 | 8,630 / 10,038 | 5,362 / 5,364 (2) |
| H12 2018-10 | 56,316 / 65,199 | 0.864 | 15 | 5,160 | 19 | 36,605 / 15,081 / 4,296 / 334 | 8,643 / 10,040 | 5,364 / 5,365 (1) |
| H12 2019-02 | 56,655 / 65,545 | 0.864 | 15 | 5,304 | 20 | 36,363 / 15,317 / 4,585 / 390 | 8,706 / 10,121 | 5,364 / 5,365 (1) |
| H12 2019-06 | 56,927 / 65,908 | 0.864 | 15 | 5,361 | 20 | 36,228 / 15,667 / 4,613 / 419 | 8,721 / 10,193 | 5,364 / 5,365 (1) |
| H12 2019-10 | 52,509 / 61,361 | 0.856 | 14 | 5,362 | 20 | 32,739 / 14,632 / 4,709 / 429 | 8,819 / 10,293 | 5,365 / 5,365 (0) |
| H12 2020-02 | 53,037 / 61,868 | 0.857 | 14 | 5,364 | 20 | 32,215 / 15,115 / 5,243 / 464 | 8,809 / 10,294 | 5,365 / 5,506 (141) |
| H12 2020-06 | 53,413 / 62,374 | 0.856 | 14 | 5,364 | 20 | 32,102 / 15,049 / 5,753 / 509 | 8,906 / 10,295 | 5,365 / 5,506 (141) |

- **Weak time support, disclosed:** H8 2018-06 has 4 label dates and H8 2018-10 has 5. Both are retained, with rounds unchanged.
- **Prevalence:** the known-origin crisis share at H4/H8 is higher than in full FIT. The resulting probability/decision shift is a disclosed confound.

## 4. Gates and checks (fail closed)

1. **Original replay first:** D34 pinned acceptance (7b2bf6f) and the D37 `gate_root`.
2. **Mask:** the exact eligible FIT mask is the original FIT rows with finite `hist_phase_o00` from the rebuilt FIT matrix.
   - It depends on the origin feature only: changing target outcomes leaves the mask unchanged.
   - Record the eligible-key sha and counts.
   - Assert no S/C/E3 keys appear.
3. **Fit:** one `nx.fit_global(X_fit[mask], y_fit[mask], plan.G_CONFIGS[g])`, unweighted and with no margin. Assert:
   - params and rounds equal `booster_params(G)`;
   - `base_score` 0.5 and 4 classes;
   - no weight or margin block.
4. **Reload:** UBJ reload is exact on FIT-known, C-known and E3-known.
5. **Saved outputs:** keyed origin-known predictions for both arms.
6. **Narrow checks:** a failed gate means no fit, and the mask depends on the feature only. These can be inline self-checks or a `--selftest` mode.
7. **Stopping:** stop at the first failure.

## 5. Metrics

- **Coverage:** per pair, and per H pooled and mean-fold, on FIT-known, C-known and E3-known.
- **Per arm:**
  - crisis F1 with TP/FP/FN (primary), and the crisis-call share;
  - fixed-four macro-F1;
  - unweighted crisis Brier;
  - four-class log loss (float64, no clipping).
- **Persistence:** crisis F1, macro-F1 and one-hot Brier only, with no log loss.
- **Secondary diagnostic:** within-root crisis AUC/AP on s = (p2 + p3)/Σp, via sklearn, mean-fold only.
- **No significance tests,** because the folds are repeatedly exposed.

## 6. Implementation

- **Script:** one self-contained one-off script, `research/d48_known_origin_fit.py`. Its exact bytes are committed before any fit.
- **Reuse:** it imports the existing package helpers — D37 acceptance, gate and rebuild; D46 pure scoring (`block`, `delta`, `ranking`, `_pool`, `_mean_fold`); `nx.fit_global`. There is no new production abstraction and no copy of the D47 runner.
- **Identity:** frozen runtime, script and package identity (git HEAD), and data hashes.
- **Process:** native implement and native check (each under 10 minutes) → producer commit → independent supervisor recheck → supervisor run release → a single run into a fresh external directory.
- **Afterwards:** stop after D48.
