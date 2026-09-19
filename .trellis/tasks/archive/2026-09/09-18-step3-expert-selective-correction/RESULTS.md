# Results: Step 3 expert selective correction (run `full_fs1_fs2_20260918`)

Run directory: `Step3ExpertCorrectionExperiment/outputs/full_fs1_fs2_20260918/` (24 folds, exit 0, 2026-09-18).
Contracts as revised 2026-09-18 (calendar-aligned expert at `O = T-H`; `V=[O-12,O)`; all gates unchanged).

## Headline

**The method does not work.** Validation-gated selective correction of the calendar-aligned
FEWS NET expert is net harmful on test. Verified as a genuine result, not an implementation
defect (see "Defect hypotheses" below).

Test set, all 2021-2024 target months, n=62,189 per scope:

| scope | expert-only | partitioned_selective_correction | pooled (GeoRF) |
|---|---|---|---|
| fs1 | **0.8070** (P 0.8398 / R 0.7768) | 0.8053 (**-0.0017**) | 0.6411 (P 0.7656 / R 0.5514) |
| fs2 | **0.7630** (P 0.8077 / R 0.7230) | 0.7576 (**-0.0054**) | 0.6159 (P 0.7547 / R 0.5202) |

Selection: 11/24 folds `corrected`, 13 `no_correction`. Validation F1 gains +0.00095..+0.00845.
Flips: fs1 214 (86 fixed / 128 damaged, 40.2% test precision); fs2 230 (50 / 180, 21.7%).

## Why it fails

Not absence of signal — **the selector picks the wrong direction.** Per-fold oracle headroom
over expert-only (threshold AND direction chosen on test labels, matching the mechanism's own
per-fold freedom):

| scope | oracle both directions | oracle `0->1` only | oracle `1->0` only | actual |
|---|---|---|---|---|
| fs1 | +0.0082 | +0.0078 | +0.0034 | -0.0017 |
| fs2 | +0.0180 | +0.0171 | +0.0020 | -0.0054 |

Nearly all available headroom is in `0->1` (flip "expert says no crisis" to crisis). The approved
gates enabled `1->0` in **7 of 11** corrected folds, sending 386 of 444 flips into the
near-worthless direction.

Validation -> test flip precision, pooled over enabled directions:

| direction | folds | val flips / precision | test flips / precision |
|---|---|---|---|
| `0->1` | 4 | 212 / 0.778 | 58 / **0.690** |
| `1->0` | 7 | 548 / 0.779 | 386 / **0.249** |

Mechanisms behind the collapse:

1. **Three observed validation months only.** The source is tri-annual, so `V=[O-12,O)` yields
   exactly 3 label months in every fold. The `distinct months >= 2` gate therefore has almost no
   power to detect temporal instability. Selected validation precisions cluster at 0.752-0.800,
   right on the 0.75 gate boundary - the winner's-curse configuration.
2. **Selection signal inside the noise.** Median per-fold bootstrap SE of test F1 is 0.0080 while
   validation gains are +0.001..+0.008. Validation gain vs realised test gain correlates
   *negatively* (Pearson -0.133, Spearman -0.118); 4/11 folds positive, 6/11 negative.
3. **Regime shift.** 88.3% of damaged flips come from 3 folds (2021-10, 2022-02). `P(expert wrong | e==1)`
   fs1 2022-02: 0.277 (validation) -> 0.068 (target); fs2 2022-02: 0.402 -> 0.125. Crisis prevalence
   rose 0.18 -> 0.30 into 2022, so the expert became *more* right about crises exactly when the
   frozen `1->0` rule was flipping them off. The two folds where `P(expert wrong | e==1)` instead
   *rose* (fs1 2021-10, fs1 2022-10) are the only ones where flips generalised (test precision
   0.784, 0.841).
4. **Design-mandated two-stage fit amplifies it** (contract-compliant, not a defect): the threshold
   is calibrated on validation-fit scores but applied to final-refit scores. Test/validation
   tail-mass ratio swings 0.00-4.14 and is 2.12 / 3.41 / 3.93 in the three worst-damage folds, so
   the refit ensemble proposes 2-4x more flips than the threshold was calibrated for.

## Defect hypotheses - all ruled out

Score polarity (independently confirmed: `P(expert wrong | score bin)` monotone
0.034 -> 0.111 -> 0.225 -> 0.442 on fs1 across four large bins; the `[0.75,1]` reversal to 0.023 is
171 rows, all `e==1`); flip direction (0 violations across 124,378 audit rows); partition keying
(eligible test set == refit-trained partition set in all 24 folds; single `assign_partitions()`
call shared by validation and test); threshold scaling (100-tree `predict_proba` is already on a
0.01 grid, so 2-dp rounding is a no-op; all final predictions reproduce from unrounded scores);
refit feature/label alignment (167 features + expert as final column, panel sort asserted,
imputer fit on `refit_index` only); test-time leakage (rule frozen before the refit ensemble exists).

Four regression guards were added because the suite had no test for the highest-risk invariant:
a change from `np.where(classes == 1)[0]` to positional `proba[:, 1]` would have passed all 55
prior tests. All four passed against the unmodified implementation. Suite is now 59 tests,
passing on both `python3` (pandas 3.0.0) and `.venv-geodt-diagnostic` (pandas 2.3.3).

## Secondary finding - affects the paper's comparison

With the correctly dated expert, **FEWS NET's own projections beat GeoRF pooled by a wide margin**:
fs1 +0.166 F1, fs2 +0.147 F1, driven by recall (0.777 vs 0.551; 0.723 vs 0.520). The monthly
`correction - pooled` deltas (+0.02..+0.43, positive in 23 of 24 months) are therefore almost
entirely the expert's contribution, not the partitioning or the correction layer.

The archived paper baseline used the legacy record shifts (fs1 F1 0.6239, fs2 0.5246 vs calendar
0.8070 / 0.7630), so any paper text citing those numbers understates the expert by 0.18-0.24 F1.
**Correcting frozen paper artifacts was explicitly deferred by the user on 2026-09-18** and was not
done; see `Step3ExpertCorrectionExperiment/README.md` and `run_manifest.json` -> `expert.deferred_known_issue`.

## Contract compliance - all PASS

31 protected hashes unchanged; pooled and fs3 per-row equal to the frozen archive (62,189 rows each,
`y_pred`/`y_true`); fs3 flagged uncorrected; no `y_prob_partitioned` or crisis-probability column;
gates left at 20 / 2 / 0.75 / strict-F1 (fs2 2022-10 `0->1` at precision 0.7222 correctly *not*
enabled); `V=[O-12,O)` with 3 observed months in 24/24 folds, all equal `{O-4, O-8, O-12}`; outer
window 35 timestamps with `outer_end == O` and `fit_cutoff == V_start - H`; expert lag a single
bucket at `H` with 100% coverage on the evaluation support; legacy record-shift series firewalled
from the correction layer while still reproducing 39/39 archived quarters per scope at atol 1e-12.
No tracked file modified; no commits.

## Not verified / out of scope

- Bit-for-bit retraining of pooled/fs3 (historical package environment unavailable; per-row and
  metric equality at 1e-12 is the strongest available check).
- Within-month expert publication timing: the source carries publication month, not day, so
  availability at the `O`-month decision remains an explicit assumption recorded in the manifest.
- Whether a different mechanism could succeed. The per-fold oracle bounds *this* mechanism
  (one shared threshold, two directions, per-partition wrong-label RF) at +0.008 / +0.018 F1. It
  says nothing about `0->1`-only correction, per-partition thresholds, cost-sensitive objectives,
  or a recency-weighted validation window - the oracle decomposition above suggests
  `0->1`-only is the one variant with a plausible case.

---

# Variant B: `0->1`-only correction (run `full_fs1_fs2_up_only_20260918`)

Run directory: `Step3ExpertCorrectionExperiment/outputs/full_fs1_fs2_up_only_20260918/` (24 folds, exit 0).
Method ID `partitioned_selective_correction_up_only`. `1->0` forced off before candidate scoring;
all other gates identical to Variant A. Variant A's run is untouched and reproduces byte-for-byte.

## EPISTEMIC STATUS - read before citing any number below

Variant B's direction restriction was specified **after** Variant A's test results were known.
Its test metric is therefore **post-hoc and test-informed, not an out-of-sample estimate.**
Variant A remains the only genuinely out-of-sample result for this mechanism.

The restriction's stated justification is **asymmetric cost**, which is independent of any test
result: a `1->0` flip silences an already-issued crisis warning, and a missed food-security crisis
carries materially higher cost than a false alarm. The oracle decomposition that first pointed at
`0->1` was computed **with test labels** and is post-hoc corroboration only.

## Result: also null

| scope | expert-only | Variant B | Variant A | pooled |
|---|---|---|---|---|
| fs1 | 0.8070 | 0.8080 (**+0.0009**) | 0.8053 (-0.0017) | 0.6411 |
| fs2 | 0.7630 | 0.7630 (**+0.0000**) | 0.7576 (-0.0054) | 0.6159 |

- **fs2 applied zero flips.** Variant B at fs2 is a bit-for-bit reproduction of expert-only, not an
  improvement on it. No fold enabled `0->1`.
- fs1: 58 flips, 40 fixed / 18 damaged, test precision 0.690. Zero `1->0` flips;
  `enable_1_to_0 = False` in 24/24 folds. 4/24 folds `corrected`, all fs1.
- Because no Variant A fold ever enabled both directions, Variant B is exactly Variant A minus its
  seven `1->0`-only folds, with thresholds unchanged in the four survivors.

## The fs1 gain is one month and is inside the noise

| fold | flips | fixed | damaged | test precision | delta vs expert |
|---|---|---|---|---|---|
| fs1 2022-10 | 44 | 37 | 7 | 0.841 | **+0.0115** |
| fs1 2023-02 | 1 | 0 | 1 | 0.000 | -0.0003 |
| fs1 2024-02 | 13 | 3 | 10 | 0.231 | -0.0013 |

Robustness (computed on the saved predictions, 2000 resamples over the 12 fs1 folds):

| check | delta |
|---|---|
| all 12 months | +0.0009 |
| leave out fs1 2022-10 | **-0.0002** |
| leave out fs1 2024-02 | +0.0012 |
| leave out fs1 2023-02 | +0.0011 |
| fold bootstrap | mean +0.0009, sd 0.0010, **95% CI [-0.0004, +0.0033]**, P(delta>0)=0.645 |

Dropping the single 2022-10 fold turns the aggregate negative, and the fold-level bootstrap CI
spans zero. The two other flipping folds are both negative (3 fixes in 14 flips combined). The
realised gain is ~12% of the `0->1` oracle bound (+0.0078).

## Conclusion across both variants

Per-partition RF selective correction of the calendar-aligned FEWS NET expert shows **no
demonstrable benefit** on this data. Variant A is net harmful (fs1 -0.0017, fs2 -0.0054).
Variant B removes the harm but adds nothing measurable: exactly zero at fs2, and at fs1 a
one-fold artefact indistinguishable from zero. Disabling the damaging direction is not the same as
finding a working mechanism.

The binding constraint is the data cadence, not the learner: three observed tri-annual validation
label months give the `distinct months >= 2` gate almost no power to detect temporal instability,
selected validation precisions sit on the 0.75 gate boundary, and validation gain correlates
*negatively* with realised test gain (Pearson -0.133 across Variant A's corrected folds).

## Verification

78 tests pass on both `python3` (pandas 3.0.0) and `.venv-geodt-diagnostic` (pandas 2.3.3), up from
59 (+19 Variant B tests: forced-off in every fold, force-off defeat caught by an independent
re-check, no `1->0` reachable even when 30 `e==1` rows would be proposed, `0->1` gate boundaries
unchanged under both modes, Variant A default unaffected, method-id and column naming, manifest
epistemic assertions, invalid-mode rejection before any write).

Variant A regression: `variantA_regression_fs1_2021_02` reproduces `bounded_cal_fs1_2021_02`
byte-for-byte across all 10 data artifacts; the `direction_mode` columns are emitted only in
restricted mode so Variant A's schema stays byte-identical. Support identical across variants
(124,378 rows; keys, `y_true`, `expert`, `y_pred_pooled` per-row equal). 31/31 protected hashes
unchanged. Legacy series still reproduces 39/39 archived quarters per scope at 1e-12. Determinism
and run-directory immutability hold. fs3 outputs byte-identical across variants.

GitNexus `impact`/`detect_changes` were unavailable inside the sub-agent session; a repo-wide
caller search confirmed no production file references the touched symbols, and the orchestrator ran
`detect-changes --repo Food_Crisis_Cluster` at commit time with "No changes detected".
