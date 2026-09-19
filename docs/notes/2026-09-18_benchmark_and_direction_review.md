# 2026-09-18 — Benchmark reality check and project direction

Status: analysis note. No code or frozen artifact was modified to produce it.
All numbers below were recomputed in-session from saved artifacts and source data.

## 1. Why the FEWS NET benchmark "got higher"

The archived paper baseline used the legacy per-admin **record** shift
(`shift(4)`/`shift(8)`), which on a quarterly-then-tri-annual source resolves to
12-16 / 24-32 calendar months. See [[fewsnet-source-is-tri-annual-not-monthly]].

| FEWS NET expert, crisis-class F1, 2021-2024 | fs1 | fs2 |
|---|---|---|
| Archived legacy series (mean of per-quarter F1 in `fewsnet_baseline_results_fs*.csv`) | 0.588 | 0.491 |
| Same legacy series, pooled recomputation | 0.624 | 0.525 |
| **Calendar-aligned (correct) expert** | **0.807** | **0.763** |

The original "GeoRF beats FEWS NET" result compared GeoRF (~0.68 partitioned) against the
mis-dated ~0.59. Against the correctly dated expert that claim does not survive.

## 2. The expert is not beaten at any operating point

GeoRF Stage 3 saves `y_prob_partitioned`, so the comparison can be made on the PR curve
rather than at a single threshold.

| fs1, 2021-2024, n=62189 | precision | recall | F1 |
|---|---|---|---|
| Expert (calendar-aligned) | 0.840 | 0.777 | **0.807** |
| GeoRF partitioned @ expert precision | 0.840 | 0.425 | — |
| GeoRF partitioned @ expert recall | 0.657 | 0.777 | — |
| GeoRF partitioned, best F1 on curve (test-tuned, optimistic) | — | — | 0.716 |

fs2 is the same shape: AP 0.750, best-on-curve 0.691 vs expert 0.763. The expert's operating
point lies outside the model's PR curve. This is not a threshold or loss-function problem.

Stratifying by an **observable** condition (last observed phase at `T-H`):

| fs1 | expert F1 | GeoRF best-F1 on curve |
|---|---|---|
| Onset set (last phase 0, n=45706, crisis rate 0.103) | 0.340 | 0.435 (test-tuned) |
| Remission set (last phase 1, n=16483, crisis rate 0.816) | 0.910 | — |

On the onset set the expert's *point* is still above the model's curve (at the expert's
recall 0.227 the model reaches precision 0.456 vs the expert's 0.679), but the expert
operates conservatively, so a threshold-tuned model can post a higher onset F1. Thin, and
test-tuned.

> Caveat on an earlier cut: stratifying by whether `y_true` changed makes GeoRF look better
> than the expert on "changed" rows. That conditions on the test label and is not achievable
> at prediction time. The observable stratification above reverses it. Do not cite the
> `y_true`-conditioned version.

## 3. The real problem: GeoRF loses to a one-line persistence rule

Persistence = carry forward the last observed IPC phase at `T-H`.

| crisis-class F1, 2021-2024 | persistence | GeoRF pooled | GeoRF partitioned | expert |
|---|---|---|---|---|
| fs1 | **0.776** | 0.641 | 0.682 | 0.807 |
| fs2 | **0.709** | 0.616 | 0.664 | 0.763 |

Not a threshold artifact: GeoRF's best F1 anywhere on the fs1 curve is 0.716 < 0.776.
`fews_ipc_crisis_lag_4` **is** in the feature matrix (index 73 of 166, confirmed in
`feature_columns_debug.csv`), and it is correctly dated — on a monthly panel with tri-annual
labels, `shift(4)` from a Feb/Jun/Oct target lands on an observed month.

Ruled out during this review:
- Forward-filled labels diluting training. `impute_gap_months` defaults to `None`
  (`src/preprocess/preprocess.py:229`); the gap-fill is off in these runs.
- Record-vs-calendar shift in the model's own lag features. The analysis panel
  `FEWSNET_forecast_unadjusted_bm.csv` is genuinely monthly (180 consecutive month stamps),
  so `groupby().shift(lag)` there is a true calendar lag. This bug is specific to the
  legacy expert loader.

What the model actually does (fs1, vs persistence):

```
model             0      1
persist(T-4)
0             41513   4193
1              5421  11062
```

- They disagree on 15.5% of rows. When they disagree, **persistence is right 65.0%** of the
  time and the model 35.0%.
- Worst cell: `persist=1, model=0` (5,421 rows) — persistence right 70.4%, model right 29.6%.
  The model switches off warnings that should have stayed on.
- Mean predicted probability is 0.623 where `persist=1` (actual crisis rate there: 0.816) and
  0.209 where `persist=0`. The signal is used, but the model is badly under-confident on the
  persistence-positive side and deviates too freely.

## 4. Ceiling of direction 1 (persistence base + model override)

Base = persistence; override to 1 when `persist=0` and `prob > up_thr`; override to 0 when
`persist=1` and `prob < dn_thr`. Thresholds swept on the test set, so these are **optimistic
ceilings**, not achievable estimates.

| | persistence | best override | expert | gap to expert |
|---|---|---|---|---|
| fs1 | 0.776 | **0.786** (up_thr 0.67, 1458 up-flips, 1 down-flip) | 0.807 | −0.021 |
| fs2 | 0.709 | **0.741** (up_thr 0.53, 3354 up-flips, 2 down-flips) | 0.763 | −0.022 |

Two things to carry forward:
- "Slightly better than persistence" is achievable (+0.010 fs1, +0.033 fs2 at the ceiling);
  "beats the expert" is not, and the shortfall is a consistent ~0.02 in both scopes.
- The optimal down-flip count is 1-2, i.e. the `1->0` direction is useless here. This
  independently reproduces the Variant A / Variant B result from the Step 3 correction
  experiment, which reached the same conclusion against the expert base.

## 5. IPCCH as an alternative base (direction 2)

`1.Source Data/assembled_IPCCH/raw/IPCCH_2026_completed.csv`: 1,219,868 rows, 6,227 admins,
53 countries, 2010-2026, 143 columns. Label is `overall_phase`.

- Only **43,713 rows (3.6%) are labeled**. Median 6 observations per admin, p10 = 3, and
  **39.9% of admins have fewer than 5 observations**. Coverage grows over time
  (461 labeled rows in 2014 -> 9,825 in 2025).
- `overall_phase` contains 102 rows coded `9.0` (almost certainly a missing/NA sentinel) and
  540 rows coded `0.0`. Both need an explicit decision before use.
- Crisis rate (phase >= 3) is 0.414, much higher than FEWS NET's 0.292, so raw F1 is **not**
  comparable across the two datasets.

Persistence strength, base-rate adjusted (kappa, and lift over the all-positive baseline):

| | base rate | persistence F1 | all-positive F1 | lift | kappa |
|---|---|---|---|---|---|
| FEWS NET fs1 (H=4) | 0.292 | 0.776 | 0.452 | +0.324 | 0.690 |
| FEWS NET fs2 (H=8) | 0.292 | 0.709 | 0.452 | +0.256 | 0.601 |
| IPCCH, prev labeled obs, 2021+ | 0.442 | 0.763 | 0.613 | +0.150 | 0.574 |

So IPCCH persistence really is weaker once the base rate is accounted for — the user's
premise holds. Caveat: the IPCCH gap between consecutive observations is heterogeneous
(modal gaps 4 and 8 months, but 3/5/6/7/9/12 all common), so this is not a clean fixed-horizon
comparison and should be redone per horizon before it is quoted.

## 6. Where this leaves the project

- **Dead:** "GeoRF outperforms the FEWS NET expert projection." Dominated at every operating
  point in both scopes. Not worth another angle.
- **Alive and stable:** partitioned > pooled, consistent across all three scopes
  (fs1 +0.041, fs2 +0.048, fs3 +0.037). Supports a spatial-partitioning *method* paper, not a
  food-security-forecasting-beats-experts paper.
- **Publishable on its own:** the 12/24-month mis-dated benchmark, which also kills the
  original headline.
- **Open and worth fixing regardless of direction:** GeoRF losing to persistence. Any future
  claim on either dataset is undermined while a 166-feature model ranks worse than one of its
  own inputs.

## 7. Two artifact facts established while planning the persistence-correction work

**2018-2020 GeoRF probabilities do not exist.** Stage 1 hard-codes its output schema to
`year,month,adm_code,fews_ipc_crisis_pred,fews_ipc_crisis_true` (`app/main_model_GF.py:233`) and
never calls `predict_proba`; no fitted 2018-2020 model was saved either. Regenerating them is a
pure configuration change to `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`, which already
writes `y_prob_pooled` / `y_prob_partitioned` (`:1301-1310`) and accepts an arbitrary
`--start-month`/`--end-month` with no year floor (`:1016-1026`). Expect roughly 6 minutes per
scope for 2018-2020 (9 labeled months), extrapolated from consecutive manifest timestamps in
`archived/release_20260624_reproducibility_inputs/`.

**Stage 3 ran with `--month-ind`, which its manifest does not record.** Distinct `partition_id`
counts are Feb 14 / Jun 12 / Oct 17, matching `nc13_m2` / `nc11_m6` / `nc16_m10` at `nc + 1`, and
5,350 of 5,713 admins change partition across months — yet the manifest lists only the `nc17`
general map and has no `month_ind_enabled` key. Consequences: a comparable re-run must pass
`--month-ind` and all three month maps by explicit path (the repo-root `refined/` directory is
empty; the live copies are under `paper_reproducibility_package/`), and "partition" is three
different partitionings selected by calendar month, not 17 stable units.

Also confirmed: no probability-calibration code exists anywhere in the repo (`sklearn.calibration`
is never imported). Only reliability *diagnostics* exist, at
`scripts/paper_artifacts/analyze_georf_probability_uncertainty.py:115-148`.

## 8. Outcome of direction 1 (2026-09-19): null, and the ceiling is the reason

The persistence-correction 2-layer gate ran and **failed**. Full record:
`.trellis/tasks/09-18-persistence-correction-2layer/` and `PersistenceCorrectionExperiment/RESULTS.md`.

| | persistence | 2-layer override | delta | bootstrap CI95 | verdict |
|---|---|---|---|---|---|
| fs1 | 0.7761 | 0.7741 | **-0.0020** | [-0.0139, +0.0112] | fail |
| fs2 | 0.7085 | 0.7217 | **+0.0131** | [-0.0029, +0.0322] | fail |

Pre-registered criteria: MDE `+0.02`, bootstrap CI excluding zero, leave-one-fold-out. fs2 missed on
both effect size and CI; fs1 moved the wrong way with 3 of 12 folds flipping the sign.

**The decisive number is the ceiling, not the transfer.** Sweeping the threshold on the calibrated
probability *with test labels* — the best the mechanism could possibly have done — gives
**+0.0054 (fs1)** and **+0.0140 (fs2)**. Both are below the MDE. fs2's frozen threshold captured 94%
of its ceiling; there was simply nothing more to capture. (An earlier draft of this conclusion
blamed threshold instability, citing +0.010/+0.033 — those are ceilings on the *raw* probability,
which layer 2 never consumes. The correction is recorded as C11.)

Three things worth carrying forward:

- **Per-`(month, partition)` calibration cost ceiling.** Raw-probability ceilings are higher in both
  scopes (+0.0095 vs +0.0054 at fs1; +0.0331 vs +0.0140 at fs2). Calibration was added so one global
  threshold would be meaningful across 40 differently-scaled local models; it was frozen before the
  run and never adjudicated. First thing to revisit in any successor design.
- **Down-flipping is useless, not harmful.** Swept properly the optimum is zero flips, delta exactly
  +0.0000 in both scopes — a third independent confirmation, after Step 3 Variant A and B.
- **The model handicap was never lifted.** The Phase 3 calibration gate failed (`persist=0` gap
  0.0874 fs1 / 0.0738 fs2 against a 0.05 bar) and the task proceeded deliberately, so mechanism
  failure and model failure cannot be separated by this run.

Per the pre-committed switch trigger, FEWS NET work on this mechanism stops and **IPCCH becomes the
active direction**. The unresolved GeoRF-loses-to-persistence problem (section 3) is unchanged and
now blocks that direction too.
