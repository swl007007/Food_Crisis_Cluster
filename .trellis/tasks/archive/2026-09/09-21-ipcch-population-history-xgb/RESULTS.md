# IPCCH population-history and pooled XGBoost objective comparison — results

Run `pop-v1`, complete. All three pre-registered claims are **not supported**.
That is a complete negative result with one genuine positive inside it, not
missing evidence: every required comparison was computed, all 2,000 bootstrap
draws were valid, and every leave-one-year-out cell is defined.

## The three claims

| claim | required comparisons | verdict |
|---|---|---|
| **prediction gain** | `share_xgb` vs `rich_rf` **and** vs `persistence` | **not supported** |
| **formulation advantage** | `share_xgb` vs `rich_direct_xgb` **and** vs `fullpool_xgb` | **not supported** |
| **information gain** | `rich_direct_xgb` vs `binary_history_xgb` | **not supported** |

Required comparisons are conjunctive: one failing baseline fails the claim.

| comparison | mean ΔF1 | 95% CI | per-horizon Δ (1/3/6/12) | leave-year-out (23/24/25) | verdict |
|---|---:|---|---|---|---|
| share_xgb vs persistence | **+0.00737** | [+0.00136, +0.01310] | +.0017 / +.0048 / +.0182 / +.0047 | +.0082 / +.0068 / +.0068 | **stable gain** |
| share_xgb vs rich_direct_xgb | **+0.00871** | [+0.00327, +0.01414] | +.0006 / +.0041 / +.0211 / +.0089 | +.0087 / +.0081 / +.0105 | **stable gain** |
| share_xgb vs rich_rf | −0.00306 | [−0.00915, +0.00430] | −.0048 / −.0079 / −.0022 / +.0026 | −.0036 / −.0051 / +.0046 | no stable gain |
| share_xgb vs fullpool_xgb | −0.00263 | [−0.00914, +0.00342] | −.0114 / −.0025 / +.0025 / +.0008 | −.0013 / −.0052 / +.0011 | no stable gain |
| rich_direct_xgb vs binary_history_xgb | −0.00108 | [−0.00527, +0.00276] | +.0010 / +.0010 / −.0021 / −.0042 | −.0004 / −.0012 / −.0027 | no stable gain |

## The headline negative

**The 468 rich history columns do not improve the direct binary classifier.**
`rich_direct_xgb` vs `binary_history_xgb` is −0.00108 on average, and the sign
flips against the rich features as the horizon lengthens: +0.0010 at h=1 and
h=3, but −0.0021 at h=6 and −0.0042 at h=12. All three leave-one-year-out means
are negative. Adding continuous phase-share levels, changes, window statistics,
trends and threshold margins to a schema that already carries binary history
buys nothing, and at long lead times it costs.

This does not isolate continuous shares: the rich schema adds continuous shares
**and** a longer binary history together, and each arm's hyperparameters were
tuned per arm. It is a tuned-pipeline contrast, not a fixed-hyperparameter
ablation.

## The one real gain

`share_xgb` — regressing the continuous normalized P3+ share and thresholding
it — **beats persistence stably**: +0.00737, CI [+0.00136, +0.01310], positive
at every horizon and after dropping each target year. It also beats the direct
rich classifier stably (+0.00871).

That is worth recording against this project's history, where persistence has
repeatedly been the thing that could not be beaten. But it does not survive the
full claim: `rich_rf` and `fullpool_xgb` are both better than `share_xgb`, so
"the selected primary family beats everything it must" is false.

## Main-schedule F1, E_history, 2023-2025

| method | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| persistence | .68139 | .67682 | .67186 | .67592 |
| binary_history_xgb | .68150 | .67652 | .67103 | .67592 |
| rich_direct_xgb | .68249 | .67749 | .66897 | .67170 |
| correction_xgb | .68258 | .67839 | .67515 | .67615 |
| share_xgb | .68314 | .68163 | .69007 | .68063 |
| rich_rf | .68793 | **.68948** | **.69228** | .67803 |
| fullpool_xgb | **.69449** | .68411 | .68757 | **.67982** |

n = 16,002 / 15,907 / 14,960 / 12,404. Every arm shares these keys exactly.

The two strongest arms are the ones with the *least* constrained fitting
support: `fullpool_xgb`, which trains on the superset including rows without
persistence, and `rich_rf`. The same ordering appeared in the earlier CH/gate
ablation, where XGB was the strongest learned arm.

## Five findings worth carrying forward

**1. Persistence at h=1 and h=3 is the same vector.** On all 9,667 shared
development keys the latest valid observation at T−1 and at T−3 is literally
the same record, so the two horizons have identical confusion counts
(tp 2471, fp 492, fn 600) and identical F1, 0.81902552. IPCCH publishes in
bursts; a three-month lookback almost never reaches a newer record than a
one-month lookback. Any h=1 vs h=3 comparison of persistence is therefore not
measuring lead time.

**2. The tuned threshold pair is not clearly better than the definitional
cutoff.** `share_xgb` with its frozen, development-optimised `(t0, t1)` gets
.68314 at h=1; the fixed "predicted share > .20" rule gets **.70226** on the
same rows. The fixed rule also wins at h=3 (.68583 vs .68163) and loses at h=6
and h=12. Thresholds fitted on 2020-2022 transfer worse than the definition
they are approximating. This is reported, not acted on: §6 forbids letting a
supplemental score reselect the method after the freeze.

**3. The correction arm is almost a no-op.** Out of ~15,000 evaluation rows it
flips 36 (h=1), 108 (h=3), 211 (h=6) and 24 (h=12). At h=1, h=3 and h=12 the
selected `t0` is high enough that **no** b=0 row ever flips to crisis — the
model only ever learned to talk persistence *out* of a crisis call. Where it
does flip both ways (h=6), the 0→1 direction is net harmful: 59 beneficial
against 83 harmful.

**4. The gains live in Cadre Harmonisé.** At h=1, persistence and
`binary_history_xgb` score identically on CH (.5969) while the rich arms lift
it (`rich_direct` .6154, `rich_rf` .6163, `fullpool` .6349). On IPC everything
sits at .694-.704 and the learned arms barely move. CH remains the harder,
more improvable regime — consistent with the earlier ablation, which found CH
about 0.13 F1 below IPC across every arm.

**5. Development F1 is not out-of-sample F1.** Development numbers sit at
.78-.84 against main-stage .67-.69, because the thresholds are chosen on the
same development predictions they are scored on. The selection ledger is a
selection ledger, not a result.

## Selection and the freeze

Frozen from 2020-2022 development predictions only, before any main-schedule
prediction existed. Information cutoff 2022-12; the latest training target
reachable by any development fold is 2022-11.

Primary family, by equal mean of four horizon development deltas vs persistence:

| candidate | mean dev Δ vs persistence |
|---|---:|
| **share_xgb (selected)** | **+0.012880** |
| rich_direct_xgb | +0.012815 |
| correction_xgb | +0.004543 |

The margin is 6.5e-5 — essentially a coin flip, decided by the declared rule and
frozen before the main stage. Had it gone the other way, claim 2 would have been
"not applicable" instead of "not supported", and claim 1 would have been tested
against a *worse* arm: on main data `share_xgb` beats `rich_direct_xgb` by
+0.0087. The thinness of that margin is a limitation of the design, not a
result; a 6.5e-5 development difference does not identify a better family.

Selected configurations: `rich_rf` took R3 at h=1/3/6 and R4 at h=12; every XGB
arm's choice varies by horizon. Full ledger in
`runs/pop-v1/development/selection_ledger.csv` (144 rows).

## Budget and runtime

| | folds | fits | bound |
|---|---:|---:|---:|
| development | 136 non-empty (8 empty) | 4,506 | 5,184 |
| main | 110 non-empty (12 empty) | 660 | 732 |
| verification replay | 8 | 48 | reported separately |

21.0 h of development fit time and 4.3 h of main fit time, compressed to 97 and
24 minutes wall by 12 fold-workers. Every fit took the `model` route; no
constant-target or single-class fallback was reached anywhere in the run.

13 development folds and 12 main folds have evaluation rows but **no** rows with
persistence — new areas appearing for the first time. Only `fullpool_xgb` runs
there, which is exactly the routing the contract specifies.

## Recorded deviation from the contract

`technical-contract.md` §7 asks for sequential folds and candidates with
`n_jobs=1`, and also says a resource problem returns for a bounded adjustment.
The measured schedule was ~21 h sequential. It was run with `--workers 12`,
**with user approval**, keeping the frozen `n_jobs=1` inside every estimator.

This is verified rather than asserted. `--stage verify` re-ran eight main folds
— the first and last non-empty fold at each horizon, chosen without looking at
any score — **sequentially, one at a time**, 48 fits. All eight are byte
identical to the parallel run (`runs/pop-v1/validation/replay.json`).

## Reproduction

Everything below runs from committed evidence in a fresh clone, without the
731 MB feature matrix (hash-bound in `manifest.json` and `freeze.json`,
rebuildable by `prepare_data.py` from the pinned source):

```
python -B IPCCHPopulationHistoryExperiment/test_contracts.py
python -B IPCCHPopulationHistoryExperiment/run_pipeline.py \
    --run-dir IPCCHPopulationHistoryExperiment/runs/pop-v1 --stage select
python -B IPCCHPopulationHistoryExperiment/report_results.py \
    --run-dir IPCCHPopulationHistoryExperiment/runs/pop-v1 --out-dir OUT
```

Verified in a clean clone at `45b74fb`: 41/41 contract tests pass; the selection
replay reproduces `selections`, `primary_family`, `primary_mean_delta`,
`persistence_development_f1`, spec identity, matrix hash, information cutoff and
selection cohort exactly; the report replay reproduces `summary.json` exactly
apart from `run_dir`, which is the path it ran in.

`freeze.json`'s `code_sha256` names `prepare_data.py` and `run_pipeline.py` as
they stood at selection time. Both were edited afterwards, for replay plumbing
only — matrix-free loading and reading the matrix hash from the manifest. The
clean-clone replay is what shows those edits changed no selected value.

## Data and QC

Source gate passes exactly: 42,695 valid, 15,206 positive, 27,489 negative,
6,227 areas. 170,780 supervised rows (42,695 outcomes × 4 horizons), 139,567
with persistence and 31,213 without. Maximum 25 observations for any area.

The rich block has **0** infinities, **0** all-NaN columns and one constant
column (`hist_q5_m06_slope`, which needs three P5 observations inside six
months and never gets them). Median NaN rate across the 468 appended columns is
0.432 — high by design: a missing derived value stays missing and no row is
dropped for it.

## Limitations

* Retrospective. 2023-2025 outcomes were inspected before this design existed.
* Source-month alignment does not establish publication-time availability.
* Claim 3 is a tuned-pipeline contrast, not a fixed-hyperparameter ablation,
  and does not isolate continuous shares from longer binary history.
* The interval is conditional on the trained predictions; not a refit bootstrap.
  2,000/2,000 draws valid, 0 rejections, over a 52-country axis shared by every
  method and horizon.
* The primary family was chosen on a 6.5e-5 development margin.
* The RF imputer keeps the pinned `max_plus` rule; for a negative column its
  fill lands inside the column's range. Retained for baseline continuity and
  disclosed, not silently changed.
* A good F1 decision threshold is not calibration, and a mean predicted share is
  not a crisis probability.

## What this does not license

A negative result on these three claims is a complete outcome, not a mandate to
start another feature or model search. Two things it does point at, if anyone
wants them: `fullpool_xgb`'s consistent edge says the binding constraint is
*fitting support*, not features; and finding 2 says the threshold-selection
step, not the model, may be where this design loses ground.
