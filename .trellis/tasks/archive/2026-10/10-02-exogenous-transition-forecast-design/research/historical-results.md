# Historical 2021–2024 results (fixed recipe; spec metrics)

2026-10-03, Claude executor. This document covers historical execution plus an independent recount. It is **not** a whole-task pass and not the spot/close audit. Nothing was retuned from these scores. The actual-2025 phase is still pending (availability facts).

## Provenance

- **Run:** `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1`.
  - Selection c0b3c967…b8b → frozen recipe 2344ab59…573e → `historical.json` 67153c53…9d79 → `report.json` 4c0efad0…7b2b.
  - Recipe: strategy **A** at both horizons, frozen map `8965af6d6a724ba5d61d`; H4 G1+L1, H8 G4+L1.
- **Calendar:** H4 10 targets (2021-10..2024-10); H8 9 targets (2022-02..2024-10); × k = 0/1/2 = 57 folds. Excluded targets, with reasons: H4 2021-02/06; H8 2021-02/06/10 (an origin or a missed cycle not after the 2020-12 freeze).
- **Reconciliations, all with 0 problems:**
  - identity/inventory (`probes/historical_identity_check_summary.json` 4bce3b8c…);
  - independent fit keys 654/654 (`scenario_fit_keys_review_final.json` 1f02f807…);
  - local/gate (`probes/local_gate_reconcile_scenario_historical.json` 7751ff1e…).
- **Independent metric recount by the coordinator** (exit 0, problems = []):
  - 325,926 forecast rows, 138 country rows, 36 paired comparisons (including 12 unavailable-expert routes);
  - output hashes and bindings; forecast-to-report row equality; target and exact-origin truth; the argmax rule; Study1/2 counts, F1, precision and recall; the paired 2,000-draw country-block CIs; every country numeric field.
  - Copies preserved byte-exact: `probes/historical_metric_review.py` (759a4a69dc8437d6ec71dbd12dc4aebfb2003ecf64d8c608f0607d29484fa1ff) and `historical_metric_review.json` (88d21073430f6e565bd8da108d97ac0c0908eb99d6398da00b5e57e0570d2bf9).

## Metric definitions

- Primary metric: pooled crisis F1 (IPC ≥ 3) from summed confusion counts, with equal weight per area-target key.
- "Matched" means the identical keys where both genuine truth and the lawful persistence comparator exist.
- Uncertainty: paired country-block bootstrap, 2,000 draws, seed 42, 22 countries. CIs are linear 95%.
- Pooled comparator: the same-input fold-global model. It isolates the effect of the local models.

## Study 1: all eligible keys

Cohort: H4 57,180 keys (10 × 5,718), of which 52,379 have genuine truth. H8 51,462 keys, of which 47,405 have genuine truth.

Denominators:
- "Model F1/P/R" and "Δ vs pooled" use **all genuine-truth keys** (52,379 / 47,405).
- "Matched model F1", "Persistence F1" and "Δ vs persistence" use the **persistence-matched** subset ("Matched n").

| H | k | Model F1 / P / R (all genuine keys) | Matched n | Matched model F1 | Persistence F1 (matched) | **Δ vs persistence [95% CI] (matched keys)** | Δ vs pooled [95% CI] (all genuine keys) |
|---|---|---|---|---|---|---|---|
| 4 | 0 | .7864 / .8027 / .7706 | 52,174 | .7868 | .8028 | **−.015986 [−.0373, +.0057]** | +.001571 [−.0003, +.0051] |
| 4 | 1 | .7588 / .7458 / .7722 | 51,969 | .7597 | .7525 | **+.007265 [−.0091, +.0292]** | +.001067 [−.0001, +.0027] |
| 4 | 2 | .7136 / .7200 / .7074 | 51,764 | .7154 | .7198 | **−.004412 [−.0165, +.0064]** | −.000306 [−.0020, +.0009] |
| 8 | 0 | .7088 / .7376 / .6821 | 47,060 | .7096 | .7559 | **−.046298 [−.0961, −.0015]** | +.000995 [−.0020, +.0055] |
| 8 | 1 | .6964 / .6872 / .7058 | 46,855 | .6980 | .7313 | **−.033383 [−.0669, +.0032]** | +.000224 [−.0009, +.0022] |
| 8 | 2 | .6430 / .6641 / .6232 | 46,790 | .6448 | .6632 | **−.018419 [−.0628, +.0220]** | −.000487 [−.0034, +.0016] |

## Study 2: exact-origin non-crisis keys (onset)

Denominators:
- The F1 and Δ-vs-persistence columns use the **persistence-matched** risk-set keys.
- "Δ vs pooled" uses **all genuine risk-set keys**.
- Model onset recall uses all genuine risk-set onsets (3,245 at H4 / 3,681 at H8). Persistence onset recall uses only the matched onsets (e.g. 3,239 at H4 k1).
- The two recalls are therefore **not on identical keys**, and no direct model-versus-persistence recall comparison is claimed.

Study 2 uses exact-origin non-crisis keys only. Keys lacking exact-origin truth are excluded and counted, never filled:
- H4: 2,434 missing origin; 14,658 origin in crisis.
- H8: 2,971 missing origin; 12,927 origin in crisis.

| H | k | Keys: risk set (persistence-matched) | Model F1 (matched) | Persistence F1 (matched) | Δ vs persistence [95% CI] (matched) | Onsets: risk set / matched | Onset recall: model (all risk-set keys) / persistence (matched keys) | Δ vs pooled [95% CI] (all genuine risk-set keys) |
|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 35,287 (35,287) | .1500 | **.0000** | +.150049 [+.0815, +.2096] | 3,245 | .0946 / .0000 | +.009301 [−.0066, +.0355] |
| 4 | 1 | 35,287 (35,099) | .3997 | .3009 | +.098781 [+.0438, +.1482] | 3,245 / 3,239 | .4265 / .2661 | +.001794 [−.0001, +.0047] |
| 4 | 2 | 35,287 (34,917) | .3664 | .3450 | +.021382 [−.0039, +.0440] | 3,245 / 3,226 | .4065 / .3283 | −.000922 [−.0050, +.0023] |
| 8 | 0 | 31,507 (31,507) | .3313 | **.0000** | +.331319 [+.2479, +.4173] | 3,681 | .2532 / .0000 | +.008126 [−.0081, +.0305] |
| 8 | 1 | 31,507 (31,319) | .4098 | .3824 | +.027355 [−.0244, +.0861] | 3,681 / 3,662 | .4137 / .3045 | +.000894 [−.0010, +.0041] |
| 8 | 2 | 31,507 (31,264) | .3319 | .2703 | +.061552 [+.0176, +.1090] | 3,681 / 3,661 | .3399 / .2289 | +.002642 [+.0005, +.0057] |

**The k = 0 persistence F1 of 0 is mechanical.** Study 2 keeps only keys whose exact-origin label is non-crisis. Ordinary persistence repeats that label, so it can never predict an onset. The large k = 0 Study 2 deltas therefore do not show, by themselves, a useful warning capability. For k ≥ 1 the persistence comparator falls back to an older lawful label, and the comparison is informative.

## Coverage

- **Expert comparator.** No documented same-horizon expert table was supplied. Every key carries `no_documented_expert_table`; the 12 expert comparison routes are unavailable, not zero.
- **Unlabelled keys.** Target keys without genuine truth are kept as forecast-only rows (coverage). They are excluded from the metrics, never filled.
- **Country tables.** `scenario_report/country_h{H}_k{k}.csv` covers every cohort country. There are 138 country rows in total = 6 tables × 23 rows (22 countries plus one "unknown country / coverage only" row for forecast-only keys without a country). It is descriptive only, with no country-level significance claims.

## Negative results (preserved)

- **Study 1: no cell has a positive Δ-vs-persistence CI that excludes zero.**
  - The only positive point estimate is H4 k1 (+.0073), and its CI includes 0; it remains a positive estimate.
  - At **H8 k0 the deficit is significant**: Δ −.0463 with CI [−.0961, −.0015], entirely negative.
  - H8 k1 (−.0334) and H4 k0 (−.0160) lean negative, with CIs that include 0.
- **Historical parity screen.** At H8 k0, historical normal-scenario parity (−.046) is worse than the −0.02 screen used in development selection. This is reported, not acted on: historical scores never retune the recipe.
- **Local models add almost nothing over the pooled global** (Δ vs pooled on all genuine keys).
  - Study 1 Δ vs pooled ranges from −.000487 to +.001571, and every CI includes 0.
  - In Study 2 only H8 k2 has a CI excluding 0, at a tiny +.0026.
- **Interruption scenarios.** These are the operational purpose of the study.
  - In Study 1 the model is within about ±.02 of persistence at k = 1/2 for H4, and below it for H8.
  - In Study 2 at k ≥ 1, on the persistence-matched keys, the F1 delta CI excludes 0 only at H4 k1 and H8 k2. The k = 0 cells also exclude 0, but mechanically (persistence F1 = 0, see above). Onset recalls are reported on different denominators (above), so they are not compared directly.

## Supplementary: descriptive per-target-month Study 1 (matched keys)

Derived from the saved historical fold predictions with no fit or tuning: model crisis F1 minus persistence F1 on the matched keys of each target month. These are descriptive only, with no CIs. Each target month is a single cross-section.

| H | T | k0 | k1 | k2 |
|---|---|---|---|---|
| 4 | 2021-10 | −.0502 | −.0702 | −.0388 |
| 4 | 2022-02 | −.0350 | +.0118 | +.0148 |
| 4 | 2022-06 | −.0032 | +.0115 | +.0218 |
| 4 | 2022-10 | +.0048 | +.0046 | −.0070 |
| 4 | 2023-02 | −.0134 | −.0073 | +.0023 |
| 4 | 2023-06 | +.0074 | +.0390 | −.0387 |
| 4 | 2023-10 | −.0166 | +.0072 | +.0063 |
| 4 | 2024-02 | −.0340 | −.0029 | +.0058 |
| 4 | 2024-06 | −.0379 | +.0432 | −.0011 |
| 4 | 2024-10 | +.0097 | −.0016 | −.0094 |
| 8 | 2022-02 | −.1134 | −.0988 | −.0741 |
| 8 | 2022-06 | −.1314 | −.0780 | −.0863 |
| 8 | 2022-10 | −.1391 | −.0812 | −.0946 |
| 8 | 2023-02 | +.0099 | −.0357 | −.1019 |
| 8 | 2023-06 | +.0313 | −.0448 | +.0322 |
| 8 | 2023-10 | −.0169 | −.0012 | −.0140 |
| 8 | 2024-02 | −.0700 | +.0078 | −.0002 |
| 8 | 2024-06 | +.0001 | +.0152 | +.0685 |
| 8 | 2024-10 | −.0516 | −.0452 | −.0105 |

Months where the model beats persistence: H4 3/10 (k0), 6/10 (k1), 5/10 (k2); H8 3/9, 2/9, 2/9. The H8 deficit is concentrated in the three 2022 targets (−.07 to −.14). The months are not stable enough to support a claim of temporal consistency in either direction.

## Caveats (required disclosures)

1. **D18 conditional map-selection bias.** The historical gate replays use the map known at the outer origin. That map need not have existed at the internal origin V. Gate scores are not an independent temporal test.
2. **Earlier historical exposure.** 2021–2024 results from earlier experiments in this project were seen before this design. The recipe was frozen from 2018–2020 evidence only (cutoff 2020-12), but the exposure is disclosed.
3. **Availability reconstruction.**
   - Historical IPC CS availability uses the user-confirmed `reference_month_end` convention, with `evidence=reconstructed`. These are assumed dates, not verified publication timestamps.
   - Covariates are revised/latest values, not real-time vintages.
   - Excluded sources follow the committed alignment (129 features).
4. **Limited bootstrap.**
   - Resampling covers only 22 country blocks. The largest single-country share of crisis events is about 0.19.
   - The CIs are descriptive, with no multiplicity adjustment across the 12 H × k × study cells (24 paired comparisons with persistence and pooled).
   - Pooled crisis F1 weights each area-target key equally. It is neither country-balanced nor population-weighted.
5. **Fit budget.** No exact realized fit count exists. The schedule bound of 147,889 applies, plus a separately reported crash overhead of ≤ 1,323.
6. **Evidence limits.** Internal gate locals have support-only evidence. Feature values and booster internals are not verified.

_Documentation clarification (2026-10-03, coordinator review). Denominators were labelled: matched vs all-genuine keys, and the onset recall bases. The categorical "does not beat persistence" was replaced by the CI statement. No numbers were changed and no new analysis was run._
