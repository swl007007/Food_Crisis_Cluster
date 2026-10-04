# Results: confirmatory pooled-vs-persistence onset test (task 10-03)

2026-10-03/04, Claude executor. The study is pre-registered: prd.md D1–D12 and R1–R10 were fixed before any 2026-02 CS value was read. There was no tuning and no product-code change (identity fd25e2f7…), and no `trellis-audit` registration.

## Pre-registered outcome

**H1 (primary, S3): INCONCLUSIVE.**
- Pooled crisis F1 **.559** vs stale persistence **.376** on the onset risk set.
- Δ **+.183**, country-block 95% CI **[−.027, +.400]**.
- The point estimate favours pooled, but the interval includes 0, so H1 is not confirmed. This is not a refutation either.

**H2: not tested.** Under the fixed sequence (D5), H2 is tested only if H1 is CONFIRMED.

**Cohort and scenario.**
- Scenario S3: H4 forecast of 2026-02 from origin 2025-10, with Feb/Jun/Oct 2025 all treated as missed. Persistence therefore comes from 2024-10 (SD 2024-06). Gate intensity 3.
- Onset risk set: 2,910 keys (Oct-2025 non-crisis with genuine Feb-2026 truth, panel countries); 622 onsets; 18 countries.
- Excluded: 1,433 with crisis at origin, 0 without origin truth, 0 without persistence.

## All pre-registered cells

Columns: pooled F1 minus comparator F1, with 95% country-block CI (2,000 draws, seed 42).

| Scenario | Set | vs persistence | vs transition | vs IPC-history logistic | vs partitioned system |
|---|---|---|---|---|---|
| **S3 (primary)** | onset (n = 2,910; 622 onsets) | **+.183 [−.027, +.400]** (.559 vs .376) | +.244 [−.027, +.491] (vs .315) | +.119 [−.009, +.249] (vs .440) | .000 [.000, .000] |
| S3 | Study 1, all keys with truth (n = 4,343) | +.072 [−.024, +.198] (.804 vs .732) | +.116 [−.024, +.299] | +.108 [+.014, +.207] | .000 |
| S0 (secondary; Oct 2025 visible) | onset | +.583 [+.394, +.670]: **mechanical** (persistence onset F1 = 0) | +.583: mechanical (transition maps the non-crisis origin to non-crisis) | +.003 [−.204, +.208] (.583 vs .580) | +.004 [−.017, +.038] |
| S0 | Study 1 | +.071 [+.007, +.149] (.852 vs .780) | +.074 [+.010, +.156] | +.057 [−.029, +.152] | +.004 [−.004, +.019] |

S0 is reported only. Its onset comparisons with persistence and transition are mechanical by construction and are not evidence.

**Onset precision / recall in S3:**
- pooled .52 / .60;
- persistence .48 / .31;
- transition .43 / .25;
- logistic .28 / .99 (it flags nearly every onset-set key as crisis);
- partitioned system: identical to pooled.

## Partition versus pooled

- In S3 the partitioned system equals pooled exactly on both the onset set and Study 1. The local models covered 837 rows but changed the crisis decision on only 4 of them, with no net change to the confusion counts.
- In S0 the difference is +.004 (CI includes 0).
- This again shows no benefit from spatial partitioning, consistent with 10-02.

## Descriptive per-country onset table (S3; no tests)

Onsets are concentrated: Kenya 196, Afghanistan 154, Zimbabwe 77, Somalia 74 and Mozambique 41 make up 542 of the 622 onsets (87%).

| Country | Pooled vs persistence | Interpretation |
|---|---|---|
| Kenya | .71 vs .09 | pooled far ahead |
| Afghanistan | .69 vs .39 | pooled ahead |
| Madagascar | .59 vs .00 | pooled ahead |
| Zimbabwe, Mozambique, Somalia | equal | no difference |
| Haiti | .95 vs .98 | persistence slightly ahead |

Per-country values are in `S3/country_onset.csv`. With 18 blocks and this concentration, the interval is wide. Single-country patterns are hypothesis-generating only; for example, Kenya was the clearest negative case in 10-02.

## Interpretation (evidence, not recommendation)

- In the only blind window available, a 3-cycle simulated outage, the pooled model's onset F1 was higher than every simple comparator by point estimate (+.12 to +.24). All intervals for the decision comparisons include 0, so the pre-registered confirmation was **not achieved**.
- The evidence is directionally consistent with the 10-02 historical onset finding. It does not establish it.
- Among the secondary cells, intervals excluding 0 occur for S3 Study 1 against logistic (+.108) and for S0 Study 1 against persistence (+.071) and against transition (+.074). The S0 onset comparisons with persistence and transition are mechanical. Secondary cells are not part of the decision rule and must not be promoted.

## Disclosures and limits

- **Sample:** a single cross-section (one target month, one horizon) with 18 country blocks. Onsets are concentrated in five countries, so power is limited.
- **Scenario:** S3 is a simulated outage. The genuine 2025 interruption covered Feb/Jun 2025; Oct 2025 was masked by design. Being a 3-cycle gap, it is not directly comparable with the historical k = 1/2.
- **Availability:** reconstructed `reference_month_end` convention. Covariates are revised/latest values.
- **ACLED:** excluded by availability (D3), giving 110 features against 10-02's 129.
- **Truth:** exact full-name + country + canonical DBF-name mapping. Geometric continuity is not certified.
  - Feb 2026: 4,343 admitted; 1,223 unmatched names; 278 rows from six countries outside the panel; 21 with no genuine phase; 2 Kenya 2995/2996 cases.
- **Transition baseline:** in S3, 1,854 of 5,718 prediction rows fell back to overall frequencies, because target ages beyond 12 months are rare in training. This follows the pre-registered rule but weakens that baseline.
- **Prior exposure:** the 10-02 historical results were known; they generated the hypothesis.
- **Execution note:** predict attempt 1 failed in the driver's synthetic self-check before any fit. The driver was fixed (transition tables, self-check sizing) before attempt 2. The driver hash at predict is recorded in `predictions_frozen.json` and differs from the `pin` record. The method is unchanged.

## Artifacts

All under `C:\Users\swl00\geoxgb_runs\confirm-onset-v1\` unless noted.

| Artifact | sha256 |
|---|---|
| inputs_manifest.json | f91d7f8225c60996dc7f0010a22c37b5007fedb1e74479de56489b675f228eae |
| alignment_v3.json | 34247cbc380d0d060649993779f8255e685dc8f66daf704eb6a9333e39f764fb |
| extension_v3 CSV / manifest | 6b333599…2496c64 / ea0eb37c…bab569 |
| predictions_frozen.json | f1fd1fdcbafa464ee00a43d9f8a1fd4f0b7acbf26677b127b1c6e7e154092933 |
| truth_release_feb2026/release.json | 67b2f608fb081b38210823460fdc4df5fb80df35cef1b307709d10c781326ccb |
| truth_feb2026.csv / crosswalk_feb2026.csv | 75025655…d24819 / 822ba26b…ac74576 |
| evaluation.json | a321c97324c02cc6d224e65e182c2e7bffaf7dd156ff0ad2a1cded9f1909b93f |
| steps/check.json | 1891d18f…f37c |
| S3 fold.json / predictions / pooled / baselines | 85ec072a… / 8e482602… / c6f1a930… / f8f2160f… |
| S0 fold.json / baselines | 85558c1c… / 9d45ba43… |
| Driver `research/run_confirm.py` | recorded in predictions_frozen.json |

**Independent check** (`steps/check.json`, no package imports): prediction hashes equal the freeze; the primary F1 recount from keyed rows matches `evaluation.json`; 4,343 truth rows were joined with 0 unmatched; problems [].
