# Next-phase readiness: scen-freeze → scen-historical → scen-report; actual-2025 inputs

2026-10-03, Claude executor, written while the sole development launcher (PID 3552757) is running. Sources: the existing CLI `FEWSNETGeoXGBExperiment/scripts/run_experiment.py`, design.md and implement.md §6. No fit, no 2025 outcome read, no source tracing, no product/test edit.

## Commands (same pinned interpreter, diagnostic env, sequential; freeze and historical only after the full selection review)

```bash
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe   # from FEWSNETGeoXGBExperiment/
RUN='C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1'
$PY -B scripts/run_experiment.py --run-dir "$RUN" scen-freeze       # no fit: final consensus map(s) + frozen.json
$PY -B scripts/run_experiment.py --run-dir "$RUN" scen-historical   # fits: frozen recipe, 2021-2024, k=0/1/2
$PY -B scripts/run_experiment.py --run-dir "$RUN" scen-report       # no fit; optional --expert-table CSV
```

## Qualification and stop handling per H

`ab_select` (run_experiment.py:541–600) works from exact rationals:
- It first checks that the 72 fold identities equal `scenario_dev_plan()`.
- A strategy qualifies at a horizon when both of these hold:
  - its normal-scenario matched crisis F1 minus matched persistence F1 is defined and ≥ −0.02 (`SCEN_PARITY`, :477);
  - its k1 and k2 F1 are defined.
- Qualifiers are ranked by the mean of their k1 and k2 F1; exact ties go to A.
- With no qualifier, the horizon has no winner and the unmet criteria are recorded.

Downstream stop behaviour for a horizon with no winner:
- `scen-freeze` (:817) writes `{"released": false, "reason": …}` for that H and builds no map.
- `scen-historical` (:854) records the same and fits nothing for that H.
- `scen-actual` and `scen-report` carry `released: false` forward.

Design.md:107 governs this: do not pick a winner anyway. This is the existing code path; nothing can be skipped or retuned manually.

## Expected artifacts

| Phase | Writes (write-once; refuses to overwrite) |
|---|---|
| scen-select | `scenario_development/selection.json`: per-H decisions, per-strategy parity, k1/k2/mean F1, unmet criteria, 72 fold SHAs |
| scen-freeze | `scenario_maps/<map_id>/` (one strategy@final map per released H, from that strategy's candidates with E3 target ≤ 2020-12, `strict=False`, `k_max=0`, run_stage2.py:169) and `scenario_final/frozen.json` (strategy, map_id, route, G1/G4, L1, `selection_sha256`). It re-runs `accept_scenario_stage1` (~4.5 min) |
| scen-historical | `scenario_historical/<S>/h<H>/k<k>/<T>/` per fold: fold.json, predictions, pooled predictions, gate. Plus `historical.json`: per H the eligible targets, the excluded targets with reasons, truth coverage, and `frozen_sha256` |
| scen-report | `scenario_report/report.json` (Study1/Study2 per H×k; paired country-block crisis-F1 bootstrap vs persistence, same-input pooled and optional expert), `country_h<H>_k<k>.csv`, `keyed_h<H>_k<k>.csv.gz` |

Expected historical calendar (design.md:28 illustration; matches the month-end ledger):
- H4: 10 targets, 2021-10 to 2024-10;
- H8: 9 targets, 2022-02 to 2024-10;
- excluded: the Feb-2021-origin targets (H4 2021-06, H8 2021-10), whose two hidden cycles 2020-10/2021-02 are not after the freeze.

If both horizons are released, that is (10 + 9) × 3 = **57 folds**. `historical.json` (computed by `historical_targets`, :592) is authoritative.

## Checks before each launch

1. **After select:**
   - selection.json is accepted and its 72 fold SHAs match the files (`_accept_selection`).
   - Report the per-H decisions with their unmet reasons to the coordinator before freeze.
   - Also report the full development aggregates for both strategies, including negative results (implement.md §6).
2. **Before freeze:**
   - the launcher is gone;
   - the package tree is clean at the same HEAD (ca86bf7);
   - no `scenario_final/` exists.
3. **Before historical:**
   - frozen.json is accepted and bound to the selection SHA.
   - Fit budget. 147,889 total and Stage 1 ≤ 40,824 (launch-readiness.md:64) are the **pre-launch schedule bound** for one attempt per scheduled root. They do not automatically include repeated attempts, and the spec budget is not redefined here.
     - Failed attempts are kept separate: 21 partial native failures, preserved in `failed_stage1/`.
     - Their conservative overhead uses the per-attempt maximum implied by the Stage 1 bound: 40,824 / 648 = 63 fits per root attempt, so 21 × 63 = **≤ 1,323 fits**.
     - The operational check total is therefore schedule bound + failed-attempt overhead = 147,889 + 1,323 = **≤ 149,212**, reported as two separate components. The 21 successful resume attempts are within the schedule bound.
     - Retry fits are never discarded. No additional experiment or fit authorisation is implied.
     - There is **no exact actual fit count**: no saved record states one. This limitation is carried into the final check.
     - Stage 1 `completion.json` keys are candidates/code/g_selection/outputs/prepared/root/runtime/status, with no fit count.
     - Development `fold.json` has no fit count; its fold directories hold fold/gate/gate_pairs/predictions/pooled files.
     - `scenario_globals/` holds stored globals only. It misses the internal and local fits, and the partial fits of the 21 failed first attempts (7–39 saved checkpoints each, preserved in `failed_stage1/`).
     - Record the available evidence (schedule-derived Stage 1 ceiling, failed-attempt checkpoint counts, stored globals, gate.json local enablement) and disclose the unavailable exact count. Do not infer exact fits from files.
     - The 57 historical folds each add up to 1 global, 6 gate globals and locals.
   - Disk space.
   - The same diagnostic env.
4. **Open implement.md item:** reconcile the saved development/historical fold, role and internal-gate ledgers against lawful keys/masks. This is a post-run check, not a launch blocker.

## Local inputs still needed for scen-actual / scen-evaluate (absent today; each phase refuses without them)

| Input | Contract (code) | Needed for |
|---|---|---|
| `--actual-availability CSV` | Columns `country, product, origin_month, missed_cycles, evidence∈{verified_vintage, reconstructed}, source` (:479, :617). Product CS rows for every cohort country at origins **2025-06** (Oct H4), **2025-02** (Oct H8, Jun H4) and **2024-10** (Jun H8); non-negative integer counts; one count per country | scen-actual gate intensity |
| `--actual-scaffold JSON` + covariate CSV | Documented contract: manifest fields `path, sha256, first_month, last_month, overlap_months, source` (:480). The CSV has `FEWSNET_admin_code, date` plus the schema covariates, equal to the pinned panel on the overlap months and extending without gaps through the 2025 origins (monthly L=1 → 2025-05; annual GDP/CC per the alignment rules). Enforced checks are listed below the table | scen-actual features |
| `--truth-release DIR` | `release.json` with `approved: true, approved_by, crosswalk, truth_file, truth_sha256, frozen_actual` (= SHA of `scenario_actual/actual.json`), plus a truth CSV of unique `area, target_month, class_code 0..3` (:1017) | scen-evaluate, only after actual predictions are frozen. June 2025 is coverage-only without truth |
| 2025 admin crosswalk | Named in release.json; maps 2025 units to `FEWSNET_admin_code` (design D7 item 1) | the truth release |
| `--expert-table CSV` (optional) | Columns `area, issue_month, product, horizon, validity_start, validity_end, class_code, release_date, evidence, source` (report_fourclass.py:369). Same horizon, exact origin, released by origin-month end; otherwise each key keeps `no_documented_expert_table` | scen-report / scen-evaluate expert comparison |

Extension: documented contract versus enforced checks (line evidence; no fix in this pass).
- Enforced in `load_extension` (run_experiment.py:658–694):
  - manifest field presence and non-empty `overlap_months`;
  - the file SHA;
  - unique area-month keys;
  - overlap keys equal to the pinned panel, with covariates equal within 1e-9 (NaN = NaN);
  - at least one month after the pinned panel.
- Enforced downstream: `scenario_context` builds `ff.Scaffold` on the concatenated panel (:721–722). `Scaffold.__init__` (fourclass_features.py:58–70) refuses anything other than a complete area × month grid (`len(panel) == areas × (max − min + 1)`, no duplicate keys). That refuses any internal month gap, including a gap between the pinned end and the first extension month, as well as any missing area-month row or any new area lacking the pinned months.
- **Not enforced anywhere found:**
  - that `first_month`/`last_month` in the manifest equal the file's actual range;
  - that the extension reaches each required origin/source month. `Scaffold.at` (:77–83) returns NaN outside the grid, so a short extension silently yields native-NaN features rather than refusing;
  - per-covariate non-missingness. Native NaN is allowed by design, so source coverage must be reviewed, not assumed.
- **Explicit pre-actual check (manual, before scen-actual):** verify the extension's actual first/last month against the manifest. Verify that it covers every required source month for each 2025 origin (static sources at O through 2025-06; monthly dynamic at O−1 through 2025-05; annual reference years per alignment.json). Tabulate per-covariate missingness for the actual cases against the pinned-panel baseline. Do not expand product checks speculatively.

**Open point for coordinator review (not investigated further):** the prepared release ledger ends at the 2024-10 cycle. Before scen-actual, confirm that this ledger, as used by the k=0 outer forecast in `scenario_context`, agrees with the actual-availability table for the 2025 origins. Any 2025 cycle rows would need their own evidence. A ledger ending at 2024-10 is consistent with "no CS input after 2024-10" only if the actual availability table and its source coverage support that for every cohort country. This reconciliation stays a pre-actual check. It needs no protected outcome values and no source tracing beyond the existing records.

## Addendum: actual-2025 local input readiness (metadata/keys only, 2026-10-03)

Bounded local pass. It read headers only, plus the existing D7 key results (`probes/d7_admin_keys_results.json`, `d7-sources-admin.md`, `d7-panel-climate-lineage.md`, `2025-outcome-metadata.md`, `probes/d7_oct2025_name_join_log.txt`). No covariate, CS, phase, forecast or truth values were read; there was no web access, no assembly and no launch. `S = Analysis/1.Source Data`.

### Required months under the frozen rules

| Case (target, H) | Origin O | Static sources read at O | Monthly source month (L=1, O−1) | GDP ref year (Y−1 from July) | CC ref year (Y−2) |
|---|---|---|---|---|---|
| 2025-10, 4 | 2025-06 | 2025-06 | 2025-05 | 2023 | 2023 |
| 2025-10, 8 / 2025-06, 4 | 2025-02 | 2025-02 | 2025-01 | 2023 | 2023 |
| 2025-06, 8 | 2024-10 | 2024-10 (pinned) | 2024-09 (pinned) | 2023 | 2022 |

The annual years come from `annual_reference_year` (fourclass_features.py:163–172) with alignment.json, read at value_month 12; the 2022-12 and 2023-12 rows are inside the pinned panel (2010-01..2024-12, 5,718 ids × 180 months, no duplicates).

Static sources are read at O itself, not O−1: `covariate_features` docstring (fourclass_features.py:197) and `end(name)` returning `origins` for `kind == "static"` (:215–217). `Scaffold.at` returns NaN beyond the grid (:77–83).

The extension scaffold therefore needs a complete grid for all 5,718 areas through **2025-06**, for the static sources at the Oct-2025 H4 origin. Monthly dynamic input still ends at **2025-05** (O−1). The annual reference dates are as listed.

### Covariate extension candidates (keys/headers)

| Candidate | Keys/schema | Coverage | Usable by `load_extension` as-is? |
|---|---|---|---|
| `S/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.csv` (sha 41f02be9…, unpinned) | `FEWSNET_admin_code`, `date` (YYYY-MM-DD; parsed by `str[:7]`); 88 columns; all 51 admitted sources present | 2010-01..2026-04; 5,718 ids/month; keys identical to pinned over 2010-01..2024-12 (d7-sources-admin.md:117–119); 2025-01..06 complete (5,718 ids each; duplicates only in 2025-10/2026-02) | **No.** The id-2996 duplicates in 2025-10 and 2026-02 (d7-sources-admin.md:59, 65) trip the whole-file duplicate refusal (run_experiment.py:678) |
| `S/assembled_FEWSNET/…_2025_combined.normalized-v1.csv` (sha 510375f5…) | same keys/88 columns; all admitted sources present | 2010-01..2026-04, deduplicated, 5,718/month | Keys pass. Values differ by construction for the recomputed climate z-scores (global rolling; d7-panel-climate-lineage.md:16–22, 40) |
| `S/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025.csv` | `admin_code`/`year`/`month` with no `date` or `FEWSNET_admin_code`; lacks GDP and CC | 2025-01..2026-04 only | **No:** no overlap months, renamed keys, missing annual sources |

What can be assembled under the frozen rules (not assembled in this metadata-only pass; pending pre-actual checks; routine assembly is within the user-authorised spec execution):
- a truncated extension from the combined panel running at least through **2025-06**, so the static sources at O are covered, and at most through 2025-09, which avoids the 2996 duplicates;
- with its CSV SHA and a manifest of `path, sha256, first_month, last_month, overlap_months` (a subset of 2010-01..2024-12) and `source`.

Unresolved pre-actual checks (not done; values not read):
1. **Value agreement on the overlap.** `load_extension` compares all of `_covariate_columns(schema)` (run_experiment.py:644–645; 28 static + 41 dynamic = 69 schema sources). That includes the 18 sources excluded from features, such as Rainf/Tair z-scores. If the combined panel's full-sample z-scores differ from the pinned panel's, the extension is refused even though those columns never enter a feature. This was already listed as unchecked in d7-panel-climate-lineage.md:38. It needs either a passing value comparison or a reviewed decision; no product change now.
2. The month-reach and missingness checks above, kept separate for the two kinds:
   - **static sources at O:** all 5,718 areas present at 2025-02 and 2025-06 (2024-10 pinned);
   - **monthly dynamic sources at O−1:** 2025-01 and 2025-05 (2024-09 pinned), plus any trailing window months back from those endpoints.
   - Missingness is tabulated per kind against the pinned-panel baseline. Native NaN stays allowed, and no value is manufactured.

### Actual availability table (`--actual-availability`)

| Origin | Assemblable from existing records? |
|---|---|
| 2024-10 (case 2025-06, H8) | Yes, under the frozen convention. The committed ledger has the 2024-10 CS cycle for its countries (reference_month_end, `reconstructed`), so `missed_cycles=0` per ledger country can cite the ledger. Countries in the forecast cohort without a 2024-10 ledger row need their own entry |
| 2025-02, 2025-06 | **No. Missing field: `missed_cycles` with `evidence` and `source`, per country, for the Feb-2025 and Jun-2025 CS cycles.** Local metadata only shows file coverage: no Feb/Jun-2025 CS reporting dates in `2025_2026_FEWSNET.csv`, and the chunk folder has non-empty CS files only for 2025-10, 2025-12, 2026-01 and 2026-02 (d7-sources-admin.md:38). Per instruction, an absent file is not evidence of non-release. Branch stopped |

Also reconcile the prepared ledger (ends 2024-10) with this table before scen-actual (above).

### Truth release and expert table (not opened)

- **October 2025 truth:**
  - Candidates are `S/Outcome/FEWSNET_IPC/2025_2026_FEWSNET.csv` (CS, 5,573 raw rows, 21 countries) and the derived `FEWS_2025.csv`.
  - The crosswalk is unresolved. fnid never matches by direct key (d7-sources-admin.md:76). The existing name join matched 4,480 and left 1,093 unmatched; DRC is wholly unmatched and Ethiopia has 645 unmatched (d7_oct2025_name_join_log.txt).
  - Missing: the reviewed crosswalk file, plus the `release.json` fields `approved_by`, `crosswalk`, `truth_file` and `truth_sha256`. `frozen_actual` exists only after scen-actual.
- **June 2025 truth:** no local source was found (2025-outcome-metadata.md). It stays forecast/coverage-only; nothing is substituted.
- **Expert table:** no documented same-horizon table exists locally. Each key keeps `no_documented_expert_table`.

### Summary of genuinely missing inputs

1. Per-country Feb/Jun-2025 CS `missed_cycles` evidence and source.
2. A value-agreement result, or a reviewed decision, for the extension overlap (all 69 schema sources).
3. The truncated extension CSV, running at least through 2025-06, and its manifest. Not assembled in this metadata-only pass; pending pre-actual checks.
4. The reviewed 2025 crosswalk (DRC and Ethiopia gaps) and an approved truth release.
5. An optional documented expert table.

## Addendum: covariate-only overlap and actual-month missingness (2026-10-03)

**What was run.** Probe `research/probes/extension_overlap_check.py`, pinned Windows Python, 33 s. Output: `research/probes/extension_overlap_check_summary.json`, which holds counts only and no raw values. Each file was read with explicit `usecols` = `FEWSNET_admin_code, date` plus the 69 schema covariate sources. No IPC, outcome or expert column; no fit, no RUN write, no assembly.

**Files compared.** The pinned panel `S/FEWSNET_forecast_unadjusted_bm.csv` against the combined `S/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.csv`.

**Shared keys.** All 1,029,240 pinned keys (5,718 areas × 180 months, 2010-01..2024-12) are present one-to-one in the combined panel. The combined panel's only duplicate-key months are 2025-10 and 2026-02.

### Overlap agreement, `load_extension` rule (`np.isclose(equal_nan=True, rtol=0, atol=1e-9)`, so equal signed infinities agree), all shared keys

Corrected 2026-10-03 after coordinator review. The first run used `abs(x−y) ≤ 1e-9`, which wrongly counted equal same-sign infinities as mismatches. The probe now uses the `np.allclose` semantics of `load_extension` and was rerun on all shared keys.

| Group | Sources | Sources with any mismatch | Mismatching cells |
|---|---|---|---|
| Admitted (28 static + 21 monthly + 2 annual) | 51 | **0** | 0 |
| Excluded | 18 | **2** | 2,058,480 |

The two excluded z-scores still mismatch on every shared key under the corrected rule. **Same-signed infinities on both sides: 0 for each**, so the earlier inf semantics did not cause the result. The real mismatches are:

| Source | Mismatching keys | Both finite and different | Involving an inf (other side finite, NaN or opposite sign) | NaN pattern differs | Max finite abs diff | Months |
|---|---|---|---|---|---|---|
| `Tair_zscore` | 1,029,240 | 997,920 | 30,780 | 540 | 13.80 | all 180 |
| `Rainf_zscore` | 1,029,240 | 997,380 | 30,960 | 1,080 | 12.25 | all 180 |

The categories can overlap: an inf against a NaN counts both as inf-involved and as a NaN-pattern difference. The other 16 excluded sources agree exactly.

**Blocker 1.** `load_extension` compares all 69 `_covariate_columns(schema)` (run_experiment.py:644–645, 684–687). So an extension taken from the combined panel is refused on these two **excluded** z-scores, even though all 51 admitted sources agree (within atol 1e-9) and the z-scores never enter a feature. Truncating the extension does not help: the mismatch spans the whole overlap. Under the corrected `np.allclose` comparison, this rests on finite-value differences (about 997k keys per source, max |diff| up to 13.8), not on infinity handling. Resolving it needs a reviewed decision, for example limiting the overlap comparison to the admitted sources, which would be a product change requiring impact/check and review. Substituting pinned-equal values into the extension would manufacture agreement and is not done.

### Actual-month missingness (combined panel) versus the pinned panel at the same 2024 month

| Admitted sources | Required month(s) | Combined missing / 5,718 rows | Pinned 2024 baseline missing |
|---|---|---|---|
| 28 static (read at O) | 2025-02, 2025-06 | 0 and 0 (every source) | 0 and 0 |
| 2 FLDAS monthly (`Rainf_f_tavg_mean`, `Tair_f_tavg_mean`; O−1) | 2025-01, 2025-05 | 0 and 0 | 0 and 0 |
| 19 ACLED monthly (O−1) | 2025-01 | **1,304** per source | 0 |
| 19 ACLED monthly (O−1) | 2025-05 | **5,718** per source (all areas) | 0 |
| 2 annual (GDP, CC) | reference rows 2022-12 / 2023-12 | inside the pinned panel | — |

The ACLED gap grows through 2025: one source (`event_count_battles`) is missing for 0 areas each month 2024-10..2024-12, then 1,304, 1,417, 1,455 and 2,586 areas in 2025-01..04. From 2025-05 onward every area is missing, through 2026-04. The 19 ACLED sources share the same pattern at the two required months.

**Blocker/limitation 2.** Under the frozen policy, unavailable values stay native NaN with no backfill:
- **Oct-2025 H4** (origin 2025-06, source month 2025-05) would run with **all 19 ACLED features missing for every area**. Those features were never missing in 2010–2024 fitting.
- **Origin 2025-02** (Oct-2025 H8 and Jun-2025 H4) has 1,304 areas without ACLED.

This file coverage does not show whether ACLED values were actually unreleased at those dates. That is a source question, not traced here. The consequence is a covariate-coverage shift for the actual cases. It must be disclosed, and it is not repaired by imputation.

### Pre-actual status after this check

Done:
- admitted-source overlap agreement (within rtol 0, atol 1e-9, equal_nan; not byte-exact);
- static and FLDAS month reach and missingness (complete).

Open:
1. A reviewed resolution of the excluded z-score overlap refusal.
2. A reviewed disclosure/acceptance of the ACLED 2025 coverage gap (native NaN, no imputation).
3. The pending user question on Feb/Jun-2025 CS missed cycles.
4. The truth crosswalk and release.

Extension assembly waits for item 1.

## Adversarial review: two-source covariate splice (2026-10-03; no assembly, no product edit)

**Proposal.** A 2024-12 overlap block taken from the PINNED panel, plus the 2025-01..06 rows from the COMBINED panel. All 69 sources, no patching, missingness preserved. The manifest names both raw files and hashes, and links the full-key admitted check.

**Verdict: valid, with conditions. No counterexample found.**

- **No spec requires one raw file.** Neither design.md nor prd.md requires the overlap rows to come from the same raw file as the later rows.
  - The overlap equality is a code contract (`load_extension`, run_experiment.py:658–694; PROGRESS line 251).
  - What D1/D7 require is an origin-legal admitted input, fixed-source identity for static predictors (design.md:13) and source/hash provenance.
- **No admitted input changes.**
  - `load_extension` appends only months after the pinned maximum (`later = ext[date > pinned.max]`). The pinned panel stays authoritative for ≤ 2024-12 whatever the overlap block holds.
  - Excluded sources and their legacy derivatives never enter features (`covariate_features` skips `kind == "excluded"`, fourclass_features.py:221–223; `aligned_feature_names`, :156–160). So the combined panel's differently standardised 2025 z-scores sit in the scaffold arrays but in no feature.
  - Features are read only at origin-based months (availability.py:236). The scaffold needs rows through 2025-06 only, which avoids the 2996 duplicates in 2025-10/2026-02.
- **Leakage boundary unchanged.** The pinned labels/IPC are untouched. The file holds keys plus covariates only.
- **The alternative is worse.** A product edit restricting the overlap comparison to the admitted sources would change `code_identity()`. `_identity_problems` (acceptance.py:44–50) would then refuse every accepted Stage 1, fold, selection and frozen record. The splice keeps code and model identity unchanged.

**Weakness that must be disclosed, not hidden.** With the splice, the in-code overlap check passes **by construction** and gives no evidence about the combined source. The identity evidence moves entirely to the external probe:
- 51/51 admitted sources agree on all 1,029,240 keys (rtol 0, atol 1e-9, equal_nan; not byte-exact, e.g. Rainf finite max |diff| 1.6e-27);
- static identity across the boundary.

Neither proves that the combined panel's 2025 rows came from the same producer run as its history. Admitted history agreeing within 1e-9 is the strongest local evidence available.

**Static identity across the boundary: checked now, PASS.** For all 28 static sources and all 5,718 areas, the combined panel's 2025-01..06 values equal the pinned 2024-12 values (`np.isclose`, equal_nan, rtol 0, atol 1e-9): 0 differences in every month.

### Facts the assembly probe must record (and the manifest must cite)

1. **The file.**
   - Columns: `FEWSNET_admin_code`, `date` (one format, `YYYY-MM`) and exactly the 69 schema sources. No IPC, outcome, expert or other column.
   - Keys: 5,718 × 7 months (2024-12..2025-06). Unique, with the same id set as the pinned panel.
2. **Overlap block.** 2024-12 equals the pinned 2024-12 for all 69 sources, by both value (NaN- and inf-equal) and key set.
3. **2025 block.** 2025-01..06 equals the combined panel's rows for all 69 sources, by value and key set, with no transformation. Its missingness counts equal the source counts; the ACLED counts are as tabulated above.
4. **Hashes.** Pinned raw sha (611f9e77…, the prepared `sources.json` value). Combined raw sha (41f02be9…). The assembled file sha. The probe script and summary shas, including `extension_overlap_check_summary.json`.
5. **Manifest.**
   - Required fields: `path, sha256, first_month = 2024-12, last_month = 2025-06, overlap_months = ["2024-12"], source`.
   - `source` states:
     - that this is a two-source splice, with component file, sha and month range for each;
     - that the overlap check is satisfied by construction;
     - that the combined panel's historical `Tair_zscore`/`Rainf_zscore` differ from the pinned panel and are **not** certified identical (excluded, never in features);
     - that the 51 admitted sources and the static boundary identity were verified separately (cite the summary sha).
   - Extra keys are tolerated by `load_extension`, which checks only that the required fields are present.
6. **Before scen-actual:** first/last month and month reach are verified against the manifest (they are not enforced in code), and the ACLED 2025 coverage gap is disclosed.

## Assembled: actual-2025 covariate extension (two-source splice, 2026-10-03)

**Directory.** `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.actual-inputs-v1\` (new, written once; WSL `/mnt/c/Users/swl00/geoxgb_runs/scen-b43ef6a-v1.actual-inputs-v1/`). No fit, no outcome/IPC/expert column, no product/test edit, no raw modification.

| Artifact | sha256 |
|---|---|
| `covariate_extension_2024-12_2025-06.csv` (40,026 rows) | 0d6b78ca2fda21f73b8bfaeb6b916dd72fcb250f020fbea7be54299be9038b05 |
| `extension_manifest.json` (the `--actual-scaffold` input) | da9b3084cdd4cf41361e02fcfc5c46ae803bb433ca5e03c7cd26b11a20aa9c9c |
| `assembly_report.json` | 26884dffd5f5c0ee0582e4f2ec70fa2762d9af4622e98cd6fb9d67aff96ff0b3 |
| `verification_report.json` | 103a363f19628644f28bb450812c37ee0fa61559f538d8baf60722d708fee943 |
| probe `research/probes/assemble_actual_extension.py` | 50ac84118ad44246e79ca93f66555a4f56a3033eaf79b19433f420e058be39cc |
| probe `research/probes/verify_actual_extension.py` | 421860dfd54810b9a99aec63fe5742ee739d94d9100f1ff33d0404a2cd298345 |

**Assembly** (stdlib csv, 21 s):
- Components: pinned rows for 2024-12 (5,718) and combined rows for 2025-01..06 (5,718 each).
- Columns are exactly `FEWSNET_admin_code, date` plus the 69 schema sources.
- Value fields are the original source strings. Only the date key is written as YYYY-MM.
- Raw hashes were identical before and after: pinned 611f9e77…f651, combined 41f02be9…6178.
- Note: the AEZ booleans are spelled `false` in pinned and `False` in combined. Both are kept verbatim and parse to the same booleans, as the loader run below confirms.

**Manifest.**
- Fields: `first_month` 2024-12, `last_month` 2025-06, `overlap_months` ["2024-12"]; the alignment sha (the committed ca9e9a66… alignment as prepared); both components with file, sha and months; evidence shas for the full-overlap probe and summary, the assembly and verify scripts, and the assembly report.
- The `source` text states:
  - that the file is an explicit two-source splice;
  - that the loader overlap check is **true by construction**;
  - that the combined panel's historical Tair/Rainf z-scores **differ and are not certified identical**;
  - the ACLED gap, the revised/latest-value limitation, and that the 2025 producer run is not independently verified.

**Verification (Windows pinned stack, 30 s), all PASS:**
- **Raw and output hashes** match the assembly record.
- **Key and column inventory:** 40,026 unique keys; each of the 7 months has exactly the same 5,718 ids; the header equals keys plus the 69 sources.
- **Each emitted block** equals its named source string-for-string: 0 mismatching rows; key sets equal.
- **Static boundary:** 0 differences, 2024-12 against every month 2025-01..06, for all 28 static sources.
- **Real loader path:**
  - `_certified_panel` (pinned sha re-checked), then `load_extension(manifest)`, then `Scaffold`: 5,718 areas, complete grid 2010-01..2025-06.
  - Then `covariate_features` under the frozen alignment at origins 2025-02, 2025-06 and 2024-10.
- **Every admitted value equals the expected month** (0 mismatches):
  - static sources at O (2025-02 / 2025-06);
  - monthly sources at exactly O−1 (2025-01 / 2025-05);
  - annual GDP at the 2023-12 reference row and CC at 2023-12 (2024-10 origin: 2022-12), all pinned rows.
- **Origin 2024-10:** covariate features are identical with and without the extension.
- **Dimensions:** `aligned_feature_names` = **129**. `covariate_features` returns 54 columns: the 51 admitted sources plus the 3 target-calendar terms (`target_year`, `target_month_sin`, `target_month_cos`). The other 75 aligned features are IPC-history features built separately. No feature-policy change.
- **Monthly NaN cells** (19 ACLED sources, native NaN): 24,776 at O 2025-02 (1,304 areas × 19) and 108,642 at O 2025-06 (5,718 × 19). These match the missingness tabulated earlier.

**Status.** The covariate-extension input for `--actual-scaffold` is assembled and verified. Still open before scen-actual:
1. The actual availability table (Feb/Jun-2025 CS missed cycles; pending user question).
2. The ledger reconciliation for the 2025 origins.
3. Disclosure of the ACLED 2025 gap in reporting.
4. The truth crosswalk and release (for scen-evaluate only).

### Manifest wording correction (v2)

The coordinator noted that the v1 manifest said the admitted history was "equal … exactly". The overlap probe uses `np.isclose(rtol=0, atol=1e-9, equal_nan=True)`, and some sources differ by tiny amounts (e.g. Rainf finite max |diff| 1.6e-27). v1 is preserved unchanged. The corrected manifest changes only the disclosure.

| File | sha256 |
|---|---|
| `extension_manifest.json` (v1, superseded wording) | da9b3084cdd4cf41361e02fcfc5c46ae803bb433ca5e03c7cd26b11a20aa9c9c |
| **`extension_manifest.v2.json`** (use for `--actual-scaffold`) | 72a766c626d699814aa532f8f973d9c5dde9fd041bf15bb64bd99d6ef44c81d6 |
| `manifest_binding_v2.json` | 5a457939835592c191977bec3811d1c800b461d3ef27c8263b553b667f4c439a |

- v2 differs from v1 only in `source`, which now says the admitted and static sources agree within rtol 0, atol 1e-9 and equal_nan, not byte-exact. It also adds a `supersedes` field naming v1 and its sha.
- The CSV (0d6b78ca…), components, months and evidence are unchanged.
- `load_extension(v2)` runs, and its panel is identical to the v1 result. The scaffold is 5,718 areas over 2010-01..2025-06.
- `verification_report.json` (103a363f…) stays valid for the CSV.

### Crosswalk key feasibility (coordinator metadata probe, 2026-10-03; not boundary certification or a truth release)

Oct-2025 CS has 5,573 unique fnid/full-name rows. An exact full-name join against all FEWSNET.csv history, excluding names that map to several area codes, gives **4,479 one-to-one matches and 1,094 unmatched or ambiguous**. There are 0 duplicate matched area keys, and the names agree for all 20 matched countries.

This supersedes the earlier 4,480. That count included the ambiguous Kenyan name "Northwestern Pastoral Zone, Kerio Delta, Turkana Central, Turkana, Kenya", which maps to both 2995 and 2996. Restricting the lookup to 2024-10 names gives 4,478 / 1,095.

Constraints: no join is selected by resulting score; no labels are duplicated across 2995/2996; outcome values stay unopened. The reviewed crosswalk and the approved truth release remain open.

## Addendum: actual CS availability resolved; scen-actual preflight (2026-10-03)

**Supersedes** the pending statements above: the open point on line 99, item 3 of the covariate pre-actual status, and item 1 of the assembled-extension status.

**User confirmation.** The user answered "可以确认" to the explicit all-covered-countries question. The answer was given in the coordinator conversation and relayed to the executor via Herdr. It says: no new CS after the 2024-10 cycle was available at the 2025-02 or 2025-06 origins.

**Table.** `research/launch/actual_availability.csv`, sha 185e2c790d0b9d11102bceea0a5925947b99116ad4d3531a1148974eb9804b59.

Wording correction: the attestation is attributed to the coordinator conversation, relayed via Herdr. The previous wording ("in the executor session") is preserved as `launch/actual_availability.v1_superseded.csv` (sha 673651082fd1ee33be390f66ce7ce596e2756971ae4bb66a318f34316acb556c), with its preflight `probes/actual_preflight_summary.v1_superseded.json` (da5f4301…).
- 66 rows: 22 cohort countries × origins 2024-10 / 2025-02 / 2025-06, with missed service cycles 0 / 1 / 2.
- Product CS; `evidence=reconstructed`.
- Source column: the 2024-10 rows cite the inherited month-end convention and the committed ledger sha; the 2025 rows cite the user attestation. Neither claims verified vintage timestamps.
- Note: `*.csv` is git-ignored here, so committing needs `git add -f` (as for the ledger).

**Reconciliation.**
- The prepared release ledger has no CS cycle after 2024-10. Every country's last released cycle is 2024-10 except **SD (2024-06)**.
- Latest lawful CS label per country:
  - origin 2024-10: 2024-10, age 0 (21 countries); SD 2024-06, age 4;
  - origin 2025-02: age 4; SD age 8;
  - origin 2025-06: age 8; SD age 12.
- SD keeps its forecast keys and its older June label at the true date and age. No October label is manufactured. The attested service-cycle count is applied uniformly and is not read as an extra missed SD publication; no separate evidence of one exists.

**Preflight.** Probe `research/probes/actual_preflight.py` (388e8e95…). Summary `actual_preflight_summary.json` (2a78935695bdc681102b3b9d52ded764a9638378ea5760c927a7adf1406154c9), rerun on the final table: 0 problems.
- `actual_gate_intensity` (the code's own refusal path) returns {22 countries: 0} at 2024-10, {22: 1} at 2025-02 and {22: 2} at 2025-06.
- Cohort countries = 22 (Availability.countries).
- Extension manifest v2 (72a766c6…): scaffold of 5,718 areas, 2010-01..2025-06.
- `_accept_frozen` passes: A on map 8965af6d… for both H.
- `scenario_actual` is absent (fresh output). Code identity fd25e2f7… (65 files) equals the selection's.
- **Initial preparation outcome, preserved:** the first 63-row draft omitted SD while its 2024-10 status was being checked. The code refused all three origins for the missing SD entry (`actual_preflight_summary.initial_63rows.json`, d977878b…; draft `launch/actual_availability.pending_SD.csv`, aee0df58…). This was a preparation outcome, not a source blocker. After coordinator steering on the user's all-countries answer, SD was included with the attested counts.

**Cases.** `ACTUAL_CASES` gives 4 cases: (2025-10, H4) at O 2025-06, (2025-10, H8) at O 2025-02, (2025-06, H4) at O 2025-02 and (2025-06, H8) at O 2024-10. Each is a k = 0 outer forecast on the real ledger, with the per-country gate intensity from the table. Truth is never loaded.

**Exact command (not launched).** Single launcher `research/probes/actual_launch.sh` (sha bd713324…473); pinned Windows Python, diagnostic env; it refuses a dirty package tree or an existing `scenario_actual`:

```bash
L=.trellis/tasks/10-02-exogenous-transition-forecast-design/research/probes/actual_launch.sh
G=/mnt/c/Users/swl00/geoxgb_runs
setsid bash -c 'echo $$ > "$0"; exec bash "$1" "$2" "$3"' "$G/scen-b43ef6a-v1.actual.pid" "$PWD/$L" \
  'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\.trellis\tasks\10-02-exogenous-transition-forecast-design\research\launch\actual_availability.csv' \
  'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.actual-inputs-v1\extension_manifest.v2.json' \
  > "$G/scen-b43ef6a-v1.actual.nohup.log" 2>&1 < /dev/null &
```

This runs `python3.12.exe -B scripts/run_experiment.py --run-dir 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1' scen-actual --actual-availability <table> --actual-scaffold <manifest v2>`.

**Prerequisite.** The package tree currently shows `README.md` modified: documentation-only status fixes, sha c5ff299a…9941. It is outside code_identity, which remains fd25e2f7…. The launcher refuses until that is committed.

**Still open.** Disclosure of the ACLED 2025 coverage gap in reporting; the truth crosswalk and approved release (scen-evaluate only); the expert table (optional).
