# PROGRESS: task 10-03 (confirmatory pooled-vs-persistence onset)

- **2026-10-03 planning.** Planning artifacts were completed after three AskUserQuestion rounds: prd.md (D1–D12, R1–R10, AC1–AC6), design.md, implement.md, implement/check.jsonl. No `trellis-audit` registration (user).
- **2026-10-03 approval.** The user set `/goal 按照预先设定的spec边界和步骤严格执行。`, taken as approval of the final planning summary. `task.py start` was run.
- **Run directory and driver.** Run dir `C:\Users\swl00\geoxgb_runs\confirm-onset-v1`; driver `research/run_confirm.py`. Steps are order-enforced and refuse overwrites; product code is not modified.
- **Step `pin`** (inputs_manifest f91d7f82…): git HEAD f10b327; code identity fd25e2f7… (package clean); runtime equal to the 10-02 stack. The raw 2025–2026 file was hashed as bytes only.
- **Step `alignment`** (alignment_v3 34247cbc…): 19 ACLED sources excluded; 110 features (10-02: 129).
- **Step `extension`** (CSV 6b333599…, manifest v3 ea0eb37c…):
  - 62,898 rows = 5,718 × 11 months (2024-12 pinned + 2025-01..10 combined).
  - Admin 2996 at 2025-10 deduplicated, first occurrence kept. The two rows differ only in the excluded Tair/Rainf z-scores.
  - Loader path verified: scaffold through 2025-10; 35 covariate columns (no ACLED); static at 2025-10, monthly at 2025-09; 0 missing monthly cells.
- **Step `inputs`.**
  - S3: latest persistence 2024-10, SD 2024-06; gate_k 3; 22 countries; availability digest e46cb391….
  - S0: latest 2025-10 (SD 2025-10); gate_k 0; availability digest 28f91168….
  - 5,716 areas have persistence in each scenario.
- **Step `predict`, attempt 1 FAILED** (log `confirm-onset-v1.predict.attempt1_selfcheck_failed.log`). It failed in the driver's synthetic self-check, before any fit or write: `pandas groupby.agg` cannot return arrays. No predict artifacts or step record were written; S3/S0 still held only their input files.
  - **Fixes (driver only, before any fit):**
    1. The transition frequency tables now use `groupby().size().unstack()`.
    2. The self-check synthetic was too sparse for the specified ≥ 30-row rule. It now uses n = 4,000, explicitly tests both fallback levels, and sizes the logistic synthetic to n.
  - The self-check now passes.
  - **Disclosure:** the driver hash differs from the one recorded at `pin`. `predictions_frozen.json` records the driver hash used for predict. The fixes do not change the pre-registered method.
- **Step `predict`, attempt 2** launched at about 2026-10-03 23:2x, PID 3199229, via `research/predict_launch.sh`. Exact-PID watcher armed.
- **Step `predict`, attempt 2 COMPLETE.**
  - S3: fit rows 62,189; routes include 837 local rows.
  - S0: 1,609 local rows.
  - Logistic converged in 325 iterations in each scenario.
  - Transition baseline levels in S3: country 582 / pooled 3,282 / overall 1,854.
  - `predictions_frozen.json` f1fd1fdc… was written before any 2026-02 value was read.
- **Step `truth`** (release 67b2f608…, truth 75025655…, crosswalk 822ba26b…): 4,343 admitted; 1,223 unmatched; 278 outside the panel; 21 with no phase; 1 ambiguous 2995/2996; 1 DBF mismatch. Classes 1,069 / 1,337 / 1,660 / 277.
- **Step `evaluate`** (evaluation a321c973…). **PRIMARY S3 H1 INCONCLUSIVE.**
  - Pooled .559 vs persistence .376 on 2,910 onset keys (622 onsets, 18 countries): Δ +.183 [−.027, +.400].
  - H2 not tested (fixed sequence).
  - Secondary cells are reported in research/results.md. The S0 onset comparisons with persistence and transition are mechanical.
- **Step `check`** (steps/check.json 1891d18f…): frozen hashes match; primary recount matches; 0 unmatched truth rows; problems [].
  - Sanity check: in S3 the partitioned system equals pooled because the locals changed only 4 crisis decisions, with net-zero counts.
- **Report:** research/results.md. All implement.md steps 1–9 are done within the spec. No commit, no audit registration, no tuning, no product edit.
- Implementation note: the Feb-2026 truth builder was implemented inside the driver's `truth` step (`research/run_confirm.py::step_truth`), with the same 10-02 rule, rather than as a separate script in research/probes/ as implement.md step 6 wording suggests. This is a location difference only; the rule is unchanged.
