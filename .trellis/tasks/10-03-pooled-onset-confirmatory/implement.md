# Implement: confirmatory pooled-vs-persistence onset test (task 10-03)

Execution starts only after the final planning summary is approved and `task.py start` has run. No `trellis-audit` registration. Product code stays byte-identical (identity fd25e2f7…), and no 2026-02 CS value is read before step 6.

## Ordered checklist

1. [x] **Pin and preflight.** Done: inputs_manifest f91d7f82…; HEAD f10b327; code fd25e2f7…; runtime equal to 10-02.
   - Verify HEAD, a clean package tree, code identity fd25e2f7… and the pinned Windows Python 3.12.10 stack.
   - Hash every input listed in design.md into `confirm-onset-v1/inputs_manifest.json`, refusing to overwrite.
2. [x] **Alignment v3.** Done: 34247cbc…; 110 features. Copy the committed 10-02 alignment, set the 19 ACLED sources to `excluded`, record the feature count, and run `check_alignment(schema, v3, real=True)`.
3. [x] **Extension v3.** Done: CSV 6b333599…, manifest ea0eb37c…; 2996 deduplicated; loader verified.
   - Splice pinned 2024-12 with combined 2025-01..10 (keys + 69 sources, original strings).
   - Dedupe admin 2996 at 2025-10, keeping the first occurrence, after asserting the two rows are identical on all 51 admitted sources.
   - Write the manifest (splice disclosure, both raw hashes, months, overlap true by construction).
   - Verify through `load_extension` → Scaffold (through 2025-10) → `covariate_features` under alignment v3: static at 2025-10, monthly at 2025-09, annual reference rows, ACLED absent.
4. [x] **Scenario inputs.** Done: S3 persistence ≤ 2024-10 (SD 2024-06), gate_k 3; S0 2025-10 visible, gate_k 0.
   - S3 uses the prepared observations and ledger as-is.
   - S0 appends the Oct-2025 truth v2 rows as observations, plus ledger rows for the 2025-10 cycle (one per country present; release 2025-10-31; `reconstructed`; source = 10-02 truth release v2).
   - For each scenario, build Availability (development_truth = False for the 2026 target) and assert:
     - persistence source months and ages (S3: ≤ 2024-10, SD 2024-06; S0: 2025-10 where present);
     - gate_k (S3: 3 for 22 countries; S0: 0).
5. [x] **Fit and predict (no truth).** Done on attempt 2 (attempt 1 failed in the driver self-check before any fit; see PROGRESS); predictions_frozen f1fd1fdc…. For each scenario:
   - Run `scen_dev_fold` with the frozen map, strategy A, G1/L1 and the scenario gate_k. This saves the pooled-arm and system predictions, `gate.json`, `gate_pairs` and the global records.
   - Fit the transition baseline and the IPC-history logistic on the identical fitting-pool rows and predict all 5,718 areas.
   - Write all prediction files, then `predictions_frozen.json` (hashes of every prediction file, the code identity, the inputs manifest and this implement.md).
6. [x] **Truth release (after the freeze only).** Done: release 67b2f608…; 4,343 admitted.
   - Build the Feb-2026 crosswalk and truth with the 10-02 rule (adapted builder in `research/probes/`), into a new external release directory.
   - `release.json` binds `predictions_frozen.json` and records the source hashes (raw CSV, FEWSNET.csv, .dbf, .shp pin) and every exclusion count, including the six out-of-universe countries.
7. [x] **Evaluate.** Done: evaluation a321c973…; S3 H1 INCONCLUSIVE; H2 not tested.
   - Onset risk set from the Oct-2025 truth v2 (class < 2) ∩ the Feb-2026 truth.
   - Crisis F1 and the paired bootstraps for H1 → H2 (fixed sequence) and the partition comparison.
   - Outcome categories (CONFIRMED / INCONCLUSIVE / CONTRADICTED / NA).
   - Secondary: Study 1 (S3), all comparisons in S0, per-country descriptive.
   - Write `evaluation.json` and the keyed and country tables.
8. [x] **Independent checks (bounded).** Done: steps/check.json, problems [].
   - Re-verify the prediction hashes against `predictions_frozen.json`.
   - Recount the primary confusion counts and F1 from the keyed rows without package imports.
   - Confirm truth and origin-truth joins (no fill; exclusions counted).
9. [x] **Report.** Done: research/results.md. `research/results.md`: pre-registered outcome per hypothesis, all cells, precision/recall context, disclosures; artifact manifest with hashes.

## Validation commands (pinned interpreter)

- `PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe`, run from `FEWSNETGeoXGBExperiment/` with `-B`.
- Driver: `$PY -B ../.trellis/tasks/10-03-pooled-onset-confirmatory/research/run_confirm.py --run-dir 'C:\Users\swl00\geoxgb_runs\confirm-onset-v1' <step>`, with steps `pin | alignment | extension | inputs | predict | truth | evaluate | check`. Each step refuses to overwrite and refuses to run out of order (`truth` requires `predictions_frozen.json`).
- No broad test rerun. Product code is unchanged, so the 10-02 test evidence still applies to the reused modules. The driver's own new functions (transition baseline, logistic wrapper) get a small synthetic self-check inside the driver.

## Risky points and rollback

- **Blinding breach risk.** No step before `truth` may open the raw CSV's value/description columns. The `truth` step asserts that `predictions_frozen.json` exists and that the hashes match.
- The external run directory is the only write target. Rollback = delete it. The 10-02 run and the repository are unchanged.
- Any input or hash mismatch, missing artifact or unexpected exclusion pattern stops the run; nothing is silently filled.
