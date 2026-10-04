# D48: known-origin FIT restriction contrast (2026-10-02)

**Scope:** contract `d48-known-origin-fit-plan.md`, planning commit `140da65`. The crisis-F1 endpoint is primary and unchanged.
- **Change:** each of the 21 D34 roots is re-fitted only on the ORIGINAL FIT rows whose exact-origin `hist_phase_o00` is finite. No target-outcome values are used to choose rows; selection uses availability of the already-known origin-label feature within the original labelled FIT pool.
- **Unchanged:** G (H4 G1 / H8 G4 / H12 G2), rounds, seed, the 162 features, the four-class objective; no weights or margins; no S/C/E3 rows in fitting.
- **Arms:** the saved original root, the fresh known-origin root, and persistence, all on exactly the origin-known keys of each part. Missing-origin E3 rows are counted only.
- **What it is not:** an identifiable missingness test. Sample size, calendar regime, era, composition, missingness selection and optimisation path change together.

## Producer, tests and run (executor factual record, `9a1aa11`)

- **Producer** `2326483`: task-research script `research/d48_known_origin_fit.py` (474 lines), git blob `86264d865dc37432a4afff50927d0392700b738f`, sha256 `120a885b48aec683a6da944b9db4381a8f447910e1c38bbfc0cc000da7a7b753`. It imports the D37 gate/rebuild/persistence helpers, the D46 pure scoring helpers and D47's arm-parameterised aggregation helpers. No package, src or default edits.
- **Native implement** (about 3 min) and **native check** (about 4 min, no result-affecting finding, no edits).
- **Tests:** `--selftest` OK (a failed gate means no fit; the mask is unchanged under different `y` and row permutation; known-only scoring); package suite 116 OK, exit 0, at HEAD. Script blob and package code both equal HEAD before the run.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d48-known-origin-fit-20261002`, frozen Windows Python, exit 0 in 185 s. Log: `C:\Users\swl00\geoxgb_runs\d48-run.log`, sha256 `255bd003e8deb512d3b89fa3fece63ddd7bb764a05d53748d78c1374811d2d7f`.
  - **Fits:** exactly 21 known-origin fits, 839,486 FIT rows in total (per-root counts equal the plan's support table), all UBJ hashes different from the originals. All 21 original-root gates passed first.
  - **Checks:** rounds 200 (H4) / 400 (H8, H12), `base_score` `5E-1`, 4 classes, no weight or margin; 0 reload mismatches on FIT-known, C-known and E3-known; 0 S/C/E3 overlaps with the eligible FIT keys; `dev_baselines` check 15 checked / 6 not covered (same as D47).

## Verification (supervisor, distinct from the executor record)

The supervisor's independent check, `research/d48_supervisor_check.py` → `research/d48_supervisor_results.json`, passed: 3,319 checks, 216 metric cells and 1,081,803 origin-known FIT/C/E3 rows.
- No producer imports and no fits.
- Feature-availability masks rebuilt independently from the frozen snapshots and the original membership; eligible sorted-key hash, rows, label dates, areas and classes matched.
- G unchanged, no weights or margins; both raw UBJs replayed, with exact known-only probabilities and argmax.
- Persistence and coverage, per-pair and per-H metrics, crisis-call shares and AUC/AP matched.

These are numerical checks, not statistical tests.

## Results (E3, origin-known keys; FIT/C in `d48_summary.json`)

Excluded missing-origin E3 rows: 142 / 284 / 287 of 37,836 per H (coverage .99625 / .99249 / .99241). No all-population claim is made.

| H | Pooled crisis F1: original / known / persistence | Fold wins, known > original | Crisis-call share, original → known |
|---|---|---|---|
| 4 | .629050 / .620930 / .651697 | 2/7 | .149865 → .160450 |
| 8 | .533375 / .519291 / .555614 | 3/7 | .156716 → .146437 |
| 12 | .478355 / .464853 / .549808 | 3/7 | .086713 → .085488 |

- **Probability quality, original → known:** crisis Brier H4 .090807 → .092096, H8 .110801 → .114930, H12 .104087 → .107515; log loss H4 .615151 → .617065, H8 .715178 → .736135, H12 .733590 → .737150. Worse at every H.
- **Mean-fold AUC / AP (secondary), original → known:** H4 .906903/.732831 → .903796/.724391; H8 .865053/.650255 → .867211/.632435; H12 .858335/.644472 → .852914/.627790. AP is lower at every H; AUC is lower at H4/H12 and slightly higher at H8 only.
  - These original-root AUC/AP values differ from the D46/D47 all-key values because the cohort here is origin-known keys only.
- **FIT-known and C-known:** crisis F1, Brier and log loss all improve at every H (pooled F1 FIT H4/H8/H12 .7258/.7658/.6093 → .7399/.8146/.6312; C .7156/.7277/.5891 → .7239/.7577/.6098).
- **Weak H8 time support (heterogeneous):** H8 2018-06 (4 label dates) E3 F1 .424116 → .495510, still below persistence .520685; H8 2018-10 (5 label dates) .501471 → .491549, against persistence .588729.

## Supervisor synthesis and decision

**D48 is complete, and the known-origin FIT restriction is not adopted.** No further era/availability pool sequence follows.
- On origin-known E3 keys the known-origin root is below both the original and persistence at every horizon.
- FIT-known and C-known F1 and both losses improve at every H, yet E3 worsens at every H. Forward generalisation is not solved.
- Composition and reduced support are confounded with the restriction. There is no causal proof that missing-origin rows help.
- The weak H8 roots are heterogeneous (one improves, one worsens, both below persistence); no blanket weak-support attribution.
- Stage 1 remains unresolved; the task and audit stay active. No D49 starts automatically; a finite checkpoint comes first.

## Evidence

**Task research holds exact byte copies** (CRLF preserved):

| File | Bytes | sha256 |
|---|---|---|
| `d48_summary.json` | 490,836 | `feac01022daac3bcfa86da6217507bf992b8fc5f020eb69a2e0ca40440ef0ab6` |
| `d48_identity.json` | 42,224 | `aa0eb32fa8581f556a83f275a182c5fc82de0f8634cf897ded455f75b959c4b3` |
| `d48_gate.json` | 60,989 | `4ddce8d08fb7ddbfd4c6f646178ed5a1bb872a54d89f429772a69e946601ba6b` |
| `d48_supervisor_check.py` | 9,436 | `b7dc056f28578e9ee6a96fffc5abeaade9d189629ad25bdf251a8e56e4489aff` |
| `d48_supervisor_results.json` | 149,384 | `09a800bf0603b243040286867b52d0e3fca84814e0e3d98bc63195b3a52daf86` |

**Kept external** in the run directory: the known-origin UBJ files and records, the keyed origin-known FIT/C/E3 rows and `completion.json` (`8dd0ba2df6d164a622a3390a824423fe03ac93e9fdfdecae127ef6b964d101cb`).
