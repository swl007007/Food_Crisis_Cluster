# D50 / A24: zero-fit exact-origin-phase ranking diagnostic (completed; no policy adopted; supervisor verification passed)

2026-10-02. Supervisor selection under the user's delegated Stage 1 research authority, after D49. Approved after supervisor review with two edits (sklearn-only producer AUC; per-H date and per-root count gates); the planning commit precedes any code, and real scoring waits for a separate supervisor release. Same active task, executor, audit run and base.
- **Endpoint:** the primary crisis-F1 endpoint (four-class argmax → code ≥ 2) and the final criterion are unchanged.
- **Out of scope:** model fits, calibrators, thresholds or decision policies, maps, Stage 2/3, full 648, final evaluation, audit close, default adoption and any automatic D51.

## 1. Question and what it is not

D39 reported within-stratum ranking only for the binary origin split (crisis vs non-crisis). Within the non-crisis stratum, phase 1 and phase 2 origins differ strongly in onset rate, so binary-stratum AUC can partly reflect exact-phase information the root already holds as `hist_phase_o00`. D50 asks: on the exposed folds, do the frozen root scores order E3 crisis outcomes **within each exact origin phase**?

**Reading a result:**
- Within-cell AUC above .5 is descriptive ranking beyond constant exact-phase information only. It can reflect other history, country or era features. It is not deployability, validation, stable forward transfer or an overfitting resolution.
- Low within-cell AUC alone does not show that global AUC or the D49 headroom is mostly persistence. That would need a separate decomposition; no causal attribution is made.
- There is no numeric success threshold and no adoption. The result informs only the next hypothesis: whether the frozen scores contain within-state ordering, or whether model inputs and objective need reconsidering.
- Within an origin-phase cohort persistence predicts a constant class. Its crisis F1 is defined; cell F1 is omitted as redundant to the reported counts.

## 2. Frozen design

- **Inputs:** the same 21 saved D38 `rows_E3.csv.gz` files (`C:\Users\swl00\geoxgb_runs\geoxgb-d38-persistence-margin-root-20261002\<root>\`). Each file's sha256 must equal the `input_hashes` in `research/d39_probability_diagnostic.json`, `research/d40_summary.json` and `research/d49_identity.json`. Read with `float_precision="round_trip"`. No snapshots or other sources.
- **Validation:** unique `(area, target_month, horizon)` keys; `target_month ≤ 2020-12`; on known rows `persistence_code` is an integer in 0..3 and `origin_phase == persistence_code + 1`. The 21 roots contain exactly 7 distinct target dates for each of H4/H8/H12. Per root, the matched n and excluded count equal D49 (`research/d49_summary.json` `per_root[<root>]["n"]` and `["excluded_missing_origin"]`); totals 112,795 known and 713 excluded.
- **Arms:** `original`, then `anchored` (D38). Score s = (p2 + p3)/(p0 + p1 + p2 + p3) in float64, identical to D49.
- **Cells:** each root is split by exact `persistence_code` 0 / 1 / 2 / 3 (origin phase 1 / 2 / 3 / 4-or-5). All four cells are reported for every root, including empty and one-class cells. No outcome-dependent removal and no minimum-sample cutoff.

### Per root, cell and arm

1. **Counts:** n, positives P (truth code ≥ 2), negatives N.
2. **Crisis AUC on s, with half credit for ties:** computed with `sklearn.metrics.roc_auc_score` only. When P·N = 0, AUC is `null` with an explicit reason (`empty`, `no_positive` or `no_negative`). The exact pair-count numerator belongs to the independent supervisor verifier, not the producer.
3. **Fixed argmax confusion** (TP/FP/FN/TN) for original, anchored and persistence. Before cell counting, the argmax is recomputed from the four saved probabilities and must equal the saved `y_<arm>` on every row, as in D49. The four cells must add up exactly to D49's per-root matched confusions (`research/d49_summary.json`: `per_root[<root>]["arms"][<arm>]["argmax"]` and `per_root[<root>]["persistence"]`), for every root.

### Aggregation

- **Per date:** all cells for all 21 roots.
- **Per H:** a mean-fold AUC for each exact phase and arm, over the valid folds only. Report `n_valid / 7` and the per-fold P/N supports. Validity depends only on P and N, so it is the same for both arms.
- **Not reported:** no pooled AUC, AP, threshold frontier or country slicing. Models, dates and cells are not selected on results.
- Phases 2 and 3 are scientifically central, but phases 1 and 4-or-5 are always reported with their actual P/N.
- The D39 binary-origin finding remains a reference and is not recomputed.

## 3. Checks (fail closed)

1. The input hashes equal all three records before any computation; 21 roots with 7 distinct dates per H; the key, month, code and phase-relation checks pass; per-root n and excluded equal D49; 112,795 known and 713 excluded; recomputed argmax equals saved labels.
2. **Inline synthetic checks before reading real data:** expected AUC on tied scores (including all-tied = .5); `empty`, `no_positive` and `no_negative` null reasons; one small brute-force pair-count check of the AUC; JSON dumps with `allow_nan=False`.
3. The cell confusions add up to D49's matched per-root confusions.
4. Fresh external output directory; stop at the first failure; no rerun.

## 4. Implementation and order

- **Script:** one standalone task-research script, `research/d50_origin_phase_ranking.py`. It uses installed numpy/pandas and `sklearn.metrics.roc_auc_score`, with no framework or package changes. Its exact bytes are committed before execution, and the script git blob must equal HEAD at run time. No fit or model APIs.
- **Outputs (external):** under `C:\Users\swl00\geoxgb_runs\geoxgb-d50-origin-phase-ranking-20261002\`: a compact `summary.json` (per root, per cell and per H) and `identity.json` (script blob and sha, input hashes, runtime).
- **Order:** supervisor review → align pointers and commit the planning (GitNexus attempt) → native implement and native check (each ≤ 10 min) → producer commit → supervisor release → single zero-fit run (after a separate supervisor release) → factual report → independent supervisor pair-count verification → evidence and synthesis → stop. No D51 automatically.

## 5. Result (2026-10-02)

- **Producer and run:** producer `386b25a` (blob `5611ced8…`; selftest OK). Run `C:\Users\swl00\geoxgb_runs\geoxgb-d50-origin-phase-ranking-20261002`: exit 0, assertions on, zero fits, 112,795 known / 713 excluded; phase 4-or-5 null (`no_negative`) on 3/7 dates per H. Executor factual record `7c398c3`.
- **Supervisor verification:** PASS (`research/d50_supervisor_verify.py` / `research/d50_supervisor_verification.json`; 1,016 checks, 35,537,568 direct pair comparisons, exact integer numerators).
- **Findings:** [research/d50-origin-phase-ranking-findings.md](research/d50-origin-phase-ranking-findings.md).
- **Decision:** no policy or model adopted. Original within-phase-2 mean-fold AUC .655702 / .598012 / .605804 (> .5 on 7/7, 4/7, 7/7 dates); phase 3 .736468 / .745844 / .728402 (7/7 at every H). This refutes "no within-state ordering anywhere" but identifies no causal feature contribution, transfer, date routing or overfitting solution; H8 onset ranking is unstable, and a monotone calibration of the same scores cannot repair a rank instability. Small phase 1 and phase 4-or-5 supports are disclosed, not claimed robust.
- **Stage 1** remains unresolved; no D51.
