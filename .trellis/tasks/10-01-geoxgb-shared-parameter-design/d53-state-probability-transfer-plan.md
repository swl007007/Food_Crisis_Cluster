# D53 / A27: zero-fit exact-origin-state probability-level transfer (completed; supervisor verification passed; no policy adopted)

2026-10-02. A separately bounded diagnostic under the user's delegated Stage 1 research authority; not an automatic continuation of D52's fits. Approved by the supervisor with four clarifications (single-class cells, numerical identity tolerance, cell inventory and contrast scope, input-record and output-file specification); the planning commit precedes any code, and real rows are scored only after a separate supervisor release. Same active task, executor, audit run and base.
- **Unchanged:** the primary crisis-F1 endpoint, the four-class main contract and the final criterion.
- **Out of scope:** models, snapshots, fitting, calibration, grids, thresholds, new transition references, deployable corrections, Stage 2/3, final evaluation, close.

## 1. Question and what it is not

D38 already scored a FIT-only add-one 4×4 transition prior (not repeated). D39 described calibration on E3 with binary origin strata only, and D50 measured within-exact-state ranking only. D53 asks, descriptively: within each exact origin state, how do the observed crisis rate and the mean predicted score move from FIT to C to E3, for both the original four-class score and the D52 binary score?

**Limits:**
- This is descriptive probability-level / transition-rate transfer. Role composition differs (areas, label months, eras, calendar regimes), so it cannot isolate causal time drift or pure overfitting.
- The Brier residual after bias is not called "refinement", and AUC is not treated as a Brier resolution decomposition.
- Rare-state support (phase 1 onset positives, phase 4-or-5 negatives) stays visible.
- No significance tests or causal inference.

## 2. Frozen inputs

- **Exactly the 63 saved D52 row files**, `rows_{FIT,C,E3}.csv.gz` for the 21 roots under `C:\Users\swl00\geoxgb_runs\geoxgb-d52-binary-root-20261002\`. Before any scoring: (1) the external `completion.json` must be byte-identical (bytes and sha256) to the committed `research/d52_completion.json`; (2) the 63 exact expected row paths (21 roots × FIT/C/E3) and each file's sha256 must equal the entries in that record.
- **Columns used:** `area`, `target_month`, `horizon`, `truth_crisis`, `persistence_code`, `s_original`, `p_binary`. No call shares.
- **Origin state:** exact `persistence_code` 0 / 1 / 2 / 3 (IPC 1 / 2 / 3 / 4-or-5); missing-origin rows counted separately as `missing`, never assigned a state.
- No models, snapshots or other sources.

## 3. Outputs (descriptive)

1. **`role_cells.csv`:** root × role (FIT / C / E3) × all 5 groups (states 0–3 plus `missing`) = 315 cells, all retained: n, positives, n_unique_areas, n_unique_label_months, min/max label month, observed crisis rate, and for each score (`s_original`, `p_binary`) mean score, bias = mean score − rate, and Brier.
   - **Only empty cells (n = 0) have null metrics** (reason `empty`). Non-empty single-class cells keep valid rate, mean, bias and Brier; `no_positive` / `no_negative` are support flags, not null reasons.
2. **`month_cells.csv`:** for each root × role, its observed label-month inventory crossed with all 5 groups, retaining empty state-month cells; the same counts, rate, mean scores, bias and Brier. No pooling across overlapping roots.
3. **`contrasts.csv`:** for exact known states 0–3 only, per root: E3 − FIT and E3 − C for rate, mean score and bias, for both scores — 21 roots × 4 states × 2 contrasts = 168 paired contrast records, each carrying both model scores. `delta_bias = delta_mean − delta_rate` must hold numerically within absolute 1e-12. A record is null when either role-state cell is empty. `missing` stays descriptive and is never a fifth IPC state.
4. **`summary.json`:** per H and state 0–3, unweighted means over valid roots with `n_valid / 7` and sign counts (positive / negative / zero) taken from the recorded raw deltas, with no rounding before the sign decision. No pooled cross-root table, significance or causal claim.
5. **`identity.json`** (written before scoring: script blob and sha256, D52 record and input hashes, runtime) and **`completion.json`** (output hashes).

## 4. Checks (fail closed)

1. Inventory and sha256 of the 63 files equal the D52 records; 21 roots, 7 per H; keys unique per file; `target_month ≤ 2020-12`; `persistence_code` ∈ {0, 1, 2, 3} or missing; `truth_crisis` ∈ {0, 1}; scores finite in [0, 1].
2. Per root and role, the state cells plus `missing` add back to the file's row count; per-month cells add back to the state cell.
3. Inline synthetic checks: an empty cell gives null metrics with reason `empty`; a non-empty single-class cell keeps valid rate, mean, bias and Brier with the right support flag; support counts (n, positives, unique areas and months); a weighted-month recomposition case where two months with different rates and sizes mix, confirming the cell equals the size-weighted month values and the month table exposes the mix; the contrast identity within 1e-12 and sign counts from raw deltas.
4. Fresh output directory `C:\Users\swl00\geoxgb_runs\geoxgb-d53-state-probability-transfer-20261002`, no overwrite; stop at the first failure; JSON with `allow_nan=False`.

## 5. Implementation and order

- **Script:** one minimal standalone runner, `research/d53_state_probability_transfer.py`, numpy/pandas/stdlib only, no package imports or new dependencies. Exact bytes committed before the run; script git blob equal to HEAD.
- **Order:** supervisor review → planning commit (GitNexus attempt) → native implement and native check (each ≤ 10 min) → producer commit → supervisor release → single zero-fit run → factual report → independent supervisor verification → stop. No automatic D54.

## 6. Result (2026-10-02)

- **Producer and run:** producer `192f0ce`; run `C:\Users\swl00\geoxgb_runs\geoxgb-d53-state-probability-transfer-20261002`, exit 0 in 6 s, zero fits; 315 role cells, 3,535 month cells, 168 contrasts. Executor factual record `d0015fd`.
- **Supervisor verification:** PASS (`research/d53_supervisor_check.py` / `research/d53_supervisor_results.json`; 66,451 checks over 1,660,244 rows).
- **Findings:** [research/d53-state-probability-transfer-findings.md](research/d53-state-probability-transfer-findings.md).
- **Synthesis:** diagnostic complete; no policy or model adopted. Level shifts differ by H and state code: H12 code 2 (IPC 3) rate +.1919 with original mean +.1256 (bias −.0663); H12 code 1 (IPC 2) underprediction grows; H8 code 2 (IPC 3) slight E3 overprediction; H4 code 2 (IPC 3) tracks fairly closely. No uniform level-collapse diagnosis or single offset fix; a near-correct group mean does not establish individual calibration, ranking or F1; composition changes prevent causal drift or pure-overfitting claims.
- **Stage 1** remains unresolved; next policy choice not made; no automatic D54.
