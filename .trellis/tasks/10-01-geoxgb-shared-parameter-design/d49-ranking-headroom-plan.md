# D49 / A23: zero-fit ranking-headroom diagnostic (approved)

2026-10-02. Supervisor selection under the user's delegated Stage 1 research authority, after the D26–D48 finite checkpoint. Approved as written after supervisor review, with three implementation clarifications (zero-denominator F1, predict-none tie winner, direct argmax recomputation); the planning commit precedes any code, and real scoring waits for supervisor release. Same active task, executor, audit run and base.
- **Endpoint:** the primary crisis-F1 endpoint (four-class argmax → code ≥ 2) and the final criterion are unchanged.
- **Out of scope:** model fits, calibrators, any new decision policy, maps, Stage 2/3, full 648, final evaluation, audit close and default adoption.

## 1. Question and what it is not

D39 recorded that the persistence operating point was not tested against the root's curve. D49 asks, per exposed root: within the deterministic scalar-cutoff family `predict crisis iff s ≥ c` on the frozen saved scores, is there a cutoff whose E3 crisis F1 exceeds persistence's? Does any cutoff weakly dominate persistence's confusion? And where does the root stand at persistence's own call budget?

**Uses of E3 truth:** the cutoff optimum is chosen with E3 target outcomes. It is a hindsight envelope within the scalar-cutoff family only.
- **Not** an upper bound on four-class, feature-based, randomised or row-specific policies.
- **Not** a deployable rule, validation or adoption evidence. It separates threshold-family headroom on these folds from D40's learned-threshold non-transferability, but neither result shows root or partition viability in general.

**Reading a result:**
- **Failure** (no cutoff beats persistence on a root) means only that this frozen checkpoint's deterministic scalar-cutoff family cannot beat that baseline on that exposed fold. Other policies, checkpoints and periods remain untested.
- **Success** is hindsight headroom on that fold. It is not a transferable gain and does not mean overfitting is solved.

**Fixed call budget:** k_p = the number of persistence crisis calls on the same keys. It is fixed without target-outcome values, because persistence comes from the origin phase. The comparison at k_p is still evaluated on a repeatedly exposed cohort, and the design was chosen after earlier diagnostics. It is not free of selection effects and is not independent validation.

**Argmax and persistence information:**
- The four-class argmax is generally not a threshold on s (D39: 1,367 argmax crisis calls with mass below .5, 71 the reverse). The family optimum therefore need not be at least the argmax F1 on a root.
- Both arms already use persistence information: the original root holds `hist_phase_o00` as a feature, and the D38 anchored arm adds explicit persistence margins. Results for the anchored arm are not independent ranking evidence, so the original arm is reported first.

## 2. Frozen design

- **Inputs:** the 21 saved D38 `rows_E3.csv.gz` files in `C:\Users\swl00\geoxgb_runs\geoxgb-d38-persistence-margin-root-20261002\<root>\` (producer 2d4fe4e). Each file's sha256 must equal the `input_hashes` in both `research/d39_probability_diagnostic.json` and `research/d40_summary.json`. Read with `float_precision="round_trip"`. Assert unique `(area, target_month, horizon)` keys and `target_month ≤ 2020-12`. No snapshots or other sources are needed.
- **Keys:** exact-origin-known rows only (`persistence_code` finite), 713 missing-origin rows excluded in total. Report the excluded count per root.
- **Arms:** `original`, then `anchored` (D38). The native four-class probabilities are kept as saved.
- **Score (diagnostic definition, frozen):** s = (p2 + p3) / (p0 + p1 + p2 + p3) in float64, from the saved columns `p_<arm>_3` and `p_<arm>_4或5` over all four.
  - This differs from D40, which thresholded the raw mass p2 + p3. Row sums differ, so the two orders need not agree. No equivalence is claimed and no inversion study is done.
- **Same-key comparators:** saved argmax (`y_<arm> ≥ 2`) and persistence (`persistence_code ≥ 2`), each with TP/FP/FN/TN and exact F1. The argmax is also recomputed directly from the four saved probabilities and must equal `y_<arm>` on every row.
- **No other score families or threshold policies.**

### Per root and arm

1. **Frontier:** sort s descending and group into unique-score blocks. The endpoint after each block gives cumulative calls, TP, FP and FN. Prepend the predict-none endpoint. The last block end is predict-all.
   - These endpoints enumerate the entire deterministic `s ≥ c` family exactly; they do not enumerate randomised or row-specific policies.
   - F1 is an exact `Fraction` `2TP / (2TP + FP + FN)`, defined as 0 when the denominator is 0 (synthetic no-positive cases). Predict-none has F1 0.
   - Endpoint JSON: `{"kind": "none" | "cutoff", "cutoff": null | float, "cutoff_hex": null | str, "calls", "tp", "fp", "fn"}`. No `Infinity` in any JSON.
2. **Optimum:** max F1 (exact and float), the number of optimal endpoints, and the winner with the highest cutoff (fewest calls), plus its confusion. Predict-none ranks above every finite cutoff, so it wins a tie that includes it; it is reported as `{"kind": "none", "cutoff": null}`, never as a non-finite value.
3. **Weak dominance of persistence:** whether there is an endpoint with TP ≥ TP_p and FP ≤ FP_p and at least one strict inequality. Report a boolean, the count of dominating endpoints, and one deterministic witness: the dominating endpoint with the highest cutoff.
4. **Fixed persistence call budget k_p:** if some endpoint has exactly k_p calls, report that point (status `exact`). Otherwise report the adjacent whole-block endpoints just below and just above k_p, with their actual calls and confusion (status `bracket`). No tie block is split and no random tie-break or expected TP is used.
5. **Kept externally:** all compact frontier counts and cutoff values. The summary stays small.

### Aggregation

- **Per root, then per H as mean-fold only** (7 folds per H):
  - original argmax, anchored argmax and persistence F1;
  - each arm's hindsight max F1;
  - the paired gaps (optimum − persistence, optimum − argmax, argmax − persistence);
  - fold counts of optimum > persistence and of dominance;
  - budget-point results with their exact/bracket status.
- **No pooled heterogeneous ranking and no pooled oracle headline.** Secondary AUC/AP are not recomputed (already in D39 and D48).
- **No significance tests:** the folds overlap and are repeatedly exposed.

## 3. Checks (fail closed)

1. Input hashes match both D39 and D40 records before any computation; 21 roots; the keys are unique; ≤ 2020-12; excluded counts total 713.
2. **Inline self-checks before reading real data**, on tied synthetic cases:
   - the frontier optimum equals a brute-force `s ≥ t` search over every unique value plus predict-none;
   - the dominance boolean, count and witness equal brute force;
   - the budget point (exact and bracket cases) equals brute force;
   - the predict-none, predict-all and tied-optimum cases are correct.
3. The argmax recomputed from the four probabilities equals the saved `y_<arm>` on every row, and the argmax and persistence confusions on the matched keys equal D39's `per_root[<root>]["matched"][<arm or persistence>]["argmax_crisis"]` counts.
4. Fresh external output directory; stop at the first failure; no rerun.

## 4. Implementation and order

- **Script:** one small task-research script, `research/d49_ranking_headroom.py`. It reuses the D40 hash and CSV reading pattern and installed numpy/pandas (sklearn only if useful). Its exact bytes are committed before execution, and the script git blob must equal HEAD at run time. It has no fit API calls.
- **Outputs (external):** under `C:\Users\swl00\geoxgb_runs\geoxgb-d49-ranking-headroom-20261002\`: the frontier file, a compact summary and an identity JSON (script blob and sha, input hashes, runtime).
- **Order:** supervisor review → align pointers and commit the planning (GitNexus attempt) → native implement and native check (each ≤ 10 min) → producer commit → supervisor release → single zero-fit run → factual report → supervisor independent check and synthesis → stop after D49.
