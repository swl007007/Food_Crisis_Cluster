# D49: zero-fit ranking-headroom diagnostic (2026-10-02)

**Scope:** contract `d49-ranking-headroom-plan.md`, planning commit `b284440`. The primary crisis-F1 endpoint and the final criterion are unchanged.
- **Inputs:** the 21 saved D38 `rows_E3.csv.gz` files, hash-bound to both the D39 and D40 records; exact-origin-known keys only (713 missing-origin rows excluded).
- **Arms:** the original root, then the D38 anchored root. Score s = (p2 + p3)/Σp in float64, the frozen diagnostic definition (D40 thresholded the raw mass p2 + p3).
- **Per root and arm:** the full deterministic `s ≥ c` family (tie-block endpoints plus predict-none/all), the hindsight max F1 chosen with E3 truth, weak dominance of persistence's confusion, and the cutoff at persistence's own call count k_p.
- **No fits,** calibrators, new decision policies, maps or adoption.

## Producer and run (executor factual record, `4bc18fb`)

- **Producer** `879c335`: task-research script `research/d49_ranking_headroom.py` (269 lines), git blob `67a9867e288d4ffd8e3ec2b5a9b6b43a76272a70`, sha256 `0a9bd888a97440c324b526811c91f01753bd03d567e49f1ca84cfa1731e8e5d6`. No package edits, no fit or model APIs.
- **Native implement** (under 2 min) and **native check** (about 1 min, no result-affecting finding, no edits). Brute-force selftest (300 tied random cases plus fixed edge cases) OK at HEAD.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d49-ranking-headroom-20261002`, frozen Windows Python, exit 0 in 4 s. The recomputed argmax matched the saved labels on every row, and the argmax and persistence confusions matched D39. 7 folds per H; all 42 budget points landed exactly on k_p (no brackets); every root has a single optimal cutoff and predict-none never wins.

## Verification (supervisor, distinct from the executor record)

`research/d49_supervisor_verify.py` → `research/d49_supervisor_verification.json`: **PASS**.
- 950 checks; 113,508 source rows, 112,795 matched, 713 excluded; 225,241 frontier endpoints.
- No producer imports and no fits. The method is different from the producer's: label-specific sorted scores with `searchsorted` per unique cutoff, rather than cumulative tie blocks.
- Every root-arm optimum, tie count, dominance result, budget point and per-H mean recomputed; input, producer and output hashes checked; exactly seven folds per H, per-root exclusions and budget denominators confirmed.

These are numerical checks, not statistical tests.

## Results (E3, origin-known keys, mean-fold; fold counts out of 7)

| H | Persistence | Original: argmax / hindsight max | Anchored: argmax / hindsight max |
|---|---|---|---|
| 4 | .655255 | .626608 / .667136 | .633752 / .672049 |
| 8 | .555568 | .534031 / .587542 | .528513 / .599142 |
| 12 | .547711 | .479043 / .565459 | .515010 / .581233 |

| H | Hindsight max > persistence (original, anchored) | Weakly dominates persistence | TP at k_p ≥ persistence TP | Mean TP at k_p − persistence TP |
|---|---|---|---|---|
| 4 | 6, 5 | 3, 4 | 3, 4 | −19.429, −10.000 |
| 8 | 5, 5 | 3, 3 | 3, 3 | −1.571, +7.429 |
| 12 | 3, 4 | 1, 1 | 1, 1 | −21.000, −8.571 |

- **Mean-fold hindsight max − persistence:** original +.011881 / +.031974 / +.017748; anchored +.016795 / +.043575 / +.033522 (H4 / H8 / H12).
- **Mean-fold argmax − persistence:** original −.028646 / −.021537 / −.068668; anchored −.021503 / −.027055 / −.032700.
- **Argmax outside the family:** one H4 original root has its hindsight max below its argmax F1.

## Aggregation: mean-fold here vs pooled earlier

D49 reports mean-fold values, as the plan requires. Persistence F1 here (.655255 / .555568 / .547711) differs from the pooled figures in D46–D48 (.651697 / .555614 / .549808) **by aggregation only**: D49's per-root confusions, pooled, reproduce .651696974 / .555613969 / .549808135. No pooled oracle headline is published, because each root's optimum uses its own cutoff.

## Supervisor synthesis

- **No policy is adopted and no overfitting resolution is claimed.**
- With E3 hindsight, the original root's best scalar cutoff beats persistence on 6/7, 5/7 and 3/7 folds (H4/H8/H12), with mean-fold gaps +.011881 / +.031974 / +.017748. Weak dominance holds on only 3/7, 3/7 and 1/7.
- At persistence's own call count the original root's mean TP gaps are −19.429 / −1.571 / −21. Hindsight headroom therefore does not establish uniformly better ranking than persistence.
- The anchored arm's H8 mean of +7.429 TP comes with only 3/7 folds at or above persistence; it is not robust.
- Both arms already use persistence information (the original holds `hist_phase_o00` as a feature; the anchored arm adds explicit persistence margins).
- **Finite reading:** these are hindsight envelopes within this frozen checkpoint's deterministic scalar-cutoff family on exposed, overlapping folds. They are not upper bounds on four-class, feature-based, randomised or row-specific policies, not transferable gains and not validation.
- **D40 relation:** D40's historical-threshold result is non-transferability evidence for that finite test. It is not a direct test of this normalised score or of every policy. D40's "All 21 roots remain below persistence" refers to the all-21 aggregate, not to each root.
- Stage 1 remains unresolved; the task and audit stay active; no D50. A possible next question, conceptual only and not specified: how temporally stable is the root's incremental information over origin-state persistence?

## Evidence

**Task research holds exact byte copies** (line endings preserved):

| File | Bytes | sha256 |
|---|---|---|
| `d49_summary.json` | 59,972 | `f10bec781c14e08f1c50e9734efb0ca76b8da189f1a36bdd03de03d062f4b5ef` |
| `d49_identity.json` | 2,791 | `407c2ff8fbc4bd885cc37feac18e34fa4802e004d5d9c4e7a54e7795d4f37363` |
| `d49_supervisor_verification.json` | 4,529 | `37afaa500bafd58c853df16d1ea4a8c3822e9b5feead2a8c6bb3bf46f0d8834b` |
| `d49_supervisor_verify.py` | 8,194 | `a7205484fe92d5c3ee10d17bc3fcfe27ef9a614ad08e0a92e0fcdaa2881878ff` |
| `d49-run.log` | 88 | `3b8fded3fc8350ae3c96c3e4bb9ab6eeb058094802349d37aa4c61d925f2794a` |

**Kept external:** `C:\Users\swl00\geoxgb_runs\geoxgb-d49-ranking-headroom-20261002\frontier.csv.gz` (6,304,835 bytes, sha256 `db0035848e2602c6b38b7b923f8521ac4950dc333486be1ada7340e6db07ae6e`).
