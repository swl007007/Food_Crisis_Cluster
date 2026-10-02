# D46: fixed 2:1 crisis-class weighting of the root (2026-10-02)

**Scope:** contract `d46-crisis-weight-root-plan.md`, planning commit `e086535`. The crisis-F1 endpoint is unchanged.
- **Data:** 21 D34 pairs (H4/H8/H12 × seven targets 2018-06 … 2020-06) with unchanged FIT/C/E3 keys, G (H4 G1 / H8 G4 / H12 G2), seed, rounds, the 162 features and the four-class objective.
- **Arms:** the saved original root; one fresh weighted root per pair, with weights from FIT labels only, `(1 + I[y ≥ 2])/mean` in float64 then float32; a zero-fit post-hoc ×2 control; persistence on the same keys.
- **Excluded:** no weight search or sequence, D38 combination, partition propagation, Stage 2/3, full 648, final evaluation or close.

## Producer, tests and run

- **Producer** `b8f8550c6455aac3c3bb2a0374ee4297a71d818f`: `scripts/stage1_class_weight_root.py`, reusing the D37 gate, rebuild, persistence and scoring helpers. No adapter or model-semantics edits.
- **Native implement:** about 6.1 min.
- **Native check:** about 5.9 min. At supervisor request it made two changes:
  - the post-hoc ranking check now fails closed. A map error or order inversion above 1e-12 stops the run; exact float ties and sub-tolerance rounding inversions are counted separately, never treated as learned changes;
  - the wording of the degenerate-FIT guard was corrected.
- **Tests:** 114 OK, exit 0, after the commit. `ClassWeightRoot` has 4 tests: a failed gate prevents fitting; weights depend only on FIT labels; post-hoc ratio and ranking preservation, including enforcement; scoring and the log-loss stop.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d46-class-weight-root-20261002`, frozen Windows Python, exit 0 in 280 s. Log: `C:\Users\swl00\geoxgb_runs\d46-run.log`, sha256 `bc069f865803b32498f44afb5a8b4463961c3456d7535df0f5892f3d4e235baf`.
  - **Fits:** exactly 21 weighted fits, all with UBJ hashes different from the originals.
  - **Gates:** all 21 original-root replay gates passed before fitting.
  - **Model checks:** `base_score` `5E-1`, and rounds 200 (H4) / 400 (H8, H12).
  - **Reload:** exact on FIT, C and E3.
  - **Weights (float32):** crisis 1.733–1.835, non-crisis 0.8665308–0.9172514 across the 21 roots.
  - **Post-hoc ranking:** preserved, with a largest map error of 2.2e-16 and no inversions.

## Verification

The supervisor's independent check, `research/d46_supervisor_check.py` → `research/d46_supervisor_results.json`, passed: 5,839 checks, 504 metric cells and 1,660,244 FIT/C/E3 rows.
- **Method:** native raw UBJ replay on independently keyed frozen snapshots, with no producer imports and no fits.
- **Matched exactly:**
  - the probabilities of all three arms, and the argmax labels;
  - the FIT labels and the weight bytes, hash, sum, min and max;
  - the original and weighted hashes, and the rounds, base score and class count;
  - the persistence cohorts, all per-pair scores, and the per-H pooled and mean-fold scores and AUC/AP.

These are numerical checks, not statistical tests.

## Results

**E3, persistence-matched cohort, crisis F1:**

| H | Original | Weighted | Post-hoc ×2 | Persistence |
|---|---:|---:|---:|---:|
| 4 | .629050 | .622356 | .614925 | .651697 |
| 8 | .533375 | .544705 | .541565 | .555614 |
| 12 | .478355 | .514900 | .521261 | .549808 |

**E3 per-fold F1 wins** (out of 7 targets per horizon; the same on all keys and matched keys):

| | H4 | H8 | H12 |
|---|---|---|---|
| Weighted > original | 2 | 5 | 5 |
| Weighted > post-hoc ×2 | 6 | 3 | 2 |

**E3 mean-fold crisis F1 differences:**

| | H4 | H8 | H12 |
|---|---|---|---|
| Weighted − original | −.005908 | +.013217 | +.035934 |
| Weighted − post-hoc ×2 | +.007600 | +.003009 | −.006948 |

**E3 pooled probability quality, all keys** (crisis Brier / four-class log loss): the weighted arm is worse than the original at every horizon.

| H | Original | Weighted |
|---|---|---|
| 4 | .090834 / .616084 | .097629 / .632723 |
| 8 | .110754 / .716542 | .118805 / .729384 |
| 12 | .104083 / .734396 | .107415 / .737774 |

**Within-root crisis ranking, mean-fold AUC / AP:** the weighted arm is lower than the original at every horizon. The post-hoc control equals the original by construction.

| H | Original | Weighted |
|---|---|---|
| 4 | .906427 / .732046 | .904833 / .727960 |
| 8 | .863654 / .648368 | .862872 / .647390 |
| 12 | .857193 / .642707 | .856946 / .638885 |

**FIT and C, pooled crisis F1, original → weighted:** rises at every horizon.

| H | FIT | C |
|---|---|---|
| 4 | .693452 → .730123 | .679029 → .712570 |
| 8 | .743506 → .773602 | .702297 → .732678 |
| 12 | .646005 → .701366 | .622801 → .681765 |

## Supervisor synthesis and decision

**D46 is complete, and fixed 2:1 weighting is not adopted.** No weight search or sequence follows.
- Weighted − original improves crisis F1 at H8 and H12 but harms it at H4, and every arm remains below persistence at every horizon.
- Within-root AUC/AP and unweighted E3 Brier/log loss are worse for the weighted arm at every horizon.
- Weighted − post-hoc ×2 captures the whole effect of re-fitting under weights. It shows small F1 gains at H4 and H8, a loss at H12, and no consistent gain in discrimination.
- FIT and C F1 rise at every horizon while forward gains are mixed. The raw gaps between roles are not causal overfitting estimates, because prevalence and populations differ.
- Stage 1 remains unresolved. This is not universal proof that partitions fail.
- This is a finite checkpoint. Any next research needs its own supervisor-led specification, and no D47 starts automatically.

## Evidence

**Task research holds exact byte copies** (CRLF preserved):
- `d46_summary.json` — sha256 `4c9fb365a6f44570f1c80084900074d4880c15d52a4cfb4ca33b41bc57244b93`
- `d46_identity.json` — `053b497cd34d872d4406f88dbb3d2fedd15801aa28af26dbb38f6d554c2ff351`
- `d46_gate.json` — `930c5d52f5cf4d9ad81e21e47ce8f6a160df9fb9544a99f897ea227bc95aaba8`
- `d46_supervisor_check.py` — `440863384b7cb2f68405c9f956629733b4dc6f800fedb36b57455c0bb72a3e9f`
- `d46_supervisor_results.json` — `cbb7982ebbf854e3cad3c502c40a648ff8cb225db0aebf3d7189ca563a353d15`

**Kept external** in the run directory: the weighted UBJ files and records, the keyed FIT/C/E3 rows and `completion.json` (`8f5a8a5ece1d71b98bbada73af37869329e204643e11b8c0ac4b20036ad473e4`).
