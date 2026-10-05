# D52: independent binary-objective root diagnostic (2026-10-02)

**Scope:** contract `d52-binary-root-proposal.md`, proposal commit `1108cc9`, user approval ("可以") recorded in `6b16bec`. Diagnostic only: the four-class main model, endpoint (four-class argmax → code ≥ 2), contract (D26/R11) and final criterion are unchanged.
- **Change:** one standalone `binary:logistic` root per D34 pair (target: original code ≥ 2), with the G parameters copied and only `objective` replaced, `num_class` and `multi_strategy` removed and `base_score=0.5` added.
- **Unchanged:** full 162 features, all original FIT rows, order and labels (including missing-origin rows), W59, G (H4 G1 / H8 G4 / H12 G2), rounds 200 / 400 / 400, seed; unweighted; no margins or local trees.
- **Arms:** binary (p ≥ .5); original normalised mass (s = (p2 + p3)/Σp ≥ .5); original pipeline argmax (existing endpoint); exact-origin persistence. Contrast A = binary vs original mass (same .5 rule); contrast B = binary vs original argmax, and each vs persistence.
- **Interpretation limit:** the objective changes together with capacity (1 tree per round vs 4), the Hessians that `min_child_weight` and `reg_lambda` act on, and G selected for the four-class model. No pure objective attribution.

## Producer, tests and run (executor factual record, `171d328`)

- **Producer** `39acbaf`: task-research runner `research/d52_binary_root.py` (892 lines), git blob `ef817e155e32c0961ce13e28da0fb9e2f953ed38`, sha256 `6bbab83c67301a5a3f57d4e8acc1559a050e7834839c0ad9ac74226baebfc5b7`. No package or schema edits.
- **Native implement** (about 3.4 min). The supervisor's source review found the initial per-root gate+fit order violated plan §4 step 2. A native fix (about 1.7 min) moved to two passes and added early-pass/later-fail (zero fits) and call-order selftests. A native check of the final file (about 1.4 min) found no result-affecting issue and made no edits. Synthetic selftest OK at HEAD; no package tests.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d52-binary-root-20261002`, frozen Windows Python 3.12.10 with assertions on (`sys.flags.optimize = 0` recorded by a launch of the same interpreter immediately before), exit 0 in 161 s. Log: `research/d52-run.log`.
  - **Order:** pass 1 gated all 21 roots (replay, G lock, FIT keys, D49/D50 consistency) before the first fit.
  - **Fits:** exactly 21, on 1,339,197 FIT rows in total; parameter diff exactly as approved; recorded `base_score` 5E-1; 162 features; 0 reload mismatches; 0 log-loss clips. `dev_baselines` 15 checked / 6 not covered.

## Verification (supervisor, distinct from the executor record)

`research/d52_supervisor_check.py` → `research/d52_supervisor_results.json` (log `research/d52-supervisor.log`): **PASS**.
- 9,111 checks, 504 metric cells, 1,660,244 FIT/C/E3 rows; no producer imports and no fits.
- Both raw UBJs replay exactly; full FIT keys, order and labels and the binary mapping; params, dimensions and tree counts; all-key and matched metrics; exact F1 contrasts; within-phase pair-count AUC; 107 output hashes and the inventory.
- The run log separately confirms that 21 gate-pass records precede 21 fit-pass records, with exit 0 in 161 s.

These are numerical checks, not statistical tests.

## Results

**Matched E3 crisis F1 (excluded 142 / 284 / 287 per H):**

| H | Pooled: binary / mass / argmax / persistence | Mean-fold: binary / mass / argmax / persistence | Crisis-call share (pooled) |
|---|---|---|---|
| 4 | .601942 / .611056 / .629050 / .651697 | .598453 / .607522 / .626608 / .655255 | .1344 / .1380 / .1499 / .1778 |
| 8 | .516022 / .518645 / .533375 / .555614 | .513690 / .518383 / .534031 / .555568 | .1396 / .1450 / .1567 / .1718 |
| 12 | .465093 / .470746 / .478355 / .549808 | .463434 / .472407 / .479043 / .547711 | .0847 / .0758 / .0867 / .1643 |

**Fold wins (wins / ties / losses of 7):**

| H | Binary vs mass (A) | Binary vs argmax (B) | Binary vs persistence | Argmax vs persistence |
|---|---|---|---|---|
| 4 | 3 / 0 / 4 | 2 / 0 / 5 | 2 / 0 / 5 | 4 / 0 / 3 |
| 8 | 3 / 0 / 4 | 2 / 0 / 5 | 2 / 0 / 5 | 2 / 0 / 5 |
| 12 | 2 / 0 / 5 | 3 / 0 / 4 | 1 / 0 / 6 | 1 / 0 / 6 |

**Shared binary probability scores and ranking (mean-fold, binary p vs original s):**

| H | Crisis Brier | Binary log loss | AUC | AP |
|---|---|---|---|---|
| 4 | .090377 vs .090825 | .287017 vs .287657 | .908362 vs .906903 | .736960 vs .732831 |
| 8 | .111247 vs .110801 | .353370 vs .346810 | .864793 vs .865053 | .650606 vs .650255 |
| 12 | .108613 vs .104086 | .348471 vs .334192 | .852295 vs .858335 | .630329 vs .644472 |

Persistence one-hot Brier: .125517 / .157566 / .156223. The four-class macro-F1 and log loss remain original-only references.

**FIT and C pooled crisis F1, binary / original argmax** (binary is higher at every H):

| H | FIT | C |
|---|---|---|
| 4 | .726732 / .725812 | .716539 / .715605 |
| 8 | .771007 / .765759 | .733109 / .727744 |
| 12 | .631686 / .609325 | .613890 / .589058 |

**All-key E3 F1, binary / mass / argmax:** .601254 / .610363 / .628357; .514865 / .517498 / .532228; .463935 / .469526 / .477156.

**D50 within-phase E3 AUC (valid-fold means, binary / original):**

| H | Phase 1 | Phase 2 | Phase 3 | Phase 4-or-5 (4/7 valid) |
|---|---|---|---|---|
| 4 | .748762 / .773484 | .668271 / .655702 | .745143 / .736468 | .782122 / .803153 |
| 8 | .812793 / .824208 | .606128 / .598012 | .740289 / .745844 | .962314 / .938756 |
| 12 | .827144 / .818068 | .593962 / .605804 | .693676 / .728402 | .839840 / .854736 |

Phases 1–3 are valid on 7/7 dates at every H. Supports: phase 1 P 4–159, N 2,294–2,916; phase 2 P 167–624, N 911–1,588; phase 3 P 312–798, N 176–688; phase 4-or-5 negatives 0–38 (null on 3/7 dates per H, not robust).

## Supervisor synthesis and decision

**D52 is complete and the binary root is not adopted.**
- Matched E3 crisis F1 is below both original decision rules and persistence at every H, pooled and mean-fold. Binary beats the original argmax on only 2/7, 2/7 and 3/7 folds, and persistence on 2/7, 2/7 and 1/7.
- The other metrics are not uniformly negative: at H4, Brier, log loss, AUC and AP improve slightly despite the F1 fall; at H8, Brier and log loss worsen, AUC is slightly lower and AP slightly higher; at H12, Brier, log loss, AUC and AP all worsen.
- Binary FIT and C F1 are **higher** than the original argmax at every H, while E3 F1 falls at every H, so the historical-to-forward gap widens. C is in-window interpolation, and no pure causal overfitting claim is made. This strengthens the not-adopted decision.
- Capacity and Hessian differences and G chosen under the four-class model prevent pure objective attribution. The result is not proof that the partition idea is impossible.
- Stage 1 remains unresolved. The Stage 2 formula, Stage 3, final evaluation and close remain deferred; no automatic D53 and no binary tuning grid.

## Evidence

**Task research holds exact byte copies** (byte-compared with the originals):

| File | Bytes | sha256 |
|---|---|---|
| `d52_summary.json` | 637,195 | `af3764bf467e2862c77a53e60279b313cf1b663d5945c74e8ba49f2fd22693ba` |
| `d52_identity.json` | 45,149 | `a35065c6a926fdfc4e748d2dcca431efabc6b9832b8b6916b6fa412c3085de9d` |
| `d52_gate.json` | 140,641 | `ade97c0cd57127a3775f18dfd67937f7abbe6e40fca13859f3bc136679316702` |
| `d52_completion.json` | 12,923 | `ae551e62066e66a8305125448e3b3a254ce1a8544f107b03598a55c7c59315d2` |
| `d52_supervisor_check.py` | 17,149 | `d6da208c929d8d006623bef1e0a7edcc07ea8664ce1f577aa1212e3dccf081f4` |
| `d52_supervisor_results.json` | 285,180 | `82b4285b2f97ae4f6a3e091528a8a8f808f83b9a48a5ac73b54c0b28806b95e5` |
| `d52-supervisor.log` | 1,915 | `bca4b9bc17842cc83227ce4958f8453b68c6e853659233ef8da5bd3dde82f7b1` |
| `d52-run.log` | 37,282 | `18589afc1ef39fce83727f5076afd89f511b84dd39428ddbd8f9d599a158a0c6` |

**Kept external** in the run directory (about 109 MB): the 21 `binary_root.ubj` files and `binary_root.json` records and the keyed `rows_{FIT,C,E3}.csv.gz`; their 107 hashes are recorded in `d52_completion.json`.
