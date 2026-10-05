# D47: depth-1 (stump) root contrast (2026-10-02)

**Scope:** contract `d47-stump-root-plan.md`, planning commit `44e8409`. The crisis-F1 endpoint is primary and unchanged.
- **Data:** 21 D34 pairs with unchanged FIT/C/E3 keys, the 59-month window, the 162 features, the four-class objective, seed, eta, subsampling and regularisers.
- **Change:** only a copied G config (H4 G1 / H8 G4 / H12 G2) with `max_depth=1`. Rounds stay 200 for H4 and 400 for H8/H12.
- **Arms:** the saved original roots, the fresh stump roots, and persistence on the same keys.

## Producer, tests and run

- **Producer** `5f6eb4307c1bded78c89ff27ab76d04431a5995d`: `scripts/stage1_stump_root.py` (384 lines), reusing the D37 gate and rebuild helpers and D46's pure, arm-agnostic helpers. No src, adapter or default edits.
- **Native implement** (about 4.6 min) and **native check** (about 4 min, no result-affecting finding, no edits). The check confirmed the depth parser rejects a depth-3 booster.
- **Tests:** 2 narrow tests, `StumpRoot` (a failed gate means no fit; the config copy and depth check). The suite went from 114 to 116 tests, all OK with exit 0 after the commit.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d47-stump-root-20261002`, frozen Windows Python, exit 0 in 189 s. Log: `C:\Users\swl00\geoxgb_runs\d47-run.log`, sha256 `83b84566a813ec67804b09a97f3d11661d42ee96e8021dada6fa903b18d5b43a`.
  - **Fits:** exactly 21 stump fits, all with UBJ hashes different from the originals. All 21 original-root gates passed.
  - **Tree structure:** every tree has depth 1 with exactly one split and two leaves (800 trees per H4 root, 1,600 per H8/H12 root); there are no deeper or zero-split trees.
  - **Other checks:** `base_score` `5E-1`, exact UBJ reload on FIT/C/E3, and plan defaults unchanged.

## Verification

The supervisor's independent check, `research/d47_supervisor_check.py` → `research/d47_supervisor_results.json`, passed:
- 172,573 checks (many of them saved-tree node checks), 360 metric cells and 1,660,244 FIT/C/E3 rows;
- no producer imports and no fits; independent snapshot keys; exact raw UBJ replay of the original and stump roots;
- params equal to the original except `max_depth` 1;
- per-tree depths and counts, and metadata split and leaf counts;
- base score 0.5, 4 classes and rounds, with no weight or margin;
- per-pair and per-H scores, AUC/AP and crisis-call shares.

These are numerical checks, not statistical tests.

## Results (E3; FIT/C in `d47_summary.json`)

| H | Matched crisis F1: original / stump / persistence | Fold wins, stump > original (all keys) | Matched crisis-call share, original → stump |
|---|---|---|---|
| 4 | .629050 / .613158 / .651697 | 3/7 | .149865 → .132780 |
| 8 | .533375 / .471098 / .555614 | 1/7 | .156716 → .100075 |
| 12 | .478355 / .399743 / .549808 | 0/7 | .086713 → .066154 |

- Mean-fold within-root AUC/AP is lower for the stump at every horizon (secondary diagnostic).
- **Crisis Brier, all keys, original → stump:** H4 .090834 → .089238 and H8 .110754 → .105221, but H12 .104083 → .107724.
- **Log loss:** better for the stump only at H8 (.716542 → .707943).
- **FIT and C:** crisis F1 falls substantially at every horizon (FIT H4/H8/H12 .6935/.7435/.6460 → .5797/.4730/.4188).

## Supervisor synthesis and decision

**D47 is complete, and a fixed-round depth-1 replacement is not adopted.** The depth/round sequence stops, and no D48 starts automatically.
- On E3 the stump is below both the original and persistence at every horizon.
- This fixed budget sheds crisis detection and some ranking. It is not a successful overfitting remedy.
- Smaller role gaps are not evidence that overfitting is solved.
- The result cannot refute additive models in general or isolate an interaction effect, because capacity, optimisation, sampling paths and effective regularisation all changed together.
- Stage 1 remains unresolved, and this is not a universal claim about whether partitions are feasible.

## Evidence

**Task research holds exact byte copies:**

| File | Bytes | sha256 |
|---|---|---|
| `d47_summary.json` | 695,643 | `d33a94316db638b60c04a8869119683f92f729cf6d39efb882778cd227721606` |
| `d47_identity.json` | 41,865 | `56590764e5059927dba0d478fd69c3390a913795cc5a062ce818b73f3e3b4b75` |
| `d47_gate.json` | 60,990 | `3c72cc9a71f766b7a808dbd86ad60f28fcdb58ac7dab2cecc302bb92086e406a` |
| `d47_supervisor_check.py` | 9,597 | `e3503f827026e4334e3c3f75f701e408417ae0befebfc6e5d36ee375b2c55509` |
| `d47_supervisor_results.json` | 169,361 | `593bfeb4c8eb0c383157bdb55757f1fa8495bd0b5d657f7767823c260237028a` |

**Kept external** in the run directory: the stump UBJ files and records, the keyed FIT/C/E3 rows and `completion.json` (`580916660cb5089266bf22e408b4dbe2991949467cb0e4a8059d3f4428df222c`).
