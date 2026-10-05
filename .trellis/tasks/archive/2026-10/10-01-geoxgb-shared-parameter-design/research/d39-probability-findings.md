# D39 saved-probability diagnostic: findings (2026-10-02; zero fits)

**Scope:** the 21 verified D38 `rows_E3.csv.gz` only (producer 2d4fe4e, input sha256 recorded in the main JSON). No model fits, threshold search, calibrator fits, final ledgers or 2021+ data. Exposed development E3. Folds overlap and are reused, so there is no significance inference.

## Evidence

- **Supervisor main diagnostic:** `research/d39_probability_diagnostic.py` (sha256 `6f4c0946…`) → `research/d39_probability_diagnostic.json` (line endings normalised; original sha256 `143a2293…`). Originals in `C:\Users\swl00\geoxgb_runs\`.
- **Executor independent check:** `research/d39_executor_check.py` (sha256 `e368b11e…`) → `research/d39_executor_check.json` (`fde791f9…`). Originals in `C:\Users\swl00\geoxgb_runs\d39_executor_check\`. It reads the saved rows only, asserting their hashes against the main JSON. It uses no sklearn and none of the supervisor's code.
  - **AUC:** Mann–Whitney with average ranks over tie blocks (O(n log n)).
  - **AP:** step definition over distinct, grouped score thresholds.
  - **Bins:** comparisons against the literal edges .1…9, with [.9, 1] closed; the bin index uses a clamped p, while Brier uses the raw values.
- **Verification scope (exact):**
  - **Coverage:** overall, H4, H8 and H12, and six individual roots, the first and last of each horizon (2018-06 and 2020-06).
  - **Row sets:** all, matched, missing, origin-0 and origin-1.
  - **Models:** original, anchored, post-hoc and prior-only, plus persistence on matched rows and within each stratum.
  - **Quantities:** n, positives, mean p, Brier, AUC, AP, argmax confusion, the argmax-vs-0.5 table with corrected/spoiled counts and F1, and all 10 bins.
  - **Result:** 2,414 comparisons, 0 issues (tolerance 1e-12).
  - The remaining 15 roots were checked only through the overall and per-H aggregates, not individually.
- **Identities checked:**
  - All 713 missing-origin prior-only rows have `p_crisis` exactly 0.5 and argmax code0. Prior-only's argmax and the 0.5 rule disagree on exactly those 713 rows, a tie-convention artifact, not a learned effect.
  - Prior-only takes only the values .25 and .75.
  - Prior-only argmax equals persistence on matched keys.
  - Within each origin stratum, prior-only and persistence have AUC .5 and AP equal to the stratum's crisis rate (.0952 / .6059).
  - Pooled matched AUC is identical for prior-only and persistence.

## Results (matched keys unless stated)

**Ranking:**
- Pooled AUC/AP: original .8695/.6522; anchored .8728/.6611; post-hoc .8718/.6610; persistence .7428/.4231.
- Persistence's figures come from a two-level score with ties. They are mathematically comparable references, but they do not establish ranking within a level or dominance of a whole curve.
- The persistence operating point was not tested against the root's curve.

**Within origin strata** (persistence AUC .5 by construction):

| Stratum | Model | AUC | AP | Crisis rate |
|---|---|---|---|---|
| Non-crisis origin | original | .8033 | .2408 | .0952 |
| | anchored | .8043 | .2392 | |
| | post-hoc | .7974 | .2314 | |
| Crisis origin | original | .7456 | .8442 | .6059 |
| | anchored | .7506 | .8466 | |
| | post-hoc | .7392 | .8391 | |

**Argmax vs fixed `p_crisis ≥ .5`** (original root):
- The two rules disagree on 1,438 rows. Argmax calls crisis with mass below .5 on 1,367 of them; the reverse happens on 71.
- So the four-class argmax is more liberal than the 0.5 mass rule.
- The fixed rule's F1 is worse: .5378 vs .5515 (corrected 775, spoiled 663).

**Fixed-bin reliability, original root, mean p vs crisis rate:**
- **Pooled:** under-prediction at low p (bin 0 .024 vs .042; bin 1 .143 vs .223); over-prediction in bins 3–8 (bin 6 .648 vs .530).
- **H8:** over-confident in mid-high bins (bin 6 .648 vs .389; bin 7 .745 vs .488).
- **H12:** under-predicts overall (mean p .152 vs crisis rate .183; crisis-call rate .087). Its upper bins are under-confident (bin 8 .856 vs .946).
- **H4:** close to calibrated in mean (.180 vs .183).
- **Bin limitation:** fixed equal-width bins on exposed E3 describe calibration coarsely. They do not authorise fitting a calibrator.

**Missing-origin rows** (713, of which 84 crises): original AUC .458, anchored .403. Too small to interpret.

## Reading (supervisor synthesis)

- A ranking signal exists in both origin directions (onset and persistence vs relief).
- This does not prove that the decision rule alone causes the F1 gap to persistence, nor that Stage 1 partition overfitting is solved.
- Collapsing to a fixed crisis mass ≥ .5 worsens F1, because argmax is already more liberal.
- Miscalibration differs by horizon (H8 over-confident in mid-high bins; H12 under-confident overall and at the top), so per-horizon results come before pooled ones.
- D38 keeps a modest root signal and is not the default.
- A pre-origin sequential threshold diagnostic on saved predictions remains design discussion until a separate spec exists.
