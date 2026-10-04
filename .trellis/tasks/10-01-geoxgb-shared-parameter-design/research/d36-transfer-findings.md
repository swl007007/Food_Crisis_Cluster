# D36 transfer diagnostic: findings (analysis only, no fits; 2026-10-02)

**Scope:** existing D34/D35 C and E3 predictions only. Snapshot rows were read up to 2020-12 only; no 2021+ labels or scores. No production or model changes. Rows are repeated keys across horizons (different model predictions, not independent observations). Transition grouping uses target truth, so it is a post-hoc error grouping, not a deployable rule.

## Inputs

- **D34:** `C:\Users\swl00\geoxgb_runs\geoxgb-d34-e1-brier-20261002`, producer `7b2bf6fe482d0a77a664f3627e493934d976696d`.
- **D35:** `C:\Users\swl00\geoxgb_runs\geoxgb-d35-global-increment-20261002`, producer `be5f485`.
- **Supervisor script (the checked final run):** `research/d36_transfer_diagnostic.py` (sha256 `b118c4bc…f08a461`), copied from `C:\Users\swl00\geoxgb_runs\d36_transfer_diagnostic.py`. Before the successful run it was fixed for the numpy-2 `np.select` default and for the ledger month-index join; the output is that final run, not the failed attempts.
- **Outputs**, kept outside the repo in `C:\Users\swl00\geoxgb_runs\d36-transfer-diagnostic-20261002\`:
  - `decomposition.csv` (sha256 `5adfb532…`)
  - `persistence_matched.csv` (`70d8a442…`)
  - `summary.json` (`230bb9b1…`)
- **Supervisor checks:** 321,047 C/E3 rows. All stratified counts add back to the totals. D35 all-population F1 is reproduced. 81,321 E3 keys match the existing 15-fold persistence ledger.
- **Executor evidence** (independent code, retained in `research/`):
  - `d36_executor_verify.py` → `d36_executor_verify.json`
  - `d36_executor_review.py` → `d36_executor_review.json`
  - `d36_executor_era.py` → `d36_executor_era.txt`

**Persistence:** the exact origin label at O = T − H, `hist_phase_o00 − 1`, with phase 5 folded into code 3. It is not `hist_latest_observed_phase`. Missing origins are kept separate.

## Headline evidence

All figures match between the supervisor's run and the executor's independent recomputation.

**Brier-local vs root:**
- E3: ΔTP +78, ΔFP +162; corrected 248, spoiled 332.
- C: ΔTP +414, ΔFP +197; corrected 598, spoiled 381.

**Crisis Brier loss (p3 + p4/5), root → Brier-local:**
- E3: .101890546 → .101950206 (worse).
- C: .047913908 → .047123033 (better).

So the small E3 F1 gain is not an argmax view hiding a strong probability gain.

**Persistence coverage:** C .62409, E3 .99372.

**Matched-key crisis F1:**

| Set | Root | Brier-local | Persistence |
|---|---|---|---|
| E3 | .551516 | .552179 | .586406 |
| C | .677720 | .686491 | .519468 |

On the ledger, 80,613 E3 keys have persistence known in both sources, and all are identical.

**E3 transition groups**, Brier vs root (00 stays non-crisis, 01 onset, 10 relief, 11 stays crisis):

| Group | Change | Net corrected − spoiled |
|---|---|---|
| 00 | +130 FP | −130 |
| 01 | +28 TP | +28 |
| 10 | +32 FP | −32 |
| 11 | +50 TP | +50 |

On C the same groups net −75 / +169 / −50 / +80, with +93 from missing-origin rows.

**Root vs persistence on E3** (known keys):
- 00: root 1,585 FP vs persistence 0.
- 01: root 701 TP vs persistence 0.
- 10: root 3,444 FP vs persistence 7,614.
- 11: root 9,060 TP and 2,647 FN vs persistence 11,707 TP.

These recompute root F1 = 19,522/35,397 and persistence F1 = 23,414/39,928.

**Country** (net corrected − spoiled):
- ET flips from C 171/88 (+83) to E3 34/81 (−47).
- In C, the top three countries (ET, KE, NG) account for 65.1% of the absolute net.
- In E3, the largest negatives are ET −47, SO −16 and GT −15.

**Structure of C** (`d36_executor_review.json`, `d36_executor_era.txt`):
- 207,497 of 207,539 C rows (99.98%) have fitting rows from the same country and month. E3 rows have none, by construction.
- 141,365 of 207,539 C rows (68.1%) are from 2012–2016, while E3 targets run 2018-06 to 2020-06.
- Persistence is missing for 52–66% of C rows in 2013–2015.
- The largest C net by year is 2015 (+82).

## Alternative explanations (hypotheses, not established)

1. **C measures interpolation inside the training window.** Contemporaneous fitting rows in the same country and month give the root and the locals information that is unavailable at E3. This plausibly explains why the root beats persistence on C but not on E3.
2. **An era-specific signal.** C is dominated by 2013–2016, including the largest net in 2015, and ET swings from positive to negative. A regional pattern learned from that era may not recur.
3. **The onset gain doesn't transfer, while false-positive cost scales up.** The 00 group is 75% of E3 rows against 51% of C rows.
4. **Overfitting of the local search remains possible,** but the gap cannot be attributed to it alone. Partition capacity and spatial structure are also still confounded.

## Reading

The best partition E3 gain on matched keys (+.0007) is about 2% of the root's deficit to persistence (.0349). That deficit concentrates in missed persisting crises (2,647 FN in group 11) and false positives in stable non-crisis areas. C cannot certify temporal skill.

This motivates examining the root's temporal robustness against persistence before any further partition tuning. That is a design question for the supervisor; no experiment is started.
