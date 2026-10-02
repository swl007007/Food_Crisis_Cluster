# D45: historical improvement does not consistently transfer to forward probability quality

Supervisor synthesis, 2026-10-02. Planning commit `950566e`; contract `d45-root-prefix-diagnostic-plan.md`. External run `C:\Users\swl00\geoxgb_runs\geoxgb-d45-root-prefix-20261002` completed in 48 seconds: 21 saved D34 roots, 63 prefix evaluations across FIT/C/E3 (189 scored part predictions, plus replay gates), zero fits. No Stage2/3 or final-period evaluation.

## Verification

All full-root C/E3 probabilities and labels replayed exactly; `(0,total_rounds)` equalled default full prediction. The native scoped reviewer found a missing comparison between snapshot SHA and the prepared record; the executor added that comparison and the outputs-manifest binding before running. Frozen runtime, schema, producer, model bytes, parameters/rounds, memberships and matrices passed the run's checks.

The supervisor independently used `Booster[:r]` for quarter/half and default full prediction, instead of the producer's `iteration_range`, on every FIT/C/E3 row for all 21 roots. `d45_supervisor_check.py` reproduced all 189 role/prefix metric cells and the persistence-matched E3 subsets. `d45_compare_checks.py` reconciled per-pair and per-H pooled/mean metrics, differences, support/prevalence and decision changes, then raw-replayed every persisted C/E3 prefix probability on all **321,047 keyed rows**. Both independent checks passed with no discrepancy. These are numerical checks, not independent statistical tests. No production code changed or new model was fitted.

The prefixes are observations along saved fitted paths; equivalence to separately trained shorter models was not established. Four trees form one boosting round here. No zero-round prediction, scalar probability shrinkage, round selection or early-stopping policy was used.

## Forward results

All-key E3: seven dates per H, **37,836 rows per H**. The grids have different absolute rounds, so results remain per H.

| H; rounds quarter / half / full | Crisis F1 | Crisis Brier | Four-class log loss |
|---|---|---|---|
| H4; 50 / 100 / 200 | .625567 / .631478 / .628357 | .093226 / .089693 / .090834 | .677607 / .627415 / .616084 |
| H8; 100 / 200 / 400 | .509841 / .520911 / .532228 | .103984 / .106987 / .110754 | .707589 / .701772 / .716542 |
| H12; 100 / 200 / 400 | .416341 / .440524 / .477156 | .104296 / .103321 / .104083 | .743098 / .726406 / .734396 |

Losses are lower-is-better; crisis F1 remains the scientific target. Half minus full, all-key E3:

| H | Pooled F1 difference | Mean-fold F1 difference | Brier difference | Log-loss difference | TP / FP difference |
|---|---:|---:|---:|---:|---|
|4|+.003120|+.002441|−.001141|+.011331|0 / −62|
|8|−.011316|−.013829|−.003768|−.014770|−201 / −293|
|12|−.036632|−.035941|−.000762|−.007990|−288 / −175|

H8 illustrates why counting corrected errors is insufficient for the chosen objective: halving corrects 425 decisions and spoils 333, but removes 201 true positives along with 293 false positives, reducing crisis F1. Neither probability loss nor overall error count can silently replace the F1 objective.

On exact-persistence-available E3 keys:

| H | Quarter F1 | Half F1 | Full F1 | Persistence F1 |
|---|---:|---:|---:|---:|
|4|.626255|.632178|.629050|.651697|
|8|.511051|.522073|.533375|.555614|
|12|.417412|.441636|.478355|.549808|

Every tested prefix remains below same-key persistence at every H. Persistence is scored by crisis/fixed-four F1 and one-hot Brier only, without log loss. These are the 21 D34 cases, not D44's different 15-case aggregate or historical Stage3 scores.

## What the historical roles show

From half to full, both FIT and C improve both losses in **all 21 pairs**. Their pooled crisis F1 also rises for every H:

| H | FIT F1 half → full | C F1 half → full | E3 cases with worse Brier / worse log loss (out of 7) |
|---|---|---|---|
|4|.660307 → .693452|.648804 → .679029|5 / 1|
|8|.677755 → .743506|.650107 → .702297|6 / 5|
|12|.566825 → .646005|.550255 → .622801|4 / 4|

Thus historical held-out improvement is an unreliable guide to the observed forward probability-loss changes. This adds evidence about the root itself, beyond the earlier local-map diagnostics. It does not prove a unique causal source of overfitting or distribution drift.

The populations differ: pooled FIT/C crisis prevalence is approximately H4 **12.28%/12.17%**, H8 **11.46%/11.38%**, H12 **10.71%/10.73%**, versus E3 **18.23%** for each H. FIT/C contain 15–18 real label dates per pair; E3 contains one. Training/holdout row counts overlap across roots and are not independent sample sizes. Raw FIT-to-E3 score gaps therefore mix population differences with generalization; do not call the gap a causal overfitting estimate. G selection and repeated development exposure still apply, and C remains historical interpolation.

## Decision

**Diagnostic completed; no prefix/early-stopping policy adopted.** The simple claim that all roots merely need fewer boosting rounds is not supported: H4 has a small descriptive F1 benefit at half, while H8/H12 lose F1 despite improved probability losses relative to full. No further round grid follows this diagnostic.

Stage1 overfitting/generalization remains unresolved. D45 narrows the next root hypothesis: explain the forward failure of historical probability improvements while preserving useful crisis detection. It does not authorize changing the objective to log loss, treating C as forward validation, adding SHAP/other diagnostics, fitting another model, propagating D38 to partitions, restarting map variants, entering Stage2/3/full648/final, or closing the active task/audit. Any next bounded investigation needs its own supervisor-led specification.

## Evidence

Task research preserves the producer script and `d45_summary.json` / `d45_identity.json`, plus `d45_supervisor_check.py`, `d45_supervisor_results.json`, `d45_compare_checks.py`, and `d45_comparison_results.json`. The 35.8 MB joined C/E3 probability file and original models remain external. Summary SHA256: `2067978d533dd49843c6b3a5cddff0566967f74347bf30c4eddc4344ea341e57`; identity SHA256: `ca6f63cc33d68cc7d9e60c6a70b63ada1f536bd565c0d2636afa112242528a61`. Exact remaining hashes are in the identity and independent-check records.
