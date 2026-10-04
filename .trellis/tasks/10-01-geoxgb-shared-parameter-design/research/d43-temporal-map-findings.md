# D43: common forecasting does not rescue temporal map learning

Supervisor synthesis, 2026-10-02. Producer `1be4e3b`; external run `C:\Users\swl00\geoxgb_runs\geoxgb-d43-temporal-map-refit-20261002`. Three production-equivalence searches and all 21 prescribed H/T cases completed: 21 search roots, 24 searches including checks, 490 search child fits, 306 common regional refits, 926 seconds. No Stage2/3 or final-period evaluation.

## Verification

Independent checker `d43_independent_check.py` uses raw XGBoost and filtered pre-2021 snapshots, no production runner imports or fits. Result `d43_independent_results.json`: 5,761 checks, zero issues; all 306 regional models, 113,508 keyed E3 rows and 454,032 probability rows across four arms. It reconstructed original/temporal roles, search/current fitting identities, overlap, spatial routes and support, checked immutable current-root prefixes, replayed probabilities and recomputed all-key/persistence-matched scores per case, H, target and overall. Search algorithm itself was not independently reimplemented. Separate byte comparison of the three equivalence cases found identical assignment/correspondence/target/score files and all 77 UBJs. Unit suite: 110 passed after producer commit; native check found no result-affecting issue.

## Results

All 21 cases, persistence-matched **112,795 rows**, crisis-positive F1 from four-class argmax; Brier uses summed crisis probabilities and is lower-is-better. These are Stage1 development metrics, not the user's historical Stage3 ~.75/.8 metrics.

| Arm | Crisis F1 | Pooled row Brier |
|---|---:|---:|
| Current D34 root | .551516 | .101884595 |
| Global +20 | .551734 | .101944273 |
| Random-map common refit | .551979 | .101864403 |
| Temporal-map common refit | .551334 | .103006280 |
| Persistence | .586406 | .146407199 |

Persistence probabilities here are its one-hot class predictions, not a calibrated probabilistic model. Its worse Brier does not negate its better crisis F1.

All-key E3 (113,508 rows), temporal minus random: pooled F1 **−.000615**, Brier **+.001135**. Mean-fold F1 difference is **+.000610**: pooling confusion counts and averaging fold F1 answer different aggregation questions, and both are reported. Across 2,356 changed binary decisions, 953 errors are corrected and 1,403 correct decisions spoiled; TP +251, FP +701. These counts are not independent trials.

| H | Root F1 | Random refit F1 | Temporal refit F1 | Temporal−random Brier |
|---|---:|---:|---:|---:|
|4|.628357|.627861|.624110|+.000044630|
|8|.532228|.533740|.533937|+.001532165|
|12|.477156|.476931|.485699|+.001828459|

H12 has a genuine descriptive F1 increase; do not erase it with an overall null statement. Relative to random, its first three dates lose F1 and its last four gain; pooled TP +242 and FP +571. Greater recall with additional false alarms can improve F1 while worsening Brier. H4 loses F1; H8 is near-flat with worse Brier. No horizon's temporal arm exceeds matched persistence. This is insufficient for adoption or a claim that overfitting is solved.

Random/temporal maps have 166/140 eligible named regions in total; all pass existing fitting floors. E3 root fallback is 1.79%/.76%. Passing support floors does not establish statistical sufficiency; row counts share only a few actual label dates and spatial dependence. Search-root age, validation sample size, region coverage and search/refit label reuse differ between pipelines. Current forecasting root and fitting pool are held fixed. No causal attribution to chronology alone, no significance claim from 21 correlated repeatedly used development cases, no comparison against D42's different 12-case aggregate.

## Decision and Stage1 synthesis

**D43 is not adopted. Stop the time-split/map-generation variant sequence as predeclared.** Retain artifacts and the H12 signal as research evidence, without selecting H-specific map families post hoc.

- D32 fixed the engineering distinction between predictive root fallback and absent spatial evidence. It did not change predictions or solve numerical overfitting.
- D28 removed cumulative ancestral increments; D33 showed that even shallow accepted splits can fail forward transfer. Excess recursive capacity is not the only problem.
- D29/D36 showed that historical random confirmation largely measures interpolation among observed dates/countries. A small S-to-C gap cannot establish forward generalization.
- D34 removed hard-F1 zero-mass/tie degeneration; its future gain remained tiny. Fixing E1 numerics was useful but insufficient.
- D35/D41/D42/D43 controlled global extra rounds, increment amplitude, map vintage and common forecasting in bounded ways. None demonstrated a robust partition correction that closes the persistence gap. Effects are heterogeneous, so neither a universal failure of spatial partitioning nor a unique cause of overfitting has been established.

The engineering work is complete within its tested scope; the Stage1 scientific problem remains unresolved. Further local complexity is not currently supported by the evidence. Next planning should start from the remaining pooled/root prediction problem and a specific new mechanism, using existing evidence first. D38's fixed persistence-margin root is a candidate to understand, not a default or an authorization to combine it with partitions. Do not launch another split/window/shrinkage grid, reopen Stage2 formulas, consume Stage3, or close the task/audit from this synthesis.

## Persisted evidence (task research)

- `research/d43_independent_check.py` (sha256 `7e19607c…`), `research/d43_input_reference.json` (`b62ae55c…`) and `research/d43_equivalence_independent.json` (`d88c6d08…`): exact byte copies of the originals in `C:\Users\swl00\geoxgb_runs\`. The input reference is also byte-equal to `/tmp/d43_input_reference.json`.
- `research/d43_independent_results.json`: the original (sha256 `b4111bde…`) with only its CRLF line endings normalised (copy `e076409b…`). The parsed JSON is identical.
- **Kept external:** bulky models and rows in run `C:\Users\swl00\geoxgb_runs\geoxgb-d43-temporal-map-refit-20261002`. Hashes: `summary.json` sha256 `e6f0eca393518e446f1776e2c9fb16fffbafa7bc91847fda38f14aec7b5e869d`, `gate.json` `aec0a8a8…`, `identity.json` `6105536…`, `completion.json` `147d2ca1…`. Producer `1be4e3bb2c1e28216b5c395e0215e4576fdcac5b`.
