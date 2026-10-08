# IPCCH fixed-map yearly GeoXGB — design v1.0

Status: approved v1.0 on 2026-10-07 (“确认，可以进入执行。”). Execution: verified Claude Opus 5.5 1M, with Codex supervision. Implementation may start after the planning commit and verified lifecycle start; scientific fitting remains behind the P0 preflight checkpoint.

## 1. Question, arms and unchanged sources

Test whether the original frozen P6 regions add value under annual full-history, time-decayed XGB fitting. The primary contrast is annual gated GeoXGB G minus its matched annual pooled quartet P. The supported-cohort diagnostic is ungated local L minus P. L is a fresh continuation of its own immutable P global, not a second independently trained global or an MLP residual model.

Also report P versus original monthly P6 pooled, G versus original P6 GeoXGB, and each applicable model versus persistence on matched keys. These describe a bundled protocol change (refit frequency, history length, decay, annual gate); they do not identify an individual ingredient's effect. Previously viewed years remain exploratory.

Source run: `IPCCHGeoXGBExperiment/runs/p6-formal-20261004b`. Preserve original prepared arrays, ordered keys, feature order, targets/QC, derived truth phase, persistence, frozen maps and selected G/L recipes. No raw-panel rebuilding, imputation, feature selection, tuning, map learning, Stage2 or donors. Keep the original package and all runs unchanged. Use four scalar targets q2/q3/q4/q5, P6 bounded isotonic projection and highest unrounded projected q>=0.20 decoding. Regions outside a map remain in evaluation and use P.

| H | Frozen recipe | Global depth / rounds | Local depth / added rounds | Map areas / regions |
| --- | --- | --- | --- | --- |
| 1 | G1L2 | 3 / 200 | 2 / 40 | 3,264 / 9 |
| 3 | G3L2 | 4 / 200 | 2 / 40 | 3,264 / 7 |
| 6 | G4L2 | 4 / 400 | 2 / 40 | 3,264 / 6 |
| 12 | G2L2 | 3 / 400 | 2 / 40 | 3,264 / 9 |

All other parameters remain those in the original contract: objective reg:squarederror, CPU hist, seed42, nthread4, eta0.05, global base_score0.5; global min_child_weight10/lambda10/alpha0/subsample0.8/colsample0.8, local min_child_weight20/lambda20/alpha1/subsample1/colsample1. No early stopping, extra seeds or target-specific searches. Fit constant targets normally.

## 2. Annual blocks and feature availability

Dates are integer calendar-month ordinals. For H, let F_H be the first scheduled main target month, regardless of whether that month has truth. Define the historical/current model origin function:

```text
fit_origin(H, U) = F_H − H, if year(U)=year(F_H) and U>=F_H
                   January(year(U)) − H, otherwise
```

Main first targets are H1 2023-02, H3 2023-04, H6 2023-07, H12 2024-01. Their first block origins all equal 2023-01. Current blocks group scheduled targets by H, period and target year: main through 2025-12 and supplementary 2026-01..04. Later current blocks use January anchors. Historical months before the special first block use their ordinary January anchor; e.g. H6 2023-01..06 uses origin2022-07, not the future first-block origin2023-01.

Retain all 138 planned folds (122 main, 16 supplementary), 126 with valid target rows. Empty truth months stay in the calendar and never shift anchors, create observed gate dates, or produce fake predictions. Do not score missing truths or drop eligible rows for missing predictors/persistence.

For each unique global model origin O, fit all original prepared valid rows with target month t<=O. There is no rolling lower bound. Each fitting/evaluation row retains its original rich561 features at its own row forecast origin t−H; an annual model does not mean all feature vectors use January values. Save `row_origin_ord`, `fit_origin_ord`, annual anchor and block identity separately. Current-model origins follow the post-map-freeze schedule above. Historical replay's earlier map/recipe conditioning is disclosed in section 4.

## 3. Weighted global fitting and local continuation

In the pinned numerical environment, compute float64 `w=0.5**((O−t)/24)` using the ordered fitting rows. Require finite positive weights, age>=0, and identical row/target alignment. No normalization, renormalization within regions, age re-anchoring, or class weights. Record both the float64 protocol-weight digest and the effective float32 DMatrix weight digest. The latter verifies what XGBoost receives; it does not replace the former.

Fit each global target from scratch on that complete pool. A regional pool is its exact mapped-row subset, with identical indexed weights. A local quartet requires >=500 keys, >=50 areas and >=6 distinct target months. These remain raw-count gates; sum(w) and ESS=sum(w)^2/sum(w^2) are diagnostics only. Empty required global pools or numerical failures stop the run.

Every local target loads a fresh copy of its own matching global booster and adds exactly 40 L2 rounds. Preserve original root bytes, base score, first-G-tree structure and prefix-margin checks. Never continue from another target, another region, an earlier annual local model, or a local ancestor. The frozen global is not mutated by any local fit. Decay can change the effect of fixed regularization/min_child_weight; that is part of the adopted protocol, not a reason to retune.

## 4. Frozen annual gate

At current block origin O, choose the latest six globally observed target months U<O from the original valid ledger. Do not substitute six consecutive calendar months, choose dates by region/performance, or use in-block outcomes to update the gate.

For each U, use the model origin defined in section 2. Fit/reuse its full-history weighted global and region-local quartets, then predict the saved U rows using their own U−H features. A historical local request is needed only if the region has U validation rows. If its fitting support fails, retain those rows and use the historical pooled quartet on the local side; the date does not count as successfully locally supported. Technical errors stop rather than becoming fallback.

Merge validation confusion counts over all selected dates per region. Original validation support remains >=100 keys, >=20 areas, >=3 observed target months, >=20 crisis keys, >=20 noncrisis keys, and >=3 dates with successful local support. Count dates, not distinct fitted models, for the last rule. Also report distinct global/local model identities: the enumerated six validation months all use one historical annual model pair per region/block, so they are not six independent fits.

Compare projected/decoded local-routed versus pooled crisis F1 with exact count fractions; accept only if difference >1/100. Equality or an undefined necessary score does not pass. Gate scoring and final metrics are unweighted, as in P6; fitting decay weights do not become evaluation weights. Save support, confusion counts, exact F1 fractions, dates, providers and rejection reasons once per region×block. Gate and current fitting support are fixed for the block.

Historical models use the frozen through-2022 maps/recipes even when their fitting origin is earlier. Their predictions hold out each scored row from model fitting, but not necessarily from map/recipe selection. This is conditional internal validation, not fully historical deployment or an independent partition-selection test. No new claims about true publication vintages are made.

## 5. Current prediction and diagnostics

Fit/reuse P once per current block. Fit each mapped current region with any evaluation keys and sufficient block-start fitting support even if its gate rejects it. Preserve its ungated L quartet predictions. This is request scheduling only: future features/truth are never used in training, support or gate decisions.

For every evaluated row, G uses the complete L quartet only if annual gate and current fitting support pass; otherwise G equals P exactly. Route atomically across four targets. L exists only on the supported mapped cohort; do not encode unsupported fallback as an observed local effect. Save per-key raw and projected shares, phases, true shares/phases, country, row origin, block fit origin, route and model identities; copy persistence provenance unchanged.

Report full-cohort G−P and supported-cohort L−P separately, including historical-support-rejected, gain-rejected and adopted groups. Count decisions as region×annual-block, not repeated month rows. Report coverage by evaluation keys as well as distinct areas. Never extrapolate the L subset to unmapped areas or infer equal predictions from equal F1.

## 6. Exact workload and source freeze

Planning-only independent stdlib enumeration is saved in `research/enumerate_yearly.py` and `research/fit-enumeration.json`. It uses keys/calendar/maps, never fits or predicts. The executor must reproduce it before scientific fitting.

| H | Main unique global / local quartets | All periods global / local | All scalar fits |
| --- | --- | --- | ---: |
| 1 | 4 / 36 | 5 / 45 | 200 |
| 3 | 4 / 28 | 5 / 35 | 160 |
| 6 | 5 / 30 | 6 / 36 | 168 |
| 12 | 4 / 35 | 5 / 44 | 196 |
| Total | 17 / 129 | 21 / 160 | **724** |

Main requires584 scalar fits; supplementary adds140. All current mapped pools are training-supported. Main L diagnostic key counts are 9,943/9,866/9,522/7,904; supplementary has1,829 per H. Full main counts remain17,322/16,919/16,413/14,087 and supplementary4,092 per H. Historical gate support does not imply gain acceptance; the latter needs predictions.

The only unsupported unique historical local request is H12/origin2021-01/r0011 (443 keys/164 areas/27 months); it is retained as support fallback, not fitted. All four targets still count even if a target is constant. Reuse current/historical identical models; request purpose is not fitting identity. No extra project-data pilots, seeds, recipe searches or replacements for failed fits. Reconcile any source/count mismatch before fitting.

Input freeze binds P6 prepared manifest and all14 artifacts, ordered schema, four maps/frozen metadata, Stage1 summary/selection ledger, original contracts/runtime and four comparator prediction files. Preflight rehashes actual bytes. Bind reused source code and the complete new numerical/control/evaluation source inventory; do not rely solely on quartet.py's hash. The old source contract validates old maps, while the new annual contract separately defines this run.

## 7. Evaluation

Retain the requested P6 panel: four-class accuracy/macro-F1 (1/2/3/4–5), binary accuracy/F1/precision/recall/F2, projected q3 R² and raw q3 R² diagnostic. Retain confusion/per-class counts and original NA semantics, including constant-truth R² and missing-class macro-F1. Pool keyed counts within each H/period; do not average monthly F1 or create a cross-H headline.

Compare P/G with saved P6 pooled/GeoXGB on identical E_all keys and truth. Persistence comparisons use the same original E_persist subset for every arm; do not mix E_all model metrics with E_persist baseline metrics. Ungated L comparisons restrict every comparator to precisely L-eligible keys. Main and2026 remain separate; seed42 is fixed by the retained XGB recipe, with no new multiseed study.

For main G−P and G−persistence crisis F1, retain paired country bootstrap, 2,000 draws, RNG seed42 and fixed shared multiplicities. Preserve undefined-draw policy: no redraw/drop; interval only with >=2 countries and all draws defined. Other contrasts and2026/region/month diagnostics report point estimates. These intervals are conditional on fitted predictions and omit model/map selection and shared-shock uncertainty.

Positive gain is not a completion criterion. Discuss the bundled annual protocol, fixed map, same-model historical dates, weight mass, supported/unmapped coverage and already-viewed period explicitly. Do not interpret small effects as equivalence or broad failure of geographic learning.

## 8. Minimal isolated implementation and runtime

Use a sibling `IPCCHYearlyGeoXGBExperiment/`, namespace `ipcch_yearly_xgb`, with a small CLI for preflight/predict/report/replay. Reuse attributed copies of only needed P6 pure projection/metrics/errors and weighted adaptations of quartet/modelstore; retain root and record-integrity protections. Implement the annual scheduler/runner and small report/replay wrappers. No general backend framework, cross-repository runtime imports or sys.path injection; no copied scanner, feature builder, Stage1 learner or complete source run. File names can follow existing package conventions; the protocol is the contract.

Use the original Windows Python3.12.10 runtime and pinned numpy2.2.6/pandas2.2.3/xgboost3.0.0 CPU hist/nthread4/seed42. The original runtime lock also lists the remaining installed dependencies; verify it without upgrading or substituting WSL/IPCCH-global/MLP environments. Execute fits serially; exact cache reuse suffices for this 724-fit experiment. Stage inputs/models/logs under a new Windows LocalAppData directory outside Dropbox; inspect its actual disk space. Do not overwrite or clean prior runs.

P0 runs synthetic tests and a bounded weighted global/local synthetic timing check, freezes code/runtime/source identities and reports the 724-fit inventory plus resource estimate. No real-data pilot. Codex checks this concrete evidence before releasing the agreed run; mismatch or material implementation changes require reconciliation, not silent recipe changes.

## 9. Identity, replay and failure evidence

Each fitting identity includes protocol/code/runtime/source digests, H/target role, annual anchor/fit origin, selected recipe/parameters/rounds, exact ordered row/key and X/y digests, weight formula/half-life and weight-array digests. Local identities also bind region members/map and own global parent digests. Do not include current/historical request purpose in the identity. Save actual fit keys, row references, per-target records, UBJ states, requested/resolved params and request ledgers. Atomic complete entries only; corrupt/conflicting entries stop, never silently refit.

Replay reads saved models without fitting, reconstructs keys/cutoffs/weights and all required prediction requests, then re-predicts in the locked path and checks saved raw values exactly. Independently implement the short annual calendar, support/count-based gate, projection/decode and metric checks rather than invoking the producer wholesale and calling that independent verification. Test projection numerics at threshold using an independent method with accurately summed blocks; no post-result tolerance that can hide changed labels. Recompute report/CI from keyed cohorts and fixed draws.

Preserve P6 fitting-input CSV parsing semantics in the locked pandas version; identify/hash the actual parsed training arrays. For new saved prediction/weight evidence use lossless representations or round-trip float parsing. Replay must use the recorded representation of each artifact, not silently change the prepared target parser. Record digest policy and dtype.

Meaningful synthetic checks cover partial-first-year anchoring despite an empty first month, monthly feature origin versus fixed fit origin, same-year historical model reuse, strict gate equality, unchanged gates after later outcomes, raw decay/subset equality, weighted root-prefix preservation, current diagnostic under rejected gate, unsupported/unmapped fallback, cache reuse and zero-fit replay. Negative cases must catch changed fit origins/weights, omitted validation keys, wrong parent/map and within-block routing changes. Use the existing test framework without generating an unrelated full-repo suite.

Any fitting/prediction exception, nonfinite required value, missing required model/global pool, key/schema/source mismatch or corruption writes INCOMPLETE evidence and stops. Statistical fallback is separate. Final inventory covers original outputs and replay evidence and labels its coverage; distinguish request/cache-hit/disk-load counts and byte versus normalized-content comparisons accurately.

## 10. Release and delivery

After final approval, commit the approved planning artifacts before implementation. Verify the actual Claude executor and requested model, then follow the enrolled lifecycle at its exact registered lowercase repository path; do not reuse the old recorded Claude identity or another task's waiver. No task start/audit/fit occurs during planning. Close actions follow the user's current scope; scientific acceptance and an audit result are distinct.

Deliver source/protocol/runtime/provenance, tests, no-fit and final fit inventories, keyed predictions, gates/diagnostics, report/replay evidence, and a concise note for the existing IPCCH discussion documents. Preserve unrelated working-tree edits. Approval of a positive scientific result, push or PR is not implied by completion.
