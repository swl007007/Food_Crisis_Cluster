# IPCCH fixed-map MLP residual adaptation — design v0.2

Status: approved 2026-10-05. PRD R1–R25 records adopted requirements, including the user-approved Fable review revisions. Implementation is released through the P0 checkpoint; project-data fitting needs the user's explicit release afterwards. Executor: Claude Opus 5.5 1M; the user supervises. Trellis audit waived for this task only.

## 1. Question and comparisons

Test whether regional neural residual correction adds value on the original XGB-selected maps. This is not MLP-driven partition learning and cannot establish that these maps are optimal for MLP.

For each target q in {q2,q3,q4,q5}, horizon H, origin O, and replicate s:

- B: base global network `g_q(X)`.
- P: pooled residual correction `g_q(X) + p_q(X)`.
- L: supported region r correction `g_q(X) + l_q,r(X)`.
- G: deployed regional system, using the complete L quartet only where the historical gate and current support pass, otherwise the complete P quartet.

P and L correct the same immutable B; L is not added on top of P. Gate routing is atomic across q2–q5. The three full-cohort report arms are B, P, G. L is an additional same-key supported-cohort diagnostic, not a fourth full-coverage deployment arm.

Primary geographic deployment contrast: G−P on the full cohort. Diagnostic: L−P on eligible current keys, interpreted alongside historical paired results and support coverage. These answer different population-level questions; the diagnostic is not promoted to the primary contrast. Additional contrasts: P−B, B−original pooled XGB, G−original GeoXGB, and model−persistence on matched persistence-available keys. Do not label B an independently optimized global-only baseline: its architecture comes from selection of P.

## 2. Package, input authority, and immutability

Implement a separate sibling `IPCCHMLPExperiment/`, with import namespace `ipcch_mlp`. Keep the old `IPCCHGeoXGBExperiment/`, all its configurations and runs, and other experiment packages unchanged. Do not introduce a general multi-backend framework.

Reuse minimal, attributed copies of the original schedule, projection, metric, support, and reporting logic as appropriate; adapt the Stage3 workflow to B/P/L/G. Do not copy the scanner, learn-map machinery, XGB booster continuation, UBJ store, or complete old runs. No runtime imports from another experiment package and no `sys.path` injection of old `src`/`config` modules.

Authority is the original `p6-formal-20261004b` prepared dataset and maps, not the later split2024 sensitivity. The original implementation was frozen at 6798df2. Current input/source hashes are listed in [research/source-inventory.json](research/source-inventory.json); check them before consuming artifacts. All 14 prepared artifacts were rehashed successfully during planning. The executor must repeat verification, not trust this historical check alone.

Read or byte-copy verified prepared data into immutable scratch inputs. Reuse the original four frozen membership CSVs, preserving node IDs as strings; 3,264 mapped areas per H and 9/7/6/9 terminal regions. Do not apply original XGB leaf providers, G/L recipes, or enabled gate flags to MLP. Original predictions are read-only comparator artifacts, never training inputs.

Required lineage: prepared manifest, all ordered feature/key arrays, feature schema/order, development F/S ledger, fold calendar, target ledger, four membership maps and frozen-map metadata, and original comparator prediction hashes. Record original paths, hashes, code identity, and runtime identity in the new run manifest. Excluded regions stay in evaluation and use P; no donor assignment or map repairs.

## 3. Data and temporal boundaries

Use the original QC-valid 42,695 target records, rich561 values/order, q2–q5 targets, truth phase, country keys, and persistence provenance without regeneration or reinterpretation. Infinite prepared values or identity/key mismatches are errors; valid missing feature values remain eligible for preprocessing.

Development uses original F=8,561 and S=9,558 keys, excluding 1,472 singletons from development fitting/selection. F/S is chronological within each area, not a single global calendar split. Fit B and P on F only; choose architecture on complete common S. Do not refit on F+S after choosing the recipe. No development regional fits are needed, since maps are fixed and selection uses P only.

S is adaptive internal development evidence: it already informed XGB maps and now selects MLP capacity. It is not independent forward validation. All map/recipe development is bounded by 2022-12.

Stage3 retains H={1,3,6,12}, O=T−H and fitting targets in inclusive [O−35,O]. Each row's features remain those built for its own origin. Main target calendars start 2023-02/2023-04/2023-07/2024-01 and end 2025-12. Supplementary targets are 2026-01..04, reported separately. Retain all 138 planned folds and 126 scored folds; 2024-12, 2025-11 and 2025-12 have no valid truth in every applicable H and trigger no fits.

Historical gate dates are the latest six observed target months U<O, from the global valid ledger, not chosen by regional performance or restricted to the current 36-month window. Refit at V=U−H using [V−35,V], with the frozen map/recipe. Historical map/recipe information may postdate V: this is a conditional replay gate, not proof the complete system could have been deployed at historical V. Preserve the original observation-month-end availability assumption and its unverified release-vintage limitation.

2023–2026 has already been inspected. All new model comparisons are exploratory; there is no fresh confirmatory holdout claim.

## 4. Preprocessing

For each unique lawful global fitting pool, compute float64 per-column medians on observed values, replacing training-all-missing column medians with zero. Impute fitting values; compute their means and population standard deviations (ddof=0), substituting scale 1 exactly where variance is zero. Preserve all 561 columns.

Transform any associated row with those frozen values, then append 561 missingness flags in original feature order. Order is `[standardized values, binary flags]`; flags remain unscaled. Output is contiguous float32 of shape (n,1122), with explicit finite checks including after conversion. Do not clip extreme finite inputs, drop columns, refit transformations on local subsets, or use S/gate/test statistics. Share the transform across targets and all residual networks attached to that global fit.

Record transform fitting keys, medians, means, scales, all-missing flags, dtype and hashes. When a training-all-missing feature is observed later, apply the same zero-fill/mean and unit-scale transform; do not learn a new scale from future data. For each evaluation population/transform, record per-feature observed counts on training-all-missing columns and the number of affected rows. These are diagnostics only; retain the specified transform and do not infer numerical failure from an unseen missingness pattern alone.

## 5. Networks and training

| Role | Candidate ID | Hidden widths | Scalar parameter count |
| --- | --- | --- | ---: |
| Global | G1 | 64,32 | 73,985 |
| Global | G2 | 128,64 | 152,065 |
| Residual | R1 | 16 | 17,985 |
| Residual | R2 | 32 | 35,969 |

Each hidden block is Linear(with bias) → ReLU → Dropout(0.10); output is Linear(with bias), without clipping or activation. Four independent scalar networks per quartet; no cross-target parameter sharing. Construct layers in forward order using PyTorch Linear's explicit default initialization: weight uniform ±1/sqrt(fan_in), bias uniform with the same bound. For P/L residual networks only, then set the final output-layer weight and bias to exactly zero before optimizer construction; retain random hidden layers. B keeps default initialization in every layer. Both residual arms therefore begin at zero correction for finite inputs, including with hidden dropout active. Record the initial state digest after this override. The residual hidden-layer data gradient is zero on the first update, but the output layer can learn immediately on nonzero targets; this is not whole-network zero initialization. This rule supersedes v0.1's default residual output initialization without changing epochs, architecture, or fit counts.

Training is float32, unweighted mean squared error, AdamW(lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01). Apply decay to weight matrices only; bias groups have decay zero. Disable AMSGrad and fix `foreach=False`, `fused=False` to avoid backend-dependent optimizer selection. No scheduler, target scaling, BatchNorm, extra dropout, augmentation, SMOTE or class weighting.

Batch size min(256,n_fit); visit every fitting row exactly once per epoch, retaining the last partial batch. Use a recorded deterministic permutation stream each epoch. Train B for 100 epochs and residual networks for 40; save final weights only for selection/prediction. Retain per-epoch weighted-by-row training loss and final evaluation-mode fitting MSE, but never use logs to choose another epoch or alter the recipe. Constant targets still use the same network procedure.

Record actual optimizer updates as epochs × ceil(n_fit/batch_size). For residual networks, n=500/2000 gives 80/320 updates; the enumerated Stage3 global pools span 9,577–19,052 rows across H, giving P 1,520–3,000 updates. Equal epochs do not equalize optimization or accumulated decay. L−P is a comparison of the specified learning procedures, not a controlled effect of geography alone; neither fewer updates nor zero initialization proves undertraining, convergence, or a negative result.

Freeze B weights and keep B in evaluation mode. Compute B predictions once on the complete ordered fitting pool using the fixed inference batching. Subtract float64-converted B output from stored float64 truth, then cast residual targets to float32 with finite checks. P uses the complete residual pool; L takes the exact indexed regional subset of those same stored residuals. P/L have fresh parameters and optimizer states, with no inheritance from each other. Dropout in B remains off during residual fitting; dropout in the residual network is on only during its own training.

Prediction uses evaluation mode, no gradients, fixed inference batch size 256 and saved row ordering. Convert each scalar output to float64 before adding B+P or B+L; apply the original bounded equal-weight isotonic projection to the sum. No per-component clipping. Decode the highest phase with unrounded q>=0.20. Persist B and residual components, raw sums, projected values, and labels. Share predictions by exact model/input identity rather than changing inference batches during replay.

## 6. Selection and randomness

Replicate labels are 42,43,44. Derive actual per-model seeds deterministically from a canonical JSON identity containing stage, H, replicate, target, role, relevant candidate widths, origin, fitting-key digest and region when applicable, using SHA256 rather than Python hash. A B identity/seed includes its global width only, never the downstream residual width, so B is genuinely shared across R1/R2. Training permutations and dropout must be reproducible independent of request order or cache hits. Pin exact derivation in code/tests and record seeds; exclude caller-purpose fields from fitting identity.

Execute fits serially within each process. By the user's post-P0 decision (PRD R25) the three replicates run as three concurrent spawned processes, one replicate each, in development and Stage3; each process applies everything in this section to its own fits, writes its own request ledger (`model_requests_<stage>_rep<r>.jsonl`), and never shares a model identity with another replicate. Transforms are identical across replicates for the same pool and are stored by exclusive create (rename only if absent, otherwise verify the existing file). Development selection and the Stage3 summary are finalized in the parent after every worker succeeds; a failed worker leaves INCOMPLETE evidence and stops the run. Synthetic tests require parallel and serial runs to produce byte-identical artifacts and identical model tensors; `--serial` keeps the serial mode available. Derive and record separate initialization, training and permutation seeds from that identity with fixed stream labels. Reset PyTorch CPU and CUDA RNGs immediately before construction using the initialization seed, and again immediately before training using the training seed. Use a dedicated CPU torch.Generator for epoch permutations; use no DataLoader multiprocessing (num_workers=0 if a DataLoader is used). During a fit, no unrelated operation may consume the global RNG used by dropout; deterministic evaluation and logging must not consume it. A synthetic check must reproduce final tensors for the same fit after intervening fits/cache hits. This training reproducibility check is distinct from evaluation-mode saved-model prediction replay.

For each H and replicate, fit only two B quartets and four P residual quartets for G1R1/G1R2/G2R1/G2R2. Reuse each B across R1/R2. Score complete S with common keys and projected labels. Select one combination per H by arithmetic mean of the three exact count-based F1 fractions. Do not average predictions or pool replicated counts across seeds. Ties: fewer combined B+P parameters, then lexicographically smaller fixed candidate ID. Undefined required F1 makes the candidate ineligible with a reason; if none is eligible, stop with selection_unavailable. A technical failure is not an ineligible candidate to skip.

Freeze one winning recipe for all three Stage3 replicates of an H. Recompute gates separately per seed. Report mean/range of per-seed metrics and paired deltas, never best-seed results or a three-seed prediction ensemble.

## 7. Gate, fallback, and diagnostic

Every required global and pooled-residual fit uses a nonempty full lawful pool. Empty required pools stop the run. A regional fit needs >=500 keys, >=50 areas, >=6 target months; no fitting-class minimum is added.

Historical validation support requires >=100 keys, >=20 areas, >=3 target months, >=20 crisis keys, >=20 noncrisis keys, and >=3 dates with supported successful regional fits. Preserve unsupported historical dates and all their validation keys: on those dates the local side equals P, and the date does not count as a successful regional fit.

For each region/origin, merge confusion counts across the selected historical dates. Compare projected/decoded L-routed versus P using exact F1 fractions, requiring difference strictly >1/100. Equality fails. Undefined necessary F1 fails statistically; it is not zero. Support and gain decisions are separately recorded.

Fit every current mapped region with current evaluation keys and sufficient fitting support, including gate-rejected regions. Produce L diagnostic predictions on this supported cohort. G uses L only when historical gate also passes; otherwise G uses P bit-for-bit. Unmapped keys also use P. Do not delete keys based on missing history/persistence or a model's score.

The diagnostic report compares L with B/P/G and available original XGB predictions on the exact same eligible keys, with explicit denominators. Split diagnostic rows by historical-support rejection, gain rejection, and acceptance without selecting favorable regions. Never call unsupported fallback an observed zero regional effect. Diagnostic test outcomes do not revise models, maps, thresholds, or seed selection.

## 8. Evaluation and inference limits

Reuse the original metric semantics: four-class accuracy and fixed-axis macro-F1 (1/2/3/4-or-5), binary accuracy/F1/precision/recall/F2, projected q3 R² plus raw q3 R². Preserve per-class counts/confusion and NA reasons, including missing-class macro-F1 and constant-truth R². Pool keys within each H/period; no mean-of-month or cross-H headline F1.

Full E_all comparisons use identical keys for B/P/G and original XGB arms. E_persist is the same subset for every model, with original last-available true-population persistence, q3, source month and age. Model absence/corruption must not silently reduce the comparison set.

For each seed/H in the main period, retain the original 2,000-draw paired country bootstrap for G−P and G−persistence crisis F1, RNG seed42, full country blocks and fixed shared multiplicities per cohort. Retain original undefined-draw policy: no redrawing or dropping undefined draws; CI requires >=2 countries and all draws defined. Supplementary 2026 and supported/region/month diagnostics are point-only. Seed means/ranges are descriptive; no invented combined CI or independent-seed t-test. Other model contrasts report all point differences.

Diagnostics must include ungated L results, mapped/unsupported/rejected/adopted coverage, fitting support and validation support separately, raw/projected q3 MSE/R², crisis-label flip counts, and final train-versus-evaluation error summaries. Distinguish internal S, historical gate, and current test populations. These support interpretation, not post-result model tuning or causal attribution.

Predeclared main-period coverage, independently counted from original P6 prediction routes:

| H | Full cohort keys | Historically support-eligible keys | Share of full cohort |
| --- | ---: | ---: | ---: |
| 1 | 17,322 | 3,211 | 18.54% |
| 3 | 16,919 | 4,997 | 29.53% |
| 6 | 16,413 | 3,423 | 20.86% |
| 12 | 14,087 | 845 | 6.00% |

These counts exclude unmapped and historical-support-rejected keys, but do not require a positive model-dependent gain. With unchanged keys/maps/support and successful required fits, they upper-bound the number of current rows that could route to L. Recompute the keyed support sets without fitting in preflight, then report their intersection with current fitting support, actual adoption, and the distinct ungated L diagnostic population. Original XGB gain decisions are not transferred. These row shares do not bound F1 change or imply that a confidence interval must contain zero. G−P remains the full-cohort deployment estimand; L−P and historical pairs characterize supported regional correction without extrapolation to uncovered rows.

Delivery success means complete verified evidence, not positive effect. Negative and mixed seed/horizon results are valid research outcomes. Do not infer equal predictions from equal F1, or declare all geographic learning useless from this fixed-map comparison.

## 9. Runtime, storage, and computation

Create an isolated Windows Python 3.12.10 virtual environment outside Dropbox, based on the existing interpreter without modifying its site-packages. Pin numpy2.2.6/pandas2.2.3 and PyTorch2.6.0+cu124 from the official Windows cp312 wheel identified in [research/baseline-and-runtime.md](research/baseline-and-runtime.md). P0 records exact dependency versions/wheel hashes; do not import XGB or geospatial libraries unless a documented reused component actually requires them. Installing a new package version or changing the scientific recipe after freeze requires a recorded plan revision.

P0 tests CPU (four threads) and available CUDA on fixed synthetic data only, with float32, TF32/autocast/AMP off, deterministic algorithms on, cuDNN benchmark off and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initialization. Validate repeatable train/inference and saved-model reload. Choose the faster eligible device from fixed synthetic timings; ties choose CPU. With replicate-parallel execution the rule is re-applied to a three-process synthetic probe on the same case list. This is operational device selection, not model-score selection. If neither passes, stop. Freeze device, versions, precision, batch/order rules and deterministic settings before any project-data fitting; no silent mid-run CPU/GPU switch. Do not promise identical training across devices.

Keep models, scratch inputs, caches and logs under a unique Windows LocalAppData Temp run directory, outside Dropbox. Check free space on the resolved Windows volume holding the environment and run (normally C:), not WSL filesystem free space. Store small reviewed evidence and manifests under the task; large artifacts remain ignored with absolute locations and SHA256 inventories. New experiment/run IDs cannot overwrite P6 or another run. Preserve partial failure evidence.

The no-fit enumeration in [research/fit-enumeration.json](research/fit-enumeration.json) specifies 288 development scalar fits and 12,972 Stage3 scalar fits across three seeds: **13,260 project-data scalar fits total**, including the ungated diagnostic. Exact-identity cache reuse is mandatory. Reproduce these counts in executor preflight. The approved study is bounded by these requests; no extra seeds, recipes, failed-model replacements or optional refits. Source/count drift must be reconciled before fitting rather than silently expanding scope.

P0 synthetic timing uses both global sizes and both residual sizes at n=500/2000/8561/19052, fixed synthetic seed, at most one warmup and three measured repeats per device/case, using the specified initialization and 100/40 epochs. The 19,052-row case is the maximum over the 172 enumerated Stage3 global pools, not a selected example; reproduce that maximum in preflight. Report measured fit/inference/persistence cost and extrapolated full-run time/storage with uncertainty; it is not a promise based on XGB timings. Current planning established GPU availability and wheel existence, not working PyTorch or measured MLP performance. Executor sends the runtime/count/timing checkpoint to the supervisor before scientific fits.

## 10. Model identity, evidence, and replay

Each scalar state binds stage/H/q/replicate/role/origin, exact ordered fit keys and row references, X/Y/residual digests, feature schema, preprocessing digest, all parameters/epochs/initialization/RNG settings, environment/device/code identity. P/L additionally bind the frozen B state and residual-target computation. L binds frozen map and members. Request-purpose metadata is separate so historical/current uses can reuse identical fits. No cross-stage, cross-target, cross-seed or incompatible-source reuse.

Persist model state_dict, architecture and fit record, immutable transform, initial/final state digests, parent state digest, losses, fit keys and model request ledger. Hash tensors canonically; do not equate archive-byte differences with changed tensor weights. Write atomic complete entries; a referenced corrupt or conflicting entry stops, never silently refits. Save complete four-target prediction groups and provider links.

Replay must rebuild transforms from referenced fitting rows, verify B immutability and residual targets, reload saved models under the frozen runtime, and reproduce every saved development S, historical-pair and current B/P/L prediction using the saved inference call ordering. Require exact reproduction within the same locked numerical path; any platform-specific discrepancy must be investigated before claiming replay success, not covered by a post-result tolerance increase. Independently recompute projection/phase, architecture ranking, all gate fields, diagnostic eligibility, routes, reports and bootstrap counts from keyed artifacts.

At least one synthetic end-to-end fixture must exercise a genuine accepted regional route, support fallback, gain fallback, unmapped keys and an empty fold. Include tests for threshold equality, missing/all-missing columns, constant targets/NA metrics, state mutation, temporal cutoff, missing/duplicate keys, corrupted state and changed map/source identity. Verify gate-rejected supported L predictions exist and cannot alter G. Do not rerun the entire old scanner/GeoXGB fit pipeline.

Verify both residual architectures initially output exactly zero in train/eval modes on finite synthetic inputs; a nonzero residual target must permit output-weight updates and subsequent hidden-layer data gradients. A zero-residual target must retain zero output under the fixed training procedure. These test initialization/gradient semantics, not an empirically chosen convergence threshold or a new scientific tuning gate.

## 11. Failure and release boundaries

Any fitting exception, nonfinite tensor/loss/prediction, required empty global pool, shape/key mismatch, source/schema/map drift, immutable-parent mutation, or model-store integrity failure stops the affected run and writes INCOMPLETE evidence. Do not hide technical failures behind normal P fallback, alter the seed, reduce epochs, clip bad values, or remove a failed date/candidate. Successful partial artifacts remain partial.

After final user approval, the verified Claude Opus 5.5 1M executor implements and performs synthetic validation, then submits P0 runtime/inventory/timing and frozen code evidence to the user. The user releases the planned scientific run when these prerequisites match the approved contract; discrepancies are reported before dependent work. This supervisor checkpoint is not a new scientific design or a blanket instruction to tune until positive.

Completion includes full three-seed evidence, no unresolved technical failures, replay, a concise results note and updates to the existing meeting/future-direction notes with appropriate limitations. Source/plan commits follow the repository rules. The Trellis audit lifecycle is waived for this task only: native `task.py start`/`archive`, no close or spot audit, and no claim of an audit pass.
