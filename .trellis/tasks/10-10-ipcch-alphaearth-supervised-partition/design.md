# IPCCH AlphaEarth supervised partitions — design for final review

Status: planning, 2026-10-10. D1–D29 are accepted user decisions in `decisions.md`. This consolidated design also makes routine implementation choices explicit for final review; it is not an execution release. No model or extraction has run. Operational cloud limits must be established at the pre-extraction checkpoint in `implement.md`.

## 1. Question, contrasts and scope

Test whether supervised, fixed environmental groups improve population-derived IPCCH forecasts beyond a matched, strong pooled backbone and shared calibration. Evaluate H1, H3, H6 and H12 separately. Any subset of horizons with credible benefit is a useful contribution; all four need not win. Report every horizon and every prespecified comparison, including fallback, losses and inconclusive evidence.

| Arm | Backbone | Additional response model |
|---|---|---|
| B0 | compact296 pooled XGB quartet | none |
| B1 | compact296 + fixed 2017 AlphaEarth128 pooled XGB quartet | none |
| B2 | exactly the fitted B1 and its raw predictions | shared ridge calibration |
| G | exactly the fitted B1 and its raw predictions | jointly fitted shared calibration + shrunk regional deviations |
| Persistence | latest eligible population-derived phase/q3 at row origin | none |

Primary contrast is G−B2. Report G−B1/G−B0, B1−B0 and B2−B1 to expose baseline, feature and calibration contributions. Persistence comparisons use their common available-key subset and show coverage. IPCCH has no expert comparator. A G−B2 gain is the incremental value of this regional-response procedure; it does not establish that AlphaEarth is superior to every possible grouping or capacity-matched nonlinear calibrator. No extra placebo, country-map, neural, oracle-weather or alternative-learner arm is implied.

## 2. Package and reuse boundary

Implement an isolated `IPCCHAlphaEarthExperiment/` package (`ipcch_alphaearth`). Keep configs/schema/runtime/source-provenance in that package and task evidence in this task. Do not alter completed experiment recipes or sibling IPCCH code. Prefer attributed, bounded reuse of existing target/projection/metric, compact feature and model-record helpers, with source SHA and adaptation reasons, to importing historical runners and their gates/cohorts.

Existing reuse anchors:

- P6 `IPCCHGeoXGBExperiment/ipcch_geoxgb/{targets,projection,metrics}.py`: population truth, equal-weight monotone projection, metric/NA semantics.
- Sibling `../IPCCH/src/ipcch/{compact_features,origin_safe,climate2015_features,retained_feature_recipes}.py`: ordered compact feature definitions and origin windows. H1 and the new target cohort require explicit adaptation.
- Yearly `IPCCHYearlyGeoXGBExperiment/ipcch_yearly_xgb/{schedule,engine,report}.py`: scheduled block identities, keyed provenance and reporting patterns. Do not inherit its old regional boosting, gates, historical request pool or model parameters.
- `IPCCHMLflow/` and the local MLflow import/summary specs: completed-result registration, model bundles and same-key dataset identities.

The repository's FEWSNET/ETH-specific horizons, experts, SMOTE and macro-F1-zero-fill conventions do not apply. This experiment's explicit IPCCH contract has precedence over those examples in `local-forecasting-experiments.md`.

## 3. Truth, keys and timing

Canonical key is `(admin_code:int64, target_month:YYYY-MM, H:int)`, with `row_origin = target−H`; preserve country identity separately. No duplicate/null keys or positional row joins. Rebuild population truth with the P6 rules: finite positive population; P1–P4 required; missing P5=0; each share in [0,1]; sum in [0.90,1.10]; normalized cumulative q2–q5. Truth phase is the largest k with the exact normalized cumulative share >=.20, using the existing exact decimal comparison. Report classes 1/2/3/4-or-5 and crisis phase>=3. Reported overall_phase is not truth and cannot gate this cohort.

D29: O is the end of observation/report month O. At a fit, labels U<=fit_origin are eligible; dynamic training inputs are cut off at each training row's own origin, not the later fit origin. Yearly XGB uses all eligible historical labels, including 2014–2017, with unnormalized weight `2**(-(fit_origin−U)/24)`. No 24- or 36-month truncation. Static 2017 AlphaEarth may be attached to old training rows as information available at the later 2018+ fit, never called an honest prediction at a pre-2018 origin.

Origin windows include month O. IDP uses reporting-month<=O; seasons have `end_exclusive <= first day of O+1`. These are observation-time proxies, not historical release/revision certification. No blanket publication lag is invented. Unknown supplier climatology fitting periods, fixed static snapshots, original XGB tuning history and AlphaEarth publication/pretraining history remain disclosed limitations. New supervision, feature construction, preprocessing and selection still obey their enforceable cutoffs.

Keep all canonical valid rows despite unavailable features. Raw/source corruption, duplicate conflicts or missing required files stop preparation; legitimate missing observations remain missing. Save invalid-target reasons as well as the valid ledger. Reproduce the previously observed 42,695 valid rows / 6,224 admins from pinned inputs or stop to explain source drift. No row removal to match old compact tables, which omit 14 canonical target keys.

## 4. Features and numerical environment

B0 ordered compact296 is static29 + ordinary120 + same-month historical-z56 + stress28 + completed-season29 + coordinate/calendar27 + reported-history5 + IDP2. Freeze the source feature-name order (existing baseline list digest `e97388f848e7095bf0535f0cfee1a2d798e0c1863dee843462e60cf6a36f9909`) and explicitly verify the adapted H1 order. Same names do not mean H0 values can stand in for H1. Rebuild row-origin values from the source grids; no future-weather oracle, rich561 expansion, interpolation or new feature/window search.

Reported history5 means the latest three eligible reported phases and their two differences. This input is distinct from population truth. Preserve existing window minimum support, missingness, prior-year-only same-month z, season and IDP rules. Static29 is extracted by admin from the pinned inherited static source with invariance checks; do not fill its largely missing popdensity from a different snapshot. Source files and hashes are listed in `research/source-runtime-availability.md`; stage and freshly rehash complete inputs before use. Legitimate feature NaN reaches XGB natively. Nonfinite source values follow the pinned compact recipe with an audit count; no unrecorded fill/clipping or silent predictor deletion.

Append `ae2017_mean_A00..A63`, then `ae2017_sd_A00..A63`, in that order, to form 424 inputs for B1/B2/G. All use the identical descriptor table. Coverage and source metadata are diagnostics, not additional predictors.

Use existing WSL `/home/swl007007/.venvs/ipcch-geo/bin/python`: Python3.12.3, numpy2.4.4, pandas3.0.3, sklearn1.8.0, xgboost3.2.0, scipy1.17.1. Pin relevant geometry dependencies from this environment. EE aggregation runs separately with `/opt/earthengine/bin/python` / earthengine-api1.7.36, exchanging hashed files; do not install EE into the numerical environment or upgrade either environment. MLflow uses its existing separate venv. Record resolved full package versions and interpreter paths in runtime-lock before execution. Serialize numerical inputs/results with round-trip precision, canonical key order and explicit dtypes.

## 5. AlphaEarth extraction and routing

Source is fixed `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, calendar2017. Expected catalog dataset1.1/model2.1 and processing metadata must be checked on the actual assets, with start/end timestamps, asset IDs, native CRS/affine grid and metadata saved. Do not mix years/versions or silently substitute assets. Different actual versions are a checkpoint issue before extraction, not a score-dependent choice.

Use the existing 6,227 repaired geometries, whose five component hashes match P6. Pin full IDs/order and geometry bytes. Their valid topology and complete key coverage do not verify upstream administrative matching or boundary vintage. Do not rebuild geometry or use nearest-neighbour/point substitution.

In EE compute each piece on its source native 10 m grid with an explicit CRS/transform and `bestEffort=false`. Freeze deterministic source-footprint ownership in lexicographic asset-ID order: intersect each footprint with the admin and subtract previously assigned footprint area. This prevents full-tile double counting without reprojecting all embeddings to a new common grid. First-owner masked areas remain masked; no unrecorded filling from another overlapping source. Record ownership and masked-area consequences in pilot diagnostics. Adjacent/source-edge intersection weights use EE's documented approximate fractional arithmetic; do not call them exact polygon intersections. Work-piece batching must preserve this fixed ownership and moments.

Use the common finite-valid mask of A00–A63. Convert values/areas to float64 before products, squaring and accumulation. Retain valid covered area A, first moments S_j and second moments Q_j, source-piece IDs and full polygon area in square metres. Require finite positive polygon area and finite nonnegative A. Completed extraction with A=0 is genuine no coverage: skip division, retain missing moments and follow the low-coverage route. For A>0 compute `mean_j=S_j/A`, `var_j=Q_j/A−mean_j**2`, `sd_j=sqrt(var_j)`. A small negative variance within 1e-12 is float64 cancellation and may be zeroed with its count; more negative/nonfinite results are technical failures. No averaging of per-piece SDs, population/cropland weighting or mean-vector renormalization. Preserve raw valid-area/full-polygon ratios and inspect numerical area consistency in the pilot; no score-dependent clipping/coverage correction.

A descriptor is usable only for raw coverage>=.80 and 128 finite moments. Denominator is the entire frozen polygon, including its water area. Low/no-coverage admins get 128 NaNs, never a zero embedding; they remain in every otherwise eligible cohort. They do not route to a partition leaf and do not count toward regional support. Their honest OOT rows still enter shared correction/scaling, with regional deviation fixed zero; G uses its own shared beta, which need not equal B2. Unsupported routed leaves have the same shared-response behavior. Fewer than two supported leaves or an unselected partition means exact whole-model B2 reuse.

Keep sufficient-statistic tables, coverage/route ledgers, masks/ownership recipe, source inventory and all export-completion records; no full raw-raster archive. These support offline reconstruction of descriptors and modeling, while raw-pixel extraction replay remains dependent on EE/provider assets. Failed exports, missing pieces or corrupt sources must never masquerade as legitimate low coverage. Cloud destination, account/quota and pilot/full-volume resource limits are checked at the explicit execution checkpoint; scientific planning alone is not cloud spending authorization.

## 6. Annual fitting and honest residual providers

Use `research/annual-fit-calendar.md` as the full block table. There are 19 historical B1 blocks through target2022, four accepted 2023 bridge blocks, 11 main blocks per backbone and four 2026 Jan–Apr supplement blocks per backbone. B0 additionally needs eight development-year blocks, not all early OOT history. Fresh-fit ceiling: B1 38 + B0 23 = **61 quartets / 244 scalar XGB fits**. Reuse the same B1 for correction candidates; no local XGB continuation and no additional per-leaf backbone.

Main origins>=2023-01, targets<=2025-12. First main target is 2023-02/04/07 for H1/3/6 and 2024-01 for H12; every first main fit is at 2023-01. First historical origin is 2018-01 after the fixed annual representation ends. Anchors use scheduled months, including no-target months, not the first observed label. Main expected keys are 17322/16919/16413/14087; supplement 4092 per H. Preserve all 122 main scheduled folds, including three no-valid-target months per H, plus 16 supplement folds; never score empty folds as zeros.

One B1 OOT provider per H/admin/target: historical blocks through2022, only the 22 H/month pre-main bridge slots in2023, then the actual formal B1 provider from the main boundary onward. Store fit origin separately from row origin. Every provider's target must be absent from its base fitting keys and after its fit origin. No overlapping alternative 2023 providers or duplicate residual weights. Bridge labels may enter an eligible subsequent parameter update, never pre-2023 map learning/selection. At the first main fit, January2023 residual labels may already be eligible under D29; this is an annual coefficient update, not a map update.

If an actual backbone fitting pool is empty, preserve a technical/unsupported record and stop affected experiment fitting for review; do not invent labels, silently skip evaluation rows or train an in-sample residual substitute. Constant regression targets are valid. A correction pool with zero eligible OOT rows yields a declared identity correction (B2=B1, G=B2), while nonempty but region-sparse pools still fit the shared B2. These distinctions must be tested and recorded.

## 7. Fixed pooled quartet recipe

| Parameter | q2/q4/q5 | q3 |
|---|---:|---:|
| max_depth | 11 | 9 |
| n_estimators | 200 | 200 |
| learning_rate | .1 | .1 |
| subsample | 1 | .5 |
| colsample_bytree | .5 | .7 |
| gamma | 0 | .1 |
| min_child_weight | 0 | 0 |
| reg_alpha / reg_lambda | 0 / 1 | 0 / 1 |
| objective | reg:squarederror | reg:squarederror |
| seed / n_jobs | 42 / 16 | 42 / 16 |

No early stopping, XGB search or target-specific new choices. Use CPU hist (the inherited recipe's resolved updater) and record all effective parameters including base-score resolution. Weights are unnormalized D7 weights. B0 and B1 differ in the 128 appended inputs only; B2/G do not refit the backbone. Save each scalar booster and its complete fitting identity. Old fitted results can be reused only with exact source/keys/targets/features/weights/config/runtime/fit-origin compatibility; the execution budget assumes fresh matched B0/B1. Incompatible old compact scores remain labelled historical references, never fill missing matched predictions.

## 8. Shared and regional correction

For each q, use raw unprojected B1 prediction p and three existing compact inputs: `prcp_anom_month_ensmean__ma3_at_origin`, `tmean_anom_month_ensmean__ma3_at_origin`, `event_count_violence__ma3_at_origin`. Fit on eligible historical OOT residuals r=q−p, not new-base in-sample predictions. Prediction rows use their current B1 p and current origin inputs.

Train-only weighted means/population SD are common to B2/G and all candidate maps in that fit. Standardize four core inputs; fill missing dynamic values with training means (zero standardized) and add three unscaled missing flags. Intercept + four cores + three flags = eight slots per q. Inactive constant/all-missing columns and constant flags have zero coefficients and remain zero at inference. Nonfinite p is an error. No region-specific scaling or additional transformations.

B2: `p + v^T beta_B2`. G: `p + v^T(beta_G+u_R)`, with `sum_R pi_R*u_R=0` coefficient-wise. pi uses each leaf's fraction of mapped fitting weight, including leaves whose deviations are fixed zero. Missing-AE rows have u=0 and remain in the shared loss/total W. Per-q objective:

`sum_i (w_i/W)*(r_i−v_i^T(beta+u_R(i)))**2 + .01*||beta_nonintercept||**2 + lambda_region*sum_R||u_R||**2`.

Quartet split objective is the equal mean of the four minimized objectives. Shared intercept is unpenalized; every regional slot, including intercept, is penalized. Regional penalty is an unweighted sum over leaves. Lambdas {.01,.1,1} are common across q within each recipe/H. No-partition result reuses B2 exactly. beta_G may differ from beta_B2; coefficient centering does not impose mean-zero prediction corrections.

A leaf needs weighted row `n_eff=(sum w)**2/sum(w**2)` >=200, >=50 distinct admins and >=6 target months using actual eligible mapped OOT rows. Both proposed children must pass. Annual refits never redraw the map; unsupported leaves have u=0. Fewer than two supported leaves means whole-model B2 reuse. No regional F1 gate or synthetic augmentation.

## 9. Deterministic supervised partition search

The following bounded implementation choices complete D8/D12/D14/D19; they are presented for final review, not a further hyperparameter grid.

- Axis-aligned binary tree over the 128 frozen descriptors only. One map per H, shared across q. Whole admins route by saved thresholds; disconnected/cross-country groups are allowed. No adjacency, country constraints, K-means, learned edge weights or post-hoc component splitting.
- Start at one mapped root with B2-equivalent response. Only fitting admins with eligible OOT observations determine thresholds; unlabeled/future admins are routed afterwards, not used for candidate quantiles or support.
- At each splittable leaf, each nonconstant descriptor supplies at most nine cuts. Sort one value per distinct fitting admin; for j=1..9 take rank ceil(j*n_admin/10), use that observed value as the <= left threshold, deduplicate and remove maximum/empty cuts. Tie admin order is numeric admin_code. No interpolated/adaptive cut grid or validation-driven threshold proposal.
- For every supported cut, refit the **whole joint shared/regional model** and compare its quartet penalized objective to the current map. Do not score a cut by mean residual or isolated child OLS: centering and shared coefficients affect other leaves. Choose largest reduction; gain must exceed `1e-12*max(1,abs(current_objective))`. Gains within that tolerance tie by node path, descriptor order then threshold order. Node IDs r/r0/r1 preserve leading path zeros as text.
- Best-first growth; root depth0, maximum depth3. For each lambda grow once up to eight leaves, recording states at budgets2/4/8; an early terminal state is reused for any later ceiling. No lookahead, pruning or alternate search seeds. If no beneficial root split, that recipe is B2.
- Cache per-admin weighted sufficient statistics, not score results whose joint fit changes after another split. Stable constrained ridge solves must satisfy stationarity/centering checks; failed numerical solves are technical errors, not unsupported leaves or permission to adjust penalties. Tests compare against a direct constrained reference solver on synthetic data.

At most128*9 candidate thresholds per splittable leaf. A loose bound for one eight-leaf growth path is1152*(1+...+7)=32256 proposed cuts. Development uses eight H/year contexts * three lambda paths; final learning uses at most four selected-lambda paths, giving at most903168 quartet candidate evaluations (3612672 per-q candidate solves), before support/depth pruning. This is a conservative search ceiling, not a timing estimate; P0 benchmarks the intended sufficient-statistic implementation. Ridge work is separate from the244 scalar XGB ceiling.

## 10. Forward selection, freeze and annual updates

Reconstruct each of nine recipes at each H's earliest safe origin for development targets2021 and2022. Scaling, maps and correction coefficients use only historical residual labels available by that origin. Evaluate the year with the same current B1 backbone and origin-updated row inputs. Require both annual G−B2 crisis-F1 gains>=0, their equal-year mean>0, and D23 guards in **each** year. Undefined required scores cannot qualify.

Rank eligible recipes by exact-fraction mean crisis-F1 gain, then smaller sum of realized leaf counts across both development maps, smaller maximum realized depth, smaller leaf ceiling, stronger lambda, and recipe ID. Preserve the full nine-recipe/year ledger even if maps duplicate. No eligible recipe selects exact B2 for that H.

Using only the selected recipe and residual labels available through2022-12, relearn that H's final map and freeze tree, routable-admin membership, source descriptors and recipe hashes. A selected recipe can still produce no final split; record B2 fallback, never reopen selection. After freeze, fit shared/regional coefficients annually on all eligible saved residual providers with the same half-life/support/scaling rules. Annual scaling can update on fitting rows; map thresholds/descriptors cannot. Never use2023+ outcomes to redraw maps, choose recipes or extend the grid. A new admin with valid fixed descriptors follows the frozen tree regardless of whether it appeared in development; no new map fitting is needed.

## 11. Projection, metrics and per-H interpretation

For every model apply equal-weight decreasing isotonic projection, then clip to [0,1], so `1>=q2>=q3>=q4>=q5>=0`. Do not clip before the isotonic step. Decode unrounded projected shares with inclusive .20. Preserve raw quartet, projected quartet and decoded phase; no threshold tuning. Persistence carries its own latest population-derived phase/q3 and source month<=row origin, without a new age cutoff.

Report seven classification metrics (four-class accuracy/macroF1; binary accuracy/precision/recall/F1/F2), projected q3 R² and diagnostic raw q3 R². Use unweighted pooled confusion counts per requested cohort; fitting weights never enter evaluation. Fixed four-class axis stays1/2/3/4-or-5. A class absent in both truth and prediction has undefined F1 and makes macroF1 undefined; do not substitute sklearn zero-fill. Preserve legitimate zeros and negative R². Constant/insufficient truth makes R² undefined. Every NA has a reason; undefined required guards never pass. Existing P6 `metrics.py:94-180` is the metric-semantic reference.

For each H, a main-period protected stable-benefit finding requires all:

1. Main same-key crisis `F1(G)−F1(B2) >= .005`.
2. D24 corrected paired lower bound strictly>0.
3. Each actually scheduled/nonempty main target year has nonnegative G−B2 F1; H12 has no2023 panel.
4. G F1>=B1 and matched B0. Against each of B1 and B2, each of the other six classification scores may fall by at most .01; projected q3 R² may fall by at most .02. Required undefined guards are inconclusive. Raw R² cannot substitute.

Apply item4 in each development year as well, before recipe ranking. Main year-specific secondary scores are diagnostics. Persistence has no added mandatory superiority/significance gate; show all metrics and model−persistence deltas on identical E_persist keys with coverage. Guard tolerances are point-estimate preferences, not formal noninferiority/equivalence tests.

Evaluate all four H, retaining D24 multiplicity while making a separate decision for each. Any qualifying subset is a valuable finding; no all-four-success gate. Report effect size, corrected bound, annual signs and each guard separately, distinguishing negligible gain, imprecision, reversal, metric trade-off and undefined evidence. A nonsignificant H does not prove zero effect; significance at one H and not another does not itself test a between-H difference. Supplement2026 is point/descriptive evidence, cannot rescue a main failure or tune the design. The historical period has been examined before: these remain retrospective findings, not untouched prospective confirmation.

## 12. Paired uncertainty

For each H's main E_all G/B2 pair, sort its observed countries and sample K countries K times with replacement for each of20,000 draws, using numpy.random.default_rng(seed42). Each country's complete cohort shares its multiplicity, applied identically to both arms. Compute F1 from aggregate TP/FP/FN counts, not equal-country mean F1. Save per-country counts, country multiplicities, both arm F1s and deltas for all draws.

Fixed family has four H even if an H falls back to B2. Use linear1.25th percentile as its nominal one-sided98.75% lower bound (Bonferroni .05/4). This is approximate bootstrap family coverage under country-sampling assumptions, not exact finite-sample coverage. Country sets need not match across H and H independence is not assumed. K<2, undefined point or any undefined replicate means inconclusive bound; never drop/redraw to obtain a pass. B2 identity fallback yields exactly zero delta where F1 is defined.

Uncorrected95% intervals may be shown only as labelled descriptive context; they cannot replace the corrected bound. There is no first-experiment synchronized time-block bootstrap or retraining bootstrap. Inference conditions on the fitted predictions and observed calendar, not future years, unseen countries, model selection or cross-country shared shocks. Keep country/time concentration diagnostics (currently52–53 countries,2–3 years, approximately40% rows in three countries and57–70% in2025).

## 13. Identity, artifacts and replay

Use an immutable new run directory on the WSL filesystem outside Dropbox, such as `~/ipcch-alphaearth-runs/<run-id>`. Stage all consumed data/manifests there with hashes. Do not touch old run directories or live artifacts. Create a new run for changed scientific inputs/configs; interrupted attempts retain complete/incomplete evidence and partial outputs, not silently overwrite them.

Freeze contract/schema, numerical/extraction runtime, source manifest, source-code inventory and prepared-input inventory before each released stage. A fit identity binds arm/H/block/fit origin, ordered fitting keys and row refs, target and feature/weight digests with dtype/shape, schema, resolved parameters, runtime and fitting-source identity. B1 provider identity is also bound in each correction fit/map-training record. Record the actual Y/X/weight slices used, not only request labels. Existing corrupt/conflicting model records stop rather than refit under the same identity. Store quartet boosters, correction coefficients/transforms and membership/thresholds as loadable local artifacts; MLflow catalog entries can remain external bundles.

Required retained outputs:

- Full target QC ledger; key/cohort/schedule ledgers; raw source/staged hashes; ordered296/424 schemas and prepared arrays; feature-missingness/origin audits.
- EE asset/geometry/projection/ownership manifest; planned/completed source-piece exports; sufficient statistics; descriptors; raw coverage and route reasons. Record pilot resource evidence and the actual authorized full-export envelope.
- Request/reuse/fit ledger and all244-or-fewer scalar model identities; distinct historical-provider ledger. Do not represent ridge searches as additional XGB fits.
- For each development context: eligible residual-key refs, scaling, candidate support/objective/split decisions, map states, recipe/year predictions and exact metric/guard/selection records. Retain evaluated cut scores and deterministic candidate ordering sufficient to reconstruct selection; don't serialize a full model for every rejected cut.
- Frozen maps/recipes, all annual transforms/coefficients and support/fallback records; keyed raw/projected/phase predictions for B0/B1/B2/G and persistence provenance.
- Full report JSON/CSV; country/year/month and AE coverage diagnostics; paired draws; zero-fit replay result; final artifact inventory after replay. Machine-readable raw metric precision, denominators and NA reasons accompany formatted tables.

Independent replay reconstructs target/cohort/time eligibility, model fitting identities and unique provider refs, re-predicts all saved OOT/evaluation/development predictions from saved backbones/corrections, independently checks projection/decoding, selection/guards and metric/bootstrap results. It checks map routing and re-evaluates winning split objectives/support against saved sufficient statistics and candidate-score ordering. A small synthetic reference exhaustively checks search; full production replay need not rerun millions of rejected ridge fits. This limitation is disclosed; replay is zero-XGB-fit and does not claim raw-EE extraction verification. Offline reaggregation must reproduce descriptor tables from saved moments; external pixel reproduction is outside D26.

Technical errors (missing pieces, changed hashes, duplicate keys, wrong temporal ancestry, nonfinite predictions or failed solves) stop the stage with durable context and an INCOMPLETE marker. Scientific null results, sparse regional support and documented product-mask gaps take their specified reporting/fallback paths. No technical exception is silently converted into B2 success or missing data.

## 14. Report, MLflow and delivery

Produce a concise Chinese group-meeting discussion note in the existing `docs/notes/` convention: four-H comparison table, confidence/guard evidence, maps with coverage, per-year effects, one explanation of B0/B1/B2/G, relation to earlier null results and explicit limitations. Keep retrospective explanation separate from tested mechanism; no claim that successful partitions are causal environmental mechanisms. Preserve all horizons and reference whether each old comparison is compatible reuse or historical context. No PowerPoint is required by this task.

After formal outputs pass replay and receive result acceptance, add this completed family through `IPCCHMLflow` plan→import→verify and refresh Summary through plan→inventory→backup→apply→verify. Use the existing nine metric names and period/cohort tags, register external composite model bundles and row-free dataset Inputs, and compare only identical dataset key/truth digests. Avoid logging every diagnostic cell as a new metric or inflating the main UI. Full diagnostic files remain artifacts. Freeze expected records/model versions/dataset descriptors in the import plan before live writes. Do not alter or rescore historical families. MLflow adapter/config changes get their own focused existing-suite checks; no new tracking service.

A scientific negative result can still satisfy every delivery criterion. No branch push/merge, external communication or lifecycle close is implied by planning approval; follow the authorized execution/handoff and repository lifecycle at the relevant checkpoint. Partial/full extraction and fitting each require their concrete release evidence described in `implement.md`.

## 15. Known limitations and stopping boundaries

The accepted study has reviewed historical evaluation years; observation/revision/product/pretraining and inherited-tuning limitations; unverified administrative matching/vintage; a single2017 representation; approximate boundary weights; limited admin summaries; finite tree/regularization search; and no automatic guarantee of a strong pooled model or transferable regional signal. Report these alongside the actual scores.

P0 must surface source drift, incomplete H1 reconstruction, unsupported earliest fits or an impractical candidate-search benchmark before project-data fits. P1 must surface unverified asset version, overlap/coverage defects, missing export pieces or an unacceptable resource envelope before bulk extraction/modeling. Do not change scientific thresholds, feature sets, sampling years, model grids or evaluation keys to work around a negative result or execution cost. Changes affecting those contracts return to a concrete spec review; routine implementation fixes retain the frozen scientific recipe and evidence.
