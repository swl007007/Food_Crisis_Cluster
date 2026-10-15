# Planning progress

- 2026-10-10: User authorized task creation, docs-grounded grilling, then spec. Task created in `planning`; no implementation/audit start.
- Official AlphaEarth catalog reviewed; read-only local baseline and geometry facts recorded in `research/`.
- PRD seeded with inherited population-derived truth, partition-gain objective, matched-input controls and unresolved decisions.
- D1 accepted (user: “采用”): fixed AlphaEarth representation and supervised partition frozen after learning; no annual representation/partition updates in the first experiment. Predictor refitting subsequently decided in D7.
- D2 accepted (user: “采用”): retrospective representation/mechanism research, with satellite-observation and supervised-learning time constraints retained; no claim of historical product availability or historical operational performance.
- D3 accepted (user: “采用”): fixed 2017 annual AlphaEarth representation; no score-based year selection. Exact asset version and coverage remain to be verified.
- Read-only count probe on the saved P6 canonical valid ledger and keyed horizon views: 2018+ origins with targets through 2022 supply 14,195–15,708 candidate supervision keys by H; these are row counts, not independent samples or confirmed fit support. Counts and source ledger SHA in research/temporal-support.md.
- D4 accepted (user: “采用”): learn/select/freeze partitions by 2022-12; main evaluation origins >=2023-01 and targets <=2025-12; 2026 targets separate. Inner splitting and predictive-model updating subsequently decided in D6/D7.
- D5 accepted (user: “采用”): retain pre-2018 outcomes for later fits and attach the frozen 2017 descriptor as fit-time geographic metadata; dynamic features remain row-origin-safe. These old fitted rows are not honest partition-supervision residuals.
- Additional no-fit counts for proposed target-year 2021/2022 validation: 3,246/4,153 validation keys; per-H earlier candidate supervision counted at each block's earliest origin. H12 has 3,683/7,793 before feature/fit-support filtering. See research/temporal-support.md.
- D6 accepted (user: “采用”): two forward validation blocks for target years 2021/2022; reconstruct candidate learned maps/corrections within each block's safe training boundary, select the recipe, then relearn/freeze the final map through end-2022.
- Coordinator read the full neighboring compact design. Its annual expanding-history, 24-month half-life fitter is a reusable protocol; its reported-phase truth/raw decoder/H0 are separate, not automatically adopted.
- D7 accepted (user: “采用”): annual target-year fits, all eligible history, 24-month half-life, consistent development/formal protocol. Pooled/correction parameters update together; final map fixed; row inputs remain origin-aligned. First partial years use the earliest eligible scheduled origin.
- D8 accepted (user: “其他的赞同”): strong pooled XGB, shallow supervised AlphaEarth partition tree, shrunk regional intercept/slope corrections, and unpartitioned calibration control.
- D9 explicit user addition: include pooled XGB without AlphaEarth; previously trained results may be reused, with provenance and comparability clearly explained. Four model roles B0/B1/B2/G recorded; mismatched historical results cannot establish an isolated partition/feature effect.
- D10 accepted (user: “采用”): partitions may include disconnected and cross-country admins; no geographic-connectivity/country-boundary constraint in the first experiment. Group maps and composition remain reportable diagnostics.
- D11 accepted (user: “采用”): H1/3/6/12, one map per horizon shared across q2–q5, separate target regressions/correction coefficients, four final maps. Cross-H outcome views do not count as independent samples.
- D12 accepted (user: “采用”): equal-weight quartet squared error plus shrinkage for candidate generation; choose recipes by forward-validation crisis F1 after common bounded monotone projection and fixed unrounded >=.20 decoding. No target-weight or threshold search.
- D13 accepted (user: “采用”): both development-year G−B2 crisis F1 gains >=0 and their equal-year mean >0; select the largest eligible mean, simpler map on ties; otherwise retain B2 for that H. This is selection, not a significance claim or a per-region deployment gate.
- D14 accepted (user: “采用”): partition-tree maximum depth 3 (root 0), leaf-budget candidates 2/4/8, early stop allowed and explicit B2 fallback. This does not cap the pooled XGB trees. Support and shrinkage remain open.
- D15 accepted (user: “采用”): common compact296 non-oracle feature base, retaining five safe reported-phase history inputs with population-derived truth; no rich561 history expansion. B0 has 296 columns; B1/B2/G add identical fixed AlphaEarth descriptors. Explicit H1 adaptation; old truth/QC/cohort/decoder/runtime not inherited. Actual row-level coverage remains unverified.
- D16 accepted (user: “采用”): admin area-weighted means64 + standard deviations64, fixed 2017 valid pixels; no human-population/cropland weights or mean-vector renormalization. B1/B2/G share 424 pooled inputs. Exact extraction/coverage rules remain open; these 128 descriptors do not become regional slope inputs by implication.
- D17 accepted (user: “采用”): each target's own raw pooled prediction plus existing origin-aligned three-month precipitation anomaly, temperature anomaly and violent-event count; identical core inputs for shared B2 and regional G corrections. No correction-feature/window search. Scaling, missingness and penalty units remain open.
- D18 accepted (user: “采用”): pooled XGB frozen during map/correction fitting; jointly estimate shared calibration and regional deviations, with training-weighted-zero-mean coefficient deviations and shrinkage. No-partition result exactly reuses B2; G's shared coefficients may differ from separately fitted B2, which must be disclosed.
- D19 accepted (user: “采用”): train-only weighted scaling/mean imputation plus three missing indicators; weighted-mean squared-error ridge, shared lambda=.01, regional lambda in {.01,.1,1}, shared across q2-q5 within H. Leaf budgets {2,4,8} yield nine recipes per H plus B2. No post-evaluation grid expansion. Exact penalty normalization and inactive-column handling recorded in research/partition-mechanism-options.md.
- Two read-only scouts checked old XGB support and recomputed candidate training support by H/year. All eight prior row counts reproduced; earliest H12 has 2,065 admins across 12 observed target months, with 71.46% of rows in three months. Coordinator spot-checked these H12 facts and hashed the four source keyed tables; evidence saved in research/temporal-support.md. No fit/feature build.
- D20 accepted (user: “采用”): per-leaf support weighted row n_eff>=200, >=50 admins, >=6 target months; both children must pass when learning. Sparse annual-refit leaves keep their map but have zero regional deviations (shared G response only); fewer than two supported leaves means exact B2 reuse. No regional F1 gate or support-threshold grid. Thresholds are not empirical power guarantees.
- Read-only scout traced sibling compact configs, fit_model, and saved metadata; coordinator checked both config JSONs and effective metadata. Existing q2/q4/q5 depth11 and q3 depth9, 200 rounds each; original tuning years/keys remain unknown. Facts and provenance limits saved in research/pooled-xgb-recipe.md.
- D21 accepted (user: “采用”): fixed inherited compact target-specific XGB recipes for B0/B1 at all four H, shared B1 backbone for B2/G, no extra base search. Explicitly disclose unknown original tuning years/keys; new fit-time compliance is not certification of the inherited selection history. Runtime and old-result compatibility still require freezing/verification.
- D22 accepted (user: “采用”): per H, main G−B2 crisis F1 >=.005, corrected paired lower bound >0 for the four-H family, and nonnegative gain in each existing main target-year panel. Claims limited to qualifying H; all four required for all-horizon language. Supplementary 2026 cannot rescue main failure. Concrete resampling and secondary guards remain open.
- D23 accepted (user: “采用”): in each development year before D13 selection and each H's aggregate main evaluation, crisis F1 G>=B1 and strictly matched B0; versus each of B1/B2, each of six secondary classification scores may fall by at most .01 and projected q3 R² by at most .02. Undefined required metrics cannot pass. Raw q3 R² remains diagnostic; persistence is reported on common available keys with coverage, without a new significance gate. Incompatible old B0 results require a matched D21 fit after execution authorization. These are point tolerances, not formal noninferiority tests.
- Read-only scouts traced the paired country-bootstrap helper and counted main country/time support; coordinator checked the helper and independently reproduced H12 counts/hash. Main cohorts have 53/53/53/52 countries, 3/3/3/2 target years, roughly 40% of rows in the largest three countries, and 57–70% in 2025. Evidence saved in research/temporal-support.md and research/evaluation-criteria.md; no predictions scored or bootstrap run.
- D24 accepted (user: “采用”): paired whole-country bootstrap, 20,000 draws/seed42, four-H Bonferroni one-sided lower bounds at the 1.25th percentile, retain all undefined draws as inconclusive, plus the accepted annual sign check. Inference remains conditional on the observed calendar/frozen predictions; synchronous time-block sensitivity is outside first-experiment scope. The procedure does not cover training/selection uncertainty, shared cross-country shocks or future-year stability.
- Read-only geometry scout found all 42,695 valid rows (6,224 admins) and all main H cohorts mapped to unique geometry IDs; five shapefile component hashes match inputs. Coordinator checked the saved valid-geometry preflight and unresolved administrative-identity/vintage caveat. Official product coverage includes land and shallow water, not guaranteed full polygons. No AlphaEarth assets/pixels were read; evidence in research/local-reuse-facts.md and alphaearth-documentation.md.
- D25 accepted (user: “80%”): common 64-band valid-area coverage >=80% of full polygon to use 128 moments; below threshold retain all cohort rows, use native-NaN AlphaEarth inputs and zero regional deviation/shared G response. No silent extraction-error fallback, subset-based rescue or coverage-based model selection. The 90% proposal was not selected. Details and caveats in research/alphaearth-documentation.md.
- Official EE regional-reduction and table-export docs checked read-only: native grid must be explicit; polygon/pixel overlap weights are approximate at 1/256 resolution; bestEffort can change scale; table export does not require downloading rasters. Evidence and proposed D26 in research/extraction-options.md.
- D26 accepted (user: “采用”): native-10m EE aggregation, preserve sufficient-statistic tables/manifests locally without a full raw-raster archive, explicitly accepting EE boundary approximation and provider-dependent extraction replay. Local pixel extraction/archive not selected. Resource/destination/pilot details remain to freeze before execution; this planning choice does not authorize cloud operations.
- Three read-only scouts checked time-availability code, actual source/runtime metadata and annual calendar gaps; coordinator checked inclusive monthly fit masks, compact vintage limitations/runtime gate and existing yearly anchors. Existing compact omits 14 canonical target keys; H1 requires adaptation; ipcch-geo matches its numerical runtime while a separate EE interpreter is available. Facts saved in research/source-runtime-availability.md. No matrices, fits, installs or cloud calls.
- D27 accepted (user: “采用”): fill the 22 H/month slots for target2023 whose origins precede 2023-01 with four honest B1 quartets; use these only for later eligible correction supervision, not map learning/main evaluation. Reuse formal B1 providers after the boundary, one provider per key. Total fresh backbone ceiling is 61 quartets / 244 scalar fits; exact source/support identity enumeration remains before fitting. See research/annual-fit-calendar.md.
- D28 accepted (user explicit correction): evaluate and report each H separately; benefit in only some H is an important contribution, without an all-four-win experiment-level gate. Revise D22 interpretation and report wording accordingly. Keep each H's D22/D23 tests and D24 four-H multiplicity correction; separate evaluation is not statistical independence. Report all favorable, negative and inconclusive H honestly.
- D29 accepted (user: “采用”): observation/report month-end proxy, inclusive label/row-origin cutoffs and completed-season boundary, without an invented blanket lag. Historical releases/revisions/static snapshots and supplier climatology remain unverified; see research/source-runtime-availability.md.
- Consolidation: preserved D1–D29 in decisions.md; drafted design.md and implement.md with deterministic search/ties, numerical environments, source/feature reconstruction, annual providers, per-H inference and staged resource checkpoints. These routine implementation choices are part of final review, not additional accepted scientific decisions or execution release.
- Planning consistency check complete: two read-only checks found no contract/budget conflict; corrected one inclusive-origin wording ambiguity. PRD convergence preserved all 29 decision paragraphs exactly in decisions.md. Native context validation passed (11 implement / 13 check entries). Evidence and verification limits: planning-check.md.
- Ready for final review of the consolidated PRD/design/implement. No downloads, EE exports, implementation, project-data fitting, lifecycle start or live MLflow writes. Task remains planning; final review and explicit execution scope are still required. Proposed first release is P0 offline implementation/readiness; cloud pilot/full extraction and real fitting retain the concrete later checkpoints in implement.md.

# Execution progress

- 2026-10-10: User activated the execution goal (execution-handoff.md). Executor Claude Opus 5.5 /1M session `3546490e-8e08-43f2-9f1d-9c72bccda1e8` in Herdr w14:p2 (terminal term_65d7ebc256c188), registered by supervisor Codex w14:p1 (executor-identity.json). Release: P0 offline implementation/readiness only; no EE contact or project-data fitting before the P0 checkpoint release. Branch `ipcch-alphaearth-supervised-partition` created from main 38adbda; task specification committed as the lifecycle base before audited start.
- 2026-10-10 USER CLOUD PREAUTH (relayed by supervisor w14:p1, recorded verbatim): "预授权云端操作，但只授权必要的EE以及最多GCS的查询和下载，不涉及其他IAM。" Scope as bounded by the supervisor: necessary EE query/extraction/download and, at most, GCS queries/downloads using existing identity/resources. Not authorized: IAM changes, billing/API enablement, new projects/buckets, GCS uploads/exports/writes. If an export destination needs a cloud write, report the exact requirement before doing it. Effective after the P0 checkpoint; P0 completion first.
- 2026-10-10 P0 complete (offline, no EE/GCS, no project-data fit). Isolated package `IPCCHAlphaEarthExperiment/` (45 files, evidence/p0/p0-source-inventory.sha256). Developer run `~/ipcch-alphaearth-runs/p0-local-20261010b`: 21 staged/rehashed inputs; 42,695/6,224 canonical keys identical to P6; compact296 H3/6/12 equal to inherited tables on 42,681 shared keys; explicit H1 rebuild; direct value-at-origin and post-cutoff perturbation checks pass; cohorts/calendar/fit ceiling 61/244 confirmed; runs a/b byte-identical. 71 synthetic tests pass (p0-unittest-final.log). Benchmarks: XGB ~1.9 h estimate, search 8-leaf path 3.98 s. All supervisor P0 samples repaired and closed at bounded scope (receipts in evidence/p0). Summary: evidence/p0/P0-evidence.md. Awaiting supervisor P0 checkpoint release before any EE metadata query.
- 2026-10-10 P0 ACCEPTED at f000113 (supervisor receipt in evidence/p0). P1-METADATA-20261010 executed: existing project manuel-remote-sensing; 14 of 20 high-level calls (2 failed server histograms: user memory limit; 1 empty listing from the exclusive endTime filter); 1,683 study-area 2017 assets listed, all MODEL 2.1 / DATASET 1.1 / PROCESSING 1.02, 64 DOUBLE bands, 10 m UTM (39 zones), 16,384^2 tiles. Local ownership plan: 17,276 pieces, 4.67e11 px. P0 helper defect fixed (sliver negative geodesic area treated as numerical zero within 1e-3 m2) with test; full suite 72 OK. Staged pilot proposal: evidence/p1/P1-metadata-and-pilot-proposal.md. Supervisor released P1-PILOT-A-20261010 (two requests only).
- 2026-10-10 P1-PILOT-A executed as released (2 of 6 calls): init ok; admin 3447 reduction rejected server-side (maxPixels counts band-pixels: 66,040,811 = 511,944 native px x 129 bands > 2e6); fixture not attempted; no moments. Wrapper size-rejection classification fixed offline with tests. Result: evidence/p1/pilot-a-20261010/PILOT-A-result.md; awaiting supervisor decision on a band-pixel cap for the same two requests.
- 2026-10-10 P1-PILOT-A2 executed (3 of 6 calls; cumulative pilot calls 5): admin 3447 usable (coverage 0.99998, 511,951 px, finite 128 moments); open Red Sea fixture returned exact zeros (not nulls) -> genuine no coverage; EE pixel-area sum = local geodesic area within 0.0016%; positive-scaleY native UTM handled. Timing 7.87 s / 5.1e5 px projects median pieces beyond the 120 s deadline: B requires a measured-envelope review. Result: evidence/p1/pilot-a2-20261010/PILOT-A2-result.md.
- 2026-10-10 P1-PILOT-B1 executed (6 of 8 calls; cumulative pilot calls 11): Cabo Verde coverage 0.99999; all sums finite. Owner valid 60.8% vs other-zone tile valid 41.0% over the same 223.55 km2 Somalia geometry: possible first-owner coverage loss bounded 0..87.6 km2, NOT established; a mask-intersection diagnostic is proposed before any ownership decision. Timing ~2.3 s + 4-6e-6 s/valid px. Result: evidence/p1/pilot-b1-20261010/PILOT-B1-result.md.
- 2026-10-10 Ownership mask diagnostic executed (3 of 4 calls): on the frozen admin-1722 piece donor_only = 39.2%, both = 1.75%, owner_only = 59.0%, neither ~0 on both native grids (totals within 0.004% of local area; own-grid valid areas equal B1 exactly). First-owner discards 39.2% otherwise valid coverage on this sampled piece (extent over cohort unknown) -> ownership rule needs a spec decision before bulk extraction. Next-decision outline (ownership options, batching/scaling check, existing-bucket export) in evidence/p1/NEXT-DECISION-OUTLINE.md.
- 2026-10-10 D30 accepted (user “确认”; release ipcch-alphaearth-C-amendment-20261010): extraction ownership amended to one fixed EPSG:6933 10 m analysis grid with ascending-ID first-valid whole-64-band vectors (nearest), mutually exclusive cells; estimand amendment explicit; denominator, .80, year, moments and geometry unchanged; edge-convention mismatch recorded separately. Spec amended (decisions/design/PRD/implement) before code.
- 2026-10-10 D30 implemented (smallest connected change): contract analysis_grid block pinned; extraction Lattice/windows/plan/recipe binding; descriptor_table refuses non-D30 plans; new scripts/ee_pilot_c.py (mask before descending-id mosaic, per-rank area check). Tests 88 OK; supervisor focused checks 14 PASS. Bounded C pilot manifest prepared (10 requests, release_id null): evidence/p1/c-amendment/C-PILOT-PROPOSAL.md.
- 2026-10-10 C pilot (release ipcch-alphaearth-p1-C-pilot-20261010) stopped before computation: EE cannot parse 'EPSG:6933' (3 of 13 calls, 0 responses). Transport repair within approved C: identical CRS sent as pinned WKT1 with an EE projection-equivalence check (<=1 mm) before reductions; parse errors non-retryable. Tests 89 OK. Rev2 manifest (same 10 requests, +1 check call) prepared for release.
- 2026-10-10 C pilot rev2 executed (12 of 14 calls): projection check 2.8e-9 m; C01 valid 99.9996% with ranks 1/4/6 owning 135.889/0.015/87.642 km2 (sum = valid area); C02 vs A2 descriptors differ <= 1.4e-4; windows vs whole A rel 7.7e-9; Red Sea exact zeros; 2.0e-6 s/cell. Production plan: 57,801 pieces, 4.67e11 cells; proposal evidence/p1/C-PRODUCTION-PROPOSAL.md (not started).
- 2026-10-10 Offline production C connection: the runner gained durable per-piece outputs, a hash-bound ledger and
  resume, a cumulative call cap, and a shared per-attempt stop/deadline control. Ingest (release, plan pin, binding and
  numerical completeness checks run before COMPLETE) feeds attach_descriptors under fresh bindings; there is a stale
  check in load_run_inputs and a CLI ingest-extraction command. Sliver maxPixels sizing was fixed before any live run.
  Supervisor fixed list (per-attempt stop, plan pin, null-before-COMPLETE, barrier fixture) was accepted at the pinned
  hashes. Tests: 16 focused / 105 full OK. Enumeration: 57,801 pieces for 6,227 admins, 4.67e11 cells, 0 zero-source
  admins. Full manifest is a DRAFT and held. Small-live S1–S3 proposal (36 pieces, ≤ 84 calls):
  evidence/p1/c-production-offline/C-PRODUCTION-OFFLINE.md.
- 2026-10-10 C small-live (release codex-C-small-20261010) executed and all gates PASS:
  - S1 14/26 calls, 12 pieces, max 45.96 s; S2 18/34 calls, 16 pieces, 75.1 s wall at concurrency 4 (3.5x);
    S3 batch 4/6 and single 10/18 calls. Total 46/84 calls, 0 failures.
  - Invariants hold on all 44 records. Batch vs single: counts identical; sums differ by ≤ 3.1e-16 relative
    (summation order).
  - Rate t ≈ 4.0 s + 1.65e-6 s/cell gives ≈ 279 request-hours for the full plan (≈ 80 h at concurrency 4).
  - Full extraction and fits remain held. Evidence: evidence/p1/c-small-20261010/C-SMALL-result.md. The offline doc's
    request-sha table was corrected to the receipt's body hashes (documentation only).
- 2026-10-10 Full extraction envelope (held; awaiting release):
  - Code: `max_invocations` counted at invocation start; null/nonfinite responses stay pending (kept in
    `failed_responses/`); `check_c_outputs.py` for checkpoint and completeness checks.
  - Limits: concurrency 4, batch 1; 20 × 6 h request-start windows; hard kill 22,200 s per invocation (≤ 123 h 20 min
    total); one invocation at a time.
  - Cumulative calls 59,576; cells ≤ 4.95e11 including re-requests; output ≤ 1 GB outside Dropbox.
    Correction: ≤ 4.95e11 is a geometry-based estimate, not an unconditional bound. Σ lattice boxes is 7.6056e11, and
    repeats are not limited to 1,735. The unconditional guard is call cap × the largest band-px guard = 1.35261e14
    band-px (about 1.0485e12 cell-equivalent); see evidence/p1/c-full-20261010/ipcch-alphaearth-full-volume-accounting-20261010.json
    and C-FULL-ENVELOPE.md. No limit changed.
  - Draft manifest body 63d0c7ab… (pieces and plan unchanged). Tests: 108 full OK; supervisor 19-test PASS.
  - Doc: evidence/p1/c-full-envelope/C-FULL-ENVELOPE.md.
- 2026-10-11 Full D30 extraction (release codex-D30-full-20261010) invocation 1 is running.
  - Started 2026-10-10T23:59:01Z at c4/b1 after all hash checks passed; inputs are unchanged in
    ~/ipcch-alphaearth-runs/p1-c-full-20261010.
  - Early prefix: 30/30 pieces ok, 1.45e6 cells/s, parallelism 3.7; projects to about 90 h (about 15 invocations).
- The user's parallel test request (relayed) arrived after the launch, so the running invocation was not touched.
  A bounded c8 probe manifest was prepared offline and is NOT run: 24 fresh second-half pieces, 3.378e8 cells, ≤ 50
  calls, body 5eab5d21…; see evidence/p1/c8-probe/C8-PROBE-PROPOSAL.md.
- Full release receipt and volume-accounting receipt are retained in evidence/p1/c-full-20261010/.
- 2026-10-11 Full invocation 1 completed normally: c4, 3,807 pieces ok, 3,811 calls, 2 recovered transient errors,
  checker clean.
  - c8 probe (codex-D30-c8-probe-20261011): 23/24 ok. 4 attempts were rejected "Too many concurrent aggregations." and
    1 piece failed after retry. The stop control did not fire because the text was not in FATAL. FAIL for promotion; no
    second attempt, no migration.
  - Execution-guard repair: FATAL += "too many concurrent aggregations" (shared fatal stop, zero retry, no new
    reservations after publication). "Earth Engine capacity exceeded." stays transient; maxPixels/size rejections stay
    non-retryable.
  - Focused 36 OK and full 110 OK; the supervisor confirmed the old classifier fails both new tests.
  - Invocation 2 stays HELD for re-release of the same c4 manifest/run.
- 2026-10-11 Full invocation 2 ended early at 07:27:57Z (4,263 s) on an exhausted transient "Earth Engine capacity
  exceeded." for 378|w001_014.
  - Cause: run_production matched "cap" as a substring, so "capacity" triggered a stop. Over-conservative; no
    numerical effect.
  - Cumulative: 2/20 starts, 4,513/59,576 reservations, 4,505 ok, 1 failed pending, checker clean.
  - My live "continues" report was wrong: the PID was alive and in-flight successes were still landing, which is not
    continuation.
  - Offline class repair:
    - exact SHARED_STOP_PREFIXES ("fatal service error", "request cap ", "non-retryable request-size rejection");
    - Budget publishes the shared stop before raising a non-retryable size rejection (as it already did for fatal).
  - Tests: the new tests fail under 77aa4bf; focused 48 OK; stress 25/25. Resume is HELD for a supervisor receipt.
  - Evidence: evidence/p1/c-full-20261010/invocation2/ and stop-prefix-repair/.
- Not changed: scripts/ee_pilot.py (superseded native pilot, comparison only) has the same substring pattern.
- 2026-10-11 USER EMERGENCY DIRECTION: stop the extraction. The supervisor killed invocation 4 (exit 143 at 12:45:56Z;
  checker 0).
  - Preserved: 4 starts, 8,310 reservations, 8,280 window outputs (774 admins complete, 14.6% of cells).
  - Old resume authority revoked.
  - USER STANDING INSTRUCTION: whenever a decision obviously drifts from the objective, challenge it immediately and
    return to discussion instead of incrementally optimizing or continuing it.
  - Next: a bounded whole-admin direct reduceRegion pilot on the unchanged D30 estimand. Offline proposal: 3 admins
    (REF 658 completed, MED 4711, LARGE 1050), ONE aggregation per admin (sums only; diagnostics dropped), sequential,
    no automatic retry, ≤ 5 calls, body 65efd14b…; evidence/p1/whole-admin-pilot/WHOLE-ADMIN-PILOT-PROPOSAL.md.
  - Tiny opt-in diff: sums-only reduction, Budget attempts.
- 2026-10-11 Whole-admin pilot (codex-D30-whole-admin-pilot-20261011), 5/5 calls, one aggregation per admin:
  - REF 658 ok in 30.4 s, identical to its 2 windows (A rel 2.3e-16; S/Q ≤ 4.4e-14; descriptors ≤ 2.5e-14).
  - MED 4711 ok in 98.0 s (coverage 0.999995).
  - LARGE 1050 read timeout at 120 s, not retried.
  - About 3e-6 s/cell, so fewer requests did not cut per-pixel cost; 10 m exhaustive computation dominates.
  - HOLD for the user discussion (route choice). Evidence: evidence/p1/whole-admin-pilot/WHOLE-ADMIN-PILOT-RESULT.md.
- 2026-10-11 User chose exhaustive 10 m statistics plus whole-admin batch extraction (larger admins split, smaller
  admins batched). Offline design proposal written (nothing implemented or submitted):
  evidence/p1/whole-admin-batch/BATCH-DESIGN-PROPOSAL.md.
  - D30 science and contract unchanged.
  - Split at whole-admin lattice box > 2.56e8 (827 admins → 3,236 pieces; 8,636 items in total).
  - Tasks ≤ 5 MB / 5e9 cells (about 96); ≤ 2 tasks at once.
  - Existing bucket prefix gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/<run-id>/.
  - One-task pilot (658, 4711, 1050 whole, plus 4505 split into 7).
  - Old window outputs proposed as a verification set, not reused.
  - Cloud write awaits user authorization.
- 2026-10-11 User priority: shorten extraction to about one day. Speed-pilot proposal (offline; nothing submitted):
  evidence/p1/whole-admin-batch/SPEED-PILOT-PROPOSAL.md.
  - Remaining 5,453 admins / 3.9887e11 cells after reusing 774 complete admins (14.609%).
  - Required 4.617e6 cells/s over 24 h (5.036e6 over 22 h), about 3x the interactive c4 rate.
  - Pilot: 2 balanced batch tasks (95 fresh size-representative items, 4.99e9 cells), plus 2 references and 4
    reuse-validation admins.
  - Fixed tolerance gates; ≤ 22 h projection at the slower task rate needed before bulk is discussed.
  - Supervisor baseline retained.
- 2026-10-11 Speed pilot: offline implementation (new files only; nothing submitted):
  - Code: scripts/ee_batch_c.py, build_speed_pilot.py, validate_speed_pilot.py; tests/test_speed_pilot.py.
  - Exact adaptive plan: 9,138 items (1,329 admins split into 4,240 pieces); remaining 7,996 items / 3.98874e11 cells.
  - Pilot body 7b8145b6…: A 50 fresh items / 2.418e9 cells, B 50 / 2.562e9, plus 6 pinned non-fresh admins.
    Prefix gs://manuel-remote-sensing-rasters-645415198431/ipcch-alphaearth/speed-pilot-20261011/.
  - Body ≤ 8 MB guard before any start; 60 min cancel.
  - Tests: 10 focused / 60 total OK.
  - Evidence: evidence/p1/whole-admin-batch/speed-pilot-offline/.
  - The pilot write still needs USER authorization.
- 2026-10-11 Speed-pilot repair after the narrow review of 8b00a88 (offline):
  - Validator: bound to the released manifest, MANIFEST pin, plan sha and pinned remaining workload; requires COMPLETED
    tasks with monotonic timestamps; clock stops after all gates; exit 1 on failure.
  - EE/GCS socket timeouts.
  - Per-ID counted status reads (120 s, ≤ 75), cancel reserve, failure handoff; known ids cancelled even with an
    ambiguous start; lookup removed.
  - Resumable download with record-before-rename.
  - Body f66a1be8… (selection and plan unchanged).
  - Tests: 15 / 65 focused OK.
  - Evidence: evidence/p1/whole-admin-batch/speed-pilot-offline/r2/REPAIR-8b00a88.md.
- 2026-10-11 Cancel-attempt accounting repair after the delta review of 7a4ad15 (offline):
  - Cancel attempts are reserved and recorded before the call, counted against max_cancels, and never reissued (also
    after a restart); the other task is still attempted.
  - Unresolved cancels lead to HOLD_HANDOFF.
  - Tests: 17 / 67 focused OK. Manifest unchanged (f66a1be8…). release_id null.
- 2026-10-11 Batch speed pilot (codex-D30-batch-speed-pilot-20261011) FAILED: both tasks hit server
  "Computation timed out." (A 1,651 s / 5.12e5 EECU-s; B 1,247 s / 1.72e5 EECU-s). No download or validation.
  - User steering: the Mexico EVI extraction is reference only, and "实在不行的话就继续原来的流程" (if nothing else
    works, continue the original flow).
  - Supervisor comparison: Mexico was a 30 m single-band selection rescued by smaller exports; no shortcut for ours.
  - Decision: fall back to the original c4 window flow on the SAME run; no further batch pilot.
  - Evidence: evidence/p1/whole-admin-batch/speed-pilot-run-20261011/.
- 2026-10-11 Fallback-resume prelaunch checkpoint (offline; NOT launched):
  - Verified state: 4 starts, 8,310 reservations, 8,280 outputs, 49,521 pending; latest exit 143 (user stop); checker
    0 violations; no runner; lock free; 648 GB free.
  - Source delta vs the invocation-4 release: only an opt-in pilot branch, a Budget(attempts) default, and tests;
    production unchanged.
  - External gate v2 (resume_check_v2.py c2573686…): pins the single invocation-4 summary gap (exit 143 at 12:45:56Z,
    8,904 s) and uses corrected elapsed = summed summaries + 8,904 s, giving 70.4 h remaining vs a 96 h window.
  - Fixture: 7 OK. prelaunch_inv5.py dry run passes. The v1 gate refuses as expected.
  - Proposed receipt fields: evidence/p1/c-full-20261010/fallback-resume/.
- 2026-10-11 USER STEERING (binding): "既然回到原路线了就不用考虑一天完成。" (Since we are back on the original route,
  completing in one day no longer matters.)
  - The one-day target is withdrawn for the c4 window fallback. Continue the released extraction reliably.
  - No further speed pilots. Missing the former one-day target is not reported as a limitation or failure.
  - Ordinary progress, a planning ETA, anomalies, the release caps and the normal-wall resume gate are all unchanged;
    no cap is enlarged or reset.
  - Invocation 5 launched 2026-10-11T14:46:35Z under codex-D30-c4-user-fallback-20261011; supervisor launch receipt
    retained.
- 2026-10-11 USER STEERING: long-running task; lower the monitoring frequency to save tokens.
  - Routine LLM status check every 30 min, at :07 and :37 local, as a session cron that expires after 7 days. It reads
    local counters only.
  - Run-local `watch_events.py` (polls every 120 s, no LLM) wakes the executor only on: wrapper exit (terminal), a new
    failed/null/halted row, a request error not recovered within 300 s, no reservation for 900 s, more than 1 GB, or
    crossing 25/50/75% of total cells. Recovered transients are only logged (monitor/auto-recovered-errors.log).
  - Runtime fatal/consecutive/fraction/cap guards and the 6 h normal-wall resume gate are unchanged.
- 2026-10-12 User-authorized local streaming D30 parity + speed preflight (user asleep; supervisor decides): NOT
  VIABLE, so EE-only continues.
  - Stage 1 (admin 3447; one pinned COG, 1.325 GB, requester-pays):
    - the pre-registered gate failed on the selection reference C02 (A_rel 1.6e-5 vs 1e-7; centre count −23 cells);
    - every held-out C03 window failed too;
    - tested densification changes did not resolve the discrepancy; the mechanism is unisolated.
  - Stages 2/3 were not downloaded, and no hybrid was released.
  - Indicative speed: whole-file download (32 MB/s) dominates; 2.2 s local reduce for 512k cells.
  - The live EE invocation 8 was untouched throughout.
  - Evidence: evidence/p1/local-stream-preflight-20261012/.
- 2026-10-12 Supervisor verified the local preflight (all seven references reproduced exactly, all fail).
  - Decision: EE-only. No stage 2/3 downloads or hybrid integration; the audit COG is kept.
  - Wording bounded: the tested variants fail the fixed gates; the mechanism is unisolated; C03 descriptor comparisons
    are uninformative (NaN on both routes); the speed figure is for one small admin only.
- 2026-10-12 User fallback preference, conditional only (relayed by the supervisor; the current run is unchanged):
  - User idea: process the GCS AlphaEarth TIFFs directly with xarray on Cloud Run, bypassing EE, "现在就先这样就可以，如果EE额度用完了可以这么处理".
  - Now: EE-only. No Cloud Run deployment, no resource creation, no resumed local TIFF work.
  - Trigger: only an actual EE quota exhaustion. This is distinct from:
    - our own manifest caps (20 starts / 59,576 calls);
    - transient capacity or concurrency errors.
  - Caveat: moving the same reducer to Cloud Run does not by itself fix the observed D30 numeric mismatch from the local preflight.
  - Before any later use, either establish compatible statistics and provenance, or make an explicit estimand amendment.
  - Cloud-native and EE outputs are never silently merged.
  - Existing guards and normal continuation remain intact.
- 2026-10-12 Invocation 9 capacity burst (15:39-15:40Z). Watcher burst alert; supervisor verified independently; run continued unchanged.
  - Six pieces failed after their single retry ("Earth Engine capacity exceeded." / "memory capacity exceeded."), in admins 2598/2599/2600.
  - Four of them were consecutive: one short of the max_consecutive_failures guard.
  - Request processing then resumed: new success rows, and new reservations beyond the last failed retry n=28449.
  - Resumed processing is NOT recovery of the six pieces. They stay pending for the normal next pass, with no standalone retry.
  - Failed fraction is about 0.00135, within the released guard.
  - This is not EE quota exhaustion, so the Cloud Run fallback is not triggered.
  - Receipt: run receipt/ipcch-alphaearth-inv9-capacity-burst-supervisor-20261012.json (includes raw errors).
- 2026-10-12 Invocation 9 ended at its normal wall stop (5,050 rows = 5,043 ok + 7 failed). Checker and resume_check_v2 passed.
  - Settled unique: 28,991. Corrected remaining: 36.6 h vs a 66 h window.
  - Invocation 10 launched at 16:31:05Z with the wrapper unchanged.
  - Supervisor verified the checkpoint: run receipt/ipcch-alphaearth-inv9-checkpoint-supervisor-20261012.json (24531980...).
  - All 7 inv9 failed pieces now have ok rows and saved outputs from inv10's normal pass, so all 7 are recovered.
  - Wording: "non-ok rows" (15 at the inv9 stop) is a historical row count, not the number of pieces currently pending as failed.
  - Received the supervisor's 50% receipt (48a122ae...); the run crossed 50% at 16:09:16Z.
- 2026-10-12 Invocation 10 ended at its normal wall stop (5,950 ok, 0 failed). Checker and resume_check_v2 passed.
  - Unique successes: 34,941. Corrected remaining: 27.8 h vs a 60 h window.
  - Invocation 11 launched at 22:34:44Z with the wrapper unchanged.
  - Supervisor verified the checkpoint: run receipt/ipcch-alphaearth-inv10-checkpoint-supervisor-20261012.json (f7783567...).
  - Every historical failed piece has since succeeded.
- 2026-10-13 Invocation 11 ended at its normal wall stop (5,859 ok + 1 failed). Checker and resume_check_v2 passed.
  - Unique successes: 40,800. Corrected remaining: 21.8 h vs a 54 h window.
  - Invocation 12 launched at 04:38:47Z with the wrapper unchanged.
  - The failed piece 3727|w000_001 succeeded in inv12, with a saved output.
  - Supervisor verified the checkpoint: run receipt/ipcch-alphaearth-inv11-checkpoint-supervisor-20261013.json (b77ac9d0...).
- 2026-10-13 The 75% plan-cell milestone was reached during inv12 (first ledger crossing 08:17:39Z; 43,651 successes at the 08:19:22Z alert).
  - Supervisor verified it: run receipt/ipcch-alphaearth-75pct-supervisor-20261013.json (c366d20e...).
  - 4343|w006_001 (failed at 07:02Z, capacity) is pending for the normal pass.
- 2026-10-13 Invocation 12 ended at its normal wall stop (4,919 ok + 1 failed). Checker and resume_check_v2 passed.
  - Unique successes: 45,719. Corrected remaining: 15.0 h vs a 48 h window.
  - Invocation 13 launched at 10:41:13Z with the wrapper unchanged.
  - The failed piece 4343|w006_001 succeeded in inv13, with a saved output.
  - Supervisor verified the checkpoint: run receipt/ipcch-alphaearth-inv12-checkpoint-supervisor-20261013.json (fac49874...).
- 2026-10-13 Invocation 13 capacity burst (watcher alert at 14:37:43Z). Supervisor checked it; the run continued unchanged.
  - Four failed pieces in large admin 100228 (13:57-14:37Z); 6 failed of 3,279 inv13 rows (~0.18%), none consecutive.
  - Pending for the normal next pass.
  - Receipt: run receipt/ipcch-alphaearth-inv13-capacity-burst-supervisor-20261013.json (d4b9806e...).
- 2026-10-13 Invocation 13 ended at its normal wall stop (4,872 ok + 11 failed, all capacity). Checker and resume_check_v2 passed.
  - Unique successes: 50,591; pending: 7,210. Corrected remaining: 9.6 h vs a 42 h window.
  - Invocation 14 launched at 16:44:03Z with the wrapper unchanged.
  - All 11 inv13 failures have saved successful outputs in inv14.
  - Error trend: request errors per invocation were inv9 30, inv10 12, inv11 18, inv12 15, inv13 ~54. All were transient capacity; no quota exhaustion.
  - Supervisor receipts:
    - run receipt/ipcch-alphaearth-inv13-checkpoint-supervisor-20261013.json (6d78cb98...);
    - second-burst receipt/ipcch-alphaearth-inv13-burst2-supervisor-20261013.json (a2fc82d8...).
- 2026-10-13 Supervisor transition checklist, retained for the terminal P1 completion (offline, already authorized; no change to the live extraction or pins):
  1. Final checker gate: complete:true, pending:0, n_violations:0 (exit 0 alone is insufficient), and a settled runner (no live EE process or lock).
  2. Fresh WSL run: `preflight --kind formal --hash-inputs`, then `prepare --local-only` exactly ONCE.
  3. `ingest-extraction` with the full-run inputs/production_plan.csv.gz and inputs/production_manifest.json, and the outputs argument pointing to ~/ipcch-alphaearth-runs/p1-c-full-20261010/out; then prepare to attach.
  4. Preserve the old p0-local-20261010b; its config predates D30.
  5. Before fits, report:
     - descriptor, coverage and cohort counts plus hashes;
     - proof that load_run_inputs assembles successfully without fitting;
     - reconciliation of the 61-quartet / 244-scalar inventory.
     The existing fit checkpoint remains.
  6. Downstream bookkeeping: correct the README's native-grid command examples and its D1-D29 label to the implemented D30 path. implement.md lines 98-110 have the right commands.
- 2026-10-13 Invocation 14 stopped NON-wall at 20:50:50Z: fatal "Too many concurrent aggregations." on 101076|w003_001 (the third such stop, after inv3 and inv7).
  - Counts: 2,921 ok, 1 failed, 2 halted; 53,512 done / 4,289 pending; 53,727 reservations.
  - Just before the fatal, two 120 s read timeouts (w002_003, w002_004) recovered on attempt 2. w002_004 completed as an in-flight request after the fatal.
  - My HOLD report wrongly called w002_003 unresolved; I corrected it within minutes.
  - Evidence: run monitor/inv14-terminal/.
- 2026-10-13 Supervisor one-time release codex-D30-c4-post-concurrency-inv15-once-20261013 (0c6b89d8...), with an inv14 terminal receipt (ab10ae5d...).
  - Not before 21:10:18Z (20 min quiet), on the unchanged c4/b1 allocation.
  - prelaunch_inv15.py all-pass at 21:10:33Z: hashes, counters, fresh checker, no runner, lock free, and the gate's only refusal equal to the exact inv14 fatal.
  - Invocation 15 launched once at 21:11:44Z (attempt-line 43).
  - Future normal walls use the original fallback gate; any other stop is HOLD.
  - Supervisor verified the inv15 launch: run receipt/ipcch-alphaearth-inv15-launch-supervisor-20261013.json (655dc229...). All three inv14 terminal pieces have saved ok outputs.
- 2026-10-13 Invocation 15 stopped NON-wall after 652 s: the same fatal on 101089|w002_001, again ~20 s after a 120 s read timeout.
  - Counts: 53,601 done / 4,200 pending / 53,822 reservations / 15 starts.
  - Supervisor: HOLD; no further unchanged-c4 start. Inv15 terminal receipt 50183b33....
  - User: "时间上没关系" (stable completion first, no speed target).
- 2026-10-13 Offline c2 operational restriction implemented (HOLD; no launch until a reviewed supervisor receipt binds it).
  - Runner: ee_pilot_c.py adds --concurrency-ceiling. Effective workers = min(manifest 4, N); invalid values are refused before any start, reservation or EE call.
  - Each start is recorded in out/invocation_concurrency.jsonl, and the summary records manifest and effective concurrency.
  - Hashes: ee_pilot_c.py c18513e2 -> d82b3e9a; test_c_production.py 5b83b787 -> 0cccdf8e.
  - Tests: 4 new tests; full suite 136 OK.
  - New run-local run_invocation_c2.sh 2fd621ea; the old wrapper is unchanged.
  - Manifest, pieces, plan, outputs, counters and the gate code are untouched.
  - Evidence and proposed receipt fields: run monitor/c2-offline-delta/.
  - Conservative planning figure (supervisor): 13.26 h of c2 work vs 30 h of start windows.
- 2026-10-13 Supervisor accepted the c2 delta at the reviewed hashes.
  - Release: codex-D30-c2-operational-ceiling-continuation-20261013 (18d73373...).
  - Supervisor evidence: 25 focused tests pass; structural review a5312659; gate review b3145cef.
  - Uncommitted state is recorded; no repository cleanliness is claimed.
- 2026-10-13 Invocation 16 launched once at 22:23:50Z with run_invocation_c2.sh (effective concurrency 2, manifest 4) after prelaunch_inv16 all-pass. Quiet interval: 60 min from 21:22:22Z.
  - Initial c2 projection: 47,735 s vs a 108,000 s window.
  - Later normal walls need c2_wall_check.py (v2 against the c2 receipt, plus 4/2 start and summary records, plus the c2 projection guard).
  - The old c4 wrapper and releases no longer authorize starts.
  - Supervisor verified the inv16 c2 launch: receipt/ipcch-alphaearth-inv16-launch-supervisor-20261013.json (2f769a95...). All 3 inv15 terminal pieces have saved ok outputs.
  - Supervisor reviewed c2_wall_check.py at a2cdf511... (pinned before use; 3 offline fixtures pass): receipt/ipcch-alphaearth-c2-wall-check-review-20261013.json (6427a4ee...).
- 2026-10-14 Invocation 16 (c2) ended at its normal wall stop (1,855 ok + 1 failed, no fatal). Done 55,456 / pending 2,345.
  - c2_wall_check passed: observed c2 rate 724,527 cells/s (below the 897,617 planning rate); projected 37,474 s vs an 86,400 s window. Output: monitor/c2_wall_check-after-inv16.json.
  - Invocation 17 launched at 04:27:49Z with run_invocation_c2.sh.
  - Supervisor verified the inv16 checkpoint: receipt/ipcch-alphaearth-inv16-checkpoint-supervisor-20261014.json (03b62522...); 10.41 h remaining vs a 24 h window.
- 2026-10-14 Invocation 17 (c2) ended at its normal wall stop (1,760 ok, 0 failed). Done 57,216 / pending 585.
  - c2_wall_check passed: observed rate 819,290 cells/s; projected 8,994 s vs a 64,800 s window. Output: monitor/c2_wall_check-after-inv17.json.
  - Invocation 18 launched at 10:32:05Z with run_invocation_c2.sh. It is expected to complete the extraction.
  - Supervisor verified the inv17 checkpoint: receipt/ipcch-alphaearth-inv17-checkpoint-supervisor-20261014.json (800fe1ec...).
- 2026-10-14 P1 D30 EXTRACTION COMPLETE. Invocation 18 (c2) did 585/585.
  - Final checker check-20261014T123221Z.json: 57,801/57,801 done; pending 0; complete true; 0 violations.
  - Totals: 58,037 of 59,576 reservations; 18 of 20 invocations; out/ 476 MB.
  - Evidence: run monitor/final-complete/.
  - No further EE starts. Next: the authorized local formal preparation/ingest checkpoint (no fits).
- 2026-10-14 Supervisor accepted the final extraction: receipt ipcch-alphaearth-final-extraction-supervisor-20261014.json (5f491547...).
- 2026-10-14 FORMAL D30 INPUT FREEZE complete (local only, no fits) in fresh run ~/ipcch-alphaearth-runs/p1-formal-d30-20261014.
  - Stages, each once: preflight formal --hash-inputs (pass; 61 quartets / 244 scalar) -> prepare --local-only -> ingest-extraction (57,801 complete, 6,227 admins) -> attach (6,227 usable).
  - Precision check (default vs round_trip plan parsing): descriptor table bitwise equal; routes unchanged; the ledger differs only at <=3.3e-16; minimum margin to .80 is 0.189.
  - Freeze report:
    - all 73 artifact hashes;
    - geometry = plan = descriptors = ledger = 6,227;
    - load_run_inputs assembles: 296/424 digests 0b8757a4/0163a677; each H has 42,695 rows / 6,224 admins;
    - schedule 61 / 244 equals the preflight ceiling.
  - Observations: coverage > 1 for 1,674 admins (max 1.0058); 3 admins without labels.
  - Evidence: evidence/p1/formal-d30-20261014/.
  - p0-local-20261010b preserved. Awaiting the supervisor's next release.
- 2026-10-14 P2 release codex-D30-formal-P2-development-mapfreeze-20261014 (ae558aab...): pre-fit check PASS; develop launched once at 13:19:36Z (pid 3319861/3319863). Then freeze-map after ledger reconciliation; stop before predict.
- 2026-10-14 P2 develop COMPLETE (13:19-13:31Z, exit 0): 27 quartets (19 B1 + 8 B0) / 108 scalar fits, exactly as scheduled.
  - Development NEGATIVE at all H: 0 of 72 recipe-years pass the guards; every H falls back to B2 (D13/D23).
  - freeze-map COMPLETE at 13:33:44Z (cutoff 2022-12): B2 for all H, no map. MAP_FREEZE 7e5778ea.
  - STOPPED for the supervisor's map-freeze review; predict/bridge/main/supplement (34 quartets) not run.
  - Evidence: evidence/p1/formal-d30-20261014/p2/.
- 2026-10-14 P3 release codex-D30-formal-P3-predict-report-replay-20261014 (efceca32...): pre-predict check PASS; predict launched once at 13:51:18Z (pid 3480736/3480738).
- 2026-10-14 P3 predict COMPLETE: 34 fits (61/244 total, exactly as scheduled); G==B2 bitwise at all H; provider ancestry clean.
  - report COMPLETE: no qualifying horizon (all no_gain_or_reversal; G-B2 = 0).
  - replay FAILED 16/488 checks -> HOLD. Two classes, both evidence/provenance only (no number changes):
    - (A) 15 canonical_cohort_fields: persistence_source_month missing written as NaN vs '';
    - (B) 1 h03 combined providers: one corrupted target_ord cell (2431 vs 24314) in predict/providers_all_h03.csv.gz. The zero-fit re-prediction is otherwise bitwise equal; cause unknown.
  - Awaiting the supervisor's scoped decision. Meeting note not started.
  - Evidence: evidence/p1/formal-d30-20261014/p3/.
- 2026-10-14 Scoped offline replay repair (release codex-D30-P3-replay-evidence-scoped-repair-20261014, 7e9c5f5d...). Done for review only:
  - replay.py prediction read now uses the canonical NA policy (cd5eeb38 -> a00067a0).
  - 4 regression tests added; the first failed before the fix with the production signature. Replay-related suites pass: 13 + 5 + 2.
  - 45/46 code, 73 inputs and 404 science artifacts unchanged.
  - Staged h03 provider reconstruction 06eefdc6 differs from the original by exactly one cell (target_ord 2431 -> 24314).
  - Archive proposal drafted. Not installed; no formal replay rerun.
  - Evidence: evidence/p1/formal-d30-20261014/p3-replay-repair/.
- 2026-10-14 Release codex-D30-P3-replay-amendment-install-and-recheck-20261014 (991ce446...): archive and install applied.
  - Failed attempt 1 and the corrupt provider e454ce15 are archived under archive/replay-attempt-01-20261014T141645Z/ (ARCHIVE.json 6f9c534e, ANALYSIS_AMENDMENT.json 488d966d). The reconstructed provider 06eefdc6 is installed.
  - Formal replay rerun ONCE: PASS 488/488, zero XGB fits (replay.json 1bbdd949, COMPLETE 48ec2a51). 404 science and 73 inputs unchanged.
  - Next: meeting note / README / inventory / offline MLflow plan.
- 2026-10-14 Formal result accepted (receipt codex-D30-P3-replay-recheck-acceptance-20261014, 0b050601...): no qualifying H; B2 fallback at every H; G == B2. Delivery pending.
- 2026-10-14 Delivery work (documentation and an offline plan only; no MLflow writes):
  - Chinese meeting note: docs/notes/2026-10-14_组会讨论稿_IPCCH_AlphaEarth监督分区.md.
  - README: D30 commands replace the native-grid examples; formal result added.
  - PRD/implement checkboxes reconciled with receipts, with appended status lines; the original planning text is kept.
  - Artifact inventory: evidence/p1/formal-d30-20261014/ARTIFACT_INVENTORY.json.
  - Offline MLflow plan: mlflow-plan.md. The target is the existing detailed + dashboard store, with an additive predecessor-fingerprint extension.
  - All 16 AlphaEarth evaluation cohorts have keys/truth digests identical to existing dashboard datasets (delivery/mlflow_dataset_compat.json).
  - Awaiting supervisor review of the plan; adapter code changes and live writes need a separate release.
- 2026-10-14 Delivery plan review r1 (codex-D30-delivery-and-mlflow-plan-review-r1-20261014, 271032d1...): revision r2 prepared (docs, plan and read-only evidence only).
  - M1: exact saved-text identity. 16/16 keys equal, 0/16 truth text equal (q3 differs <= 9.9e-17). 12 family-qualified evaluation descriptors frozen (dataset_descriptors_v2.json 2efda436); no supplement persistence dataset.
  - M2: native AlphaEarth model-ledger contract specified (61 identities / 244 own boosters).
  - M3: all 1,298 sha256 leaves typed. External bindings by exact path (list_images, production pieces, receipts); no generic sources/** exemption.
  - M4: versioned inventory v2 with self-exclusion.
  - M5: dashboard extension state machine S0-S3 with resume/backup rules.
  - M6: 8 composite training descriptors (training_descriptors_v2.json b76c2ab2).
  - D1: meeting-note corrections. The reviewed note is snapshotted.
  - Plan: mlflow-plan.md r2 (r1 in mlflow-plan-history/).
  - No adapter/store/server change.
- 2026-10-14 MLflow offline implementation (release b9cdee29...): the five IPCCHMLflow files, tests and docs were implemented.
  - The empty-artifact verifier fix was accepted (2c91952e...).
  - The scratch rehearsal stopped twice: once on the empty-artifact listing, once on missing legacy Temp sources (four roots had 0 files).
  - The surviving yearly-formal-20261007 source was preserved, SHA-verified and made read-only under ~/ipcch-legacy-source-preservation/.
- 2026-10-14 User stop, then the user research-storage policy (supersedes the old retention/rehearsal release).
  - B0, the scratch stores and the old models.tar files were removed by the supervisor; they are not recreated.
  - Code delta (no live writes):
    - model bundles are listed by member manifest and not uploaded by default (opt-in `upload_model_bundles`);
    - bundle-mode mismatch refused;
    - AE model descriptors carry an empty bundle URI; legacy tar tags documented as historical;
    - `apply --extend` no longer needs a full-store backup;
    - restore-check skips absent tars;
    - AE extras limited to code + required inputs (no task docs/evidence/receipts).
  - README, both specs and mlflow-plan.md updated in place.
  - Next: review of the diff, then the live AE import + dashboard extension on explicit go-ahead.
- 2026-10-14 Research-retention delta passed scoped review. Final config-only adjustments:
  - AE extras are code only. list_images and production pieces are no longer copied; they stay external and SHA-pinned.
  - extraction/{plan,pieces}.csv.gz moved to excluded (path + SHA).
  - Dashboard guide now states that old bundle URI/SHA tags are historical and the weights were intentionally removed.
  - Superseded B0/scratch commands were removed from mlflow-plan.md. Focused wording test: OK.
- 2026-10-14 Live additive registration (store ~/.local/share/ipcch-mlflow, http://127.0.0.1:5000, server via manage.sh; no backup, scratch or legacy verify):
  - AE plan: fp 4f96ff09..., upload estimate 79,281,769 bytes.
  - AE import: created 21, 1,080 metrics, 260 artifacts. AE verify: 21 records, 79,443,114 bytes read back.
  - Dashboard plan --extend: legacy reproduces c7abf4cd; new fp cb3db908...
  - Dashboard apply --extend reached S3; built-in verify passed (legacy 136 rows / 100 versions / 68 registered models unchanged; detailed runs unchanged; catalog_status complete).
  - Counts:
    - runs: exp1 126->147, exp2 136->156, exp3/4 202/174 unchanged;
    - registered models / versions / logged models: 180/226/226 -> 196/242/242;
    - datasets: +20 (12 family-qualified evaluation + 8 training).
  - Log: ~/.local/share/ipcch-mlflow/logs/alphaearth-registration-20261014.log.
  - Task lifecycle remains open (no commit/archive).
- 2026-10-14 Delivery accepted by the supervisor (independent read of ae-dashboard-apply.json, SQLite and health):
  - catalog_status complete; experiments 147/156/202/174; registered models/versions/logged models 196/242/242.
  - No AE models.tar; AE parent artifacts 79,443,114 bytes; local store about 2.25 GB; health HTTP 200.
  - Research delivery complete. Lifecycle (commit, close/archive, audit) remains separate scope and not done.
- 2026-10-14 Branch integration authorized by the user: commit, push, and merge into main, preserving Claude's dashboard, permissions and remote-assistant integration.
  - Final integration checks: MLflow offline suite 81/81 passed; AlphaEarth replay/pipeline/synthetic regressions 20/20 passed; `bash -n IPCCHMLflow/manage.sh` passed.
  - Read-only service check: local health 200, anonymous root 401; MLflow and Cloudflare services both active and enabled. WSL shutdown/reopen was not exercised in this session.
  - Current delivery also retains model-card lookup, the assistant read-only command allowlist, and WSL-session autostart. Runtime credentials remain outside Git.
  - Only code, documentation, tests and compact task evidence enter Git; no raw inputs, model bundles, live stores, rehearsal copies or cloud-only legacy files are added. Root AGENTS.md/CLAUDE.md remain intentionally ignored by existing repository policy.
