# AlphaEarth official documentation — 2026-10-10

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

Primary source read by coordinator:
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

## Documented facts

- EE collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
- The catalog currently displays availability `2017-01-01` through `2025-01-01`; this is not evidence of a complete 2025 annual embedding. Actual image inventory has not been queried.
- Every 10 m pixel has 64 dimensions, `A00`–`A63`, describing observations over a **single calendar year**. All dimensions collectively describe one embedding and are not independently interpretable physical bands.
- Pixel embeddings have unit norm; space is consistent across years. Documentation says they are linearly composable and may be aggregated to coarser resolution.
- These statements do not make an aggregated mean unit length; aggregation, dispersion summaries and any renormalization still need an explicit design decision.
- Images are in local UTM projections. Calendar coverage is reflected in `system:time_start` / `system:time_end`.
- The catalog identifies AlphaEarth Foundations v2.1 as the producer model; the paper evaluated v2.0. Since 2025-11-17, layers have `DATASET_VERSION=1.1`, including regenerated 2017 Sentinel-1 coverage.
- Image properties include `MODEL_VERSION`, `PROCESSING_SOFTWARE_VERSION`, `DATASET_VERSION`, and `UTM_ZONE`; pin actual downloaded image metadata, not just the collection name.
- GCS bucket: `gs://alphaearth_foundations`; catalog states provider pays as of July 2026. This does not establish that arbitrary downstream EE computations or exports have no quotas/cost.
- License CC-BY 4.0; attribution: “The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.”

## Consequences for the design (inferences, not demonstrated efficacy)

1. Never treat annual Y embeddings as available for an origin inside Y merely because the layer is labelled Y. The layer uses that full year's observations.
2. Year-before-origin is an observation-time condition, not proof that the asset/version was published by that historical origin. A retrospective representation experiment and a strict historical operational simulation must be distinguished.
3. A static geographical representation is one identifiable first experiment; annual representation updates jointly test environmental state updates and can change assignments. User accepted fixed representation and a learned-then-frozen partition on 2026-10-10 (D1); D3 selects 2017, and D4 freezes the final map through end-2022.
4. Aggregation feasibility supports an admin-level feature pipeline; it does not demonstrate that embedding-based groups transfer food-crisis response relationships.
5. Pooled + AlphaEarth must be a principal comparator, otherwise improved predictions cannot be attributed to partitioning.

## Not checked

Live EE asset inventory, actual publication timestamps, SDK access/project, model pretraining temporal scope, download sizes, per-admin coverage, and prediction effects. No EE export or data download was launched.

## Accepted research time semantics (D2)

The existing P6 calendar (`IPCCHGeoXGBExperiment/config/experiment-contract.json:20-31`) uses development through 2022 and evaluation from 2023, with observation-month availability rather than verified release vintages. The current catalog documents 2025 product revisions. Historical observation dates therefore do not demonstrate historical availability of this representation product or its pretrained model.

User accepted on 2026-10-10: first conduct an explicitly retrospective representation/mechanism experiment. Its primary comparison gives the same frozen AlphaEarth information to pooled and partitioned arms. Satellite observation coverage must end before the earliest evaluated forecast origin; supervised partition/model selection must respect their own historical label cutoffs. Product publication and foundation-model pretraining limitations must remain disclosed, and this experiment cannot establish real-time historical deployment performance. A later independent prospective test is separate from this retrospective evidence.

## Accepted fixed representation year (D3)

User accepted on 2026-10-10: use the 2017 annual representation, the earliest year listed in the official catalog. At the calendar-month level this permits origin-safe use of its annual observation coverage from 2018 onward; exact timestamp boundary semantics remain to be specified. Compared with a later annual or multi-year representation, this leaves more potential historical origins for out-of-time supervision before the end-2022 development cutoff. D4 subsequently adopts main origins >=2023-01 and targets <=2025-12, with 2026 targets separate; D6 sets forward development blocks in 2021/2022.

Trade-off: a single 2017 representation may encode year-specific conditions and become stale for later urban/agricultural changes. More recent/multi-year embeddings could improve geographical characterization, but their full observation interval delays eligible historical origins. No predictive superiority is established for either choice. Do not select the year by scanning the previously inspected evaluation scores. D5 subsequently retains pre-2018 outcomes for later fits with explicit fit-time attachment of the 2017 descriptor; old fitted rows do not become honest residual supervision.

## Accepted admin aggregation (D16)

User accepted on 2026-10-10: use 64 area-weighted means plus 64 area-weighted population standard deviations of valid 2017 pixel embeddings within each admin (128 descriptors). For pixel overlap weights w normalized within the valid area, mu_j = sum(w * e_j), sd_j = sqrt(sum(w * (e_j - mu_j)^2)); "population" here means the full pixel distribution, not human-population weighting or a sample degrees-of-freedom correction. Each physical pixel contributes once. Do not renormalize the mean vector. Exact extraction, masks, overlap accounting and minimum coverage remain to be frozen before execution.

The documented spatial composability supports the mean. Per-dimension dispersion is the selected summary of within-admin heterogeneity, not a documented food-crisis predictor or a complete description of the spatial distribution. It omits cross-dimension covariance and sub-admin spatial arrangement. All 128 descriptors are to be supplied identically to B1/B2/G (424 XGB input columns with D15); the partition tree may use these descriptors to route whole admins. This does not add 128 regional slope parameters: correction inputs remain a separate pending decision.

Area weighting avoids requiring a separately dated population/cropland weighting product in this first experiment, but extensive low-population land may dominate an admin's representation. Mean-only aggregation was the smaller 64-descriptor alternative; mean plus dispersion was selected to distinguish some admins with similar average embeddings but different internal environments. The extra descriptors expand the tree's split search and may increase selection noise; D6 forward validation and D14 tree budgets still apply. Coverage diagnostics are to be retained as extraction evidence, not silently appended as predictors.

## Accepted coverage and missing-region rule (D25)

The official catalog says the product covers terrestrial land surfaces and shallow waters, including intertidal/reef zones and inland/coastal waterways; polar coverage is constrained by sensors/orbits. It does not guarantee complete coverage of every administrative polygon (catalog URL above; saved text `/tmp/ipcch-alphaearth-catalog-20261010.txt:100-103`, re-read by coordinator). A read-only local probe found that every canonical valid-target admin has an existing geometry, with component hashes matching the frozen inputs; see `local-reuse-facts.md`. Neither fact establishes actual 2017 embedding coverage.

User selected the **80%** option on 2026-10-10 with the following common-cohort/fallback protocol:

1. After confirming extraction completed correctly for the frozen 2017 assets, a valid pixel has the product mask valid and finite values in all A00–A63. Use the common 64-band valid footprint; do not calculate each band's moments over a different footprint. Follow D16's unique physical coverage/overlap-area weighting. Define coverage as valid covered overlap area divided by the full frozen administrative polygon area. No additional land/water/population mask or silent polygon clipping changes that denominator.
2. Require unrounded coverage **>=0.80** and 128 finite moments to use the admin's AlphaEarth descriptor for either XGB input or partition routing. Means/SD are normalized over the valid covered area, not the full polygon area, and are not filled with zero for missing pixels. For a legitimate low/no-coverage admin, all 128 AlphaEarth columns are NaN together; compact296 is retained, giving the same 424-column schema. B1 uses XGB native missing-value handling; B2/G inherit this same B1 prediction, without a new missingness predictor or neighbour/point embedding substitution.
3. Keep every otherwise eligible training/evaluation row under the common cohort protocol. A low/no-coverage admin cannot contribute to regional split membership or leaf-support counts. Its eligible historical OOT rows still contribute to shared-correction fitting and preprocessing in B2/G. For G fix its regional deviation to zero: `p + v^T beta_G`. As already allowed by D18/D20, that shared response need not equal separately fitted B2. Save a distinct missing-AlphaEarth route; do not describe it as exact B2 fallback. For no selected partition or fewer than two supported leaves, retain D20's whole-model exact B2 reuse.
4. In D19's loss, these rows remain in W and the shared-fit sum with `u=0`; D18's regional centering weights remain fractions of mapped correction-training weight. Missing admins do not form a learned extra leaf. Thus this extension preserves the B2 nesting and uses the same pooled training keys and same correction-supervision keys across B2/G.
5. D22/D23 main success and D13 development selection still use the full common eligible cohort. Report coverage by admins/rows/country and report identical-key covered/uncovered subsets descriptively; a favorable covered subset cannot rescue a full-cohort failure. Freeze the mask/threshold independently of outcomes and do not tune it on scores.
6. Missing files, failed exports, unresolved source tiles, unexpected schema/hash mismatch or damaged geometry are technical failures, not legitimate low coverage to conceal behind this fallback. Actual product mask absence must be distinguishable from an incomplete extraction.

The accepted .80 cutoff trades greater routing coverage for more partial-area representations relative to the unselected .90 proposal. It is a user-selected quality tolerance, not an empirically validated threshold. The full-polygon denominator may conservatively exclude coastal/island polygons containing deep water even when their land is well covered; it is not a sensor-quality score. Neither cutoff has been tested against this experiment's outcomes, and no asset extraction has begun. Geometry identity/vintage limitations remain even at 100% pixel coverage.
