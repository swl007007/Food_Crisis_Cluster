# GeoRF m2 Cluster-Level Profile Diagnostics Design

## Goal

Address the reviewer request for local-model cluster-level characterization with
a representative GeoRF example. The artifact will describe one implemented
local-model partition, not all model families or all partition maps.

## Scope

- Model family: GeoRF only.
- Partition map: GeoRF m2 refined mapping,
  `result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc13_m2_refined_contig3.csv`.
- Unit of analysis: the 13 local-model clusters in that mapping.
- Output location: `final_artifacts_in_paper_updated/`.
- Interpretation: descriptive cluster profiles and error analysis, not feature
  importance, causal attribution, or local model internals.

## Data Sources

- Cluster membership:
  `cluster_mapping_k40_nc13_m2_refined_contig3.csv`.
- Local prediction outcomes:
  `result_partition_k40_compare_GF_fs1/predictions_monthly.csv`, filtered to
  February target months when using the m2 partition.
- Countries and regions:
  FEWSNET Admin Boundaries shapefile, using `ADMIN0` and the existing
  reviewer-artifact region mapping.
- Crisis prevalence:
  `y_true` from the Stage 3 predictions, summarized within each cluster.
- Error modes:
  `y_true` and `y_pred_partitioned`, summarized as TP, FP, FN, and TN shares.
- Market-access profile:
  raw panel columns `market_access` and `market_distance`.
- Conflict-exposure profile:
  raw panel ACLED event and fatality columns, including base, 5 km, and 10 km
  windows.
- Dominant AEZ:
  AEZ one-hot columns from the raw panel data.

## Output Artifacts

1. `georf_m2_cluster_profile_table.csv`
   - One row per cluster.
   - Columns include cluster id, polygon count, countries/regions included,
     dominant region, crisis prevalence, dominant AEZ, market-access profile,
     conflict-exposure profile, main error mode, and local error counts.

2. `georf_m2_cluster_profile_similarity.png`
   - A compact `3 x 2` diagnostic figure:
     - row 1: market-access inter-cluster similarity heatmap and
       intra-cluster cohesion bar.
     - row 2: conflict-exposure inter-cluster similarity heatmap and
       intra-cluster cohesion bar.
     - row 3: error-mode inter-cluster similarity heatmap and intra-cluster
       cohesion bar.

3. `georf_m2_cluster_profile_similarity_matrices.csv`
   - Long-format inter-cluster similarities:
     `profile_type, cluster_i, cluster_j, similarity`.

4. `georf_m2_cluster_profile_cohesion.csv`
   - Intra-cluster cohesion by cluster and profile type:
     `profile_type, cluster_id, cohesion`.

5. `georf_m2_cluster_profile_note.md`
   - Chinese reviewer-facing note plus concise English appendix text.

## Metric Definitions

### Inter-Cluster Similarity

For market-access and conflict-exposure profiles:

1. Construct polygon-month profile vectors from the relevant raw panel columns.
2. Standardize each feature across the analysis sample.
3. Compute each cluster centroid as the mean standardized vector.
4. Compute pairwise cosine similarity between cluster centroids.

For error-mode profiles:

1. Compute each cluster's TP, FP, FN, and TN shares from Stage 3 partitioned
   predictions.
2. Compute pairwise cosine similarity between these four-element error-mode
   composition vectors.

### Intra-Cluster Cohesion

For market-access and conflict-exposure profiles:

1. Compute cosine similarity between each polygon-month vector and its cluster
   centroid.
2. Report the cluster mean as the cohesion score.

For error-mode profiles:

1. Assign each evaluated polygon-month observation to one of TP, FP, FN, or TN.
2. Report the dominant error-mode share within the cluster as the cohesion
   score.

This keeps heatmap off-diagonal values focused on between-cluster similarity,
while the right-side bars separately show within-cluster consistency.

## Profile Table Definitions

- `n_polygons`: number of FEWSNET polygons in the refined m2 cluster.
- `countries_or_regions_included`: compact list of countries or, if too long,
  region names with country counts.
- `dominant_region`: region with the largest polygon count in the cluster.
- `crisis_prevalence`: mean `y_true` over evaluated February target observations
  in the cluster.
- `dominant_AEZ`: AEZ one-hot column with the largest cluster-level polygon or
  polygon-month share.
- `market_access_profile`: short categorical label based on cluster percentile
  rank of market access and market distance, such as high access / low distance.
- `conflict_exposure_profile`: short categorical label based on standardized
  event and fatality exposure, such as low, moderate, or high exposure.
- `main_error_mode`: dominant non-TN error mode when present; otherwise
  TN-dominant.

## Reviewer-Safe Language

Use this framing in the appendix:

> We summarize cluster-level descriptive profiles for one representative GeoRF
> m2 local-model partition. Similarity is computed from standardized descriptive
> profiles and from observed prediction error compositions; it is not derived
> from model internals and should not be interpreted as feature importance or
> causal attribution.

## Out of Scope

- GeoDT and GeoXGB cluster profiles.
- All general/m6/m10 mappings.
- Claiming that clusters are optimized for market access, conflict exposure, or
  AEZ composition.
- Re-training local models or changing Stage 3 evaluation.

## Validation

- Unit tests for:
  - cluster membership joins;
  - TP/FP/FN/TN error-mode calculation;
  - cosine similarity matrix shape and diagonal;
  - cohesion range between 0 and 1;
  - February-only filtering for m2.
- Smoke run that writes all five output artifacts to
  `final_artifacts_in_paper_updated/`.
