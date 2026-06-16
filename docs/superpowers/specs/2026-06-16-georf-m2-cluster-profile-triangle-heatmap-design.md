# GeoRF m2 Cluster Profile Triangle Heatmap Design

## Goal

Reduce the checkerboard appearance of the GeoRF m2 cluster-profile similarity
figure while preserving the same diagnostic content.

## Scope

- Modify only the representative GeoRF m2 cluster-profile diagnostic figure.
- Keep the existing three profile types: market access, conflict exposure, and
  error mode.
- Keep the existing cohesion bars.
- Do not introduce hierarchical clustering, new similarity metrics, or new
  model outputs.

## Design

The three inter-cluster similarity matrices are symmetric, so the figure will
show only the lower triangle, including the diagonal. The upper triangle will be
masked to the plot background. This removes duplicated visual information and
reduces the checkerboard effect without changing the underlying CSV matrices.

Cluster order remains `0..12`. This avoids adding a new ordering method that
would need extra explanation in the paper.

## Artifacts

Regenerate:

- `final_artifacts_in_paper_updated/georf_m2_cluster_profile_similarity.png`

No CSV schema changes are required. The long-format matrix CSV should remain
complete with all pairwise similarities.

## Validation

- Add a unit test for the upper-triangle mask.
- Run the existing cluster-profile test suite.
- Regenerate the figure and visually confirm that only one triangle is shown
  for each similarity matrix.
