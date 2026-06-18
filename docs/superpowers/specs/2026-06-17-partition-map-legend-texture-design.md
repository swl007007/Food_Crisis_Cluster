# Partition Map Legend and Texture Redesign

## Context

The current paper artifact figures are readable at full resolution but weak for paper/appendix use after scaling:

- `final_artifacts_in_paper_updated/05_partition_diagnostics/georf_m2_adjacency_refinement_1x3.png`
- `final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_georf_refined.png`
- `final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_geodt_refined.png`

The main problems are small legends, long legend labels, and partition colors that are too similar when viewed small, printed, or read by color-limited readers.

## Confirmed Design Direction

Use **region-family colors plus partition-level hatch patterns**.

- Dominant region remains encoded by hue family.
- Different partitions within the same region are separated by shade plus hatch pattern.
- Legends show only compact partition IDs such as `c0`, `c1`, `c2`.
- Cluster IDs in the 2x2 global figures are interpreted within each panel; this constraint should be stated in the figure note or caption, not in the legend text.
- The design optimizes for paper/appendix readability, including reduced-size viewing and black-and-white printing.

## Figure-Specific Behavior

### Global 2x2 GeoRF and GeoDT Maps

Target script:

- `scripts/plot_global_cluster_map_2x2_refined.py`

Behavior:

- Keep the current 2x2 panel structure: `General`, `m2`, `m6`, `m10`.
- Keep region-family hue assignment based on dominant region.
- Add a deterministic hatch assignment for partitions.
- Draw map polygons with both face color and hatch pattern.
- Replace long shared legend labels with compact labels: `c0`, `c1`, etc.
- Increase legend readability through larger font, larger handles, and less compressed spacing.
- Add a concise note in the figure or companion metadata that cluster IDs are panel-specific.

### GeoRF m2 Adjacency Refinement Figure

Target script:

- `scripts/plot_georf_m2_adjacency_refinement.py`

Behavior:

- Apply color plus hatch encoding consistently to before/after cluster panels.
- Use compact legend labels: `c0`, `c1`, etc.
- Keep the reassigned-polygons panel visually distinct.
- Highlight reassigned polygons with strong red plus a distinct pattern or outline.
- Keep the reassignment count annotation visible and readable.

## Scope

In scope:

- Update the two plotting scripts above.
- Add or update tests for palette/hatch assignment and compact legend labels.
- Regenerate the three PNG artifacts listed in Context.
- Preserve existing output filenames unless implementation evidence shows a filename change is necessary.

Out of scope:

- Model reruns.
- Cluster mapping changes.
- Metric or table changes.
- Reworking unrelated paper figures.
- Changing shapefile or basemap inputs.

## Data Flow

The redesign is presentation-only:

1. Existing cluster mapping CSVs are loaded exactly as before.
2. Existing cluster-to-region dominance logic remains the source of region color family assignment.
3. A deterministic style builder assigns a face color and hatch to each plotted cluster.
4. Plotting uses the existing geometries and output paths.
5. Tests validate style assignment and legend labels without requiring full shapefile rendering.

## Error Handling

- If a region has more partitions than available shade/hatch combinations, fail with a clear error naming the panel and region.
- If hatch rendering is unsupported by a plotting backend, the script should still save a valid figure with color and edge outlines; tests should cover the deterministic style data, not pixel-perfect hatch rendering.
- Basemap failures remain non-fatal as in the current global-map script.

## Testing

Focused tests should cover:

- The style builder can support current high-count regions by combining shades and hatch patterns.
- Legend labels are compact cluster IDs, not long `panel cN (region)` strings.
- The adjacency-refinement reassigned legend entry remains present.
- Existing mapping-selection behavior is unchanged.

Manual verification should include:

- Open the regenerated PNGs at paper-like reduced size.
- Confirm legends are readable.
- Confirm partitions within the same region are distinguishable by both shade and hatch.
- Confirm reassigned polygons remain easy to identify in the adjacency figure.
