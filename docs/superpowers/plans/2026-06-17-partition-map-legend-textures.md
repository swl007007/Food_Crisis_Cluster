# Partition Map Legend Textures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the selected paper partition-map figures so partitions remain distinguishable in reduced-size paper/appendix use through compact `cN` legends plus color-and-hatch encoding.

**Architecture:** Keep all changes presentation-only inside the two existing plotting scripts. Add deterministic style builders that map cluster IDs to face colors and hatch patterns, then render each cluster subset separately so GeoPandas/Matplotlib can apply per-partition hatches. Tests exercise style generation and legend-label construction without requiring full shapefile rendering.

**Tech Stack:** Python 3.12, pandas, geopandas, matplotlib, contextily, unittest.

---

### Task 1: Add Compact Global-Map Style Tests

**Files:**
- Modify: `src/tests/test_plot_global_cluster_map_2x2_refined.py`
- Later modified by implementation: `scripts/plot_global_cluster_map_2x2_refined.py`

- [ ] **Step 1: Add tests for hatch support and compact labels**

Append these tests to `GlobalClusterMapSelectionTests` in `src/tests/test_plot_global_cluster_map_2x2_refined.py`:

```python
    def test_partition_styles_support_many_east_africa_partitions_with_hatches(self):
        module = load_script_module()
        panel_data = {}
        for panel in module.PANEL_ORDER:
            if panel == "m10":
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": list(range(12)),
                        "region_group": ["East Africa"] * 12,
                    }
                )
            else:
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": [0],
                        "region_group": ["West Africa"],
                    }
                )
        cluster_regions = {
            (panel, int(row.cluster_id)): str(row.region_group)
            for panel, df in panel_data.items()
            for row in df.itertuples(index=False)
        }

        key_to_style, summary = module.build_partition_styles(panel_data, cluster_regions)

        m10_styles = [key_to_style[("m10", cluster_id)] for cluster_id in range(12)]
        self.assertEqual(len(m10_styles), 12)
        self.assertEqual(summary["m10"][11], "East Africa")
        self.assertGreater(len({style.hatch for style in m10_styles}), 1)
        self.assertGreater(len({style.facecolor for style in m10_styles}), 1)

    def test_compact_legend_labels_are_cluster_ids_only(self):
        module = load_script_module()
        labels = module.compact_cluster_labels([0, 2, 13])

        self.assertEqual(labels, ["c0", "c2", "c13"])
        self.assertNotIn("m2", " ".join(labels))
        self.assertNotIn("WA", " ".join(labels))
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m unittest src.tests.test_plot_global_cluster_map_2x2_refined -v
```

Expected: FAIL because `build_partition_styles` and `compact_cluster_labels` are not defined yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add src/tests/test_plot_global_cluster_map_2x2_refined.py
git commit -m "test global partition map compact texture legend"
```

---

### Task 2: Implement Global-Map Color-and-Hatch Styling

**Files:**
- Modify: `scripts/plot_global_cluster_map_2x2_refined.py`
- Test: `src/tests/test_plot_global_cluster_map_2x2_refined.py`

- [ ] **Step 1: Add style data structures and hatch constants**

In `scripts/plot_global_cluster_map_2x2_refined.py`, update imports:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
```

Remove the unused `ListedColormap` import:

```python
from matplotlib.colors import ListedColormap
```

Add this after `REGION_SUBPALETTES`:

```python
HATCH_PATTERNS = ("", "///", "\\\\\\", "xxx", "...", "++", "--", "||", "oo", "**")


@dataclass(frozen=True)
class PartitionStyle:
    """Matplotlib style assigned to one panel-local cluster."""

    facecolor: str
    hatch: str
```

- [ ] **Step 2: Replace palette builder with style builder**

Replace `build_partition_palette` with:

```python
def build_partition_styles(
    panel_data: Dict[str, gpd.GeoDataFrame],
    cluster_regions: Dict[Tuple[str, int], str],
) -> Tuple[Dict[Tuple[str, int], PartitionStyle], Dict[str, Dict[int, str]]]:
    key_to_style: Dict[Tuple[str, int], PartitionStyle] = {}
    summary: Dict[str, Dict[int, str]] = {}
    for panel in PANEL_ORDER:
        panel_clusters = sorted(panel_data[panel]["cluster_id"].unique().tolist())
        summary[panel] = {int(cluster_id): cluster_regions[(panel, int(cluster_id))] for cluster_id in panel_clusters}
        region_to_clusters: Dict[str, list[int]] = {region: [] for region in REGION_ORDER}
        for cluster_id in panel_clusters:
            region_to_clusters[cluster_regions[(panel, int(cluster_id))]].append(int(cluster_id))
        for region in REGION_ORDER:
            clusters = region_to_clusters[region]
            subpalette = REGION_SUBPALETTES[region]
            capacity = len(subpalette) * len(HATCH_PATTERNS)
            if len(clusters) > capacity:
                raise ValueError(
                    f"Not enough color/hatch styles for {panel} {region}: "
                    f"{len(clusters)} clusters, {capacity} available combinations"
                )
            for idx, cluster_id in enumerate(clusters):
                color = subpalette[idx % len(subpalette)]
                hatch = HATCH_PATTERNS[idx // len(subpalette)]
                key_to_style[(panel, cluster_id)] = PartitionStyle(facecolor=color, hatch=hatch)
    return key_to_style, summary
```

- [ ] **Step 3: Add compact legend helper**

Add this helper below `build_partition_styles`:

```python
def compact_cluster_labels(cluster_ids: Iterable[int]) -> list[str]:
    return [f"c{int(cluster_id)}" for cluster_id in sorted(cluster_ids)]
```

- [ ] **Step 4: Remove obsolete color-index helper**

Delete `add_partition_color_index`; it is no longer needed because each cluster subset is plotted with a direct style.

- [ ] **Step 5: Update panel plotting to draw one cluster at a time**

In `plot_model_grid`, replace:

```python
    cmap, key_to_idx, summary, key_to_color = build_partition_palette(panel_data, cluster_regions)

    plot_data = {
        panel: add_partition_color_index(gdf, panel, key_to_idx).to_crs(epsg=3857)
        for panel, gdf in panel_data.items()
    }
```

with:

```python
    key_to_style, summary = build_partition_styles(panel_data, cluster_regions)

    plot_data = {panel: gdf.to_crs(epsg=3857) for panel, gdf in panel_data.items()}
```

Then replace the `gdf.plot(...)` block inside the panel loop with:

```python
        for cluster_id in sorted(gdf["cluster_id"].unique().tolist()):
            style = key_to_style[(panel, int(cluster_id))]
            subset = gdf[gdf["cluster_id"].eq(cluster_id)]
            subset.plot(
                ax=ax,
                color=style.facecolor,
                edgecolor="#f7f7f7",
                linewidth=0.10,
                hatch=style.hatch,
                alpha=0.92 if add_basemap else 1.0,
                zorder=2,
            )
```

- [ ] **Step 6: Replace long shared legend with compact cluster-ID legend**

Replace the `color_to_labels` and `legend_handles` construction with:

```python
    legend_clusters = sorted(
        {int(cluster_id) for panel in PANEL_ORDER for cluster_id in panel_data[panel]["cluster_id"].unique().tolist()}
    )
    representative_styles: Dict[int, PartitionStyle] = {}
    for cluster_id in legend_clusters:
        for panel in PANEL_ORDER:
            key = (panel, cluster_id)
            if key in key_to_style:
                representative_styles[cluster_id] = key_to_style[key]
                break

    legend_handles = [
        mpatches.Patch(
            facecolor=representative_styles[cluster_id].facecolor,
            hatch=representative_styles[cluster_id].hatch,
            edgecolor="black",
            linewidth=0.35,
            label=f"c{cluster_id}",
        )
        for cluster_id in legend_clusters
    ]
```

Update the `fig.legend(...)` call to:

```python
    fig.legend(
        handles=legend_handles,
        title="Partition ID (panel-specific)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.040),
        ncol=min(10, max(1, len(legend_handles))),
        frameon=True,
        fontsize=8.5,
        title_fontsize=10,
        columnspacing=1.1,
        handlelength=1.8,
        handleheight=1.0,
        handletextpad=0.45,
    )
    fig.text(
        0.5,
        0.012,
        "Cluster IDs are interpreted within each panel; hue families indicate dominant geographic region.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
```

Update `plt.tight_layout(...)` to reserve enough bottom space:

```python
    plt.tight_layout(rect=(0.02, 0.16, 0.98, 0.94))
```

- [ ] **Step 7: Update old palette test name if needed**

Replace the existing `test_partition_palette_supports_ten_east_africa_clusters` body so it calls `build_partition_styles`:

```python
    def test_partition_styles_support_ten_east_africa_clusters(self):
        module = load_script_module()
        panel_data = {}
        for panel in module.PANEL_ORDER:
            if panel == "m10":
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": list(range(10)),
                        "region_group": ["East Africa"] * 10,
                    }
                )
            else:
                panel_data[panel] = pd.DataFrame(
                    {
                        "cluster_id": [0],
                        "region_group": ["West Africa"],
                    }
                )
        cluster_regions = {
            (panel, int(row.cluster_id)): str(row.region_group)
            for panel, df in panel_data.items()
            for row in df.itertuples(index=False)
        }

        key_to_style, summary = module.build_partition_styles(panel_data, cluster_regions)

        self.assertEqual(len([key for key in key_to_style if key[0] == "m10"]), 10)
        self.assertEqual(summary["m10"][9], "East Africa")
```

- [ ] **Step 8: Run global-map unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m unittest src.tests.test_plot_global_cluster_map_2x2_refined -v
```

Expected: PASS.

- [ ] **Step 9: Commit global-map implementation**

```bash
git add scripts/plot_global_cluster_map_2x2_refined.py src/tests/test_plot_global_cluster_map_2x2_refined.py
git commit -m "improve global partition map legends"
```

---

### Task 3: Add Adjacency-Refinement Style Tests

**Files:**
- Modify: `src/tests/test_plot_georf_m2_adjacency_refinement.py`
- Later modified by implementation: `scripts/plot_georf_m2_adjacency_refinement.py`

- [ ] **Step 1: Add tests for cluster styles and legend labels**

Append these tests to `GeoRFM2AdjacencyRefinementTests`:

```python
    def test_cluster_styles_assign_color_and_hatch(self):
        styles = refinement.cluster_style_map(range(15))

        self.assertEqual(sorted(styles), list(range(15)))
        self.assertGreater(len({style.facecolor for style in styles.values()}), 1)
        self.assertGreater(len({style.hatch for style in styles.values()}), 1)

    def test_compact_cluster_label_uses_c_prefix(self):
        self.assertEqual(refinement.compact_cluster_label(12), "c12")
```

- [ ] **Step 2: Run adjacency tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m unittest src.tests.test_plot_georf_m2_adjacency_refinement -v
```

Expected: FAIL because `cluster_style_map` and `compact_cluster_label` are not defined yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add src/tests/test_plot_georf_m2_adjacency_refinement.py
git commit -m "test adjacency refinement texture legend"
```

---

### Task 4: Implement Adjacency-Refinement Color-and-Hatch Styling

**Files:**
- Modify: `scripts/plot_georf_m2_adjacency_refinement.py`
- Test: `src/tests/test_plot_georf_m2_adjacency_refinement.py`

- [ ] **Step 1: Add dataclass and hatch constants**

In `scripts/plot_georf_m2_adjacency_refinement.py`, update imports:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
```

Add this after `CLUSTER_PALETTE`:

```python
HATCH_PATTERNS = ("", "///", "\\\\\\", "xxx", "...", "++", "--", "||", "oo", "**")


@dataclass(frozen=True)
class ClusterStyle:
    """Matplotlib style assigned to one GeoRF m2 cluster."""

    facecolor: str
    hatch: str
```

- [ ] **Step 2: Replace color map helper with style helpers**

Replace `cluster_color_map` with:

```python
def compact_cluster_label(cluster_id: int) -> str:
    return f"c{int(cluster_id)}"


def cluster_style_map(cluster_ids: Iterable[int]) -> dict[int, ClusterStyle]:
    clusters = sorted({int(cluster_id) for cluster_id in cluster_ids})
    capacity = len(CLUSTER_PALETTE) * len(HATCH_PATTERNS)
    if len(clusters) > capacity:
        raise ValueError(f"Style set supports {capacity} clusters, got {len(clusters)}")
    return {
        cluster_id: ClusterStyle(
            facecolor=CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)],
            hatch=HATCH_PATTERNS[idx // len(CLUSTER_PALETTE)],
        )
        for idx, cluster_id in enumerate(clusters)
    }
```

- [ ] **Step 3: Apply styles in before/after panels**

In `plot_refinement_figure`, replace:

```python
    color_lookup = cluster_color_map(all_clusters)
```

with:

```python
    style_lookup = cluster_style_map(all_clusters)
```

Replace the before/after cluster loop with:

```python
        for cluster_id, style in style_lookup.items():
            subset = gdf[gdf[cluster_column].eq(cluster_id)]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=style.facecolor,
                    edgecolor="white",
                    linewidth=0.08,
                    hatch=style.hatch,
                )
```

- [ ] **Step 4: Make reassigned polygons red and patterned**

Replace:

```python
        reassigned.plot(ax=ax, color="#d73027", edgecolor="#111111", linewidth=0.18)
```

with:

```python
        reassigned.plot(
            ax=ax,
            color="#d73027",
            edgecolor="#111111",
            linewidth=0.22,
            hatch="xxx",
        )
```

- [ ] **Step 5: Use compact legend labels and visible handles**

Replace legend handle construction with:

```python
    handles = [
        mpatches.Patch(
            facecolor=style_lookup[cluster_id].facecolor,
            hatch=style_lookup[cluster_id].hatch,
            edgecolor="black",
            linewidth=0.35,
            label=compact_cluster_label(cluster_id),
        )
        for cluster_id in sorted(style_lookup)
    ]
    handles.append(
        mpatches.Patch(
            facecolor="#d73027",
            hatch="xxx",
            edgecolor="black",
            linewidth=0.35,
            label="Reassigned",
        )
    )
```

Update `fig.legend(...)` to:

```python
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        fontsize=9,
        handlelength=1.8,
        handleheight=1.0,
        columnspacing=1.0,
        handletextpad=0.45,
        bbox_to_anchor=(0.5, -0.010),
    )
```

- [ ] **Step 6: Run adjacency unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m unittest src.tests.test_plot_georf_m2_adjacency_refinement -v
```

Expected: PASS.

- [ ] **Step 7: Commit adjacency implementation**

```bash
git add scripts/plot_georf_m2_adjacency_refinement.py src/tests/test_plot_georf_m2_adjacency_refinement.py
git commit -m "improve adjacency refinement map legend"
```

---

### Task 5: Regenerate Paper Figures and Verify Outputs

**Files:**
- Regenerate: `final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_georf_refined.png`
- Regenerate: `final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_geodt_refined.png`
- Regenerate: `final_artifacts_in_paper_updated/05_partition_diagnostics/georf_m2_adjacency_refinement_1x3.png`

- [ ] **Step 1: Regenerate global GeoRF/GeoDT 2x2 figures**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/plot_global_cluster_map_2x2_refined.py --output-dir final_artifacts_in_paper_updated/01_main_results
```

Expected:

- Console reports both GeoRF and GeoDT figures saved.
- Existing output filenames are overwritten in `final_artifacts_in_paper_updated/01_main_results/`.
- No model training command runs.

- [ ] **Step 2: Regenerate GeoRF m2 adjacency-refinement figure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/plot_georf_m2_adjacency_refinement.py --output-dir final_artifacts_in_paper_updated/05_partition_diagnostics
```

Expected:

- Console reports summary and figure written.
- Existing output filename is overwritten in `final_artifacts_in_paper_updated/05_partition_diagnostics/`.
- The CSV and note are preserved or regenerated with identical metric semantics.

- [ ] **Step 3: Run focused unit tests together**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m unittest src.tests.test_plot_global_cluster_map_2x2_refined src.tests.test_plot_georf_m2_adjacency_refinement -v
```

Expected: PASS.

- [ ] **Step 4: Run artifact reproducibility checker**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/verify_current_results_reproducibility.py
```

Expected: PASS or a report that only expected regenerated image hashes changed. If the checker treats hash changes as failures, record the exact changed PNGs and confirm no CSV/table/model artifacts changed unexpectedly.

- [ ] **Step 5: Manually inspect generated PNGs**

Use local image inspection to check:

- `global_cluster_map_2x2_georf_refined.png`: compact `cN` legend, visible hatches, readable note.
- `global_cluster_map_2x2_geodt_refined.png`: compact `cN` legend, visible hatches, readable note.
- `georf_m2_adjacency_refinement_1x3.png`: before/after use color+hatch, `Reassigned` remains red and obvious.

- [ ] **Step 6: Commit regenerated artifacts**

```bash
git add final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_georf_refined.png \
        final_artifacts_in_paper_updated/01_main_results/global_cluster_map_2x2_geodt_refined.png \
        final_artifacts_in_paper_updated/05_partition_diagnostics/georf_m2_adjacency_refinement_1x3.png
git commit -m "regenerate partition maps with compact legends"
```

---

### Task 6: Final Working-Tree and Diff Review

**Files:**
- Review all files changed by Tasks 1-5.

- [ ] **Step 1: Check git status**

Run:

```bash
git status --short
```

Expected: no uncommitted changes, or only `.superpowers/brainstorm/` local companion files that are ignored.

- [ ] **Step 2: Review recent commits**

Run:

```bash
git log --oneline -6
```

Expected: recent commits include the spec commit plus test, implementation, and artifact-regeneration commits from this plan.

- [ ] **Step 3: Summarize verification evidence**

Final response should report:

- The regenerated PNG paths.
- Unit test command and PASS status.
- Reproducibility checker outcome.
- Any artifact hash changes expected from image regeneration.
