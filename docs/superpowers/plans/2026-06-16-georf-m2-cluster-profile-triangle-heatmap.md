# GeoRF m2 Cluster Profile Triangle Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-render the GeoRF m2 cluster-profile similarity figure with lower-triangle heatmaps to reduce the checkerboard visual pattern.

**Architecture:** Add one small mask helper to `scripts/analyze_georf_m2_cluster_profiles.py`, use it inside `plot_similarity_figure()`, and add one unit test. The CSV matrices remain full pairwise outputs; only the rendered PNG masks duplicate upper-triangle cells.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib/seaborn, unittest.

---

### Task 1: Lower-Triangle Similarity Heatmaps

**Files:**
- Modify: `scripts/analyze_georf_m2_cluster_profiles.py`
- Modify: `src/tests/test_georf_m2_cluster_profiles.py`

- [ ] **Step 1: Add the mask unit test**

Add a test that builds a `3 x 3` matrix and verifies that cells above the
diagonal are masked while the lower triangle and diagonal remain visible.

- [ ] **Step 2: Run the test suite**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected before implementation: failure because `upper_triangle_mask` is not
defined.

- [ ] **Step 3: Implement the mask helper and apply it to plotting**

Add `upper_triangle_mask(matrix)` and use it in both seaborn and matplotlib
fallback paths inside `plot_similarity_figure()`.

- [ ] **Step 4: Run verification and regenerate artifacts**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
python3 -m py_compile scripts/analyze_georf_m2_cluster_profiles.py
"/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe" scripts/analyze_georf_m2_cluster_profiles.py
```

Expected: tests pass, script compiles, and the figure is regenerated.

- [ ] **Step 5: Inspect output**

Check:

```bash
file final_artifacts_in_paper_updated/georf_m2_cluster_profile_similarity.png
git diff --check -- scripts/analyze_georf_m2_cluster_profiles.py src/tests/test_georf_m2_cluster_profiles.py final_artifacts_in_paper_updated/georf_m2_cluster_profile_similarity.png
```

Expected: PNG exists and `git diff --check` reports no whitespace errors.
