# GeoRF False-Negative Error-Mode Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a GeoRF partitioned-model false-negative error-mode note and compact evidence tables for crisis-state misses in specified hotspots.

**Architecture:** Add one standalone analysis script that loads GeoRF predictions, joins evaluated observations to the FEWSNET model input panel, filters partitioned false negatives, computes hotspot and error-mode proxy summaries, and writes paper artifacts under `final_artifacts_in_paper_updated/10_false_negative_error_modes/`. Add focused unittest coverage for filtering and proxy logic.

**Tech Stack:** Python 3, pandas, numpy, unittest.

---

### Task 1: Add Tests First

**Files:**
- Create: `src/tests/test_georf_false_negative_error_modes.py`
- Create: `scripts/analyze_georf_false_negative_error_modes.py`

- [ ] Test that hotspot assignment handles exact countries and latitude-based Afghanistan/Mozambique subregions.
- [ ] Test that false-negative summaries only count `y_true = 1` and `y_pred_partitioned = 0`.
- [ ] Test that the lagged-outcome proxy maps scopes to the expected lag phase column and flags lagged non-crisis states.
- [ ] Test that neighbor-context summaries compute mixed-neighbor crisis-state shares from a small adjacency dictionary.
- [ ] Run `python3 -m unittest src.tests.test_georf_false_negative_error_modes` and confirm it fails because the script is not implemented.

### Task 2: Implement Analysis Script

**Files:**
- Create: `scripts/analyze_georf_false_negative_error_modes.py`

- [ ] Implement prediction loading for GeoRF `fs1`, `fs2`, and `fs3`.
- [ ] Implement FEWSNET model-panel loading with selected conflict, price, lagged outcome, population, country, and coordinate columns.
- [ ] Validate model-panel keys by `FEWSNET_admin_code` and `month_start`.
- [ ] Join predictions to the model panel and require full coverage.
- [ ] Assign hotspot labels using exact country and latitude-proxy definitions.
- [ ] Filter to partitioned false negatives.
- [ ] Compute hotspot-by-horizon summary metrics.
- [ ] Compute hotspot-level error-mode proxy metrics.
- [ ] Write CSV, Markdown, and bilingual note outputs.

### Task 3: Register Artifacts

**Files:**
- Modify: `final_artifacts_in_paper_updated/README.md`

- [ ] Add `10_false_negative_error_modes/` to the folder index.
- [ ] Add a new section listing the four generated false-negative error-mode files.

### Task 4: Generate And Verify

**Files:**
- Generate: `final_artifacts_in_paper_updated/10_false_negative_error_modes/*`

- [ ] Run `python3 -m unittest src.tests.test_georf_false_negative_error_modes`.
- [ ] Run `python3 -m py_compile scripts/analyze_georf_false_negative_error_modes.py`.
- [ ] Run `python3 scripts/analyze_georf_false_negative_error_modes.py`.
- [ ] Verify README paths exist.
- [ ] Run `git diff --check`.
