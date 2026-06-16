# GeoRF Humanitarian Population Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GeoRF-only appendix diagnostic for population-weighted humanitarian metrics over evaluated polygon-month predictions.

**Architecture:** Create one standalone analysis script that loads GeoRF Stage 3 prediction files, joins raw FEWSNET `pop` by admin code and target month, computes population-month metrics, and writes compact paper artifacts under `final_artifacts_in_paper_updated/09_humanitarian_metrics/`. Add focused unittest coverage for formulas and join validation.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, unittest.

---

### Task 1: Add Metric Tests

**Files:**
- Create: `src/tests/test_georf_humanitarian_population_metrics.py`
- Create: `scripts/analyze_georf_humanitarian_population_metrics.py`

- [ ] Write tests for population confusion totals, population-weighted recall and precision, compact delta table formatting, and duplicate FEWSNET key rejection.
- [ ] Run `python3 -m unittest src.tests.test_georf_humanitarian_population_metrics` and confirm the tests fail because the script does not exist yet.

### Task 2: Implement Analysis Script

**Files:**
- Create: `scripts/analyze_georf_humanitarian_population_metrics.py`

- [ ] Implement raw FEWSNET loading from `FEWSNET.csv`, normalizing `admin_code`, `year`, `month`, and `pop`.
- [ ] Implement prediction loading for `fs1`, `fs2`, and `fs3`, normalizing `FEWSNET_admin_code` and `month_start`.
- [ ] Join predictions to population with `many_to_one` validation and raise an explicit error if population coverage is incomplete.
- [ ] Compute horizon-level and month-level population-month metrics for pooled and partitioned models.
- [ ] Build one-row-per-horizon compact table with pooled, partitioned, and delta values.
- [ ] Write CSV, Markdown, PNG, and bilingual note outputs to `final_artifacts_in_paper_updated/09_humanitarian_metrics/`.

### Task 3: Register Artifacts

**Files:**
- Modify: `final_artifacts_in_paper_updated/README.md`

- [ ] Add `09_humanitarian_metrics/` to the folder index.
- [ ] Add a new section listing the six generated humanitarian population metric files.

### Task 4: Generate And Verify

**Files:**
- Generate: `final_artifacts_in_paper_updated/09_humanitarian_metrics/*`

- [ ] Run `python3 -m unittest src.tests.test_georf_humanitarian_population_metrics`.
- [ ] Run `python3 -m py_compile scripts/analyze_georf_humanitarian_population_metrics.py`.
- [ ] Run `python3 scripts/analyze_georf_humanitarian_population_metrics.py`.
- [ ] Verify README paths exist.
- [ ] Run `git diff --check`.
