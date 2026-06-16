# GeoRF Probability and Uncertainty Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export GeoRF class-1 probabilities from Stage 3 and generate reviewer-facing paired bootstrap, Brier score, reliability, and uncertainty diagnostics for GeoRF fs1/fs2/fs3.

**Architecture:** Patch the existing Stage 3 GeoRF comparison script to add probability columns without changing hard predictions. Add a separate diagnostics script that consumes prediction artifacts, joins country/region lookup, computes clustered bootstrap CIs and probability diagnostics, and writes final appendix artifacts.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, geopandas for region lookup, unittest.

---

### Task 1: Stage 3 Probability Export

**Files:**
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- Create/modify: `src/tests/test_partition_comparison_contract.py`

- [ ] Add tests for class-1 probability extraction and partitioned probability fallback.
- [ ] Implement `predict_class1_probability`, `predict_pooled_probability`, and `predict_partitioned_probability`.
- [ ] Add `y_prob_pooled` and `y_prob_partitioned` to `predictions_monthly.csv`.
- [ ] Verify existing hard predictions are unchanged by the probability helper.

### Task 2: Probability Diagnostics Script

**Files:**
- Create: `scripts/analyze_georf_probability_uncertainty.py`
- Create: `src/tests/test_georf_probability_uncertainty.py`

- [ ] Add tests for Brier score, hard-label metrics, reliability bins, country-clustered bootstrap, and region filtering.
- [ ] Implement pure helper functions first.
- [ ] Implement CLI defaults for GeoRF fs1/fs2/fs3 and `final_artifacts_in_paper_updated/`.
- [ ] Write CSV, PNG, and markdown note artifacts.

### Task 3: Regenerate GeoRF Stage 3 Predictions

**Files:**
- Regenerate: `result_partition_k40_compare_GF_fs1/predictions_monthly.csv`
- Regenerate: `result_partition_k40_compare_GF_fs2/predictions_monthly.csv`
- Regenerate: `result_partition_k40_compare_GF_fs3/predictions_monthly.csv`

- [ ] Rerun the standard GeoRF Stage 3 comparison for fs1/fs2/fs3 using the existing refined fixed partitions and month-indicator setup.
- [ ] Confirm each regenerated predictions file has `y_prob_pooled` and `y_prob_partitioned`.
- [ ] Confirm monthly hard-label metrics stay compatible with the current outputs.

### Task 4: Generate Final Appendix Artifacts

**Files:**
- Modify: `final_artifacts_in_paper_updated/README.md`
- Create: `final_artifacts_in_paper_updated/georf_probability_bootstrap_ci.csv`
- Create: `final_artifacts_in_paper_updated/georf_probability_bootstrap_region_ci.csv`
- Create: `final_artifacts_in_paper_updated/georf_probability_brier_reliability.csv`
- Create: `final_artifacts_in_paper_updated/georf_probability_uncertainty_summary.csv`
- Create: `final_artifacts_in_paper_updated/georf_probability_reliability.png`
- Create: `final_artifacts_in_paper_updated/georf_probability_uncertainty_note.md`

- [ ] Run the probability diagnostics script.
- [ ] Inspect row counts and required columns.
- [ ] Visually inspect the reliability figure.
- [ ] Run `git diff --check` on touched files.
