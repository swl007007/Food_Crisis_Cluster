# GeoRF Probability Compact Appendix Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an appendix-ready 3-row compact table from the GeoRF probability bootstrap CI diagnostics.

**Architecture:** Extend `scripts/analyze_georf_probability_uncertainty.py` with pure helpers for formatting compact delta cells and building the compact table. The existing full CSV outputs remain unchanged; the compact CSV/Markdown files are additional presentation artifacts.

**Tech Stack:** Python 3.12, pandas, unittest.

---

### Task 1: Compact Appendix Table

**Files:**
- Modify: `scripts/analyze_georf_probability_uncertainty.py`
- Modify: `src/tests/test_georf_probability_uncertainty.py`
- Modify: `final_artifacts_in_paper_updated/README.md`

- [ ] Add tests for formatted cells and 3-row compact table shape.
- [ ] Implement compact table helpers.
- [ ] Write compact CSV and Markdown outputs in `main()`.
- [ ] Regenerate compact artifacts from the existing diagnostics.
- [ ] Run tests and `git diff --check`.
