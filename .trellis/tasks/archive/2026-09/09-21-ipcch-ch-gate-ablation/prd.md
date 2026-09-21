# IPCCH CH-heterogeneity and split-gate ablation

## Why

The IPCCH GeoRF baseline (`IPCCHGeoRFExperiment`, run `ipcch-v1-20260920d`) learned no
partition and did not beat persistence. Two candidate explanations were never
separated:

1. **The split gate is too strict.** Class-1 F1 must improve by strictly more than
   `MIN_CLASS_1_IMPROVEMENT_THRESHOLD = 0.01` for a split to be accepted. In the FEWS NET
   line, 16 of 19 root-only candidates missed that gate with gains in [0.005, 0.01) —
   near-misses, not degenerate failures.
2. **The cohort is too heterogeneous to pool.** The panel mixes IPC areas with Cadre
   Harmonisé (CH) areas. Measured from the source:

   | cohort | areas | rows with a phase | class-1 (crisis) rate |
   |---|---|---|---|
   | non-CH (`admin_code < 100000`) | 4,919 | 25,155 | **0.6207** |
   | CH (`admin_code >= 100000`) | 1,308 | 18,558 | **0.1367** |

   A 4.5x difference in base rate across 42.5% of the labeled data. A single pooled
   model, and a single global split gate, are being asked to serve both regimes.

   These counts are a raw-source approximation (`overall_phase` non-null), which totals
   43,713 against the pipeline's stricter validity rule of `EXPECTED_VALID = 42,695`.
   The authoritative per-cohort split is recomputed from the built target ledger under
   R4 and must be reported from there, not from this table.

This task separates the two explanations instead of confounding them.

## Frozen design

**Full factorial: 3 cohorts x 2 gates = 6 cells.** Every cell is one complete IPCCH
pipeline run; nothing else changes.

| | gate 0.01 (released) | gate 0.005 (relaxed) |
|---|---|---|
| **all** areas | C1 (reproduces `ipcch-v1-20260920d`) | C2 |
| **non-CH** only | C3 | C4 |
| **CH** only | C5 | C6 |

The 2x2 over {all, non-CH} x {0.01, 0.005} answers "gate or cohort?". The CH-only row
answers the follow-up the 2x2 cannot: whether CH is a *different problem* with its own
learnable structure, or simply noisier data. That row was added because the measured
base-rate gap makes "pull CH out and model it separately" a live option, not because
dropping CH is assumed to be the fix.

### Cohort definition

CH is exactly `admin_code >= 100000`. No country-level special-casing: the Central
African Republic is split by this rule (72 of its 298 areas are >= 100000) and that is
accepted as-is. The rule covers 19 countries — Benin, Burkina Faso, Cabo Verde,
Cameroon, CAR (partial), Chad, Cote d'Ivoire, Gambia, Ghana, Guinea, Guinea-Bissau,
Liberia, Mali, Mauritania, Niger, Nigeria, Senegal, Sierra Leone, Togo.

The filter is applied to the **cohort**, i.e. before Stage 1 candidate learning, Stage 3
fitting and evaluation alike. It is not an evaluation-time mask over a model trained on
everything.

### Everything else is held fixed

Carried unchanged from `ipcch-v1-20260920d`, so that the only moving parts are the two
factors above:

* Horizons 1, 3, 6, 12; main target schedule `h1 2023-02..2025-12`,
  `h3 2023-04..2025-12`, `h6 2023-07..2025-12`, `h12 2024-01..2025-12`.
* Four arms: `partitioned_rf`, `pooled_rf`, `xgb`, `persistence`. XGB is retained so
  every cell lines up arm-for-arm with the prior baseline's report structure, even
  though the FEWS NET line treats GeoXGB as legacy.
* Two cohorts in the reporting sense: `E_all` (every valid test key) and `E_persist`
  (the history-available subset, where all four arms share identical keys).
* Decision threshold 0.5, reference arm `partitioned_rf`.
* The pinned `GeoRFBaseline` release, extracted and hash-verified per run.

## Requirements

* **R1.** Run all six cells. Each is a fresh run directory; none overwrites another.
* **R2.** The only intended differences between cells are the cohort filter and the gate
  value. Record the effective value of every `REPORTED_CONFIG_KEYS` entry per cell and
  fail the cell if anything other than `MIN_CLASS_1_IMPROVEMENT_THRESHOLD` differs from
  C1's.
* **R3.** The gate override must be verified from the **consuming** modules, not from
  `config`. `GeoRFBaseline/src/partition/transformation.py:13` and
  `src/tests/sig_test.py:12` both use `from config import *`, and `transformation.py:17`
  re-exports `sig_test`'s copy, so the value is bound into three namespaces at import
  time. Setting it on `config` alone would leave the run gating at 0.01 while reporting
  0.005. A cell whose consuming-module readback does not equal its declared gate fails.
* **R4.** The cohort filter is applied once, at cohort construction, and its effect is
  recorded per cell: areas kept/dropped, labeled rows kept/dropped, class-1 counts and
  rate for each side. A cell whose recorded cohort does not match the declared filter
  fails.
* **R5.** Produce a like-for-like reference for the CH-dropped cells by recomputing the
  prior baseline's metrics on the non-CH key subset of its **existing stored per-row
  predictions**. No model is refitted for this. This is what distinguishes "dropping CH
  improved the model" from "dropping CH swapped in an easier cohort".
* **R6.** Report per cell, per horizon, per arm: class-1 precision/recall/F1 from pooled
  confusion counts, plus the number of terminal partitions actually learned and the
  split-gate evaluation counts (how many candidate splits were considered, how many
  cleared the gate). The partition count is the direct measurement of whether the gate
  change did anything.
* **R7.** Persistence is computed independently of the RF arms and on the same keys
  within each cell, exactly as in the prior baseline.
* **R8.** No new feature engineering, no threshold tuning, no recipe search, no model
  family beyond the four existing arms. This is a diagnostic ablation, not a new
  experiment.

## Acceptance

* **A1.** Six run directories exist, each with a complete manifest, and R2's
  cross-cell settings comparison passes.
* **A2.** For each cell, the consuming-module gate readback equals the declared gate
  (R3), demonstrated for all three namespaces.
* **A3.** Cohort counts per cell reconcile exactly to the `admin_code >= 100000` rule
  against the source, including the CAR split.
* **A4.** C1 reproduces `ipcch-v1-20260920d`'s headline metrics. If it does not, the
  discrepancy is investigated and reported before any other cell is interpreted — a
  failed reproduction invalidates the whole comparison.
* **A5.** The non-CH-restricted recomputation of the prior baseline (R5) is reported
  beside C3/C4.
* **A6.** A single results table carries all six cells with partition counts, and states
  plainly which factor moved the result and by how much — including "neither" if that is
  what the numbers say.

## Out of scope

Deciding what to do about CH. This task measures whether CH is separable and whether the
gate binds; choosing a modelling response (separate models, hierarchical pooling,
CH-specific thresholds, or abandoning CH) is a later decision informed by these numbers.

## Status of the surrounding programme

The FEWS NET persistence-correction line was declared dead on 2026-09-21 after a
pre-registered `complete_fail`. This task is diagnostic work on the IPCCH line, which
together with the expert-gap window (2025) is the current direction. It is deliberately
smaller and cheaper than the experiments that preceded it: the prior full IPCCH run took
19 minutes, so six cells is roughly two hours.
