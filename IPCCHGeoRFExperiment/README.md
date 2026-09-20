# IPCCH binary crisis GeoRF pipeline

Baseline experiment for the task `.trellis/tasks/09-19-ipcch-binary-georf-pipeline`.
Specification: that task's `prd.md` (authoritative), `design.md`, `implement.md`.
Every scope change and finding during implementation: that task's `DECISIONS_LOG.md`.

Authoritative run: `runs/ipcch-v1-20260920d/`. Run artifacts are git-ignored.

---

# Verdict

**Stage 1 learned no spatial partition, and no learned arm beat persistence reliably.**

Crisis-class F1 on `E_persist` (all four arms scored on identical keys, main period):

| horizon | partitioned RF | pooled RF | XGB | persistence |
|---|---|---|---|---|
| 1 | 0.6619 | 0.6664 | 0.6789 | **0.6814** |
| 3 | 0.6470 | 0.6143 | **0.6807** | 0.6768 |
| 6 | 0.6566 | 0.6597 | 0.6671 | **0.6719** |
| 12 | 0.6612 | 0.6580 | **0.6787** | 0.6759 |

Of twelve paired contrasts (partitioned RF minus each baseline, 1000 country-cluster
bootstrap draws, seed 42), **three have a 95% interval excluding zero and all three are
losses**, plus one that is a gain and must not be read as one:

| contrast | delta | 95% CI |
|---|---|---|
| h1 vs persistence | -0.0195 | [-0.0325, -0.0065] |
| h1 vs XGB | -0.0171 | [-0.0334, -0.0024] |
| h12 vs XGB | -0.0175 | [-0.0326, -0.0007] |
| **h3 vs pooled RF** | **+0.0327** | **[+0.0149, +0.0466]** |

## Read this before citing the h3 result

The h3 partitioned-over-pooled gain has an interval excluding zero and looks like evidence
that spatial partitioning works. **It is not.**

Stage 1 accepted no split: the candidate improved class-1 F1 from 0.803606 to 0.811040, a
gain of 0.007434 against the released `MIN_CLASS_1_IMPROVEMENT_THRESHOLD = 0.01`, so
`branch_table.sum() == 1` and the learned map is the root branch alone.

With one partition, the partitioned and pooled arms still differ — but only because the
**local model's training rows exclude the 596 areas the frozen map could not assign**
(no eligible donor within 100 km), while the pooled model includes them. That exclusion is
strongly geographic: **five countries are entirely unassigned** (Lebanon, Bangladesh,
Ecuador, Timor-Leste, Palestine), and Angola 85.7%, Dominican Republic 68.8%, Ethiopia
66.7%, Pakistan 36.5%, Mozambique 24.0% are partly so. The top three countries hold 55.1%
of unresolved areas against a much flatter learned distribution (45 countries, top three
37.4%).

So the h3 contrast measures *a model trained with five countries removed* against *a model
trained on everything*. It is a training-set composition effect whose composition happens
to be geographic. See `DECISIONS_LOG.md` D9 and D13.

## The horizon axis is weaker than it looks

Performance barely decays with lead time (persistence 0.6814 at h1 to 0.6759 at h12). That
is not crisis stability — it is that **the information set barely changes**. On the same
(area, target month), h1 and h12 draw persistence from the **same source month 33.2%** of
the time and produce the **same prediction 84.5%** of the time, because IPCCH labels are
sparse and irregular (median 6 observations per area). The observation cadence is coarser
than the horizon differences being tested.

---

# What was built

Five modules, no new framework, no new dependency. The released GeoRF baseline is extracted
fresh into each run and never modified in place.

| file | role |
|---|---|
| `prepare_data.py` | R1 target/QC ledger, the 93-column feature matrix, the R4 Stage 1 split, and geography (Q8g repair, adjacency, donor coordinates) |
| `baseline_runtime.py` | verified extraction of the pinned release, the single approved source patch, and isolated imports |
| `run_pipeline.py` | run scaffolding, one Stage 1 fit, map export and donor completion, and the Stage 3 rolling forecasts |
| `report_results.py` | cohorts, confusion-count metrics, paired deltas and the country-cluster bootstrap; never fits a model |
| `test_contracts.py`, `test_stage1_contracts.py`, `test_stage3_contracts.py`, `test_report_contracts.py` | 122 contract checks |

## Gates that passed, independently re-verified in the main session

| gate | result |
|---|---|
| pinned source SHA256 | matches |
| release ZIP SHA256, CRC, MANIFEST | matches; 45/45 payload hashes in-archive and on-disk |
| runtime | Python 3.12.10, NumPy 2.2.6, pandas 2.2.3, sklearn 1.6.1, XGBoost 3.0.0, GeoPandas 1.0.1, Shapely 2.1.0 — all exact |
| R1 target ledger | 42,695 valid / 15,206 positive / 27,489 negative / 6,227 areas / 84 P5-fills / 2,601 shares exactly at .20 |
| feature matrix | 170,780 rows x 93 columns; `origin == target - horizon` on every row |
| R4 Stage 1 split | 19,591 labels / 3,264 multi-areas / 8,561 fit / 9,558 validation / 1,472 singleton / 1,491 zero-label |
| geometry | 6,227 features, EPSG:4326, 253 invalid (212/30/6/5); after repair 0 invalid, 0 empty, 0 non-polygonal |
| Stage 3 schedule | 122 main folds, exactly 35/33/30/24 at h=1/3/6/12; +16 partial-2026; 126 fitted, 12 empty |
| leakage | train window exactly 36 calendar months with `end == origin`; every origin >= 2023-01; persistence source month <= origin on all rows |
| coverage | 81,109 rows, **zero missing** learned predictions; persistence 75,580 present / 5,529 absent (91.55%) |

## The single source patch

`src/model/GeoRF.py`'s final grid-only refinement is restricted to non-polygon mode:
`if CONTIGUITY:` becomes `if CONTIGUITY and contiguity_type != 'polygon':`. Diff against
the pristine archive is **-1/+6 lines** (one condition, five comment lines).

The block is grid-only by its own comment, yet it runs *after* `s_branch.pkl` (:438),
`branch_table.npy` (:439) and `X_branch_id.npy` (:447) are written, and its output feeds
`build_terminal` (:471). Under polygon contiguity it would desynchronise the correspondence
table from the accepted state and the trained checkpoints, while the prediction paths
(:1000, :1021, :1084) recompute assignments from `s_branch` and would not see the change.
`get_refined_partitions_all` has exactly one caller repo-wide, so the blast radius is that
call site.

---

# Reproducing

Use the pinned Windows interpreter; WSL `python3` lacks GeoPandas.

```text
PY='/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe'

# contract checks
"$PY" -B IPCCHGeoRFExperiment/test_contracts.py         # 33
"$PY" -B IPCCHGeoRFExperiment/test_stage1_contracts.py  # 31
"$PY" -B IPCCHGeoRFExperiment/test_stage3_contracts.py  # 30
"$PY" -B IPCCHGeoRFExperiment/test_report_contracts.py  # 28

# full run (~19 min: Stage 1 ~100 s, Stage 3 ~1,018 s)
"$PY" -B IPCCHGeoRFExperiment/run_pipeline.py \
  --source-root "C:\...\1.Source Data\assembled_IPCCH" --run-id <fresh-id>

# report, independently reconstructible into a fresh directory
"$PY" -B IPCCHGeoRFExperiment/report_results.py --run-dir <run> --out-dir <fresh>
```

Run IDs are refused if they already exist. A failed run keeps its artifacts and is never
marked complete.

---

# Limitations

These constrain how the numbers may be read and are not resolved by any check above.

- **Publication timing is unverified.** Features assume observation-month availability.
  This is retrospective alignment, not proven real-time availability.
- **Five covariates have literally zero missingness** across all 170,780 origin-month reads
  — `EVI_mean`, `nightlight_mean`, `nightlight_std`, `Rainf_f_tavg_mean`,
  `Tair_f_tavg_mean` — while `GPP_mean` in the same approved family misses 31.4%. This is
  the fingerprint `research/secondary-predictors.md` flags for an upstream ungrouped
  forward-fill, whose link to the selected CSV is unverified. Nothing was reconstructed,
  substituted or dropped. The own-origin contract governs *which month is read*, not what
  upstream placed in that month.
- **Topology repair does not establish administrative identity.** Two areas lose ~97-98% of
  their footprint to plain `make_valid` repair (`admin_code` 162 and 133); 35 of the 217
  extracted collections change by more than 1%, worst 29.21%. An invalid original's area is
  diagnostic only. The upstream builder allowed unrestricted nearest-neighbour fallback
  with no saved per-area match provenance.
- **Stage 1 internal validation is a development score**, computed under per-area
  chronological cutoffs. The 0.8036 parent F1 is **not** comparable to any Stage 3 number.
- **1,231 of 6,227 polygons have no adjacency neighbour**, so inherited polygon refinement
  cannot act on a fifth of the universe.
- **Intervals describe country-composition uncertainty only**, conditional on saved
  predictions — not future prediction, not partition or training uncertainty.
- **Singleton scores are supplementary post-map diagnostics**, not learning validation and
  not independent Stage 3 evidence.

# Next round

Out of scope here and requiring a new task with its own pre-registration: relaxing the
0.01 gate, parameter tuning, and feature engineering. The rejected split's margin is known
(0.007434), so a relaxed threshold must be declared before the run rather than set just
below that number afterwards. `DECISIONS_LOG.md` D14.
