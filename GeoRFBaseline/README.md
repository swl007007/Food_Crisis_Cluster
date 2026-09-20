# GeoRF baseline 0.1.0 — F1, no SMOTE

Standalone source baseline for subsequent IPCCH specification and planning.
This release changes the Stage 1 F1 objective and disables SMOTE; it does not
contain a new scientific experiment or an IPCCH model. Parent source commit and
original file hashes are recorded in `SOURCE_PROVENANCE.json`. Original sources,
experiments and published results are unchanged.

## Model contract

The target remains the original binary crisis target (`fews_ipc_crisis`, class 1).
Validation membership remains the existing random split within each area/group,
with at least one validation and one training row when possible; singleton
groups remain in training. This is not temporal validation.

For group g, aggregate confusion counts before computing F1:

```text
D_g = 2 TP_g + FP_g + FN_g
A_g = 2 TP_g
C_g = D_g - A_g
B_g = D_g * sum(C) / sum(D)
q(S) = sum(C_S) / sum(B_S)
     = (1 - F1_S) / (1 - F1_parent)
```

The ratio identity applies when denominators are nonzero. True-negative-only
groups have zero F1 exposure. Undefined scan ratios are neutral and finite.
The existing spatial scan/refinement proposes partitions; it does not guarantee
the globally best partition for final F1. Minimum-support and spatial rules are
retained. Empty candidate children are rejected.

At every depth, including the first, compare the parent and all three possible
child/parent checkpoint combinations on the **same complete validation rows of
the current parent branch**. Use aggregate class-1 F1, including false positives,
not a mean of child F1 scores. The parent wins ties. Accept a split only when
F1 improves **strictly more than 0.01**, using exact count-based arithmetic at the
boundary. This is a performance gate, not a statistical significance test.
Rejected parents do not activate descendants; inherited checkpoints and reported
predictions agree. Stage 2 retains its original positive logit-F1 gain weights.

Stage 1 and both Stage 3 arms have SMOTE disabled. Explicit `use_smote=True` raises
an error. **The user-approved Stage 1 class-recovery behavior remains:** one
zero-feature row per class is appended to fitting (two rows for the binary target).
These artificial observations can influence small or single-class branches; their
effect is not measured here. Stage 3 trains on original rows only, using its
existing pooled fallback for small/single-class partitions.

## Install and verify

Use Python 3.12. `requirements.txt` records the direct package versions used in
the verified Windows Python 3.12.10 environment; it is not a historical or fully
transitive environment lock. Optional basemap rendering also requires contextily.

```bash
cd GeoRFBaseline
python -m pip install -r requirements.txt
python -B tests/test_baseline.py
```

See `VALIDATION.md` for checks actually executed and their limits. Dependencies
were already installed for local validation; a clean installation is not claimed.

## Inputs and runnable stages

Supply the original-format FEWS NET panel CSV, boundaries with shapefile sidecars,
and a coordinate CSV with `FEWSNET_admin_code,lat,lon`. These data are external and
not included. Input rows must be sorted by `FEWSNET_admin_code,date` before
preprocessing/group creation. Feature preparation rejects unsorted rows to prevent
spatial/metadata misalignment; it does not silently reorder groups. Panel preprocessing and feature schema are inherited from
`src/preprocess/preprocess.py` and `src/feature/feature.py`; an IPCCH table is not
interchangeable with this input. Stage 2 retains the original admin universe
0–5717. Every in-scope code needs coordinates.

Run from the extracted package root. The examples below use explicit paths;
replace `/data/...` with your paths (native Windows paths when using Windows
Python). Source filenames retain their legacy names.

Stage 1: one month/scope per invocation. The wrapper uses a fresh run directory,
sets `GEORF_POLYGONS`, saves the full log/command, and copies the exact layout
required by Stage 2. It rejects existing output and missing correspondence files.

```bash
python scripts/run_stage1.py --data /data/panel.csv --polygons /data/boundaries.shp --experiment-dir work/GeoRFExperiment --year 2018 --month 2 --scope 1
```

For the inherited main workflow, repeat for the approved learning years,
months and scopes (original main batch: 2018–2020, all twelve months,
fs1/fs2/fs3). Scope mapping is fs0=1 month, fs1=4, fs2=8, fs3=12.
Keep fs0 in a separate experiment directory, with general consensus only.
These are reproduction instructions, not a newly approved experiment execution.

Stage 1 output is under `work/GeoRFExperiment/runs/`; the Stage 2 handoff lives in
`GeoRFResults/`, containing monthly metrics and named model directories with
correspondence tables. Unsplit roots export as `root`; labels such as `00` remain
strings. NaN row removal is synchronized with correspondence metadata.

Stage 2: copy your coordinate CSV to
`work/GeoRFExperiment/FEWSNET_admin_code_lat_lon.csv`, then run:

```bash
python scripts/step1_merge_results.py --experiment-dir work/GeoRFExperiment --model-type georf
python scripts/step3_create_linked_tables.py --experiment-dir work/GeoRFExperiment
python scripts/step4_similarity_matrix.py --experiment-dir work/GeoRFExperiment
python scripts/step5_sparsification.py --experiment-dir work/GeoRFExperiment
python scripts/step6_complete_clustering_pipeline.py --experiment-dir work/GeoRFExperiment
```

No step2 script is needed: step3 reads step1's pickle directly. Step6 uses the
existing report's recommended cluster count; its output is
`knn_sparsification_results/cluster_mapping_k40_ncN_general.csv`, with actual N.
Month-specific consensus is available through the retained stage CLIs; use
`--help` for matching month/suffix/similarity-directory arguments.

Stage 3: original partitioned versus pooled RF comparison, with no additional arms:

```bash
python scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py --data /data/panel.csv --partition-map work/GeoRFExperiment/knn_sparsification_results/cluster_mapping_k40_ncN_general.csv --out-dir work/comparison_fs1 --start-month 2021-01 --end-month 2024-12 --forecasting-scope 1 --lower-model rf
```

Use a fresh output directory. Add `--polygons /data/boundaries.shp --visual` only
for optional maps. Do not enable validation-threshold/expert-correction arms for
this baseline. Non-RF adapters and legacy alternative switches are outside this
release's supported workflow. Default machine-specific data paths retained in
legacy entrypoints must be overridden as above.

## Release and IPCCH handoff

`releases/georf-baseline-v0.1.0.zip` is a local source release.
`MANIFEST.json` hashes package payload files; `SHA256SUMS` verifies the archive.
Raw data, fitted scientific models and historical outputs are excluded.

Before IPCCH modeling, specify its area keys, sparse observation calendar,
target, forecast-origin availability, observed-history construction, temporal
validation and class-support policy. The current preprocessing, calendar lags,
fixed admin universe and repeated use of validation for split selection are
inherited limitations, not resolved by this release. No claim of complete
forecast-time safety or improved test performance is made.
