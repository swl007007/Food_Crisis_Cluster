# Baseline and runtime evidence — 2026-10-05

Read-only planning research. No project-data fitting, new predictions, dependency installation, lifecycle start, or executor dispatch occurred.

## Source evidence

- Original scientific contract: `IPCCHGeoXGBExperiment/config/experiment-contract.json`.
- Complete original design read directly: `.trellis/tasks/archive/2026-10/10-04-ipcch-cumulative-share-geoxgb/design.md`.
- Existing run: `IPCCHGeoXGBExperiment/runs/p6-formal-20261004b`.
- `source-inventory.json` captures 56 existing source/input files. All 14 prepared artifacts were rehashed and matched `prepared/prepared-manifest.json`; four original prediction files, maps, frozen-map metadata, configuration, and package Python sources were also hashed. Other captured hashes establish current byte identity, not an independent audit of the original experiment.
- Current source has no working NN backend in `src/model` / `app`. The neural imports at `src/model/GeoRF.py:16` are commented; adapters are GF/XGB/DT. Do not interpret historical neural configuration names as implemented MLP support.
- `quartet.py:258-278` requires a true global parent for every target. `stage3.py:119-135` binds local fits to the global for the same fitting origin; `stage3.py:144-170` defines exact-count gate semantics.
- GitNexus query failed with the existing LadybugDB shadow-page replay error during exploration; no reindex was performed. A new exploration-agent dispatch later hit the thread limit, so the bounded runtime/artifact checks were completed directly.

## Runtime observations

| Item | Observed |
| --- | --- |
| Existing Windows Python | 3.12.10, Windows Store python3.12.exe |
| Existing Windows packages | numpy 2.2.6, pandas 2.2.3, xgboost 3.0.0; torch absent |
| Current WSL Python | 3.12.3; numpy 2.4.2, pandas 3.0.0; torch absent |
| GPU via nvidia-smi | NVIDIA GeForce RTX 4070, 12,282 MiB, driver 560.94 |
| Logical CPUs | 32 reported to WSL |
| WSL scratch free space | about 789 GB at inspection; recheck before execution |

The official PyTorch CUDA 12.4 index was read without downloading/installing a wheel:

`https://download.pytorch.org/whl/cu124/torch/`

It lists Windows CPython 3.12 wheel `torch-2.6.0+cu124-cp312-cp312-win_amd64.whl`, SHA256 `3313061c1fec4c7310cf47944e84513dcd27b6173b72a349bb7ca68d0ee6e9c0`.
This verifies wheel availability, not local CUDA import, numerical behavior, or training speed. Those require executor P0 checks in a separate environment. Do not install into either existing interpreter environment.

## Fit enumeration

`fit-enumeration.json` was produced with Python stdlib csv/gzip over the saved keys, calendar, and maps. No feature matrix or targets were used to train a model.

Algorithm: for each nonempty scheduled H/T fold, include current global origin O=T-H and the six latest observed target months U<O with internal origins U-H. At each historical U, include a regional residual request iff the region has U keys and its [U-H-35,U-H] fitting pool meets 500 keys / 50 areas / 6 target months. For current diagnostic requests, require current keys and the same fitting support, irrespective of historical gate. Deduplicate globals by (H,origin) and regional fits by (H,origin,region); each global also requires one pooled residual quartet. Multiply scalar models by four targets and three seeds. No candidate-dependent gate score affects these fit counts.

| H | Scored/planned folds | Unique globals | Unique pooled residuals | Historical regional fits | Current-only regional fits | All regional fits | Scalar fits per seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36/39 | 43 | 43 | 183 | 12 | 195 | 1,124 |
| 3 | 34/37 | 43 | 43 | 143 | 18 | 161 | 988 |
| 6 | 31/34 | 43 | 43 | 129 | 25 | 154 | 960 |
| 12 | 25/28 | 43 | 43 | 157 | 70 | 227 | 1,252 |

The 172 globals and 612 unique historical regional fits reproduce the previous P6 enumeration totals. The new diagnostic adds 125 unique current-only regional fits. Thus Stage3 needs 4,324 scalar fits per seed, or 12,972 for three seeds. Development needs 4 horizons × 3 seeds × 4 targets × (2 global + 4 pooled residual fits) = 288 scalar fits. Total: **13,260**. These are neural-model counts, not XGB quartet counts or elapsed-time estimates.

The executor must implement a no-fit enumerator and reproduce this inventory from verified inputs before scientific fitting. Exact-identity caching must also reconcile requested versus unique fits after completion. Synthetic tests and fixed synthetic timing probes are accounted for separately, not included in the project-data fit count.

## Fable review disposition and supervisor checks — 2026-10-05

The existing Claude Fable 5.1 session in Herdr wN:p2 (session 174ea213-be60-4f2a-9bcb-e5e3523ad738) returned a read-only adversarial review of this specification. It found no blocker and raised optimization-budget and coverage concerns. This is a planning review, not an implementation audit or execution release. The user subsequently approved revising the specification according to the supervisor's recommendations.

- Adopted: initialize P/L output-layer weights and biases to zero; keep random hidden layers, B initialization and the 100/40 epoch budgets. Record actual update counts. Unequal updates do not prove undertraining; no fixed-step replacement, new convergence threshold or additional search was approved.
- Adopted: predeclare historical-support coverage, keep G−P as full-cohort deployment contrast and L−P as supported-cohort diagnostic. Do not infer inevitable nonsignificance from low coverage.
- Clarified: serial independent initialization/training/permutation RNG streams; maximum-pool synthetic timing; Windows run-volume space checks; training-all-missing feature observations; R1–R24 numbering. Existing missingness encoding and no-clipping choices remain unchanged. The review's all-missing-column observations are not evidence of numerical explosion or prediction failure.

Supervisor independently read `stage3.py:144-170` and counted main-period routes in `stage3/h{01,03,06,12}/predictions.csv.gz`. For each H, take every main key except `unmapped_area_global` and routes beginning `global_fallback:gate_support`. In these artifacts all remaining routes are `local` or `global_fallback:gain_not_above_threshold`. This yields:

| H | Main keys | Historical-support eligible | Share |
| --- | ---: | ---: | ---: |
| 1 | 17,322 | 3,211 | 18.54% |
| 3 | 16,919 | 4,997 | 29.53% |
| 6 | 16,413 | 3,423 | 20.86% |
| 12 | 14,087 | 845 | 6.00% |

These are historical-support counts, not current-support intersections or MLP gain decisions. The executor must independently rebuild keyed support sets from original keys/maps before fitting. A required technical failure stops the run rather than redefining support. The source prediction files are already covered by `source-inventory.json`.

Supervisor also enumerated every scheduled current O plus each selected historical U−H from `fold_calendar.csv` and `keys_h{01,03,06,12}.csv.gz`, deduplicated by H/origin, then counted keys in inclusive [origin−35,origin]. This gives 43 global pools per H:

| H | Minimum pool | Maximum pool |
| --- | ---: | ---: |
| 1 | 10,465 | 19,012 |
| 3 | 10,465 | 19,052 |
| 6 | 10,339 | 19,052 |
| 12 | 9,577 | 16,394 |

The review's 20,829-row maximum was not reproduced for the actual request set and is not used. The timing grid now includes the verified maximum 19,052. With 40 residual epochs and batch size 256, these global pools entail 1,520–3,000 updates, versus 80/320 at local n=500/2,000. This is count arithmetic, not a fitted convergence result. No project-data fitting or new predictions were performed for these checks.
