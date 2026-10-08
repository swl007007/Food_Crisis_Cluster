# Source report layout — read-only research 2026-10-07

R = current repository root; T = `/mnt/c/Users/swl00/AppData/Local/Temp`. Two bounded agents inspected source JSON structures and retained files; no import or model fit. Parent architecture and numeric metric layout remain proposed in design.md.

| Family | JSON authority | Panel extraction |
| --- | --- | --- |
| P6 | `R/IPCCHGeoXGBExperiment/runs/p6-formal-20261004b/report/report.json` | `horizons[H][period][cohort][arm]` |
| MLP | `T/ipcch-mlp-runs/mlp-formal-20261005/report/report.json` | `replicates[seed][H][period][cohort].panels[arm]` |
| Yearly | `T/ipcch-yearly-xgb-runs/yearly-formal-20261007/report/report.json` | `horizons[H][period][cohort].panels[arm]` |
| Climate | `T/ipcch-climate-runs/climate-20261005/report/report.json` | P6 format; additionally `compare-p6/comparison.json` |
| Window | `T/ipcch-window-probe-20261005-zckpq9hf/summary.json` | `aggregates[]` by H/cohort with `panels`; `cells[]` by H/U |
| Split2024 | `T/ipcch-split2024-20261005-3moikuh3/IPCCHGeoXGBExperiment/runs/split2024-20261005/report/report.json` | P6 format; outer `combined-and-matched-report.json` by H/combined/2025/2026 |

Standard panels include n; binary accuracy/precision/recall/f1/f2/counts/na_reasons; four_class accuracy/macro_f1/per_class/confusion/na_reasons; q3_r2_projected/raw. MLP seeds42/43/44 are model seeds; other retained model seeds42, distinct from bootstrap seed42. Preserve per-class `4/5` meaning if metric keys encode it.

P6/climate E_all provides geo/pool; E_persist provides geo/persistence only. MLP also reports base and embedded xgbgeo/xgbpool; yearly embeds p6geo/p6pool. Local panels are in ungated_local_diagnostic, never E_all. Yearly additionally reports local_persistence_matched. Record already-saved panels only.

P6/climate bootstrap keys use `geo_vs_*`; MLP/yearly use `geo_minus_*`. All available formal main CIs are country bootstrap with2000 draws; supplementary has no CI. MLP descriptive seed_mean_range_descriptive is mean/min/max, not uncertainty CI. Embedded P6 panels come from keyed joins (`IPCCHMLPExperiment/ipcch_mlp/report.py:125-134`, yearly report.py:128-140), not new fits.

Window:11 selected cells; H1/3/6 targets2023/2024/2025 September, H12 targets2024-09/2025-01.18 aggregate records include H='all'; cohorts all/mapped/common_local_support/new_local_support (last absent for H3/H12). Four arms base_global/exp_global/base_local/exp_local. Local predictions fall back to corresponding global (`run_probe.py:148-180`); no historical deployment gate. Existing source contains380 UBJ+95 model records+11 saved prediction tables. Existing494-file inventory plus itself is495 files/165,460,097 bytes. Old baseline full models live in P6, not this folder. Saved start timestamp2026-10-05T18:39:41.831067Z;291.1202557s elapsed. Runtime/source hashes exist but separate Git HEAD unknown.

Split2024: model origin>=2025-01; stage1 cutoff2024-12. Main/supplementary E_all counts H1=7820/4092,H3=7371/4092,H6=4860/4092,H12=0/4092. Outer combined report has old_split_matched geo/pool on exactly the new keys (`run_experiment.py:70-87`); it is not the original full P6 score. Existing run inventory excluding prepared is4364 files/1,485,147,097 bytes; retained root+source/config estimate4410 files/1,486,936,528bytes. Model store836 quartets,3344UBJ; source HEAD afc88390e4750da52c4e44ed8662b83925642003, explicit variant patch/hashes.3330.344s elapsed; absolute start/end unknown. Old Linux/tmp report.md files missing, but retained JSON is sufficient; never fabricate missing prose or times.

Archive includes maps/recipes/ledgers, MLP model transforms, configs/source patch evidence and reports. Existing input/prepared X matrices are excluded with identity references. Original inventories are reference evidence, not assertions that every original file was copied. Scope counts/bytes above derive from existing inventories/stat and do not replace implementation-time checksum validation.

## MLflow interface research

PyPI `https://pypi.org/pypi/mlflow/json` returned stable3.17.0 (release2026-10-07T06:07:39Z), Python>=3.10. Full package supplies server/UI/SQLite dependencies; no extras needed. Resolve and lock dependencies in the isolated environment during implementation.

Context7 `/mlflow/mlflow` official-source documentation confirms SQLite tracking backend; artifact proxy with `--artifacts-destination`; clients using HTTP/`mlflow-artifacts:/`; MlflowClient.create_run/log_batch/log_artifact/set_terminated and tag-based search. References: `https://github.com/mlflow/mlflow/blob/master/docs/docs/self-hosting/architecture/tracking-server.mdx` and `https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/client.py`. These are unversioned docs; actual installed3.17.0 CLI/API smoke checks remain necessary. Use explicit run IDs; no training or active-run global side effects are needed for historical import.
