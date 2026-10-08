# Initial read-only inventory — 2026-10-07

Two read-only probes examined environments and bounded historical experiment locations. No installation, server startup, historical import, model fitting or source-run modification occurred.

## Environment

- WSL Python `/usr/bin/python3` 3.12.3; `uv` 0.11.8 at `/home/swl007007/.local/bin/uv`. No MLflow executable on PATH.
- Windows Store Python is 3.12.10, with numpy2.2.6/pandas2.2.3/xgboost3.0.0; Windows `.venvs/ipcch-mlp` is another distinct environment. WSL system numpy2.4.2/pandas3.0.0 is not the frozen model runtime. Package versions were checked through metadata, not training-library imports.
- No MLflow/mlflow-skinny/mlflow-tracing distribution found in the two system interpreters, eight Linux home venvs, the Windows MLP venv or repository GeoDT diagnostic venv. This was bounded discovery, not an exhaustive search of remote/conda/container environments.
- No listener on5000/5001 on either platform. Linux home has about735GiB free; mounted C: about299GiB. Linux home is writable and outside Dropbox.
- User systemd is accessible but degraded due to an unrelated paseo.service; Linger=no. No MLflow unit found. Do not alter unrelated services. Windows-browser localhost access and MLflow dependencies remain to be verified after installation is authorized.
- Recommended architecture for discussion: isolated WSL venv; SQLite metadata and artifacts on Linux home; historical importer reads Windows artifacts via `/mnt/c`, with explicit original-path metadata. Do not install into frozen model runtimes or system Python. This is a proposal, not an adopted host/storage decision.

## Experiment families

| Family | Candidate source | Known boundary |
| --- | --- | --- |
| IPCCH original P6 GeoXGB | `IPCCHGeoXGBExperiment/runs/p6-formal-20261004b`; archived `10-04-ipcch-cumulative-share-geoxgb` evidence | Supervisor accepted; first run without b is incomplete and retained. About2GB models. |
| IPCCH fixed-map MLP | Windows LocalAppData/Temp `ipcch-mlp-runs/mlp-formal-20261005`; archived `10-05-ipcch-mlp-spatial-adaptation/evidence/formal` | Three replicates42/43/44, B/P/L/G arms; distinct frozen runtime. Final inventory about2.59GB. |
| IPCCH yearly XGB | Windows LocalAppData/Temp `ipcch-yearly-xgb-runs/yearly-formal-20261007`; task `10-07-ipcch-fixed-map-yearly-xgb/evidence/formal` | Supervisor accepted at76fdedf; final doc5de8fac merged/pushed main. Task remains in_progress because lifecycle closure is separate. Final inventory1,127,063,505B. |
| IPCCH climate perturbation | Windows LocalAppData/Temp `ipcch-climate-runs/climate-20261005`; archived `10-05-ipcch-climate-perturbation/evidence` | Exploratory rich601 with RELEARNED maps, not a fixed-map feature substitution. About2.43GB retained run. |
| Adjacent IPCCH global climate/history/IDP | `../IPCCH/results/experiments/origin_safe_climate_idp_v1`; reports in matching `reports/` directory | Three arms x H0/3/6/12, reported-phase truth; not paired with P6 scores without alignment. |
| Adjacent IPCCH oracle/no-weather | `../IPCCH/results/experiments/origin_safe_weather_oracle_v1`; matching reports | Located; coverage/acceptance/lineage need per-family verification if included. |
| Adjacent IPCCH older annual/threshold families | `../IPCCH/results/experiments/deep_feature_weight_decay_forecasting`; matching reports | Includes climate2015_v1 and threshold0.12/0.15/0.20 variants. Final vs context-only status unresolved. |
| FEWSNET four-class RF | `FEWSNETFourClassBaseline/runs/fourclass-v7-20260928` | README names v7 authoritative even though v8/v9 directories also exist; do not select by max version. About1GB retained checkpoints per VALIDATION.md. |
| FEWSNET clean persistence | `FEWSNETCleanPersistenceExperiment/runs/20260920_stage1/report/final_report.json` | WinnerABDE; prepare runs differ from pipeline runs. Current acceptance not independently checked. |
| Other FEWSNET/ETH/correction | `FEWSNETGeoXGBExperiment`, `Step3ExpertCorrectionExperiment/outputs`, `PersistenceCorrectionExperiment/outputs`, `EthiopiaForecastingExperiment/outputs/local_partition_experiment` | Mix of incomplete, bounded, repeated and older runs; further curation needed before formal import. |

The inventory did not locate all-history-window and split2024 sensitivity raw packages within its bounded search. Do not infer that they are absent or silently omit them if the user includes those studies; locate their evidence in the next scoped pass. Counts/volumes here come from existing manifests/documents, not a new full recursive scan.

## Import implications, not yet a selected schema

- Preserve scientific acceptance separately from task lifecycle and MLflow ingestion status. `in_progress` is not proof that yearly-XGB science is unaccepted; explicit supervisor acceptance is in `10-07-ipcch-fixed-map-yearly-xgb/evidence/final-acceptance-76fdedf.md`. A passed replay likewise is not an independent close audit.
- IPCCH recent families distinguish H1/3/6/12, main/supplementary2026, E_all/E_persist/local-eligible cohorts, arms and seeds. P6/MLP population-share truth differs from adjacent IPCCH reported phase; raw-vs-projected decode must be explicit.
- FEWSNET fs1/fs2/fs3 correspond to H4/H8/H12; ETH fs0 is H1. Stage1 development results must not be silently labeled test performance.
- Historical imports need stable source identities and idempotence; import time is not original fitting time. Missing/NA metrics must not become zeros. Models alone are insufficient without saved source/config/input/report lineage.
- Small reports/configs/manifests are available for the four recent IPCCH families. Copying all checkpoints would add several GB; references vs managed artifact copies remains a user decision.

## Next user-owned decision

User selected option1: recent current-repository IPCCH studies and related sensitivities. Next choose artifact retention/storage policy; local install version/dependencies and host/persistence policy remain to be settled.

## Scope follow-up: sensitivity storage

The old Linux `/tmp/ipcch-window-probe-20261005/report.md` and `/tmp/ipcch-split2024-20261005/report.md` links in the meeting note no longer resolve. A bounded Windows Temp listing located candidate retained directories `/mnt/c/Users/swl00/AppData/Local/Temp/ipcch-window-probe-20261005-zckpq9hf` and `/mnt/c/Users/swl00/AppData/Local/Temp/ipcch-split2024-20261005-3moikuh3`. Their exact source completeness/identity must be checked before import. This demonstrates why copying retained artifacts to a durable managed store is preferable to relying on Temp paths alone.
