# Docs-based reverse pressure test

| Failure condition | Result | Evidence |
|---|---|---|
| Wrong Ethiopia boundary | PASS | Exact `ISO3 == "ETH"`; 187,200 rows, 1,040 admins, 180 months. |
| Wrong input boundary | PASS | Profiles the assembled 88-column panel before loader; excludes Stage 2/3 artifacts from feature statistics. |
| Source drift | PASS | Authoritative SHA-256 matches the frozen evidence and is unchanged after execution. |
| Duplicate or incomplete panel keys | PASS | `(FEWSNET_admin_code,date)` is unique and the 1,040 x 180 grid is complete. |
| Missing labels treated as negatives | PASS | Model probes use only non-null `fews_ipc_crisis`; missing label months remain missing. |
| Infinity overlooked | PASS | Raw null and pipeline-effective missingness are separate; z-score infinities are explicitly inventoried. |
| Obvious outcome/projection leakage in probe | PASS | IPC phase, HA, adjusted IPC and near/medium provider projections are excluded. |
| Forecast-time safety assumed | WARN | Contemporaneous assembled covariates remain; availability classes are undeclared. Probe is diagnostic only. |
| Random row validation presented as generalization | PASS | Random split is labeled sensitivity only and admin overlap is recorded. |
| Test labels used for threshold/model selection | PASS | Threshold is selected on 2019-2020 validation only; fixed candidate definitions are not chosen on test. |
| Lightweight RF called GeoRF | PASS | Reports distinguish diagnostic pooled RF fits from frozen released GeoRF predictions. |
| GeoRF overfitting declared without train metrics | WARN | Frozen providers contain test predictions but no comparable GeoRF train/OOF metrics; conclusion remains risk, not proof. |
| Low-support 2021-06 concealed | PASS | Monthly and released-result tables retain the one-row ETH month and stability summaries exclude it explicitly. |

The WARN items are unresolved evidence limitations, not execution failures.
