# 2025 outcome-source metadata probe

Date: 2026-10-02. Planning-only read-only source inspection. No phase/crisis/value values, outcome summaries or scores were inspected. CSV parsing selected only header, date and source/scenario/status metadata for reporting. External files unchanged.

## Search scope and method

Filename search under `/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/1.Source Data/Outcome`, followed by FEWSNET_IPC candidate headers and date/scenario/source metadata. Checked current repository README, PIPELINE_WORKFLOW, config files and task references. Scout dispatch failed at the agent thread limit; coordinator performed the bounded probe directly.

Base directory: `/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/1.Source Data/Outcome/FEWSNET_IPC/`.

## Candidate metadata

`2025_2026_FEWSNET.csv` includes source_organization, source_document, datasourcedocument, country/country_code, fnid, scenario_name, scenario, reporting_date, projection_start/end, collection_status and value columns. Only metadata was inspected; value/description were not used or output.

| Reporting month | Scenario | Raw row count |
|---|---|---:|
| 2024-10 | ML1 | 323 |
| 2024-10 | ML2 | 323 |
| 2025-10 | CS | 5,573 |
| 2026-02 | CS | 5,867 |

For October 2025, metadata identifies Current Situation, Published, FEWS NET source organizations and Food Security Outlook country documents. All observed October CS validity intervals were 2025-10-01 through 2025-10-31. There are no February/June 2025 reporting-date rows in this file. This is file coverage, not proof of a universal publication outage or absence of other products.

`FEWS_2025.csv` header: admin_code, fews_ipc, year, month, fews_ipc_crisis. Only year/month were summarized:

- 2025-10: 4,481 rows.
- 2026-02: 4,367 rows.
- Empty year/month: 1,238 rows.
- No dated 2025-06 rows.

Filename discovery also found `fewsnet_ipcphase_2025-01_2026-04.geojson`, monthly CS/ML1/ML2 chunk filenames (including June), and external IPC/CH collections. Their contents were not opened. A monthly filename does not establish nonempty genuine labels. `scrape_fewsnet.py` is a candidate provenance lead, not reviewed here.

## Interpretation and unresolved checks

October 2025 has a plausible FEWS NET Current Situation evaluation source. Raw CS counts and processed CSV counts differ; no unique-region or valid-label coverage claim is supported yet. Verify source-to-existing-admin mapping, current-situation versus forecast semantics, duplicates, valid labels and extraction/transformation lineage before scoring. reporting_date and Published status alone do not establish exact historical release/vintage availability.

June 2025 evaluation truth remains unresolved; neither inspected CSV supplies it. Do not substitute projections or carry-forward labels for missing outcomes. Other sources require explicit comparability decisions. 2026 rows are outside the adopted target scope.

The older `FEWSNET.csv` ending 2024-10 is not the only local FEWS NET source. This probe supersedes any inference of global absence from that old file's cutoff. No source is certified ready for evaluation by this metadata probe alone.
