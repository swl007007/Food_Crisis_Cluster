# D7 FLDAS release corroboration

2026-10-03, Codex; public documentation and local notebook **source cells only**. No source values, protected outcomes, fitting, or product changes. This narrows the release/lineage uncertainties; it does not certify the ordered schema or pass D7.

## Direct provider evidence

- NASA specifications: https://ldas.gsfc.nasa.gov/fldas/specifications (HTTP 200). Global final-run latency: "~20th (final run) and ~5th (preliminary run) of the next month". Do not confuse preliminary CP with final C.
- GEE product catalog: https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001 (HTTP 200). Names collection `NASA/FLDAS/NOAH01/C/GL/M/V001`, monthly cadence, Noah 3.6.1, MERRA-2/CHIRPS forcings, DOI `10.5067/5NHC22T9375G`; links the provider README below.
- Provider README: https://hydro1.gesdisc.eosdis.nasa.gov/data/FLDAS/FLDAS_NOAH01_C_GL_M.001/doc/README_FLDAS.pdf (HTTP 200, 485145 bytes; SHA256 `5050cbbbcbec2d4365c8c0a29180a25ac9479520cb3dad70cffd362baeb7f379`; document last revised July 12, 2022).
  - Page 6, section 1.2: "FLDAS 'C' data is delivered about three weeks after the month concludes." CP is within one week; G daily is next day. This is direct evidence for a **reconstructed L=1** at an origin-month-end cutoff, not proof of every actual granule's timely publication.
  - Page 8, section 1.3: global C has "~1 month latency".
  - Page 8, section 1.4: November 2020 MOD44W post-processing affected model output over inland water; "all of the meteorological forcing variables (denoted by a _f_ in their short names) were unchanged." Thus that specific reprocessing is **not evidence that `Rainf_f_tavg` or `Tair_f_tavg` changed**. This corrects an overly broad implication in the earlier family note; it does not certify all historic vintages or upstream revisions.
  - Page 3: global products added October 15, 2018. Page 8: regional monthly C products decommissioned September 16, 2019. Coverage back to 1982 does not by itself prove the global product existed at earlier prediction origins. A reconstruction using predecessor products needs an explicit identity/compatibility basis.

## Local lineage located, not yet reconciled to pinned panel

Paths below are relative to external `Analysis/2.source_code/`:

- `Step1_FEWNET_predictor/first_method/00_download_FEWSNET.ipynb`, code cell 1 line 3: GES DISC URL names `FLDAS_NOAH01_C_GL_M.001/2010/FLDAS_NOAH01_C_GL_M.A201001.001.nc`. The example requests `Evap_tavg`, so this proves a product pointer, not Rainf/Tair extraction completeness.
- `Step1_FEWNET_predictor/02_alternative_extract_FEWSNET.py:166–168,222–230`: reads `Processed_Dataset/FLDAS_Monthly/Part5` TIFFs and writes mean/std extraction CSVs. Filename/path evidence alone does not bind those outputs to the pinned panel.
- `Step1_FEWNET_predictor/first_method/01_combine_all_FEWSNET.ipynb`, code cell 2: reads four `IPC_fldas_completed_part*.csv`; cell 10 reads `IPC_scaffold_FEWSNET_2024_completed.csv`; cell 22 writes `IPC_fldas_fewsnet_completed.csv`.
- `Step1_ACLED/00_add_ACLED_FEWS.ipynb`, code cell 0 lines 11–33: current source reads an April 2026 export, selects event fields without `timestamp`, and groups by event date's month/year. This does not implement historical timestamp filtering and does not prove it produced the older pinned panel. No ACLED values were opened. A direct request to the previously cited ACLED release-cadence page returned HTTP 403; no new cadence evidence was obtained.

## Admission consequence

The documented final-C L=1 rule is supported as **retrospective reconstruction**, retaining native missing values and the existing exclusion of z-scores. Still needed before family admission: bind raw-column extraction to the pinned historical/2025 covariate panel, resolve pre-global-product compatibility or eligibility, and record revised-value limitations. Do not relabel this as verified-vintage replay. ACLED and other family certification is unchanged. No release rule or feature schema was silently activated.
