# D7 covariate release rules — family-level evidence (2026-10-02)

Scope: non-IPC covariates in `FEWSNETGeoXGBExperiment/feature-schema.json`. Policy (fixed, not changed here): cutoff = end of origin month O; monthly family -> one fixed documented lag L, feature = source month O-L, no backward search; annual -> latest reference year whose release is eligible at cutoff; no documented rule -> EXCLUDE.
Time-boxed scout (~10 min after coordinator tightening). Read-only. No `fews_*` values read. Notebook outputs not opened.

Legend: **[V]** verified from a local file or a quoted provider page; **[I]** inference by this scout.

## Local identity evidence (root `SRC = Analysis/1.Source Data`)

- `variable_construction.xlsx` (2025-06-15) maps families to GEE IDs: FLDAS `NASA/FLDAS/NOAH01/C/GL/M/V001`; nightlight `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4`; EVI `MODIS/061/MOD13A2`; AEZ `ESA/WorldCereal/AEZ/v100`; slope `CSP/ERGo/1_0/Global/ALOS_topoDiversity`; elevation `USGS/GMTED2010_FULL`; rivers `WWF/HydroSHEDS/v1/Basins/hybas_12`; ruggedness Nunn–Puga; crop/range ASAP; CPI/GDP/Gini/CC WBG; soil ISRIC. [V]
- `variable_construction_notes_description.xlsx` (2025-10-31) **conflicts** on several: nightlight "VIIRS"; EVI `MOD13A1`; elevation `USGS/SRTMGL1_003`; slope Geomorpho90m; rivers WB "Major Rivers of the World" (units km); market access Nelson (2019) travel time to cities/ports 2015 (minutes). [V]
- `assembled_IPCCH/metadata/variable_codebook_reorganized.csv`: FAO_price = FPMA domestic market-level, matched on market; WFP_Price/_std = country-level **ALPS** product; Food_CPI / Food_food_inflation = FAOSTAT country-level; CC = WGI Control of Corruption percentile; nightlight "VIIRS and DMSP/OLS". [V]
- Nightlight extraction `DMSP_OLS/FEWSNET/nightlight_mean_extraction_results_FEWSNET.csv` columns start `2012_04`, end `2024_12` [V] -> consistent with VIIRS DNB monthly (VCMCFG starts 2012-04), not DMSP (ends 2013) [I]. No local evidence for NASA Black Marble; source of 2025 nightlight values not located. [V/I]
- EVI `EVI/FEWSNET/...` monthly 2010_01–2024_12; `EVI/output_update/` 2025_01–2026_03 (file date 2026-05-11). GPP `GOSIF_GPP/FEWSNET/...` monthly `2010.M01`–`2024.M12`. FLDAS `output_updates` 2025_01–2026_03, 28 bands (2026-05-11). [V]
- ACLED raw exports `ACLED/raw/ACLED Data_2026-04-23.csv`, `ACLED Data_2026-05-11_ipcch.csv`, `2009-01-01-2025-01-01.csv` (2025-04-09); raw header includes a `timestamp` (record last-modified epoch) column. [V]
- WBG: `GDP.csv` = `NY.GDP.PCAP.PP.KD` (GDP per capita PPP, constant 2021 intl $), `CPI.csv` = `FP.CPI.TOTL.ZG` (inflation, consumer prices annual %), both to 2023 (downloaded 2025-04-09); `CC_percentile.csv` = `CC.PER.RNK` to 2023 (2025-04-09); `GDP_CPI_GINI.csv` 2025 column (2026-04-23). [V]
- FAOSTAT `FAO/Food_CPI_2025.csv` (download 2026-05-01): last non-missing month per country is 2025-09 for 140 countries, 2025-02..2025-08 for others [V] -> one observed vintage, ≥7-month staleness; not a documented rule.
- FAO prices: `FAO/FOOD_CRISIS_FAO_PRICE_DATA_04012026.xlsx` (2026-05-01); WFP: `WFP/Price_ALPS.xlsx` (2026-04-23) plus DataViz exports 2025-03-27, 2025-12-02, 2026-05-11. [V]
- Statics: ASAP `asap_mask_crop_v02.tif`, `asap_mask_rangeland_v02.tif` (2023-04-20); `Market_access/acc_50k.tif` (2023-12-28); per-unit static CSVs (no time column) for soil, elevation, slope, ruggedness, rivers, AEZ, market_access, popdensity. [V]
- Descriptives (`FEWSNET_forecast_unadjusted_bm_predictor_descriptives.md`): crop/range max 200; gpp_mean max 65,535 (fill value); ruggedness uniform 0–1 (median 0.5001); slope 0–1; distance_to_river max 6,774.8; Rainf/Tair_zscore have 30,780 ±inf; pop 71.7% missing, mean 140k, max 13.3M. [V]

## Provider documentation (retrieved 2026-10-02)

- ACLED — acleddata.com/use-access/when-are-acled-data-updated : "Africa: Weekly on Monday | Event-level data covering Saturday to Friday of the previous week." Living dataset: "supplementation of historic periods is also ongoing". [V]
- FLDAS — ldas.gsfc.nasa.gov/fldas/specifications and GES DISC README_FLDAS.pdf : latency "~20th (final run)" of next month; "~1 month latency"; Nov 2020 global reprocessing with MOD44W land mask. [V]
- GOSIF GPP — globalecology.unh.edu/data/GOSIF-GPP.html : update log "Update on August 16, 2026: GOSIF GPP has been extended to December 2025"; 2025-03-30 to Dec 2024 (fix 2025-05-31 seasonal-cycle error in 2024); 2024-05-13 to Dec 2023; 2023-04-02 to Dec 2022 (fix 2023-05-14); 2022-04-29; 2021-03-15; v2 2019-12-01 extended to 2018. [V]
- MOD13A2 v061 — LP DAAC user guide / Earthdata catalog: 16-day composite; no production-latency figure found. [V: absence]
- VIIRS DNB monthly (EOG/GEE) — no documented monthly-composite latency found. [V: absence]
- FAO FPMA — data.apps.fao.org/catalog/dataset/domestic-market-prices-fpma : "updated on a monthly basis with latest available data"; FPMA bulletin warnings only "if latest available price data is not older than two months" (usage rule, not a publication lag). [V]
- FAOSTAT CPI — fao.org/statistics/data-releases : CP domain "Release interval · Quarterly"; e.g. March-2025 update released 18 June 2025; latest 11 June 2026. Reference-month coverage per release not documented in sources found. [V]
- WDI — blogs.worldbank.org "What's new in the WDI: July 2026 update": July update "includes national accounts statistics for 2025"; same pattern July 2024 (2023 data) and July 2025 (2024 data). CPI inflation sourced from IMF IFS. PPP revised (ICP 2021, May 2024). [V]
- WGI — datacatalog.worldbank.org WGI: 2023 edition public "Friday September 29, 2023"; WGI 2026 last updated Sep 18, 2026; 2024/2025 editions Oct–Dec; 2025 = WGI 2.0 methodology with recalculated history and new 0–100 scale. [V]
- WFP ALPS — dataviz.vam.wfp.org methodology: ALPS "calculated monthly" and includes forecast months (ARIMA/Holt-Winters, six months ahead); no upload-latency documentation; HDX global food prices dataset discontinued 2021. [V]

## Family table

| Column(s) | Product | Resolution as built | Documented latency | Proposed rule | Vintage status | Caveat |
|---|---|---|---|---|---|---|
| event_count_*, sum_fatalities_*, *_w5/_w10, distance_to_nearest_acled | ACLED event data | monthly aggregate of events | weekly Monday release of prior Sat–Fri (ACLED) | **L=1** (O-1 fully released by end of O) [I from V] | reconstruction; `timestamp` permits a partial as-of filter, but edited/deleted record history is lost | back-coding inflates historic counts vs real-time |
| Rainf_f_tavg_mean, Tair_f_tavg_mean (+Rainf/Tair_zscore) | FLDAS Noah01 C GL M (MERRA-2+CHIRPS) | monthly | final ~20th of next month | **L=1** | reconstruction with revised values (Nov 2020 reprocessing; CHIRPS final) | z-score baseline must use only eligible months; sidecar rolling-order ambiguity unresolved; ±inf values |
| EVI, EVI_l1..l12 | MODIS MOD13A2 v061 (notes say MOD13A1) | 16-day -> monthly | none found | **EXCLUDE: no documented rule** (unlock path: granule production dates) | unknown | product ID conflict |
| nightlight, nightlight_sd, nightlight_m12 | VIIRS DNB monthly (inferred from 2012-04 start); spreadsheet says DMSP | monthly | none found | **EXCLUDE: no documented rule** | unknown | source identity conflict; 2025 source not located |
| gpp_mean | GOSIF GPP v2 | monthly | annual batch, Mar–Aug of Y+1 (log 2019–2026) | **EXCLUDE recommended**; if kept, fixed L≥23 (Jan 2018 data released 2019-12-01) | reconstruction with revised values (2022, 2024 fixes) | 65,535 fill value in panel |
| FAO_price, market_distance | FAO GIEWS FPMA domestic prices | monthly, nearest matched market | "updated monthly"; 2-month staleness rule (usage) | **EXCLUDE: no documented publication lag**; candidate L=2 needs lead sign-off | reconstruction (single 2026-04-01 download) | market match set changes with reporting |
| Food_CPI, Food_food_inflation | FAOSTAT CP (food CPI, food inflation) | monthly, country | quarterly release; coverage lag undocumented | **EXCLUDE: no documented rule** (observed ≥7 months in one vintage) | reconstruction | estimated "E/X" flags revised |
| WFP_Price, WFP_Price_std, WFP_Price_m4/_m12 | WFP ALPS country-level | monthly | none | **EXCLUDE** | unknown | ALPS includes model forecast months -> look-ahead risk |
| GDP | WDI NY.GDP.PCAP.PP.KD | annual repeated monthly | July update adds Y-1 | **annual: ref year Y-1 if O ≥ Aug, else Y-2** | reconstruction with revised values (ICP 2021 rebasing) | constant-2021$ series did not exist before May 2024 |
| CPI | WDI FP.CPI.TOTL.ZG | annual | July update (conservative) | **same as GDP** | reconstruction | IMF IFS revisions |
| CC | WGI CC.PER.RNK | annual | edition Sep–Dec of Y covers Y-1 | **annual: ref year Y-2 at any O in year Y** (conservative) | fixed-source: April-2025 download (pre-WGI 2.0) | do not mix with WGI 2.0 vintages |
| gini | WDI SI.POV.GINI | irregular survey years | none (survey-dependent) | **EXCLUDE: no documented rule** | unknown | 51.6% missing |
| pop | unknown (not `popdensity`) | unclear | — | **EXCLUDE: semantics unknown** | — | magnitude resembles IPC `estimated_population`; possible outcome-pipeline leakage |
| crop, range | JRC ASAP crop/rangeland masks v02 | static | n/a | static OK (0–200 = 0–100% x2) [I] | fixed file 2023-04-20 | — |
| market_access | Nelson 2019 travel time to cities 50k (2015) | static | n/a | static OK (minutes) | fixed file | reference year 2015 |
| sg_* (5) | ISRIC SoilGrids 5–15 cm | static | n/a | static OK (mapped units) | fixed extraction | — |
| AEZ_* | ESA WorldCereal AEZ v100 (codebook says FAO/IIASA) | static dummies | n/a | static OK | — | label conflict only |
| elevation | GMTED2010 vs SRTMGL1 (conflict) | static | n/a | static, **identity unresolved** | — | pick one |
| slope | ALOS topoDiversity vs Geomorpho90m (conflict) | static 0–1 | n/a | static, **identity/units unresolved** | — | values look normalised |
| ruggedness | Nunn–Puga, percentile-transformed [I] | static 0–1 | n/a | static OK if percentile set fixed | — | uniform distribution |
| distance_to_river | HydroSHEDS vs WB Major Rivers (conflict) | static | n/a | static, **units unresolved** (max 6,775 not km-plausible) | — | — |
| lat, lon | unit centroid | static | n/a | static OK | — | — |

## Implied source month at end-of-month cutoff

| Rule | Origin Jun 2025 | Origin Feb 2025 |
|---|---|---|
| L=1 (ACLED, FLDAS) | May 2025 | Jan 2025 |
| L=2 (FPMA, only if approved) | Apr 2025 | Dec 2024 |
| L=23 (GOSIF, only if kept) | Jul 2023 | Mar 2023 |
| WDI annual (GDP, CPI) | ref 2023 | ref 2023 |
| WGI annual (CC) | ref 2023 | ref 2023 |

## Unresolved / excluded

- EXCLUDED (no documented rule): EVI family; nightlight family; WFP family; FAOSTAT Food_CPI/Food_food_inflation; gini; pop (semantics). FAO_price/market_distance excluded unless the 2-month FPMA staleness rule is accepted as a lag basis. GOSIF GPP excluded by recommendation (documented lag ≥23 months; fill values).
- Identity conflicts to resolve before freezing statics: elevation (GMTED vs SRTM), slope (ALOS vs Geomorpho90m), distance_to_river (HydroSHEDS vs WB rivers; units), nightlight (DMSP vs VIIRS), EVI (MOD13A2 vs MOD13A1).
- Not verified: actual 2025 nightlight provenance; whether ACLED `timestamp` filtering is used; FLDAS z-score producer rolling order.
