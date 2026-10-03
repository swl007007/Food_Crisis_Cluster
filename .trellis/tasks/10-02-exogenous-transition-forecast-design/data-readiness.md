# D7 data-readiness record

Status: **Bounded D7 pass consolidated, 2026-10-02 (awaiting coordinator review).** Executor: Claude, audit run `5c4dede7bc6e44f0835fbde1ad2f473d`, base `e84368274241e6becc67d5a7e69dc39db23bda66`. Metadata/provenance only. No product code changed, no model fitted, no score computed. No 2025+ IPC value, description, HA or projection-value column was read. Raw sources were stream-hashed, and GeoJSON files were not parsed for values.

Evidence notes (task research):

- `research/d7-sources-admin.md`: hashes, keys, admin crosswalk and 2025 join lineage. Probes `probes/d7_hash_files.py` and `probes/d7_admin_keys.py`.
- `research/d7-ipc-release-ledger.md`: IPC cycle identities, chunk metadata and FDW timestamps.
- `research/d7-panel-climate-lineage.md`: panel producers and climate rolling.
- `research/d7-covariate-release-rules.md`: source/product release rules.
- `probes/d7_label_calendar_n.py` → `d7_label_calendar_n.json`: historical label calendar and pre-mask N.

## Verdict summary

Real fitting BLOCKED (historical IPC release rule; dependent 2025 truth crosswalk). Environment and pinned sources PASS. See section 9.

## 1. Code identity and numerical environment — VERIFIED

- Checkout: branch `geoxgb-shared-parameter-experiment`, HEAD `e84368274241e6becc67d5a7e69dc39db23bda66`, which is the audit base. A real run must pin its own later producer commit. The existing `prepare_fourclass.py` refuses code that differs from the committed HEAD (`FEWSNETGeoXGBExperiment/README.md:87–89`).
- Windows interpreter: `C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe` resolves to `PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe` (Python 3.12.10, MSC v.1943, Windows-11-10.0.26200).
  - numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.15.2, geopandas 1.0.1, shapely 2.1.0, polars 1.27.1 and xgboost 3.0.0 are all equal to `PINNED_RUNTIME` (`scripts/prepare_fourclass.py:59–61`).
  - pyogrio 0.11.0 and pyarrow 19.0.1 are present. fiona is absent.
- `xgboost.dll` SHA256 is `019fb46ce4d0424e7adfdd0e6f96401e127ea88828ae2d273dc37547d4cd8fed`. This equals `xgboost_library_sha256` in the prior run manifests under `/mnt/c/Users/swl00/geoxgb_runs/`, which is outside Dropbox.
- Linux python3 was used only for read-only metadata probes. It is not a fitting environment.

## 2. Source identities, hashes and keys — pinned training sources VERIFIED; 2025 sources IDENTIFIED, not certified

Pinned sources (`prepare_fourclass.py:48–57`) all recompute to their pinned SHA256:

| Source | SHA256 | Notes |
|---|---|---|
| `1.Source Data/FEWSNET_forecast_unadjusted_bm.csv` | `611f9e77…f8c651` | 1,029,240 rows = 5,718 areas × 180 months, 2010-01..**2024-12**; no duplicate keys |
| `Outcome/FEWSNET_IPC/FEWSNET.csv` | `8fdd4cca…7b38d0` | 53 label months (quarterly 2009-07..2015-10, Feb/Jun/Oct 2016-02..2024-10); 5,716 codes (216, 2786 absent); no vintage field. `FEWS October 2024 Update TrueBoundaries_12-09-24.csv` is byte-identical |
| `FEWSNET_admin_code_lat_lon.csv` | `a06be858…a4bcb` | 5,718 codes |
| `FEWS_Admin_LZ_v3.shp` (+.shx/.dbf/.prj/.cpg hashed) | `3aba66a6…f5f742` | 5,718 records; same code space and `unit_name` as the panel |

Unpinned sources required for 2025 origins, because the pinned panel ends in 2024-12:

| Source | SHA256 (prefix) | Key facts |
|---|---|---|
| `assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025.csv` | `5c8aefc1` | 2025-01..2026-04; **4 duplicate key rows (code 2996, 2025-10 and 2026-02)** |
| `…_2025_combined.csv` | `41f02be9` | 2010-01..2026-04; keys for 2010-01..2024-12 are identical to the pinned panel; same 4 duplicates |
| `…_2025_combined.normalized-v1.csv` | `510375f5` | equals the sidecar value (independently recomputed); duplicates removed by an uninspected tie-break; values not certified (section 5) |
| `Outcome/FEWSNET_IPC/2025_2026_FEWSNET.csv` | `a64ed4bb` | raw FDW extract: CS 2025-10 (5,573 rows) and 2026-02 (5,867); ML1/ML2 reported 2024-10 (323 each, 6 countries) |
| `Outcome/FEWSNET_IPC/FEWS_2025.csv` | `4987356a` | derived by a name join; see section 3 |

**Administrative keys.** All modelling sources share one integer code space (0–5717). `unit_name` is equal across the panel, the DBF and the geoidentifier. FEWSNET.csv anomalies:

- codes 1667 and 2996 carry two names each;
- one Turkana name maps to codes 2995 and 2996, which is the source of the 2996 duplicates in the 2025 panels.

The historical join is unambiguous apart from these. The pinned-panel preflight already enforces equality with FEWSNET.csv on shared keys (`prepare_fourclass.py:249–275`).

## 3. Final-truth source (Study1/Study2 2025 evaluation) — BLOCKED for evaluation readiness; metadata identified

- **October 2025.** Genuine FEWS NET Current Situation exists in the raw FDW extract: `reporting_date` 2025-10-01, validity 2025-10-01..31, 21 countries, `Food Security Outlook <country>` documents, status Published.
- **The only local bridge to admin codes is a name join.** `fnid` never equals an admin code. The geoidentifier has no fnid, and the DBF `LZCODE` is blank.
  - The bridge used is `append_2025_2026.ipynb` cells 5–14: `admin_name == geographic_unit_full_name`, with no dedup and no write cell for FEWS_2025.csv.
  - **October 2025 only** (`research/probes/d7_oct2025_name_join.py`; scenario/date/country/name metadata only): 21 countries, 5,573 CS rows, 4,480 name-matched, 1,093 unmatched.
    - **Only DRC is wholly unmatched (345/345).**
    - Partly unmatched: Ethiopia 645 of 1,141, Sudan 30/391, Nigeria 21/305, Mali 20/116, Somalia 19/217, Malawi 8/84, Kenya 3/642, Chad 2/92.
    - Burkina Faso (65/65) and Niger (182/182) match fully.
    - Uganda has no October 2025 CS rows.
  - The combined 2025-10 + 2026-02 counts in `research/d7-sources-admin.md` (e.g. CAR, El Salvador, Honduras, Lebanon, Syria and Venezuela wholly unmatched; Ethiopia 665/1,161) include the 2026 additions. They are outside this target and labelled as two-year counts.
  - 502 names map to more than one fnid (2020-vs-2026 boundary versions), while the shapefile coverage dates are 2020–2021.
  - Code inference (no values read): unmatched rows have NaN `fews_ipc`, and cell 14 sets `fews_ipc_crisis = (fews_ipc >= 3)`, giving **0 for unmatched rows**. Using FEWS_2025.csv as-is would therefore fabricate non-crisis truth.
- Memory/prior planning recorded that normalized-v1 carries 2025-10 truth for all 5,718 codes, which exceeds the 4,480 name matches. That lineage is unexplained; see section 5 for the producer status.
- **Decision rule (D5/D7): an absent or ambiguous crosswalk blocks only the dependent 2025 truth keys.** It does not block historical implementation, which uses the pinned, already-joined FEWSNET.csv. October 2025 evaluation requires an approved, documented crosswalk before outcome release.
  - The historical file states "Joined to districts using largest spatial overlap rule" (`Column Descriptions.xlsx`), but its producer is not local.
  - **No unapproved geometric aggregation of IPC is introduced to recover coverage.**
  - The name and boundary evidence comes partly from 2026 records, so it does not establish that a 2025 mapping is valid.
  - Every unmatched or ambiguous unit and country stays excluded with a coverage reason, never zero-filled.
- **June 2025.** No CS in any local source: the raw CSV, the processed files and the empty CS chunks for 2025-01..2025-09. By the approved rule it is forecast/coverage-only unless a genuine source appears.

## 4. IPC publication cycles, release rule and outage — BLOCKED (no defensible historical release rule yet)

Three distinct notions are kept separate:

| Notion | Evidence | What it is not |
|---|---|---|
| Release-cycle identity | FEWSNET.csv `year_month`; FDW `reporting_date` (Feb/Jun/Oct outlook cycle) | not a publication date |
| Reporting/validity interval | FDW `projection_start/end`: CS = month M; ML1 = D..D+3; ML2 = D+4..D+7 | not availability |
| Collection timestamp | FDW `created`, `modified`, `status_changed`, `collection_status_changed` (database events) | not first public availability; re-uploads overwrite them (e.g. Sudan 2024-10 re-created 2026-01-13) |

Facts:

- **Historical cycles.** Quarterly through 2015-10, then tri-annual Feb/Jun/Oct 2016-02..2024-10. FEWSNET.csv shows country-cycle gaps: AFG 2021-10..2023-02, SSD 2020-10..2022-02, SOM 2022-06/2023-02, SDN 2024-10, YEM 2017-02/2023-10, BDI 2022-10; BFA absent before 2021-10, CMR before 2020-02. Whether these are unpublished cycles or extraction losses is not determinable locally.
- **2025 outage (local evidence).**
  - No CS for 2025-01..2025-09.
  - No ML1/ML2 with `reporting_date` Feb–Jul 2025.
  - Aug/Sep 2025 updates exist for 6 countries.
  - The full October 2025 outlook covers 21 countries; Feb 2026 CS covers 27.
  - The scrape window appears to filter on `projection_end` (inference), so absence from the scrape is file-coverage evidence, not proof that the products were never published.
- **Collection timestamps (2024-10, 2025-10, 2026-02 only).** Most countries' records were created between M day ~24 and M+1 day ~12. For example, only 7/21 countries had records created by 2025-10-31.
  - This is database-event evidence for three recent cycles.
  - It is **not** a documented historical publication lag, and it cannot be extrapolated to 2016–2024 or across countries.
  - The scout's "M+1" proposal (`d7-ipc-release-ledger.md:122–146`) is an unapproved extrapolation and is **not adopted** here.
  - FDW API fields include no publication-date field (`d7-ipc-release-ledger.md` section 2, verified by an API metadata fetch).
  - Feb CS `created` dates for 2021–2023 in four countries (ETH, KEN, SSD, YEM) range from 2021-02-26 to 2021-09-10, and Ethiopia Oct 2023 shows M+2. Pre-2021 `created` values are bulk re-ingestion dates (2020-11), so they carry no information.
  - The scout's alternative "M+2" candidate is violated by South Sudan in 2021 and 2022, and it also rests on database events. **Neither M+1 nor M+2 is adopted.**
  - Web relaunch evidence (FEWS NET relaunch 2025-06-24; select-country Key Messages from July 2025) is paraphrased from search summaries and not fully verified.
- **Consequence under the approved contracts (Codex ruling, 2026-10-02).** The cutoff stays at origin-month end.
  - Origin-month CS enters `hist_phase_o00` and ordinary persistence only if it was actually released by the cutoff. Otherwise it stays NaN, and persistence uses the last genuinely released label with its date and age. k=0 means no simulated extra outage; it does not guarantee fresh origin IPC.
  - Implementation note: `fourclass_features._area_history` uses `searchsorted(om, origins, side="right")` (origin month included), and `build_baselines` uses exact-origin persistence. Both must consume a release-eligibility ledger rather than reference months.
  - **Without a defensible documented or verified release rule for historical CS, the cycle mask (G2), the gate dates, the historical calendar and every IPC-derived feature are undefined. Dependent real fitting remains BLOCKED per D7.**
- **Expert comparators.**
  - FEWSNET.csv describes `fews_proj_near` as "Estimated IPC phase for the following 4 month period" and `fews_proj_med` as "…following 8 month period" (`Column Descriptions.xlsx`).
  - The raw FDW validity windows are ML1 = D..D+3 and ML2 = D+4..D+7. Under these windows, the legacy T−4 near join targets a month outside ML1's stated interval.
  - The FEWSNET.csv producer, i.e. how its row month relates to the FDW `reporting_date`, is not available locally, so the mismatch is **unresolved**, not declared.
  - Per D5, a comparator whose genuine projected interval excludes T is unavailable/mismatched with a coverage reason. It is never relabelled to another horizon, and model horizons/cutoffs do not move. The legacy expert F1 (~0.81) is not evidence of comparator validity.
  - FDW timestamps also suggest that outlook projections issued at O land after O's month end. This needs the same release rule.

## 5. Panel producers and climate rolling — VERIFIED defect; z-scores EXCLUDED

See `research/d7-panel-climate-lineage.md`.

- Producer: `Step5_Geo_RF_trial/assemble_latest_FEWSNET/02_preprocess_and_combine.ipynb`.
  - Cell 16 is a global `rolling_mean(12, min_periods=1)` with no per-admin grouping.
  - Cell 17 standardises with the per-admin **full-sample** mean and standard deviation.
- Empirical reproduction ("global roll, then admin full-sample z"): `Tair_zscore` matches 1.0000 in both panels; `Rainf_zscore` matches 1.0000 in normalized-v1 and 0.9944 in the pinned panel, where the residual is ±inf rows. This is not a literal all-value match there; every per-admin variant matches about 0%. The z-scores therefore carry cross-admin contamination and look-ahead, and are **excluded**. Raw FLDAS means are unaffected.
- Value agreement between the pinned panel and the 2025 combined/normalized-v1 panels for 2010–2024 covariates was not checked. It must pass before any 2025 input is drawn from the unpinned file.
- The 2025-only panel lacks `nightlight`, `nightlight_sd`, `EVI`, `gpp_mean`, `CPI`, `GDP`, `CC`, `gini`, `pop` and the z-scores. Normalized-v1 has these columns, but `CPI`/`GDP`/`CC`/`gini` are empty from 2024-06 onward and `gpp_mean` is empty from 2025-01.

## 6. Covariate families: candidate / excluded (no silent omission; NOT a frozen allowlist)

Source: `research/d7-covariate-release-rules.md` (table at lines 37–60, URLs retrieved 2026-10-02). No family has archived vintages, so anything later admitted is a **disclosed retrospective reconstruction from revised values**. "Candidate" means not ready. A candidate becomes admissible only after its provider URL and quoted rule, and its raw-to-panel feature lineage, are recorded and reviewed. Pending proposals stay out of any frozen schema.

| Family (schema columns) | Status | Evidence / reason |
|---|---|---|
| ACLED: 6 event/fatality counts × {plain, `_w5`, `_w10`} + `distance_to_nearest_acled` (19) | CANDIDATE, monthly L=1 | provider: weekly Monday release covering the prior Sat–Fri (acleddata.com update-schedule page). Lineage from raw export to panel not yet recorded. 2025 rows fall to 0 by 2025-05, so a June 2025 origin would be natively NaN. |
| FLDAS `Rainf_f_tavg_mean`, `Tair_f_tavg_mean` (2) | CANDIDATE, monthly L=1 | provider: final release ~20th of the next month (NASA LDAS specifications; GES DISC README). Lineage not yet recorded; values revised (Nov 2020 reprocessing). |
| `Tair_zscore`, `Rainf_zscore` (2) | EXCLUDE | cross-admin rolling plus full-sample standardisation (section 5) |
| GDP (`NY.GDP.PCAP.PP.KD`) (1) | CANDIDATE, annual | WDI July update adds Y−1. Also requires verifying that the panel stores annual values at their reference year; panel cells are empty from 2024-06. PPP series rebased May 2024. |
| CPI (`FP.CPI.TOTL.ZG`) (1) | EXCLUDE unless a CPI-specific source rule is verified | the GDP/WDI release pattern does not establish CPI release dates |
| CC (WGI percentile rank) (1) | CANDIDATE, annual | edition released Sep–Dec of Y covers Y−1; fixed April 2025 pre-WGI-2.0 download; same reference-year condition as GDP |
| EVI + `EVI_l1..l12` (13) | EXCLUDE | no documented latency; product ID conflict |
| nightlight, nightlight_sd, nightlight_m12 (3) | EXCLUDE | no documented latency; source identity conflict; 2025 source not located |
| gpp_mean (1) | EXCLUDE | annual batch release; 65,535 fill values; empty from 2025-01 |
| FAO_price, market_distance (2) | EXCLUDE | no documented publication lag. The FPMA bulletin's 2-month staleness criterion is a usage rule, not release-lag evidence. |
| Food_CPI, Food_food_inflation (2) | EXCLUDE | quarterly release, coverage lag undocumented |
| WFP_Price, WFP_Price_std, WFP_Price_m4/_m12 (4) | EXCLUDE | no documented latency; ALPS includes model-forecast months |
| gini (1) | EXCLUDE | irregular survey years |
| pop (1) | EXCLUDE | semantics unknown; populated only on IPC label months, so IPC-pipeline-linked |
| crop, range (2) | CANDIDATE static, **as-of flag** | JRC ASAP masks v02, file dated 2023. Land-cover reference content is time-varying, so a 2023 mask is not automatically as-of eligible for 2018 origins merely because the column repeats. Needs defensible fixed semantics plus disclosure, or exclusion. |
| AEZ_* (17) | CANDIDATE static, **as-of flag** | WorldCereal AEZ v100 is a recent product (the codebook says FAO/IIASA: identity conflict); same as-of caution |
| market_access (1) | CANDIDATE static, **as-of flag** | Nelson 2019 (2015 reference year); reference-year content predates or postdates some origins |
| lat, lon (2); sg_* SoilGrids (5); ruggedness (1) | CANDIDATE static | fixed geometry/terrain/soil semantics; ruggedness appears percentile-transformed (fixed percentile set required) |
| elevation, slope, distance_to_river (3) | CANDIDATE static, **identity unresolved** | conflicting source records and units; a decision is required before freezing |
| calendar: target_year, target_month_sin/cos (3) | RETAIN | known calendar |
| IPC history blocks (75) | gated by the release ledger | **BLOCKED** with section 4 |

Counts over the 87 non-IPC schema columns (asserted from schema names by `research/probes/d7_covariate_counts.py`):

- excluded: 30, including the legacy-derived WFP_m4/m12, nightlight_m12 and EVI_l1..l12;
- calendar retained: 3;
- candidates: 54 = 21 monthly + 2 annual (GDP, CC) + 31 static (28 static sources + crop, range, market_access). Of the static candidates, 23 carry an as-of or identity flag: 17 AEZ, crop, range, market_access, elevation, slope and distance_to_river.

No covariate family is currently certified ready. Before any real fit, Codex must review the candidate list and approve the frozen ordered schema.

## 7. Schedules, roles and fit ceiling

- Label months and per-month labelled areas: `probes/d7_label_calendar_n.json`. 2018–2020 targets have 5,364–5,506 areas in 20–21 countries; 2021-06 drops to 4,385.
- **Provisional fit bound (pre-mask upper bound, not certified lawful N).**
  - Computed keys-only, from historical FEWSNET.csv admin code, year/month and the non-null presence of the CS field; no label value, distribution or score was used.
  - The maximum number of distinct labelled areas in any `[O−59, O)` window, over origins 2010-01..2025-06, is N ≤ **5,714** (argmax origin 2024-03). floor(N/50) = 114, so the provisional total is ≤ **40,824 + 931 × 115 = 147,889 fits**. The full 5,718-code universe gives the same floor.
  - Lawful N can only shrink after release/outage masks and the 2025 crosswalk.
  - Support floors are counted on original keys only (rows/areas/dates/classes), never on augmented copies.
  - Certified N and fit bound await sections 3–4.
- The schedules themselves are fixed by the approved design: 648 candidates; 72 development folds; at most 57 historical and at most 4 actual-2025 folds. Their exact eligible dates (historical H4/H8 first targets, the latest-six gate dates, and which 2025 cycles count as missed per country/product) depend on the release rule. **BLOCKED with section 4.** Illustrative only, not adopted:
  - if origin-month CS is released by the cutoff, the first common-eligible origin is 2021-06 (H4 from 2021-10, H8 from 2022-02: 10/9 targets);
  - if it is released one month later, the latest two cycles at origin O are O−4 and O−8, so the first eligible origin is 2021-10 (H4 from 2022-02, H8 from 2022-06: 9/8 targets, at most 51 folds).

## 8. Minimal change boundary for later implementation (recorded, not executed)

Recorded only, ahead of eventual product edits. GitNexus `impact` is required per symbol before editing; `detect_changes` currently fails with the LadybugDB read-only shadow-page replay error, so fallback source inspection is disclosed.

- `src/feature/fourclass_features.py` `history_features/_area_history`, and `scripts/prepare_fourclass.py` `build_observations/build_snapshot/build_baselines/gate_dates/build_schedule`. Gap: IPC visibility keyed by reference month. Needed: a release-eligibility ledger plus outer/inner cycle masks; keys carry a scenario; persistence source month/age comes from the last released label.
- `covariate_features`: gap: exact-origin covariates. Needed: per-family source month O−L and an annual latest-eligible rule; excluded families are removed from the ordered schema.
- `src/model/native_xgb.py` continuation (`:211–223`) lacks sample weights; support (`:259–265`) counts rows. Needed: B weights w/3, and support counted on original keys.
- `src/experiment/stage3.py`:
  - `gate_decision` uses macro-F1 → crisis F1, gain strictly > 0.01.
  - **Codex trap 1:** `GlobalStore.get` memo key `(H, origin, G)` (`:94–96`) returns before any identity check. Both the memory key and the disk identity must include strategy, scenario k, the outer mask, fitting keys, feature boundary and weights.
  - **Codex trap 2:** `Panel` casts `class_code` to int64 (`:57`), and `run_fold` raises on targets without labels (`:194–195`). June 2025 forecast-only targets need a prediction cohort separate from evaluator truth, never fake labels.
- Stage 1 schedule validators are hard-coded to old roots (`scripts/run_stage1.py:72,101–120`). A new schedule identity is needed; do not relabel the old one.
- Code can proceed against **synthetic source-ledger fixtures** while real-run readiness is blocked. No real fits until sections 3–4 resolve.

## 9. Verdict (bounded D7 pass, 2026-10-02)

**Real fitting: BLOCKED.** Code implementation can proceed against synthetic source-ledger fixtures.

| D7 item | Verdict | Blocking evidence / next fact needed |
|---|---|---|
| Code identity, frozen Windows environment | PASS | section 1 |
| Pinned training source hashes, historical admin keys | PASS | section 2; 2996/1667 name anomalies recorded |
| Historical IPC release rule (cycle mask G2, gate dates, calendar, IPC-derived features, ordinary persistence) | **BLOCKED** | No publication-date field; FDW `created` values are database events (pre-2021 bulk re-ingested; 2021–2023 Feb cycles range M+0..M+7 for 4 countries). No official historical publication-lag source found in the bounded search. Needed: official FEWS NET release evidence (e.g. dated report postings) supporting a documented reconstruction. M+1 and M+2 are not adopted. |
| Actual 2025 availability per country/product | PARTIAL | local files show no CS 2025-01..09 and no projections reported Feb–Jul 2025; relaunch evidence is unverified. Exact per-country 2025 missed-cycle intensity depends on the release rule. |
| October 2025 truth crosswalk | **BLOCKED (dependent 2025 keys only)** | name join only: 1,093 of 5,573 October 2025 CS rows unmatched (DRC wholly; Ethiopia 645/1,141; smaller losses in 7 others); FEWS_2025.csv zero-fills unmatched crisis flags. Needed: an approved, documented crosswalk. June 2025: forecast/coverage-only. |
| Expert comparator alignment | UNRESOLVED | the raw ML1 window D..D+3 vs the legacy T−4 join; the FEWSNET.csv producer is not local; mismatched keys remain unavailable with a reason (D5) |
| Climate rolling producer | PASS as a finding → z-scores EXCLUDED | section 5 |
| Covariate release rules | PARTIAL | 54 candidates (21 monthly L=1, 2 annual, 31 static with 23 flagged) and 30 excluded; no family certified until URL/quote/lineage records and Codex schema approval. 2025 covariates need pinned-vs-2025 value agreement (not run). |
| Schedules/roles | design fixed; exact eligible dates **BLOCKED** with the release rule | section 7 |
| N-based fit ceiling | PROVISIONAL ≤ 147,889 | section 7 |
| Change boundary | recorded | section 8, including both Codex traps |

GitNexus `detect_changes` was not used for this packet: no product code changed, and the known LadybugDB shadow-page error persists. Direct source inspection was used and is disclosed.
