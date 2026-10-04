# D7 — FEWS NET IPC publication-cycle / release ledger (read-only scout)

Date: 2026-10-02. Read-only. No IPC phase/value/description/HA value was read for any row dated 2025+.
Historical (<=2024) `FEWSNET.csv` used only for date/admin/country and non-null presence.
Geojson chunks were NOT json-parsed: a byte-level regex (mmap) extracted only these keys:
`reporting_date, scenario, country, projection_start, projection_end, collection_status,
source_document, datasourcedocument, collection_schedule, created, modified, status_changed,
collection_status_changed`, plus a count of `"type": "Feature"`. Probe scripts: `/tmp/d7/a1.py`
(FEWSNET.csv presence), `/tmp/d7/a2.py` (2025_2026 CSV metadata), `/tmp/d7/a3.py` (geojson metadata).

`SRC=/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data`

## 1. Verified facts (local files)

### 1a. Historical CS cycles in `FEWSNET.csv` (presence only)

53 `year_month` values: quarterly 2009_07..2015_10, then Feb/Jun/Oct 2016_02..2024_10. No publication,
vintage, created or modified field (header: `country, admin_code, year_month, year, month, fews_ipc,
fews_ha, fews_proj_near, fews_proj_near_ha, fews_proj_med, fews_proj_med_ha, pop, pop_source,
fews_ipc_adjusted, fews_proj_med_adjusted, admin_name`). 5,716 admin rows per month; CS non-null
4,292–5,697; `fews_proj_near/med` non-null from 2011_04.

Country-cycle gaps 2016-02..2024-10 (0 or ~0 admins with CS that cycle, while present before/after):

| Country | Missing CS cycles |
|---|---|
| Afghanistan | 2021-10, 2022-02, 2022-06, 2022-10, 2023-02 |
| Burkina Faso | absent 2016-02..2021-06 (enters 2021-10) |
| Burundi | 1 admin only 2016-02..2019-10; missing 2022-10 |
| Cameroon | absent before 2020-02 |
| DRC | absent 2016-02, 2016-06; 200–205 admins to 2023-10, 345 from 2024-02 |
| Ethiopia | 2021-06 (1 of 1,040 admins) |
| Somalia | 2022-06, 2023-02 |
| South Sudan | 2020-10, 2021-02, 2021-06, 2021-10, 2022-02 |
| Sudan | 2024-10 |
| Yemen | 2017-02, 2023-10 |
| Malawi | admin count drifts 83→66 |

Other countries (Chad, Guatemala, Haiti, Kenya, Madagascar, Mali, Mozambique, Niger, Nigeria, Uganda,
Zimbabwe) present every cycle. Whether gaps are unpublished cycles or extraction losses is not
determinable from this file (no vintage field).

### 1b. `2025_2026_FEWSNET.csv` metadata (12,086 rows)

| reporting_date | scenario | projection_start..end | rows | countries | source_document |
|---|---|---|---:|---:|---|
| 2024-10-01 | ML1 | 2024-10-01..2025-01-31 | 323 | 6 (Uganda 318 + 5 RM 1-row) | FSO Uganda; Remote Monitoring Report (Angola, Lesotho, Mauritania, Nicaragua, Togo) |
| 2024-10-01 | ML2 | 2025-02-01..2025-05-31 | 323 | same 6 | same |
| 2025-10-01 | CS | 2025-10-01..2025-10-31 | 5,573 | 21 | "Food Security Outlook, <country>" |
| 2026-02-01 | CS | 2026-02-01..2026-02-28 | 5,867 | 27 (adds CAR, El Salvador, Honduras, Lebanon, Syria, Venezuela) | "Food Security Outlook, <country>" |

All rows `collection_status=Published`, `collection_schedule=Ad Hoc`. No created/modified/publication
column in this CSV. No Feb 2025 or Jun 2025 CS rows. Record `id` ranges increase with cycle
(37.17M / 39.21–40.21M / 41.12–41.62M) — ordering only, not dates.

Note: the CSV's ML1 for reporting month D covers D..D+3 and ML2 covers D+4..D+7 (validity windows).
The project convention "ML1 published D targets D+4" therefore scores ML1 one month past its stated
validity window; ML2 (D+4..D+7) does cover D+4, not D+8. Flag for D7/E-design review; not resolved here.

### 1c. Scraper provenance (`$SRC/Outcome/FEWSNET_IPC/scrape_fewsnet.py`, `.log`)

- Endpoint `https://fdw.fews.net/api/ipcphase.geojson`, params `start_date`, `end_date`, `scenario`
  (CS/ML1/ML2), monthly windows 2025-01-01..2026-04-01 (exclusive), run 2026-04-29.
- Inferred filter semantics (from chunk contents): the date window selects on **projection_end**
  (e.g. ML2 of reporting 2024-10 with validity Feb–May 2025 appears only in the May-2025 window).
  Consequence: the scrape omits products whose projection_end is >= 2026-04-01 (e.g. Feb 2026 ML1/ML2,
  Oct 2025 ML2), and includes 2024 vintages whose windows end in 2025.
- **The API exposes per-record `created`, `modified`, `status_changed`, `collection_status_changed`
  timestamps** (also `datacollection`, `datacollectionperiod`, `dataseries`, `preference_rating`).
- Merge bug: final `fewsnet_ipcphase_2025-01_2026-04.geojson` (3.29 GB) reports "merged 23628 unique
  features" though chunks hold ~78k; dedup key `(scenario, properties.id)` collapsed records. Use chunks,
  not the merged file.

### 1d. Chunk ledger (`fewsnet_chunks_2025_2026/`, 45 files)

35 chunks are 45-byte empty FeatureCollections — including **all CS windows Jan–Sep 2025** (so no Feb
or Jun 2025 CS), CS Nov 2025 and Mar 2026. Non-empty chunks:

| chunk | bytes | features | reporting_date / scenario / validity | countries | `created` range | `collection_status_changed` range |
|---|---:|---:|---|---:|---|---|
| CS_2025-10 | 208,952,587 | 5,573 | 2025-10 CS, Oct 2025 | 21 | 2025-10-29..2026-01-13 | 2025-10-31..2026-01-14 |
| CS_2025-12 | 19,050,963 | 20 | 2025-12 CS ("Food Assistance Outlook Brief, Highest FIC", country-level) | 20 | 2025-12-23 | 2026-01-26 |
| CS_2026-01 | 19,261,502 | 21 | 2026-01 CS (country-level) | 21 | 2026-01-20 | 2026-01-20 |
| CS_2026-02 | 223,704,905 | 5,867 | 2026-02 CS | 27 | 2026-02-24..2026-03-12 | 2026-02-26..2026-03-12 |
| ML1_2025-01 | 340,087,496 | 11,177 | ML1 rep. 2024-10 (5,903; Oct–Jan), 2024-11 (2,638), 2024-12 (2,636) | 32/11/10 | 2024-10-25..2026-01-13 | 2024-10-25..2026-01-16 |
| ML1_2025-09 | 182,740,411 | 3,024 | ML1 rep. 2025-08 (FSO Update) and 2025-09 (Key Message Update) | 6 | 2025-09-04..2025-10-08 | to 2025-10-08 |
| ML1_2026-01 | 837,061,614 | 22,289 | ML1 rep. 2025-10, 2025-11, 2025-12, 2026-01 | 21 each | 2025-10-29..2026-02-10 | to 2026-02-11 |
| ML2_2025-01 | 933,340,170 | 23,628 | ML2 rep. 2024-06..2024-09 (validity Oct 2024–Jan 2025) | 32 | 2024-06-28..2024-10-18 | to 2024-10-18 |
| ML2_2025-05 | 340,132,597 | 11,177 | ML2 rep. 2024-10..2024-12 (Feb–May 2025) | 32/11/10 | 2024-10-25..2026-01-14 | to 2026-01-16 |
| ML2_2026-01 | 182,752,209 | 3,024 | ML2 rep. 2025-08, 2025-09 (Oct 2025–Jan 2026) | 6 | 2025-09-04..2025-10-08 | to 2025-10-08 |

2025 products present locally: no CS until Oct 2025; **no ML1/ML2 with reporting_date Feb–Jul 2025**;
Aug/Sep 2025 ML1/ML2 for 6 countries only (Burundi, Guatemala, Kenya, Madagascar, Malawi, Mozambique;
"Food Security Outlook Update" Aug, "Key Message Update" Sep); full Oct 2025 outlook (CS+ML1, 21
countries) and monthly ML1 updates Nov 2025–Jan 2026. Feb 2025/Jun 2025 ML2 absent.

Per-country first `created` date (record-level DB timestamps):

| Cycle | earliest | most countries | latest |
|---|---|---|---|
| ML1 2024-10 (published with Oct 2024 CS) | 2024-10-25 (Haiti, Malawi, Nigeria) | 2024-10-31..2024-11-08 | Ethiopia 2024-11-21; Sudan 2026-01-13 (re-created) |
| CS 2025-10 | 2025-10-29 (Niger) | 2025-10-30..2025-11-07 | Somalia 11-10; Sudan 2026-01-13 (Kenya status change 2025-12-10) |
| CS 2026-02 | 2026-02-24 (Cameroon) | 2026-02-26..2026-03-10 | Venezuela 2026-03-12 |

By the end of the reporting month, records existed for only 7/21 countries for Oct 2025 CS
(created by 2025-10-31: Burundi, Cameroon, Guatemala, Madagascar, Malawi, Mali, Niger) and 7/27 for
Feb 2026 CS (created by 2026-02-28: Burkina Faso, Cameroon, El Salvador, Guatemala, Mali, Niger,
Zimbabwe); by `collection_status_changed` (Published) only 4/21 and 4/27. All others landed in M+1. Caveat: `created` is the current record's DB creation;
re-uploads overwrite (Sudan 2024-10 re-created 2026-01-13), so these are **lower-bound-quality evidence
of availability, not certified first-publication vintages**; website PDF release can precede DB load.

### 1e. Repo expert sources (metadata only)

`Step3ExpertCorrectionExperiment/step3correction/expert.py` joins expert at calendar origin
`O = T-H` (fs1 `fews_proj_near` at T-4, fs2 `fews_proj_med` at T-8) from `FEWSNET.csv`; legacy
record-shift kept as validation artifact only. No vintage information anywhere in the repo.

## 2. Web evidence (publication timing, suspension)

Web sub-scout, time-capped. No fews.net report pages were opened; no dated "Food Security Outlook"
posting dates were obtained. Items below are tagged.

**Verified (FDW API fetched directly, metadata only):**
`https://fdw.fews.net/api/ipcphase/?format=json&country_code=ET&page_size=2` returns fields
`... projection_start, projection_end, status, value, pct_phase3..5, description, id,
datacollectionperiod, datacollection, scenario, ..., datasourcedocument, document_type, dataseries,
..., created, modified, status_changed, collection_status, collection_status_changed,
collection_schedule, reporting_date`. **No publication-date field**; `created` /
`collection_status_changed` are the only vintage proxies. CS and ML1 rows of one outlook share
timestamps (Ethiopia Oct 2023 CS + ML1: created 2023-12-01, `collection_status_changed`
2023-12-01T18:48:20 — i.e. M+2). `/api/datasourcedocument/6537/` (FSO Ethiopia):
`"current_schedule":"Ad Hoc"`, `"last_collection_date":"2026-06-01"`. `reporting_date` as a filter is
ignored; `page_size>=1000` returns 403.

FDW `created` for **Feb CS** (source FEWS NET):

| Country | Feb 2021 | Feb 2022 | Feb 2023 | Feb 2026 |
|---|---|---|---|---|
| Ethiopia | 2021-03-25 | 2022-04-26 | 2023-04-24 | 2026-03-06 |
| Kenya | 2021-02-26 | 2022-04-01 | 2023-03-31 | 2026-03-06 |
| South Sudan | 2021-09-10 | 2022-05-10 | 2023-03-27 | 2026-03-05 |
| Yemen | 2021-03-21 | 2022-04-11 (KMU source) | 2023-04-17 | 2026-03-10 |

Pre-2021 cycles show bulk re-ingestion dates (2020-11-19/25; Ethiopia 2022-02-11 / 2024-12-09), so
`created` is uninformative for 2016–2020.

**Documented, paraphrased from search summaries (not quoted, not fully verified):**
- FEWS NET relaunch announcement (https://fews.net/fews-net-relaunches-website-resumes-global-food-security-analysis):
  relaunch 2025-06-24 with a Global Food Security Update through Sep 2025; regular monthly reporting
  to resume, starting with Key Messages for select countries in July 2025.
- Devex (https://www.devex.com/news/fews-net-once-usaid-s-flagship-famine-warning-system-is-back-online-110357):
  offline from January 2025 except one report in May; management moving to State Department.
- CNN 2025-03-09 (https://www.cnn.com/2025/03/09/world/us-foreign-aid-freeze-famine-fewsnet-intl):
  website and data offline. NPR/NEPM 2026-08-31 (https://www.nepm.org/2026-08-31/whatever-happened-to-the-u-s-famine-alert-system-that-trump-shut-down-then-reopened):
  reduced country coverage after reopening.
- Exact stop-work order date: not verified (late Jan 2025 per Wikipedia).

Consistency with local data: no Feb/Jun 2025 CS; first 2025 projections Aug 2025 (6 countries, FSO
Update) and Sep 2025 (KMU) fit "select countries" resumption; first full outlook Oct 2025.

## 3. Proposed conservative release rule (proposal, not a decision)

Two candidates; FDW `created` evidence differs by era:

- **Rule R2 (recommended conservative): CS and ML1/ML2 of outlook cycle M available at end of M+2.**
  Covers 2021–2023 FDW ingestion (Feb CS created Feb 26 – Apr 26; Ethiopia Oct 2023 on Dec 1 = M+2)
  and all 2024-10..2026-02 cycles. Known violations: South Sudan Feb 2021 (Sep 2021), Feb 2022
  (May 10, M+3); Sudan re-creates (2026-01). These are DB-ingestion dates, probably later than web
  PDF posting, so R2 is conservative w.r.t. public PDFs but not necessarily w.r.t. machine-readable
  data.
- **Rule R1: end of M+1.** Fits 2024-10, 2025-10, 2026-02 (nearly all countries created M day 24 –
  M+1 day 12) but is violated by most 2022–2023 Feb cycles in FDW (late Mar–Apr).
- Monthly KMU/Outlook-Update projections for month M landed within M+1 in 2025
  (Aug → Sep 4–9; Sep → Sep 30–Oct 8; Nov → Nov 20–25).
- Verification status: verified-vintage = none (no publication field; `created` is overwritten by
  re-uploads); reconstructed = R1/R2 from FDW timestamps for 2021–2023 Feb cycles (4 countries) and
  2024-10..2026-02 (all countries); unknown = 2016–2020 (bulk re-ingested), Jun/Oct cycles before 2023,
  website posting dates, and whether 1a gaps were late, skipped or lost.

### Consequences for the calendar (origins O ∈ {Feb, Jun, Oct}, cutoff = end of O)

| Origin cutoff | Origin-month CS (cycle O) | Latest CS available under R1 or R2 | Interruption k=1 hides | k=2 hides |
|---|---|---|---|---|
| end Feb Y | NOT available (lands ~early Mar) | Oct Y-1 | Oct Y-1 → latest Jun Y-1 | Oct Y-1, Jun Y-1 → latest Feb Y-1 |
| end Jun Y | NOT available | Feb Y | Feb Y → Oct Y-1 | → Jun Y-1 |
| end Oct Y | NOT available | Jun Y | Jun Y → Feb Y | → Oct Y-1 |

The table holds identically under R1 and R2 (both put cycle O after end of O and cycle O-4 before
end of O, since O-4 + 2 = O-2 < O). So under either rule the effective CS lag at cutoff is 4 months (H=4 target is 8 months
past the latest CS; H=8 is 12 months). Treating cycle-O CS as known at end of O would be look-ahead
for ~67–74% of countries (by `created`) in the two observed CS cycles. Expert ML1/ML2 issued at O are likewise not available
at end of O; the expert comparator "published at O" is in fact released ~O+1.

## 4. Inferences (not verified)

- The ~1–6 week post-M lag in 2024–2026 likely resembles 2016–2024 practice (FEWS NET outlook cycle
  structure unchanged), but this is unverified for older cycles.
- The 2025 hole (Feb/Jun 2025 CS missing, Feb–Jul 2025 projections missing) is consistent with the
  USAID stop-work; partial Aug/Sep 2025 updates for 6 countries suggest staged resumption.

## Unresolved

1. Historical (2016–2024) CS publication dates per country; could be pulled from FDW `created`
   via the API for older reporting dates (metadata-only query) — but re-uploads overwrite `created`,
   so web-page posting dates are the better source.
2. Whether `start_date/end_date` filters on projection_end (inferred, not documented).
3. Whether Feb/Jun 2025 CS were ever retro-published (none in the 2026-04-29 scrape).
4. Feb 2026 ML1/ML2 and Oct 2025 ML2 not scraped (outside window).
5. Gaps in 1a: skipped publications vs extraction loss.
6. ML1 validity window (D..D+3) vs project target convention (D+4).
