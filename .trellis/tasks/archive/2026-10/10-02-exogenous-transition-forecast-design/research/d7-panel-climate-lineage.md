# D7: panel producer and climate rolling lineage (partial; scout stopped at its time limit)

Date: 2026-10-02. Consolidated by the executor from the stopped scout's saved probes plus one bounded producer read. Covariate columns only: no `fews_*` value column was read, and notebook outputs were not opened (source cells only, via `probes/d7_nb_grep.py`).

## Verified

- **Producer** of `assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.csv`: `Analysis/2.source_code/Step5_Geo_RF_trial/assemble_latest_FEWSNET/02_preprocess_and_combine.ipynb`. Its `to_csv` is in cell 20. The same folder holds `00_prepare.ipynb` and `01_extract_invariables_and_scaffold.ipynb`.
  - Cell 8 drops the old `Rainf_zscore`/`Tair_zscore` from `df_previous`.
  - Cell 16: `pl.col(variant).rolling_mean(window_size=12, min_periods=1)` for each `time_variant`, with **no `.over(admin)`**. This is a global rolling mean over the admin/date-sorted frame, so each admin's first 11 months mix the previous admin's last months.
  - Cell 17: `Tair_zscore`/`Rainf_zscore = (x_m12 − mean().over(admin)) / std().over(admin)`. This is a **full-sample per-admin standardisation**, i.e. it uses every month in the file, including months after any origin.
- **Empirical reproduction** (`probes/d7_zscore_variants.py`, log `probes/d7_zscore_variants_log.txt`). The variant "global roll, then admin full-sample z" reproduces the stored values exactly; every per-admin rolling variant matches about 0%.

  | Panel | `Tair_zscore` match | `Rainf_zscore` match |
  |---|---|---|
  | Pinned panel (2010-01..2024-12) | 1.0000 (max abs 8e-11) | 0.9944 (±inf rows) |
  | normalized-v1 (2010-01..2026-04) | 1.0000 | 1.0000 |

  So the pinned panel's z-scores carry the same construction, standardised over 2010–2024.
- **Consequence:** stored `Tair_zscore`/`Rainf_zscore` contain cross-admin contamination plus look-ahead from full-sample standardisation, in both the pinned and the 2025 panels. Under D1 they cannot be certified and are **excluded**. Building an origin-legal replacement z-score would be a new feature construction, which needs design review; it is not proposed here.
- **Non-null counts by month** (`probes/d7_nonnull_by_month.py`, log `probes/d7_nonnull_log.txt`; covariates only).
  - The 2025-only panel header lacks `nightlight, nightlight_sd, EVI, gpp_mean, CPI, GDP, CC, gini, pop, Tair_zscore, Rainf_zscore`.
  - normalized-v1, 2025-01 → 2025-06:

    | Column(s) | Non-null rows |
    |---|---|
    | ACLED (e.g. `event_count_battles`) | 4,414 / 4,301 / 4,263 / 3,132 / **0** / **0** |
    | `gpp_mean` | 0 from 2025-01 |
    | `CPI`, `GDP`, `CC`, `gini` | 0 from 2024-06 onward |
    | `FAO_price` | ~5,450 |
    | `WFP_Price` | ~2,993 |
    | `Food_CPI` | 4,796, then falls to 0 by 2025-10 |
    | FLDAS raw, `crop`, `market_access` | 5,718 throughout |

  - `pop` is non-null only in 2024-06 and 2024-10 (and earlier label months): it is populated only on IPC label months. That fits the `pop` column of FEWSNET.csv (outcome file). Inference: `pop` is IPC-pipeline-linked, so its presence encodes label presence.

## Unresolved

1. Value agreement between the pinned panel and the 2025 combined / normalized-v1 panels over 2010-01..2024-12 for the retained covariates. Keys are identical (`d7-sources-admin.md` section 5), but values were not compared before the stop. This must pass before 2025 inputs are drawn from an unpinned file.
2. Which `time_variants` cell 16 covers, beyond the climate means. Legacy `_m12` features in the GeoXGB schema are recomputed per area by `fourclass_features.trailing_sum`, so they do not inherit cell 16.
3. The normalized-v1 producer (deduplicate-then-global-rolling) itself was not located. Only the combined-panel producer was read. The sidecar's "global_after_stable_admin_date_sort" agrees with cell 16.
4. Producer of the pinned 2024-12 panel: not read. It is inferred to be the same lineage from the identical z-score signature.
