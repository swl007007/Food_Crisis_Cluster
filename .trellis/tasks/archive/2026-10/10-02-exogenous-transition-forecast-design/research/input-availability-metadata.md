# Input availability — metadata-only probe, 2026-10-02

Scope: CSV headers and date counts, plus explicitly selected schema/provenance fields of the normalisation sidecar. No outcome values, predictions or scores inspected. Source row existence does not establish nonmissing covariates, genuine labels, unique regions or historical availability.

External source directory: `Analysis/1.Source Data/assembled_FEWSNET/`.

- `FEWSNET_forecast_unadjusted_bm_2025.csv`: year/month rows cover January 2025–April 2026; 5,718 rows per month except October 2025 and February 2026 (5,719). Header has no obvious release/publication/vintage/as-of field.
- `FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.csv`: date rows cover January 2010–April 2026, with 5,718 rows per month (196 months). Header has no obvious release/publication/vintage/as-of field. Presence of origin-month rows does not certify usable origin-available features.
- Sidecar `FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.audit.json`: schema `fewsnet-panel-normalization-v1`; normalisation `deduplicate-before-global-rolling-zscore-v1`; key columns `FEWSNET_admin_code, feature_month`; sort columns `FEWSNET_admin_code, date, source_row_number`. Reports output 1,120,728 rows/88 columns and SHA256 `510375f58cd835e694b6e287cce9439bbe1b6246d752daabc8151df8ffdda61d`. This hash is sidecar-reported, not independently recomputed here.

The sidecar describes climate derivation with grouping column `FEWSNET_admin_code`, window 12, minimum periods 1, ddof 1, but rolling order `global_after_stable_admin_date_sort`. This is ambiguous: inspect the producer before reuse. If rolling calculations cross administration boundaries, they could mix one region's late observations into another region's early rows. This probe does not establish that such contamination occurred. Do not certify precomputed climate z-scores from this metadata alone.

## Confirmed G2 availability policy

Use verified historical release/vintage information where available. Where archived vintages cannot be recovered, permit a clearly labelled retrospective reconstruction using source-documented publication lags and a prespecified conservative cutoff. Disclose use of revised/latest values and do not call this an exact real-time replay. If even a defensible release lag cannot be established, omit/mask that feature at the relevant cutoff rather than assuming source month equals availability. Apply the policy consistently to development and final cases. Freeze specific source-family lags before fitting; no arbitrary common lag is approved here.

User adopted this policy on 2026-10-02. This changes the evidentiary claim where archived versions are absent: reconstruction under documented release assumptions, not verified real-time replay. It does not relax the prohibition on future IPC, hidden IPC derivatives or final-outcome tuning. Specific source lags, source eligibility and cutoff dates still require verification before fitting.
