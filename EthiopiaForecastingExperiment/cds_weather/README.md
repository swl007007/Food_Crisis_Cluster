# CDS Weather Workstream

## Status

Deferred until the baseline diagnosis and freeze gates are complete. No formal
CDS product has been selected or downloaded.

## Locked preparation boundary

- Formal CDS rasters will be aggregated to the frozen Ethiopia
  `FEWSNET_admin_code` polygons.
- Reference-table `area_id` values will not be used as experiment join keys.
- Forecast records must distinguish forecast origin, issue/vintage, valid
  month, and lead.
- Lead 0 through lead 6 is inclusive and contains seven monthly values.
- The future harmonization contract must document units, transformations,
  ensemble statistic, spatial weighting, missingness, and any bias adjustment.
- Forecast-derived variables must not silently overwrite observed/reanalysis
  variables without preserving measurement-domain provenance.

The reference `cds_api_tif_values_by_area_time.csv` is evidence about a possible
shape and naming convention only; its area counts and provenance are not part
of the formal experiment contract.

