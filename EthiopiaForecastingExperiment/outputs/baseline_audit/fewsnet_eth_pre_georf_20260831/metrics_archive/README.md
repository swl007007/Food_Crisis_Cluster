# FEWSNET-ETH metric archive

This directory archives metric tables produced by the read-only
`fewsnet_eth_pre_georf_20260831` audit.

- No production GeoRF pipeline was run.
- The diagnostic random forests are lightweight overfitting probes, not GeoRF
  reproductions or candidate-selection evidence.
- Released GeoRF tables were recalculated after filtering frozen global
  prediction artifacts to the authoritative `ISO3 == "ETH"` cohort via
  `FEWSNET_admin_code`.
- Horizon summaries use pooled-micro counts across the retained ETH rows.
  Target-month macro metrics are intentionally not included.
- `SHA256SUMS.txt` covers every archived CSV plus this README and the manifest.

See the parent directory's `README.md`, `SOURCE_MANIFEST.csv`,
`reproduction.json`, and `grill_reverse_stress_test.md` for the full data,
lineage, assumptions, and reverse stress test.
