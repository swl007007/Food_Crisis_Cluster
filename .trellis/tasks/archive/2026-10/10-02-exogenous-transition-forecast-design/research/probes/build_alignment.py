"""Launch prep: the minimal covariate alignment (D7 decisions; no new choices).

Monthly ACLED (19) and FLDAS raw means (2): lag 1 (provider rules; FLDAS corroborated in
research/d7-fldas-release-corroboration.md). Annual GDP: WDI July update adds Y-1. Annual CC:
WGI edition covers Y-1 released Sep-Dec of Y -> conservative Y-2. Static sources: user-attested
pinned-panel identity (as-of content disclosed in data-readiness section 6). Everything else
excluded per D7 section 6. value_month for annual sources is set by probes/annual_semantics.py."""
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[5] / "FEWSNETGeoXGBExperiment"
schema = json.loads((PKG / "feature-schema.json").read_text(encoding="utf-8"))
value_month = int(sys.argv[1]) if len(sys.argv) > 1 else None
ACLED = [c for c in schema["dynamic_sources_at_origin"] if c.startswith(("event_count", "sum_fatalities",
                                                                        "distance_to_nearest_acled"))]
FLDAS = ["Rainf_f_tavg_mean", "Tair_f_tavg_mean"]
STATIC_LIKE = list(schema["static_sources"])   # stable terrain/soil/geometry + fixed AEZ grouping
#: crop/range (2023 masks) and market_access (2015 reference) have unestablished historical as-of
#: semantics -> excluded under D1 (coordinator review 2026-10-03), no alternate arm.
EXCLUDED = ["Tair_zscore", "Rainf_zscore", "CPI", "EVI", "nightlight", "nightlight_sd", "gpp_mean", "FAO_price",
            "market_distance", "Food_CPI", "Food_food_inflation", "WFP_Price", "WFP_Price_std", "gini", "pop",
            "crop", "range", "market_access"]
out = {}
for c in ACLED:
    out[c] = {"kind": "monthly", "lag": 1, "status": "reconstructed",
              "evidence": "ACLED weekly Monday release of prior Sat-Fri (research/d7-covariate-release-rules.md)"}
for c in FLDAS:
    out[c] = {"kind": "monthly", "lag": 1, "status": "reconstructed",
              "evidence": "FLDAS Noah01 C ~3-week release; _f_ forcing unchanged by Nov2020 reprocessing "
                          "(research/d7-fldas-release-corroboration.md)"}
for c in STATIC_LIKE:
    out[c] = {"kind": "static", "status": "reconstructed",
              "evidence": "user-attested pinned panel source identity; static reconstruction semantics; "
                          "identity/units caveats disclosed (data-readiness s6)"}
for c, delay, month, ev in (("GDP", 1, 7, "WDI July update adds reference year Y-1"),
                            ("CC", 2, 1, "WGI edition Sep-Dec of Y covers Y-1; conservative Y-2 at any origin")):
    out[c] = {"kind": "annual", "release_delay_years": delay, "release_month": month,
              "value_month": value_month, "status": "reconstructed", "evidence": ev}
for c in EXCLUDED:
    why = ("historical as-of semantics unestablished (D1)" if c in ("crop", "range", "market_access")
           else "excluded per D7 data-readiness section 6")
    out[c] = {"kind": "excluded", "status": "reconstructed", "evidence": why}
sources = schema["static_sources"] + schema["dynamic_sources_at_origin"]
assert set(out) == set(sources), sorted(set(sources) ^ set(out))
print(json.dumps(out, indent=1))
