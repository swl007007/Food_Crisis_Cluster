"""D7: reconcile section-6 covariate classification against feature-schema.json names.

Semantic classes (candidate monthly/annual/static-like, excluded) are kept distinct from the
legacy schema groups (static_sources, dynamic_sources_at_origin, ...)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
d = json.loads((ROOT / "FEWSNETGeoXGBExperiment" / "feature-schema.json").read_text())
non_ipc = d["static_sources"] + d["dynamic_sources_at_origin"] + d["legacy_covariate_derived"] + d["known_calendar"]
excluded = {"Tair_zscore", "Rainf_zscore", "CPI", "EVI", "nightlight", "nightlight_sd", "nightlight_m12",
            "gpp_mean", "FAO_price", "market_distance", "Food_CPI", "Food_food_inflation", "WFP_Price",
            "WFP_Price_std", "WFP_Price_m4", "WFP_Price_m12", "gini", "pop"} | {f"EVI_l{k}" for k in range(1, 13)}
calendar = set(d["known_calendar"])
monthly = {c for c in non_ipc if c.startswith(("event_count", "sum_fatalities", "distance_to_nearest_acled"))} \
    | {"Rainf_f_tavg_mean", "Tair_f_tavg_mean"}
annual = {"GDP", "CC"}
static_like = set(d["static_sources"]) | {"crop", "range", "market_access"}
flagged = {c for c in static_like if c.startswith("AEZ_")} | {"crop", "range", "market_access", "elevation",
                                                             "slope", "distance_to_river"}
classes = [excluded, calendar, monthly, annual, static_like]
assert set(non_ipc) == set().union(*classes), set(non_ipc) ^ set().union(*classes)
assert sum(map(len, classes)) == len(set(non_ipc)) == len(non_ipc) == 87
counts = {"non_ipc": len(non_ipc), "excluded": len(excluded), "calendar": len(calendar),
          "monthly": len(monthly), "annual": len(annual), "static_like": len(static_like),
          "candidates": len(monthly) + len(annual) + len(static_like), "flagged_static": len(flagged),
          "aez": sum(c.startswith("AEZ_") for c in non_ipc)}
assert counts == {"non_ipc": 87, "excluded": 30, "calendar": 3, "monthly": 21, "annual": 2, "static_like": 31,
                  "candidates": 54, "flagged_static": 23, "aez": 17}, counts
print(json.dumps(counts))
