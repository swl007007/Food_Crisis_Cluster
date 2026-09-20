"""Independent exact-rational classification of all source target candidates."""
from pathlib import Path
from fractions import Fraction
import json
import pandas as pd

BASE = Path("C:/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis")
SOURCE = BASE / "1.Source Data/assembled_IPCCH/raw/IPCCH_2026_completed.csv"
RUN = BASE / "2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d"
columns = ["admin_code", "year", "month", "estimated_population"] + [f"phase{i}_percent" for i in range(1, 6)]
source = pd.read_csv(SOURCE, usecols=columns, dtype=str, keep_default_na=False)[columns]
valid = {}
reasons = {}
filled = 0
ties = 0
for row in source.itertuples(index=False, name=None):
    area, year, month, population, *raw = row
    key = (int(area), f"{int(year):04d}-{int(month):02d}")
    try:
        parts = [Fraction(value) if value else None for value in raw]
        pop = Fraction(population) if population else None
    except (ValueError, ZeroDivisionError):
        reasons["invalid_numeric"] = reasons.get("invalid_numeric", 0) + 1
        continue
    if any(value is None for value in parts[:4]):
        reason = "missing_phase_1_to_4"
    elif any(value is not None and not 0 <= value <= 1 for value in parts):
        reason = "component_bounds"
    elif pop is None or pop <= 0:
        reason = "population"
    else:
        p5_filled = parts[4] is None
        parts[4] = parts[4] or Fraction(0)
        total = sum(parts)
        if not Fraction(9, 10) <= total <= Fraction(11, 10):
            reason = "total_bounds"
        else:
            crisis = 5 * sum(parts[2:])
            valid[key] = int(crisis > total)
            ties += crisis == total
            filled += p5_filled
            continue
    reasons[reason] = reasons.get(reason, 0) + 1
saved = pd.read_csv(RUN / "data/target_ledger_valid.csv.gz")
actual = {(int(r.admin_code), str(r.target_month)[:7]): int(r.ipcch_food_crisis) for r in saved.itertuples()}
if valid != actual:
    print('independent_counts',len(valid),sum(valid.values()),'saved',len(actual),'reasons',reasons)
    print('missing_examples',list(actual.keys()-valid.keys())[:8])
    print('extra_examples',list(valid.keys()-actual.keys())[:8])
    print('different_examples',[(k,valid[k],actual[k]) for k in valid.keys()&actual.keys() if valid[k]!=actual[k]][:8])
assert valid == actual
assert len(source) == len(valid) + sum(reasons.values())
assert (len(valid), sum(valid.values()), filled, ties) == (42695, 15206, 84, 2601)
result = {"source_rows": len(source), "valid": len(valid), "positive": sum(valid.values()),
          "valid_label_areas": len({k[0] for k in valid}), "p5_fills": filled, "exact_threshold_ties": ties,
          "invalid_counts_independent_reason_order": reasons, "all_saved_valid_keys_and_labels_equal": True}
(Path(__file__).parent / "fraction-target-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
