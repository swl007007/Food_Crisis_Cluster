"""D7: October-2025-only name-join coverage. Metadata columns only (scenario, reporting_date,
country, geographic_unit_full_name; FEWSNET.csv admin_name). No value/description column read."""
import pandas as pd
SRC = "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/Outcome/FEWSNET_IPC/"
raw = pd.read_csv(SRC + "2025_2026_FEWSNET.csv", encoding="utf-8-sig",
                  usecols=["scenario", "reporting_date", "country", "geographic_unit_full_name"])
names = set(pd.read_csv(SRC + "FEWSNET.csv", usecols=["admin_name"])["admin_name"].dropna())
oct25 = raw[(raw["scenario"] == "CS") & raw["reporting_date"].astype(str).str.startswith("2025-10")].copy()
oct25["matched"] = oct25["geographic_unit_full_name"].isin(names)
t = oct25.groupby("country")["matched"].agg(total="size", matched="sum")
t["unmatched"] = t["total"] - t["matched"]
print("rows", len(oct25), "countries", oct25["country"].nunique(), "matched", int(oct25["matched"].sum()),
      "unmatched", int((~oct25["matched"]).sum()))
print(t[t["unmatched"] > 0].sort_values("unmatched", ascending=False).to_string())
print("wholly unmatched:", list(t.index[t["matched"] == 0]))
