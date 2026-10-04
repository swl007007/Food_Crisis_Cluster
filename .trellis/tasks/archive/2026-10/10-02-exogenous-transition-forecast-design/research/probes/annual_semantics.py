"""Launch prep: is each annual covariate constant within calendar year in the pinned panel
(i.e. stored at its reference year)? Reads key + GDP/CC columns only."""
import sys
import pandas as pd
path = sys.argv[1]
p = pd.read_csv(path, usecols=["FEWSNET_admin_code", "date", "GDP", "CC"])
p["year"] = p["date"].astype(str).str[:4].astype(int)
for c in ("GDP", "CC"):
    g = p.dropna(subset=[c]).groupby(["FEWSNET_admin_code", "year"])[c].nunique()
    yrs = sorted(p.dropna(subset=[c])["year"].unique())
    print(c, "area-years", len(g), "varying within year", int((g > 1).sum()),
          "years with values", yrs[0] if yrs else None, "-", yrs[-1] if yrs else None)
