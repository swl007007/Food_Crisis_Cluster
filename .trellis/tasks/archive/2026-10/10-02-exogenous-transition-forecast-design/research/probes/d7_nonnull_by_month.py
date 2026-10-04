# Covariate-only non-null counts per month (2024-06..2026-04) for schema columns in the 2025 panels.
import json, sys, pandas as pd
SRC = "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"
R = "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster"
sch = json.load(open(f"{R}/FEWSNETGeoXGBExperiment/feature-schema.json"))
def flat(v):
    if isinstance(v, dict): return list(v.keys())
    return [e if isinstance(e, str) else (e.get("column") or e.get("name") or e.get("source")) for e in v]
static = flat(sch["static_sources"]); dyn = flat(sch["dynamic_sources_at_origin"])
BAN = ("fews_", "ipc", "crisis", "phase", "class")
covs = [c for c in static + dyn if not any(b in c.lower() for b in BAN)]
print("static", len(static), "dynamic", len(dyn), "covariates kept", len(covs))
print("dropped as IPC-like:", [c for c in static + dyn if c not in covs])
for name, path, datecol in [("2025", f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025.csv", None),
                            ("normv1", f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.csv", "date")]:
    hdr = pd.read_csv(path, nrows=0).columns
    use = [c for c in covs if c in hdr]
    missing = [c for c in covs if c not in hdr]
    keys = ["year", "month"] if datecol is None else ["date"]
    df = pd.read_csv(path, usecols=use + keys)
    if datecol is None:
        df["ym"] = df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2)
    else:
        df["ym"] = df["date"].astype(str).str[:7]
    df = df[(df.ym >= "2024-06") & (df.ym <= "2026-04")]
    t = df.groupby("ym")[use].apply(lambda x: x.notna().sum()).T
    t.insert(0, "rows", "")
    rows = df.groupby("ym").size()
    print(f"\n## {name}: missing schema covariate columns in header: {missing}")
    print("rows per month:", rows.to_dict())
    pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    print(t.drop(columns="rows").to_string())
