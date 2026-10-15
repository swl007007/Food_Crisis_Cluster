"""Read-only input-freeze report for the formal D30 run (no fitting, no network). Prints JSON."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.runinputs import load_run_inputs
from ipcch_alphaearth import descriptors

run = Path(sys.argv[1]); c = load_contract()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
dig = lambda xs: hashlib.sha256(json.dumps(list(xs)).encode()).hexdigest()
rep = {"run": str(run)}
rep["artifact_sha256"] = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
rep["n_artifacts"] = len(rep["artifact_sha256"])
# stage COMPLETE records: every recorded file hash equals the bytes on disk
mism = []
for stage in ("prepared", "extraction", "descriptors"):
    rec = json.loads((run / stage / "COMPLETE.json").read_text())
    for name, h in (rec.get("files") or {}).items():
        if sha(run / stage / name) != h:
            mism.append(f"{stage}/{name}")
pm = json.loads((run / "prepared" / "prepared-manifest.json").read_text())
for name, h in pm["artifacts_sha256"].items():
    if sha(run / "prepared" / name) != h:
        mism.append(f"prepared/{name} (manifest)")
rep["recorded_hash_mismatches"] = mism
rep["prepared_manifest_artifacts"] = len(pm["artifacts_sha256"])
# 6,227 admin equality: geometry / plan / descriptor table / coverage ledger
import geopandas as gpd
geo_path = next((run / "sources" / "geometry").glob("*.shp"))
geo = gpd.read_file(geo_path)
gcol = "admin_code" if "admin_code" in geo.columns else [k for k in geo.columns if "code" in k.lower()][0]
plan = pd.read_csv(run / "extraction" / "plan.csv.gz", dtype={"piece_id": str, "asset_id": str, "recipe": str})
table = pd.read_csv(run / "descriptors" / "descriptor_table.csv.gz", float_precision="round_trip")
ledger = pd.read_csv(run / "descriptors" / "coverage_ledger.csv.gz", float_precision="round_trip")
sets = {"geometry": set(geo[gcol].astype(int)), "plan": set(plan["admin_code"].astype(int)),
        "descriptor_table": set(table["admin_code"].astype(int)), "coverage_ledger": set(ledger["admin_code"].astype(int))}
rep["admin_sets"] = {k: len(v) for k, v in sets.items()}
rep["admin_sets_equal"] = len({frozenset(v) for v in sets.values()}) == 1
rep["geometry_rows"], rep["geometry_code_column"], rep["geometry_file"] = len(geo), gcol, geo_path.name
rep["descriptor_table_duplicates"] = int(table["admin_code"].duplicated().sum())
names = descriptors.ae_names(c)
rep["descriptor_columns"] = len(names)
rep["descriptor_nonfinite"] = int((~np.isfinite(table[names].to_numpy(float))).sum())
rep["coverage_min"], rep["coverage_max"] = float(ledger["coverage"].min()), float(ledger["coverage"].max())
rep["usable"] = int(ledger["usable"].sum())
rep["negative_variance_zeroed_total"] = int(ledger["negative_variance_zeroed"].sum())
# load_run_inputs assembles (no fitting)
inp = load_run_inputs(run, c)
rep["inputs_fields"] = ["horizons", "features296", "features424", "admin_codes", "D_admin", "source", "descriptor_names"]
rep["schema"] = {"features296": {"n": len(inp.features296), "sha256": dig(inp.features296)},
                 "features424": {"n": len(inp.features424), "sha256": dig(inp.features424)},
                 "descriptor_names": {"n": len(inp.descriptor_names), "sha256": dig(inp.descriptor_names)},
                 "features424_is_296_plus_descriptors": inp.features424 == inp.features296 + inp.descriptor_names}
rep["D_admin_shape"], rep["admin_codes_n"] = list(inp.D_admin.shape), int(len(inp.admin_codes))
rep["horizons"] = {}
for h, hi in sorted(inp.horizons.items()):
    keys = hi.keys
    prov = pd.read_csv(run / "prepared" / f"providers_h{h:02d}.csv.gz")
    rep["horizons"][str(h)] = {"rows": int(len(keys)), "admins": int(keys["admin_code"].nunique()),
        "countries": int(keys["country_key"].nunique()),
        "X296_shape": list(hi.X296.shape), "D_shape": list(hi.D.shape), "routable": int(hi.routable.sum()),
        "D_nan_rows_where_routable": int(np.isnan(hi.D[hi.routable]).any(axis=1).sum()),
        "Y_shape": list(hi.Y.shape), "crisis_truth_rows": int(keys["crisis_truth"].notna().sum()),
        "persistence_available": int(keys["persistence_available"].astype(bool).sum()),
        "target_months": [str(keys["target_month"].min()), str(keys["target_month"].max())],
        "providers_rows": int(len(prov)), "providers_columns": list(prov.columns)}
rep["inputs_source"] = inp.source
# 61-quartet / 244-scalar inventory
s = pd.read_csv(run / "prepared" / "schedule_requests.csv.gz")
pre = json.loads(sorted((run / "preflight").glob("*.json"))[-1].read_text())
rep["schedule"] = {"requests": int(len(s)), "scalar_fits": int(s["scalar_fits"].sum()),
                   "by_arm": s.groupby("arm")["scalar_fits"].agg(["count", "sum"]).to_dict("index"),
                   "by_horizon_arm": {f"h{h}_{a}": int(n) for (h, a), n in s.groupby(["horizon", "arm"]).size().items()},
                   "fit_empty": int(s["fit_empty"].sum()), "preflight_fit_ceiling": pre.get("fit_ceiling")}
print(json.dumps(rep, indent=1, default=str))
