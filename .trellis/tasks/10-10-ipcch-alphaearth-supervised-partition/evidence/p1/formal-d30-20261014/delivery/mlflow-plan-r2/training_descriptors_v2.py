"""Read-only (r2): 8 training-pool descriptors. B0 = saved prepared X296 + keys (hashes from the prepared manifest,
re-verified on disk). B1/B2/G = honest COMPOSITE pool: the same X296 + keys + the frozen descriptor table joined on
admin_code with the ordered 424 schema; no X424 file exists and none is created."""
import hashlib, json, sys
from pathlib import Path
import pandas as pd
run = Path(sys.argv[1]); TEXT = sys.argv[2]
sha = lambda b: hashlib.sha256(b).hexdigest(); fsha = lambda p: sha(Path(p).read_bytes())
canon = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()
pm = json.loads((run / "prepared/prepared-manifest.json").read_text())["artifacts_sha256"]
order296 = pd.read_csv(run / "prepared/feature_order_296.csv.gz")["name"].tolist()
desc_done = json.loads((run / "descriptors/COMPLETE.json").read_text())
table = run / "descriptors/descriptor_table.csv.gz"
names128 = [c for c in pd.read_csv(table, nrows=0).columns if c != "admin_code"]
out, problems = [], []
for h in (1, 3, 6, 12):
    x, k = f"X296_h{h:02d}.npy", f"keys_h{h:02d}.csv.gz"
    for f in (x, k):
        if pm.get(f) != fsha(run / "prepared" / f):
            problems.append(f"{f}: manifest hash != disk")
    base = {"schema": "ipcch-training-pool-v1", "context": "training", "lead_months": h, "first_target_month": "2014-01",
            "definition": TEXT, "x_sha256": pm[x], "keys_sha256": pm[k], "x_path": f"prepared/{x}", "keys_path": f"prepared/{k}",
            "prepared_manifest_sha256": fsha(run / "prepared/prepared-manifest.json")}
    b0 = {**base, "features": "compact296", "ordered_features_sha256": sha("\n".join(order296).encode()), "n_features": 296}
    b1 = {**base, "features": "compact296+alphaearth128 (composite)", "composite": True,
          "descriptor_table_sha256": fsha(table), "descriptor_table_path": "descriptors/descriptor_table.csv.gz",
          "descriptors_complete_sha256": fsha(run / "descriptors/COMPLETE.json"),
          "join": "row admin_code -> descriptor_table.admin_code (exact int64); the 128 descriptor columns appended after the 296 in this order",
          "ordered_features_sha256": sha("\n".join(order296 + names128).encode()), "n_features": 424,
          "materialized_x424": "none (assembled in memory by runinputs.load_run_inputs; no file is created for tracking)"}
    for d in (b0, b1):
        full = sha(canon(d))
        out.append({"name": f"IPCCH training pool | {d['features']} | {h}-month", "digest": full[:32], "full_sha256": full,
                    "arms": ["B0 pooled"] if d is b0 else ["B1 pooled_plus_alphaearth", "B2 shared_calibration", "G regional_calibration"],
                    "descriptor": d})
print(json.dumps({"training_descriptors": out, "n": len(out), "problems": problems,
                  "descriptor_column_count": len(names128)}, indent=1))
