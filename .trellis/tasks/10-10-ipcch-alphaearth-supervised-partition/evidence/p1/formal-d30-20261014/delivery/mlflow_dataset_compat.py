"""Read-only: AlphaEarth evaluation cohorts hashed with summary_catalog.eval_descriptor's key/truth formula, compared
with the datasets of the current dashboard plan (by keys+truth digest and by the name that would be generated)."""
import hashlib, json, sys
from pathlib import Path
import pandas as pd
run = Path(sys.argv[1]); plan = json.load(open(sys.argv[2]))
existing = {}
for d in plan["datasets"]:
    de = d["descriptor"]
    if de["context"] == "evaluation":
        existing.setdefault((de["lead_months"], de["cohort"], de["keys_sha256"], de["truth_sha256"]), []).append(d["name"])
by_name = {d["name"]: d["descriptor"] for d in plan["datasets"]}
out = []
for h in (1, 3, 6, 12):
    p = pd.read_csv(run / f"predict/predictions_h{h:02d}.csv.gz", keep_default_na=True)
    for period in ("main", "supplement"):
        sub = p[p.period == period]
        for coh, fr in (("all_scored", sub), ("persistence_available", sub[sub.persistence_available.astype(int) == 1])):
            keys = sorted(f"{int(a)}|{int(t)}" for a, t in zip(fr.admin_code, fr.target_ord))
            ks = hashlib.sha256("\n".join(keys).encode()).hexdigest()
            lines = sorted(f"{int(a)}|{int(t)}|{ph}|{q}" for a, t, ph, q in zip(fr.admin_code, fr.target_ord, fr.phase_truth, fr.q3))
            ts = hashlib.sha256("\n".join(lines).encode()).hexdigest()
            span = f"{fr.target_month.min()}..{fr.target_month.max()}"
            name = f"IPCCH eval | {h}-month | {span} | {coh}"
            match = existing.get((h, coh, ks, ts))
            ex = by_name.get(name)
            out.append({"H": h, "period": period, "cohort": coh, "n": len(fr), "span": span, "keys_sha256": ks, "truth_sha256": ts,
                        "same_keys_truth_as_existing": match, "generated_name_exists": ex is not None,
                        "existing_same_name_keys_equal": None if ex is None else ex["keys_sha256"] == ks,
                        "existing_same_name_truth_equal": None if ex is None else ex["truth_sha256"] == ts})
print(json.dumps(out, indent=1))
