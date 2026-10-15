"""Read-only (r2): exact saved-text identity of the AlphaEarth evaluation cohorts, read as the dashboard reads
predictions (dtype=str, keep_default_na=False), compared with every existing dashboard dataset descriptor; then the
12 family-qualified descriptors actually backed by saved report scores (main all_scored, main persistence_available,
supplement all_scored; no supplement persistence panel is saved, so that cohort is NOT registered or rescored)."""
import hashlib, json, sys
from pathlib import Path
import pandas as pd
run, plan_path = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads(plan_path.read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()
canon = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()
existing = [d for d in plan["datasets"] if d["descriptor"]["context"] == "evaluation"]
TRUTH_DEF = existing[0]["descriptor"]["truth_definition"]
COH_DEF = {d["descriptor"]["cohort"]: d["descriptor"]["cohort_definition"] for d in existing}
report = json.loads((run / "report/report.json").read_text())
FAMILY_SHORT = "AlphaEarth supervised partition"
lead = lambda h: f"{h}-month"
compare, descriptors = [], []
for h in (1, 3, 6, 12):
    p = pd.read_csv(run / f"predict/predictions_h{h:02d}.csv.gz", dtype=str, keep_default_na=False)
    for period, role in (("main", "primary"), ("supplement", "holdout")):
        sub = p[p.period == period]
        for coh, fr in (("all_scored", sub), ("persistence_available", sub[sub.persistence_available == "1"])):
            keys = sorted(f"{int(a)}|{int(t)}" for a, t in zip(fr.admin_code, fr.target_ord))
            kp = sorted(f"{int(a)}|{int(t)}|{ph}" for a, t, ph in zip(fr.admin_code, fr.target_ord, fr.phase_truth))
            lines = sorted(f"{int(a)}|{int(t)}|{ph}|{q}" for a, t, ph, q in zip(fr.admin_code, fr.target_ord, fr.phase_truth, fr.q3))
            ks, kps, ts = sha("\n".join(keys).encode()), sha("\n".join(kp).encode()), sha("\n".join(lines).encode())
            months = sorted(set(fr.target_month))
            span = f"{months[0]}..{months[-1]}"
            plain = f"IPCCH eval | {lead(h)} | {span} | {coh}"
            ex = next((d for d in existing if d["name"] == plain), None)
            row = {"H": h, "period": period, "cohort": coh, "n": len(fr), "span": span, "keys_sha256": ks,
                   "keys_phase_sha256": kps, "truth_text_sha256": ts, "existing_name": plain if ex else None,
                   "existing_keys_equal": ex is not None and ex["descriptor"]["keys_sha256"] == ks,
                   "existing_truth_text_equal": ex is not None and ex["descriptor"]["truth_sha256"] == ts}
            if ex is not None and not row["existing_truth_text_equal"]:  # locate the first differing text line
                compare.append(row)
            else:
                compare.append(row)
            scored = coh == "all_scored" or period == "main"
            if period == "supplement" and coh == "persistence_available":
                row["registered"] = False
                row["why_not"] = "no supplement persistence panel is saved in report.json; not rescored"
                continue
            row["registered"] = True
            desc = {"schema": "ipcch-evaluation-dataset-v2", "context": "evaluation", "lead_months": h, "period": span,
                    "cohort": coh, "cohort_definition": COH_DEF[coh], "num_rows": len(fr), "keys_sha256": ks, "truth_sha256": ts,
                    "truth_columns": ["phase_truth", "q3_truth"], "truth_definition": TRUTH_DEF,
                    "key_definition": "one row per admin_code|target_ord; truth text as saved in the predictions",
                    "target_month_min": months[0], "target_month_max": months[-1], "rows_depend_on": FAMILY_SHORT,
                    "truth_text_source": "predict/predictions_hXX.csv.gz columns phase_truth, q3 (saved text)"}
            full = sha(canon(desc))
            descriptors.append({"name": f"{plain} ({FAMILY_SHORT})", "digest": full[:32], "full_sha256": full, "descriptor": desc,
                                "report_panel": f"horizons.{h}.{'panels' if (period, coh) == ('main', 'all_scored') else ('persistence.panels' if period == 'main' else 'supplement_2026.panels')}"})
# one example of differing text (first H, main, all) for the record
h = 1
p = pd.read_csv(run / "predict/predictions_h01.csv.gz", dtype=str, keep_default_na=False)
print(json.dumps({"comparison": compare, "registered_descriptors": descriptors,
                  "counts": {"compared": len(compare), "registered": len(descriptors),
                             "keys_equal": sum(r["existing_keys_equal"] for r in compare),
                             "truth_text_equal": sum(r["existing_truth_text_equal"] for r in compare)}}, indent=1))
