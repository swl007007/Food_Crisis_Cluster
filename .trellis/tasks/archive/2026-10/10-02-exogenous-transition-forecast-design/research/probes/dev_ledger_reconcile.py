"""Read-only reconciliation of the 72 saved development folds against lawful keys/calendars/masks.

SAVED evidence read: fold.json, gate.json, gate_pairs.csv.gz, predictions.csv.gz and the
scenario_globals/*.json fit records (identity fields: origin, fit_label_months, excluded/masked
months, scenario/intensity k, strategy, rows/original_keys, weights, key/label digests).
EXPECTED values are recomputed from the prepared inputs by the package's own lawful definitions
(ReleaseLedger.hidden, Availability.gate_dates/visible, plan.WINDOW): agreement shows the saved
records are consistent with those definitions; it is not independent lineage, and per-key fit
lists are not saved (digests only). No fit, no write to RUN. Usage: python dev_ledger_reconcile.py RUN_DIR
"""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[5] / "FEWSNETGeoXGBExperiment"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import run_experiment as rx  # noqa: E402
from src.experiment import plan, stage3 as s3  # noqa: E402

run = Path(sys.argv[1])
dev = run / "scenario_development"
records = {}
for p in (run / "scenario_globals").rglob("*.json"):
    r = json.loads(p.read_text(encoding="utf-8"))
    records.setdefault(r["booster_sha256"], []).append(r)
ctx = {h: rx.scenario_context(run, h) for h in plan.SCENARIO_HORIZONS}
lab = lambda ms: [s3.ml(int(m)) for m in sorted(ms)]   # noqa: E731
rows, problems = [], []


def check_global(rec, origin, k, strategy, excluded, masked, where):
    bad = []
    if s3.mi(rec["origin_month"]) != origin: bad.append("origin")
    if rec["strategy"] != strategy or int(rec["scenario_k"]) != k or int(rec["intensity_k"]) != k: bad.append("strategy/k")
    if rec["excluded_months"] != excluded: bad.append("excluded_months")
    if rec["masked_months"] != masked: bad.append("masked_months")
    lo, hi = (s3.mi(m) for m in rec["fit_label_months"])
    if lo < origin - plan.WINDOW or hi >= origin: bad.append("fit_label_months outside [O-59, O)")
    if set(rec["fit_label_months"]) & set(masked): bad.append("fit label range endpoint in a masked month")
    mult = 1 if strategy == "A" else 3
    if rec["rows"] != mult * rec["original_keys"]: bad.append("rows vs original keys")
    if (rec["weights_sha256"] is None) != (strategy == "A"): bad.append("weights presence")
    return bad


for fold in rx.scenario_dev_plan():
    s, h, k, t = fold["strategy"], fold["horizon"], fold["scenario_k"], fold["target_month"]
    base = rx._fold_dir(dev, fold)
    av = ctx[h]
    T, O = s3.mi(t), s3.mi(fold["origin_month"])
    where = f"{s}/h{h}/k{k}/{t}"
    rec = json.loads((base / "fold.json").read_text(encoding="utf-8"))
    gate = json.loads((base / "gate.json").read_text(encoding="utf-8"))
    outer_hidden = av.hidden_for(O, k)
    row = {"fold": where, "map_route": rec["map_route"], "routes": rec["routes"]}
    # outer global
    g = gate["global"]
    cands = [r for r in records.get(g["booster_sha256"], []) if r["fit_keys_sha256"] == g["fit_keys_sha256"]]
    outer_bad = [check_global(r, O, k, s, [], lab(outer_hidden), f"{where} outer") for r in cands]
    row["outer_identity_records"] = len(cands)
    row["outer_record"] = any(not b for b in outer_bad)   # identical boosters may carry several identity records
    if not row["outer_record"]:
        problems.append(f"{where} outer: no matching identity record {outer_bad}")
    # internal gate calendar (non-learned map routes have no gate: pooled global only)
    if not (base / "gate_pairs.csv.gz").exists():
        row.update(gate="absent", gate_note=f"no gate_pairs (map_route={rec['map_route']}); gate keys {sorted(gate)}",
                   gate_calendar_ok=rec["map_route"] != "learned_map",
                   internal_records_ok=rec["map_route"] != "learned_map", internal_globals=0, gate_val_months=[])
        if rec["map_route"] == "learned_map":
            problems.append(f"{where}: learned map without gate pairs")
    else:
        pairs = pd.read_csv(base / "gate_pairs.csv.gz")
        want_val = av.gate_dates(O, k)
        got = pairs[["validation_month", "internal_origin", "global_sha256"]].drop_duplicates()
        row["gate_val_months"] = sorted(got["validation_month"].unique().tolist())
        row["gate_calendar_ok"] = row["gate_val_months"] == lab(want_val) and \
            all(s3.mi(v) - h == s3.mi(o) for v, o in zip(got["validation_month"], got["internal_origin"])) and \
            all(s3.mi(v) < O and s3.mi(v) not in outer_hidden for v in got["validation_month"])
        if not row["gate_calendar_ok"]:
            problems.append(f"{where}: gate calendar {row['gate_val_months']} vs lawful {lab(want_val)}")
        ok_int = True
        for _, r in got.drop_duplicates(["internal_origin", "global_sha256"]).iterrows():
            oi = s3.mi(r["internal_origin"])
            cs = records.get(r["global_sha256"], [])
            masked = lab(frozenset(outer_hidden) | av.hidden_for(oi, k))
            bads = [check_global(c, oi, k, s, lab(outer_hidden), masked, "") for c in cs]
            hit = any(not b for b in bads)
            if not hit:
                problems.append(f"{where} internal {r['internal_origin']}: no matching identity record {bads}")
            ok_int &= hit
        row["internal_records_ok"] = bool(ok_int)
        row["internal_globals"] = int(got["global_sha256"].nunique())
    # locals continue the outer global with the single 20-round increment
    loc = gate.get("locals") or {}
    row["locals"] = len(loc)
    row["locals_ok"] = all(v.get("parent_sha256") == g["booster_sha256"] and int(v.get("rounds_added", -1)) == 20
                           for v in loc.values())
    if not row["locals_ok"]:
        problems.append(f"{where}: local parent/rounds")
    # prediction keys and persistence (latest lawful label at O for the area)
    pr = pd.read_csv(base / "predictions.csv.gz")
    keys_ok = (len(pr) == pr["area"].nunique() == 5718 and (pr["target_month"] == t).all()
               and (pr["origin_month"] == fold["origin_month"]).all() and (pr["horizon"] == h).all()
               and (pr["scenario_k"] == k).all())
    vis = av.visible(O, outer_hidden)
    vis = vis[vis["month"] <= O]
    latest = vis.sort_values("month").groupby("area")["month"].last()
    src = pr.set_index("area")["persistence_source_month"]
    exp = latest.reindex(src.index)
    both = src.notna() & exp.notna()
    row["persistence_rows"] = int(src.notna().sum())
    row["persistence_latest_lawful_mismatch"] = int((src[both] != exp[both]).sum() + (src.notna() != exp.notna()).sum())
    row["persistence_in_hidden"] = int(src.dropna().astype(int).isin(list(outer_hidden)).sum())
    age = pr.set_index("area")["persistence_age"]
    row["persistence_age_mismatch"] = int(((O - src[src.notna()]) != age[src.notna()]).sum())   # age at the origin
    row["keys_ok"] = bool(keys_ok)
    if not keys_ok or row["persistence_latest_lawful_mismatch"] or row["persistence_in_hidden"] or row["persistence_age_mismatch"]:
        problems.append(f"{where}: prediction keys/persistence")
    rows.append(row)
out = {"folds": len(rows), "problems": problems, "rows": rows,
       "global_records_indexed": sum(len(v) for v in records.values())}
HERE.with_name("dev_ledger_reconcile_summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps({"folds": len(rows), "problems": len(problems), "first_problems": problems[:10],
                  "outer_ok": sum(r["outer_record"] for r in rows), "gate_calendar_ok": sum(r["gate_calendar_ok"] for r in rows),
                  "internal_ok": sum(r["internal_records_ok"] for r in rows), "locals_ok": sum(r["locals_ok"] for r in rows),
                  "keys_ok": sum(r["keys_ok"] for r in rows),
                  "internal_globals_total": sum(r["internal_globals"] for r in rows),
                  "locals_total": sum(r["locals"] for r in rows)}, indent=1))
