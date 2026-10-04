"""Saved-artifact Stage 1 diagnostic summary (read-only; no fit). Usage: python stage1_diagnostics.py RUN_DIR

Reuses accept_scenario_stage1, run_stage2.scenario_candidate_row (E3 recount + NA status),
crisis_plan_weights (existing E4 matched-E3 utility) and run_stage2.diagnostics (coverage-matched
canonical partitions). Per candidate it reads the saved role scores (F = fit_diagnostic,
S = scores.validation, C = confirmation, E3 = target) and the original-key role counts from
fold_membership. Candidate distributions are diagnostic only: repeated evaluation keys across
candidates are never pooled as independent observations. Writes only beside this script.
"""
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[5] / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PKG))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import run_stage2 as s2  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402

run = Path(sys.argv[1])
stage = run / "stage1_scenario"
accepted = acc.accept_scenario_stage1(run)
rows = []
for cand, v in sorted(accepted.items()):
    e = v["entry"]
    row = s2.scenario_candidate_row(stage, e)
    roles = pd.read_csv(stage / "roots" / e["root"] / "fold_membership.csv.gz")
    for role, n in roles.drop_duplicates(["area", "target_month", "role"])["role"].value_counts().items():
        row[f"keys_{role}"] = int(n)
    row["dup_role_keys"] = int(roles.duplicated(["area", "target_month", "role"]).sum())
    row["overlap_keys_across_roles"] = int(roles.drop_duplicates(["area", "target_month", "role"])
                                           .duplicated(["area", "target_month"]).sum())
    if row["status"] in ("scored", "e3_undefined"):
        rec = json.loads((stage / "candidates" / cand / "candidate.json").read_text(encoding="utf-8"))
        for tag, block in (("F", rec["fit_diagnostic"]), ("S", rec["scores"]["validation"]),
                           ("C", rec["confirmation"])):
            for side in ("final", "root"):
                c = block[side]["crisis"]
                row[f"{tag}_{side}_f1"] = c["f1"] if c.get("f1_na_reason", "") == "" else np.nan
                row[f"{tag}_{side}_na"] = c.get("f1_na_reason", "")
            row[f"{tag}_n"] = block["n"]
        tp = pd.read_csv(stage / "candidates" / cand / "target_predictions.csv", usecols=["routing"])
        row["e3_rows"] = int(len(tp))
        row["e3_routing"] = json.dumps(tp["routing"].value_counts().to_dict(), sort_keys=True)
        row["accepted_splits"] = int(rec["partition"]["accepted_splits"])
    rows.append(row)
led = pd.DataFrame(rows)
w = s2.crisis_plan_weights(led)
led["weight"] = w["weight"]
out = Path(__file__).with_suffix("")
led.to_csv(f"{out}_ledger.csv", index=False)

summary = {"candidates": int(len(led)), "status_counts": led["status"].value_counts().to_dict(), "cells": [],
           "map_diagnostics": {}}
for (s, h, k), g in led.groupby(["strategy", "horizon", "scenario_k"]):
    sc = g[g["status"] == "scored"]
    cell = {"strategy": s, "horizon": int(h), "scenario_k": int(k), "candidates": int(len(g)),
            "status": g["status"].value_counts().to_dict(),
            "na_reasons": g.loc[g["na_reason"].fillna("") != "", "na_reason"].value_counts().to_dict(),
            "root_only_n_terminal_1": int((g["n_terminal"] == 1).sum()),
            "positive_weight": int((sc["weight"] > 0).sum())}
    for role in ("fitting", "validation", "confirmation", "heldout_target"):
        col = f"keys_{role}"
        cell[f"orig_keys_{role}_median"] = float(g[col].median()) if col in g else None
        cell[f"orig_keys_{role}_range"] = [int(g[col].min()), int(g[col].max())] if col in g else None
    cell["dup_role_keys_total"] = int(g["dup_role_keys"].sum())
    cell["overlap_keys_across_roles_total"] = int(g["overlap_keys_across_roles"].sum())
    gaps = {"E3": sc["crisis_f1"] - sc["crisis_f1_base"]}
    for tag in ("F", "S", "C"):
        gaps[tag] = sc[f"{tag}_final_f1"] - sc[f"{tag}_root_f1"]
        cell[f"{tag}_na_scored_rows"] = int(sc[f"{tag}_final_f1"].isna().sum() + sc[f"{tag}_root_f1"].isna().sum())
    for tag, d in gaps.items():
        d = d.dropna()
        cell[f"{tag}_gap"] = {"n": int(len(d)), "median": float(d.median()) if len(d) else None,
                              "q10": float(d.quantile(.1)) if len(d) else None,
                              "q90": float(d.quantile(.9)) if len(d) else None,
                              "share_gt0": float((d > 0).mean()) if len(d) else None}
    routing = {}
    for r in g["e3_routing"].dropna():
        for key, n in json.loads(r).items():
            routing[key] = routing.get(key, 0) + n
    cell["e3_routing_rows_summed_over_candidates"] = routing
    summary["cells"].append(cell)
paths = {n: stage / "candidates" / n / "correspondence_table.csv" for n in led.loc[led["status"] == "scored", "name"]}
for s in sorted(led["strategy"].unique()):
    for scope, sub in (("all", led[led["strategy"] == s]),
                       *[(f"h{h}_k{k}", led[(led["strategy"] == s) & (led["horizon"] == h) & (led["scenario_k"] == k)])
                         for h in (4, 8) for k in (0, 1, 2)]):
        sc = sub[sub["status"] == "scored"]
        d = s2.diagnostics(sc, paths)
        d["genuine_unique_partitions"] = d["candidates"] - d["same_coverage_duplicate_partitions"]
        summary["map_diagnostics"][f"{s}_{scope}"] = d
Path(f"{out}_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
print(json.dumps({"status_counts": summary["status_counts"], "written": f"{out}_summary.json"}))
