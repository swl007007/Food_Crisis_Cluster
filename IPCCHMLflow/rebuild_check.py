"""Old-store vs new-store value check for the readable-naming rebuild (read-only on both DBs).

  python rebuild_check.py --old OLD/mlflow.db --new NEW/mlflow.db --out REPORT.json

1. Every metric of every old detailed record (experiment ``IPCCH``) has the identical value under
   its readable name (naming.metric_name) in the new detailed record with the same source key,
   and the new record has no other metric.
2. Every value of every old Summary row (``IPCCH Summary``) has the identical value in the new
   dashboard row of the same source key.
3. Every other dashboard value is either a seed mean (equal to the mean of the seed rows) or a
   contrast equal to its source value in the new detailed record.
4. Every dataset name in the new store has exactly one digest.
Counts per family / arm / lead are reported for both stores.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import naming  # noqa: E402
from summary_catalog import COHORTS, contrast_values  # noqa: E402

OLD_DETAIL, OLD_SUMMARY = "IPCCH", "IPCCH Summary"
NEW_DETAIL, NEW_DASHBOARD = "IPCCH - detailed runs", "IPCCH - dashboard"
SUMMARY_LEAF = {"q3_r2_projected": "share_phase3plus_r2", "n": "n_rows"}


def load(db: Path) -> dict:
    """{experiment name: {run_id: {"name", "tags", "metrics"}}} for active runs."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        exps = dict(con.execute("select experiment_id, name from experiments"))
        out = defaultdict(dict)
        for rid, eid, name in con.execute("select run_uuid, experiment_id, name from runs where lifecycle_stage='active'"):
            out[exps[eid]][rid] = {"name": name, "tags": {}, "metrics": {}}
        idx = {rid: e for e, runs in out.items() for rid in runs}
        for k, v, rid in con.execute("select key, value, run_uuid from tags"):
            if rid in idx:
                out[idx[rid]][rid]["tags"][k] = v
        for k, v, rid in con.execute("select key, value, run_uuid from latest_metrics"):
            if rid in idx:
                out[idx[rid]][rid]["metrics"][k] = v
        datasets = defaultdict(set)
        for name, digest in con.execute("select name, digest from datasets"):
            datasets[name].add(digest)
        return {"experiments": dict(out), "datasets": {k: sorted(v) for k, v in datasets.items()}}
    finally:
        con.close()


def compare(old: dict, new: dict) -> dict:
    rep = {"detailed": Counter(), "summary": Counter(), "dashboard_extra": Counter(), "problems": []}
    bad = rep["problems"].append
    new_detail = {r["tags"]["_prov.source_key"]: r for r in new["experiments"][NEW_DETAIL].values()}
    for r in old["experiments"][OLD_DETAIL].values():
        key, family = r["tags"]["source_key"], r["tags"]["family"]
        n = new_detail.get(key)
        if n is None:
            bad(f"detailed {key}: missing in new store")
            continue
        names = naming.check_one_to_one(family, r["metrics"])
        for k, v in r["metrics"].items():
            if n["metrics"].get(names[k]) != v:
                bad(f"detailed {key}: {k} -> {names[k]} {n['metrics'].get(names[k])!r} != {v!r}")
            else:
                rep["detailed"]["values_equal"] += 1
        extra = set(n["metrics"]) - set(names.values())
        if extra:
            bad(f"detailed {key}: new metrics without an old source {sorted(extra)[:3]}")
        rep["detailed"]["records"] += 1
    if len(new_detail) != rep["detailed"]["records"]:
        bad(f"new detailed store has {len(new_detail)} records, old {rep['detailed']['records']}")

    dash = list(new["experiments"][NEW_DASHBOARD].values())
    by_source = {r["tags"]["_prov.original_source_key"]: r for r in dash if r["tags"]["seed"] != "mean"}
    explained = defaultdict(set)
    for r in old["experiments"][OLD_SUMMARY].values():
        key, ns = r["tags"]["projection_key"].split("#")
        family = r["tags"]["family"]
        d = by_source.get(key)
        if d is None:
            bad(f"summary {key}#{ns}: no dashboard row")
            continue
        new_ns = naming.namespace(family, ns)
        for k, v in r["metrics"].items():
            nk = f"{new_ns}.{SUMMARY_LEAF.get(k, k)}"
            if d["metrics"].get(nk) != v:
                bad(f"summary {key}#{ns}: {k} -> {nk} {d['metrics'].get(nk)!r} != {v!r}")
            else:
                rep["summary"]["values_equal"] += 1
            explained[key].add(nk)
        rep["summary"]["rows"] += 1

    seeds = defaultdict(list)
    for d in dash:
        if d["tags"]["seed"] not in ("mean", "none") and naming.multi_seed(d["tags"]["_prov.original_source_key"].split("/")[0]):
            seeds[(d["tags"]["family"], d["tags"]["arm"], d["tags"]["lead_months"])].append(d)
    for d in dash:
        t = d["tags"]
        if t["seed"] == "mean":
            group = seeds[(t["family"], t["arm"], t["lead_months"])]
            for k, v in d["metrics"].items():
                vals = [g["metrics"].get(k) for g in group]
                if len(group) != 3 or None in vals or abs(sum(vals) / 3 - v) > 1e-12:
                    bad(f"dashboard {t['_prov.projection_key']}: seed mean {k} {v!r} vs {vals}")
                else:
                    rep["dashboard_extra"]["seed_mean_values"] += 1
            continue
        src = new_detail[t["_prov.original_source_key"]]
        for k, v in d["metrics"].items():
            if k in explained[t["_prov.original_source_key"]]:
                continue
            ns, _, contrast = k.partition(".binary.f1.minus_")
            other, _, suffix = contrast.partition(".")
            role, coh = ns.split(".", 1)
            srcs = contrast_values({"metrics": src["metrics"]}, role, coh, t["arm"], other) if contrast else {}
            s = srcs.get(f".{suffix}" if suffix else "")
            if s is None and coh in COHORTS and not contrast:
                s = k                                      # a dashboard panel value absent from the old Summary
            if s is None or src["metrics"].get(s) != v:
                bad(f"dashboard {t['_prov.projection_key']}: {k} has no matching source value ({s})")
            else:
                rep["dashboard_extra"]["contrast_values" if contrast else "panel_values_new_in_dashboard"] += 1
    rep["dashboard_rows"] = len(dash)

    multi = {k: v for k, v in new["datasets"].items() if len(v) != 1}
    if multi:
        bad(f"dataset names with several digests: {sorted(multi)[:3]}")
    rep["datasets"] = {"names": len(new["datasets"]), "names_with_several_digests": len(multi)}

    def counts(runs, fam_key, arm_key, lead_key):
        return Counter(f"{r['tags'].get(fam_key)}|{r['tags'].get(arm_key)}|{r['tags'].get(lead_key)}"
                       for r in runs if r["tags"].get(lead_key))
    rep["counts"] = {
        "old_detailed_by_family_arm_lead": len(counts(old["experiments"][OLD_DETAIL].values(), "family", "arm", "horizon")),
        "new_detailed_by_family_arm_lead": len(counts(new["experiments"][NEW_DETAIL].values(), "family", "arm", "lead_months")),
        "old_detailed_records": len(old["experiments"][OLD_DETAIL]),
        "new_detailed_records": len(new["experiments"][NEW_DETAIL]),
        "old_summary_rows": len(old["experiments"][OLD_SUMMARY]), "new_dashboard_rows": len(dash)}
    rep["passed"] = not rep["problems"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = compare(load(Path(a.old)), load(Path(a.new)))
    rep["problems_total"] = len(rep["problems"])
    rep["problems"] = rep["problems"][:50]
    text = json.dumps(rep, indent=1, sort_keys=True, default=dict)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
