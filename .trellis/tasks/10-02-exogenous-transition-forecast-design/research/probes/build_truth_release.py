"""Evaluator-only October-2025 truth release candidate (no fit; never feeds fitting/selection).

Source: the raw FEWS NET export 2025_2026_FEWSNET.csv, rows with scenario CS and reporting_date 2025-10
(NOT the derived/zero-filled FEWS_2025.csv). Crosswalk: EXACT full-name equality of
``geographic_unit_full_name`` with FEWSNET.csv ``admin_name`` over all history. Admission requires:
  1. the name maps to exactly one admin code (ambiguous names, e.g. Kenya 2995/2996, are excluded);
  2. that code receives exactly one October row (one-to-one);
  3. the raw country_code equals the code's country in the prepared observations;
  4. the canonical shapefile DBF admin_name for the code equals the matched name;
  5. a genuine IPC phase 1..5 (missing / Not Projected / Not Available excluded).
No fuzzy or spatial repair, no zero fill, no June label. Name equality is key evidence, not certified
geometric identity. class_code = min(phase, 4) - 1 (the four-class axis of the prepared labels).
Every raw row is kept in the crosswalk with its status. Writes a new external release directory
(refuses overwrite) with release.json approved=false pending coordinator approval.
Usage: python build_truth_release.py RUN_DIR OUT_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
if out_dir.exists():
    raise SystemExit(f"{out_dir} exists; refusing to overwrite")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()   # noqa: E731
sources = json.loads((run / "prepared/manifests/sources.json").read_text(encoding="utf-8"))["sources"]
FEWS = Path(sources["fewsnet"]["path"])
RAW = FEWS.parent / "2025_2026_FEWSNET.csv"
SHP = Path(sources["shapefile"]["path"])
if sha(FEWS) != sources["fewsnet"]["sha256"]: raise SystemExit("FEWSNET.csv differs from its pin")
if sha(SHP) != sources["shapefile"]["sha256"]: raise SystemExit("shapefile differs from its pin")
raw_sha = sha(RAW)
raw = pd.read_csv(RAW, encoding="utf-8-sig", dtype=str)
o = raw[(raw.scenario == "CS") & raw.reporting_date.str.startswith("2025-10")].copy()
o = o[["id", "fnid", "country", "country_code", "geographic_unit_full_name", "value", "description",
       "is_allowing_for_assistance", "status", "projection_start", "projection_end"]].reset_index(drop=True)
hist = pd.read_csv(FEWS, usecols=["country", "admin_code", "admin_name", "year_month"], dtype={"admin_code": "Int64"})
hist = hist.dropna(subset=["admin_code", "admin_name"])
codes_of = hist.groupby("admin_name").admin_code.agg(lambda s: sorted(set(int(x) for x in s)))
last_seen = hist.groupby("admin_name").year_month.max()
import pyogrio   # noqa: E402  (pinned Windows stack; reads attributes only)
dbf = pyogrio.read_dataframe(SHP, read_geometry=False, columns=["admin_code", "admin_name"])
dbf_name = dict(zip(dbf.admin_code.astype(int), dbf.admin_name))
obs = pd.read_csv(run / "prepared/ledgers/observations.csv", usecols=["area", "country"])
obs_country = obs.drop_duplicates("area").set_index("area").country.to_dict()
universe = set(dbf_name)

o["name_codes"] = o.geographic_unit_full_name.map(lambda n: codes_of.get(n, []))
o["area"] = o.name_codes.map(lambda c: c[0] if len(c) == 1 else np.nan).astype("Int64")
o["fewsnet_name_last_seen"] = o.geographic_unit_full_name.map(last_seen)
o["dbf_admin_name"] = o.area.map(lambda a: dbf_name.get(int(a)) if pd.notna(a) else None)
o["obs_country"] = o.area.map(lambda a: obs_country.get(int(a)) if pd.notna(a) else None)
o["phase"] = pd.to_numeric(o.value, errors="coerce")
dup_codes = set(o.area.dropna()[o.area.dropna().duplicated()].astype(int))


def status(r):
    if len(r.name_codes) == 0: return "excluded_unmatched_name"
    if len(r.name_codes) > 1: return "excluded_ambiguous_name_multiple_codes:" + "/".join(map(str, r.name_codes))
    a = int(r.area)
    if a not in universe: return "excluded_code_outside_universe"
    if a in dup_codes: return "excluded_code_receives_multiple_rows"
    if r.obs_country is not None and r.obs_country != r.country_code: return "excluded_country_mismatch"
    if r.dbf_admin_name != r.geographic_unit_full_name: return "excluded_dbf_name_mismatch"
    if not (r.phase in (1, 2, 3, 4, 5)): return f"excluded_no_genuine_phase:{r.status}"
    return "admitted"


o["match_status"] = o.apply(status, axis=1)
o["class_code"] = np.where(o.match_status == "admitted", np.minimum(o.phase.fillna(0), 4) - 1, np.nan)
o["name_codes"] = o.name_codes.map(lambda c: "/".join(map(str, c)))
out_dir.mkdir(parents=True)
cross = out_dir / "crosswalk_oct2025.csv"
o.to_csv(cross, index=False, lineterminator="\n")
adm = o[o.match_status == "admitted"]
truth = pd.DataFrame({"area": adm.area.astype(int), "target_month": "2025-10", "class_code": adm.class_code.astype(int),
                      "raw_phase": adm.phase.astype(int), "is_allowing_for_assistance": adm.is_allowing_for_assistance,
                      "fnid": adm.fnid, "source_id": adm.id}).sort_values("area")
assert not truth.duplicated(["area", "target_month"]).any() and truth.class_code.isin([0, 1, 2, 3]).all()
truth_path = out_dir / "truth_oct2025.csv"
truth.to_csv(truth_path, index=False, lineterminator="\n")
actual_path = run / "scenario_actual/actual.json"
release = {"approved": False, "approved_by": None, "crosswalk": cross.name, "truth_file": truth_path.name,
           "truth_sha256": sha(truth_path), "frozen_actual": sha(actual_path),
           "status": "CANDIDATE - pending coordinator approval; scen-evaluate refuses while approved is false",
           "sources": {"raw_2025_cs": {"path": str(RAW), "sha256": raw_sha},
                       "fewsnet_history_names": {"path": str(FEWS), "sha256": sources["fewsnet"]["sha256"]},
                       "shapefile_dbf": {"path": str(SHP), "sha256": sources["shapefile"]["sha256"]}},
           "crosswalk_sha256": sha(cross),
           "rules": __doc__.split("Usage:")[0].strip(),
           "june_2025": "no local genuine June-2025 CS source; June targets stay forecast/coverage-only (no truth rows)",
           "expert": "no documented same-horizon expert table; keys keep no_documented_expert_table"}
with open(out_dir / "release.json", "x", encoding="utf-8") as f:
    json.dump(release, f, indent=1, ensure_ascii=False)
pred_areas = set(range(5718))
summary = {"raw_oct2025_cs_rows": len(o), "raw_countries": int(o.country.nunique()),
           "status_counts": o.match_status.value_counts().to_dict(),
           "admitted_keys": len(truth), "admitted_in_prediction_universe": int(truth.area.isin(pred_areas).sum()),
           "prediction_areas": len(pred_areas),
           "class_counts": truth.class_code.value_counts().sort_index().to_dict(),
           "raw_phase_counts": truth.raw_phase.value_counts().sort_index().to_dict(),
           "admitted_allowing_for_assistance": int((truth.is_allowing_for_assistance == "True").sum()),
           "by_country": {c: {k: int(v) for k, v in row.items() if v} for c, row in
                          pd.crosstab(o.country, o.match_status.str.split(":").str[0]).iterrows()},
           "files": {"crosswalk": sha(cross), "truth": sha(truth_path), "release": sha(out_dir / "release.json")},
           "frozen_actual_sha256": release["frozen_actual"]}
(out_dir / "release_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "by_country"}, indent=1, ensure_ascii=False, default=str))
