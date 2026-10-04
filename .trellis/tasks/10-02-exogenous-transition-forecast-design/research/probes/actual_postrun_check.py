"""Post-run check of scen-actual (read-only; no truth/expert values, no fit, no RUN write).

Explicit expectations plus the saved acceptors:
- actual.json: accepted record; binds sha256(frozen.json); code/runtime = the selection's; exactly
  the four ACTUAL_CASES, all released, each fold sha equal to its fold.json;
- each fold.json: phase scenario_actual, frozen strategy/map for its H, scenario_k 0, gate_k = the
  table's per-country counts at its origin (22 countries), prepared sha, availability-table sha and
  extension-manifest sha equal to the launched inputs, truth "not loaded", 5,718 rows, outputs hashed;
- predictions: 5,718 unique areas at (T, O, H); y_true_code entirely NaN (truth not loaded);
  persistence source month/class/age = the latest lawful label <= O (release <= O, k = 0 outer),
  from the prepared observations with explicit filters; per-country ages reported (SD older);
- gate.json outer global origin = O.
Usage: python actual_postrun_check.py RUN_DIR TABLE_CSV MANIFEST_JSON
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[5] / "FEWSNETGeoXGBExperiment"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import run_experiment as rx  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402

run, table_path, manifest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()   # noqa: E731
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))   # noqa: E731
mi = lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1   # noqa: E731
lab = lambda x: f"{x // 12}-{x % 12 + 1:02d}"   # noqa: E731
problems, out = [], {}
base = run / "scenario_actual"
actual, frozen = load(base / "actual.json"), load(run / "scenario_final/frozen.json")
selection = load(run / "scenario_development/selection.json")
table = pd.read_csv(table_path, dtype={"origin_month": str})
out["inputs"] = {"table_sha256": sha(table_path), "manifest_sha256": sha(manifest)}
out["actual_json_sha256"] = sha(base / "actual.json")
try:
    acc.accept_record(base, "actual.json")
    rx._accept_frozen(run)
    out["acceptors"] = "ok"
except Exception as e:   # noqa: BLE001
    out["acceptors"] = repr(e); problems.append("acceptor")
if actual["frozen_sha256"] != sha(run / "scenario_final/frozen.json"): problems.append("actual.json does not bind frozen.json")
if actual["code"] != selection["code"] or actual["runtime"] != selection["runtime"]: problems.append("actual code/runtime")
want_cases = {f"h{h}_{t}" for t, h in rx.ACTUAL_CASES}
if set(actual["cases"]) != want_cases: problems.append(f"cases {sorted(actual['cases'])}")
obs = pd.read_csv(run / "prepared/ledgers/observations.csv", usecols=["area", "month", "country", "class_code"])
led = pd.read_csv(run / "prepared/manifests/release_ledger.csv", dtype=str)
led = led[led["product"] == "CS"].assign(month=lambda d: d.reference_month.map(mi), release=lambda d: d.release_date.map(mi))
obs = obs.merge(led[["country", "month", "release"]], on=["country", "month"], how="left")
prepared_sha = sha(run / "prepared/manifests/outputs.json")
out["cases"], out["output_hashes"] = {}, {}
for case, info in sorted(actual["cases"].items()):
    h, t = int(case.split("_")[0][1:]), case.split("_", 1)[1]
    O = mi(t) - h
    fdir = base / f"h{h}" / t
    rec, recipe = load(fdir / "fold.json"), frozen["recipe"][str(h)]
    bad = []
    if not info.get("released") or info.get("fold_sha256") != sha(fdir / "fold.json"): bad.append("actual.json fold sha")
    gate_k = table[(table.origin_month == lab(O)) & (table["product"] == "CS")].set_index("country").missed_cycles.astype(int).to_dict()
    want = {"phase": "scenario_actual", "strategy": recipe["strategy"], "horizon": h, "scenario_k": 0, "target_month": t,
            "map_id": recipe["map_id"], "prepared_outputs_sha256": prepared_sha, "gate_k": gate_k, "rows": 5718,
            "availability_table_sha256": out["inputs"]["table_sha256"],
            "extension_manifest_sha256": out["inputs"]["manifest_sha256"],
            "code": selection["code"], "runtime": selection["runtime"]}
    bad += [k for k, v in want.items() if rec.get(k) != v]
    if "not loaded" not in str(rec.get("truth", "")): bad.append("truth note")
    for rel, hs in rec["outputs"].items():
        if sha(fdir / rel) != hs: bad.append(f"output {rel}")
        out["output_hashes"][f"h{h}/{t}/{rel}"] = hs
    out["output_hashes"][f"h{h}/{t}/fold.json"] = sha(fdir / "fold.json")
    try:
        acc.accept_record(fdir, "fold.json", ["predictions.csv.gz", "gate.json"])
    except Exception as e:   # noqa: BLE001
        bad.append(f"accept_record {e!r}")
    gate = load(fdir / "gate.json")
    if gate["global"]["origin_month"] != lab(O): bad.append("outer global origin")
    pr = pd.read_csv(fdir / "predictions.csv.gz", usecols=["area", "target_month", "origin_month", "horizon", "y_true_code",
                                                           "country", "persistence_class_code", "persistence_source_month",
                                                           "persistence_age"])
    if not (len(pr) == pr.area.nunique() == 5718 and (pr.target_month == t).all() and (pr.origin_month == lab(O)).all()
            and (pr.horizon == h).all()): bad.append("prediction keys")
    truth_nan = bool(pr.y_true_code.isna().all())
    if not truth_nan: bad.append("truth present in predictions")
    vis = obs[(obs.month <= O) & (obs.release <= O)]
    last = vis.sort_values("month").groupby("area").tail(1).set_index("area")
    for name, want_v, got in (("source", pr.area.map(last.month), pr.persistence_source_month),
                              ("class", pr.area.map(last.class_code), pr.persistence_class_code),
                              ("age", O - pr.area.map(last.month), pr.persistence_age)):
        if not np.array_equal(want_v.to_numpy(dtype=float), got.to_numpy(dtype=float), equal_nan=True):
            bad.append(f"persistence {name}")
    ages = pr.dropna(subset=["persistence_age"]).groupby("country").persistence_age.agg(lambda s: sorted(set(int(x) for x in s)))
    out["cases"][case] = {"origin": lab(O), "gate_k_values": sorted(set(gate_k.values())), "countries": len(gate_k),
                          "truth_all_nan": truth_nan, "routes": rec.get("routes"),
                          "persistence_rows": int(pr.persistence_source_month.notna().sum()),
                          "ages_by_country": {c: v for c, v in ages.items()}, "problems": bad}
    if bad: problems.append(f"{case}: {bad}")
out["problems"] = problems
HERE.with_name("actual_postrun_check_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: (v if k != "cases" else {c: {x: y for x, y in d.items() if x != "ages_by_country"} for c, d in v.items()})
                  for k, v in out.items() if k != "output_hashes"}, indent=1))
if problems:
    raise SystemExit(1)
