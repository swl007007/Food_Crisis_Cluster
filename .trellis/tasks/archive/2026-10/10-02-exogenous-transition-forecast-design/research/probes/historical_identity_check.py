"""Post-run identity and inventory check of scen-historical / scen-report (read-only; no fit).

Explicit expectations (independent of the saved acceptors):
- the fold inventory equals the historical.json calendar x k = 0/1/2 for each released H, with
  no missing or extra fold directory, and the calendar equals the frozen design calendar
  (H4 10 targets 2021-10..2024-10, H8 9 targets 2022-02..2024-10);
- every fold.json: phase scenario_historical, strategy/map_id/route = frozen.json recipe for its
  H, prepared_outputs_sha256 = sha256(prepared/manifests/outputs.json), the availability-input
  digest equal to the development folds of the same H, code = the selection's code identity,
  runtime = the selection's runtime, status fitted, 5,718 rows, every recorded output present
  with its hash;
- historical.json binds sha256(frozen.json); report.json binds sha256(historical.json);
  frozen.json binds sha256(selection.json); every recorded report output present with its hash.
Then the saved acceptors are run as well (acceptance.accept_fold / accept_record, _accept_frozen).
Usage: python historical_identity_check.py RUN_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[5] / "FEWSNETGeoXGBExperiment"))

run = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()   # noqa: E731
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))   # noqa: E731
problems = []
sel_path = run / "scenario_development" / "selection.json"
frozen_path = run / "scenario_final" / "frozen.json"
hist_path = run / "scenario_historical" / "historical.json"
rep_path = run / "scenario_report" / "report.json"
selection, frozen, hist, rep = load(sel_path), load(frozen_path), load(hist_path), load(rep_path)
prepared_sha = sha(run / "prepared" / "manifests" / "outputs.json")
DESIGN = {"4": [f"{y}-{m:02d}" for y in range(2021, 2025) for m in (2, 6, 10) if f"{y}-{m:02d}" >= "2021-10"],
          "8": [f"{y}-{m:02d}" for y in range(2022, 2025) for m in (2, 6, 10)]}

if frozen["selection_sha256"] != sha(sel_path): problems.append("frozen.json does not bind selection.json")
if hist["frozen_sha256"] != sha(frozen_path): problems.append("historical.json does not bind frozen.json")
if rep["historical_sha256"] != sha(hist_path): problems.append("report.json does not bind historical.json")
for name, rec, base in (("historical", hist, hist_path.parent), ("report", rep, rep_path.parent)):
    if rec["code"] != selection["code"] or rec["runtime"] != selection["runtime"]:
        problems.append(f"{name}.json code/runtime differ from the selection")
    for rel, h in rec["outputs"].items():
        if not (base / rel).is_file() or sha(base / rel) != h: problems.append(f"{name} output {rel} missing/changed")
dev_digest = {}
for f in (run / "scenario_development").rglob("fold.json"):
    d = load(f)
    dev_digest.setdefault(str(d["horizon"]), set()).add(d["availability_inputs_sha256"])
expected, summary = set(), {"calendar": {}, "folds": 0}
for h, cal in hist["calendar"].items():
    recipe = frozen["recipe"][h]
    if not recipe["released"] or not cal["released"]:
        problems.append(f"H{h} unexpectedly not released")
        continue
    if cal["targets"] != DESIGN[h]: problems.append(f"H{h} calendar {cal['targets']} != design {DESIGN[h]}")
    if cal["strategy"] != recipe["strategy"]: problems.append(f"H{h} calendar strategy differs from frozen recipe")
    summary["calendar"][h] = {"targets": len(cal["targets"]), "excluded": [e["target_month"] for e in cal["excluded"]],
                              "truth_available": sum(v["truth_available"] for v in cal["coverage"].values())}
    for k in (0, 1, 2):
        for t in cal["targets"]:
            expected.add((recipe["strategy"], h, k, t))
found = {}
for f in (run / "scenario_historical").rglob("fold.json"):
    parts = f.parent.relative_to(run / "scenario_historical").parts   # strategy/hH/kK/T
    found[(parts[0], parts[1][1:], int(parts[2][1:]), parts[3])] = f
summary["folds"] = len(found)
if set(found) != expected:
    problems.append(f"fold inventory: missing {sorted(expected - set(found))[:5]}, extra {sorted(set(found) - expected)[:5]}")
for (s, h, k, t), f in sorted(found.items()):
    d, recipe = load(f), frozen["recipe"][h]
    want = {"phase": "scenario_historical", "strategy": recipe["strategy"], "horizon": int(h), "scenario_k": k,
            "target_month": t, "map_id": recipe["map_id"], "prepared_outputs_sha256": prepared_sha,
            "status": "fitted", "rows": 5718, "code": selection["code"], "runtime": selection["runtime"]}
    bad = [x for x, v in want.items() if d.get(x) != v]
    if dev_digest.get(h) != {d["availability_inputs_sha256"]}: bad.append("availability_inputs_sha256")
    for rel, hsh in d["outputs"].items():
        if not (f.parent / rel).is_file() or sha(f.parent / rel) != hsh: bad.append(f"output {rel}")
    if bad: problems.append(f"{s}/h{h}/k{k}/{t}: {bad}")
# the saved acceptors as well
from scripts import run_experiment as rx  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
acceptor = {"_accept_frozen": "ok", "historical.json": "ok", "report.json": "ok", "accept_fold": 0}
try:
    rx._accept_frozen(run)
except Exception as e:   # noqa: BLE001
    acceptor["_accept_frozen"] = repr(e); problems.append("acceptor _accept_frozen")
for name, base, rec in (("historical.json", hist_path.parent, "historical.json"), ("report.json", rep_path.parent, "report.json")):
    try:
        acc.accept_record(base, rec)
    except Exception as e:   # noqa: BLE001
        acceptor[name] = repr(e); problems.append(f"acceptor {name}")
for (s, h, k, t), f in sorted(found.items()):
    d = load(f)
    identity = {x: d[x] for x in ("strategy", "horizon", "scenario_k", "target_month", "phase", "map_id",
                                  "prepared_outputs_sha256", "availability_inputs_sha256")}
    try:
        acc.accept_fold(f.parent, identity)
        acceptor["accept_fold"] += 1
    except Exception as e:   # noqa: BLE001
        problems.append(f"acceptor accept_fold {s}/h{h}/k{k}/{t}: {e!r}")
out = {"selection_sha256": sha(sel_path), "frozen_sha256": sha(frozen_path), "historical_sha256": sha(hist_path),
       "report_sha256": sha(rep_path), "prepared_outputs_sha256": prepared_sha, "summary": summary,
       "acceptors": acceptor, "expected_folds": len(expected), "problems": problems}
HERE.with_name("historical_identity_check_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
if problems:
    raise SystemExit(1)
