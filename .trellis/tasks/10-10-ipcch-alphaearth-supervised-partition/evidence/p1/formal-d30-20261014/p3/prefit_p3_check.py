"""Pre-predict P3 verification (local, read-only): receipt, executor, config/code/input byte pins, runtime lock, no
existing fit outputs. Exit 0 = all pass."""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster")
REC = Path.home() / "ipcch-alphaearth-runs/p1-formal-d30-logs/receipts/ipcch-alphaearth-formal-P3-release-20261014.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks = {}
ok = lambda n, c, d=None: checks.__setitem__(n, {"pass": bool(c), "detail": d})
ok("receipt_sha256", sha(REC) == "efceca323d3b7a561687e22ed190bafddfbc0fcd84663666f82ce093cec53c52")
r = json.loads(REC.read_text()); run = Path(r["run_directory"]); L0 = Path.home() / "ipcch-alphaearth-runs/p1-formal-d30-logs"
ok("executor", os.environ.get("HERDR_PANE_ID") == r["executor"]["pane"], os.environ.get("HERDR_PANE_ID"))
ok("prior_P2_release", sha(L0 / "receipts/ipcch-alphaearth-formal-P2-release-20261014.json") == r["prior_P2_release_sha256"])
L = Path.home() / "ipcch-alphaearth-runs/p1-formal-d30-logs"
bad = [k for k, v in r["config_sha256"].items() if sha(ROOT / "IPCCHAlphaEarthExperiment/config" / k) != v]
ok("config_sha256", not bad, bad)
bad = [k for k, v in r["code_inventory"].items() if sha(ROOT / k) != v]
ok("code_inventory_46", not bad and len(r["code_inventory"]) == 46, bad)
bad = [k for k, v in r["frozen_input_artifact_sha256"].items() if sha(run / k) != v]
ok("frozen_inputs_73", not bad and len(r["frozen_input_artifact_sha256"]) == 73, bad)
bad = [k for k, v in r["accepted_P2_artifact_sha256"].items() if sha(run / k) != v]
ok("accepted_P2_195", not bad and len(r["accepted_P2_artifact_sha256"]) == 195, bad[:20])
for name, key in (("maps/MAP_FREEZE.json", "map_freeze_sha256"), ("maps/COMPLETE.json", "map_complete_sha256"),
                  ("develop/COMPLETE.json", "develop_complete_sha256"), ("develop/request_ledger.csv.gz", "development_request_ledger_sha256")):
    ok(key, sha(run / name) == r[key])
ok("map_freeze_content", json.loads((run / "maps/MAP_FREEZE.json").read_text()) == r["map_freeze"])
present = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file())
extra = [p for p in present if p not in r["frozen_input_artifact_sha256"] and p not in r["accepted_P2_artifact_sha256"]]
ok("no_files_beyond_inputs_and_P2", not extra, extra[:20])
sys.path.insert(0, str(ROOT / "IPCCHAlphaEarthExperiment"))
from ipcch_alphaearth.preflight import probe_numerical  # noqa: E402
rt = probe_numerical()
ok("runtime_matches_lock", rt.get("matches_lock"), {k: rt[k] for k in rt if k != "packages"} if isinstance(rt, dict) else rt)
ok("interpreter", Path(sys.executable).resolve() == Path(r["numerical_python"]).resolve() or sys.executable == r["numerical_python"], sys.executable)
rep = {"checks": checks, "pass": all(v["pass"] for v in checks.values())}
print(json.dumps(rep, indent=1, default=str))
sys.exit(0 if rep["pass"] else 1)
