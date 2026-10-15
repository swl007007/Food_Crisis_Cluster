"""Pre-fit P2 verification (local, read-only): receipt, executor, config/code/input byte pins, runtime lock, no
existing fit outputs. Exit 0 = all pass."""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster")
REC = Path.home() / "ipcch-alphaearth-runs/p1-formal-d30-logs/receipts/ipcch-alphaearth-formal-P2-release-20261014.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks = {}
ok = lambda n, c, d=None: checks.__setitem__(n, {"pass": bool(c), "detail": d})
ok("receipt_sha256", sha(REC) == "ae558aab981164bb4d75242359b99adea83a713166ad501bc581d934f011a2c6")
r = json.loads(REC.read_text()); run = Path(r["run_directory"])
ok("executor", os.environ.get("HERDR_PANE_ID") == r["executor"]["pane"], os.environ.get("HERDR_PANE_ID"))
ok("extraction_acceptance", sha(Path.home() / "ipcch-alphaearth-runs/p1-c-full-20261010/receipt" / Path(r["extraction_acceptance_receipt"]).name) == r["extraction_acceptance_sha256"])
L = Path.home() / "ipcch-alphaearth-runs/p1-formal-d30-logs"
ok("freeze_report", sha(L / "freeze_report.json") == r["input_freeze_report_sha256"])
ok("precision_check", sha(L / "precision_check.json") == r["precision_check_sha256"])
ok("stage_log", sha(L / "stages.log") == r["stage_log_sha256"])
bad = [k for k, v in r["config_sha256"].items() if sha(ROOT / "IPCCHAlphaEarthExperiment/config" / k) != v]
ok("config_sha256", not bad, bad)
bad = [k for k, v in r["code_inventory"].items() if sha(ROOT / k) != v]
ok("code_inventory_46", not bad and len(r["code_inventory"]) == 46, bad)
bad = [k for k, v in r["frozen_input_artifact_sha256"].items() if sha(run / k) != v]
ok("frozen_inputs_73", not bad and len(r["frozen_input_artifact_sha256"]) == 73, bad)
present = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file())
extra = [p for p in present if p not in r["frozen_input_artifact_sha256"]]
ok("no_existing_fit_outputs", not extra, extra[:20])
sys.path.insert(0, str(ROOT / "IPCCHAlphaEarthExperiment"))
from ipcch_alphaearth.preflight import probe_numerical  # noqa: E402
rt = probe_numerical()
ok("runtime_matches_lock", rt.get("matches_lock"), {k: rt[k] for k in rt if k != "packages"} if isinstance(rt, dict) else rt)
ok("interpreter", Path(sys.executable).resolve() == Path(r["numerical_python"]).resolve() or sys.executable == r["numerical_python"], sys.executable)
rep = {"checks": checks, "pass": all(v["pass"] for v in checks.values())}
print(json.dumps(rep, indent=1, default=str))
sys.exit(0 if rep["pass"] else 1)
