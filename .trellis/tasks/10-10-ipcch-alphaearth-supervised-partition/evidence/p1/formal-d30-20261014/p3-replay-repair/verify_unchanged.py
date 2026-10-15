import hashlib, json, sys
from pathlib import Path
ROOT = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster")
r = json.loads(Path(sys.argv[1]).read_text()); run = Path(r["run_directory"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
code = {k: sha(ROOT / k) == v for k, v in r["original_code_inventory"].items()}
changed = [k for k, ok in code.items() if not ok]
out = {"code_entries": len(code), "code_changed": changed,
       "config_ok": all(sha(ROOT / "IPCCHAlphaEarthExperiment/config" / k) == v for k, v in r["config_sha256"].items()),
       "frozen_inputs_ok": sum(sha(run / k) == v for k, v in r["frozen_input_artifact_sha256"].items()), "frozen_inputs": len(r["frozen_input_artifact_sha256"]),
       "science_ok": sum(sha(run / k) == v for k, v in r["immutable_P2_P3_science_artifact_sha256"].items()), "science": len(r["immutable_P2_P3_science_artifact_sha256"]),
       "failed_replay_ok": {k: sha(run / k) == v for k, v in r["failed_replay_artifact_sha256"].items()},
       "replay_py_old_pin_ok": r["class_A"]["old_sha256"] == r["original_code_inventory"]["IPCCHAlphaEarthExperiment/ipcch_alphaearth/replay.py"],
       "replay_py_new": sha(ROOT / "IPCCHAlphaEarthExperiment/ipcch_alphaearth/replay.py"),
       "test_replay_contracts_old_pin": r["class_A"]["tests_old_sha256"], "test_replay_contracts_new": sha(ROOT / "IPCCHAlphaEarthExperiment/tests/test_replay_contracts.py"),
       "run_files": sum(1 for p in run.rglob("*") if p.is_file())}
print(json.dumps(out, indent=1))
