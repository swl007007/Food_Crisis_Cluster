"""Gate for the next c4 invocation of release codex-D30-full-20261010 (continuation receipt 77aa4bf). Exit 0 = may resume."""
import glob, hashlib, json, subprocess, sys
from pathlib import Path

F = Path("/home/swl007007/ipcch-alphaearth-runs/p1-c-full-20261010")
REPO = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHAlphaEarthExperiment")
rec = json.loads((F / "receipt" / "ipcch-alphaearth-c4-continuation-77aa4bf-20261011.json").read_text())
fails, out = [], {}
exits = (F / "invocations.exit.log").read_text().splitlines()
runner = [l for l in exits if l.startswith("exit ")]
checker = [l for l in exits if l.startswith("checker exit ")]
starts = [l for l in exits if l.startswith("start ")]
out["last_lines"] = exits[-3:]
if not (len(runner) == len(checker) == len(starts)):
    fails.append("unbalanced start/exit/checker lines (an invocation may still be running or was killed)")
if not runner or not runner[-1].startswith("exit 0 "):
    fails.append(f"last runner exit not 0: {runner[-1:] }")
if not checker or not checker[-1].startswith("checker exit 0 "):
    fails.append(f"last checker exit not 0: {checker[-1:]}")
summ = [json.loads(x) for x in (F / "out" / "run_summaries.jsonl").read_text().splitlines()]
if len(summ) != len(runner) or "invocation wall budget" not in str(summ[-1].get("stopped")):
    fails.append(f"last invocation did not end on its wall budget: {summ[-1].get('stopped') if summ else None}")
c = json.loads(Path(sorted(glob.glob(str(F / "check-*.json")))[-1]).read_text())
out["checker"] = {k: c[k] for k in ("done", "pending", "done_cells", "pending_cells", "ledger_statuses", "call_reservations", "call_cap",
                                     "invocations", "max_invocations", "summed_invocation_seconds", "observed_cells_per_wall_second",
                                     "projected_remaining_wall_s", "projected_within_invocation_cap", "piece_seconds_max", "last_stop",
                                     "complete", "n_violations")}
if c["n_violations"]:
    fails.append("checker violations")
if c["complete"]:
    fails.append("COMPLETE: no further invocation (proceed to ingest checkpoint)")
if c["call_reservations"] >= c["call_cap"] or c["invocations"] >= c["max_invocations"]:
    fails.append("call or invocation cap reached")
remaining_window = (c["max_invocations"] - c["invocations"]) * 21600
if c["projected_remaining_wall_s"] is None or c["projected_remaining_wall_s"] > remaining_window:
    fails.append(f"projection {c['projected_remaining_wall_s']} s exceeds remaining window {remaining_window} s")
if (c["call_cap"] - c["call_reservations"]) < c["pending"] + 2 * (c["max_invocations"] - c["invocations"]):
    fails.append("remaining calls cannot cover pending pieces plus setup calls")
ps = "\n".join(l for l in subprocess.run(["pgrep", "-af", "ee_pilot_c.py"], capture_output=True, text=True).stdout.splitlines()
               if l.split(None, 1)[1].startswith(("/opt/earthengine/bin/python", "timeout")))  # real runners only
if ps:
    fails.append(f"active EE runner: {ps}")
for name, want in rec["source_sha256"].items():
    if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != want:
        fails.append(f"source hash differs: {name}")
m = json.loads((F / "inputs" / "production_manifest.json").read_text())
if m["request_sha256"] != "63d0c7abc11a02a974c3d5c0aca2c05b124505a6ff3d40081c8e25015f42496a":
    fails.append("manifest body differs")
errs = [json.loads(x) for x in (F / "out" / "request_log.jsonl").read_text().splitlines() if '"status": "error"' in x]
out["request_errors_total"] = len(errs)
out["request_errors_last"] = errs[-3:]
out["fails"], out["pass"] = fails, not fails
print(json.dumps(out, indent=1, default=str))
sys.exit(0 if not fails else 1)
