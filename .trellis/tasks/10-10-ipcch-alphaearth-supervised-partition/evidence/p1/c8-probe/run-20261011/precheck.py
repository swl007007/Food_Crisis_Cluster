"""Preconditions for the released c8 probe (codex-D30-c8-probe-20261011). Exit 0 only if all hold; prints a JSON record."""
import gzip, hashlib, json, subprocess, sys
from pathlib import Path

F = Path("/home/swl007007/ipcch-alphaearth-runs/p1-c-full-20261010")
R = Path("/home/swl007007/ipcch-alphaearth-runs/p1-c8-probe-20261011")
REPO = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHAlphaEarthExperiment")
rel = json.loads((R / "receipt" / "release.json").read_text())
fails, rec = [], {}

def sig(p):
    return hashlib.sha256(json.dumps({"g": p["geometry"], "c": p["candidates"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

# 1 full invocation 1: runner exit code, normal wall stop, clean checker
exits = (F / "invocations.exit.log").read_text().splitlines()
rec["full_exit_log"] = exits
runner = [l for l in exits if l.startswith("exit ")]
checker = [l for l in exits if l.startswith("checker exit ")]
if len(runner) != 1 or not runner[0].startswith("exit 0 "):
    fails.append(f"runner exit not exactly one normal exit 0: {runner}")
if len(checker) != 1 or not checker[0].startswith("checker exit 0 "):
    fails.append(f"checker exit not 0: {checker}")
summ = [json.loads(x) for x in (F / "out" / "run_summaries.jsonl").read_text().splitlines()] if (F / "out" / "run_summaries.jsonl").exists() else []
rec["full_last_summary"] = summ[-1] if summ else None
if not summ or "invocation wall budget" not in str(summ[-1].get("stopped")):
    fails.append("full invocation 1 did not stop on its invocation wall budget")
checks = sorted(F.glob("check-*.json"))
if not checks:
    fails.append("no checker JSON")
else:
    c = json.loads(checks[-1].read_text())
    rec["full_checker"] = {k: c[k] for k in ("done", "pending", "n_violations", "call_reservations", "call_cap", "invocations", "max_invocations", "projected_within_invocation_cap", "last_stop")}
    if c["n_violations"] or not c["projected_within_invocation_cap"]:
        fails.append("full checker reports violations or projection beyond cap")
    # the checker's exit status does not enforce the budget fields: compare them to the caps explicitly
    if c["call_reservations"] >= c["call_cap"]:
        fails.append(f"full call reservations {c['call_reservations']} reached cap {c['call_cap']}")
    if c["invocations"] != 1 or c["invocations"] >= c["max_invocations"]:
        fails.append(f"full invocations {c['invocations']} (expected exactly 1, below cap {c['max_invocations']})")
# 2 no active EE runner
ps = subprocess.run(["pgrep", "-af", "ee_pilot_c.py"], capture_output=True, text=True).stdout.strip()
rec["active_ee_runners"] = ps
if ps:
    fails.append(f"active EE runner: {ps}")
# 3 hashes
for name, want in rel["source_sha256"].items():
    if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != want:
        fails.append(f"source hash differs: {name}")
m = R / "inputs" / "C8_probe_manifest.json"
if hashlib.sha256(m.read_bytes()).hexdigest() != rel["released_manifest_file_sha256"]:
    fails.append("released manifest file hash differs")
man = json.loads(m.read_text())
body = {k: v for k, v in man.items() if k not in ("request_sha256", "release_id")}
if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest() != rel["body_sha256"]:
    fails.append("body hash differs")
pp = R / "inputs" / "C8_probe_pieces.jsonl.gz"
if hashlib.sha256(pp.read_bytes()).hexdigest() != rel["pieces_sha256"]:
    fails.append("pieces hash differs")
if (R / "out").exists():
    fails.append("probe output dir already exists")
# 4 signatures of actually attempted full requests vs the 24 probe pieces
probe = [json.loads(x) for x in gzip.decompress(pp.read_bytes()).decode().splitlines()]
full = {json.loads(x)["request_id"]: json.loads(x) for x in gzip.decompress((F / "inputs" / "production_pieces.jsonl.gz").read_bytes()).decode().splitlines()}
attempted = {json.loads(x)["piece_id"] for x in (F / "out" / "ledger.jsonl").read_text().splitlines()}
attempted |= {json.loads(x)["purpose"] for x in (F / "out" / "request_log.jsonl").read_text().splitlines()} & set(full)
asig = {sig(full[p]): p for p in attempted}
overlap = [(p["request_id"], asig[sig(p)]) for p in probe if sig(p) in asig]
rec["attempted_full_requests"] = len(attempted)
rec["probe_signature_overlap"] = overlap
if overlap:
    fails.append(f"probe geometry+candidate signatures already attempted in the full run: {overlap}")
rec["fails"] = fails
rec["pass"] = not fails
print(json.dumps(rec, indent=1, default=str))
sys.exit(0 if not fails else 1)
