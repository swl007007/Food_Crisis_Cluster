"""Prelaunch for the ONE-TIME exception invocation 5 (user fallback to the c4 window flow after the batch pilot FAIL).
Exit 0 = launch the unchanged run_invocation.sh once. Checks every receipt pin; the v2 gate must refuse ONLY because the
latest invocation is the pinned user-stopped one. Usage: python prelaunch_inv5.py --receipt FALLBACK_RECEIPT_JSON"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

import resume_check_v2 as gate

F = gate.RUN
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", required=True)
    a = p.parse_args(argv)
    rec = json.loads(Path(a.receipt).read_text())
    fails = []
    for name, want in rec["terminal_log_sha256"].items():
        if sha(F / name) != want:
            fails.append(f"terminal log changed: {name}")
    starts = len((F / "out" / "invocations.log").read_text().splitlines())
    calls = len((F / "out" / "call_reservations.log").read_text().splitlines())
    outs = len(list((F / "out" / "pieces").glob("*.json")))
    e = rec["expected"]
    if (starts, calls, outs) != (e["starts"], e["reservations"], e["unique_outputs"]):
        fails.append(f"counters differ: {(starts, calls, outs)}")
    if subprocess.run(["flock", "-n", str(F / "invocation.lock"), "true"]).returncode != 0:
        fails.append("invocation.lock is held")
    for name, want in {"inputs/production_manifest.json": rec["manifest_file_sha256"], "inputs/production_pieces.jsonl.gz": rec["pieces_sha256"],
                       "inputs/production_plan.csv.gz": rec["plan_sha256"], "run_invocation.sh": rec["run_invocation_sha256"],
                       "resume_check_v2.py": rec["gate_sha256"]}.items():
        if sha(F / name) != want:
            fails.append(f"pinned file differs: {name}")
    ps = subprocess.run(["pgrep", "-af", "ee_pilot_c.py"], capture_output=True, text=True).stdout
    g = gate.evaluate(F, rec, gate.REPO, ps)
    expected = ["the latest invocation is the pinned user-stopped one: a normal resume needs a later normal-wall invocation"]
    if g["fails"] != expected:
        fails.append(f"v2 gate refusals are not exactly the accepted user stop: {g['fails']}")
    out = {"starts": starts, "reservations": calls, "unique_outputs": outs, "gate": g, "fails": fails, "pass": not fails}
    print(json.dumps(out, indent=1, default=str))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
