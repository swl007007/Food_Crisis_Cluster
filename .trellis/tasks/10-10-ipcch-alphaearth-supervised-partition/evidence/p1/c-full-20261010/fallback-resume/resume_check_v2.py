"""Normal-wall resume gate v2 for release codex-D30-full-20261010 (external, run-local; replaces resume_check.py only
after the fallback receipt binds it). Exit 0 = the next normal resume may start.

Identical checks to v1, except the summary/exit bookkeeping: invocation 4 was user-stopped (runner exit 143) before
writing a run summary. The receipt pins that single historical gap by its exact start and exit lines; every other
runner exit must have exactly one run summary, in order. The LATEST invocation must have runner exit 0, checker exit 0
and its own summary ending on the invocation wall budget. The projection's elapsed time is the checker's summed summary
seconds plus the receipt-pinned wall seconds of the gap (invocation 4: 10:17:32Z -> 12:45:56Z = 8,904 s), because the
gap's outputs are counted in done_cells while its time is in no summary. No time, start or call budget is recredited.
Usage: python resume_check_v2.py --receipt RECEIPT_JSON [--run RUN_DIR] [--repo PKG_DIR]
"""
import argparse, glob, hashlib, json, subprocess, sys
from pathlib import Path

RUN = Path("/home/swl007007/ipcch-alphaearth-runs/p1-c-full-20261010")
REPO = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHAlphaEarthExperiment")
BODY = "63d0c7abc11a02a974c3d5c0aca2c05b124505a6ff3d40081c8e25015f42496a"


def invocations(run: Path) -> list:
    """Exit log -> [{start, exit, checker}] in order (lock-busy lines ignored)."""
    out = []
    for line in (run / "invocations.exit.log").read_text().splitlines():
        if line.startswith("start "):
            out.append({"start": line, "exit": None, "checker": None})
        elif line.startswith("exit ") and out:
            out[-1]["exit"] = line
        elif line.startswith("checker exit ") and out:
            out[-1]["checker"] = line
    return out


def runners(pgrep_output: str) -> str:
    return "\n".join(l for l in pgrep_output.splitlines() if l.split(None, 1)[1].startswith(("/opt/earthengine/bin/python", "timeout")))


def evaluate(run: Path, receipt: dict, repo: Path, pgrep_output: str) -> dict:
    fails, out = [], {}
    invs = invocations(run)
    out["invocations"] = len(invs)
    starts = len((run / "out" / "invocations.log").read_text().splitlines())
    if starts != len(invs) or any(i["exit"] is None or i["checker"] is None for i in invs):
        fails.append("start/exit/checker records incomplete (an invocation may be running or was killed without an exit line)")
    gaps = receipt["accepted_summary_gaps"]
    gap_idx = []
    for g in gaps:
        k = int(g["invocation"]) - 1
        if not (0 <= k < len(invs) and invs[k]["start"] == g["start_line"] and invs[k]["exit"] == g["exit_line"]):
            fails.append(f"pinned summary gap for invocation {g['invocation']} does not match the exit log")
        gap_idx.append(k)
    summ = [json.loads(x) for x in (run / "out" / "run_summaries.jsonl").read_text().splitlines()]
    with_summary = [k for k in range(len(invs)) if k not in gap_idx]
    if len(summ) != len(with_summary):
        fails.append(f"{len(invs)} runner exits, {len(summ)} summaries, {len(gap_idx)} pinned gap(s): an unapproved summary is missing or extra")
    last = invs[-1] if invs else None
    if not last or (len(invs) - 1) in gap_idx:
        fails.append("the latest invocation is the pinned user-stopped one: a normal resume needs a later normal-wall invocation")
    elif not (last["exit"] or "").startswith("exit 0 ") or not (last["checker"] or "").startswith("checker exit 0 "):
        fails.append(f"latest invocation did not end with runner exit 0 and checker exit 0: {last['exit']} / {last['checker']}")
    elif not summ or "invocation wall budget" not in str(summ[-1].get("stopped")):
        fails.append(f"latest invocation did not end on its wall budget: {summ[-1].get('stopped') if summ else None}")
    c = json.loads(Path(sorted(glob.glob(str(run / "check-*.json")))[-1]).read_text())
    if c["n_violations"] or c["complete"]:
        fails.append("checker reports violations, or the extraction is complete (go to the ingest checkpoint)")
    if c["call_reservations"] >= c["call_cap"] or len(invs) >= c["max_invocations"]:
        fails.append("call or invocation cap reached")
    # elapsed = checker's summed summary seconds + the receipt-pinned wall seconds of each accepted summary gap
    # (the gap's outputs are in done_cells, its time is not in any summary); nothing is recredited
    wall = float(c["summed_invocation_seconds"]) + sum(float(gp["wall_s"]) for gp in gaps)
    rate = c["done_cells"] / wall if wall > 0 else 0
    projected = c["pending_cells"] / rate if rate else float("inf")
    window = (c["max_invocations"] - len(invs)) * 21600
    out.update(corrected_elapsed_s=wall, cells_per_elapsed_s=rate, projected_remaining_s=projected, remaining_window_s=window)
    if projected > window:
        fails.append(f"projection {projected:.0f} s exceeds the remaining window {window} s")
    if c["call_cap"] - c["call_reservations"] < c["pending"] + 2 * (c["max_invocations"] - len(invs)):
        fails.append("remaining calls cannot cover pending pieces plus setup calls")
    ps = runners(pgrep_output)
    if ps:
        fails.append(f"active EE runner: {ps}")
    for name, want in receipt["source_sha256"].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != want:
            fails.append(f"source hash differs: {name}")
    if json.loads((run / "inputs" / "production_manifest.json").read_text())["request_sha256"] != BODY:
        fails.append("manifest body differs")
    out["checker"] = {k: c[k] for k in ("done", "pending", "call_reservations", "invocations", "n_violations")}
    out["fails"], out["pass"] = fails, not fails
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", required=True)
    p.add_argument("--run", default=str(RUN))
    p.add_argument("--repo", default=str(REPO))
    a = p.parse_args(argv)
    ps = subprocess.run(["pgrep", "-af", "ee_pilot_c.py"], capture_output=True, text=True).stdout
    rep = evaluate(Path(a.run), json.loads(Path(a.receipt).read_text()), Path(a.repo), ps)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
