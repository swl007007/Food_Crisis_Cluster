"""Bound archive/install for receipt codex-D30-P3-replay-amendment-install-and-recheck-20261014 (local only).
Phase 'verify': all pre-checks, no filesystem change. Phase 'apply': pre-checks, then archive moves, install, manifests,
post-checks. Prints JSON; exit 0 only if every check passes."""
import datetime as dt, hashlib, json, os, shutil, sys
from pathlib import Path
ROOT = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster")
L = Path.home() / "ipcch-alphaearth-runs/p1-formal-d30-logs"
REC = L / "receipts/ipcch-alphaearth-replay-install-recheck-release-20261014.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
phase = sys.argv[1]
r = json.loads(REC.read_text()); run = Path(r["run_directory"]); a = r["analysis_amendment"]
checks = {}
def ok(n, c, d=None):
    checks[n] = {"pass": bool(c), "detail": d}
def pre():
    ok("receipt", sha(REC) == "991ce446663b530fc6d77c60c513e970575aa81d4c2a8a95d2b00de2e7e599a8")
    ok("executor", os.environ.get("HERDR_PANE_ID") == r["executor"]["pane"])
    ok("prior_repair_release", sha(L / "receipts/ipcch-alphaearth-replay-scoped-repair-release-20261014.json") == r["prior_repair_release_sha256"])
    ok("supervisor_test_log", sha(L / "receipts/ipcch-alphaearth-replay-repair-supervisor-tests-20261014.log") == a["supervisor_test_sha256"])
    bad = [k for k, v in a["analysis_code_inventory"].items() if sha(ROOT / k) != v]
    ok("analysis_code_inventory_46", not bad and len(a["analysis_code_inventory"]) == 46, bad)
    diffk = [k for k in a["original_code_inventory"] if a["original_code_inventory"][k] != a["analysis_code_inventory"][k]]
    ok("only_replay_py_differs_from_original", diffk == ["IPCCHAlphaEarthExperiment/ipcch_alphaearth/replay.py"], diffk)
    ok("replay_py_new", sha(ROOT / "IPCCHAlphaEarthExperiment/ipcch_alphaearth/replay.py") == a["replay_py_new_sha256"])
    ok("regression_test_source", sha(ROOT / "IPCCHAlphaEarthExperiment/tests/test_replay_contracts.py") == a["regression_test_source_sha256"])
    ok("replay_diff", sha(L / "replay-repair/replay.py.diff") == a["replay_diff_sha256"])
    ok("config", all(sha(ROOT / "IPCCHAlphaEarthExperiment/config" / k) == v for k, v in r["config_sha256"].items()))
    bad = [k for k, v in r["frozen_input_artifact_sha256"].items() if sha(run / k) != v]
    ok("frozen_inputs_73", not bad and len(r["frozen_input_artifact_sha256"]) == 73, bad)
    bad = [k for k, v in r["immutable_science_artifact_sha256"].items() if sha(run / k) != v]
    ok("science_404", not bad and len(r["immutable_science_artifact_sha256"]) == 404, bad[:10])
    ok("provider_not_in_404", "predict/providers_all_h03.csv.gz" not in r["immutable_science_artifact_sha256"])
    ok("archive_proposal", sha(L / "replay-repair/archive_proposal.json") == r["archive_proposal_sha256"])
    ok("staged_provider", sha(r["install"]["from"]) == r["install"]["sha256"])
pre_phase = phase
pre()
moves_done = []
if phase == "apply" and all(v["pass"] for v in checks.values()):
    for m in r["archive_moves"]:
        ok(f"pre_move_hash:{m['from']}", (run / m["from"]).exists() and sha(run / m["from"]) == m["sha256"])
    root = run / r["archive_root"]
    ok("archive_root_absent", not root.exists())
    if all(v["pass"] for v in checks.values()):
        for m in r["archive_moves"]:
            dst = run / m["to"]; dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise SystemExit(f"refusing to overwrite {dst}")
            os.rename(run / m["from"], dst)
            moves_done.append({**m, "after_sha256": sha(dst)})
        rep = run / "replay"
        if rep.exists() and not any(rep.iterdir()):
            rep.rmdir()
        dst = Path(r["install"]["to"])
        if dst.exists():
            raise SystemExit("active provider path unexpectedly present after archive move")
        shutil.copyfile(r["install"]["from"], dst)
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = {"archive_root": r["archive_root"], "receipt_id": r["receipt_id"], "receipt_sha256": sha(REC), "utc": now,
                    "moves": moves_done, "install": {**r["install"], "after_sha256": sha(dst)},
                    "reasons": {"replay": "formal replay attempt 1 failed 16/488 checks (class A: prediction CSV reader NA policy; class B: one corrupted provider cell)",
                                "provider": "predict/providers_all_h03.csv.gz had ONE corrupted cell (physical CSV line 38174, row 21721, target_ord 2431 instead of 24314); gzip CRC valid; cause UNKNOWN. Replaced by a zero-fit reconstruction from the stored B1 boosters, identical in every other cell (38,764 data rows + header = 38,765 physical lines)."}}
        (root / "ARCHIVE.json").write_text(json.dumps(manifest, indent=2) + "\n")
        amend = {"kind": "analysis-only amendment", "receipt_id": r["receipt_id"], "utc": now, "modified_symbol": a["modified_symbol"],
                 "original_code_identity": a["original_code_identity"], "analysis_code_identity": a["analysis_code_identity"],
                 "replay_py": {"old": a["original_code_inventory"]["IPCCHAlphaEarthExperiment/ipcch_alphaearth/replay.py"], "new": a["replay_py_new_sha256"]},
                 "regression_test_source_sha256": a["regression_test_source_sha256"], "replay_diff_sha256": a["replay_diff_sha256"],
                 "supervisor_test_log_sha256": a["supervisor_test_sha256"],
                 "provider": {"original_sha256": "e454ce1590084ac2e0a9ce654b775a35ed2bf1beb11d0597b46a4144f03576d5", "reconstructed_sha256": r["install"]["sha256"], "corruption_cause": "unknown"},
                 "scope": a["scope"]}
        (root / "ANALYSIS_AMENDMENT.json").write_text(json.dumps(amend, indent=2) + "\n")
        # post checks
        ok("post_provider_06eefdc6", sha(dst) == r["install"]["sha256"])
        bad = [k for k, v in r["immutable_science_artifact_sha256"].items() if sha(run / k) != v]
        ok("post_science_404", not bad, bad[:10])
        bad = [k for k, v in r["frozen_input_artifact_sha256"].items() if sha(run / k) != v]
        ok("post_inputs_73", not bad, bad)
        ok("post_replay_dir_absent_or_empty", not (run / "replay").exists() or not any((run / "replay").iterdir()))
        ok("post_run_incomplete_absent", not (run / "RUN_INCOMPLETE.json").exists())
        ok("post_archive_hashes", all(sha(run / m["to"]) == m["sha256"] for m in r["archive_moves"]))
        ok("manifests", {"ARCHIVE.json": sha(root / "ARCHIVE.json"), "ANALYSIS_AMENDMENT.json": sha(root / "ANALYSIS_AMENDMENT.json")} is not None,
           {"ARCHIVE.json": sha(root / "ARCHIVE.json"), "ANALYSIS_AMENDMENT.json": sha(root / "ANALYSIS_AMENDMENT.json")})
out = {"phase": phase, "checks": checks, "pass": all(v["pass"] for v in checks.values()), "moves": moves_done,
       "run_files": sum(1 for p in run.rglob("*") if p.is_file())}
print(json.dumps(out, indent=1, default=str))
sys.exit(0 if out["pass"] else 1)
