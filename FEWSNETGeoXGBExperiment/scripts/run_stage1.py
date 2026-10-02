"""Run every scheduled Stage 1 root and collect its four candidates (experiment-plan s.4).

python scripts/run_stage1.py --run-dir RUN [--workers 6] [--only h4_2018-02_G2_r80_s42,...]
python scripts/run_stage1.py --run-dir RUN --split-mode tb3 [--workers 6]

``--split-mode tb3`` (D27, experiment-plan A2) runs exactly the six scheduled
``stage1_tb3_roots`` (one L1/gt0 candidate each) into ``RUN/stage1_tb3/`` with the same
layout and completion record; the random-split 648 schedule stays in ``RUN/stage1/``.

Needs the accepted preparation and the frozen G selection (``RUN/gscreen``). Each root
(H, T, selected G, ratio, split seed) runs app/main_model_GF.py in its own process and a
scratch working directory outside Dropbox; boosters stay in ``RUN/stage1/checkpoints``.
Evidence is copied to ``RUN/stage1/roots/<root>/`` and ``RUN/stage1/candidates/<cand>/``;
``RUN/stage1/roots/<root>/completion.json`` is written LAST with the SHA-256 of every
root, candidate and checkpoint file. Existing output is never continued.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from src.experiment import plan  # noqa: E402
from src.utils.run_identity import (SCHEMA_PATH as SCHEMA, code_identity,  # noqa: E402
                                    output_hashes, require_prepared, runtime_identity, write_json_atomic)

ROOT_FILES = ("root.json", "fold_membership.csv.gz", "root_target_predictions.csv", "command.json", "run.log")
CANDIDATE_FILES = ("candidate.json", "correspondence_table.csv", "target_predictions.csv",
                   "heldout_scores.csv", "e2_predictions.csv.gz", "validation_predictions.csv.gz",
                   "s_branch.pkl", "branch_table.npy", "X_branch_id.npy")


#: split mode -> (schedule lists, Stage 1 output directory under the run)
SPLIT_MODES = {"random": ("stage1_roots", "stage1_candidates", "stage1"),
               plan.TIME_BLOCK: ("stage1_tb3_roots", "stage1_tb3_candidates", "stage1_tb3")}


def scheduled_roots(schedule: dict, g_of: dict, mode: str = "random") -> dict:
    """root name -> schedule entry with its selected G (the frozen 162, or the six tb3)."""
    out = {}
    for r in schedule[SPLIT_MODES[mode][0]]:
        g = g_of[str(r["horizon"])]
        out[plan.root_name(r["horizon"], r["target_month"], g, r["ratio"], r["split_seed"])] = {**r, "g_config": g}
    return out


def scheduled_candidates(schedule: dict, g_of: dict, mode: str = "random") -> dict:
    out = {}
    for c in schedule[SPLIT_MODES[mode][1]]:
        g = g_of[str(c["horizon"])]
        name = plan.candidate_name(c["horizon"], c["target_month"], g, c["local_config"], c["ratio"],
                                   c["split_seed"], c["threshold_family"])
        out[name] = {**c, "g_config": g,
                     "root": plan.root_name(c["horizon"], c["target_month"], g, c["ratio"], c["split_seed"])}
    return out


def require_tb3_g(g_of: dict) -> None:
    """D27 locks the D26 development crisis-F1 G selection; tb3 never reselects G."""
    if dict(g_of) != plan.TB3_G:
        raise SystemExit(f"tb3 requires the locked D26 G selection {plan.TB3_G}, got {dict(g_of)}")


def run_root(run: Path, name: str, root: dict, python: str, prepared_identity: dict, g_record: str,
             stage_dir: str = "stage1") -> dict:
    stage1 = run / stage_dir
    out_root = stage1 / "roots" / name
    if out_root.exists():
        raise FileExistsError(f"{name}: output exists; Stage 1 never continues a root")
    work = Path(tempfile.gettempdir()) / "geoxgb_stage1" / run.name / name  # tb3 names cannot collide
    if work.exists():
        shutil.rmtree(work)  # scratch from an interrupted attempt; nothing was collected
    work.mkdir(parents=True)
    ckpt = stage1 / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    prepared = run / "prepared"
    command = [python, "-B", str(PACKAGE / "app" / "main_model_GF.py"),
               "--data", str(prepared / f"snapshot_h{root['horizon']}.parquet"),
               "--geometry-dir", str(prepared / "geometry"), "--schema", str(SCHEMA),
               "--forecasting_scope", str(plan.SCOPE_OF[root["horizon"]]), "--desired_terms", root["target_month"],
               "--g-config", root["g_config"], "--ratio", root["ratio"], "--split-seed", str(root["split_seed"]),
               "--checkpoint-dir", str(ckpt)]
    (work / "command.json").write_text(json.dumps({"command": command, "cwd": str(work)}, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="5")
    env.pop("PYTHONPATH", None)
    started = time.time()
    with (work / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {"root": name, "status": "failed", "returncode": result.returncode, "log": str(work / "run.log")}
    record = json.loads((work / "root.json").read_text(encoding="utf-8"))
    out_root.mkdir(parents=True)
    for fname in ROOT_FILES:
        if (work / fname).exists():
            shutil.copy2(work / fname, out_root / fname)
    outputs = {f"roots/{name}/{rel}": sha for rel, sha in output_hashes(out_root).items()}
    if record["status"] == "completed":
        for cand in record["candidates"]:
            dest = stage1 / "candidates" / cand
            if dest.exists():
                raise FileExistsError(f"{cand}: candidate output exists")
            dest.mkdir(parents=True)
            for fname in CANDIDATE_FILES:
                shutil.copy2(work / cand / fname, dest / fname)
            outputs.update({f"candidates/{cand}/{rel}": sha for rel, sha in output_hashes(dest).items()})
            for rel, sha in json.loads((dest / "candidate.json").read_text(encoding="utf-8"))["checkpoints"]["sha256"].items():
                outputs[f"checkpoints/{cand}/{rel}"] = sha
    shutil.rmtree(work, ignore_errors=True)
    write_json_atomic(out_root / "completion.json", {
        "root": name, "status": record["status"], "candidates": record["candidates"],
        "prepared": prepared_identity["outputs_sha256"], "g_selection": g_record,
        "code": code_identity(), "runtime": runtime_identity(), "outputs": outputs})
    return {"root": name, "status": record["status"], "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", default="", help="comma list of root names (timing sample)")
    parser.add_argument("--split-mode", choices=sorted(SPLIT_MODES), default="random",
                        help="random: the 648 r80/r50 schedule; tb3: the six D27 time-block roots")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    from src.utils.acceptance import accept_g_selection
    prepared_identity = require_prepared(run)
    g_of, g_record = accept_g_selection(run)
    if args.split_mode == plan.TIME_BLOCK:
        require_tb3_g(g_of)
    schedule = json.loads((run / "prepared" / "manifests" / "schedule.json").read_text(encoding="utf-8"))
    roots = scheduled_roots(schedule, g_of, args.split_mode)
    stage_dir = SPLIT_MODES[args.split_mode][2]
    wanted = [s for s in args.only.split(",") if s]
    if wanted:
        unknown = set(wanted) - set(roots)
        if unknown:
            raise SystemExit(f"--only names unscheduled roots: {sorted(unknown)}")
        roots = {n: roots[n] for n in wanted}
    ledger = run / stage_dir / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_root, run, n, r, python, prepared_identity, g_record, stage_dir)
                for n, r in roots.items() if not (run / stage_dir / "roots" / n / "completion.json").exists() or wanted]
        for job in jobs:
            outcome = job.result()
            print(json.dumps(outcome), flush=True)
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(outcome) + "\n")
            failures += outcome["status"] == "failed"
    if failures:
        raise SystemExit(f"{failures} Stage 1 roots failed")


if __name__ == "__main__":
    main()
