"""Stage 2 general consensus over every Stage 1 candidate (design.md "Stage 2").

python scripts/run_stage2.py --run-dir runs/<id>

Requires a complete candidate ledger: every scheduled Stage 1 fold must have a
completion record, a correspondence table and finite held-out macro-F1 scores.
Then computes D9 weights. If all weights are zero (D15) it records the
null-consensus route and builds nothing. Otherwise it runs the release steps
1/3/4/5/6 (general suffix, k40, sigma 5, recommended cluster count, seed 42).
Output: ``<run>/stage2/`` with ``consensus.json`` as the Stage 3 contract.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts.step4_similarity_matrix import compute_plan_weights  # noqa: E402
from scripts.run_stage1 import verify_fold  # noqa: E402
from src.utils.run_identity import (code_identity, output_hashes, refuse_existing,  # noqa: E402
                                    require_prepared, runtime_identity, write_json_atomic)

STEPS = (
    ("step1_merge_results.py", ["--model-type", "georf"]),
    ("step3_create_linked_tables.py", []),
    ("step4_similarity_matrix.py", []),
    ("step5_sparsification.py", ["--suffix", "general"]),
    ("step6_complete_clustering_pipeline.py", ["--suffix", "general"]),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_ledger(run: Path, prepared_identity=None) -> pd.DataFrame:
    schedule = json.loads((run / "prepared" / "manifests" / "schedule.json").read_text(encoding="utf-8"))
    rows, problems = [], []
    for fold in schedule["stage1"]:
        name = f"fs{fold['scope']}_{fold['target_month']}"
        if fold["status"] != "scheduled":
            rows.append({"candidate": name, "status": fold["status"]})
            continue
        fold_dir = run / "stage1" / "folds" / name
        record = fold_dir / "candidate.json"
        try:
            if prepared_identity is None:
                raise RuntimeError("no verified preparation")
            plan = json.loads((run / "stage1" / "retain_plan.json").read_text(encoding="utf-8"))
            verify_fold(run, name, prepared_identity,
                        retain_expected=f"fs{fold['scope']}:{fold['target_month']}" in plan)
        except Exception as exc:  # missing, partial, stale or foreign fold
            problems.append(f"{name}: {exc}")
            continue
        data = json.loads(record.read_text(encoding="utf-8"))
        scores = data["scores"]
        if not np.isfinite([scores["macro_f1"], scores["macro_f1_base"]]).all():
            problems.append(f"{name}: non-finite held-out score")
        rows.append({"candidate": name, "status": "completed", "scope": fold["scope"],
                     "target_month": fold["target_month"], "macro_f1": scores["macro_f1"],
                     "macro_f1_base": scores["macro_f1_base"],
                     "n_terminal": data["partition"]["n_terminal"],
                     "heldout_rows": data["rows"]["heldout_target"]})
    if problems:
        raise RuntimeError("incomplete Stage 1 candidate ledger (not a D15 null consensus):\n"
                           + "\n".join(problems))
    return pd.DataFrame(rows)


def finish(out: Path, record: dict) -> None:
    """consensus.json is the completion record, written last with every output hash."""
    record.update(code=code_identity(), runtime=runtime_identity(),
                  outputs={rel: sha for rel, sha in output_hashes(out).items() if rel != "consensus.json"})
    write_json_atomic(out / "consensus.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "stage2"
    refuse_existing(out, "Stage 2")
    ledger = candidate_ledger(run, require_prepared(run))
    out.mkdir(parents=True)
    ledger.to_csv(out / "candidate_ledger.csv", index=False)
    completed = ledger[ledger["status"] == "completed"].copy()
    weights = compute_plan_weights(completed)
    weights.to_csv(out / "plan_weights.csv", index=False)
    positive = int((weights["weight"] > 0).sum())
    record = {"candidates_scheduled": int(len(ledger)), "candidates_completed": int(len(completed)),
              "candidates_skipped_empty": int((ledger["status"] != "completed").sum()),
              "positive_weight_candidates": positive,
              "weight_rule": "max(0, logit(clip(F_part)) - logit(clip(F_pooled))), clip [1e-6, 1-1e-6] (D9)",
              "consensus_pool": "general, fs1-fs3 together (month-specific consensus not run)"}
    if positive == 0:
        record.update(route="null_consensus",
                      note=("D15: the complete candidate ledger has only zero weights. No similarity, "
                            "graph or clustering is built; Stage 3's partitioned arm reuses the pooled RF."))
        finish(out, record)
        print("null consensus", flush=True)
        return

    experiment = out / "experiment"
    results = experiment / "GeoRFResults"
    shutil.copytree(run / "stage1" / "GeoRFResults", results)
    shutil.copy2(run / "prepared" / "geometry" / "FEWSNET_admin_code_lat_lon.csv", experiment)
    steps = []
    for script, extra in STEPS:
        command = [sys.executable, "-B", str(PACKAGE / "scripts" / script), "--experiment-dir", str(experiment)] + extra
        with (out / f"{script}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        steps.append({"script": script, "returncode": code})
        if code != 0:
            raise RuntimeError(f"{script} failed; see {out / (script + '.log')}")

    main_index = pd.read_csv(experiment / "linked_tables" / "main_index.csv")
    if len(main_index) != len(completed):
        raise RuntimeError("step1/step3 did not carry every completed candidate")
    report = json.loads((experiment / "knn_sparsification_results" / "knn_analysis_report_k40_general.json")
                        .read_text(encoding="utf-8"))
    n_clusters = int(report["recommended_clusters"])
    cluster_map = experiment / "knn_sparsification_results" / f"cluster_mapping_k40_nc{n_clusters}_general.csv"
    mapping = pd.read_csv(cluster_map)
    summary = json.loads((experiment / "similarity_matrices" / "summary_statistics.json").read_text(encoding="utf-8"))
    record.update(
        route="learned_map", cluster_map=str(cluster_map), cluster_map_sha256=sha256(cluster_map),
        recommended_clusters=n_clusters, actual_clusters=int(mapping["cluster_id"].nunique()),
        areas_in_scope=int(len(mapping)), outlier_areas_1nn=int(mapping["is_outlier"].sum()),
        connectivity=report["connectivity"], similarity=summary, steps=steps,
        cluster_sizes={str(k): int(v) for k, v in mapping["cluster_id"].value_counts().sort_index().items()},
        scope_rule="in-scope areas = any non-s-1 assignment in some candidate (step4 default)",
        admin_universe="0..5717 (step3); out-of-scope areas are unmapped and use the pooled RF in Stage 3")
    finish(out, record)
    print(f"learned map with {n_clusters} clusters", flush=True)


if __name__ == "__main__":
    main()
