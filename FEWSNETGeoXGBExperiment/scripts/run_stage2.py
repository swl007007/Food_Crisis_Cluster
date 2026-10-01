"""Stage 2 general consensus over an explicit, complete candidate set (design "Stage 2").

Library entry ``build_consensus(out_dir, candidates)`` — used by the development and
final drivers — plus a CLI wrapper. ``candidates`` is the COMPLETE expected list for one
map (every H pooled into one general map): name, horizon, target_month, macro_f1,
macro_f1_base, n_terminal and the accepted correspondence-table path. Missing or
non-finite evidence is an error before this call; the routes are distinct:

* ``no_prior_candidates`` - the schedule legitimately offers no candidate before O;
* ``null_consensus``      - a complete pool whose E4 weights are all zero (inherited rule);
* ``learned_map``         - release steps 3/4/5/6 (k40, sigma 5, recommended clusters,
                            seed 42) over the step-1 merge of these candidates.

Step 1's merge function is reused in-process with each candidate's unique name as its
"variant", so many candidates of one horizon/month cannot collide. ``consensus.json``
is written LAST with the hash of every output. Large intermediates (the similarity
NPZ) are hashed in the record and, when ``keep_matrices`` is False, deleted afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts.step1_merge_results import merge_results_with_correspondence  # noqa: E402
from scripts.step4_similarity_matrix import compute_plan_weights  # noqa: E402
from src.utils.run_identity import (code_identity, file_sha256, output_hashes,  # noqa: E402
                                    runtime_identity, write_json_atomic)

STEPS = (
    ("step3_create_linked_tables.py", []),
    ("step4_similarity_matrix.py", []),
    ("step5_sparsification.py", ["--suffix", "general"]),
    ("step6_complete_clustering_pipeline.py", ["--suffix", "general"]),
)
LEDGER_COLUMNS = ["name", "horizon", "target_month", "macro_f1", "macro_f1_base", "n_terminal",
                  "correspondence_sha256", "source"]


def pool_identity(candidates: pd.DataFrame) -> str:
    """Map identity = the exact ordered candidate list, scores and partition contents."""
    rows = candidates[LEDGER_COLUMNS].sort_values("name").astype(str).to_numpy().tolist()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:20]


def canonical_partition(path: Path):
    """Coverage-keyed partition with labels renamed in sorted-area order."""
    table = pd.read_csv(path, dtype={"partition_id": str}, keep_default_na=False)
    table = table.sort_values("FEWSNET_admin_code")
    relabel, labels = {}, []
    for pid in table["partition_id"]:
        relabel.setdefault(pid, len(relabel))
        labels.append(relabel[pid])
    return tuple(table["FEWSNET_admin_code"].astype(int)), tuple(labels)


def diagnostics(weights: pd.DataFrame, paths: dict) -> dict:
    """Plan section 4: counts, positive weights, same-coverage duplicates, concentration."""
    w = weights["weight"].to_numpy(float)
    positive = w > 0
    signatures = {}
    for name in weights["name"]:
        coverage, labels = canonical_partition(paths[name])
        signatures.setdefault(coverage, []).append((name, labels))
    duplicates = 0
    for members in signatures.values():
        seen = {}
        for name, labels in members:
            duplicates += labels in seen
            seen.setdefault(labels, name)
    total = float(w.sum())
    return {"candidates": int(len(w)), "split_candidates": int((weights["n_terminal"] > 1).sum()),
            "positive_weight": int(positive.sum()),
            "coverage_groups": int(len(signatures)),
            "same_coverage_duplicate_partitions": int(duplicates),
            "max_weight_share": float(w.max() / total) if total > 0 else None,
            "weight_concentration_sum_sq": float(total ** 2 / float((w ** 2).sum())) if total > 0 else None,
            "concentration_note": "(sum w)^2 / sum w^2 describes concentration only; not an effective N"}


def build_consensus(out: Path, candidates: pd.DataFrame, paths: dict, geometry_csv: Path,
                    label: str, keep_matrices: bool = False) -> dict:
    """Build (or accept an identical existing) map for exactly ``candidates``."""
    out = Path(out)
    record_path = out / "consensus.json"
    if record_path.exists():
        return accept_consensus(out, candidates)
    if out.exists():
        raise FileExistsError(f"{out} exists without a completion record; never continued")
    started = time.time()
    out.mkdir(parents=True)
    candidates = candidates.sort_values("name").reset_index(drop=True)
    candidates[LEDGER_COLUMNS].to_csv(out / "candidate_ledger.csv", index=False, float_format="%.17g")
    record = {"label": label, "pool_identity": pool_identity(candidates),
              "candidates": int(len(candidates)),
              "consensus_pool": "general, all horizons together",
              "weight_rule": "max(0, logit(clip(F_part)) - logit(clip(F_pooled))), clip [1e-6, 1-1e-6] (D9)"}
    if candidates.empty:
        record.update(route="no_prior_candidates", positive_weight_candidates=0,
                      note="no Stage 1 candidate target precedes this origin; Stage 3 uses the pooled global")
        return _finish(out, record, started)
    weights = compute_plan_weights(candidates)
    weights[LEDGER_COLUMNS + ["weight"]].to_csv(out / "plan_weights.csv", index=False, float_format="%.17g")
    positive = int((weights["weight"] > 0).sum())
    record.update(positive_weight_candidates=positive, diagnostics=diagnostics(weights, paths))
    if positive == 0:
        record.update(route="null_consensus",
                      note="complete candidate pool with only zero weights (inherited null-consensus rule): no graph is built")
        return _finish(out, record, started)

    experiment = out / "experiment"
    experiment.mkdir()
    shutil.copy2(geometry_csv, experiment / "FEWSNET_admin_code_lat_lon.csv")
    results = pd.DataFrame({"model": weights["name"], "year": weights["target_month"].str[:4],
                            "month": weights["target_month"].str[5:7],
                            "forecasting_scope": "fs" + weights["horizon"].astype(str),
                            "macro_f1": weights["macro_f1"], "macro_f1_base": weights["macro_f1_base"]})
    tables = [{"name": f"{n}_{t[:4]}_{t[5:7]}_fs{h}", "variant": n, "year": t[:4], "month": t[5:7],
               "forecasting_scope": f"fs{h}", "file_path": str(paths[n]),
               "dataframe": pd.read_csv(paths[n], dtype={"partition_id": str}, keep_default_na=False)}
              for n, t, h in zip(weights["name"], weights["target_month"], weights["horizon"])]
    merged, stats = merge_results_with_correspondence(results, tables)
    if len(merged) != len(weights) or (stats["status"] != "success").any():
        raise RuntimeError("step-1 merge did not carry every candidate exactly once")
    with (experiment / "merged_correspondence_tables.pkl").open("wb") as handle:
        pickle.dump(merged, handle)
    steps = []
    for script, extra in STEPS:
        command = [sys.executable, "-B", str(PACKAGE / "scripts" / script), "--experiment-dir", str(experiment)] + extra
        step_started = time.time()
        with (out / f"{script}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        steps.append({"script": script, "returncode": code, "seconds": round(time.time() - step_started, 1)})
        if code != 0:
            raise RuntimeError(f"{script} failed; see {out / (script + '.log')}")
    main_index = pd.read_csv(experiment / "linked_tables" / "main_index.csv")
    if len(main_index) != len(weights):
        raise RuntimeError("step3 did not carry every candidate")
    report = json.loads((experiment / "knn_sparsification_results" / "knn_analysis_report_k40_general.json")
                        .read_text(encoding="utf-8"))
    n_clusters = int(report["recommended_clusters"])
    cluster_map = experiment / "knn_sparsification_results" / f"cluster_mapping_k40_nc{n_clusters}_general.csv"
    mapping = pd.read_csv(cluster_map)
    summary = json.loads((experiment / "similarity_matrices" / "summary_statistics.json").read_text(encoding="utf-8"))
    npz = experiment / "similarity_matrices" / "similarity_matrices.npz"
    record.update(route="learned_map", cluster_map=cluster_map.relative_to(out).as_posix(),
                  cluster_map_sha256=file_sha256(cluster_map), recommended_clusters=n_clusters,
                  actual_clusters=int(mapping["cluster_id"].nunique()), areas_in_scope=int(len(mapping)),
                  outlier_areas_1nn=int(mapping["is_outlier"].sum()), connectivity=report["connectivity"],
                  similarity=summary, steps=steps, similarity_npz_sha256=file_sha256(npz),
                  similarity_npz_bytes=npz.stat().st_size, similarity_npz_retained=keep_matrices,
                  cluster_sizes={str(k): int(v) for k, v in mapping["cluster_id"].value_counts().sort_index().items()})
    if not keep_matrices:
        npz.unlink()
        for extra in experiment.glob("**/*.npz"):
            record.setdefault("deleted_intermediates", {})[extra.relative_to(out).as_posix()] = file_sha256(extra)
            extra.unlink()
    return _finish(out, record, started)


def _finish(out: Path, record: dict, started: float) -> dict:
    record.update(seconds=round(time.time() - started, 1), code=code_identity(), runtime=runtime_identity(),
                  outputs={rel: sha for rel, sha in output_hashes(out).items() if rel != "consensus.json"})
    write_json_atomic(out / "consensus.json", record)
    return record


def accept_consensus(out: Path, candidates: pd.DataFrame | None = None) -> dict:
    """Accept a map: completion record, current code/runtime, every output hash, and (if
    given) the exact expected candidate pool; returns the record."""
    from src.utils.run_identity import check_inventory
    out = Path(out)
    record = json.loads((out / "consensus.json").read_text(encoding="utf-8"))
    if record.get("code") != code_identity() or record.get("runtime") != runtime_identity():
        raise RuntimeError(f"{out}: map made by different code or runtime")
    required = ["candidate_ledger.csv"] + (["plan_weights.csv"] if record["route"] != "no_prior_candidates" else [])
    if record["route"] == "learned_map":
        required.append(record["cluster_map"])
    problems = check_inventory(out, record.get("outputs") or {}, required)
    if problems:
        raise RuntimeError(f"{out}: {problems[:5]}")
    if candidates is not None and record["pool_identity"] != pool_identity(candidates.reset_index(drop=True)):
        raise RuntimeError(f"{out}: map pool differs from the expected candidate pool")
    ledger = pd.read_csv(out / "candidate_ledger.csv", float_precision="round_trip")
    if record["route"] != "no_prior_candidates":
        weights = compute_plan_weights(ledger)
        saved = pd.read_csv(out / "plan_weights.csv", float_precision="round_trip")
        if not np.array_equal(saved["weight"].to_numpy(float), weights["weight"].to_numpy(float)):
            raise RuntimeError(f"{out}: plan weights differ from weights recomputed from the ledger")
        positive = int((weights["weight"] > 0).sum())
        if (record["route"] == "null_consensus") != (positive == 0):
            raise RuntimeError(f"{out}: route {record['route']} contradicts {positive} positive weights")
    return record


def cluster_map(out: Path, record: dict):
    """area -> cluster for a learned map, else None."""
    if record["route"] != "learned_map":
        return None
    mapping = pd.read_csv(Path(out) / record["cluster_map"])
    if mapping["FEWSNET_admin_code"].duplicated().any():
        raise RuntimeError("cluster map assigns an area twice")
    return dict(zip(mapping["FEWSNET_admin_code"].astype(int), mapping["cluster_id"].astype(int)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--accept-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(accept_consensus(args.map_dir), indent=1, default=str)[:2000])


if __name__ == "__main__":
    main()
