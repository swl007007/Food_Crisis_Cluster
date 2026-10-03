"""Stage 2 general consensus over an explicit, complete candidate set (design "Stage 2").

Library entry ``build_consensus(out_dir, candidates)`` — used by the development and
final drivers — plus a CLI wrapper. ``candidates`` is the COMPLETE expected list for one
map (every H pooled into one general map): name, horizon, target_month, macro_f1,
macro_f1_base, n_terminal and the accepted correspondence-table path. Missing or
non-finite evidence is an error before this call; the routes are distinct:

* ``no_prior_candidates`` - the schedule legitimately offers no candidate before O;
* ``null_consensus``      - a complete pool whose E4 weights are all zero (inherited rule);
* ``no_scorable_evidence``- (crisis metric) candidates exist but none has a defined E3 score;
* ``learned_map``         - release steps 3/4/5/6 (k40, sigma 5, recommended clusters,
                            seed 42) over the step-1 merge of these candidates.

Interruption task (10-02 design G4): ``metric="crisis"`` uses the matched E3 crisis F1 of each
candidate against its own root. Scheduled candidates without a defined score (genuine NA,
``no_e3_target_labels``, ``root_insufficient_support``) stay in the ledger as ineligible with a
reason and never enter the graph; missing or inconsistent artifacts are errors.

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
#: Interruption task crisis-metric ledger (scored = defined matched E3 crisis F1 on both sides).
CRISIS_LEDGER_COLUMNS = ["name", "strategy", "horizon", "target_month", "scenario_k", "status", "crisis_f1",
                         "crisis_f1_base", "na_reason", "n_terminal", "evidence_last_month", "correspondence_sha256",
                         "source"]
CRISIS_TEXT = ("name", "strategy", "target_month", "status", "na_reason", "evidence_last_month",
               "correspondence_sha256", "source")
INELIGIBLE_ROOT_STATUS = ("no_e3_target_labels", "root_insufficient_support")


def ledger_columns(metric: str) -> list:
    if metric not in ("macro", "crisis"):
        raise ValueError(metric)
    return CRISIS_LEDGER_COLUMNS if metric == "crisis" else LEDGER_COLUMNS


def canonical(frame: pd.DataFrame, metric: str = "macro") -> pd.DataFrame:
    """Ledger rows as comparable text (empty text and NaN coincide after a CSV round trip)."""
    out = frame[ledger_columns(metric)].copy()
    if metric == "crisis":
        for c in CRISIS_TEXT:
            out[c] = out[c].fillna("").astype(str)
        for c in ("horizon", "scenario_k", "crisis_f1", "crisis_f1_base", "n_terminal"):
            out[c] = out[c].astype(float)
    return out.sort_values("name").reset_index(drop=True).astype(str)


def pool_identity(candidates: pd.DataFrame, metric: str = "macro") -> str:
    """Map identity = the exact ordered candidate list, scores and partition contents."""
    rows = canonical(candidates, metric).to_numpy().tolist()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:20]


def crisis_plan_weights(ledger: pd.DataFrame) -> pd.DataFrame:
    """G4 E4 on matched E3 crisis F1 versus the candidate's own root, with legitimate NA.

    Scored rows: w = max(0, logit(clip F) - logit(clip F_root)), clip [1e-6, 1-1e-6]. Ineligible
    rows (genuine NA or an ineligible scheduled status) keep w = NaN and their reason. A scored
    row with a non-finite/out-of-range score, or an ineligible row without a reason or with a
    score, is a corrupt ledger (error), never a zero weight."""
    from scipy.special import logit
    missing = set(CRISIS_LEDGER_COLUMNS) - set(ledger.columns)
    if missing:
        raise ValueError(f"crisis ledger lacks {sorted(missing)}")
    scored = ledger["status"].eq("scored").to_numpy()
    f = ledger["crisis_f1"].to_numpy(dtype=float)
    b = ledger["crisis_f1_base"].to_numpy(dtype=float)
    reason = ledger["na_reason"].fillna("").astype(str).str.strip().to_numpy()
    ok_scored = np.isfinite(f) & np.isfinite(b) & (f >= 0) & (f <= 1) & (b >= 0) & (b <= 1) & (reason == "")
    ok_na = np.isnan(f) & np.isnan(b) & (reason != "")
    bad = (scored & ~ok_scored) | (~scored & ~ok_na)
    if bad.any():
        raise ValueError(f"corrupt crisis ledger rows: {ledger.loc[bad, 'name'].tolist()[:5]}")
    eps = 1e-6
    w = np.full(len(ledger), np.nan)
    w[scored] = np.maximum(logit(np.clip(f[scored], eps, 1 - eps)) - logit(np.clip(b[scored], eps, 1 - eps)), 0.0)
    return ledger.assign(weight=w)


def scenario_candidate_row(stage_dir: Path, entry: dict) -> dict:
    """One crisis-ledger row for a scheduled scenario candidate from its saved Stage 1 evidence.

    The E3 score is recomputed from the saved matched target rows (target_predictions.csv:
    truth, partitioned, own-root pooled) with the reporting convention (undefined -> NA) and
    must agree with the candidate record's counts. A missing completion/candidate file or a
    disagreement is an error; ``no_e3_target_labels``/``root_insufficient_support`` roots are
    legitimate ineligible outcomes."""
    from src.metrics import fourclass
    stage_dir = Path(stage_dir)
    base = {"name": entry["candidate"], "strategy": entry["strategy"], "horizon": int(entry["horizon"]),
            "target_month": entry["target_month"], "scenario_k": int(entry["scenario_k"]),
            "source": f"{stage_dir.name}/candidates/{entry['candidate']}"}
    root_dir = stage_dir / "roots" / entry["root"]
    if not (root_dir / "completion.json").is_file():
        raise RuntimeError(f"{entry['root']}: no completion record")
    root = json.loads((root_dir / "root.json").read_text(encoding="utf-8"))
    completion = json.loads((root_dir / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != root.get("status") or root.get("candidates") != [entry["candidate"]]:
        raise RuntimeError(f"{entry['root']}: root record and completion disagree")
    if root.get("root") != entry["root"] or any(root.get(f) != entry[f] for f in
                                                ("strategy", "horizon", "target_month", "scenario_k")):
        raise RuntimeError(f"{entry['root']}: root.json identity differs from the schedule entry")
    if root["status"] in INELIGIBLE_ROOT_STATUS:
        return {**base, "status": root["status"], "crisis_f1": np.nan, "crisis_f1_base": np.nan,
                "na_reason": root["status"], "n_terminal": np.nan, "evidence_last_month": entry["target_month"],
                "correspondence_sha256": ""}
    if root["status"] != "completed":
        raise RuntimeError(f"{entry['root']}: unexpected root status {root['status']!r}")
    cand_dir = stage_dir / "candidates" / entry["candidate"]
    record = json.loads((cand_dir / "candidate.json").read_text(encoding="utf-8"))
    if record.get("candidate") != entry["candidate"]:
        raise RuntimeError(f"{entry['candidate']}: candidate.json describes another candidate")
    preds = pd.read_csv(cand_dir / "target_predictions.csv")
    truth = preds["y_true_code"].to_numpy(dtype=np.int64)
    part = preds["y_pred_partitioned_code"].to_numpy(dtype=np.int64)
    pool = preds["y_pred_pooled_code"].to_numpy(dtype=np.int64)
    for side, pred in (("partitioned_crisis", part), ("pooled_crisis", pool)):
        counts = fourclass.crisis_counts(truth, pred)
        if any(counts[k] != record["scores"][side][k] for k in ("tp", "fp", "fn", "tn")):
            raise RuntimeError(f"{entry['candidate']}: saved E3 rows disagree with the candidate record ({side})")
    f, fb = fourclass.crisis_f1_exact_or_none(truth, part), fourclass.crisis_f1_exact_or_none(truth, pool)
    path = cand_dir / "correspondence_table.csv"
    # Full evidence span from the saved role lineage (fitting/S/C/E3 keys); IPC inputs of every
    # key were released by its own origin, which precedes these label months.
    roles = pd.read_csv(root_dir / "fold_membership.csv.gz")
    last = max(roles["target_month"].astype(str).max(), entry["target_month"])
    row = {**base, "n_terminal": int(record["partition"]["n_terminal"]), "evidence_last_month": last,
           "correspondence_sha256": file_sha256(path)}
    if f is None or fb is None:
        side = "partitioned" if f is None else "root"
        return {**row, "status": "e3_undefined", "crisis_f1": np.nan, "crisis_f1_base": np.nan,
                "na_reason": f"e3_crisis_f1_undefined:{side} (2TP+FP+FN=0)"}
    return {**row, "status": "scored", "crisis_f1": float(f), "crisis_f1_base": float(fb), "na_reason": ""}


def scenario_map_pool(ledger: pd.DataFrame, strategy: str, cutoff: int, release_ledger, strict: bool,
                      k_max: int = 2) -> pd.DataFrame:
    """G4 common origin-legal candidate pool for one strategy's general map at a cutoff month.

    Keeps that strategy's candidates (all horizons and scenarios) whose E3 target precedes the
    cutoff (strictly for a development origin, ``strict``; at or before it for the 2020-12
    final freeze), whose target cycle is released by the cutoff in every country, and whose
    full evidence span (``evidence_last_month``: fitting/S/C/E3 label months; their IPC inputs
    were released before their own origins) ends before every cycle hidden at the cutoff by
    its latest ``k_max`` cycles, i.e. the common pool across compared scenarios k = 0..k_max.
    The release ledger refuses publication orders that disagree with reference order."""
    from src.experiment.stage3 import mi
    targets = ledger["target_month"].map(mi).to_numpy()
    last = np.maximum(ledger["evidence_last_month"].map(mi).to_numpy(), targets)
    hidden = release_ledger.hidden(cutoff, k_max) if k_max else frozenset()
    first_hidden = min(hidden) if hidden else np.inf
    released = np.array([release_ledger.fully_released(t, cutoff) for t in targets], dtype=bool)
    before = targets < cutoff if strict else targets <= cutoff
    keep = (ledger["strategy"] == strategy).to_numpy() & before & released & (last < first_hidden)
    return ledger.loc[keep].reset_index(drop=True)


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
    weights = weights[np.isfinite(weights["weight"].to_numpy(float))]   # crisis: eligible rows only
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
                    label: str, keep_matrices: bool = False, metric: str = "macro") -> dict:
    """Build (or accept an identical existing) map for exactly ``candidates``."""
    out = Path(out)
    record_path = out / "consensus.json"
    columns = ledger_columns(metric)
    if record_path.exists():
        return accept_consensus(out, candidates, metric)
    if out.exists():
        raise FileExistsError(f"{out} exists without a completion record; never continued")
    started = time.time()
    out.mkdir(parents=True)
    candidates = candidates.sort_values("name").reset_index(drop=True)
    candidates[columns].to_csv(out / "candidate_ledger.csv", index=False, float_format="%.17g")
    record = {"label": label, "metric": metric, "pool_identity": pool_identity(candidates, metric),
              "candidates": int(len(candidates)),
              "consensus_pool": "general, all horizons together",
              "weight_rule": "max(0, logit(clip(F_part)) - logit(clip(F_pooled))), clip [1e-6, 1-1e-6] (D9)"
                             + (" on matched E3 crisis F1 vs the candidate's own root; NA ineligible (G4)"
                                if metric == "crisis" else "")}
    if candidates.empty:
        record.update(route="no_prior_candidates", positive_weight_candidates=0,
                      note="no Stage 1 candidate target precedes this origin; Stage 3 uses the pooled global")
        return _finish(out, record, started)
    weights = crisis_plan_weights(candidates) if metric == "crisis" else compute_plan_weights(candidates)
    weights[columns + ["weight"]].to_csv(out / "plan_weights.csv", index=False, float_format="%.17g")
    eligible = np.isfinite(weights["weight"].to_numpy(float))
    positive = int((weights["weight"] > 0).sum())
    record.update(positive_weight_candidates=positive, eligible_candidates=int(eligible.sum()),
                  ineligible_reasons={str(k): int(v) for k, v in
                                      weights.loc[~eligible, "na_reason"].value_counts().items()}
                  if metric == "crisis" else {})
    if not eligible.any():
        record.update(route="no_scorable_evidence",
                      note="candidates exist but none has a defined E3 score: Stage 3 uses the pooled global")
        return _finish(out, record, started)
    weights = weights[eligible].reset_index(drop=True)
    record.update(diagnostics=diagnostics(weights, paths))
    if metric == "crisis":   # step 1-6 read the legacy score columns; they carry crisis F1 here
        weights = weights.assign(macro_f1=weights["crisis_f1"], macro_f1_base=weights["crisis_f1_base"])
        record["legacy_score_columns"] = "macro_f1/macro_f1_base carry matched E3 crisis F1 (eligible rows only)"
    if positive == 0:
        record.update(route="null_consensus",
                      note="complete candidate pool with only zero weights (inherited null-consensus rule): no graph is built")
        return _finish(out, record, started)

    experiment = out / "experiment"
    experiment.mkdir()
    shutil.copy2(geometry_csv, experiment / "FEWSNET_admin_code_lat_lon.csv")
    record["geometry"] = {"path": str(geometry_csv), "sha256": file_sha256(geometry_csv)}
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


def accept_consensus(out: Path, candidates: pd.DataFrame | None = None, metric: str = "macro") -> dict:
    """Accept a map: completion record, current code/runtime, every output hash, and (if
    given) the exact expected candidate pool; returns the record."""
    from src.utils.run_identity import check_inventory
    out = Path(out)
    record = json.loads((out / "consensus.json").read_text(encoding="utf-8"))
    if record.get("metric", "macro") != metric:
        raise RuntimeError(f"{out}: map built with metric {record.get('metric', 'macro')!r}, expected {metric!r}")
    LEDGER = ledger_columns(metric)
    if record.get("code") != code_identity() or record.get("runtime") != runtime_identity():
        raise RuntimeError(f"{out}: map made by different code or runtime")
    required = ["candidate_ledger.csv"] + (["plan_weights.csv"] if record["route"] != "no_prior_candidates" else [])
    if record["route"] == "learned_map":
        required.append(record["cluster_map"])
    problems = check_inventory(out, record.get("outputs") or {}, required)
    if problems:
        raise RuntimeError(f"{out}: {problems[:5]}")
    if candidates is not None and record["pool_identity"] != pool_identity(candidates.reset_index(drop=True), metric):
        raise RuntimeError(f"{out}: map pool differs from the expected candidate pool")
    ledger = pd.read_csv(out / "candidate_ledger.csv", float_precision="round_trip")
    if (record["route"] == "no_prior_candidates") != ledger.empty or record.get("candidates") != len(ledger):
        raise RuntimeError(f"{out}: route {record['route']} contradicts a ledger of {len(ledger)} candidates")
    if candidates is not None:
        want = canonical(candidates, metric)
        got = canonical(ledger, metric) if len(ledger) else pd.DataFrame(columns=LEDGER)
        if len(want) != len(got) or not (want.to_numpy() == got.to_numpy()).all():
            raise RuntimeError(f"{out}: persisted ledger differs from the expected candidate pool")
    if record["route"] != "no_prior_candidates":
        weights = crisis_plan_weights(ledger) if metric == "crisis" else compute_plan_weights(ledger)
        saved = pd.read_csv(out / "plan_weights.csv", float_precision="round_trip")
        if not np.array_equal(saved["weight"].to_numpy(float), weights["weight"].to_numpy(float), equal_nan=True):
            raise RuntimeError(f"{out}: plan weights differ from weights recomputed from the ledger")
        positive = int((weights["weight"] > 0).sum())
        eligible = int(np.isfinite(weights["weight"].to_numpy(float)).sum())
        if (record["route"] == "no_scorable_evidence") != (eligible == 0):
            raise RuntimeError(f"{out}: route {record['route']} contradicts {eligible} scorable candidates")
        if eligible and (record["route"] == "null_consensus") != (positive == 0):
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
